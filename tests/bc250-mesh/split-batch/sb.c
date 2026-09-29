/* RADV_BC250_SPLIT_BATCH_PREP offline IB harness (derived from mesh-timer/timer.c).
 * drm-shim ONLY: refuses to run when the noop drm-shim is not preloaded or when the device is not the
 * shim's GFX1013 (run.sh also hides /dev/dri).
 * usage: sb <script> [submits]
 * Shaders from the work directory: nanite.mesh.spv / nanite.frag.spv (256V/128P, per-primitive output:
 * split=3), als32.mesh.spv / als32.frag.spv (64V/124P: split=2), small.mesh.spv / small.frag.spv
 * (32V/32P, not split), tri_vert.spv / tri_frag.spv (VS).
 * The script is a list of space-separated tokens, "tok*N" repeats a token N times:
 *   B Bs Br Brs Bx   vkCmdBeginRendering (flags: none, SUSPENDING, RESUMING, both, SECONDARY contents)
 *   E                vkCmdEndRendering
 *   R N RE           vkCmdBeginRenderPass2 (2 subpasses, subpass 0 has a self-dependency) / NextSubpass2 /
 *                    EndRenderPass2
 *   P                vkCmdPipelineBarrier2 inside subpass 0 (the self-dependency, framebuffer-local)
 *   C c              vkCmdBeginConditionalRenderingEXT / End
 *   X<n>             vkCmdExecuteCommands of a secondary holding n "si" draws (inside a Bx instance)
 *   si  nanite vkCmdDrawMeshTasksIndirectEXT, 2 records, stride 12
 *   s1  nanite vkCmdDrawMeshTasksIndirectEXT, 1 record, stride 12
 *   sc  nanite vkCmdDrawMeshTasksIndirectCountEXT, max 4, stride 16, count buffer value (k % 6)
 *   sL  nanite vkCmdDrawMeshTasksIndirectCountEXT, max 1000, stride 12, count value 700 (16 workgroups of
 *       setup per draw; records at args + 128 KiB, shared by all sL draws)
 *   ai  als32  vkCmdDrawMeshTasksIndirectEXT, 3 records, stride 20
 *   ac  als32  vkCmdDrawMeshTasksIndirectCountEXT, max 3, stride 12, count buffer value 5 (> max)
 *   sd  nanite vkCmdDrawMeshTasksEXT (direct split, never batched)
 *   m / mi  small Mesh direct / indirect (2 records)
 *   v   VS draw (3 vertices)
 * Indirect draw number k (in recording order, over all command buffers of one submit) reads its records at
 * args + 256*k and its count at args + 256*k + 128. Prints SB_ARGS_VA and one SB_DRAW line per indirect
 * draw (stdout). */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d @%d\n", #x, r_, __LINE__); fflush(stdout); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice pd;

static VkShaderModule load(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) { perror(path); exit(2); }
   fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc(n); if (fread(code, 1, n, f) != (size_t)n) exit(2); fclose(f);
   VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, n, code};
   VkShaderModule m; CK(vkCreateShaderModule(dev, &ci, NULL, &m)); free(code); return m;
}

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
   exit(2);
}

enum { SH_NANITE, SH_ALS32, SH_SMALL, SH_VS, SH_COUNT };
enum { CTX_DYN, CTX_RP0, CTX_RP1, CTX_COUNT };
static const char *shape_names[SH_COUNT] = {"nanite", "als32", "small", "tri"};
static VkPipeline pipes[SH_COUNT][CTX_COUNT];
static VkPipelineLayout pl;
static VkRenderPass rpass;
static VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;

static VkPipeline pipeline(int shape, int ctx)
{
   if (pipes[shape][ctx])
      return pipes[shape][ctx];
   char m[64], f[64];
   VkPipelineRenderingCreateInfo prci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &cf};
   VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
   VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
   rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_BACK_BIT; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   rs.polygonMode = VK_POLYGON_MODE_FILL;
   VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkPipelineColorBlendAttachmentState cba = {0}; cba.colorWriteMask = 0xf;
   VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &cba};
   VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyns};
   VkPipelineShaderStageCreateInfo st[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_MESH_BIT_EXT, VK_NULL_HANDLE, "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, VK_NULL_HANDLE, "main"}};
   VkPipelineVertexInputStateCreateInfo vis = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ias = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
                                                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkGraphicsPipelineCreateInfo gpi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, ctx == CTX_DYN ? &prci : NULL, 0, 2, st,
                                       NULL, NULL, NULL, &vps, &rs, &ms, NULL, &cbs, &ds, pl};
   if (ctx != CTX_DYN) {
      gpi.renderPass = rpass;
      gpi.subpass = ctx == CTX_RP1;
   }
   if (shape == SH_VS) {
      st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
      st[0].module = load("tri_vert.spv");
      st[1].module = load("tri_frag.spv");
      gpi.pVertexInputState = &vis; gpi.pInputAssemblyState = &ias;
   } else {
      snprintf(m, sizeof(m), "%s.mesh.spv", shape_names[shape]);
      snprintf(f, sizeof(f), "%s.frag.spv", shape_names[shape]);
      st[0].module = load(m);
      st[1].module = load(f);
   }
   VkResult pr = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipes[shape][ctx]);
   printf("PIPELINE %s ctx=%d RESULT=%d\n", shape_names[shape], ctx, pr); fflush(stdout);
   if (pr) exit(1);
   return pipes[shape][ctx];
}

static PFN_vkCmdDrawMeshTasksEXT drawMesh;
static PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshInd;
static PFN_vkCmdDrawMeshTasksIndirectCountEXT drawMeshCnt;
static PFN_vkCmdBeginConditionalRenderingEXT beginCond;
static PFN_vkCmdEndConditionalRenderingEXT endCond;
static VkBuffer args;
static uint32_t *argmap;
static VkDeviceAddress args_va;
static unsigned draw_k;
static int ctx;

static uint32_t rnd_state = 12345;
static uint32_t rnd(void) { rnd_state = rnd_state * 1103515245u + 12345u; return (rnd_state >> 16) & 0x7fff; }

/* Fills the records of indirect draw k at byte offset off: dims 0..4 (x, y, z), some zero records. */
static void fill(unsigned k, unsigned off, unsigned records, unsigned stride, int count_value)
{
   for (unsigned r = 0; r < records; r++) {
      uint32_t *p = argmap + (off + r * stride) / 4;
      for (unsigned c = 0; c < 3; c++)
         p[c] = rnd() % 5;
      if ((k + r) % 7 == 3) p[0] = 0;
      if ((k + r) % 4 != 1) { p[0] |= 1; p[1] |= 1; p[2] |= 1; }
   }
   if (count_value >= 0)
      argmap[(256 * k + 128) / 4] = count_value;
}

static void indirect(VkCommandBuffer cb, const char *tok, int shape, unsigned records, unsigned stride, int count_value)
{
   unsigned k = draw_k++;
   if (256 * (k + 1) > 128 * 1024) { printf("FAIL too many draws\n"); exit(1); }
   const unsigned off = !strcmp(tok, "sL") ? 128 * 1024 : 256 * k;
   fill(k, off, records, stride, count_value);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(shape, ctx));
   printf("SB_DRAW k=%u tok=%s shape=%s input=0x%llx records=%u stride=%u count=0x%llx count_value=%d\n", k, tok,
          shape_names[shape], (unsigned long long)(args_va + off), records, stride,
          count_value >= 0 ? (unsigned long long)(args_va + 256 * k + 128) : 0ull, count_value);
   if (count_value >= 0)
      drawMeshCnt(cb, args, off, args, 256 * k + 128, records, stride);
   else
      drawMeshInd(cb, args, off, records, stride);
}

int main(int argc, char **argv)
{
   if (argc < 2) { fprintf(stderr, "usage: %s script [submits]\n", argv[0]); return 2; }
   const int submits = argc > 2 ? atoi(argv[2]) : 2;
   const char *pre = getenv("LD_PRELOAD");
   if (!pre || !strstr(pre, "drm_shim")) {
      printf("REFUSED: drm-shim only (hide /dev/dri and preload libamdgpu_noop_drm_shim.so)\n");
      return 9;
   }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "splitbatch", 1, NULL, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
   VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1; if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || !n) return 3;
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
   if (!strstr(pp.deviceName, "GFX1013")) { printf("REFUSED: device %s is not the shim GFX1013\n", pp.deviceName); return 9; }

   VkPhysicalDeviceConditionalRenderingFeaturesEXT condf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
   condf.conditionalRendering = VK_TRUE;
   VkPhysicalDeviceMeshShaderFeaturesEXT meshf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, &condf};
   meshf.meshShader = VK_TRUE;
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &meshf};
   f12.drawIndirectCount = VK_TRUE; f12.bufferDeviceAddress = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
   f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
   const char *exts[] = {"VK_EXT_mesh_shader", "VK_EXT_conditional_rendering"};
   float prio = 1;
   VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, NULL, 2, exts, NULL};
   CK(vkCreateDevice(pd, &dci, NULL, &dev));
   drawMesh = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   drawMeshInd = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
   drawMeshCnt = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectCountEXT");
   beginCond = (void *)vkGetDeviceProcAddr(dev, "vkCmdBeginConditionalRenderingEXT");
   endCond = (void *)vkGetDeviceProcAddr(dev, "vkCmdEndConditionalRenderingEXT");
   VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

   {
      const VkDeviceSize size = 256 * 1024;
      VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT, VK_SHARING_MODE_EXCLUSIVE};
      CK(vkCreateBuffer(dev, &bi, NULL, &args));
      VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, args, &mr);
      VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
      VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi, mr.size,
                                 memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
      VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, NULL, &m)); CK(vkBindBufferMemory(dev, args, m, 0));
      CK(vkMapMemory(dev, m, 0, size, 0, (void **)&argmap));
      memset(argmap, 0, size);
      VkBufferDeviceAddressInfo bai = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, args};
      args_va = vkGetBufferDeviceAddress(dev, &bai);
      printf("SB_ARGS_VA 0x%llx\n", (unsigned long long)args_va);
   }

   VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, cf,
                           {256, 256, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage img; CK(vkCreateImage(dev, &ii, NULL, &img));
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, 0)};
   VkDeviceMemory im; CK(vkAllocateMemory(dev, &ai, NULL, &im)); CK(vkBindImageMemory(dev, img, im, 0));
   VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D,
                               cf, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView rtv; CK(vkCreateImageView(dev, &vi, NULL, &rtv));

   /* Legacy render pass: 2 subpasses on the same color attachment, subpass 0 has a by-region
    * self-dependency (color attachment output -> fragment shader, as a framebuffer-local barrier). */
   VkAttachmentDescription2 ad = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, NULL, 0, cf, VK_SAMPLE_COUNT_1_BIT,
                                  VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                  VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL};
   VkAttachmentReference2 ar = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT};
   VkSubpassDescription2 sd[2] = {
      {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, NULL, 0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, 0, NULL, 1, &ar},
      {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, NULL, 0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, 0, NULL, 1, &ar}};
   VkMemoryBarrier2 self_mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL,
                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT};
   VkSubpassDependency2 dep[2] = {
      {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2, &self_mb, 0, 0, 0, 0, 0, 0, VK_DEPENDENCY_BY_REGION_BIT, 0},
      {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2, NULL, 0, 1, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, 0}};
   VkRenderPassCreateInfo2 rpci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2, NULL, 0, 1, &ad, 2, sd, 2, dep};
   CK(vkCreateRenderPass2(dev, &rpci, NULL, &rpass));
   VkFramebufferCreateInfo fbci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, NULL, 0, rpass, 1, &rtv, 256, 256, 1};
   VkFramebuffer fb; CK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkPushConstantRange pcr = {VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
   VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pcr};
   CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

   VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
   VkCommandPool cpool; CK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));
   VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
   VkCommandBuffer cmd; CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkCommandBuffer secs[16]; unsigned nsec = 0;
   cai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY; cai.commandBufferCount = 16;
   CK(vkAllocateCommandBuffers(dev, &cai, secs));
   VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, rtv, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   VkViewport vp = {0, 0, 256, 256, 0, 1};
   VkRect2D sc = {{0, 0}, {256, 256}};
   uint32_t base = 0;

   /* tokens */
   char *script = strdup(argv[1]);
   char *toks[4096]; unsigned ntok = 0;
   for (char *t = strtok(script, " "); t; t = strtok(NULL, " ")) {
      char *star = strchr(t, '*');
      unsigned rep = 1;
      if (star) { *star = 0; rep = atoi(star + 1); }
      for (unsigned r = 0; r < rep && ntok < 4096; r++) toks[ntok++] = t;
   }

   for (int s = 0; s < submits; s++) {
      CK(vkResetCommandPool(dev, cpool, 0));
      draw_k = 0; nsec = 0; ctx = CTX_DYN; rnd_state = 12345;
      VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      CK(vkBeginCommandBuffer(cmd, &bi));
      vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
      vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &base);
      for (unsigned i = 0; i < ntok; i++) {
         const char *t = toks[i];
         VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {256, 256}}, 1, 0, 1, &att};
         if (!strcmp(t, "B") || !strcmp(t, "Bs") || !strcmp(t, "Br") || !strcmp(t, "Brs") || !strcmp(t, "Bx")) {
            if (strchr(t + 1, 's')) ri.flags |= VK_RENDERING_SUSPENDING_BIT;
            if (strchr(t + 1, 'r')) ri.flags |= VK_RENDERING_RESUMING_BIT;
            if (t[1] == 'x') ri.flags |= VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
            vkCmdBeginRendering(cmd, &ri);
            ctx = CTX_DYN;
         } else if (!strcmp(t, "E")) {
            vkCmdEndRendering(cmd);
         } else if (!strcmp(t, "R")) {
            VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, NULL, rpass, fb, {{0, 0}, {256, 256}}, 0, NULL};
            VkSubpassBeginInfo sbi = {VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO, NULL, VK_SUBPASS_CONTENTS_INLINE};
            vkCmdBeginRenderPass2(cmd, &rbi, &sbi);
            ctx = CTX_RP0;
         } else if (!strcmp(t, "N")) {
            VkSubpassBeginInfo sbi = {VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO, NULL, VK_SUBPASS_CONTENTS_INLINE};
            VkSubpassEndInfo sei = {VK_STRUCTURE_TYPE_SUBPASS_END_INFO};
            vkCmdNextSubpass2(cmd, &sbi, &sei);
            ctx = CTX_RP1;
         } else if (!strcmp(t, "RE")) {
            VkSubpassEndInfo sei = {VK_STRUCTURE_TYPE_SUBPASS_END_INFO};
            vkCmdEndRenderPass2(cmd, &sei);
            ctx = CTX_DYN;
         } else if (!strcmp(t, "P")) {
            VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, VK_DEPENDENCY_BY_REGION_BIT, 1, &self_mb};
            vkCmdPipelineBarrier2(cmd, &di);
         } else if (!strcmp(t, "C")) {
            VkConditionalRenderingBeginInfoEXT cbi = {VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT, NULL, args, 256 * 1023, 0};
            argmap[256 * 1023 / 4] = 1;
            beginCond(cmd, &cbi);
         } else if (!strcmp(t, "c")) {
            endCond(cmd);
         } else if (t[0] == 'X') {
            unsigned nd = atoi(t + 1);
            VkCommandBuffer sec = secs[nsec++];
            VkCommandBufferInheritanceRenderingInfo inh_r = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO, NULL, 0, 0, 1, &cf,
                                                             VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT};
            VkCommandBufferInheritanceInfo inh = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, &inh_r};
            VkCommandBufferBeginInfo sbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &inh};
            CK(vkBeginCommandBuffer(sec, &sbi));
            vkCmdSetViewport(sec, 0, 1, &vp); vkCmdSetScissor(sec, 0, 1, &sc);
            vkCmdPushConstants(sec, pl, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &base);
            for (unsigned d = 0; d < nd; d++)
               indirect(sec, "si@secondary", SH_NANITE, 2, 12, -1);
            CK(vkEndCommandBuffer(sec));
            vkCmdExecuteCommands(cmd, 1, &sec);
         } else if (!strcmp(t, "si")) {
            indirect(cmd, t, SH_NANITE, 2, 12, -1);
         } else if (!strcmp(t, "s1")) {
            indirect(cmd, t, SH_NANITE, 1, 12, -1);
         } else if (!strcmp(t, "sc")) {
            indirect(cmd, t, SH_NANITE, 4, 16, (int)(draw_k % 6));
         } else if (!strcmp(t, "sL")) {
            indirect(cmd, t, SH_NANITE, 1000, 12, 700);
         } else if (!strcmp(t, "ai")) {
            indirect(cmd, t, SH_ALS32, 3, 20, -1);
         } else if (!strcmp(t, "ac")) {
            indirect(cmd, t, SH_ALS32, 3, 12, 5);
         } else if (!strcmp(t, "sd")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(SH_NANITE, ctx));
            drawMesh(cmd, 2, 1, 1);
         } else if (!strcmp(t, "m")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(SH_SMALL, ctx));
            drawMesh(cmd, 3, 1, 1);
         } else if (!strcmp(t, "mi")) {
            indirect(cmd, t, SH_SMALL, 2, 12, -1);
         } else if (!strcmp(t, "v")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(SH_VS, ctx));
            vkCmdDraw(cmd, 3, 1, 0, 0);
         } else {
            printf("FAIL unknown token %s\n", t);
            return 1;
         }
      }
      CK(vkEndCommandBuffer(cmd));
      VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd};
      CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
      CK(vkQueueWaitIdle(q));
      printf("SUBMIT_OK %d\n", s); fflush(stdout);
   }
   vkDestroyDevice(dev, NULL);
   printf("DONE\n");
   return 0;
}

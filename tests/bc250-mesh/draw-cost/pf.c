/* Draw-cost switches offline IB harness (RADV_BC250_SPLIT_PREP_FREE, RADV_BC250_SCRATCH_REUSE,
 * RADV_BC250_SPLIT_LEAN_SETUP; derived from split-batch/sb.c).
 * drm-shim ONLY: refuses to run when the noop drm-shim is not preloaded or when the device is not the
 * shim's GFX1013 (run.sh also hides /dev/dri).
 * usage: pf <script> [submits]      env PF_ONE_TIME=1: begin the primary with ONE_TIME_SUBMIT (default:
 *                                    no usage flags, as vkd3d-proton records its command lists)
 * Shaders from the work directory: nanite.mesh.spv (256V/128P, per-primitive output: split=3) with the
 * fragment shaders vis/used/exch/store.frag.spv (pf.frag KIND 0..3) and col.frag.spv (color output),
 * small.mesh.spv / small.frag.spv (32V/32P, not split), tri_vert.spv / tri_frag.spv (VS), cs.comp.spv.
 * Two pipeline contexts: color only (B) and color + D32S8 depth/stencil (BD). Depth test, depth write,
 * depth compare (LESS) and stencil test are dynamic, all off at command buffer begin.
 * Tokens ("tok*N" repeats a token N times):
 *   B BD E          vkCmdBeginRendering (color / color + depth-stencil) / vkCmdEndRendering
 *   dt0 dt1 dw0 dw1 st0 st1   vkCmdSetDepthTestEnable / DepthWriteEnable / StencilTestEnable
 *   si  nanite+vis   vkCmdDrawMeshTasksIndirectEXT, 2 records, stride 12
 *   s1  nanite+vis   1 record, stride 12
 *   sc  nanite+vis   vkCmdDrawMeshTasksIndirectCountEXT, max 4, stride 16, count value k % 6
 *   ci cc  nanite+col   indirect (2 records) / indirect count (max 3, stride 12, count 2)
 *   ui xi wi  nanite+used / exch / store   indirect, 2 records, stride 20
 *   sd  nanite+vis   vkCmdDrawMeshTasksEXT (2,1,1) (direct split)
 *   m mi  small direct (3,1,1) / indirect (2 records)
 *   v   VS draw
 *   p   vkCmdPushConstants (graphics + compute stages, value = token index)
 *   Q q   vkCmdBeginQuery / vkCmdEndQuery (pipeline statistics: mesh + fragment + compute invocations;
 *         occlusion queries crash the RADV_DEBUG=dumpibs packet printer of this tree)
 *   cb  bind the application compute pipeline (+ its set); cd  vkCmdDispatch(2,1,1) (outside rendering)
 *   X<n>  vkCmdExecuteCommands of a secondary holding n "si" draws (inside a "Bx" instance)
 *   Bx  vkCmdBeginRendering with SECONDARY_COMMAND_BUFFERS contents (color only)
 * Indirect draw k reads its records at args + 256*k and its count at args + 256*k + 128.
 * Prints PF_ARGS_VA and one PF_DRAW line per indirect draw of the primary (stdout). */
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

enum { P_VIS, P_USED, P_EXCH, P_STORE, P_COL, P_SMALL, P_VS, P_COUNT };
static const char *pnames[P_COUNT] = {"vis", "used", "exch", "store", "col", "small", "tri"};
static VkPipeline pipes[P_COUNT][2];
static VkPipelineLayout pl;
static VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM, dsf = VK_FORMAT_D32_SFLOAT_S8_UINT;
static int ctx; /* 0 color only, 1 color + depth/stencil */

static VkPipeline pipeline(int kind)
{
   if (pipes[kind][ctx])
      return pipes[kind][ctx];
   char m[64], f[64];
   VkPipelineRenderingCreateInfo prci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &cf,
                                         ctx ? dsf : VK_FORMAT_UNDEFINED, ctx ? dsf : VK_FORMAT_UNDEFINED};
   VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
   VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
   rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_BACK_BIT; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   rs.polygonMode = VK_POLYGON_MODE_FILL;
   VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkPipelineColorBlendAttachmentState cba = {0}; cba.colorWriteMask = 0xf;
   VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &cba};
   VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
   dss.depthCompareOp = VK_COMPARE_OP_LESS;
   dss.front.failOp = dss.front.passOp = dss.front.depthFailOp = VK_STENCIL_OP_INCREMENT_AND_CLAMP;
   dss.front.compareOp = VK_COMPARE_OP_ALWAYS; dss.front.writeMask = dss.front.compareMask = 0xff;
   dss.back = dss.front;
   VkDynamicState dyns[5] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                             VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE};
   VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 5, dyns};
   VkPipelineShaderStageCreateInfo st[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_MESH_BIT_EXT, VK_NULL_HANDLE, "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, VK_NULL_HANDLE, "main"}};
   VkPipelineVertexInputStateCreateInfo vis = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ias = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
                                                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkGraphicsPipelineCreateInfo gpi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st,
                                       NULL, NULL, NULL, &vps, &rs, &ms, &dss, &cbs, &ds, pl};
   if (kind == P_VS) {
      st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
      st[0].module = load("tri_vert.spv");
      st[1].module = load("tri_frag.spv");
      gpi.pVertexInputState = &vis; gpi.pInputAssemblyState = &ias;
   } else {
      snprintf(m, sizeof(m), "%s.mesh.spv", kind == P_SMALL ? "small" : "nanite");
      snprintf(f, sizeof(f), "%s.frag.spv", pnames[kind]);
      st[0].module = load(m);
      st[1].module = load(f);
   }
   VkResult pr = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipes[kind][ctx]);
   printf("PIPELINE %s ctx=%d RESULT=%d\n", pnames[kind], ctx, pr); fflush(stdout);
   if (pr) exit(1);
   return pipes[kind][ctx];
}

static PFN_vkCmdDrawMeshTasksEXT drawMesh;
static PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshInd;
static PFN_vkCmdDrawMeshTasksIndirectCountEXT drawMeshCnt;
static VkBuffer args;
static uint32_t *argmap;
static VkDeviceAddress args_va;
static unsigned draw_k;
static VkDescriptorSet dset;

static uint32_t rnd_state = 12345;
static uint32_t rnd(void) { rnd_state = rnd_state * 1103515245u + 12345u; return (rnd_state >> 16) & 0x7fff; }

static void fill(unsigned k, unsigned records, unsigned stride, int count_value)
{
   for (unsigned r = 0; r < records; r++) {
      uint32_t *p = argmap + (256 * k + r * stride) / 4;
      for (unsigned c = 0; c < 3; c++)
         p[c] = rnd() % 5;
      if ((k + r) % 7 == 3) p[0] = 0;
      if ((k + r) % 4 != 1) { p[0] |= 1; p[1] |= 1; p[2] |= 1; }
   }
   if (count_value >= 0)
      argmap[(256 * k + 128) / 4] = count_value;
}

static void indirect(VkCommandBuffer cb, const char *tok, int kind, unsigned records, unsigned stride, int count_value)
{
   unsigned k = draw_k++;
   if (256 * (k + 1) > 128 * 1024) { printf("FAIL too many draws\n"); exit(1); }
   fill(k, records, stride, count_value);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(kind));
   printf("PF_DRAW k=%u tok=%s fs=%s ctx=%d input=0x%llx records=%u stride=%u count=0x%llx count_value=%d\n", k, tok,
          pnames[kind], ctx, (unsigned long long)(args_va + 256 * k), records, stride,
          count_value >= 0 ? (unsigned long long)(args_va + 256 * k + 128) : 0ull, count_value);
   if (count_value >= 0)
      drawMeshCnt(cb, args, 256 * k, args, 256 * k + 128, records, stride);
   else
      drawMeshInd(cb, args, 256 * k, records, stride);
}

int main(int argc, char **argv)
{
   if (argc < 2) { fprintf(stderr, "usage: %s script [submits]\n", argv[0]); return 2; }
   const int submits = argc > 2 ? atoi(argv[2]) : 2;
   const char *ot = getenv("PF_ONE_TIME");
   const int one_time = ot && atoi(ot);
   const char *pre = getenv("LD_PRELOAD");
   if (!pre || !strstr(pre, "drm_shim")) {
      printf("REFUSED: drm-shim only (hide /dev/dri and preload libamdgpu_noop_drm_shim.so)\n");
      return 9;
   }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "drawcost", 1, NULL, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
   VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1; if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || !n) return 3;
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
   if (!strstr(pp.deviceName, "GFX1013")) { printf("REFUSED: device %s is not the shim GFX1013\n", pp.deviceName); return 9; }

   VkPhysicalDeviceMeshShaderFeaturesEXT meshf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   meshf.meshShader = VK_TRUE;
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &meshf};
   f12.drawIndirectCount = VK_TRUE; f12.bufferDeviceAddress = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
   f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
   f2.features.fragmentStoresAndAtomics = VK_TRUE;
   f2.features.pipelineStatisticsQuery = VK_TRUE;
   const char *exts[] = {"VK_EXT_mesh_shader"};
   float prio = 1;
   VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, NULL, 1, exts, NULL};
   CK(vkCreateDevice(pd, &dci, NULL, &dev));
   drawMesh = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   drawMeshInd = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
   drawMeshCnt = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectCountEXT");
   VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

   VkBuffer vbuf;
   {
      const VkDeviceSize size = 256 * 1024;
      VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
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
      printf("PF_ARGS_VA 0x%llx\n", (unsigned long long)args_va);
      bi.size = 4096; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      CK(vkCreateBuffer(dev, &bi, NULL, &vbuf));
      vkGetBufferMemoryRequirements(dev, vbuf, &mr);
      VkMemoryAllocateInfo ai2 = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, 0)};
      VkDeviceMemory m2; CK(vkAllocateMemory(dev, &ai2, NULL, &m2)); CK(vkBindBufferMemory(dev, vbuf, m2, 0));
   }

   VkImageView views[2];
   for (int i = 0; i < 2; i++) {
      VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, i ? dsf : cf,
                              {256, 256, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                              i ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
      VkImage img; CK(vkCreateImage(dev, &ii, NULL, &img));
      VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img, &mr);
      VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, 0)};
      VkDeviceMemory im; CK(vkAllocateMemory(dev, &ai, NULL, &im)); CK(vkBindImageMemory(dev, img, im, 0));
      VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D,
                                  i ? dsf : cf, {0},
                                  {i ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      CK(vkCreateImageView(dev, &vi, NULL, &views[i]));
   }

   VkDescriptorSetLayoutBinding b0 = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                      VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b0};
   VkDescriptorSetLayout sl; CK(vkCreateDescriptorSetLayout(dev, &dl, NULL, &sl));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
   VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &sl, 1, &pcr};
   CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
   {
      VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
      VkDescriptorPoolCreateInfo dp = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
      VkDescriptorPool pool; CK(vkCreateDescriptorPool(dev, &dp, NULL, &pool));
      VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, pool, 1, &sl};
      CK(vkAllocateDescriptorSets(dev, &da, &dset));
      VkDescriptorBufferInfo dbi = {vbuf, 0, 4096};
      VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi};
      vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
   }
   VkPipeline cpipe;
   {
      VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
         {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, load("cs.comp.spv"), "main"}, pl};
      CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &cpipe));
   }
   VkQueryPool qpool;
   {
      VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_PIPELINE_STATISTICS, 16,
                                   VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
                                   VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT};
      CK(vkCreateQueryPool(dev, &qpi, NULL, &qpool));
   }

   VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
   VkCommandPool cpool; CK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));
   VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
   VkCommandBuffer cmd; CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkCommandBuffer secs[16]; unsigned nsec = 0;
   cai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY; cai.commandBufferCount = 16;
   CK(vkAllocateCommandBuffers(dev, &cai, secs));
   VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, views[0], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   VkRenderingAttachmentInfo datt = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, views[1],
                                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
   datt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; datt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   VkViewport vp = {0, 0, 256, 256, 0, 1};
   VkRect2D sc = {{0, 0}, {256, 256}};

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
      draw_k = 0; nsec = 0; ctx = 0; rnd_state = 12345;
      unsigned nq = 0;
      VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                     one_time ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0};
      CK(vkBeginCommandBuffer(cmd, &bi));
      vkCmdResetQueryPool(cmd, qpool, 0, 16);
      vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
      vkCmdSetDepthTestEnable(cmd, VK_FALSE); vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
      vkCmdSetStencilTestEnable(cmd, VK_FALSE);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &dset, 0, NULL);
      uint32_t base = 0;
      vkCmdPushConstants(cmd, pl, pcr.stageFlags, 0, 4, &base);
      for (unsigned i = 0; i < ntok; i++) {
         const char *t = toks[i];
         VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {256, 256}}, 1, 0, 1, &att};
         if (!strcmp(t, "B") || !strcmp(t, "BD") || !strcmp(t, "Bx")) {
            ctx = !strcmp(t, "BD");
            if (ctx) ri.pDepthAttachment = ri.pStencilAttachment = &datt;
            if (t[1] == 'x') ri.flags |= VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
            vkCmdBeginRendering(cmd, &ri);
         } else if (!strcmp(t, "E")) {
            vkCmdEndRendering(cmd);
         } else if (!strcmp(t, "dt0") || !strcmp(t, "dt1")) {
            vkCmdSetDepthTestEnable(cmd, t[2] == '1');
         } else if (!strcmp(t, "dw0") || !strcmp(t, "dw1")) {
            vkCmdSetDepthWriteEnable(cmd, t[2] == '1');
         } else if (!strcmp(t, "st0") || !strcmp(t, "st1")) {
            vkCmdSetStencilTestEnable(cmd, t[2] == '1');
         } else if (!strcmp(t, "p")) {
            base = i;
            vkCmdPushConstants(cmd, pl, pcr.stageFlags, 0, 4, &base);
         } else if (!strcmp(t, "Q")) {
            vkCmdBeginQuery(cmd, qpool, nq, 0);
         } else if (!strcmp(t, "q")) {
            vkCmdEndQuery(cmd, qpool, nq++);
         } else if (!strcmp(t, "cb")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, NULL);
         } else if (!strcmp(t, "cd")) {
            vkCmdDispatch(cmd, 2, 1, 1);
         } else if (t[0] == 'X') {
            unsigned nd = atoi(t + 1);
            VkCommandBuffer sec = secs[nsec++];
            VkCommandBufferInheritanceRenderingInfo inh_r = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO, NULL, 0, 0, 1, &cf,
                                                             VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT};
            VkCommandBufferInheritanceInfo inh = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, &inh_r};
            VkCommandBufferBeginInfo sbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                            VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &inh};
            CK(vkBeginCommandBuffer(sec, &sbi));
            vkCmdSetViewport(sec, 0, 1, &vp); vkCmdSetScissor(sec, 0, 1, &sc);
            vkCmdSetDepthTestEnable(sec, VK_FALSE); vkCmdSetDepthWriteEnable(sec, VK_FALSE);
            vkCmdSetStencilTestEnable(sec, VK_FALSE);
            vkCmdBindDescriptorSets(sec, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &dset, 0, NULL);
            vkCmdPushConstants(sec, pl, pcr.stageFlags, 0, 4, &base);
            unsigned save = draw_k;
            for (unsigned d = 0; d < nd; d++) {
               draw_k = 400 + nsec * 16 + d; /* records of secondaries away from the primary's */
               fill(draw_k, 2, 12, -1);
               vkCmdBindPipeline(sec, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(P_VIS));
               drawMeshInd(sec, args, 256 * draw_k, 2, 12);
            }
            draw_k = save;
            CK(vkEndCommandBuffer(sec));
            vkCmdExecuteCommands(cmd, 1, &sec);
         } else if (!strcmp(t, "si")) {
            indirect(cmd, t, P_VIS, 2, 12, -1);
         } else if (!strcmp(t, "s1")) {
            indirect(cmd, t, P_VIS, 1, 12, -1);
         } else if (!strcmp(t, "sc")) {
            indirect(cmd, t, P_VIS, 4, 16, (int)(draw_k % 6));
         } else if (!strcmp(t, "ci")) {
            indirect(cmd, t, P_COL, 2, 12, -1);
         } else if (!strcmp(t, "cc")) {
            indirect(cmd, t, P_COL, 3, 12, 2);
         } else if (!strcmp(t, "ui")) {
            indirect(cmd, t, P_USED, 2, 20, -1);
         } else if (!strcmp(t, "xi")) {
            indirect(cmd, t, P_EXCH, 2, 20, -1);
         } else if (!strcmp(t, "wi")) {
            indirect(cmd, t, P_STORE, 2, 20, -1);
         } else if (!strcmp(t, "sd")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(P_VIS));
            drawMesh(cmd, 2, 1, 1);
         } else if (!strcmp(t, "m")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(P_SMALL));
            drawMesh(cmd, 3, 1, 1);
         } else if (!strcmp(t, "mi")) {
            indirect(cmd, t, P_SMALL, 2, 12, -1);
         } else if (!strcmp(t, "v")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(P_VS));
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

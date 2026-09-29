/* BC250_MESH_TIMER offline IB harness (derived from mesh-autocull/autocull.c and mesh-merge/merge.c).
 * drm-shim ONLY: refuses to run when the noop drm-shim is not preloaded or when the device is not the
 * shim's GFX1013 (run.sh also hides /dev/dri).
 * usage: timer <mesh.spv> <frag.spv> <gx> [option...]
 *   task=<task.spv>   Task shader (the base driver hybrid Task replay)
 *   submits=<n>       record + submit the command buffer n times (default 2)
 *   secondary=1       record the Mesh draws in a secondary command buffer executed by the primary
 * Records in one primary command buffer (tri_vert.spv / tri_frag.spv from the work directory):
 *   Mesh direct (gx,1,1) -> VS draw (3 vertices) -> Mesh indirect (2 records) ->
 *   Mesh indirect count (4 records, count buffer = 2) -> Mesh direct (gx,1,1)
 * then submits (the shim executes nothing) and waits. Prints SUBMIT_OK per submit. */
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

static VkBuffer mkbuf(VkDeviceSize size, VkBufferUsageFlags usage, void **map)
{
   VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, usage, VK_SHARING_MODE_EXCLUSIVE};
   VkBuffer b; CK(vkCreateBuffer(dev, &bi, NULL, &b));
   VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
   VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
                              memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
   VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, NULL, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
   if (map) CK(vkMapMemory(dev, m, 0, size, 0, map));
   return b;
}

static const char *opt(int argc, char **argv, const char *key)
{
   size_t n = strlen(key);
   for (int i = 4; i < argc; i++)
      if (!strncmp(argv[i], key, n) && argv[i][n] == '=')
         return argv[i] + n + 1;
   return "";
}

int main(int argc, char **argv)
{
   if (argc < 4) { fprintf(stderr, "usage: %s mesh.spv frag.spv gx [option...]\n", argv[0]); return 2; }
   const uint32_t gx = atoi(argv[3]);
   const char *task = opt(argc, argv, "task");
   const char *submits_s = opt(argc, argv, "submits");
   const int submits = submits_s[0] ? atoi(submits_s) : 2;
   const int secondary = !strcmp(opt(argc, argv, "secondary"), "1");
   const char *pre = getenv("LD_PRELOAD");
   if (!pre || !strstr(pre, "drm_shim")) {
      printf("REFUSED: drm-shim only (hide /dev/dri and preload libamdgpu_noop_drm_shim.so)\n");
      return 9;
   }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "meshtimer", 1, NULL, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
   VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1; if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || !n) return 3;
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
   if (!strstr(pp.deviceName, "GFX1013")) { printf("REFUSED: device %s is not the shim GFX1013\n", pp.deviceName); return 9; }

   VkPhysicalDeviceMeshShaderFeaturesEXT meshf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   meshf.meshShader = VK_TRUE; meshf.taskShader = VK_TRUE;
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &meshf};
   f12.drawIndirectCount = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
   f13.dynamicRendering = VK_TRUE;
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
   const char *exts[] = {"VK_EXT_mesh_shader"};
   float prio = 1;
   VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, NULL, 1, exts, NULL};
   CK(vkCreateDevice(pd, &dci, NULL, &dev));
   PFN_vkCmdDrawMeshTasksEXT drawMesh = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshInd = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
   PFN_vkCmdDrawMeshTasksIndirectCountEXT drawMeshCnt = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectCountEXT");
   VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

   void *argmap;
   VkBuffer args = mkbuf(4096, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &argmap);
   uint32_t *a = argmap;
   for (unsigned i = 0; i < 4; i++) { a[3 * i] = gx; a[3 * i + 1] = 1; a[3 * i + 2] = 1; }
   const uint32_t count_off = 256; a[count_off / 4] = 2;

   VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM,
                           {256, 256, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage img; CK(vkCreateImage(dev, &ii, NULL, &img));
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, 0)};
   VkDeviceMemory im; CK(vkAllocateMemory(dev, &ai, NULL, &im)); CK(vkBindImageMemory(dev, img, im, 0));
   VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D,
                               VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView rtv; CK(vkCreateImageView(dev, &vi, NULL, &rtv));

   VkPushConstantRange pcr = {VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT, 0, 4};
   VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pcr};
   VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

   VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
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
   const int has_task = task[0] != 0;
   VkPipelineShaderStageCreateInfo mst[3] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_MESH_BIT_EXT, load(argv[1]), "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, load(argv[2]), "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_TASK_BIT_EXT,
       has_task ? load(task) : VK_NULL_HANDLE, "main"}};
   VkGraphicsPipelineCreateInfo gpi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, has_task ? 3 : 2, mst, NULL, NULL, NULL,
                                       &vps, &rs, &ms, NULL, &cbs, &ds, pl};
   VkPipeline meshp;
   VkResult pr = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &meshp);
   printf("PIPELINE_RESULT=%d\n", pr); fflush(stdout);
   if (pr) return 1;

   VkPipelineShaderStageCreateInfo vst[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, load("tri_vert.spv"), "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, load("tri_frag.spv"), "main"}};
   VkPipelineVertexInputStateCreateInfo vis = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ias = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
                                                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   gpi.stageCount = 2; gpi.pStages = vst; gpi.pVertexInputState = &vis; gpi.pInputAssemblyState = &ias;
   VkPipeline vsp; CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &vsp));

   VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
   VkCommandPool cpool; CK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));
   VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
   VkCommandBuffer cmd; CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkCommandBuffer sec = VK_NULL_HANDLE;
   if (secondary) {
      cai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
      CK(vkAllocateCommandBuffers(dev, &cai, &sec));
   }
   VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, rtv, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {256, 256}}, 1, 0, 1, &att};
   VkViewport vp = {0, 0, 256, 256, 0, 1};
   VkRect2D sc = {{0, 0}, {256, 256}};
   uint32_t base = 0;
   for (int s = 0; s < submits; s++) {
      CK(vkResetCommandPool(dev, cpool, 0));
      VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      CK(vkBeginCommandBuffer(cmd, &bi));
      VkCommandBuffer rec = cmd;
      if (secondary) {
         VkCommandBufferInheritanceRenderingInfo inh_r = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO, NULL, 0, 0, 1, &cf,
                                                          VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT};
         VkCommandBufferInheritanceInfo inh = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, &inh_r};
         VkCommandBufferBeginInfo sbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &inh};
         CK(vkBeginCommandBuffer(sec, &sbi));
         ri.flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
         vkCmdBeginRendering(cmd, &ri);
         rec = sec;
      } else {
         vkCmdBeginRendering(cmd, &ri);
      }
      vkCmdSetViewport(rec, 0, 1, &vp); vkCmdSetScissor(rec, 0, 1, &sc);
      fprintf(stderr, "TIMER_MARK mesh_direct\n");
      vkCmdBindPipeline(rec, VK_PIPELINE_BIND_POINT_GRAPHICS, meshp);
      vkCmdPushConstants(rec, pl, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT, 0, 4, &base);
      drawMesh(rec, gx, 1, 1);
      fprintf(stderr, "TIMER_MARK vs_draw\n");
      vkCmdBindPipeline(rec, VK_PIPELINE_BIND_POINT_GRAPHICS, vsp);
      vkCmdDraw(rec, 3, 1, 0, 0);
      fprintf(stderr, "TIMER_MARK mesh_indirect\n");
      vkCmdBindPipeline(rec, VK_PIPELINE_BIND_POINT_GRAPHICS, meshp);
      drawMeshInd(rec, args, 0, 2, 12);
      fprintf(stderr, "TIMER_MARK mesh_indirect_count\n");
      drawMeshCnt(rec, args, 0, args, count_off, 4, 12);
      fprintf(stderr, "TIMER_MARK mesh_direct2\n");
      drawMesh(rec, gx, 1, 1);
      if (secondary) {
         CK(vkEndCommandBuffer(sec));
         vkCmdExecuteCommands(cmd, 1, &sec);
      }
      vkCmdEndRendering(cmd);
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

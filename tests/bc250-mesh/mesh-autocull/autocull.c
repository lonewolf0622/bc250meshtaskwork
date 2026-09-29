/* RADV_BC250_MESH_AUTOCULL offline IB harness (derived from mesh-merge/merge.c). drm-shim ONLY: refuses
 * to run when the noop drm-shim is not preloaded or when the device is not the shim's GFX1013
 * (run.sh also hides /dev/dri).
 * usage: autocull <mesh.spv> <frag.spv> <gx> [option...]
 *   cull=none|back|front|dyn   cull mode (dyn: VK_DYNAMIC_STATE_CULL_MODE + FRONT_FACE)
 *   ff=ccw|cw                  static front face
 *   poly=fill|line|dyn         polygon mode (dyn: VK_DYNAMIC_STATE_POLYGON_MODE_EXT)
 *   discard=1|dyn              rasterizer discard (static on, or dynamic)
 *   cons=1                     conservative rasterization, overestimate
 *   mv=1                       multiview (view mask 0x3)
 *   yflip=1                    negative viewport height (y-inverted)
 *   task=<task.spv>            Task shader
 * Records in one command buffer: Mesh direct (gx,1,1) -> Mesh indirect (2 records) and, for the
 * dynamic cases, the extra draws listed below; then submits once (the shim executes nothing).
 *   cull=dyn:    direct draws with (BACK, CCW), (FRONT, CW), (NONE, CCW)
 *   discard=dyn: direct draws with discard on, then off
 * Every draw is preceded by an "AUTOCULL_MARK <n> <what>" line on stderr. */
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

static int mark_no;
static void mark(const char *what)
{
   fprintf(stderr, "AUTOCULL_MARK %d %s\n", mark_no++, what);
}

int main(int argc, char **argv)
{
   if (argc < 4) { fprintf(stderr, "usage: %s mesh.spv frag.spv gx [option...]\n", argv[0]); return 2; }
   const uint32_t gx = atoi(argv[3]);
   const char *cull = opt(argc, argv, "cull"), *ff = opt(argc, argv, "ff"), *poly = opt(argc, argv, "poly");
   const char *discard = opt(argc, argv, "discard"), *task = opt(argc, argv, "task");
   const int cons = !strcmp(opt(argc, argv, "cons"), "1"), mv = !strcmp(opt(argc, argv, "mv"), "1");
   const int yflip = !strcmp(opt(argc, argv, "yflip"), "1");
   const char *pre = getenv("LD_PRELOAD");
   if (!pre || !strstr(pre, "drm_shim")) {
      printf("REFUSED: drm-shim only (hide /dev/dri and preload libamdgpu_noop_drm_shim.so)\n");
      return 9;
   }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "meshautocull", 1, NULL, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
   VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1; if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || !n) return 3;
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
   if (!strstr(pp.deviceName, "GFX1013")) { printf("REFUSED: device %s is not the shim GFX1013\n", pp.deviceName); return 9; }

   if (mv) {
      VkPhysicalDeviceMeshShaderFeaturesEXT have = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
      VkPhysicalDeviceFeatures2 hf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &have};
      vkGetPhysicalDeviceFeatures2(pd, &hf);
      if (!have.multiviewMeshShader) {
         printf("UNSUPPORTED multiviewMeshShader\n");
         return 0;
      }
   }
   VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
   eds3.extendedDynamicState3PolygonMode = VK_TRUE;
   VkPhysicalDeviceMeshShaderFeaturesEXT meshf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, &eds3};
   meshf.meshShader = VK_TRUE; meshf.taskShader = VK_TRUE; meshf.multiviewMeshShader = mv;
   VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &meshf};
   f11.multiview = mv;
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11};
   f12.drawIndirectCount = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
   f13.dynamicRendering = VK_TRUE;
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
   f2.features.fillModeNonSolid = VK_TRUE; f2.features.multiViewport = VK_TRUE;
   const char *exts[] = {"VK_EXT_mesh_shader", "VK_EXT_extended_dynamic_state3", "VK_EXT_conservative_rasterization"};
   float prio = 1;
   VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, NULL, 3, exts, NULL};
   CK(vkCreateDevice(pd, &dci, NULL, &dev));
   PFN_vkCmdDrawMeshTasksEXT drawMesh = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshInd = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
   PFN_vkCmdSetPolygonModeEXT setPoly = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetPolygonModeEXT");
   VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

   void *argmap;
   VkBuffer args = mkbuf(4096, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &argmap);
   uint32_t *a = argmap; a[0] = gx; a[1] = 1; a[2] = 1; a[3] = 64; a[4] = 1; a[5] = 1;

   const uint32_t layers = mv ? 2 : 1;
   VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM,
                           {256, 256, 1}, 1, layers, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage img; CK(vkCreateImage(dev, &ii, NULL, &img));
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, 0)};
   VkDeviceMemory im; CK(vkAllocateMemory(dev, &ai, NULL, &im)); CK(vkBindImageMemory(dev, img, im, 0));
   VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img,
                               mv ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
                               VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
   VkImageView rtv; CK(vkCreateImageView(dev, &vi, NULL, &rtv));

   VkPushConstantRange pcr = {VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT, 0, 4};
   VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pcr};
   VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

   VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo prci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, mv ? 3 : 0, 1, &cf};
   VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
   VkPipelineRasterizationConservativeStateCreateInfoEXT crs = {
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT, NULL, 0,
      VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT, 0.0f};
   VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, cons ? &crs : NULL};
   rs.lineWidth = 1;
   rs.cullMode = !strcmp(cull, "back") ? VK_CULL_MODE_BACK_BIT : !strcmp(cull, "front") ? VK_CULL_MODE_FRONT_BIT : VK_CULL_MODE_NONE;
   rs.frontFace = !strcmp(ff, "cw") ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
   rs.polygonMode = !strcmp(poly, "line") ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
   rs.rasterizerDiscardEnable = !strcmp(discard, "1");
   VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkPipelineColorBlendAttachmentState cba = {0}; cba.colorWriteMask = 0xf;
   VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &cba};
   VkDynamicState dyns[8] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   uint32_t ndyn = 2;
   if (!strcmp(cull, "dyn")) { dyns[ndyn++] = VK_DYNAMIC_STATE_CULL_MODE; dyns[ndyn++] = VK_DYNAMIC_STATE_FRONT_FACE; }
   if (!strcmp(poly, "dyn")) dyns[ndyn++] = VK_DYNAMIC_STATE_POLYGON_MODE_EXT;
   if (!strcmp(discard, "dyn")) dyns[ndyn++] = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE;
   VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, ndyn, dyns};
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

   VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
   VkCommandPool cpool; CK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));
   VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
   VkCommandBuffer cmd; CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, rtv, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {256, 256}}, 1, mv ? 3 : 0, 1, &att};
   VkViewport vp = {0, yflip ? 256 : 0, 256, yflip ? -256 : 256, 0, 1};
   VkRect2D sc = {{0, 0}, {256, 256}};
   uint32_t base = 0;
   VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   CK(vkBeginCommandBuffer(cmd, &bi));
   vkCmdBeginRendering(cmd, &ri);
   vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshp);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT, 0, 4, &base);
   if (!strcmp(cull, "dyn")) { vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE); vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE); }
   if (!strcmp(poly, "dyn")) setPoly(cmd, VK_POLYGON_MODE_FILL);
   if (!strcmp(discard, "dyn")) vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
   mark("mesh_direct");
   drawMesh(cmd, gx, 1, 1);
   mark("mesh_indirect");
   drawMeshInd(cmd, args, 0, 2, 12);
   if (!strcmp(cull, "dyn")) {
      static const struct { VkCullModeFlags cm; VkFrontFace ff; const char *what; } d[] = {
         {VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, "dyn_back_ccw"},
         {VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_CLOCKWISE, "dyn_front_cw"},
         {VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, "dyn_none_ccw"}};
      for (unsigned i = 0; i < 3; i++) {
         vkCmdSetCullMode(cmd, d[i].cm); vkCmdSetFrontFace(cmd, d[i].ff);
         mark(d[i].what);
         drawMesh(cmd, gx, 1, 1);
      }
   }
   if (!strcmp(discard, "dyn")) {
      vkCmdSetRasterizerDiscardEnable(cmd, VK_TRUE);
      mark("dyn_discard_on");
      drawMesh(cmd, gx, 1, 1);
      vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
      mark("dyn_discard_off");
      drawMesh(cmd, gx, 1, 1);
   }
   vkCmdEndRendering(cmd);
   CK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd};
   CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
   CK(vkQueueWaitIdle(q));
   printf("SUBMIT_OK\n");
   return 0;
}

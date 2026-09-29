/* BC250 Mesh pipeline probe (drm-shim only, copy of piece-ceiling/pipe.c): pipe MESH.spv FRAG.spv [TASK.spv|-] [attachments]
 * Creates one Mesh (+Task) + fragment graphics pipeline, records a direct, an indirect (2 records) and an
 * indirect-count Mesh draw, submits them to the noop drm-shim queue and waits. Prints PIPELINE_RESULT and
 * SUBMIT_OK. Refuses any device that is not the shim's GFX1013. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { VkResult r_ = (x); if (r_) { printf("FAIL %s = %d\n", #x, r_); return 1; } } while (0)
static VkDevice dev;
static VkShaderModule load(const char *p)
{
   FILE *f = fopen(p, "rb");
   if (!f) { perror(p); exit(2); }
   fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
   void *b = malloc(n);
   if (fread(b, 1, n, f) != (size_t)n) exit(2);
   fclose(f);
   VkShaderModuleCreateInfo c = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, n, b};
   VkShaderModule m;
   if (vkCreateShaderModule(dev, &c, 0, &m)) exit(2);
   return m;
}
static uint32_t memtype(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
   return __builtin_ctz(bits);
}
int main(int argc, char **argv)
{
   if (argc < 3) { fprintf(stderr, "usage: pipe MESH.spv FRAG.spv [TASK.spv|-] [attachments]\n"); return 2; }
   const char *task = argc > 3 && strcmp(argv[3], "-") ? argv[3] : NULL;
   unsigned rts = argc > 4 ? (unsigned)atoi(argv[4]) : 1;
   if (rts < 1 || rts > 8) rts = 1;
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance in;
   if (vkCreateInstance(&ici, 0, &in)) { puts("NO_INSTANCE"); return 3; }
   uint32_t n = 1; VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(in, &n, &pd) < 0 || !n) { puts("NO_DEVICE"); return 3; }
   VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(pd, &pr);
   if (!strstr(pr.deviceName, "GFX1013")) { puts("NOT_SHIM_GFX1013"); return 3; }
   float q = 1;
   VkDeviceQueueCreateInfo qc = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &q};
   VkPhysicalDeviceMeshShaderFeaturesEXT en = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, .meshShader = 1, .taskShader = 1};
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &en, .drawIndirectCount = 1};
   VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12, .shaderDrawParameters = 1};
   VkPhysicalDeviceFeatures fe = {.multiDrawIndirect = 1};
   const char *ext = "VK_EXT_mesh_shader";
   VkDeviceCreateInfo dc = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f11, 0, 1, &qc, 0, 0, 1, &ext, &fe};
   CK(vkCreateDevice(pd, &dc, 0, &dev));
   VkQueue queue; vkGetDeviceQueue(dev, 0, 0, &queue);
   VkPushConstantRange pc = {VK_SHADER_STAGE_ALL, 0, 16};
   VkPipelineLayoutCreateInfo lc = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 0, 0, 1, &pc};
   VkPipelineLayout lay; CK(vkCreatePipelineLayout(dev, &lc, 0, &lay));
   VkAttachmentDescription at[8]; VkAttachmentReference ar[8]; VkPipelineColorBlendAttachmentState ba[8];
   for (unsigned i = 0; i < 8; i++) {
      at[i] = (VkAttachmentDescription){0, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
                                       VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, 0, VK_IMAGE_LAYOUT_GENERAL};
      ar[i] = (VkAttachmentReference){i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      ba[i] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = 15};
   }
   VkSubpassDescription sp = {0, 0, 0, 0, rts, ar};
   VkRenderPassCreateInfo rc = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, 0, 0, rts, at, 1, &sp};
   VkRenderPass rp; CK(vkCreateRenderPass(dev, &rc, 0, &rp));
   VkPipelineShaderStageCreateInfo st[3] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_MESH_BIT_EXT, load(argv[1]), "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_FRAGMENT_BIT, load(argv[2]), "main"}};
   if (task)
      st[2] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_TASK_BIT_EXT, load(task), "main"};
   VkViewport vp = {0, 0, 64, 64, 0, 1}; VkRect2D sc = {{0, 0}, {64, 64}};
   VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, 0, 0, 1, &vp, 1, &sc};
   VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = 1};
   VkPipelineColorBlendStateCreateInfo bs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = rts, .pAttachments = ba};
   VkGraphicsPipelineCreateInfo gp = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = task ? 3 : 2, .pStages = st,
      .pViewportState = &vs, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &bs, .layout = lay, .renderPass = rp};
   VkPipeline p; VkResult r = vkCreateGraphicsPipelines(dev, 0, 1, &gp, 0, &p);
   printf("PIPELINE_RESULT=%d\n", r); fflush(stdout);
   if (r) return 1;

   /* Indirect records {x,y,z} at 0 and 16 (stride 16), count (=2) at 64. */
   VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, 256, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
   VkBuffer ib; CK(vkCreateBuffer(dev, &bci, 0, &ib));
   VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, ib, &mr);
   VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size,
      memtype(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   VkDeviceMemory im; CK(vkAllocateMemory(dev, &ma, 0, &im)); CK(vkBindBufferMemory(dev, ib, im, 0));
   uint32_t *rec; CK(vkMapMemory(dev, im, 0, 256, 0, (void **)&rec));
   memset(rec, 0, 256);
   rec[0] = 3; rec[1] = 1; rec[2] = 1; rec[4] = 2; rec[5] = 2; rec[6] = 1; rec[16] = 2;

   VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, 0, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {64, 64, 1}, 1, 1, 1, 0,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
   VkImage img; CK(vkCreateImage(dev, &ic, 0, &img));
   vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo ma2 = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size, memtype(pd, mr.memoryTypeBits, 0)};
   VkDeviceMemory mem; CK(vkAllocateMemory(dev, &ma2, 0, &mem)); CK(vkBindImageMemory(dev, img, mem, 0));
   VkImageViewCreateInfo iv = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, 0, 0, img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {1, 0, 1, 0, 1}};
   VkImageView view[8];
   for (unsigned i = 0; i < rts; i++) CK(vkCreateImageView(dev, &iv, 0, &view[i]));
   VkFramebufferCreateInfo fc = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, 0, 0, rp, rts, view, 64, 64, 1};
   VkFramebuffer fb; CK(vkCreateFramebuffer(dev, &fc, 0, &fb));

   VkCommandPoolCreateInfo cp = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool; CK(vkCreateCommandPool(dev, &cp, 0, &pool));
   VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, pool, 0, 1};
   VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &ca, &cb));
   VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CK(vkBeginCommandBuffer(cb, &bi));
   VkClearValue cv[8] = {0};
   VkRenderPassBeginInfo rb = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, 0, rp, fb, sc, rts, cv};
   vkCmdBeginRenderPass(cb, &rb, 0);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   uint32_t base = 5;
   vkCmdPushConstants(cb, lay, VK_SHADER_STAGE_ALL, 0, 4, &base);
   PFN_vkCmdDrawMeshTasksEXT drawMesh = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   PFN_vkCmdDrawMeshTasksIndirectEXT drawInd = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
   PFN_vkCmdDrawMeshTasksIndirectCountEXT drawCnt = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectCountEXT");
   drawMesh(cb, 7, 1, 1);
   drawInd(cb, ib, 0, 2, 16);
   drawCnt(cb, ib, 0, ib, 64, 2, 16);
   vkCmdEndRenderPass(cb);
   CK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cb};
   CK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CK(vkQueueWaitIdle(queue));
   printf("SUBMIT_OK\n");
   return 0;
}

/* M1 regression (Tree C port): one direct Task draw in a primary command buffer,
 * optionally SIMULTANEOUS_USE, submitted so RADV_DEBUG=dumpibs shows whether a
 * VS/PS partial flush precedes the first Task producer dispatch.
 * Usage: simul 0|1 (1 = SIMULTANEOUS_USE). Run only under the amdgpu noop drm-shim. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_) { printf("FAIL %s = %d\n", #x, r_); exit(1); } } while (0)

static VkShaderModule load(VkDevice d, const char *p)
{
   FILE *f = fopen(p, "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
   void *b = malloc(n); fread(b, 1, n, f); fclose(f);
   VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, n, b};
   VkShaderModule m; CK(vkCreateShaderModule(d, &ci, 0, &m)); return m;
}

int main(int argc, char **argv)
{
   const int simultaneous = argc > 1 && argv[1][0] == '1';
   VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "g3", 0, 0, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai};
   VkInstance inst; CK(vkCreateInstance(&ici, 0, &inst));
   uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &n, &pd);
   if (!n) { printf("no device\n"); return 1; }
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
   printf("device=%s maxDrawIndirectCount=%u\n", pp.deviceName, pp.limits.maxDrawIndirectCount);
   if (!strstr(pp.deviceName, "GFX1013")) { printf("not shim gfx1013\n"); return 1; }
   VkPhysicalDeviceMeshShaderFeaturesEXT mf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &mf};
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   printf("taskShader=%u meshShader=%u drawIndirectCount=%u multiDrawIndirect=%u\n", mf.taskShader, mf.meshShader,
          f12.drawIndirectCount, f2.features.multiDrawIndirect);
   if (!mf.taskShader) return 1;
   VkPhysicalDeviceMeshShaderFeaturesEXT emf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, 0, 1, 1};
   VkPhysicalDeviceVulkan13Features e13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &emf};
   e13.dynamicRendering = 1;
   VkPhysicalDeviceVulkan12Features e12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &e13};
   e12.drawIndirectCount = 1; e12.bufferDeviceAddress = 1;
   VkPhysicalDeviceFeatures2 e2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &e12};
   e2.features.multiDrawIndirect = 1;
   float pr = 1; VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, 0, 1, &pr};
   const char *ext[] = {"VK_EXT_mesh_shader"};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &e2, 0, 1, &qci, 0, 0, 1, ext};
   VkDevice d; CK(vkCreateDevice(pd, &dci, 0, &d));
   PFN_vkCmdDrawMeshTasksIndirectEXT di = (void *)vkGetDeviceProcAddr(d, "vkCmdDrawMeshTasksIndirectEXT");
   PFN_vkCmdDrawMeshTasksIndirectCountEXT dic = (void *)vkGetDeviceProcAddr(d, "vkCmdDrawMeshTasksIndirectCountEXT");

   VkPipelineShaderStageCreateInfo st[3] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_TASK_BIT_EXT, load(d, "t.spv"), "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_MESH_BIT_EXT, load(d, "m.spv"), "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_FRAGMENT_BIT, load(d, "f.spv"), "main"}};
   VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout pl; CK(vkCreatePipelineLayout(d, &lci, 0, &pl));
   VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, 0, 0, 0, &fmt};
   VkViewport vp = {0, 0, 64, 64, 0, 1}; VkRect2D sc = {{0, 0}, {64, 64}};
   VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, 0, 0, 1, &vp, 1, &sc};
   VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
   rs.lineWidth = 1;
   VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   ms.rasterizationSamples = 1;
   VkPipelineColorBlendAttachmentState ba = {0}; ba.colorWriteMask = 0xf;
   VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
   cb.attachmentCount = 0; cb.pAttachments = &ba;
   VkGraphicsPipelineCreateInfo gci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci};
   gci.stageCount = 2; gci.pStages = st; gci.pViewportState = &vs; gci.pRasterizationState = &rs;
   gci.pMultisampleState = &ms; gci.pColorBlendState = &cb; gci.layout = pl;
   VkPipeline pipe; VkResult pr_ = vkCreateGraphicsPipelines(d, 0, 1, &gci, 0, &pipe);
   printf("pipeline=%d\n", pr_);
   if (pr_) return 1;

   VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
   VkCommandPool cp; CK(vkCreateCommandPool(d, &cpci, 0, &cp));
   PFN_vkCmdDrawMeshTasksEXT dm = (void *)vkGetDeviceProcAddr(d, "vkCmdDrawMeshTasksEXT");
   VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, cp, 0, 1};
   VkCommandBuffer c; CK(vkAllocateCommandBuffers(d, &cai, &c));
   VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   if (simultaneous)
      bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
   CK(vkBeginCommandBuffer(c, &bi));
   VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO}; ri.renderArea.extent.width = 64; ri.renderArea.extent.height = 64; ri.layerCount = 1;
   vkCmdBeginRendering(c, &ri);
   vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   dm(c, 1, 1, 1);
   vkCmdEndRendering(c);
   CK(vkEndCommandBuffer(c));
   VkQueue q; vkGetDeviceQueue(d, 0, 0, &q);
   VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &c};
   CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
   CK(vkQueueWaitIdle(q));
   printf("SUBMITTED simultaneous=%d\n", simultaneous);
   return 0;
}

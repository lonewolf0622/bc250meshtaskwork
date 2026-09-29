/* BC250_MESH_TIMER async-compute harness (drm-shim ONLY; refuses to run otherwise).
 * usage: ace <ace.comp.spv> [submits=<n>]
 * One graphics queue and two queues of the compute family. Per iteration s (value base v = 4 s):
 *   compute q0: dispatch                     signal T = v+1
 *   graphics  : vkCmdFillBuffer    wait T = v+1, signal T = v+2
 *   compute q1: dispatch           wait T = v+2, signal T = v+3 and binary B
 *   graphics  : vkCmdFillBuffer    wait B
 *   graphics  : no command buffer  signal T = v+4   (a semaphore-only submission; the drm-shim cannot
 *                                                    export the sync files a CB-less wait needs)
 * then vkDeviceWaitIdle. Prints SUBMIT_OK per iteration, DONE at the end. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d @%d\n", #x, r_, __LINE__); fflush(stdout); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice pd;

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
   exit(2);
}

int main(int argc, char **argv)
{
   if (argc < 2) { fprintf(stderr, "usage: %s ace.comp.spv [submits=n]\n", argv[0]); return 2; }
   int submits = 2;
   for (int i = 2; i < argc; i++)
      if (!strncmp(argv[i], "submits=", 8)) submits = atoi(argv[i] + 8);
   const char *pre = getenv("LD_PRELOAD");
   if (!pre || !strstr(pre, "drm_shim")) {
      printf("REFUSED: drm-shim only (hide /dev/dri and preload libamdgpu_noop_drm_shim.so)\n");
      return 9;
   }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "acetimer", 1, NULL, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
   VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1; if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || !n) return 3;
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
   if (!strstr(pp.deviceName, "GFX1013")) { printf("REFUSED: device %s is not the shim GFX1013\n", pp.deviceName); return 9; }

   uint32_t nf = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, NULL);
   VkQueueFamilyProperties qfp[8]; if (nf > 8) nf = 8; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, qfp);
   uint32_t gfam = UINT32_MAX, cfam = UINT32_MAX;
   for (uint32_t i = 0; i < nf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfam == UINT32_MAX) gfam = i;
      if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && cfam == UINT32_MAX) cfam = i;
   }
   if (gfam == UINT32_MAX || cfam == UINT32_MAX) { printf("FAIL no compute family\n"); return 1; }
   const uint32_t ncq = qfp[cfam].queueCount >= 2 ? 2 : 1;
   printf("FAMILIES gfx=%u compute=%u compute_queues=%u\n", gfam, cfam, qfp[cfam].queueCount);

   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
   f12.timelineSemaphore = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
   f13.synchronization2 = VK_TRUE;
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
   float prio[2] = {1, 1};
   VkDeviceQueueCreateInfo qci[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, gfam, 1, prio},
                                     {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, cfam, ncq, prio}};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 2, qci, 0, NULL, 0, NULL, NULL};
   CK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue gq, cq[2];
   vkGetDeviceQueue(dev, gfam, 0, &gq);
   vkGetDeviceQueue(dev, cfam, 0, &cq[0]);
   vkGetDeviceQueue(dev, cfam, ncq - 1, &cq[1]);

   /* storage buffer + descriptor */
   VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_CONCURRENT, 2,
                             (uint32_t[]){gfam, cfam}};
   VkBuffer buf; CK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
   VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, 0)};
   VkDeviceMemory mem; CK(vkAllocateMemory(dev, &mai, NULL, &mem)); CK(vkBindBufferMemory(dev, buf, mem, 0));
   VkDescriptorSetLayoutBinding b0 = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
   VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b0};
   VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
   VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
   VkDescriptorSet set; CK(vkAllocateDescriptorSets(dev, &dsai, &set));
   VkDescriptorBufferInfo dbi = {buf, 0, VK_WHOLE_SIZE};
   VkWriteDescriptorSet wds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi};
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   /* compute pipeline */
   FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 2; }
   fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc(sz); if (fread(code, 1, sz, f) != (size_t)sz) return 2; fclose(f);
   VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sz, code};
   VkShaderModule sm; CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
   VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr};
   VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                       {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"}, pl};
   VkPipeline cp; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &cp));

   /* semaphores */
   VkSemaphoreTypeCreateInfo stci = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
   VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &stci};
   VkSemaphore tl; CK(vkCreateSemaphore(dev, &sci, NULL, &tl));
   VkSemaphoreCreateInfo bsci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   VkSemaphore bin; CK(vkCreateSemaphore(dev, &bsci, NULL, &bin));

   /* command buffers */
   VkCommandPoolCreateInfo gpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, gfam};
   VkCommandPoolCreateInfo ccpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, cfam};
   VkCommandPool gpool, cpool; CK(vkCreateCommandPool(dev, &gpci, NULL, &gpool)); CK(vkCreateCommandPool(dev, &ccpci, NULL, &cpool));
   VkCommandBufferAllocateInfo gai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, gpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
   VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
   VkCommandBuffer gcmd, ccmd[2]; CK(vkAllocateCommandBuffers(dev, &gai, &gcmd)); CK(vkAllocateCommandBuffers(dev, &cai, ccmd));

   for (int s = 0; s < submits; s++) {
      const uint64_t v = 4ull * s;
      CK(vkResetCommandPool(dev, gpool, 0)); CK(vkResetCommandPool(dev, cpool, 0));
      VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      for (int c = 0; c < 2; c++) {
         CK(vkBeginCommandBuffer(ccmd[c], &bi));
         fprintf(stderr, "TIMER_MARK ace_dispatch %d\n", c);
         vkCmdBindPipeline(ccmd[c], VK_PIPELINE_BIND_POINT_COMPUTE, cp);
         vkCmdBindDescriptorSets(ccmd[c], VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
         uint32_t base = 100 * c;
         vkCmdPushConstants(ccmd[c], pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &base);
         vkCmdDispatch(ccmd[c], 16, 1, 1);
         CK(vkEndCommandBuffer(ccmd[c]));
      }
      CK(vkBeginCommandBuffer(gcmd, &bi));
      fprintf(stderr, "TIMER_MARK gfx_fill\n");
      vkCmdFillBuffer(gcmd, buf, 0, 4096, 0x12345678);
      CK(vkEndCommandBuffer(gcmd));

      VkCommandBufferSubmitInfo c0 = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, ccmd[0]};
      VkCommandBufferSubmitInfo c1 = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, ccmd[1]};
      VkCommandBufferSubmitInfo g0 = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, gcmd};
      VkSemaphoreSubmitInfo sig1 = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, tl, v + 1, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
      VkSemaphoreSubmitInfo wait1 = sig1;
      VkSemaphoreSubmitInfo sig2 = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, tl, v + 2, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
      VkSemaphoreSubmitInfo wait2 = sig2;
      VkSemaphoreSubmitInfo sig3[2] = {{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, tl, v + 3, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
                                       {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, bin, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
      VkSemaphoreSubmitInfo waitb = sig3[1];
      VkSemaphoreSubmitInfo sig4 = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, tl, v + 4, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
      VkSubmitInfo2 s1 = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 0, NULL, 1, &c0, 1, &sig1};
      VkSubmitInfo2 s2 = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 1, &wait1, 1, &g0, 1, &sig2};
      VkSubmitInfo2 s3 = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 1, &wait2, 1, &c1, 2, sig3};
      VkSubmitInfo2 s4 = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 1, &waitb, 1, &g0, 0, NULL};
      VkSubmitInfo2 s5 = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 0, NULL, 0, NULL, 1, &sig4};
      CK(vkQueueSubmit2(cq[0], 1, &s1, VK_NULL_HANDLE));
      CK(vkQueueSubmit2(gq, 1, &s2, VK_NULL_HANDLE));
      CK(vkQueueSubmit2(cq[1], 1, &s3, VK_NULL_HANDLE));
      CK(vkQueueSubmit2(gq, 1, &s4, VK_NULL_HANDLE));
      CK(vkQueueSubmit2(gq, 1, &s5, VK_NULL_HANDLE));
      CK(vkDeviceWaitIdle(dev));
      printf("SUBMIT_OK %d\n", s); fflush(stdout);
   }
   vkDestroyCommandPool(dev, gpool, NULL); vkDestroyCommandPool(dev, cpool, NULL);
   vkDestroySemaphore(dev, tl, NULL); vkDestroySemaphore(dev, bin, NULL);
   vkDestroyPipeline(dev, cp, NULL); vkDestroyPipelineLayout(dev, pl, NULL); vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL); vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyBuffer(dev, buf, NULL); vkFreeMemory(dev, mem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("DONE\n");
   return 0;
}

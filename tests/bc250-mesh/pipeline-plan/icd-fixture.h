/* SPDX-License-Identifier: MIT
 * Direct ICD fixture: probe hidden entrypoints without loader extension
 * trampolines or advertising unproven features to applications. */
#include <dlfcn.h>
static PFN_vkGetInstanceProcAddr fixture_get_proc;
static VkInstance fixture_instance;
static VkResult
fixture_create_instance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *alloc, VkInstance *instance)
{
   const char *path = getenv("TEST_ICD_LIBRARY");
   if (!path) return VK_ERROR_INITIALIZATION_FAILED;
   void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!library) { fputs(dlerror(), stderr); return VK_ERROR_INITIALIZATION_FAILED; }
   fixture_get_proc = (void *)dlsym(library, "vk_icdGetInstanceProcAddr");
   if (!fixture_get_proc) return VK_ERROR_INITIALIZATION_FAILED;
   PFN_vkCreateInstance create = (void *)fixture_get_proc(NULL, "vkCreateInstance");
   VkResult result = create(info, alloc, instance);
   if (result == VK_SUCCESS) fixture_instance = *instance;
   return result;
}
#define vkCreateInstance fixture_create_instance
#define vkGetInstanceProcAddr fixture_get_proc
#define vkGetDeviceProcAddr(device, name) fixture_get_proc(fixture_instance, name)
#define vkAllocateCommandBuffers ((PFN_vkAllocateCommandBuffers)fixture_get_proc(fixture_instance, "vkAllocateCommandBuffers"))
#define vkAllocateDescriptorSets ((PFN_vkAllocateDescriptorSets)fixture_get_proc(fixture_instance, "vkAllocateDescriptorSets"))
#define vkAllocateMemory ((PFN_vkAllocateMemory)fixture_get_proc(fixture_instance, "vkAllocateMemory"))
#define vkBeginCommandBuffer ((PFN_vkBeginCommandBuffer)fixture_get_proc(fixture_instance, "vkBeginCommandBuffer"))
#define vkBindBufferMemory ((PFN_vkBindBufferMemory)fixture_get_proc(fixture_instance, "vkBindBufferMemory"))
#define vkBindImageMemory ((PFN_vkBindImageMemory)fixture_get_proc(fixture_instance, "vkBindImageMemory"))
#define vkCmdBeginRenderPass ((PFN_vkCmdBeginRenderPass)fixture_get_proc(fixture_instance, "vkCmdBeginRenderPass"))
#define vkCmdBindDescriptorSets ((PFN_vkCmdBindDescriptorSets)fixture_get_proc(fixture_instance, "vkCmdBindDescriptorSets"))
#define vkCmdBindPipeline ((PFN_vkCmdBindPipeline)fixture_get_proc(fixture_instance, "vkCmdBindPipeline"))
#define vkCmdEndRenderPass ((PFN_vkCmdEndRenderPass)fixture_get_proc(fixture_instance, "vkCmdEndRenderPass"))
#define vkCmdPushConstants ((PFN_vkCmdPushConstants)fixture_get_proc(fixture_instance, "vkCmdPushConstants"))
#define vkCreateBuffer ((PFN_vkCreateBuffer)fixture_get_proc(fixture_instance, "vkCreateBuffer"))
#define vkCreateCommandPool ((PFN_vkCreateCommandPool)fixture_get_proc(fixture_instance, "vkCreateCommandPool"))
#define vkCreateDescriptorPool ((PFN_vkCreateDescriptorPool)fixture_get_proc(fixture_instance, "vkCreateDescriptorPool"))
#define vkCreateDescriptorSetLayout ((PFN_vkCreateDescriptorSetLayout)fixture_get_proc(fixture_instance, "vkCreateDescriptorSetLayout"))
#define vkCreateDevice ((PFN_vkCreateDevice)fixture_get_proc(fixture_instance, "vkCreateDevice"))
#define vkCreateFramebuffer ((PFN_vkCreateFramebuffer)fixture_get_proc(fixture_instance, "vkCreateFramebuffer"))
#define vkCreateGraphicsPipelines ((PFN_vkCreateGraphicsPipelines)fixture_get_proc(fixture_instance, "vkCreateGraphicsPipelines"))
#define vkCreateImage ((PFN_vkCreateImage)fixture_get_proc(fixture_instance, "vkCreateImage"))
#define vkCreateImageView ((PFN_vkCreateImageView)fixture_get_proc(fixture_instance, "vkCreateImageView"))
#define vkCreatePipelineLayout ((PFN_vkCreatePipelineLayout)fixture_get_proc(fixture_instance, "vkCreatePipelineLayout"))
#define vkCreateRenderPass ((PFN_vkCreateRenderPass)fixture_get_proc(fixture_instance, "vkCreateRenderPass"))
#define vkCreateShaderModule ((PFN_vkCreateShaderModule)fixture_get_proc(fixture_instance, "vkCreateShaderModule"))
#define vkDestroyPipeline ((PFN_vkDestroyPipeline)fixture_get_proc(fixture_instance, "vkDestroyPipeline"))
#define vkEndCommandBuffer ((PFN_vkEndCommandBuffer)fixture_get_proc(fixture_instance, "vkEndCommandBuffer"))
#define vkEnumeratePhysicalDevices ((PFN_vkEnumeratePhysicalDevices)fixture_get_proc(fixture_instance, "vkEnumeratePhysicalDevices"))
#define vkGetBufferMemoryRequirements ((PFN_vkGetBufferMemoryRequirements)fixture_get_proc(fixture_instance, "vkGetBufferMemoryRequirements"))
#define vkGetDeviceQueue ((PFN_vkGetDeviceQueue)fixture_get_proc(fixture_instance, "vkGetDeviceQueue"))
#define vkGetImageMemoryRequirements ((PFN_vkGetImageMemoryRequirements)fixture_get_proc(fixture_instance, "vkGetImageMemoryRequirements"))
#define vkGetPhysicalDeviceMemoryProperties ((PFN_vkGetPhysicalDeviceMemoryProperties)fixture_get_proc(fixture_instance, "vkGetPhysicalDeviceMemoryProperties"))
#define vkGetPhysicalDeviceProperties ((PFN_vkGetPhysicalDeviceProperties)fixture_get_proc(fixture_instance, "vkGetPhysicalDeviceProperties"))
#define vkMapMemory ((PFN_vkMapMemory)fixture_get_proc(fixture_instance, "vkMapMemory"))
#define vkQueueSubmit ((PFN_vkQueueSubmit)fixture_get_proc(fixture_instance, "vkQueueSubmit"))
#define vkQueueWaitIdle ((PFN_vkQueueWaitIdle)fixture_get_proc(fixture_instance, "vkQueueWaitIdle"))
#define vkUpdateDescriptorSets ((PFN_vkUpdateDescriptorSets)fixture_get_proc(fixture_instance, "vkUpdateDescriptorSets"))

/* SPDX-License-Identifier: MIT
 * Exercise serialized application-cache restoration after the original
 * graphics and private compute pipelines have been destroyed. The inherited
 * harness records direct, indirect and indirect-count Mesh/Task draws. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>

static VkResult
cache_roundtrip(VkDevice device, VkPipelineCache unused_cache, uint32_t count,
                const VkGraphicsPipelineCreateInfo *infos,
                const VkAllocationCallbacks *allocator, VkPipeline *pipelines)
{
   VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
   VkPipelineCache cache;
   VkResult result = vkCreatePipelineCache(device, &ci, NULL, &cache);
   if (result != VK_SUCCESS)
      return result;
   result = vkCreateGraphicsPipelines(device, cache, count, infos, allocator, pipelines);
   if (result != VK_SUCCESS)
      return result;

   size_t size = 0;
   result = vkGetPipelineCacheData(device, cache, &size, NULL);
   if (result != VK_SUCCESS)
      return result;
   void *data = malloc(size);
   if (!data)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   result = vkGetPipelineCacheData(device, cache, &size, data);
   if (result != VK_SUCCESS)
      return result;
   for (uint32_t i = 0; i < count; i++)
      vkDestroyPipeline(device, pipelines[i], allocator);
   vkDestroyPipelineCache(device, cache, NULL);

   ci.initialDataSize = size;
   ci.pInitialData = data;
   result = vkCreatePipelineCache(device, &ci, NULL, &cache);
   free(data);
   if (result != VK_SUCCESS)
      return result;
   VkPipelineCreationFeedback feedback = {0};
   VkPipelineCreationFeedbackCreateInfo feedback_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
      .pNext = infos->pNext,
      .pPipelineCreationFeedback = &feedback,
   };
   VkGraphicsPipelineCreateInfo info = *infos;
   info.pNext = &feedback_info;
   if (count != 1)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   result = vkCreateGraphicsPipelines(device, cache, 1, &info, allocator, pipelines);
   vkDestroyPipelineCache(device, cache, NULL);
   printf("CACHE_ROUNDTRIP bytes=%zu hit=%u\n", size,
          !!(feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT));
   if (result == VK_SUCCESS &&
       !(feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT))
      return VK_ERROR_UNKNOWN;
   return result;
}

#define vkCreateGraphicsPipelines cache_roundtrip
#include "../safe-direct/pipe.c"

/* SPDX-License-Identifier: MIT
 * Offline fixture for binary code paths while feature advertisement remains
 * disabled. Instance entrypoints are used, as in RADV's internal tests. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "icd-fixture.h"

static VkInstance binary_instance;
static VkResult
binary_create_instance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *alloc, VkInstance *instance)
{
   VkResult result = vkCreateInstance(info, alloc, instance);
   if (result == VK_SUCCESS)
      binary_instance = *instance;
   return result;
}

static VkResult
binary_roundtrip(VkDevice device, VkPipelineCache cache, uint32_t count,
                 const VkGraphicsPipelineCreateInfo *infos,
                 const VkAllocationCallbacks *alloc, VkPipeline *pipelines)
{
   if (count != 1)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   PFN_vkCreatePipelineBinariesKHR create = (void *)vkGetInstanceProcAddr(binary_instance, "vkCreatePipelineBinariesKHR");
   PFN_vkGetPipelineBinaryDataKHR get = (void *)vkGetInstanceProcAddr(binary_instance, "vkGetPipelineBinaryDataKHR");
   PFN_vkDestroyPipelineBinaryKHR destroy = (void *)vkGetInstanceProcAddr(binary_instance, "vkDestroyPipelineBinaryKHR");
   if (!create || !get || !destroy)
      return VK_ERROR_EXTENSION_NOT_PRESENT;

   VkPipelineCreateFlags2CreateInfo flags = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
      .pNext = infos->pNext, .flags = infos->flags | VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR};
   VkGraphicsPipelineCreateInfo info = *infos;
   info.pNext = &flags;
   VkResult result = vkCreateGraphicsPipelines(device, cache, 1, &info, alloc, pipelines);
   if (result != VK_SUCCESS)
      return result;
   VkPipelineBinaryCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
                                      .pipeline = *pipelines};
   VkPipelineBinaryHandlesInfoKHR handles = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
   result = create(device, &ci, NULL, &handles);
   if (result != VK_SUCCESS || handles.pipelineBinaryCount > 8)
      return result != VK_SUCCESS ? result : VK_ERROR_UNKNOWN;
   VkPipelineBinaryKHR binaries[8];
   VkPipelineBinaryKeyKHR keys[8];
   VkPipelineBinaryDataKHR data[8];
   handles.pPipelineBinaries = binaries;
   result = create(device, &ci, NULL, &handles);
   if (result != VK_SUCCESS)
      return result;
   const unsigned n = handles.pipelineBinaryCount;
   for (unsigned i = 0; i < n; i++) {
      keys[i] = (VkPipelineBinaryKeyKHR){.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
      VkPipelineBinaryDataInfoKHR di = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
                                       .pipelineBinary = binaries[i]};
      data[i].dataSize = 0;
      result = get(device, &di, &keys[i], &data[i].dataSize, NULL);
      if (result != VK_SUCCESS)
         return result;
      data[i].pData = malloc(data[i].dataSize);
      if (!data[i].pData)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      result = get(device, &di, &keys[i], &data[i].dataSize, data[i].pData);
      if (result != VK_SUCCESS)
         return result;
      destroy(device, binaries[i], NULL);
   }
   vkDestroyPipeline(device, *pipelines, alloc);
   VkPipelineBinaryKeysAndDataKHR kd = {.binaryCount = n, .pPipelineBinaryKeys = keys,
                                       .pPipelineBinaryData = data};
   ci.pipeline = VK_NULL_HANDLE;
   ci.pKeysAndDataInfo = &kd;
   handles.pipelineBinaryCount = n;
   result = create(device, &ci, NULL, &handles);
   if (result != VK_SUCCESS)
      return result;
   VkPipelineBinaryInfoKHR bi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR,
                                .pNext = infos->pNext, .binaryCount = n, .pPipelineBinaries = binaries};
   info = *infos;
   info.pNext = &bi;
   VkPipelineShaderStageCreateInfo stages[3];
   if (info.stageCount > 3)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   memcpy(stages, info.pStages, info.stageCount * sizeof(stages[0]));
   for (unsigned i = 0; i < info.stageCount; i++)
      stages[i].module = VK_NULL_HANDLE;
   info.pStages = stages;
   VkPipeline refused = VK_NULL_HANDLE;
   bi.binaryCount = n - 1;
   VkResult missing = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, alloc, &refused);
   if (missing != VK_ERROR_FEATURE_NOT_PRESENT || refused != VK_NULL_HANDLE)
      return VK_ERROR_UNKNOWN;
   bi.binaryCount = n;
   printf("BINARY_MISSING_PLAN_REFUSED\n");
   result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, alloc, pipelines);
   for (unsigned i = 0; i < n; i++) {
      destroy(device, binaries[i], NULL);
   }
   /* A plan with a changed version and its original content key is refused
    * before private compute programs can be created or bound. */
   if (n < 1 || data[n - 1].dataSize < 12)
      return VK_ERROR_UNKNOWN;
   ((unsigned char *)data[n - 1].pData)[8] ^= 1;
   handles.pipelineBinaryCount = n;
   VkResult corrupt = create(device, &ci, NULL, &handles);
   if (corrupt != VK_SUCCESS)
      return corrupt;
   refused = VK_NULL_HANDLE;
   corrupt = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, alloc, &refused);
   for (unsigned i = 0; i < n; i++) {
      destroy(device, binaries[i], NULL);
      free(data[i].pData);
   }
   if (corrupt != VK_ERROR_FEATURE_NOT_PRESENT || refused != VK_NULL_HANDLE)
      return VK_ERROR_UNKNOWN;
   printf("BINARY_CORRUPT_PLAN_REFUSED\n");
   printf("BINARY_ROUNDTRIP count=%u result=%d\n", n, result);
   return result;
}

#undef vkCreateInstance
#define vkCreateInstance binary_create_instance
#undef vkCreateGraphicsPipelines
#define vkCreateGraphicsPipelines binary_roundtrip
#include "../safe-direct/pipe.c"

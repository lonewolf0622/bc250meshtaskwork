/* SPDX-License-Identifier: MIT
 * Offline fixture for binary code paths while feature advertisement remains
 * disabled. Instance entrypoints are used, as in RADV's internal tests. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "icd-fixture.h"

static VkInstance binary_instance;
static VkShaderModule load(const char *path);
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
   const char *write_path = getenv("PLAN_BINARY_WRITE");
   const char *read_path = getenv("PLAN_BINARY_READ");
   if (write_path) {
      FILE *file = fopen(write_path, "wb");
      if (!file || fwrite(&n, sizeof(n), 1, file) != 1)
         return VK_ERROR_UNKNOWN;
      for (unsigned i = 0; i < n; i++) {
         const uint64_t size = data[i].dataSize;
         if (fwrite(&keys[i], sizeof(keys[i]), 1, file) != 1 ||
             fwrite(&size, sizeof(size), 1, file) != 1 ||
             fwrite(data[i].pData, 1, size, file) != size)
            return VK_ERROR_UNKNOWN;
      }
      if (fclose(file))
         return VK_ERROR_UNKNOWN;
   }
   if (read_path) {
      FILE *file = fopen(read_path, "rb");
      unsigned stored_count;
      if (!file || fread(&stored_count, sizeof(stored_count), 1, file) != 1 || stored_count != n)
         return VK_ERROR_UNKNOWN;
      for (unsigned i = 0; i < n; i++) {
         uint64_t size;
         if (fread(&keys[i], sizeof(keys[i]), 1, file) != 1 ||
             fread(&size, sizeof(size), 1, file) != 1 || !size || size > (UINT64_C(128) << 20))
            return VK_ERROR_UNKNOWN;
         free(data[i].pData);
         data[i].dataSize = size;
         data[i].pData = malloc(size);
         if (!data[i].pData || fread(data[i].pData, 1, size, file) != size)
            return VK_ERROR_UNKNOWN;
      }
      if (fgetc(file) != EOF || fclose(file))
         return VK_ERROR_UNKNOWN;
   }
   vkDestroyPipeline(device, *pipelines, alloc);
   *pipelines = VK_NULL_HANDLE;
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
   VkPipelineRasterizationStateCreateInfo changed_raster;
   if (getenv("PLAN_BINARY_BAD_STATE")) {
      if (!infos->pRasterizationState)
         return VK_ERROR_UNKNOWN;
      changed_raster = *infos->pRasterizationState;
      changed_raster.polygonMode = VK_POLYGON_MODE_LINE;
      info.pRasterizationState = &changed_raster;
   }
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
   const char *swap_mesh = getenv("PLAN_SWAP_MESH");
   if (swap_mesh) {
      /* Both plans are individually valid. Mixing one pipeline's graphics
       * binaries with another pipeline's plan must still fail closed. */
      VkPipelineShaderStageCreateInfo other_stages[3];
      memcpy(other_stages, infos->pStages, infos->stageCount * sizeof(other_stages[0]));
      VkShaderModule module = load(swap_mesh);
      for (unsigned i = 0; i < infos->stageCount; i++)
         if (other_stages[i].stage == VK_SHADER_STAGE_MESH_BIT_EXT)
            other_stages[i].module = module;
      VkGraphicsPipelineCreateInfo other_info = *infos;
      other_info.pStages = other_stages;
      other_info.pNext = &flags;
      VkPipeline other = VK_NULL_HANDLE;
      VkResult swap_result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &other_info, alloc, &other);
      if (swap_result != VK_SUCCESS)
         return swap_result;
      VkPipelineBinaryCreateInfoKHR other_ci = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR, .pipeline = other};
      VkPipelineBinaryHandlesInfoKHR other_handles = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
      swap_result = create(device, &other_ci, NULL, &other_handles);
      if (swap_result != VK_SUCCESS || !other_handles.pipelineBinaryCount || other_handles.pipelineBinaryCount > 8)
         return VK_ERROR_UNKNOWN;
      VkPipelineBinaryKHR other_binaries[8];
      other_handles.pPipelineBinaries = other_binaries;
      swap_result = create(device, &other_ci, NULL, &other_handles);
      if (swap_result != VK_SUCCESS)
         return swap_result;
      VkPipelineBinaryKHR saved = binaries[n - 1];
      binaries[n - 1] = other_binaries[other_handles.pipelineBinaryCount - 1];
      refused = VK_NULL_HANDLE;
      swap_result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, alloc, &refused);
      binaries[n - 1] = saved;
      for (unsigned i = 0; i < other_handles.pipelineBinaryCount; i++)
         destroy(device, other_binaries[i], NULL);
      vkDestroyPipeline(device, other, alloc);
      vkDestroyShaderModule(device, module, NULL);
      if (swap_result != VK_ERROR_FEATURE_NOT_PRESENT || refused != VK_NULL_HANDLE)
         return VK_ERROR_UNKNOWN;
      printf("BINARY_FOREIGN_PLAN_REFUSED\n");
   }
   result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, alloc, pipelines);
   if (getenv("PLAN_BINARY_EXPECT_REFUSAL")) {
      for (unsigned i = 0; i < n; i++) {
         destroy(device, binaries[i], NULL);
         free(data[i].pData);
      }
      if (result != VK_ERROR_FEATURE_NOT_PRESENT || *pipelines != VK_NULL_HANDLE)
         return VK_ERROR_UNKNOWN;
      printf("BINARY_POLICY_PLAN_REFUSED\n");
      return result;
   }
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
#ifndef BC250_PIPELINE_PLAN_FIXTURE_ONLY
#include "../safe-direct/pipe.c"
#endif

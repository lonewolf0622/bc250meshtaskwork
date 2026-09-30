/* SPDX-License-Identifier: MIT
 * Hidden-entrypoint probe for conservative GPL source linking. */
#include <stdbool.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "icd-fixture.h"
#define BC250_PIPELINE_PLAN_FIXTURE_ONLY
#include "pipe.c"
#undef vkCreateGraphicsPipelines
#define vkCreateGraphicsPipelines ((PFN_vkCreateGraphicsPipelines)fixture_get_proc(fixture_instance, "vkCreateGraphicsPipelines"))
#include "binary-pipe.c"

static VkResult
gpl_complete_binary_link(VkDevice device, VkPipelineCache cache, const VkGraphicsPipelineCreateInfo *infos,
                         const VkAllocationCallbacks *alloc, VkPipeline *pipelines)
{
   PFN_vkCreateGraphicsPipelines create = (void *)fixture_get_proc(fixture_instance, "vkCreateGraphicsPipelines");
   VkGraphicsPipelineLibraryCreateInfoEXT subset = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
      .pNext = infos->pNext,
      .flags = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT |
               VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT |
               VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT |
               VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
   VkGraphicsPipelineCreateInfo info = *infos;
   info.pNext = &subset;
   info.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
   VkPipeline library = VK_NULL_HANDLE;
   VkResult result = binary_roundtrip(device, cache, 1, &info, alloc, &library);
   if (result != VK_SUCCESS)
      return result;
   VkPipelineLibraryCreateInfoKHR link = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
      .pNext = infos->pNext, .libraryCount = 1, .pLibraries = &library};
   info = *infos;
   info.pNext = &link;
   info.stageCount = 0;
   info.pStages = NULL;
   /* No retained source and compilation is forbidden at this link. */
   info.flags = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
   result = create(device, VK_NULL_HANDLE, 1, &info, alloc, pipelines);
   if (result == VK_SUCCESS) {
      VkPipeline refused = VK_NULL_HANDLE;
      info.flags = VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
      VkResult refusal = create(device, VK_NULL_HANDLE, 1, &info, alloc, &refused);
      if (refusal != VK_ERROR_FEATURE_NOT_PRESENT || refused != VK_NULL_HANDLE)
         result = VK_ERROR_UNKNOWN;
      else
         printf("GPL_BINARY_LTO_REFUSED\n");
   }
   vkDestroyPipeline(device, library, alloc);
   printf("GPL_BINARY_FAST_LINK result=%d\n", result);
   return result;
}

static VkResult
gpl_link(VkDevice device, VkPipelineCache cache, uint32_t count,
         const VkGraphicsPipelineCreateInfo *infos, const VkAllocationCallbacks *alloc,
         VkPipeline *pipelines)
{
   if (count != 1 || infos->stageCount > 3)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   PFN_vkCreateGraphicsPipelines create = (void *)fixture_get_proc(fixture_instance, "vkCreateGraphicsPipelines");
   const char *mode = getenv("GPL_MODE");
   if (mode && !strcmp(mode, "compiled"))
      return gpl_complete_binary_link(device, cache, infos, alloc, pipelines);
   const bool lto = mode && !strcmp(mode, "lto");
   const bool nested = mode && !strcmp(mode, "nested");
   VkPipelineShaderStageCreateInfo pre[2], fs;
   unsigned pre_count = 0;
   for (unsigned i = 0; i < infos->stageCount; i++) {
      if (infos->pStages[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT)
         fs = infos->pStages[i];
      else if (pre_count < 2)
         pre[pre_count++] = infos->pStages[i];
      else
         return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   VkPipeline libs[3] = {VK_NULL_HANDLE};
   VkGraphicsPipelineLibraryCreateInfoEXT subset = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
      .pNext = infos->pNext};
   VkGraphicsPipelineCreateInfo info = *infos;
   info.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR |
                (lto ? VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT : 0);
   info.pNext = &subset;
   info.stageCount = pre_count;
   info.pStages = pre;
   subset.flags = VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;
   VkResult result = create(device, cache, 1, &info, alloc, &libs[0]);
   if (result != VK_SUCCESS)
      return result;
   subset.flags = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT |
                  VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT;
   info.stageCount = 1;
   info.pStages = &fs;
   result = create(device, cache, 1, &info, alloc, &libs[1]);
   if (result != VK_SUCCESS)
      goto finish;
   VkPipelineLibraryCreateInfoKHR link = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
      .pNext = infos->pNext, .libraryCount = 2, .pLibraries = libs};
   info = *infos;
   info.stageCount = 0;
   info.pStages = NULL;
   info.pNext = &link;
   info.flags = lto ? VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT : 0;
   if (nested) {
      info.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
      result = create(device, cache, 1, &info, alloc, &libs[2]);
      if (result != VK_SUCCESS)
         goto finish;
      link.libraryCount = 1;
      link.pLibraries = &libs[2];
      info.flags = 0;
   }
   if (getenv("GPL_FINAL_BINARY"))
      result = binary_roundtrip(device, cache, 1, &info, alloc, pipelines);
   else
      result = cache_roundtrip(device, cache, 1, &info, alloc, pipelines);
   printf("GPL_SOURCE_LINK mode=%s result=%d\n", mode ? mode : "fast", result);
finish:
   for (unsigned i = 0; i < 3; i++)
      vkDestroyPipeline(device, libs[i], alloc);
   return result;
}
#undef vkCreateGraphicsPipelines
#define vkCreateGraphicsPipelines gpl_link
#include "../safe-direct/pipe.c"

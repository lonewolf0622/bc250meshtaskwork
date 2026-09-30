/* SPDX-License-Identifier: MIT
 * Hidden-entrypoint, noop-only probe for linked shader-object ownership. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "icd-fixture.h"

static struct {
   VkShaderModule handle;
   size_t size;
   void *code;
} modules[16];
static unsigned module_count;
static VkDescriptorSetLayout object_sets[8];
static VkPushConstantRange object_push[8];
static unsigned object_set_count, object_push_count;
static VkShaderEXT objects[2];

static VkResult
object_module(VkDevice device, const VkShaderModuleCreateInfo *info, const VkAllocationCallbacks *alloc,
              VkShaderModule *module)
{
   PFN_vkCreateShaderModule create = (void *)fixture_get_proc(fixture_instance, "vkCreateShaderModule");
   if (module_count == 16)
      return VK_ERROR_UNKNOWN;
   VkResult result = create(device, info, alloc, module);
   if (result != VK_SUCCESS)
      return result;
   modules[module_count].handle = *module;
   modules[module_count].size = info->codeSize;
   modules[module_count].code = malloc(info->codeSize);
   if (!modules[module_count].code)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memcpy(modules[module_count].code, info->pCode, info->codeSize);
   module_count++;
   return result;
}

static VkResult
object_layout(VkDevice device, const VkPipelineLayoutCreateInfo *info, const VkAllocationCallbacks *alloc,
              VkPipelineLayout *layout)
{
   PFN_vkCreatePipelineLayout create = (void *)fixture_get_proc(fixture_instance, "vkCreatePipelineLayout");
   if (info->setLayoutCount > 8 || info->pushConstantRangeCount > 8)
      return VK_ERROR_UNKNOWN;
   object_set_count = info->setLayoutCount;
   object_push_count = info->pushConstantRangeCount;
   if (object_set_count)
      memcpy(object_sets, info->pSetLayouts, object_set_count * sizeof(object_sets[0]));
   if (object_push_count)
      memcpy(object_push, info->pPushConstantRanges, object_push_count * sizeof(object_push[0]));
   return create(device, info, alloc, layout);
}

static VkResult
object_pipeline(VkDevice device, VkPipelineCache cache, uint32_t count,
                const VkGraphicsPipelineCreateInfo *infos, const VkAllocationCallbacks *alloc, VkPipeline *pipelines)
{
   PFN_vkCreateGraphicsPipelines create_pipeline = (void *)fixture_get_proc(fixture_instance, "vkCreateGraphicsPipelines");
   PFN_vkCreateShadersEXT create = (void *)fixture_get_proc(fixture_instance, "vkCreateShadersEXT");
   PFN_vkGetShaderBinaryDataEXT get = (void *)fixture_get_proc(fixture_instance, "vkGetShaderBinaryDataEXT");
   PFN_vkDestroyShaderEXT destroy = (void *)fixture_get_proc(fixture_instance, "vkDestroyShaderEXT");
   if (count != 1 || infos->stageCount != 2)
   {
      printf("OBJECT_SHAPE_REFUSED\n");
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   /* A pipeline supplies the inherited fixture's initial graphics state; the
    * draw uses the objects after vkCmdBindShadersEXT clears that pipeline. */
   VkResult result = create_pipeline(device, cache, count, infos, alloc, pipelines);
   if (result != VK_SUCCESS)
      return result;
   VkShaderCreateInfoEXT shader_infos[2];
   for (unsigned i = 0; i < 2; i++) {
      const VkPipelineShaderStageCreateInfo *stage = &infos->pStages[i];
      shader_infos[i] = (VkShaderCreateInfoEXT){
         .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
         .flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT |
            (stage->stage == VK_SHADER_STAGE_MESH_BIT_EXT ? VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT : 0),
         .stage = stage->stage,
         .nextStage = stage->stage == VK_SHADER_STAGE_MESH_BIT_EXT ? VK_SHADER_STAGE_FRAGMENT_BIT : 0,
         .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .pName = stage->pName,
         .pSpecializationInfo = stage->pSpecializationInfo,
         .setLayoutCount = object_set_count, .pSetLayouts = object_sets,
         .pushConstantRangeCount = object_push_count, .pPushConstantRanges = object_push};
      unsigned j;
      for (j = 0; j < module_count; j++)
         if (modules[j].handle == stage->module)
            break;
      if (j == module_count)
         return VK_ERROR_UNKNOWN;
      shader_infos[i].pCode = modules[j].code;
      shader_infos[i].codeSize = modules[j].size;
   }
   VkShaderCreateInfoEXT original_infos[2];
   memcpy(original_infos, shader_infos, sizeof(original_infos));
   const char *write_path = getenv("OBJECT_BINARY_WRITE"), *read_path = getenv("OBJECT_BINARY_READ");
   void *data[2] = {NULL, NULL};
   size_t sizes[2] = {0, 0};
   if (!read_path) {
      result = create(device, 2, shader_infos, alloc, objects);
      printf("OBJECT_LINK result=%d\n", result);
      if (result != VK_SUCCESS)
         return result;
   }
   for (unsigned i = 0; !read_path && i < 2; i++) {
      result = get(device, objects[i], &sizes[i], NULL);
      if (result != VK_SUCCESS)
         return result;
      data[i] = malloc(sizes[i]);
      if (!data[i])
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      result = get(device, objects[i], &sizes[i], data[i]);
      if (result != VK_SUCCESS)
         return result;
      destroy(device, objects[i], alloc);
      objects[i] = VK_NULL_HANDLE;
      shader_infos[i].codeType = VK_SHADER_CODE_TYPE_BINARY_EXT;
      shader_infos[i].pCode = data[i];
      shader_infos[i].codeSize = sizes[i];
   }
   if (write_path) {
      FILE *f = fopen(write_path, "wb");
      if (!f)
         return VK_ERROR_UNKNOWN;
      for (unsigned i = 0; i < 2; i++) {
         uint64_t size = sizes[i];
         if (fwrite(&size, sizeof(size), 1, f) != 1 || fwrite(data[i], 1, sizes[i], f) != sizes[i])
            return VK_ERROR_UNKNOWN;
      }
      if (fclose(f))
         return VK_ERROR_UNKNOWN;
   }
   if (read_path) {
      FILE *f = fopen(read_path, "rb");
      if (!f)
         return VK_ERROR_UNKNOWN;
      for (unsigned i = 0; i < 2; i++) {
         uint64_t size;
         if (fread(&size, sizeof(size), 1, f) != 1 || !size || size > (UINT64_C(128) << 20))
            return VK_ERROR_UNKNOWN;
         free(data[i]);
         data[i] = malloc(size);
         if (!data[i] || fread(data[i], 1, size, f) != size)
            return VK_ERROR_UNKNOWN;
         sizes[i] = size;
         shader_infos[i].pCode = data[i];
         shader_infos[i].codeSize = sizes[i];
         shader_infos[i].codeType = VK_SHADER_CODE_TYPE_BINARY_EXT;
      }
      if (fgetc(f) != EOF || fclose(f))
         return VK_ERROR_UNKNOWN;
   }
   result = create(device, 2, shader_infos, alloc, objects);
   if (getenv("OBJECT_EXPECT_REFUSAL")) {
      if (result == VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT &&
          objects[0] == VK_NULL_HANDLE && objects[1] == VK_NULL_HANDLE)
         printf("OBJECT_BINARY_POLICY_REFUSED\n");
      return result;
   }
   if (result != VK_SUCCESS)
      return result;
   printf("OBJECT_BINARY_ROUNDTRIP\n");
   for (unsigned i = 0; i < 2; i++) {
      VkShaderEXT refused = VK_NULL_HANDLE;
      ((unsigned char *)data[i])[sizes[i] - 1] ^= 1;
      VkResult bad = create(device, 1, &shader_infos[i], alloc, &refused);
      ((unsigned char *)data[i])[sizes[i] - 1] ^= 1;
      if (bad != VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT || refused != VK_NULL_HANDLE)
         return VK_ERROR_UNKNOWN;
      shader_infos[i].codeSize--;
      bad = create(device, 1, &shader_infos[i], alloc, &refused);
      shader_infos[i].codeSize++;
      if (bad != VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT || refused != VK_NULL_HANDLE)
         return VK_ERROR_UNKNOWN;
      VkPushConstantRange changed = {VK_SHADER_STAGE_ALL, 0, 32};
      VkShaderCreateInfoEXT bad_layout = shader_infos[i];
      bad_layout.pushConstantRangeCount = 1;
      bad_layout.pPushConstantRanges = &changed;
      bad = create(device, 1, &bad_layout, alloc, &refused);
      if (bad != VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT || refused != VK_NULL_HANDLE)
         return VK_ERROR_UNKNOWN;
      free(data[i]);
   }
   printf("OBJECT_CORRUPT_TRUNCATED_LAYOUT_REFUSED\n");
   const char *foreign_path = getenv("OBJECT_FOREIGN_FS");
   if (foreign_path) {
      FILE *f = fopen(foreign_path, "rb");
      if (!f || fseek(f, 0, SEEK_END))
         return VK_ERROR_UNKNOWN;
      long size = ftell(f);
      if (size <= 0 || fseek(f, 0, SEEK_SET))
         return VK_ERROR_UNKNOWN;
      void *code = malloc(size);
      if (!code || fread(code, 1, size, f) != (size_t)size || fclose(f))
         return VK_ERROR_UNKNOWN;
      original_infos[1].pCode = code;
      original_infos[1].codeSize = size;
      VkShaderEXT foreign[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
      result = create(device, 2, original_infos, alloc, foreign);
      free(code);
      if (result != VK_SUCCESS)
         return result;
      destroy(device, objects[1], alloc);
      objects[1] = foreign[1];
      destroy(device, foreign[0], alloc);
      printf("OBJECT_FOREIGN_PAIR_BOUND\n");
   }
   return VK_SUCCESS;
}

static void
object_bind(VkCommandBuffer cb, VkPipelineBindPoint bind_point, VkPipeline pipeline)
{
   PFN_vkCmdBindPipeline bind = (void *)fixture_get_proc(fixture_instance, "vkCmdBindPipeline");
   PFN_vkCmdBindShadersEXT bind_objects = (void *)fixture_get_proc(fixture_instance, "vkCmdBindShadersEXT");
   bind(cb, bind_point, pipeline);
   const VkShaderStageFlagBits stages[2] = {VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT};
   bind_objects(cb, 2, stages, objects);
   if (getenv("OBJECT_BAD_STATE")) {
      PFN_vkCmdSetPolygonModeEXT set = (void *)fixture_get_proc(fixture_instance, "vkCmdSetPolygonModeEXT");
      set(cb, VK_POLYGON_MODE_LINE);
   }
   printf("OBJECTS_BOUND\n");
}
#undef vkCreateShaderModule
#define vkCreateShaderModule object_module
#undef vkCreatePipelineLayout
#define vkCreatePipelineLayout object_layout
#undef vkCreateGraphicsPipelines
#define vkCreateGraphicsPipelines object_pipeline
#undef vkCmdBindPipeline
#define vkCmdBindPipeline object_bind
#include "../safe-direct/pipe.c"

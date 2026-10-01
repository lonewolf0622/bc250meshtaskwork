/*
 * Copyright © 2024 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_pipeline_binary.h"
#include "radv_bc250.h"
#include "tools/radv_debug.h"
#include "util/blob.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/u_atomic.h"
#include "radv_device.h"
#include "radv_entrypoints.h"
#include "radv_physical_device.h"
#include "radv_pipeline_cache.h"
#include "radv_pipeline_graphics.h"
#include "radv_pipeline_rt.h"
#include "radv_shader.h"
#include "vk_log.h"
#include "vk_pipeline.h"

static VkResult
radv_get_pipeline_key(struct radv_device *device, const VkPipelineCreateInfoKHR *pPipelineCreateInfo,
                      unsigned char *key)
{
   VkResult result = VK_SUCCESS;

   switch (((VkBaseInStructure *)pPipelineCreateInfo->pNext)->sType) {
   case VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO: {
      const VkGraphicsPipelineCreateInfo *graphics_create_info =
         (VkGraphicsPipelineCreateInfo *)pPipelineCreateInfo->pNext;
      struct radv_graphics_pipeline_state gfx_state;

      result = radv_generate_graphics_pipeline_state(device, graphics_create_info, &gfx_state);
      if (result != VK_SUCCESS)
         return result;

      radv_graphics_pipeline_hash(device, &gfx_state, key);
      radv_graphics_pipeline_state_finish(device, &gfx_state);
      break;
   }
   case VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: {
      const VkComputePipelineCreateInfo *compute_create_info =
         (VkComputePipelineCreateInfo *)pPipelineCreateInfo->pNext;

      radv_compute_pipeline_hash(device, compute_create_info, key);
      break;
   }
   case VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR: {
      const VkRayTracingPipelineCreateInfoKHR *rt_create_info =
         (VkRayTracingPipelineCreateInfoKHR *)pPipelineCreateInfo->pNext;
      struct radv_ray_tracing_state_key rt_state;

      result = radv_generate_ray_tracing_state_key(device, rt_create_info, &rt_state);
      if (result != VK_SUCCESS)
         return result;

      radv_ray_tracing_pipeline_hash(device, rt_create_info, &rt_state, key);
      radv_ray_tracing_state_key_finish(&rt_state);
      break;
   }
   default:
      UNREACHABLE("unsupported pipeline create info struct");
   }

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPipelineKeyKHR(VkDevice _device, const VkPipelineCreateInfoKHR *pPipelineCreateInfo,
                       VkPipelineBinaryKeyKHR *pPipelineKey)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   VkResult result;

   memset(pPipelineKey->key, 0, sizeof(pPipelineKey->key));

   /* Return the global key that applies to all pipelines. */
   if (!pPipelineCreateInfo) {
      struct mesa_blake3 ctx;

      static_assert(sizeof(blake3_hash) <= sizeof(pPipelineKey->key), "mismatch pipeline binary key size");

      _mesa_blake3_init(&ctx);
      _mesa_blake3_update(&ctx, pdev->cache_uuid, sizeof(pdev->cache_uuid));
      _mesa_blake3_update(&ctx, device->cache_hash, sizeof(device->cache_hash));
      _mesa_blake3_final(&ctx, pPipelineKey->key);

      pPipelineKey->keySize = sizeof(blake3_hash);

      return VK_SUCCESS;
   }

   result = radv_get_pipeline_key(device, pPipelineCreateInfo, pPipelineKey->key);
   if (result != VK_SUCCESS)
      return result;

   pPipelineKey->keySize = BLAKE3_KEY_LEN;

   return VK_SUCCESS;
}

static VkResult
radv_create_pipeline_binary(struct radv_device *device, const VkAllocationCallbacks *pAllocator, const blake3_hash key,
                            const void *data, size_t data_size, struct radv_pipeline_binary **pipeline_binary_out)
{
   struct radv_pipeline_binary *pipeline_binary;

   pipeline_binary =
      vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pipeline_binary), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pipeline_binary == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   vk_object_base_init(&device->vk, &pipeline_binary->base, VK_OBJECT_TYPE_PIPELINE_BINARY_KHR);

   pipeline_binary->data = (void *)data;
   pipeline_binary->size = data_size;

   memcpy(pipeline_binary->key, key, BLAKE3_OUT_LEN);

   *pipeline_binary_out = pipeline_binary;
   return VK_SUCCESS;
}

/* A separate binary carries private draw state; ordinary shader binaries
 * and all switch-off formats remain untouched. */
#define RADV_BC250_PLAN_BINARY_MAGIC UINT64_C(0x32304e414c503242)

bool
radv_bc250_pipeline_binary_is_plan(const struct radv_pipeline_binary *binary)
{
   uint64_t magic = 0;
   if (binary->size >= sizeof(magic))
      memcpy(&magic, binary->data, sizeof(magic));
   return magic == RADV_BC250_PLAN_BINARY_MAGIC;
}

VkResult
radv_create_pipeline_binary_from_bc250_plan(struct radv_device *device,
                                           const VkAllocationCallbacks *allocator,
                                           const struct radv_bc250_pipeline_plan *plan,
                                           struct radv_shader *producer, struct radv_shader *setup,
                                           struct util_dynarray *binaries, uint32_t *num_binaries)
{
   if (!radv_bc250_pipeline_plan_valid(plan) ||
       (!!producer != !!(plan->flags & RADV_BC250_PLAN_TASK)) || (!!producer != !!setup))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (!binaries) {
      (*num_binaries)++;
      return VK_SUCCESS;
   }
   struct blob blob;
   blob_init(&blob);
   blob_write_uint64(&blob, RADV_BC250_PLAN_BINARY_MAGIC);
   blob_write_bytes(&blob, plan, sizeof(*plan));
   struct radv_shader *shaders[2] = {producer, setup};
   for (unsigned i = 0; producer && i < 2; i++) {
      struct blob shader_blob;
      blob_init(&shader_blob);
      radv_shader_serialize(shaders[i], &shader_blob);
      blob_write_uint64(&blob, shader_blob.size);
      blob_write_bytes(&blob, shaders[i]->hash, sizeof(shaders[i]->hash));
      blob_write_bytes(&blob, shader_blob.data, shader_blob.size);
      if (shader_blob.out_of_memory)
         blob.out_of_memory = true;
      blob_finish(&shader_blob);
   }
   if (blob.out_of_memory) {
      blob_finish(&blob);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   blake3_hash key;
   _mesa_blake3_compute(blob.data, blob.size, key);
   void *data;
   size_t size;
   blob_finish_get_buffer(&blob, &data, &size);
   struct radv_pipeline_binary *binary;
   VkResult result = radv_create_pipeline_binary(device, allocator, key, data, size, &binary);
   if (result != VK_SUCCESS) {
      free(data);
      return result;
   }
   util_dynarray_append(binaries, binary);
   return VK_SUCCESS;
}

bool
radv_bc250_shader_binary_valid(const void *data, size_t size, unsigned stage)
{
   if (size < sizeof(struct radv_shader_binary_legacy))
      return false;
   struct radv_shader_binary_legacy legacy;
   memcpy(&legacy, data, sizeof(legacy));
   const uint64_t total = (uint64_t)sizeof(legacy) + legacy.code_size + legacy.ir_size +
                          legacy.disasm_size + legacy.stats_size + legacy.debug_info_size;
   if (legacy.base.type != RADV_BINARY_TYPE_LEGACY || legacy.base.info.stage != stage ||
       total != legacy.base.total_size || total > size || legacy.exec_size > legacy.code_size)
      return false;
   struct blob_reader debug;
   blob_reader_init(&debug, data, size);
   blob_skip_bytes(&debug, total);
   if (debug.current != debug.end) {
      uint32_t spirv_size = blob_read_uint32(&debug);
      blob_skip_bytes(&debug, spirv_size);
      blob_read_string(&debug);
      blob_read_string(&debug);
   }
   return !debug.overrun && debug.current == debug.end;
}

VkResult
radv_bc250_pipeline_binary_restore(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                   const struct radv_pipeline_layout *layout,
                                   const struct radv_pipeline_binary *binary)
{
   blake3_hash key;
   _mesa_blake3_compute(binary->data, binary->size, key);
   if (!device->bc250_env.pipeline_plan || memcmp(key, binary->key, sizeof(key)))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct blob_reader blob;
   blob_reader_init(&blob, binary->data, binary->size);
   if (blob_read_uint64(&blob) != RADV_BC250_PLAN_BINARY_MAGIC)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct radv_bc250_pipeline_plan plan;
   blob_copy_bytes(&blob, &plan, sizeof(plan));
   if (blob.overrun || !radv_bc250_pipeline_plan_valid(&plan))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   const void *shader_data[2] = {NULL, NULL};
   size_t shader_size[2] = {0, 0};
   blake3_hash hashes[2];
   for (unsigned i = 0; (plan.flags & RADV_BC250_PLAN_TASK) && i < 2; i++) {
      uint64_t size = blob_read_uint64(&blob);
      if (blob.overrun || size > SIZE_MAX)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      shader_size[i] = size;
      blob_copy_bytes(&blob, hashes[i], sizeof(hashes[i]));
      shader_data[i] = blob_read_bytes(&blob, size);
      if (blob.overrun || !radv_bc250_shader_binary_valid(shader_data[i], size, MESA_SHADER_COMPUTE))
         return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   if (blob.overrun || blob.current != blob.end)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct radv_shader *shaders[2] = {NULL, NULL};
   VkResult result = VK_SUCCESS;
   for (unsigned i = 0; (plan.flags & RADV_BC250_PLAN_TASK) && i < 2; i++) {
      struct blob_reader shader_blob;
      blob_reader_init(&shader_blob, shader_data[i], shader_size[i]);
      shaders[i] = radv_shader_deserialize(device, hashes[i], sizeof(hashes[i]), &shader_blob);
      if (!shaders[i]) {
         result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
         goto done;
      }
   }
   result = radv_bc250_restore_cached_plan(device, pipeline, layout, &plan,
      pipeline->base.shaders[MESA_SHADER_MESH], pipeline->base.shaders[MESA_SHADER_FRAGMENT],
      shaders[0], shaders[1]);
done:
   for (unsigned i = 0; i < 2; i++) {
      if (shaders[i])
         radv_shader_unref(device, shaders[i]);
   }
   return result;
}

static VkResult
radv_create_pipeline_binary_from_data(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                                      const VkPipelineBinaryDataKHR *pData, const VkPipelineBinaryKeyKHR *pKey,
                                      struct util_dynarray *pipeline_binaries, uint32_t *num_binaries)
{
   struct radv_pipeline_binary *pipeline_binary;
   VkResult result;
   void *data;

   if (!pipeline_binaries) {
      (*num_binaries)++;
      return VK_SUCCESS;
   }

   data = malloc(pData->dataSize);
   if (!data)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   memcpy(data, pData->pData, pData->dataSize);

   result = radv_create_pipeline_binary(device, pAllocator, pKey->key, data, pData->dataSize, &pipeline_binary);
   if (result != VK_SUCCESS) {
      free(data);
      return result;
   }

   util_dynarray_append(pipeline_binaries, pipeline_binary);
   return result;
}

VkResult
radv_create_pipeline_binary_from_shader(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                                        struct radv_shader *shader, struct util_dynarray *pipeline_binaries,
                                        uint32_t *num_binaries)
{
   struct radv_pipeline_binary *pipeline_binary;
   struct blob blob;
   size_t data_size;
   VkResult result;
   void *data;

   if (!pipeline_binaries) {
      (*num_binaries)++;
      return VK_SUCCESS;
   }

   blob_init(&blob);
   radv_shader_serialize(shader, &blob);
   blob_finish_get_buffer(&blob, &data, &data_size);

   result = radv_create_pipeline_binary(device, pAllocator, shader->hash, data, data_size, &pipeline_binary);
   if (result != VK_SUCCESS) {
      free(data);
      return result;
   }

   util_dynarray_append(pipeline_binaries, pipeline_binary);
   return result;
}

VkResult
radv_create_pipeline_binary_from_rt_shader(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                                           struct radv_shader *shader, bool is_traversal_shader,
                                           const uint8_t stage_blake3[BLAKE3_KEY_LEN],
                                           const struct radv_ray_tracing_stage_info *rt_stage_info, uint32_t stack_size,
                                           struct vk_pipeline_cache_object *nir,
                                           struct util_dynarray *pipeline_binaries, uint32_t *num_binaries)
{
   struct radv_pipeline_binary *pipeline_binary;
   struct mesa_blake3 ctx;
   struct blob blob;
   size_t data_size;
   blake3_hash key;
   VkResult result;
   void *data;

   if (!pipeline_binaries) {
      (*num_binaries)++;
      return VK_SUCCESS;
   }

   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, stage_blake3, sizeof(*stage_blake3));
   _mesa_blake3_final(&ctx, key);

   struct radv_ray_tracing_binary_header header = {
      .is_traversal_shader = is_traversal_shader,
      .has_shader = !!shader,
      .has_nir = !!nir,
      .stack_size = stack_size,
   };

   memcpy(header.stage_blake3, stage_blake3, sizeof(header.stage_blake3));
   if (rt_stage_info)
      memcpy(&header.stage_info, rt_stage_info, sizeof(header.stage_info));

   blob_init(&blob);
   blob_write_bytes(&blob, &header, sizeof(header));
   if (header.has_shader)
      radv_shader_serialize(shader, &blob);
   if (header.has_nir) {
      struct vk_raw_data_cache_object *nir_object = container_of(nir, struct vk_raw_data_cache_object, base);
      blob_write_bytes(&blob, nir_object->data, nir_object->data_size);
   }
   blob_finish_get_buffer(&blob, &data, &data_size);

   result = radv_create_pipeline_binary(device, pAllocator, key, data, data_size, &pipeline_binary);
   if (result != VK_SUCCESS) {
      free(data);
      return result;
   }

   util_dynarray_append(pipeline_binaries, pipeline_binary);
   return result;
}

static VkResult
radv_create_pipeline_binary_from_pipeline(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                                          struct radv_pipeline *pipeline, struct util_dynarray *pipeline_binaries,
                                          uint32_t *num_binaries)
{
   VkResult result = VK_SUCCESS;

   if (pipeline->type == RADV_PIPELINE_RAY_TRACING) {
      struct radv_ray_tracing_pipeline *rt_pipeline = radv_pipeline_to_ray_tracing(pipeline);

      for (uint32_t i = 0; i < rt_pipeline->non_imported_stage_count; i++) {
         struct radv_ray_tracing_stage *rt_stage = &rt_pipeline->stages[i];

         result = radv_create_pipeline_binary_from_rt_shader(device, pAllocator, rt_stage->shader, false,
                                                             rt_stage->blake3, &rt_stage->info, rt_stage->stack_size,
                                                             rt_stage->nir, pipeline_binaries, num_binaries);
         if (result != VK_SUCCESS)
            return result;
      }

      struct radv_shader *traversal_shader = rt_pipeline->base.base.shaders[MESA_SHADER_INTERSECTION];
      if (traversal_shader) {
         result = radv_create_pipeline_binary_from_rt_shader(device, pAllocator, traversal_shader, true,
                                                             traversal_shader->hash, NULL, 0, NULL, pipeline_binaries,
                                                             num_binaries);
         if (result != VK_SUCCESS)
            return result;
      }
   } else {
      if (pipeline->type == RADV_PIPELINE_GRAPHICS_LIB &&
          radv_pipeline_to_graphics_lib(pipeline)->bc250_source_only)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      for (uint32_t i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         if (!pipeline->shaders[i])
            continue;

         result = radv_create_pipeline_binary_from_shader(device, pAllocator, pipeline->shaders[i], pipeline_binaries,
                                                          num_binaries);
         if (result != VK_SUCCESS)
            return result;
      }

      if (pipeline->gs_copy_shader) {
         result = radv_create_pipeline_binary_from_shader(device, pAllocator, pipeline->gs_copy_shader,
                                                          pipeline_binaries, num_binaries);
         if (result != VK_SUCCESS)
            return result;
      }
      const bool complete_lib = pipeline->type == RADV_PIPELINE_GRAPHICS_LIB &&
         device->bc250_env.gpl_binary_link &&
         radv_pipeline_to_graphics_lib(pipeline)->lib_flags ==
            (VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT |
             VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT |
             VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT |
             VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT);
      if (device->bc250_env.pipeline_plan &&
          (pipeline->type == RADV_PIPELINE_GRAPHICS || complete_lib) && pipeline->shaders[MESA_SHADER_MESH]) {
         struct radv_graphics_pipeline *graphics = complete_lib ?
            &radv_pipeline_to_graphics_lib(pipeline)->base : radv_pipeline_to_graphics(pipeline);
         VK_FROM_HANDLE(radv_pipeline, producer, graphics->bc250_task_pipeline);
         VK_FROM_HANDLE(radv_pipeline, setup, graphics->bc250_setup_pipeline);
         result = radv_create_pipeline_binary_from_bc250_plan(device, pAllocator, &graphics->bc250_plan,
                     producer ? producer->shaders[MESA_SHADER_COMPUTE] : NULL,
                     producer && setup ? setup->shaders[MESA_SHADER_COMPUTE] : NULL, pipeline_binaries, num_binaries);
      }
   }

   return result;
}

static VkResult
radv_create_pipeline_binary_from_cache(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                                       const VkPipelineCreateInfoKHR *pPipelineCreateInfo,
                                       struct util_dynarray *pipeline_binaries, uint32_t *num_binaries)
{
   unsigned char key[BLAKE3_KEY_LEN];
   bool found_in_internal_cache;
   VkResult result;

   assert(pPipelineCreateInfo);

   result = radv_get_pipeline_key(device, pPipelineCreateInfo, key);
   if (result != VK_SUCCESS)
      return result;

   result = radv_pipeline_cache_get_binaries(device, pAllocator, key, pipeline_binaries, num_binaries,
                                             &found_in_internal_cache);
   if (result != VK_SUCCESS)
      return result;

   return found_in_internal_cache ? VK_SUCCESS : VK_PIPELINE_BINARY_MISSING_KHR;
}

static VkResult
radv_create_pipeline_binaries(struct radv_device *device, const VkPipelineBinaryCreateInfoKHR *pCreateInfo,
                              const VkAllocationCallbacks *pAllocator, struct util_dynarray *pipeline_binaries,
                              uint32_t *num_binaries)
{
   VkResult result = VK_SUCCESS;

   if (pCreateInfo->pKeysAndDataInfo) {
      const VkPipelineBinaryKeysAndDataKHR *pKeysAndDataInfo = pCreateInfo->pKeysAndDataInfo;

      for (uint32_t i = 0; i < pKeysAndDataInfo->binaryCount; i++) {
         const VkPipelineBinaryDataKHR *pData = &pKeysAndDataInfo->pPipelineBinaryData[i];
         const VkPipelineBinaryKeyKHR *pKey = &pKeysAndDataInfo->pPipelineBinaryKeys[i];

         result =
            radv_create_pipeline_binary_from_data(device, pAllocator, pData, pKey, pipeline_binaries, num_binaries);
         if (result != VK_SUCCESS)
            return result;
      }
   } else if (pCreateInfo->pipeline) {
      VK_FROM_HANDLE(radv_pipeline, pipeline, pCreateInfo->pipeline);

      result = radv_create_pipeline_binary_from_pipeline(device, pAllocator, pipeline, pipeline_binaries, num_binaries);
   } else {
      result = radv_create_pipeline_binary_from_cache(device, pAllocator, pCreateInfo->pPipelineCreateInfo,
                                                      pipeline_binaries, num_binaries);
   }

   return result;
}

static void
radv_destroy_pipeline_binary(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                             struct radv_pipeline_binary *pipeline_binary)
{
   if (!pipeline_binary)
      return;

   free(pipeline_binary->data);

   vk_object_base_finish(&pipeline_binary->base);
   vk_free2(&device->vk.alloc, pAllocator, pipeline_binary);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_CreatePipelineBinariesKHR(VkDevice _device, const VkPipelineBinaryCreateInfoKHR *pCreateInfo,
                               const VkAllocationCallbacks *pAllocator, VkPipelineBinaryHandlesInfoKHR *pBinaries)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   struct util_dynarray pipeline_binaries;
   VkResult result;

   if (!pBinaries->pPipelineBinaries) {
      result = radv_create_pipeline_binaries(device, pCreateInfo, pAllocator, NULL, &pBinaries->pipelineBinaryCount);
      return result;
   }

   for (uint32_t i = 0; i < pBinaries->pipelineBinaryCount; i++)
      pBinaries->pPipelineBinaries[i] = VK_NULL_HANDLE;

   pipeline_binaries = UTIL_DYNARRAY_INIT;

   /* Get all pipeline binaries from the pCreateInfo first to simplify the creation. */
   result = radv_create_pipeline_binaries(device, pCreateInfo, pAllocator, &pipeline_binaries, NULL);
   if (result != VK_SUCCESS) {
      util_dynarray_foreach (&pipeline_binaries, struct radv_pipeline_binary *, pipeline_binary)
         radv_destroy_pipeline_binary(device, pAllocator, *pipeline_binary);
      util_dynarray_fini(&pipeline_binaries);
      return result;
   }

   const uint32_t num_binaries = util_dynarray_num_elements(&pipeline_binaries, struct radv_pipeline_binary *);

   for (uint32_t i = 0; i < num_binaries; i++) {
      struct radv_pipeline_binary **pipeline_binary =
         util_dynarray_element(&pipeline_binaries, struct radv_pipeline_binary *, i);

      if (i < pBinaries->pipelineBinaryCount) {
         pBinaries->pPipelineBinaries[i] = radv_pipeline_binary_to_handle(*pipeline_binary);
      } else {
         /* Free the pipeline binary that couldn't be returned. */
         radv_destroy_pipeline_binary(device, pAllocator, *pipeline_binary);
      }
   }

   result = pBinaries->pipelineBinaryCount < num_binaries ? VK_INCOMPLETE : result;
   pBinaries->pipelineBinaryCount = MIN2(num_binaries, pBinaries->pipelineBinaryCount);

   util_dynarray_fini(&pipeline_binaries);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
radv_DestroyPipelineBinaryKHR(VkDevice _device, VkPipelineBinaryKHR pipelineBinary,
                              const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(radv_pipeline_binary, pipeline_binary, pipelineBinary);
   VK_FROM_HANDLE(radv_device, device, _device);

   radv_destroy_pipeline_binary(device, pAllocator, pipeline_binary);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPipelineBinaryDataKHR(VkDevice _device, const VkPipelineBinaryDataInfoKHR *pInfo,
                              VkPipelineBinaryKeyKHR *pPipelineBinaryKey, size_t *pPipelineBinaryDataSize,
                              void *pPipelineBinaryData)
{
   VK_FROM_HANDLE(radv_pipeline_binary, pipeline_binary, pInfo->pipelineBinary);
   const size_t size = pipeline_binary->size;

   memcpy(pPipelineBinaryKey->key, pipeline_binary->key, sizeof(pipeline_binary->key));
   pPipelineBinaryKey->keySize = sizeof(pipeline_binary->key);

   if (!pPipelineBinaryData) {
      *pPipelineBinaryDataSize = size;
      return VK_SUCCESS;
   }

   if (*pPipelineBinaryDataSize < size) {
      *pPipelineBinaryDataSize = size;
      return VK_ERROR_NOT_ENOUGH_SPACE_KHR;
   }

   memcpy(pPipelineBinaryData, pipeline_binary->data, size);
   *pPipelineBinaryDataSize = size;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_ReleaseCapturedPipelineDataKHR(VkDevice _device, const VkReleaseCapturedPipelineDataInfoKHR *pInfo,
                                    const VkAllocationCallbacks *pAllocator)
{
   /* no-op */
   return VK_SUCCESS;
}

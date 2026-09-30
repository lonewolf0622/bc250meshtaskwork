/*
 * Copyright © 2024 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "vk_log.h"

#include "util/blob.h"
#include "radv_bc250.h"
#include "radv_device.h"
#include "radv_entrypoints.h"
#include "radv_physical_device.h"
#include "radv_pipeline_cache.h"
#include "radv_pipeline_compute.h"
#include "radv_pipeline_graphics.h"
#include "radv_shader_object.h"

struct radv_shader_object_metadata {
   uint32_t dynamic_offset_count;
};

#define RADV_BC250_OBJECT_MAGIC UINT64_C(0x354a424f30353242)

static void
radv_bc250_object_layout_hash(const VkShaderCreateInfoEXT *info, uint8_t hash[32])
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, &info->setLayoutCount, sizeof(info->setLayoutCount));
   const uint8_t absent[32] = {0};
   for (unsigned i = 0; i < info->setLayoutCount; i++) {
      VK_FROM_HANDLE(radv_descriptor_set_layout, layout, info->pSetLayouts[i]);
      _mesa_blake3_update(&ctx, layout ? layout->hash : absent, sizeof(absent));
   }
   _mesa_blake3_update(&ctx, &info->pushConstantRangeCount, sizeof(info->pushConstantRangeCount));
   for (unsigned i = 0; i < info->pushConstantRangeCount; i++) {
      const VkPushConstantRange *range = &info->pPushConstantRanges[i];
      _mesa_blake3_update(&ctx, &range->stageFlags, sizeof(range->stageFlags));
      _mesa_blake3_update(&ctx, &range->offset, sizeof(range->offset));
      _mesa_blake3_update(&ctx, &range->size, sizeof(range->size));
   }
   _mesa_blake3_final(&ctx, hash);
}

static void
radv_bc250_shader_object_policy(struct radv_device *device, struct radv_shader_object *obj)
{
   obj->bc250_policy_valid = true;
   memcpy(obj->bc250_route_key, &device->compiler_info.key, sizeof(obj->bc250_route_key));
   memcpy(obj->bc250_hardware_key, &device->compiler_info.hw, sizeof(obj->bc250_hardware_key));
}

static void
radv_shader_object_destroy_variant(struct radv_device *device, VkShaderCodeTypeEXT code_type,
                                   struct radv_shader *shader, struct radv_shader_binary *binary)
{
   if (shader)
      radv_shader_unref(device, shader);

   if (code_type == VK_SHADER_CODE_TYPE_SPIRV_EXT)
      free(binary);
}

static void
radv_shader_object_destroy(struct radv_device *device, struct radv_shader_object *shader_obj,
                           const VkAllocationCallbacks *pAllocator)
{
   radv_shader_object_destroy_variant(device, shader_obj->code_type, shader_obj->as_ls.shader,
                                      shader_obj->as_ls.binary);
   radv_shader_object_destroy_variant(device, shader_obj->code_type, shader_obj->as_es.shader,
                                      shader_obj->as_es.binary);
   radv_shader_object_destroy_variant(device, shader_obj->code_type, shader_obj->gs.copy_shader,
                                      shader_obj->gs.copy_binary);
   radv_shader_object_destroy_variant(device, shader_obj->code_type, shader_obj->shader, shader_obj->binary);

   vk_object_base_finish(&shader_obj->base);
   vk_free2(&device->vk.alloc, pAllocator, shader_obj);
}

VKAPI_ATTR void VKAPI_CALL
radv_DestroyShaderEXT(VkDevice _device, VkShaderEXT shader, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_shader_object, shader_obj, shader);

   if (!shader)
      return;

   radv_shader_object_destroy(device, shader_obj, pAllocator);
}

static void
radv_shader_layout_add_set(struct radv_shader_layout *layout, uint32_t set_idx,
                           struct radv_descriptor_set_layout *set_layout)
{
   if (layout->set[set_idx].layout)
      return;

   layout->num_sets = MAX2(set_idx + 1, layout->num_sets);

   layout->set[set_idx].layout = set_layout;
   layout->set[set_idx].dynamic_offset_start = layout->dynamic_offset_count;

   layout->dynamic_offset_count += set_layout->dynamic_offset_count;
}

static void
radv_get_shader_layout(const VkShaderCreateInfoEXT *pCreateInfo, struct radv_shader_layout *layout)
{
   uint16_t dynamic_shader_stages = 0;

   layout->dynamic_offset_count = 0;

   for (uint32_t i = 0; i < pCreateInfo->setLayoutCount; i++) {
      VK_FROM_HANDLE(radv_descriptor_set_layout, set_layout, pCreateInfo->pSetLayouts[i]);

      if (set_layout == NULL)
         continue;

      radv_shader_layout_add_set(layout, i, set_layout);

      dynamic_shader_stages |= set_layout->dynamic_shader_stages;
   }

   if (layout->dynamic_offset_count && (dynamic_shader_stages & pCreateInfo->stage)) {
      layout->use_dynamic_descriptors = true;
   }

   layout->independent_sets = !!(pCreateInfo->flags & VK_SHADER_CREATE_INDEPENDENT_SETS_BIT_KHR);
}

static void
radv_merge_shader_layout(const struct radv_shader_layout *src, struct radv_shader_layout *dst)
{
   for (uint32_t s = 0; s < src->num_sets; s++) {
      if (!src->set[s].layout)
         continue;

      radv_shader_layout_add_set(dst, s, src->set[s].layout);
   }

   dst->use_dynamic_descriptors |= src->use_dynamic_descriptors;
}

static void
radv_shader_stage_init(const VkShaderCreateInfoEXT *sinfo, struct radv_shader_stage *out_stage)
{
   memset(out_stage, 0, sizeof(*out_stage));

   out_stage->stage = vk_to_mesa_shader_stage(sinfo->stage);
   out_stage->next_stage = MESA_SHADER_NONE;
   out_stage->entrypoint = sinfo->pName;
   out_stage->spec_info = sinfo->pSpecializationInfo;
   out_stage->feedback.flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT;
   out_stage->spirv.data = (const char *)sinfo->pCode;
   out_stage->spirv.size = sinfo->codeSize;

   radv_get_shader_layout(sinfo, &out_stage->layout);

   const VkShaderDescriptorSetAndBindingMappingInfoEXT *mapping =
      vk_find_struct_const(sinfo->pNext, SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT);
   out_stage->layout.mapping = mapping;

   const VkShaderRequiredSubgroupSizeCreateInfoEXT *const subgroup_size =
      vk_find_struct_const(sinfo->pNext, SHADER_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT);

   if (subgroup_size) {
      if (subgroup_size->requiredSubgroupSize == 32)
         out_stage->key.subgroup_required_size = RADV_REQUIRED_WAVE32;
      else if (subgroup_size->requiredSubgroupSize == 64)
         out_stage->key.subgroup_required_size = RADV_REQUIRED_WAVE64;
      else
         UNREACHABLE("Unsupported required subgroup size.");
   }

   if (sinfo->flags & VK_SHADER_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT) {
      out_stage->key.subgroup_require_full = 1;
   }

   if (sinfo->flags & VK_SHADER_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT) {
      out_stage->key.subgroup_allow_varying = 1;
   }

   if (sinfo->flags & VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT)
      out_stage->key.indirect_bindable = 1;

   if (sinfo->flags & VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT)
      out_stage->key.descriptor_heap = 1;

   if (out_stage->stage == MESA_SHADER_MESH) {
      out_stage->key.has_task_shader = !(sinfo->flags & VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT);
   }
}

static VkResult
radv_shader_object_init_graphics(struct radv_shader_object *shader_obj, struct radv_device *device,
                                 const VkShaderCreateInfoEXT *pCreateInfo)
{
   mesa_shader_stage stage = vk_to_mesa_shader_stage(pCreateInfo->stage);
   struct radv_shader_stage stages[MESA_VULKAN_SHADER_STAGES];

   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      stages[i].stage = MESA_SHADER_NONE;
      stages[i].gs_copy_shader = NULL;
      stages[i].nir = NULL;
      stages[i].spirv.size = 0;
      stages[i].next_stage = MESA_SHADER_NONE;
   }

   radv_shader_stage_init(pCreateInfo, &stages[stage]);

   struct radv_graphics_state_key gfx_state = {0};

   gfx_state.vs.has_prolog = true;
   gfx_state.ps.has_epilog = true;
   gfx_state.dynamic_rasterization_samples = true;
   gfx_state.dynamic_provoking_vtx_mode = true;
   gfx_state.bc250_ps_dynamic_provoking = true;
   gfx_state.dynamic_line_rast_mode = true;
   gfx_state.rs.polygon_mode_unknown = true;
   gfx_state.ps.exports_mrtz_via_epilog = true;

   for (uint32_t i = 0; i < MAX_RTS; i++)
      gfx_state.ps.epilog.color_map[i] = i;

   struct radv_shader *shader = NULL;
   struct radv_shader_binary *binary = NULL;
   VkResult result;

   if (!pCreateInfo->nextStage) {
      struct radv_shader *shaders[MESA_VULKAN_SHADER_STAGES] = {NULL};
      struct radv_shader_binary *binaries[MESA_VULKAN_SHADER_STAGES] = {NULL};
      struct radv_shader_debug_info debug[MESA_VULKAN_SHADER_STAGES] = {0};
      struct radv_shader_debug_info gs_copy_debug = {0};

      result = radv_graphics_shaders_compile(&device->compiler_info, NULL, stages, &gfx_state, false, NULL, false,
                                             debug, binaries, &gs_copy_debug, &shader_obj->gs.copy_binary);
      if (result != VK_SUCCESS)
         goto fail;
      radv_graphics_shaders_create(device, NULL, true, shaders, binaries, debug, &shader_obj->gs.copy_shader,
                                   shader_obj->gs.copy_binary, &gs_copy_debug);

      shader = shaders[stage];
      binary = binaries[stage];

      ralloc_free(stages[stage].nir);
      ralloc_free(stages[MESA_SHADER_GEOMETRY].gs_copy_shader);

      shader_obj->shader = shader;
      shader_obj->binary = binary;
   } else {
      VkShaderStageFlags next_stages = pCreateInfo->nextStage;

      /* The last VGT stage can always be used with rasterization enabled and a null fragment shader
       * (ie. depth-only rendering). Because we don't want to have two variants for NONE and
       * FRAGMENT, let's compile only one variant that works for both.
       */
      if (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_EVAL || stage == MESA_SHADER_GEOMETRY)
         next_stages |= VK_SHADER_STAGE_FRAGMENT_BIT;

      radv_foreach_stage (next_stage, next_stages) {
         struct radv_shader *shaders[MESA_VULKAN_SHADER_STAGES] = {NULL};
         struct radv_shader_binary *binaries[MESA_VULKAN_SHADER_STAGES] = {NULL};
         struct radv_shader_debug_info debug[MESA_VULKAN_SHADER_STAGES] = {0};
         struct radv_shader_debug_info gs_copy_debug = {0};

         radv_shader_stage_init(pCreateInfo, &stages[stage]);
         stages[stage].next_stage = next_stage;

         result = radv_graphics_shaders_compile(&device->compiler_info, NULL, stages, &gfx_state, false, NULL, false,
                                                debug, binaries, &gs_copy_debug, &shader_obj->gs.copy_binary);
         if (result != VK_SUCCESS)
            goto fail;
         radv_graphics_shaders_create(device, NULL, true, shaders, binaries, debug, &shader_obj->gs.copy_shader,
                                      shader_obj->gs.copy_binary, &gs_copy_debug);

         shader = shaders[stage];
         binary = binaries[stage];

         ralloc_free(stages[stage].nir);
         ralloc_free(stages[MESA_SHADER_GEOMETRY].gs_copy_shader);

         if (stage == MESA_SHADER_VERTEX) {
            if (next_stage == MESA_SHADER_TESS_CTRL) {
               shader_obj->as_ls.shader = shader;
               shader_obj->as_ls.binary = binary;
            } else if (next_stage == MESA_SHADER_GEOMETRY) {
               shader_obj->as_es.shader = shader;
               shader_obj->as_es.binary = binary;
            } else {
               shader_obj->shader = shader;
               shader_obj->binary = binary;
            }
         } else if (stage == MESA_SHADER_TESS_EVAL) {
            if (next_stage == MESA_SHADER_GEOMETRY) {
               shader_obj->as_es.shader = shader;
               shader_obj->as_es.binary = binary;
            } else {
               shader_obj->shader = shader;
               shader_obj->binary = binary;
            }
         } else {
            shader_obj->shader = shader;
            shader_obj->binary = binary;
         }
      }
   }

   return VK_SUCCESS;

fail:
   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++)
      ralloc_free(stages[i].nir);
   ralloc_free(stages[MESA_SHADER_GEOMETRY].gs_copy_shader);
   return result;
}

static VkResult
radv_shader_object_init_compute(struct radv_shader_object *shader_obj, struct radv_device *device,
                                const VkShaderCreateInfoEXT *pCreateInfo)
{
   struct radv_shader_stage stage = {0};

   radv_shader_stage_init(pCreateInfo, &stage);

   struct radv_shader_debug_info cs_dbg = {0};
   struct radv_shader_binary *cs_binary = radv_compile_cs(&device->compiler_info, &stage, false, &cs_dbg);
   struct radv_shader *cs_shader = radv_shader_create(device, NULL, cs_binary, true, &cs_dbg);

   ralloc_free(stage.nir);

   shader_obj->shader = cs_shader;
   shader_obj->binary = cs_binary;

   return VK_SUCCESS;
}

static VkResult
radv_shader_object_init_binary(struct radv_device *device, struct blob_reader *blob, struct radv_shader **shader_out,
                               struct radv_shader_binary **binary_out)
{
   const char *binary_blake3 = blob_read_bytes(blob, BLAKE3_KEY_LEN);
   const uint32_t binary_size = blob_read_uint32(blob);
   if (blob->overrun || binary_size < sizeof(struct radv_shader_binary) ||
       binary_size > (size_t)(blob->end - blob->current))
      return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
   const struct radv_shader_binary *binary = blob_read_bytes(blob, binary_size);
   unsigned char blake3[BLAKE3_KEY_LEN];

   if (binary->total_size != binary_size)
      return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
   _mesa_blake3_compute(binary, binary_size, blake3);
   if (memcmp(blake3, binary_blake3, BLAKE3_KEY_LEN))
      return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;

   *shader_out = radv_shader_create(device, NULL, binary, true, NULL);
   *binary_out = (struct radv_shader_binary *)binary;

   return VK_SUCCESS;
}

static VkResult
radv_shader_object_init(struct radv_shader_object *shader_obj, struct radv_device *device,
                        const VkShaderCreateInfoEXT *pCreateInfo)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   VkResult result;

   shader_obj->stage = vk_to_mesa_shader_stage(pCreateInfo->stage);
   shader_obj->code_type = pCreateInfo->codeType;
   if (device->bc250_env.shader_object_plan && pCreateInfo->codeType == VK_SHADER_CODE_TYPE_SPIRV_EXT &&
       (shader_obj->stage == MESA_SHADER_MESH || shader_obj->stage == MESA_SHADER_TASK))
      return VK_ERROR_FEATURE_NOT_PRESENT;

   if (pCreateInfo->codeType == VK_SHADER_CODE_TYPE_BINARY_EXT) {
      if (pCreateInfo->codeSize < VK_UUID_SIZE + sizeof(uint32_t)) {
         return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
      }

      struct blob_reader blob;
      blob_reader_init(&blob, pCreateInfo->pCode, pCreateInfo->codeSize);

      const uint8_t *cache_uuid = blob_read_bytes(&blob, VK_UUID_SIZE);

      if (memcmp(cache_uuid, pdev->cache_uuid, VK_UUID_SIZE))
         return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;

      const struct radv_shader_object_metadata *md =
         (struct radv_shader_object_metadata *)blob_read_bytes(&blob, sizeof(struct radv_shader_object_metadata));
      if (blob.overrun)
         return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;

      shader_obj->dynamic_offset_count = md->dynamic_offset_count;

      const bool has_main_binary = blob_read_uint32(&blob);

      if (has_main_binary) {
         result = radv_shader_object_init_binary(device, &blob, &shader_obj->shader, &shader_obj->binary);
         if (result != VK_SUCCESS)
            return result;
      }

      if (shader_obj->stage == MESA_SHADER_VERTEX) {
         const bool has_es_binary = blob_read_uint32(&blob);
         if (has_es_binary) {
            result =
               radv_shader_object_init_binary(device, &blob, &shader_obj->as_es.shader, &shader_obj->as_es.binary);
            if (result != VK_SUCCESS)
               return result;
         }

         const bool has_ls_binary = blob_read_uint32(&blob);
         if (has_ls_binary) {
            result =
               radv_shader_object_init_binary(device, &blob, &shader_obj->as_ls.shader, &shader_obj->as_ls.binary);
            if (result != VK_SUCCESS)
               return result;
         }
      } else if (shader_obj->stage == MESA_SHADER_TESS_EVAL) {
         const bool has_es_binary = blob_read_uint32(&blob);
         if (has_es_binary) {
            result =
               radv_shader_object_init_binary(device, &blob, &shader_obj->as_es.shader, &shader_obj->as_es.binary);
            if (result != VK_SUCCESS)
               return result;
         }
      } else if (shader_obj->stage == MESA_SHADER_GEOMETRY) {
         const bool has_gs_copy_binary = blob_read_uint32(&blob);
         if (has_gs_copy_binary) {
            result =
               radv_shader_object_init_binary(device, &blob, &shader_obj->gs.copy_shader, &shader_obj->gs.copy_binary);
            if (result != VK_SUCCESS)
               return result;
         }
      }
      if (blob.overrun)
         return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
      if (device->bc250_env.shader_object_plan) {
         if (blob_read_uint64(&blob) != RADV_BC250_OBJECT_MAGIC ||
             blob_read_uint32(&blob) != shader_obj->stage || blob_read_uint32(&blob))
            return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         blob_copy_bytes(&blob, shader_obj->bc250_route_key, sizeof(shader_obj->bc250_route_key));
         blob_copy_bytes(&blob, shader_obj->bc250_hardware_key, sizeof(shader_obj->bc250_hardware_key));
         blob_copy_bytes(&blob, shader_obj->bc250_layout_hash, sizeof(shader_obj->bc250_layout_hash));
         blob_copy_bytes(&blob, &shader_obj->bc250_plan, sizeof(shader_obj->bc250_plan));
         if (blob.overrun)
            return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         uint8_t hash[32];
         radv_bc250_object_layout_hash(pCreateInfo, hash);
         if (memcmp(hash, shader_obj->bc250_layout_hash, sizeof(hash)))
            return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         _mesa_blake3_compute(blob.data, blob.current - blob.data, hash);
         const uint8_t *stored_hash = blob_read_bytes(&blob, sizeof(hash));
         if (blob.overrun || blob.current != blob.end || memcmp(hash, stored_hash, sizeof(hash)) ||
             memcmp(shader_obj->bc250_route_key, &device->compiler_info.key, sizeof(shader_obj->bc250_route_key)) ||
             memcmp(shader_obj->bc250_hardware_key, &device->compiler_info.hw, sizeof(shader_obj->bc250_hardware_key)))
            return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         if (shader_obj->bc250_plan.version) {
            const struct radv_bc250_pipeline_plan *plan = &shader_obj->bc250_plan;
            const uint8_t *executable_hash = shader_obj->stage == MESA_SHADER_MESH ? plan->mesh_hash : plan->fragment_hash;
            if (!radv_bc250_pipeline_plan_valid(plan) || !shader_obj->shader ||
                (shader_obj->stage != MESA_SHADER_MESH && shader_obj->stage != MESA_SHADER_FRAGMENT) ||
                (plan->flags & RADV_BC250_PLAN_TASK) || plan->direct_pieces || plan->split_pieces ||
                memcmp(executable_hash, shader_obj->shader->hash, 32) ||
                memcmp(plan->route_key, shader_obj->bc250_route_key, sizeof(plan->route_key)) ||
                memcmp(plan->hardware_key, shader_obj->bc250_hardware_key, sizeof(plan->hardware_key)))
               return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         } else if (shader_obj->stage == MESA_SHADER_MESH || shader_obj->stage == MESA_SHADER_TASK) {
            return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         }
         shader_obj->bc250_policy_valid = true;
      } else if (blob.current != blob.end) {
         /* A portable object must not silently lose its ownership checks when
          * imported with the experimental switch disabled. */
         if (blob_read_uint64(&blob) == RADV_BC250_OBJECT_MAGIC)
            return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
      }
   } else {
      struct radv_shader_layout layout = {0};

      assert(pCreateInfo->codeType == VK_SHADER_CODE_TYPE_SPIRV_EXT);

      radv_get_shader_layout(pCreateInfo, &layout);

      shader_obj->dynamic_offset_count = layout.dynamic_offset_count;

      if (pCreateInfo->stage == VK_SHADER_STAGE_COMPUTE_BIT) {
         result = radv_shader_object_init_compute(shader_obj, device, pCreateInfo);
      } else {
         result = radv_shader_object_init_graphics(shader_obj, device, pCreateInfo);
      }

      if (result != VK_SUCCESS)
         return result;
      if (device->bc250_env.shader_object_plan) {
         radv_bc250_shader_object_policy(device, shader_obj);
         radv_bc250_object_layout_hash(pCreateInfo, shader_obj->bc250_layout_hash);
      }
   }

   return VK_SUCCESS;
}

static VkResult
radv_shader_object_create(VkDevice _device, const VkShaderCreateInfoEXT *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator, VkShaderEXT *pShader)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   struct radv_shader_object *shader_obj;
   VkResult result;

   shader_obj = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*shader_obj), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (shader_obj == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   vk_object_base_init(&device->vk, &shader_obj->base, VK_OBJECT_TYPE_SHADER_EXT);

   result = radv_shader_object_init(shader_obj, device, pCreateInfo);
   if (result != VK_SUCCESS) {
      radv_shader_object_destroy(device, shader_obj, pAllocator);
      return result;
   }

   radv_bc250_report_mesh_route(device, shader_obj->shader, "shader_object", 0, false, false);
   *pShader = radv_shader_object_to_handle(shader_obj);

   return VK_SUCCESS;
}

static VkResult
radv_shader_object_create_linked(VkDevice _device, uint32_t createInfoCount, const VkShaderCreateInfoEXT *pCreateInfos,
                                 const VkAllocationCallbacks *pAllocator, VkShaderEXT *pShaders)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   struct radv_shader_stage stages[MESA_VULKAN_SHADER_STAGES];

   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      stages[i].stage = MESA_SHADER_NONE;
      stages[i].gs_copy_shader = NULL;
      stages[i].nir = NULL;
      stages[i].spirv.size = 0;
      stages[i].next_stage = MESA_SHADER_NONE;
   }

   struct radv_graphics_state_key gfx_state = {0};

   gfx_state.vs.has_prolog = true;
   gfx_state.ps.has_epilog = true;
   gfx_state.dynamic_rasterization_samples = true;
   gfx_state.dynamic_provoking_vtx_mode = true;
   gfx_state.bc250_ps_dynamic_provoking = true;
   gfx_state.dynamic_line_rast_mode = true;
   gfx_state.rs.polygon_mode_unknown = true;
   gfx_state.ps.exports_mrtz_via_epilog = true;

   for (uint32_t i = 0; i < MAX_RTS; i++)
      gfx_state.ps.epilog.color_map[i] = i;

   for (unsigned i = 0; i < createInfoCount; i++) {
      const VkShaderCreateInfoEXT *pCreateInfo = &pCreateInfos[i];
      mesa_shader_stage s = vk_to_mesa_shader_stage(pCreateInfo->stage);

      if (radv_device_physical(device)->bc250_native_mesh && s == MESA_SHADER_TASK)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      radv_shader_stage_init(pCreateInfo, &stages[s]);
   }

   if (device->bc250_env.shader_object_plan && stages[MESA_SHADER_MESH].stage == MESA_SHADER_MESH) {
      /* The first object proof is restricted to this state at every draw.
       * Keeping unknown provoking/polygon state would prevent the existing
       * owned-corner route from proving its output capacity. */
      gfx_state.dynamic_provoking_vtx_mode = false;
      gfx_state.bc250_ps_dynamic_provoking = false;
      gfx_state.rs.polygon_mode_unknown = false;
      gfx_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;
   }

   /* Determine next stage. */
   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      if (stages[i].stage == MESA_SHADER_NONE)
         continue;

      switch (stages[i].stage) {
      case MESA_SHADER_VERTEX:
         if (stages[MESA_SHADER_TESS_CTRL].stage != MESA_SHADER_NONE) {
            stages[i].next_stage = MESA_SHADER_TESS_CTRL;
         } else if (stages[MESA_SHADER_GEOMETRY].stage != MESA_SHADER_NONE) {
            stages[i].next_stage = MESA_SHADER_GEOMETRY;
         } else if (stages[MESA_SHADER_FRAGMENT].stage != MESA_SHADER_NONE) {
            stages[i].next_stage = MESA_SHADER_FRAGMENT;
         }
         break;
      case MESA_SHADER_TESS_CTRL:
         stages[i].next_stage = MESA_SHADER_TESS_EVAL;
         break;
      case MESA_SHADER_TESS_EVAL:
         if (stages[MESA_SHADER_GEOMETRY].stage != MESA_SHADER_NONE) {
            stages[i].next_stage = MESA_SHADER_GEOMETRY;
         } else if (stages[MESA_SHADER_FRAGMENT].stage != MESA_SHADER_NONE) {
            stages[i].next_stage = MESA_SHADER_FRAGMENT;
         }
         break;
      case MESA_SHADER_GEOMETRY:
      case MESA_SHADER_MESH:
         if (stages[MESA_SHADER_FRAGMENT].stage != MESA_SHADER_NONE) {
            stages[i].next_stage = MESA_SHADER_FRAGMENT;
         }
         break;
      case MESA_SHADER_FRAGMENT:
         stages[i].next_stage = MESA_SHADER_NONE;
         break;
      case MESA_SHADER_TASK:
         stages[i].next_stage = MESA_SHADER_MESH;
         break;
      default:
         assert(0);
      }

      if (stages[i].layout.independent_sets) {
         /* Merge layouts for merged stages with independent sets. */
         if (stages[i].stage == MESA_SHADER_VERTEX) {
            if (stages[i].next_stage == MESA_SHADER_TESS_CTRL) {
               radv_merge_shader_layout(&stages[MESA_SHADER_VERTEX].layout, &stages[MESA_SHADER_TESS_CTRL].layout);
            } else if (stages[i].next_stage == MESA_SHADER_GEOMETRY) {
               radv_merge_shader_layout(&stages[MESA_SHADER_VERTEX].layout, &stages[MESA_SHADER_GEOMETRY].layout);
            }
         }

         if (stages[i].stage == MESA_SHADER_TESS_EVAL && stages[i].next_stage == MESA_SHADER_GEOMETRY) {
            radv_merge_shader_layout(&stages[MESA_SHADER_TESS_EVAL].layout, &stages[MESA_SHADER_GEOMETRY].layout);
         }
      }
   }

   struct radv_shader *shaders[MESA_VULKAN_SHADER_STAGES] = {NULL};
   struct radv_shader_binary *binaries[MESA_VULKAN_SHADER_STAGES] = {NULL};
   struct radv_shader_debug_info debug[MESA_VULKAN_SHADER_STAGES] = {0};
   struct radv_shader *gs_copy_shader = NULL;
   struct radv_shader_binary *gs_copy_binary = NULL;
   struct radv_shader_debug_info gs_copy_debug = {0};

   VkResult compile_result = radv_graphics_shaders_compile(&device->compiler_info, NULL, stages, &gfx_state, false, NULL, false, debug, binaries,
                                 &gs_copy_debug, &gs_copy_binary);
   if (compile_result != VK_SUCCESS) {
      for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++)
         ralloc_free(stages[s].nir);
      return compile_result;
   }
   radv_graphics_shaders_create(device, NULL, true, shaders, binaries, debug, &gs_copy_shader, gs_copy_binary,
                                &gs_copy_debug);

   struct radv_bc250_pipeline_plan bc250_plan = {0};
   if (device->bc250_env.shader_object_plan && stages[MESA_SHADER_MESH].stage != MESA_SHADER_NONE) {
      struct radv_shader_stage *ms = &stages[MESA_SHADER_MESH];
      if (!shaders[MESA_SHADER_MESH] || !shaders[MESA_SHADER_FRAGMENT] ||
          !shaders[MESA_SHADER_MESH]->info.ms.bc250_safe_direct || ms->key.has_task_shader ||
          ms->bc250_split_pieces || ms->bc250_task_replay || ms->bc250_ordered_export) {
         for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
            if (shaders[s])
               radv_shader_unref(device, shaders[s]);
            free(binaries[s]);
            ralloc_free(stages[s].nir);
         }
         if (gs_copy_shader)
            radv_shader_unref(device, gs_copy_shader);
         free(gs_copy_binary);
         ralloc_free(stages[MESA_SHADER_GEOMETRY].gs_copy_shader);
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      bc250_plan = (struct radv_bc250_pipeline_plan) {
         .version = RADV_BC250_PIPELINE_PLAN_VERSION,
         .flags = ms->bc250_safe_owned ? RADV_BC250_PLAN_CORNERS : 0,
         .bary_ref_mask = ms->bc250_bary_ref_mask,
         .per_primitive_locations = ms->bc250_pp_locations,
         .bary_slots = ms->bc250_bary_ref_mask ? (ms->bc250_bary_raw_slot + 1) |
            ((ms->bc250_bary_flat_slot + 1) << 8) : 0,
      };
      memcpy(bc250_plan.route_key, &device->compiler_info.key, sizeof(bc250_plan.route_key));
      memcpy(bc250_plan.hardware_key, &device->compiler_info.hw, sizeof(bc250_plan.hardware_key));
      memcpy(bc250_plan.mesh_hash, shaders[MESA_SHADER_MESH]->hash, sizeof(bc250_plan.mesh_hash));
      memcpy(bc250_plan.fragment_hash, shaders[MESA_SHADER_FRAGMENT]->hash, sizeof(bc250_plan.fragment_hash));
      _mesa_blake3_compute(&gfx_state, sizeof(gfx_state), bc250_plan.state_hash);
   }

   for (unsigned i = 0; i < createInfoCount; i++) {
      const VkShaderCreateInfoEXT *pCreateInfo = &pCreateInfos[i];
      mesa_shader_stage s = vk_to_mesa_shader_stage(pCreateInfo->stage);
      struct radv_shader_object *shader_obj;

      shader_obj = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*shader_obj), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (shader_obj == NULL)
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

      vk_object_base_init(&device->vk, &shader_obj->base, VK_OBJECT_TYPE_SHADER_EXT);

      shader_obj->stage = s;
      shader_obj->code_type = pCreateInfo->codeType;
      shader_obj->dynamic_offset_count = stages[s].layout.dynamic_offset_count;
      if (device->bc250_env.shader_object_plan) {
         radv_bc250_shader_object_policy(device, shader_obj);
         radv_bc250_object_layout_hash(pCreateInfo, shader_obj->bc250_layout_hash);
         if (s == MESA_SHADER_MESH || s == MESA_SHADER_FRAGMENT)
            shader_obj->bc250_plan = bc250_plan;
      }

      if (s == MESA_SHADER_VERTEX) {
         if (stages[s].next_stage == MESA_SHADER_TESS_CTRL) {
            shader_obj->as_ls.shader = shaders[s];
            shader_obj->as_ls.binary = binaries[s];
         } else if (stages[s].next_stage == MESA_SHADER_GEOMETRY) {
            shader_obj->as_es.shader = shaders[s];
            shader_obj->as_es.binary = binaries[s];
         } else {
            shader_obj->shader = shaders[s];
            shader_obj->binary = binaries[s];
         }
      } else if (s == MESA_SHADER_TESS_EVAL) {
         if (stages[s].next_stage == MESA_SHADER_GEOMETRY) {
            shader_obj->as_es.shader = shaders[s];
            shader_obj->as_es.binary = binaries[s];
         } else {
            shader_obj->shader = shaders[s];
            shader_obj->binary = binaries[s];
         }
      } else {
         shader_obj->shader = shaders[s];
         shader_obj->binary = binaries[s];
      }

      if (s == MESA_SHADER_GEOMETRY) {
         shader_obj->gs.copy_shader = gs_copy_shader;
         shader_obj->gs.copy_binary = gs_copy_binary;
      }

      ralloc_free(stages[s].nir);

      radv_bc250_report_mesh_route(device, shader_obj->shader, "linked_shader_object", 0, false, false);
      pShaders[i] = radv_shader_object_to_handle(shader_obj);
   }

   ralloc_free(stages[MESA_SHADER_GEOMETRY].gs_copy_shader);

   return VK_SUCCESS;
}

struct stage_idx {
   mesa_shader_stage stage;
   uint32_t idx;
};

VKAPI_ATTR VkResult VKAPI_CALL
radv_CreateShadersEXT(VkDevice _device, uint32_t createInfoCount, const VkShaderCreateInfoEXT *pCreateInfos,
                      const VkAllocationCallbacks *pAllocator, VkShaderEXT *pShaders)
{
   VkResult final_result = VK_SUCCESS;

   /* From the Vulkan 1.3.274 spec:
    *
    *    "When this function returns, whether or not it succeeds, it is
    *    guaranteed that every element of pShaders will have been overwritten
    *    by either VK_NULL_HANDLE or a valid VkShaderEXT handle."
    *
    * Zeroing up-front makes the error path easier.
    */
   memset(pShaders, 0, createInfoCount * sizeof(*pShaders));

   VkShaderStageFlagBits linked_stages = 0;

   for (uint32_t i = 0; i < createInfoCount; i++) {
      const VkShaderCreateInfoEXT *pCreateInfo = &pCreateInfos[i];

      if (pCreateInfo->codeType == VK_SHADER_CODE_TYPE_SPIRV_EXT &&
          (pCreateInfo->flags & VK_SHADER_CREATE_LINK_STAGE_BIT_EXT)) {
         linked_stages |= pCreateInfo->stage;
      }
   }

   uint32_t linked_count = 0;
   struct stage_idx linked[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];

   for (uint32_t i = 0; i < createInfoCount; i++) {
      const VkShaderCreateInfoEXT *pCreateInfo = &pCreateInfos[i];
      VkResult result = VK_SUCCESS;

      switch (pCreateInfo->codeType) {
      case VK_SHADER_CODE_TYPE_BINARY_EXT: {
         result = radv_shader_object_create(_device, &pCreateInfos[i], pAllocator, &pShaders[i]);
         break;
      }
      case VK_SHADER_CODE_TYPE_SPIRV_EXT: {
         bool is_linking_enabled = !!(pCreateInfo->flags & VK_SHADER_CREATE_LINK_STAGE_BIT_EXT);

         /* Force disable shaders linking when the next stage of VS/TES isn't present because the
          * driver would need to compile all shaders twice due to shader variants. This is probably
          * less optimal than compiling unlinked shaders.
          */
         if ((pCreateInfo->stage & VK_SHADER_STAGE_VERTEX_BIT) &&
             (pCreateInfo->nextStage & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_GEOMETRY_BIT)) &&
             !(linked_stages & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_GEOMETRY_BIT)))
            is_linking_enabled = false;

         if ((pCreateInfo->stage & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) &&
             (pCreateInfo->nextStage & VK_SHADER_STAGE_GEOMETRY_BIT) && !(linked_stages & VK_SHADER_STAGE_GEOMETRY_BIT))
            is_linking_enabled = false;

         if (is_linking_enabled) {
            /* Stash it and compile later */
            assert(linked_count < ARRAY_SIZE(linked));
            linked[linked_count++] = (struct stage_idx){
               .stage = vk_to_mesa_shader_stage(pCreateInfo->stage),
               .idx = i,
            };
         } else {
            result = radv_shader_object_create(_device, &pCreateInfos[i], pAllocator, &pShaders[i]);
         }
         break;
      }
      default:
         UNREACHABLE("Unknown shader code type");
      }

      if (final_result == VK_SUCCESS)
         final_result = result;
   }

   if (linked_count > 0) {
      VkShaderCreateInfoEXT linked_infos[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];
      VkShaderEXT linked_shaders[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];
      VkResult result = VK_SUCCESS;

      for (uint32_t l = 0; l < linked_count; l++)
         linked_infos[l] = pCreateInfos[linked[l].idx];

      result = radv_shader_object_create_linked(_device, linked_count, linked_infos, pAllocator, linked_shaders);
      if (result == VK_SUCCESS) {
         for (uint32_t l = 0; l < linked_count; l++)
            pShaders[linked[l].idx] = linked_shaders[l];
      }

      if (final_result == VK_SUCCESS)
         final_result = result;
   }

   return final_result;
}

static size_t
radv_get_shader_binary_size(const struct radv_shader_binary *binary)
{
   size_t size = sizeof(uint32_t); /* has_binary */

   if (binary)
      size += BLAKE3_KEY_LEN + 4 + align(binary->total_size, 4);

   return size;
}

static size_t
radv_get_shader_object_size(const struct radv_shader_object *shader_obj)
{
   size_t size = VK_UUID_SIZE;

   size += sizeof(struct radv_shader_object_metadata);
   size += radv_get_shader_binary_size(shader_obj->binary);

   if (shader_obj->stage == MESA_SHADER_VERTEX) {
      size += radv_get_shader_binary_size(shader_obj->as_es.binary);
      size += radv_get_shader_binary_size(shader_obj->as_ls.binary);
   } else if (shader_obj->stage == MESA_SHADER_TESS_EVAL) {
      size += radv_get_shader_binary_size(shader_obj->as_es.binary);
   } else if (shader_obj->stage == MESA_SHADER_GEOMETRY) {
      size += radv_get_shader_binary_size(shader_obj->gs.copy_binary);
   }

   if (shader_obj->bc250_policy_valid)
      size = align(size, 8) + 16 + sizeof(shader_obj->bc250_route_key) +
         sizeof(shader_obj->bc250_hardware_key) + sizeof(shader_obj->bc250_layout_hash) +
         sizeof(shader_obj->bc250_plan) + 32;

   return size;
}

static void
radv_write_shader_object_metadata(struct blob *blob, const struct radv_shader_object *shader_obj)
{
   struct radv_shader_object_metadata md = {
      .dynamic_offset_count = shader_obj->dynamic_offset_count,
   };

   blob_write_bytes(blob, &md, sizeof(md));
}

static void
radv_write_shader_binary(struct blob *blob, const struct radv_shader_binary *binary)
{
   unsigned char binary_blake3[BLAKE3_KEY_LEN];

   blob_write_uint32(blob, !!binary);

   if (binary) {
      _mesa_blake3_compute(binary, binary->total_size, binary_blake3);

      blob_write_bytes(blob, binary_blake3, sizeof(binary_blake3));
      blob_write_uint32(blob, binary->total_size);
      blob_write_bytes(blob, binary, binary->total_size);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetShaderBinaryDataEXT(VkDevice _device, VkShaderEXT shader, size_t *pDataSize, void *pData)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_shader_object, shader_obj, shader);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const size_t size = radv_get_shader_object_size(shader_obj);

   if (!pData) {
      *pDataSize = size;
      return VK_SUCCESS;
   }

   if (*pDataSize < size) {
      *pDataSize = 0;
      return VK_INCOMPLETE;
   }

   struct blob blob;
   blob_init_fixed(&blob, pData, *pDataSize);
   blob_write_bytes(&blob, pdev->cache_uuid, VK_UUID_SIZE);

   radv_write_shader_object_metadata(&blob, shader_obj);
   radv_write_shader_binary(&blob, shader_obj->binary);

   if (shader_obj->stage == MESA_SHADER_VERTEX) {
      radv_write_shader_binary(&blob, shader_obj->as_es.binary);
      radv_write_shader_binary(&blob, shader_obj->as_ls.binary);
   } else if (shader_obj->stage == MESA_SHADER_TESS_EVAL) {
      radv_write_shader_binary(&blob, shader_obj->as_es.binary);
   } else if (shader_obj->stage == MESA_SHADER_GEOMETRY) {
      radv_write_shader_binary(&blob, shader_obj->gs.copy_binary);
   }

   if (shader_obj->bc250_policy_valid) {
      blob_write_uint64(&blob, RADV_BC250_OBJECT_MAGIC);
      blob_write_uint32(&blob, shader_obj->stage);
      blob_write_uint32(&blob, 0);
      blob_write_bytes(&blob, shader_obj->bc250_route_key, sizeof(shader_obj->bc250_route_key));
      blob_write_bytes(&blob, shader_obj->bc250_hardware_key, sizeof(shader_obj->bc250_hardware_key));
      blob_write_bytes(&blob, shader_obj->bc250_layout_hash, sizeof(shader_obj->bc250_layout_hash));
      blob_write_bytes(&blob, &shader_obj->bc250_plan, sizeof(shader_obj->bc250_plan));
      uint8_t hash[32];
      _mesa_blake3_compute(blob.data, blob.size, hash);
      blob_write_bytes(&blob, hash, sizeof(hash));
   }

   assert(!blob.out_of_memory);
   if (shader_obj->bc250_policy_valid)
      *pDataSize = blob.size;

   return VK_SUCCESS;
}

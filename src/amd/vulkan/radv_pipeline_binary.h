/*
 * Copyright © 2024 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_PIPELINE_BINARY_H
#define RADV_PIPELINE_BINARY_H

#include "vk_object.h"

#include "util/mesa-blake3.h"
#include "util/mesa-blake3.h"

struct radv_device;
struct radv_ray_tracing_stage_info;
struct radv_shader;
struct util_dynarray;
struct vk_pipeline_cache_object;
struct radv_bc250_pipeline_plan;
struct radv_graphics_pipeline;
struct radv_pipeline_layout;

struct radv_pipeline_binary {
   struct vk_object_base base;

   blake3_hash key;
   void *data;
   size_t size;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(radv_pipeline_binary, base, VkPipelineBinaryKHR, VK_OBJECT_TYPE_PIPELINE_BINARY_KHR)

bool radv_bc250_pipeline_binary_is_plan(const struct radv_pipeline_binary *binary);
bool radv_bc250_shader_binary_valid(const void *data, size_t size, unsigned stage);
VkResult radv_bc250_pipeline_binary_restore(struct radv_device *device,
                                           struct radv_graphics_pipeline *pipeline,
                                           const struct radv_pipeline_layout *layout,
                                           const struct radv_pipeline_binary *binary);
VkResult radv_create_pipeline_binary_from_bc250_plan(struct radv_device *device,
                                                    const VkAllocationCallbacks *allocator,
                                                    const struct radv_bc250_pipeline_plan *plan,
                                                    struct radv_shader *producer, struct radv_shader *setup,
                                                    struct util_dynarray *binaries, uint32_t *num_binaries);

VkResult radv_create_pipeline_binary_from_shader(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                                                 struct radv_shader *shader, struct util_dynarray *pipeline_binaries,
                                                 uint32_t *num_binaries);

VkResult radv_create_pipeline_binary_from_rt_shader(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                                                    struct radv_shader *shader, bool is_traversal_shader,
                                                    const uint8_t stage_blake3[BLAKE3_KEY_LEN],
                                                    const struct radv_ray_tracing_stage_info *rt_stage_info,
                                                    uint32_t stack_size, struct vk_pipeline_cache_object *nir,
                                                    struct util_dynarray *pipeline_binaries, uint32_t *num_binaries);

#endif /* RADV_PIPELINE_BINARY_H */

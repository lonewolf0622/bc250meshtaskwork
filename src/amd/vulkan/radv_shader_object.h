/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_SHADER_OBJECT_H
#define RADV_SHADER_OBJECT_H

#include "radv_shader.h"
#include "radv_bc250_pipeline_plan.h"

struct radv_bc250_shader_object_context;
struct radv_graphics_pipeline;

struct radv_shader_object {
   struct vk_object_base base;

   mesa_shader_stage stage;

   VkShaderCodeTypeEXT code_type;

   /* Main shader */
   struct radv_shader *shader;
   struct radv_shader_binary *binary;

   /* Shader variants */
   /* VS before TCS */
   struct {
      struct radv_shader *shader;
      struct radv_shader_binary *binary;
   } as_ls;

   /* VS/TES before GS */
   struct {
      struct radv_shader *shader;
      struct radv_shader_binary *binary;
   } as_es;

   /* GS copy shader */
   struct {
      struct radv_shader *copy_shader;
      struct radv_shader_binary *copy_binary;
   } gs;

   uint32_t dynamic_offset_count;

   bool bc250_policy_valid;
   uint8_t bc250_route_key[24];
   uint8_t bc250_hardware_key[8];
   uint8_t bc250_extended_key[4];
   uint8_t bc250_layout_hash[32];
   struct radv_bc250_pipeline_plan bc250_plan;
   struct radv_bc250_shader_object_context *bc250_context;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(radv_shader_object, base, VkShaderEXT, VK_OBJECT_TYPE_SHADER_EXT);

struct radv_graphics_pipeline *radv_bc250_shader_object_pipeline(const struct radv_shader_object *object);

#endif /* RADV_SHADER_OBJECT_H */

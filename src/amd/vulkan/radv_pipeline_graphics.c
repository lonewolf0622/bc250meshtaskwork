/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "meta/radv_meta.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "nir/nir_serialize.h"
#include "nir/nir_xfb_info.h"
#include "nir/radv_nir.h"
#include "tools/radv_rmv.h"
#include "util/mesa-blake3.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "radv_device.h"
#include "radv_bc250.h"
#include "radv_entrypoints.h"
#include "radv_formats.h"
#include "radv_physical_device.h"
#include "radv_pipeline_binary.h"
#include "radv_pipeline_cache.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "shader_enums.h"
#include "vk_pipeline.h"
#include "vk_util.h"

#include "ac_binary.h"
#include "ac_formats.h"
#include "ac_nir.h"
#include "ac_shader_util.h"
#include "aco_interface.h"

static bool
radv_is_static_vrs_enabled(const struct vk_graphics_pipeline_state *state)
{
   if (!state->fsr)
      return false;

   return state->fsr->fragment_size.width != 1 || state->fsr->fragment_size.height != 1 ||
          state->fsr->combiner_ops[0] != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR ||
          state->fsr->combiner_ops[1] != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
}

static bool
radv_pipeline_has_ds_attachments(const struct vk_render_pass_state *rp)
{
   return rp->depth_attachment_format != VK_FORMAT_UNDEFINED || rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;
}

static bool
radv_pipeline_has_color_attachments(const struct vk_render_pass_state *rp)
{
   for (uint32_t i = 0; i < rp->color_attachment_count; ++i) {
      if (rp->color_attachment_formats[i] != VK_FORMAT_UNDEFINED)
         return true;
   }

   return false;
}

/**
 * Get rid of DST in the blend factors by commuting the operands:
 *    func(src * DST, dst * 0) ---> func(src * 0, dst * SRC)
 */
void
radv_blend_remove_dst(VkBlendOp *func, VkBlendFactor *src_factor, VkBlendFactor *dst_factor, VkBlendFactor expected_dst,
                      VkBlendFactor replacement_src)
{
   if (*src_factor == expected_dst && *dst_factor == VK_BLEND_FACTOR_ZERO) {
      *src_factor = VK_BLEND_FACTOR_ZERO;
      *dst_factor = replacement_src;

      /* Commuting the operands requires reversing subtractions. */
      if (*func == VK_BLEND_OP_SUBTRACT)
         *func = VK_BLEND_OP_REVERSE_SUBTRACT;
      else if (*func == VK_BLEND_OP_REVERSE_SUBTRACT)
         *func = VK_BLEND_OP_SUBTRACT;
   }
}

static unsigned
radv_choose_spi_color_format(const struct radv_compiler_info *compiler_info, VkFormat vk_format, bool blend_enable,
                             bool blend_need_alpha)
{
   const struct util_format_description *desc = radv_format_description(vk_format);
   bool use_rbplus = compiler_info->hw.rbplus_allowed;
   struct ac_spi_color_formats formats = {0};
   unsigned format, ntype, swap;

   format = ac_get_cb_format(compiler_info->ac->gfx_level, desc->format);
   ntype = ac_get_cb_number_type(desc->format);
   swap = ac_translate_colorswap(compiler_info->ac->gfx_level, desc->format, false);

   ac_choose_spi_color_formats(format, swap, ntype, false, use_rbplus, &formats);

   if (blend_enable && blend_need_alpha)
      return formats.blend_alpha;
   else if (blend_need_alpha)
      return formats.alpha;
   else if (blend_enable)
      return formats.blend;
   else
      return formats.normal;
}

static bool
format_is_int8(VkFormat format)
{
   const struct util_format_description *desc = radv_format_description(format);
   int channel = vk_format_get_first_non_void_channel(format);

   return channel >= 0 && desc->channel[channel].pure_integer && desc->channel[channel].size == 8;
}

static bool
format_is_int10(VkFormat format)
{
   const struct util_format_description *desc = radv_format_description(format);

   if (desc->nr_channels != 4)
      return false;
   for (unsigned i = 0; i < 4; i++) {
      if (desc->channel[i].pure_integer && desc->channel[i].size == 10)
         return true;
   }
   return false;
}

static bool
format_is_float32(VkFormat format)
{
   const struct util_format_description *desc = radv_format_description(format);
   int channel = vk_format_get_first_non_void_channel(format);

   return channel >= 0 && desc->channel[channel].type == UTIL_FORMAT_TYPE_FLOAT && desc->channel[channel].size == 32;
}

static bool
format_ignores_signed_zero(VkFormat format)
{
   const struct util_format_description *desc = radv_format_description(format);

   /* Unsigned float formats don't care about signed zeros. */
   if (desc->format == PIPE_FORMAT_R11G11B10_FLOAT || desc->format == PIPE_FORMAT_R9G9B9E5_FLOAT)
      return true;

   for (unsigned i = 0; i < desc->nr_channels; i++) {
      if (desc->channel[i].pure_integer || desc->channel[i].type == UTIL_FORMAT_TYPE_FLOAT)
         return false;
   }

   return true;
}

static bool
radv_pipeline_needs_ps_epilog(const struct vk_graphics_pipeline_state *state,
                              VkGraphicsPipelineLibraryFlagBitsEXT lib_flags)
{
   /* Use a PS epilog when the fragment shader is compiled without the fragment output interface. */
   if ((state->shader_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
       (lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) &&
       !(lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT))
      return true;

   /* These dynamic states need to compile PS epilogs on-demand. */
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ENABLES) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_WRITE_MASKS) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE))
      return true;

   return false;
}

static bool
radv_pipeline_uses_vrs_attachment(const struct radv_graphics_pipeline *pipeline,
                                  const struct vk_graphics_pipeline_state *state)
{
   VkPipelineCreateFlags2 create_flags = pipeline->base.create_flags;
   if (state->rp)
      create_flags |= state->pipeline_flags;

   return (create_flags & VK_PIPELINE_CREATE_2_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR) != 0;
}

static void
radv_pipeline_init_multisample_state(const struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                     const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                     const struct vk_graphics_pipeline_state *state)
{
   struct radv_multisample_state *ms = &pipeline->ms;

   /* From the Vulkan 1.1.129 spec, 26.7. Sample Shading:
    *
    * "Sample shading is enabled for a graphics pipeline:
    *
    * - If the interface of the fragment shader entry point of the
    *   graphics pipeline includes an input variable decorated
    *   with SampleId or SamplePosition. In this case
    *   minSampleShadingFactor takes the value 1.0.
    * - Else if the sampleShadingEnable member of the
    *   VkPipelineMultisampleStateCreateInfo structure specified
    *   when creating the graphics pipeline is set to VK_TRUE. In
    *   this case minSampleShadingFactor takes the value of
    *   VkPipelineMultisampleStateCreateInfo::minSampleShading.
    *
    * Otherwise, sample shading is considered disabled."
    */
   if (state->ms && state->ms->sample_shading_enable) {
      ms->sample_shading_enable = true;
      ms->min_sample_shading = state->ms->min_sample_shading;
   }
}

static uint64_t
radv_dynamic_state_mask(VkDynamicState state)
{
   switch (state) {
   case VK_DYNAMIC_STATE_VIEWPORT:
      return RADV_DYNAMIC_VIEWPORT;
   case VK_DYNAMIC_STATE_SCISSOR:
      return RADV_DYNAMIC_SCISSOR;
   case VK_DYNAMIC_STATE_LINE_WIDTH:
      return RADV_DYNAMIC_LINE_WIDTH;
   case VK_DYNAMIC_STATE_DEPTH_BIAS:
      return RADV_DYNAMIC_DEPTH_BIAS;
   case VK_DYNAMIC_STATE_BLEND_CONSTANTS:
      return RADV_DYNAMIC_BLEND_CONSTANTS;
   case VK_DYNAMIC_STATE_DEPTH_BOUNDS:
      return RADV_DYNAMIC_DEPTH_BOUNDS;
   case VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK:
      return RADV_DYNAMIC_STENCIL_COMPARE_MASK;
   case VK_DYNAMIC_STATE_STENCIL_WRITE_MASK:
      return RADV_DYNAMIC_STENCIL_WRITE_MASK;
   case VK_DYNAMIC_STATE_STENCIL_REFERENCE:
      return RADV_DYNAMIC_STENCIL_REFERENCE;
   case VK_DYNAMIC_STATE_DISCARD_RECTANGLE_EXT:
      return RADV_DYNAMIC_DISCARD_RECTANGLE;
   case VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_EXT:
      return RADV_DYNAMIC_SAMPLE_LOCATIONS;
   case VK_DYNAMIC_STATE_LINE_STIPPLE:
      return RADV_DYNAMIC_LINE_STIPPLE;
   case VK_DYNAMIC_STATE_CULL_MODE:
      return RADV_DYNAMIC_CULL_MODE;
   case VK_DYNAMIC_STATE_FRONT_FACE:
      return RADV_DYNAMIC_FRONT_FACE;
   case VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY:
      return RADV_DYNAMIC_PRIMITIVE_TOPOLOGY;
   case VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE:
      return RADV_DYNAMIC_DEPTH_TEST_ENABLE;
   case VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE:
      return RADV_DYNAMIC_DEPTH_WRITE_ENABLE;
   case VK_DYNAMIC_STATE_DEPTH_COMPARE_OP:
      return RADV_DYNAMIC_DEPTH_COMPARE_OP;
   case VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE:
      return RADV_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE;
   case VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE:
      return RADV_DYNAMIC_STENCIL_TEST_ENABLE;
   case VK_DYNAMIC_STATE_STENCIL_OP:
      return RADV_DYNAMIC_STENCIL_OP;
   case VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE:
      return RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE;
   case VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR:
      return RADV_DYNAMIC_FRAGMENT_SHADING_RATE;
   case VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT:
      return RADV_DYNAMIC_PATCH_CONTROL_POINTS;
   case VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE:
      return RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE;
   case VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE:
      return RADV_DYNAMIC_DEPTH_BIAS_ENABLE;
   case VK_DYNAMIC_STATE_LOGIC_OP_EXT:
      return RADV_DYNAMIC_LOGIC_OP;
   case VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE:
      return RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE;
   case VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT:
      return RADV_DYNAMIC_COLOR_WRITE_ENABLE;
   case VK_DYNAMIC_STATE_VERTEX_INPUT_EXT:
      return RADV_DYNAMIC_VERTEX_INPUT;
   case VK_DYNAMIC_STATE_POLYGON_MODE_EXT:
      return RADV_DYNAMIC_POLYGON_MODE;
   case VK_DYNAMIC_STATE_TESSELLATION_DOMAIN_ORIGIN_EXT:
      return RADV_DYNAMIC_TESS_DOMAIN_ORIGIN;
   case VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT:
      return RADV_DYNAMIC_LOGIC_OP_ENABLE;
   case VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT:
      return RADV_DYNAMIC_LINE_STIPPLE_ENABLE;
   case VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT:
      return RADV_DYNAMIC_ALPHA_TO_COVERAGE_ENABLE;
   case VK_DYNAMIC_STATE_SAMPLE_MASK_EXT:
      return RADV_DYNAMIC_SAMPLE_MASK;
   case VK_DYNAMIC_STATE_DEPTH_CLIP_ENABLE_EXT:
      return RADV_DYNAMIC_DEPTH_CLIP_ENABLE;
   case VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT:
      return RADV_DYNAMIC_CONSERVATIVE_RAST_MODE;
   case VK_DYNAMIC_STATE_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT:
      return RADV_DYNAMIC_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE;
   case VK_DYNAMIC_STATE_PROVOKING_VERTEX_MODE_EXT:
      return RADV_DYNAMIC_PROVOKING_VERTEX_MODE;
   case VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT:
      return RADV_DYNAMIC_DEPTH_CLAMP_ENABLE;
   case VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT:
      return RADV_DYNAMIC_COLOR_WRITE_MASK;
   case VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT:
      return RADV_DYNAMIC_COLOR_BLEND_ENABLE;
   case VK_DYNAMIC_STATE_RASTERIZATION_SAMPLES_EXT:
      return RADV_DYNAMIC_RASTERIZATION_SAMPLES;
   case VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT:
      return RADV_DYNAMIC_LINE_RASTERIZATION_MODE;
   case VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT:
      return RADV_DYNAMIC_COLOR_BLEND_EQUATION;
   case VK_DYNAMIC_STATE_DISCARD_RECTANGLE_ENABLE_EXT:
      return RADV_DYNAMIC_DISCARD_RECTANGLE_ENABLE;
   case VK_DYNAMIC_STATE_DISCARD_RECTANGLE_MODE_EXT:
      return RADV_DYNAMIC_DISCARD_RECTANGLE_MODE;
   case VK_DYNAMIC_STATE_ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT:
      return RADV_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE;
   case VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_ENABLE_EXT:
      return RADV_DYNAMIC_SAMPLE_LOCATIONS_ENABLE;
   case VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT:
      return RADV_DYNAMIC_ALPHA_TO_ONE_ENABLE;
   case VK_DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT:
      return RADV_DYNAMIC_DEPTH_CLAMP_RANGE;
   case VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT:
      return RADV_DYNAMIC_VIEWPORT | RADV_DYNAMIC_VIEWPORT_WITH_COUNT;
   case VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT:
      return RADV_DYNAMIC_SCISSOR | RADV_DYNAMIC_SCISSOR_WITH_COUNT;
   default:
      UNREACHABLE("Unhandled dynamic state");
   }
}

#define RADV_DYNAMIC_CB_STATES                                                                                         \
   (RADV_DYNAMIC_LOGIC_OP_ENABLE | RADV_DYNAMIC_LOGIC_OP | RADV_DYNAMIC_COLOR_WRITE_ENABLE |                           \
    RADV_DYNAMIC_COLOR_WRITE_MASK | RADV_DYNAMIC_COLOR_BLEND_ENABLE | RADV_DYNAMIC_COLOR_BLEND_EQUATION |              \
    RADV_DYNAMIC_BLEND_CONSTANTS)

static bool
radv_pipeline_is_blend_enabled(const struct radv_graphics_pipeline *pipeline, const struct vk_color_blend_state *cb)
{
   /* If we don't know then we have to assume that blend may be enabled. cb may also be NULL in this
    * case.
    */
   if (pipeline->dynamic_states & (RADV_DYNAMIC_COLOR_BLEND_ENABLE | RADV_DYNAMIC_COLOR_WRITE_MASK))
      return true;

   /* If we have the blend enable state, then cb being NULL indicates no attachments are written. */
   if (cb) {
      for (uint32_t i = 0; i < cb->attachment_count; i++) {
         if (cb->attachments[i].write_mask && cb->attachments[i].blend_enable)
            return true;
      }
   }

   return false;
}

static uint64_t
radv_pipeline_needed_dynamic_state(const struct radv_device *device, const struct radv_graphics_pipeline *pipeline,
                                   const struct vk_graphics_pipeline_state *state)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   bool has_color_att = radv_pipeline_has_color_attachments(state->rp);
   bool raster_enabled =
      !state->rs->rasterizer_discard_enable || (pipeline->dynamic_states & RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE);
   uint64_t states = RADV_DYNAMIC_ALL;

   if (pdev->info.gfx_level < GFX10_3)
      states &= ~RADV_DYNAMIC_FRAGMENT_SHADING_RATE;

   /* Disable dynamic states that are useless to mesh shading. */
   if (radv_pipeline_has_stage(pipeline, MESA_SHADER_MESH)) {
      if (!raster_enabled)
         return RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE;

      states &= ~(RADV_DYNAMIC_VERTEX_INPUT | RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE |
                  RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE | RADV_DYNAMIC_PRIMITIVE_TOPOLOGY);
   }

   /* Disable dynamic states that are useless when rasterization is disabled. */
   if (!raster_enabled) {
      states = RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE;

      if (state->ia)
         states |= RADV_DYNAMIC_PRIMITIVE_TOPOLOGY | RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE;

      if (state->vi)
         states |= RADV_DYNAMIC_VERTEX_INPUT | RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE;

      if (pipeline->active_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
         states |= RADV_DYNAMIC_PATCH_CONTROL_POINTS | RADV_DYNAMIC_TESS_DOMAIN_ORIGIN;

      return states;
   }

   if (!state->rs->depth_bias.enable && !(pipeline->dynamic_states & RADV_DYNAMIC_DEPTH_BIAS_ENABLE))
      states &= ~RADV_DYNAMIC_DEPTH_BIAS;

   if (!(pipeline->dynamic_states & RADV_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE) &&
       (!state->ds || !state->ds->depth.bounds_test.enable))
      states &= ~RADV_DYNAMIC_DEPTH_BOUNDS;

   if (!(pipeline->dynamic_states & RADV_DYNAMIC_STENCIL_TEST_ENABLE) &&
       (!state->ds || !state->ds->stencil.test_enable))
      states &= ~(RADV_DYNAMIC_STENCIL_COMPARE_MASK | RADV_DYNAMIC_STENCIL_WRITE_MASK | RADV_DYNAMIC_STENCIL_REFERENCE |
                  RADV_DYNAMIC_STENCIL_OP);

   if (!(pipeline->dynamic_states & RADV_DYNAMIC_DISCARD_RECTANGLE_ENABLE) && !state->dr->rectangle_count)
      states &= ~RADV_DYNAMIC_DISCARD_RECTANGLE;

   if (!(pipeline->dynamic_states & RADV_DYNAMIC_SAMPLE_LOCATIONS_ENABLE) &&
       (!state->ms || !state->ms->sample_locations_enable))
      states &= ~RADV_DYNAMIC_SAMPLE_LOCATIONS;

   if (!has_color_att || !radv_pipeline_is_blend_enabled(pipeline, state->cb))
      states &= ~RADV_DYNAMIC_BLEND_CONSTANTS;

   if (!(pipeline->active_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT))
      states &= ~(RADV_DYNAMIC_PATCH_CONTROL_POINTS | RADV_DYNAMIC_TESS_DOMAIN_ORIGIN);

   if (pipeline->dynamic_states & RADV_DYNAMIC_VERTEX_INPUT)
      states &= ~RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE;

   return states;
}

struct radv_ia_multi_vgt_param_helpers
radv_compute_ia_multi_vgt_param(const struct radv_device *device, struct radv_shader *const *shaders)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_ia_multi_vgt_param_helpers ia_multi_vgt_param = {0};

   if (shaders[MESA_SHADER_TESS_CTRL]) {
      const struct radv_shader *tes = radv_get_shader(shaders, MESA_SHADER_TESS_EVAL);

      /* SWITCH_ON_EOI must be set if PrimID is used. */
      if (shaders[MESA_SHADER_TESS_CTRL]->info.uses_prim_id || tes->info.uses_prim_id ||
          (tes->info.merged_shader_compiled_separately && shaders[MESA_SHADER_GEOMETRY]->info.uses_prim_id))
         ia_multi_vgt_param.ia_switch_on_eoi = true;
      if (shaders[MESA_SHADER_FRAGMENT] && shaders[MESA_SHADER_FRAGMENT]->info.ps.prim_id_input)
         ia_multi_vgt_param.ia_switch_on_eoi = true;
      if (shaders[MESA_SHADER_GEOMETRY] && shaders[MESA_SHADER_GEOMETRY]->info.uses_prim_id)
         ia_multi_vgt_param.ia_switch_on_eoi = true;
   }

   if (shaders[MESA_SHADER_TESS_CTRL]) {
      /* Bug with tessellation and GS on Bonaire and older 2 SE chips. */
      if ((pdev->info.family == CHIP_TAHITI || pdev->info.family == CHIP_PITCAIRN ||
           pdev->info.family == CHIP_BONAIRE) &&
          shaders[MESA_SHADER_GEOMETRY])
         ia_multi_vgt_param.partial_vs_wave = true;
      /* Needed for 028B6C_DISTRIBUTION_MODE != 0 */
      if (pdev->info.has_distributed_tess) {
         if (shaders[MESA_SHADER_GEOMETRY]) {
            if (pdev->info.gfx_level <= GFX8)
               ia_multi_vgt_param.partial_es_wave = true;
         } else {
            ia_multi_vgt_param.partial_vs_wave = true;
         }
      }
   }

   if (shaders[MESA_SHADER_GEOMETRY]) {
      /* On these chips there is the possibility of a hang if the
       * pipeline uses a GS and partial_vs_wave is not set.
       *
       * This mostly does not hit 4-SE chips, as those typically set
       * ia_switch_on_eoi and then partial_vs_wave is set for pipelines
       * with GS due to another workaround.
       *
       * Reproducer: https://bugs.freedesktop.org/show_bug.cgi?id=109242
       */
      if (pdev->info.family == CHIP_TONGA || pdev->info.family == CHIP_FIJI || pdev->info.family == CHIP_POLARIS10 ||
          pdev->info.family == CHIP_POLARIS11 || pdev->info.family == CHIP_POLARIS12 ||
          pdev->info.family == CHIP_VEGAM) {
         ia_multi_vgt_param.partial_vs_wave = true;
      }
   }

   ia_multi_vgt_param.base =
      /* The following field was moved to VGT_SHADER_STAGES_EN in GFX9. */
      S_028AA8_MAX_PRIMGRP_IN_WAVE(pdev->info.gfx_level == GFX8 ? 2 : 0) |
      S_030960_EN_INST_OPT_BASIC(pdev->info.gfx_level >= GFX9) | S_030960_EN_INST_OPT_ADV(pdev->info.gfx_level >= GFX9);

   return ia_multi_vgt_param;
}

static uint32_t
radv_get_attrib_stride(const VkPipelineVertexInputStateCreateInfo *vi, uint32_t attrib_binding)
{
   for (uint32_t i = 0; i < vi->vertexBindingDescriptionCount; i++) {
      const VkVertexInputBindingDescription *input_binding = &vi->pVertexBindingDescriptions[i];

      if (input_binding->binding == attrib_binding)
         return input_binding->stride;
   }

   return 0;
}

#define ALL_GRAPHICS_LIB_FLAGS                                                                                         \
   (VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT |                                                      \
    VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT |                                                   \
    VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT |                                                             \
    VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT)

static VkGraphicsPipelineLibraryFlagBitsEXT
shader_stage_to_pipeline_library_flags(VkShaderStageFlagBits stage)
{
   assert(util_bitcount(stage) == 1);
   switch (stage) {
   case VK_SHADER_STAGE_VERTEX_BIT:
   case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
   case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
   case VK_SHADER_STAGE_GEOMETRY_BIT:
   case VK_SHADER_STAGE_TASK_BIT_EXT:
   case VK_SHADER_STAGE_MESH_BIT_EXT:
      return VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;
   case VK_SHADER_STAGE_FRAGMENT_BIT:
      return VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;
   default:
      UNREACHABLE("Invalid shader stage");
   }
}

static void
radv_graphics_pipeline_import_layout(struct radv_pipeline_layout *dst, const struct radv_pipeline_layout *src)
{
   for (uint32_t s = 0; s < src->num_sets; s++) {
      if (!src->set[s].layout)
         continue;

      radv_pipeline_layout_add_set(dst, s, src->set[s].layout);
   }

   dst->independent_sets |= src->independent_sets;
}

static void
radv_pipeline_import_graphics_info(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                   const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   /* Mark all states declared dynamic at pipeline creation. */
   if (pCreateInfo->pDynamicState) {
      uint32_t count = pCreateInfo->pDynamicState->dynamicStateCount;
      for (uint32_t s = 0; s < count; s++) {
         pipeline->dynamic_states |= radv_dynamic_state_mask(pCreateInfo->pDynamicState->pDynamicStates[s]);
      }
   }

   /* Mark all active stages at pipeline creation. */
   for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *sinfo = &pCreateInfo->pStages[i];

      pipeline->active_stages |= sinfo->stage;
   }

   if (pipeline->active_stages & VK_SHADER_STAGE_MESH_BIT_EXT) {
      pipeline->last_vgt_api_stage = MESA_SHADER_MESH;
   } else {
      pipeline->last_vgt_api_stage = util_last_bit(pipeline->active_stages & BITFIELD_MASK(MESA_SHADER_FRAGMENT)) - 1;
   }
}

static bool
radv_should_import_lib_binaries(const VkPipelineCreateFlags2 create_flags)
{
   return !(create_flags & (VK_PIPELINE_CREATE_2_LINK_TIME_OPTIMIZATION_BIT_EXT |
                            VK_PIPELINE_CREATE_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT));
}

static void
radv_graphics_pipeline_import_lib(const struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                  struct radv_graphics_lib_pipeline *lib, bool import_pipeline_binaries)
{
   bool import_binaries = false;

   /* There should be no common blocks between a lib we import and the current
    * pipeline we're building.
    */
   assert((pipeline->active_stages & lib->base.active_stages) == 0);

   pipeline->dynamic_states |= lib->base.dynamic_states;
   pipeline->active_stages |= lib->base.active_stages;

   /* Import binaries when LTO is disabled and when the library doesn't retain any shaders. */
   if (!import_pipeline_binaries &&
       (lib->base.has_pipeline_binaries || radv_should_import_lib_binaries(pipeline->base.create_flags))) {
      import_binaries = true;
   }

   if (import_binaries) {
      /* Import the compiled shaders. */
      for (uint32_t s = 0; s < ARRAY_SIZE(lib->base.base.shaders); s++) {
         if (!lib->base.base.shaders[s])
            continue;

         pipeline->base.shaders[s] = radv_shader_ref(lib->base.base.shaders[s]);
      }

      /* Import the GS copy shader if present. */
      if (lib->base.base.gs_copy_shader) {
         assert(!pipeline->base.gs_copy_shader);
         pipeline->base.gs_copy_shader = radv_shader_ref(lib->base.base.gs_copy_shader);
      }
   }
}

static void
radv_pipeline_init_input_assembly_state(const struct radv_device *device, struct radv_graphics_pipeline *pipeline)
{
   pipeline->ia_multi_vgt_param = radv_compute_ia_multi_vgt_param(device, pipeline->base.shaders);
}

static bool
radv_pipeline_uses_ds_feedback_loop(const struct radv_graphics_pipeline *pipeline,
                                    const struct vk_graphics_pipeline_state *state)
{
   VkPipelineCreateFlags2 create_flags = pipeline->base.create_flags;
   if (state->rp)
      create_flags |= state->pipeline_flags;

   return (create_flags & VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) != 0;
}

void
radv_get_viewport_xform(const VkViewport *viewport, float scale[3], float translate[3])
{
   float x = viewport->x;
   float y = viewport->y;
   float half_width = 0.5f * viewport->width;
   float half_height = 0.5f * viewport->height;
   double n = viewport->minDepth;
   double f = viewport->maxDepth;

   scale[0] = half_width;
   translate[0] = half_width + x;
   scale[1] = half_height;
   translate[1] = half_height + y;

   scale[2] = (f - n);
   translate[2] = n;
}

void
radv_translate_blend_equation(const struct radv_physical_device *pdev, VkBlendOp eqRGB, VkBlendFactor srcRGB,
                              VkBlendFactor dstRGB, VkBlendOp eqA, VkBlendFactor srcA, VkBlendFactor dstA,
                              uint32_t *cb_blend_control_out, uint32_t *sx_mrt_blend_opt_out)
{
   unsigned srcRGB_opt, dstRGB_opt, srcA_opt, dstA_opt;
   uint32_t cb_blend_control = 0, sx_mrt_blend_opt = 0;

   radv_normalize_blend_factor(eqRGB, &srcRGB, &dstRGB);
   radv_normalize_blend_factor(eqA, &srcA, &dstA);

   /* Blending optimizations for RB+.
    * These transformations don't change the behavior.
    *
    * First, get rid of DST in the blend factors:
    *    func(src * DST, dst * 0) ---> func(src * 0, dst * SRC)
    */
   radv_blend_remove_dst(&eqRGB, &srcRGB, &dstRGB, VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_SRC_COLOR);

   radv_blend_remove_dst(&eqA, &srcA, &dstA, VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_SRC_COLOR);

   radv_blend_remove_dst(&eqA, &srcA, &dstA, VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA);

   /* Look up the ideal settings from tables. */
   srcRGB_opt = radv_translate_blend_opt_factor(srcRGB, false);
   dstRGB_opt = radv_translate_blend_opt_factor(dstRGB, false);
   srcA_opt = radv_translate_blend_opt_factor(srcA, true);
   dstA_opt = radv_translate_blend_opt_factor(dstA, true);

   /* Handle interdependencies. */
   if (radv_blend_factor_uses_dst(srcRGB))
      dstRGB_opt = V_028760_BLEND_OPT_PRESERVE_NONE_IGNORE_NONE;
   if (radv_blend_factor_uses_dst(srcA))
      dstA_opt = V_028760_BLEND_OPT_PRESERVE_NONE_IGNORE_NONE;

   if (srcRGB == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE &&
       (dstRGB == VK_BLEND_FACTOR_ZERO || dstRGB == VK_BLEND_FACTOR_SRC_ALPHA ||
        dstRGB == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE))
      dstRGB_opt = V_028760_BLEND_OPT_PRESERVE_NONE_IGNORE_A0;

   /* Set the final value. */
   sx_mrt_blend_opt = S_028760_COLOR_SRC_OPT(srcRGB_opt) | S_028760_COLOR_DST_OPT(dstRGB_opt) |
                      S_028760_COLOR_COMB_FCN(radv_translate_blend_opt_function(eqRGB)) |
                      S_028760_ALPHA_SRC_OPT(srcA_opt) | S_028760_ALPHA_DST_OPT(dstA_opt) |
                      S_028760_ALPHA_COMB_FCN(radv_translate_blend_opt_function(eqA));

   cb_blend_control |= S_028780_ENABLE(1);
   cb_blend_control |= S_028780_COLOR_COMB_FCN(radv_translate_blend_function(eqRGB));
   cb_blend_control |= S_028780_COLOR_SRCBLEND(radv_translate_blend_factor(pdev->info.gfx_level, srcRGB));
   cb_blend_control |= S_028780_COLOR_DESTBLEND(radv_translate_blend_factor(pdev->info.gfx_level, dstRGB));
   if (srcA != srcRGB || dstA != dstRGB || eqA != eqRGB) {
      cb_blend_control |= S_028780_SEPARATE_ALPHA_BLEND(1);
      cb_blend_control |= S_028780_ALPHA_COMB_FCN(radv_translate_blend_function(eqA));
      cb_blend_control |= S_028780_ALPHA_SRCBLEND(radv_translate_blend_factor(pdev->info.gfx_level, srcA));
      cb_blend_control |= S_028780_ALPHA_DESTBLEND(radv_translate_blend_factor(pdev->info.gfx_level, dstA));
   }

   *cb_blend_control_out = cb_blend_control;
   *sx_mrt_blend_opt_out = sx_mrt_blend_opt;
}

static void
radv_pipeline_init_vertex_input_state(const struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                      const struct vk_graphics_pipeline_state *state)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader *vs = radv_get_shader(pipeline->base.shaders, MESA_SHADER_VERTEX);
   struct radv_dynamic_state *dynamic = &pipeline->dynamic_state;

   if (vs->info.vs.use_per_attribute_vb_descs) {
      const enum amd_gfx_level gfx_level = pdev->info.gfx_level;
      const bool alpha_adjust = pdev->info.compiler_info.has_vtx_format_alpha_adjust_bug;
      const struct ac_vtx_format_info *vtx_info_table = ac_get_vtx_format_info_table(gfx_level, alpha_adjust);

      dynamic->vertex_input.bindings_match_attrib = true;

      u_foreach_bit (i, state->vi->attributes_valid) {
         uint32_t binding = state->vi->attributes[i].binding;
         uint32_t offset = state->vi->attributes[i].offset;

         dynamic->vertex_input.attribute_mask |= BITFIELD_BIT(i);
         dynamic->vertex_input.bindings[i] = binding;
         dynamic->vertex_input.bindings_match_attrib &= binding == i;

         if (state->vi->bindings[binding].stride) {
            dynamic->vertex_input.attrib_index_offset[i] = offset / state->vi->bindings[binding].stride;
         }

         if (state->vi->bindings[binding].input_rate) {
            dynamic->vertex_input.instance_rate_inputs |= BITFIELD_BIT(i);
            dynamic->vertex_input.divisors[i] = state->vi->bindings[binding].divisor;

            if (state->vi->bindings[binding].divisor == 0) {
               dynamic->vertex_input.zero_divisors |= BITFIELD_BIT(i);
            } else if (state->vi->bindings[binding].divisor > 1) {
               dynamic->vertex_input.nontrivial_divisors |= BITFIELD_BIT(i);
            }
         }

         dynamic->vertex_input.offsets[i] = offset;

         enum pipe_format format = radv_format_to_pipe_format(state->vi->attributes[i].format);
         const struct ac_vtx_format_info *vtx_info = &vtx_info_table[format];
         const uint32_t hw_format = vtx_info->hw_format[vtx_info->num_channels - 1];

         dynamic->vertex_input.formats[i] = format;
         uint8_t format_align_req_minus_1 = vtx_info->chan_byte_size >= 4 ? 3 : (vtx_info->element_size - 1);
         dynamic->vertex_input.format_align_req_minus_1[i] = format_align_req_minus_1;
         uint8_t component_align_req_minus_1 =
            MIN2(vtx_info->chan_byte_size ? vtx_info->chan_byte_size : vtx_info->element_size, 4) - 1;
         dynamic->vertex_input.component_align_req_minus_1[i] = component_align_req_minus_1;
         dynamic->vertex_input.format_sizes[i] = vtx_info->element_size;
         dynamic->vertex_input.alpha_adjust_lo |= (vtx_info->alpha_adjust & 0x1) << i;
         dynamic->vertex_input.alpha_adjust_hi |= (vtx_info->alpha_adjust >> 1) << i;
         if (G_008F0C_DST_SEL_X(vtx_info->dst_sel) == V_008F0C_SQ_SEL_Z) {
            dynamic->vertex_input.post_shuffle |= BITFIELD_BIT(i);
         }

         if (vtx_info->has_hw_format & BITFIELD_BIT(vtx_info->num_channels - 1)) {
            if (pdev->info.gfx_level >= GFX10) {
               dynamic->vertex_input.non_trivial_format[i] =
                  vtx_info->dst_sel | S_008F0C_FORMAT_GFX10(hw_format) |
                  S_008F0C_RESOURCE_LEVEL(pdev->info.compiler_info.has_desc_resource_level);
            } else {
               dynamic->vertex_input.non_trivial_format[i] = vtx_info->dst_sel |
                                                             S_008F0C_NUM_FORMAT((hw_format >> 4) & 0x7) |
                                                             S_008F0C_DATA_FORMAT(hw_format & 0xf);
            }
         } else {
            dynamic->vertex_input.nontrivial_formats |= BITFIELD_BIT(i);
         }
      }

      dynamic->vertex_input.vbo_misaligned_mask_invalid = dynamic->vertex_input.attribute_mask;
   } else {
      u_foreach_bit (i, vs->info.vs.vb_desc_usage_mask) {
         dynamic->vertex_input.bindings[i] = i;
      }
   }
}

static void
radv_pipeline_init_dynamic_state(const struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                 const struct vk_graphics_pipeline_state *state)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   uint64_t needed_states = radv_pipeline_needed_dynamic_state(device, pipeline, state);
   struct radv_dynamic_state *dynamic = &pipeline->dynamic_state;
   uint64_t states = needed_states;

   /* Initialize non-zero values for default dynamic state. */
   dynamic->vk.rs.line.width = 1.0f;
   dynamic->vk.fsr.fragment_size.width = 1u;
   dynamic->vk.fsr.fragment_size.height = 1u;
   dynamic->vk.ds.depth.bounds_test.max = 1.0f;
   dynamic->vk.ds.stencil.front.compare_mask = ~0;
   dynamic->vk.ds.stencil.front.write_mask = ~0;
   dynamic->vk.ds.stencil.back.compare_mask = ~0;
   dynamic->vk.ds.stencil.back.write_mask = ~0;
   dynamic->vk.ms.rasterization_samples = VK_SAMPLE_COUNT_1_BIT;

   pipeline->needed_dynamic_state = needed_states;

   states &= ~pipeline->dynamic_states;

   /* Input assembly. */
   if (states & RADV_DYNAMIC_PRIMITIVE_TOPOLOGY) {
      dynamic->vk.ia.primitive_topology = radv_translate_prim(state->ia->primitive_topology);
   }

   if (states & RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE) {
      dynamic->vk.ia.primitive_restart_enable = state->ia->primitive_restart_enable;
   }

   /* Tessellation. */
   if (states & RADV_DYNAMIC_PATCH_CONTROL_POINTS) {
      dynamic->vk.ts.patch_control_points = state->ts->patch_control_points;
   }

   if (states & RADV_DYNAMIC_TESS_DOMAIN_ORIGIN) {
      dynamic->vk.ts.domain_origin = state->ts->domain_origin;
   }

   /* Viewport. */
   if (states & RADV_DYNAMIC_VIEWPORT) {
      typed_memcpy(dynamic->vk.vp.viewports, state->vp->viewports, state->vp->viewport_count);
      for (unsigned i = 0; i < state->vp->viewport_count; i++)
         radv_get_viewport_xform(&dynamic->vk.vp.viewports[i], dynamic->vp_xform[i].scale,
                                 dynamic->vp_xform[i].translate);
   }

   if (states & RADV_DYNAMIC_VIEWPORT_WITH_COUNT) {
      dynamic->vk.vp.viewport_count = state->vp->viewport_count;
   }

   /* Scissor. */
   if (states & RADV_DYNAMIC_SCISSOR) {
      typed_memcpy(dynamic->vk.vp.scissors, state->vp->scissors, state->vp->scissor_count);
   }

   if (states & RADV_DYNAMIC_SCISSOR_WITH_COUNT) {
      dynamic->vk.vp.scissor_count = state->vp->scissor_count;
   }

   if (states & RADV_DYNAMIC_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) {
      dynamic->vk.vp.depth_clip_negative_one_to_one = state->vp->depth_clip_negative_one_to_one;
   }

   if (states & RADV_DYNAMIC_DEPTH_CLAMP_RANGE) {
      dynamic->vk.vp.depth_clamp_mode = state->vp->depth_clamp_mode;
      dynamic->vk.vp.depth_clamp_range = state->vp->depth_clamp_range;
   }

   /* Discard rectangles. */
   if (needed_states & RADV_DYNAMIC_DISCARD_RECTANGLE) {
      dynamic->vk.dr.rectangle_count = state->dr->rectangle_count;
      if (states & RADV_DYNAMIC_DISCARD_RECTANGLE) {
         typed_memcpy(dynamic->vk.dr.rectangles, state->dr->rectangles, state->dr->rectangle_count);
      }
   }

   /* Rasterization. */
   if (states & RADV_DYNAMIC_LINE_WIDTH) {
      dynamic->vk.rs.line.width = state->rs->line.width;
   }

   if (states & RADV_DYNAMIC_DEPTH_BIAS) {
      dynamic->vk.rs.depth_bias.constant_factor = state->rs->depth_bias.constant_factor;
      dynamic->vk.rs.depth_bias.clamp = state->rs->depth_bias.clamp;
      dynamic->vk.rs.depth_bias.slope_factor = state->rs->depth_bias.slope_factor;
      dynamic->vk.rs.depth_bias.representation = state->rs->depth_bias.representation;
   }

   if (states & RADV_DYNAMIC_CULL_MODE) {
      dynamic->vk.rs.cull_mode = state->rs->cull_mode;
   }

   if (states & RADV_DYNAMIC_FRONT_FACE) {
      dynamic->vk.rs.front_face = state->rs->front_face;
   }

   if (states & RADV_DYNAMIC_LINE_STIPPLE) {
      dynamic->vk.rs.line.stipple.factor = state->rs->line.stipple.factor;
      dynamic->vk.rs.line.stipple.pattern = state->rs->line.stipple.pattern;
   }

   if (states & RADV_DYNAMIC_DEPTH_BIAS_ENABLE) {
      dynamic->vk.rs.depth_bias.enable = state->rs->depth_bias.enable;
   }

   if (states & RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE) {
      dynamic->vk.rs.rasterizer_discard_enable = state->rs->rasterizer_discard_enable;
   }

   if (states & RADV_DYNAMIC_POLYGON_MODE) {
      dynamic->vk.rs.polygon_mode = radv_translate_fill(state->rs->polygon_mode);
   }

   if (states & RADV_DYNAMIC_LINE_STIPPLE_ENABLE) {
      dynamic->vk.rs.line.stipple.enable = state->rs->line.stipple.enable;
   }

   if (states & RADV_DYNAMIC_DEPTH_CLIP_ENABLE) {
      dynamic->vk.rs.depth_clip_enable = state->rs->depth_clip_enable;
   }

   if (states & RADV_DYNAMIC_CONSERVATIVE_RAST_MODE) {
      dynamic->vk.rs.conservative_mode = state->rs->conservative_mode;
   }

   if (states & RADV_DYNAMIC_PROVOKING_VERTEX_MODE) {
      dynamic->vk.rs.provoking_vertex = state->rs->provoking_vertex;
   }

   if (states & RADV_DYNAMIC_DEPTH_CLAMP_ENABLE) {
      dynamic->vk.rs.depth_clamp_enable = state->rs->depth_clamp_enable;
   }

   if (states & RADV_DYNAMIC_LINE_RASTERIZATION_MODE) {
      dynamic->vk.rs.line.mode = state->rs->line.mode;
   }

   /* Fragment shading rate. */
   if (states & RADV_DYNAMIC_FRAGMENT_SHADING_RATE) {
      dynamic->vk.fsr = *state->fsr;
   }

   /* Multisample. */
   if (states & RADV_DYNAMIC_ALPHA_TO_COVERAGE_ENABLE) {
      dynamic->vk.ms.alpha_to_coverage_enable = state->ms->alpha_to_coverage_enable;
   }

   if (states & RADV_DYNAMIC_ALPHA_TO_ONE_ENABLE) {
      dynamic->vk.ms.alpha_to_one_enable = state->ms->alpha_to_one_enable;
   }

   if (states & RADV_DYNAMIC_SAMPLE_MASK) {
      dynamic->vk.ms.sample_mask = state->ms->sample_mask & 0xffff;
   }

   if (states & RADV_DYNAMIC_RASTERIZATION_SAMPLES) {
      dynamic->vk.ms.rasterization_samples = state->ms->rasterization_samples;
   }

   if (states & RADV_DYNAMIC_SAMPLE_LOCATIONS_ENABLE) {
      dynamic->vk.ms.sample_locations_enable = state->ms->sample_locations_enable;
   }

   if (states & RADV_DYNAMIC_SAMPLE_LOCATIONS) {
      unsigned count = state->ms->sample_locations->per_pixel * state->ms->sample_locations->grid_size.width *
                       state->ms->sample_locations->grid_size.height;

      dynamic->sample_location.per_pixel = state->ms->sample_locations->per_pixel;
      dynamic->sample_location.grid_size = state->ms->sample_locations->grid_size;
      dynamic->sample_location.count = count;
      typed_memcpy(&dynamic->sample_location.locations[0], state->ms->sample_locations->locations, count);
   }

   /* Depth stencil. */
   /* If there is no depthstencil attachment, then don't read
    * pDepthStencilState. The Vulkan spec states that pDepthStencilState may
    * be NULL in this case. Even if pDepthStencilState is non-NULL, there is
    * no need to override the depthstencil defaults in
    * radv_pipeline::dynamic_state when there is no depthstencil attachment.
    *
    * Section 9.2 of the Vulkan 1.0.15 spec says:
    *
    *    pDepthStencilState is [...] NULL if the pipeline has rasterization
    *    disabled or if the subpass of the render pass the pipeline is created
    *    against does not use a depth/stencil attachment.
    */
   if (needed_states && radv_pipeline_has_ds_attachments(state->rp)) {
      if (states & RADV_DYNAMIC_DEPTH_BOUNDS) {
         dynamic->vk.ds.depth.bounds_test.min = state->ds->depth.bounds_test.min;
         dynamic->vk.ds.depth.bounds_test.max = state->ds->depth.bounds_test.max;
      }

      if (states & RADV_DYNAMIC_STENCIL_COMPARE_MASK) {
         dynamic->vk.ds.stencil.front.compare_mask = state->ds->stencil.front.compare_mask;
         dynamic->vk.ds.stencil.back.compare_mask = state->ds->stencil.back.compare_mask;
      }

      if (states & RADV_DYNAMIC_STENCIL_WRITE_MASK) {
         dynamic->vk.ds.stencil.front.write_mask = state->ds->stencil.front.write_mask;
         dynamic->vk.ds.stencil.back.write_mask = state->ds->stencil.back.write_mask;
      }

      if (states & RADV_DYNAMIC_STENCIL_REFERENCE) {
         dynamic->vk.ds.stencil.front.reference = state->ds->stencil.front.reference;
         dynamic->vk.ds.stencil.back.reference = state->ds->stencil.back.reference;
      }

      if (states & RADV_DYNAMIC_DEPTH_TEST_ENABLE) {
         dynamic->vk.ds.depth.test_enable = state->ds->depth.test_enable;
      }

      if (states & RADV_DYNAMIC_DEPTH_WRITE_ENABLE) {
         dynamic->vk.ds.depth.write_enable = state->ds->depth.write_enable;
      }

      if (states & RADV_DYNAMIC_DEPTH_COMPARE_OP) {
         dynamic->vk.ds.depth.compare_op = state->ds->depth.compare_op;
      }

      if (states & RADV_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE) {
         dynamic->vk.ds.depth.bounds_test.enable = state->ds->depth.bounds_test.enable;
      }

      if (states & RADV_DYNAMIC_STENCIL_TEST_ENABLE) {
         dynamic->vk.ds.stencil.test_enable = state->ds->stencil.test_enable;
      }

      if (states & RADV_DYNAMIC_STENCIL_OP) {
         dynamic->vk.ds.stencil.front.op.compare = state->ds->stencil.front.op.compare;
         dynamic->vk.ds.stencil.front.op.fail = radv_translate_stencil_op(state->ds->stencil.front.op.fail);
         dynamic->vk.ds.stencil.front.op.pass = radv_translate_stencil_op(state->ds->stencil.front.op.pass);
         dynamic->vk.ds.stencil.front.op.depth_fail = radv_translate_stencil_op(state->ds->stencil.front.op.depth_fail);

         dynamic->vk.ds.stencil.back.op.compare = state->ds->stencil.back.op.compare;
         dynamic->vk.ds.stencil.back.op.fail = radv_translate_stencil_op(state->ds->stencil.back.op.fail);
         dynamic->vk.ds.stencil.back.op.pass = radv_translate_stencil_op(state->ds->stencil.back.op.pass);
         dynamic->vk.ds.stencil.back.op.depth_fail = radv_translate_stencil_op(state->ds->stencil.back.op.depth_fail);
      }
   }

   /* Color blend. */
   /* Section 9.2 of the Vulkan 1.0.15 spec says:
    *
    *    pColorBlendState is [...] NULL if the pipeline has rasterization
    *    disabled or if the subpass of the render pass the pipeline is
    *    created against does not use any color attachments.
    */
   if (states & RADV_DYNAMIC_BLEND_CONSTANTS) {
      typed_memcpy(dynamic->vk.cb.blend_constants, state->cb->blend_constants, 4);
   }

   if (radv_pipeline_has_color_attachments(state->rp)) {
      if (states & RADV_DYNAMIC_LOGIC_OP) {
         if ((pipeline->dynamic_states & RADV_DYNAMIC_LOGIC_OP_ENABLE) || state->cb->logic_op_enable) {
            dynamic->vk.cb.logic_op = radv_translate_blend_logic_op(state->cb->logic_op);
         }
      }

      if (states & RADV_DYNAMIC_COLOR_WRITE_ENABLE) {
         u_foreach_bit (i, state->cb->color_write_enables) {
            dynamic->color_write_enable |= BITFIELD_RANGE(i * 4, 4);
         }
      }

      if (states & RADV_DYNAMIC_LOGIC_OP_ENABLE) {
         dynamic->vk.cb.logic_op_enable = state->cb->logic_op_enable;
      }

      if (states & RADV_DYNAMIC_COLOR_WRITE_MASK) {
         for (unsigned i = 0; i < state->cb->attachment_count; i++) {
            dynamic->color_write_mask |= (uint32_t)state->cb->attachments[i].write_mask << (4 * i);
         }
      }

      if (states & RADV_DYNAMIC_COLOR_BLEND_ENABLE) {
         for (unsigned i = 0; i < state->cb->attachment_count; i++) {
            dynamic->color_blend_enable |= state->cb->attachments[i].blend_enable << i;
         }
      }

      if (states & RADV_DYNAMIC_COLOR_BLEND_EQUATION) {
         for (unsigned i = 0; i < state->cb->attachment_count; i++) {
            const struct vk_color_blend_attachment_state *att = &state->cb->attachments[i];

            radv_translate_blend_equation(pdev, att->color_blend_op, att->src_color_blend_factor,
                                          att->dst_color_blend_factor, att->alpha_blend_op, att->src_alpha_blend_factor,
                                          att->dst_alpha_blend_factor, &dynamic->blend_eq.att[i].cb_blend_control,
                                          &dynamic->blend_eq.att[i].sx_mrt_blend_opt);
         }

         dynamic->blend_eq.mrt0_is_dual_src = radv_can_enable_dual_src(&state->cb->attachments[0]);
      }
   }

   if (states & RADV_DYNAMIC_DISCARD_RECTANGLE_ENABLE) {
      dynamic->vk.dr.enable = state->dr->rectangle_count > 0;
   }

   if (states & RADV_DYNAMIC_DISCARD_RECTANGLE_MODE) {
      dynamic->vk.dr.mode = state->dr->mode;
   }

   if (states & RADV_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE) {
      bool uses_ds_feedback_loop = radv_pipeline_uses_ds_feedback_loop(pipeline, state);

      dynamic->feedback_loop_aspects =
         uses_ds_feedback_loop ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_NONE;
   }

   if (states & RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE) {
      u_foreach_bit (i, state->vi->bindings_valid) {
         dynamic->vk.vi_binding_strides[i] = state->vi->bindings[i].stride;
      }
   }

   if (states & RADV_DYNAMIC_VERTEX_INPUT) {
      radv_pipeline_init_vertex_input_state(device, pipeline, state);
   }

   for (uint32_t i = 0; i < MAX_RTS; i++) {
      dynamic->vk.cal.color_map[i] = state->cal ? state->cal->color_map[i] : i;
      dynamic->vk.ial.color_map[i] = state->ial ? state->ial->color_map[i] : i;
   }

   dynamic->vk.ial.depth_att = state->ial ? state->ial->depth_att : MESA_VK_ATTACHMENT_UNUSED;
   dynamic->vk.ial.stencil_att = state->ial ? state->ial->stencil_att : MESA_VK_ATTACHMENT_UNUSED;

   pipeline->dynamic_state.mask = states;
}

struct radv_shader *
radv_get_shader(struct radv_shader *const *shaders, mesa_shader_stage stage)
{
   if (stage == MESA_SHADER_VERTEX) {
      if (shaders[MESA_SHADER_VERTEX])
         return shaders[MESA_SHADER_VERTEX];
      if (shaders[MESA_SHADER_TESS_CTRL])
         return shaders[MESA_SHADER_TESS_CTRL];
      if (shaders[MESA_SHADER_GEOMETRY])
         return shaders[MESA_SHADER_GEOMETRY];
   } else if (stage == MESA_SHADER_TESS_EVAL) {
      if (!shaders[MESA_SHADER_TESS_CTRL])
         return NULL;
      if (shaders[MESA_SHADER_TESS_EVAL])
         return shaders[MESA_SHADER_TESS_EVAL];
      if (shaders[MESA_SHADER_GEOMETRY])
         return shaders[MESA_SHADER_GEOMETRY];
   }
   return shaders[stage];
}

static bool
radv_should_export_multiview(const struct radv_shader_stage *stage, const struct radv_graphics_state_key *gfx_state)
{
   /* Export the layer in the last VGT stage if multiview is used.
    * Also checks for NONE stage, which happens when we have depth-only rendering.
    * When the next stage is unknown (with GPL or ESO), the layer is exported unconditionally.
    */
   return gfx_state->has_multiview_view_index && radv_is_last_vgt_stage(stage) &&
          !(stage->nir->info.outputs_written & VARYING_BIT_LAYER);
}

static void
merge_tess_info(struct shader_info *tes_info, struct shader_info *tcs_info)
{
   /* The Vulkan 1.0.38 spec, section 21.1 Tessellator says:
    *
    *    "PointMode. Controls generation of points rather than triangles
    *     or lines. This functionality defaults to disabled, and is
    *     enabled if either shader stage includes the execution mode.
    *
    * and about Triangles, Quads, IsoLines, VertexOrderCw, VertexOrderCcw,
    * PointMode, SpacingEqual, SpacingFractionalEven, SpacingFractionalOdd,
    * and OutputVertices, it says:
    *
    *    "One mode must be set in at least one of the tessellation
    *     shader stages."
    *
    * So, the fields can be set in either the TCS or TES, but they must
    * agree if set in both.  Our backend looks at TES, so bitwise-or in
    * the values from the TCS.
    */
   assert(tcs_info->tess.tcs_vertices_out == 0 || tes_info->tess.tcs_vertices_out == 0 ||
          tcs_info->tess.tcs_vertices_out == tes_info->tess.tcs_vertices_out);
   tes_info->tess.tcs_vertices_out |= tcs_info->tess.tcs_vertices_out;

   assert(tcs_info->tess.spacing == TESS_SPACING_UNSPECIFIED || tes_info->tess.spacing == TESS_SPACING_UNSPECIFIED ||
          tcs_info->tess.spacing == tes_info->tess.spacing);
   tes_info->tess.spacing |= tcs_info->tess.spacing;

   assert(tcs_info->tess._primitive_mode == TESS_PRIMITIVE_UNSPECIFIED ||
          tes_info->tess._primitive_mode == TESS_PRIMITIVE_UNSPECIFIED ||
          tcs_info->tess._primitive_mode == tes_info->tess._primitive_mode);
   tes_info->tess._primitive_mode |= tcs_info->tess._primitive_mode;
   tes_info->tess.ccw |= tcs_info->tess.ccw;
   tes_info->tess.point_mode |= tcs_info->tess.point_mode;

   /* Copy the merged info back to the TCS */
   tcs_info->tess.tcs_vertices_out = tes_info->tess.tcs_vertices_out;
   tcs_info->tess._primitive_mode = tes_info->tess._primitive_mode;
}

static const mesa_shader_stage graphics_shader_order[] = {
   MESA_SHADER_VERTEX,   MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL, MESA_SHADER_GEOMETRY,

   MESA_SHADER_TASK,     MESA_SHADER_MESH,

   MESA_SHADER_FRAGMENT,
};

static bool
radv_pipeline_needs_noop_fs(struct radv_graphics_pipeline *pipeline, const struct radv_graphics_state_key *gfx_state)
{
   if (pipeline->base.type == RADV_PIPELINE_GRAPHICS &&
       !(radv_pipeline_to_graphics(&pipeline->base)->active_stages & VK_SHADER_STAGE_FRAGMENT_BIT))
      return true;

   if (pipeline->base.type == RADV_PIPELINE_GRAPHICS_LIB &&
       (gfx_state->lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) &&
       !(radv_pipeline_to_graphics_lib(&pipeline->base)->base.active_stages & VK_SHADER_STAGE_FRAGMENT_BIT))
      return true;

   return false;
}

static void
radv_graphics_shaders_fill_linked_vs_io_info(struct radv_shader_stage *vs_stage,
                                             struct radv_shader_stage *consumer_stage)
{
   const unsigned num_reserved_slots = util_bitcount64(consumer_stage->nir->info.inputs_read);
   vs_stage->info.vs.num_linked_outputs = num_reserved_slots;
   vs_stage->info.outputs_linked = true;

   switch (consumer_stage->stage) {
   case MESA_SHADER_TESS_CTRL: {
      consumer_stage->info.tcs.num_linked_inputs = num_reserved_slots;
      consumer_stage->info.inputs_linked = true;
      break;
   }
   case MESA_SHADER_GEOMETRY: {
      consumer_stage->info.gs.num_linked_inputs = num_reserved_slots;
      consumer_stage->info.inputs_linked = true;
      break;
   }
   default:
      UNREACHABLE("invalid next stage for VS");
   }
}

static void
radv_graphics_shaders_fill_linked_tcs_tes_io_info(struct radv_shader_stage *tcs_stage,
                                                  struct radv_shader_stage *tes_stage)
{
   assume(tes_stage->stage == MESA_SHADER_TESS_EVAL);

   /* Count the number of per-vertex output slots we need to reserve for the TCS and TES. */
   const uint64_t per_vertex_mask =
      tes_stage->nir->info.inputs_read & ~(VARYING_BIT_TESS_LEVEL_OUTER | VARYING_BIT_TESS_LEVEL_INNER);
   const unsigned num_reserved_slots = util_bitcount64(per_vertex_mask);

   /* Count the number of per-patch output slots we need to reserve for the TCS and TES.
    * This is necessary because we need it to determine the patch size in VRAM.
    */
   const uint64_t tess_lvl_mask =
      tes_stage->nir->info.inputs_read & (VARYING_BIT_TESS_LEVEL_OUTER | VARYING_BIT_TESS_LEVEL_INNER);
   const unsigned num_reserved_patch_slots =
      util_bitcount64(tess_lvl_mask) + util_bitcount64(tes_stage->nir->info.patch_inputs_read);

   tcs_stage->info.outputs_linked = true;

   tes_stage->info.tes.num_linked_inputs = num_reserved_slots;
   tes_stage->info.tes.num_linked_patch_inputs = num_reserved_patch_slots;
   tes_stage->info.inputs_linked = true;
}

static void
radv_graphics_shaders_fill_linked_tes_gs_io_info(struct radv_shader_stage *tes_stage,
                                                 struct radv_shader_stage *gs_stage)
{
   assume(gs_stage->stage == MESA_SHADER_GEOMETRY);

   const unsigned num_reserved_slots = util_bitcount64(gs_stage->nir->info.inputs_read);
   tes_stage->info.tes.num_linked_outputs = num_reserved_slots;
   tes_stage->info.outputs_linked = true;
   gs_stage->info.gs.num_linked_inputs = num_reserved_slots;
   gs_stage->info.inputs_linked = true;
}

static void
radv_graphics_shaders_fill_linked_io_info(struct radv_shader_stage *producer_stage,
                                          struct radv_shader_stage *consumer_stage)
{
   /* We don't need to fill this info for the last pre-rasterization stage. */
   if (consumer_stage->stage == MESA_SHADER_FRAGMENT)
      return;

   switch (producer_stage->stage) {
   case MESA_SHADER_VERTEX:
      radv_graphics_shaders_fill_linked_vs_io_info(producer_stage, consumer_stage);
      break;

   case MESA_SHADER_TESS_CTRL:
      radv_graphics_shaders_fill_linked_tcs_tes_io_info(producer_stage, consumer_stage);
      break;

   case MESA_SHADER_TESS_EVAL:
      radv_graphics_shaders_fill_linked_tes_gs_io_info(producer_stage, consumer_stage);
      break;

   default:
      break;
   }
}

/**
 * Varying optimizations performed on lowered shader I/O.
 *
 * We do this after lowering shader I/O because this is more effective
 * than running the same optimizations on I/O derefs.
 */
static void
radv_graphics_shaders_link_varyings(struct radv_shader_stage *stages, enum amd_gfx_level gfx_level)
{
   /* Prepare shaders before running nir_opt_varyings. */
   for (int i = 0; i < ARRAY_SIZE(graphics_shader_order); ++i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      if (!stages[s].nir)
         continue;

      if (stages[s].key.optimisations_disabled)
         continue;

      nir_shader *shader = stages[s].nir;

      /* It is expected by nir_opt_varyings that no undefined stores are present in the shader. */
      NIR_PASS(_, shader, nir_opt_undef);

      /* Update load/store alignments because inter-stage code motion may move instructions used to deduce this info. */
      NIR_PASS(_, shader, nir_opt_load_store_update_alignments);
   }

   int highest_changed_producer = -1;

   /* Optimize varyings from first to last stage. */
   for (int i = 0; i < ARRAY_SIZE(graphics_shader_order); ++i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      const mesa_shader_stage next = stages[s].info.next_stage;
      if (!stages[s].nir || next == MESA_SHADER_NONE || !stages[next].nir)
         continue;

      if (stages[s].key.optimisations_disabled || stages[next].key.optimisations_disabled)
         continue;

      nir_shader *producer = stages[s].nir;
      nir_shader *consumer = stages[next].nir;

      const nir_opt_varyings_progress p = nir_opt_varyings(producer, consumer, true, 0, 0, false);

      /* Run algebraic optimizations on shaders that changed. */
      if (p & nir_progress_producer) {
         radv_optimize_nir_algebraic(producer, false, false, gfx_level);
         NIR_PASS(_, producer, nir_opt_undef);

         highest_changed_producer = i;
      }
      if (p & nir_progress_consumer) {
         radv_optimize_nir_algebraic(consumer, false, false, gfx_level);
         NIR_PASS(_, consumer, nir_opt_undef);
      }
   }

   /* Optimize varyings from last to first stage. */
   for (int i = highest_changed_producer; i >= 0; --i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      const mesa_shader_stage next = stages[s].info.next_stage;
      if (!stages[s].nir || next == MESA_SHADER_NONE || !stages[next].nir)
         continue;

      if (stages[s].key.optimisations_disabled || stages[next].key.optimisations_disabled)
         continue;

      nir_shader *producer = stages[s].nir;
      nir_shader *consumer = stages[next].nir;

      const nir_opt_varyings_progress p = nir_opt_varyings(producer, consumer, true, 0, 0, false);

      /* Run algebraic optimizations on shaders that changed. */
      if (p & nir_progress_producer) {
         radv_optimize_nir_algebraic(producer, true, false, gfx_level);
         NIR_PASS(_, producer, nir_opt_undef);
      }
      if (p & nir_progress_consumer) {
         radv_optimize_nir_algebraic(consumer, true, false, gfx_level);
         NIR_PASS(_, consumer, nir_opt_undef);
      }
   }

   /* Run optimizations and fixups after linking. */
   for (int i = 0; i < ARRAY_SIZE(graphics_shader_order); ++i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      if (!stages[s].nir)
         continue;

      nir_shader *shader = stages[s].nir;

      /* Re-vectorize I/O for stages that use memory for I/O (LDS or VRAM).
       * Don't vectorize FS I/O, doing so just regresses shader stats without any benefit.
       */
      if (s != MESA_SHADER_FRAGMENT && !stages[s].key.optimisations_disabled) {
         /* Delete dead instructions to prevent them from being vectorized. */
         NIR_PASS(_, shader, nir_opt_dce);

         /* Vectorize inputs. Non-FS inputs are always read from memory. */
         nir_variable_mode vec_mode = nir_var_shader_in;

         /* There is also no benefit from re-vectorizing the outputs of the last pre-rasterization
          * stage here, because ac_nir_lower_ngg/legacy already takes care of that.
          */
         if (!radv_is_last_vgt_stage(&stages[s]))
            vec_mode |= nir_var_shader_out;

         /* Scalarize and revectorize VS inputs to make sure every VS input is loaded by a
          * single *_load_format_* instruction. Those instructions can't skip loading unused
          * components before the last used component, so loading X, Y, Z, W separately
          * actually loads X, XY, XYZ, XYZW, which unnecessarily increases VMEM return data
          * transfers between the VMEM cache and the SIMDs, which wastes SIMD<->VMEM cache bandwidth.
          * By allowing holes during VS input vectorization, VS input loads loading different
          * components are always merged, so that no used or unused component is ever loaded twice.
          */
         if (s == MESA_SHADER_VERTEX) {
            NIR_PASS(_, shader, nir_opt_vectorize_io, nir_var_shader_in, true);
            vec_mode &= ~nir_var_shader_in;
         }

         if (vec_mode)
            NIR_PASS(_, shader, nir_opt_vectorize_io, vec_mode, false);
      }

      /* Gather shader info; at least the I/O info likely changed
       * and changes to only the I/O info are not reflected in nir_opt_varyings_progress.
       */
      nir_shader_gather_info(shader, nir_shader_get_entrypoint(shader));

      /* Recreate XFB info from intrinsics (nir_opt_varyings may have changed it). */
      if (shader->xfb_info) {
         nir_gather_xfb_info_from_intrinsics(shader);
      }
   }

   /* Fill linked I/O info.
    * This needs to be done after all optimizations are done and shader info gathered.
    */
   for (int i = 0; i < ARRAY_SIZE(graphics_shader_order); ++i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      const mesa_shader_stage next = stages[s].info.next_stage;
      if (!stages[s].nir || next == MESA_SHADER_NONE || !stages[next].nir)
         continue;

      radv_graphics_shaders_fill_linked_io_info(&stages[s], &stages[next]);
   }
}

struct radv_ps_epilog_key
radv_generate_ps_epilog_key(const struct radv_compiler_info *compiler_info, const struct radv_ps_epilog_state *state)
{
   unsigned col_format = 0, is_int8 = 0, is_int10 = 0, is_float32 = 0, z_format = 0, no_signed_zero = 0;
   struct radv_ps_epilog_key key;

   memset(&key, 0, sizeof(key));
   memset(key.color_map, MESA_VK_ATTACHMENT_UNUSED, sizeof(key.color_map));

   for (unsigned i = 0; i < state->color_attachment_count; ++i) {
      unsigned cf;
      unsigned cb_idx = state->color_attachment_mappings[i];
      VkFormat fmt = state->color_attachment_formats[i];

      if (fmt == VK_FORMAT_UNDEFINED || !(state->color_write_mask & (0xfu << (i * 4))) ||
          cb_idx == MESA_VK_ATTACHMENT_UNUSED) {
         cf = V_028714_SPI_SHADER_ZERO;
      } else {
         const bool blend_enable = (state->color_blend_enable >> i) & 0x1u;

         cf = radv_choose_spi_color_format(compiler_info, fmt, blend_enable, state->need_src_alpha & (1 << i));

         uint32_t comp_used = util_format_colormask(vk_format_description(fmt));

         comp_used &= (state->color_write_mask >> (i * 4));
         comp_used |= ((state->need_src_alpha >> i) & 0x1) << 3;

         key.colors_needed |= comp_used << (4 * i);

         if (format_ignores_signed_zero(fmt) || blend_enable)
            no_signed_zero |= 1 << i;
         if (format_is_int8(fmt))
            is_int8 |= 1 << i;
         if (format_is_int10(fmt))
            is_int10 |= 1 << i;
         if (format_is_float32(fmt))
            is_float32 |= 1 << i;
      }

      col_format |= cf << (4 * i);

      key.color_map[i] = state->color_attachment_mappings[i];
   }

   if (!(col_format & 0xf) && state->need_src_alpha & (1 << 0)) {
      /* When a subpass doesn't have any color attachments, write the alpha channel of MRT0 when
       * alpha coverage is enabled because the depth attachment needs it.
       */
      col_format |= V_028714_SPI_SHADER_32_AR;
      key.color_map[0] = 0;
      key.colors_needed |= 0x8;
   }

   /* The output for dual source blending should have the same format as the first output. */
   if (state->mrt0_is_dual_src) {
      assert(!(col_format >> 4));
      col_format |= (col_format & 0xf) << 4;
      key.color_map[1] = 1;
      key.colors_needed |= (key.colors_needed & 0xf) << 4;
      no_signed_zero |= 0x2;
   }

   z_format = ac_get_spi_shader_z_format(state->export_depth, state->export_stencil, state->export_sample_mask,
                                         state->alpha_to_coverage_via_mrtz);

   key.spi_shader_col_format = col_format;
   key.color_is_int8 = compiler_info->ac->has_cb_lt16bit_int_clamp_bug ? is_int8 : 0;
   key.color_is_int10 = compiler_info->ac->has_cb_lt16bit_int_clamp_bug ? is_int10 : 0;
   key.enable_mrt_output_nan_fixup = compiler_info->key.enable_mrt_output_nan_fixup ? is_float32 : 0;
   key.no_signed_zero = no_signed_zero;
   key.colors_written = state->colors_written;
   key.mrt0_is_dual_src = state->mrt0_is_dual_src && key.colors_needed & 0xf;
   key.export_depth = state->export_depth;
   key.export_stencil = state->export_stencil;
   key.export_sample_mask = state->export_sample_mask;
   key.alpha_to_coverage_via_mrtz = state->alpha_to_coverage_via_mrtz;
   key.spi_shader_z_format = z_format;
   key.alpha_to_one = state->alpha_to_one;

   return key;
}

static struct radv_ps_epilog_key
radv_pipeline_generate_ps_epilog_key(const struct radv_compiler_info *compiler_info,
                                     const struct vk_graphics_pipeline_state *state)
{
   struct radv_ps_epilog_state ps_epilog = {0};

   if (state->ms && state->ms->alpha_to_coverage_enable)
      ps_epilog.need_src_alpha |= 0x1;

   if (state->cb) {
      for (uint32_t i = 0; i < state->cb->attachment_count; i++) {
         VkBlendOp eqRGB = state->cb->attachments[i].color_blend_op;
         VkBlendFactor srcRGB = state->cb->attachments[i].src_color_blend_factor;
         VkBlendFactor dstRGB = state->cb->attachments[i].dst_color_blend_factor;

         /* Ignore other blend targets if dual-source blending is enabled to prevent wrong
          * behaviour.
          */
         if (i > 0 && ps_epilog.mrt0_is_dual_src)
            continue;

         ps_epilog.color_write_mask |= (unsigned)state->cb->attachments[i].write_mask << (4 * i);
         if (!((ps_epilog.color_write_mask >> (i * 4)) & 0xf))
            continue;

         ps_epilog.color_blend_enable |= state->cb->attachments[i].blend_enable << i;

         if (!((ps_epilog.color_blend_enable >> i) & 0x1u))
            continue;

         if (i == 0 && radv_can_enable_dual_src(&state->cb->attachments[i])) {
            ps_epilog.mrt0_is_dual_src = true;
         }

         radv_normalize_blend_factor(eqRGB, &srcRGB, &dstRGB);

         if (srcRGB == VK_BLEND_FACTOR_SRC_ALPHA || dstRGB == VK_BLEND_FACTOR_SRC_ALPHA ||
             srcRGB == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE || dstRGB == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE ||
             srcRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA || dstRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA ||
             srcRGB == VK_BLEND_FACTOR_SRC1_ALPHA || dstRGB == VK_BLEND_FACTOR_SRC1_ALPHA ||
             srcRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA || dstRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA)
            ps_epilog.need_src_alpha |= 1 << i;
      }
   }

   if (state->rp) {
      ps_epilog.color_attachment_count = state->rp->color_attachment_count;

      for (uint32_t i = 0; i < ps_epilog.color_attachment_count; i++) {
         ps_epilog.color_attachment_formats[i] = state->rp->color_attachment_formats[i];
      }
   }

   if (state->ms)
      ps_epilog.alpha_to_one = state->ms->alpha_to_one_enable;

   for (uint32_t i = 0; i < MAX_RTS; i++) {
      ps_epilog.color_attachment_mappings[i] = state->cal ? state->cal->color_map[i] : i;
   }

   return radv_generate_ps_epilog_key(compiler_info, &ps_epilog);
}

static struct radv_graphics_state_key
radv_generate_graphics_state_key(const struct radv_compiler_info *compiler_info,
                                 const struct vk_graphics_pipeline_state *state,
                                 VkGraphicsPipelineLibraryFlagBitsEXT lib_flags, uint32_t custom_blend_mode)
{
   struct radv_graphics_state_key key;

   memset(&key, 0, sizeof(key));

   key.lib_flags = lib_flags;
   key.has_multiview_view_index = state->mv ? !!state->mv->view_mask : 0;

   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI)) {
      key.vs.has_prolog = true;
   }

   /* Make sure to require a VS prolog when the VS is compiled without the vertex input state (this
    * can happen with GPL).
    */
   if ((state->shader_stages & VK_SHADER_STAGE_VERTEX_BIT) && !state->vi) {
      key.vs.has_prolog = true;
   }

   /* Vertex input state */
   if (state->vi) {
      key.vi.attributes_valid = state->vi->attributes_valid;

      u_foreach_bit (i, state->vi->attributes_valid) {
         uint32_t binding = state->vi->attributes[i].binding;
         uint32_t offset = state->vi->attributes[i].offset;
         enum pipe_format format = radv_format_to_pipe_format(state->vi->attributes[i].format);

         key.vi.vertex_attribute_formats[i] = format;
         key.vi.vertex_attribute_bindings[i] = binding;
         key.vi.vertex_attribute_offsets[i] = offset;
         key.vi.instance_rate_divisors[i] = state->vi->bindings[binding].divisor;

         /* vertex_attribute_strides is only needed to workaround GFX6/7 offset>=stride checks. */
         if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDING_STRIDES) && compiler_info->ac->gfx_level < GFX8) {
            /* From the Vulkan spec 1.2.157:
             *
             * "If the bound pipeline state object was created with the
             * VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE dynamic state enabled then pStrides[i]
             * specifies the distance in bytes between two consecutive elements within the
             * corresponding buffer. In this case the VkVertexInputBindingDescription::stride state
             * from the pipeline state object is ignored."
             *
             * Make sure the vertex attribute stride is zero to avoid computing a wrong offset if
             * it's initialized to something else than zero.
             */
            key.vi.vertex_attribute_strides[i] = state->vi->bindings[binding].stride;
         }

         if (state->vi->bindings[binding].input_rate) {
            key.vi.instance_rate_inputs |= 1u << i;
         }

         const struct ac_vtx_format_info *vtx_info = ac_get_vtx_format_info(
            compiler_info->ac->gfx_level, compiler_info->ac->has_vtx_format_alpha_adjust_bug, format);
         unsigned attrib_align = vtx_info->chan_byte_size ? vtx_info->chan_byte_size : vtx_info->element_size;

         /* If offset is misaligned, then the buffer offset must be too. Just skip updating
          * vertex_binding_align in this case.
          */
         if (offset % attrib_align == 0) {
            key.vi.vertex_binding_align[binding] = MAX2(key.vi.vertex_binding_align[binding], attrib_align);
         }
      }
   }

   if (state->ts)
      key.ts.patch_control_points = state->ts->patch_control_points;

   const bool alpha_to_coverage_unknown =
      !state->ms || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE);
   const bool alpha_to_coverage_enabled = alpha_to_coverage_unknown || state->ms->alpha_to_coverage_enable;
   const bool alpha_to_one_unknown = !state->ms || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE);
   const bool alpha_to_one_enabled = alpha_to_one_unknown || state->ms->alpha_to_one_enable;

   /* alpha-to-coverage is always exported via MRTZ on GFX11 but it's also using MRTZ when
    * alpha-to-one is enabled (alpha to MRTZ.a and one to MRT0.a).
    */
   key.ms.alpha_to_coverage_via_mrtz =
      alpha_to_coverage_enabled && (compiler_info->ac->gfx_level >= GFX11 || alpha_to_one_enabled);

   if (state->ms) {
      key.ms.sample_shading_enable = state->ms->sample_shading_enable;
      key.ms.max_sample_shading_enable = state->ms->sample_shading_enable && state->ms->min_sample_shading == 1;

      if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
          state->ms->rasterization_samples > 1) {
         key.ms.rasterization_samples = state->ms->rasterization_samples;

         if (state->ms->sample_shading_enable) {
            key.ms.ps_iter_samples =
               util_next_power_of_two(ceilf(state->ms->rasterization_samples * state->ms->min_sample_shading));
         }
      }
   }

   if (state->ia) {
      key.ia.topology = radv_translate_prim(state->ia->primitive_topology);
   }

   key.rs.polygon_mode_unknown = !state->rs || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE);

   if (!key.rs.polygon_mode_unknown) {
      switch (state->rs->polygon_mode) {
      case VK_POLYGON_MODE_FILL:
      case VK_POLYGON_MODE_LINE:
      case VK_POLYGON_MODE_POINT:
         key.rs.polygon_mode = state->rs->polygon_mode;
         break;
      default:
         UNREACHABLE("unexpected polygon mode");
      }
   }

   key.bc250_ps_dynamic_provoking =
      compiler_info->hw.bc250_barycentrics && BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_PROVOKING_VERTEX);

   if (state->rs) {
      if (compiler_info->ac->gfx_level >= GFX10)
         key.rs.provoking_vtx_last = state->rs->provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;

      if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_CULL_MODE))
         key.rs.cull_mode = state->rs->cull_mode;
   }

   key.dynamic_rasterization_samples = BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) ||
                                       (!!(state->shader_stages & VK_SHADER_STAGE_FRAGMENT_BIT) && !state->ms);

   const bool vrs_disabled_by_msaa_8x = !key.dynamic_rasterization_samples && key.ms.rasterization_samples == 8;
   const bool vrs_disabled_by_sample_shading = key.ms.sample_shading_enable;
   const bool vrs_is_possible = !vrs_disabled_by_msaa_8x && !vrs_disabled_by_sample_shading;

   key.ps.force_vrs_enabled = vrs_is_possible && compiler_info->force_vrs_enabled && !radv_is_static_vrs_enabled(state);
   /* RADV_BC250_VRS_NOOP: the rate is always 1x1, whatever the pipeline or dynamic state says. */
   key.vrs_may_be_enabled =
      !compiler_info->hw.bc250_vrs_noop && vrs_is_possible &&
      (radv_is_static_vrs_enabled(state) || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_FSR) ||
       key.ps.force_vrs_enabled);

   if (key.vrs_may_be_enabled && compiler_info->ac->has_vrs_frag_pos_z_bug)
      key.adjust_frag_coord_z = true;

   if (radv_pipeline_needs_ps_epilog(state, lib_flags))
      key.ps.has_epilog = true;

   key.ps.epilog = radv_pipeline_generate_ps_epilog_key(compiler_info, state);

   /* Alpha to coverage is exported via MRTZ when depth/stencil/samplemask are also exported.
    * Though, when a PS epilog is needed and the MS state is NULL (with dynamic rendering), it's not
    * possible to know the info at compile time and MRTZ needs to be exported in the epilog.
    */
   if (key.ps.has_epilog) {
      if (compiler_info->ac->gfx_level >= GFX11) {
         key.ps.exports_mrtz_via_epilog = alpha_to_coverage_unknown;
      } else {
         key.ps.exports_mrtz_via_epilog =
            (alpha_to_coverage_unknown && alpha_to_one_enabled) || (alpha_to_one_unknown && alpha_to_coverage_enabled);
      }
   }

   /* Set whether alpha_to_one makes MRT0 alpha dead. */
   key.ps.mrt0_alpha_is_dead = !alpha_to_one_unknown && !alpha_to_coverage_unknown && state->ms->alpha_to_one_enable &&
                               !state->ms->alpha_to_coverage_enable;

   if (compiler_info->key.use_ngg) {
      VkShaderStageFlags ngg_stage;

      if (state->shader_stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
         ngg_stage = VK_SHADER_STAGE_GEOMETRY_BIT;
      } else if (state->shader_stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
         ngg_stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
      } else if (state->shader_stages & VK_SHADER_STAGE_MESH_BIT_EXT) {
         ngg_stage = VK_SHADER_STAGE_MESH_BIT_EXT;
      } else {
         ngg_stage = VK_SHADER_STAGE_VERTEX_BIT;
      }

      /* Mesh: the BC250 Mesh paths that place per-primitive data on one provoking corner
       * (the base driver's compact vertex map, RADV_BC250_MESH_MERGE stage 2,
       * RADV_BC250_MESH_COMPACT owned corners) must know that the mode can change at draw
       * time. Mesh pipelines always had the flag (their NGG stage used to fall through to
       * the vertex case above); it is spelled out so that it cannot get lost. */
      key.dynamic_provoking_vtx_mode =
         BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_PROVOKING_VERTEX) &&
         (ngg_stage == VK_SHADER_STAGE_VERTEX_BIT || ngg_stage == VK_SHADER_STAGE_GEOMETRY_BIT ||
          ngg_stage == VK_SHADER_STAGE_MESH_BIT_EXT);
   }

   if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY) && state->ia &&
       state->ia->primitive_topology != VK_PRIMITIVE_TOPOLOGY_POINT_LIST &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE) && state->rs &&
       state->rs->polygon_mode != VK_POLYGON_MODE_POINT) {
      key.enable_remove_point_size = true;
   }

   if (compiler_info->smooth_lines) {
      /* Make the line rasterization mode dynamic for smooth lines to conditionally enable the lowering at draw time.
       * This is because it's not possible to know if the graphics pipeline will draw lines at this point and it also
       * simplifies the implementation.
       */
      if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_LINE_MODE) ||
          (state->rs && state->rs->line.mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH))
         key.dynamic_line_rast_mode = true;

      /* For GPL, when the fragment shader is compiled without any pre-rasterization information,
       * ensure the line rasterization mode is considered dynamic because we can't know if it's
       * going to draw lines or not.
       */
      key.dynamic_line_rast_mode |= !!(lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) &&
                                    !(lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT);
   }

   key.dcc_decompress_gfx11 =
      compiler_info->ac->gfx_level >= GFX11 && custom_blend_mode == V_028808_CB_DCC_DECOMPRESS_GFX11;

   return key;
}

static struct radv_graphics_pipeline_key
radv_generate_graphics_pipeline_key(const struct radv_device *device, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                    const struct vk_graphics_pipeline_state *state,
                                    VkGraphicsPipelineLibraryFlagBitsEXT lib_flags)
{
   const struct radv_compiler_info *compiler_info = &device->compiler_info;
   VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   struct radv_graphics_pipeline_key key = {0};
   uint32_t custom_blend_mode = 0;

   const VkGraphicsPipelineCreateInfoRADV *radv_info =
      vk_find_struct_const(pCreateInfo->pNext, GRAPHICS_PIPELINE_CREATE_INFO_RADV);
   if (radv_info) {
      custom_blend_mode = radv_info->custom_blend_mode;
   }

   key.gfx_state = radv_generate_graphics_state_key(compiler_info, state, lib_flags, custom_blend_mode);

   for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *stage = &pCreateInfo->pStages[i];
      mesa_shader_stage s = vk_to_mesa_shader_stage(stage->stage);

      key.stage_info[s] = radv_pipeline_get_shader_key(compiler_info, stage, create_flags, pCreateInfo->pNext);

      if (s == MESA_SHADER_MESH && (state->shader_stages & VK_SHADER_STAGE_TASK_BIT_EXT))
         key.stage_info[s].has_task_shader = true;
   }

   return key;
}

static void
radv_fill_shader_info_ngg(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                          VkShaderStageFlagBits active_nir_stages)
{
   if (!compiler_info->key.use_ngg)
      return;

   if (stages[MESA_SHADER_VERTEX].nir && stages[MESA_SHADER_VERTEX].info.next_stage != MESA_SHADER_TESS_CTRL) {
      stages[MESA_SHADER_VERTEX].info.is_ngg = true;
   } else if (stages[MESA_SHADER_TESS_EVAL].nir) {
      stages[MESA_SHADER_TESS_EVAL].info.is_ngg = true;
   } else if (stages[MESA_SHADER_MESH].nir) {
      stages[MESA_SHADER_MESH].info.is_ngg = true;
   }

   if (compiler_info->ac->gfx_level >= GFX11) {
      if (stages[MESA_SHADER_GEOMETRY].nir)
         stages[MESA_SHADER_GEOMETRY].info.is_ngg = true;
   } else {
      /* GFX10/GFX10.3 can't always enable NGG due to HW bugs/limitations. */
      if (stages[MESA_SHADER_TESS_EVAL].nir && stages[MESA_SHADER_GEOMETRY].nir &&
          stages[MESA_SHADER_GEOMETRY].nir->info.gs.invocations *
                stages[MESA_SHADER_GEOMETRY].nir->info.gs.vertices_out >
             256) {
         /* Fallback to the legacy path if tessellation is
          * enabled with extreme geometry because
          * EN_MAX_VERT_OUT_PER_GS_INSTANCE doesn't work and it
          * might hang.
          */
         stages[MESA_SHADER_TESS_EVAL].info.is_ngg = false;
      }

      struct radv_shader_stage *last_vgt_stage = NULL;
      radv_foreach_stage (i, active_nir_stages) {
         if (radv_is_last_vgt_stage(&stages[i])) {
            last_vgt_stage = &stages[i];
         }
      }

      if ((last_vgt_stage && last_vgt_stage->nir->xfb_info) ||
          (compiler_info->key.no_ngg_gs && stages[MESA_SHADER_GEOMETRY].nir)) {
         /* NGG needs to be disabled on GFX10/GFX10.3 when:
          * - streamout is used because NGG streamout isn't supported
          * - NGG GS is explictly disabled to workaround performance issues
          */
         if (stages[MESA_SHADER_TESS_EVAL].nir)
            stages[MESA_SHADER_TESS_EVAL].info.is_ngg = false;
         else
            stages[MESA_SHADER_VERTEX].info.is_ngg = false;
      }

      if (stages[MESA_SHADER_GEOMETRY].nir) {
         if (stages[MESA_SHADER_TESS_EVAL].nir)
            stages[MESA_SHADER_GEOMETRY].info.is_ngg = stages[MESA_SHADER_TESS_EVAL].info.is_ngg;
         else
            stages[MESA_SHADER_GEOMETRY].info.is_ngg = stages[MESA_SHADER_VERTEX].info.is_ngg;
      }

      /* When pre-rasterization stages are compiled separately with shader objects, NGG GS needs to
       * be disabled because if the next stage of VS/TES is GS and GS is unknown, it might use
       * streamout but it's not possible to know that when compiling VS or TES only.
       */
      if (stages[MESA_SHADER_VERTEX].nir && stages[MESA_SHADER_VERTEX].info.next_stage == MESA_SHADER_GEOMETRY &&
          !stages[MESA_SHADER_GEOMETRY].nir) {
         stages[MESA_SHADER_VERTEX].info.is_ngg = false;
      } else if (stages[MESA_SHADER_TESS_EVAL].nir &&
                 stages[MESA_SHADER_TESS_EVAL].info.next_stage == MESA_SHADER_GEOMETRY &&
                 !stages[MESA_SHADER_GEOMETRY].nir) {
         stages[MESA_SHADER_TESS_EVAL].info.is_ngg = false;
      } else if (stages[MESA_SHADER_GEOMETRY].nir &&
                 (!stages[MESA_SHADER_VERTEX].nir && !stages[MESA_SHADER_TESS_EVAL].nir)) {
         stages[MESA_SHADER_GEOMETRY].info.is_ngg = false;
      }
   }

   /* Now that we know if ngg is used for geometry shaders, determine the subgroup size. */
   if (stages[MESA_SHADER_GEOMETRY].nir) {
      unsigned wave_size = stages[MESA_SHADER_GEOMETRY].info.is_ngg
                              ? stages[MESA_SHADER_GEOMETRY].nir->info.min_subgroup_size
                              : stages[MESA_SHADER_GEOMETRY].nir->info.max_subgroup_size;
      stages[MESA_SHADER_GEOMETRY].nir->info.max_subgroup_size = wave_size;
      stages[MESA_SHADER_GEOMETRY].nir->info.min_subgroup_size = wave_size;
   }
}

static bool
radv_consider_force_vrs(const struct radv_graphics_state_key *gfx_state, const struct radv_shader_stage *last_vgt_stage,
                        const struct radv_shader_stage *fs_stage)
{
   if (!gfx_state->ps.force_vrs_enabled)
      return false;

   /* Mesh shaders aren't considered. */
   if (last_vgt_stage->info.stage == MESA_SHADER_MESH)
      return false;

   if (last_vgt_stage->nir->info.outputs_written & VARYING_BIT_PRIMITIVE_SHADING_RATE)
      return false;

   /* VRS has no effect if there is no pixel shader. */
   if (last_vgt_stage->info.next_stage == MESA_SHADER_NONE)
      return false;

   return true;
}

static mesa_shader_stage
radv_get_next_stage(mesa_shader_stage stage, VkShaderStageFlagBits active_nir_stages)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      if (active_nir_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
         return MESA_SHADER_TESS_CTRL;
      } else if (active_nir_stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
         return MESA_SHADER_GEOMETRY;
      } else if (active_nir_stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
         return MESA_SHADER_FRAGMENT;
      } else {
         return MESA_SHADER_NONE;
      }
   case MESA_SHADER_TESS_CTRL:
      return MESA_SHADER_TESS_EVAL;
   case MESA_SHADER_TESS_EVAL:
      if (active_nir_stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
         return MESA_SHADER_GEOMETRY;
      } else if (active_nir_stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
         return MESA_SHADER_FRAGMENT;
      } else {
         return MESA_SHADER_NONE;
      }
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_MESH:
      if (active_nir_stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
         return MESA_SHADER_FRAGMENT;
      } else {
         return MESA_SHADER_NONE;
      }
   case MESA_SHADER_TASK:
      return MESA_SHADER_MESH;
   case MESA_SHADER_FRAGMENT:
      return MESA_SHADER_NONE;
   default:
      UNREACHABLE("invalid graphics shader stage");
   }
}

static void
radv_fill_shader_info(const struct radv_compiler_info *compiler_info, const enum radv_pipeline_type pipeline_type,
                      const struct radv_graphics_state_key *gfx_state, struct radv_shader_stage *stages,
                      VkShaderStageFlagBits active_nir_stages)
{
   radv_foreach_stage (i, active_nir_stages) {
      bool consider_force_vrs = false;

      if (radv_is_last_vgt_stage(&stages[i])) {
         consider_force_vrs = radv_consider_force_vrs(gfx_state, &stages[i], &stages[MESA_SHADER_FRAGMENT]);
      }

      if (i == MESA_SHADER_FRAGMENT)
         NIR_PASS(_, stages[i].nir, ac_nir_assign_fs_input_locations);

      radv_nir_shader_info_pass(compiler_info, stages[i].nir, &stages[i].layout, &stages[i].key, gfx_state,
                                pipeline_type, consider_force_vrs, &stages[i].info);
   }

   radv_nir_shader_info_link(compiler_info, gfx_state, stages);
}

static void
radv_declare_pipeline_args(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                           const struct radv_graphics_state_key *gfx_state, VkShaderStageFlagBits active_nir_stages,
                           struct radv_shader_debug_info *debug)
{
   enum amd_gfx_level gfx_level = compiler_info->ac->gfx_level;

   if (gfx_level >= GFX9 && stages[MESA_SHADER_TESS_CTRL].nir) {
      radv_declare_shader_args(compiler_info, gfx_state, &stages[MESA_SHADER_TESS_CTRL], MESA_SHADER_VERTEX,
                               &debug[MESA_SHADER_TESS_CTRL]);
      stages[MESA_SHADER_TESS_CTRL].info.user_sgprs_locs = stages[MESA_SHADER_TESS_CTRL].args.user_sgprs_locs;
      stages[MESA_SHADER_TESS_CTRL].info.inline_push_constant_mask =
         stages[MESA_SHADER_TESS_CTRL].args.ac.inline_push_const_mask;

      stages[MESA_SHADER_VERTEX].info.user_sgprs_locs = stages[MESA_SHADER_TESS_CTRL].info.user_sgprs_locs;
      stages[MESA_SHADER_VERTEX].info.inline_push_constant_mask =
         stages[MESA_SHADER_TESS_CTRL].info.inline_push_constant_mask;
      stages[MESA_SHADER_VERTEX].args = stages[MESA_SHADER_TESS_CTRL].args;

      active_nir_stages &= ~(1 << MESA_SHADER_VERTEX);
      active_nir_stages &= ~(1 << MESA_SHADER_TESS_CTRL);
   }

   if (gfx_level >= GFX9 && stages[MESA_SHADER_GEOMETRY].nir) {
      mesa_shader_stage pre_stage = stages[MESA_SHADER_TESS_EVAL].nir ? MESA_SHADER_TESS_EVAL : MESA_SHADER_VERTEX;
      radv_declare_shader_args(compiler_info, gfx_state, &stages[MESA_SHADER_GEOMETRY], pre_stage,
                               &debug[MESA_SHADER_GEOMETRY]);
      stages[MESA_SHADER_GEOMETRY].info.user_sgprs_locs = stages[MESA_SHADER_GEOMETRY].args.user_sgprs_locs;
      stages[MESA_SHADER_GEOMETRY].info.inline_push_constant_mask =
         stages[MESA_SHADER_GEOMETRY].args.ac.inline_push_const_mask;

      stages[pre_stage].info.user_sgprs_locs = stages[MESA_SHADER_GEOMETRY].info.user_sgprs_locs;
      stages[pre_stage].info.inline_push_constant_mask = stages[MESA_SHADER_GEOMETRY].info.inline_push_constant_mask;
      stages[pre_stage].args = stages[MESA_SHADER_GEOMETRY].args;
      active_nir_stages &= ~(1 << pre_stage);
      active_nir_stages &= ~(1 << MESA_SHADER_GEOMETRY);
   }

   u_foreach_bit (i, active_nir_stages) {
      radv_declare_shader_args(compiler_info, gfx_state, &stages[i], MESA_SHADER_NONE, &debug[i]);
      stages[i].info.user_sgprs_locs = stages[i].args.user_sgprs_locs;
      stages[i].info.inline_push_constant_mask = stages[i].args.ac.inline_push_const_mask;
   }
}

static struct radv_shader_binary *
radv_create_gs_copy_shader(const struct radv_compiler_info *compiler_info, struct vk_pipeline_cache *cache,
                           struct radv_shader_stage *gs_stage, const struct radv_graphics_state_key *gfx_state,
                           struct radv_shader_debug_info *gs_copy_debug)
{

   const struct radv_shader_info *gs_info = &gs_stage->info;
   nir_shader *nir = gs_stage->gs_copy_shader;

   nir_validate_shader(nir, "after ac_nir_create_gs_copy_shader");
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   struct radv_shader_stage gs_copy_stage = {
      .stage = MESA_SHADER_VERTEX,
      .shader_blake3 = {0},
      .key =
         {
            .optimisations_disabled = gs_stage->key.optimisations_disabled,
            .keep_statistic_info = gs_stage->key.keep_statistic_info,
            .keep_executable_info = gs_stage->key.keep_executable_info,
         },
   };
   radv_nir_shader_info_init(gs_copy_stage.stage, MESA_SHADER_FRAGMENT, &gs_copy_stage.info);
   radv_nir_shader_info_pass(compiler_info, nir, &gs_stage->layout, &gs_stage->key, gfx_state, RADV_PIPELINE_GRAPHICS,
                             false, &gs_copy_stage.info);
   gs_copy_stage.info.wave_size = 64;      /* Wave32 not supported. */
   gs_copy_stage.info.workgroup_size = 64; /* HW VS: separate waves, no workgroups */
   gs_copy_stage.info.so = gs_info->so;
   gs_copy_stage.info.outinfo = gs_info->outinfo;
   gs_copy_stage.info.force_vrs_per_vertex = gs_info->force_vrs_per_vertex;
   gs_copy_stage.info.type = RADV_SHADER_TYPE_GS_COPY;

   radv_declare_shader_args(compiler_info, gfx_state, &gs_copy_stage, MESA_SHADER_NONE, gs_copy_debug);
   gs_copy_stage.info.user_sgprs_locs = gs_copy_stage.args.user_sgprs_locs;
   gs_copy_stage.info.inline_push_constant_mask = gs_copy_stage.args.ac.inline_push_const_mask;

   NIR_PASS(_, nir, ac_nir_lower_intrinsics_to_args, &gs_copy_stage.args.ac,
            &(ac_nir_lower_intrinsics_to_args_options){.gfx_level = compiler_info->ac->gfx_level,
                                                       .has_ls_vgpr_init_bug = compiler_info->ac->has_ls_vgpr_init_bug,
                                                       .hw_stage = AC_HW_VERTEX_SHADER,
                                                       .wave_size = 64,
                                                       .workgroup_size = 64,
                                                       .use_llvm = compiler_info->key.use_llvm});
   NIR_PASS(_, nir, radv_nir_lower_abi, compiler_info->ac->gfx_level, &gs_copy_stage, gfx_state, compiler_info->hw.address32_hi);

   NIR_PASS(_, nir, ac_nir_lower_global_access, compiler_info->ac->gfx_level);
   NIR_PASS(_, nir, nir_lower_int64);

   struct radv_graphics_pipeline_key key = {0};
   gs_copy_debug->dump_shader = radv_can_dump_shader(compiler_info, nir);

   if (gs_copy_debug->dump_shader)
      simple_mtx_lock(compiler_info->debug.shader_dump_mtx);

   struct radv_shader_binary *gs_copy_binary =
      radv_shader_nir_to_asm(compiler_info, &gs_copy_stage, &nir, 1, &key.gfx_state);

   char *nir_string = NULL;
   if (gs_copy_stage.key.keep_executable_info)
      nir_string = radv_dump_nir_shaders(compiler_info, &nir, 1);

   gs_copy_debug->nir_string = nir_string;
   gs_copy_debug->stages = 1 << MESA_SHADER_VERTEX;
   radv_shader_dump_asm(compiler_info, gs_copy_debug, gs_copy_binary, &gs_copy_stage.info);

   if (gs_copy_debug->dump_shader)
      simple_mtx_unlock(compiler_info->debug.shader_dump_mtx);

   return gs_copy_binary;
}

/* RADV_BC250_ASYNC_COMPILE: the Mesh and fragment stages are compiled by ACO without optimizations first;
 * the job keeps a clone of their final NIR (ACO modifies its input) and the stage state, and compiles the
 * optimized binaries in the background. The NIR, route, plan, LDS layout and shader info are identical; only
 * the machine code differs. */
struct radv_bc250_async_job {
   struct util_queue_fence fence;
   struct radv_device *device;
   struct radv_graphics_pipeline *pipeline;
   bool skip_shaders_cache;
   uint32_t stage_mask;
   nir_shader *nir[MESA_VULKAN_SHADER_STAGES];
   struct radv_shader_stage stage[MESA_VULKAN_SHADER_STAGES];
   struct radv_graphics_state_key gfx_state;
};

static __thread struct radv_bc250_async_job *bc250_async_capture;

static void
radv_graphics_shaders_nir_to_asm(const struct radv_compiler_info *compiler_info, struct vk_pipeline_cache *cache,
                                 struct radv_shader_stage *stages, const struct radv_graphics_state_key *gfx_state,
                                 VkShaderStageFlagBits active_nir_stages, struct radv_shader_debug_info *debug,
                                 struct radv_shader_binary **binaries, struct radv_shader_debug_info *gs_copy_debug,
                                 struct radv_shader_binary **gs_copy_binary)
{
   for (int s = MESA_VULKAN_SHADER_STAGES - 1; s >= 0; s--) {
      if (!(active_nir_stages & (1 << s)))
         continue;

      nir_shader *nir_shaders[2] = {stages[s].nir, NULL};
      unsigned shader_count = 1;

      /* On GFX9+, TES is merged with GS and VS is merged with TCS or GS. */
      if (compiler_info->ac->gfx_level >= GFX9 &&
          ((s == MESA_SHADER_GEOMETRY &&
            (active_nir_stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT))) ||
           (s == MESA_SHADER_TESS_CTRL && (active_nir_stages & VK_SHADER_STAGE_VERTEX_BIT)))) {
         mesa_shader_stage pre_stage;

         if (s == MESA_SHADER_GEOMETRY && (active_nir_stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)) {
            pre_stage = MESA_SHADER_TESS_EVAL;
         } else {
            pre_stage = MESA_SHADER_VERTEX;
         }

         nir_shaders[0] = stages[pre_stage].nir;
         nir_shaders[1] = stages[s].nir;
         shader_count = 2;
      }

      int64_t stage_start = os_time_get_nano();

      for (unsigned i = 0; i < shader_count; ++i)
         debug[s].dump_shader |= radv_can_dump_shader(compiler_info, nir_shaders[i]);

      bool dump_nir = debug[s].dump_shader && compiler_info->debug.dump_nir;

      if (debug[s].dump_shader) {
         simple_mtx_lock(compiler_info->debug.shader_dump_mtx);

         if (dump_nir) {
            for (uint32_t i = 0; i < shader_count; i++)
               nir_print_shader(nir_shaders[i], stderr);
         }
      }

      if (bc250_async_capture && shader_count == 1 && (s == MESA_SHADER_MESH || s == MESA_SHADER_FRAGMENT)) {
         struct radv_bc250_async_job *job = bc250_async_capture;
         job->nir[s] = nir_shader_clone(NULL, stages[s].nir);
         job->stage[s] = stages[s];
         job->stage[s].nir = NULL;
         job->stage_mask |= 1u << s;
         struct radv_shader_stage quick = stages[s];
         quick.key.optimisations_disabled = 1;
         binaries[s] = radv_shader_nir_to_asm(compiler_info, &quick, nir_shaders, shader_count, gfx_state);
      } else {
         binaries[s] = radv_shader_nir_to_asm(compiler_info, &stages[s], nir_shaders, shader_count, gfx_state);
      }

      /* Dump NIR after nir_to_asm, because ACO modifies it. */
      char *nir_string = NULL;
      if (stages[s].key.keep_executable_info)
         nir_string = radv_dump_nir_shaders(compiler_info, nir_shaders, shader_count);

      debug[s].nir_string = nir_string;
      for (uint32_t i = 0; i < shader_count; i++)
         debug[s].stages |= 1 << nir_shaders[i]->info.stage;

      radv_shader_dump_asm(compiler_info, &debug[s], binaries[s], &stages[s].info);

      if (debug[s].dump_shader)
         simple_mtx_unlock(compiler_info->debug.shader_dump_mtx);

      if (stages[s].key.keep_executable_info && stages[s].spirv.size) {
         debug[s].spirv = malloc(stages[s].spirv.size);
         memcpy(debug[s].spirv, stages[s].spirv.data, stages[s].spirv.size);
         debug[s].spirv_size = stages[s].spirv.size;
      }

      if (s == MESA_SHADER_GEOMETRY && !stages[s].info.is_ngg) {
         *gs_copy_binary =
            radv_create_gs_copy_shader(compiler_info, cache, &stages[MESA_SHADER_GEOMETRY], gfx_state, gs_copy_debug);
      }

      stages[s].feedback.duration += os_time_get_nano() - stage_start;

      active_nir_stages &= ~(1 << nir_shaders[0]->info.stage);
      if (nir_shaders[1])
         active_nir_stages &= ~(1 << nir_shaders[1]->info.stage);
   }
}

static void
radv_pipeline_retain_shaders(struct radv_retained_shaders *retained_shaders, struct radv_shader_stage *stages)
{
   for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
      if (stages[s].stage == MESA_SHADER_NONE)
         continue;

      int64_t stage_start = os_time_get_nano();

      /* Serialize the NIR shader to reduce memory pressure. */
      struct blob blob;

      blob_init(&blob);
      nir_serialize(&blob, stages[s].nir, true);
      blob_finish_get_buffer(&blob, &retained_shaders->stages[s].serialized_nir,
                             &retained_shaders->stages[s].serialized_nir_size);

      memcpy(retained_shaders->stages[s].shader_blake3, stages[s].shader_blake3, sizeof(stages[s].shader_blake3));
      memcpy(&retained_shaders->stages[s].key, &stages[s].key, sizeof(stages[s].key));

      stages[s].feedback.duration += os_time_get_nano() - stage_start;
   }
}

static void
radv_pipeline_import_retained_shaders(const struct radv_device *device, struct radv_graphics_lib_pipeline *lib,
                                      struct radv_shader_stage *stages)
{
   struct radv_retained_shaders *retained_shaders = &lib->retained_shaders;

   /* Import the stages (SPIR-V only in case of cache hits). */
   for (uint32_t i = 0; i < lib->stage_count; i++) {
      const VkPipelineShaderStageCreateInfo *sinfo = &lib->stages[i];
      mesa_shader_stage s = vk_to_mesa_shader_stage(sinfo->stage);

      radv_pipeline_stage_init(lib->base.base.create_flags, sinfo, &lib->layout, &lib->stage_keys[s], &stages[s]);
   }

   /* Import the NIR shaders (after SPIRV->NIR). */
   for (uint32_t s = 0; s < ARRAY_SIZE(lib->base.base.shaders); s++) {
      if (!retained_shaders->stages[s].serialized_nir_size)
         continue;

      int64_t stage_start = os_time_get_nano();

      /* Deserialize the NIR shader. */
      const struct nir_shader_compiler_options *options = &device->compiler_info.nir_options[s];
      struct blob_reader blob_reader;
      blob_reader_init(&blob_reader, retained_shaders->stages[s].serialized_nir,
                       retained_shaders->stages[s].serialized_nir_size);

      stages[s].stage = s;
      stages[s].nir = nir_deserialize(NULL, options, &blob_reader);
      stages[s].bc250_imported_nir = true;
      stages[s].entrypoint = nir_shader_get_entrypoint(stages[s].nir)->function->name;
      memcpy(stages[s].shader_blake3, retained_shaders->stages[s].shader_blake3, sizeof(stages[s].shader_blake3));
      memcpy(&stages[s].key, &retained_shaders->stages[s].key, sizeof(stages[s].key));

      radv_shader_layout_init(&lib->layout, s, &stages[s].layout);

      stages[s].feedback.flags |= VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT;

      stages[s].feedback.duration += os_time_get_nano() - stage_start;
   }
}

static void
radv_pipeline_load_retained_shaders(const struct radv_device *device, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                    struct radv_shader_stage *stages)
{
   const VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);

   /* Nothing to load if no libs are imported. */
   if (!libs_info)
      return;

   /* Nothing to load if fast-linking is enabled and if there is no retained shaders. */
   if (radv_should_import_lib_binaries(create_flags) && !device->bc250_env.gpl_source_link)
      return;

   for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
      VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
      struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

      radv_pipeline_import_retained_shaders(device, gfx_pipeline_lib, stages);
   }
}

static unsigned
radv_get_num_raster_vertices_per_prim(const struct radv_shader_stage *stages,
                                      const struct radv_graphics_state_key *gfx_state)
{
   unsigned vgt_outprim_type;

   /* If VS or MS is present, it means we have all pre-rasterization shaders. We can't determine
    * the raster primitive type without them.
    */
   if (!stages[MESA_SHADER_VERTEX].nir && !stages[MESA_SHADER_MESH].nir)
      return 0; /* unknown */

   /* The pre-raster primitive type is determined from enabled shaders and the input topology. */
   if (stages[MESA_SHADER_GEOMETRY].nir) {
      vgt_outprim_type = radv_conv_gl_prim_to_gs_out(stages[MESA_SHADER_GEOMETRY].nir->info.gs.output_primitive);
   } else if (stages[MESA_SHADER_TESS_EVAL].nir) {
      if (stages[MESA_SHADER_TESS_EVAL].nir->info.tess.point_mode) {
         vgt_outprim_type = V_028A6C_POINTLIST;
      } else {
         vgt_outprim_type = radv_conv_tess_prim_to_gs_out(stages[MESA_SHADER_TESS_EVAL].nir->info.tess._primitive_mode);
      }
   } else if (stages[MESA_SHADER_MESH].nir) {
      vgt_outprim_type = radv_conv_gl_prim_to_gs_out(stages[MESA_SHADER_MESH].nir->info.mesh.primitive_type);
   } else {
      if (gfx_state->ia.topology == V_008958_DI_PT_NONE)
         return 0; /* unknown */

      vgt_outprim_type = radv_conv_prim_to_gs_out(gfx_state->ia.topology, false);
   }

   /* The rasterized primitive type is determined from the pre-raster primitive type and the polygon mode. */
   switch (vgt_outprim_type) {
   case V_028A6C_POINTLIST:
      return 1;

   case V_028A6C_LINESTRIP:
      return 2;

   case V_028A6C_TRISTRIP:
      if (!gfx_state->rs.polygon_mode_unknown) {
         switch (gfx_state->rs.polygon_mode) {
         case VK_POLYGON_MODE_POINT:
            return 1;
         case VK_POLYGON_MODE_LINE:
            return 2;
         default:
            return 3;
         }
      }
      break;

   default:
      UNREACHABLE("invalid vgt_outprim_type");
   }

   return 0;
}

static bool
radv_is_fast_linking_enabled(const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   const VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);

   if (!libs_info)
      return false;

   return !(create_flags & VK_PIPELINE_CREATE_2_LINK_TIME_OPTIMIZATION_BIT_EXT);
}

static bool
radv_skip_graphics_pipeline_compile(const struct radv_device *device, const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   const VkPipelineBinaryInfoKHR *binary_info = vk_find_struct_const(pCreateInfo->pNext, PIPELINE_BINARY_INFO_KHR);
   const VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   VkShaderStageFlagBits binary_stages = 0;
   VkShaderStageFlags active_stages = 0;

   /* No compilation when pipeline binaries are imported. */
   if (binary_info && binary_info->binaryCount > 0)
      return true;

   /* Do not skip for libraries. */
   if (create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)
      return false;

   /* Do not skip when fast-linking isn't enabled. */
   if (!radv_is_fast_linking_enabled(pCreateInfo))
      return false;

   for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *sinfo = &pCreateInfo->pStages[i];
      active_stages |= sinfo->stage;
   }

   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
   if (libs_info) {
      for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

         assert(pipeline_lib->type == RADV_PIPELINE_GRAPHICS_LIB);

         active_stages |= gfx_pipeline_lib->base.active_stages;

         for (uint32_t s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
            if (!gfx_pipeline_lib->base.base.shaders[s])
               continue;

            binary_stages |= mesa_to_vk_shader_stage(s);
         }
      }
   }

   if (pdev->info.gfx_level >= GFX9) {
      /* On GFX9+, TES is merged with GS and VS is merged with TCS or GS. */
      if (binary_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
         binary_stages |= VK_SHADER_STAGE_VERTEX_BIT;
      }

      if (binary_stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
         if (binary_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
            binary_stages |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
         } else {
            binary_stages |= VK_SHADER_STAGE_VERTEX_BIT;
         }
      }
   }

   /* Only skip compilation when all binaries have been imported. */
   return binary_stages == active_stages;
}

/* BC250 (GFX1013) barycentrics: export the vertex-order reference from the last
 * pre-rasterization stage and resolve the fragment shader's rotation
 * (radv_nir_bc250_lower_bary_rotation). Only fragment shaders that read
 * barycentric coordinates or strict per-vertex inputs are affected; every other
 * pipeline compiles exactly as before. Runs on linked, lowered I/O, after
 * nir_opt_varyings, so that no varying optimization merges or drops the two
 * reference slots, and before the FS input locations are assigned.
 */
static VkResult
radv_bc250_link_bary_rotation(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                              const struct radv_graphics_state_key *gfx_state)
{
   nir_shader *fs = stages[MESA_SHADER_FRAGMENT].nir;

   if (!fs || !compiler_info->hw.bc250_barycentrics || compiler_info->ac->gfx_level >= GFX10_3 ||
       !radv_nir_bc250_fs_needs_bary_rotation(fs))
      return VK_SUCCESS;

   struct radv_shader_stage *producer = NULL;
   for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
      if (s != MESA_SHADER_FRAGMENT && stages[s].nir && stages[s].info.next_stage == MESA_SHADER_FRAGMENT)
         producer = &stages[s];
   }

   const unsigned num_raster_vertices_per_prim = radv_get_num_raster_vertices_per_prim(stages, gfx_state);
   const char *fallback = NULL;
   int raw_slot = -1, flat_slot = -1;

   if (!producer) {
      /* Fragment shader library or unlinked shader object. */
      fallback = "no producer";
   } else if (compiler_info->hw.bc250_bary_no_ref) {
      /* Diagnostic only: wrong for the triangles the hardware rotates. */
      fallback = "RADV_BC250_DIAG_BARY_NO_REF";
   } else if (num_raster_vertices_per_prim && num_raster_vertices_per_prim != 3) {
      /* Points and lines are not rotated. */
      fallback = "points or lines";
   } else if (producer->stage == MESA_SHADER_MESH && producer->bc250_safe_owned && compiler_info->key.bc250_mesh_safe_corners &&
              debug_get_bool_option("RADV_BC250_BARY_PRIVATE_ROT0", false)) {
      /* Hardware experiment: private corners export every triangle as three fresh ascending
       * vertices (3p, 3p+1, 3p+2); if the parameter cache then never rotates, no reference
       * parameters are needed (also when all 32 PARAMs are used). Proven only by the GPU CTS. */
      fallback = "private corners (RADV_BC250_BARY_PRIVATE_ROT0)";
   } else {
      nir_shader *nir = producer->nir;
      /* Admission and linking use the same conservative PARAM-capacity proof. */
      int cid_slot = -1;
      /* RADV_BC250_BARY_CORNER_ID_FORCE=1 (validation only): use the corner id even when two
       * reference slots are free, so the barycentric CTS exercises it on hardware. */
      const bool cid_force = debug_get_bool_option("RADV_BC250_BARY_CORNER_ID_FORCE", false);
      if ((cid_force || !radv_bc250_bary_ref_slots(nir, fs, &raw_slot, &flat_slot)) &&
          producer->stage == MESA_SHADER_MESH && producer->bc250_safe_owned &&
          compiler_info->key.bc250_mesh_safe_corners && debug_get_bool_option("RADV_BC250_BARY_CORNER_ID", false) &&
          radv_bc250_bary_cid_slot(nir, fs, &cid_slot)) {
         /* RADV_BC250_BARY_CORNER_ID: one slot carries each private corner's number (0, 1, 2);
          * the fragment shader finds the raw slot holding corner 0. */
         raw_slot = cid_slot;
         flat_slot = -1;
         NIR_PASS(_, nir, radv_nir_bc250_export_bary_ref, cid_slot, cid_slot);
         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
         producer->bc250_bary_ref_mask = BITFIELD64_BIT(cid_slot);
      } else if (raw_slot < 0) {
         if (compiler_info->key.bc250_mesh_safe_bary_last && producer->stage == MESA_SHADER_MESH) {
            /* Split uses the same FS ABI and cannot repair missing rotation
             * information. Fail closed in the opt-in conformance mode rather
             * than silently executing the old, incorrect rotation-zero path. */
            fprintf(stderr, "BC250 Mesh pipeline refused: bary_reference_capacity_unresolved; "
                            "SAFE_BARY_LAST requires a proven rotation reference\n");
            return VK_ERROR_FEATURE_NOT_PRESENT;
         }
         fallback = "no two free parameters";
      } else {
         NIR_PASS(_, nir, radv_nir_bc250_export_bary_ref, raw_slot, flat_slot);
         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
         producer->bc250_bary_ref_mask = BITFIELD64_BIT(raw_slot) | BITFIELD64_BIT(flat_slot);
      }
   }

   if (getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 BARYCENTRICS: %s producer=%s raw=%d flat=%d raster_vertices=%u provoking=%s%s%s\n",
              fallback ? "rotation 0" : flat_slot < 0 ? "rotation from corner id" : "rotation from reference", producer ? mesa_shader_stage_name(producer->stage) : "none",
              raw_slot >= 0 ? raw_slot - VARYING_SLOT_VAR0 : -1, flat_slot >= 0 ? flat_slot - VARYING_SLOT_VAR0 : -1,
              num_raster_vertices_per_prim,
              gfx_state->bc250_ps_dynamic_provoking ? "dynamic" : gfx_state->rs.provoking_vtx_last ? "last" : "first",
              fallback ? " reason=" : "", fallback ? fallback : "");

   NIR_PASS(_, fs, radv_nir_bc250_lower_bary_rotation, raw_slot, flat_slot, num_raster_vertices_per_prim,
            gfx_state->bc250_ps_dynamic_provoking, gfx_state->rs.provoking_vtx_last);
   if (producer) {
      producer->bc250_bary_raw_slot = raw_slot;
      producer->bc250_bary_flat_slot = flat_slot;
   }
   nir_shader_gather_info(fs, nir_shader_get_entrypoint(fs));
   return VK_SUCCESS;
}

static enum radv_bc250_mesh_route_reason
radv_bc250_mesh_no_split_reason(const struct radv_compiler_info *compiler_info,
                                 struct radv_shader_stage *stages,
                                 const struct radv_graphics_state_key *gfx_state)
{
   struct radv_shader_stage *ms = &stages[MESA_SHADER_MESH];
   nir_shader *mesh = ms->nir;
   nir_shader *fs = stages[MESA_SHADER_FRAGMENT].nir;
   if (ms->bc250_safe_direct)
      return RADV_BC250_ROUTE_REASON_UNKNOWN;
   if (!compiler_info->key.bc250_mesh_safe_fast && !compiler_info->key.bc250_mesh_safe_pieces &&
       !compiler_info->key.bc250_mesh_safe_owned && !compiler_info->key.bc250_mesh_safe_direct)
      return RADV_BC250_ROUTE_REASON_SWITCHES_OFF;
   if (stages[MESA_SHADER_TASK].stage != MESA_SHADER_NONE || ms->key.has_task_shader || ms->bc250_task_replay)
      return RADV_BC250_ROUTE_REASON_TASK_STAGE;
   if (gfx_state->has_multiview_view_index)
      return RADV_BC250_ROUTE_REASON_MULTIVIEW;
   if (gfx_state->vrs_may_be_enabled)
      return RADV_BC250_ROUTE_REASON_VRS;
   if (!fs)
      return RADV_BC250_ROUTE_REASON_NO_FRAGMENT;
   if (compiler_info->key.bc250_mesh_safe_bary && radv_bc250_mesh_fs_refused(compiler_info, fs)) {
      int raw_slot, flat_slot;
      if (compiler_info->hw.bc250_bary_no_ref || !mesh ||
          !radv_bc250_bary_ref_slots(mesh, fs, &raw_slot, &flat_slot))
         return RADV_BC250_ROUTE_REASON_BARY_REFERENCE;
      if (mesh->info.mesh.primitive_type == MESA_PRIM_TRIANGLES && mesh->info.mesh.max_primitives_out < 24)
         return RADV_BC250_ROUTE_REASON_BARY_COST;
      return RADV_BC250_ROUTE_REASON_BARY_PRIVATE;
   }
   if (radv_bc250_merge_fs_refusal(fs))
      return RADV_BC250_ROUTE_REASON_FRAGMENT_ABI;
   if (mesh && mesh->info.shared_size > 16 * 1024)
      return RADV_BC250_ROUTE_REASON_SHARED_MEMORY;
   if (ms->bc250_pp_locations || (mesh && radv_bc250_mesh_needs_expansion(mesh)))
      return RADV_BC250_ROUTE_REASON_PER_PRIMITIVE_OUTPUT;
   if (mesh && (mesh->info.outputs_written & RADV_BC250_CLIPCULL_OUTPUTS))
      return RADV_BC250_ROUTE_REASON_CLIPCULL_EXPORT;
   if (!mesh || mesh->info.mesh.primitive_type != MESA_PRIM_TRIANGLES)
      return RADV_BC250_ROUTE_REASON_TOPOLOGY;
   if (!mesh->info.mesh.max_vertices_out || mesh->info.mesh.max_vertices_out > 256 ||
       !mesh->info.mesh.max_primitives_out || mesh->info.mesh.max_primitives_out > 256)
      return RADV_BC250_ROUTE_REASON_API_BOUND;
   if (mesh->info.workgroup_size_variable ||
       mesh->info.workgroup_size[0] * mesh->info.workgroup_size[1] * mesh->info.workgroup_size[2] > 256)
      return RADV_BC250_ROUTE_REASON_WORKGROUP_BOUND;
   const unsigned v = mesh->info.mesh.max_vertices_out;
   const unsigned p = mesh->info.mesh.max_primitives_out;
   const unsigned bound = v <= 32 ? MIN2(v, 3 * p) : 3 * p;
   if (bound > 256)
      return RADV_BC250_ROUTE_REASON_LAUNCH_BOUND;
   if (mesh->info.outputs_read || mesh->info.outputs_written_16bit)
      return RADV_BC250_ROUTE_REASON_MESH_IO;
   if (mesh->info.shared_size > 16 * 1024)
      return RADV_BC250_ROUTE_REASON_SHARED_MEMORY;
   const uint64_t special = VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_PRIMITIVE_COUNT;
   const uint64_t primitive_id_vertex =
      (mesh->info.outputs_written & VARYING_BIT_PRIMITIVE_ID) &&
      !(mesh->info.per_primitive_outputs & VARYING_BIT_PRIMITIVE_ID) ? VARYING_BIT_PRIMITIVE_ID : 0;
   const uint64_t vertex = mesh->info.outputs_written & ~(special | primitive_id_vertex);
   const uint64_t allowed = VARYING_BIT_POS | primitive_id_vertex | (UINT64_C(0xffffffff) << VARYING_SLOT_VAR0);
   if (!(vertex & VARYING_BIT_POS) || (vertex & ~allowed))
      return RADV_BC250_ROUTE_REASON_MESH_IO;
   const unsigned max_slots = compiler_info->key.bc250_mesh_safe_fast ||
                              compiler_info->key.bc250_mesh_safe_pieces ? 16 : 8;
   if (util_bitcount64(vertex) > max_slots)
      return RADV_BC250_ROUTE_REASON_VARYING_LAYOUT;
   nir_foreach_variable_with_modes(var, mesh, nir_var_shader_out) {
      if (var->data.location < VARYING_SLOT_VAR0)
         continue;
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!(glsl_type_is_scalar(type) || glsl_type_is_vector(type) || glsl_type_is_matrix(type)) ||
          glsl_get_bit_size(type) != 32)
         return RADV_BC250_ROUTE_REASON_VARYING_LAYOUT;
   }
   const bool parallel = p <= 85;
   const unsigned lds_bound = align(mesh->info.shared_size, 16) + 128 +
      16 * util_bitcount64(vertex) * v + 6 * p +
      (parallel ? MAX2(4 * v, align(3 * p, 4) + 32) : 4 * v) + bound;
   if (lds_bound >= 30 * 1024)
      return RADV_BC250_ROUTE_REASON_SHARED_MEMORY;
   if (!radv_bc250_mesh_safe_direct_candidate_slots(mesh, true, max_slots))
      return RADV_BC250_ROUTE_REASON_VARYING_LAYOUT;
   return RADV_BC250_ROUTE_REASON_CANDIDATE_UNPROVEN;
}

VkResult
radv_graphics_shaders_compile(const struct radv_compiler_info *compiler_info, struct vk_pipeline_cache *cache,
                              struct radv_shader_stage *stages, const struct radv_graphics_state_key *gfx_state,
                              bool is_internal, struct radv_retained_shaders *retained_shaders, bool noop_fs,
                              struct radv_shader_debug_info *debug, struct radv_shader_binary **binaries,
                              struct radv_shader_debug_info *gs_copy_debug, struct radv_shader_binary **gs_copy_binary)
{
   for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
      if (stages[s].stage == MESA_SHADER_NONE)
         continue;

      int64_t stage_start = os_time_get_nano();

      /* NIR might already have been imported from a library. */
      if (!stages[s].nir) {
         struct radv_spirv_to_nir_options options = {
            .lower_view_index_to_zero = !gfx_state->has_multiview_view_index,
            .lower_view_index_to_device_index = stages[s].key.view_index_from_device_index,
         };
         stages[s].nir = radv_shader_spirv_to_nir_cached(compiler_info, cache, &stages[s],
                                                        &options, is_internal);
      }

      stages[s].feedback.duration += os_time_get_nano() - stage_start;
   }

   /* RADV_BC250_MESH_CLIPCULL_CONST (radv_bc250_mesh_clip_cull_const): constant non-negative
    * clip/cull distances are removed before any BC250 rewrite and route decision (merge, AMD
    * route, split, expansion, autocull). radv_bc250_prepare_task already did this for the Mesh
    * shaders it rewrites; a second run finds nothing to remove. */
   /* RADV_BC250_MESH_REFERENCE with RADV_BC250_MESH_REFERENCE_CULLDIST=keep: a Mesh shader the raw
    * route admits keeps its clip/cull distances (exported as POS1), as in the reference driver. */
   const bool bc250_reference_keep_cd =
      compiler_info->hw.bc250_mesh_reference_keep_cd && stages[MESA_SHADER_MESH].nir &&
      !stages[MESA_SHADER_MESH].bc250_split_mesh &&
      radv_bc250_mesh_amd_route(true, false, false, stages[MESA_SHADER_MESH].nir,
                                stages[MESA_SHADER_TASK].stage != MESA_SHADER_NONE ||
                                   stages[MESA_SHADER_MESH].key.has_task_shader ||
                                   stages[MESA_SHADER_MESH].bc250_task_replay,
                                gfx_state->has_multiview_view_index, true);
   if (compiler_info->hw.bc250_mesh_cc_const && stages[MESA_SHADER_MESH].nir && !bc250_reference_keep_cd)
      radv_bc250_mesh_clip_cull_const(stages[MESA_SHADER_MESH].nir, stages[MESA_SHADER_FRAGMENT].nir,
                                      stages[MESA_SHADER_FRAGMENT].nir || noop_fs);

   /* gl_CullPrimitiveEXT lowers to the NGG null-primitive bit in the prim-exp
    * argument (ac_nir_lower_ngg_mesh -> ac_nir_pack_ngg_prim_exp_arg), not to
    * the quarantined per-primitive attribute export path, and the GFX10
    * fully-culled allocation workaround covers 100% culling. */

   /* RADV_BC250_MESH_MERGE: K small triangle Mesh workgroups become one raw
    * subgroup of K*P >= 65 triangles (radv_bc250_merge_mesh), before the route
    * decision. Mesh-only pipelines whose NIR is compiled here and not retained:
    * no Task stage or replay, no split, no multiview, no mesh queries, no
    * graphics pipeline library NIR (a retained merged NIR would be merged
    * again at link time). */
   /* Clip/cull distances left after RADV_BC250_MESH_CLIPCULL_CONST keep the expanded route, which
    * culls cull distances in the shader (the merged raw shape would export them as POS1). */
   struct radv_shader_stage *owned_ms = &stages[MESA_SHADER_MESH];
   /* A fixed private-index proof depends only on the Mesh output. It is also
    * valid for a depth-only pipeline and after the existing Task transport has
    * lowered its payload; neither needs a new transport or export planner. */
   if (compiler_info->key.bc250_mesh_safe_bary_last && !retained_shaders &&
       !owned_ms->bc250_imported_nir && !owned_ms->bc250_split_mesh &&
       !owned_ms->bc250_ordered_export && !owned_ms->bc250_safe_fast && !owned_ms->bc250_safe_owned &&
       stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE &&
       !gfx_state->has_multiview_view_index && !gfx_state->vrs_may_be_enabled &&
       !compiler_info->key.bc250_mesh_amd_size && !compiler_info->key.bc250_mesh_min2waves)
      radv_bc250_prepare_bary_affine(compiler_info, owned_ms, stages[MESA_SHADER_FRAGMENT].nir);
   const bool private_bary = radv_bc250_mesh_private_bary(compiler_info, stages[MESA_SHADER_FRAGMENT].nir);
   /* The private expansion already references every vertex with a small backjump. For tiny classes
    * without shader culling the ordinary exporter needs no survivor map.
    * AUTOCULL_ALL retains the existing route until separately costed. */
   const bool tiny_bary = compiler_info->key.bc250_mesh_safe_bary &&
      compiler_info->key.bc250_mesh_safe_bary_tiny && compiler_info->key.bc250_mesh_safe_corners &&
      owned_ms->nir && !compiler_info->key.bc250_mesh_autocull_all &&
      ((owned_ms->nir->info.mesh.max_primitives_out <
         (compiler_info->key.bc250_mesh_safe_bary_last ? 86u : 24u)) ||
       owned_ms->nir->info.mesh.max_primitives_out > 85) &&
      (private_bary || (compiler_info->key.bc250_mesh_safe_bary_last &&
       (radv_bc250_mesh_has_per_primitive_data(owned_ms->nir) ||
        (owned_ms->nir->info.outputs_written & (RADV_BC250_CLIPCULL_OUTPUTS | VARYING_BIT_CULL_PRIMITIVE)))) ||
       owned_ms->nir->info.mesh.primitive_type != MESA_PRIM_TRIANGLES ||
       owned_ms->nir->info.mesh.max_primitives_out > 85 ||
       /* RADV_BC250_MESH_SAFE_PIECES_EXT: every split piece (Task pieces included) takes the
        * private-corner export, not only pieces with barycentrics or per-primitive data. */
       (compiler_info->bc250x.safe_pieces_ext && compiler_info->key.bc250_mesh_safe_bary_last &&
        owned_ms->bc250_split_mesh && debug_get_bool_option("RADV_BC250_MESH_SAFE_SPLIT_PIECES", false))) &&
      owned_ms->nir->info.mesh.max_vertices_out <= 256;
   /* RADV_BC250_MESH_SAFE_SPLIT_PIECES (default 0): every split piece (Task replay or the direct
    * Mesh-only split) is its own workgroup exporting its own slice, like the one-piece case, so it
    * takes the private-corner export too; a missing fragment shader (rasterizer discard) is allowed. */
   const bool safe_split_pieces = debug_get_bool_option("RADV_BC250_MESH_SAFE_SPLIT_PIECES", false) &&
      compiler_info->key.bc250_mesh_safe_bary_last && tiny_bary && owned_ms->bc250_split_mesh;
   const bool no_fs_ok = safe_split_pieces && !stages[MESA_SHADER_FRAGMENT].nir;
   /* RADV_BC250_MESH_SAFE_PIECES_EXT: an unsplit subgroup-free wave32 Mesh shader with 33..64 primitives
    * (above the wave32 owned capacity) tries the owned route in wave64; if it is not admitted it keeps
    * wave32 and its route. */
   bool owned_wave64 = false;
   if (compiler_info->bc250x.safe_pieces_ext && owned_ms->nir && !owned_ms->bc250_split_mesh &&
       !owned_ms->bc250_safe_fast && !owned_ms->bc250_safe_owned &&
       owned_ms->nir->info.mesh.max_primitives_out > 32 && owned_ms->nir->info.mesh.max_primitives_out <= 64 &&
       radv_bc250_mesh_wave64_promotable(owned_ms)) {
      radv_bc250_mesh_set_wave(owned_ms->nir, 64);
      owned_wave64 = true;
   }
   if (compiler_info->key.bc250_mesh_safe_owned && !retained_shaders &&
       (private_bary || no_fs_ok || !radv_bc250_mesh_fs_refused(compiler_info, stages[MESA_SHADER_FRAGMENT].nir)) &&
       (!owned_ms->bc250_split_mesh || safe_split_pieces ||
        (compiler_info->key.bc250_mesh_safe_bary_last && tiny_bary &&
         owned_ms->bc250_task_replay && owned_ms->bc250_split_pieces == 1)) &&
       !owned_ms->bc250_ordered_export &&
       !owned_ms->bc250_imported_nir && !owned_ms->bc250_safe_fast && !owned_ms->bc250_safe_owned &&
       !gfx_state->has_multiview_view_index && !gfx_state->vrs_may_be_enabled &&
       (private_bary || tiny_bary || (!gfx_state->dynamic_provoking_vtx_mode && !gfx_state->rs.provoking_vtx_last)) &&
       !compiler_info->key.bc250_mesh_amd_size && !compiler_info->key.bc250_mesh_min2waves) {
      for (unsigned attempt = 0; attempt < 2; attempt++) {
         /* A failed constant-count specialization must not displace the
          * existing safe owned route (HB2's counts can be dynamic). */
         if (attempt && !(compiler_info->key.bc250_mesh_safe_bary_last && tiny_bary &&
                          (private_bary || (!gfx_state->dynamic_provoking_vtx_mode &&
                                            !gfx_state->rs.provoking_vtx_last))))
            break;
         const bool use_tiny = tiny_bary && !attempt;
         radv_bc250_owned_lds_direct = compiler_info->bc250x.safe_pieces_ext && use_tiny &&
            owned_ms->bc250_split_mesh &&
            (compiler_info->key.bc250_mesh_direct_read & RADV_BC250_DIRECT_READ_ALL) == RADV_BC250_DIRECT_READ_ALL;
         const bool owned_ok = radv_bc250_mesh_safe_owned(&owned_ms->nir, &stages[MESA_SHADER_FRAGMENT].nir,
                                        compiler_info->key.bc250_mesh_safe_corners,
                                        private_bary, use_tiny, compiler_info->key.bc250_mesh_safe_bary_last,
                                        compiler_info->key.bc250_bary_io16, safe_split_pieces,
                                        compiler_info->key.mesh_shader_queries,
                                        compiler_info->key.bc250_mesh_direct_read & RADV_BC250_DIRECT_READ_ALL,
                                        owned_ms->bc250_index_staging, &owned_ms->bc250_pp_locations);
         radv_bc250_owned_lds_direct = false;
         if (!owned_ok)
            continue;
         owned_ms->bc250_safe_owned = true;
         owned_ms->bc250_safe_bary_tiny = use_tiny;
         if (use_tiny) {
            owned_ms->bc250_expanded = true;
            owned_ms->bc250_private_tris = true;
         }
         /* RADV_BC250_MESH_PP_SHARE needs the real provoking corner (0x5: dynamic, not shared). */
         owned_ms->bc250_compact_owned = !compiler_info->bc250x.pp_share ? 1 :
            gfx_state->dynamic_provoking_vtx_mode ? 0x5 : gfx_state->rs.provoking_vtx_last ? 0x4 : 0x1;
         if (getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 MESH SAFE OWNED: logical_V=%u P=%u bound=%u task_transport=%u\n",
                    owned_ms->nir->info.mesh.max_vertices_out, owned_ms->nir->info.mesh.max_primitives_out,
                    3 * owned_ms->nir->info.mesh.max_primitives_out, owned_ms->bc250_task_replay);
         break;
      }
   }
   if (owned_wave64) {
      if (!owned_ms->bc250_safe_owned)
         radv_bc250_mesh_set_wave(owned_ms->nir, 32);
      else if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE PIECES EXT: owned route in wave64\n");
   }

   const bool bc250_merge_clipcull = stages[MESA_SHADER_MESH].nir && !compiler_info->hw.bc250_mesh_allow_pos1 &&
      (stages[MESA_SHADER_MESH].nir->info.outputs_written & RADV_BC250_CLIPCULL_OUTPUTS);
   if (compiler_info->key.bc250_mesh_merge && bc250_merge_clipcull && getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH MERGE: not merged reason=clip/cull distance output\n");
   if (compiler_info->key.bc250_mesh_merge && !bc250_merge_clipcull && stages[MESA_SHADER_MESH].nir &&
       !stages[MESA_SHADER_MESH].bc250_split_mesh && !stages[MESA_SHADER_MESH].bc250_task_replay &&
       !stages[MESA_SHADER_MESH].bc250_imported_nir && !stages[MESA_SHADER_MESH].bc250_ordered_export &&
       !stages[MESA_SHADER_MESH].bc250_safe_fast && !stages[MESA_SHADER_MESH].bc250_safe_owned &&
       !retained_shaders &&
       stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE && !stages[MESA_SHADER_MESH].key.has_task_shader &&
       !gfx_state->has_multiview_view_index && !compiler_info->key.mesh_shader_queries) {
      /* Stage 2 (per-primitive outputs): the data rides on each primitive's
       * provoking vertex and the fragment shader reads it flat, so the
       * provoking mode must be static, the fragment shader must be linked
       * here and must not read explicit per-vertex inputs, and VRS stays
       * excluded (as for the base driver's compact vertex map). Outputs the fragment
       * shader never reads are dropped first, as the expansion does. */
      nir_shader *merge_fs = stages[MESA_SHADER_FRAGMENT].nir;
      const char *pp_refusal = NULL;
      if (radv_bc250_mesh_has_per_primitive_data(stages[MESA_SHADER_MESH].nir)) {
         if (merge_fs) {
            NIR_PASS(_, merge_fs, nir_remove_dead_variables, nir_var_shader_in, NULL);
            NIR_PASS(_, stages[MESA_SHADER_MESH].nir, nir_remove_unused_varyings, merge_fs);
            nir_shader_gather_info(stages[MESA_SHADER_MESH].nir,
                                   nir_shader_get_entrypoint(stages[MESA_SHADER_MESH].nir));
         }
         pp_refusal = radv_bc250_merge_fs_refusal(merge_fs);
         if (!pp_refusal && gfx_state->dynamic_provoking_vtx_mode)
            pp_refusal = "dynamic provoking vertex mode";
         if (!pp_refusal && gfx_state->vrs_may_be_enabled)
            pp_refusal = "VRS may be enabled";
      }
      struct radv_bc250_merge_plan plan;
      const bool merge = radv_bc250_mesh_merge_plan(stages[MESA_SHADER_MESH].nir, pp_refusal,
                                                    gfx_state->rs.provoking_vtx_last, &plan);
      if (getenv("BC250_TRACE_COMPILE")) {
         char stage2[96] = "";
         if (plan.pp)
            snprintf(stage2, sizeof(stage2), " stage2=1 wave=%u provoking=%s cull=%u physical_per_instance=%u",
                     plan.wave, plan.provoking ? (plan.provoking_last ? "last" : "first") : "none", plan.cull, plan.c);
         fprintf(stderr, "BC250 MESH MERGE: %s K=%u S=%u L=%u V=%u P=%u merged_V=%u merged_P=%u lanes=%u lds_bound=%u subgroup_ops=%u%s%s%s\n",
                 merge ? "merged" : "not merged", plan.k, plan.s, plan.l, plan.v, plan.p, plan.vm, plan.k * plan.p,
                 plan.lanes, plan.lds, plan.subgroup_ops, stage2, merge ? "" : " reason=", merge ? "" : plan.reason);
      }
      plan.dims = compiler_info->key.bc250_mesh_merge_prep;
      plan.address32_hi = compiler_info->hw.address32_hi;
      if (merge && radv_bc250_merge_mesh(stages[MESA_SHADER_MESH].nir, merge_fs, &plan)) {
         stages[MESA_SHADER_MESH].bc250_merge_k = plan.k;
         stages[MESA_SHADER_MESH].bc250_merge_s = plan.s;
         stages[MESA_SHADER_MESH].bc250_merge_dims = plan.dims;
      }
   }

   /* RADV_BC250_MESH_AMD: eligible Mesh shaders skip the split and the
    * expansion and run as one NGG subgroup of T lanes (radv_bc250.c). A Mesh
    * shader that radv_bc250_prepare_task already split or attached to a
    * Task replay is rewritten (e.g. the split consumes CullPrimitive) and
   * keeps the base driver's route. */
   struct radv_shader_stage *safe_ms = &stages[MESA_SHADER_MESH];
   safe_ms->bc250_safe_direct = safe_ms->bc250_safe_fast || safe_ms->bc250_safe_owned || (compiler_info->key.bc250_mesh_safe_direct &&
      !safe_ms->bc250_ordered_export && !safe_ms->bc250_split_mesh && !safe_ms->bc250_task_replay && safe_ms->bc250_merge_k <= 1 &&
      stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE && !safe_ms->key.has_task_shader &&
      !gfx_state->has_multiview_view_index && !gfx_state->vrs_may_be_enabled &&
      !compiler_info->key.bc250_mesh_amd_size && !compiler_info->key.bc250_mesh_min2waves &&
      radv_bc250_mesh_safe_direct_candidate(safe_ms->nir, compiler_info->key.bc250_mesh_safe_parallel) &&
      stages[MESA_SHADER_FRAGMENT].nir && !radv_bc250_mesh_fs_refused(compiler_info, stages[MESA_SHADER_FRAGMENT].nir));
   if (compiler_info->key.bc250_mesh_safe_direct && safe_ms->nir && getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH SAFE DIRECT: %s V=%u P=%u bound=%u\n",
              safe_ms->bc250_safe_direct ? "admitted" : "fallback",
              safe_ms->nir->info.mesh.max_vertices_out, safe_ms->nir->info.mesh.max_primitives_out,
              safe_ms->nir->info.mesh.max_vertices_out <= 32 ?
                 MIN2(safe_ms->nir->info.mesh.max_vertices_out, 3 * safe_ms->nir->info.mesh.max_primitives_out) :
                 3 * safe_ms->nir->info.mesh.max_primitives_out);

   stages[MESA_SHADER_MESH].bc250_amd_mesh = safe_ms->bc250_ordered_export || safe_ms->bc250_safe_direct || (
      !stages[MESA_SHADER_MESH].bc250_split_mesh &&
      radv_bc250_mesh_amd_route(compiler_info->key.bc250_mesh_amd, compiler_info->key.bc250_mesh_fast,
                                stages[MESA_SHADER_MESH].bc250_merge_k > 1, stages[MESA_SHADER_MESH].nir,
                                stages[MESA_SHADER_TASK].stage != MESA_SHADER_NONE ||
                                   stages[MESA_SHADER_MESH].key.has_task_shader ||
                                   stages[MESA_SHADER_MESH].bc250_task_replay,
                                gfx_state->has_multiview_view_index,
                                compiler_info->hw.bc250_mesh_allow_pos1 || compiler_info->hw.bc250_mesh_reference_keep_cd));

   if (compiler_info->key.bc250_expand_primitives && stages[MESA_SHADER_MESH].nir &&
       !stages[MESA_SHADER_MESH].bc250_amd_mesh) {
      nir_shader *bc250_mesh = stages[MESA_SHADER_MESH].nir;

      /* The split and the expansion below stage every declared Mesh output in
       * at most 16 KiB of LDS, and they run before the ordinary varying
       * linking. Drop the outputs the linked fragment shader never reads
       * first: an unread output has no observable effect, and without this
       * Hellblade 2's Nanite shader (256V/128P, a dead flat uvec3 output)
       * needs 18.9 KiB of staging and every such pipeline is refused. Outputs
       * the Mesh shader reads back itself and built-ins other than PrimitiveId
       * are kept (nir_remove_unused_io_vars).
       */
      nir_shader *bc250_fs = stages[MESA_SHADER_FRAGMENT].nir;
      if (bc250_fs) {
         NIR_PASS(_, bc250_fs, nir_remove_dead_variables, nir_var_shader_in, NULL);
         NIR_PASS(_, bc250_mesh, nir_remove_unused_varyings, bc250_fs);
      }
      if (compiler_info->key.bc250_split_mesh && !stages[MESA_SHADER_MESH].bc250_split_mesh &&
          ((bc250_mesh->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) ||
           (bc250_mesh->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
            (bc250_mesh->info.mesh.max_primitives_out > 85 || stages[MESA_SHADER_MESH].bc250_fit_min_pieces > 1 ||
             radv_bc250_mesh_culldist_split(compiler_info, bc250_mesh))))) {
         unsigned pieces = 0;
         if (!radv_bc250_split_mesh(bc250_mesh, NULL, stages[MESA_SHADER_FRAGMENT].nir, false, false, compiler_info->key.bc250_balanced_slices, compiler_info->key.bc250_parallel_cull, compiler_info->key.bc250_output_regions,
                                    compiler_info->key.bc250_mesh_compact_lds, compiler_info->key.bc250_piece_prims, stages[MESA_SHADER_MESH].bc250_fit_min_pieces, &pieces))
            return VK_ERROR_FEATURE_NOT_PRESENT;
         stages[MESA_SHADER_MESH].bc250_split_mesh = true;
         stages[MESA_SHADER_MESH].bc250_split_pieces = pieces;
      }
      bool bc250_over_budget = false;
      /* The per-primitive outputs the expansion turns into flat per-vertex outputs. */
      stages[MESA_SHADER_MESH].bc250_pp_locations =
         bc250_mesh->info.per_primitive_outputs & bc250_mesh->info.outputs_written &
         ((UINT64_C(0xffffffff) << VARYING_SLOT_VAR0) | VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_VIEWPORT |
          VARYING_BIT_LAYER);
      /* RADV_BC250_MESH_SPLIT_ANY: a Mesh shader without a fragment shader (rasterizer discard)
       * expands against an empty fragment shader: no varying is read, only positions export. */
      nir_shader *bc250_empty_fs = NULL;
      if (!stages[MESA_SHADER_FRAGMENT].nir && debug_get_bool_option("RADV_BC250_MESH_SPLIT_ANY", false))
         bc250_empty_fs = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, bc250_mesh->options,
                                                         "bc250_empty_fs").shader;
      bool bc250_expanded = radv_bc250_expand_primitive_attributes(bc250_mesh, bc250_empty_fs ? bc250_empty_fs : stages[MESA_SHADER_FRAGMENT].nir, compiler_info->key.bc250_compact_vertices &&
         stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE &&
         !gfx_state->has_multiview_view_index && !gfx_state->vrs_may_be_enabled &&
         !gfx_state->dynamic_provoking_vtx_mode && !gfx_state->rs.provoking_vtx_last,
         stages[MESA_SHADER_MESH].bc250_fit_reclaim || compiler_info->key.bc250_mesh_compact_lds,
         (compiler_info->key.bc250_mesh_direct_read & RADV_BC250_DIRECT_READ_ALL), &bc250_over_budget,
         &stages[MESA_SHADER_MESH].bc250_dead_shared, stages[MESA_SHADER_MESH].bc250_index_staging,
         &stages[MESA_SHADER_MESH].bc250_compact_map, compiler_info->key.bc250_mesh_compact);
      ralloc_free(bc250_empty_fs);
      stages[MESA_SHADER_MESH].bc250_expanded = bc250_expanded;
      /* RADV_BC250_MESH_IMPLICIT_TRIS: without the compact vertex map every primitive has
       * its own consecutive vertices (radv_bc250_expand_primitive_attributes). */
      stages[MESA_SHADER_MESH].bc250_private_tris = bc250_expanded && !stages[MESA_SHADER_MESH].bc250_compact_map;
      /* Per-primitive generic attributes are only representable through the
       * expansion epilogue. If it was required but not applied (e.g. the mesh
       * shared-memory + staging budget is exceeded), reject now instead of
       * leaving raw per-primitive output intrinsics that later NIR passes
       * assert on and crash in. */
      if (!bc250_expanded && (stages[MESA_SHADER_MESH].bc250_split_mesh ||
                              radv_bc250_mesh_needs_expansion(bc250_mesh))) {
         if (stages[MESA_SHADER_MESH].bc250_split_mesh)
            fprintf(stderr, "BC250 mesh split rejected: required vertex expansion unavailable\n");
         /* Output staging over the LDS budget: smaller pieces stage less. */
         stages[MESA_SHADER_MESH].bc250_lds_refused = bc250_over_budget;
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
   }

   /* RADV_BC250_MESH_FAIL_CLOSED: every protected route has been decided by now. A Mesh shader on
    * none of them would draw the API connectivity unchecked (the route the hang rules forbid):
    * refuse it. The refused-pipeline retry may still split it; otherwise pipeline creation fails. */
   if (compiler_info->hw.bc250_mesh_fail_closed && stages[MESA_SHADER_MESH].nir &&
       !radv_bc250_mesh_protected_route(stages[MESA_SHADER_MESH].bc250_safe_direct,
                                       stages[MESA_SHADER_MESH].bc250_ordered_export,
                                       stages[MESA_SHADER_MESH].bc250_split_mesh,
                                       stages[MESA_SHADER_MESH].bc250_expanded,
                                       stages[MESA_SHADER_MESH].bc250_merge_k,
                                       stages[MESA_SHADER_MESH].bc250_empty_output)) {
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH FAIL CLOSED: no protected route (V=%u P=%u prim=%u), refused\n",
                 stages[MESA_SHADER_MESH].nir->info.mesh.max_vertices_out,
                 stages[MESA_SHADER_MESH].nir->info.mesh.max_primitives_out,
                 stages[MESA_SHADER_MESH].nir->info.mesh.primitive_type);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }

   /* RADV_BC250_MESH_AMD size and reuse parts: they apply to AMD-route Mesh
    * shaders and, switched on alone, to the other native mesh-only shaders
    * that the base driver did not split (expanded or not), with T taken from the final
    * NIR. Task, replayed, split and multiview Mesh shaders are excluded.
    * With the route part on (RADV_BC250_MESH_AMD=1), only AMD-route Mesh
    * shaders get them, so every base-route pipeline stays unchanged. */
   if (stages[MESA_SHADER_MESH].nir) {
      const bool bc250_plain_mesh =
         stages[MESA_SHADER_MESH].bc250_amd_mesh ||
         (!compiler_info->key.bc250_mesh_amd && !stages[MESA_SHADER_MESH].bc250_split_mesh && !stages[MESA_SHADER_MESH].bc250_task_replay &&
          stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE && !stages[MESA_SHADER_MESH].key.has_task_shader &&
          !gfx_state->has_multiview_view_index && !stages[MESA_SHADER_MESH].nir->info.mesh.nv);
      const bool bc250_merged = stages[MESA_SHADER_MESH].bc250_merge_k > 1;
      /* Merged subgroups keep THDS_PER_SUBGRP = 0 and fast launch 0. */
      stages[MESA_SHADER_MESH].bc250_amd_size =
         bc250_plain_mesh && compiler_info->key.bc250_mesh_amd_size && !bc250_merged;
      stages[MESA_SHADER_MESH].bc250_amd_reuse = bc250_plain_mesh && compiler_info->key.bc250_mesh_amd_reuse;
      /* Fast launch 0 for every Mesh pipeline: plain, expanded, split pieces,
       * hybrid-Task replay consumers and multiview. Their NIR reads the
       * workgroup index (readfirstlane of lane 0's input), draw id, view index
       * and grid SGPRs only, never per-lane launch VGPRs, and their draws use
       * the same DRAW_INDEX_AUTO / DISPATCH_MESH_INDIRECT_MULTI packets. Only a
       * native hardware Task stage (not used by the base driver's policy) and the
       * experimental AMD size part keep fast launch 1 (see radv_device.c). */
      stages[MESA_SHADER_MESH].bc250_fl0 =
         (compiler_info->key.bc250_mesh_fl0 || stages[MESA_SHADER_MESH].bc250_amd_mesh || bc250_merged) &&
         !stages[MESA_SHADER_MESH].bc250_amd_size &&
         stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE && !stages[MESA_SHADER_MESH].key.has_task_shader &&
         !stages[MESA_SHADER_MESH].nir->info.mesh.nv;

      /* RADV_BC250_MESH_AUTOCULL (radv_bc250_mesh_autocull_candidate). RADV_BC250_MESH_CULLDIST_CULL:
       * a Mesh shader that writes cull distances (constant non-negative ones are already gone) is
       * a candidate also without RADV_BC250_MESH_AUTOCULL: the shader culls with them and does
       * not export them, as VS NGG culling (ac_nir_lower_ngg_mesh bc250_autocull_culldist). */
      const bool bc250_culldist = compiler_info->hw.bc250_mesh_culldist_cull &&
                                  stages[MESA_SHADER_MESH].nir->info.cull_distance_array_size &&
                                  (stages[MESA_SHADER_MESH].nir->info.outputs_written & RADV_BC250_CLIPCULL_OUTPUTS);
      if (compiler_info->key.bc250_mesh_autocull || bc250_culldist) {
         const char *reason;
         stages[MESA_SHADER_MESH].bc250_autocull = radv_bc250_mesh_autocull_candidate(
            stages[MESA_SHADER_MESH].nir, stages[MESA_SHADER_MESH].bc250_expanded, bc250_merged,
            stages[MESA_SHADER_MESH].bc250_amd_mesh &&
               !(compiler_info->key.bc250_mesh_safe_bary_last &&
                 stages[MESA_SHADER_MESH].bc250_safe_bary_tiny), gfx_state, &reason);
         stages[MESA_SHADER_MESH].bc250_autocull_culldist = stages[MESA_SHADER_MESH].bc250_autocull && bc250_culldist;
         if (getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 MESH AUTOCULL candidate: %s V=%u P=%u split=%u replay=%u%s%s%s\n",
                    stages[MESA_SHADER_MESH].bc250_autocull ? "yes" : "no",
                    stages[MESA_SHADER_MESH].nir->info.mesh.max_vertices_out,
                    stages[MESA_SHADER_MESH].nir->info.mesh.max_primitives_out,
                    stages[MESA_SHADER_MESH].bc250_split_mesh, stages[MESA_SHADER_MESH].bc250_task_replay,
                    bc250_culldist ? " culldist=1" : "", reason ? " reason=" : "", reason ? reason : "");
      }
   }

   /* RADV_BC250_MESH_COMPACT (ac_nir_lower_ngg_mesh.c, ms_compact_vertices): the base driver's
    * plain expansion (3 private vertices per triangle, not the compact vertex map), no
    * multiview. The per-primitive payload the expansion copies to every corner stays on
    * a vertex owned by its primitive: the provoking corner (0 first, 2 last), both for a
    * dynamic provoking mode. The lowering applies its own checks (trace line
    * BC250 MESH COMPACT). */
   if (compiler_info->key.bc250_mesh_compact && stages[MESA_SHADER_MESH].nir) {
      struct radv_shader_stage *ms = &stages[MESA_SHADER_MESH];
      ms->bc250_compact = ms->bc250_expanded && !ms->bc250_compact_map && !ms->bc250_amd_mesh &&
                          ms->bc250_merge_k <= 1 && !gfx_state->has_multiview_view_index &&
                          ms->nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES;
      ms->bc250_compact_owned = !ms->bc250_pp_locations ? 0 :
                                gfx_state->dynamic_provoking_vtx_mode ? 0x5 :
                                gfx_state->rs.provoking_vtx_last ? 0x4 : 0x1;
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH COMPACT candidate: %s V=%u P=%u expanded=%u compact_map=%u pp_locations=0x%" PRIx64
                 " owned=0x%x provoking=%s\n", ms->bc250_compact ? "yes" : "no",
                 ms->nir->info.mesh.max_vertices_out, ms->nir->info.mesh.max_primitives_out, ms->bc250_expanded,
                 ms->bc250_compact_map, ms->bc250_pp_locations, ms->bc250_compact_owned,
                 gfx_state->dynamic_provoking_vtx_mode ? "dynamic" : gfx_state->rs.provoking_vtx_last ? "last" : "first");
   }

   if (retained_shaders) {
      radv_pipeline_retain_shaders(retained_shaders, stages);
   }

   VkShaderStageFlagBits active_nir_stages = 0;
   for (int i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      if (stages[i].nir)
         active_nir_stages |= mesa_to_vk_shader_stage(i);
   }

   if (compiler_info->ac->gfx_level < GFX11 && stages[MESA_SHADER_MESH].nir &&
       BITSET_TEST(stages[MESA_SHADER_MESH].nir->info.system_values_read, SYSTEM_VALUE_WORKGROUP_ID)) {
      nir_shader *mesh = stages[MESA_SHADER_MESH].nir;
      nir_shader *task = stages[MESA_SHADER_TASK].nir;

      /* Mesh shaders only have a 1D "vertex index" which we use
       * as "workgroup index" to emulate the 3D workgroup ID.
       */
      nir_lower_compute_system_values_options o = {
         .lower_workgroup_id_to_index = true,
         .shortcut_1d_workgroup_id = true,
         .num_workgroups[0] = task ? task->info.mesh.ts_mesh_dispatch_dimensions[0] : 0,
         .num_workgroups[1] = task ? task->info.mesh.ts_mesh_dispatch_dimensions[1] : 0,
         .num_workgroups[2] = task ? task->info.mesh.ts_mesh_dispatch_dimensions[2] : 0,
      };

      NIR_PASS(_, mesh, nir_lower_compute_system_values, &o);
   }

   radv_foreach_stage (i, active_nir_stages) {
      mesa_shader_stage next_stage;

      if (stages[i].next_stage != MESA_SHADER_NONE) {
         next_stage = stages[i].next_stage;
      } else {
         next_stage = radv_get_next_stage(i, active_nir_stages);
      }

      radv_nir_shader_info_init(i, next_stage, &stages[i].info);
   }
   if (stages[MESA_SHADER_MESH].nir) {
      stages[MESA_SHADER_MESH].info.ms.bc250_amd_mesh = stages[MESA_SHADER_MESH].bc250_amd_mesh;
      stages[MESA_SHADER_MESH].info.ms.bc250_safe_fast = stages[MESA_SHADER_MESH].bc250_safe_fast;
      stages[MESA_SHADER_MESH].info.ms.bc250_pp_direct = stages[MESA_SHADER_MESH].bc250_pp_direct;
      stages[MESA_SHADER_MESH].bc250_safe_stats =
         (compiler_info->key.bc250_mesh_direct_read & RADV_BC250_MESH_SAFE_STATS_KEY) &&
         stages[MESA_SHADER_MESH].bc250_safe_fast;
      stages[MESA_SHADER_MESH].info.ms.bc250_safe_stats = stages[MESA_SHADER_MESH].bc250_safe_stats;
      stages[MESA_SHADER_MESH].info.ms.bc250_safe_direct = stages[MESA_SHADER_MESH].bc250_safe_direct;
      stages[MESA_SHADER_MESH].info.ms.bc250_route_reason = radv_bc250_mesh_no_split_reason(compiler_info, stages, gfx_state);
      stages[MESA_SHADER_MESH].info.ms.bc250_safe_autocull = stages[MESA_SHADER_MESH].bc250_safe_direct &&
         (compiler_info->key.bc250_mesh_safe_parallel ||
          (stages[MESA_SHADER_MESH].bc250_safe_fast &&
           ((compiler_info->key.bc250_mesh_direct_read & RADV_BC250_MESH_SAFE_LOCAL_KEY) ||
            compiler_info->key.bc250_mesh_safe_corners))) &&
         compiler_info->key.bc250_mesh_safe_autocull &&
         stages[MESA_SHADER_MESH].nir->info.mesh.max_primitives_out <= 85 &&
         !gfx_state->rs.polygon_mode_unknown && gfx_state->rs.polygon_mode == VK_POLYGON_MODE_FILL;
      stages[MESA_SHADER_MESH].info.ms.bc250_amd_size = stages[MESA_SHADER_MESH].bc250_amd_size;
      stages[MESA_SHADER_MESH].info.ms.bc250_amd_reuse = stages[MESA_SHADER_MESH].bc250_amd_reuse;
      stages[MESA_SHADER_MESH].info.ms.bc250_fl0 = stages[MESA_SHADER_MESH].bc250_fl0;
      stages[MESA_SHADER_MESH].info.ms.bc250_autocull = stages[MESA_SHADER_MESH].bc250_autocull;
      stages[MESA_SHADER_MESH].info.ms.bc250_autocull_culldist = stages[MESA_SHADER_MESH].bc250_autocull_culldist;
      stages[MESA_SHADER_MESH].info.ms.bc250_compact = stages[MESA_SHADER_MESH].bc250_compact;
      stages[MESA_SHADER_MESH].info.ms.bc250_expanded = stages[MESA_SHADER_MESH].bc250_expanded;
      stages[MESA_SHADER_MESH].info.ms.bc250_merge_k = stages[MESA_SHADER_MESH].bc250_merge_k;
      stages[MESA_SHADER_MESH].info.ms.bc250_merge_s = stages[MESA_SHADER_MESH].bc250_merge_s;
      stages[MESA_SHADER_MESH].info.ms.bc250_merge_dims = stages[MESA_SHADER_MESH].bc250_merge_dims;
      stages[MESA_SHADER_MESH].info.ms.bc250_alloc_barrier = compiler_info->key.bc250_mesh_alloc_barrier;
      stages[MESA_SHADER_MESH].info.ms.bc250_min2waves =
         compiler_info->key.bc250_mesh_min2waves && stages[MESA_SHADER_MESH].bc250_amd_mesh;
   }

   /* Determine if shaders uses NGG before linking because it's needed for some NIR pass. */
   radv_fill_shader_info_ngg(compiler_info, stages, active_nir_stages);

   if (stages[MESA_SHADER_GEOMETRY].nir) {
      unsigned nir_gs_flags = nir_lower_gs_intrinsics_per_stream;

      if (stages[MESA_SHADER_GEOMETRY].info.is_ngg) {
         nir_gs_flags |= nir_lower_gs_intrinsics_count_primitives |
                         nir_lower_gs_intrinsics_count_vertices_per_primitive |
                         nir_lower_gs_intrinsics_overwrite_incomplete;
      }

      NIR_PASS(_, stages[MESA_SHADER_GEOMETRY].nir, nir_lower_gs_intrinsics, nir_gs_flags);
      NIR_PASS(_, stages[MESA_SHADER_GEOMETRY].nir, nir_lower_vars_to_ssa);
   }

   if (stages[MESA_SHADER_TESS_CTRL].nir && stages[MESA_SHADER_TESS_EVAL].nir) {
      /* Copy TCS info into the TES info */
      merge_tess_info(&stages[MESA_SHADER_TESS_EVAL].nir->info, &stages[MESA_SHADER_TESS_CTRL].nir->info);
   }

   if (stages[MESA_SHADER_FRAGMENT].nir) {
      unsigned num_raster_vertices_per_prim = radv_get_num_raster_vertices_per_prim(stages, gfx_state);

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_lower_fs_barycentric, gfx_state,
               num_raster_vertices_per_prim, compiler_info->ac->gfx_level >= GFX10_3);

      /* frag_depth = gl_FragCoord.z broadcasts to all samples of the fragment shader invocation,
       * so only optimize it away if we know there is only one sample per invocation.
       * Because we don't know if sample shading is used with factor 1.0f, this means
       * we only optimize single sampled shaders.
       */
      if ((gfx_state->lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT) &&
          !gfx_state->dynamic_rasterization_samples && gfx_state->ms.rasterization_samples == 0)
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_fragdepth);

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_opt_fs_builtins, gfx_state, num_raster_vertices_per_prim);
   }

   if (stages[MESA_SHADER_VERTEX].nir && !gfx_state->vs.has_prolog)
      NIR_PASS(_, stages[MESA_SHADER_VERTEX].nir, radv_nir_optimize_vs_inputs_to_const, gfx_state);

   radv_foreach_stage (i, active_nir_stages) {
      int64_t stage_start = os_time_get_nano();

      if (i == MESA_SHADER_FRAGMENT && stages[MESA_SHADER_MESH].nir) {
         nir_foreach_shader_in_variable (var, stages[i].nir) {
            /* These variables are implicitly per-primitive when used with mesh->fragment stages
             * and this can't be determined with only the FS.
             * nir_opt_varyings relies on inputs and outputs agreeing on per-primitive.
             */
            if (var->data.location == VARYING_SLOT_PRIMITIVE_ID || var->data.location == VARYING_SLOT_VIEWPORT ||
                var->data.location == VARYING_SLOT_LAYER) {
               const nir_shader *mesh = stages[MESA_SHADER_MESH].nir;
               const uint64_t bit = BITFIELD64_BIT(var->data.location);
               /* BC250 expansion carries these builtins on duplicated vertices. */
               bool expanded_builtin = compiler_info->key.bc250_expand_primitives &&
                                       (mesh->info.outputs_written & bit) &&
                                       !(mesh->info.per_primitive_outputs & bit);
               var->data.per_primitive = !expanded_builtin;
               if (expanded_builtin)
                  var->data.interpolation = INTERP_MODE_FLAT;
            }
         }
      }

      radv_nir_lower_io(stages[i].nir);

      if (!stages[i].key.optimisations_disabled) {
         /* Scalarize all I/O, because nir_opt_varyings and nir_opt_vectorize_io expect all I/O to be scalarized. */
         NIR_PASS(_, stages[i].nir, nir_lower_io_to_scalar, nir_var_shader_in | nir_var_shader_out, NULL, NULL);

         /* Eliminate useless vec->mov copies resulting from scalarization. */
         NIR_PASS(_, stages[i].nir, nir_opt_copy_prop);
         NIR_PASS(_, stages[i].nir, nir_opt_constant_folding);
      }

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   if (stages[MESA_SHADER_FRAGMENT].nir) {
      if (gfx_state->dynamic_line_rast_mode)
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_lower_poly_line_smooth, RADV_NUM_SMOOTH_AA_SAMPLES);

      if (!gfx_state->ps.has_epilog) {
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_remap_color_attachment, gfx_state);

         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_trim_fs_color_exports, &gfx_state->ps.epilog,
                  gfx_state->ps.mrt0_alpha_is_dead);

         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_copy_prop);
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_dce);
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_dead_cf);
      }

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_lower_fs_input_attachment);

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_cse);
      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_copy_prop);
      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_dce);

      const bool vrs_may_be_enabled =
         gfx_state->vrs_may_be_enabled && !stages[MESA_SHADER_FRAGMENT].nir->info.fs.sample_mask_in_declared;
      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_lower_opt_fs_frag_pos, vrs_may_be_enabled,
               gfx_state->ms.sample_shading_enable || stages[MESA_SHADER_FRAGMENT].nir->info.fs.uses_sample_shading);

      ac_nir_lower_sample_mask_in_options lower_sample_mask_in_options = {0};

      if (stages[MESA_SHADER_FRAGMENT].nir->info.fs.uses_sample_shading || gfx_state->ms.max_sample_shading_enable) {
         lower_sample_mask_in_options.behavior = ac_nir_lower_samplemask_sample_shading_max;
      } else if (gfx_state->ms.sample_shading_enable) {
         lower_sample_mask_in_options.behavior = ac_nir_lower_samplemask_sample_shading_partial;
         lower_sample_mask_in_options.ps_iter_samples = gfx_state->ms.ps_iter_samples;
      } else if (!gfx_state->dynamic_rasterization_samples && gfx_state->ms.rasterization_samples == 0) {
         lower_sample_mask_in_options.behavior = ac_nir_lower_samplemask_1sample_no_vrs;
      } else {
         lower_sample_mask_in_options.behavior = ac_nir_lower_samplemask_unknown_states_no_sample_shading;
      }

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, ac_nir_lower_sample_mask_in, &lower_sample_mask_in_options);
   }

   radv_foreach_stage (i, active_nir_stages) {
      if (!radv_is_last_vgt_stage(&stages[i]))
         continue;

      /* Lower mesh shader draw ID to zero prevent app bugs from triggering undefined behaviour. */
      if (i == MESA_SHADER_MESH && stages[i].info.ms.has_task &&
          BITSET_TEST(stages[i].nir->info.system_values_read, SYSTEM_VALUE_DRAW_ID))
         radv_nir_lower_draw_id_to_zero(stages[i].nir);

      if (i != MESA_SHADER_MESH && radv_should_export_multiview(&stages[i], gfx_state))
         NIR_PASS(_, stages[i].nir, radv_nir_export_multiview);

      uint64_t remove_as_varying = 0;
      uint64_t remove_as_sysval = 0;

      /* Remove all varyings when the fragment shader is a noop. */
      if (noop_fs)
         remove_as_varying |= ~0ull;

      /* Remove PSIZ from shaders when it's not needed.
       * This is typically produced by translation layers like Zink or d3d9 DXVK.
       */
      if (gfx_state->enable_remove_point_size && (i != MESA_SHADER_TESS_EVAL || !stages[i].nir->info.tess.point_mode) &&
          (i != MESA_SHADER_GEOMETRY || stages[i].nir->info.gs.output_primitive != MESA_PRIM_POINTS) &&
          (i != MESA_SHADER_MESH || stages[i].nir->info.mesh.primitive_type != MESA_PRIM_POINTS)) {
         remove_as_varying |= VARYING_BIT_PSIZ;
         remove_as_sysval |= VARYING_BIT_PSIZ;
      }

      if (!remove_as_varying && !remove_as_sysval)
         continue;

      NIR_PASS(_, stages[i].nir, nir_remove_outputs, MESA_SHADER_FRAGMENT, remove_as_varying, remove_as_sysval);
      break;
   }

   radv_foreach_stage (i, active_nir_stages) {
      int64_t stage_start = os_time_get_nano();

      ac_nir_lower_indirect_derefs_early(stages[i].nir);
      NIR_PASS(_, stages[i].nir, nir_lower_vars_to_ssa);

      radv_optimize_nir(stages[i].nir, stages[i].key.optimisations_disabled);

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   /* Optimize varyings on lowered shader I/O (more efficient than optimizing I/O derefs). */
   radv_graphics_shaders_link_varyings(stages, compiler_info->ac->gfx_level);
   if (stages[MESA_SHADER_MESH].nir && stages[MESA_SHADER_MESH].bc250_pp_direct && stages[MESA_SHADER_FRAGMENT].nir)
      radv_bc250_pp_direct_fs_inputs(stages[MESA_SHADER_FRAGMENT].nir);

   VkResult bary_result = radv_bc250_link_bary_rotation(compiler_info, stages, gfx_state);
   if (bary_result != VK_SUCCESS)
      return bary_result;

   /* Optimize constant clip/cull distance after linking to operate on scalar io in the last
    * pre raster stage.
    */
   radv_foreach_stage (
      i, active_nir_stages &
            (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT)) {
      if (stages[i].key.optimisations_disabled)
         continue;

      int64_t stage_start = os_time_get_nano();

      NIR_PASS(_, stages[i].nir, nir_opt_clip_cull_const);

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   radv_foreach_stage (i, active_nir_stages) {
      int64_t stage_start = os_time_get_nano();

      /* Indirect lowering must be called after the radv_optimize_nir() loop
       * has been called at least once. Otherwise indirect lowering can
       * bloat the instruction count of the loop and cause it to be
       * considered too large for unrolling.
       *
       * We want to do this as late as possible because scratch access isn't
       * very optimizable. We lower smaller arrays to SSA earlier with
       * ac_nir_lower_indirect_derefs_early, because that can actually enable
       * optimizations.
       */
      bool indirect_derefs_lowered = false;
      NIR_PASS(indirect_derefs_lowered, stages[i].nir, ac_nir_lower_indirect_derefs);
      NIR_PASS(_, stages[i].nir, nir_lower_vars_to_ssa);
      if (indirect_derefs_lowered && !stages[i].key.optimisations_disabled)
         radv_optimize_nir(stages[i].nir, false);

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   radv_fill_shader_info(compiler_info, RADV_PIPELINE_GRAPHICS, gfx_state, stages, active_nir_stages);

   /* Remove the primitive shading rate output if VRS flat shading overrides it. */
   radv_foreach_stage (i, active_nir_stages) {
      if (!radv_is_last_vgt_stage(&stages[i]))
         continue;

      struct radv_shader_stage *fs_stage = &stages[MESA_SHADER_FRAGMENT];

      /* RADV_BC250_VRS_NOOP strips the output right after spirv_to_nir (radv_nir_bc250_vrs_noop);
       * this is only a backstop for NIR that did not come from there. */
      if (fs_stage &&
          (fs_stage->info.ps.allow_flat_shading || fs_stage->info.ps.force_disable_vrs || compiler_info->hw.bc250_vrs_noop)) {
         stages[i].info.force_vrs_per_vertex = false;

         if (stages[i].info.outinfo.writes_primitive_shading_rate ||
             stages[i].info.outinfo.writes_primitive_shading_rate_per_primitive) {
            NIR_PASS(_, stages[i].nir, nir_remove_outputs, MESA_SHADER_FRAGMENT, 0, VARYING_BIT_PRIMITIVE_SHADING_RATE);

            stages[i].info.outinfo.writes_primitive_shading_rate = false;
            stages[i].info.outinfo.writes_primitive_shading_rate_per_primitive = false;
            stages[i].nir->info.outputs_written &= ~VARYING_BIT_PRIMITIVE_SHADING_RATE;
            stages[i].nir->info.per_primitive_outputs &= ~VARYING_BIT_PRIMITIVE_SHADING_RATE;
         }
      } else if (fs_stage && fs_stage->info.ps.disallow_force_vrs_per_vertex) {
         stages[i].info.force_vrs_per_vertex = false;
      }
      break;
   }

   radv_declare_pipeline_args(compiler_info, stages, gfx_state, active_nir_stages, debug);

   radv_foreach_stage (i, active_nir_stages) {
      int64_t stage_start = os_time_get_nano();

      radv_postprocess_nir(compiler_info, gfx_state, &stages[i]);

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   /* BC250: expanded Mesh shaders may stage up to 28 KiB of outputs in LDS
    * (radv_bc250_expand_primitive_attributes). Fail closed if the NGG lowering
    * then had to move output storage to the Mesh scratch ring, which has no
    * hardware validation on GFX1013.
    */
   if (stages[MESA_SHADER_MESH].nir &&
       (stages[MESA_SHADER_MESH].bc250_expanded || stages[MESA_SHADER_MESH].bc250_merge_k > 1) &&
       stages[MESA_SHADER_MESH].info.ms.needs_ms_scratch_ring) {
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 expanded Mesh rejected: output storage would spill to the Mesh scratch ring\n");
      stages[MESA_SHADER_MESH].bc250_lds_refused = stages[MESA_SHADER_MESH].bc250_expanded;
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }

   /* BC250 Mesh position exports (radv_bc250_mesh_pos_export_refusal): the lowered shader must
    * export exactly the position vectors SPI_SHADER_POS_FORMAT declares, and clip/cull distance
    * vectors (POS1/POS2) are refused until the hardware gates pass (RADV_BC250_MESH_ALLOW_POS1). */
   if (stages[MESA_SHADER_MESH].nir && compiler_info->key.bc250_native_mesh) {
      const char *refusal = radv_bc250_mesh_pos_export_refusal(compiler_info, stages[MESA_SHADER_MESH].nir,
                                                              &stages[MESA_SHADER_MESH].info);
      if (refusal) {
         fprintf(stderr, "BC250 Mesh pipeline refused: %s\n", refusal);
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
   }

   if (stages[MESA_SHADER_VERTEX].nir || stages[MESA_SHADER_TESS_EVAL].nir) {
      struct radv_shader_stage *es_stage =
         stages[MESA_SHADER_TESS_EVAL].nir ? &stages[MESA_SHADER_TESS_EVAL] : &stages[MESA_SHADER_VERTEX];
      struct radv_shader_stage *gs_stage = stages[MESA_SHADER_GEOMETRY].nir ? &stages[MESA_SHADER_GEOMETRY] : NULL;
      struct radv_shader_stage *stage = gs_stage ? gs_stage : es_stage;

      if ((gs_stage ? gs_stage : es_stage)->info.is_ngg) {
         gfx10_get_ngg_info(compiler_info, &es_stage->info, gs_stage ? &gs_stage->info : NULL, &stage->info.ngg_info);
         stage->info.nir_shared_size = stage->info.ngg_info.lds_size;
      }
   }

   if (stages[MESA_SHADER_GEOMETRY].nir && !stages[MESA_SHADER_GEOMETRY].info.is_ngg)
      radv_get_legacy_gs_info(compiler_info, NULL, &stages[MESA_SHADER_GEOMETRY].info);

   /* Compile NIR shaders to AMD assembly. */
   radv_graphics_shaders_nir_to_asm(compiler_info, cache, stages, gfx_state, active_nir_stages, debug, binaries,
                                    gs_copy_debug, gs_copy_binary);
   return VK_SUCCESS;
}

void
radv_graphics_shaders_create(struct radv_device *device, struct vk_pipeline_cache *cache, bool skip_shaders_cache,
                             struct radv_shader **shaders, struct radv_shader_binary **binaries,
                             struct radv_shader_debug_info *debug, struct radv_shader **gs_copy_shader,
                             struct radv_shader_binary *gs_copy_binary, struct radv_shader_debug_info *gs_copy_debug)
{
   for (int i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      struct radv_shader_binary *binary = binaries[i];
      if (binary)
         shaders[i] = radv_shader_create(device, cache, binary, skip_shaders_cache, &debug[i]);
   }
   if (gs_copy_binary)
      *gs_copy_shader = radv_shader_create(device, cache, gs_copy_binary, skip_shaders_cache, gs_copy_debug);
}

static bool
radv_should_compute_pipeline_hash(const struct radv_device *device, const enum radv_pipeline_type pipeline_type,
                                  bool fast_linking_enabled)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   /* Skip computing the pipeline hash when GPL fast-linking is enabled because these shaders aren't
    * supposed to be cached and computing the hash is costly. Though, make sure it's always computed
    * when RGP is enabled, otherwise ISA isn't reported.
    */
   return !fast_linking_enabled ||
          ((instance->vk.trace_mode & RADV_TRACE_MODE_RGP) && pipeline_type == RADV_PIPELINE_GRAPHICS);
}

void
radv_graphics_pipeline_state_finish(struct radv_device *device, struct radv_graphics_pipeline_state *gfx_state)
{
   radv_pipeline_layout_finish(device, &gfx_state->layout);
   vk_free(&device->vk.alloc, gfx_state->vk_data);

   if (gfx_state->stages) {
      for (uint32_t i = 0; i < MESA_VULKAN_SHADER_STAGES; i++)
         radv_pipeline_stage_finish(&gfx_state->stages[i]);

      ralloc_free(gfx_state->stages[MESA_SHADER_GEOMETRY].gs_copy_shader);

      free(gfx_state->stages);
   }
}

VkResult
radv_generate_graphics_pipeline_state(struct radv_device *device, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                      struct radv_graphics_pipeline_state *gfx_state)
{
   VK_FROM_HANDLE(radv_pipeline_layout, pipeline_layout, pCreateInfo->layout);
   const VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   const bool fast_linking_enabled = radv_is_fast_linking_enabled(pCreateInfo);
   enum radv_pipeline_type pipeline_type = RADV_PIPELINE_GRAPHICS;
   VkResult result;

   memset(gfx_state, 0, sizeof(*gfx_state));

   VkGraphicsPipelineLibraryFlagBitsEXT needed_lib_flags = ALL_GRAPHICS_LIB_FLAGS;
   if (create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
      const VkGraphicsPipelineLibraryCreateInfoEXT *lib_info =
         vk_find_struct_const(pCreateInfo->pNext, GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
      needed_lib_flags = lib_info ? lib_info->flags : 0;
      pipeline_type = RADV_PIPELINE_GRAPHICS_LIB;
   }

   radv_pipeline_layout_init(device, &gfx_state->layout, false);

   /* If we have libraries, import them first. */
   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
   if (libs_info) {
      for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
         const struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

         vk_graphics_pipeline_state_merge(&gfx_state->vk, &gfx_pipeline_lib->graphics_state);

         radv_graphics_pipeline_import_layout(&gfx_state->layout, &gfx_pipeline_lib->layout);

         needed_lib_flags &= ~gfx_pipeline_lib->lib_flags;
      }
   }

   result = vk_graphics_pipeline_state_fill(&device->vk, &gfx_state->vk, pCreateInfo,
                                            NULL /* driver_mv */, NULL /* driver_rp */,
                                            0, NULL, NULL,
                                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, &gfx_state->vk_data);
   if (result != VK_SUCCESS)
      goto fail;

   if (pipeline_layout)
      radv_graphics_pipeline_import_layout(&gfx_state->layout, pipeline_layout);

   if (radv_should_compute_pipeline_hash(device, pipeline_type, fast_linking_enabled))
      radv_pipeline_layout_hash(&gfx_state->layout);

   gfx_state->compilation_required = !radv_skip_graphics_pipeline_compile(device, pCreateInfo);
   if (gfx_state->compilation_required) {
      gfx_state->key = radv_generate_graphics_pipeline_key(device, pCreateInfo, &gfx_state->vk, needed_lib_flags);

      gfx_state->stages = calloc(MESA_VULKAN_SHADER_STAGES, sizeof(struct radv_shader_stage));
      if (!gfx_state->stages) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }

      for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         gfx_state->stages[i].stage = MESA_SHADER_NONE;
         gfx_state->stages[i].next_stage = MESA_SHADER_NONE;
      }

      for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
         const VkPipelineShaderStageCreateInfo *sinfo = &pCreateInfo->pStages[i];
         mesa_shader_stage stage = vk_to_mesa_shader_stage(sinfo->stage);

         radv_pipeline_stage_init(create_flags, sinfo, &gfx_state->layout, &gfx_state->key.stage_info[stage],
                                  &gfx_state->stages[stage]);
      }

      radv_pipeline_load_retained_shaders(device, pCreateInfo, gfx_state->stages);
   }

   return VK_SUCCESS;

fail:
   radv_graphics_pipeline_state_finish(device, gfx_state);
   return result;
}

/* Normalize the library subsets: a complete library and its final link must
 * describe the same compiler state even when the link contributes no subset. */
static void
radv_bc250_graphics_state_hash(const struct radv_device *device,
                               const VkGraphicsPipelineCreateInfo *info,
                               const struct radv_graphics_pipeline_state *gfx_state, uint8_t hash[32])
{
   struct radv_pipeline_layout layout = gfx_state->layout;
   radv_pipeline_layout_hash(&layout);
   const struct radv_graphics_pipeline_key key =
      radv_generate_graphics_pipeline_key(device, info, &gfx_state->vk, ALL_GRAPHICS_LIB_FLAGS);
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   radv_pipeline_hash(device, &layout, &ctx);
   _mesa_blake3_update(&ctx, &key.gfx_state, sizeof(key.gfx_state));
   _mesa_blake3_final(&ctx, hash);
}

void
radv_graphics_pipeline_hash(const struct radv_device *device, const struct radv_graphics_pipeline_state *gfx_state,
                            unsigned char *hash)
{
   blake3_hasher ctx;

   _mesa_blake3_init(&ctx);
   radv_pipeline_hash(device, &gfx_state->layout, &ctx);

   _mesa_blake3_update(&ctx, &gfx_state->key.gfx_state, sizeof(gfx_state->key.gfx_state));

   if (gfx_state->stages) {
      for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
         const struct radv_shader_stage *stage = &gfx_state->stages[s];

         if (stage->stage == MESA_SHADER_NONE)
            continue;

         _mesa_blake3_update(&ctx, stage->shader_blake3, sizeof(stage->shader_blake3));
         _mesa_blake3_update(&ctx, &stage->key, sizeof(stage->key));
      }
   }

   _mesa_blake3_final(&ctx, hash);
}

static void
radv_bc250_async_free_nir(struct radv_bc250_async_job *job)
{
   for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; ++s) {
      ralloc_free(job->nir[s]);
      job->nir[s] = NULL;
   }
}

/* Background: the optimized binaries of the same NIR. Published only if every stage compiled without scratch
 * and with shader info and LDS equal to the quick binaries; otherwise the quick binaries stay bound. */
static void
radv_bc250_async_execute(void *data, void *gdata, int thread_index)
{
   struct radv_bc250_async_job *job = data;
   struct radv_device *device = job->device;
   struct radv_graphics_pipeline *pipeline = job->pipeline;
   struct radv_shader *opt[MESA_VULKAN_SHADER_STAGES] = {NULL};
   bool ok = true;

   /* RADV_BC250_ASYNC_NEVER_SWAP=1 (testing): keep the quick binaries bound for the pipeline's lifetime. */
   if (debug_get_bool_option("RADV_BC250_ASYNC_NEVER_SWAP", false)) {
      radv_bc250_async_free_nir(job);
      return;
   }

   u_foreach_bit (s, job->stage_mask) {
      nir_shader *nir = job->nir[s];
      struct radv_shader_binary *binary = radv_shader_nir_to_asm(&device->compiler_info, &job->stage[s], &nir, 1,
                                                                 &job->gfx_state);
      struct radv_shader_debug_info debug = {0};
      /* Cached like the synchronous path's shaders, so the pipeline entry below resolves on the next run. */
      opt[s] = binary ? radv_shader_create(device, NULL, binary, job->skip_shaders_cache, &debug) : NULL;
      free(binary);
      const struct radv_shader *quick = pipeline->base.shaders[s];
      if (!opt[s] || !quick || opt[s]->config.scratch_bytes_per_wave || opt[s]->config.lds_size != quick->config.lds_size ||
          memcmp(&opt[s]->info, &quick->info, sizeof(quick->info))) {
         ok = false;
         break;
      }
   }
   radv_bc250_async_free_nir(job);

   if (!ok) {
      for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; ++s) {
         if (opt[s])
            radv_shader_unref(device, opt[s]);
      }
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 ASYNC COMPILE: optimized binaries not published (quick binaries stay)\n");
      return;
   }

   for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; ++s)
      pipeline->bc250_opt_shaders[s] = opt[s] ? opt[s] : pipeline->base.shaders[s];
   __atomic_store_n(&pipeline->bc250_opt_ready, 1, __ATOMIC_RELEASE);

   /* The disk cache gets the entry the synchronous path would have written, so the next run loads the
    * optimized binaries directly. */
   if (!job->skip_shaders_cache) {
      struct radv_graphics_pipeline *copy = malloc(sizeof(*copy));
      if (copy) {
         memcpy(copy, pipeline, sizeof(*copy));
         memcpy(copy->base.shaders, pipeline->bc250_opt_shaders, sizeof(copy->base.shaders));
         copy->base.cache_object = NULL;
         radv_pipeline_cache_insert(device, NULL, &copy->base);
         /* The entry reached the disk cache when it was added; this copy holds no reference. */
         if (copy->base.cache_object)
            vk_pipeline_cache_object_unref(&device->vk, copy->base.cache_object);
         free(copy);
      }
   }
   if (getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 ASYNC COMPILE: optimized binaries published (stages 0x%x)\n", job->stage_mask);
}

static bool
radv_bc250_async_allowed(const struct radv_device *device, const struct radv_graphics_pipeline *pipeline,
                         const struct radv_shader_stage *stages, bool fast_linking_enabled)
{
   const VkPipelineCreateFlags2 excluded = VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR |
      VK_PIPELINE_CREATE_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT |
      VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT | VK_PIPELINE_CREATE_2_CAPTURE_STATISTICS_BIT_KHR |
      VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT;
   return device->bc250_async && !pipeline->base.is_internal && !fast_linking_enabled &&
          !(pipeline->base.create_flags & excluded) &&
          (pipeline->active_stages & VK_SHADER_STAGE_MESH_BIT_EXT) &&
          !stages[MESA_SHADER_MESH].key.keep_executable_info &&
          !stages[MESA_SHADER_FRAGMENT].key.keep_executable_info;
}

static VkResult
radv_graphics_pipeline_compile(struct radv_graphics_pipeline *pipeline, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                               const struct radv_graphics_pipeline_state *gfx_state, struct radv_device *device,
                               struct vk_pipeline_cache *cache, bool fast_linking_enabled)
{
   const struct radv_compiler_info *compiler_info = &device->compiler_info;
   struct radv_shader_binary *binaries[MESA_VULKAN_SHADER_STAGES] = {NULL};
   struct radv_shader_binary *gs_copy_binary = NULL;
   bool skip_shaders_cache = radv_pipeline_skip_shaders_cache(device, &pipeline->base);
   const bool bc250_prepare_before_cache = radv_device_physical(device)->bc250_native_mesh &&
      ((pipeline->active_stages & VK_SHADER_STAGE_TASK_BIT_EXT) ||
       (compiler_info->key.bc250_split_mesh && (pipeline->active_stages & VK_SHADER_STAGE_MESH_BIT_EXT)));
   struct radv_shader_stage *stages = gfx_state->stages;
   const VkPipelineCreationFeedbackCreateInfo *creation_feedback =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_CREATION_FEEDBACK_CREATE_INFO);
   VkPipelineCreationFeedback pipeline_feedback = {
      .flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT,
   };
   VkResult result = VK_SUCCESS;
   const bool retain_shaders =
      !!(pipeline->base.create_flags & VK_PIPELINE_CREATE_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT);
   struct radv_retained_shaders *retained_shaders = NULL;

   int64_t pipeline_start = os_time_get_nano();

   if (radv_should_compute_pipeline_hash(device, pipeline->base.type, fast_linking_enabled)) {
      radv_graphics_pipeline_hash(device, gfx_state, pipeline->base.blake3);

      pipeline->base.pipeline_hash = *(uint64_t *)pipeline->base.blake3;
   }

   /* Skip the shaders cache when any of the below are true:
    * - fast-linking is enabled because it's useless to cache unoptimized pipelines
    * - graphics pipeline libraries are created with the RETAIN_LINK_TIME_OPTIMIZATION flag and
    *   module identifiers are used (ie. no SPIR-V provided).
    */
   if (fast_linking_enabled) {
      skip_shaders_cache = true;
   } else if (retain_shaders) {
      assert(pipeline->base.create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR);
      for (uint32_t i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         if (stages[i].stage != MESA_SHADER_NONE && !stages[i].spirv.size) {
            skip_shaders_cache = true;
            break;
         }
      }
   }

   /* BC250 Mesh LDS fit retry (radv_bc250_fit_retry): without the cached plan,
    * a cache hit would pair the retried shaders with the first attempt's split
    * draw multiplier (radv_bc250_prepare_task runs before the lookup). */
   if (stages[MESA_SHADER_MESH].bc250_fit_reclaim && !compiler_info->key.bc250_cache_plan)
      skip_shaders_cache = true;

   /* Cached native split shaders still require their draw multiplier and
    * argument helper. Reconstruct those before lookup; task-backed pipelines
    * retain their existing cache exclusion until their metadata is covered. */
   bool found_in_application_cache = true;
   const bool restored_bc250 = bc250_prepare_before_cache &&
      (compiler_info->key.bc250_cache_plan || device->bc250_env.pipeline_plan) &&
      pipeline->base.type == RADV_PIPELINE_GRAPHICS && !skip_shaders_cache &&
      radv_graphics_pipeline_cache_search(device, cache, pipeline, &gfx_state->layout, &found_in_application_cache);
   if (bc250_prepare_before_cache && !restored_bc250) {
      if (pipeline->base.create_flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)
         return VK_PIPELINE_COMPILE_REQUIRED;
      result = radv_bc250_prepare_task(device, pipeline, gfx_state);
      if (result != VK_SUCCESS)
         return result;
      skip_shaders_cache |= pipeline->bc250_task_pipeline != VK_NULL_HANDLE && !device->bc250_env.pipeline_plan;
   }

   if (restored_bc250 || (!skip_shaders_cache &&
       radv_graphics_pipeline_cache_search(device, cache, pipeline, &gfx_state->layout, &found_in_application_cache))) {
      if (bc250_prepare_before_cache && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 graphics cache hit: split_pieces=%u\n", pipeline->bc250_direct_split_pieces);
      if (found_in_application_cache)
         pipeline_feedback.flags |= VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;

      if (retain_shaders) {
         /* For graphics pipeline libraries created with the RETAIN_LINK_TIME_OPTIMIZATION flag, we
          * need to retain the stage info because we can't know if the LTO pipelines will
          * be find in the shaders cache.
          */
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(&pipeline->base);

         gfx_pipeline_lib->stages = radv_copy_shader_stage_create_info(device, pCreateInfo->stageCount,
                                                                       pCreateInfo->pStages, gfx_pipeline_lib->mem_ctx);
         if (!gfx_pipeline_lib->stages)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

         gfx_pipeline_lib->stage_count = pCreateInfo->stageCount;

         for (unsigned i = 0; i < pCreateInfo->stageCount; i++) {
            mesa_shader_stage s = vk_to_mesa_shader_stage(pCreateInfo->pStages[i].stage);
            gfx_pipeline_lib->stage_keys[s] = gfx_state->key.stage_info[s];
         }
      }

      result = VK_SUCCESS;
      goto done;
   }

   if (pipeline->base.create_flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)
      return VK_PIPELINE_COMPILE_REQUIRED;

   if (retain_shaders) {
      struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(&pipeline->base);
      retained_shaders = &gfx_pipeline_lib->retained_shaders;
   }

   if (radv_device_physical(device)->bc250_native_mesh && !bc250_prepare_before_cache) {
      VkResult hybrid_result = radv_bc250_prepare_task(device, pipeline, gfx_state);
      if (hybrid_result != VK_SUCCESS)
         return hybrid_result;
   }
   const bool noop_fs = radv_pipeline_needs_noop_fs(pipeline, &gfx_state->key.gfx_state);

   struct radv_shader_debug_info debug[MESA_VULKAN_SHADER_STAGES] = {0};
   struct radv_shader_debug_info gs_copy_debug = {0};
   struct radv_bc250_async_job *async_job = NULL;
   /* The pipeline plan is captured from the final binaries: no quick/optimized swap with it. */
   if (!retained_shaders && !device->bc250_env.pipeline_plan &&
       radv_bc250_async_allowed(device, pipeline, stages, fast_linking_enabled))
      async_job = calloc(1, sizeof(*async_job));
   bc250_async_capture = async_job;
   VkResult compile_result = radv_graphics_shaders_compile(compiler_info, cache, stages, &gfx_state->key.gfx_state, pipeline->base.is_internal,
                                 retained_shaders, noop_fs, debug, binaries, &gs_copy_debug, &gs_copy_binary);
   bc250_async_capture = NULL;
   if (async_job && (compile_result != VK_SUCCESS || !async_job->stage_mask)) {
      radv_bc250_async_free_nir(async_job);
      free(async_job);
      async_job = NULL;
   }
   if (compile_result != VK_SUCCESS)
      return compile_result;
   /* RADV_BC250_ASYNC_COMPILE: the quick binaries are never cached. */
   radv_graphics_shaders_create(device, cache, skip_shaders_cache || async_job, pipeline->base.shaders, binaries, debug,
                                &pipeline->base.gs_copy_shader, gs_copy_binary, &gs_copy_debug);

   if (device->bc250_env.pipeline_plan && pipeline->base.shaders[MESA_SHADER_MESH]) {
      radv_bc250_capture_pipeline_plan(device, pipeline, &stages[MESA_SHADER_MESH]);
      radv_bc250_graphics_state_hash(device, pCreateInfo, gfx_state, pipeline->bc250_plan.state_hash);
      if (!radv_bc250_pipeline_plan_admitted(&pipeline->bc250_plan, pipeline->base.shaders[MESA_SHADER_MESH]))
         return VK_ERROR_FEATURE_NOT_PRESENT;
   }

   if (async_job) {
      async_job->device = device;
      async_job->pipeline = pipeline;
      async_job->skip_shaders_cache = skip_shaders_cache;
      async_job->gfx_state = gfx_state->key.gfx_state;
      util_queue_fence_init(&async_job->fence);
      pipeline->bc250_async = async_job;
      util_queue_add_job(&device->bc250_async_queue, async_job, &async_job->fence, radv_bc250_async_execute, NULL, 0);
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 ASYNC COMPILE: quick binaries bound, optimized queued (stages 0x%x)\n",
                 async_job->stage_mask);
   } else if (!skip_shaders_cache) {
      radv_pipeline_cache_insert(device, cache, &pipeline->base);
   }

   free(gs_copy_binary);
   for (int i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      free(binaries[i]);
      if (stages[i].nir) {
         if (radv_can_dump_shader_stats(&device->compiler_info, stages[i].nir) && pipeline->base.shaders[i]) {
            radv_dump_shader_stats(device, &pipeline->base, pipeline->base.shaders[i], stderr);
         }
      }
   }

done:
   pipeline_feedback.duration = os_time_get_nano() - pipeline_start;

   if (creation_feedback) {
      *creation_feedback->pPipelineCreationFeedback = pipeline_feedback;

      if (creation_feedback->pipelineStageCreationFeedbackCount > 0) {
         uint32_t num_feedbacks = 0;

         for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
            mesa_shader_stage s = vk_to_mesa_shader_stage(pCreateInfo->pStages[i].stage);
            creation_feedback->pPipelineStageCreationFeedbacks[num_feedbacks++] = stages[s].feedback;
         }

         /* Stages imported from graphics pipeline libraries are defined as additional entries in the
          * order they were imported.
          */
         const VkPipelineLibraryCreateInfoKHR *libs_info =
            vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
         if (libs_info) {
            for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
               VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
               struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

               if (!gfx_pipeline_lib->base.active_stages)
                  continue;

               radv_foreach_stage (s, gfx_pipeline_lib->base.active_stages) {
                  creation_feedback->pPipelineStageCreationFeedbacks[num_feedbacks++] = stages[s].feedback;
               }
            }
         }

         assert(num_feedbacks == creation_feedback->pipelineStageCreationFeedbackCount);
      }
   }

   return result;
}

struct radv_vgt_shader_key
radv_get_vgt_shader_key(const struct radv_device *device, struct radv_shader **shaders,
                        const struct radv_shader *gs_copy_shader)
{
   uint8_t hs_size = 64, gs_size = 64, vs_size = 64;
   struct radv_shader *last_vgt_shader = NULL;
   struct radv_vgt_shader_key key;

   memset(&key, 0, sizeof(key));

   if (shaders[MESA_SHADER_GEOMETRY]) {
      last_vgt_shader = shaders[MESA_SHADER_GEOMETRY];
   } else if (shaders[MESA_SHADER_TESS_EVAL]) {
      last_vgt_shader = shaders[MESA_SHADER_TESS_EVAL];
   } else if (shaders[MESA_SHADER_VERTEX]) {
      last_vgt_shader = shaders[MESA_SHADER_VERTEX];
   } else {
      assert(shaders[MESA_SHADER_MESH]);
      last_vgt_shader = shaders[MESA_SHADER_MESH];
   }

   vs_size = gs_size = last_vgt_shader->info.wave_size;
   if (gs_copy_shader)
      vs_size = gs_copy_shader->info.wave_size;

   if (shaders[MESA_SHADER_TESS_CTRL])
      hs_size = shaders[MESA_SHADER_TESS_CTRL]->info.wave_size;

   key.tess = !!shaders[MESA_SHADER_TESS_CTRL];
   key.gs = !!shaders[MESA_SHADER_GEOMETRY];
   key.mesh = !!shaders[MESA_SHADER_MESH];
   key.bc250_fl0 = shaders[MESA_SHADER_MESH] && shaders[MESA_SHADER_MESH]->info.ms.bc250_fl0;
   if (last_vgt_shader->info.is_ngg) {
      key.ngg = 1;
      key.ngg_passthrough = last_vgt_shader->info.is_ngg_passthrough;
      key.ngg_wave_id_en = last_vgt_shader->info.ngg_wave_id_en;
   }

   key.hs_wave32 = hs_size == 32;
   key.vs_wave32 = vs_size == 32;
   key.gs_wave32 = gs_size == 32;

   return key;
}

static void
radv_pipeline_init_shader_stages_state(const struct radv_device *device, struct radv_graphics_pipeline *pipeline)
{
   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      bool shader_exists = !!pipeline->base.shaders[i];
      if (shader_exists || i < MESA_SHADER_COMPUTE) {
         if (shader_exists) {
            pipeline->base.need_indirect_descriptors |=
               radv_shader_need_indirect_descriptors(pipeline->base.shaders[i]);
            pipeline->base.need_push_constants_upload |=
               radv_shader_need_push_constants_upload(pipeline->base.shaders[i]);
            pipeline->base.need_dynamic_descriptors_offset_addr |=
               radv_shader_need_dynamic_descriptors_offset_addr(pipeline->base.shaders[i]);
         }
      }
   }
}

static void
radv_pipeline_init_extra(struct radv_graphics_pipeline *pipeline, const VkGraphicsPipelineCreateInfoRADV *radv_info,
                         const struct vk_graphics_pipeline_state *state)
{
   pipeline->custom_blend_mode = radv_info->custom_blend_mode;
}

bool
radv_needs_null_export_workaround(const struct radv_device *device, const struct radv_shader *ps,
                                  unsigned custom_blend_mode)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const enum amd_gfx_level gfx_level = pdev->info.gfx_level;

   if (!ps)
      return false;

   /* Ensure that some export memory is always allocated, for two reasons:
    *
    * 1) Correctness: The hardware ignores the EXEC mask if no export
    *    memory is allocated, so KILL and alpha test do not work correctly
    *    without this.
    * 2) Performance: Every shader needs at least a NULL export, even when
    *    it writes no color/depth output. The NULL export instruction
    *    stalls without this setting.
    *
    * Don't add this to CB_SHADER_MASK.
    *
    * GFX10 supports pixel shaders without exports by setting both the
    * color and Z formats to SPI_SHADER_ZERO. The hw will skip export
    * instructions if any are present.
    *
    * GFX11 requires one color output, otherwise the DCC decompression does nothing.
    *
    * Primitive Ordered Pixel Shading also requires an export, otherwise interlocking doesn't work
    * correctly before GFX11, and a hang happens on GFX11.
    */
   return (gfx_level <= GFX9 || ps->info.ps.can_discard || ps->info.ps.pops ||
           (custom_blend_mode == V_028808_CB_DCC_DECOMPRESS_GFX11 && gfx_level >= GFX11)) &&
          !ps->info.ps.writes_z && !ps->info.ps.writes_stencil && !ps->info.ps.writes_sample_mask;
}

static VkResult
radv_graphics_pipeline_import_binaries(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                       const struct radv_pipeline_layout *layout,
                                       const VkPipelineBinaryInfoKHR *binary_info)
{
   blake3_hash pipeline_hash;
   struct mesa_blake3 ctx;

   _mesa_blake3_init(&ctx);
   const struct radv_pipeline_binary *plan_binary = NULL;

   for (uint32_t i = 0; i < binary_info->binaryCount; i++) {
      VK_FROM_HANDLE(radv_pipeline_binary, pipeline_binary, binary_info->pPipelineBinaries[i]);
      struct radv_shader *shader;
      struct blob_reader blob;

      if (radv_bc250_pipeline_binary_is_plan(pipeline_binary)) {
         const bool complete_lib = pipeline->base.type == RADV_PIPELINE_GRAPHICS_LIB &&
            device->bc250_env.gpl_binary_link &&
            radv_pipeline_to_graphics_lib(&pipeline->base)->lib_flags == ALL_GRAPHICS_LIB_FLAGS;
         if (!device->bc250_env.pipeline_plan || plan_binary ||
             (pipeline->base.type != RADV_PIPELINE_GRAPHICS && !complete_lib))
            return VK_ERROR_FEATURE_NOT_PRESENT;
         plan_binary = pipeline_binary;
         _mesa_blake3_update(&ctx, pipeline_binary->key, sizeof(pipeline_binary->key));
         continue;
      }

      blob_reader_init(&blob, pipeline_binary->data, pipeline_binary->size);

      shader = radv_shader_deserialize(device, pipeline_binary->key, sizeof(pipeline_binary->key), &blob);
      if (!shader)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      if (device->bc250_env.pipeline_plan &&
          (blob.overrun || shader->info.stage >= MESA_VULKAN_SHADER_STAGES ||
           shader->info.stage == MESA_SHADER_TASK || shader->info.stage == MESA_SHADER_COMPUTE ||
           pipeline->base.shaders[shader->info.stage])) {
         radv_shader_unref(device, shader);
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }

      if (shader->info.stage == MESA_SHADER_VERTEX && i > 0) {
         /* The GS copy-shader is a VS placed after all other stages. */
         pipeline->base.gs_copy_shader = shader;
      } else {
         pipeline->base.shaders[shader->info.stage] = shader;
      }

      _mesa_blake3_update(&ctx, pipeline_binary->key, sizeof(pipeline_binary->key));
   }

   _mesa_blake3_final(&ctx, pipeline_hash);

   if (device->bc250_env.pipeline_plan && pipeline->base.shaders[MESA_SHADER_MESH]) {
      if (!plan_binary)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      VkResult result = radv_bc250_pipeline_binary_restore(device, pipeline, layout, plan_binary);
      if (result != VK_SUCCESS)
         return result;
   } else if (plan_binary) {
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }

   pipeline->base.pipeline_hash = *(uint64_t *)pipeline_hash;

   pipeline->has_pipeline_binaries = true;

   return VK_SUCCESS;
}

/* BC250 Mesh LDS fit. An expanded Mesh shader (every triangle owns three
 * private vertices) keeps its output staging and the expanded outputs in LDS.
 * When they do not fit, the NGG lowering would move outputs to the Mesh
 * scratch ring, which is not validated on GFX1013, and the pipeline is refused
 * (bc250_lds_refused). Instead of failing pipeline creation, compile it again:
 *   1. same pieces, dead shared-variable copies dropped (bc250_fit_reclaim);
 *   2. then one more piece per retry (bc250_fit_min_pieces), each piece
 *      handling fewer triangles: fewer expanded vertices, the same launch
 *      class (expanded private vertices, fast launch 0, no scratch ring),
 *      until it fits or the split refuses (at most 5 pieces).
 * Only pipelines refused today take this path: every pipeline that fits is
 * compiled exactly as before. Each retry is a deterministic function of the
 * pipeline's inputs and the compiler key, so a cached result stays valid;
 * the split draw multiplier is restored from the cached plan
 * (BC250_CACHE_PLAN), and without it a retried pipeline is not cached (see
 * radv_graphics_pipeline_compile). Returns true when gfx_state is ready for
 * another compile. */
static bool
radv_bc250_fit_reset(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                     const VkGraphicsPipelineCreateInfo *pCreateInfo, struct radv_graphics_pipeline_state *gfx_state,
                     VkShaderStageFlags active_stages, bool reclaim, unsigned min_pieces);

static bool
radv_bc250_fit_retry(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                     const VkGraphicsPipelineCreateInfo *pCreateInfo, struct radv_graphics_pipeline_state *gfx_state,
                     VkShaderStageFlags active_stages, unsigned attempt)
{
   if (!radv_device_physical(device)->bc250_native_mesh || !gfx_state->stages ||
       pipeline->base.type != RADV_PIPELINE_GRAPHICS ||
       vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR))
      return false;
   const struct radv_shader_stage *ms = &gfx_state->stages[MESA_SHADER_MESH];
   if (!ms->bc250_lds_refused)
      return false;
   /* The previous attempt must have produced the pieces it asked for;
    * otherwise the split cannot go further for this shader. */
   const unsigned pieces = MAX2(ms->bc250_split_pieces, 1);
   if (ms->bc250_fit_min_pieces > pieces)
      return false;
   /* Skip the same-pieces step when there is nothing to drop, or when the
    * refused attempt already had the compact layout (RADV_BC250_MESH_COMPACT_LDS)
    * or dropped the dead copies itself (RADV_BC250_MESH_DIRECT_READ dead part). */
   const bool reclaimed = ms->bc250_fit_reclaim || device->compiler_info.key.bc250_mesh_compact_lds ||
                          (device->compiler_info.key.bc250_mesh_direct_read & RADV_BC250_DIRECT_READ_DEAD) ||
                          !ms->bc250_dead_shared;
   const bool reclaim = true;
   const unsigned min_pieces = reclaimed ? pieces + 1 : pieces;
   if (min_pieces > 5 || attempt > 6)
      return false;
   if (getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 Mesh LDS fit: retry %u after refusal with pieces=%u reclaim=%u: min_pieces=%u reclaim=%u\n",
              attempt, pieces, ms->bc250_fit_reclaim, min_pieces, reclaim);
   return radv_bc250_fit_reset(device, pipeline, pCreateInfo, gfx_state, active_stages, reclaim, min_pieces);
}

/* Undo a refused attempt's pipeline state (radv_bc250_prepare_task) and regenerate gfx_state for
 * another compile with the given fit parameters. Returns true when gfx_state is ready. */
static bool
radv_bc250_fit_reset(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                     const VkGraphicsPipelineCreateInfo *pCreateInfo, struct radv_graphics_pipeline_state *gfx_state,
                     VkShaderStageFlags active_stages, bool reclaim, unsigned min_pieces)
{
   /* Undo the refused attempt's pipeline state (radv_bc250_prepare_task). */
   VkDevice _device = radv_device_to_handle(device);
   if (pipeline->bc250_task_pipeline)
      radv_DestroyPipeline(_device, pipeline->bc250_task_pipeline, NULL);
   if (pipeline->bc250_setup_pipeline && !pipeline->bc250_shared_setup)
      radv_DestroyPipeline(_device, pipeline->bc250_setup_pipeline, NULL);
   if (pipeline->bc250_task_layout && !pipeline->bc250_shared_setup)
      radv_DestroyPipelineLayout(_device, pipeline->bc250_task_layout, NULL);
   pipeline->bc250_task_pipeline = VK_NULL_HANDLE;
   pipeline->bc250_setup_pipeline = VK_NULL_HANDLE;
   pipeline->bc250_task_layout = VK_NULL_HANDLE;
   pipeline->bc250_shared_setup = false;
   pipeline->bc250_payload_stride = 0;
   pipeline->bc250_ordered = false;
   pipeline->bc250_direct_split_pieces = 0;
   pipeline->active_stages = active_stages;
   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      if (pipeline->base.shaders[i]) {
         radv_shader_unref(device, pipeline->base.shaders[i]);
         pipeline->base.shaders[i] = NULL;
      }
   }
   if (pipeline->base.gs_copy_shader) {
      radv_shader_unref(device, pipeline->base.gs_copy_shader);
      pipeline->base.gs_copy_shader = NULL;
   }

   radv_graphics_pipeline_state_finish(device, gfx_state);
   if (radv_generate_graphics_pipeline_state(device, pCreateInfo, gfx_state) != VK_SUCCESS || !gfx_state->stages) {
      /* The caller finishes gfx_state again: leave it empty. */
      memset(gfx_state, 0, sizeof(*gfx_state));
      radv_pipeline_layout_init(device, &gfx_state->layout, false);
      return false;
   }
   gfx_state->stages[MESA_SHADER_MESH].bc250_fit_reclaim = reclaim;
   gfx_state->stages[MESA_SHADER_MESH].bc250_fit_min_pieces = min_pieces;
   return true;
}

static VkResult
radv_graphics_pipeline_init(struct radv_graphics_pipeline *pipeline, struct radv_device *device,
                            struct vk_pipeline_cache *cache, const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   bool fast_linking_enabled = radv_is_fast_linking_enabled(pCreateInfo);
   struct radv_graphics_pipeline_state gfx_state;
   VkResult result = VK_SUCCESS;

   /* A partial BC250 interface cannot prove barycentric slot ownership or
    * the Task consumer route. Recompile the merged source at the final link,
    * including when the application requests a link without LTO. */
   if (device->bc250_env.gpl_source_link)
      fast_linking_enabled = false;

   pipeline->last_vgt_api_stage = MESA_SHADER_NONE;

   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);

   const VkPipelineBinaryInfoKHR *binary_info = vk_find_struct_const(pCreateInfo->pNext, PIPELINE_BINARY_INFO_KHR);
   const bool import_pipeline_binaries = binary_info && binary_info->binaryCount > 0;

   /* If we have libraries, import them first. */
   if (libs_info) {
      for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

         assert(pipeline_lib->type == RADV_PIPELINE_GRAPHICS_LIB);

         radv_graphics_pipeline_import_lib(device, pipeline, gfx_pipeline_lib, import_pipeline_binaries);
      }
   }

   radv_pipeline_import_graphics_info(device, pipeline, pCreateInfo);

   result = radv_generate_graphics_pipeline_state(device, pCreateInfo, &gfx_state);
   if (result != VK_SUCCESS)
      return result;

   /* A fast link owns its private handles, even after library destruction.
    * Partial compiled interfaces are refused until their route compatibility
    * has been proven independently. */
   if (!import_pipeline_binaries && libs_info && device->bc250_env.gpl_binary_link) {
      struct radv_graphics_lib_pipeline *compiled = NULL;
      for (unsigned i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, imported, libs_info->pLibraries[i]);
         struct radv_graphics_lib_pipeline *lib = radv_pipeline_to_graphics_lib(imported);
         if (lib->base.bc250_plan.version)
            compiled = lib;
      }
      if (compiled) {
         uint8_t hash[32];
         radv_bc250_graphics_state_hash(device, pCreateInfo, &gfx_state, hash);
         if (libs_info->libraryCount != 1 || pCreateInfo->stageCount ||
             compiled->lib_flags != ALL_GRAPHICS_LIB_FLAGS || !radv_is_fast_linking_enabled(pCreateInfo) ||
             memcmp(hash, compiled->base.bc250_plan.state_hash, sizeof(hash))) {
            result = VK_ERROR_FEATURE_NOT_PRESENT;
         } else {
            VK_FROM_HANDLE(radv_pipeline, producer, compiled->base.bc250_task_pipeline);
            VK_FROM_HANDLE(radv_pipeline, setup, compiled->base.bc250_setup_pipeline);
            result = radv_bc250_restore_cached_plan(device, pipeline, &gfx_state.layout,
               &compiled->base.bc250_plan, pipeline->base.shaders[MESA_SHADER_MESH],
               pipeline->base.shaders[MESA_SHADER_FRAGMENT],
               producer ? producer->shaders[MESA_SHADER_COMPUTE] : NULL,
               producer && setup ? setup->shaders[MESA_SHADER_COMPUTE] : NULL);
         }
         if (result != VK_SUCCESS) {
            radv_graphics_pipeline_state_finish(device, &gfx_state);
            return result;
         }
      }
   }

   if (import_pipeline_binaries) {
      result = radv_graphics_pipeline_import_binaries(device, pipeline, &gfx_state.layout, binary_info);
      if (result == VK_SUCCESS && pipeline->bc250_plan.version) {
         uint8_t hash[32];
         radv_bc250_graphics_state_hash(device, pCreateInfo, &gfx_state, hash);
         if (memcmp(hash, pipeline->bc250_plan.state_hash, sizeof(hash)))
            result = VK_ERROR_FEATURE_NOT_PRESENT;
      }
   } else {
      if (gfx_state.compilation_required) {
         const VkShaderStageFlags active_stages = pipeline->active_stages;
         result =
            radv_graphics_pipeline_compile(pipeline, pCreateInfo, &gfx_state, device, cache, fast_linking_enabled);
         /* BC250 Mesh LDS fit: see radv_bc250_fit_retry. */
         unsigned retries = 0;
         while (result == VK_ERROR_FEATURE_NOT_PRESENT &&
                radv_bc250_fit_retry(device, pipeline, pCreateInfo, &gfx_state, active_stages, retries + 1)) {
            retries++;
            result =
               radv_graphics_pipeline_compile(pipeline, pCreateInfo, &gfx_state, device, cache, fast_linking_enabled);
         }
         /* RADV_BC250_MESH_SPLIT_ANY: one last attempt for a Mesh pipeline every route refused,
          * letting Mesh-only lines/points split. Pipelines that already compile never get here. */
         if (result == VK_ERROR_FEATURE_NOT_PRESENT && (active_stages & VK_SHADER_STAGE_MESH_BIT_EXT) &&
             debug_get_bool_option("RADV_BC250_MESH_SPLIT_ANY", false) &&
             radv_device_physical(device)->bc250_native_mesh && pipeline->base.type == RADV_PIPELINE_GRAPHICS &&
             !vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR) &&
             radv_bc250_fit_reset(device, pipeline, pCreateInfo, &gfx_state, active_stages, false, 0)) {
            radv_bc250_split_refused_retry = true;
            result =
               radv_graphics_pipeline_compile(pipeline, pCreateInfo, &gfx_state, device, cache, fast_linking_enabled);
            /* The same LDS fit steps (more, smaller pieces) for the retried form. */
            unsigned split_retries = 0;
            while (result == VK_ERROR_FEATURE_NOT_PRESENT &&
                   radv_bc250_fit_retry(device, pipeline, pCreateInfo, &gfx_state, active_stages, ++split_retries))
               result =
                  radv_graphics_pipeline_compile(pipeline, pCreateInfo, &gfx_state, device, cache, fast_linking_enabled);
            radv_bc250_split_refused_retry = false;
            if (getenv("BC250_TRACE_COMPILE"))
               fprintf(stderr, "BC250 Mesh refused retry (split any topology): %s\n",
                       result == VK_SUCCESS ? "admitted" : "refused");
         }
         if (retries && getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 Mesh LDS fit: %s after %u retries (pieces=%u reclaim=%u)\n",
                    result == VK_SUCCESS ? "admitted" : "refused", retries,
                    gfx_state.stages ? gfx_state.stages[MESA_SHADER_MESH].bc250_split_pieces : 0,
                    gfx_state.stages ? gfx_state.stages[MESA_SHADER_MESH].bc250_fit_reclaim : 0);
      }
   }

   if (result != VK_SUCCESS) {
      radv_graphics_pipeline_state_finish(device, &gfx_state);
      return result;
   }

   radv_pipeline_init_multisample_state(device, pipeline, pCreateInfo, &gfx_state.vk);

   if (!radv_pipeline_has_stage(pipeline, MESA_SHADER_MESH))
      radv_pipeline_init_input_assembly_state(device, pipeline);
   radv_pipeline_init_dynamic_state(device, pipeline, &gfx_state.vk);

   radv_pipeline_init_shader_stages_state(device, pipeline);

   pipeline->uses_out_of_order_rast = gfx_state.vk.rs->rasterization_order_amd == VK_RASTERIZATION_ORDER_RELAXED_AMD;
   pipeline->uses_vrs_attachment = radv_pipeline_uses_vrs_attachment(pipeline, &gfx_state.vk);

   uint32_t push_constant_size = 0;
   for (uint32_t i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      const struct radv_shader *shader = pipeline->base.shaders[i];

      if (!shader)
         continue;

      push_constant_size = MAX2(push_constant_size, shader->info.push_constant_size);
   }

   pipeline->base.push_constant_size = align(push_constant_size, 4);
   pipeline->base.dynamic_offset_count = gfx_state.layout.dynamic_offset_count;

   const VkGraphicsPipelineCreateInfoRADV *radv_info =
      vk_find_struct_const(pCreateInfo->pNext, GRAPHICS_PIPELINE_CREATE_INFO_RADV);
   if (radv_info) {
      radv_pipeline_init_extra(pipeline, radv_info, &gfx_state.vk);
   }

   radv_graphics_pipeline_state_finish(device, &gfx_state);
   return result;
}

static VkResult
radv_graphics_pipeline_create(VkDevice _device, VkPipelineCache _cache, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                              const VkAllocationCallbacks *pAllocator, VkPipeline *pPipeline)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(vk_pipeline_cache, cache, _cache);
   struct radv_graphics_pipeline *pipeline;
   VkResult result;

   pipeline = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pipeline), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pipeline == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   radv_pipeline_init(device, &pipeline->base, RADV_PIPELINE_GRAPHICS);
   pipeline->base.create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   pipeline->base.is_internal = _cache == device->meta_state.cache;

   result = radv_graphics_pipeline_init(pipeline, device, cache, pCreateInfo);
   if (result != VK_SUCCESS) {
      radv_pipeline_destroy(device, &pipeline->base, pAllocator);
      return result;
   }

   radv_bc250_report_mesh_route(device, pipeline->base.shaders[MESA_SHADER_MESH],
      "pipeline", pipeline->bc250_direct_split_pieces, pipeline->bc250_task_pipeline != VK_NULL_HANDLE,
      pipeline->bc250_ordered, pipeline->bc250_plan.flags & RADV_BC250_PLAN_EMPTY);
   /* RADV_BC250_MESH_IDXPASS: a declined or failed index route keeps the Mesh route. */
   if (device->compiler_info.bc250x.idxpass && pipeline->base.shaders[MESA_SHADER_MESH] && !pipeline->bc250_task_pipeline)
      radv_bc250_idx_create(device, pipeline, pCreateInfo);

   radv_pipeline_report_pso_history(device, &pipeline->base);

   *pPipeline = radv_pipeline_to_handle(&pipeline->base);
   radv_rmv_log_graphics_pipeline_create(device, &pipeline->base, pipeline->base.is_internal);
   return VK_SUCCESS;
}

void
radv_destroy_graphics_pipeline(struct radv_device *device, struct radv_graphics_pipeline *pipeline)
{
   if (pipeline->bc250_async) {
      /* Removes a queued job, or waits for a running one. */
      util_queue_drop_job(&device->bc250_async_queue, &pipeline->bc250_async->fence);
      util_queue_fence_destroy(&pipeline->bc250_async->fence);
      radv_bc250_async_free_nir(pipeline->bc250_async);
      free(pipeline->bc250_async);
      pipeline->bc250_async = NULL;
      if (__atomic_load_n(&pipeline->bc250_opt_ready, __ATOMIC_ACQUIRE)) {
         for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
            if (pipeline->bc250_opt_shaders[i] && pipeline->bc250_opt_shaders[i] != pipeline->base.shaders[i])
               radv_shader_unref(device, pipeline->bc250_opt_shaders[i]);
         }
      }
   }
   if (pipeline->bc250_task_pipeline)
      radv_DestroyPipeline(radv_device_to_handle(device), pipeline->bc250_task_pipeline, NULL);
   if (pipeline->bc250_idx_cs)
      radv_DestroyPipeline(radv_device_to_handle(device), pipeline->bc250_idx_cs, NULL);
   if (pipeline->bc250_idx_gfx)
      radv_DestroyPipeline(radv_device_to_handle(device), pipeline->bc250_idx_gfx, NULL);
   if (pipeline->bc250_idx_layout)
      radv_DestroyPipelineLayout(radv_device_to_handle(device), pipeline->bc250_idx_layout, NULL);
   if (pipeline->bc250_idx_setup)
      radv_DestroyPipeline(radv_device_to_handle(device), pipeline->bc250_idx_setup, NULL);
   if (pipeline->bc250_idx_setup_layout)
      radv_DestroyPipelineLayout(radv_device_to_handle(device), pipeline->bc250_idx_setup_layout, NULL);
   if (pipeline->bc250_idx_cs_ind)
      radv_DestroyPipeline(radv_device_to_handle(device), pipeline->bc250_idx_cs_ind, NULL);
   if (pipeline->bc250_setup_pipeline && !pipeline->bc250_shared_setup)
      radv_DestroyPipeline(radv_device_to_handle(device), pipeline->bc250_setup_pipeline, NULL);
   if (pipeline->bc250_task_layout && !pipeline->bc250_shared_setup)
      radv_DestroyPipelineLayout(radv_device_to_handle(device), pipeline->bc250_task_layout, NULL);
   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      if (pipeline->base.shaders[i])
         radv_shader_unref(device, pipeline->base.shaders[i]);
   }

   if (pipeline->base.gs_copy_shader)
      radv_shader_unref(device, pipeline->base.gs_copy_shader);
}

static VkResult
radv_graphics_lib_pipeline_init(struct radv_graphics_lib_pipeline *pipeline, struct radv_device *device,
                                struct vk_pipeline_cache *cache, const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   VK_FROM_HANDLE(radv_pipeline_layout, pipeline_layout, pCreateInfo->layout);
   VkResult result;

   const VkGraphicsPipelineLibraryCreateInfoEXT *lib_info =
      vk_find_struct_const(pCreateInfo->pNext, GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
   bool fast_linking_enabled = radv_is_fast_linking_enabled(pCreateInfo);

   struct vk_graphics_pipeline_state *state = &pipeline->graphics_state;

   pipeline->base.last_vgt_api_stage = MESA_SHADER_NONE;
   pipeline->lib_flags = lib_info ? lib_info->flags : 0;

   radv_pipeline_layout_init(device, &pipeline->layout, false);

   const VkPipelineBinaryInfoKHR *binary_info = vk_find_struct_const(pCreateInfo->pNext, PIPELINE_BINARY_INFO_KHR);
   const bool import_pipeline_binaries = binary_info && binary_info->binaryCount > 0;

   /* If we have libraries, import them first. */
   if (libs_info) {
      for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

         vk_graphics_pipeline_state_merge(state, &gfx_pipeline_lib->graphics_state);

         radv_graphics_pipeline_import_layout(&pipeline->layout, &gfx_pipeline_lib->layout);

         radv_graphics_pipeline_import_lib(device, &pipeline->base, gfx_pipeline_lib, import_pipeline_binaries);

         pipeline->lib_flags |= gfx_pipeline_lib->lib_flags;
      }
   }

   result = vk_graphics_pipeline_state_fill(&device->vk, state, pCreateInfo,
                                            NULL /* driver_mv */, NULL /* driver_rp */, 0, NULL, NULL,
                                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, &pipeline->state_data);
   if (result != VK_SUCCESS)
      return result;

   radv_pipeline_import_graphics_info(device, &pipeline->base, pCreateInfo);

   if (pipeline_layout)
      radv_graphics_pipeline_import_layout(&pipeline->layout, pipeline_layout);

   if (import_pipeline_binaries) {
      result = radv_graphics_pipeline_import_binaries(device, &pipeline->base, &pipeline->layout, binary_info);
      if (result == VK_SUCCESS && pipeline->base.bc250_plan.version) {
         struct radv_graphics_pipeline_state gfx_state;
         result = radv_generate_graphics_pipeline_state(device, pCreateInfo, &gfx_state);
         if (result == VK_SUCCESS) {
            uint8_t hash[32];
            radv_bc250_graphics_state_hash(device, pCreateInfo, &gfx_state, hash);
            if (memcmp(hash, pipeline->base.bc250_plan.state_hash, sizeof(hash)))
               result = VK_ERROR_FEATURE_NOT_PRESENT;
            radv_graphics_pipeline_state_finish(device, &gfx_state);
         }
      }
   } else {
      struct radv_graphics_pipeline_state gfx_state;

      result = radv_generate_graphics_pipeline_state(device, pCreateInfo, &gfx_state);
      if (result != VK_SUCCESS)
         return result;

      if (device->bc250_env.gpl_binary_link && pipeline->lib_flags == ALL_GRAPHICS_LIB_FLAGS &&
          (pipeline->base.active_stages & VK_SHADER_STAGE_MESH_BIT_EXT) && !libs_info) {
         /* The entire interface is present. Use the monolithic admission and
          * preparation path, then retain only the executables and their plan.
          * Partial libraries keep the conservative source fallback below. */
         VkPipelineCreateFlags2 flags = pipeline->base.base.create_flags;
         pipeline->base.base.type = RADV_PIPELINE_GRAPHICS;
         pipeline->base.base.create_flags &= ~(VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR |
            VK_PIPELINE_CREATE_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT);
         result = radv_graphics_pipeline_compile(&pipeline->base, pCreateInfo, &gfx_state, device, cache, false);
         pipeline->base.base.type = RADV_PIPELINE_GRAPHICS_LIB;
         pipeline->base.base.create_flags = flags;
      } else if (device->bc250_env.gpl_source_link) {
         VkPipelineShaderStageCreateInfo source[MESA_VULKAN_SHADER_STAGES];
         unsigned count = 0;
         if (libs_info) {
            for (unsigned i = 0; i < libs_info->libraryCount; i++) {
               VK_FROM_HANDLE(radv_pipeline, imported, libs_info->pLibraries[i]);
               struct radv_graphics_lib_pipeline *lib = radv_pipeline_to_graphics_lib(imported);
               if (!lib->bc250_source_only || count + lib->stage_count > ARRAY_SIZE(source)) {
                  result = VK_ERROR_FEATURE_NOT_PRESENT;
                  goto source_done;
               }
               if (lib->stage_count)
                  memcpy(source + count, lib->stages, lib->stage_count * sizeof(source[0]));
               count += lib->stage_count;
            }
         }
         if (count + pCreateInfo->stageCount > ARRAY_SIZE(source)) {
            result = VK_ERROR_FEATURE_NOT_PRESENT;
            goto source_done;
         }
         if (pCreateInfo->stageCount)
            memcpy(source + count, pCreateInfo->pStages, pCreateInfo->stageCount * sizeof(source[0]));
         count += pCreateInfo->stageCount;
         for (unsigned i = 0; i < count; i++) {
            mesa_shader_stage stage = vk_to_mesa_shader_stage(source[i].stage);
            if (!gfx_state.stages[stage].spirv.size || gfx_state.stages[stage].layout.mapping) {
               result = VK_ERROR_FEATURE_NOT_PRESENT;
               goto source_done;
            }
            pipeline->stage_keys[stage] = gfx_state.stages[stage].key;
         }
         pipeline->stages = radv_copy_shader_stage_create_info(device, count, source, pipeline->mem_ctx);
         if (!pipeline->stages) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto source_done;
         }
         pipeline->stage_count = count;
         pipeline->bc250_source_only = true;
         result = VK_SUCCESS;
      } else {
         result =
            radv_graphics_pipeline_compile(&pipeline->base, pCreateInfo, &gfx_state, device, cache, fast_linking_enabled);
      }

source_done:
      radv_graphics_pipeline_state_finish(device, &gfx_state);
   }

   return result;
}

static VkResult
radv_graphics_lib_pipeline_create(VkDevice _device, VkPipelineCache _cache,
                                  const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                  const VkAllocationCallbacks *pAllocator, VkPipeline *pPipeline)
{
   VK_FROM_HANDLE(vk_pipeline_cache, cache, _cache);
   VK_FROM_HANDLE(radv_device, device, _device);
   struct radv_graphics_lib_pipeline *pipeline;
   VkResult result;

   pipeline = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pipeline), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pipeline == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   radv_pipeline_init(device, &pipeline->base.base, RADV_PIPELINE_GRAPHICS_LIB);
   pipeline->base.base.create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);

   pipeline->mem_ctx = ralloc_context(NULL);

   result = radv_graphics_lib_pipeline_init(pipeline, device, cache, pCreateInfo);
   if (result != VK_SUCCESS) {
      radv_pipeline_destroy(device, &pipeline->base.base, pAllocator);
      return result;
   }

   radv_bc250_report_mesh_route(device, pipeline->base.base.shaders[MESA_SHADER_MESH],
      "library", 0, false, false, pipeline->base.bc250_plan.flags & RADV_BC250_PLAN_EMPTY);

   radv_pipeline_report_pso_history(device, &pipeline->base.base);

   *pPipeline = radv_pipeline_to_handle(&pipeline->base.base);

   return VK_SUCCESS;
}

void
radv_destroy_graphics_lib_pipeline(struct radv_device *device, struct radv_graphics_lib_pipeline *pipeline)
{
   struct radv_retained_shaders *retained_shaders = &pipeline->retained_shaders;

   radv_pipeline_layout_finish(device, &pipeline->layout);

   vk_free(&device->vk.alloc, pipeline->state_data);

   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      free(retained_shaders->stages[i].serialized_nir);
   }

   ralloc_free(pipeline->mem_ctx);

   radv_destroy_graphics_pipeline(device, &pipeline->base);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_CreateGraphicsPipelines(VkDevice _device, VkPipelineCache pipelineCache, uint32_t count,
                             const VkGraphicsPipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator,
                             VkPipeline *pPipelines)
{
   VkResult result = VK_SUCCESS;
   unsigned i = 0;

   VK_FROM_HANDLE(radv_device, device, _device);
   for (; i < count; i++) {
      /* The experimental producer is a separate compute pipeline and is not
       * serialized by the graphics pipeline binary format.
       */
      const VkPipelineBinaryInfoKHR *bc250_binaries =
         vk_find_struct_const(pCreateInfos[i].pNext, PIPELINE_BINARY_INFO_KHR);
      if (radv_device_physical(device)->bc250_hybrid_task && !device->bc250_env.pipeline_plan &&
          bc250_binaries && bc250_binaries->binaryCount) {
         pPipelines[i] = VK_NULL_HANDLE;
         result = VK_ERROR_FEATURE_NOT_PRESENT;
         if (vk_graphics_pipeline_create_flags(&pCreateInfos[i]) & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT)
            goto done_pipelines;
         goto next_pipeline;
      }

      if (radv_device_physical(device)->bc250_native_mesh &&
          !radv_device_physical(device)->bc250_hybrid_task &&
          !radv_device_physical(device)->bc250_native_task) {
         for (uint32_t j = 0; j < pCreateInfos[i].stageCount; j++) {
            if (pCreateInfos[i].pStages[j].stage == VK_SHADER_STAGE_TASK_BIT_EXT) {
               pPipelines[i] = VK_NULL_HANDLE;
               result = VK_ERROR_FEATURE_NOT_PRESENT;
               if (vk_graphics_pipeline_create_flags(&pCreateInfos[i]) & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT)
                  goto done_pipelines;
               goto next_pipeline;
            }
         }
      }
      const VkPipelineCreateFlagBits2 create_flags = vk_graphics_pipeline_create_flags(&pCreateInfos[i]);
      VkResult r;
      if (create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
         r = radv_graphics_lib_pipeline_create(_device, pipelineCache, &pCreateInfos[i], pAllocator, &pPipelines[i]);
      } else {
         r = radv_graphics_pipeline_create(_device, pipelineCache, &pCreateInfos[i], pAllocator, &pPipelines[i]);
      }
      if (r != VK_SUCCESS) {
         result = r;
         pPipelines[i] = VK_NULL_HANDLE;

         if (create_flags & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT)
            break;
      }

   next_pipeline:;
   }

done_pipelines:
   for (; i < count; ++i)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

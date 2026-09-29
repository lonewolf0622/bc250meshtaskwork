/*
 * Copyright © 2023 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "radv_nir.h"
#include "radv_pipeline.h"
#include "radv_shader.h"

#include "amdgfxregs.h"

typedef struct {
   bool dynamic_rasterization_samples;
   unsigned num_rasterization_samples;
   unsigned num_raster_vertices_per_prim;
   /* The hardware reports the per-quad provoking vertex (GFX10.3+
    * SPI_SHADER_PGM_RSRC1_PS.LOAD_PROVOKING_VTX). BC250 (GFX1013, GFX10.1) has
    * neither that nor PS_INPUT_CNTL.ROTATE_PC_PTR: the parameter-cache slot
    * holding API vertex 0 differs per primitive. The triangle case then emits
    * load_provoking_vtx_in_prim_amd as a placeholder, resolved after linking by
    * radv_nir_bc250_lower_bary_rotation.
    */
   bool has_provoking_vtx;
} lower_fs_barycentric_state;

static nir_def *
lower_interp_center_smooth(nir_builder *b, nir_def *offset)
{
   nir_def *pull_model = nir_load_barycentric_model(b, 32);

   nir_def *deriv_x =
      nir_vec3(b, nir_ddx_fine(b, nir_channel(b, pull_model, 0)), nir_ddx_fine(b, nir_channel(b, pull_model, 1)),
               nir_ddx_fine(b, nir_channel(b, pull_model, 2)));
   nir_def *deriv_y =
      nir_vec3(b, nir_ddy_fine(b, nir_channel(b, pull_model, 0)), nir_ddy_fine(b, nir_channel(b, pull_model, 1)),
               nir_ddy_fine(b, nir_channel(b, pull_model, 2)));

   nir_def *offset_x = nir_channel(b, offset, 0);
   nir_def *offset_y = nir_channel(b, offset, 1);

   nir_def *adjusted_x = nir_fadd(b, pull_model, nir_fmul(b, deriv_x, offset_x));
   nir_def *adjusted = nir_fadd(b, adjusted_x, nir_fmul(b, deriv_y, offset_y));

   nir_def *ij = nir_vec2(b, nir_channel(b, adjusted, 0), nir_channel(b, adjusted, 1));

   /* Get W by using the reciprocal of 1/W. */
   nir_def *w = nir_frcp(b, nir_channel(b, adjusted, 2));

   return nir_fmul(b, ij, w);
}

static nir_def *
lower_barycentric_coord_at_offset(nir_builder *b, nir_def *src, enum glsl_interp_mode mode)
{
   if (mode == INTERP_MODE_SMOOTH)
      return lower_interp_center_smooth(b, src);

   return nir_load_barycentric_at_offset(b, 32, src, .interp_mode = mode);
}

static nir_def *
lower_barycentric_coord_at_sample(nir_builder *b, lower_fs_barycentric_state *state, nir_intrinsic_instr *intrin)
{
   const enum glsl_interp_mode mode = (enum glsl_interp_mode)nir_intrinsic_interp_mode(intrin);
   nir_def *num_samples = nir_load_rasterization_samples_amd(b);
   nir_def *new_dest;

   if (state->dynamic_rasterization_samples) {
      nir_def *res1, *res2;

      nir_push_if(b, nir_ieq_imm(b, num_samples, 1));
      {
         res1 = nir_load_barycentric_pixel(b, 32, .interp_mode = nir_intrinsic_interp_mode(intrin));
      }
      nir_push_else(b, NULL);
      {
         nir_def *sample_pos = nir_load_sample_positions_amd(b, 32, intrin->src[0].ssa, num_samples);

         /* sample_pos -= 0.5 */
         sample_pos = nir_fadd_imm(b, sample_pos, -0.5f);

         res2 = lower_barycentric_coord_at_offset(b, sample_pos, mode);
      }
      nir_pop_if(b, NULL);

      new_dest = nir_if_phi(b, res1, res2);
   } else {
      if (!state->num_rasterization_samples) {
         new_dest = nir_load_barycentric_pixel(b, 32, .interp_mode = nir_intrinsic_interp_mode(intrin));
      } else {
         nir_def *sample_pos = nir_load_sample_positions_amd(b, 32, intrin->src[0].ssa, num_samples);

         /* sample_pos -= 0.5 */
         sample_pos = nir_fadd_imm(b, sample_pos, -0.5f);

         new_dest = lower_barycentric_coord_at_offset(b, sample_pos, mode);
      }
   }

   return new_dest;
}

static nir_def *
get_interp_param(nir_builder *b, lower_fs_barycentric_state *state, nir_intrinsic_instr *intrin)
{
   const enum glsl_interp_mode mode = (enum glsl_interp_mode)nir_intrinsic_interp_mode(intrin);

   if (intrin->intrinsic == nir_intrinsic_load_barycentric_coord_pixel) {
      return nir_load_barycentric_pixel(b, 32, .interp_mode = mode);
   } else if (intrin->intrinsic == nir_intrinsic_load_barycentric_coord_at_offset) {
      return lower_barycentric_coord_at_offset(b, intrin->src[0].ssa, mode);
   } else if (intrin->intrinsic == nir_intrinsic_load_barycentric_coord_at_sample) {
      return lower_barycentric_coord_at_sample(b, state, intrin);
   } else if (intrin->intrinsic == nir_intrinsic_load_barycentric_coord_centroid) {
      return nir_load_barycentric_centroid(b, 32, .interp_mode = mode);
   } else {
      assert(intrin->intrinsic == nir_intrinsic_load_barycentric_coord_sample);
      return nir_load_barycentric_sample(b, 32, .interp_mode = mode);
   }

   return NULL;
}

static nir_def *
lower_point(nir_builder *b)
{
   nir_def *coords[3];

   coords[0] = nir_imm_float(b, 1.0f);
   coords[1] = nir_imm_float(b, 0.0f);
   coords[2] = nir_imm_float(b, 0.0f);

   return nir_vec(b, coords, 3);
}

static nir_def *
lower_line(nir_builder *b, nir_def *p1, nir_def *p2)
{
   nir_def *coords[3];

   coords[1] = nir_fadd(b, p1, p2);
   coords[0] = nir_fsub_imm(b, 1.0f, coords[1]);
   coords[2] = nir_imm_float(b, 0.0f);

   return nir_vec(b, coords, 3);
}

static nir_def *
lower_triangle(nir_builder *b, const lower_fs_barycentric_state *state, nir_def *p1, nir_def *p2)
{
   nir_def *v0_bary[3], *v1_bary[3], *v2_bary[3];
   nir_def *coords[3];

   /* Compute the provoking vertex ID:
    *
    * quad_id = thread_id >> 2
    * provoking_vtx_id = (provoking_vtx >> (quad_id << 1)) & 3
    */
   nir_def *provoking_vtx_id;
   if (state->has_provoking_vtx) {
      nir_def *quad_id = nir_ushr_imm(b, nir_load_subgroup_invocation(b), 2);
      nir_def *provoking_vtx = nir_load_provoking_vtx_amd(b);
      provoking_vtx_id = nir_ubfe(b, provoking_vtx, nir_ishl_imm(b, quad_id, 1), nir_imm_int(b, 2));
   } else {
      provoking_vtx_id = nir_load_provoking_vtx_in_prim_amd(b);
   }

   /* Compute barycentrics. */
   v0_bary[0] = nir_fsub(b, nir_fsub_imm(b, 1.0f, p2), p1);
   v0_bary[1] = p1;
   v0_bary[2] = p2;

   v1_bary[0] = p1;
   v1_bary[1] = p2;
   v1_bary[2] = nir_fsub(b, nir_fsub_imm(b, 1.0f, p2), p1);

   v2_bary[0] = p2;
   v2_bary[1] = nir_fsub(b, nir_fsub_imm(b, 1.0f, p2), p1);
   v2_bary[2] = p1;

   /* Select barycentrics for the given provoking vertex ID. */
   for (unsigned i = 0; i < 3; i++) {
      coords[i] = nir_bcsel(b, nir_ieq_imm(b, provoking_vtx_id, 2), v2_bary[i],
                            nir_bcsel(b, nir_ieq_imm(b, provoking_vtx_id, 1), v1_bary[i], v0_bary[i]));
   }

   return nir_vec(b, coords, 3);
}

static bool
lower_load_barycentric_coord(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   lower_fs_barycentric_state *state = data;
   if (intrin->intrinsic != nir_intrinsic_load_barycentric_coord_pixel &&
       intrin->intrinsic != nir_intrinsic_load_barycentric_coord_centroid &&
       intrin->intrinsic != nir_intrinsic_load_barycentric_coord_sample &&
       intrin->intrinsic != nir_intrinsic_load_barycentric_coord_at_offset &&
       intrin->intrinsic != nir_intrinsic_load_barycentric_coord_at_sample)
      return false;

   nir_def *interp, *p1, *p2;
   nir_def *new_dest;

   b->cursor = nir_after_instr(&intrin->instr);

   /* When the rasterization primitive isn't known at compile time (GPL), load it. */
   if (!state->num_raster_vertices_per_prim) {
      nir_def *vgt_outprim_type = nir_load_rasterization_primitive_amd(b);
      nir_def *res1, *res2;

      nir_def *is_point = nir_ieq_imm(b, vgt_outprim_type, V_028A6C_POINTLIST);
      nir_if *if_point = nir_push_if(b, is_point);
      {
         res1 = lower_point(b);
      }
      nir_push_else(b, if_point);
      {
         nir_def *res_line, *res_triangle;

         interp = get_interp_param(b, state, intrin);
         p1 = nir_channel(b, interp, 0);
         p2 = nir_channel(b, interp, 1);

         nir_def *is_line = nir_ieq_imm(b, vgt_outprim_type, V_028A6C_LINESTRIP);
         nir_if *if_line = nir_push_if(b, is_line);
         {
            res_line = lower_line(b, p1, p2);
         }
         nir_push_else(b, if_line);
         {
            res_triangle = lower_triangle(b, state, p1, p2);
         }
         nir_pop_if(b, if_line);

         res2 = nir_if_phi(b, res_line, res_triangle);
      }
      nir_pop_if(b, if_point);

      new_dest = nir_if_phi(b, res1, res2);
   } else {
      if (state->num_raster_vertices_per_prim == 1) {
         new_dest = lower_point(b);
      } else {
         interp = get_interp_param(b, state, intrin);
         p1 = nir_channel(b, interp, 0);
         p2 = nir_channel(b, interp, 1);

         if (state->num_raster_vertices_per_prim == 2) {
            new_dest = lower_line(b, p1, p2);
         } else {
            assert(state->num_raster_vertices_per_prim == 3);
            new_dest = lower_triangle(b, state, p1, p2);
         }
      }
   }

   nir_def_replace(&intrin->def, new_dest);

   return true;
}

bool
radv_nir_lower_fs_barycentric(nir_shader *shader, const struct radv_graphics_state_key *gfx_state,
                              unsigned num_raster_vertices_per_prim, bool has_provoking_vtx)
{
   lower_fs_barycentric_state state = {
      .has_provoking_vtx = has_provoking_vtx,
      .dynamic_rasterization_samples = gfx_state->dynamic_rasterization_samples,
      .num_rasterization_samples = gfx_state->ms.rasterization_samples,
      .num_raster_vertices_per_prim = num_raster_vertices_per_prim,
   };

   return nir_shader_intrinsics_pass(shader, lower_load_barycentric_coord, nir_metadata_none, &state);
}

/* BC250 (GFX1013, GFX10.1) barycentric vertex order.
 *
 * Without ROTATE_PC_PTR/LOAD_PROVOKING_VTX the parameter cache holds each
 * triangle's vertices in a per-primitive cyclic rotation of the API order, and
 * the hardware I/J are relative to that rotated order. Measured on GFX1013
 * (2026-09-23): a triangle list's (v0,v1,v2) arrived as (v0,v1,v2) and
 * (v3,v4,v5) as (v4,v5,v3) in both provoking modes, while a flat read returned
 * the provoking vertex (v0/v3 first, v2/v5 last).
 *
 * The rotation is recovered per pixel: the last pre-rasterization stage also
 * exports its clip-space position to two generic slots; the FS reads one of
 * them per vertex (passthrough, raw slot order) and the other flat (provoking
 * vertex). The raw slot matching the flat value holds the provoking vertex,
 * which is API vertex 0 (first-vertex convention) or 2 (last-vertex
 * convention). Two vertices of a triangle that produces fragments cannot have
 * the same clip-space x, y and w, so the match is exact.
 */

static bool
bc250_export_bary_ref_instr(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   const unsigned *slots = data;

   if (intrin->intrinsic != nir_intrinsic_store_output && intrin->intrinsic != nir_intrinsic_store_per_vertex_output)
      return false;

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intrin);
   if (sem.location != VARYING_SLOT_POS)
      return false;

   b->cursor = nir_after_instr(&intrin->instr);

   /* One slot (RADV_BC250_BARY_CORNER_ID): raw_slot == flat_slot. */
   for (unsigned i = 0; i < (slots[0] == slots[1] ? 1 : 2); i++) {
      nir_intrinsic_instr *copy = nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intrin->instr));
      nir_io_semantics copy_sem = sem;

      copy_sem.location = slots[i];
      copy_sem.num_slots = 1;
      copy_sem.no_varying = 0;
      copy_sem.no_sysval_output = 1;
      nir_intrinsic_set_io_semantics(copy, copy_sem);

      if (nir_intrinsic_has_io_xfb(copy))
         nir_intrinsic_set_io_xfb(copy, (nir_io_xfb){0});

      nir_builder_instr_insert(b, &copy->instr);
   }

   return true;
}

/* Copy every position store of the last pre-rasterization stage to the two
 * generic reference slots. Runs on lowered, linked I/O. Mesh shaders keep the
 * copies only for the output info: ac_nir_lower_ngg_mesh removes them and
 * exports the position it already loaded for the vertex
 * (ac_nir_lower_ngg_options::bc250_bary_ref_mask), so they cost no LDS.
 */
bool
radv_nir_bc250_export_bary_ref(nir_shader *producer, unsigned raw_slot, unsigned flat_slot)
{
   unsigned slots[2] = {raw_slot, flat_slot};

   return nir_shader_intrinsics_pass(producer, bc250_export_bary_ref_instr, nir_metadata_control_flow, slots);
}

static bool
bc250_is_strict_vertex_load(const nir_intrinsic_instr *intrin)
{
   return intrin->intrinsic == nir_intrinsic_load_input_vertex &&
          nir_intrinsic_io_semantics(intrin).interp_explicit_strict;
}

/* Whether the FS reads barycentric coordinates or strict per-vertex inputs
 * (PerVertexKHR), the two things that depend on the API vertex order.
 * AMD_shader_explicit_vertex_parameter loads are not strict and keep the raw
 * order, as on the base driver's GFX10.
 */
bool
radv_nir_bc250_fs_needs_bary_rotation(const nir_shader *fs)
{
   nir_foreach_function_impl (impl, fs) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic == nir_intrinsic_load_provoking_vtx_in_prim_amd ||
                bc250_is_strict_vertex_load(intrin))
               return true;
         }
      }
   }

   return false;
}

/* The parameter-cache slot (0..2) holding API vertex 0 of the pixel's
 * triangle; 0 for points and lines, and when the producer is unknown.
 */
static nir_def *
bc250_build_rotation(nir_builder *b, int raw_slot, int flat_slot, unsigned num_raster_vertices_per_prim,
                     bool dynamic_provoking, bool provoking_last)
{
   if (raw_slot < 0 || (num_raster_vertices_per_prim && num_raster_vertices_per_prim != 3))
      return nir_imm_int(b, 0);

   const nir_io_semantics raw_sem = {.location = raw_slot, .num_slots = 1};
   if (flat_slot < 0) {
      /* RADV_BC250_BARY_CORNER_ID: component 0 of raw slot k is the private corner number of
       * the vertex there; API vertex 0 is the slot holding 0. Independent of the provoking mode. */
      nir_def *cid[3];
      for (unsigned i = 1; i < 3; i++)
         cid[i] = nir_load_input_vertex(b, 1, 32, nir_imm_int(b, i), nir_imm_int(b, 0), .base = 0, .component = 0,
                                        .dest_type = nir_type_uint32, .io_semantics = raw_sem);
      nir_def *slot = nir_bcsel(b, nir_ieq_imm(b, cid[1], 0), nir_imm_int(b, 1),
                                nir_bcsel(b, nir_ieq_imm(b, cid[2], 0), nir_imm_int(b, 2), nir_imm_int(b, 0)));
      if (!num_raster_vertices_per_prim) {
         nir_def *is_triangle = nir_ieq_imm(b, nir_load_rasterization_primitive_amd(b), V_028A6C_TRISTRIP);
         slot = nir_bcsel(b, is_triangle, slot, nir_imm_int(b, 0));
      }
      return slot;
   }
   const nir_io_semantics flat_sem = {.location = flat_slot, .num_slots = 1};
   /* Clip-space x, y and w identify a vertex of a triangle that produces
    * fragments: two vertices with equal x, y and w project to the same screen
    * point (for any z), so the triangle would be degenerate. z is not compared.
    */
   const unsigned xyw = 0xb;
   nir_def *raw[3];

   /* Slot 0 needs no load: it is the provoking vertex when slots 1 and 2 are not. */
   for (unsigned i = 1; i < 3; i++) {
      raw[i] = nir_channels(b,
                            nir_load_input_vertex(b, 4, 32, nir_imm_int(b, i), nir_imm_int(b, 0), .base = 0,
                                                  .component = 0, .dest_type = nir_type_uint32, .io_semantics = raw_sem),
                            xyw);
   }

   nir_def *provoking =
      nir_channels(b,
                   nir_load_input(b, 4, 32, nir_imm_int(b, 0), .base = 0, .range = 1, .component = 0,
                                  .dest_type = nir_type_uint32, .io_semantics = flat_sem),
                   xyw);

   /* Raw slot of the provoking vertex. */
   nir_def *slot = nir_bcsel(b, nir_ball_iequal(b, raw[1], provoking), nir_imm_int(b, 1),
                             nir_bcsel(b, nir_ball_iequal(b, raw[2], provoking), nir_imm_int(b, 2), nir_imm_int(b, 0)));

   /* The provoking vertex is API vertex 2 with the last-vertex convention;
    * API vertex 0 is then the next slot of the cyclic order.
    */
   if (dynamic_provoking || provoking_last) {
      nir_def *slot_last = nir_bcsel(b, nir_ieq_imm(b, slot, 2), nir_imm_int(b, 0), nir_iadd_imm(b, slot, 1));
      slot = dynamic_provoking ? nir_bcsel(b, nir_ine_imm(b, nir_load_provoking_last(b), 0), slot_last, slot)
                               : slot_last;
   }

   if (!num_raster_vertices_per_prim) {
      nir_def *is_triangle = nir_ieq_imm(b, nir_load_rasterization_primitive_amd(b), V_028A6C_TRISTRIP);
      slot = nir_bcsel(b, is_triangle, slot, nir_imm_int(b, 0));
   }

   return slot;
}

/* Strict per-vertex input k lives in raw slot (k + rotation) % 3. ACO needs a
 * constant vertex index, so load all three and select.
 */
static void
bc250_rotate_vertex_load(nir_builder *b, nir_intrinsic_instr *intrin, nir_def *rotation)
{
   if (!nir_src_is_const(intrin->src[0]))
      return;

   const unsigned k = nir_src_as_uint(intrin->src[0]);
   if (k > 2)
      return;

   b->cursor = nir_after_instr(&intrin->instr);

   nir_def *loads[3];
   loads[k] = &intrin->def;
   for (unsigned i = 0; i < 3; i++) {
      if (i == k)
         continue;

      nir_def *vertex = nir_imm_int(b, i);
      nir_intrinsic_instr *copy = nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intrin->instr));
      nir_builder_instr_insert(b, &copy->instr);
      nir_src_rewrite(&copy->src[0], vertex);
      loads[i] = &copy->def;
   }

   nir_def *res = nir_bcsel(b, nir_ieq_imm(b, rotation, 1), loads[(k + 1) % 3],
                            nir_bcsel(b, nir_ieq_imm(b, rotation, 2), loads[(k + 2) % 3], loads[k]));
   nir_def_rewrite_uses_after(&intrin->def, res);
}

/* Resolve the rotation placeholders of radv_nir_lower_fs_barycentric and
 * reorder strict per-vertex inputs. raw_slot < 0 means that the producer is
 * not known (fragment shader library, unlinked shader object) or has no two
 * free parameters: the rotation is then 0, which is exact only for the
 * primitives the hardware does not rotate.
 */
bool
radv_nir_bc250_lower_bary_rotation(nir_shader *fs, int raw_slot, int flat_slot, unsigned num_raster_vertices_per_prim,
                                   bool dynamic_provoking, bool provoking_last)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *rotation =
      bc250_build_rotation(&b, raw_slot, flat_slot, num_raster_vertices_per_prim, dynamic_provoking, provoking_last);

   nir_foreach_block_safe (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic == nir_intrinsic_load_provoking_vtx_in_prim_amd) {
            nir_def_replace(&intrin->def, rotation);
         } else if (bc250_is_strict_vertex_load(intrin)) {
            bc250_rotate_vertex_load(&b, intrin, rotation);
         }
      }
   }

   return nir_progress(true, impl, nir_metadata_none);
}

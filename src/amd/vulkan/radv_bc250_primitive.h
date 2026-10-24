/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_PRIMITIVE_H
#define RADV_BC250_PRIMITIVE_H

#include "nir/nir_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A logical primitive, before an exporter chooses physical vertex slots.
 * indices keep API corner order. culled is an optional boolean; attributes
 * selects the original per-primitive output record, independently of any
 * compaction or private provoking vertex. This view introduces no storage.
 */
struct radv_bc250_primitive {
   nir_def *indices;
   nir_def *culled;
   nir_def *attributes;
};

static inline struct radv_bc250_primitive
radv_bc250_primitive_make(nir_def *indices, nir_def *culled, nir_def *attributes)
{
   assert(!indices || (indices->bit_size == 32 && indices->num_components <= 3));
   assert(!culled || (culled->bit_size == 1 && culled->num_components == 1));
   assert(!attributes || (attributes->bit_size == 32 && attributes->num_components == 1));
   struct radv_bc250_primitive primitive = {indices, culled, attributes};
   return primitive;
}

static inline nir_def *
radv_bc250_primitive_keep(nir_builder *b, struct radv_bc250_primitive primitive, nir_def *allocated)
{
   return primitive.culled ? nir_iand(b, allocated, nir_inot(b, primitive.culled)) : allocated;
}

static inline nir_def *
radv_bc250_primitive_survives(nir_builder *b, struct radv_bc250_primitive primitive)
{
   return primitive.culled ? nir_ieq_imm(b, primitive.culled, 0) : nir_imm_true(b);
}

static inline nir_def *
radv_bc250_primitive_corner(nir_builder *b, struct radv_bc250_primitive primitive, unsigned corner)
{
   assert(primitive.indices && corner < primitive.indices->num_components);
   return nir_channel(b, primitive.indices, corner);
}

static inline nir_def *
radv_bc250_primitive_local_corner(nir_builder *b, struct radv_bc250_primitive primitive, unsigned corner,
                                  nir_def *instance, unsigned vertices)
{
   assert(vertices);
   return nir_umin(b,
                   nir_isub(b, radv_bc250_primitive_corner(b, primitive, corner), nir_imul_imm(b, instance, vertices)),
                   nir_imm_int(b, vertices - 1));
}

/* Both vector staging and the packed 16-bit plane represent the same logical
 * connectivity. Preserve selected-corner loads for the direct-read exporter.
 */
static inline struct radv_bc250_primitive
radv_bc250_primitive_load(nir_builder *b, nir_variable *indices, nir_variable *narrow, unsigned corners,
                          nir_def *primitive)
{
   nir_def *tuple;
   if (!narrow) {
      tuple = nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, indices), primitive));
   } else {
      nir_def *comps[3];
      assert(corners && corners <= ARRAY_SIZE(comps));
      nir_def *first = nir_imul_imm(b, primitive, corners);
      for (unsigned c = 0; c < corners; c++)
         comps[c] = nir_u2u32(
            b, nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, narrow), nir_iadd_imm(b, first, c))));
      tuple = nir_vec(b, comps, corners);
   }
   return radv_bc250_primitive_make(tuple, NULL, primitive);
}

static inline nir_def *
radv_bc250_primitive_load_corner(nir_builder *b, nir_variable *indices, nir_variable *narrow, unsigned corners,
                                 nir_def *primitive, nir_def *corner, bool selected_load)
{
   if (corners == 1 || !selected_load) {
      struct radv_bc250_primitive p = radv_bc250_primitive_load(b, indices, narrow, corners, primitive);
      return corners == 1 ? p.indices : nir_vector_extract(b, p.indices, corner);
   }
   if (narrow)
      return nir_u2u32(
         b, nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, narrow),
                                                    nir_iadd(b, nir_imul_imm(b, primitive, corners), corner))));
   return nir_load_deref(
      b, nir_build_deref_array(b, nir_build_deref_array(b, nir_build_deref_var(b, indices), primitive), corner));
}

/* Indexed exporters give a primitive's flat attributes a distinct provoking
 * vertex while keeping its original logical vertex recoverable.
 */
static inline unsigned
radv_bc250_primitive_vertex_stride(unsigned vertices, unsigned primitives, bool attributes)
{
   return vertices * (attributes ? primitives + 1 : 1);
}

static inline nir_def *
radv_bc250_primitive_encode(nir_builder *b, struct radv_bc250_primitive primitive, unsigned vertices,
                            unsigned provoking)
{
   assert(primitive.attributes && primitive.indices->num_components == 3 && provoking < 3);
   nir_def *corner[3];
   for (unsigned j = 0; j < 3; j++) {
      corner[j] = radv_bc250_primitive_corner(b, primitive, j);
      if (j == provoking)
         corner[j] = nir_iadd(b, corner[j], nir_imul_imm(b, nir_iadd_imm(b, primitive.attributes, 1), vertices));
   }
   return nir_vec3(b, corner[0], corner[1], corner[2]);
}

static inline nir_def *
radv_bc250_primitive_decode_attribute(nir_builder *b, nir_def *slot, unsigned vertices, bool default_zero)
{
   /* Replay VS needs the -1 sentinel for ordinary vertices. Fetch-only VS
    * reads record zero there; only the provoking corner contributes flat data.
    */
   if (!default_zero)
      return nir_iadd_imm(b, nir_udiv_imm(b, slot, vertices), -1);
   return nir_isub(b, nir_umax_imm(b, nir_udiv_imm(b, slot, vertices), 1), nir_imm_int(b, 1));
}

#ifdef __cplusplus
}
#endif

#endif

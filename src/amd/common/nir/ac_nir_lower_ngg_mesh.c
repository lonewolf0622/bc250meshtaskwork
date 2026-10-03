/*
 * Copyright © 2021 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "ac_nir.h"
#include "ac_nir_helpers.h"
#include "ac_gpu_info.h"

#include "nir_builder.h"
#include "util/u_debug.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define SPECIAL_MS_OUT_MASK \
   (VARYING_BIT_PRIMITIVE_COUNT | \
    VARYING_BIT_PRIMITIVE_INDICES | \
    VARYING_BIT_CULL_PRIMITIVE)

#define MS_PRIM_ARG_EXP_MASK \
   (VARYING_BIT_LAYER | \
    VARYING_BIT_VIEWPORT | \
    VARYING_BIT_PRIMITIVE_SHADING_RATE)

#define MS_VERT_ARG_EXP_MASK \
   (VARYING_BIT_CULL_DIST0 | \
    VARYING_BIT_CULL_DIST1 | \
    VARYING_BIT_CLIP_DIST0 | \
    VARYING_BIT_CLIP_DIST1 | \
    VARYING_BIT_PSIZ)

/* LDS layout of Mesh Shader workgroup info. */
enum {
   /* DW0: number of primitives */
   lds_ms_num_prims = 0,
   /* DW1: number of vertices */
   lds_ms_num_vtx = 4,
   /* DW2: workgroup index within the current dispatch */
   lds_ms_wg_index = 8,
   /* DW3: number of API workgroups in flight */
   lds_ms_num_api_waves = 12,
};

/* Potential location for Mesh Shader outputs. */
typedef enum {
   ms_out_mode_lds,
   ms_out_mode_scratch_ring,
   ms_out_mode_attr_ring,
   ms_out_mode_var,
} ms_out_mode;

typedef struct
{
   uint64_t mask; /* Mask of output locations */
   uint32_t addr; /* Base address */
} ms_out_part;

typedef struct
{
   /* Mesh shader LDS layout. For details, see ms_calculate_output_layout. */
   struct {
      uint32_t workgroup_info_addr;
      ms_out_part vtx_attr;
      /* RADV_BC250_MESH_COMPACT_LDS: packed per-vertex records. A vertex record
       * holds vtx_comps[location] dwords for every location of vtx_attr.mask, in
       * location order, instead of one 16-byte slot per location. */
      bool vtx_packed;
      uint8_t vtx_comps[VARYING_SLOT_MAX];
      ms_out_part prm_attr;
      uint32_t indices_addr;
      uint32_t cull_flags_addr;
      uint32_t total_size;
   } lds;

   /* VRAM "mesh shader scratch ring" layout for outputs that don't fit into the LDS.
    * Not to be confused with scratch memory.
    */
   struct {
      ms_out_part vtx_attr;
      ms_out_part prm_attr;
   } scratch_ring;

   /* VRAM attributes ring (supported GPUs only) for all non-position outputs.
    * We don't have to reload attributes from this ring at the end of the shader.
    */
   struct {
      ms_out_part vtx_attr;
      ms_out_part prm_attr;
   } attr_ring;

   /* Outputs without cross-invocation access can be stored in variables. */
   struct {
      ms_out_part vtx_attr;
      ms_out_part prm_attr;
   } var;
} ms_out_mem_layout;

typedef struct
{
   const ac_nir_lower_ngg_options *options;
   const struct ac_compiler_info *ac;
   bool vert_multirow_export;
   bool prim_multirow_export;

   ms_out_mem_layout layout;
   uint64_t per_vertex_outputs;
   uint64_t per_primitive_outputs;
   unsigned vertices_per_prim;

   unsigned wave_size;
   unsigned api_workgroup_size;
   unsigned hw_workgroup_size;

   nir_def *workgroup_index;
   nir_variable *out_variables[VARYING_SLOT_MAX * 4];
   nir_variable *primitive_count_var;
   nir_variable *vertex_count_var;

   ac_nir_prerast_out out;

   /* True if the shader has waves that don't execute the API shader at all. */
   bool has_non_api_waves;
   /* True if the lowering needs to insert the layer output. */
   bool insert_layer_output;
   /* True if cull flags are used */
   bool uses_cull_flags;
   /* Experimental, opt-in, single-wave survivor mapping. */
   bool compact_cull;
   nir_def *compact_source_index;
   bool pack_triangle_vertices;
   nir_def *packed_vertex_source_prim;
   nir_def *original_vertex_count;
   /* RADV_BC250_MESH_AUTOCULL (implies compact_cull and pack_triangle_vertices):
    * the epilogue culls triangles itself, see ms_autocull_compact. */
   bool autocull;
   /* compact_cull / pack_triangle_vertices of the switch-off lowering: used by the
    * epilogue side of the runtime branch that does not cull (emit_ms_finale). */
   bool plain_compact_cull;
   bool plain_pack_triangle_vertices;
   bool autocull_skip_viewport_state;
   uint32_t autocull_lds_addr;
   /* RADV_BC250_MESH_CULLDIST_CULL: the cull distances are culled in the shader and not
    * exported (ac_nir_lower_ngg_options::bc250_autocull_culldist). */
   bool dont_export_cull;
   /* The position vectors every exported vertex writes (SPI_SHADER_POS_FORMAT), also
    * written by the GFX10 fully-culled workaround's dummy vertex (ms_num_pos_exports). */
   unsigned num_pos_exports;
   /* True if the output vertex and primitive counts are workgroup-uniform. */
   bool output_counts_workgroup_uniform;

   const uint8_t *vs_output_param_offset;
   bool has_param_exports;

   /* RADV_BC250_MESH_DIRECT_READ (ms_direct_read_analyze), NULL when off. */
   struct ms_direct_read *direct;

   /* RADV_BC250_MESH_IMPLICIT_TRIS: the primitive indices are implicit, see
    * ms_implicit_index. Their stores are dropped and they get no LDS area. */
   bool implicit_indices;

   /* BC250 barycentrics (ac_nir_lower_ngg_options::bc250_bary_ref_mask): generic
    * slots exported with the vertex position, see ms_bary_ref_remove_stores. */
   uint64_t bary_ref_mask;

   /* RADV_BC250_MESH_COMPACT (ms_compact_vertices). */
   bool compact;
   uint32_t compact_lds_addr;
   unsigned compact_table; /* entries of the first-use table = declared logical vertices */
   /* Set by ms_compact_vertices for the export: vertex and primitive lanes read
    * their expanded source vertex and their corner slots from the LDS maps. */
   bool compact_active;
   /* Every lane holds the final output counts (no LDS distribution needed). */
   bool counts_in_all_lanes;
   bool safe_direct;
   nir_def *fast_clean;
   /* RADV_BC250_MESH_SAFE_COMPACT: enabled for this shader (LDS permitting), and the per-workgroup
    * flag (published flag bit 1): referenced vertices renumbered in order, survivors from indices. */
   bool safe_compact;
   nir_def *fast_compact;
   /* RADV_BC250_MESH_PP_SHARE: shared-vertex plan for the owned private-corner route. */
   bool pp_share;
   nir_variable *fast_prim_arg;
   uint32_t safe_direct_map_addr;
   uint32_t safe_direct_latest_addr;
   uint32_t safe_direct_indices_addr;
   uint32_t safe_direct_counts_addr;
   nir_def *safe_query_prims;
} lower_ngg_ms_state;

/* RADV_BC250_MESH_IMPLICIT_TRIS.
 *
 * The base driver's expanded Mesh shaders (radv_bc250_expand_primitive_attributes
 * without its compact vertex map) give every primitive private consecutive
 * vertices: the terminal expansion loop writes, for every primitive p below the
 * primitive count, PrimitiveIndices[p] = (N p, N p + 1, .., N p + N - 1) with N
 * the vertices per primitive, and the vertex count is N x the primitive count.
 * Nothing else writes or reads that output. So corner c of primitive p is
 * N p + c, a function of the primitive index the reader already has: the NGG
 * lowering does not need to store the indices to LDS in the loop and reload them
 * for the primitive export (and, with RADV_BC250_MESH_AUTOCULL, for the culling
 * test and the packed triangle vertices).
 *
 * Every reader of an index only reads the indices of an existing primitive
 * (p < primitive count: the export lanes, the compacted survivors' source
 * primitives, the autocull test lanes), whose tuple the loop stored, so the
 * derived value is the value the reader loaded before (8-bit storage cannot
 * truncate it: N x max primitives <= 256). The readers keep their clamps against
 * the vertex count. RADV only requests this for shaders its expansion produced
 * (bc250_implicit_tris, keyed); the lowering checks the shape it can see
 * (ms_implicit_indices_ok) and otherwise keeps the stored indices. */
static nir_def *
ms_implicit_index(nir_builder *b, nir_def *prim, unsigned corner, const lower_ngg_ms_state *s)
{
   return nir_iadd_imm(b, nir_imul_imm(b, prim, s->vertices_per_prim), corner);
}

/* RADV_BC250_MESH_DIRECT_READ.
 *
 * The expanded Mesh shaders (radv_bc250_expand_primitive_attributes) end with
 * one loop that copies the application's outputs, staged in LDS, to the private
 * expanded vertices:
 *
 *    for (v = local_invocation_index; v < 3 * primitives; v += api_workgroup_size)
 *       out[location][v] = f_location(v)        (store_per_vertex_output, index v)
 *
 * where f_location(v) reads the staging of the vertex (or primitive) the
 * expanded vertex v copies, through the staged primitive indices. The NGG
 * lowering then keeps every out[location][v] in LDS (16 bytes per location per
 * expanded vertex, the expansion writes them across invocations) and the
 * epilogue reloads out[location][e] for the vertex e its lane exports.
 *
 * A location is exported directly when its value is such a pure function of v:
 *  - all its stores are store_per_vertex_output in one block at the top level
 *    of that terminal loop's body (run once per iteration that passed the bound
 *    check v >= vertex count of SetMeshOutputs), one store per component, with
 *    arrayed index v (the loop's only phi, starting at the local invocation
 *    index and stepping by the API workgroup size: every v is written by at
 *    most one lane and iteration), a zero slot offset and 32-bit data;
 *  - the stored values only depend on v, constants, ALU and non-volatile
 *    load_shared instructions that are inside the loop;
 *  - the loop has no other side effect: no LDS or memory stores, atomics or
 *    barriers, only one break (the bound check) and nothing runs after it;
 *  - no other store or load of that location exists, it is not accessed
 *    indirectly, per-primitive or read back, and it is Position, a clip/cull
 *    distance slot (CLIP_DIST0/1: the merged ClipDistance/CullDistance array,
 *    one component per element) or VARn.
 * LDS is never written after the loop's loads: the loop and the rest of the
 * shader write no LDS, and the lowering's own stores before the epilogue go to
 * its areas above the API shared memory. Recomputing f_location(e) in the epilogue
 * (after the finale barrier, which follows every loop iteration of every lane) therefore
 * gives the value the loop stored for e. When the loop did not store e the
 * expanded record was never written, so its value was undefined anyway.
 *
 * Direct locations get no LDS record and no store. Their output metadata
 * (update_ms_output_info) comes from the removed stores, so exports, parameter
 * offsets and masks are unchanged. Every other location keeps its layout (per
 * location fallback). */
typedef struct ms_direct_read {
   uint64_t mask;
   nir_def *vertex;
   nir_loop *loop;
   /* The stored value of every component of a direct location. */
   nir_scalar value[VARYING_SLOT_MAX][4];
   /* Clones of the value computations for the vertex index remap_index. */
   struct hash_table *remap;
   nir_def *remap_index;
   nir_block *remap_block;
   unsigned stores;
} ms_direct_read;

static void
ms_store_prim_indices(nir_builder *b,
                      nir_intrinsic_instr *intrin,
                      lower_ngg_ms_state *s)
{
   /* EXT_mesh_shader primitive indices: array of vectors.
    * They don't count as per-primitive outputs, but the array is indexed
    * by the primitive index, so they are practically per-primitive.
    */
   assert(nir_src_is_const(*nir_get_io_offset_src(intrin)));
   assert(nir_src_as_uint(*nir_get_io_offset_src(intrin)) == 0);

   /* RADV_BC250_MESH_IMPLICIT_TRIS: every reader derives the value (ms_implicit_index). */
   if (s->implicit_indices)
      return;

   const unsigned write_mask = nir_intrinsic_write_mask(intrin);
   const unsigned component_offset = nir_intrinsic_component(intrin);
   nir_def *store_val = intrin->src[0].ssa;
   assert(store_val->num_components <= 3);
   assert(write_mask && write_mask <= BITFIELD_MASK(s->vertices_per_prim));

   if (store_val->num_components > s->vertices_per_prim)
      store_val = nir_trim_vector(b, store_val, s->vertices_per_prim);

   if (s->layout.var.prm_attr.mask & VARYING_BIT_PRIMITIVE_INDICES) {
      for (unsigned c = 0; c < store_val->num_components; ++c) {
         if (!(write_mask & BITFIELD_BIT(c)))
            continue;

         const unsigned i = VARYING_SLOT_PRIMITIVE_INDICES * 4 + c + component_offset;
         nir_store_var(b, s->out_variables[i], nir_channel(b, store_val, c), 0x1);
      }
      return;
   }

   nir_def *arr_index = nir_get_io_arrayed_index_src(intrin)->ssa;
   nir_def *offset = nir_imul_imm(b, arr_index, s->vertices_per_prim);

   /* The max vertex count is 256, so these indices always fit 8 bits.
    * To reduce LDS use, store these as a flat array of 8-bit values.
    */
   nir_store_shared(b, nir_u2u8(b, store_val), offset,
                    .base = s->layout.lds.indices_addr + component_offset,
                    .write_mask = write_mask);
}

static void
ms_store_cull_flag(nir_builder *b,
                   nir_intrinsic_instr *intrin,
                   lower_ngg_ms_state *s)
{
   /* EXT_mesh_shader cull primitive: per-primitive bool. */
   assert(nir_src_is_const(*nir_get_io_offset_src(intrin)));
   assert(nir_src_as_uint(*nir_get_io_offset_src(intrin)) == 0);
   assert(nir_intrinsic_component(intrin) == 0);
   assert(nir_intrinsic_write_mask(intrin) == 1);

   nir_def *store_val = nir_b2b1(b, intrin->src[0].ssa);

   assert(store_val->num_components == 1);

   if (s->layout.var.prm_attr.mask & VARYING_BIT_CULL_PRIMITIVE) {
      nir_store_var(b, s->out_variables[VARYING_SLOT_CULL_PRIMITIVE * 4], nir_b2i32(b, store_val), 0x1);
      return;
   }

   nir_def *arr_index = nir_get_io_arrayed_index_src(intrin)->ssa;

   /* To reduce LDS use, store these as an array of 8-bit values: one byte per
    * primitive (see ms_calculate_output_layout). */
   nir_store_shared(b, nir_b2i8(b, store_val), arr_index, .base = s->layout.lds.cull_flags_addr);
}

static nir_def *
ms_arrayed_output_base_addr(nir_builder *b,
                            nir_def *arr_index,
                            unsigned mapped_location,
                            unsigned num_arrayed_outputs)
{
   /* Address offset of the array item (vertex or primitive). */
   unsigned arr_index_stride = num_arrayed_outputs * 16u;
   nir_def *arr_index_off = nir_imul_imm(b, arr_index, arr_index_stride);

   /* IO address offset within the vertex or primitive data. */
   unsigned io_offset = mapped_location * 16u;
   nir_def *io_off = nir_imm_int(b, io_offset);

   return nir_iadd_nuw(b, arr_index_off, io_off);
}

/* RADV_BC250_MESH_COMPACT_LDS packed vertex record: byte stride and the byte
 * offset of a location inside the record. */
static unsigned
ms_packed_vertex_stride(const ms_out_mem_layout *l)
{
   unsigned dwords = 0;
   u_foreach_bit64(loc, l->lds.vtx_attr.mask)
      dwords += l->lds.vtx_comps[loc];
   return dwords * 4;
}

static unsigned
ms_packed_vertex_offset(const ms_out_mem_layout *l, unsigned location)
{
   unsigned dwords = 0;
   u_foreach_bit64(loc, l->lds.vtx_attr.mask & BITFIELD64_MASK(location))
      dwords += l->lds.vtx_comps[loc];
   return dwords * 4;
}

static void
update_ms_output_info(const nir_io_semantics io_sem,
                      const nir_src *base_offset_src,
                      const uint32_t write_mask,
                      const unsigned component_offset,
                      const unsigned bit_size,
                      const ms_out_part *out,
                      lower_ngg_ms_state *s)
{
   const uint32_t components_mask = write_mask << component_offset;

   /* 64-bit outputs should have already been lowered to 32-bit. */
   assert(bit_size <= 32);
   assert(components_mask <= 0xf);

   /* When the base offset is constant, only mark the components of the current slot as used.
    * Otherwise, mark the components of all possibly affected slots as used.
    */
   const unsigned base_off_start = nir_src_is_const(*base_offset_src) ? nir_src_as_uint(*base_offset_src) : 0;
   const unsigned num_slots = nir_src_is_const(*base_offset_src) ? 1 : io_sem.num_slots;

   for (unsigned base_off = base_off_start; base_off < num_slots; ++base_off) {
      ac_nir_prerast_per_output_info *info = &s->out.infos[io_sem.location + base_off];
      info->components_mask |= components_mask;

      if (!io_sem.no_sysval_output)
         info->as_sysval_mask |= components_mask;
      if (!io_sem.no_varying)
         info->as_varying_mask |= components_mask;
   }
}

static const ms_out_part *
ms_get_out_layout_part(unsigned location,
                       shader_info *info,
                       ms_out_mode *out_mode,
                       lower_ngg_ms_state *s)
{
   uint64_t mask = BITFIELD64_BIT(location);

   if (info->per_primitive_outputs & mask) {
      if (mask & s->layout.lds.prm_attr.mask) {
         *out_mode = ms_out_mode_lds;
         return &s->layout.lds.prm_attr;
      } else if (mask & s->layout.scratch_ring.prm_attr.mask) {
         *out_mode = ms_out_mode_scratch_ring;
         return &s->layout.scratch_ring.prm_attr;
      } else if (mask & s->layout.attr_ring.prm_attr.mask) {
         *out_mode = ms_out_mode_attr_ring;
         return &s->layout.attr_ring.prm_attr;
      } else if (mask & s->layout.var.prm_attr.mask) {
         *out_mode = ms_out_mode_var;
         return &s->layout.var.prm_attr;
      }
   } else {
      if (mask & s->layout.lds.vtx_attr.mask) {
         *out_mode = ms_out_mode_lds;
         return &s->layout.lds.vtx_attr;
      } else if (mask & s->layout.scratch_ring.vtx_attr.mask) {
         *out_mode = ms_out_mode_scratch_ring;
         return &s->layout.scratch_ring.vtx_attr;
      } else if (mask & s->layout.attr_ring.vtx_attr.mask) {
         *out_mode = ms_out_mode_attr_ring;
         return &s->layout.attr_ring.vtx_attr;
      } else if (mask & s->layout.var.vtx_attr.mask) {
         *out_mode = ms_out_mode_var;
         return &s->layout.var.vtx_attr;
      }
   }

   UNREACHABLE("Couldn't figure out mesh shader output mode.");
}

static void
ms_store_arrayed_output(nir_builder *b,
                        nir_src *base_off_src,
                        nir_def *store_val,
                        nir_def *arr_index,
                        const nir_io_semantics io_sem,
                        const unsigned component_offset,
                        const unsigned write_mask,
                        lower_ngg_ms_state *s)
{
   ms_out_mode out_mode;
   const ms_out_part *out = ms_get_out_layout_part(io_sem.location, &b->shader->info, &out_mode, s);
   update_ms_output_info(io_sem, base_off_src, write_mask, component_offset, store_val->bit_size, out, s);

   bool hi_16b = io_sem.high_16bits;
   bool lo_16b = !hi_16b && store_val->bit_size == 16;

   unsigned mapped_location = util_bitcount64(out->mask & BITFIELD64_MASK(io_sem.location));
   unsigned num_outputs = util_bitcount64(out->mask);
   unsigned const_off = out->addr + component_offset * 4 + (hi_16b ? 2 : 0);

   nir_def *base_addr = ms_arrayed_output_base_addr(b, arr_index, mapped_location, num_outputs);
   nir_def *base_offset = base_off_src->ssa;
   nir_def *base_addr_off = nir_imul_imm(b, base_offset, 16u);
   nir_def *addr = nir_iadd_nuw(b, base_addr, base_addr_off);

   if (out_mode == ms_out_mode_lds && out == &s->layout.lds.vtx_attr && s->layout.lds.vtx_packed) {
      /* Packed record: every access has a zero slot offset (ms_packed_vertex_components). */
      assert(component_offset + util_last_bit(write_mask) <= s->layout.lds.vtx_comps[io_sem.location]);
      addr = nir_imul_imm(b, arr_index, ms_packed_vertex_stride(&s->layout));
      const_off = out->addr + ms_packed_vertex_offset(&s->layout, io_sem.location) +
                  component_offset * 4 + (hi_16b ? 2 : 0);
      nir_store_shared(b, store_val, addr, .base = const_off,
                     .write_mask = write_mask, .align_mul = 4, .align_offset = const_off % 4);
   } else if (out_mode == ms_out_mode_lds) {
      nir_store_shared(b, store_val, addr, .base = const_off,
                     .write_mask = write_mask, .align_mul = 16,
                     .align_offset = const_off % 16);
   } else if (out_mode == ms_out_mode_scratch_ring) {
      nir_def *ring = nir_load_ring_mesh_scratch_amd(b);
      nir_def *off = nir_load_ring_mesh_scratch_offset_amd(b);
      nir_def *zero = nir_imm_int(b, 0);
      nir_store_buffer_amd(b, store_val, ring, addr, off, zero,
                           .base = const_off,
                           .write_mask = write_mask,
                           .memory_modes = nir_var_shader_out,
                           .access = ACCESS_COHERENT);
   } else if (out_mode == ms_out_mode_attr_ring) {
      /* Store params straight to the attribute ring.
       * Even though the access pattern may not be the most optimal,
       * this is still much better than reserving LDS and losing waves.
       * (Also much better than storing and reloading from the scratch ring.)
       */
      unsigned param_offset = s->vs_output_param_offset[io_sem.location];
      nir_def *ring = nir_load_ring_attr_amd(b);
      nir_def *soffset = nir_load_ring_attr_offset_amd(b);
      nir_store_buffer_amd(b, store_val, ring, base_addr_off, soffset, arr_index,
                           .base = const_off + param_offset * 16,
                           .write_mask = write_mask,
                           .memory_modes = nir_var_shader_out,
                           .access = ACCESS_COHERENT | ACCESS_IS_SWIZZLED_AMD,
                           .align_mul = 16, .align_offset = const_off % 16u);
   } else if (out_mode == ms_out_mode_var) {
      u_foreach_bit(comp, write_mask) {
         unsigned idx = io_sem.location * 4 + comp + component_offset;
         nir_def *val = nir_channel(b, store_val, comp);
         nir_def *v = nir_load_var(b, s->out_variables[idx]);

         if (lo_16b) {
            nir_def *var_hi = nir_unpack_32_2x16_split_y(b, v);
            val = nir_pack_32_2x16_split(b, val, var_hi);
         } else if (hi_16b) {
            nir_def *var_lo = nir_unpack_32_2x16_split_x(b, v);
            val = nir_pack_32_2x16_split(b, var_lo, val);
         }

         nir_store_var(b, s->out_variables[idx], val, 0x1);
      }
   } else {
      UNREACHABLE("Invalid MS output mode for store");
   }
}

static void
ms_store_arrayed_output_intrin(nir_builder *b,
                               nir_intrinsic_instr *intrin,
                               lower_ngg_ms_state *s)
{
   const nir_io_semantics io_sem = nir_intrinsic_io_semantics(intrin);

   if (io_sem.location == VARYING_SLOT_PRIMITIVE_INDICES) {
      ms_store_prim_indices(b, intrin, s);
      return;
   } else if (io_sem.location == VARYING_SLOT_CULL_PRIMITIVE) {
      ms_store_cull_flag(b, intrin, s);
      return;
   }

   unsigned component_offset = nir_intrinsic_component(intrin);
   unsigned write_mask = nir_intrinsic_write_mask(intrin);

   nir_def *store_val = intrin->src[0].ssa;
   nir_def *arr_index = nir_get_io_arrayed_index_src(intrin)->ssa;
   nir_src *base_off_src = nir_get_io_offset_src(intrin);

   if (store_val->bit_size < 32) {
      /* Split 16-bit output stores to ensure each 16-bit component is stored
       * in the correct location, without overwriting the other 16 bits there.
       */
      u_foreach_bit(c, write_mask) {
         nir_def *store_component = nir_channel(b, store_val, c);
         ms_store_arrayed_output(b, base_off_src, store_component, arr_index, io_sem, c + component_offset, 1, s);
      }
   } else {
      ms_store_arrayed_output(b, base_off_src, store_val, arr_index, io_sem, component_offset, write_mask, s);
   }
}

/* RADV_BC250_MESH_DIRECT_READ: clone the computation of def for the vertex
 * index of the remap table (sources first, so every source is remapped). */
static nir_def *
ms_direct_clone(nir_builder *b, nir_def *def, ms_direct_read *dr)
{
   struct hash_entry *he = _mesa_hash_table_search(dr->remap, def);
   if (he)
      return he->data;

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
         ms_direct_clone(b, alu->src[i].src.ssa, dr);
   } else if (instr->type == nir_instr_type_intrinsic) {
      assert(nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_shared);
      ms_direct_clone(b, nir_instr_as_intrinsic(instr)->src[0].ssa, dr);
   } else {
      assert(instr->type == nir_instr_type_load_const);
   }

   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, dr->remap);
   nir_builder_instr_insert(b, clone);
   nir_def *result = nir_instr_def(clone);
   _mesa_hash_table_insert(dr->remap, def, result);
   return result;
}

static nir_def *
ms_direct_read_load(nir_builder *b, nir_def *index, unsigned location, unsigned component_offset,
                    unsigned num_components, lower_ngg_ms_state *s)
{
   ms_direct_read *dr = s->direct;
   nir_block *block = nir_cursor_current_block(b->cursor);

   /* Clones are shared by the loads of one vertex emitted in sequence. */
   if (dr->remap_index != index || dr->remap_block != block) {
      _mesa_hash_table_clear(dr->remap, NULL);
      _mesa_hash_table_insert(dr->remap, dr->vertex, index);
      dr->remap_index = index;
      dr->remap_block = block;
   }

   nir_def *comps[4];
   for (unsigned i = 0; i < num_components; i++) {
      nir_scalar value = dr->value[location][component_offset + i];
      /* Components never stored: the expanded record was never written either. */
      comps[i] = value.def ? nir_channel(b, ms_direct_clone(b, value.def, dr), value.comp) :
                             nir_undef(b, 1, 32);
   }
   return nir_vec(b, comps, num_components);
}

static nir_def *
ms_load_arrayed_output(nir_builder *b,
                       nir_def *arr_index,
                       nir_def *base_offset,
                       unsigned location,
                       unsigned component_offset,
                       unsigned num_components,
                       unsigned load_bit_size,
                       lower_ngg_ms_state *s)
{
   if (s->direct && (s->direct->mask & BITFIELD64_BIT(location))) {
      assert(load_bit_size == 32 && nir_src_is_const(nir_src_for_ssa(base_offset)) &&
             nir_src_as_uint(nir_src_for_ssa(base_offset)) == 0);
      return ms_direct_read_load(b, arr_index, location, component_offset, num_components, s);
   }

   ms_out_mode out_mode;
   const ms_out_part *out = ms_get_out_layout_part(location, &b->shader->info, &out_mode, s);

   unsigned num_outputs = util_bitcount64(out->mask);
   unsigned const_off = out->addr + component_offset * 4;

   /* Use compacted location instead of the original semantic location. */
   unsigned mapped_location = util_bitcount64(out->mask & BITFIELD64_MASK(location));

   nir_def *base_addr = ms_arrayed_output_base_addr(b, arr_index, mapped_location, num_outputs);
   nir_def *base_addr_off = nir_imul_imm(b, base_offset, 16);
   nir_def *addr = nir_iadd_nuw(b, base_addr, base_addr_off);

   if (out_mode == ms_out_mode_lds && out == &s->layout.lds.vtx_attr && s->layout.lds.vtx_packed) {
      assert(component_offset + num_components <= s->layout.lds.vtx_comps[location]);
      addr = nir_imul_imm(b, arr_index, ms_packed_vertex_stride(&s->layout));
      return nir_load_shared(b, num_components, load_bit_size, addr, .align_mul = 4, .align_offset = 0,
                             .base = out->addr + ms_packed_vertex_offset(&s->layout, location) +
                                     component_offset * 4);
   } else if (out_mode == ms_out_mode_lds) {
      return nir_load_shared(b, num_components, load_bit_size, addr, .align_mul = 16,
                             .align_offset = (component_offset * 4) % 16,
                             .base = const_off);
   } else if (out_mode == ms_out_mode_scratch_ring) {
      nir_def *ring = nir_load_ring_mesh_scratch_amd(b);
      nir_def *off = nir_load_ring_mesh_scratch_offset_amd(b);
      nir_def *zero = nir_imm_int(b, 0);
      return nir_load_buffer_amd(b, num_components, load_bit_size, ring, addr, off, zero,
                                 .base = const_off,
                                 .memory_modes = nir_var_shader_out,
                                 .access = ACCESS_COHERENT);
   } else if (out_mode == ms_out_mode_var) {
      assert(load_bit_size == 32);
      nir_def *arr[8] = {0};
      for (unsigned comp = 0; comp < num_components; ++comp) {
         unsigned idx = location * 4 + comp + component_offset;
         arr[comp] = nir_load_var(b, s->out_variables[idx]);
      }
      return nir_vec(b, arr, num_components);
   } else {
      UNREACHABLE("Invalid MS output mode for load");
   }
}

static nir_def *
lower_ms_load_workgroup_index(nir_builder *b,
                              UNUSED nir_intrinsic_instr *intrin,
                              lower_ngg_ms_state *s)
{
   return s->workgroup_index;
}

static nir_def *
lower_ms_set_vertex_and_primitive_count(nir_builder *b,
                                        nir_intrinsic_instr *intrin,
                                        lower_ngg_ms_state *s)
{
   /* Remember if the output vertex and primitive counts are both workgroup-uniform.
    * This assumes that the divergence info contains workgroup divergence.
    */
   s->output_counts_workgroup_uniform &=
      !nir_src_is_divergent(&intrin->src[0]) && !nir_src_is_divergent(&intrin->src[1]);

   /* If either the number of vertices or primitives is zero, set both of them to zero. */
   nir_def *num_vtx = nir_read_first_invocation(b, intrin->src[0].ssa);
   nir_def *num_prm = nir_read_first_invocation(b, intrin->src[1].ssa);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *is_either_zero = nir_ieq(b, nir_umin(b, num_vtx, num_prm), zero);
   num_vtx = nir_bcsel(b, is_either_zero, zero, num_vtx);
   num_prm = nir_bcsel(b, is_either_zero, zero, num_prm);

   nir_store_var(b, s->vertex_count_var, num_vtx, 0x1);
   nir_store_var(b, s->primitive_count_var, num_prm, 0x1);

   return NIR_LOWER_INSTR_PROGRESS_REPLACE;
}

static nir_def *
update_ms_barrier(nir_builder *b,
                         nir_intrinsic_instr *intrin,
                         lower_ngg_ms_state *s)
{
   /* Output loads and stores are lowered to shared memory access,
    * so we have to update the barriers to also reflect this.
    */
   unsigned mem_modes = nir_intrinsic_memory_modes(intrin);
   if (mem_modes & nir_var_shader_out)
      mem_modes |= nir_var_mem_shared;
   else
      return NULL;

   nir_intrinsic_set_memory_modes(intrin, mem_modes);

   return NIR_LOWER_INSTR_PROGRESS;
}

static nir_def *
lower_ms_intrinsic(nir_builder *b, nir_instr *instr, void *state)
{
   lower_ngg_ms_state *s = (lower_ngg_ms_state *) state;

   if (instr->type != nir_instr_type_intrinsic)
      return NULL;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output:
      ms_store_arrayed_output_intrin(b, intrin, s);
      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   case nir_intrinsic_barrier:
      return update_ms_barrier(b, intrin, s);
   case nir_intrinsic_load_workgroup_index:
      return lower_ms_load_workgroup_index(b, intrin, s);
   case nir_intrinsic_load_num_subgroups:
      return nir_imm_int(b, DIV_ROUND_UP(s->api_workgroup_size, s->wave_size));
   case nir_intrinsic_set_vertex_and_primitive_count:
      return lower_ms_set_vertex_and_primitive_count(b, intrin, s);
   default:
      UNREACHABLE("Not a lowerable mesh shader intrinsic.");
   }
}

static bool
filter_ms_intrinsic(const nir_instr *instr,
                    UNUSED const void *s)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   return intrin->intrinsic == nir_intrinsic_store_output ||
          intrin->intrinsic == nir_intrinsic_load_output ||
          intrin->intrinsic == nir_intrinsic_store_per_vertex_output ||
          intrin->intrinsic == nir_intrinsic_store_per_primitive_output ||
          intrin->intrinsic == nir_intrinsic_barrier ||
          intrin->intrinsic == nir_intrinsic_load_workgroup_index ||
          intrin->intrinsic == nir_intrinsic_load_num_subgroups ||
          intrin->intrinsic == nir_intrinsic_set_vertex_and_primitive_count;
}

static void
lower_ms_intrinsics(nir_shader *shader, lower_ngg_ms_state *s)
{
   nir_shader_lower_instructions(shader, filter_ms_intrinsic, lower_ms_intrinsic, s);
}

static void
ms_emit_arrayed_outputs(nir_builder *b,
                        nir_def *invocation_index,
                        uint64_t mask,
                        lower_ngg_ms_state *s)
{
   nir_def *zero = nir_imm_int(b, 0);

   u_foreach_bit64(slot, mask) {
      /* Should not occur here, handled separately. */
      assert(slot != VARYING_SLOT_PRIMITIVE_COUNT && slot != VARYING_SLOT_PRIMITIVE_INDICES);

      unsigned component_mask = s->out.infos[slot].components_mask;

      while (component_mask) {
         int start_comp = 0, num_components = 1;
         u_bit_scan_consecutive_range(&component_mask, &start_comp, &num_components);

         nir_def *load =
            ms_load_arrayed_output(b, invocation_index, zero, slot, start_comp,
                                   num_components, 32, s);

         for (int i = 0; i < num_components; i++)
            s->out.outputs[slot][start_comp + i] = nir_channel(b, load, i);
      }
   }
}

static void
ms_create_same_invocation_vars(nir_builder *b, lower_ngg_ms_state *s)
{
   /* Initialize NIR variables for same-invocation outputs. */
   uint64_t same_invocation_output_mask = s->layout.var.prm_attr.mask | s->layout.var.vtx_attr.mask;

   u_foreach_bit64(slot, same_invocation_output_mask) {
      for (unsigned comp = 0; comp < 4; ++comp) {
         unsigned idx = slot * 4 + comp;
         s->out_variables[idx] = nir_local_variable_create(b->impl, glsl_uint_type(), "ms_var_output");
      }
   }
}

static void
ms_emit_legacy_workgroup_index(nir_builder *b, lower_ngg_ms_state *s)
{
   /* Workgroup ID should have been lowered to workgroup index. */
   assert(!BITSET_TEST(b->shader->info.system_values_read, SYSTEM_VALUE_WORKGROUP_ID));

   /* No need to do anything if the shader doesn't use the workgroup index. */
   if (!BITSET_TEST(b->shader->info.system_values_read, SYSTEM_VALUE_WORKGROUP_INDEX))
      return;

   b->cursor = nir_before_impl(b->impl);

   /* Legacy fast launch mode (FAST_LAUNCH=1):
    *
    * The HW doesn't support a proper workgroup index for vertex processing stages,
    * so we use the vertex ID which is equivalent to the index of the current workgroup
    * within the current dispatch.
    *
    * Due to the register programming of mesh shaders, this value is only filled for
    * the first invocation of the first wave. To let other waves know, we use LDS.
    */
   nir_def *workgroup_index = nir_load_vertex_id_zero_base(b);

   if (s->api_workgroup_size <= s->wave_size) {
      /* API workgroup is small, so we don't need to use LDS. */
      s->workgroup_index = nir_read_first_invocation(b, workgroup_index);
      return;
   }

   unsigned workgroup_index_lds_addr = s->layout.lds.workgroup_info_addr + lds_ms_wg_index;

   nir_def *zero = nir_imm_int(b, 0);
   nir_def *dont_care = nir_undef(b, 1, 32);
   nir_def *loaded_workgroup_index = NULL;

   /* Use elect to make sure only 1 invocation uses LDS. */
   nir_if *if_elected = nir_push_if(b, nir_elect(b, 1));
   {
      nir_def *wave_id = nir_load_subgroup_id(b);
      nir_if *if_wave_0 = nir_push_if(b, nir_ieq_imm(b, wave_id, 0));
      {
         nir_store_shared(b, workgroup_index, zero, .base = workgroup_index_lds_addr);
         nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                               .memory_scope = SCOPE_WORKGROUP,
                               .memory_semantics = NIR_MEMORY_ACQ_REL,
                               .memory_modes = nir_var_mem_shared);
      }
      nir_push_else(b, if_wave_0);
      {
         nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                               .memory_scope = SCOPE_WORKGROUP,
                               .memory_semantics = NIR_MEMORY_ACQ_REL,
                               .memory_modes = nir_var_mem_shared);
         loaded_workgroup_index = nir_load_shared(b, 1, 32, zero, .base = workgroup_index_lds_addr);
      }
      nir_pop_if(b, if_wave_0);

      workgroup_index = nir_if_phi(b, workgroup_index, loaded_workgroup_index);
   }
   nir_pop_if(b, if_elected);

   workgroup_index = nir_if_phi(b, workgroup_index, dont_care);
   s->workgroup_index = nir_read_first_invocation(b, workgroup_index);
}

/* The position vectors ac_nir_export_position writes for every exported vertex (POS0, the
 * misc vector, one or two packed clip/cull distance vectors), the count RADV declares in
 * SPI_SHADER_POS_FORMAT (radv_get_num_pos_exports). */
static unsigned
ms_num_pos_exports(uint64_t per_vertex_outputs, uint32_t clipdist_mask)
{
   unsigned num = 1;
   if (per_vertex_outputs & (VARYING_BIT_PSIZ | VARYING_BIT_EDGE | VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT |
                             VARYING_BIT_PRIMITIVE_SHADING_RATE))
      num++;
   const unsigned comps = util_bitcount(clipdist_mask);
   return num + (comps > 0) + (comps > 4);
}

static void
set_ms_final_output_counts(nir_builder *b,
                           lower_ngg_ms_state *s,
                           nir_def **out_num_prm,
                           nir_def **out_num_vtx)
{
   /* The spec allows the numbers to be divergent, and in that case we need to
    * use the values from the first invocation. Also the HW requires us to set
    * both to 0 if either was 0.
    *
    * These are already done by the lowering.
    */
   nir_def *num_prm = nir_load_var(b, s->primitive_count_var);
   nir_def *num_vtx = nir_load_var(b, s->vertex_count_var);

   *out_num_prm = num_prm;
   *out_num_vtx = num_vtx;

   /* RADV_BC250_MESH_REFERENCE: the reference driver's fully-culled dummy exports POS0 only. */
   const unsigned dummy_pos_exports = s->options->bc250_reference ? 1 : s->num_pos_exports;

   if (s->hw_workgroup_size <= s->wave_size) {
      /* Single-wave mesh shader workgroup. */
      ac_nir_ngg_alloc_vertices_and_primitives_pos(b, num_vtx, num_prm, s->ac->has_ngg_fully_culled_bug,
                                                   dummy_pos_exports);
      return;
   }

   if (s->counts_in_all_lanes) {
      /* RADV_BC250_MESH_COMPACT: every lane computed the final counts from LDS. */
      nir_if *if_wave_0 = nir_push_if(b, nir_ieq_imm(b, nir_load_subgroup_id(b), 0));
      ac_nir_ngg_alloc_vertices_and_primitives_pos(b, num_vtx, num_prm, s->ac->has_ngg_fully_culled_bug,
                                                   s->num_pos_exports);
      nir_pop_if(b, if_wave_0);
      return;
   }

   if (s->output_counts_workgroup_uniform && !s->has_non_api_waves && !s->options->bc250_reference) {
      /* Output counts are workgroup-uniform and all waves execute the API shader.
       * All waves calculated the same output count values, so we don't
       * have to distribute the value from the first active invocation.
       *
       * Note that this can't be done when there are non-API waves
       * because those don't execute the API shader and therefore
       * can't know the value of the output counts and must read it
       * from LDS.
       *
       * We can simply use the values from the first wave when
       * allocating space for vertices/primitives.
       */
      nir_if *if_wave_0 = nir_push_if(b, nir_ieq_imm(b, nir_load_subgroup_id(b), 0));
      ac_nir_ngg_alloc_vertices_and_primitives_pos(b, num_vtx, num_prm, s->ac->has_ngg_fully_culled_bug,
                                                   s->num_pos_exports);
      nir_pop_if(b, if_wave_0);
      return;
   }

   /* Multi-wave mesh shader workgroup with workgroup-divergent output counts:
    * We need to use LDS to distribute the correct values to the other waves.
    */

   nir_def *zero = nir_imm_int(b, 0);

   nir_if *if_wave_0 = nir_push_if(b, nir_ieq_imm(b, nir_load_subgroup_id(b), 0));
   {
      nir_if *if_elected = nir_push_if(b, nir_elect(b, 1));
      {
         nir_store_shared(b, nir_vec2(b, num_prm, num_vtx), zero,
                          .base = s->layout.lds.workgroup_info_addr + lds_ms_num_prims);
      }
      nir_pop_if(b, if_elected);

      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                            .memory_scope = SCOPE_WORKGROUP,
                            .memory_semantics = NIR_MEMORY_ACQ_REL,
                            .memory_modes = nir_var_mem_shared);

      ac_nir_ngg_alloc_vertices_and_primitives_pos(b, num_vtx, num_prm, s->ac->has_ngg_fully_culled_bug,
                                                   dummy_pos_exports);
   }
   nir_push_else(b, if_wave_0);
   {
      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                            .memory_scope = SCOPE_WORKGROUP,
                            .memory_semantics = NIR_MEMORY_ACQ_REL,
                            .memory_modes = nir_var_mem_shared);

      nir_def *prm_vtx = NULL;
      nir_def *dont_care_2x32 = nir_undef(b, 2, 32);
      nir_if *if_elected = nir_push_if(b, nir_elect(b, 1));
      {
         prm_vtx = nir_load_shared(b, 2, 32, zero,
                                   .base = s->layout.lds.workgroup_info_addr + lds_ms_num_prims);
      }
      nir_pop_if(b, if_elected);

      prm_vtx = nir_if_phi(b, prm_vtx, dont_care_2x32);
      num_prm = nir_read_first_invocation(b, nir_channel(b, prm_vtx, 0));
      num_vtx = nir_read_first_invocation(b, nir_channel(b, prm_vtx, 1));

      nir_store_var(b, s->primitive_count_var, num_prm, 0x1);
      nir_store_var(b, s->vertex_count_var, num_vtx, 0x1);
   }
   nir_pop_if(b, if_wave_0);

   *out_num_prm = nir_load_var(b, s->primitive_count_var);
   *out_num_vtx = nir_load_var(b, s->vertex_count_var);
}

static void
ms_emit_attribute_ring_output_stores(nir_builder *b, const uint64_t outputs_mask,
                                     nir_def *idx, lower_ngg_ms_state *s)
{
   if (!outputs_mask)
      return;

   nir_def *ring = nir_load_ring_attr_amd(b);
   nir_def *off = nir_load_ring_attr_offset_amd(b);
   nir_def *zero = nir_imm_int(b, 0);

   u_foreach_bit64 (slot, outputs_mask) {
      if (s->vs_output_param_offset[slot] > AC_EXP_PARAM_OFFSET_31)
         continue;

      nir_def *soffset = nir_iadd_imm(b, off, s->vs_output_param_offset[slot] * 16 * 32);
      nir_def *store_val = nir_undef(b, 4, 32);
      unsigned store_val_components = 0;
      for (unsigned c = 0; c < 4; ++c) {
         if (s->out.outputs[slot][c]) {
            store_val = nir_vector_insert_imm(b, store_val, s->out.outputs[slot][c], c);
            store_val_components = c + 1;
         }
      }

      store_val = nir_trim_vector(b, store_val, store_val_components);
      nir_store_buffer_amd(b, store_val, ring, zero, soffset, idx,
                           .memory_modes = nir_var_shader_out,
                           .access = ACCESS_COHERENT | ACCESS_IS_SWIZZLED_AMD,
                           .align_mul = 16, .align_offset = 0);
   }
}

static nir_def *
ms_prim_exp_arg_ch1(nir_builder *b, nir_def *invocation_index, nir_def *num_vtx, lower_ngg_ms_state *s)
{
   if (s->safe_direct) {
      if (s->pp_share) {
         /* RADV_BC250_MESH_PP_SHARE: the planned slot triple of surviving triangle r. */
         nir_def *planned = nir_u2u32(b, nir_load_shared(b, 3, 8, nir_imul_imm(b, invocation_index, 3),
                                                        .base = s->safe_direct_indices_addr));
         nir_def *slot[3] = {nir_channel(b, planned, 0), nir_channel(b, planned, 1), nir_channel(b, planned, 2)};
         return ac_nir_pack_ngg_prim_exp_arg(b, 3, slot, NULL, s->ac->gfx_level);
      }
      if (s->options->bc250_safe_corners) {
         nir_def *first = nir_imul_imm(b, invocation_index, 3);
         nir_def *slots[3] = {first, nir_iadd_imm(b, first, 1), nir_iadd_imm(b, first, 2)};
         nir_def *corners = ac_nir_pack_ngg_prim_exp_arg(b, 3, slots, NULL, s->ac->gfx_level);
         /* RADV_BC250_MESH_SAFE_ADAPTIVE: a clean workgroup exports surviving triangle r with the
          * API indices the checker published at map[3r..3r+2] (readable by every wave). */
         if (s->options->bc250_safe_adaptive && s->fast_clean) {
            nir_def *mapped = nir_u2u32(b, nir_load_shared(b, 3, 8, first, .base = s->safe_direct_map_addr));
            nir_def *api[3] = {nir_channel(b, mapped, 0), nir_channel(b, mapped, 1), nir_channel(b, mapped, 2)};
            nir_def *arg = nir_bcsel(b, s->fast_clean, ac_nir_pack_ngg_prim_exp_arg(b, 3, api, NULL, s->ac->gfx_level),
                                     corners);
            if (s->fast_compact) {
               /* RADV_BC250_MESH_SAFE_COMPACT: the renumbered survivor indices. */
               nir_def *ranked = nir_u2u32(b, nir_load_shared(b, 3, 8, first, .base = s->safe_direct_indices_addr));
               nir_def *slot[3] = {nir_channel(b, ranked, 0), nir_channel(b, ranked, 1), nir_channel(b, ranked, 2)};
               arg = nir_bcsel(b, s->fast_compact, ac_nir_pack_ngg_prim_exp_arg(b, 3, slot, NULL, s->ac->gfx_level),
                               arg);
            }
            return arg;
         }
         return corners;
      }
      nir_if *repair = s->fast_clean ? nir_push_if(b, nir_inot(b, s->fast_clean)) : NULL;
      nir_def *mapped = nir_u2u32(b, nir_load_shared(b, 3, 8, nir_imul_imm(b, invocation_index, 3),
                                                   .base = s->safe_direct_indices_addr));
      nir_def *slots[3] = {nir_channel(b, mapped, 0), nir_channel(b, mapped, 1), nir_channel(b, mapped, 2)};
      nir_def *arg = ac_nir_pack_ngg_prim_exp_arg(b, 3, slots, NULL, s->ac->gfx_level);
      if (repair) {
         nir_push_else(b, repair);
         nir_def *raw = nir_load_var(b, s->fast_prim_arg);
         nir_pop_if(b, repair);
         arg = nir_if_phi(b, arg, raw);
      }
      return arg;
   }

   /* Primitive connectivity data: describes which vertices the primitive uses. */
   nir_def *prim_idx_addr = nir_imul_imm(b, invocation_index, s->vertices_per_prim);
   nir_def *indices_loaded = NULL;
   nir_def *cull_flag = NULL;

   if (s->implicit_indices) {
      nir_def *indices[3] = {0};
      for (unsigned c = 0; c < s->vertices_per_prim; ++c)
         indices[c] = ms_implicit_index(b, invocation_index, c, s);
      indices_loaded = nir_vec(b, indices, s->vertices_per_prim);
   } else if (s->layout.var.prm_attr.mask & VARYING_BIT_PRIMITIVE_INDICES) {
      nir_def *indices[3] = {0};
      for (unsigned c = 0; c < s->vertices_per_prim; ++c)
         indices[c] = nir_load_var(b, s->out_variables[VARYING_SLOT_PRIMITIVE_INDICES * 4 + c]);
      indices_loaded = nir_vec(b, indices, s->vertices_per_prim);
   } else {
      indices_loaded = nir_load_shared(b, s->vertices_per_prim, 8, prim_idx_addr, .base = s->layout.lds.indices_addr);
      indices_loaded = nir_u2u32(b, indices_loaded);
   }

   if (s->uses_cull_flags && !s->compact_cull) {
      nir_def *loaded_cull_flag = NULL;
      if (s->layout.var.prm_attr.mask & VARYING_BIT_CULL_PRIMITIVE)
         loaded_cull_flag = nir_load_var(b, s->out_variables[VARYING_SLOT_CULL_PRIMITIVE * 4]);
      else
         /* One byte per primitive; prim_idx_addr is the vertex-indexed offset. */
         loaded_cull_flag = nir_u2u32(b, nir_load_shared(b, 1, 8, invocation_index, .base = s->layout.lds.cull_flags_addr));

      cull_flag = nir_i2b(b, loaded_cull_flag);
   }

   nir_def *indices[3];
   nir_def *max_vtx_idx = nir_iadd_imm(b, num_vtx, -1u);

   for (unsigned i = 0; i < s->vertices_per_prim; ++i) {
      indices[i] = nir_channel(b, indices_loaded, i);
      indices[i] = nir_umin(b, indices[i], max_vtx_idx);
   }

   return ac_nir_pack_ngg_prim_exp_arg(b, s->vertices_per_prim, indices, cull_flag,
                                       s->ac->gfx_level);
}

static nir_def *
ms_prim_exp_arg_ch2(nir_builder *b, uint64_t outputs_mask, lower_ngg_ms_state *s)
{
   nir_def *prim_exp_arg_ch2 = NULL;

   if (outputs_mask) {
      /* When layer, viewport etc. are per-primitive, they need to be encoded in
       * the primitive export instruction's second channel. The encoding is:
       *
       * --- GFX10.3 ---
       * bits 31..30: VRS rate Y
       * bits 29..28: VRS rate X
       * bits 23..20: viewport
       * bits 19..17: layer
       *
       * --- GFX11 ---
       * bits 31..28: VRS rate enum
       * bits 23..20: viewport
       * bits 12..00: layer
       */
      prim_exp_arg_ch2 = nir_imm_int(b, 0);

      if (outputs_mask & VARYING_BIT_LAYER) {
         nir_def *layer = nir_ishl_imm(b, s->out.outputs[VARYING_SLOT_LAYER][0],
                                       s->ac->gfx_level >= GFX11 ? 0 : 17);
         prim_exp_arg_ch2 = nir_ior(b, prim_exp_arg_ch2, layer);
      }

      if (outputs_mask & VARYING_BIT_VIEWPORT) {
         nir_def *view = nir_ishl_imm(b, s->out.outputs[VARYING_SLOT_VIEWPORT][0], 20);
         prim_exp_arg_ch2 = nir_ior(b, prim_exp_arg_ch2, view);
      }

      if (outputs_mask & VARYING_BIT_PRIMITIVE_SHADING_RATE) {
         nir_def *rate = s->out.outputs[VARYING_SLOT_PRIMITIVE_SHADING_RATE][0];
         prim_exp_arg_ch2 = nir_ior(b, prim_exp_arg_ch2, rate);
      }
   }

   return prim_exp_arg_ch2;
}

static void
ms_prim_gen_query(nir_builder *b,
                  nir_def *invocation_index,
                  nir_def *num_prm,
                  lower_ngg_ms_state *s)
{
   if (!s->options->has_gen_prim_query)
      return;

   nir_if *if_invocation_index_zero = nir_push_if(b, nir_ieq_imm(b, invocation_index, 0));
   {
      nir_if *if_shader_query = nir_push_if(b, nir_load_prim_gen_query_enabled_amd(b));
      {
         nir_atomic_add_gen_prim_count_amd(b, num_prm, .stream_id = 0);
      }
      nir_pop_if(b, if_shader_query);
   }
   nir_pop_if(b, if_invocation_index_zero);
}

static void
ms_invocation_query(nir_builder *b,
                    nir_def *invocation_index,
                    lower_ngg_ms_state *s)
{
   if (!s->options->has_ms_gs_invocations_query)
      return;

   nir_if *if_invocation_index_zero = nir_push_if(b, nir_ieq_imm(b, invocation_index, 0));
   {
      nir_if *if_pipeline_query = nir_push_if(b, nir_load_pipeline_stat_query_enabled_amd(b));
      {
         /* Count API shader invocations; extra hardware waves are bookkeeping. */
         nir_atomic_add_shader_invocation_count_amd(b, nir_imm_int(b, s->api_workgroup_size));
      }
      nir_pop_if(b, if_pipeline_query);
   }
   nir_pop_if(b, if_invocation_index_zero);
}

static nir_def *ms_compact_vertex_source(nir_builder *b, nir_def *slot, lower_ngg_ms_state *s);
static void ms_compact_primitive_slots(nir_builder *b, nir_def *prim, nir_def *slots[3], lower_ngg_ms_state *s);

static void
emit_ms_vertex(nir_builder *b, nir_def *index, nir_def *row, bool exports, bool parameters,
               uint64_t per_vertex_outputs, lower_ngg_ms_state *s)
{
   /* Export slot of this vertex; private corners put corner c of primitive p in slot 3p + c. */
   nir_def *export_index = index;
   if (s->safe_direct) {
      if (s->fast_clean) {
         nir_def *raw = index;
         nir_if *repair = nir_push_if(b, nir_inot(b, s->fast_clean));
         nir_def *mapped = nir_u2u32(b, nir_load_shared(b, 1, 8, index, .base = s->safe_direct_map_addr));
         nir_push_else(b, repair);
         nir_pop_if(b, repair);
         index = nir_if_phi(b, mapped, raw);
      } else {
         index = nir_u2u32(b, nir_load_shared(b, 1, 8, index, .base = s->safe_direct_map_addr));
      }
   } else if (s->compact_active) {
      /* RADV_BC250_MESH_COMPACT: slot `index` exports the expanded vertex that created it. */
      index = ms_compact_vertex_source(b, index, s);
   } else if (s->pack_triangle_vertices) {
      nir_def *addr = nir_iadd(b, nir_imul_imm(b, s->packed_vertex_source_prim, 3),
                              nir_umod_imm(b, index, 3));
      /* RADV_BC250_MESH_IMPLICIT_TRIS: corner c of primitive p is 3p + c, i.e. addr. */
      nir_def *source = s->implicit_indices ? addr :
         nir_u2u32(b, nir_load_shared(b, 1, 8, addr, .base = s->layout.lds.indices_addr));
      index = nir_umin(b, source, nir_iadd_imm(b, s->original_vertex_count, -1));
   }
   ms_emit_arrayed_outputs(b, index, per_vertex_outputs, s);
   /* RADV_BC250_MESH_MULTIVIEW_VTX: every vertex of the draw carries the view index as its layer. */
   if (s->insert_layer_output && s->options->multiview_layer_per_vertex) {
      s->out.outputs[VARYING_SLOT_LAYER][0] = nir_load_view_index(b);
      s->out.infos[VARYING_SLOT_LAYER].as_sysval_mask |= 1;
   }

   /* BC250 barycentrics: the reference slots carry the position just loaded. */
   if (s->bary_ref_mask && parameters) {
      if (util_bitcount64(s->bary_ref_mask) == 1) {
         /* RADV_BC250_BARY_CORNER_ID: one slot with the private corner number. */
         const unsigned slot = ffsll(s->bary_ref_mask) - 1;
         s->out.outputs[slot][0] = nir_umod_imm(b, export_index, 3);
         for (unsigned c = 1; c < 4; c++)
            s->out.outputs[slot][c] = nir_imm_int(b, 0);
      } else {
         u_foreach_bit64(slot, s->bary_ref_mask) {
            for (unsigned c = 0; c < 4; c++)
               s->out.outputs[slot][c] = s->out.outputs[VARYING_SLOT_POS][c];
         }
      }
      per_vertex_outputs |= s->bary_ref_mask;
   }

   if (exports) {
      ac_nir_export_position(b, s->ac->gfx_level, s->options->export_clipdist_mask, s->dont_export_cull, false,
                             !s->has_param_exports, false, s->per_vertex_outputs | VARYING_BIT_POS,
                             &s->out, row);
   }

   if (parameters) {
      /* Export generic attributes when there is no attribute ring. */
      if (s->has_param_exports && !s->ac->has_attr_ring) {
         ac_nir_export_parameters(b, s->vs_output_param_offset, per_vertex_outputs, 0, &s->out);
      }

      /* Also store special outputs to the attribute ring so PS can load them. */
      if (s->ac->has_attr_ring && (per_vertex_outputs & MS_VERT_ARG_EXP_MASK))
         ms_emit_attribute_ring_output_stores(b, per_vertex_outputs & MS_VERT_ARG_EXP_MASK, index, s);
   }
}

static void
emit_ms_primitive(nir_builder *b, nir_def *index, nir_def *row, bool exports, bool parameters,
                  uint64_t per_primitive_outputs, lower_ngg_ms_state *s)
{
   nir_def *export_index = index;
   /* Export lane stays compact; all source attributes and indices follow
    * the surviving original primitive. No in-place array overwrite. */
   if (s->compact_cull)
      index = s->compact_source_index;
   ms_emit_arrayed_outputs(b, index, per_primitive_outputs, s);

   /* Insert layer output store if the pipeline uses multiview but the API shader doesn't write it. */
   if (s->insert_layer_output && !s->options->multiview_layer_per_vertex) {
      s->out.outputs[VARYING_SLOT_LAYER][0] = nir_load_view_index(b);
      s->out.infos[VARYING_SLOT_LAYER].as_sysval_mask |= 1;
   }

   if (exports) {
      const uint64_t outputs_mask = per_primitive_outputs & MS_PRIM_ARG_EXP_MASK;
      nir_def *num_vtx = nir_load_var(b, s->vertex_count_var);
      nir_def *prim_exp_arg_ch1;
      /* RADV_BC250_MESH_COMPACT takes precedence: its renumbered connectivity replaces
       * the expanded private-vertex connectivity (3j, 3j+1, 3j+2), whether stored or
       * derived (RADV_BC250_MESH_IMPLICIT_TRIS, ms_prim_exp_arg_ch1), in every workgroup
       * of a shader it applies to. */
      if (s->compact_active) {
         nir_def *indices[3];
         ms_compact_primitive_slots(b, export_index, indices, s);
         prim_exp_arg_ch1 = ac_nir_pack_ngg_prim_exp_arg(b, 3, indices, NULL, s->ac->gfx_level);
      } else if (s->pack_triangle_vertices) {
         nir_def *first = nir_imul_imm(b, export_index, 3);
         nir_def *indices[3] = {first, nir_iadd_imm(b, first, 1), nir_iadd_imm(b, first, 2)};
         prim_exp_arg_ch1 = ac_nir_pack_ngg_prim_exp_arg(b, 3, indices, NULL, s->ac->gfx_level);
      } else {
         prim_exp_arg_ch1 = ms_prim_exp_arg_ch1(b, index, num_vtx, s);
      }
      nir_def *prim_exp_arg_ch2 = ms_prim_exp_arg_ch2(b, outputs_mask, s);

      nir_def *prim_exp_arg = prim_exp_arg_ch2 ?
         nir_vec2(b, prim_exp_arg_ch1, prim_exp_arg_ch2) : prim_exp_arg_ch1;

      ac_nir_export_primitive(b, prim_exp_arg, row);
   }

   if (parameters) {
      /* Export generic attributes when there is no attribute ring. */
      if (s->has_param_exports && !s->ac->has_attr_ring) {
         ac_nir_export_parameters(b, s->vs_output_param_offset, per_primitive_outputs, 0, &s->out);
      }

      /* Also store special outputs to the attribute ring so PS can load them. */
      if (s->ac->has_attr_ring)
         ms_emit_attribute_ring_output_stores(b, per_primitive_outputs & MS_PRIM_ARG_EXP_MASK, index, s);
   }
}

static void
emit_ms_outputs(nir_builder *b, nir_def *invocation_index, nir_def *row_start,
                nir_def *count, bool exports, bool parameters, uint64_t mask,
                void (*cb)(nir_builder *, nir_def *, nir_def *, bool, bool,
                           uint64_t, lower_ngg_ms_state *),
                lower_ngg_ms_state *s)
{
   if (cb == &emit_ms_primitive ? s->prim_multirow_export : s->vert_multirow_export) {
      assert(s->hw_workgroup_size % s->wave_size == 0);
      const unsigned num_waves = s->hw_workgroup_size / s->wave_size;

      nir_loop *row_loop = nir_push_loop(b);
      {
         nir_block *preheader = nir_cf_node_as_block(nir_cf_node_prev(&row_loop->cf_node));

         nir_phi_instr *index = nir_phi_instr_create(b->shader);
         nir_phi_instr *row = nir_phi_instr_create(b->shader);
         nir_def_init(&index->instr, &index->def, 1, 32);
         nir_def_init(&row->instr, &row->def, 1, 32);

         nir_phi_instr_add_src(index, preheader, invocation_index);
         nir_phi_instr_add_src(row, preheader, row_start);

         nir_break_if(b, nir_uge(b, &index->def, count));

         cb(b, &index->def, &row->def, exports, parameters, mask, s);

         nir_block *body = nir_cursor_current_block(b->cursor);
         nir_phi_instr_add_src(index, body,
                               nir_iadd_imm(b, &index->def, s->hw_workgroup_size));
         nir_phi_instr_add_src(row, body,
                               nir_iadd_imm(b, &row->def, num_waves));

         nir_instr_insert_before_cf_list(&row_loop->body, &row->instr);
         nir_instr_insert_before_cf_list(&row_loop->body, &index->instr);
      }
      nir_pop_loop(b, row_loop);
   } else {
      nir_def *has_output = nir_ilt(b, invocation_index, count);
      nir_if *if_has_output = nir_push_if(b, has_output);
      {
         cb(b, invocation_index, row_start, exports, parameters, mask, s);
      }
      nir_pop_if(b, if_has_output);
   }
}

/* Deliberately bounded prototype: every lane computes the same prefix counts
 * from LDS, then selects the original primitive for its compact export slot.
 * This is O(max_primitives) per lane, not a production performance algorithm.
 * The existing finale barrier precedes this helper. */
static void
ms_compact_culled_single_wave(nir_builder *b, lower_ngg_ms_state *s)
{
   nir_def *old_count = nir_load_var(b, s->primitive_count_var);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *count = nir_imm_int(b, 0);
   nir_def *source = nir_imm_int(b, 0);
   nir_def *vertex_source_prim = nir_imm_int(b, 0);
   nir_def *vertex_primitive_slot = s->pack_triangle_vertices ? nir_udiv_imm(b, lane, 3) : NULL;
   for (unsigned i = 0; i < b->shader->info.mesh.max_primitives_out; ++i) {
      nir_def *flag = nir_load_shared(b, 1, 8, nir_imm_int(b, i),
                                     .base = s->layout.lds.cull_flags_addr);
      nir_def *keep = nir_iand(b, nir_ult(b, nir_imm_int(b, i), old_count),
                               nir_ieq_imm(b, flag, 0));
      source = nir_bcsel(b, nir_iand(b, keep, nir_ieq(b, lane, count)),
                         nir_imm_int(b, i), source);
      if (s->pack_triangle_vertices)
         vertex_source_prim = nir_bcsel(b, nir_iand(b, keep, nir_ieq(b, vertex_primitive_slot, count)),
                                        nir_imm_int(b, i), vertex_source_prim);
      count = nir_iadd(b, count, nir_b2i32(b, keep));
   }
   s->compact_source_index = source;
   nir_store_var(b, s->primitive_count_var, count, 1);
   nir_def *vertices = nir_load_var(b, s->vertex_count_var);
   s->original_vertex_count = vertices;
   s->packed_vertex_source_prim = vertex_source_prim;
   if (s->pack_triangle_vertices)
      vertices = nir_imul_imm(b, count, 3);
   nir_store_var(b, s->vertex_count_var,
                 nir_bcsel(b, nir_ieq_imm(b, count, 0), nir_imm_int(b, 0), vertices), 1);
}

/* RADV_BC250_MESH_AUTOCULL LDS area (at autocull_lds_addr): the surviving
 * primitive count of each wave that holds primitives (ms_autocull_max_waves
 * dwords), the original primitive and vertex counts, then one byte per surviving
 * primitive slot holding its original primitive index (max_primitives bytes). */
enum {
   ms_autocull_max_waves = 8,
   ms_autocull_wave_counts = 0,
   ms_autocull_old_prims = 4 * ms_autocull_max_waves,
   ms_autocull_old_vtx = ms_autocull_old_prims + 4,
   ms_autocull_map = ms_autocull_old_vtx + 4,
};

static unsigned
ms_autocull_prim_waves(nir_builder *b, lower_ngg_ms_state *s)
{
   return DIV_ROUND_UP(b->shader->info.mesh.max_primitives_out, s->wave_size);
}

/* Multi-wave primitives: the original counts are only known by wave 0 (the other
 * waves may not run the API shader). Store them before the finale barrier. */
static void
ms_autocull_store_counts(nir_builder *b, lower_ngg_ms_state *s)
{
   if (ms_autocull_prim_waves(b, s) <= 1)
      return;

   nir_if *if_first = nir_push_if(b, nir_iand(b, nir_ieq_imm(b, nir_load_subgroup_id(b), 0), nir_elect(b, 1)));
   {
      nir_def *counts = nir_vec2(b, nir_load_var(b, s->primitive_count_var), nir_load_var(b, s->vertex_count_var));
      nir_store_shared(b, counts, nir_imm_int(b, 0), .base = s->autocull_lds_addr + ms_autocull_old_prims);
   }
   nir_pop_if(b, if_first);
}

/* RADV_BC250_MESH_AUTOCULL: the clip and cull distances the epilogue culls with
 * (ms_autocull_accept), the exported ones (cull_clipdist_mask) that the shader
 * writes as per-vertex outputs. */
static uint32_t
ms_autocull_clipdist_mask(const lower_ngg_ms_state *s)
{
   uint32_t mask = 0;
   u_foreach_bit(i, s->options->cull_clipdist_mask & 0xff) {
      const unsigned slot = VARYING_SLOT_CLIP_DIST0 + i / 4;
      if ((s->per_vertex_outputs & BITFIELD64_BIT(slot)) &&
          (s->out.infos[slot].components_mask & BITFIELD_BIT(i % 4)))
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

/* Visibility of primitive `prim` (only called for existing primitives). */
static nir_def *
ms_autocull_accept(nir_builder *b, nir_def *prim, nir_def *old_vtx, lower_ngg_ms_state *s,
                   nir_def *indices)
{
   nir_def *zero = nir_imm_int(b, 0);
   /* The direct checker can supply connectivity it already loaded. */
   if (!indices) {
      if (s->implicit_indices) {
         /* RADV_BC250_MESH_IMPLICIT_TRIS */
         indices = nir_vec3(b, ms_implicit_index(b, prim, 0, s), ms_implicit_index(b, prim, 1, s),
                            ms_implicit_index(b, prim, 2, s));
      } else {
         indices = nir_u2u32(b, nir_load_shared(b, 3, 8, nir_imul_imm(b, prim, 3),
                                                .base = s->layout.lds.indices_addr));
      }
   }
   nir_def *max_vtx_idx = nir_iadd_imm(b, old_vtx, -1);
   nir_def *pos[3][4] = {0};
   nir_def *vtx[3];
   for (unsigned v = 0; v < 3; ++v) {
      vtx[v] = nir_umin(b, nir_channel(b, indices, v), max_vtx_idx);
      nir_def *p = ms_load_arrayed_output(b, vtx[v], zero, VARYING_SLOT_POS, 0, 4, 32, s);
      pos[v][3] = nir_channel(b, p, 3);
      pos[v][0] = nir_fdiv(b, nir_channel(b, p, 0), pos[v][3]);
      pos[v][1] = nir_fdiv(b, nir_channel(b, p, 1), pos[v][3]);
   }

   /* Clip and cull distances (cull_clipdist_mask: every exported clip and cull
    * distance, bit i = component i % 4 of CLIP_DIST0 + i / 4, the numbering of
    * ac_nir_export_position): a triangle with one distance negative at all three
    * corners is entirely outside that clip half-space or culled, as in VS NGG
    * culling (add_clipdist_bit, ac_nir_lower_ngg.c). The survivors still export
    * all their distances, so the hardware keeps clipping and culling with them;
    * this only drops triangles that it would drop. */
   nir_def *initially_accepted = nir_imm_true(b);
   const uint32_t clipdist_mask = ms_autocull_clipdist_mask(s);
   if (clipdist_mask) {
      nir_def *all_negative = nir_imm_int(b, clipdist_mask);
      for (unsigned v = 0; v < 3; ++v) {
         nir_def *negative = nir_imm_int(b, 0);
         u_foreach_bit(i, clipdist_mask) {
            nir_def *dist = ms_load_arrayed_output(b, vtx[v], zero, VARYING_SLOT_CLIP_DIST0 + i / 4, i % 4, 1, 32, s);
            negative = nir_ior(b, negative, nir_ishl_imm(b, nir_b2i32(b, nir_flt_imm(b, dist, 0)), i));
         }
         all_negative = nir_iand(b, all_negative, negative);
      }
      initially_accepted = nir_ieq_imm(b, all_negative, 0);
   }

   /* Only reached on the culling side of emit_ms_finale's runtime branch (backface
    * culling enabled), so no further check of the culling settings is needed. */
   if (!s->uses_cull_flags)
      return ac_nir_cull_primitive(b, s->autocull_skip_viewport_state,
                                   s->options->use_point_tri_intersection,
                                   initially_accepted, pos, 3, NULL, NULL);

   nir_def *flag = nir_load_shared(b, 1, 8, prim, .base = s->layout.lds.cull_flags_addr);
   nir_def *app_accepted = nir_ieq_imm(b, flag, 0);
   nir_def *culled_accepted;
   nir_if *if_app = nir_push_if(b, app_accepted);
   {
      culled_accepted = ac_nir_cull_primitive(b, s->autocull_skip_viewport_state,
                                              s->options->use_point_tri_intersection,
                                              initially_accepted, pos, 3, NULL, NULL);
   }
   nir_pop_if(b, if_app);
   return nir_if_phi(b, culled_accepted, app_accepted);
}

/* RADV_BC250_MESH_AUTOCULL: driver-side per-triangle culling of the Mesh output.
 *
 * Runs after the finale barrier, when every position, the primitive indices and
 * the application's CullPrimitive flags are in LDS. Lane p (of the whole
 * workgroup) owns primitive p: it loads the three positions and runs the same
 * tests as VS NGG culling (ac_nir_cull_primitive: w <= 0, backface per the
 * runtime cull mode / front face, frustum, small primitive), all driven by the
 * NGG culling settings user SGPRs, so dynamic cull mode, front face, rasterizer
 * discard and conservative rasterization behave as on the VS path. An
 * application-culled primitive stays culled. Surviving primitives get compact
 * slots in their original order (ballot + mbcnt, plus the counts of the lower
 * waves when the primitives span several waves) and the slot -> primitive map
 * goes to LDS; after a workgroup barrier every lane reads the surviving count and
 * its source primitive. The export then uses the existing compaction +
 * packed-triangle-vertices path: primitive slot j exports vertices 3j, 3j+1,
 * 3j+2, copies of the original corners in their original order, so every
 * exported vertex is private and referenced (the expanded shape class), the
 * provoking vertex and the winding are unchanged, and per-primitive outputs
 * follow their primitive. A workgroup whose triangles are all culled allocates
 * 0/0 (on GFX10 the fully-culled workaround's dummy).
 *
 * Only runs on the culling side of emit_ms_finale's runtime branch, i.e. when
 * front or back face culling is enabled at draw time. Frustum and small-primitive
 * culling apply there as well, but never alone.
 */
static void
ms_autocull_compact(nir_builder *b, lower_ngg_ms_state *s)
{
   const unsigned base = s->autocull_lds_addr;
   const unsigned max_primitives = b->shader->info.mesh.max_primitives_out;
   const unsigned prim_waves = ms_autocull_prim_waves(b, s);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *wave_id = nir_load_subgroup_id(b);
   nir_def *not_a_primitive = nir_imm_false(b);

   if (prim_waves <= 1) {
      /* Every primitive is in wave 0: one barrier. */
      nir_if *if_wave_0 = nir_push_if(b, nir_ieq_imm(b, wave_id, 0));
      {
         nir_def *old_prims = nir_load_var(b, s->primitive_count_var);
         nir_def *old_vtx = nir_load_var(b, s->vertex_count_var);
         /* RADV_BC250_MESH_COMPACT: never more survivors than declared primitives. */
         if (s->compact)
            old_prims = nir_umin(b, old_prims, nir_imm_int(b, max_primitives));
         nir_def *accepted;
         nir_if *if_prim = nir_push_if(b, nir_ult(b, lane, old_prims));
         {
            accepted = ms_autocull_accept(b, lane, old_vtx, s, NULL);
         }
         nir_pop_if(b, if_prim);
         accepted = nir_if_phi(b, accepted, not_a_primitive);

         nir_def *mask = nir_ballot(b, 1, s->wave_size, accepted);
         nir_def *slot = nir_mbcnt_amd(b, mask, zero);
         nir_if *if_accepted = nir_push_if(b, accepted);
         {
            nir_store_shared(b, nir_u2u8(b, lane), slot, .base = base + ms_autocull_map);
         }
         nir_pop_if(b, if_accepted);

         nir_if *if_elected = nir_push_if(b, nir_elect(b, 1));
         {
            nir_store_shared(b, nir_u2u32(b, nir_bit_count(b, mask)), zero, .base = base + ms_autocull_wave_counts);
            nir_store_shared(b, nir_vec2(b, old_prims, old_vtx), zero, .base = base + ms_autocull_old_prims);
         }
         nir_pop_if(b, if_elected);
      }
      nir_pop_if(b, if_wave_0);
   } else {
      /* The original counts were stored before the finale barrier. */
      nir_def *old = nir_load_shared(b, 2, 32, zero, .base = base + ms_autocull_old_prims);
      nir_def *old_prims = nir_channel(b, old, 0);
      nir_def *old_vtx = nir_channel(b, old, 1);
      if (s->compact)
         old_prims = nir_umin(b, old_prims, nir_imm_int(b, max_primitives));
      nir_def *accepted;
      nir_if *if_prim = nir_push_if(b, nir_iand(b, nir_ult(b, lane, old_prims),
                                                nir_ult_imm(b, lane, max_primitives)));
      {
         accepted = ms_autocull_accept(b, lane, old_vtx, s, NULL);
      }
      nir_pop_if(b, if_prim);
      accepted = nir_if_phi(b, accepted, not_a_primitive);

      nir_def *mask = nir_ballot(b, 1, s->wave_size, accepted);
      nir_if *if_count = nir_push_if(b, nir_iand(b, nir_ult_imm(b, wave_id, prim_waves), nir_elect(b, 1)));
      {
         nir_store_shared(b, nir_u2u32(b, nir_bit_count(b, mask)), nir_imul_imm(b, wave_id, 4),
                          .base = base + ms_autocull_wave_counts);
      }
      nir_pop_if(b, if_count);

      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);

      /* Slot = surviving primitives of the lower waves + those below this lane. */
      nir_def *slot = nir_mbcnt_amd(b, mask, zero);
      for (unsigned w = 0; w + 1 < prim_waves; ++w) {
         nir_def *count = nir_load_shared(b, 1, 32, zero, .base = base + ms_autocull_wave_counts + 4 * w);
         slot = nir_iadd(b, slot, nir_bcsel(b, nir_ult(b, nir_imm_int(b, w), wave_id), count, zero));
      }
      nir_if *if_accepted = nir_push_if(b, accepted);
      {
         nir_store_shared(b, nir_u2u8(b, lane), slot, .base = base + ms_autocull_map);
      }
      nir_pop_if(b, if_accepted);
   }

   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);

   nir_def *count = zero;
   for (unsigned w = 0; w < prim_waves; ++w)
      count = nir_iadd(b, count, nir_load_shared(b, 1, 32, zero, .base = base + ms_autocull_wave_counts + 4 * w));
   nir_def *max_slot = nir_imm_int(b, max_primitives - 1);

   /* Slot j < count holds the j-th surviving primitive; lanes beyond the counts
    * read a stale slot and never export it. */
   s->compact_source_index =
      nir_u2u32(b, nir_load_shared(b, 1, 8, nir_umin(b, lane, max_slot), .base = base + ms_autocull_map));
   s->packed_vertex_source_prim =
      nir_u2u32(b, nir_load_shared(b, 1, 8, nir_umin(b, nir_udiv_imm(b, lane, 3), max_slot),
                                   .base = base + ms_autocull_map));
   s->original_vertex_count = nir_load_shared(b, 1, 32, zero, .base = base + ms_autocull_old_vtx);

   nir_store_var(b, s->primitive_count_var, count, 1);
   nir_store_var(b, s->vertex_count_var,
                 nir_bcsel(b, nir_ieq_imm(b, count, 0), zero, nir_imul_imm(b, count, 3)), 1);
}

/* RADV_BC250_MESH_COMPACT: GE-like shared-vertex renumbering of the expanded Mesh output.
 *
 * The base driver's expanded shaders (radv_bc250_expand_primitive_attributes) export
 * primitive j as private vertices (3j, 3j+1, 3j+2); expanded vertex 3p+c copies the
 * application's logical vertex L = indices[p][c] (per-vertex outputs) and primitive p's
 * per-primitive outputs (flat). GFX1013 hangs on raw shared-vertex connectivity unless
 * (MESH_PERF/rawroot) (all referenced) every exported vertex is referenced by an exported
 * primitive and (window) no corner references a vertex more than W below the highest vertex index
 * referenced before it (the backjump; W = AC_NIR_BC250_COMPACT_W = 8 is proven).
 *
 * The epilogue here (after autocull, when on) exports shared vertices instead, with a
 * layout that satisfies both rules for any index data:
 *  - key k = 3j + c is corner c of the j-th exported primitive (source primitive p =
 *    j, or the j-th autocull survivor), in the order the primitives are exported; key
 *    lane k (k < 3P) reads L_k from the application's staged indices;
 *  - anchors, AC_NIR_BC250_COMPACT_LEVELS levels: level 1 anchor first1[L] = the smallest
 *    key with that L (LDS atomic umin); a key within D = W + 1 keys of it (k - first1 <= D)
 *    takes it as its anchor; the others go to level 2, whose anchor first2[L] is the
 *    smallest remaining key with that L, and so on; a key left after the last level is
 *    its own anchor. Every anchor is its own anchor (distance 0);
 *  - key k is FRESH (gets a new vertex slot) when it is its own anchor, when L_k is not a
 *    declared logical vertex (garbage: private copy), or when corner c owns the
 *    primitive's flat payload (per-primitive outputs: the provoking corner, 0 or 2, both
 *    for a dynamic provoking mode);
 *  - slot(k) = number of fresh keys below k (ballot + mbcnt + per-wave counts);
 *    V' = number of fresh keys; slot s exports the expanded vertex of the fresh key
 *    that created it (same values as the expanded export of that corner);
 *  - a non-fresh key references the slot of its anchor (an anchor is fresh).
 * All referenced: slot s is referenced by its creating key's primitive; no null primitive exists
 * (the lowering refuses CullPrimitive) and P = 0 takes the GFX10 fully-culled dummy.
 * Window: the highest slot referenced before key k is (#fresh keys < k) - 1, so the
 * backjump of a non-fresh key is #fresh keys in [anchor, k) - 1 <= k - anchor - 1 <= W.
 * V' <= 3P <= the workgroup lanes: the expanded launch (3P lanes) is unchanged, and a
 * workgroup where every key is fresh is exactly the expanded (private-vertex) shape.
 * Primitive order, corner order (winding, provoking vertex) and every exported value
 * of every exported corner are those of the expanded export.
 *
 * The anchor tables are cleared before the finale barrier; an anchor is only taken when the
 * key it names has the same logical vertex (keyl[], written after the barrier), so even a
 * table that an undefined application store overwrote can only make keys fresh.
 *
 * LDS area (compact_lds_addr): the clamped primitive count, one fresh count per wave,
 * one anchor table per level (u32 per declared logical vertex), then u16 arrays refkey[3P] (the key whose
 * slot a key references), slot[3P] (slot of a fresh key), map[3P] (slot -> expanded
 * vertex) and keyl[3P] (logical vertex of a key). */
enum {
   ms_compact_prims = 0,
   ms_compact_wave_counts = 4,
   ms_compact_first = 4 + 4 * ms_autocull_max_waves,
};

static unsigned
ms_compact_refkey_addr(const lower_ngg_ms_state *s)
{
   return s->compact_lds_addr + ms_compact_first + 4 * AC_NIR_BC250_COMPACT_LEVELS * s->compact_table;
}

static unsigned
ms_compact_slot_addr(nir_builder *b, const lower_ngg_ms_state *s)
{
   return ms_compact_refkey_addr(s) + 2 * 3 * b->shader->info.mesh.max_primitives_out;
}

static unsigned
ms_compact_map_addr(nir_builder *b, const lower_ngg_ms_state *s)
{
   return ms_compact_slot_addr(b, s) + 2 * 3 * b->shader->info.mesh.max_primitives_out;
}

static unsigned
ms_compact_keyl_addr(nir_builder *b, const lower_ngg_ms_state *s)
{
   return ms_compact_map_addr(b, s) + 2 * 3 * b->shader->info.mesh.max_primitives_out;
}

static unsigned
ms_compact_lds_size(unsigned table, unsigned max_primitives)
{
   return ms_compact_first + 4 * AC_NIR_BC250_COMPACT_LEVELS * table + 4 * 2 * 3 * max_primitives;
}

static void
ms_compact_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
}

/* Before the finale barrier: wave 0 (an API wave) publishes the clamped primitive count
 * and every lane clears its part of the first-use table. */
static void
ms_compact_prepare(nir_builder *b, lower_ngg_ms_state *s)
{
   const unsigned max_primitives = b->shader->info.mesh.max_primitives_out;
   nir_if *if_first = nir_push_if(b, nir_iand(b, nir_ieq_imm(b, nir_load_subgroup_id(b), 0), nir_elect(b, 1)));
   {
      nir_def *prims = nir_umin(b, nir_load_var(b, s->primitive_count_var), nir_imm_int(b, max_primitives));
      nir_store_shared(b, prims, nir_imm_int(b, 0), .base = s->compact_lds_addr + ms_compact_prims);
   }
   nir_pop_if(b, if_first);

   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *ones = nir_imm_int(b, -1);
   const unsigned entries = AC_NIR_BC250_COMPACT_LEVELS * s->compact_table;
   for (unsigned i = 0; i < entries; i += s->hw_workgroup_size) {
      nir_def *entry = nir_iadd_imm(b, lane, i);
      nir_if *if_entry = nir_push_if(b, nir_ult_imm(b, entry, entries));
      nir_store_shared(b, ones, nir_imul_imm(b, entry, 4), .base = s->compact_lds_addr + ms_compact_first);
      nir_pop_if(b, if_entry);
   }
}

/* After the finale barrier (and autocull): the renumbering. Sets the output counts. */
static void
ms_compact_vertices(nir_builder *b, lower_ngg_ms_state *s, bool culled)
{
   const unsigned max_primitives = b->shader->info.mesh.max_primitives_out;
   const unsigned num_waves = DIV_ROUND_UP(s->hw_workgroup_size, s->wave_size);
   const uint32_t *ix = s->options->bc250_compact_index_staging;
   const unsigned d = AC_NIR_BC250_COMPACT_W + 1;
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *wave_id = nir_load_subgroup_id(b);
   assert(num_waves <= ms_autocull_max_waves);

   /* Primitives exported (already clamped) and the source primitive of each key lane. */
   nir_def *prims, *src_prim;
   if (culled) {
      prims = nir_umin(b, nir_load_var(b, s->primitive_count_var), nir_imm_int(b, max_primitives));
      src_prim = s->packed_vertex_source_prim;
   } else {
      prims = nir_load_shared(b, 1, 32, zero, .base = s->compact_lds_addr + ms_compact_prims);
      src_prim = nir_umin(b, nir_udiv_imm(b, lane, 3), nir_imm_int(b, max_primitives - 1));
   }
   prims = nir_read_first_invocation(b, prims);
   nir_def *corner = nir_umod_imm(b, lane, 3);
   nir_def *is_key = nir_ult(b, lane, nir_imul_imm(b, prims, 3));

   /* The logical vertex of the key. */
   nir_def *logical, *valid;
   nir_def *not_valid = nir_imm_false(b);
   nir_if *if_key = nir_push_if(b, is_key);
   {
      nir_def *addr = nir_iadd(b, nir_imul_imm(b, src_prim, ix[1]), nir_imul_imm(b, corner, ix[2]));
      logical = nir_u2u32(b, nir_load_shared(b, 1, ix[2] * 8, addr, .base = ix[0], .align_mul = ix[2]));
      valid = nir_ult_imm(b, logical, s->compact_table);
      nir_store_shared(b, nir_u2u16(b, logical), nir_imul_imm(b, lane, 2), .base = ms_compact_keyl_addr(b, s),
                       .align_mul = 2);
   }
   nir_pop_if(b, if_key);
   logical = nir_if_phi(b, logical, zero);
   valid = nir_if_phi(b, valid, not_valid);

   /* Anchors, level by level: a pending key competes for the smallest key of its logical
    * vertex among the pending keys; it keeps that anchor when it lies within D keys. */
   nir_def *anchor = lane;
   nir_def *pending = valid;
   for (unsigned level = 0; level < AC_NIR_BC250_COMPACT_LEVELS; ++level) {
      const unsigned table = s->compact_lds_addr + ms_compact_first + 4 * level * s->compact_table;
      nir_if *if_pending = nir_push_if(b, pending);
      nir_shared_atomic(b, 32, nir_imul_imm(b, logical, 4), lane, .base = table, .atomic_op = nir_atomic_op_umin);
      nir_pop_if(b, if_pending);

      ms_compact_barrier(b);

      nir_def *first, *same;
      nir_def *no = nir_imm_false(b);
      if_pending = nir_push_if(b, pending);
      {
         first = nir_load_shared(b, 1, 32, nir_imul_imm(b, logical, 4), .base = table);
         /* The named key must be a key of the same logical vertex (else: fresh). */
         nir_def *key = nir_umin(b, first, nir_imm_int(b, 3 * max_primitives - 1));
         nir_def *l = nir_u2u32(b, nir_load_shared(b, 1, 16, nir_imul_imm(b, key, 2),
                                                   .base = ms_compact_keyl_addr(b, s), .align_mul = 2));
         same = nir_iand(b, nir_ieq(b, l, nir_iand_imm(b, logical, 0xffff)), nir_uge(b, lane, first));
      }
      nir_pop_if(b, if_pending);
      first = nir_if_phi(b, first, lane);
      same = nir_if_phi(b, same, no);
      /* A key within D keys of its anchor (first <= lane: the key itself competed). */
      nir_def *near = nir_iand(b, same, nir_ule_imm(b, nir_isub(b, lane, first), d));
      /* A key whose table entry does not name a key of its vertex stops competing: fresh. */
      nir_def *broken = nir_iand(b, pending, nir_inot(b, same));
      anchor = nir_bcsel(b, near, first, anchor);
      pending = nir_iand(b, pending, nir_inot(b, nir_ior(b, near, broken)));
   }

   /* Fresh keys and the key each key references. */
   const uint8_t owned = s->options->bc250_compact_owned_corners & 0x5;
   nir_def *owner = nir_ine_imm(b, nir_iand_imm(b, nir_ushr(b, nir_imm_int(b, owned), corner), 1), 0);
   nir_def *fresh = nir_iand(b, is_key, nir_ior(b, nir_ieq(b, anchor, lane), owner));
   nir_def *first = anchor;
   nir_def *refkey = nir_bcsel(b, fresh, lane, first);
   nir_if *if_key2 = nir_push_if(b, is_key);
   nir_store_shared(b, nir_u2u16(b, refkey), nir_imul_imm(b, lane, 2), .base = ms_compact_refkey_addr(s),
                    .align_mul = 2);
   nir_pop_if(b, if_key2);

   nir_def *mask = nir_ballot(b, 1, s->wave_size, fresh);
   nir_if *if_elected = nir_push_if(b, nir_elect(b, 1));
   nir_store_shared(b, nir_u2u32(b, nir_bit_count(b, mask)), nir_imul_imm(b, wave_id, 4),
                    .base = s->compact_lds_addr + ms_compact_wave_counts);
   nir_pop_if(b, if_elected);

   ms_compact_barrier(b);

   /* Slots: fresh keys of the lower waves + those below this lane. */
   nir_def *slot = nir_mbcnt_amd(b, mask, zero);
   nir_def *total = zero;
   for (unsigned w = 0; w < num_waves; ++w) {
      nir_def *count = nir_load_shared(b, 1, 32, zero, .base = s->compact_lds_addr + ms_compact_wave_counts + 4 * w);
      slot = nir_iadd(b, slot, nir_bcsel(b, nir_ult(b, nir_imm_int(b, w), wave_id), count, zero));
      total = nir_iadd(b, total, count);
   }
   nir_if *if_fresh = nir_push_if(b, fresh);
   {
      nir_def *expanded = nir_iadd(b, nir_imul_imm(b, src_prim, 3), corner);
      nir_store_shared(b, nir_u2u16(b, slot), nir_imul_imm(b, lane, 2), .base = ms_compact_slot_addr(b, s),
                       .align_mul = 2);
      nir_store_shared(b, nir_u2u16(b, expanded), nir_imul_imm(b, slot, 2), .base = ms_compact_map_addr(b, s),
                       .align_mul = 2);
   }
   nir_pop_if(b, if_fresh);

   ms_compact_barrier(b);

   total = nir_read_first_invocation(b, total);
   nir_store_var(b, s->primitive_count_var, prims, 1);
   nir_store_var(b, s->vertex_count_var, nir_bcsel(b, nir_ieq_imm(b, prims, 0), zero, total), 1);
   s->compact_active = true;
   s->counts_in_all_lanes = true;
}

static nir_def *
ms_compact_vertex_source(nir_builder *b, nir_def *slot, lower_ngg_ms_state *s)
{
   return nir_u2u32(b, nir_load_shared(b, 1, 16, nir_imul_imm(b, slot, 2), .base = ms_compact_map_addr(b, s),
                                       .align_mul = 2));
}

static void
ms_compact_primitive_slots(nir_builder *b, nir_def *prim, nir_def *slots[3], lower_ngg_ms_state *s)
{
   for (unsigned c = 0; c < 3; ++c) {
      nir_def *key = nir_iadd_imm(b, nir_imul_imm(b, prim, 3), c);
      nir_def *refkey = nir_u2u32(b, nir_load_shared(b, 1, 16, nir_imul_imm(b, key, 2),
                                                     .base = ms_compact_refkey_addr(s), .align_mul = 2));
      slots[c] = nir_u2u32(b, nir_load_shared(b, 1, 16, nir_imul_imm(b, refkey, 2),
                                              .base = ms_compact_slot_addr(b, s), .align_mul = 2));
   }
}

/* Parallel source-key chains. A copy anchored at key a may serve triangle p
 * only while 3*p+2-a <= 31. Ranking the live anchors can only shorten this
 * distance, including the last corner of p. Each anchor is itself referenced.
 * V<=32 needs no expiry: its complete first-use map fits in the window.
 *
 * Logical vertices own independent chains; no lane serializes all allocations.
 * The conservative source-key window may duplicate more than exact closure.
 */
static nir_def *
ms_safe_parallel_rank(nir_builder *b, nir_def *key, nir_def **words)
{
   nir_def *word = nir_ushr_imm(b, key, 5);
   nir_def *bit = nir_iand_imm(b, key, 31);
   nir_def *below = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), bit), -1);
   nir_def *rank = nir_imm_int(b, 0);
   for (unsigned i = 0; i < 8; ++i) {
      nir_def *mask = nir_bcsel(b, nir_ult_imm(b, word, i), nir_imm_int(b, 0),
                              nir_bcsel(b, nir_ieq_imm(b, word, i),
                                        nir_iand(b, words[i], below), words[i]));
      rank = nir_iadd(b, rank, nir_bit_count(b, mask));
   }
   return rank;
}

static void ms_safe_stats_add(nir_builder *b, lower_ngg_ms_state *s, unsigned field, nir_def *value);

static void
ms_safe_parallel_vertices(nir_builder *b, lower_ngg_ms_state *s)
{
   const unsigned vmax = b->shader->info.mesh.max_vertices_out;
   const unsigned pmax = b->shader->info.mesh.max_primitives_out;
   const unsigned refs = s->safe_direct_latest_addr;
   const unsigned masks = refs + align(3 * pmax, 4);
   const unsigned live_masks = s->safe_direct_counts_addr + 8;
   const bool cull = s->options->bc250_safe_autocull;
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *planner_start = s->options->bc250_safe_stats ?
      nir_channel(b, nir_shader_clock(b, SCOPE_SUBGROUP), 0) : NULL;
   nir_if *first = nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_def *v0 = nir_umin_imm(b, nir_load_var(b, s->vertex_count_var), vmax);
   nir_def *p0 = nir_umin_imm(b, nir_load_var(b, s->primitive_count_var), pmax);
   p0 = nir_bcsel(b, nir_ieq_imm(b, v0, 0), zero, p0);
   nir_store_shared(b, nir_vec2(b, v0, p0), zero,
                    .base = s->safe_direct_counts_addr, .align_mul = 4);
   nir_pop_if(b, first);
   nir_if *init = nir_push_if(b, nir_ult_imm(b, lane, 8));
   nir_store_shared(b, zero, nir_imul_imm(b, lane, 4), .base = masks, .align_mul = 4);
   nir_pop_if(b, init);
   if (cull) {
      nir_if *clear = nir_push_if(b, nir_ult_imm(b, lane, 3));
      nir_store_shared(b, zero, nir_imul_imm(b, lane, 4), .base = live_masks, .align_mul = 4);
      nir_pop_if(b, clear);
   }
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   nir_def *counts = nir_load_shared(b, 2, 32, zero,
                                    .base = s->safe_direct_counts_addr, .align_mul = 4);
   nir_def *vc = nir_channel(b, counts, 0), *pc = nir_channel(b, counts, 1);
   nir_def *limit = nir_iadd_imm(b, nir_umax_imm(b, vc, 1), -1);
   nir_def *survivors[8] = {zero, zero, zero, zero, zero, zero, zero, zero};
   nir_def *live_count = zero;
   if (cull) {
      s->safe_query_prims = pc;
      nir_if *prim = nir_push_if(b, nir_ult(b, lane, pc));
      nir_def *keep = nir_imm_true(b);
      nir_if *enabled = nir_push_if(b, nir_load_cull_any_enabled_amd(b));
      nir_def *accepted = ms_autocull_accept(b, lane, vc, s, NULL);
      nir_pop_if(b, enabled);
      accepted = nir_if_phi(b, accepted, keep);
      nir_if *live = nir_push_if(b, accepted);
      nir_shared_atomic(b, 32, nir_imul_imm(b, nir_ushr_imm(b, lane, 5), 4),
                        nir_ishl(b, nir_imm_int(b, 1), nir_iand_imm(b, lane, 31)),
                        .base = live_masks, .atomic_op = nir_atomic_op_ior);
      nir_pop_if(b, live);
      nir_pop_if(b, prim);
      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
      for (unsigned i = 0; i < 3; ++i) {
         survivors[i] = nir_load_shared(b, 1, 32, zero, .base = live_masks + 4 * i, .align_mul = 4);
         live_count = nir_iadd(b, live_count, nir_bit_count(b, survivors[i]));
      }
   }

   for (unsigned base = 0; base < vmax; base += s->hw_workgroup_size) {
      nir_def *v = nir_iadd_imm(b, lane, base);
      nir_if *active = nir_push_if(b, nir_ult(b, v, vc));
      nir_variable *anchor = nir_local_variable_create(b->impl, glsl_uint_type(), "safe_anchor");
      nir_variable *prim = nir_local_variable_create(b->impl, glsl_uint_type(), "safe_scan");
      nir_store_var(b, anchor, nir_imm_int(b, -1), 1);
      nir_store_var(b, prim, zero, 1);
      nir_loop *loop = nir_push_loop(b);
      nir_def *p = nir_load_var(b, prim);
      nir_break_if(b, nir_uge(b, p, pc));
      nir_def *key = nir_imul_imm(b, p, 3);
      nir_def *indices = nir_u2u32(b, nir_load_shared(b, 3, 8, key, .base = s->layout.lds.indices_addr));
      nir_def *eq[3];
      for (unsigned c = 0; c < 3; ++c)
         eq[c] = nir_ieq(b, v, nir_umin(b, nir_channel(b, indices, c), limit));
      nir_def *is_live = nir_imm_true(b);
      if (cull) {
         nir_def *word = nir_load_shared(b, 1, 32, nir_imul_imm(b, nir_umin_imm(b, nir_ushr_imm(b, p, 5), 2), 4),
                                        .base = live_masks, .align_mul = 4);
         is_live = nir_ine_imm(b, nir_iand(b, nir_ushr(b, word, nir_iand_imm(b, p, 31)), nir_imm_int(b, 1)), 0);
      }
      nir_if *used = nir_push_if(b, nir_iand(b, is_live, nir_ior(b, eq[0], nir_ior(b, eq[1], eq[2]))));
      nir_def *old = nir_load_var(b, anchor);
      nir_def *fresh = nir_ieq_imm(b, old, -1);
      if (vmax > 32)
         fresh = nir_ior(b, fresh, nir_ugt_imm(b, nir_isub(b, nir_iadd_imm(b, key, 2), old), 31));
      nir_if *create = nir_push_if(b, fresh);
      nir_def *corner = nir_bcsel(b, eq[0], zero,
                                 nir_bcsel(b, eq[1], nir_imm_int(b, 1), nir_imm_int(b, 2)));
      nir_def *a = nir_iadd(b, key, corner);
      nir_store_var(b, anchor, a, 1);
      nir_shared_atomic(b, 32, nir_imul_imm(b, nir_ushr_imm(b, a, 5), 4),
                        nir_ishl(b, nir_imm_int(b, 1), nir_iand_imm(b, a, 31)),
                        .base = masks, .atomic_op = nir_atomic_op_ior);
      nir_pop_if(b, create);
      for (unsigned c = 0; c < 3; ++c) {
         nir_if *owns = nir_push_if(b, eq[c]);
         nir_store_shared(b, nir_u2u8(b, nir_load_var(b, anchor)), nir_iadd_imm(b, key, c), .base = refs);
         nir_pop_if(b, owns);
      }
      nir_pop_if(b, used);
      nir_store_var(b, prim, nir_iadd_imm(b, p, 1), 1);
      nir_pop_loop(b, loop);
      nir_pop_if(b, active);
   }
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   nir_def *words[8], *total = zero;
   for (unsigned i = 0; i < 8; ++i) {
      words[i] = nir_load_shared(b, 1, 32, zero, .base = masks + 4 * i, .align_mul = 4);
      total = nir_iadd(b, total, nir_bit_count(b, words[i]));
   }
   for (unsigned base = 0; base < pmax; base += s->hw_workgroup_size) {
      nir_def *p = nir_iadd_imm(b, lane, base);
      nir_def *valid_prim = nir_ult(b, p, pc);
      if (cull) {
         nir_def *word = nir_load_shared(b, 1, 32, nir_imul_imm(b, nir_umin_imm(b, nir_ushr_imm(b, p, 5), 2), 4),
                                        .base = live_masks, .align_mul = 4);
         valid_prim = nir_iand(b, valid_prim,
            nir_ine_imm(b, nir_iand(b, nir_ushr(b, word, nir_iand_imm(b, p, 31)), nir_imm_int(b, 1)), 0));
      }
      nir_if *valid = nir_push_if(b, valid_prim);
      nir_def *key = nir_imul_imm(b, p, 3);
      nir_def *a = nir_u2u32(b, nir_load_shared(b, 3, 8, key, .base = refs));
      nir_def *indices = nir_u2u32(b, nir_load_shared(b, 3, 8, key, .base = s->layout.lds.indices_addr));
      nir_def *slots[3];
      for (unsigned c = 0; c < 3; ++c) {
         nir_def *ac = nir_channel(b, a, c);
         slots[c] = ms_safe_parallel_rank(b, ac, words);
         nir_if *owns = nir_push_if(b, nir_ieq(b, ac, nir_iadd_imm(b, key, c)));
         nir_store_shared(b, nir_u2u8(b, nir_umin(b, nir_channel(b, indices, c), limit)), slots[c],
                          .base = s->safe_direct_map_addr);
         nir_pop_if(b, owns);
      }
      nir_def *output_key = cull ? nir_imul_imm(b, ms_safe_parallel_rank(b, p, survivors), 3) : key;
      nir_store_shared(b, nir_u2u8(b, nir_vec3(b, slots[0], slots[1], slots[2])), output_key,
                       .base = s->safe_direct_indices_addr);
      nir_pop_if(b, valid);
   }
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   if (s->options->bc250_safe_stats) {
      nir_if *leader = nir_push_if(b, nir_ieq_imm(b, lane, 0));
      nir_def *planner_end = nir_channel(b, nir_shader_clock(b, SCOPE_SUBGROUP), 0);
      ms_safe_stats_add(b, s, 5, nir_isub(b, planner_end, planner_start));
      nir_pop_if(b, leader);
   }
   nir_store_var(b, s->vertex_count_var, total, 1);
   nir_store_var(b, s->primitive_count_var, cull ? live_count : pc, 1);
   s->counts_in_all_lanes = true;
}

/* Ordered latest-copy allocation with triangle-wide closure.
 *
 * One lane plans the complete stream before any allocation/export. A triangle
 * is only committed after closure: trial slots never enter LDS/latest[].
 * A missing/stale corner becomes fresh monotonically; at most three decisions
 * can change, so an initial trial plus three refinements suffice. Each committed
 * slot belongs to a corner of this triangle, so every slot is referenced. Closure uses the final
 * high-water mark of the whole triangle, proving H - corner <= 31 (triangle-wide backjump).
 *
 * At most three slots are created per primitive. For V<=32 no first-use slot
 * can become stale, so the bound is min(V,3P); otherwise it is 3P. Admission and
 * launch provisioning establish this bound statically. There is no truncation,
 * late overflow draw, primitive reordering, or API-body reexecution.
 *
 * The serial planner is a correctness-first implementation, not a claim of
 * performance parity with raw Mesh. Original outputs stay in LDS.
 */
static void
ms_safe_direct_vertices(nir_builder *b, lower_ngg_ms_state *s)
{
   const unsigned max_vertices = b->shader->info.mesh.max_vertices_out;
   const unsigned max_primitives = b->shader->info.mesh.max_primitives_out;
   const unsigned bound = s->options->bc250_safe_direct_bound;
   assert(max_vertices && max_vertices <= 256 && max_primitives && max_primitives <= 256);
   assert(bound && bound <= s->hw_workgroup_size && bound <= 256);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *lane = nir_load_local_invocation_index(b);

   /* All hardware lanes initialize disjoint latest-copy entries. */
   for (unsigned first = 0; first < max_vertices; first += s->hw_workgroup_size) {
      nir_def *v = nir_iadd_imm(b, lane, first);
      nir_if *valid = nir_push_if(b, nir_ult_imm(b, v, max_vertices));
      nir_store_shared(b, nir_imm_int(b, -1), nir_imul_imm(b, v, 4),
                       .base = s->safe_direct_latest_addr, .align_mul = 4);
      nir_pop_if(b, valid);
   }
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);

   nir_if *first_lane = nir_push_if(b, nir_ieq_imm(b, lane, 0));
   {
      /* Only invocation zero owns the original counts; extra export lanes may
       * have undef count variables and must never branch on those values. */
      nir_def *vc = nir_umin_imm(b, nir_load_var(b, s->vertex_count_var), max_vertices);
      nir_def *pc = nir_umin_imm(b, nir_load_var(b, s->primitive_count_var), max_primitives);
      pc = nir_bcsel(b, nir_ieq_imm(b, vc, 0), zero, pc);
      nir_def *limit = nir_iadd_imm(b, nir_umax_imm(b, vc, 1), -1);
      nir_variable *count = nir_local_variable_create(b->impl, glsl_uint_type(), "safe_vertex_count");
      nir_variable *primitive = nir_local_variable_create(b->impl, glsl_uint_type(), "safe_primitive");
      nir_store_var(b, count, zero, 1);
      nir_store_var(b, primitive, zero, 1);

      nir_loop *loop = nir_push_loop(b);
      {
         nir_def *p = nir_load_var(b, primitive);
         nir_break_if(b, nir_uge(b, p, pc));
         nir_def *indices = nir_u2u32(b, nir_load_shared(b, 3, 8, nir_imul_imm(b, p, 3),
                                                       .base = s->layout.lds.indices_addr));
         nir_def *v[3], *old[3], *fresh[3], *slot[3];
         for (unsigned c = 0; c < 3; ++c) {
            v[c] = nir_umin(b, nir_channel(b, indices, c), limit);
            old[c] = nir_load_shared(b, 1, 32, nir_imul_imm(b, v[c], 4),
                                     .base = s->safe_direct_latest_addr, .align_mul = 4);
            fresh[c] = nir_ieq_imm(b, old[c], -1);
            for (unsigned j = 0; j < c; ++j)
               fresh[c] = nir_iand(b, fresh[c], nir_ine(b, v[c], v[j]));
         }

         nir_def *base = nir_load_var(b, count);
         nir_def *end = NULL;
         for (unsigned pass = 0; pass < 4; ++pass) {
            end = base;
            for (unsigned c = 0; c < 3; ++c) {
               nir_def *reuse = old[c];
               /* Repeated logical corners see this trial's newest copy. */
               for (unsigned j = 0; j < c; ++j)
                  reuse = nir_bcsel(b, nir_ieq(b, v[c], v[j]), slot[j], reuse);
               slot[c] = nir_bcsel(b, fresh[c], end, reuse);
               end = nir_iadd(b, end, nir_b2i32(b, fresh[c]));
            }
            if (pass != 3) {
               nir_def *high = nir_iadd_imm(b, end, -1);
               for (unsigned c = 0; c < 3; ++c) {
                  nir_def *stale = nir_ugt_imm(b, nir_isub(b, high, slot[c]), 31);
                  fresh[c] = nir_ior(b, fresh[c], stale);
               }
            }
         }

         for (unsigned c = 0; c < 3; ++c) {
            nir_if *create = nir_push_if(b, fresh[c]);
            nir_store_shared(b, nir_u2u8(b, v[c]), slot[c], .base = s->safe_direct_map_addr);
            nir_pop_if(b, create);
            nir_store_shared(b, slot[c], nir_imul_imm(b, v[c], 4),
                             .base = s->safe_direct_latest_addr, .align_mul = 4);
         }
         nir_store_shared(b, nir_u2u8(b, nir_vec3(b, slot[0], slot[1], slot[2])),
                          nir_imul_imm(b, p, 3), .base = s->safe_direct_indices_addr);
         nir_store_var(b, count, end, 1);
         nir_store_var(b, primitive, nir_iadd_imm(b, p, 1), 1);
      }
      nir_pop_loop(b, loop);
      nir_store_shared(b, nir_vec2(b, nir_load_var(b, count), pc), zero,
                       .base = s->safe_direct_counts_addr, .align_mul = 4);
   }
   nir_pop_if(b, first_lane);

   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   nir_def *counts = nir_load_shared(b, 2, 32, zero,
                                     .base = s->safe_direct_counts_addr, .align_mul = 4);
   nir_store_var(b, s->vertex_count_var, nir_channel(b, counts, 0), 1);
   nir_store_var(b, s->primitive_count_var, nir_channel(b, counts, 1), 1);
   s->counts_in_all_lanes = true;
}

/* One wave64 checks at most 64 triangles; one publication barrier. No map on the
 * clean path. The wave maximum includes all three corners of each triangle. */
static void
ms_safe_stats_add(nir_builder *b, lower_ngg_ms_state *s, unsigned field, nir_def *value)
{
   if (!s->options->bc250_safe_stats)
      return;

   nir_def *va = nir_load_ngg_safe_stats_buf_va_amd(b);
   nir_if *enabled = nir_push_if(b, nir_ine_imm(b, nir_u2u32(b, va), 0));
   nir_global_atomic_amd(b, 32, va, value, nir_imm_int(b, field * 4), .atomic_op = nir_atomic_op_iadd);
   nir_pop_if(b, enabled);
}

/* One wave owns at most 64 primitives. Transpose the three index streams into
 * bit planes, then match each corner against every live triangle using only
 * register operations. The first occurrence in each ten-triangle interval owns
 * the vertex; an interval has at most 30 corners, hence every reference is at
 * most 29 slots behind its triangle-wide high-water mark. Ranking first uses
 * preserves primitive and corner order, and every allocated slot is referenced.
 * For V<=32 the whole wave is one interval, retaining the min(V,3P) bound.
 * No cross-wave scan, shared atomics, vertex-by-primitive loop or intermediate
 * LDS table is required. The caller's publication barrier covers these stores.
 */
static nir_def *
ms_safe_local_vertices(nir_builder *b, lower_ngg_ms_state *s, nir_def *idx,
                       nir_def *live, nir_def *vc)
{
   assert(b->shader->info.mesh.max_primitives_out <= s->wave_size);
   const unsigned bits = util_logbase2_ceil(b->shader->info.mesh.max_vertices_out);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *limit = nir_iadd_imm(b, nir_umax_imm(b, vc, 1), -1);
   nir_def *v[3];
   nir_def *live_mask = nir_ballot(b, 1, 64, live);
   for (unsigned c = 0; c < 3; ++c) {
      v[c] = nir_umin(b, nir_channel(b, idx, c), limit);
   }
   nir_def *interval = nir_imm_int64(b, -1);
   if (b->shader->info.mesh.max_vertices_out > 32) {
      nir_def *start = nir_imul_imm(b, nir_udiv_imm(b, lane, 10), 10);
      interval = nir_bcsel(b, nir_ule_imm(b, vc, 32), interval,
                          nir_ishl(b, nir_imm_int64(b, 1023), start));
   }
   nir_def *owner_p[3], *owner_c[3], *fresh[3];
   nir_def *owner[3] = {nir_imm_int(b, 192), nir_imm_int(b, 192), nir_imm_int(b, 192)};
   /* Consume each plane immediately. Holding all 24 ballots live together
    * needlessly increases SGPR pressure, including on clean workgroups. */
   for (unsigned d = 0; d < 3; ++d) {
      nir_def *matches[3];
      for (unsigned c = 0; c < 3; ++c)
         matches[c] = nir_iand(b, live_mask, interval);
      for (unsigned k = 0; k < bits; ++k) {
         nir_def *plane = nir_ballot(b, 1, 64, nir_ine_imm(b, nir_iand_imm(b, v[d], 1u << k), 0));
         for (unsigned c = 0; c < 3; ++c) {
            nir_def *set = nir_ine_imm(b, nir_iand_imm(b, v[c], 1u << k), 0);
            matches[c] = nir_iand(b, matches[c], nir_bcsel(b, set, plane, nir_inot(b, plane)));
         }
      }
      for (unsigned c = 0; c < 3; ++c) {
         /* find_lsb(0)=-1 produces an unsigned key above every valid key. */
         nir_def *key = nir_iadd_imm(b, nir_imul_imm(b, nir_find_lsb(b, matches[c]), 3), d);
         owner[c] = nir_umin(b, owner[c], key);
      }
   }
   nir_def *fresh_bits = zero;
   nir_def *fresh_count = zero;
   for (unsigned c = 0; c < 3; ++c) {
      owner_p[c] = nir_bcsel(b, live, nir_udiv_imm(b, owner[c], 3), zero);
      owner_c[c] = nir_umod_imm(b, owner[c], 3);
      fresh[c] = nir_iand(b, live, nir_ieq(b, owner[c], nir_iadd_imm(b, nir_imul_imm(b, lane, 3), c)));
      fresh_bits = nir_ior(b, fresh_bits, nir_ishl_imm(b, nir_b2i32(b, fresh[c]), c));
      fresh_count = nir_iadd(b, fresh_count, nir_b2i32(b, fresh[c]));
   }
   nir_def *prefix = nir_exclusive_scan(b, fresh_count, .reduction_op = nir_op_iadd);
   nir_def *slots[3];
   for (unsigned c = 0; c < 3; ++c) {
      nir_def *base = nir_shuffle(b, prefix, owner_p[c]);
      nir_def *owned = nir_shuffle(b, fresh_bits, owner_p[c]);
      nir_def *below = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), owner_c[c]), -1);
      slots[c] = nir_iadd(b, base, nir_bit_count(b, nir_iand(b, owned, below)));
      nir_if *create = nir_push_if(b, fresh[c]);
      nir_store_shared(b, nir_u2u8(b, v[c]), slots[c], .base = s->safe_direct_map_addr);
      nir_pop_if(b, create);
   }
   nir_if *accepted = nir_push_if(b, live);
   nir_def *p = nir_mbcnt_amd(b, live_mask, zero);
   nir_store_shared(b, nir_u2u8(b, nir_vec3(b, slots[0], slots[1], slots[2])), nir_imul_imm(b, p, 3),
                    .base = s->safe_direct_indices_addr);
   nir_pop_if(b, accepted);
   return nir_vec2(b, nir_read_invocation(b, nir_iadd(b, prefix, fresh_count), nir_imm_int(b, s->wave_size - 1)),
                   nir_u2u32(b, nir_bit_count(b, live_mask)));
}

/* RADV_BC250_MESH_PP_SHARE, owned private corners. Key k = 3j + c is corner c of the j-th surviving triangle
 * (lane j), x[c] its private expanded vertex and L[c] the application's logical vertex (index staging). Each
 * key refers to the smallest key with the same valid logical vertex in its 10-triangle window, as
 * ms_safe_local_vertices does; the provoking corner of every triangle (and every invalid index) refers to
 * itself. A fresh key gets the next export slot, whose source is its own private vertex x: every copy of L
 * carries the same per-vertex values, and a provoking copy carries its own triangle's per-primitive values,
 * which flat interpolation reads only there. Every slot is referenced by the triangle that created it, and a
 * reference stays inside the window (at most 30 keys back), so the triangle-wide backjump stays <= 31.
 * Primitive and corner order are unchanged. Writes the slot map and the slot triples; returns (V', P'). */
static nir_def *
ms_pps_vertices(nir_builder *b, lower_ngg_ms_state *s, nir_def **L, nir_def **valid, nir_def **x, nir_def *live,
                unsigned provoking)
{
   const unsigned table = s->options->bc250_compact_index_staging[3];
   const unsigned bits = MAX2(util_logbase2_ceil(table), 1);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *live_mask = nir_ballot(b, 1, 64, live);
   nir_def *start = nir_imul_imm(b, nir_udiv_imm(b, lane, 10), 10);
   nir_def *interval = nir_ishl(b, nir_imm_int64(b, 1023), start);
   nir_def *owner[3] = {nir_imm_int(b, 192), nir_imm_int(b, 192), nir_imm_int(b, 192)};
   if (table > 64) {
      /* Large tables: compare with the (at most 30) keys of the window directly. The bit-plane match below
       * would keep 3 * 8 wave masks live and exceed the scalar register budget. Same result: the smallest
       * key in the window with the same valid logical vertex (never above the key itself). */
      nir_def *self_lane = lane;
      nir_variable *owner_var[3];
      for (unsigned c = 0; c < 3; ++c) {
         owner_var[c] = nir_local_variable_create(b->impl, glsl_uint_type(), "pps_owner");
         nir_store_var(b, owner_var[c], nir_iadd_imm(b, nir_imul_imm(b, self_lane, 3), c), 1);
      }
      /* A real loop over the window, so only one window lane's values are live at a time. */
      nir_variable *i_var = nir_local_variable_create(b->impl, glsl_uint_type(), "pps_i");
      nir_store_var(b, i_var, nir_imm_int(b, 0), 1);
      nir_loop *loop = nir_push_loop(b);
      {
         nir_def *i = nir_load_var(b, i_var);
         nir_break_if(b, nir_uge_imm(b, i, 10));
         nir_def *src = nir_umin_imm(b, nir_iadd(b, start, i), s->wave_size - 1);
         nir_def *before = nir_uge(b, self_lane, src);
         nir_def *own[3];
         for (unsigned c = 0; c < 3; ++c)
            own[c] = nir_load_var(b, owner_var[c]);
         for (unsigned d = 0; d < 3; ++d) {
            nir_def *other_l = nir_shuffle(b, L[d], src);
            nir_def *other_ok = nir_ine_imm(b, nir_shuffle(b, nir_b2i32(b, valid[d]), src), 0);
            nir_def *key = nir_iadd_imm(b, nir_imul_imm(b, src, 3), d);
            for (unsigned c = 0; c < 3; ++c) {
               nir_def *same = nir_iand(b, nir_iand(b, before, other_ok),
                                        nir_iand(b, valid[c], nir_ieq(b, other_l, L[c])));
               own[c] = nir_bcsel(b, same, nir_umin(b, own[c], key), own[c]);
            }
         }
         for (unsigned c = 0; c < 3; ++c)
            nir_store_var(b, owner_var[c], own[c], 1);
         nir_store_var(b, i_var, nir_iadd_imm(b, i, 1), 1);
      }
      nir_pop_loop(b, loop);
      for (unsigned c = 0; c < 3; ++c)
         owner[c] = nir_load_var(b, owner_var[c]);
   }
   for (unsigned d = 0; d < 3 && table <= 64; ++d) {
      nir_def *candidates = nir_iand(b, nir_ballot(b, 1, 64, nir_iand(b, live, valid[d])), interval);
      nir_def *matches[3];
      for (unsigned c = 0; c < 3; ++c)
         matches[c] = candidates;
      for (unsigned k = 0; k < bits; ++k) {
         nir_def *plane = nir_ballot(b, 1, 64, nir_ine_imm(b, nir_iand_imm(b, L[d], 1u << k), 0));
         for (unsigned c = 0; c < 3; ++c) {
            nir_def *set = nir_ine_imm(b, nir_iand_imm(b, L[c], 1u << k), 0);
            matches[c] = nir_iand(b, matches[c], nir_bcsel(b, set, plane, nir_inot(b, plane)));
         }
      }
      for (unsigned c = 0; c < 3; ++c) {
         /* find_lsb(0) = -1 gives an unsigned key above every valid key. */
         nir_def *key = nir_iadd_imm(b, nir_imul_imm(b, nir_find_lsb(b, matches[c]), 3), d);
         owner[c] = nir_umin(b, owner[c], key);
      }
   }
   nir_def *owner_p[3], *owner_c[3], *fresh[3];
   nir_def *fresh_bits = zero, *fresh_count = zero;
   for (unsigned c = 0; c < 3; ++c) {
      nir_def *self = nir_iadd_imm(b, nir_imul_imm(b, lane, 3), c);
      if (c == provoking)
         owner[c] = self;
      else
         owner[c] = nir_bcsel(b, valid[c], owner[c], self);
      owner_p[c] = nir_bcsel(b, live, nir_udiv_imm(b, owner[c], 3), zero);
      owner_c[c] = nir_umod_imm(b, owner[c], 3);
      fresh[c] = nir_iand(b, live, nir_ieq(b, owner[c], self));
      fresh_bits = nir_ior(b, fresh_bits, nir_ishl_imm(b, nir_b2i32(b, fresh[c]), c));
      fresh_count = nir_iadd(b, fresh_count, nir_b2i32(b, fresh[c]));
   }
   nir_def *prefix = nir_exclusive_scan(b, fresh_count, .reduction_op = nir_op_iadd);
   nir_def *slots[3];
   for (unsigned c = 0; c < 3; ++c) {
      nir_def *base = nir_shuffle(b, prefix, owner_p[c]);
      nir_def *owned = nir_shuffle(b, fresh_bits, owner_p[c]);
      nir_def *below = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), owner_c[c]), -1);
      slots[c] = nir_iadd(b, base, nir_bit_count(b, nir_iand(b, owned, below)));
      nir_if *create = nir_push_if(b, fresh[c]);
      nir_store_shared(b, nir_u2u8(b, x[c]), slots[c], .base = s->safe_direct_map_addr);
      nir_pop_if(b, create);
   }
   nir_if *accepted = nir_push_if(b, live);
   nir_def *p = nir_mbcnt_amd(b, live_mask, zero);
   nir_store_shared(b, nir_u2u8(b, nir_vec3(b, slots[0], slots[1], slots[2])), nir_imul_imm(b, p, 3),
                    .base = s->safe_direct_indices_addr);
   nir_pop_if(b, accepted);
   return nir_vec2(b, nir_read_invocation(b, nir_iadd(b, prefix, fresh_count), nir_imm_int(b, s->wave_size - 1)),
                   nir_u2u32(b, nir_bit_count(b, live_mask)));
}

/* If the surviving connectivity already obeys W31, deleting holes in increasing
 * original-index order can only shorten every backjump. Reuse the checker's
 * coverage masks instead of building any duplicate-vertex plan. This also
 * handles clean meshlets after culling has made some vertices unreferenced. */
static nir_def *
ms_safe_local_remove_holes(nir_builder *b, lower_ngg_ms_state *s, nir_def *idx,
                           nir_def *live, nir_def **coverage, nir_def *used)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *lane = nir_load_subgroup_invocation(b);
   const unsigned vmax = b->shader->info.mesh.max_vertices_out;
   nir_def *words[8];
   for (unsigned w = 0; w < 8; ++w)
      words[w] = w < DIV_ROUND_UP(vmax, 32) ? coverage[w] : zero;
   for (unsigned first = 0; first < vmax; first += s->wave_size) {
      nir_def *v = nir_iadd_imm(b, lane, first);
      nir_def *mask = words[first / 32];
      if (s->wave_size == 64 && first + 32 < vmax)
         mask = nir_bcsel(b, nir_ult_imm(b, lane, 32), mask, words[first / 32 + 1]);
      nir_def *referenced = nir_iand(b, nir_ult_imm(b, v, vmax),
         nir_ine_imm(b, nir_iand_imm(b, nir_ushr(b, mask, nir_iand_imm(b, v, 31)), 1), 0));
      nir_if *store = nir_push_if(b, referenced);
      nir_store_shared(b, nir_u2u8(b, v), ms_safe_parallel_rank(b, v, words), .base = s->safe_direct_map_addr);
      nir_pop_if(b, store);
   }
   nir_def *live_mask = nir_ballot(b, 1, 64, live);
   nir_if *store = nir_push_if(b, live);
   nir_def *slots[3];
   for (unsigned c = 0; c < 3; ++c)
      slots[c] = ms_safe_parallel_rank(b, nir_channel(b, idx, c), words);
   nir_store_shared(b, nir_u2u8(b, nir_vec3(b, slots[0], slots[1], slots[2])),
                    nir_imul_imm(b, nir_mbcnt_amd(b, live_mask, zero), 3), .base = s->safe_direct_indices_addr);
   nir_pop_if(b, store);
   return nir_vec2(b, used, nir_u2u32(b, nir_bit_count(b, live_mask)));
}

/* Wave-local check and repair, shared by both publication protocols. */
/* RADV_BC250_MESH_SAFE_ADAPTIVE: whether the API connectivity of this workgroup, with every
 * primitive kept, already references every vertex in [0, vc) and keeps the triangle-wide
 * backjump <= 31 with indices < vc. Runs in the single checker wave; the result is wave-uniform. */
static nir_def *
ms_adaptive_shared_clean(nir_builder *b, lower_ngg_ms_state *s, nir_def *idx, nir_def *live, nir_def *vc)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *v[3] = {nir_channel(b, idx, 0), nir_channel(b, idx, 1), nir_channel(b, idx, 2)};
   nir_def *hi = nir_umax(b, v[0], nir_umax(b, v[1], v[2]));
   nir_def *lo = nir_umin(b, v[0], nir_umin(b, v[1], v[2]));
   nir_def *range_bad = nir_iand(b, live, nir_uge(b, hi, vc));
   nir_def *prefix = nir_inclusive_scan(b, nir_bcsel(b, live, hi, zero), .reduction_op = nir_op_umax);
   nir_def *r2_bad = nir_iand(b, live, nir_ult(b, nir_iadd_imm(b, lo, 31), prefix));
   const unsigned words = DIV_ROUND_UP(b->shader->info.mesh.max_vertices_out, 32);
   assert(words <= 8);
   nir_def *used = zero;
   for (unsigned w = 0; w < words; ++w) {
      nir_def *bits = zero;
      for (unsigned c = 0; c < 3; ++c) {
         nir_def *in_word = nir_ieq_imm(b, nir_ushr_imm(b, v[c], 5), w);
         bits = nir_ior(b, bits, nir_bcsel(b, in_word, nir_ishl(b, nir_imm_int(b, 1), v[c]), zero));
      }
      bits = nir_bcsel(b, live, bits, zero);
      used = nir_iadd(b, used, nir_bit_count(b, nir_reduce(b, bits, .reduction_op = nir_op_ior)));
   }
   return nir_iand(b, nir_ieq(b, used, vc),
                   nir_inot(b, nir_vote_any(b, 1, nir_ior(b, range_bad, r2_bad))));
}

/* RADV_BC250_MESH_SAFE_COMPACT: whether the live triangles, with the vertices they do not reference
 * deleted and the rest renumbered in increasing order (their rank), keep indices < vc and the
 * triangle-wide backjump <= 31 on the new numbers. Every exported vertex is then referenced by
 * construction. Ranks are monotone, so the running maximum of the ranks is the rank of the running
 * maximum. Fills the coverage words for ms_safe_local_remove_holes. Wave-uniform. */
static nir_def *
ms_adaptive_compact_ok(nir_builder *b, lower_ngg_ms_state *s, nir_def *idx, nir_def *live, nir_def *vc,
                       nir_def **words, nir_def **used)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *v[3] = {nir_channel(b, idx, 0), nir_channel(b, idx, 1), nir_channel(b, idx, 2)};
   nir_def *hi = nir_umax(b, v[0], nir_umax(b, v[1], v[2]));
   nir_def *lo = nir_umin(b, v[0], nir_umin(b, v[1], v[2]));
   nir_def *range_bad = nir_iand(b, live, nir_uge(b, hi, vc));
   const unsigned count = DIV_ROUND_UP(b->shader->info.mesh.max_vertices_out, 32);
   assert(count <= 8);
   *used = zero;
   for (unsigned w = 0; w < 8; ++w) {
      if (w >= count) {
         words[w] = zero;
         continue;
      }
      nir_def *bits = zero;
      for (unsigned c = 0; c < 3; ++c) {
         nir_def *in_word = nir_ieq_imm(b, nir_ushr_imm(b, v[c], 5), w);
         bits = nir_ior(b, bits, nir_bcsel(b, in_word, nir_ishl(b, nir_imm_int(b, 1), v[c]), zero));
      }
      words[w] = nir_reduce(b, nir_bcsel(b, live, bits, zero), .reduction_op = nir_op_ior);
      *used = nir_iadd(b, *used, nir_bit_count(b, words[w]));
   }
   nir_def *rank_hi = ms_safe_parallel_rank(b, hi, words);
   nir_def *rank_lo = ms_safe_parallel_rank(b, lo, words);
   nir_def *prefix = nir_inclusive_scan(b, nir_bcsel(b, live, rank_hi, zero), .reduction_op = nir_op_umax);
   nir_def *r2_bad = nir_iand(b, live, nir_ult(b, nir_iadd_imm(b, rank_lo, 31), prefix));
   return nir_inot(b, nir_vote_any(b, 1, nir_ior(b, range_bad, r2_bad)));
}

static nir_def *
ms_safe_fast_check_wave(nir_builder *b, lower_ngg_ms_state *s, nir_def *vc, nir_def *pc)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *live = nir_ult(b, lane, pc);
   nir_if *active = nir_push_if(b, live);
   nir_def *loaded = nir_u2u32(b, nir_load_shared(b, 3, 8, nir_imul_imm(b, lane, 3),
                                                .base = s->layout.lds.indices_addr));
   nir_push_else(b, active);
   nir_def *empty = nir_imm_ivec3(b, 0, 0, 0);
   nir_pop_if(b, active);
   nir_def *idx = nir_if_phi(b, loaded, empty);
   nir_def *live_all = live;
   nir_def *culled = nir_imm_false(b);
   if (s->options->bc250_safe_local && s->options->bc250_safe_autocull) {
      nir_def *keep = nir_imm_true(b);
      nir_if *test = nir_push_if(b, nir_iand(b, live, nir_load_cull_any_enabled_amd(b)));
      nir_def *accepted = ms_autocull_accept(b, lane, vc, s, s->options->bc250_safe_check ? idx : NULL);
      nir_pop_if(b, test);
      accepted = nir_if_phi(b, accepted, keep);
      culled = nir_iand(b, live, nir_inot(b, accepted));
      live = nir_iand(b, live, accepted);
   }
   if (s->options->bc250_safe_corners) {
      /* Private corners, directly from the original API outputs. The ballot
       * preserves primitive order, including degenerates. Slot 3p+c belongs
       * only to surviving triangle p: all slots are referenced and the
       * triangle-wide backjump is exactly 2. No search, coverage reduction,
       * prefix-max scan or primitive-index map is needed. */
      /* RADV_BC250_MESH_SAFE_ADAPTIVE, per workgroup (all decisions wave-uniform):
       *  1. the survivors of culling reference every API vertex with backjump <= 31: export the API
       *     vertices once each and the survivors;
       *  2. else, culling orphaned vertices but removed at most a quarter of the triangles and the
       *     whole set references every vertex with backjump <= 31: export the API vertices and every triangle; the rasterizer culls
       *     the few (cheaper than private corners for the many survivors);
       *  3. otherwise the private corners of the survivors.
       * The map holds the exported triangles' API indices in order (corner sources for 3). */
      nir_def *use_all = nir_imm_false(b), *shared = nir_imm_false(b);
      if (s->options->bc250_safe_adaptive) {
         nir_def *post = ms_adaptive_shared_clean(b, s, idx, live, vc);
         nir_def *pre;
         if (s->options->bc250_lean_check) {
            /* RADV_BC250_MESH_LEAN_CHECK: the pre-cull check only matters when the survivors are not clean
             * and culling removed a triangle (otherwise it equals the survivor check). Same result. */
            nir_def *unneeded = nir_imm_false(b);
            nir_if *need = nir_push_if(b, nir_iand(b, nir_inot(b, post), nir_vote_any(b, 1, culled)));
            nir_def *checked = ms_adaptive_shared_clean(b, s, idx, live_all, vc);
            nir_pop_if(b, need);
            pre = nir_if_phi(b, checked, unneeded);
         } else {
            pre = ms_adaptive_shared_clean(b, s, idx, live_all, vc);
         }
         nir_def *kept = nir_u2u32(b, nir_bit_count(b, nir_ballot(b, 1, 64, live)));
         nir_def *few = nir_uge(b, pc, nir_imul_imm(b, nir_isub(b, pc, kept), 4));
         use_all = nir_iand(b, nir_inot(b, post), nir_iand(b, pre, few));
         shared = nir_ior(b, post, use_all);
      }
      /* RADV_BC250_MESH_SAFE_COMPACT, between cases 1 and 2: the survivors reference fewer vertices
       * (culled or split away) but keep backjump <= 31 once the others are deleted: export only the
       * referenced vertices, in increasing API order, and the survivors in order. */
      if (s->pp_share) {
         /* RADV_BC250_MESH_PP_SHARE (owned corners, never adaptive): the logical vertex of each corner. */
         const uint32_t *ix = s->options->bc250_compact_index_staging;
         nir_def *L[3], *valid[3], *x[3];
         nir_def *limit = nir_iadd_imm(b, vc, -1);
         nir_if *load = nir_push_if(b, live);
         nir_def *loaded[3];
         for (unsigned c = 0; c < 3; ++c) {
            nir_def *addr = nir_iadd_imm(b, nir_imul_imm(b, lane, ix[1]), c * ix[2]);
            loaded[c] = nir_u2u32(b, nir_load_shared(b, 1, ix[2] * 8, addr, .base = ix[0], .align_mul = ix[2]));
         }
         nir_pop_if(b, load);
         for (unsigned c = 0; c < 3; ++c)
            L[c] = nir_if_phi(b, loaded[c], zero);
         for (unsigned c = 0; c < 3; ++c) {
            valid[c] = nir_iand(b, live, nir_ult_imm(b, L[c], ix[3]));
            x[c] = nir_umin(b, nir_channel(b, idx, c), limit);
         }
         nir_def *planned = ms_pps_vertices(b, s, L, valid, x, live,
                                            s->options->bc250_compact_owned_corners == 0x4 ? 2 : 0);
         return nir_vec3(b, nir_channel(b, planned, 0), nir_channel(b, planned, 1), zero);
      }
      nir_if *compact = NULL;
      nir_def *compact_result = NULL;
      if (s->safe_compact) {
         nir_def *words[8], *used;
         nir_def *ok = nir_iand(b, nir_inot(b, shared), ms_adaptive_compact_ok(b, s, idx, live, vc, words, &used));
         compact = nir_push_if(b, ok);
         nir_def *holes = ms_safe_local_remove_holes(b, s, idx, live, words, used);
         compact_result = nir_vec3(b, nir_channel(b, holes, 0), nir_channel(b, holes, 1), nir_imm_int(b, 2));
         nir_push_else(b, compact);
      }
      nir_def *sel = nir_bcsel(b, use_all, live_all, live);
      nir_def *mask = nir_ballot(b, 1, 64, sel);
      nir_def *first = nir_imul_imm(b, nir_mbcnt_amd(b, mask, zero), 3);
      nir_def *limit = nir_iadd_imm(b, vc, -1);
      nir_if *store = nir_push_if(b, sel);
      nir_store_shared(b, nir_u2u8(b, nir_umin(b, idx, limit)), first,
                       .base = s->safe_direct_map_addr);
      nir_pop_if(b, store);
      nir_def *prims = nir_u2u32(b, nir_bit_count(b, mask));
      nir_def *corner_result = nir_vec3(b, nir_imul_imm(b, prims, 3), prims, zero);
      if (!s->options->bc250_safe_adaptive)
         return corner_result;
      nir_def *result = nir_bcsel(b, shared, nir_vec3(b, vc, prims, nir_imm_int(b, 1)), corner_result);
      if (compact) {
         nir_pop_if(b, compact);
         result = nir_if_phi(b, compact_result, result);
      }
      return result;
   }
   nir_def *v[3] = {nir_channel(b, idx, 0), nir_channel(b, idx, 1), nir_channel(b, idx, 2)};
   nir_store_var(b, s->fast_prim_arg, ac_nir_pack_ngg_prim_exp_arg(b, 3, v, NULL, s->ac->gfx_level), 1);
   nir_def *hi = nir_umax(b, v[0], nir_umax(b, v[1], v[2]));
   nir_def *range_bad = nir_iand(b, live, nir_uge(b, hi, vc));
   nir_def *r2_bad = nir_imm_false(b);
   /* With at most 32 declared vertices, valid indices are already in a
    * single W31 window. Avoid the prefix-max scan entirely for these small
    * meshlets; it is otherwise a wave scan on every clean workgroup. */
   if (b->shader->info.mesh.max_vertices_out > 32) {
      /* Counts are workgroup-uniform. Small dynamic outputs also fit one W31
       * window, even when the shader's declared maximum is larger. Branch
       * around the wave scan so those workgroups only pay the range check. */
      nir_if *wide = nir_push_if(b, nir_ugt_imm(b, vc, 32));
      nir_def *lo = nir_bcsel(b, live, nir_umin(b, v[0], nir_umin(b, v[1], v[2])), nir_imm_int(b, 255));
      nir_def *prefix = nir_inclusive_scan(b, s->options->bc250_safe_local ? nir_bcsel(b, live, hi, zero) : hi,
                                          .reduction_op = nir_op_umax);
      nir_def *wide_r2_bad = nir_iand(b, live, nir_ult(b, nir_iadd_imm(b, lo, 31), prefix));
      nir_push_else(b, wide);
      nir_def *r2_clean = nir_imm_false(b);
      nir_pop_if(b, wide);
      r2_bad = nir_if_phi(b, wide_r2_bad, r2_clean);
   }
   const unsigned coverage_words = DIV_ROUND_UP(b->shader->info.mesh.max_vertices_out, 32);
   assert(coverage_words <= 8);
   nir_def *data[10];
   for (unsigned w = 0; w < coverage_words; ++w) {
      nir_def *bits = zero;
      for (unsigned c = 0; c < 3; ++c) {
         nir_def *in_word = nir_ieq_imm(b, nir_ushr_imm(b, v[c], 5), w);
         if (!s->options->bc250_safe_check)
            in_word = nir_iand(b, live, in_word);
         bits = nir_ior(b, bits, nir_bcsel(b, in_word,
                              nir_ishl(b, nir_imm_int(b, 1), v[c]), zero));
      }
      if (s->options->bc250_safe_check)
         bits = nir_bcsel(b, live, bits, zero);
      data[w] = nir_reduce(b, bits, .reduction_op = nir_op_ior);
   }
   if (!s->options->bc250_safe_local) {
      data[coverage_words] = nir_reduce(b, nir_b2i32(b, range_bad), .reduction_op = nir_op_ior);
      data[coverage_words + 1] = nir_reduce(b, nir_b2i32(b, r2_bad), .reduction_op = nir_op_ior);
   }
   /* Range validation above proves the covered set is a subset of [0, vc).
    * Its cardinality equals vc iff every exported vertex is referenced. */
   nir_def *used = zero;
   for (unsigned w = 0; w < coverage_words; ++w)
      used = nir_iadd(b, used, nir_bit_count(b, data[w]));
   nir_def *r1_fail = s->options->bc250_safe_local ? nir_ine(b, used, vc) :
      nir_ior(b, nir_ine_imm(b, data[coverage_words], 0), nir_ine(b, used, vc));
   nir_def *r2_fail = s->options->bc250_safe_local ?
      nir_vote_any(b, 1, nir_ior(b, range_bad, r2_bad)) :
      nir_ine_imm(b, data[coverage_words + 1], 0);
   nir_def *wave_clean = nir_iand(b, nir_inot(b, r1_fail), nir_inot(b, r2_fail));
   nir_def *final_counts = nir_vec2(b, vc, pc);
   if (s->options->bc250_safe_local) {
      /* Removing even a fully shared triangle changes the primitive stream. */
      wave_clean = nir_iand(b, wave_clean, nir_inot(b, nir_vote_any(b, 1, culled)));
      nir_if *repair = nir_push_if(b, nir_inot(b, wave_clean));
      nir_if *ordered = nir_push_if(b, nir_inot(b, r2_fail));
      nir_def *holes = ms_safe_local_remove_holes(b, s, idx, live, data, used);
      nir_push_else(b, ordered);
      nir_def *copies = ms_safe_local_vertices(b, s, idx, live, vc);
      nir_pop_if(b, ordered);
      nir_def *repaired = nir_if_phi(b, holes, copies);
      if (!s->options->bc250_safe_check) {
         nir_if *publish = nir_push_if(b, nir_ieq_imm(b, lane, 0));
         nir_store_shared(b, repaired, zero, .base = s->safe_direct_counts_addr, .align_mul = 4);
         nir_pop_if(b, publish);
      }
      nir_pop_if(b, repair);
      final_counts = nir_if_phi(b, repaired, final_counts);
   }
   nir_def *flags = s->options->bc250_safe_local ? nir_b2i32(b, wave_clean) :
                            nir_ior(b, nir_b2i32(b, wave_clean),
                            nir_ior(b, nir_ishl_imm(b, nir_b2i32(b, r1_fail), 1),
                                    nir_ishl_imm(b, nir_b2i32(b, r2_fail), 2)));
   return nir_vec3(b, nir_channel(b, final_counts, 0), nir_channel(b, final_counts, 1), flags);
}

/* The first wave owns the API's count values and all primitive checks. Other
 * waves need only the final counts/map. Publish them once, after checking.
 * For a one-wave API the producer/checker memory dependency is subgroup-local;
 * the publication barrier below still covers all exporting waves. */
static void
ms_safe_check(nir_builder *b, lower_ngg_ms_state *s)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *is_checker = nir_ieq_imm(b, nir_load_subgroup_id(b), 0);
   s->fast_prim_arg = nir_local_variable_create(b->impl, glsl_uint_type(), "fast_checked_connectivity");
   nir_store_var(b, s->fast_prim_arg, zero, 1);
   nir_if *checker = nir_push_if(b, is_checker);
   nir_def *vc = nir_umin_imm(b, nir_read_invocation(b, nir_load_var(b, s->vertex_count_var), zero),
                            b->shader->info.mesh.max_vertices_out);
   nir_def *pc = nir_umin_imm(b, nir_read_invocation(b, nir_load_var(b, s->primitive_count_var), zero),
                            b->shader->info.mesh.max_primitives_out);
   pc = nir_bcsel(b, nir_ieq_imm(b, vc, 0), zero, pc);
   vc = nir_bcsel(b, nir_ieq_imm(b, pc, 0), zero, vc);
   nir_if *nonempty = nir_push_if(b, nir_ine_imm(b, pc, 0));
   nir_def *planned = ms_safe_fast_check_wave(b, s, vc, pc);
   nir_push_else(b, nonempty);
   nir_def *empty = nir_imm_ivec3(b, 0, 0, 1);
   nir_pop_if(b, nonempty);
   nir_def *result = nir_if_phi(b, planned, empty);
   if (s->hw_workgroup_size > s->wave_size) {
      nir_if *leader = nir_push_if(b, nir_ieq_imm(b, lane, 0));
      nir_store_shared(b, nir_trim_vector(b, result, 2), zero,
                       .base = s->safe_direct_counts_addr, .align_mul = 4);
      if (!s->options->bc250_safe_corners || s->options->bc250_safe_adaptive)
         nir_store_shared(b, nir_channel(b, result, 2), zero,
                          .base = s->safe_direct_latest_addr, .align_mul = 4);
      nir_pop_if(b, leader);
   }
   nir_push_else(b, checker);
   nir_def *unused = nir_imm_ivec3(b, 0, 0, 0);
   nir_pop_if(b, checker);
   result = nir_if_phi(b, result, unused);
   /* Only API invocation zero updates the primitive query. */
   s->safe_query_prims = nir_if_phi(b, pc, zero);
   nir_barrier(b, .execution_scope = s->hw_workgroup_size > s->wave_size ? SCOPE_WORKGROUP : SCOPE_SUBGROUP,
               .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
   if (s->hw_workgroup_size > s->wave_size) {
      nir_if *other = nir_push_if(b, nir_inot(b, is_checker));
      nir_def *counts = nir_load_shared(b, 2, 32, zero, .base = s->safe_direct_counts_addr, .align_mul = 4);
      nir_def *flags = s->options->bc250_safe_corners && !s->options->bc250_safe_adaptive ? zero :
         nir_load_shared(b, 1, 32, zero, .base = s->safe_direct_latest_addr, .align_mul = 4);
      nir_def *loaded = nir_vec3(b, nir_channel(b, counts, 0), nir_channel(b, counts, 1), flags);
      nir_pop_if(b, other);
      result = nir_if_phi(b, loaded, result);
   }
   /* Both arms are wave-uniform. Keep the allocator and export predicates
    * scalar after merging the register and LDS results. */
   result = nir_read_first_invocation(b, result);
   nir_store_var(b, s->vertex_count_var, nir_channel(b, result, 0), 1);
   nir_store_var(b, s->primitive_count_var, nir_channel(b, result, 1), 1);
   s->fast_clean = s->options->bc250_safe_adaptive ?
                      nir_ine_imm(b, nir_iand_imm(b, nir_channel(b, result, 2), 1), 0) :
                   s->options->bc250_safe_corners ? NULL : nir_ine_imm(b, nir_channel(b, result, 2), 0);
   if (s->safe_compact)
      s->fast_compact = nir_ine_imm(b, nir_iand_imm(b, nir_channel(b, result, 2), 2), 0);
   s->counts_in_all_lanes = true;
}

static void
ms_safe_fast_check(nir_builder *b, lower_ngg_ms_state *s)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *check_start = s->options->bc250_safe_stats ?
      nir_channel(b, nir_shader_clock(b, SCOPE_SUBGROUP), 0) : NULL;
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *counts = nir_load_shared(b, 2, 32, zero,
                                    .base = s->safe_direct_counts_addr, .align_mul = 4);
   nir_def *vc = nir_channel(b, counts, 0), *pc = nir_channel(b, counts, 1);
   s->safe_query_prims = pc;
   nir_store_var(b, s->vertex_count_var, vc, 1);
   nir_store_var(b, s->primitive_count_var, pc, 1);
   s->counts_in_all_lanes = true;
   /* No primitives proves the empty export is safe without reading indices. */
   s->fast_prim_arg = nir_local_variable_create(b->impl, glsl_uint_type(), "fast_checked_connectivity");
   nir_store_var(b, s->fast_prim_arg, zero, 1);
   nir_if *nonempty = nir_push_if(b, nir_ine_imm(b, pc, 0));
   nir_if *checker_wave = nir_push_if(b, nir_ult_imm(b, lane, 64));
   nir_def *checked = ms_safe_fast_check_wave(b, s, vc, pc);
   nir_def *flags = nir_channel(b, checked, 2);
   nir_if *leader = nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_store_shared(b, flags, zero,
                    .base = s->safe_direct_latest_addr, .align_mul = 4);
   nir_pop_if(b, leader);
   nir_pop_if(b, checker_wave);
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   nir_def *checked_published = nir_read_first_invocation(b,
      nir_load_shared(b, 1, 32, zero, .base = s->safe_direct_latest_addr, .align_mul = 4));
   nir_def *checked_clean = nir_ine_imm(b, nir_iand_imm(b, checked_published, 1), 0);
   nir_push_else(b, nonempty);
   nir_def *empty_clean = nir_imm_true(b);
   nir_def *empty_published = nir_imm_int(b, 1);
   nir_pop_if(b, nonempty);
   nir_def *published = nir_if_phi(b, checked_published, empty_published);
   nir_def *clean = nir_if_phi(b, checked_clean, empty_clean);
   s->fast_clean = clean;
   if (s->options->bc250_safe_local) {
      nir_if *repair = nir_push_if(b, nir_inot(b, clean));
      nir_def *final_counts = nir_load_shared(b, 2, 32, zero,
         .base = s->safe_direct_counts_addr, .align_mul = 4);
      nir_pop_if(b, repair);
      final_counts = nir_if_phi(b, final_counts, counts);
      nir_store_var(b, s->vertex_count_var, nir_channel(b, final_counts, 0), 1);
      nir_store_var(b, s->primitive_count_var, nir_channel(b, final_counts, 1), 1);
      return;
   }
   if (s->options->bc250_safe_stats) {
      nir_def *end = nir_channel(b, nir_shader_clock(b, SCOPE_SUBGROUP), 0);
      nir_if *leader = nir_push_if(b, nir_ieq_imm(b, lane, 0));
      nir_def *unreferenced = nir_ine_imm(b, nir_iand_imm(b, published, 2), 0);
      nir_def *backjump_bad = nir_ine_imm(b, nir_iand_imm(b, published, 4), 0);
      ms_safe_stats_add(b, s, 0, nir_b2i32(b, clean));
      ms_safe_stats_add(b, s, 1, nir_b2i32(b, unreferenced));
      ms_safe_stats_add(b, s, 2, nir_b2i32(b, backjump_bad));
      ms_safe_stats_add(b, s, 3, nir_b2i32(b, nir_inot(b, clean)));
      ms_safe_stats_add(b, s, 4, nir_isub(b, end, check_start));
      nir_pop_if(b, leader);
   }
   nir_if *repair = nir_push_if(b, nir_inot(b, clean));
   ms_safe_parallel_vertices(b, s);
   nir_pop_if(b, repair);
}

/* The epilogue at the cursor: the finale barrier, the output counts (GS_ALLOC_REQ)
 * and the exports, as selected by s->autocull / compact_cull / pack_triangle_vertices. */
static void
emit_ms_epilogue(nir_builder *b, lower_ngg_ms_state *s)
{
   if (s->autocull)
      ms_autocull_store_counts(b, s);
   if (s->compact)
      ms_compact_prepare(b, s);

   if (s->options->bc250_safe_fast && !s->options->bc250_safe_check) {
      nir_if *first = nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      nir_def *vc = nir_umin_imm(b, nir_load_var(b, s->vertex_count_var), b->shader->info.mesh.max_vertices_out);
      nir_def *pc = nir_umin_imm(b, nir_load_var(b, s->primitive_count_var), b->shader->info.mesh.max_primitives_out);
      pc = nir_bcsel(b, nir_ieq_imm(b, vc, 0), nir_imm_int(b, 0), pc);
      vc = nir_bcsel(b, nir_ieq_imm(b, pc, 0), nir_imm_int(b, 0), vc);
      nir_store_shared(b, nir_vec2(b, vc, pc), nir_imm_int(b, 0),
                       .base = s->safe_direct_counts_addr, .align_mul = 4);
      nir_pop_if(b, first);
   }

   const bool local_producer = s->options->bc250_safe_check && s->api_workgroup_size <= s->wave_size;
   nir_barrier(b, .execution_scope = local_producer ? SCOPE_SUBGROUP : SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_shader_out | nir_var_mem_shared);

   nir_def *num_prm;
   nir_def *num_vtx;

   nir_def *query_primitives = nir_load_var(b, s->primitive_count_var);
   if (s->autocull)
      ms_autocull_compact(b, s);
   else if (s->compact_cull)
      ms_compact_culled_single_wave(b, s);
   if (s->compact)
      ms_compact_vertices(b, s, s->autocull);
   if (s->safe_direct) {
      if (s->options->bc250_safe_check)
         ms_safe_check(b, s);
      else if (s->options->bc250_safe_fast)
         ms_safe_fast_check(b, s);
      else if (s->options->bc250_safe_parallel)
         ms_safe_parallel_vertices(b, s);
      else
         ms_safe_direct_vertices(b, s);
   }
   nir_def *export_start = s->options->bc250_safe_stats ?
      nir_channel(b, nir_shader_clock(b, SCOPE_SUBGROUP), 0) : NULL;
   set_ms_final_output_counts(b, s, &num_prm, &num_vtx);

   nir_def *invocation_index = nir_load_local_invocation_index(b);

   ms_prim_gen_query(b, invocation_index, s->safe_query_prims ? s->safe_query_prims :
                     s->compact_cull ? query_primitives : num_prm, s);

   nir_def *row_start = NULL;
   if (s->ac->gfx_level >= GFX11)
      row_start = s->hw_workgroup_size <= s->wave_size ? nir_imm_int(b, 0) : nir_load_subgroup_id(b);

   /* Load vertex/primitive attributes from shared memory and
    * emit store_output intrinsics for them.
    *
    * Contrary to the semantics of the API mesh shader, these are now
    * compliant with NGG HW semantics, meaning that these store the
    * current thread's vertex attributes in a way the HW can export.
    */

   uint64_t per_vertex_outputs =
      s->per_vertex_outputs & ~s->layout.attr_ring.vtx_attr.mask;
   uint64_t per_primitive_outputs =
      s->per_primitive_outputs & ~s->layout.attr_ring.prm_attr.mask & ~SPECIAL_MS_OUT_MASK;

   /* Insert layer output store if the pipeline uses multiview but the API shader doesn't write it. */
   if (s->insert_layer_output && s->options->multiview_layer_per_vertex) {
      b->shader->info.outputs_written |= VARYING_BIT_LAYER;
      per_vertex_outputs |= VARYING_BIT_LAYER;
   } else if (s->insert_layer_output) {
      b->shader->info.outputs_written |= VARYING_BIT_LAYER;
      b->shader->info.per_primitive_outputs |= VARYING_BIT_LAYER;
      per_primitive_outputs |= VARYING_BIT_LAYER;
   }

   const bool has_special_param_exports =
      (per_vertex_outputs & MS_VERT_ARG_EXP_MASK) ||
      (per_primitive_outputs & MS_PRIM_ARG_EXP_MASK);
   const bool wait_attr_ring = has_special_param_exports && s->ac->has_attr_ring_wait_bug;

   /* Export vertices. */
   if ((per_vertex_outputs & ~VARYING_BIT_POS) || !wait_attr_ring) {
      emit_ms_outputs(b, invocation_index, row_start, num_vtx, !wait_attr_ring, true,
                      per_vertex_outputs, &emit_ms_vertex, s);
   }

   /* Export primitives. */
   if (per_primitive_outputs || !wait_attr_ring) {
      emit_ms_outputs(b, invocation_index, row_start, num_prm, !wait_attr_ring, true,
                      per_primitive_outputs, &emit_ms_primitive, s);
   }

   /* When we need to wait for attribute ring stores, we emit both position and primitive
    * export instructions after a barrier to make sure both per-vertex and per-primitive
    * attribute ring stores are finished before the GPU starts rasterization.
    */
   if (wait_attr_ring) {
      /* Wait for attribute stores to finish. */
      nir_barrier(b, .execution_scope = SCOPE_SUBGROUP,
                     .memory_scope = SCOPE_DEVICE,
                     .memory_semantics = NIR_MEMORY_RELEASE,
                     .memory_modes = nir_var_shader_out);

      /* Position/primitive export only */
      emit_ms_outputs(b, invocation_index, row_start, num_vtx, true, false,
                      per_vertex_outputs, &emit_ms_vertex, s);
      emit_ms_outputs(b, invocation_index, row_start, num_prm, true, false,
                      per_primitive_outputs, &emit_ms_primitive, s);
   }
   if (s->options->bc250_safe_stats) {
      nir_if *leader = nir_push_if(b, nir_ieq_imm(b, invocation_index, 0));
      nir_def *export_end = nir_channel(b, nir_shader_clock(b, SCOPE_SUBGROUP), 0);
      ms_safe_stats_add(b, s, 6, nir_isub(b, export_end, export_start));
      nir_pop_if(b, leader);
   }
}

static void
emit_ms_finale(nir_builder *b, lower_ngg_ms_state *s)
{
   /* We assume there is always a single end block in the shader. */
   nir_block *last_block = nir_impl_last_block(b->impl);
   b->cursor = nir_after_block(last_block);

   /* RADV_BC250_MESH_CULLDIST_CULL: the cull distances are not exported, so the shader culls
    * with them on every draw (VS NGG culling: "If cull distances are present, always cull in
    * the shader"); the runtime switch-off branch below would export triangles the hardware
    * could no longer cull. */
   if (!s->autocull || s->dont_export_cull) {
      emit_ms_epilogue(b, s);
      return;
   }

   /* RADV_BC250_MESH_AUTOCULL: one workgroup-uniform branch on the NGG culling
    * settings user SGPR (load_cull_any_enabled_amd, lowered by RADV for Mesh to
    * "front or back face culling enabled"). Only backface culling can remove
    * enough triangles to pay for the culling epilogue (frustum and small-primitive
    * culling alone do not), so without it the shader takes exactly the epilogue it
    * has with the switch off: same LDS addresses for everything it reads (checked
    * by ms_autocull_layouts_compatible), no culling test, no compaction, no extra
    * barrier. The branch is uniform, so both sides run their own barriers and
    * GS_ALLOC_REQ consistently for the whole workgroup. */
   nir_if *if_cull = nir_push_if(b, nir_load_cull_any_enabled_amd(b));
   {
      emit_ms_epilogue(b, s);
   }
   nir_push_else(b, if_cull);
   {
      s->autocull = false;
      s->compact_cull = s->plain_compact_cull;
      s->pack_triangle_vertices = s->plain_pack_triangle_vertices;
      s->compact_source_index = NULL;
      s->packed_vertex_source_prim = NULL;
      s->original_vertex_count = NULL;
      s->compact_active = false;
      s->counts_in_all_lanes = false;
      emit_ms_epilogue(b, s);
   }
   nir_pop_if(b, if_cull);
}

static void
handle_smaller_ms_api_workgroup(nir_builder *b,
                                lower_ngg_ms_state *s)
{
   if (s->api_workgroup_size >= s->hw_workgroup_size) {
      /* The smaller-workgroup wrapper below also emits query accounting.
       * Full workgroups need the same accounting without that wrapper. */
      ms_invocation_query(b, nir_load_local_invocation_index(b), s);
      return;
   }

   /* Handle barriers manually when the API workgroup
    * size is less than the HW workgroup size.
    *
    * The problem is that the real workgroup launched on NGG HW
    * will be larger than the size specified by the API, and the
    * extra waves need to keep up with barriers in the API waves.
    *
    * There are 2 different cases:
    * 1. The whole API workgroup fits in a single wave.
    *    We can shrink the barriers to subgroup scope and
    *    don't need to insert any extra ones.
    * 2. The API workgroup occupies multiple waves, but not
    *    all. In this case, we emit code that consumes every
    *    barrier on the extra waves.
    */
   assert(s->hw_workgroup_size % s->wave_size == 0);
   const unsigned num_api_waves = DIV_ROUND_UP(s->api_workgroup_size, s->wave_size);
   const unsigned num_hw_waves = DIV_ROUND_UP(s->hw_workgroup_size, s->wave_size);
   const bool scan_barriers = num_api_waves < num_hw_waves;
   const bool can_shrink_barriers = s->api_workgroup_size <= s->wave_size;

   bool need_additional_barriers = scan_barriers && !can_shrink_barriers;

   unsigned api_waves_in_flight_addr = s->layout.lds.workgroup_info_addr + lds_ms_num_api_waves;

   s->has_non_api_waves = scan_barriers;

   /* Scan the shader for workgroup barriers. */
   if (scan_barriers) {
      bool has_any_workgroup_barriers = false;

      nir_foreach_block(block, b->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_barrier)
               continue;

            /* Every API invocation runs in the first wave.
             * In this case, we can change the barriers to subgroup scope
             * and avoid adding additional barriers.
             */
            if (can_shrink_barriers &&
                nir_intrinsic_execution_scope(intrin) == SCOPE_WORKGROUP) {
               nir_intrinsic_set_execution_scope(intrin, SCOPE_SUBGROUP);
            }
            if (can_shrink_barriers &&
                nir_intrinsic_memory_scope(intrin) == SCOPE_WORKGROUP) {
               nir_intrinsic_set_memory_scope(intrin, SCOPE_SUBGROUP);
            }

            if (nir_intrinsic_execution_scope(intrin) == SCOPE_WORKGROUP) {
               has_any_workgroup_barriers = true;
               if (s->safe_direct && !can_shrink_barriers && !s->options->bc250_wave_sync) {
                  /* Safe direct: extra waves sample the in-flight counter
                   * between two barriers. API waves cannot finish/decrement
                   * until every extra wave has sampled this barrier's count.
                   * The terminal barrier below has count zero and no pair. */
                  b->cursor = nir_after_instr(instr);
                  nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                              .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
               }
            }
         }
      }

      need_additional_barriers &= has_any_workgroup_barriers;
   }

   /* BC250 wave sync: count the workgroup barriers the API waves execute (a register; Vulkan requires
    * workgroup barriers in workgroup-uniform control flow, so every API wave executes the same number). */
   const bool wave_sync = need_additional_barriers && s->options->bc250_wave_sync;
   nir_variable *api_barriers = NULL;
   if (wave_sync) {
      api_barriers = nir_local_variable_create(b->impl, glsl_uint_type(), "bc250_api_barriers");
      nir_foreach_block(block, b->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_barrier ||
                nir_intrinsic_execution_scope(intrin) != SCOPE_WORKGROUP)
               continue;
            b->cursor = nir_before_instr(instr);
            nir_store_var(b, api_barriers, nir_iadd_imm(b, nir_load_var(b, api_barriers), 1), 1);
         }
      }
      b->cursor = nir_before_impl(b->impl);
      nir_store_var(b, api_barriers, nir_imm_int(b, 0), 1);
   }

   /* Extract the full control flow of the shader. */
   nir_cf_list extracted;
   nir_cf_extract(&extracted, nir_before_impl(b->impl),
                  nir_after_cf_list(&b->impl->body));
   b->cursor = nir_before_impl(b->impl);

   /* Wrap the shader in an if to ensure that only the necessary amount of lanes run it. */
   nir_def *invocation_index = nir_load_local_invocation_index(b);
   nir_def *zero = nir_imm_int(b, 0);

   if (need_additional_barriers) {
      /* First invocation stores 0 to number of API waves in flight. */
      nir_if *if_first_in_workgroup = nir_push_if(b, nir_ieq_imm(b, invocation_index, 0));
      {
         nir_store_shared(b, nir_imm_int(b, wave_sync ? 0 : num_api_waves), zero, .base = api_waves_in_flight_addr);
      }
      nir_pop_if(b, if_first_in_workgroup);

      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                            .memory_scope = SCOPE_WORKGROUP,
                            .memory_semantics = NIR_MEMORY_ACQ_REL,
                            .memory_modes = nir_var_shader_out | nir_var_mem_shared);
   }

   nir_def *has_api_ms_invocation = nir_ult_imm(b, invocation_index, s->api_workgroup_size);
   nir_if *if_has_api_ms_invocation = nir_push_if(b, has_api_ms_invocation);
   {
      nir_cf_reinsert(&extracted, b->cursor);
      b->cursor = nir_after_cf_list(&if_has_api_ms_invocation->then_list);

      if (need_additional_barriers) {
         if (wave_sync) {
            /* The first invocation publishes the exact number of workgroup barriers the API waves
             * execute: the ones in the API code plus the one below. */
            nir_if *if_first = nir_push_if(b, nir_ieq_imm(b, invocation_index, 0));
            {
               nir_store_shared(b, nir_iadd_imm(b, nir_load_var(b, api_barriers), 1), zero,
                                .base = api_waves_in_flight_addr);
            }
            nir_pop_if(b, if_first);
         } else {
            /* One invocation in each API wave decrements the number of API waves in flight. */
            nir_if *if_elected_again = nir_push_if(b, nir_elect(b, 1));
            {
               nir_shared_atomic(b, 32, zero, nir_imm_int(b, -1u),
                                 .base = api_waves_in_flight_addr,
                                 .atomic_op = nir_atomic_op_iadd);
            }
            nir_pop_if(b, if_elected_again);
         }

         nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                               .memory_scope = SCOPE_WORKGROUP,
                               .memory_semantics = NIR_MEMORY_ACQ_REL,
                               .memory_modes = nir_var_shader_out | nir_var_mem_shared);
      }

      ms_invocation_query(b, invocation_index, s);
   }
   nir_pop_if(b, if_has_api_ms_invocation);

   if (need_additional_barriers) {
      /* Make sure that waves that don't run any API invocations execute
       * the same amount of barriers as those that do.
       *
       * We do this by executing a barrier until the number of API waves
       * in flight becomes zero.
       */
      nir_def *has_api_ms_ballot = nir_ballot(b, 1, s->wave_size, has_api_ms_invocation);
      nir_def *wave_has_no_api_ms = nir_ieq_imm(b, has_api_ms_ballot, 0);
      nir_if *if_wave_has_no_api_ms = nir_push_if(b, wave_has_no_api_ms);
      {
         nir_if *if_elected = nir_push_if(b, nir_elect(b, 1));
         {
            nir_variable *own_barriers = NULL;
            if (wave_sync) {
               own_barriers = nir_local_variable_create(b->impl, glsl_uint_type(), "bc250_own_barriers");
               nir_store_var(b, own_barriers, nir_imm_int(b, 0), 1);
            }
            nir_loop *loop = nir_push_loop(b);
            {
               nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                                     .memory_scope = SCOPE_WORKGROUP,
                                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                                     .memory_modes = nir_var_shader_out | nir_var_mem_shared);

               nir_def *loaded = nir_load_shared(b, 1, 32, zero, .base = api_waves_in_flight_addr);
               if (wave_sync) {
                  /* An early read sees 0 (not published yet) or the exact total: never ends early. */
                  nir_def *own = nir_iadd_imm(b, nir_load_var(b, own_barriers), 1);
                  nir_store_var(b, own_barriers, own, 1);
                  nir_break_if(b, nir_ieq(b, loaded, own));
               } else {
                  nir_break_if(b, nir_ieq_imm(b, loaded, 0));
                  if (s->safe_direct)
                     nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                                 .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
               }
            }
            nir_pop_loop(b, loop);
         }
         nir_pop_if(b, if_elected);
      }
      nir_pop_if(b, if_wave_has_no_api_ms);
   }
}

static void
ms_move_output(ms_out_part *from, ms_out_part *to)
{
   uint64_t loc = util_logbase2_64(from->mask);
   uint64_t bit = BITFIELD64_BIT(loc);
   from->mask ^= bit;
   to->mask |= bit;
}

static void
ms_calculate_arrayed_output_layout(ms_out_mem_layout *l,
                                   unsigned max_vertices,
                                   unsigned max_primitives)
{
   uint32_t lds_vtx_attr_size = l->lds.vtx_packed ? ms_packed_vertex_stride(l) * max_vertices :
                                util_bitcount64(l->lds.vtx_attr.mask) * max_vertices * 16;
   uint32_t lds_prm_attr_size = util_bitcount64(l->lds.prm_attr.mask) * max_primitives * 16;
   l->lds.prm_attr.addr = align(l->lds.vtx_attr.addr + lds_vtx_attr_size, 16);
   l->lds.total_size = l->lds.prm_attr.addr + lds_prm_attr_size;

   uint32_t scratch_ring_vtx_attr_size =
      util_bitcount64(l->scratch_ring.vtx_attr.mask) * max_vertices * 16;
   l->scratch_ring.prm_attr.addr =
      align(l->scratch_ring.vtx_attr.addr + scratch_ring_vtx_attr_size, 16);
}

static ms_out_mem_layout
ms_calculate_output_layout(const struct ac_compiler_info *info, unsigned api_shared_size,
                           uint64_t per_vertex_output_mask, uint64_t per_primitive_output_mask,
                           uint64_t cross_invocation_output_access, unsigned max_vertices,
                           unsigned max_primitives, unsigned vertices_per_prim,
                           const uint8_t *packed_vertex_comps, bool implicit_indices, bool private_bary)
{
   /* These outputs always need export instructions and can't use the attributes ring. */
   const uint64_t always_export_mask =
      VARYING_BIT_POS | VARYING_BIT_CULL_DIST0 | VARYING_BIT_CULL_DIST1 | VARYING_BIT_CLIP_DIST0 |
      VARYING_BIT_CLIP_DIST1 | VARYING_BIT_PSIZ | VARYING_BIT_VIEWPORT |
      VARYING_BIT_PRIMITIVE_SHADING_RATE | VARYING_BIT_LAYER |
      VARYING_BIT_PRIMITIVE_COUNT |
      VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_CULL_PRIMITIVE;

   const bool use_attr_ring = info->has_attr_ring;
   const uint64_t attr_ring_per_vertex_output_mask =
      use_attr_ring ? per_vertex_output_mask & ~always_export_mask : 0;
   const uint64_t attr_ring_per_primitive_output_mask =
      use_attr_ring ? per_primitive_output_mask & ~always_export_mask : 0;

   const uint64_t lds_per_vertex_output_mask =
      per_vertex_output_mask & ~attr_ring_per_vertex_output_mask & cross_invocation_output_access &
      ~SPECIAL_MS_OUT_MASK;
   const uint64_t lds_per_primitive_output_mask =
      per_primitive_output_mask & ~attr_ring_per_primitive_output_mask &
      cross_invocation_output_access & ~SPECIAL_MS_OUT_MASK;

   const bool cross_invocation_indices =
      cross_invocation_output_access & VARYING_BIT_PRIMITIVE_INDICES;
   const bool cross_invocation_cull_primitive =
      cross_invocation_output_access & VARYING_BIT_CULL_PRIMITIVE;

   /* Shared memory used by the API shader. */
   ms_out_mem_layout l = { .lds = { .total_size = api_shared_size } };

   /* Use attribute ring for all generic attributes (on GPUs with an attribute ring). */
   l.attr_ring.vtx_attr.mask = attr_ring_per_vertex_output_mask;
   l.attr_ring.prm_attr.mask = attr_ring_per_primitive_output_mask;

   /* Outputs without cross-invocation access can be stored in variables. */
   l.var.vtx_attr.mask =
      per_vertex_output_mask & ~attr_ring_per_vertex_output_mask & ~cross_invocation_output_access;
   l.var.prm_attr.mask = per_primitive_output_mask & ~attr_ring_per_primitive_output_mask &
                         ~cross_invocation_output_access;

   /* Workgroup information, see ms_workgroup_* for the layout. */
   l.lds.workgroup_info_addr = align(l.lds.total_size, 16);
   l.lds.total_size = l.lds.workgroup_info_addr + 16;

   /* Per-vertex and per-primitive output attributes.
    * Outputs without cross-invocation access are not included here.
    * First, try to put all outputs into LDS (shared memory).
    * If they don't fit, try to move them to VRAM one by one.
    */
   l.lds.vtx_attr.addr = align(l.lds.total_size, 16);
   l.lds.vtx_attr.mask = lds_per_vertex_output_mask;
   if (packed_vertex_comps) {
      l.lds.vtx_packed = true;
      memcpy(l.lds.vtx_comps, packed_vertex_comps, sizeof(l.lds.vtx_comps));
   }
   l.lds.prm_attr.mask = lds_per_primitive_output_mask;
   ms_calculate_arrayed_output_layout(&l, max_vertices, max_primitives);

   /* NGG shaders can only address up to 32K LDS memory.
    * The spec requires us to allow the application to use at least up to 28K
    * shared memory. Additionally, we reserve 2K for driver internal use
    * (eg. primitive indices and such, see below).
    *
    * Move the outputs that do not fit LDS, to VRAM.
    * Start with per-primitive attributes, because those are grouped at the end.
    */
   /* Private bary corners were admitted with a packed-output bound. Reserve
    * indices, inverse map, counts and alignment explicitly. Reference PARAMs
    * alias position and have already been removed from the output mask. */
   const unsigned usable_lds_bytes = private_bary ? 32 * 1024 - (6 * max_primitives + 128) :
      ((cross_invocation_cull_primitive || cross_invocation_indices) ? 30 : 31) * 1024;
   while (l.lds.total_size >= usable_lds_bytes) {
      if (l.lds.prm_attr.mask)
         ms_move_output(&l.lds.prm_attr, &l.scratch_ring.prm_attr);
      else if (l.lds.vtx_attr.mask)
         ms_move_output(&l.lds.vtx_attr, &l.scratch_ring.vtx_attr);
      else
         UNREACHABLE("API shader uses too much shared memory.");

      ms_calculate_arrayed_output_layout(&l, max_vertices, max_primitives);
   }

   /* RADV_BC250_MESH_IMPLICIT_TRIS: implicit indices need no area. The limit above
    * still reserves the room, so outputs move to the scratch ring exactly as
    * without the switch. */
   if (cross_invocation_indices && !implicit_indices) {
      /* Indices: flat array of 8-bit vertex indices for each primitive. */
      l.lds.indices_addr = align(l.lds.total_size, 16);
      l.lds.total_size = l.lds.indices_addr + max_primitives * vertices_per_prim;
   }

   if (cross_invocation_cull_primitive) {
      /* Cull flags: array of 8-bit cull flags for each primitive, 1=cull, 0=keep. */
      l.lds.cull_flags_addr = align(l.lds.total_size, 16);
      l.lds.total_size = l.lds.cull_flags_addr + max_primitives;
   }

   /* NGG is only allowed to address up to 32K of LDS. */
   assert(l.lds.total_size <= 32 * 1024);
   return l;
}

/* RADV_BC250_MESH_AUTOCULL: the culling layout only appends to the switch-off
 * layout (cull flags, the autocull area), so the switch-off epilogue finds every
 * output, index and flag it reads at the same place. The base driver's expanded shaders already
 * access all their outputs across invocations, so this holds for them. */
static bool
ms_autocull_layouts_compatible(const ms_out_mem_layout *plain, const ms_out_mem_layout *cull,
                               uint64_t plain_access)
{
   if (plain->var.vtx_attr.mask != cull->var.vtx_attr.mask ||
       plain->var.prm_attr.mask != cull->var.prm_attr.mask ||
       plain->scratch_ring.vtx_attr.mask || plain->scratch_ring.prm_attr.mask ||
       plain->attr_ring.vtx_attr.mask || plain->attr_ring.prm_attr.mask ||
       plain->lds.workgroup_info_addr != cull->lds.workgroup_info_addr ||
       plain->lds.vtx_attr.mask != cull->lds.vtx_attr.mask ||
       plain->lds.vtx_attr.addr != cull->lds.vtx_attr.addr ||
       plain->lds.prm_attr.mask != cull->lds.prm_attr.mask ||
       plain->lds.prm_attr.addr != cull->lds.prm_attr.addr)
      return false;
   if ((plain_access & VARYING_BIT_PRIMITIVE_INDICES) && plain->lds.indices_addr != cull->lds.indices_addr)
      return false;
   if ((plain_access & VARYING_BIT_CULL_PRIMITIVE) && plain->lds.cull_flags_addr != cull->lds.cull_flags_addr)
      return false;
   return true;
}

/* RADV_BC250_MESH_COMPACT_LDS: the dwords of every per-vertex output location
 * that the shader stores or loads (highest component + 1). Returns false when
 * a record cannot be packed: an access with a slot offset (arrays, indirect
 * addressing) or a location wider than one slot. */
static bool
ms_packed_vertex_components(nir_shader *shader, uint8_t comps[VARYING_SLOT_MAX])
{
   memset(comps, 0, VARYING_SLOT_MAX);
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            unsigned used;
            if (intrin->intrinsic == nir_intrinsic_store_per_vertex_output)
               used = util_last_bit(nir_intrinsic_write_mask(intrin));
            else if (intrin->intrinsic == nir_intrinsic_load_per_vertex_output)
               used = intrin->def.num_components;
            else
               continue;
            const nir_io_semantics sem = nir_intrinsic_io_semantics(intrin);
            nir_src *offset = nir_get_io_offset_src(intrin);
            const unsigned bits = intrin->intrinsic == nir_intrinsic_store_per_vertex_output ?
                                  intrin->src[0].ssa->bit_size : intrin->def.bit_size;
            if (sem.num_slots != 1 || !nir_src_is_const(*offset) || nir_src_as_uint(*offset) || bits > 32)
               return false;
            used += nir_intrinsic_component(intrin);
            if (used > 4)
               return false;
            comps[sem.location] = MAX2(comps[sem.location], used);
         }
      }
   }
   return true;
}

static bool
ms_direct_in_loop(nir_block *block, nir_loop *loop)
{
   for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
      if (n == &loop->cf_node)
         return true;
   }
   return false;
}

/* The value computation of a direct location: v, constants, ALU and
 * non-volatile load_shared inside the loop. */
static bool
ms_direct_closure_ok(nir_def *def, const ms_direct_read *dr, struct set *visited, unsigned depth)
{
   if (def == dr->vertex || _mesa_set_search(visited, def))
      return true;
   if (depth > 128)
      return false;

   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
      break;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (!ms_direct_closure_ok(alu->src[i].src.ssa, dr, visited, depth + 1))
            return false;
      }
      break;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (intrin->intrinsic != nir_intrinsic_load_shared ||
          (nir_intrinsic_access(intrin) & ACCESS_VOLATILE) ||
          !ms_direct_in_loop(instr->block, dr->loop) ||
          !ms_direct_closure_ok(intrin->src[0].ssa, dr, visited, depth + 1))
         return false;
      break;
   }
   default:
      return false;
   }

   _mesa_set_add(visited, def);
   return true;
}

static bool
ms_is_output_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_load_output:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_load_per_primitive_output:
      return true;
   default:
      return false;
   }
}

/* RADV_BC250_MESH_DIRECT_READ: find the direct locations (see ms_direct_read).
 * Returns NULL and fills dr, or the reason why no location is direct. */
static const char *
ms_direct_read_analyze(nir_shader *shader, unsigned api_workgroup_size,
                       uint64_t per_vertex_outputs, ms_direct_read *dr)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   /* The terminal loop: the last control flow node before the (empty) end. */
   nir_cf_node *tail = exec_node_data(nir_cf_node, exec_list_get_tail(&impl->body), node);
   if (tail->type != nir_cf_node_block || !exec_list_is_empty(&nir_cf_node_as_block(tail)->instr_list))
      return "code after the terminal loop";
   nir_cf_node *prev = nir_cf_node_prev(tail);
   if (!prev || prev->type != nir_cf_node_loop)
      return "no terminal loop";
   nir_loop *loop = nir_cf_node_as_loop(prev);
   if (nir_loop_has_continue_construct(loop))
      return "loop continue construct";
   dr->loop = loop;

   /* v = phi(local_invocation_index, v + api_workgroup_size) */
   nir_block *header = nir_loop_first_block(loop);
   nir_block *preheader = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_phi_instr *vphi = NULL;
   nir_foreach_phi(phi, header) {
      if (vphi)
         return "several loop phis";
      vphi = phi;
   }
   if (!vphi || vphi->def.num_components != 1 || vphi->def.bit_size != 32)
      return "no vertex index phi";
   nir_def *init = NULL, *next = NULL;
   unsigned num_srcs = 0;
   nir_foreach_phi_src(src, vphi) {
      num_srcs++;
      if (src->pred == preheader)
         init = src->src.ssa;
      else
         next = src->src.ssa;
   }
   if (num_srcs != 2 || !init || !next)
      return "vertex index phi sources";
   if (nir_def_instr(init)->type != nir_instr_type_intrinsic ||
       nir_instr_as_intrinsic(nir_def_instr(init))->intrinsic != nir_intrinsic_load_local_invocation_index)
      return "vertex index does not start at the local invocation index";
   if (nir_def_instr(next)->type != nir_instr_type_alu)
      return "vertex index step";
   nir_alu_instr *step = nir_instr_as_alu(nir_def_instr(next));
   if (step->op != nir_op_iadd || step->def.num_components != 1)
      return "vertex index step";
   bool step_ok = false;
   for (unsigned i = 0; i < 2; i++) {
      nir_scalar other = nir_scalar_chase_alu_src(nir_get_scalar(&step->def, 0), 1 - i);
      nir_scalar self = nir_scalar_chase_alu_src(nir_get_scalar(&step->def, 0), i);
      if (self.def == &vphi->def && nir_scalar_is_const(other) &&
          nir_scalar_as_uint(other) == api_workgroup_size)
         step_ok = true;
   }
   if (!step_ok)
      return "vertex index step is not the API workgroup size";

   /* if (v >= bound) break; right after the header. */
   nir_cf_node *check = nir_cf_node_next(&header->cf_node);
   if (!check || check->type != nir_cf_node_if)
      return "no bound check";
   nir_if *bound_if = nir_cf_node_as_if(check);
   nir_block *then_block = nir_if_first_then_block(bound_if);
   nir_block *else_block = nir_if_first_else_block(bound_if);
   if (then_block != nir_if_last_then_block(bound_if) || else_block != nir_if_last_else_block(bound_if) ||
       !exec_list_is_empty(&else_block->instr_list) || exec_list_length(&then_block->instr_list) != 1 ||
       nir_block_first_instr(then_block)->type != nir_instr_type_jump ||
       nir_instr_as_jump(nir_block_first_instr(then_block))->type != nir_jump_break)
      return "bound check shape";
   nir_def *vertex_count = NULL;
   unsigned count_intrinsics = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
            vertex_count = nir_instr_as_intrinsic(instr)->src[0].ssa;
            count_intrinsics++;
         }
      }
   }
   if (count_intrinsics != 1)
      return "not one SetMeshOutputs";
   nir_scalar cond = nir_get_scalar(bound_if->condition.ssa, 0);
   if (!nir_scalar_is_alu(cond) || nir_scalar_alu_op(cond) != nir_op_uge ||
       nir_scalar_chase_alu_src(cond, 0).def != &vphi->def ||
       nir_scalar_chase_alu_src(cond, 1).def != vertex_count)
      return "bound check is not v >= vertex count";

   /* No side effect in the loop but the output stores. */
   unsigned jumps = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      for (nir_cf_node *n = block->cf_node.parent; n != &loop->cf_node; n = n->parent) {
         if (n->type == nir_cf_node_loop)
            return "nested loop";
      }
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump) {
            jumps++;
            continue;
         }
         if (instr->type == nir_instr_type_call)
            return "call in the loop";
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         const nir_intrinsic_info *info = &nir_intrinsic_infos[intrin->intrinsic];
         if (intrin->intrinsic == nir_intrinsic_load_shared) {
            if (nir_intrinsic_access(intrin) & ACCESS_VOLATILE)
               return "volatile LDS load in the loop";
         } else if (intrin->intrinsic != nir_intrinsic_store_per_vertex_output &&
                    intrin->intrinsic != nir_intrinsic_store_per_primitive_output &&
                    !((info->flags & NIR_INTRINSIC_CAN_ELIMINATE) && (info->flags & NIR_INTRINSIC_CAN_REORDER))) {
            return "side effect in the loop";
         }
      }
   }
   if (jumps != 1)
      return "jumps in the loop";

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump &&
             nir_instr_as_jump(instr)->type != nir_jump_break &&
             nir_instr_as_jump(instr)->type != nir_jump_continue)
            return "early exit";
      }
   }

   /* Candidate locations: every store in one block of the loop, index v. */
   dr->vertex = &vphi->def;
   uint64_t candidates = per_vertex_outputs & ~shader->info.per_primitive_outputs &
                         ~shader->info.outputs_read & ~shader->info.outputs_read_indirectly &
                         ~shader->info.outputs_written_indirectly &
                         (VARYING_BIT_POS | VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 |
                          BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32));
   uint64_t rejected = 0;
   nir_block *store_block[VARYING_SLOT_MAX] = {0};
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (!ms_is_output_intrinsic(intrin->intrinsic))
            continue;
         const nir_io_semantics sem = nir_intrinsic_io_semantics(intrin);
         const uint64_t slots = BITFIELD64_RANGE(sem.location, MAX2(sem.num_slots, 1));
         if (intrin->intrinsic != nir_intrinsic_store_per_vertex_output) {
            rejected |= slots;
            continue;
         }
         nir_src *offset = nir_get_io_offset_src(intrin);
         /* The block runs once per iteration that passed the bound check. */
         const bool per_iteration = block->cf_node.parent == &loop->cf_node && block != header;
         if (sem.num_slots != 1 || !nir_src_is_const(*offset) || nir_src_as_uint(*offset) != 0 ||
             sem.high_16bits || intrin->src[0].ssa->bit_size != 32 ||
             nir_get_io_arrayed_index_src(intrin)->ssa != &vphi->def || !per_iteration ||
             (store_block[sem.location] && store_block[sem.location] != block)) {
            rejected |= slots;
            continue;
         }
         store_block[sem.location] = block;
      }
   }
   candidates &= ~rejected;

   struct set *visited = _mesa_pointer_set_create(NULL);
   u_foreach_bit64(loc, candidates) {
      bool ok = store_block[loc] != NULL;
      if (ok) {
         nir_foreach_instr(instr, store_block[loc]) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_store_per_vertex_output ||
                nir_intrinsic_io_semantics(intrin).location != loc)
               continue;
            if (!ms_direct_closure_ok(intrin->src[0].ssa, dr, visited, 0)) {
               ok = false;
               break;
            }
            /* One store per component (the expansion copies each once). */
            const unsigned component = nir_intrinsic_component(intrin);
            u_foreach_bit(c, nir_intrinsic_write_mask(intrin)) {
               if (component + c >= 4 || dr->value[loc][component + c].def) {
                  ok = false;
                  break;
               }
               dr->value[loc][component + c] = nir_get_scalar(intrin->src[0].ssa, c);
            }
         }
      }
      if (!ok) {
         candidates &= ~BITFIELD64_BIT(loc);
         memset(dr->value[loc], 0, sizeof(dr->value[loc]));
      }
   }
   _mesa_set_destroy(visited, NULL);

   dr->mask = candidates;
   return candidates ? NULL : "no eligible location";
}

/* Remove the stores of the direct locations, keeping their output metadata. */
/* BC250 barycentrics: radv_nir_bc250_export_bary_ref copied every position
 * store to the reference slots, so that they get parameter offsets and the
 * fragment shader inputs. Drop those copies before anything looks at the
 * output stores (LDS layout, direct read): the slots take no LDS, and
 * emit_ms_vertex exports the position it loads for the vertex to them. That is
 * the value the copies stored, for every Mesh route (plain, expanded, direct
 * read, packed/autocull vertices load the position of their source vertex).
 */
static void
ms_bary_ref_remove_stores(nir_shader *shader, uint64_t mask)
{
   nir_foreach_block(block, nir_shader_get_entrypoint(shader)) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic != nir_intrinsic_store_per_vertex_output &&
             intrin->intrinsic != nir_intrinsic_store_output)
            continue;
         if (mask & BITFIELD64_BIT(nir_intrinsic_io_semantics(intrin).location))
            nir_instr_remove(instr);
      }
   }
}

static void
ms_direct_read_remove_stores(nir_shader *shader, lower_ngg_ms_state *s)
{
   nir_foreach_block(block, nir_shader_get_entrypoint(shader)) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic != nir_intrinsic_store_per_vertex_output)
            continue;
         const nir_io_semantics sem = nir_intrinsic_io_semantics(intrin);
         if (!(s->direct->mask & BITFIELD64_BIT(sem.location)))
            continue;
         update_ms_output_info(sem, nir_get_io_offset_src(intrin), nir_intrinsic_write_mask(intrin),
                               nir_intrinsic_component(intrin), intrin->src[0].ssa->bit_size, NULL, s);
         nir_instr_remove(instr);
         s->direct->stores++;
      }
   }
}

/* RADV_BC250_MESH_IMPLICIT_TRIS (see ms_implicit_index): the shape of an
 * expanded shader that the lowering can check. NULL when the indices are
 * implicit, otherwise the reason they are not. */
static const char *
ms_implicit_indices_ok(const nir_shader *shader, const ac_nir_lower_ngg_options *options,
                       unsigned vertices_per_prim)
{
   if (!options->bc250_implicit_tris)
      return "switch off";
   if (!(shader->info.outputs_written & VARYING_BIT_PRIMITIVE_INDICES))
      return "no primitive indices";
   if ((shader->info.outputs_read | shader->info.outputs_read_indirectly |
        shader->info.outputs_written_indirectly) & VARYING_BIT_PRIMITIVE_INDICES)
      return "primitive indices read back or indirect";
   if (shader->info.mesh.max_vertices_out != vertices_per_prim * shader->info.mesh.max_primitives_out ||
       shader->info.mesh.max_vertices_out > 256)
      return "not N private vertices per primitive";
   return NULL;
}

bool
ac_nir_lower_ngg_mesh(nir_shader *shader, const ac_nir_lower_ngg_options *options,
                      bool *out_needs_scratch_ring)
{
   unsigned vertices_per_prim =
      mesa_vertices_per_prim(shader->info.mesh.primitive_type);

   /* BC250 barycentrics (ms_bary_ref_remove_stores). GFX10 without attribute ring only. */
   const uint64_t bary_ref_mask = options->bc250_bary_ref_mask & shader->info.outputs_written &
                                  ~shader->info.per_primitive_outputs;
   assert(!bary_ref_mask || !options->compiler_info->has_attr_ring);
   if (bary_ref_mask)
      ms_bary_ref_remove_stores(shader, bary_ref_mask);

   uint64_t per_vertex_outputs =
      shader->info.outputs_written & ~shader->info.per_primitive_outputs & ~SPECIAL_MS_OUT_MASK & ~bary_ref_mask;
   uint64_t per_primitive_outputs =
      shader->info.per_primitive_outputs & shader->info.outputs_written;

   /* Whether the shader uses CullPrimitiveEXT */
   bool uses_cull = shader->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE;
   /* Can't handle indirect register addressing, pretend as if they were cross-invocation. */
   uint64_t cross_invocation_access = shader->info.mesh.ms_cross_invocation_output_access |
                                      (shader->info.outputs_read_indirectly |
                                       shader->info.outputs_written_indirectly);

   unsigned max_vertices = shader->info.mesh.max_vertices_out;
   unsigned max_primitives = shader->info.mesh.max_primitives_out;
   const bool safe_direct = options->bc250_safe_direct;
   if (safe_direct) {
      assert(options->compiler_info->gfx_level == GFX10 && !options->compiler_info->has_attr_ring);
      assert(!options->multiview && !uses_cull && vertices_per_prim == 3);
      assert(max_vertices > 0 && max_vertices <= 256 && max_primitives > 0 && max_primitives <= 256);
      assert(options->bc250_safe_direct_bound <= options->max_workgroup_size &&
             options->bc250_safe_direct_bound <= 256);
      cross_invocation_access |= per_vertex_outputs | VARYING_BIT_PRIMITIVE_INDICES;
   }

   /* Effective compiler policy is supplied by RADV and keyed before lookup. */
   bool compact_cull = uses_cull &&
      options->bc250_cull_compact &&
      !options->compiler_info->has_attr_ring &&
      options->max_workgroup_size <= options->wave_size &&
      max_primitives <= options->wave_size;
   /* Bounded rendering-preserving experiment: duplicate shared vertices as needed
    * into a packed triangle list. No attributes or multiwave support here.
    */
   bool pack_triangle_vertices = compact_cull &&
      options->bc250_pack_triangle_vertices &&
      options->compiler_info->gfx_level == GFX10 && !options->multiview &&
      vertices_per_prim == 3 && max_primitives > 0 &&
      max_primitives <= max_vertices / 3 && max_vertices <= options->wave_size &&
      per_vertex_outputs == VARYING_BIT_POS &&
      !(per_primitive_outputs & ~SPECIAL_MS_OUT_MASK);
   /* RADV_BC250_MESH_COMPACT (ms_compact_vertices): only the base driver's expanded
    * private-vertex triangle shape (3 vertices per declared primitive, every lane of the
    * 3P keys available), no CullPrimitive (null primitives would leave their vertices
    * unreferenced), no per-primitive export argument and no per-vertex layer/viewport
    * (taken from a vertex the renumbering may share). Every per-vertex output becomes
    * readable by any lane (a slot exports another lane's expanded vertex). */
   const uint32_t *compact_ix = options->bc250_compact_index_staging;
   const char *compact_reason = NULL;
   if (!options->bc250_compact)
      compact_reason = "switch off";
   else if (options->compiler_info->gfx_level != GFX10 || options->compiler_info->has_attr_ring || options->multiview)
      compact_reason = "not GFX10 without attribute ring, or multiview";
   else if (vertices_per_prim != 3 || !max_primitives || max_vertices != 3 * max_primitives)
      compact_reason = "not the expanded triangle shape";
   else if (3 * max_primitives > options->max_workgroup_size || options->max_workgroup_size > 256 ||
            DIV_ROUND_UP(options->max_workgroup_size, options->wave_size) > ms_autocull_max_waves)
      compact_reason = "lanes";
   else if (uses_cull || compact_cull || pack_triangle_vertices)
      compact_reason = "CullPrimitive";
   else if (per_primitive_outputs & ~SPECIAL_MS_OUT_MASK)
      compact_reason = "per-primitive export argument";
   else if (shader->info.outputs_written & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT | VARYING_BIT_VIEWPORT_MASK |
                                            VARYING_BIT_PRIMITIVE_SHADING_RATE))
      compact_reason = "layer/viewport/shading rate output";
   else if (!compact_ix[1] || (compact_ix[2] != 2 && compact_ix[2] != 4) || !compact_ix[3] || compact_ix[3] > 256 ||
            compact_ix[0] + max_primitives * compact_ix[1] > shader->info.shared_size)
      compact_reason = "index staging";
   else if (util_bitcount64(bary_ref_mask) == 1)
      /* RADV_BC250_BARY_CORNER_ID exports export_index % 3 as the corner number: it needs private
       * corners (slot 3p + c). The two-slot position reference is independent of vertex sharing. */
      compact_reason = "one-slot barycentric corner number";
   bool compact = !compact_reason;
   if (compact)
      cross_invocation_access |= per_vertex_outputs;
   uint64_t original_cross_invocation_access = cross_invocation_access;

   /* RADV_BC250_MESH_IMPLICIT_TRIS (see ms_implicit_index). */
   const char *implicit_reason = ms_implicit_indices_ok(shader, options, vertices_per_prim);
   const bool implicit_indices = !implicit_reason;
   if (options->bc250_implicit_tris && getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH IMPLICIT TRIS: %s V=%u P=%u%s%s\n", implicit_indices ? "applied" : "not applied",
              max_vertices, max_primitives, implicit_reason ? " reason=" : "", implicit_reason ? implicit_reason : "");

   /* RADV_BC250_MESH_DIRECT_READ (ms_direct_read): direct locations get no
    * LDS record; everything else keeps its layout. */
   const unsigned api_workgroup_size = shader->info.workgroup_size[0] *
                                       shader->info.workgroup_size[1] *
                                       shader->info.workgroup_size[2];
   ms_direct_read direct = {0};
   const char *direct_reason = "switch off";
   if (options->bc250_direct_read) {
      if (options->compiler_info->gfx_level != GFX10 || options->compiler_info->has_attr_ring || options->multiview)
         direct_reason = "not GFX10 without attribute ring, or multiview";
      else
         direct_reason = ms_direct_read_analyze(shader, api_workgroup_size, per_vertex_outputs, &direct);
      if (direct_reason)
         direct.mask = 0;
   }
   const uint64_t layout_per_vertex_outputs = per_vertex_outputs & ~direct.mask;

   uint8_t packed_vertex_comps[VARYING_SLOT_MAX];
   const uint8_t *packed_comps =
      options->bc250_compact_lds && !options->compiler_info->has_attr_ring &&
      ms_packed_vertex_components(shader, packed_vertex_comps) ? packed_vertex_comps : NULL;

   /* RADV_BC250_MESH_AUTOCULL (see ms_autocull_compact). RADV enables it only for
    * The base driver's private-vertex (expanded) triangle Mesh shaders; the output is always
    * packed private-vertex triangles. Every packed vertex needs a lane. The viewport index
    * is not handled (the frustum and small-primitive tests use viewport 0): such shaders
    * are left alone. Clip and cull distances are per-vertex outputs like any other: the
    * packed vertices load them from their source vertex (ms_autocull_accept also culls
    * with them). */
   const uint64_t autocull_unhandled_outputs = VARYING_BIT_VIEWPORT | VARYING_BIT_VIEWPORT_MASK;
   bool autocull = options->bc250_autocull &&
      options->compiler_info->gfx_level == GFX10 && !options->compiler_info->has_attr_ring &&
      !options->multiview && vertices_per_prim == 3 &&
      (per_vertex_outputs & VARYING_BIT_POS) &&
      !(shader->info.outputs_written & autocull_unhandled_outputs) &&
      max_primitives > 0 && DIV_ROUND_UP(max_primitives, options->wave_size) <= ms_autocull_max_waves &&
      max_vertices <= 256 && max_primitives * 3 <= options->max_workgroup_size;
   const bool plain_compact_cull = compact_cull;
   const bool plain_pack_triangle_vertices = pack_triangle_vertices;
   if (autocull) {
      compact_cull = true;
      pack_triangle_vertices = true;
      /* Packed vertices load the outputs of their source vertex. */
      cross_invocation_access |= per_vertex_outputs;
      if (uses_cull)
         cross_invocation_access |= VARYING_BIT_CULL_PRIMITIVE;
   }

   if (pack_triangle_vertices)
      cross_invocation_access |= VARYING_BIT_POS;
   if (compact_cull)
      cross_invocation_access |= per_primitive_outputs |
         VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_CULL_PRIMITIVE;

   ms_out_mem_layout layout = ms_calculate_output_layout(
      options->compiler_info, shader->info.shared_size, layout_per_vertex_outputs, per_primitive_outputs,
      cross_invocation_access, max_vertices, max_primitives, vertices_per_prim, packed_comps, implicit_indices,
      options->bc250_safe_corners && options->bc250_safe_bary);

   uint32_t autocull_lds_addr = 0;
   if (autocull) {
      autocull_lds_addr = align(layout.lds.total_size, 16);
      const uint32_t autocull_total = autocull_lds_addr + ms_autocull_map + max_primitives;
      /* The layout the shader has with the switch off: the non-culling side of the
       * runtime branch runs the switch-off epilogue on it. */
      uint64_t plain_access = original_cross_invocation_access;
      if (plain_pack_triangle_vertices)
         plain_access |= VARYING_BIT_POS;
      if (plain_compact_cull)
         plain_access |= per_primitive_outputs | VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_CULL_PRIMITIVE;
      const ms_out_mem_layout plain_layout = ms_calculate_output_layout(
         options->compiler_info, shader->info.shared_size, layout_per_vertex_outputs, per_primitive_outputs,
         plain_access, max_vertices, max_primitives, vertices_per_prim, packed_comps, implicit_indices, false);
      if (layout.scratch_ring.vtx_attr.mask || layout.scratch_ring.prm_attr.mask ||
          autocull_total > 32 * 1024 ||
          !ms_autocull_layouts_compatible(&plain_layout, &layout, plain_access)) {
         /* No autocull when outputs spill to the Mesh scratch ring or the switch-off
          * epilogue would read its outputs elsewhere: fall back to exactly what the
          * shader gets without it. */
         autocull = false;
         compact_cull = plain_compact_cull;
         pack_triangle_vertices = plain_pack_triangle_vertices;
         cross_invocation_access = original_cross_invocation_access;
         if (pack_triangle_vertices)
            cross_invocation_access |= VARYING_BIT_POS;
         if (compact_cull)
            cross_invocation_access |= per_primitive_outputs |
               VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_CULL_PRIMITIVE;
         layout = ms_calculate_output_layout(
            options->compiler_info, shader->info.shared_size, layout_per_vertex_outputs, per_primitive_outputs,
            cross_invocation_access, max_vertices, max_primitives, vertices_per_prim, packed_comps, implicit_indices, false);
      } else {
         layout.lds.total_size = autocull_total;
      }
   }

   /* RADV_BC250_MESH_CULLDIST_CULL: exported cull distances (bit i: element i of the merged
    * clip/cull array, ac_nir_export_position's numbering). */
   const uint32_t cull_export_mask = options->export_clipdist_mask &
      BITFIELD_RANGE(shader->info.clip_distance_array_size, shader->info.cull_distance_array_size);
   const bool dont_export_cull = autocull && options->bc250_autocull_culldist && cull_export_mask;
   if (options->bc250_culldist_culled)
      *options->bc250_culldist_culled = dont_export_cull;

   if (options->bc250_autocull && getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH AUTOCULL: %s V=%u P=%u wave=%u hw_workgroup=%u lds=%u app_cull=%u%s%s\n",
              autocull ? "applied" : "not applied", max_vertices, max_primitives, options->wave_size,
              options->max_workgroup_size, layout.lds.total_size, uses_cull,
              autocull && !dont_export_cull ? " runtime_skip=no_face_cull" : "",
              dont_export_cull ? " culldist=culled_not_exported always_cull=1" : "");

   /* Scratch-ring synchronization/remapping is outside this prototype. */
   if (compact_cull && (layout.scratch_ring.vtx_attr.mask || layout.scratch_ring.prm_attr.mask)) {
      compact_cull = false;
      pack_triangle_vertices = false;
      layout = ms_calculate_output_layout(
         options->compiler_info, shader->info.shared_size, layout_per_vertex_outputs, per_primitive_outputs,
         original_cross_invocation_access, max_vertices, max_primitives, vertices_per_prim, packed_comps, implicit_indices, false);
   }

   /* RADV_BC250_MESH_COMPACT LDS area, after everything else. Without it (no room, or
    * outputs on the Mesh scratch ring) the shader is compiled as with the switch off. */
   uint32_t compact_lds_addr = 0;
   if (compact) {
      compact_lds_addr = align(layout.lds.total_size, 16);
      const uint32_t total = compact_lds_addr + ms_compact_lds_size(compact_ix[3], max_primitives);
      if (layout.scratch_ring.vtx_attr.mask || layout.scratch_ring.prm_attr.mask || total > 32 * 1024) {
         compact_reason = "LDS";
         compact = false;
      } else {
         layout.lds.total_size = total;
      }
   }
   if (options->bc250_compact && getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH COMPACT: %s W=%u D=%u V=%u P=%u keys=%u lanes=%u wave=%u table=%u owned=0x%x "
              "autocull=%u direct=%u lds=%u%s%s\n",
              compact ? "applied" : "not applied", AC_NIR_BC250_COMPACT_W, AC_NIR_BC250_COMPACT_W + 1, max_vertices,
              max_primitives, 3 * max_primitives, options->max_workgroup_size, options->wave_size, compact_ix[3],
              options->bc250_compact_owned_corners, autocull, direct.mask != 0, layout.lds.total_size,
              compact_reason ? " reason=" : "", compact_reason ? compact_reason : "");
   if (options->bc250_compact_applied)
      *options->bc250_compact_applied = compact;

   if (options->bc250_direct_read && getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH DIRECT READ: %s locations=0x%" PRIx64 " (%u of %u per-vertex)%s%s\n",
              direct.mask ? "applied" : "not applied", direct.mask, util_bitcount64(direct.mask),
              util_bitcount64(per_vertex_outputs), direct_reason ? " reason=" : "", direct_reason ? direct_reason : "");

   if (getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH LDS: V=%u P=%u api_shared=%u vtx_attr=%u@%u%s prm_attr=%u@%u indices=%u total=%u "
              "scratch_vtx=%u scratch_prm=%u var_vtx=%u var_prm=%u\n",
              max_vertices, max_primitives, shader->info.shared_size,
              util_bitcount64(layout.lds.vtx_attr.mask), layout.lds.vtx_attr.addr,
              layout.lds.vtx_packed ? " packed" : "",
              util_bitcount64(layout.lds.prm_attr.mask), layout.lds.prm_attr.addr,
              (cross_invocation_access & VARYING_BIT_PRIMITIVE_INDICES) && !implicit_indices ?
                 layout.lds.indices_addr : 0,
              layout.lds.total_size, util_bitcount64(layout.scratch_ring.vtx_attr.mask),
              util_bitcount64(layout.scratch_ring.prm_attr.mask),
              util_bitcount64(layout.var.vtx_attr.mask), util_bitcount64(layout.var.prm_attr.mask));

   uint32_t safe_direct_map_addr = 0, safe_direct_latest_addr = 0;
   uint32_t safe_direct_indices_addr = 0, safe_direct_counts_addr = 0;
   bool safe_compact = false;
   bool pp_share = false;
   if (safe_direct) {
      assert(!layout.scratch_ring.vtx_attr.mask && !layout.scratch_ring.prm_attr.mask);
      /* Direct-read removes LDS copies by cloning the original per-vertex
       * expressions at export. It is compatible with the safe remap: the
       * remapped source index is passed to ms_direct_read_load. */
      assert(!compact && !autocull && !compact_cull && !implicit_indices);
      safe_direct_map_addr = align(layout.lds.total_size, 4);
      safe_direct_latest_addr = align(safe_direct_map_addr + options->bc250_safe_direct_bound, 4);
      safe_direct_indices_addr = safe_direct_latest_addr +
         (options->bc250_safe_local ? 4 :
          options->bc250_safe_parallel ? MAX2(4 * max_vertices, align(3 * max_primitives, 4) + 32) :
                                        4 * max_vertices);
      safe_direct_counts_addr = align(safe_direct_indices_addr + 3 * max_primitives, 4);
      if (options->bc250_safe_corners) {
         /* No remapped connectivity. Plain corners have no flags either, so the counts reuse
          * that word; RADV_BC250_MESH_SAFE_ADAPTIVE publishes its per-workgroup flag there,
          * so the counts follow it (other waves read both after the publication barrier). */
         safe_direct_counts_addr = options->bc250_safe_adaptive ? safe_direct_latest_addr + 4 :
                                                                  safe_direct_latest_addr;
      }
      layout.lds.total_size = safe_direct_counts_addr + 8 +
         (options->bc250_safe_autocull && !options->bc250_safe_local ? 12 : 0);
      /* RADV_BC250_MESH_SAFE_COMPACT: the renumbered survivor indices follow the counts. Without the
       * room (or with barycentric reference slots, whose corner numbers need private corners) the
       * shader keeps the three adaptive cases. The vertex map needs at most 3 * survivors entries,
       * within the corner bound. */
      /* RADV_BC250_MESH_PP_SHARE: the slot triples follow the counts (the map fits the corner bound). The
       * one-slot barycentric corner number needs private corners (slot 3p + c). */
      const uint32_t *pix = options->bc250_compact_index_staging;
      if (options->bc250_pp_share && !options->bc250_safe_adaptive && util_bitcount64(bary_ref_mask) != 1 &&
          pix[1] && (pix[2] == 2 || pix[2] == 4) && pix[3] && pix[3] <= 256 &&
          pix[0] + max_primitives * pix[1] <= shader->info.shared_size && max_primitives <= options->wave_size &&
          layout.lds.total_size + 3 * max_primitives <= 32 * 1024) {
         pp_share = true;
         safe_direct_indices_addr = layout.lds.total_size;
         layout.lds.total_size += 3 * max_primitives;
      }
      if (options->bc250_safe_compact && !bary_ref_mask &&
          layout.lds.total_size + 3 * max_primitives <= 32 * 1024) {
         safe_compact = true;
         safe_direct_indices_addr = layout.lds.total_size;
         layout.lds.total_size += 3 * max_primitives;
      }
      assert(layout.lds.total_size <= 32 * 1024);
      if (options->bc250_safe_direct_index_staging) {
         uint32_t *ix = options->bc250_safe_direct_index_staging;
         ix[0] = layout.lds.indices_addr;
         ix[1] = 3;
         ix[2] = 1;
         ix[3] = max_vertices;
      }
      if (options->bc250_safe_autocull && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE AUTOCULL: %s\n",
                 options->bc250_safe_local ? "before check and local remap" : "before parallel remap");
      if (options->bc250_safe_check && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE CHECK: wave-local counts, one publication\n");
      if (options->bc250_pp_share && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH PP SHARE: %s provoking=%u table=%u\n", pp_share ? "applied" : "not applied",
                 options->bc250_compact_owned_corners, options->bc250_compact_index_staging[3]);
      if (safe_compact && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE COMPACT: referenced vertices renumbered, W=31\n");
      if (options->bc250_safe_corners && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE CORNERS: private surviving corners, W=2, bound=%u\n", 3 * max_primitives);
      if (options->bc250_safe_local && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE LOCAL: one-wave 10-triangle intervals, packed LDS\n");
      if (options->bc250_safe_parallel && !options->bc250_safe_local && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE PARALLEL: source-key window=31\n");
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE DIRECT: applied W=31 V=%u P=%u lanes=%u wave=%u lds=%u\n",
                 max_vertices, max_primitives, options->max_workgroup_size, options->wave_size, layout.lds.total_size);
   }
   shader->info.shared_size = layout.lds.total_size;
   *out_needs_scratch_ring = layout.scratch_ring.vtx_attr.mask || layout.scratch_ring.prm_attr.mask;

   /* The workgroup size that is specified by the API shader may be different
    * from the size of the workgroup that actually runs on the HW, due to the
    * limitations of NGG: max 0/1 vertex and 0/1 primitive per lane is allowed.
    *
    * Therefore, we must make sure that when the API workgroup size is smaller,
    * we don't run the API shader on more HW invocations than is necessary.
    */

   nir_custom_divergence_analysis(shader,
      api_workgroup_size <= options->wave_size ? 0 : nir_divergence_across_subgroups);

   bool fast_launch_2 = options->compiler_info->gfx_level >= GFX11;

   unsigned hw_workgroup_size = options->max_workgroup_size;
   lower_ngg_ms_state state = {
      .options = options,
      .layout = layout,
      .wave_size = options->wave_size,
      .per_vertex_outputs = per_vertex_outputs,
      .per_primitive_outputs = per_primitive_outputs,
      .vertices_per_prim = vertices_per_prim,
      .api_workgroup_size = api_workgroup_size,
      .hw_workgroup_size = hw_workgroup_size,
      .insert_layer_output =
         options->multiview && !(shader->info.outputs_written & VARYING_BIT_LAYER),
      .uses_cull_flags = uses_cull,
      .compact_cull = compact_cull,
      .pack_triangle_vertices = pack_triangle_vertices,
      .autocull = autocull,
      .plain_compact_cull = plain_compact_cull,
      .plain_pack_triangle_vertices = plain_pack_triangle_vertices,
      .autocull_skip_viewport_state = options->skip_viewport_state_culling,
      .autocull_lds_addr = autocull_lds_addr,
      .dont_export_cull = dont_export_cull,
      .num_pos_exports = ms_num_pos_exports(per_vertex_outputs, options->export_clipdist_mask &
                                            ~(dont_export_cull ? cull_export_mask : 0)),
      .ac = options->compiler_info,
      .vert_multirow_export = fast_launch_2 && max_vertices > hw_workgroup_size,
      .prim_multirow_export = fast_launch_2 && max_primitives > hw_workgroup_size,
      .output_counts_workgroup_uniform = true,
      .vs_output_param_offset = options->vs_output_param_offset,
      .has_param_exports = options->has_param_exports,
      .implicit_indices = implicit_indices,
      .bary_ref_mask = bary_ref_mask,
      .compact = compact,
      .compact_lds_addr = compact_lds_addr,
      .compact_table = compact_ix[3],
      .safe_direct = safe_direct,
      .safe_direct_map_addr = safe_direct_map_addr,
      .safe_direct_latest_addr = safe_direct_latest_addr,
      .safe_direct_indices_addr = safe_direct_indices_addr,
      .safe_direct_counts_addr = safe_direct_counts_addr,
      .safe_compact = safe_compact,
      .pp_share = pp_share,
   };

   u_foreach_bit64(slot, bary_ref_mask) {
      state.out.infos[slot].components_mask = 0xf;
      state.out.infos[slot].as_varying_mask = 0xf;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   assert(impl);

   state.vertex_count_var =
      nir_local_variable_create(impl, glsl_uint_type(), "vertex_count_var");
   state.primitive_count_var =
      nir_local_variable_create(impl, glsl_uint_type(), "primitive_count_var");

   nir_builder builder = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &builder; /* This is to avoid the & */

   if (direct.mask) {
      direct.remap = _mesa_pointer_hash_table_create(NULL);
      state.direct = &direct;
      ms_direct_read_remove_stores(shader, &state);
   }

   handle_smaller_ms_api_workgroup(b, &state);
   if (!fast_launch_2)
      ms_emit_legacy_workgroup_index(b, &state);
   ms_create_same_invocation_vars(b, &state);

   lower_ms_intrinsics(shader, &state);

   emit_ms_finale(b, &state);

   if (direct.remap)
      _mesa_hash_table_destroy(direct.remap, NULL);
   state.direct = NULL;

   /* Take care of metadata and validation before calling other passes */
   nir_progress(true, impl, nir_metadata_none);
   nir_validate_shader(shader, "after emitting NGG MS");

   /* Cleanup */
   nir_lower_vars_to_ssa(shader);
   nir_remove_dead_variables(shader, nir_var_function_temp, NULL);
   nir_lower_alu_to_scalar(shader, NULL, NULL);
   nir_lower_phis_to_scalar(shader, ac_nir_lower_phis_to_scalar_cb, NULL);

   /* Optimize load_local_invocation_index. When the API workgroup is smaller than the HW workgroup,
    * local_invocation_id isn't initialized for all lanes and we can't perform this optimization for
    * all load_local_invocation_index.
    */
   if (fast_launch_2 && api_workgroup_size == hw_workgroup_size &&
       ((shader->info.workgroup_size[0] == 1) + (shader->info.workgroup_size[1] == 1) +
        (shader->info.workgroup_size[2] == 1)) == 2) {
      nir_lower_compute_system_values_options csv_options = {
         .lower_local_invocation_index = true,
      };
      nir_lower_compute_system_values(shader, &csv_options);
   }


   return true;
}

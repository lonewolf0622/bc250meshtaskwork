/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_IDX_H
#define RADV_BC250_IDX_H

#include "nir/nir_builder.h"
#include "radv_bc250_primitive.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Internal NIR builders, before pipeline compilation. The layout assigns an
 * output plane byte offset to driver_location on the logical Mesh variables.
 * The producer and fetch shader must use that same layout snapshot. */
#define RADV_BC250_IDX_CONSTANT_SIZE 72

/* Generic primitive attributes use the same flat fragment inputs as the direct Mesh route.
 * Encode a private provoking corner as V * (primitive + 1) + vertex within each workgroup.
 * This preserves the original vertex index while giving each primitive a distinct flat value. */
static inline bool
bc250_idx_has_pp(const nir_shader *mesh)
{
   return mesh->info.per_primitive_outputs & (UINT64_C(0xffffffff) << VARYING_SLOT_VAR0);
}

static inline unsigned
bc250_idx_vertex_stride(const nir_shader *mesh)
{
   return radv_bc250_primitive_vertex_stride(mesh->info.mesh.max_vertices_out, mesh->info.mesh.max_primitives_out,
                                           bc250_idx_has_pp(mesh));
}

struct bc250_idx_setup_args {
   uint64_t records, count, table, dims, draws, mesh, pool;
   uint32_t stride, max_count, capacity, prims;
   /* First setup of the command buffer: the pool is empty. */
   uint32_t reset, vertex_stride;
   uint32_t output_bytes;
};

/* Immutable after recording, like the side IB. Each batch has one pool and
 * disjoint per-call tables; overflow remains a suffix within each call. */
#define RADV_BC250_IDX_SETUP_BATCH_CALLS 32
#define RADV_BC250_IDX_SETUP_BATCH_HEADER_BYTES 16
#define RADV_BC250_IDX_SETUP_BATCH_BYTES \
   (RADV_BC250_IDX_SETUP_BATCH_HEADER_BYTES + RADV_BC250_IDX_SETUP_BATCH_CALLS * sizeof(struct bc250_idx_setup_args))

nir_shader *radv_bc250_idx_build_setup(const nir_shader_compiler_options *options, bool stored);
nir_shader *radv_bc250_idx_build_setup_batch(const nir_shader_compiler_options *options);

unsigned radv_bc250_idx_output_layout(nir_shader *mesh);
bool radv_bc250_idx_prune_outputs(nir_shader *mesh, const nir_shader *fragment);
nir_shader *radv_bc250_idx_build_cs(const nir_shader *mesh, bool indirect, unsigned provoking, unsigned output_bytes,
                                    nir_intrinsic_pass_cb lower_constants);
nir_shader *radv_bc250_idx_build_stored_vs(const nir_shader_compiler_options *options, const nir_shader *mesh,
                                           unsigned output_bytes);

#ifdef __cplusplus
}
#endif
#endif

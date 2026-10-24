/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_MESH_CONTRACT_H
#define RADV_BC250_MESH_CONTRACT_H

#include "nir/nir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Facts about the logical Mesh program, before any export or dispatch rewrite.
 * Route-specific admission must still check ownership, fragment inputs and the
 * cost of the chosen route. Discard this snapshot after changing the shader.
 */
struct radv_bc250_mesh_contract {
   mesa_shader_stage stage;
   enum mesa_prim primitive_type;
   unsigned workgroup_size[3];
   unsigned workgroup_invocations;
   unsigned max_vertices;
   unsigned max_primitives;
   unsigned task_payload_size;
   unsigned shared_size;
   unsigned min_subgroup_size, max_subgroup_size, api_subgroup_size;
   bool nv, workgroup_size_variable, shared_memory_explicit_layout;
   bool zero_initialize_shared_memory;
   enum gl_derivative_group derivative_group;
   uint64_t outputs_written;
   uint64_t outputs_read;
   uint64_t per_primitive_outputs;
   uint64_t outputs_written_16bit;
   uint64_t declared_per_primitive_outputs;
   bool has_per_primitive_data, has_per_primitive_cull;
   bool non_array_output, non_32bit_generic_output;
   bool triangle_index_array;
   unsigned function_count;
   bool has_calls;
   bool subgroup_ops, observes_subgroup;
   bool uses_num_workgroups, uses_task_payload;
   unsigned memory_effects;
   bool used_external_atomic_result;
   /* Output ownership is independent of output values and fragment policy. */
   const char *output_store_refusal;
   uint64_t subgroup_dependent_outputs;
   uint64_t shared_dependent_outputs;
   bool output_dependencies_unknown;

   unsigned allocation_static_sites;
   unsigned allocation_sites;
   bool allocation_analysis_complete;
   bool allocation_may_repeat;
   bool allocation_may_be_skipped;
   bool allocation_control_uniform;
   bool allocation_values_uniform;
   bool output_before_allocation;
};

enum radv_bc250_memory_effect {
   RADV_BC250_EXTERNAL_READ = 1 << 0,
   RADV_BC250_STORAGE_READ = 1 << 1,
   RADV_BC250_EXTERNAL_WRITE = 1 << 2,
   RADV_BC250_EXTERNAL_ATOMIC = 1 << 3,
   RADV_BC250_SHARED_READ = 1 << 4,
   RADV_BC250_SHARED_WRITE = 1 << 5,
   RADV_BC250_SHARED_ATOMIC = 1 << 6,
   RADV_BC250_VOLATILE_ACCESS = 1 << 7,
   RADV_BC250_EXTERNAL_BARRIER = 1 << 8,
};

bool radv_bc250_mesh_intrinsic_subgroup_op(const nir_intrinsic_instr *in);
unsigned radv_bc250_mesh_intrinsic_memory_effects(const nir_intrinsic_instr *in);
nir_deref_instr *radv_bc250_mesh_output_element(nir_deref_instr *d);
/* Gather facts without modifying instructions or metadata. Allocation CFG and
 * workgroup-uniformity proofs are separate, and only needed by some routes.
 * Dependency masks cover SSA sources and enclosing ifs; loops and indirect
 * temporary memory dependencies are explicitly reported as unknown.
 */
void radv_bc250_mesh_contract_init(const nir_shader *mesh, struct radv_bc250_mesh_contract *contract);
void radv_bc250_analyze_mesh_allocation(nir_shader *mesh, struct radv_bc250_mesh_contract *contract);
void radv_bc250_analyze_mesh_contract(nir_shader *mesh, struct radv_bc250_mesh_contract *contract);

/* Multiple static sites are usable when execution selects at most one of them.
 * An empty path is permitted: the index producer initializes its count to zero.
 * Uniform values are required by the producer's existing shared count stores.
 */
bool radv_bc250_mesh_allocation_is_canonical(const struct radv_bc250_mesh_contract *contract);

#ifdef __cplusplus
}
#endif
#endif

/* SPDX-License-Identifier: MIT */
#include "radv_bc250_mesh_contract.h"

#include <stdlib.h>
#include <string.h>

bool
radv_bc250_mesh_intrinsic_subgroup_op(const nir_intrinsic_instr *in)
{
   if (in->intrinsic == nir_intrinsic_load_subgroup_id || in->intrinsic == nir_intrinsic_load_num_subgroups)
      return false;
   static const char *const keys[] = {
      "subgroup",
      "ballot",
      "vote",
      "reduce",
      "scan",
      "shuffle",
      "quad",
      "elect",
      "read_invocation",
      "read_first_invocation",
      "rotate",
      "swizzle",
      "first_invocation",
      "last_invocation",
      "as_uniform",
      "dpp",
      "permlane",
      "mbcnt",
      "write_invocation",
   };
   const char *name = nir_intrinsic_infos[in->intrinsic].name;
   for (unsigned i = 0; i < ARRAY_SIZE(keys); i++) {
      if (strstr(name, keys[i]))
         return true;
   }
   return false;
}

unsigned
radv_bc250_mesh_intrinsic_memory_effects(const nir_intrinsic_instr *in)
{
   const char *name = nir_intrinsic_infos[in->intrinsic].name;
   unsigned effects = 0;
   bool external_read;
   const bool deref_read = in->intrinsic == nir_intrinsic_load_deref || in->intrinsic == nir_intrinsic_copy_deref;
   const unsigned source = in->intrinsic == nir_intrinsic_copy_deref ? 1 : 0;
   if (deref_read) {
      nir_deref_instr *d = nir_src_as_deref(in->src[source]);
      external_read = nir_deref_mode_may_be(d, nir_var_mem_ssbo | nir_var_mem_global | nir_var_mem_ubo | nir_var_image);
      if (nir_deref_mode_may_be(d, nir_var_mem_shared))
         effects |= RADV_BC250_SHARED_READ;
   } else {
      external_read = !strncmp(name, "load_ssbo", 9) || !strncmp(name, "load_ubo", 8) ||
                      !strncmp(name, "load_global", 11) || !strncmp(name, "load_buffer", 11) ||
                      (strstr(name, "image") && strstr(name, "load"));
   }
   if (external_read) {
      effects |= RADV_BC250_EXTERNAL_READ;
      bool ubo = deref_read ? !nir_deref_mode_may_be(nir_src_as_deref(in->src[source]),
                                                     nir_var_mem_ssbo | nir_var_mem_global | nir_var_image)
                            : !strncmp(name, "load_ubo", 8);
      unsigned access = in->intrinsic == nir_intrinsic_copy_deref ? nir_intrinsic_src_access(in)
                        : nir_intrinsic_has_access(in)            ? nir_intrinsic_access(in)
                                                                  : 0;
      if (!ubo && !(access & ACCESS_NON_WRITEABLE))
         effects |= RADV_BC250_STORAGE_READ;
   }
   if (nir_intrinsic_writes_external_memory(in)) {
      effects |= RADV_BC250_EXTERNAL_WRITE;
      if (nir_intrinsic_has_atomic_op(in) || strstr(name, "atomic"))
         effects |= RADV_BC250_EXTERNAL_ATOMIC;
   }
   if (strstr(name, "shared")) {
      if (strstr(name, "load"))
         effects |= RADV_BC250_SHARED_READ;
      if (strstr(name, "store"))
         effects |= RADV_BC250_SHARED_WRITE;
      if (strstr(name, "atomic") || (nir_intrinsic_has_access(in) && (nir_intrinsic_access(in) & ACCESS_ATOMIC)))
         effects |= RADV_BC250_SHARED_ATOMIC | RADV_BC250_SHARED_READ | RADV_BC250_SHARED_WRITE;
   }
   if (in->intrinsic == nir_intrinsic_store_deref || in->intrinsic == nir_intrinsic_copy_deref) {
      nir_deref_instr *d = nir_src_as_deref(in->src[0]);
      if (nir_deref_mode_may_be(d, nir_var_mem_shared))
         effects |= RADV_BC250_SHARED_WRITE;
      if (d->modes & ~(nir_var_shader_out | nir_var_mem_shared | nir_var_shader_temp | nir_var_function_temp))
         effects |= RADV_BC250_EXTERNAL_WRITE;
   }
   if ((nir_intrinsic_has_access(in) && (nir_intrinsic_access(in) & ACCESS_VOLATILE)) ||
       (in->intrinsic == nir_intrinsic_copy_deref &&
        ((nir_intrinsic_src_access(in) | nir_intrinsic_dst_access(in)) & ACCESS_VOLATILE)))
      effects |= RADV_BC250_VOLATILE_ACCESS;
   if (in->intrinsic == nir_intrinsic_barrier &&
       (nir_intrinsic_memory_modes(in) & ~(nir_var_mem_shared | nir_var_shader_out)))
      effects |= RADV_BC250_EXTERNAL_BARRIER;
   return effects;
}

enum output_dependency {
   dependency_subgroup = 1,
   dependency_shared = 2,
   dependency_unknown = 4,
};

struct dependency_state {
   unsigned depth, dependencies;
   uint8_t *cache;
};

static unsigned output_dependencies(nir_def *def, unsigned depth, uint8_t *cache);

static bool
source_dependencies(nir_src *src, void *data)
{
   struct dependency_state *state = data;
   state->dependencies |= output_dependencies(src->ssa, state->depth + 1, state->cache);
   return true;
}

static unsigned
output_dependencies(nir_def *def, unsigned depth, uint8_t *cache)
{
   if (depth > 32)
      return dependency_unknown;
   if (cache[def->index])
      return cache[def->index] == 1 ? dependency_unknown : cache[def->index] - 2;
   cache[def->index] = 1;
   nir_instr *instr = nir_def_instr(def);
   struct dependency_state state = {.depth = depth, .cache = cache};
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
      if (radv_bc250_mesh_intrinsic_subgroup_op(in) || in->intrinsic == nir_intrinsic_load_subgroup_id ||
          in->intrinsic == nir_intrinsic_load_num_subgroups)
         state.dependencies |= dependency_subgroup;
      const unsigned effects = radv_bc250_mesh_intrinsic_memory_effects(in);
      if (effects & RADV_BC250_SHARED_READ)
         state.dependencies |= dependency_shared | dependency_unknown;
      if (effects & (RADV_BC250_STORAGE_READ | RADV_BC250_EXTERNAL_ATOMIC))
         state.dependencies |= dependency_unknown;
      if (in->intrinsic == nir_intrinsic_load_deref &&
          nir_deref_mode_may_be(nir_src_as_deref(in->src[0]),
                                nir_var_function_temp | nir_var_shader_temp | nir_var_shader_out))
         state.dependencies |= dependency_unknown;
   }
   if (instr->type == nir_instr_type_phi) {
      nir_cf_node *prev = nir_cf_node_prev(&instr->block->cf_node);
      if (prev && prev->type == nir_cf_node_if)
         source_dependencies(&nir_cf_node_as_if(prev)->condition, &state);
      else
         state.dependencies |= dependency_unknown;
   }
   nir_foreach_src(instr, source_dependencies, &state);
   cache[def->index] = state.dependencies + 2;
   return state.dependencies;
}

enum allocation_paths {
   allocation_none = 1,
   allocation_once = 2,
   allocation_repeated = 4,
};

static unsigned
allocate(unsigned paths)
{
   return ((paths & allocation_none) ? allocation_once : 0) |
          ((paths & (allocation_once | allocation_repeated)) ? allocation_repeated : 0);
}

static bool
writes_mesh_output(const nir_intrinsic_instr *in)
{
   switch (in->intrinsic) {
   case nir_intrinsic_store_deref:
   case nir_intrinsic_copy_deref:
      return nir_deref_mode_is(nir_src_as_deref(in->src[0]), nir_var_shader_out);
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output:
      return true;
   default:
      return false;
   }
}

static void
analyze_mesh_contract(nir_shader *mesh, struct radv_bc250_mesh_contract *c)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   nir_foreach_function_impl (other, mesh) {
      /* The current routes require calls to be inlined. Do not silently analyze
       * the entrypoint while an allocation could be hidden in another function.
       */
      if (other != impl)
         return;
   }
   nir_metadata_require(impl, nir_metadata_block_index);
   unsigned *out = calloc(impl->num_blocks, sizeof(*out));
   if (!out)
      return;

   /* Track all reachable path counts, saturating at two. Union at joins keeps
    * mutually exclusive allocations distinct from sequential allocations. The
    * fixed point also detects allocations reached again through a loop edge.
    * No instructions, count values or output stores are moved by this analysis.
    */
   bool progress;
   do {
      progress = false;
      nir_foreach_block (block, impl) {
         unsigned paths = block == nir_start_block(impl) ? allocation_none : 0;
         nir_foreach_pred (pred, block)
            paths |= out[pred->index];
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count)
               paths = allocate(paths);
            else if (writes_mesh_output(in) && (paths & allocation_none))
               c->output_before_allocation = true;
         }
         c->allocation_may_repeat |= paths & allocation_repeated;
         if (out[block->index] != paths) {
            out[block->index] = paths;
            progress = true;
         }
      }
   } while (progress);

   nir_foreach_pred (pred, impl->end_block)
      c->allocation_may_be_skipped |= out[pred->index] & allocation_none;

   /* A subgroup-uniform branch can still choose different sites in different
    * waves. Mesh allocation control flow must be uniform across the workgroup.
    */
   nir_custom_divergence_analysis(mesh, nir_divergence_across_subgroups);
   nir_foreach_block (block, impl) {
      unsigned paths = block == nir_start_block(impl) ? allocation_none : 0;
      nir_foreach_pred (pred, block)
         paths |= out[pred->index];
      if (!paths)
         continue;
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
            continue;
         c->allocation_sites++;
         c->allocation_control_uniform &= !block->divergent;
         c->allocation_values_uniform &= !in->src[0].ssa->divergent && !in->src[1].ssa->divergent;
      }
   }
   c->allocation_analysis_complete = true;
   free(out);
}

void
radv_bc250_analyze_mesh_contract(nir_shader *mesh, struct radv_bc250_mesh_contract *c)
{
   radv_bc250_mesh_contract_init(mesh, c);
   radv_bc250_analyze_mesh_allocation(mesh, c);
}

void
radv_bc250_analyze_mesh_allocation(nir_shader *mesh, struct radv_bc250_mesh_contract *c)
{
   c->allocation_sites = 0;
   c->allocation_analysis_complete = false;
   c->allocation_may_repeat = c->allocation_may_be_skipped = c->output_before_allocation = false;
   c->allocation_control_uniform = c->allocation_values_uniform = true;
   /* Divergence analysis requires lowered returns. Keep the route's original
    * program and metadata intact while normalizing a private analysis copy.
    */
   nir_shader *normalized = nir_shader_clone(NULL, mesh);
   nir_lower_returns(normalized);
   analyze_mesh_contract(normalized, c);
   ralloc_free(normalized);
}

bool
radv_bc250_mesh_allocation_is_canonical(const struct radv_bc250_mesh_contract *c)
{
   return c->allocation_analysis_complete && c->allocation_sites && !c->allocation_may_repeat &&
          c->allocation_control_uniform && c->allocation_values_uniform && !c->output_before_allocation;
}

static bool
output_is_lane(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   return nir_scalar_is_intrinsic(s) && nir_scalar_intrinsic_op(s) == nir_intrinsic_load_local_invocation_index;
}

/* The vertex index is invocation + k * threads (k >= 0): the invocation that stores vertex v is v % threads. */
static bool
output_owned(nir_scalar s, unsigned threads, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (depth > 6)
      return false;
   if (output_is_lane(s))
      return true;
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      for (unsigned i = 0; i < 2; i++) {
         nir_scalar o = nir_scalar_chase_alu_src(s, !i);
         if (nir_scalar_is_const(o) && nir_scalar_as_uint(o) % threads == 0 &&
             output_owned(nir_scalar_chase_alu_src(s, i), threads, depth + 1))
            return true;
      }
   }
   if (nir_def_instr(s.def)->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(s.def));
      nir_cf_node *parent = phi->instr.block->cf_node.parent;
      if (parent->type != nir_cf_node_loop || nir_loop_first_block(nir_cf_node_as_loop(parent)) != phi->instr.block)
         return false;
      unsigned inits = 0;
      nir_foreach_phi_src (src, phi) {
         nir_scalar v = nir_scalar_chase_movs(nir_get_scalar(src->src.ssa, 0));
         if (nir_scalar_is_alu(v) && nir_scalar_alu_op(v) == nir_op_iadd) {
            nir_scalar a = nir_scalar_chase_alu_src(v, 0), c = nir_scalar_chase_alu_src(v, 1);
            if ((a.def == &phi->def && nir_scalar_is_const(c) && nir_scalar_as_uint(c) % threads == 0) ||
                (c.def == &phi->def && nir_scalar_is_const(a) && nir_scalar_as_uint(a) % threads == 0))
               continue;
         }
         if (!output_owned(v, threads, depth + 1))
            return false;
         inits++;
      }
      return inits > 0;
   }
   return false;
}

/* The deref of a Mesh output array element: the vertex (or primitive) level array deref right below the variable. */
nir_deref_instr *
radv_bc250_mesh_output_element(nir_deref_instr *d)
{
   while (d && d->deref_type != nir_deref_type_var) {
      nir_deref_instr *parent = nir_deref_instr_parent(d);
      if (parent && parent->deref_type == nir_deref_type_var)
         return d->deref_type == nir_deref_type_array ? d : NULL;
      d = parent;
   }
   return NULL;
}

void
radv_bc250_mesh_contract_init(const nir_shader *mesh, struct radv_bc250_mesh_contract *c)
{
   *c = (struct radv_bc250_mesh_contract){
      .stage = mesh->info.stage,
      .primitive_type = mesh->info.mesh.primitive_type,
      .workgroup_size = {mesh->info.workgroup_size[0], mesh->info.workgroup_size[1], mesh->info.workgroup_size[2]},
      .workgroup_invocations =
         mesh->info.workgroup_size[0] * mesh->info.workgroup_size[1] * mesh->info.workgroup_size[2],
      .max_vertices = mesh->info.mesh.max_vertices_out,
      .max_primitives = mesh->info.mesh.max_primitives_out,
      .task_payload_size = mesh->info.task_payload_size,
      .shared_size = mesh->info.shared_size,
      .min_subgroup_size = mesh->info.min_subgroup_size,
      .max_subgroup_size = mesh->info.max_subgroup_size,
      .api_subgroup_size = mesh->info.api_subgroup_size,
      .nv = mesh->info.mesh.nv,
      .workgroup_size_variable = mesh->info.workgroup_size_variable,
      .shared_memory_explicit_layout = mesh->info.shared_memory_explicit_layout,
      .zero_initialize_shared_memory = mesh->info.zero_initialize_shared_memory,
      .derivative_group = mesh->info.derivative_group,
      .outputs_written = mesh->info.outputs_written,
      .outputs_read = mesh->info.outputs_read,
      .per_primitive_outputs = mesh->info.per_primitive_outputs,
      .outputs_written_16bit = mesh->info.outputs_written_16bit,
      .allocation_control_uniform = true,
      .allocation_values_uniform = true,
   };
   nir_foreach_shader_out_variable (v, mesh) {
      c->non_array_output |= !glsl_type_is_array(v->type);
      if (v->data.per_primitive) {
         if ((unsigned)v->data.location < 64)
            c->declared_per_primitive_outputs |= BITFIELD64_BIT(v->data.location);
         if (v->data.location == VARYING_SLOT_CULL_PRIMITIVE)
            c->has_per_primitive_cull = true;
         else if (v->data.location != VARYING_SLOT_PRIMITIVE_INDICES)
            c->has_per_primitive_data = true;
      }
      if (v->data.location == VARYING_SLOT_PRIMITIVE_INDICES) {
         c->triangle_index_array =
            glsl_type_is_array(v->type) && glsl_get_vector_elements(glsl_get_array_element(v->type)) == 3;
      }
      if (v->data.location >= VARYING_SLOT_VAR0) {
         const struct glsl_type *type = glsl_without_array(v->type);
         c->non_32bit_generic_output |=
            !(glsl_type_is_scalar(type) || glsl_type_is_vector(type) || glsl_type_is_matrix(type)) ||
            glsl_get_bit_size(type) != 32;
      }
   }
   nir_foreach_function_impl (impl, mesh) {
      c->function_count++;
      uint8_t *dependencies = calloc(MAX2(impl->ssa_alloc, 1), sizeof(*dependencies));
      if (!dependencies)
         c->output_dependencies_unknown = true;
      nir_foreach_block (block, impl) {
         /* Loops and early returns can control later constant output stores
          * outside the construct, without leaving an SSA value dependency.
          */
         for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
            if (node->type == nir_cf_node_loop)
               c->output_dependencies_unknown = true;
         }
         nir_foreach_instr (instr, block) {
            c->has_calls |= instr->type == nir_instr_type_call;
            if (instr->type == nir_instr_type_jump && nir_instr_as_jump(instr)->type == nir_jump_return)
               c->output_dependencies_unknown = true;
            if (instr->type == nir_instr_type_deref)
               c->uses_task_payload |= nir_deref_mode_may_be(nir_instr_as_deref(instr), nir_var_mem_task_payload);
            if (instr->type == nir_instr_type_tex)
               c->memory_effects |= RADV_BC250_EXTERNAL_READ;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            const char *name = nir_intrinsic_infos[in->intrinsic].name;
            const unsigned effects = radv_bc250_mesh_intrinsic_memory_effects(in);
            c->memory_effects |= effects;
            c->used_external_atomic_result |= (effects & RADV_BC250_EXTERNAL_ATOMIC) && !nir_def_is_unused(&in->def);
            c->subgroup_ops |= radv_bc250_mesh_intrinsic_subgroup_op(in);
            c->observes_subgroup |= c->subgroup_ops || in->intrinsic == nir_intrinsic_load_subgroup_id ||
                                    in->intrinsic == nir_intrinsic_load_num_subgroups;
            c->uses_num_workgroups |= in->intrinsic == nir_intrinsic_load_num_workgroups;
            c->uses_task_payload |= strstr(name, "task_payload") != NULL;
            c->allocation_static_sites += in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count;
            if (in->intrinsic == nir_intrinsic_store_output || in->intrinsic == nir_intrinsic_store_per_vertex_output ||
                in->intrinsic == nir_intrinsic_store_per_primitive_output) {
               c->output_dependencies_unknown = true;
               if (!c->output_store_refusal)
                  c->output_store_refusal = "lowered output store ownership unavailable";
            }
            if (in->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_deref_instr *d = nir_src_as_deref(in->src[0]);
            nir_variable *v = nir_deref_instr_get_variable(d);
            if (!v || v->data.mode != nir_var_shader_out)
               continue;
            nir_deref_instr *element = radv_bc250_mesh_output_element(d);
            const char *refusal = NULL;
            if (!element)
               refusal = "output store shape";
            else if (v->data.location == VARYING_SLOT_PRIMITIVE_INDICES) {
               if (element != d)
                  refusal = "index store shape";
            } else if (!c->workgroup_invocations ||
                       !output_owned(nir_get_scalar(element->arr.index.ssa, 0), c->workgroup_invocations, 0)) {
               refusal = "vertex store not owned by invocation % workgroup size";
            }
            if (!c->output_store_refusal)
               c->output_store_refusal = refusal;
            if (!dependencies)
               continue;
            struct dependency_state state = {.cache = dependencies};
            nir_foreach_src(instr, source_dependencies, &state);
            for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
               if (node->type == nir_cf_node_if)
                  source_dependencies(&nir_cf_node_as_if(node)->condition, &state);
               if (node->type == nir_cf_node_loop)
                  state.dependencies |= dependency_unknown;
            }
            const uint64_t bit = (unsigned)v->data.location < 64 ? BITFIELD64_BIT(v->data.location) : 0;
            if (state.dependencies & dependency_subgroup)
               c->subgroup_dependent_outputs |= bit;
            if (state.dependencies & dependency_shared)
               c->shared_dependent_outputs |= bit;
            c->output_dependencies_unknown |=
               (state.dependencies & dependency_unknown) || (unsigned)v->data.location >= 64;
         }
      }
      free(dependencies);
   }
   c->output_dependencies_unknown |= c->has_calls;
}

/* SPDX-License-Identifier: MIT */
#include "radv_bc250_idx.h"
#include "nir/nir_builder.h"
#include "nir/nir_deref.h"
#include "radv_bc250_mesh_contract.h"
#include "radv_bc250_dispatch.h"

static void
bc250_shared_type(const struct glsl_type *type, unsigned *size, unsigned *alignment)
{
   unsigned component = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   *size = component * glsl_get_vector_elements(type);
   *alignment = component;
}

static nir_def *
bc250_pointer(nir_builder *b, unsigned offset)
{
   return nir_pack_64_2x32(
      b, nir_load_push_constant(b, 2, 32, nir_imm_int(b, offset), .base = 0, .range = RADV_BC250_IDX_CONSTANT_SIZE));
}

static nir_def *
bc250_task_index(nir_builder *b)
{
   nir_def *id = nir_load_workgroup_id(b), *size = nir_load_num_workgroups(b);
   return radv_bc250_dispatch_flatten(b, id, size);
}

static nir_def *
bc250_payload_address(nir_builder *b, nir_def *offset, bool producer)
{
   assert(producer);
   nir_def *stride =
      nir_load_push_constant(b, 1, 32, nir_imm_int(b, 16), .base = 0, .range = RADV_BC250_IDX_CONSTANT_SIZE);
   return nir_iadd(b, bc250_pointer(b, 8), nir_u2u64(b, nir_iadd(b, nir_imul(b, bc250_task_index(b), stride), offset)));
}

/* Stored-output IDX keeps the original workgroup execution, including subgroup
 * and shared-memory dependencies. Each output has a compact plane in a group
 * record. Vertex and primitive planes are separate; encoded provoking indices
 * select primitive attributes without duplicating the vertex plane.
 */
static bool
bc250_idx_stored_type(const struct glsl_type *type)
{
   if (glsl_type_is_array(type) || glsl_type_is_matrix(type))
      return glsl_get_length(type) && bc250_idx_stored_type(glsl_get_array_element(type));
   if (glsl_type_is_struct_or_ifc(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         if (!bc250_idx_stored_type(glsl_get_struct_field(type, i)))
            return false;
      }
      return true;
   }
   return (glsl_type_is_scalar(type) || glsl_type_is_vector(type)) && !glsl_type_is_boolean(type) &&
          glsl_get_bit_size(type) == 32;
}

static bool
bc250_idx_can_remove_input(nir_variable *var, void *data)
{
   return !var->data.always_active_io;
}

bool
radv_bc250_idx_prune_outputs(nir_shader *mesh, const nir_shader *fragment)
{
   /* Analyze a private consumer copy: dead fragment input loads must not keep
    * output planes alive. Keep whole locations, including partially consumed
    * vectors and aggregates, and all rasterization built-ins. Connectivity is
    * consumed by the index producer even when the fragment shader reads none.
    */
   nir_shader *consumer = nir_shader_clone(NULL, fragment);
   NIR_PASS(_, consumer, nir_opt_dce);
   NIR_PASS(_, consumer, nir_remove_dead_derefs);
   const nir_remove_dead_variables_options opts = {.can_remove_var = bc250_idx_can_remove_input};
   NIR_PASS(_, consumer, nir_remove_dead_variables, nir_var_shader_in, &opts);
   uint64_t read[4], patches_read[4] = {0};
   for (unsigned c = 0; c < ARRAY_SIZE(read); c++)
      read[c] = BITFIELD64_MASK(VARYING_SLOT_VAR0);
   nir_foreach_shader_in_variable (var, consumer) {
      if (var->data.location < VARYING_SLOT_VAR0)
         continue;
      unsigned slots = glsl_count_attribute_slots(var->type, false);
      assert(var->data.location < 64 && slots <= 64 - var->data.location);
      uint64_t mask = BITFIELD64_MASK(slots) << var->data.location;
      for (unsigned c = 0; c < ARRAY_SIZE(read); c++)
         read[c] |= mask;
   }
   bool progress = nir_remove_unused_io_vars(mesh, nir_var_shader_out, read, patches_read);
   ralloc_free(consumer);
   if (progress) {
      NIR_PASS(_, mesh, nir_opt_dce);
      NIR_PASS(_, mesh, nir_remove_dead_derefs);
      nir_shader_gather_info(mesh, nir_shader_get_entrypoint(mesh));
      nir_validate_shader(mesh, "BC250 stored output pruning");
   }
   return progress;
}

unsigned
radv_bc250_idx_output_layout(nir_shader *mesh)
{
   unsigned offset = 0;
   nir_foreach_shader_out_variable (var, mesh) {
      if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
         continue;
      if (glsl_get_length(var->type) !=
             (var->data.per_primitive ? mesh->info.mesh.max_primitives_out : mesh->info.mesh.max_vertices_out) ||
          !bc250_idx_stored_type(var->type))
         return 0;
      unsigned size, alignment;
      glsl_get_explicit_type_for_size_align(var->type, bc250_shared_type, &size, &alignment);
      /* Deref offsets and masked stores use 32-bit arithmetic. Keep every
       * plane and the complete group record comfortably within that range. */
      if (size > (1u << 20) || offset > (1u << 20) - size)
         return 0;
      var->data.driver_location = offset;
      offset += size;
   }
   return align(offset, 16);
}

static void
bc250_idx_type_size(const struct glsl_type *type, unsigned *size, unsigned *alignment)
{
   glsl_get_explicit_type_for_size_align(type, bc250_shared_type, size, alignment);
}

static void
bc250_idx_fetch_output(nir_builder *b, nir_deref_instr *d, nir_def *base)
{
   const struct glsl_type *type = d->type;
   if (glsl_type_is_array(type) || glsl_type_is_matrix(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++)
         bc250_idx_fetch_output(b, nir_build_deref_array_imm(b, d, i), base);
   } else if (glsl_type_is_struct_or_ifc(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++)
         bc250_idx_fetch_output(b, nir_build_deref_struct(b, d, i), base);
   } else {
      nir_def *offset = nir_build_deref_offset(b, d, bc250_idx_type_size);
      nir_def *value = nir_load_global(b, glsl_get_vector_elements(type), 32, nir_iadd(b, base, nir_u2u64(b, offset)),
                                       .align_mul = 4);
      nir_store_deref(b, d, value, BITFIELD_MASK(value->num_components));
   }
}

nir_shader *
radv_bc250_idx_build_stored_vs(const nir_shader_compiler_options *options, const nir_shader *mesh,
                               unsigned output_bytes)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, options, "bc250_idx_stored_vs");
   const unsigned V = mesh->info.mesh.max_vertices_out, P = mesh->info.mesh.max_primitives_out;
   nir_def *k = nir_load_vertex_id(&b);
   nir_def *group = nir_udiv_imm(&b, k, bc250_idx_vertex_stride(mesh));
   nir_def *slot = nir_umod_imm(&b, k, bc250_idx_vertex_stride(mesh));
   nir_def *vertex = nir_umod_imm(&b, slot, V);
   /* Ordinary corners have slot / V == 0: read primitive zero. Only the
    * provoking corner contributes these flat outputs to the fragment stage. */
   nir_def *primitive = radv_bc250_primitive_decode_attribute(&b, slot, V, true);
   nir_def *direct_base = bc250_pointer(&b, 32);
   nir_push_if(&b, nir_ine_imm(&b, bc250_pointer(&b, 40), 0));
   nir_def *draw = nir_load_draw_id(&b);
   nir_def *first = nir_load_global(
      &b, 1, 32, nir_iadd(&b, bc250_pointer(&b, 40), nir_u2u64(&b, nir_iadd_imm(&b, nir_imul_imm(&b, draw, 20), 8))),
      .align_mul = 4);
   nir_def *dims = nir_load_global(
      &b, 3, 32, nir_iadd(&b, bc250_pointer(&b, 32), nir_u2u64(&b, nir_imul_imm(&b, draw, 12))), .align_mul = 4);
   nir_def *groups =
      radv_bc250_dispatch_count(&b, nir_channel(&b, dims, 0), nir_channel(&b, dims, 1), nir_channel(&b, dims, 2));
   nir_def *indirect_base =
      nir_iadd(&b, bc250_pointer(&b, 8),
               nir_u2u64(&b, nir_iadd(&b, nir_imul_imm(&b, first, 4), nir_imul_imm(&b, groups, P * 12))));
   nir_pop_if(&b, NULL);
   nir_def *base = nir_if_phi(&b, indirect_base, direct_base);
   base = nir_iadd(&b, base, nir_u2u64(&b, nir_imul_imm(&b, group, output_bytes)));
   nir_foreach_shader_out_variable (var, mesh) {
      if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
         continue;
      nir_variable *out = nir_variable_clone(var, b.shader);
      out->type = glsl_get_array_element(var->type);
      out->data.per_primitive = false;
      out->data.driver_location = 0;
      if (var->data.per_primitive)
         out->data.interpolation = INTERP_MODE_FLAT;
      nir_shader_add_variable(b.shader, out);
      unsigned size, alignment;
      glsl_get_explicit_type_for_size_align(out->type, bc250_shared_type, &size, &alignment);
      nir_def *address =
         nir_iadd(&b, base,
                  nir_u2u64(&b, nir_iadd_imm(&b, nir_imul_imm(&b, var->data.per_primitive ? primitive : vertex, size),
                                             var->data.driver_location)));
      bc250_idx_fetch_output(&b, nir_build_deref_var(&b, out), address);
   }
   b.shader->info.clip_distance_array_size = mesh->info.clip_distance_array_size;
   b.shader->info.cull_distance_array_size = mesh->info.cull_distance_array_size;
   nir_shader_gather_info(b.shader, b.impl);
   nir_validate_shader(b.shader, "BC250 stored-output fetch vertex shader");
   return b.shader;
}

/* Indirect index pass (RADV_BC250_MESH_IDXPASS): one dispatch covers every record of an indirect call. The
 * setup pass published, at constants.xyz, {records n, total workgroups} and the exclusive workgroup prefix of
 * each record (n + 1 entries from +32); constants.input holds each record's size (x, y, z) and
 * constants.input_count its indexed-draw command (firstIndex at +8). A workgroup finds its record r by binary
 * search and runs as workgroup "flat - prefix[r]" of record r with DrawID r. */
static void
bc250_idx_indirect_prologue(nir_builder *b, unsigned V, unsigned P, nir_variable *wg_var, nir_variable *draw_var,
                            nir_def **vbase, nir_def **ibase, unsigned output_bytes, nir_def **obase)
{
   nir_def *tab = bc250_pointer(b, 0);
   nir_def *hdr = nir_load_global(b, 2, 32, tab, .align_mul = 4);
   nir_def *n = nir_channel(b, hdr, 0), *total = nir_channel(b, hdr, 1);
   nir_def *wid = nir_load_workgroup_id(b);
   nir_def *flat = nir_iadd(b, nir_channel(b, wid, 0), nir_imul_imm(b, nir_channel(b, wid, 1), 65535));
   nir_push_if(b, nir_uge(b, flat, total));
   nir_jump(b, nir_jump_return);
   nir_pop_if(b, NULL);
   nir_variable *lo = nir_local_variable_create(b->impl, glsl_uint_type(), "bc250_idx_lo");
   nir_variable *hi = nir_local_variable_create(b->impl, glsl_uint_type(), "bc250_idx_hi");
   nir_store_var(b, lo, nir_imm_int(b, 0), 1);
   nir_store_var(b, hi, n, 1);
   nir_def *prefix = nir_iadd_imm(b, tab, 32);
   nir_push_loop(b);
   {
      nir_def *l = nir_load_var(b, lo), *h = nir_load_var(b, hi);
      nir_break_if(b, nir_uge(b, l, h));
      nir_def *mid = nir_ushr_imm(b, nir_iadd(b, l, h), 1);
      nir_def *pv = nir_load_global(
         b, 1, 32, nir_iadd(b, prefix, nir_u2u64(b, nir_imul_imm(b, nir_iadd_imm(b, mid, 1), 4))), .align_mul = 4);
      nir_push_if(b, nir_uge(b, flat, pv));
      nir_store_var(b, lo, nir_iadd_imm(b, mid, 1), 1);
      nir_push_else(b, NULL);
      nir_store_var(b, hi, mid, 1);
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);
   nir_def *r = nir_load_var(b, lo);
   nir_def *local = nir_isub(
      b, flat, nir_load_global(b, 1, 32, nir_iadd(b, prefix, nir_u2u64(b, nir_imul_imm(b, r, 4))), .align_mul = 4));
   nir_def *dims = nir_load_global(b, 3, 32, nir_iadd(b, bc250_pointer(b, 32), nir_u2u64(b, nir_imul_imm(b, r, 12))),
                                   .align_mul = 4);
   nir_def *dx = nir_channel(b, dims, 0), *dxy = nir_imul(b, dx, nir_channel(b, dims, 1));
   nir_store_var(b, wg_var, radv_bc250_dispatch_unflatten(b, local, dx, dxy), 7);
   nir_store_var(b, draw_var, r, 1);
   nir_def *first = nir_load_global(
      b, 1, 32, nir_iadd(b, bc250_pointer(b, 40), nir_u2u64(b, nir_iadd_imm(b, nir_imul_imm(b, r, 20), 8))),
      .align_mul = 4);
   if (output_bytes) {
      nir_def *groups = nir_imul(b, dxy, nir_channel(b, dims, 2));
      *obase = nir_iadd(
         b, bc250_pointer(b, 8),
         nir_u2u64(b, nir_iadd(b, nir_imul_imm(b, first, 4),
                               nir_iadd(b, nir_imul_imm(b, groups, P * 12), nir_imul_imm(b, local, output_bytes)))));
   }
   *vbase = nir_imul_imm(b, local, V);
   *ibase = nir_iadd(b, bc250_pointer(b, 8),
                     nir_u2u64(b, nir_iadd(b, nir_imul_imm(b, first, 4), nir_imul_imm(b, local, P * 12))));
}

nir_shader *
radv_bc250_idx_build_cs(const nir_shader *mesh, bool indirect, unsigned provoking, unsigned output_bytes,
                        nir_intrinsic_pass_cb lower_constants)
{
   nir_shader *cs = nir_shader_clone(NULL, mesh);
   NIR_PASS(_, cs, nir_split_var_copies);
   NIR_PASS(_, cs, nir_lower_var_copies);
   const unsigned V = cs->info.mesh.max_vertices_out, P = cs->info.mesh.max_primitives_out;
   const unsigned threads = cs->info.workgroup_size[0];
   nir_function_impl *impl = nir_shader_get_entrypoint(cs);
   nir_builder b = nir_builder_create(impl);
   nir_variable *wg_var = NULL, *draw_var = NULL;
   if (indirect) {
      /* The application's WorkGroupID and DrawID are the record's, found in the prologue. */
      wg_var = nir_local_variable_create(impl, glsl_uvec_type(3), "bc250_idx_wg");
      draw_var = nir_local_variable_create(impl, glsl_uint_type(), "bc250_idx_draw");
      nir_foreach_block_safe (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic != nir_intrinsic_load_workgroup_id && in->intrinsic != nir_intrinsic_load_draw_id)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def *v = nir_load_var(&b, in->intrinsic == nir_intrinsic_load_draw_id ? draw_var : wg_var);
            nir_def_rewrite_uses(&in->def, nir_trim_vector(&b, v, in->def.num_components));
            nir_instr_remove(instr);
         }
      }
   }
   /* Direct: the whole grid is one dispatch, WorkGroupID and NumWorkGroups stay the hardware values. */
   if (lower_constants)
      NIR_PASS(_, cs, nir_shader_intrinsics_pass, lower_constants, nir_metadata_none, NULL);
   /* Application returns end the body only: the epilogue below runs in every invocation. The prologue's exit
    * (whole workgroup) is lowered after the epilogue exists. */
   NIR_PASS(_, cs, nir_lower_returns);
   nir_variable *count = nir_variable_create(cs, nir_var_mem_shared, glsl_uint_type(), "bc250_idx_count");
   b = nir_builder_create(impl);
   b.cursor = nir_before_impl(impl);
   nir_def *vbase, *ibase, *obase = NULL;
   if (indirect) {
      bc250_idx_indirect_prologue(&b, bc250_idx_vertex_stride(mesh), P, wg_var, draw_var, &vbase, &ibase, output_bytes,
                                  &obase);
   } else {
      nir_def *flat = nir_iadd(
         &b, nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 64), .base = 0, .range = RADV_BC250_IDX_CONSTANT_SIZE),
         bc250_task_index(&b));
      vbase = nir_imul_imm(&b, flat, bc250_idx_vertex_stride(mesh));
      ibase = bc250_payload_address(&b, nir_imm_int(&b, 0), true);
      if (output_bytes)
         obase = nir_iadd(&b, bc250_pointer(&b, 32), nir_u2u64(&b, nir_imul_imm(&b, flat, output_bytes)));
   }
   nir_foreach_block_safe (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
            b.cursor = nir_before_instr(instr);
            nir_def *pc = nir_bcsel(&b, nir_ieq_imm(&b, in->src[0].ssa, 0), nir_imm_int(&b, 0),
                                    nir_umin_imm(&b, in->src[1].ssa, P));
            if (output_bytes)
               nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
            nir_store_var(&b, count, pc, 1);
            if (output_bytes)
               nir_pop_if(&b, NULL);
            nir_instr_remove(instr);
            continue;
         }
         if (in->intrinsic != nir_intrinsic_store_deref)
            continue;
         nir_deref_instr *d = nir_src_as_deref(in->src[0]);
         nir_variable *var = nir_deref_instr_get_variable(d);
         if (!var || var->data.mode != nir_var_shader_out)
            continue;
         b.cursor = nir_before_instr(instr);
         if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES) {
            nir_def *t = d->arr.index.ssa;
            if (output_bytes)
               nir_push_if(&b, nir_ult_imm(&b, t, P));
            nir_def *indices = nir_u2u32(&b, in->src[1].ssa);
            if (output_bytes)
               indices = nir_umin_imm(&b, indices, V - 1);
            if (bc250_idx_has_pp(mesh)) {
               struct radv_bc250_primitive primitive = radv_bc250_primitive_make(indices, NULL, t);
               indices = radv_bc250_primitive_encode(&b, primitive, V, provoking);
            }
            nir_def *value = nir_iadd(&b, indices, vbase);
            nir_store_global(&b, value, nir_iadd(&b, ibase, nir_u2u64(&b, nir_imul_imm(&b, t, 12))),
                             .write_mask = nir_intrinsic_write_mask(in), .align_mul = 4);
            if (output_bytes)
               nir_pop_if(&b, NULL);
         } else if (output_bytes) {
            unsigned size, alignment;
            glsl_get_explicit_type_for_size_align(var->type, bc250_shared_type, &size, &alignment);
            nir_def *offset = nir_build_deref_offset(&b, d, bc250_idx_type_size);
            nir_deref_instr *element = radv_bc250_mesh_output_element(d);
            nir_def *valid = nir_iand(&b, nir_ult_imm(&b, element->arr.index.ssa, var->data.per_primitive ? P : V),
                                      nir_ule_imm(&b, offset, size - in->src[1].ssa->num_components * 4));
            nir_push_if(&b, valid);
            nir_store_global(&b, in->src[1].ssa,
                             nir_iadd(&b, obase, nir_u2u64(&b, nir_iadd_imm(&b, offset, var->data.driver_location))),
                             .write_mask = nir_intrinsic_write_mask(in), .align_mul = 4);
            nir_pop_if(&b, NULL);
         }
         nir_instr_remove(instr);
      }
   }
   b.cursor = nir_before_impl(impl);
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_store_var(&b, count, nir_imm_int(&b, 0), 1);
   nir_pop_if(&b, NULL);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   /* Unused primitive slots: degenerate triangles on the workgroup's first vertex (no fragments). */
   b.cursor = nir_after_impl(impl);
   /* Also orders the application's own index stores (slots at or past its primitive count are allowed) before the
    * degenerate fill of other waves. */
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared | nir_var_mem_global);
   nir_def *pc = nir_load_var(&b, count);
   nir_def *lane = nir_load_local_invocation_index(&b);
   for (unsigned first = 0; first < P; first += threads) {
      nir_def *t = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_iand(&b, nir_uge(&b, t, pc), nir_ult_imm(&b, t, P)));
      nir_store_global(&b, nir_vec3(&b, vbase, vbase, vbase),
                       nir_iadd(&b, ibase, nir_u2u64(&b, nir_imul_imm(&b, t, 12))), .write_mask = 7, .align_mul = 4);
      nir_pop_if(&b, NULL);
   }
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, cs, nir_lower_returns);
   NIR_PASS(_, cs, nir_lower_vars_to_ssa);
   NIR_PASS(_, cs, nir_opt_dce);
   NIR_PASS(_, cs, nir_remove_dead_derefs);
   nir_foreach_shader_out_variable_safe (v, cs)
      exec_node_remove(&v->node);
   cs->info.stage = MESA_SHADER_COMPUTE;
   cs->info.outputs_written = 0;
   cs->info.per_primitive_outputs = 0;
   cs->info.name = ralloc_asprintf(cs, "bc250_idx_cs%s:%u:%u", indirect ? "_ind" : "", V, P);
   NIR_PASS(_, cs, nir_lower_vars_to_explicit_types, nir_var_mem_shared, bc250_shared_type);
   NIR_PASS(_, cs, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   NIR_PASS(_, cs, nir_opt_dce);
   nir_shader_gather_info(cs, impl);
   nir_validate_shader(cs, "BC250 index pass");
   return cs;
}

/* RADV_BC250_MESH_IDXPASS setup pass for one indirect call (one invocation, records in order). */
static_assert(sizeof(struct bc250_idx_setup_args) == 88, "IDX setup entry ABI");
static_assert(offsetof(struct bc250_idx_setup_args, output_bytes) == 80, "IDX setup output ABI");

static nir_def *
bc250_idx_arg(nir_builder *b, nir_def *entry, unsigned offset, unsigned bits)
{
   if (entry)
      return nir_load_global(b, 1, bits, nir_iadd_imm(b, entry, offset), .align_mul = 4);
   nir_def *v = nir_load_push_constant(b, bits == 64 ? 2 : 1, 32, nir_imm_int(b, offset), .base = 0,
                                       .range = sizeof(struct bc250_idx_setup_args));
   return bits == 64 ? nir_pack_64_2x32(b, v) : v;
}

/* 64 records per step: wave prefix sums of the index counts (each clamped to capacity + 1, capacity <= 2^25, so
 * no 32-bit sum overflows) and of the workgroup counts, plus running carries. The offsets only grow, so a record
 * fits exactly when its inclusive end is within the capacity; every record from the first that does not fit goes
 * to the Mesh fallback. */
static nir_def *
bc250_idx_setup_body(nir_builder *b, bool stored, nir_def *entry, nir_def *prior_base)
{
   nir_def *records = bc250_idx_arg(b, entry, 0, 64), *count = bc250_idx_arg(b, entry, 8, 64);
   nir_def *table = bc250_idx_arg(b, entry, 16, 64), *dims = bc250_idx_arg(b, entry, 24, 64);
   nir_def *draws = bc250_idx_arg(b, entry, 32, 64), *mesh = bc250_idx_arg(b, entry, 40, 64), *pool = bc250_idx_arg(b, entry, 48, 64);
   nir_def *stride = bc250_idx_arg(b, entry, 56, 32), *max_count = bc250_idx_arg(b, entry, 60, 32);
   nir_def *capacity = bc250_idx_arg(b, entry, 64, 32), *prims = bc250_idx_arg(b, entry, 68, 32);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *zero = nir_imm_int(b, 0);
   nir_variable *n_var = nir_local_variable_create(b->impl, glsl_uint_type(), "n");
   nir_store_var(b, n_var, max_count, 1);
   nir_push_if(b, nir_ine_imm(b, count, 0));
   nir_store_var(
      b, n_var,
      nir_umin(b, nir_read_first_invocation(b, nir_load_global(b, 1, 32, count, .align_mul = 4)), max_count), 1);
   nir_pop_if(b, NULL);
   nir_def *n = nir_load_var(b, n_var);
   nir_variable *base_var = nir_local_variable_create(b->impl, glsl_uint_type(), "base");
   nir_store_var(b, base_var, zero, 1);
   nir_push_if(b, nir_ieq_imm(b, bc250_idx_arg(b, entry, 72, 32), 0));
   nir_store_var(b, base_var, prior_base ? prior_base :
                 nir_read_first_invocation(b, nir_load_global(b, 1, 32, pool, .align_mul = 4)), 1);
   nir_pop_if(b, NULL);
   nir_def *base = nir_load_var(b, base_var);
   nir_def *limit = nir_iadd_imm(b, capacity, 1);
   /* carry: index offset (saturated at capacity + 1); gcarry: fitting workgroups; fb: records sent to the fallback. */
   nir_variable *first_var = nir_local_variable_create(b->impl, glsl_uint_type(), "first");
   nir_variable *carry_var = nir_local_variable_create(b->impl, glsl_uint_type(), "carry");
   nir_variable *gcarry_var = nir_local_variable_create(b->impl, glsl_uint_type(), "gcarry");
   nir_variable *fb_var = nir_local_variable_create(b->impl, glsl_uint_type(), "fallback");
   nir_store_var(b, first_var, zero, 1);
   nir_store_var(b, carry_var, nir_umin(b, base, limit), 1);
   nir_store_var(b, gcarry_var, zero, 1);
   nir_store_var(b, fb_var, zero, 1);
   nir_def *prefix = nir_iadd_imm(b, table, 32);
   nir_push_loop(b);
   {
      nir_def *first = nir_load_var(b, first_var);
      nir_break_if(b, nir_uge(b, first, max_count));
      nir_def *r = nir_iadd(b, first, lane);
      nir_def *active = nir_ult(b, r, n);
      /* Records past the GPU count: zero fallback records (the fallback may read all max_count). */
      nir_push_if(b, nir_iand(b, nir_inot(b, active), nir_ult(b, r, max_count)));
      nir_store_global(b, nir_imm_ivec3(b, 0, 0, 0), nir_iadd(b, mesh, nir_u2u64(b, nir_imul_imm(b, r, 12))),
                       .align_mul = 4);
      nir_pop_if(b, NULL);
      nir_def *rec_zero = nir_imm_ivec3(b, 0, 0, 0);
      nir_push_if(b, active);
      nir_def *rec =
         nir_load_global(b, 3, 32, nir_iadd(b, records, nir_u2u64(b, nir_imul(b, r, stride))), .align_mul = 4);
      nir_pop_if(b, NULL);
      rec = nir_if_phi(b, rec, rec_zero);
      nir_def *x = nir_channel(b, rec, 0), *y = nir_channel(b, rec, 1), *z = nir_channel(b, rec, 2);
      nir_def *total = nir_imul(b, nir_imul(b, nir_u2u64(b, x), nir_u2u64(b, y)), nir_u2u64(b, z));
      nir_def *valid =
         nir_iand(b,
                  nir_iand(b, active,
                           nir_ult_imm(b, nir_imul(b, total, nir_u2u64(b, bc250_idx_arg(b, entry, 76, 32))), 1ull << 32)),
                  nir_iand(b, nir_iand(b, nir_ult_imm(b, x, 65536), nir_ult_imm(b, y, 65536)),
                           nir_iand(b, nir_ult_imm(b, z, 65536), nir_ult_imm(b, total, 4194305))));
      nir_def *g = nir_bcsel(b, valid, nir_u2u32(b, total), zero);
      nir_def *group_words = nir_imul_imm(b, prims, 3);
      if (stored)
         group_words = nir_iadd(b, group_words, nir_ushr_imm(b, bc250_idx_arg(b, entry, 80, 32), 2));
      nir_def *need64 = nir_imul(b, nir_u2u64(b, g), nir_u2u64(b, group_words));
      /* An unencodable record begins the fallback suffix, just like pool overflow. */
      nir_def *need = nir_bcsel(b, nir_iand(b, active, nir_inot(b, valid)), limit,
                                nir_u2u32(b, nir_umin(b, need64, nir_u2u64(b, limit))));
      nir_def *carry = nir_load_var(b, carry_var);
      nir_def *end = nir_iadd(b, carry, nir_inclusive_scan(b, need, .reduction_op = nir_op_iadd));
      nir_def *fits = nir_iand(b, valid, nir_uge(b, capacity, end));
      nir_def *gfit = nir_bcsel(b, fits, g, zero);
      nir_def *gstart =
         nir_iadd(b, nir_load_var(b, gcarry_var), nir_exclusive_scan(b, gfit, .reduction_op = nir_op_iadd));
      nir_def *start = nir_isub(b, end, need);
      nir_push_if(b, active);
      {
         nir_store_global(b, gstart, nir_iadd(b, prefix, nir_u2u64(b, nir_imul_imm(b, r, 4))), .align_mul = 4);
         nir_def *dz = nir_imm_ivec3(b, 0, 0, 0);
         nir_store_global(b, nir_bcsel(b, fits, rec, dz), nir_iadd(b, dims, nir_u2u64(b, nir_imul_imm(b, r, 12))),
                          .align_mul = 4);
         nir_def *daddr = nir_iadd(b, draws, nir_u2u64(b, nir_imul_imm(b, r, 20)));
         nir_def *draw =
            nir_vec4(b, nir_bcsel(b, fits, stored ? nir_imul(b, g, nir_imul_imm(b, prims, 3)) : need, zero),
                     nir_b2i32(b, nir_iand(b, fits, nir_ine_imm(b, g, 0))), nir_bcsel(b, fits, start, zero), zero);
         nir_store_global(b, draw, daddr, .align_mul = 4);
         nir_store_global(b, nir_bcsel(b, fits, nir_ior(b, x, nir_ishl_imm(b, y, 16)), zero),
                          nir_iadd_imm(b, daddr, 16), .align_mul = 4);
         nir_store_global(b, nir_bcsel(b, nir_iand(b, active, nir_inot(b, fits)), rec, dz),
                          nir_iadd(b, mesh, nir_u2u64(b, nir_imul_imm(b, r, 12))), .align_mul = 4);
      }
      nir_pop_if(b, NULL);
      nir_store_var(b, carry_var,
                    nir_umin(b, nir_iadd(b, carry, nir_reduce(b, need, .reduction_op = nir_op_iadd)), limit), 1);
      nir_store_var(b, gcarry_var,
                    nir_iadd(b, nir_load_var(b, gcarry_var), nir_reduce(b, gfit, .reduction_op = nir_op_iadd)), 1);
      nir_def *nfb =
         nir_reduce(b, nir_b2i32(b, nir_iand(b, active, nir_inot(b, fits))), .reduction_op = nir_op_iadd);
      nir_store_var(b, fb_var, nir_iadd(b, nir_load_var(b, fb_var), nfb), 1);
      nir_store_var(b, first_var, nir_iadd_imm(b, first, 64), 1);
   }
   nir_pop_loop(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, lane, 0));
   {
      nir_def *gt = nir_load_var(b, gcarry_var);
      nir_store_global(b, gt, nir_iadd(b, prefix, nir_u2u64(b, nir_imul_imm(b, n, 4))), .align_mul = 4);
      nir_def *dx = nir_umin_imm(b, gt, 65535);
      nir_def *dy = nir_bcsel(b, nir_ieq_imm(b, gt, 0), nir_imm_int(b, 1),
                              nir_udiv_imm(b, nir_iadd_imm(b, gt, 65534), 65535));
      nir_def *fallback = nir_bcsel(b, nir_ine_imm(b, nir_load_var(b, fb_var), 0), n, zero);
      nir_store_global(b, nir_vec4(b, n, gt, dx, dy), table, .align_mul = 4);
      nir_store_global(b, nir_vec2(b, nir_imm_int(b, 1), fallback), nir_iadd_imm(b, table, 16), .align_mul = 4);
      /* Pool use: the fitting records' end (the carry saturates past the capacity only on overflow). */
      if (!entry)
         nir_store_global(b, nir_umin(b, nir_load_var(b, carry_var), capacity), pool, .align_mul = 4);
   }
   nir_pop_if(b, NULL);
   return nir_umin(b, nir_load_var(b, carry_var), capacity);
}

static nir_builder
bc250_idx_setup_builder(const nir_shader_compiler_options *options, const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options, "%s", name);
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.min_subgroup_size = b.shader->info.max_subgroup_size = 64;
   b.shader->info.api_subgroup_size = 64;
   return b;
}

nir_shader *
radv_bc250_idx_build_setup(const nir_shader_compiler_options *options, bool stored)
{
   nir_builder b = bc250_idx_setup_builder(options, "bc250_idx_setup");
   bc250_idx_setup_body(&b, stored, NULL, NULL);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   nir_validate_shader(b.shader, "BC250 index setup");
   return b.shader;
}

nir_shader *
radv_bc250_idx_build_setup_batch(const nir_shader_compiler_options *options)
{
   nir_builder b = bc250_idx_setup_builder(options, "bc250_idx_setup_batch");
   nir_def *list = bc250_idx_arg(&b, NULL, 0, 64);
   nir_def *n = nir_umin_imm(&b, nir_read_first_invocation(&b, nir_load_global(&b, 1, 32, list, .align_mul = 4)),
                            RADV_BC250_IDX_SETUP_BATCH_CALLS);
   nir_variable *index = nir_local_variable_create(b.impl, glsl_uint_type(), "call");
   nir_variable *base = nir_local_variable_create(b.impl, glsl_uint_type(), "pool_base");
   nir_store_var(&b, index, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, base, nir_imm_int(&b, 0), 1);
   nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, index);
      nir_break_if(&b, nir_uge(&b, i, n));
      nir_def *entry = nir_iadd(&b, nir_iadd_imm(&b, list, RADV_BC250_IDX_SETUP_BATCH_HEADER_BYTES),
                               nir_u2u64(&b, nir_imul_imm(&b, i, sizeof(struct bc250_idx_setup_args))));
      /* Pool carry lives in this wave's registers between calls. Tables are disjoint,
       * so no global-memory round trip or inter-dispatch wait is needed between scans.
       * Only the first entry can use the pool value from an earlier batch. */
      nir_push_if(&b, nir_iand(&b, nir_ieq_imm(&b, i, 0),
                             nir_ieq_imm(&b, bc250_idx_arg(&b, entry, 72, 32), 0)));
      nir_def *pool = bc250_idx_arg(&b, entry, 48, 64);
      nir_store_var(&b, base, nir_read_first_invocation(&b, nir_load_global(&b, 1, 32, pool, .align_mul = 4)), 1);
      nir_pop_if(&b, NULL);
      /* output_bytes=0 also handles replay IDX entries in a mixed batch. */
      nir_def *used = bc250_idx_setup_body(&b, true, entry, nir_load_var(&b, base));
      nir_store_var(&b, base, used, 1);
      nir_push_if(&b, nir_iand(&b, nir_ieq_imm(&b, nir_load_subgroup_invocation(&b), 0),
                             nir_ieq(&b, nir_iadd_imm(&b, i, 1), n)));
      nir_store_global(&b, used, bc250_idx_arg(&b, entry, 48, 64), .align_mul = 4);
      nir_pop_if(&b, NULL);
      nir_store_var(&b, index, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, NULL);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   nir_validate_shader(b.shader, "BC250 batched index setup");
   return b.shader;
}

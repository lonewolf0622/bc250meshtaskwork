/* SPDX-License-Identifier: MIT */
#include "radv_bc250_idx_cost.h"
#include "nir/nir_builder.h"
#include "nir/nir_deref.h"
#include "util/hash_table.h"
#include "util/set.h"
#include <string.h>

/* A conservative, bounded CPU expression program. It reads application push
 * constants only; descriptor contents and GPU indirect records are never read
 * by the CPU. Unsupported expressions leave the established route unchanged. */
static int
cost_expression(nir_scalar value, struct radv_bc250_idx_cost *cost, unsigned depth)
{
   value = nir_scalar_resolved(value.def, value.comp);
   if (depth > 16 || value.def->bit_size != 32 || cost->node_count >= RADV_BC250_COST_NODES)
      return -1;
   struct radv_bc250_cost_node node = {0};
   nir_instr *instr = nir_def_instr(value.def);
   if (instr->type == nir_instr_type_load_const) {
      node.op = RADV_BC250_COST_CONST;
      node.value = nir_scalar_as_uint(value);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
      if (in->intrinsic != nir_intrinsic_load_push_constant || !nir_src_is_const(in->src[0]))
         return -1;
      node.op = RADV_BC250_COST_PUSH;
      uint64_t offset = (uint64_t)nir_intrinsic_base(in) + nir_src_as_uint(in->src[0]) + 4 * value.comp;
      if (offset + 4 > 256)
         return -1;
      node.value = offset;
   } else if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_iadd: node.op = RADV_BC250_COST_ADD; break;
      case nir_op_imul: node.op = RADV_BC250_COST_MUL; break;
      case nir_op_iand: node.op = RADV_BC250_COST_AND; break;
      case nir_op_ior: node.op = RADV_BC250_COST_OR; break;
      case nir_op_ixor: node.op = RADV_BC250_COST_XOR; break;
      case nir_op_ishl: node.op = RADV_BC250_COST_SHL; break;
      case nir_op_ushr: node.op = RADV_BC250_COST_USHR; break;
      case nir_op_umin: node.op = RADV_BC250_COST_UMIN; break;
      case nir_op_umax: node.op = RADV_BC250_COST_UMAX; break;
      case nir_op_ineg: node.op = RADV_BC250_COST_NEG; break;
      default: return -1;
      }
      int a = cost_expression(nir_scalar_chase_alu_src(value, 0), cost, depth + 1);
      int b = alu->op == nir_op_ineg ? a : cost_expression(nir_scalar_chase_alu_src(value, 1), cost, depth + 1);
      if (a < 0 || b < 0)
         return -1;
      node.a = a;
      node.b = b;
   } else {
      return -1;
   }
   if (cost->node_count >= RADV_BC250_COST_NODES)
      return -1;
   cost->nodes[cost->node_count] = node;
   return cost->node_count++;
}

struct mark_state {
   struct set *live;
   unsigned depth;
};

static void mark_value(nir_def *value, struct set *live, unsigned depth);

static bool
mark_source(nir_src *src, void *data)
{
   struct mark_state *state = data;
   mark_value(src->ssa, state->live, state->depth + 1);
   return true;
}

static void
mark_value(nir_def *value, struct set *live, unsigned depth)
{
   if (!value || depth > 128 || _mesa_set_search(live, value))
      return;
   _mesa_set_add(live, value);
   struct mark_state state = {.live = live, .depth = depth};
   nir_foreach_src(nir_def_instr(value), mark_source, &state);
}

/* Only canonical zero-based, increment-by-one integer loops with one exit.
 * Do not infer a minimum cost from arbitrary conditional execution. */
static void
cost_loop(nir_loop *loop, struct set *live, struct radv_bc250_idx_cost *cost)
{
   nir_loop_info *info = loop->info;
   if (!info || info->complex_loop || cost->term_count == RADV_BC250_COST_TERMS ||
       list_is_empty(&info->loop_terminator_list))
      return;
   for (nir_cf_node *p = loop->cf_node.parent; p; p = p->parent) {
      if (p->type == nir_cf_node_if)
         return;
   }
   nir_loop_terminator *term = list_first_entry(&info->loop_terminator_list, nir_loop_terminator, loop_terminator_link);
   if (term->loop_terminator_link.next != &info->loop_terminator_list)
      return;
   nir_scalar cond = nir_scalar_resolved(term->nif->condition.ssa, 0);
   bool invert = term->continue_from_then;
   if (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
      cond = nir_scalar_chase_alu_src(cond, 0);
      invert = !invert;
   }
   if (!nir_scalar_is_alu(cond))
      return;
   nir_op op = nir_scalar_alu_op(cond);
   if ((!invert && op != nir_op_uge) || (invert && op != nir_op_ult))
      return;
   nir_scalar induction = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar bound = nir_scalar_chase_alu_src(cond, 1);
   struct hash_entry *entry = _mesa_hash_table_search(info->induction_vars, induction.def);
   if (!entry)
      return;
   nir_loop_induction_variable *var = entry->data;
   if (var->basis != induction.def || induction.comp || !nir_src_is_const(*var->init_src) ||
       nir_src_as_uint(*var->init_src) != 0 || !nir_src_is_const(var->update_src->src) ||
       nir_src_as_uint(var->update_src->src) != 1 ||
       nir_def_as_alu(var->def)->op != nir_op_iadd)
      return;
   unsigned alu_count = 0;
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      if (node->type != nir_cf_node_block)
         continue;
      nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
         if (instr->type == nir_instr_type_alu && _mesa_set_search(live, &nir_instr_as_alu(instr)->def))
            alu_count += nir_instr_as_alu(instr)->def.num_components;
      }
   }
   if (alu_count < 4)
      return;
   unsigned saved_nodes = cost->node_count;
   int expression = cost_expression(bound, cost, 0);
   if (expression < 0) {
      cost->node_count = saved_nodes;
      return;
   }
   cost->terms[cost->term_count].bound = expression;
   cost->terms[cost->term_count++].alu = MIN2(alu_count, UINT16_MAX);
}

bool
radv_bc250_idx_cost_build(const nir_shader *shader, struct radv_bc250_idx_cost *cost)
{
   memset(cost, 0, sizeof(*cost));
   /* The measured useful class replays a large vertex body for multiple Mesh
    * pieces. Small meshlets remain Mesh even when their shader has a long loop.
    * These are workload properties, never game identities or shader hashes. */
   if (shader->info.stage != MESA_SHADER_MESH || shader->info.mesh.max_vertices_out <= 32 ||
       shader->info.mesh.max_vertices_out > 256 || shader->info.mesh.max_primitives_out == 0 ||
       shader->info.mesh.max_primitives_out > 256)
      return false;
   nir_shader *copy = nir_shader_clone(NULL, shader);
   NIR_PASS(_, copy, nir_lower_vars_to_ssa);
   NIR_PASS(_, copy, nir_opt_copy_prop);
   NIR_PASS(_, copy, nir_opt_constant_folding);
   NIR_PASS(_, copy, nir_opt_dce);
   struct set *live = _mesa_pointer_set_create(NULL);
   unsigned counts = 0;
   bool count_program = false;
   nir_foreach_function_impl(impl, copy) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
               counts++;
               int v = cost_expression(nir_get_scalar(in->src[0].ssa, 0), cost, 0);
               int p = cost_expression(nir_get_scalar(in->src[1].ssa, 0), cost, 0);
               count_program = v >= 0 && p >= 0;
               if (count_program) {
                  cost->vertex_count = v;
                  cost->primitive_count = p;
               }
            }
            if (in->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_variable *v = nir_deref_instr_get_variable(nir_src_as_deref(in->src[0]));
            if (v && v->data.mode == nir_var_shader_out && !v->data.per_primitive &&
                v->data.location != VARYING_SLOT_PRIMITIVE_INDICES)
               mark_value(in->src[1].ssa, live, 0);
         }
      }
      nir_loop_analyze_impl(impl, nir_var_all, false);
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent->type == nir_cf_node_loop &&
             block == nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)))
            cost_loop(nir_cf_node_as_loop(block->cf_node.parent), live, cost);
      }
   }
   cost->vertices = shader->info.mesh.max_vertices_out;
   cost->primitives = shader->info.mesh.max_primitives_out;
   _mesa_set_destroy(live, NULL);
   ralloc_free(copy);
   return cost->term_count != 0 && counts == 1 && count_program;
}

static bool
cost_evaluate(const struct radv_bc250_idx_cost *cost, const void *push, size_t size, uint64_t *work,
              bool *long_loop, unsigned minimum_loop)
{
   uint32_t values[RADV_BC250_COST_NODES];
   *work = 0;
   *long_loop = false;
   if (!cost->term_count || cost->node_count > RADV_BC250_COST_NODES || cost->term_count > RADV_BC250_COST_TERMS)
      return false;
   for (unsigned i = 0; i < cost->node_count; i++) {
      const struct radv_bc250_cost_node *n = &cost->nodes[i];
      uint32_t a = 0, b = 0;
      if (n->op >= RADV_BC250_COST_ADD) {
         if (n->a >= i || n->b >= i)
            return false;
         a = values[n->a]; b = values[n->b];
      }
      switch (n->op) {
      case RADV_BC250_COST_CONST: values[i] = n->value; break;
      case RADV_BC250_COST_PUSH:
         if (n->value > size || size - n->value < 4 || !push)
            return false;
         memcpy(&values[i], (const uint8_t *)push + n->value, 4); break;
      case RADV_BC250_COST_ADD: values[i] = a + b; break;
      case RADV_BC250_COST_MUL: values[i] = a * b; break;
      case RADV_BC250_COST_AND: values[i] = a & b; break;
      case RADV_BC250_COST_OR: values[i] = a | b; break;
      case RADV_BC250_COST_XOR: values[i] = a ^ b; break;
      case RADV_BC250_COST_SHL: if (b >= 32) return false; values[i] = a << b; break;
      case RADV_BC250_COST_USHR: if (b >= 32) return false; values[i] = a >> b; break;
      case RADV_BC250_COST_UMIN: values[i] = MIN2(a, b); break;
      case RADV_BC250_COST_UMAX: values[i] = MAX2(a, b); break;
      case RADV_BC250_COST_NEG: values[i] = 0u - a; break;
      default: return false;
      }
   }
   if (cost->vertex_count >= cost->node_count || cost->primitive_count >= cost->node_count)
      return false;
   uint32_t vertices = values[cost->vertex_count], primitives = values[cost->primitive_count];
   if (vertices > cost->vertices || primitives > cost->primitives)
      return false;
   if (vertices < 192 || primitives < 96)
      return true;
   for (unsigned i = 0; i < cost->term_count; i++) {
      if (cost->terms[i].bound >= cost->node_count)
         return false;
      /* A very large or effectively unbounded loop is outside this model. */
      uint32_t trips = values[cost->terms[i].bound];
      if (trips > 65536)
         return false;
      *long_loop |= trips >= minimum_loop;
      *work += (uint64_t)trips * cost->terms[i].alu * vertices;
   }
   return true;
}

bool
radv_bc250_idx_cost_evaluate(const struct radv_bc250_idx_cost *cost, const void *push, size_t size, uint64_t *work)
{
   bool long_loop;
   return cost_evaluate(cost, push, size, work, &long_loop, 256);
}

bool
radv_bc250_idx_cost_use_stored(const struct radv_bc250_idx_cost *cost, const void *push, size_t size)
{
   uint64_t work;
   bool long_loop;
   /* The measured long-loop class starts at 256 iterations. A large static
    * instruction count alone is not evidence that a short loop benefits. */
   return cost_evaluate(cost, push, size, &work, &long_loop, 256) && long_loop && work >= 192ull * 256 * 8;
}

bool
radv_bc250_idx_cost_use_stored_indirect(const struct radv_bc250_idx_cost *cost, const void *push, size_t size)
{
   uint64_t work;
   bool long_loop;
   /* Three stable owner repeats found the 64-iteration indirect class faster
    * with stored outputs. Its measured branch uses the same producer, fetch,
    * shared setup and synchronization; only the CPU selection threshold differs. */
   return cost_evaluate(cost, push, size, &work, &long_loop, 64) && long_loop && work >= 192ull * 64 * 8;
}

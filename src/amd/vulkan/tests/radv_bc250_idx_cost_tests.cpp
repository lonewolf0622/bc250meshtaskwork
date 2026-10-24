/* SPDX-License-Identifier: MIT */
#include "tests/nir_test.h"
#include "radv_bc250_idx_cost.h"

class radv_bc250_idx_cost_test : public nir_test {
protected:
   radv_bc250_idx_cost cost = {};
   radv_bc250_idx_cost_test() : nir_test("radv_bc250_idx_cost_test")
   {
      b->shader->info.stage = MESA_SHADER_MESH;
      b->shader->info.workgroup_size[0] = 128;
      b->shader->info.mesh.max_vertices_out = 256;
      b->shader->info.mesh.max_primitives_out = 128;
      b->shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   }

   void shader(unsigned offset, bool per_primitive = false, bool conditional = false, bool unknown = false,
               bool dynamic_counts = false)
   {
      nir_def *vertices = dynamic_counts ?
         nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .base = int(offset + 4), .range = 4) :
         nir_imm_int(b, b->shader->info.mesh.max_vertices_out);
      nir_def *primitives = dynamic_counts ?
         nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .base = int(offset + 8), .range = 4) :
         nir_imm_int(b, b->shader->info.mesh.max_primitives_out);
      nir_set_vertex_and_primitive_count(b, vertices, primitives, nir_imm_int(b, 0));
      auto *value = nir_local_variable_create(b->impl, glsl_uint_type(), "value");
      auto *step = nir_local_variable_create(b->impl, glsl_uint_type(), "step");
      nir_store_var(b, value, nir_load_local_invocation_index(b), 1);
      nir_store_var(b, step, nir_imm_int(b, 0), 1);
      nir_if *branch = conditional ? nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0)) : nullptr;
      nir_loop *loop = nir_push_loop(b);
      auto *n = nir_load_var(b, step);
      auto *limit = unknown ? nir_load_local_invocation_index(b) :
         nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .base = int(offset), .range = 4);
      nir_if *stop = nir_push_if(b, nir_uge(b, n, limit));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, stop);
      auto *x = nir_load_var(b, value);
      x = nir_ixor(b, x, nir_ishl_imm(b, x, 13));
      x = nir_ixor(b, x, nir_ushr_imm(b, x, 17));
      x = nir_ixor(b, x, nir_ishl_imm(b, x, 5));
      x = nir_iadd_imm(b, nir_imul_imm(b, x, 1664525), 1013904223);
      nir_store_var(b, value, x, 1);
      nir_store_var(b, step, nir_iadd_imm(b, n, 1), 1);
      nir_pop_loop(b, loop);
      if (branch)
         nir_pop_if(b, branch);
      auto *out = nir_variable_create(b->shader, nir_var_shader_out,
                                     glsl_array_type(glsl_uint_type(), 256, 0), "output");
      out->data.location = VARYING_SLOT_VAR0;
      out->data.per_primitive = per_primitive;
      nir_store_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, out), nir_load_local_invocation_index(b)),
                      nir_load_var(b, value), 1);
   }
};

TEST_F(radv_bc250_idx_cost_test, runtime_bounds_change_route_without_changing_shader)
{
   shader(28);
   ASSERT_TRUE(radv_bc250_idx_cost_build(b->shader, &cost));
   uint32_t push[64] = {};
   uint64_t work;
   ASSERT_TRUE(radv_bc250_idx_cost_evaluate(&cost, push, sizeof(push), &work));
   EXPECT_EQ(work, 0u);
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
   push[7] = 64;
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
   push[7] = 255;
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
   push[7] = 256;
   EXPECT_TRUE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
   EXPECT_EQ(b->shader->info.mesh.max_vertices_out, 256u);
   nir_validate_shader(b->shader, "cost analysis preserves input shader");
}

TEST_F(radv_bc250_idx_cost_test, small_meshlets_stay_mesh_even_with_long_loops)
{
   b->shader->info.mesh.max_vertices_out = 64;
   b->shader->info.mesh.max_primitives_out = 64;
   shader(12);
   ASSERT_TRUE(radv_bc250_idx_cost_build(b->shader, &cost));
   uint32_t push[64] = {};
   push[3] = 1024;
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
}

TEST_F(radv_bc250_idx_cost_test, primitive_work_does_not_count_as_vertex_work)
{
   shader(12, true);
   EXPECT_FALSE(radv_bc250_idx_cost_build(b->shader, &cost));
}

TEST_F(radv_bc250_idx_cost_test, conditional_work_does_not_supply_lower_bound)
{
   shader(12, false, true);
   EXPECT_FALSE(radv_bc250_idx_cost_build(b->shader, &cost));
}

TEST_F(radv_bc250_idx_cost_test, unknown_limit_keeps_mesh)
{
   shader(12, false, false, true);
   EXPECT_FALSE(radv_bc250_idx_cost_build(b->shader, &cost));
}

TEST_F(radv_bc250_idx_cost_test, missing_push_data_and_unbounded_limits_keep_mesh)
{
   shader(28);
   ASSERT_TRUE(radv_bc250_idx_cost_build(b->shader, &cost));
   uint32_t push[64] = {};
   push[7] = 256;
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, 28));
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, nullptr, sizeof(push)));
   push[7] = 0xffffffff;
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
}

TEST_F(radv_bc250_idx_cost_test, runtime_empty_small_and_invalid_counts_keep_mesh)
{
   shader(28, false, false, false, true);
   ASSERT_TRUE(radv_bc250_idx_cost_build(b->shader, &cost));
   uint32_t push[64] = {};
   push[7] = 256;
   push[8] = 256;
   push[9] = 128;
   EXPECT_TRUE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
   for (unsigned count : {0u, 31u, 128u, 257u}) {
      push[8] = count;
      EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
      EXPECT_FALSE(radv_bc250_idx_cost_use_stored_indirect(&cost, push, sizeof(push)));
   }
   push[8] = 256;
   for (unsigned count : {0u, 64u, 129u}) {
      push[9] = count;
      EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
      EXPECT_FALSE(radv_bc250_idx_cost_use_stored_indirect(&cost, push, sizeof(push)));
   }
}

TEST_F(radv_bc250_idx_cost_test, measured_medium_indirect_work_uses_stored_without_widening_direct)
{
   shader(28);
   ASSERT_TRUE(radv_bc250_idx_cost_build(b->shader, &cost));
   uint32_t push[64] = {};
   push[7] = 0;
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored_indirect(&cost, push, sizeof(push)));
   push[7] = 63;
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored_indirect(&cost, push, sizeof(push)));
   push[7] = 64;
   EXPECT_TRUE(radv_bc250_idx_cost_use_stored_indirect(&cost, push, sizeof(push)));
   EXPECT_FALSE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
   push[7] = 256;
   EXPECT_TRUE(radv_bc250_idx_cost_use_stored_indirect(&cost, push, sizeof(push)));
   EXPECT_TRUE(radv_bc250_idx_cost_use_stored(&cost, push, sizeof(push)));
}

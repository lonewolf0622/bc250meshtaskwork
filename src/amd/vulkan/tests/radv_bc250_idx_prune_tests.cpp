/* SPDX-License-Identifier: MIT */
#include "tests/nir_test.h"
#include "radv_bc250_idx.h"

class radv_bc250_idx_prune_test : public nir_test {
protected:
   nir_builder fs;
   radv_bc250_idx_prune_test(): nir_test("radv_bc250_idx_prune_test")
   {
      b->shader->info.stage = MESA_SHADER_MESH;
      b->shader->info.workgroup_size[0] = 64;
      b->shader->info.mesh.max_vertices_out = b->shader->info.mesh.max_primitives_out = 64;
      b->shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      fs = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, b->shader->options, "consumer");
   }
   ~radv_bc250_idx_prune_test()
   {
      ralloc_free(fs.shader);
   }

   nir_variable *output(unsigned location, const glsl_type *type, bool primitive = false)
   {
      auto *v = nir_variable_create(b->shader, nir_var_shader_out, glsl_array_type(type, 64, 0), "output");
      v->data.location = location;
      v->data.per_primitive = primitive;
      nir_deref_instr *d = nir_build_deref_array_imm(b, nir_build_deref_var(b, v), 0);
      if (glsl_type_is_vector_or_scalar(type))
         nir_store_deref(b, d, nir_undef(b, glsl_get_vector_elements(type), glsl_get_bit_size(type)),
                         BITFIELD_MASK(glsl_get_vector_elements(type)));
      return v;
   }

   nir_variable *input(unsigned location, const glsl_type *type, bool live, unsigned component = 0)
   {
      auto *v = nir_variable_create(fs.shader, nir_var_shader_in, type, "input");
      v->data.location = location;
      nir_def *value = nir_load_var(&fs, v);
      if (live) {
         auto *color = nir_variable_create(fs.shader, nir_var_shader_out, glsl_uint_type(), "color");
         color->data.location = FRAG_RESULT_DATA0;
         nir_store_var(&fs, color, nir_channel(&fs, value, component), 1);
      }
      return v;
   }

   bool contains(nir_variable *v)
   {
      nir_foreach_shader_out_variable (other, b->shader)
         if (v == other)
            return true;
      return false;
   }
};

TEST_F(radv_bc250_idx_prune_test, drops_unread_output_and_dead_input_without_mutating_consumer)
{
   auto *pos = output(VARYING_SLOT_POS, glsl_uvec_type(4));
   auto *indices = output(VARYING_SLOT_PRIMITIVE_INDICES, glsl_uvec_type(3), true);
   auto *live = output(VARYING_SLOT_VAR0, glsl_uvec_type(4));
   auto *dead = output(VARYING_SLOT_VAR0 + 1, glsl_uvec_type(4));
   auto *unread = output(VARYING_SLOT_VAR0 + 2, glsl_uvec_type(4));
   input(VARYING_SLOT_VAR0, glsl_uvec_type(4), true, 2);
   auto *dead_input = input(VARYING_SLOT_VAR0 + 1, glsl_uvec_type(4), false);
   unsigned instructions = nir_shader_get_entrypoint(fs.shader)->ssa_alloc;
   EXPECT_EQ(radv_bc250_idx_output_layout(b->shader), 4096u);
   ASSERT_TRUE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
   EXPECT_TRUE(contains(pos));
   EXPECT_TRUE(contains(indices));
   EXPECT_TRUE(contains(live));
   EXPECT_FALSE(contains(dead));
   EXPECT_FALSE(contains(unread));
   EXPECT_EQ(radv_bc250_idx_output_layout(b->shader), 2048u);
   EXPECT_EQ(nir_shader_get_entrypoint(fs.shader)->ssa_alloc, instructions);
   EXPECT_EQ(dead_input->data.location, VARYING_SLOT_VAR0 + 1);
   unsigned inputs = 0;
   nir_foreach_shader_in_variable (v, fs.shader)
      inputs++;
   EXPECT_EQ(inputs, 2u);
   EXPECT_FALSE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
   nir_validate_shader(b->shader, "pruned generic outputs");
}

TEST_F(radv_bc250_idx_prune_test, keeps_rasterization_builtins_without_fragment_inputs)
{
   nir_variable *builtins[] = {
      output(VARYING_SLOT_POS, glsl_vec_type(4)),
      output(VARYING_SLOT_PSIZ, glsl_float_type()),
      output(VARYING_SLOT_CLIP_DIST0, glsl_float_type()),
      output(VARYING_SLOT_CULL_DIST0, glsl_float_type()),
      output(VARYING_SLOT_LAYER, glsl_int_type(), true),
      output(VARYING_SLOT_VIEWPORT, glsl_int_type(), true),
      output(VARYING_SLOT_PRIMITIVE_ID, glsl_int_type(), true),
      output(VARYING_SLOT_CULL_PRIMITIVE, glsl_bool_type(), true),
      output(VARYING_SLOT_PRIMITIVE_INDICES, glsl_uvec_type(3), true),
   };
   auto *unused = output(VARYING_SLOT_VAR0, glsl_vec_type(4));
   ASSERT_TRUE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
   for (auto *v : builtins)
      EXPECT_TRUE(contains(v));
   EXPECT_FALSE(contains(unused));
}

TEST_F(radv_bc250_idx_prune_test, retains_whole_array_when_one_location_is_read)
{
   auto *array = output(VARYING_SLOT_VAR0, glsl_array_type(glsl_uvec_type(4), 3, 0));
   auto *unused = output(VARYING_SLOT_VAR0 + 3, glsl_uvec_type(4));
   input(VARYING_SLOT_VAR0 + 2, glsl_uvec_type(4), true, 1);
   ASSERT_TRUE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
   EXPECT_TRUE(contains(array));
   EXPECT_FALSE(contains(unused));
   EXPECT_EQ(radv_bc250_idx_output_layout(b->shader), 3072u);
}

TEST_F(radv_bc250_idx_prune_test, retains_other_components_in_packed_location)
{
   auto *first = output(VARYING_SLOT_VAR0, glsl_uint_type());
   auto *second = output(VARYING_SLOT_VAR0, glsl_uint_type());
   second->data.location_frac = 1;
   input(VARYING_SLOT_VAR0, glsl_uint_type(), true);
   EXPECT_FALSE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
   EXPECT_TRUE(contains(first));
   EXPECT_TRUE(contains(second));
}

TEST_F(radv_bc250_idx_prune_test, preserves_explicit_active_and_transform_feedback_outputs)
{
   auto *active = output(VARYING_SLOT_VAR0, glsl_uvec_type(4));
   auto *xfb = output(VARYING_SLOT_VAR0 + 1, glsl_uvec_type(4));
   active->data.always_active_io = true;
   xfb->data.explicit_xfb_buffer = true;
   EXPECT_FALSE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
   EXPECT_TRUE(contains(active));
   EXPECT_TRUE(contains(xfb));
}

TEST_F(radv_bc250_idx_prune_test, preserves_explicit_active_fragment_input)
{
   auto *active = output(VARYING_SLOT_VAR0, glsl_uvec_type(4));
   input(VARYING_SLOT_VAR0, glsl_uvec_type(4), false)->data.always_active_io = true;
   EXPECT_FALSE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
   EXPECT_TRUE(contains(active));
}

TEST_F(radv_bc250_idx_prune_test, preserves_side_effects_of_dead_output_value)
{
   auto *unused = output(VARYING_SLOT_VAR0, glsl_uint_type());
   b->shader->info.shared_size = 4;
   nir_def *value = nir_shared_atomic(b, 32, nir_imm_int(b, 0), nir_imm_int(b, 1), .atomic_op = nir_atomic_op_iadd);
   nir_store_deref(b, nir_build_deref_array_imm(b, nir_build_deref_var(b, unused), 0), value, 1);
   ASSERT_TRUE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
   EXPECT_FALSE(contains(unused));
   unsigned atomics = 0;
   nir_foreach_block (block, b->impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_shared_atomic)
            atomics++;
      }
   }
   EXPECT_EQ(atomics, 1u);
   nir_validate_shader(b->shader, "dead output side effect");
}

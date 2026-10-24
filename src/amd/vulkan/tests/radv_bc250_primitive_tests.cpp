/* SPDX-License-Identifier: MIT */
#include "../../../../tests/bc250-mesh/dgc/nir_interpreter.h"
#include "tests/nir_test.h"
#include "radv_bc250_primitive.h"

class radv_bc250_primitive_test : public nir_test {
protected:
   radv_bc250_primitive_test(): nir_test("radv_bc250_primitive_test")
   {
   }

   nir_def *parameter(unsigned word, unsigned components = 1)
   {
      return nir_load_push_constant(b, components, 32, nir_imm_int(b, word * 4), .range = 32);
   }

   void result(unsigned slot, nir_def *value)
   {
      nir_store_output(b, value, nir_imm_int(b, 0), .base = (int)slot,
                       .write_mask = BITFIELD_MASK(value->num_components));
   }

   void staging(unsigned corners, bool narrow, bool selected)
   {
      nir_variable *indices = nir_variable_create(
         b->shader, nir_var_mem_shared, glsl_array_type(glsl_vector_type(GLSL_TYPE_UINT, corners), 4, 0), "indices");
      nir_variable *packed = narrow
                                ? nir_variable_create(b->shader, nir_var_mem_shared,
                                                      glsl_array_type(glsl_uint16_t_type(), 4 * corners, 0), "packed")
                                : NULL;
      nir_def *slot = parameter(0), *corner = parameter(1);
      struct radv_bc250_primitive primitive = radv_bc250_primitive_load(b, indices, packed, corners, slot);
      result(0, primitive.indices);
      result(1, radv_bc250_primitive_load_corner(b, indices, packed, corners, slot, corner, selected));
      result(2, primitive.attributes);
      auto layout = [](const glsl_type *type, unsigned *size, unsigned *alignment) {
         *alignment = glsl_get_bit_size(type) / 8;
         *size = *alignment * glsl_get_vector_elements(type);
      };
      nir_lower_vars_to_explicit_types(b->shader, nir_var_mem_shared, layout);
      nir_lower_explicit_io(b->shader, nir_var_mem_shared, nir_address_format_32bit_offset);
      nir_remove_dead_derefs(b->shader);
      nir_validate_shader(b->shader, "shared primitive staging");
      bc250_dgc_nir_interpreter vm(b->shader);
      vm.shared_memory.resize(b->shader->info.shared_size, 0);
      nir_variable *storage = narrow ? packed : indices;
      for (unsigned p = 0; p < 4; p++) {
         for (unsigned c = 0; c < corners; c++) {
            uint32_t value = 7 + p * 31 + c * 11;
            unsigned bytes = narrow ? 2 : 4;
            memcpy(vm.shared_memory.data() + storage->data.driver_location + (p * corners + c) * bytes, &value, bytes);
         }
      }
      for (uint32_t p = 0; p < 4; p++) {
         for (uint32_t c = 0; c < corners; c++) {
            uint32_t constants[8] = {p, c};
            vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 8));
            vm.run(b->shader, 0, 0);
            for (unsigned j = 0; j < corners; j++)
               EXPECT_EQ(vm.outputs[j], 7 + p * 31 + j * 11);
            EXPECT_EQ(vm.outputs[4], 7 + p * 31 + c * 11);
            EXPECT_EQ(vm.outputs[8], p);
         }
      }
   }
};

TEST_F(radv_bc250_primitive_test, culling_preserves_corner_order_and_attribute_identity)
{
   nir_def *indices = parameter(0, 3), *attributes = parameter(3);
   auto primitive = radv_bc250_primitive_make(indices, nir_ine_imm(b, parameter(4), 0), attributes);
   result(0, primitive.indices);
   result(1, primitive.attributes);
   result(2, nir_b2i32(b, radv_bc250_primitive_keep(b, primitive, nir_ine_imm(b, parameter(5), 0))));
   result(3, nir_b2i32(b, radv_bc250_primitive_survives(b, primitive)));
   nir_validate_shader(b->shader, "primitive culling");
   bc250_dgc_nir_interpreter vm(b->shader);
   for (uint32_t culled = 0; culled < 2; culled++) {
      for (uint32_t allocated = 0; allocated < 2; allocated++) {
         uint32_t constants[8] = {5, 2, 7, 93, culled, allocated};
         vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 8));
         vm.run(b->shader, 0, 0);
         EXPECT_EQ(vm.outputs[0], 5u);
         EXPECT_EQ(vm.outputs[1], 2u);
         EXPECT_EQ(vm.outputs[2], 7u);
         EXPECT_EQ(vm.outputs[4], 93u);
         EXPECT_EQ(vm.outputs[8], allocated && !culled);
         EXPECT_EQ(vm.outputs[12], !culled);
      }
   }
}

TEST_F(radv_bc250_primitive_test, absent_cull_reuses_allocation_predicate)
{
   auto primitive = radv_bc250_primitive_make(parameter(0, 3), NULL, parameter(3));
   nir_def *allocated = nir_ine_imm(b, parameter(4), 0);
   EXPECT_EQ(radv_bc250_primitive_keep(b, primitive, allocated), allocated);
   result(0, nir_b2i32(b, radv_bc250_primitive_survives(b, primitive)));
   bc250_dgc_nir_interpreter vm(b->shader);
   uint32_t constants[8] = {};
   vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 8));
   vm.run(b->shader, 0, 0);
   EXPECT_EQ(vm.outputs[0], 1u);
}

TEST_F(radv_bc250_primitive_test, encoded_provoking_vertices_roundtrip)
{
   for (unsigned vertices : {1u, 32u, 64u, 128u, 256u}) {
      for (unsigned primitives : {1u, 32u, 64u, 256u}) {
         EXPECT_EQ(radv_bc250_primitive_vertex_stride(vertices, primitives, false), vertices);
         EXPECT_EQ(radv_bc250_primitive_vertex_stride(vertices, primitives, true), vertices * (primitives + 1));
         for (unsigned provoking : {0u, 2u}) {
            nir_builder *saved = b;
            b = new nir_builder(nir_builder_init_simple_shader(MESA_SHADER_VERTEX, saved->shader->options, "encoded"));
            auto primitive = radv_bc250_primitive_make(parameter(0, 3), NULL, parameter(3));
            nir_def *encoded = radv_bc250_primitive_encode(b, primitive, vertices, provoking);
            result(0, encoded);
            for (unsigned c = 0; c < 3; c++) {
               nir_def *slot = nir_channel(b, encoded, c);
               result(1 + c, nir_vec3(b, nir_umod_imm(b, slot, vertices),
                                      radv_bc250_primitive_decode_attribute(b, slot, vertices, true),
                                      radv_bc250_primitive_decode_attribute(b, slot, vertices, false)));
            }
            nir_validate_shader(b->shader, "encoded primitive");
            bc250_dgc_nir_interpreter vm(b->shader);
            for (uint32_t p = 0; p < primitives; p++) {
               for (uint32_t v = 0; v < vertices; v++) {
                  uint32_t constants[8] = {v, (v + 3) % vertices, (v + 7) % vertices, p};
                  vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 8));
                  vm.run(b->shader, 0, 0);
                  for (unsigned c = 0; c < 3; c++) {
                     EXPECT_EQ(vm.outputs[c], constants[c] + (c == provoking ? (p + 1) * vertices : 0));
                     EXPECT_EQ(vm.outputs[(1 + c) * 4], constants[c]);
                     EXPECT_EQ(vm.outputs[(1 + c) * 4 + 1], c == provoking ? p : 0);
                     EXPECT_EQ(vm.outputs[(1 + c) * 4 + 2], c == provoking ? p : UINT32_MAX);
                  }
               }
            }
            ralloc_free(b->shader);
            delete b;
            b = saved;
         }
      }
   }
}

TEST_F(radv_bc250_primitive_test, merged_instance_corners_are_local_and_bounded)
{
   auto primitive = radv_bc250_primitive_make(parameter(0, 3), NULL, parameter(3));
   nir_def *instance = parameter(4);
   result(0, nir_vec3(b, radv_bc250_primitive_local_corner(b, primitive, 0, instance, 64),
                      radv_bc250_primitive_local_corner(b, primitive, 1, instance, 64),
                      radv_bc250_primitive_local_corner(b, primitive, 2, instance, 64)));
   nir_validate_shader(b->shader, "merged primitive corners");
   bc250_dgc_nir_interpreter vm(b->shader);
   for (uint32_t instance = 0; instance < 4; instance++) {
      uint32_t constants[8] = {instance * 64 + 2, instance * 64 + 63, instance * 64 + 99, 23, instance};
      vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 8));
      vm.run(b->shader, 0, 0);
      EXPECT_EQ(vm.outputs[0], 2u);
      EXPECT_EQ(vm.outputs[1], 63u);
      EXPECT_EQ(vm.outputs[2], 63u);
      constants[0] = instance * 64 - 1;
      constants[1] = instance * 64;
      constants[2] = instance * 64 + 64;
      vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 8));
      vm.run(b->shader, 0, 0);
      EXPECT_EQ(vm.outputs[0], 63u);
      EXPECT_EQ(vm.outputs[1], 0u);
      EXPECT_EQ(vm.outputs[2], 63u);
   }
}

TEST_F(radv_bc250_primitive_test, triangle_vector_tuple)
{
   staging(3, false, false);
}
TEST_F(radv_bc250_primitive_test, triangle_vector_selected_corner)
{
   staging(3, false, true);
}
TEST_F(radv_bc250_primitive_test, triangle_packed_tuple)
{
   staging(3, true, false);
}
TEST_F(radv_bc250_primitive_test, triangle_packed_selected_corner)
{
   staging(3, true, true);
}
TEST_F(radv_bc250_primitive_test, line_vector_selected_corner)
{
   staging(2, false, true);
}
TEST_F(radv_bc250_primitive_test, line_packed_selected_corner)
{
   staging(2, true, true);
}
TEST_F(radv_bc250_primitive_test, point_vector_corner)
{
   staging(1, false, true);
}
TEST_F(radv_bc250_primitive_test, point_packed_corner)
{
   staging(1, true, true);
}

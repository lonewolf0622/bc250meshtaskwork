/* SPDX-License-Identifier: MIT */
#include "../../../../tests/bc250-mesh/dgc/nir_interpreter.h"
#include "tests/nir_test.h"
#include "radv_bc250_dispatch.h"
#include "radv_directmesh.h"

class radv_bc250_dispatch_test : public nir_test {
protected:
   radv_bc250_dispatch_test(): nir_test("radv_bc250_dispatch_test")
   {
   }

   nir_def *parameter(unsigned word, unsigned bits = 32, unsigned components = 1)
   {
      return nir_load_push_constant(b, components, bits, nir_imm_int(b, word * (bits / 8)), .range = 64);
   }

   void result(unsigned slot, nir_def *value)
   {
      nir_store_output(b, value, nir_imm_int(b, 0), .base = (int)slot,
                       .write_mask = BITFIELD_MASK(value->num_components));
   }

   template <typename T> void coordinates(const std::vector<std::array<T, 3>> &grids)
   {
      const unsigned bits = sizeof(T) * 8;
      nir_def *id = parameter(0, bits, 3), *size = parameter(3, bits, 3);
      nir_def *x = nir_channel(b, size, 0), *y = nir_channel(b, size, 1), *z = nir_channel(b, size, 2);
      nir_def *linear = radv_bc250_dispatch_flatten(b, id, size);
      nir_def *count = radv_bc250_dispatch_count(b, x, y, z);
      result(0, bits == 64 ? nir_unpack_64_2x32(b, linear) : linear);
      result(1, bits == 64 ? nir_unpack_64_2x32(b, count) : count);
      result(2, nir_u2u32(b, radv_bc250_dispatch_unflatten(b, parameter(6, bits), x, nir_imul(b, x, y))));
      result(3, nir_u2u32(b, radv_bc250_dispatch_unflatten_modulo(b, parameter(6, bits), x, y)));
      nir_validate_shader(b->shader, "logical dispatch coordinates");
      bc250_dgc_nir_interpreter vm(b->shader);
      for (auto grid : grids) {
         for (T iz : {T(0), T(grid[2] / 2), T(grid[2] - 1)}) {
            for (T iy : {T(0), T(grid[1] / 2), T(grid[1] - 1)}) {
               for (T ix : {T(0), T(grid[0] / 2), T(grid[0] - 1)}) {
                  uint64_t expected = uint64_t(ix) + uint64_t(grid[0]) * (uint64_t(iy) + uint64_t(grid[1]) * iz);
                  T constants[8] = {ix, iy, iz, grid[0], grid[1], grid[2], T(expected), 0};
                  vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 8));
                  vm.run(b->shader, 0, 0);
                  EXPECT_EQ(vm.outputs[0], uint32_t(expected));
                  uint64_t expected_count = uint64_t(grid[0]) * grid[1] * grid[2];
                  EXPECT_EQ(vm.outputs[4], uint32_t(expected_count));
                  if (bits == 64) {
                     EXPECT_EQ(vm.outputs[1], expected >> 32);
                     EXPECT_EQ(vm.outputs[5], expected_count >> 32);
                  }
                  for (unsigned slot : {2u, 3u}) {
                     EXPECT_EQ(vm.outputs[slot * 4], ix);
                     EXPECT_EQ(vm.outputs[slot * 4 + 1], iy);
                     EXPECT_EQ(vm.outputs[slot * 4 + 2], iz);
                  }
               }
            }
         }
      }
   }
};

TEST_F(radv_bc250_dispatch_test, row_major_32bit)
{
   coordinates<uint32_t>({{1, 1, 1}, {17, 1, 1}, {1, 13, 1}, {1, 1, 11}, {7, 5, 3}, {31, 19, 17}});
}

TEST_F(radv_bc250_dispatch_test, dimension_and_chunk_boundaries)
{
   coordinates<uint32_t>({{65535, 1, 1}, {1, 65535, 1}, {65535, 32, 2}, {65535, 65535, 1}, {4096, 1024, 1}});
}

TEST_F(radv_bc250_dispatch_test, task_products_use_64bit)
{
   coordinates<uint64_t>({{65535, 65535, 65535}, {65536, 65536, 3}, {UINT32_MAX, 2, 2}});
}

TEST_F(radv_bc250_dispatch_test, empty_grid_has_zero_groups)
{
   nir_def *size = parameter(0, 32, 3);
   result(0, radv_bc250_dispatch_count(b, nir_channel(b, size, 0), nir_channel(b, size, 1), nir_channel(b, size, 2)));
   bc250_dgc_nir_interpreter vm(b->shader);
   for (unsigned axis = 0; axis < 3; axis++) {
      uint32_t constants[3] = {65535, 32, 2};
      constants[axis] = 0;
      vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 3));
      vm.run(b->shader, 0, 0);
      EXPECT_EQ(vm.outputs[0], 0u);
   }
}

TEST_F(radv_bc250_dispatch_test, merge_and_split_preserve_group_and_instance)
{
   nir_def *group = parameter(0), *instance = parameter(1);
   const unsigned factors[] = {1, 2, 3, 4, 8, 32};
   for (unsigned slot = 0; slot < ARRAY_SIZE(factors); slot++) {
      unsigned factor = factors[slot];
      nir_def *physical = radv_bc250_dispatch_interleave(b, group, factor, instance);
      result(slot, nir_vec3(b, physical, nir_udiv_imm(b, physical, factor), nir_umod_imm(b, physical, factor)));
   }
   nir_validate_shader(b->shader, "interleaved dispatch groups");
   bc250_dgc_nir_interpreter vm(b->shader);
   for (uint32_t group : {0u, 1u, 65534u, 65535u, (1u << 22) - 1}) {
      for (uint32_t instance = 0; instance < 32; instance++) {
         uint32_t constants[2] = {group, instance};
         vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 2));
         vm.run(b->shader, 0, 0);
         for (unsigned slot = 0; slot < ARRAY_SIZE(factors); slot++) {
            unsigned factor = factors[slot];
            if (instance >= factor)
               continue;
            EXPECT_EQ(vm.outputs[slot * 4], group * factor + instance);
            EXPECT_EQ(vm.outputs[slot * 4 + 1], group);
            EXPECT_EQ(vm.outputs[slot * 4 + 2], instance);
         }
      }
   }
}

TEST_F(radv_bc250_dispatch_test, chunk_base_recovers_full_dispatch_coordinates)
{
   nir_def *chunk_local = radv_bc250_dispatch_flatten(b, parameter(0, 32, 3), parameter(3, 32, 3));
   nir_def *global = nir_iadd(b, parameter(6), chunk_local);
   result(0, nir_u2u32(b, radv_bc250_dispatch_unflatten(b, nir_u2u64(b, global), nir_imm_int64(b, 257),
                                                        nir_imm_int64(b, 257 * 127))));
   nir_validate_shader(b->shader, "chunk dispatch coordinates");
   bc250_dgc_nir_interpreter vm(b->shader);
   for (uint32_t base : {0u, 256u, 257u, 257u * 127 - 1, 257u * 127, (1u << 22) - 64}) {
      for (uint32_t local = 0; local < 64; local++) {
         uint32_t constants[7] = {local % 8, (local / 8) % 4, local / 32, 8, 4, 2, base};
         vm.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 7));
         vm.run(b->shader, 0, 0);
         uint32_t expected = base + local;
         EXPECT_EQ(vm.outputs[0], expected % 257);
         EXPECT_EQ(vm.outputs[1], (expected / 257) % 127);
         EXPECT_EQ(vm.outputs[2], expected / (257 * 127));
      }
   }
}

TEST(radv_directmesh, hardware_scope)
{
   for (auto family : {CHIP_NAVI10, CHIP_NAVI12, CHIP_NAVI14, CHIP_GFX1013})
      EXPECT_TRUE(radv_directmesh_family(family));
   for (auto family : {CHIP_POLARIS10, CHIP_RENOIR, CHIP_NAVI21, CHIP_NAVI31})
      EXPECT_FALSE(radv_directmesh_family(family));
   for (unsigned id : {0x13fe, 0x143f, 0x7310, 0x7312, 0x7318, 0x7319, 0x731a, 0x731b,
                       0x731e, 0x731f, 0x7340, 0x7341, 0x7347, 0x734f, 0x7360, 0x7362}) {
      EXPECT_TRUE(radv_directmesh_pci(0x1002, id));
      EXPECT_FALSE(radv_directmesh_pci(0x10de, id));
   }
   for (unsigned id : {0x13ff, 0x7311, 0x7342, 0x7361, 0x15e7, 0x73bf, 0x744c})
      EXPECT_FALSE(radv_directmesh_pci(0x1002, id));
}

TEST(radv_directmesh, activation_override)
{
   EXPECT_TRUE(radv_directmesh_preset(nullptr, true));
   EXPECT_FALSE(radv_directmesh_preset(nullptr, false));
   for (bool detected : {false, true}) {
      EXPECT_TRUE(radv_directmesh_preset("1", detected));
      EXPECT_FALSE(radv_directmesh_preset("0", detected));
      EXPECT_FALSE(radv_directmesh_preset("", detected));
      EXPECT_FALSE(radv_directmesh_preset("invalid", detected));
   }
}

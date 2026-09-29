/* SPDX-License-Identifier: MIT */
#include "nir_test.h"
class nir_opt_uub_test : public nir_test {
protected:
   nir_opt_uub_test() : nir_test("nir_opt_uub_test") {}
   void check(nir_op op, int constant, bool reverse, bool expect_constant,
              unsigned bits = 32) {
      nir_def *x = nir_iand(b, nir_undef(b, 1, bits), nir_imm_intN_t(b, 31, bits));
      nir_def *c = nir_imm_intN_t(b, constant, bits);
      nir_def *v = nir_build_alu2(b, op, reverse ? c : x, reverse ? x : c);
      nir_def *use = nir_mov(b, v);
      nir_opt_uub_options options = {};
      nir_opt_uub(b->shader, &options);
      nir_alu_instr *mov = nir_instr_as_alu(nir_def_instr(use));
      nir_def *actual = mov->src[0].src.ssa;
      if (bits == 64) {
         EXPECT_EQ(actual, v);
      } else {
         while (nir_def_instr(actual)->type == nir_instr_type_alu &&
                nir_instr_as_alu(nir_def_instr(actual))->op == nir_op_mov)
            actual = nir_instr_as_alu(nir_def_instr(actual))->src[0].src.ssa;
         EXPECT_EQ(actual, expect_constant ? c : x);
      }
      nir_validate_shader(b->shader, "uub regression");
   }
};
TEST_F(nir_opt_uub_test, signed_max) { check(nir_op_imax, 63, false, true); }
TEST_F(nir_opt_uub_test, signed_max_reversed) { check(nir_op_imax, 63, true, true); }
TEST_F(nir_opt_uub_test, signed_min) { check(nir_op_imin, 63, false, false); }
TEST_F(nir_opt_uub_test, equal_bound) { check(nir_op_imax, 31, false, true); }
TEST_F(nir_opt_uub_test, negative_max) { check(nir_op_imax, -1, false, false); }
TEST_F(nir_opt_uub_test, negative_min) { check(nir_op_imin, -1, false, true); }
TEST_F(nir_opt_uub_test, excluded_64bit) { check(nir_op_imax, 63, false, false, 64); }

/* SPDX-License-Identifier: MIT */
#include "../../../../tests/bc250-mesh/dgc/nir_interpreter.h"
#include "tests/nir_test.h"
#include "radv_bc250_idx.h"

class radv_bc250_idx_test : public nir_test {
protected:
   radv_bc250_idx_test(): nir_test("radv_bc250_idx_test")
   {
      b->shader->info.stage = MESA_SHADER_MESH;
      b->shader->info.workgroup_size[0] = 64;
      b->shader->info.workgroup_size[1] = b->shader->info.workgroup_size[2] = 1;
      b->shader->info.mesh.max_vertices_out = b->shader->info.mesh.max_primitives_out = 64;
      b->shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   }

   nir_variable *output(const glsl_type *type, unsigned location, bool primitive = false)
   {
      nir_variable *v = nir_variable_create(b->shader, nir_var_shader_out, glsl_array_type(type, 64, 0), "out");
      v->data.location = location;
      v->data.per_primitive = primitive;
      return v;
   }

   nir_deref_instr *element(nir_variable *v, nir_def *index)
   {
      return nir_build_deref_array(b, nir_build_deref_var(b, v), index);
   }

   void program(bool primitive, bool ballot, bool shared, unsigned count)
   {
      nir_def *i = nir_load_local_invocation_index(b);
      nir_def *g = nir_channel(b, nir_load_workgroup_id(b), 0);
      nir_set_vertex_and_primitive_count(b, nir_imm_int(b, 64), nir_imm_int(b, count), nir_imm_int(b, 0));
      nir_def *value = nir_iadd(b, i, nir_imul_imm(b, g, 100));
      if (ballot) {
         nir_def *votes = nir_ballot(b, 4, 32, nir_imm_true(b));
         value = nir_iadd(
            b, value,
            nir_iadd(b, nir_bit_count(b, nir_channel(b, votes, 0)), nir_bit_count(b, nir_channel(b, votes, 1))));
      }
      if (shared) {
         nir_variable *scratch =
            nir_variable_create(b->shader, nir_var_mem_shared, glsl_array_type(glsl_uint_type(), 64, 0), "scratch");
         nir_deref_instr *d = element(scratch, i);
         nir_store_deref(b, d, value, 1);
         nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
         value = nir_load_deref(b, d);
      }
      nir_variable *pos = output(glsl_vector_type(GLSL_TYPE_UINT, 4), VARYING_SLOT_POS);
      /* Separate masked writes must preserve all four stored components. */
      nir_store_deref(b, element(pos, i), nir_vec4(b, value, i, g, nir_imm_int(b, 99)), 5);
      nir_store_deref(b, element(pos, i), nir_vec4(b, value, i, g, nir_imm_int(b, 99)), 10);
      nir_variable *arr = output(glsl_array_type(glsl_uint_type(), 2, 0), VARYING_SLOT_VAR0);
      nir_store_deref(b, nir_build_deref_array_imm(b, element(arr, i), 0), nir_iadd_imm(b, value, 1), 1);
      nir_store_deref(b, nir_build_deref_array_imm(b, element(arr, i), 1), nir_iadd_imm(b, value, 2), 1);
      if (primitive) {
         nir_variable *pp = output(glsl_uint_type(), VARYING_SLOT_VAR0 + 2, true);
         nir_store_deref(b, element(pp, i), nir_iadd(b, nir_imul_imm(b, i, 11), g), 1);
      }
      nir_variable *idx = output(glsl_vector_type(GLSL_TYPE_UINT, 3), VARYING_SLOT_PRIMITIVE_INDICES, true);
      nir_store_deref(
         b, element(idx, i),
         nir_vec3(b, i, nir_iand_imm(b, nir_iadd_imm(b, i, 1), 63), nir_iand_imm(b, nir_iadd_imm(b, i, 2), 63)), 7);
      nir_shader_gather_info(b->shader, b->impl);
   }

   void roundtrip(bool primitive, bool ballot, bool shared, unsigned count, unsigned provoking, bool indirect,
                  bool prune = false, bool dead_primitive = false)
   {
      program(primitive, ballot, shared, count);
      if (prune) {
         nir_def *lane = nir_load_local_invocation_index(b);
         for (unsigned i = 0; i < 2; i++) {
            nir_variable *unused = output(glsl_uvec_type(4), VARYING_SLOT_VAR0 + 4 + i, i != 0);
            nir_store_deref(b, element(unused, lane), nir_imm_ivec4(b, 1, 2, 3, 4), 15);
         }
         nir_builder fs = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, b->shader->options, "consumer");
         nir_variable *array = nir_variable_create(fs.shader, nir_var_shader_in,
            glsl_array_type(glsl_uint_type(), 2, 0), "array");
         array->data.location = VARYING_SLOT_VAR0;
         nir_variable *color = nir_variable_create(fs.shader, nir_var_shader_out, glsl_uvec_type(2), "color");
         color->data.location = FRAG_RESULT_DATA0;
         nir_def *value = nir_load_deref(&fs, nir_build_deref_array_imm(&fs, nir_build_deref_var(&fs, array), 0));
         nir_def *pp_value = nir_imm_int(&fs, 0);
         if (primitive && !dead_primitive) {
            nir_variable *pp = nir_variable_create(fs.shader, nir_var_shader_in, glsl_uint_type(), "primitive");
            pp->data.location = VARYING_SLOT_VAR0 + 2;
            pp->data.per_primitive = true;
            pp_value = nir_load_var(&fs, pp);
         }
         nir_store_var(&fs, color, nir_vec2(&fs, value, pp_value), 3);
         ASSERT_TRUE(radv_bc250_idx_prune_outputs(b->shader, fs.shader));
         ralloc_free(fs.shader);
      }
      const bool kept_primitive = primitive && !dead_primitive;
      const unsigned bytes = radv_bc250_idx_output_layout(b->shader);
      ASSERT_EQ(bytes, kept_primitive ? 1792u : 1536u);
      EXPECT_EQ(bc250_idx_has_pp(b->shader), kept_primitive);
      nir_shader *cs = radv_bc250_idx_build_cs(b->shader, indirect, provoking, bytes, NULL);
      nir_shader *vs = radv_bc250_idx_build_stored_vs(b->shader->options, b->shader, bytes);
      nir_foreach_shader_out_variable (v, vs)
         v->data.driver_location = v->data.location;
      nir_lower_io(vs, nir_var_shader_out, glsl_count_attribute_slots, (nir_lower_io_options)0);
      nir_remove_dead_derefs(vs);
      nir_validate_shader(cs, "producer oracle");
      nir_validate_shader(vs, "fetch oracle");
      constexpr uint64_t ibva = 0x100000, outva = 0x200000, dimsva = 0x300000, drawsva = 0x400000, tabva = 0x500000;
      constexpr unsigned groups = 3;
      const unsigned first = indirect ? 101 : 0;
      std::vector<uint8_t> ib(first * 4 + groups * (64 * 12 + bytes), 0xa5), data(groups * bytes, 0xa5);
      std::vector<uint32_t> dims = {0, 0, 0, groups, 1, 1},
                            draws = {0, 0, 0, 0, 0, groups * 64 * 3, 1, first, 0, groups};
      std::vector<uint32_t> table = {2, groups, groups, 1, 1, 0, 0, 0, 0, 0, groups};
      bc250_dgc_nir_interpreter producer(cs), fetch(vs);
      producer.mappings = {{ibva, ib.size(), ib.data()},
                           {outva, data.size(), data.data()},
                           {dimsva, dims.size() * 4, (uint8_t *)dims.data()},
                           {drawsva, draws.size() * 4, (uint8_t *)draws.data()},
                           {tabva, table.size() * 4, (uint8_t *)table.data()}};
      fetch.mappings = producer.mappings;
      producer.shared_memory.resize(cs->info.shared_size, 0);
      uint64_t constants[9] = {indirect ? tabva : 0,   ibva, 64 * 12, 0, indirect ? dimsva : outva,
                               indirect ? drawsva : 0, 0,    0,       0};
      producer.push_constants.assign((uint8_t *)constants, (uint8_t *)(constants + 9));
      fetch.push_constants = producer.push_constants;
      fetch.draw_id = indirect ? 1 : 0;
      producer.num_workgroups = groups;
      for (unsigned g = 0; g < groups; g++) {
         for (unsigned lane = 0; lane < 64; lane++)
            producer.run(cs, g, lane);
         for (unsigned p = 0; p < 64; p++) {
            for (unsigned c = 0; c < 3; c++) {
               uint32_t k;
               memcpy(&k, ib.data() + (first + g * 64 * 3 + p * 3 + c) * 4, 4);
               if (p >= count) {
                  EXPECT_EQ(k, g * (kept_primitive ? 64 * 65 : 64));
                  continue;
               }
               const unsigned v = (p + c) % 64;
               EXPECT_EQ(k, g * (kept_primitive ? 64 * 65 : 64) + v + (kept_primitive && c == provoking ? (p + 1) * 64 : 0));
               fetch.vertex_id = k;
               fetch.run(vs, 0, 0);
               const unsigned expected = v + g * 100 + (ballot ? 64 : 0);
               EXPECT_EQ(fetch.outputs[VARYING_SLOT_POS * 4], expected);
               EXPECT_EQ(fetch.outputs[VARYING_SLOT_POS * 4 + 1], v);
               EXPECT_EQ(fetch.outputs[VARYING_SLOT_POS * 4 + 2], g);
               EXPECT_EQ(fetch.outputs[VARYING_SLOT_POS * 4 + 3], 99u);
               EXPECT_EQ(fetch.outputs[VARYING_SLOT_VAR0 * 4], expected + 1);
               EXPECT_EQ(fetch.outputs[(VARYING_SLOT_VAR0 + 1) * 4], expected + 2);
               if (kept_primitive && c == provoking) {
                  EXPECT_EQ(fetch.outputs[(VARYING_SLOT_VAR0 + 2) * 4], p * 11 + g);
               }
            }
         }
      }
      ralloc_free(vs);
      ralloc_free(cs);
   }
};

TEST_F(radv_bc250_idx_test, direct_masked_array_outputs)
{
   roundtrip(false, false, false, 64, 0, false);
}
TEST_F(radv_bc250_idx_test, direct_ballot_outputs)
{
   roundtrip(false, true, false, 64, 0, false);
}
TEST_F(radv_bc250_idx_test, direct_shared_outputs)
{
   roundtrip(false, false, true, 64, 0, false);
}
TEST_F(radv_bc250_idx_test, direct_first_provoking)
{
   roundtrip(true, true, false, 64, 0, false);
}
TEST_F(radv_bc250_idx_test, direct_last_provoking)
{
   roundtrip(true, true, false, 64, 2, false);
}
TEST_F(radv_bc250_idx_test, direct_unused_primitives)
{
   roundtrip(true, false, false, 48, 2, false);
}
TEST_F(radv_bc250_idx_test, indirect_ballot_outputs)
{
   roundtrip(true, true, false, 64, 0, true);
}
TEST_F(radv_bc250_idx_test, indirect_unused_primitives)
{
   roundtrip(true, false, false, 48, 2, true);
}

TEST_F(radv_bc250_idx_test, pruned_direct_shared_outputs)
{
   roundtrip(false, false, true, 64, 0, false, true);
}
TEST_F(radv_bc250_idx_test, pruned_direct_first_provoking)
{
   roundtrip(true, true, false, 64, 0, false, true);
}
TEST_F(radv_bc250_idx_test, pruned_indirect_last_provoking)
{
   roundtrip(true, false, false, 48, 2, true, true);
}
TEST_F(radv_bc250_idx_test, pruned_dead_primitive_attributes)
{
   roundtrip(true, false, false, 48, 2, true, true, true);
}

TEST_F(radv_bc250_idx_test, aggregate_layout)
{
   glsl_struct_field fields[2] = {};
   fields[0].name = "a";
   fields[0].type = glsl_uint_type();
   fields[1].name = "b";
   fields[1].type = glsl_array_type(glsl_vec_type(3), 2, 0);
   auto *a = output(glsl_struct_type(fields, 2, "aggregate", false), VARYING_SLOT_VAR0);
   auto *p = output(glsl_uint_type(), VARYING_SLOT_VAR0 + 3, true);
   EXPECT_EQ(radv_bc250_idx_output_layout(b->shader), (4u + 2 * 12) * 64 + 4 * 64);
   EXPECT_EQ(a->data.driver_location, 0);
   EXPECT_EQ(p->data.driver_location, 28 * 64);
   auto *vs = radv_bc250_idx_build_stored_vs(b->shader->options, b->shader, 32 * 64);
   nir_validate_shader(vs, "aggregate fetch");
   ralloc_free(vs);
}

TEST_F(radv_bc250_idx_test, rejects_non_32bit_output)
{
   output(glsl_float16_t_type(), VARYING_SLOT_VAR0);
   EXPECT_EQ(radv_bc250_idx_output_layout(b->shader), 0u);
}

TEST_F(radv_bc250_idx_test, rejects_mismatched_outer_array)
{
   output(glsl_uint_type(), VARYING_SLOT_VAR0);
   b->shader->info.mesh.max_vertices_out = 128;
   EXPECT_EQ(radv_bc250_idx_output_layout(b->shader), 0u);
}

/* Execute the actual setup NIR with 64 CPU invocations. Collectives rendezvous
 * at each intrinsic; their inputs are the values evaluated by the shader, not
 * values recomputed from its allocation formulas. */
#include <chrono>
#include <future>
#include <mutex>
#include <condition_variable>

class idx_wave {
   using value = bc250_dgc_nir_interpreter::value;
   std::mutex mutex;
   std::condition_variable ready;
   std::array<value, 64> inputs = {}, results = {};
   unsigned arrivals = 0, phase = 0;
   nir_intrinsic_instr *instruction = nullptr;
   bool aborted = false;

public:
   void cancel()
   {
      std::lock_guard<std::mutex> lock(mutex);
      aborted = true;
      ready.notify_all();
   }

   value collective(nir_intrinsic_instr *in, unsigned lane, const value &v)
   {
      std::unique_lock<std::mutex> lock(mutex);
      if (aborted)
         throw std::runtime_error("aborted wave");
      if (!arrivals)
         instruction = in;
      if (instruction != in)
         throw std::runtime_error("divergent collective");
      inputs[lane] = v;
      const unsigned old_phase = phase;
      if (++arrivals == 64) {
         uint32_t sum = 0, reduction = 0;
         for (const auto &input : inputs)
            reduction += input[0];
         for (unsigned l = 0; l < 64; l++) {
            results[l] = {};
            switch (in->intrinsic) {
            case nir_intrinsic_read_first_invocation:
               results[l][0] = inputs[0][0];
               break;
            case nir_intrinsic_read_invocation:
               if (inputs[l][1] >= 64)
                  throw std::runtime_error("read invocation out of bounds");
               results[l][0] = inputs[inputs[l][1]][0];
               break;
            case nir_intrinsic_reduce:
               if (nir_intrinsic_reduction_op(in) != nir_op_iadd || in->def.bit_size != 32 ||
                   (nir_intrinsic_cluster_size(in) != 0 && nir_intrinsic_cluster_size(in) != 64))
                  throw std::runtime_error("unsupported reduction");
               results[l][0] = reduction;
               break;
            case nir_intrinsic_inclusive_scan:
            case nir_intrinsic_exclusive_scan:
               if (nir_intrinsic_reduction_op(in) != nir_op_iadd || in->def.bit_size != 32)
                  throw std::runtime_error("unsupported scan");
               if (in->intrinsic == nir_intrinsic_exclusive_scan)
                  results[l][0] = sum;
               sum += inputs[l][0];
               if (in->intrinsic == nir_intrinsic_inclusive_scan)
                  results[l][0] = sum;
               break;
            default:
               throw std::runtime_error("unsupported collective");
            }
         }
         arrivals = 0;
         phase++;
         ready.notify_all();
      } else if (!ready.wait_for(lock, std::chrono::seconds(5), [&] { return phase != old_phase || aborted; })) {
         throw std::runtime_error("collective timeout");
      }
      if (aborted)
         throw std::runtime_error("aborted wave");
      return results[lane];
   }
};

static void
idx_run_setup(nir_shader *setup, const void *constants, size_t size,
              const std::vector<bc250_dgc_nir_interpreter::mapping> &mappings)
{
   idx_wave wave;
   std::mutex error_mutex;
   std::exception_ptr error;
   std::array<std::future<void>, 64> jobs;
   for (unsigned lane = 0; lane < 64; lane++) {
      jobs[lane] = std::async(std::launch::async, [&, lane] {
         try {
            bc250_dgc_nir_interpreter vm(setup);
            vm.push_constants.assign((const uint8_t *)constants, (const uint8_t *)constants + size);
            vm.mappings = mappings;
            vm.collective = [&](nir_intrinsic_instr *in, unsigned local, const bc250_dgc_nir_interpreter::value &v) {
               return wave.collective(in, local, v);
            };
            vm.run(setup, 0, lane);
         } catch (...) {
            {
               std::lock_guard<std::mutex> lock(error_mutex);
               if (!error)
                  error = std::current_exception();
            }
            wave.cancel();
            throw;
         }
      });
   }
   for (auto &job : jobs) {
      try {
         job.get();
      } catch (...) {
      }
   }
   if (error)
      std::rethrow_exception(error);
}

static void
idx_setup_check(const nir_shader_compiler_options *options, bool stored, unsigned actual_count, unsigned capacity,
                unsigned base, bool reset, bool invalid)
{
   const unsigned nmax = 65, P = 64, V = 64 * 65, output_bytes = stored ? 1792 : 0;
   nir_shader *setup = radv_bc250_idx_build_setup(options, stored);
   nir_opt_dce(setup);
   nir_remove_dead_derefs(setup);
   nir_validate_shader(setup, "setup oracle");
   constexpr uint64_t records_va = 0x100000, count_va = 0x200000, table_va = 0x300000, dims_va = 0x400000;
   constexpr uint64_t draws_va = 0x500000, mesh_va = 0x600000, pool_va = 0x700000;
   std::vector<uint32_t> records(nmax * 3), table(8 + nmax + 1, 0xa5a5a5a5), dims(nmax * 3, 0xa5a5a5a5);
   std::vector<uint32_t> draws(nmax * 5, 0xa5a5a5a5), mesh(nmax * 3, 0xa5a5a5a5), pool(64, 0);
   pool[0] = base;
   for (unsigned i = 0; i < nmax; i++) {
      records[i * 3] = i % 7 ? i % 3 + 1 : 0;
      records[i * 3 + 1] = records[i * 3 + 2] = 1;
   }
   if (invalid)
      records[2 * 3] = 65536;
   bc250_idx_setup_args args = {records_va, count_va, table_va, dims_va, draws_va, mesh_va, pool_va,
                                12,         nmax,     capacity, P,       reset,    V,       output_bytes};
   idx_run_setup(setup, &args, sizeof(args),
                 {{records_va, records.size() * 4, (uint8_t *)records.data()},
                  {count_va, 4, (uint8_t *)&actual_count},
                  {table_va, table.size() * 4, (uint8_t *)table.data()},
                  {dims_va, dims.size() * 4, (uint8_t *)dims.data()},
                  {draws_va, draws.size() * 4, (uint8_t *)draws.data()},
                  {mesh_va, mesh.size() * 4, (uint8_t *)mesh.data()},
                  {pool_va, pool.size() * 4, (uint8_t *)pool.data()}});
   const unsigned n = std::min(nmax, actual_count);
   uint64_t used = reset ? 0 : base;
   unsigned groups = 0, fallback = 0;
   bool suffix = false;
   for (unsigned i = 0; i < nmax; i++) {
      if (i >= n) {
         for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(mesh[i * 3 + j], 0u);
         continue;
      }
      const unsigned g = records[i * 3];
      const uint64_t bytes = uint64_t(g) * (P * 12 + output_bytes);
      const bool encodable = g <= 65535 && uint64_t(g) * V < (1ull << 32);
      const bool fits = !suffix && encodable && used * 4 + bytes <= uint64_t(capacity) * 4;
      EXPECT_EQ(table[8 + i], groups);
      EXPECT_EQ(draws[i * 5], fits ? g * P * 3 : 0);
      EXPECT_EQ(draws[i * 5 + 1], unsigned(fits && g != 0));
      EXPECT_EQ(draws[i * 5 + 2], fits ? used : 0);
      EXPECT_EQ(draws[i * 5 + 3], 0u);
      EXPECT_EQ(draws[i * 5 + 4], fits ? g | (1u << 16) : 0);
      for (unsigned j = 0; j < 3; j++) {
         EXPECT_EQ(dims[i * 3 + j], fits ? records[i * 3 + j] : 0);
         EXPECT_EQ(mesh[i * 3 + j], fits ? 0 : records[i * 3 + j]);
      }
      if (fits) {
         used += bytes / 4;
         groups += g;
      } else {
         suffix = true;
         fallback++;
      }
   }
   EXPECT_EQ(table[0], n);
   EXPECT_EQ(table[1], groups);
   EXPECT_EQ(table[2], std::min(groups, 65535u));
   EXPECT_EQ(table[3], groups ? (groups + 65534) / 65535 : 1);
   EXPECT_EQ(table[4], 1u);
   EXPECT_EQ(table[5], fallback ? n : 0);
   EXPECT_EQ(table[8 + n], groups);
   EXPECT_EQ(pool[0], fallback ? capacity : used);
   ralloc_free(setup);
}

TEST_F(radv_bc250_idx_test, setup_exact_capacity)
{
   idx_setup_check(b->shader->options, true, 3, 3200, 0, true, false);
}
TEST_F(radv_bc250_idx_test, setup_overflow_suffix)
{
   idx_setup_check(b->shader->options, true, 65, 3200, 0, true, false);
}
TEST_F(radv_bc250_idx_test, setup_count_zero)
{
   idx_setup_check(b->shader->options, true, 0, 3200, 0, true, false);
}
TEST_F(radv_bc250_idx_test, setup_count_clamp)
{
   idx_setup_check(b->shader->options, true, 100, 100000, 0, true, false);
}
TEST_F(radv_bc250_idx_test, setup_pool_carry)
{
   idx_setup_check(b->shader->options, true, 4, 3500, 300, false, false);
}
TEST_F(radv_bc250_idx_test, setup_invalid_grid)
{
   idx_setup_check(b->shader->options, true, 5, 100000, 0, true, true);
}
TEST_F(radv_bc250_idx_test, setup_replay_unchanged)
{
   idx_setup_check(b->shader->options, false, 65, 100000, 0, true, false);
}

/* Compare the actual batched NIR against sequential executions of the setup
 * oracle above. Distinct tables retain untouched sentinels past the GPU count.
 * Replay and stored entries share a pool and use different sizes and strides. */
static void
idx_setup_batch_check(const nir_shader_compiler_options *options, unsigned calls, unsigned capacity,
                      bool reset, bool invalid, bool count_zero)
{
   constexpr uint64_t list_va = 0x100000, pool_va = 0x200000;
   constexpr unsigned max_records = 65, table_words = 2048;
   std::vector<uint32_t> list(RADV_BC250_IDX_SETUP_BATCH_BYTES / 4, 0);
   auto *args = (bc250_idx_setup_args *)((uint8_t *)list.data() + RADV_BC250_IDX_SETUP_BATCH_HEADER_BYTES);
   std::vector<uint32_t> records(calls * max_records * 4), counts(calls);
   std::vector<uint32_t> tables(calls * table_words, 0xa5a5a5a5), pool(64, 0);
   pool[0] = 300;
   list[0] = calls;
   constexpr uint64_t records_va = 0x300000, counts_va = 0x400000, tables_va = 0x500000;
   for (unsigned i = 0; i < calls; i++) {
      counts[i] = count_zero ? 0 : i % 3 == 0 ? 3 : i % 3 == 1 ? 100 : 4;
      const unsigned stride = i % 2 ? 16 : 12;
      for (unsigned r = 0; r < max_records; r++) {
         records[i * max_records * 4 + r * stride / 4] = r % 7 ? r % 3 + 1 : 0;
         records[i * max_records * 4 + r * stride / 4 + 1] = 1;
         records[i * max_records * 4 + r * stride / 4 + 2] = 1;
      }
      if (invalid && i == 1)
         records[i * max_records * 4 + 2 * stride / 4] = 65536;
      const uint64_t table = tables_va + i * table_words * 4;
      args[i] = {records_va + i * max_records * 16, i % 4 == 3 ? 0 : counts_va + i * 4,
                 table, table + 512, table + 1536, table + 3072, pool_va,
                 stride, max_records, capacity, i % 2 ? 128u : 64u,
                 unsigned(reset && i == 0), 64 * 65, i % 2 ? 1792u : 0u};
   }
   auto batch_tables = tables, batch_pool = pool;
   auto mappings = [&](std::vector<uint32_t> &t, std::vector<uint32_t> &p) {
      return std::vector<bc250_dgc_nir_interpreter::mapping>{
         {list_va, list.size() * 4, (uint8_t *)list.data()},
         {pool_va, p.size() * 4, (uint8_t *)p.data()},
         {records_va, records.size() * 4, (uint8_t *)records.data()},
         {counts_va, counts.size() * 4, (uint8_t *)counts.data()},
         {tables_va, t.size() * 4, (uint8_t *)t.data()}};
   };
   nir_shader *replay = radv_bc250_idx_build_setup(options, false);
   nir_shader *stored = radv_bc250_idx_build_setup(options, true);
   nir_shader *batch = radv_bc250_idx_build_setup_batch(options);
   for (auto *s : {replay, stored, batch}) {
      nir_opt_dce(s);
      nir_remove_dead_derefs(s);
      nir_validate_shader(s, "batch setup oracle");
   }
   for (unsigned i = 0; i < calls; i++)
      idx_run_setup(args[i].output_bytes ? stored : replay, &args[i], sizeof(args[i]), mappings(tables, pool));
   idx_run_setup(batch, &list_va, sizeof(list_va), mappings(batch_tables, batch_pool));
   EXPECT_EQ(batch_tables, tables);
   EXPECT_EQ(batch_pool, pool);
   /* A second submission starts from the recorded reset bit, not CPU-patched
    * GPU state. For carry cases this also checks an already exhausted pool. */
   for (unsigned i = 0; i < calls; i++)
      idx_run_setup(args[i].output_bytes ? stored : replay, &args[i], sizeof(args[i]), mappings(tables, pool));
   idx_run_setup(batch, &list_va, sizeof(list_va), mappings(batch_tables, batch_pool));
   EXPECT_EQ(batch_tables, tables);
   EXPECT_EQ(batch_pool, pool);
   ralloc_free(replay);
   ralloc_free(stored);
   ralloc_free(batch);
}

TEST_F(radv_bc250_idx_test, setup_batch_mixed_layouts)
{
   idx_setup_batch_check(b->shader->options, 4, 100000, true, false, false);
}

TEST_F(radv_bc250_idx_test, setup_batch_overflow_carry)
{
   idx_setup_batch_check(b->shader->options, 4, 3500, false, false, false);
}

TEST_F(radv_bc250_idx_test, setup_batch_invalid_suffix)
{
   idx_setup_batch_check(b->shader->options, 4, 100000, true, true, false);
}

TEST_F(radv_bc250_idx_test, setup_batch_empty_counts)
{
   idx_setup_batch_check(b->shader->options, 3, 3500, true, false, true);
}

TEST_F(radv_bc250_idx_test, setup_batch_full_list)
{
   idx_setup_batch_check(b->shader->options, RADV_BC250_IDX_SETUP_BATCH_CALLS, 1000000, true, false, false);
}

TEST_F(radv_bc250_idx_test, fetch_preserves_clip_cull_lengths)
{
   auto *clip = output(glsl_array_type(glsl_float_type(), 3, 0), VARYING_SLOT_CLIP_DIST0);
   nir_store_deref(b, nir_build_deref_array_imm(b, element(clip, nir_load_local_invocation_index(b)), 0),
                   nir_imm_float(b, 1), 1);
   auto *cull = output(glsl_array_type(glsl_float_type(), 2, 0), VARYING_SLOT_CULL_DIST0);
   nir_store_deref(b, nir_build_deref_array_imm(b, element(cull, nir_load_local_invocation_index(b)), 0),
                   nir_imm_float(b, 1), 1);
   b->shader->info.clip_distance_array_size = 3;
   b->shader->info.cull_distance_array_size = 2;
   const unsigned bytes = radv_bc250_idx_output_layout(b->shader);
   auto *vs = radv_bc250_idx_build_stored_vs(b->shader->options, b->shader, bytes);
   EXPECT_EQ(vs->info.clip_distance_array_size, 3u);
   EXPECT_EQ(vs->info.cull_distance_array_size, 2u);
   ralloc_free(vs);
}

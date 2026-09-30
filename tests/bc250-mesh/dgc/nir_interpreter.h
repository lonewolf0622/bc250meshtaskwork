/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

/* Deliberately small CPU interpreter for the DGC prepare shader (the BC250
 * generator). One invocation at a time; global memory is reached through
 * address translation of the buffers the test provides (preprocess buffer,
 * token stream, params upload). Unimplemented instructions, unmapped
 * addresses and excessive execution are hard test errors, never approximated.
 * The shader must have had its variables lowered to SSA. */
#pragma once

#include "nir.h"

#include <array>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

class bc250_dgc_nir_interpreter {
   using value = std::array<uint64_t, 16>;
   std::vector<value> values;
   nir_block *previous = nullptr;
   uint64_t steps = 0;
   enum flow { normal, stop_break, stop_continue, stop_return };
   static void check(bool b, const char *why) { if (!b) throw std::runtime_error(why); }
   static uint64_t mask(unsigned bits) { return bits == 64 ? UINT64_MAX : (1ull << bits) - 1; }
   uint64_t src(nir_src s, unsigned c = 0) { return values.at(s.ssa->index).at(c); }

   static int64_t sext(uint64_t x, unsigned bits) { return bits == 64 ? (int64_t)x : (int64_t)(x << (64 - bits)) >> (64 - bits); }

   void alu(nir_alu_instr *a)
   {
      auto &out = values.at(a->def.index);
      const unsigned bits = a->def.bit_size;
      for (unsigned c = 0; c < a->def.num_components; ++c) {
         auto input = [&](unsigned s) {
            unsigned component = nir_op_infos[a->op].input_sizes[s] == 1 ? 0 : c;
            return src(a->src[s].src, a->src[s].swizzle[component]);
         };
         const unsigned in_bits = a->src[0].src.ssa->bit_size;
         uint64_t x = input(0), y = nir_op_infos[a->op].num_inputs > 1 ? input(1) : 0;
         uint64_t z = nir_op_infos[a->op].num_inputs > 2 ? input(2) : 0;
         uint64_t result;
         switch (a->op) {
         case nir_op_mov: case nir_op_u2u8: case nir_op_u2u16: case nir_op_u2u32: case nir_op_u2u64: result = x; break;
         case nir_op_i2i32: case nir_op_i2i64: result = (uint64_t)sext(x, in_bits); break;
         case nir_op_vec2: case nir_op_vec3: case nir_op_vec4: case nir_op_vec5: case nir_op_vec8: case nir_op_vec16:
            result = input(c); break;
         case nir_op_iadd: result = x + y; break;
         case nir_op_isub: result = x - y; break;
         case nir_op_imul: result = x * y; break;
         case nir_op_iadd3: result = x + y + z; break;
         case nir_op_imad: result = x * y + z; break;
         case nir_op_ineg: result = -x; break;
         case nir_op_iand: result = x & y; break;
         case nir_op_ior: result = x | y; break;
         case nir_op_ixor: result = x ^ y; break;
         case nir_op_inot: result = ~x; break;
         case nir_op_ishl: result = x << (y & (bits - 1)); break;
         case nir_op_ushr: result = x >> (y & (bits - 1)); break;
         case nir_op_ishr: result = (uint64_t)(sext(x, bits) >> (y & (bits - 1))); break;
         case nir_op_umin: result = x < y ? x : y; break;
         case nir_op_umax: result = x > y ? x : y; break;
         case nir_op_imin: result = sext(x, bits) < sext(y, bits) ? x : y; break;
         case nir_op_imax: result = sext(x, bits) > sext(y, bits) ? x : y; break;
         case nir_op_udiv: check(y != 0, "divide by zero"); result = x / y; break;
         case nir_op_umod: check(y != 0, "modulo by zero"); result = x % y; break;
         case nir_op_ieq: result = x == y ? mask(bits) : 0; break;
         case nir_op_ine: result = x != y ? mask(bits) : 0; break;
         case nir_op_ult: result = x < y ? mask(bits) : 0; break;
         case nir_op_uge: result = x >= y ? mask(bits) : 0; break;
         case nir_op_ilt: result = sext(x, in_bits) < sext(y, in_bits) ? mask(bits) : 0; break;
         case nir_op_ige: result = sext(x, in_bits) >= sext(y, in_bits) ? mask(bits) : 0; break;
         case nir_op_bcsel: result = x ? y : z; break;
         case nir_op_b2i32: case nir_op_b2i64: case nir_op_b2i8: case nir_op_b2i16: result = x != 0; break;
         case nir_op_b2b1: case nir_op_b2b32: result = x != 0 ? mask(bits) : 0; break;
         case nir_op_bit_count: result = __builtin_popcountll(x); break;
         case nir_op_ubfe: {
            unsigned offset = y & 31, width = z & 31;
            result = width ? (x >> offset) & mask(width) : 0; break;
         }
         case nir_op_ibfe: {
            unsigned offset = y & 31, width = z & 31;
            result = width ? (uint64_t)sext((x >> offset) & mask(width), width) : 0; break;
         }
         case nir_op_bfi: result = (x & y) | (~x & z); break;
         case nir_op_pack_64_2x32_split: result = (x & 0xffffffffull) | (y << 32); break;
         case nir_op_unpack_64_2x32_split_x: result = x & 0xffffffffull; break;
         case nir_op_unpack_64_2x32_split_y: result = x >> 32; break;
         case nir_op_pack_64_2x32: {
            uint64_t lo = src(a->src[0].src, a->src[0].swizzle[0]), hi = src(a->src[0].src, a->src[0].swizzle[1]);
            result = (lo & 0xffffffffull) | (hi << 32); break;
         }
         case nir_op_unpack_64_2x32: result = c == 0 ? (x & 0xffffffffull) : (x >> 32); break;
         case nir_op_extract_u8: result = (x >> (8 * (y & 7))) & 0xff; break;
         case nir_op_extract_u16: result = (x >> (16 * (y & 3))) & 0xffff; break;
         default: throw std::runtime_error(std::string("unsupported ALU ") + nir_op_infos[a->op].name);
         }
         out[c] = result & mask(bits);
      }
   }

   const nir_instr *current_instr = NULL;
   uint8_t *translate(uint64_t va, size_t bytes)
   {
      for (const mapping &m : mappings) {
         if (va >= m.va && va + bytes <= m.va + m.size)
            return m.ptr + (va - m.va);
      }
      char why[128];
      snprintf(why, sizeof(why), "unmapped global address 0x%llx (%zu bytes)%s", (unsigned long long)va, bytes,
               getenv("BC250_INTERP_TRACE") ? " [trace]" : "");
      if (getenv("BC250_INTERP_TRACE") && current_instr) {
         nir_print_instr(current_instr, stderr);
         fprintf(stderr, "\n");
      }
      throw std::runtime_error(why);
   }

   void intrinsic(nir_intrinsic_instr *i)
   {
      value unused = {};
      auto &out = nir_intrinsic_infos[i->intrinsic].has_dest ? values.at(i->def.index) : unused;
      switch (i->intrinsic) {
      case nir_intrinsic_barrier: break;
      case nir_intrinsic_load_push_constant: {
         const uint64_t offset = nir_intrinsic_base(i) + src(i->src[0]);
         const unsigned bytes = i->def.bit_size / 8;
         for (unsigned c = 0; c < i->def.num_components; ++c) {
            check(offset + (c + 1) * bytes <= push_constants.size(), "push constant read out of range");
            uint64_t v = 0;
            memcpy(&v, push_constants.data() + offset + c * bytes, bytes);
            out[c] = v;
         }
         break;
      }
      case nir_intrinsic_load_global: {
         const unsigned bytes = i->def.bit_size / 8;
         current_instr = &i->instr;
         for (unsigned c = 0; c < i->def.num_components; ++c) {
            uint64_t v = 0;
            memcpy(&v, translate(src(i->src[0]) + c * bytes, bytes), bytes);
            out[c] = v;
         }
         global_loads++;
         break;
      }
      case nir_intrinsic_store_global: {
         current_instr = &i->instr;
         const unsigned bytes = i->src[0].ssa->bit_size / 8;
         for (unsigned c = 0; c < i->src[0].ssa->num_components; ++c) {
            if (!(nir_intrinsic_write_mask(i) & (1u << c)))
               continue;
            uint64_t v = src(i->src[0], c);
            memcpy(translate(src(i->src[1]) + c * bytes, bytes), &v, bytes);
         }
         global_stores++;
         break;
      }
      case nir_intrinsic_load_workgroup_id: out[0] = workgroup_id; out[1] = 0; out[2] = 0; break;
      case nir_intrinsic_load_local_invocation_id: out[0] = local_id; out[1] = 0; out[2] = 0; break;
      case nir_intrinsic_load_local_invocation_index: out[0] = local_id; break;
      case nir_intrinsic_load_subgroup_id: out[0] = 0; break;
      case nir_intrinsic_load_num_workgroups: out[0] = num_workgroups; out[1] = 1; out[2] = 1; break;
      default:
         throw std::runtime_error(std::string("unsupported intrinsic ") + nir_intrinsic_infos[i->intrinsic].name);
      }
   }

   flow block(nir_block *b)
   {
      std::vector<std::pair<unsigned, value>> phis;
      nir_foreach_instr(instr, b) {
         if (instr->type != nir_instr_type_phi) break;
         auto phi = nir_instr_as_phi(instr);
         bool found = false;
         nir_foreach_phi_src(s, phi) {
            if (s->pred == previous) {
               phis.emplace_back(phi->def.index, values.at(s->src.ssa->index));
               found = true;
            }
         }
         check(found, "missing phi predecessor");
      }
      for (const auto &p : phis) values.at(p.first) = p.second;
      previous = b;
      nir_foreach_instr(instr, b) {
         check(++steps < 50000000, "CPU interpreter step budget exceeded");
         switch (instr->type) {
         case nir_instr_type_phi: break;
         case nir_instr_type_load_const: {
            auto c = nir_instr_as_load_const(instr);
            for (unsigned j = 0; j < c->def.num_components; ++j)
               values.at(c->def.index)[j] = nir_const_value_as_uint(c->value[j], c->def.bit_size);
            break;
         }
         case nir_instr_type_alu: alu(nir_instr_as_alu(instr)); break;
         case nir_instr_type_intrinsic: intrinsic(nir_instr_as_intrinsic(instr)); break;
         case nir_instr_type_undef: break;
         case nir_instr_type_jump:
            switch (nir_instr_as_jump(instr)->type) {
            case nir_jump_break: return stop_break;
            case nir_jump_continue: return stop_continue;
            case nir_jump_return: return stop_return;
            default: throw std::runtime_error("unsupported jump");
            }
         case nir_instr_type_deref:
            throw std::runtime_error("deref instruction: lower variables to SSA first");
         default: throw std::runtime_error("unsupported instruction type " + std::to_string((int)instr->type));
         }
      }
      return normal;
   }

   flow list(exec_list *l)
   {
      foreach_list_typed(nir_cf_node, n, node, l) {
         flow result = normal;
         switch (n->type) {
         case nir_cf_node_block: result = block(nir_cf_node_as_block(n)); break;
         case nir_cf_node_if: {
            auto i = nir_cf_node_as_if(n);
            result = list(src(i->condition) ? &i->then_list : &i->else_list);
            break;
         }
         case nir_cf_node_loop: {
            auto loop = nir_cf_node_as_loop(n);
            check(!nir_loop_has_continue_construct(loop), "unsupported continue construct");
            do { result = list(&loop->body); } while (result == normal || result == stop_continue);
            if (result == stop_break) result = normal;
            break;
         }
         default: throw std::runtime_error("unsupported CF node");
         }
         if (result != normal) return result;
      }
      return normal;
   }

public:
   struct mapping {
      uint64_t va;
      size_t size;
      uint8_t *ptr;
   };
   std::vector<mapping> mappings;
   std::vector<uint8_t> push_constants;
   unsigned workgroup_id = 0, local_id = 0, num_workgroups = 1;
   uint64_t global_loads = 0, global_stores = 0;

   explicit bc250_dgc_nir_interpreter(nir_shader *s) : values(nir_shader_get_entrypoint(s)->ssa_alloc) {}

   /* Runs one invocation (workgroup, local) of the entrypoint. */
   uint64_t run(nir_shader *s, unsigned wg, unsigned local)
   {
      workgroup_id = wg;
      local_id = local;
      previous = nullptr;
      steps = 0;
      for (auto &v : values) v.fill(0);
      list(&nir_shader_get_entrypoint(s)->body);
      return steps;
   }
};

/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_IDX_COST_H
#define RADV_BC250_IDX_COST_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

struct nir_shader;

#define RADV_BC250_COST_NODES 48
#define RADV_BC250_COST_TERMS 8

enum radv_bc250_cost_op {
   RADV_BC250_COST_CONST, RADV_BC250_COST_PUSH,
   RADV_BC250_COST_ADD, RADV_BC250_COST_MUL, RADV_BC250_COST_AND,
   RADV_BC250_COST_OR, RADV_BC250_COST_XOR, RADV_BC250_COST_SHL,
   RADV_BC250_COST_USHR, RADV_BC250_COST_UMIN, RADV_BC250_COST_UMAX, RADV_BC250_COST_NEG,
};

struct radv_bc250_cost_node {
   uint32_t value;
   uint16_t a, b;
   uint8_t op;
};

struct radv_bc250_idx_cost {
   uint16_t vertices, primitives, vertex_count, primitive_count, node_count, term_count;
   struct radv_bc250_cost_node nodes[RADV_BC250_COST_NODES];
   struct {
      uint16_t bound, alu;
   } terms[RADV_BC250_COST_TERMS];
};

#ifdef __cplusplus
extern "C" {
#endif
bool radv_bc250_idx_cost_build(const struct nir_shader *shader, struct radv_bc250_idx_cost *cost);
bool radv_bc250_idx_cost_evaluate(const struct radv_bc250_idx_cost *cost, const void *push, size_t size,
                                   uint64_t *work);
bool radv_bc250_idx_cost_use_stored(const struct radv_bc250_idx_cost *cost, const void *push, size_t size);
bool radv_bc250_idx_cost_use_stored_indirect(const struct radv_bc250_idx_cost *cost, const void *push, size_t size);
#ifdef __cplusplus
}
#endif
#endif

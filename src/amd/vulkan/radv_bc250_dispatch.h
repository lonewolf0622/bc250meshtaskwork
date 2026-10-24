/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_DISPATCH_H
#define RADV_BC250_DISPATCH_H

#include "nir/nir_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Logical groups are row-major: x + X * (y + Y * z). Physical dispatches
 * may merge groups, split primitives or process a chunk. Callers supply the
 * logical dimensions from their own direct, indirect or DGC transport.
 * Keep arithmetic at the caller's width; decoding requires nonzero X and Y
 * and a representable X * Y. Empty dispatches are rejected by the caller.
 */
static inline nir_def *
radv_bc250_dispatch_flatten(nir_builder *b, nir_def *id, nir_def *size)
{
   return nir_iadd(
      b, nir_channel(b, id, 0),
      nir_imul(b, nir_channel(b, size, 0),
               nir_iadd(b, nir_channel(b, id, 1), nir_imul(b, nir_channel(b, size, 1), nir_channel(b, id, 2)))));
}

static inline nir_def *
radv_bc250_dispatch_count(nir_builder *b, nir_def *x, nir_def *y, nir_def *z)
{
   return nir_imul(b, nir_imul(b, x, y), z);
}

/* Quotient/remainder decoding also works for the 64-bit task chunk path.
 * Accept a precomputed X * Y so record addressing can reuse the product.
 */
static inline nir_def *
radv_bc250_dispatch_unflatten(nir_builder *b, nir_def *linear, nir_def *x, nir_def *xy)
{
   nir_def *z = nir_udiv(b, linear, xy);
   nir_def *rest = nir_isub(b, linear, nir_imul(b, z, xy));
   nir_def *y = nir_udiv(b, rest, x);
   return nir_vec3(b, nir_isub(b, rest, nir_imul(b, y, x)), y, z);
}

/* Preserve the merged exporter's modulo form of the same mapping. */
static inline nir_def *
radv_bc250_dispatch_unflatten_modulo(nir_builder *b, nir_def *linear, nir_def *x, nir_def *y)
{
   return nir_vec3(b, nir_umod(b, linear, x), nir_umod(b, nir_udiv(b, linear, x), y),
                   nir_udiv(b, linear, nir_imul(b, x, y)));
}

/* Interleaved physical groups: merge maps (group, instance) to a logical
 * group; split maps (logical group, piece) to a physical group. */
static inline nir_def *
radv_bc250_dispatch_interleave(nir_builder *b, nir_def *group, unsigned factor, nir_def *instance)
{
   assert(factor);
   return nir_iadd(b, nir_imul_imm(b, group, factor), instance);
}

#ifdef __cplusplus
}
#endif

#endif

/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_ROUTE_H
#define RADV_BC250_ROUTE_H

#include <stdbool.h>

/* Shared by compiler admission and executable-plan restoration. Transport
 * and interface ownership checks are additional to route protection. A proven
 * empty output is exempt because neither vertices nor primitives are exported. */
static inline bool
radv_bc250_mesh_protected_route(bool safe_direct, bool ordered, bool split, bool expanded,
                                 unsigned merge_k, bool empty_output)
{
   return safe_direct || ordered || split || expanded || merge_k > 1 || empty_output;
}

#endif

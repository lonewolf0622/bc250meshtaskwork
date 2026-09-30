/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_PIPELINE_PLAN_H
#define RADV_BC250_PIPELINE_PLAN_H

#include <stdbool.h>
#include <stdint.h>

/* Separate from radv_shader_info: the switch-off shader binary layout and
 * compiler keys must remain unchanged. Executable shaders remain ordinary
 * cache objects; a plan references its two private compute shaders last. */
#define RADV_BC250_PIPELINE_PLAN_VERSION 4u
#define RADV_BC250_PLAN_TASK (1u << 0)
#define RADV_BC250_PLAN_ORDERED (1u << 1)
#define RADV_BC250_PLAN_ORDER_FREE (1u << 2)
#define RADV_BC250_PLAN_CORNERS (1u << 3)
#define RADV_BC250_PLAN_FLAGS 15u

struct radv_bc250_pipeline_plan {
   uint32_t version;
   uint32_t direct_pieces;
   uint32_t payload_stride;
   uint32_t flags;
   uint64_t bary_ref_mask;
   uint64_t per_primitive_locations;
   uint32_t split_pieces;
   /* Actual varying slots, each encoded as slot+1; zero means absent.
    * One raw slot and no flat slot is the private CORNER_ID representation. */
   uint32_t bary_slots;
   /* Exact compiler policy bytes, including adaptive in the primitive ceiling byte. */
   uint8_t route_key[24];
   uint8_t hardware_key[8];
   /* Bind the plan to the exact graphics executables, not just their policy. */
   uint8_t mesh_hash[32];
   uint8_t fragment_hash[32];
   uint64_t reserved;
};

static inline bool
radv_bc250_pipeline_plan_valid(const struct radv_bc250_pipeline_plan *plan)
{
   const bool task = plan->flags & RADV_BC250_PLAN_TASK;
   const uint32_t raw = plan->bary_slots & 0xff;
   const uint32_t flat = (plan->bary_slots >> 8) & 0xff;
   if ((plan->bary_slots & 0xffff0000u) || raw > 64 || flat > 64 ||
       (!raw && flat) || (raw && raw == flat) ||
       (raw && !flat && !(plan->flags & RADV_BC250_PLAN_CORNERS)))
      return false;
   const uint64_t refs = (raw ? UINT64_C(1) << (raw - 1) : 0) |
                         (flat ? UINT64_C(1) << (flat - 1) : 0);
   return plan->version == RADV_BC250_PIPELINE_PLAN_VERSION &&
          refs == plan->bary_ref_mask &&
          !(plan->flags & ~RADV_BC250_PLAN_FLAGS) && !plan->reserved &&
          plan->direct_pieces <= 5 && plan->split_pieces <= 5 &&
          (!task || (!plan->direct_pieces && plan->payload_stride >= 16 &&
                     plan->payload_stride <= ((plan->flags & RADV_BC250_PLAN_ORDERED) ? 65536 : 16384) &&
                     !(plan->payload_stride & 15))) &&
          (task || (!plan->payload_stride && !(plan->flags & RADV_BC250_PLAN_ORDERED))) &&
          (!(plan->flags & RADV_BC250_PLAN_ORDER_FREE) || plan->direct_pieces);
}

#endif

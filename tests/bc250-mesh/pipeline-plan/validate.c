/* SPDX-License-Identifier: MIT */
#include "radv_bc250_pipeline_plan.h"
#include <assert.h>

int main(void)
{
   struct radv_bc250_pipeline_plan p = {.version = RADV_BC250_PIPELINE_PLAN_VERSION};
   assert(radv_bc250_pipeline_plan_valid(&p));
   p.flags = RADV_BC250_PLAN_TASK;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.payload_stride = 16;
   assert(radv_bc250_pipeline_plan_valid(&p));
   p.direct_pieces = 1;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.direct_pieces = 0;
   p.payload_stride = 17;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.payload_stride = 16384;
   assert(radv_bc250_pipeline_plan_valid(&p));
   p.payload_stride += 16;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.flags |= RADV_BC250_PLAN_ORDERED;
   assert(radv_bc250_pipeline_plan_valid(&p));
   p.flags = 0;
   p.payload_stride = 0;
   p.direct_pieces = 5;
   p.flags = RADV_BC250_PLAN_ORDER_FREE;
   assert(radv_bc250_pipeline_plan_valid(&p));
   p.direct_pieces = 6;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.direct_pieces = 1;
   p.version++;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.version--;
   p.reserved = 1;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.reserved = 0;
   p.flags |= 1u << 31;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.flags = RADV_BC250_PLAN_CORNERS;
   p.direct_pieces = 0;
   p.bary_slots = 64;
   p.bary_ref_mask = UINT64_C(1) << 63;
   assert(radv_bc250_pipeline_plan_valid(&p));
   p.flags = 0;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   p.bary_slots |= 63u << 8;
   p.bary_ref_mask |= UINT64_C(1) << 62;
   assert(radv_bc250_pipeline_plan_valid(&p));
   p.bary_ref_mask = 0;
   assert(!radv_bc250_pipeline_plan_valid(&p));
   return 0;
}

/* SPDX-License-Identifier: MIT
 * BC250 DGC records ordinary indirect Mesh/Task PM4 with final preprocess
 * addresses. The prepare shader copies that immutable program and its initial
 * data into the caller's preprocess allocation and resolves DRAW_MESH tokens.
 * No private executable, grid encoding or barrier is synthesized a second time.
 */
#include "radv_bc250_dgc.h"
#include "radv_bc250.h"
#include "radv_cmd_buffer.h"
#include "radv_cs.h"
#include "radv_entrypoints.h"
#include "radv_pipeline_graphics.h"
#include "meta/radv_meta.h"
#include "nir_builder.h"
#include "nir_serialize.h"
#include "util/blob.h"
#include "util/u_debug.h"
#include "vk_command_pool.h"

/* Bounded experimental capture. Exceeding any bound rejects recording rather
 * than truncating a program, wrapping an offset, or emitting a raw exporter. */
#define BC250_DGC_MAX_BYTES (1u << 30)
#define BC250_DGC_CODE_MESH 65536u
#define BC250_DGC_CODE_TASK 1048576u

struct bc250_dgc_shape {
   uint32_t records, code, data, stride;
   uint64_t size;
   bool task;
};
struct bc250_dgc_params {
   uint64_t source, output, stream, count;
   uint32_t sequences, stride, code, records;
};

bool
radv_bc250_dgc_layout(const struct radv_device *device, const struct radv_indirect_command_layout *layout)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   return pdev->bc250_native_mesh && !pdev->bc250_native_task &&
          (layout->vk.dgc_info & BITFIELD_BIT(MESA_VK_DGC_DRAW_MESH));
}

static bool
bc250_dgc_shape(const struct radv_indirect_command_layout *layout, const void *next,
                uint32_t sequences, uint32_t max_draws, struct bc250_dgc_shape *s)
{
   const VkGeneratedCommandsPipelineInfoEXT *pi = vk_find_struct_const(next, GENERATED_COMMANDS_PIPELINE_INFO_EXT);
   const VkGeneratedCommandsShaderInfoEXT *si = vk_find_struct_const(next, GENERATED_COMMANDS_SHADER_INFO_EXT);
   if (!pi || si || !pi->pipeline || sequences > 4096 ||
       (layout->vk.dgc_info & ~(BITFIELD_BIT(MESA_VK_DGC_DRAW_MESH) |
          BITFIELD_BIT(MESA_VK_DGC_PC) | BITFIELD_BIT(MESA_VK_DGC_SI))))
      return false;
   VK_FROM_HANDLE(radv_pipeline, base, pi->pipeline);
   if (base->type != RADV_PIPELINE_GRAPHICS)
      return false;
   struct radv_graphics_pipeline *p = radv_pipeline_to_graphics(base);
   const struct radv_shader *ms = base->shaders[MESA_SHADER_MESH];
   if (!ms || ms->info.ms.has_task ||
       !radv_bc250_mesh_protected_route(ms->info.ms.bc250_safe_direct, p->bc250_ordered,
          p->bc250_direct_split_pieces || (p->bc250_plan.flags & RADV_BC250_PLAN_SPLIT),
          ms->info.ms.bc250_expanded, ms->info.ms.bc250_merge_k,
          p->bc250_plan.flags & RADV_BC250_PLAN_EMPTY))
      return false;
   /* Merged indirect helpers keep ordinary inlined application constants;
    * dynamic token constants on that route are not yet an admitted ABI. */
   if (ms->info.ms.bc250_merge_k > 1 && layout->push_constant_mask)
      return false;
   memset(s, 0, sizeof(*s));
   s->task = !!p->bc250_task_pipeline;
   s->records = layout->vk.draw_count ? max_draws : MAX2(sequences, 1);
   if (!s->records || s->records > 4096)
      return false;
   uint64_t draws = layout->vk.draw_count ? s->records : 1;
   uint64_t code = s->task ? draws * BC250_DGC_CODE_TASK : BC250_DGC_CODE_MESH;
   /* Every capture upload, record table and payload is preprocess-owned.
    * The upload margin covers all 1024 chunk constants and descriptor tables.
    * Task count draws reuse the ordinary shared scratch allocation. */
   uint64_t data = 65536u + (uint64_t)s->records * 32;
   if (s->task)
      data += draws * 1048576u + 4096ull * p->bc250_payload_stride;
   uint64_t stride = align64(code + data, 256);
   if (code / 4 >= (1u << 20) || stride > BC250_DGC_MAX_BYTES ||
       stride * sequences > BC250_DGC_MAX_BYTES)
      return false;
   s->code = code;
   s->data = stride - code;
   s->stride = stride;
   s->size = stride * sequences;
   return true;
}

void
radv_bc250_dgc_requirements(struct radv_device *device,
                            const VkGeneratedCommandsMemoryRequirementsInfoEXT *info,
                            VkMemoryRequirements2 *requirements)
{
   VK_FROM_HANDLE(radv_indirect_command_layout, layout, info->indirectCommandsLayout);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct bc250_dgc_shape s;
   requirements->memoryRequirements = (VkMemoryRequirements){
      .alignment = MAX2(256, radv_dgc_get_buffer_alignment(device)),
      .memoryTypeBits = pdev->memory_types_32bit,
      .size = bc250_dgc_shape(layout, info->pNext, info->maxSequenceCount, info->maxDrawCount, &s)
                 ? MAX2(s.size, 256) : 0,
   };
}

#define PC(field) nir_load_push_constant(&b, 1, sizeof(((struct bc250_dgc_params *)0)->field) * 8, \
   nir_imm_int(&b, offsetof(struct bc250_dgc_params, field)), .range = sizeof(struct bc250_dgc_params))

nir_shader *
radv_bc250_dgc_shader(struct radv_device *device, const struct radv_indirect_command_layout *layout)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &device->compiler_info.nir_options[MESA_SHADER_COMPUTE], "bc250_dgc_prepare");
   b.shader->info.workgroup_size[0] = 64;
   nir_def *seq = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *stride = PC(stride), *code = PC(code), *records = PC(records);
   nir_def *offset = nir_u2u64(&b, nir_imul(&b, seq, stride));
   nir_def *src = nir_iadd(&b, PC(source), offset);
   nir_def *dst = nir_iadd(&b, PC(output), offset);
   nir_def *stream = nir_iadd(&b, PC(stream), nir_u2u64(&b, nir_imul_imm(&b, seq, layout->vk.stride)));
   nir_def *count = PC(sequences);
   nir_def *limit = count;
   nir_push_if(&b, nir_ine_imm(&b, PC(count), 0));
   count = nir_umin(&b, count, nir_load_global(&b, 1, 32, PC(count), .align_mul = 4));
   nir_pop_if(&b, NULL);
   /* count must be SSA across the optional load. */
   count = nir_if_phi(&b, count, limit);
   nir_def *active = nir_ult(&b, seq, count);
   nir_def *input_bytes = nir_iadd_imm(&b, nir_imul_imm(&b, records, 12), 4);
   nir_def *app_offset = nir_iand_imm(&b, nir_iadd_imm(&b, input_bytes, 15), ~15u);
   nir_def *reserved = layout->push_constant_mask ? nir_iadd_imm(&b, app_offset, MAX_PUSH_CONSTANTS_SIZE) : input_bytes;
   /* Copy programs and initial uploads. Reserved token records are written
    * separately, so no workgroup lane can race a copy with a token write. */
   nir_variable *i = nir_local_variable_create(b.impl, glsl_uint_type(), "word");
   nir_store_var(&b, i, lane, 1);
   nir_push_loop(&b);
   nir_def *word = nir_load_var(&b, i);
   nir_def *byte = nir_imul_imm(&b, word, 4);
   nir_push_if(&b, nir_uge(&b, byte, stride));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ior(&b, nir_ult(&b, byte, code), nir_uge(&b, byte, nir_iadd(&b, code, reserved))));
   nir_def *v = nir_load_global(&b, 1, 32, nir_iadd(&b, src, nir_u2u64(&b, byte)), .align_mul = 4);
   v = nir_bcsel(&b, nir_iand(&b, nir_inot(&b, active), nir_ult(&b, byte, code)),
                   nir_imm_int(&b, PKT3_NOP_PAD), v);
   nir_store_global(&b, v, nir_iadd(&b, dst, nir_u2u64(&b, byte)), .align_mul = 4);
   nir_pop_if(&b, NULL);
   nir_store_var(&b, i, nir_iadd_imm(&b, word, 64), 1);
   nir_pop_loop(&b, NULL);
   nir_def *input = nir_iadd(&b, dst, nir_u2u64(&b, code));
   if (layout->push_constant_mask) {
      nir_push_if(&b, nir_ult_imm(&b, lane, MAX_PUSH_CONSTANTS_SIZE / 4));
      nir_def *pc_offset = nir_iadd(&b, app_offset, nir_imul_imm(&b, lane, 4));
      nir_def *initial = nir_load_global(&b, 1, 32, nir_iadd(&b, nir_iadd(&b, src, nir_u2u64(&b, code)),
         nir_u2u64(&b, pc_offset)), .align_mul = 4);
      nir_def *value = initial;
      u_foreach_bit64(idx, layout->push_constant_mask) {
         nir_def *pc_value = (layout->sequence_index_mask & (1ull << idx)) ? seq :
            nir_load_global(&b, 1, 32, nir_iadd_imm(&b, stream, layout->push_constant_offsets[idx]), .align_mul = 4);
         value = nir_bcsel(&b, nir_ieq_imm(&b, lane, idx), pc_value, value);
      }
      nir_store_global(&b, value, nir_iadd(&b, input, nir_u2u64(&b, pc_offset)), .align_mul = 4);
      nir_pop_if(&b, NULL);
   }
   nir_def *token = nir_iadd_imm(&b, stream, layout->vk.draw_src_offset_B);
   nir_def *draws = nir_imm_int(&b, 1), *address = token, *step = nir_imm_int(&b, 12);
   if (layout->vk.draw_count) {
      nir_def *data = nir_load_global(&b, 4, 32, token, .align_mul = 4);
      address = nir_pack_64_2x32(&b, nir_channels(&b, data, 3));
      step = nir_channel(&b, data, 2);
      draws = nir_umin(&b, records, nir_channel(&b, data, 3));
   }
   draws = nir_bcsel(&b, active, draws, nir_imm_int(&b, 0));
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_global(&b, layout->vk.draw_count ? draws : nir_bcsel(&b, active, nir_iadd_imm(&b, seq, 1), nir_imm_int(&b, 0)),
      input, .align_mul = 4);
   nir_pop_if(&b, NULL);
   nir_store_var(&b, i, lane, 1);
   nir_push_loop(&b);
   word = nir_load_var(&b, i);
   nir_push_if(&b, nir_uge(&b, word, records));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);
   nir_def *valid = layout->vk.draw_count ? nir_ult(&b, word, draws) : nir_iand(&b, active, nir_ieq(&b, word, seq));
   nir_def *xyz = nir_imm_ivec3(&b, 0, 0, 0);
   nir_def *zero_xyz = xyz;
   nir_push_if(&b, valid);
   xyz = nir_load_global(&b, 3, 32, nir_iadd(&b, address,
      layout->vk.draw_count ? nir_imul(&b, nir_u2u64(&b, word), nir_u2u64(&b, step)) : nir_imm_int64(&b, 0)), .align_mul = 4);
   nir_pop_if(&b, NULL);
   xyz = nir_if_phi(&b, xyz, zero_xyz);
   nir_store_global(&b, xyz, nir_iadd(&b, input, nir_u2u64(&b, nir_iadd_imm(&b, nir_imul_imm(&b, word, 12), 4))),
      .align_mul = 4);
   nir_store_var(&b, i, nir_iadd_imm(&b, word, 64), 1);
   nir_pop_loop(&b, NULL);
   nir_validate_shader(b.shader, "BC250 DGC token resolution");
   const char *dump = debug_get_option("BC250_DGC_DUMP", NULL);
   if (dump) {
      char path[4096];
      snprintf(path, sizeof(path), "%s/prepare-%u.nir", dump, layout->vk.draw_count);
      struct blob blob;
      blob_init(&blob);
      nir_serialize(&blob, b.shader, false);
      FILE *f = fopen(path, "wb");
      if (f) { fwrite(blob.data, 1, blob.size, f); fclose(f); }
      blob_finish(&blob);
   }
   return b.shader;
}

static bool
bc250_dgc_state_valid(const struct radv_cmd_buffer *state, const VkGeneratedCommandsInfoEXT *info)
{
   const struct radv_device *device = radv_cmd_buffer_device(state);
   const VkGeneratedCommandsPipelineInfoEXT *pi = vk_find_struct_const(info->pNext, GENERATED_COMMANDS_PIPELINE_INFO_EXT);
   VK_FROM_HANDLE(radv_pipeline, p, pi ? pi->pipeline : VK_NULL_HANDLE);
   const struct radv_cmd_state *s = &state->state;
   if (!p || state->qf != RADV_QUEUE_GENERAL || state->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY ||
       (state->vk.pool->flags & VK_COMMAND_POOL_CREATE_PROTECTED_BIT) ||
       (state->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) ||
       s->graphics_pipeline != radv_pipeline_to_graphics(p) || s->render.view_mask || state->gang.cs ||
       s->render.vrs_att.iview || s->uses_vrs_attachment || s->force_vrs_per_vertex ||
       s->cond_render.enabled || s->active_occlusion_queries || s->active_pipeline_queries ||
       s->active_emulated_pipeline_queries || s->active_pipeline_ace_queries || s->active_prims_gen_queries ||
       s->active_emulated_prims_gen_queries || s->active_prims_xfb_queries || s->active_emulated_prims_xfb_queries ||
       s->streamout.enabled_mask || device->sqtt.bo || device->utrace.context || device->bc250_timer.enabled ||
       radv_bc250_chain_enabled(device) || device->bc250_split_batch_prep)
      return false;
   for (unsigned i = 0; i < MAX_BIND_POINTS; i++) {
      if (state->descriptors[i].push_set.set.size || state->descriptors[i].valid_heaps ||
          state->descriptors[i].dynamic_offset_count)
         return false;
   }
   for (unsigned i = 0; i < MAX_SETS; i++) {
      if (state->descriptor_buffers[i])
         return false;
      for (unsigned j = 0; j < MAX_BIND_POINTS; j++)
         if (state->descriptors[j].descriptor_buffers[i])
            return false;
   }
   return true;
}

static bool
bc250_dgc_capture(struct radv_cmd_buffer *owner, const struct radv_cmd_buffer *state,
                  const VkGeneratedCommandsInfoEXT *info, const struct bc250_dgc_shape *shape,
                  uint32_t seq, uint8_t *snapshot)
{
   struct radv_device *device = radv_cmd_buffer_device(owner);
   VK_FROM_HANDLE(radv_indirect_command_layout, layout, info->indirectCommandsLayout);
   struct radv_cmd_buffer *cmd = malloc(sizeof(*cmd));
   if (!cmd)
      return false;
   *cmd = *state;
   cmd->bc250_dgc_upload_va = info->preprocessAddress + (uint64_t)seq * shape->stride + shape->code;
   memset(&cmd->upload, 0, sizeof(cmd->upload));
   list_inithead(&cmd->upload.list);
   cmd->upload.map = snapshot + shape->code;
   cmd->upload.size = shape->data;
   unsigned app_offset = align(4 + shape->records * 12, 16);
   if (layout->push_constant_mask) {
      cmd->bc250_dgc_application_va = cmd->bc250_dgc_upload_va + app_offset;
      memcpy(cmd->upload.map + app_offset, state->push_constants, MAX_PUSH_CONSTANTS_SIZE);
   }
   cmd->upload.offset = align(app_offset + (layout->push_constant_mask ? MAX_PUSH_CONSTANTS_SIZE : 0), 256);
   unsigned fence_offset = align(cmd->upload.offset, 8);
   cmd->gfx9_fence_va = cmd->bc250_dgc_upload_va + fence_offset;
   cmd->gfx9_fence_idx = 0;
   cmd->gfx9_eop_bug_va = 0;
   cmd->upload.offset = fence_offset + 8;
   cmd->bc250_small_arena = NULL;
   cmd->bc250_ordered_arena = 0;
   cmd->bc250_ordered_arena_size = 0;
   cmd->bc250_split_batch = NULL;
   cmd->utrace.trace = NULL;
   cmd->state.meta = (struct radv_meta_saved_state){0};
   cmd->state.dirty = state->state.dirty;
   cmd->state.dirty_dynamic = cmd->state.graphics_pipeline->needed_dynamic_state;
   cmd->state.emitted_ps = NULL;

   cmd->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_VS_PARTIAL_FLUSH |
                            RADV_CMD_FLAG_PS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_L2;
   cmd->usage_flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
   memset(&cmd->vs_prologs, 0, sizeof(cmd->vs_prologs));
   memset(&cmd->ps_epilogs, 0, sizeof(cmd->ps_epilogs));
   if (device->vs_prologs.ops)
      _mesa_set_init(&cmd->vs_prologs, NULL, device->vs_prologs.ops->hash, device->vs_prologs.ops->equals);
   if (device->ps_epilogs.ops)
      _mesa_set_init(&cmd->ps_epilogs, NULL, device->ps_epilogs.ops->hash, device->ps_epilogs.ops->equals);
   for (unsigned i = 0; i < MAX_BIND_POINTS; i++) {
      cmd->descriptors[i].dirty |= cmd->descriptors[i].valid;
      cmd->descriptors[i].dirty_dynamic = true;
      cmd->descriptors[i].indirect_descriptor_sets_va = 0;
      cmd->push_constant_state[i].need_upload = true;
   }
   cmd->push_constant_stages = VK_SHADER_STAGE_ALL;
   cmd->cs = NULL;
   bool ok = radv_create_cmd_stream(device, AMD_IP_GFX, false, &cmd->cs) == VK_SUCCESS;
   if (ok) {
      /* A capture is a single flat program. Reserve the entire bound before
       * emission; any larger program is rejected before it can be published. */
      radeon_check_space(device->ws, cmd->cs->b, shape->code / 4);
      struct radv_graphics_pipeline *pipeline = cmd->state.graphics_pipeline;
      cmd->state.graphics_pipeline = NULL;
      radv_foreach_stage(stage, RADV_GRAPHICS_STAGE_BITS)
         cmd->state.shaders[stage] = NULL;
      cmd->state.active_stages &= ~RADV_GRAPHICS_STAGE_BITS;
      radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          radv_pipeline_to_handle(&pipeline->base));
      const uint32_t *capture_buf = cmd->cs->b->buf;
      uint64_t records = cmd->bc250_dgc_upload_va + 4;
      uint64_t count = cmd->bc250_dgc_upload_va;
      if (shape->task && !layout->vk.draw_count) {
         radv_bc250_draw_task_dgc(cmd, records + (uint64_t)seq * 12, count, seq);
      } else {
         VkDrawIndirectCount2InfoKHR draw = {
            .sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_COUNT_2_INFO_KHR,
            .addressRange = {.address = records, .size = (uint64_t)shape->records * 12, .stride = 12},
            .countAddressRange = {.address = count, .size = 4},
            .maxDrawCount = layout->vk.draw_count ? shape->records : seq + 1,
         };
         radv_CmdDrawMeshTasksIndirectCount2EXT(radv_cmd_buffer_to_handle(cmd), &draw);
      }
      /* Preserve the ordinary helper's deferred post-consumer flush, including
       * readers that must finish before this preprocess region is reused. */
      radv_emit_cache_flush(cmd);
      ok = !vk_command_buffer_has_error(&cmd->vk) && cmd->cs->b->buf == capture_buf &&
           cmd->cs->b->cdw * 4 <= shape->code &&
           cmd->upload.offset <= shape->data && !cmd->gang.cs;
      if (ok) {
         memcpy(snapshot, cmd->cs->b->buf, cmd->cs->b->cdw * 4);
         uint32_t *padding = (uint32_t *)snapshot;
         for (uint32_t i = cmd->cs->b->cdw; i < shape->code / 4; i++)
            padding[i] = PKT3_NOP_PAD;
         /* Transfer residency only: an unfinalized capture has no IB buffers.
          * The winsys copies its residency list and no executable commands. */
         device->ws->cs_execute_secondary(owner->cs->b, cmd->cs->b, false);
         owner->queue_state.shader_upload_seq = MAX2(owner->queue_state.shader_upload_seq, cmd->queue_state.shader_upload_seq);
         owner->queue_state.compute_scratch_size_per_wave_needed = MAX2(owner->queue_state.compute_scratch_size_per_wave_needed,
            cmd->queue_state.compute_scratch_size_per_wave_needed);
         owner->queue_state.compute_scratch_waves_wanted = MAX2(owner->queue_state.compute_scratch_waves_wanted,
            cmd->queue_state.compute_scratch_waves_wanted);
         owner->queue_state.scratch_size_per_wave_needed = MAX2(owner->queue_state.scratch_size_per_wave_needed,
            cmd->queue_state.scratch_size_per_wave_needed);
         owner->queue_state.scratch_waves_wanted = MAX2(owner->queue_state.scratch_waves_wanted, cmd->queue_state.scratch_waves_wanted);
         owner->queue_state.gds_needed |= cmd->queue_state.gds_needed;
         owner->queue_state.mesh_scratch_ring_needed |= cmd->queue_state.mesh_scratch_ring_needed;
         owner->queue_state.esgs_ring_size_needed = MAX2(owner->queue_state.esgs_ring_size_needed, cmd->queue_state.esgs_ring_size_needed);
         owner->queue_state.gsvs_ring_size_needed = MAX2(owner->queue_state.gsvs_ring_size_needed, cmd->queue_state.gsvs_ring_size_needed);
      }
      radv_destroy_cmd_stream(device, cmd->cs);
   }
   _mesa_set_fini(&cmd->vs_prologs, NULL);
   _mesa_set_fini(&cmd->ps_epilogs, NULL);
   free(cmd);
   return ok;
}

static bool
bc250_dgc_info_valid(const struct radv_cmd_buffer *cmd, const VkGeneratedCommandsInfoEXT *info,
                     struct bc250_dgc_shape *shape)
{
   VK_FROM_HANDLE(radv_indirect_command_layout, layout, info->indirectCommandsLayout);
   return !info->indirectExecutionSet && bc250_dgc_state_valid(cmd, info) &&
      bc250_dgc_shape(layout, info->pNext, info->maxSequenceCount, info->maxDrawCount, shape) &&
      info->preprocessSize >= shape->size && !(info->preprocessAddress & 255) &&
      (info->preprocessAddress >> 32) == radv_device_physical(radv_cmd_buffer_device(cmd))->info.address32_hi &&
      (uint64_t)(uint32_t)info->preprocessAddress + shape->size <= (1ull << 32) &&
      info->indirectAddressSize >= (uint64_t)layout->vk.stride * info->maxSequenceCount;
}

void
radv_bc250_dgc_prepare(struct radv_cmd_buffer *cmd, const VkGeneratedCommandsInfoEXT *info,
                       struct radv_cmd_buffer *state)
{
   struct bc250_dgc_shape shape;
   if (!bc250_dgc_info_valid(state, info, &shape) || cmd->qf != RADV_QUEUE_GENERAL ||
       cmd->state.cond_render.enabled || cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (!info->maxSequenceCount)
      return;
   uint8_t *snapshot = calloc(1, shape.size);
   if (!snapshot) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   bool ok = true;
   for (uint32_t seq = 0; seq < info->maxSequenceCount && ok; seq++)
      ok = bc250_dgc_capture(cmd, state, info, &shape, seq, snapshot + (uint64_t)seq * shape.stride);
   unsigned offset = 0;
   if (ok)
      ok = radv_cmd_buffer_upload_data(cmd, shape.size, snapshot, &offset);
   if (!ok) {
      free(snapshot);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   VK_FROM_HANDLE(radv_indirect_command_layout, layout, info->indirectCommandsLayout);
   struct bc250_dgc_params params = {
      .source = radv_cmd_buffer_upload_va(cmd) + offset, .output = info->preprocessAddress,
      .stream = info->indirectAddress, .count = info->sequenceCountAddress,
      .sequences = info->maxSequenceCount, .stride = shape.stride,
      .code = shape.code, .records = shape.records,
   };
   const char *dump = debug_get_option("BC250_DGC_DUMP", NULL);
   if (dump) {
      char path[4096];
      snprintf(path, sizeof(path), "%s/capture-%u.bin", dump, layout->vk.draw_count);
      FILE *f = fopen(path, "wb");
      if (f) { fwrite(&params, 1, sizeof(params), f); fwrite(snapshot, 1, shape.size, f); fclose(f); }
   }
   free(snapshot);
   cmd->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_VS_PARTIAL_FLUSH |
                            RADV_CMD_FLAG_PS_PARTIAL_FLUSH | RADV_CMD_FLAG_WB_L2;
   radv_meta_begin(cmd);
   radv_meta_bind_compute_pipeline(cmd, layout->pipeline);
   radv_meta_push_constants(cmd, layout->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
   radv_CmdDispatchBase(radv_cmd_buffer_to_handle(cmd), 0, 0, 0, info->maxSequenceCount, 1, 1);
   radv_meta_end(cmd);
}

void
radv_bc250_dgc_execute(struct radv_cmd_buffer *cmd, VkBool32 preprocessed, const VkGeneratedCommandsInfoEXT *info)
{
   struct bc250_dgc_shape shape;
   if (!bc250_dgc_info_valid(cmd, info, &shape)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (!preprocessed)
      radv_bc250_dgc_prepare(cmd, info, cmd);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   struct radv_device *device = radv_cmd_buffer_device(cmd);
   struct radv_graphics_pipeline *p = radv_bc250_mesh_pipeline(cmd);
   if (!radv_bc250_dgc_before(cmd))
      return;
   /* Explicit preprocess buffers may be reset after completion. Scratch and
    * constants are all in preprocess memory; executable residency belongs to
    * the execution command buffer as well as the preprocessing buffer. */
   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      struct radv_shader *shader = p->base.shaders[i];
      if (shader) {
         radv_cs_add_buffer(device->ws, cmd->cs->b, shader->bo);
         cmd->queue_state.shader_upload_seq = MAX2(cmd->queue_state.shader_upload_seq, shader->upload_seq);
      }
   }
   VkPipeline helpers[] = {p->bc250_setup_pipeline, p->bc250_task_pipeline};
   for (unsigned i = 0; i < ARRAY_SIZE(helpers); i++) {
      VK_FROM_HANDLE(radv_pipeline, helper, helpers[i]);
      if (helper) {
         struct radv_shader *shader = helper->shaders[MESA_SHADER_COMPUTE];
         radv_cs_add_buffer(device->ws, cmd->cs->b, shader->bo);
         cmd->queue_state.shader_upload_seq = MAX2(cmd->queue_state.shader_upload_seq, shader->upload_seq);
         cmd->queue_state.compute_scratch_size_per_wave_needed = MAX2(cmd->queue_state.compute_scratch_size_per_wave_needed,
            shader->config.scratch_bytes_per_wave);
         cmd->queue_state.compute_scratch_waves_wanted = MAX2(cmd->queue_state.compute_scratch_waves_wanted,
            radv_get_max_scratch_waves(device, shader));
      }
   }
   cmd->queue_state.gds_needed |= shape.task;
   cmd->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_L2 | RADV_CMD_FLAG_INV_VCACHE |
                            RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_WB_L2;
   radv_emit_cache_flush(cmd);
   ac_emit_cp_pfp_sync_me(cmd->cs->b, false);
   for (uint32_t seq = 0; seq < info->maxSequenceCount; seq++) {
      radeon_check_space(device->ws, cmd->cs->b, 4);
      device->ws->cs_chain_dgc_ib(cmd->cs->b, info->preprocessAddress + (uint64_t)seq * shape.stride,
                                 shape.code / 4, 0, false);
   }
   /* Captured helper dispatches and indirect draws change tracked registers.
    * Require the next ordinary draw/dispatch to re-emit all affected state. */
   radv_init_cmd_stream(device, cmd->cs, AMD_IP_GFX);
   cmd->state.dirty |= RADV_CMD_DIRTY_GRAPHICS_PIPELINE |
                       RADV_CMD_DIRTY_COMPUTE_PIPELINE | RADV_CMD_DIRTY_PS_STATE |
                       RADV_CMD_DIRTY_NGG_STATE | RADV_CMD_DIRTY_INDEX_BUFFER;
   cmd->state.dirty_dynamic |= cmd->state.graphics_pipeline->needed_dynamic_state;

   cmd->state.emitted_ps = NULL;
   cmd->state.last_index_type = -1;
   cmd->state.last_num_instances = -1;
   cmd->state.last_drawid = -1;
   for (unsigned i = 0; i < MAX_BIND_POINTS; i++) {
      cmd->descriptors[i].dirty |= cmd->descriptors[i].valid;
      cmd->descriptors[i].dirty_dynamic = true;
   }
   cmd->push_constant_stages |= VK_SHADER_STAGE_ALL;
   cmd->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_VS_PARTIAL_FLUSH |
                            RADV_CMD_FLAG_PS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_L2;
}

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
#define BC250_DGC_CODE_PATCH 131072u

struct bc250_dgc_shape {
   uint32_t records, code, data, stride, segments, segment_code, template_offset;
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
   if (!pi || si || !pi->pipeline || !layout->bc250_pc_stages_valid || sequences > 1048576 ||
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
   /* Hybrid Task requires its private producer/consumer constants. A merged
    * Mesh-only helper instead retains the application's ordinary ABI. */
   if (p->bc250_task_pipeline && ms->info.ms.bc250_merge_k > 1)
      return false;
   memset(s, 0, sizeof(*s));
   s->task = !!p->bc250_task_pipeline;
   s->records = layout->vk.draw_count ? max_draws : MAX2(sequences, 1);
   if (!s->records || (layout->vk.draw_count && s->records > 4096))
      return false;
   uint64_t draws = layout->vk.draw_count ? s->records : 1;
   bool reuse = s->task && layout->vk.draw_count;
   uint64_t code = s->task ? BC250_DGC_CODE_TASK + (reuse ? draws * BC250_DGC_CODE_PATCH : 0) : BC250_DGC_CODE_MESH;
   /* Every capture upload, record table and payload is preprocess-owned.
    * The upload margin covers all 1024 chunk constants and descriptor tables.
    * Task count draws reuse the ordinary shared scratch allocation. */
   uint64_t data = 65536u + (uint64_t)s->records * 32;
   if (s->task)
      data += 1048576u + 4096ull * p->bc250_payload_stride;
   uint64_t stride = align64(code + data, 256);
   if (stride > BC250_DGC_MAX_BYTES ||
       stride * sequences > BC250_DGC_MAX_BYTES)
      return false;
   s->code = code;
   s->data = stride - code;
   s->stride = stride;
   s->size = stride * sequences;
   s->segments = s->task ? draws : 1;
   s->segment_code = reuse ? BC250_DGC_CODE_PATCH : s->code;
   s->template_offset = reuse ? draws * BC250_DGC_CODE_PATCH : 0;
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
   if (!radv_device_physical(device)->bc250_expose_dgc)
      return false;
   const VkGeneratedCommandsPipelineInfoEXT *pi = vk_find_struct_const(info->pNext, GENERATED_COMMANDS_PIPELINE_INFO_EXT);
   VK_FROM_HANDLE(radv_pipeline, p, pi ? pi->pipeline : VK_NULL_HANDLE);
   const struct radv_cmd_state *s = &state->state;
   VK_FROM_HANDLE(radv_indirect_command_layout, layout, info->indirectCommandsLayout);
   if (state->bc250_dgc_nonuniform_pc & ~layout->push_constant_mask)
      return false;
   if (!p || state->qf != RADV_QUEUE_GENERAL || state->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY ||
       (state->vk.pool->flags & VK_COMMAND_POOL_CREATE_PROTECTED_BIT) ||
       (state->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) ||
       s->graphics_pipeline != radv_pipeline_to_graphics(p) || s->render.view_mask || state->gang.cs ||
       s->render.vrs_att.iview || s->uses_vrs_attachment || s->force_vrs_per_vertex ||
       device->force_vrs_enabled || s->dynamic.vk.fsr.fragment_size.width != 1 ||
       s->dynamic.vk.fsr.fragment_size.height != 1 ||
       s->dynamic.vk.fsr.combiner_ops[0] != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR ||
       s->dynamic.vk.fsr.combiner_ops[1] != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR ||
       s->active_pipeline_ace_queries || s->active_prims_xfb_queries || s->active_emulated_prims_xfb_queries ||
       s->streamout.enabled_mask || device->sqtt.bo || device->utrace.context || device->bc250_timer.enabled ||
       radv_bc250_chain_enabled(device) || device->bc250_split_batch_prep)
      return false;
   for (unsigned i = 0; i < MAX_BIND_POINTS; i++) {
      if (state->descriptors[i].valid_heaps || state->descriptors[i].dynamic_offset_count)
         return false;
   }
   return true;
}

struct bc250_dgc_query_patch { uint64_t offset; uint32_t header, event; };
struct bc250_dgc_program {
   VkIndirectCommandsLayoutEXT layout;
   VkPipeline pipeline;
   uint64_t size;
   uint32_t sequences, draws, count;
   struct bc250_dgc_query_patch patches[];
};

static bool
bc250_dgc_record_program(struct radv_device *device, const VkGeneratedCommandsInfoEXT *info,
                         const struct bc250_dgc_shape *shape, const uint8_t *snapshot)
{
   unsigned count = 0;
   struct bc250_dgc_program *program = NULL;
   for (unsigned pass = 0; pass < 2; pass++) {
      unsigned index = 0;
      for (uint32_t seq = 0; seq < info->maxSequenceCount; seq++) {
         const uint32_t *words = (const uint32_t *)(snapshot + (uint64_t)seq * shape->stride);
         for (unsigned i = 0; i < shape->code / 4;) {
            uint32_t header = words[i];
            unsigned len = header == PKT3_NOP_PAD || header >> 30 == 2 ? 1 : ((header >> 16) & 0x3fff) + 2;
            if ((header >> 30 != 2 && header >> 30 != 3) || i + len > shape->code / 4) {
               free(program); return false;
            }
            if ((header >> 8 & 255) == PKT3_EVENT_WRITE && len > 1 &&
                ((words[i + 1] & 63) == V_028A90_PIPELINESTAT_START ||
                 (words[i + 1] & 63) == V_028A90_PIPELINESTAT_STOP)) {
               if (len != 2) { free(program); return false; }
               if (pass)
                  program->patches[index] = (struct bc250_dgc_query_patch){
                     (uint64_t)seq * shape->stride + i * 4, header, words[i + 1]};
               index++;
            }
            i += len;
         }
      }
      if (!pass) {
         count = index;
         program = malloc(sizeof(*program) + (size_t)count * sizeof(program->patches[0]));
         if (!program) return false;
      }
   }
   const VkGeneratedCommandsPipelineInfoEXT *pi = vk_find_struct_const(info->pNext, GENERATED_COMMANDS_PIPELINE_INFO_EXT);
   *program = (struct bc250_dgc_program){.layout = info->indirectCommandsLayout, .pipeline = pi->pipeline,
      .size = shape->size, .sequences = info->maxSequenceCount, .draws = info->maxDrawCount, .count = count};
   mtx_lock(&device->meta_state.mtx);
   if (!device->bc250_dgc_query_states)
      device->bc250_dgc_query_states = _mesa_hash_table_u64_create(NULL);
   bool ok = device->bc250_dgc_query_states != NULL;
   if (ok) {
      free(_mesa_hash_table_u64_search(device->bc250_dgc_query_states, info->preprocessAddress));
      _mesa_hash_table_u64_insert(device->bc250_dgc_query_states, info->preprocessAddress, program);
   }
   mtx_unlock(&device->meta_state.mtx);
   if (!ok) free(program);
   return ok;
}

static bool
bc250_dgc_patch_queries(struct radv_cmd_buffer *cmd, const VkGeneratedCommandsInfoEXT *info,
                        const struct bc250_dgc_shape *shape)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd);
   mtx_lock(&device->meta_state.mtx);
   const struct bc250_dgc_program *program = device->bc250_dgc_query_states ?
      _mesa_hash_table_u64_search(device->bc250_dgc_query_states, info->preprocessAddress) : NULL;
   const VkGeneratedCommandsPipelineInfoEXT *pi = vk_find_struct_const(info->pNext, GENERATED_COMMANDS_PIPELINE_INFO_EXT);
   bool ok = program && program->pipeline == pi->pipeline && program->layout == info->indirectCommandsLayout && program->size == shape->size &&
      program->sequences == info->maxSequenceCount && program->draws == info->maxDrawCount;
   if (ok) {
      bool native_stats = radv_get_num_pipeline_stat_queries(cmd) > 0;
      FILE *packets = NULL;
      const char *dump = debug_get_option("BC250_DGC_DUMP", NULL);
      if (dump) {
         VK_FROM_HANDLE(radv_indirect_command_layout, layout, info->indirectCommandsLayout);
         char path[4096];
         snprintf(path, sizeof(path), "%s/query-owner-%u.bin", dump, layout->vk.draw_count);
         packets = fopen(path, "wb");
      }
      for (unsigned i = 0; i < program->count; i++) {
         const struct bc250_dgc_query_patch *patch = &program->patches[i];
         /* Restore both words: inactive sequences were filled entirely with
          * NOPs by preprocessing. Their stop/start pairs contain no draw and
          * preserve the caller's query state. Helpers are never counted. */
         uint32_t words[] = {native_stats ? patch->header : PKT3(PKT3_NOP, 0, false), patch->event};
         radv_cs_write_data(device, cmd->cs, V_371_MICRO_ENGINE,
            info->preprocessAddress + patch->offset, 2, words, false);
         /* Preserve every emitted write even if command-stream growth chains
          * away the earlier IB. The publication tail is appended below. */
         if (packets) fwrite(cmd->cs->b->buf + cmd->cs->b->cdw - 6, 4, 6, packets);
      }
      if (packets) fclose(packets);
   }
   mtx_unlock(&device->meta_state.mtx);
   return ok;
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
   /* The shallow copy must not touch the owner's task tails; a capture keeps its own. */
   util_dynarray_init(&cmd->bc250_task_tails, NULL);
   cmd->bc250_dgc_tail = NULL;
   cmd->bc250_dgc_tail_va = shape->task && device->compiler_info.bc250x.task_tail ?
      info->preprocessAddress + (uint64_t)seq * shape->stride + shape->template_offset + BC250_DGC_TAIL_OFFSET : 0;
   cmd->bc250_dgc_inherit_graphics_state = true;
   /* Capture all native meta query stop/start pairs. Execution patches these
    * typed packets according to its query scope, independently of preprocessing. */
   cmd->state.active_pipeline_queries = 1;
   cmd->bc250_dgc_merged_constants = cmd->state.graphics_pipeline->base.shaders[MESA_SHADER_MESH]->info.ms.bc250_merge_k > 1;
   cmd->bc250_dgc_upload_va = info->preprocessAddress + (uint64_t)seq * shape->stride + shape->code;
   memset(&cmd->upload, 0, sizeof(cmd->upload));
   list_inithead(&cmd->upload.list);
   cmd->upload.map = snapshot + shape->code;
   cmd->upload.size = shape->data;
   unsigned app_offset = align(4 + shape->records * 12, 16);
   const struct radv_shader *fs = cmd->state.graphics_pipeline->base.shaders[MESA_SHADER_FRAGMENT];
   bool app_constants = layout->push_constant_mask ||
      (fs && (fs->info.loads_push_constants || fs->info.inline_push_constant_mask));
   if (app_constants) {
      cmd->bc250_dgc_application_va = cmd->bc250_dgc_upload_va + app_offset;
      memcpy(cmd->upload.map + app_offset, state->push_constants, MAX_PUSH_CONSTANTS_SIZE);
   }
   cmd->upload.offset = align(app_offset + (app_constants ? MAX_PUSH_CONSTANTS_SIZE : 0), 256);
   unsigned fence_offset = align(cmd->upload.offset, 8);
   cmd->gfx9_fence_va = cmd->bc250_dgc_upload_va + fence_offset;
   cmd->gfx9_fence_idx = 0;
   cmd->gfx9_eop_bug_va = 0;
   cmd->upload.offset = fence_offset + 8;
   /* Descriptor-buffer addresses are application-owned and copied with the
    * descriptor state. Push sets instead point inside the source command
    * buffer, and their VA normally points at that buffer's upload BO. Copy
    * their bytes into the final arena and rebase every bound header pointer.
    * The hybrid producer receives this same graphics state, so both compute
    * and graphics bind the frozen push-set upload, not a preparation lifetime. */
   bool descriptors_ok = true;
   for (unsigned bp = 0; bp < MAX_BIND_POINTS && descriptors_ok; bp++) {
      struct radv_descriptor_state *ds = &cmd->descriptors[bp];
      const struct radv_descriptor_set_header *original = &state->descriptors[bp].push_set.set;
      if (!original->size)
         continue;
      unsigned offset;
      descriptors_ok = radv_cmd_buffer_upload_data(cmd, original->size, original->mapped_ptr, &offset);
      if (!descriptors_ok)
         break;
      ds->push_set.set.va = cmd->bc250_dgc_upload_va + offset;
      for (unsigned set = 0; set < MAX_SETS; set++)
         if ((void *)ds->sets[set] == (const void *)original)
            ds->sets[set] = (struct radv_descriptor_set *)&ds->push_set.set;
   }
   cmd->bc250_dgc_task_uploads = NULL;
   if (shape->template_offset)
      cmd->bc250_dgc_task_uploads = calloc(1, sizeof(*cmd->bc250_dgc_task_uploads));
   descriptors_ok &= !shape->template_offset || cmd->bc250_dgc_task_uploads;
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
   bool ok = descriptors_ok && radv_create_cmd_stream(device, AMD_IP_GFX, false, &cmd->cs) == VK_SUCCESS;
   if (ok) {
      /* Each Task draw is a separate IB. No packet or local COND_EXEC spans
       * that boundary. Private constants remain unique; scratch is shared
       * using the ordinary backend's consumer drains between count draws. */
      radeon_check_space(device->ws, cmd->cs->b, (shape->task ? BC250_DGC_CODE_TASK : shape->code) / 4);
      struct radv_graphics_pipeline *pipeline = cmd->state.graphics_pipeline;
      cmd->state.graphics_pipeline = NULL;
      radv_foreach_stage(stage, RADV_GRAPHICS_STAGE_BITS)
         cmd->state.shaders[stage] = NULL;
      cmd->state.active_stages &= ~RADV_GRAPHICS_STAGE_BITS;
      radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          radv_pipeline_to_handle(&pipeline->base));
      const char *dump = debug_get_option("BC250_DGC_DUMP", NULL);
      if (dump) {
         char path[4096];
         snprintf(path, sizeof(path), "%s/bindings-%u-%u.json", dump, layout->vk.draw_count, seq);
         FILE *f = fopen(path, "w");
         if (f) {
            const struct radv_descriptor_state *ds = radv_get_descriptors_state(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
            VK_FROM_HANDLE(radv_pipeline, producer, pipeline->bc250_task_pipeline);
            const struct radv_shader *shaders[] = {pipeline->base.shaders[MESA_SHADER_MESH],
               pipeline->base.shaders[MESA_SHADER_FRAGMENT], producer ? producer->shaders[MESA_SHADER_COMPUTE] : NULL};
            fprintf(f, "[");
            bool first = true;
            for (unsigned stage = 0; stage < ARRAY_SIZE(shaders); stage++) {
               const struct radv_shader *shader = shaders[stage];
               if (!shader)
                  continue;
               u_foreach_bit(set, shader->info.user_sgprs_locs.descriptor_sets_enabled & ds->valid) {
                  uint64_t va = ds->sets[set] ? ds->sets[set]->header.va : ds->descriptor_buffers[set];
                  unsigned reg = shader->info.user_data_0 + shader->info.user_sgprs_locs.descriptor_sets[set].sgpr_idx * 4;
                  fprintf(f, "%s{\"stage\":%u,\"set\":%u,\"reg\":%u,\"va\":%llu}",
                     first ? "" : ",", stage, set, reg, (unsigned long long)va);
                  first = false;
               }
            }
            fprintf(f, "]\n");
            fclose(f);
         }
         snprintf(path, sizeof(path), "%s/constants-%u-%u.json", dump, layout->vk.draw_count, seq);
         f = fopen(path, "w");
         const struct radv_shader *fragment = pipeline->base.shaders[MESA_SHADER_FRAGMENT];
         if (f) {
            fprintf(f, "{\"app\":%llu,\"fragment_inline\":%u,\"fragment_mask\":%llu,\"fragment_pointer\":%u}\n",
               (unsigned long long)cmd->bc250_dgc_application_va,
               fragment ? radv_get_user_sgpr_loc(fragment, AC_UD_INLINE_PUSH_CONSTANTS) : 0,
               (unsigned long long)(fragment ? fragment->info.inline_push_constant_mask : 0),
               fragment ? radv_get_user_sgpr_loc(fragment, AC_UD_PUSH_CONSTANTS) : 0);
            fclose(f);
         }
      }
      const uint32_t *capture_buf = cmd->cs->b->buf;
      uint64_t records = cmd->bc250_dgc_upload_va + 4;
      uint64_t count = cmd->bc250_dgc_upload_va;
      uint64_t scratch = 0;
      {
         unsigned template_code = shape->task ? BC250_DGC_CODE_TASK : shape->code;
         if (shape->task) {
            unsigned draw_id = layout->vk.draw_count ? 0 : seq;
            radv_bc250_draw_task_dgc(cmd, records + (uint64_t)draw_id * 12, count, draw_id, &scratch);
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
              cmd->cs->b->cdw * 4 <= (cmd->bc250_dgc_tail ? BC250_DGC_TAIL_OFFSET : template_code) &&
              (!cmd->bc250_dgc_tail_va || cmd->bc250_dgc_tail) &&
              (!cmd->bc250_dgc_task_uploads || !cmd->bc250_dgc_task_uploads->overflow) &&
              cmd->upload.offset <= shape->data && !cmd->gang.cs;
         if (!ok && dump)
            fprintf(stderr, "BC250 DGC capture refused: words=%u code=%u uploads=%u data=%u sites=%u overflow=%u moved=%u\n",
               cmd->cs->b->cdw, template_code, cmd->upload.offset, shape->data,
               cmd->bc250_dgc_task_uploads ? cmd->bc250_dgc_task_uploads->count : 0,
               cmd->bc250_dgc_task_uploads ? cmd->bc250_dgc_task_uploads->overflow : 0,
               cmd->cs->b->buf != capture_buf);
         if (ok) {
            uint8_t *program = snapshot + shape->template_offset;
            memcpy(program, cmd->cs->b->buf, cmd->cs->b->cdw * 4);
            uint32_t *padding = (uint32_t *)program;
            for (uint32_t i = cmd->cs->b->cdw; i < template_code / 4; i++)
               padding[i] = PKT3_NOP_PAD;
            /* RADV_BC250_TASK_TAIL: the chunk slots after the first, reached by one CHAIN. */
            if (cmd->bc250_dgc_tail) {
               memcpy(program + BC250_DGC_TAIL_OFFSET, cmd->bc250_dgc_tail->b->buf, cmd->bc250_dgc_tail->b->cdw * 4);
               device->ws->cs_execute_secondary(owner->cs->b, cmd->bc250_dgc_tail->b, false);
            }
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
         cmd->cs->b->cdw = 0;
      }
      if (ok && shape->template_offset) {
         ok = cmd->bc250_dgc_task_uploads->count >= 1025;
         for (unsigned u = 0; u < cmd->bc250_dgc_task_uploads->count && ok; u++) {
            unsigned offset = cmd->bc250_dgc_task_uploads->offsets[u];
            const uint32_t *pc = (const uint32_t *)(snapshot + shape->code + offset);
            ok = offset + 72 <= shape->data && pc[5] == 0 &&
                 ((uint64_t)pc[8] | (uint64_t)pc[9] << 32) == records &&
                 ((uint64_t)pc[10] | (uint64_t)pc[11] << 32) == count;
         }
         /* Each patch is bounded and calls the same immutable ordinary IB.
          * A full drain precedes writes into constants read by the last draw;
          * publish CP writes before either inline SGPR loads or shader loads.
          * Descriptor tables, application constants and chunk coordinates
          * keep their captured values. Only DrawID and input change. */
         for (unsigned draw = 0; draw < shape->segments && ok; draw++) {
            cmd->cs->b->cdw = 0;
            cmd->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_VS_PARTIAL_FLUSH |
               RADV_CMD_FLAG_PS_PARTIAL_FLUSH | RADV_CMD_FLAG_WB_L2;
            radv_emit_cache_flush(cmd);
            for (unsigned u = 0; u < cmd->bc250_dgc_task_uploads->count; u++) {
               unsigned offset = cmd->bc250_dgc_task_uploads->offsets[u];
               const uint32_t *original = (const uint32_t *)(snapshot + shape->code + offset);
               uint64_t input = records + (uint64_t)draw * 12;
               uint32_t patch[] = {draw, original[6], original[7], input, input >> 32};
               radv_cs_write_data(device, cmd->cs, V_371_MICRO_ENGINE,
                  cmd->bc250_dgc_upload_va + offset + 20, ARRAY_SIZE(patch), patch, false);
            }
            cmd->state.flush_bits |= RADV_CMD_FLAG_INV_L2 | RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_VCACHE;
            radv_emit_cache_flush(cmd);
            ac_emit_cp_pfp_sync_me(cmd->cs->b, false);
            /* Tail-chain at IB2 rather than nesting an unsupported IB3. The
             * original IB1 return address is retained by CHAIN. */
            ac_emit_cp_indirect_buffer(cmd->cs->b, info->preprocessAddress + (uint64_t)seq * shape->stride +
               shape->template_offset, BC250_DGC_CODE_TASK / 4, AC_CP_INDIRECT_BUFFER_CHAIN, false);
            ok = cmd->cs->b->buf == capture_buf && cmd->cs->b->cdw * 4 <= shape->segment_code &&
                 !vk_command_buffer_has_error(&cmd->vk);
            if (ok) {
               uint32_t *program = (uint32_t *)(snapshot + (uint64_t)draw * shape->segment_code);
               memcpy(program, capture_buf, cmd->cs->b->cdw * 4);
               for (unsigned i = cmd->cs->b->cdw; i < shape->segment_code / 4; i++)
                  program[i] = PKT3_NOP_PAD;
            }
         }
         if (dump) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/task-uploads-%u.bin", dump, seq);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(cmd->bc250_dgc_task_uploads->offsets, sizeof(uint32_t), cmd->bc250_dgc_task_uploads->count, f); fclose(f); }
         }
      }
      radv_destroy_cmd_stream(device, cmd->cs);
   }
   if (cmd->bc250_dgc_tail)
      radv_destroy_cmd_stream(device, cmd->bc250_dgc_tail);
   _mesa_set_fini(&cmd->vs_prologs, NULL);
   _mesa_set_fini(&cmd->ps_epilogs, NULL);
   free(cmd->bc250_dgc_task_uploads);
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
       cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
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
   bool recorded = bc250_dgc_record_program(radv_cmd_buffer_device(cmd), info, &shape, snapshot);
   free(snapshot);
   if (!recorded) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   cmd->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_VS_PARTIAL_FLUSH |
                            RADV_CMD_FLAG_PS_PARTIAL_FLUSH | RADV_CMD_FLAG_WB_L2;
   radv_meta_begin(cmd);
   radv_meta_bind_compute_pipeline(cmd, layout->pipeline);
   radv_meta_push_constants(cmd, layout->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
   bool conditional = cmd->state.cond_render.enabled;
   cmd->state.cond_render.enabled = false;
   radv_CmdDispatchBase(radv_cmd_buffer_to_handle(cmd), 0, 0, 0, info->maxSequenceCount, 1, 1);
   cmd->state.cond_render.enabled = conditional;
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
   if (!info->maxSequenceCount)
      return;
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
   if (!bc250_dgc_patch_queries(cmd, info, &shape)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   cmd->state.flush_bits |= RADV_CMD_FLAG_WB_L2 | RADV_CMD_FLAG_INV_L2;
   radv_emit_cache_flush(cmd);
   ac_emit_cp_pfp_sync_me(cmd->cs->b, false);
   const char *dump = debug_get_option("BC250_DGC_DUMP", NULL);
   if (dump) {
      VK_FROM_HANDLE(radv_indirect_command_layout, layout, info->indirectCommandsLayout);
      char path[4096];
      snprintf(path, sizeof(path), "%s/query-owner-%u.bin", dump, layout->vk.draw_count);
      FILE *f = fopen(path, "ab");
      if (f) { fwrite(cmd->cs->b->buf, 4, cmd->cs->b->cdw, f); fclose(f); }
   }
   for (uint32_t seq = 0; seq < info->maxSequenceCount; seq++) {
      for (uint32_t segment = 0; segment < shape.segments; segment++) {
         radeon_check_space(device->ws, cmd->cs->b, 4);
         device->ws->cs_chain_dgc_ib(cmd->cs->b, info->preprocessAddress + (uint64_t)seq * shape.stride +
                                    (uint64_t)segment * shape.segment_code, shape.segment_code / 4, 0,
                                    cmd->state.cond_render.enabled);
      }
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

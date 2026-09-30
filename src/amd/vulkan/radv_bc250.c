/* SPDX-License-Identifier: MIT
 * Experimental BC250 TASK scheduling on the graphics queue.
 * No ACE queue, task rings, native TASK packets, or device-generation spoofing.
 */
#include "radv_bc250.h"
#include "radv_descriptor_set.h"
#include "meta/radv_meta.h"
#include "nir/nir_builder.h"
#include "nir/nir_serialize.h"
#include "nir/radv_nir.h"
#include "radv_pipeline_graphics.h"
#include "radv_pipeline_cache.h"
#include "radv_constants.h"
#include "radv_cs.h"
#include "tools/radv_rmv.h"
#include "vk_shader_module.h"

#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_dynarray.h"
#include <math.h>
#include <stdarg.h>
#include <unistd.h>

void
radv_bc250_device_env_init(struct radv_device *device, const struct radv_physical_device *pdev)
{
   struct radv_bc250_device_env *env = &device->bc250_env;
   env->chain_trace = pdev->info.family == CHIP_GFX1013 && debug_get_bool_option("BC250_CHAIN_TRACE", false);
   env->chain_shader_only = debug_get_bool_option("BC250_CHAIN_SHADER_ONLY", false);
   env->chain_arguments_only = debug_get_bool_option("BC250_CHAIN_ARGUMENTS_ONLY", false);
   env->chain_sample_output = debug_get_bool_option("BC250_CHAIN_SAMPLE_OUTPUT", false);
   env->transient_arena = debug_get_bool_option("BC250_TRANSIENT_ARENA", false);
   env->trace_compile = getenv("BC250_TRACE_COMPILE") != NULL;
   env->trace_regs = getenv("BC250_TRACE_REGS") != NULL;
   env->trace_usage = getenv("BC250_TRACE_USAGE") != NULL;
   env->omit_launch = getenv("BC250_DIAGNOSTIC_OMIT_LAUNCH") != NULL;
   env->post_mesh_vgt_flush = debug_get_bool_option("BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH", false);
   env->skip_inactive_chunks = debug_get_bool_option("BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS", false);
   env->no_split_l2_inv = debug_get_bool_option("RADV_BC250_PERF_NO_SPLIT_L2_INV", false);
   env->split_batch_trace = debug_get_bool_option("RADV_BC250_SPLIT_BATCH_TRACE", false);
   env->merge_prep_l2_inv = debug_get_bool_option("RADV_BC250_MESH_MERGE_PREP_L2_INV", false);
   env->native_task_direct = debug_get_bool_option("BC250_NATIVE_TASK_DIRECT", false) ||
                             debug_get_bool_option("RADV_BC250_NATIVE_TASK", false);
   env->native_task_indirect = debug_get_bool_option("BC250_NATIVE_TASK_INDIRECT", false) ||
                               debug_get_bool_option("RADV_BC250_NATIVE_TASK", false);
   env->scratch_reuse = pdev->bc250_native_mesh && debug_get_bool_option("RADV_BC250_SCRATCH_REUSE", false);
   env->split_lean_setup = pdev->bc250_native_mesh && debug_get_bool_option("RADV_BC250_SPLIT_LEAN_SETUP", false);
   env->split_prep_free = pdev->bc250_native_mesh && debug_get_bool_option("RADV_BC250_SPLIT_PREP_FREE", false);
   env->mesh_no_split = pdev->bc250_native_mesh && debug_get_bool_option("RADV_BC250_MESH_NO_SPLIT", false);
   /* Submission-cost switches (GFX1013 only, default off). */
   const bool bc250 = pdev->info.family == CHIP_GFX1013;
   env->local_bos = bc250 && debug_get_bool_option("RADV_BC250_LOCAL_BOS", false);
   env->submit_profile = bc250 && debug_get_bool_option("RADV_BC250_SUBMIT_PROFILE", false);
   device->vk.bc250_submit_profile = env->submit_profile;
   device->vk.bc250_known_signals = bc250 && debug_get_bool_option("RADV_BC250_SUBMIT_KNOWN_SIGNALS", false);
   if (env->submit_profile) {
      radv_bc250_ws_prof_enabled = true;
      simple_mtx_init(&device->bc250_prof_mtx, mtx_plain);
      device->bc250_prof_start_ns = device->bc250_prof_last_ns = os_time_get_nano();
      const char *path = getenv("RADV_BC250_SUBMIT_PROFILE_FILE");
      if (path && path[0]) {
         device->bc250_prof_file = fopen(path, "a");
         if (!device->bc250_prof_file)
            fprintf(stderr, "radv/bc250: cannot open RADV_BC250_SUBMIT_PROFILE_FILE %s\n", path);
      }
   }
   if (env->local_bos || env->submit_profile || device->vk.bc250_known_signals)
      fprintf(stderr, "radv/bc250: submission switches local_bos=%d known_signals=%d submit_profile=%d\n",
              env->local_bos, device->vk.bc250_known_signals, env->submit_profile);
}

/* Report final shaders, including cache/binary imports. This is an observer:
 * neither admission, shader keys nor command recording depend on this switch. */
void
radv_bc250_report_mesh_route(const struct radv_device *device, const struct radv_shader *shader,
                              const char *object, unsigned pieces, bool task, bool ordered)
{
   if (!device->bc250_env.mesh_no_split || !shader || shader->info.stage != MESA_SHADER_MESH)
      return;
   const struct radv_shader_info *info = &shader->info;
   const bool safe = info->ms.bc250_safe_direct && !ordered;
   static const char *const reasons[] = {
      [RADV_BC250_ROUTE_REASON_UNKNOWN] = "route_policy_or_shader_object",
      [RADV_BC250_ROUTE_REASON_SWITCHES_OFF] = "safe_direct_switches_off",
      [RADV_BC250_ROUTE_REASON_TASK_STAGE] = "task_stage_not_admitted",
      [RADV_BC250_ROUTE_REASON_MULTIVIEW] = "multiview_not_admitted",
      [RADV_BC250_ROUTE_REASON_VRS] = "vrs_may_be_enabled",
      [RADV_BC250_ROUTE_REASON_NO_FRAGMENT] = "no_linked_fragment_shader",
      [RADV_BC250_ROUTE_REASON_PER_PRIMITIVE_OUTPUT] = "per_primitive_output_not_owned",
      [RADV_BC250_ROUTE_REASON_TOPOLOGY] = "non_triangle_topology",
      [RADV_BC250_ROUTE_REASON_API_BOUND] = "api_shape_out_of_range",
      [RADV_BC250_ROUTE_REASON_WORKGROUP_BOUND] = "workgroup_bound_unproven",
      [RADV_BC250_ROUTE_REASON_MESH_IO] = "mesh_output_read_or_16bit_io",
      [RADV_BC250_ROUTE_REASON_VARYING_LAYOUT] = "varying_layout_or_count_unproven",
      [RADV_BC250_ROUTE_REASON_SHARED_MEMORY] = "shared_memory_budget",
      [RADV_BC250_ROUTE_REASON_FRAGMENT_ABI] = "fragment_interface_not_admitted",
      [RADV_BC250_ROUTE_REASON_LAUNCH_BOUND] = "closure_bound_exceeds_256",
      [RADV_BC250_ROUTE_REASON_CANDIDATE_UNPROVEN] = "safe_candidate_not_admitted",
      [RADV_BC250_ROUTE_REASON_CLIPCULL_EXPORT] = "clip_cull_distance_export_needs_POS1",
      [RADV_BC250_ROUTE_REASON_BARY_REFERENCE] = "bary_two_reference_parameters_unavailable",
      [RADV_BC250_ROUTE_REASON_BARY_PRIVATE] = "bary_private_corner_shape_or_budget_unproven",
      [RADV_BC250_ROUTE_REASON_BARY_COST] = "bary_small_class_cost_not_better_than_split",
   };
   const unsigned reason_id = info->ms.bc250_route_reason;
   const char *reason = ordered ? "ordered_materialization_route" : safe ? "none" :
      reason_id < ARRAY_SIZE(reasons) && reasons[reason_id] ? reasons[reason_id] : "unclassified";
   char shader_hash[17];
   for (unsigned i = 0; i < 8; i++)
      snprintf(shader_hash + 2 * i, 3, "%02x", shader->hash[i]);
   fprintf(stderr, "BC250 NO_SPLIT: object=%s fallback_needed=%u route=%s reason=%s V=%u P=%u lanes=%u "
                   "pieces=%u task_transport=%u scratch=%u fallback_retained=1 shader_hash=%s\n",
           object, !safe, ordered ? "ordered_materialization" : safe ? (pieces ? "safe_direct_pieces" : "safe_direct") :
           pieces ? "split" : info->ms.bc250_expanded ? "expansion" : "raw_unproven", reason,
           info->ms.bc250_api_vertices, info->ms.bc250_api_primitives, info->workgroup_size,
           pieces, task || info->ms.has_task, info->ms.needs_ms_scratch_ring, shader_hash);
}

bool
radv_bc250_chain_enabled(const struct radv_device *device)
{
   return device->bc250_env.chain_trace;
}

uint64_t
radv_bc250_chain_event(const struct radv_device *device, const char *event, const char *format, ...)
{
   if (!radv_bc250_chain_enabled(device))
      return 0;
   /* Allocation-only diagnostic: no command/descriptor/packet history.
    * Filter before IDs and budgets so shader lifetime records remain complete. */
   if (device->bc250_env.chain_shader_only && strncmp(event, "SHADER_", 7))
      return 0;
   if (device->bc250_env.chain_arguments_only &&
       strncmp(event, "SHADER_", 7) && strncmp(event, "TARGET_QUEUE_", 13) &&
       strcmp(event, "CMD_BEGIN") && strcmp(event, "CMD_STORAGE_RELEASE_CPU") &&
       strcmp(event, "CHAIN_BEGIN") && strcmp(event, "OUTPUT_CPU_MAPPING") &&
       strcmp(event, "OUTPUT_AT_RELEASE_CPU") && strcmp(event, "HELPER_RECORD") &&
       strcmp(event, "HELPER_ARGUMENT_BYTES_CPU") && strcmp(event, "CONSUME_RECORD"))
      return 0;
   static uint64_t serial, class_count[3];
   /* Routine queue traffic must not exhaust shader lifetime/target records.
    * A truncated category remains incomplete even while others continue. */
   unsigned category = (!strncmp(event, "SHADER_", 7) || !strcmp(event, "CMD_BEGIN")) ? 0 :
      (!strncmp(event, "QUEUE_", 6) || !strncmp(event, "CMD_", 4)) ? 1 : 2;
   uint64_t ordinal = p_atomic_inc_return(&class_count[category]);
   if (ordinal > 200000)
      return 0;
   uint64_t id = p_atomic_inc_return(&serial);
   struct radv_device *mutable_device = (struct radv_device *)device;
   p_atomic_cmpxchg(&mutable_device->bc250_trace_device_id, 0, id);
   uint64_t device_id = p_atomic_read(&mutable_device->bc250_trace_device_id);
   char line[3072];
   int n = snprintf(line, sizeof(line), "BC250CHAIN pid=%u device=%p device_generation=%llu id=%llu ns=%llu category=%u event=%s ",
                    (unsigned)getpid(), (const void *)device, (unsigned long long)device_id, (unsigned long long)id,
                    (unsigned long long)os_time_get_nano(), category, ordinal == 200000 ? "TRUNCATED" : event);
   va_list ap;
   va_start(ap, format);
   int m = vsnprintf(line + n, sizeof(line) - n - 1, format, ap);
   va_end(ap);
   if (m < 0 || (size_t)m >= sizeof(line) - n - 1)
      n += snprintf(line + n, sizeof(line) - n - 1, "RECORD_TOO_LONG");
   else
      n += m;
   line[n++] = '\n';
   /* CPU metadata only. A short/failed write leaves an unavailable event ID. */
   return write(STDERR_FILENO, line, n) == n && ordinal != 200000 ? id : 0;
}

void
radv_bc250_chain_state(struct radv_cmd_buffer *cmd, const char *event, const struct radv_shader *shader)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd);
   if (device->bc250_env.chain_arguments_only)
      return;
   if (!radv_bc250_chain_enabled(device) || !shader)
      return;
   radv_bc250_chain_event(device, event,
      "cmd=%p epoch=%llu chain=%llu producer=%llu cs=%p buf=%p cdw=%u shader=%p allocation=%llu upload=%llu va=%llx code=%s stage=%u flush_requested=%llx recording_only=1",
      (void *)cmd, (unsigned long long)cmd->bc250_trace_epoch, (unsigned long long)cmd->bc250_trace_chain,
      (unsigned long long)cmd->bc250_trace_producer, (void *)cmd->cs->b, (void *)cmd->cs->b->buf, cmd->cs->b->cdw,
      (const void *)shader, (unsigned long long)(shader->alloc ? shader->alloc->bc250_trace_allocation : 0),
      (unsigned long long)shader->bc250_trace_upload, (unsigned long long)radv_shader_get_va(shader),
      shader->bc250_trace_code, shader->info.stage, (unsigned long long)cmd->state.flush_bits);
   if (strcmp(event, "MESH_IDENTITY") && strcmp(event, "HELPER_BEFORE"))
      return;
   struct radv_descriptor_state *ds = radv_get_descriptors_state(cmd,
      shader->info.stage == MESA_SHADER_COMPUTE ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS);
   for (unsigned i = 0; i < MAX_SETS; ++i) {
      if (!(ds->valid & (1u << i)))
         continue;
      radv_bc250_chain_event(device, "DESCRIPTOR_IDENTITY",
         "cmd=%p epoch=%llu chain=%llu stage=%u set=%u object=%p va=%llx buffer_va=%llx contents_and_version=UNOBSERVED",
         (void *)cmd, (unsigned long long)cmd->bc250_trace_epoch, (unsigned long long)cmd->bc250_trace_chain,
         shader->info.stage, i, (void *)ds->sets[i],
         (unsigned long long)(ds->sets[i] ? ds->sets[i]->header.va : 0),
         (unsigned long long)ds->descriptor_buffers[i]);
   }
}

static void
bc250_chain_begin(struct radv_cmd_buffer *cmd, const char *kind, uint64_t input, uint64_t count,
                  uint32_t records, uint32_t stride)
{
   const struct radv_shader *mesh = cmd->state.shaders[MESA_SHADER_MESH];
   cmd->bc250_trace_chain = cmd->bc250_trace_producer = 0;
   if (!radv_bc250_chain_enabled(radv_cmd_buffer_device(cmd)) || !mesh || !mesh->bc250_trace_target)
      return;
   cmd->bc250_trace_has_target = true;
   cmd->bc250_trace_start_buf = cmd->cs->b->buf;
   cmd->bc250_trace_start_dw = cmd->cs->b->cdw;
   cmd->bc250_trace_chain = radv_bc250_chain_event(radv_cmd_buffer_device(cmd), "CHAIN_BEGIN",
      "cmd=%p epoch=%llu kind=%s input=%llx count=%llx records=%u stride=%u external_input_generation=UNKNOWN",
      (void *)cmd, (unsigned long long)cmd->bc250_trace_epoch, kind,
      (unsigned long long)input, (unsigned long long)count, records, stride);
   radv_bc250_chain_state(cmd, "MESH_IDENTITY", mesh);
}


void
radv_bc250_chain_packets(struct radv_cmd_buffer *cmd)
{
   /* The cheap guards first: bc250_trace_chain is only set while chain tracing. */
   if (!cmd->bc250_inside_mesh_draw || !cmd->bc250_trace_chain)
      return;
   if (radv_cmd_buffer_device(cmd)->bc250_env.chain_arguments_only)
      return;
   uint32_t begin = cmd->bc250_trace_start_dw, end = cmd->cs->b->cdw;
   if (cmd->bc250_trace_start_buf != cmd->cs->b->buf || end < begin) {
      radv_bc250_chain_event(radv_cmd_buffer_device(cmd), "CHAIN_PACKETS_UNAVAILABLE",
         "cmd=%p epoch=%llu chain=%llu reason=BUFFER_CHANGED",
         (void *)cmd, (unsigned long long)cmd->bc250_trace_epoch,
         (unsigned long long)cmd->bc250_trace_chain);
      return;
   }
   uint32_t stop = MIN2(end, begin + 1024);
   for (uint32_t offset = begin; offset < stop; offset += 128) {
      unsigned count = MIN2(128, stop - offset);
      char words[128 * 9 + 1];
      for (unsigned i = 0; i < count; ++i)
         snprintf(words + 9 * i, 10, "%08x,", cmd->cs->b->buf[offset + i]);
      radv_bc250_chain_event(radv_cmd_buffer_device(cmd), "CHAIN_PACKETS_CPU",
         "cmd=%p epoch=%llu chain=%llu cs=%p buf=%p begin=%u end=%u offset=%u count=%u clipped=%u words=%s gpu_execution=UNOBSERVED",
         (void *)cmd, (unsigned long long)cmd->bc250_trace_epoch,
         (unsigned long long)cmd->bc250_trace_chain, (void *)cmd->cs->b,
         (void *)cmd->cs->b->buf, begin, end, offset, count, stop != end, words);
   }
}

static void
bc250_chain_dispatch(struct radv_cmd_buffer *cmd, const struct radv_dispatch_info *dispatch, const char *role)
{
   if (cmd->bc250_trace_chain) {
      cmd->bc250_trace_producer = radv_bc250_chain_event(radv_cmd_buffer_device(cmd), "HELPER_RECORD",
         "cmd=%p epoch=%llu chain=%llu role=%s indirect=%llx x=%u y=%u z=%u generation_is_recorded_write_intent=1",
         (void *)cmd, (unsigned long long)cmd->bc250_trace_epoch, (unsigned long long)cmd->bc250_trace_chain,
         role, (unsigned long long)dispatch->indirect_va, dispatch->blocks[0], dispatch->blocks[1], dispatch->blocks[2]);
      unsigned argument_bytes = !strcmp(role, "split_arguments") ? 36 : 68;
      char push_hex[68 * 2 + 1];
      for (unsigned i = 0; i < argument_bytes; ++i)
         snprintf(push_hex + 2 * i, 3, "%02x", ((const unsigned char *)cmd->push_constants)[i]);
      radv_bc250_chain_event(radv_cmd_buffer_device(cmd), "HELPER_ARGUMENT_BYTES_CPU",
         "cmd=%p epoch=%llu chain=%llu producer=%llu bytes=%u hex=%s gpu_values=UNOBSERVED",
         (void *)cmd, (unsigned long long)cmd->bc250_trace_epoch,
         (unsigned long long)cmd->bc250_trace_chain, (unsigned long long)cmd->bc250_trace_producer,
         argument_bytes, push_hex);
      radv_bc250_chain_state(cmd, "HELPER_BEFORE", cmd->state.shaders[MESA_SHADER_COMPUTE]);
   }
   radv_compute_dispatch(cmd, dispatch);
   if (cmd->bc250_trace_chain)
      radv_bc250_chain_state(cmd, "HELPER_PACKETS_RECORDED", cmd->state.shaders[MESA_SHADER_COMPUTE]);
}

static void
bc250_chain_consume(struct radv_cmd_buffer *cmd, uint64_t records, uint64_t count, uint64_t payload,
                    uint32_t maximum, uint32_t stride, uint32_t chunk)
{
   if (!cmd->bc250_trace_chain)
      return;
   radv_bc250_chain_event(radv_cmd_buffer_device(cmd), "CONSUME_RECORD",
      "cmd=%p epoch=%llu chain=%llu expected_helper=%llu records=%llx count=%llx payload=%llx maximum=%u stride=%u chunk=%u gpu_completion=UNOBSERVED",
      (void *)cmd, (unsigned long long)cmd->bc250_trace_epoch, (unsigned long long)cmd->bc250_trace_chain,
      (unsigned long long)cmd->bc250_trace_producer, (unsigned long long)records, (unsigned long long)count,
      (unsigned long long)payload, maximum, stride, chunk);
   radv_bc250_chain_state(cmd, "CONSUMER_BEFORE", cmd->state.shaders[MESA_SHADER_MESH]);
}




/* Experimental parameter-cache fallback. Give every primitive private vertices
 * so flat vertex attributes can represent arbitrary primitive attributes even
 * when the application's vertices are shared. The original program writes a
 * shared staging array; an epilogue copies only referenced vertices.
 */
static void
bc250_shared_type(const struct glsl_type *type, unsigned *size, unsigned *alignment)
{
   unsigned component = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   *size = component * glsl_get_vector_elements(type);
   *alignment = component;
}

/* True when the mesh shader declares per-primitive generic attributes that can
 * only be represented through the expansion epilogue below. Used to reject such
 * pipelines early at creation when expansion cannot run, instead of leaving raw
 * per-primitive output intrinsics for later NIR passes to assert on and crash in. */
bool
radv_bc250_mesh_needs_expansion(nir_shader *mesh)
{
   const uint64_t generic = mesh->info.per_primitive_outputs &
      ((UINT64_C(0xffffffff) << VARYING_SLOT_VAR0) | VARYING_BIT_PRIMITIVE_ID |
       VARYING_BIT_VIEWPORT | VARYING_BIT_LAYER);
   return generic != 0;
}

/* Plain triangles with a proven latest-copy/closure bound <=256 export slots. The NGG epilogue removes unreferenced
 * vertices before allocation; admission alone never makes raw connectivity safe.
 * The conservative storage bound keeps remapped outputs in LDS, not the scratch
 * ring. Task, multiview, split and linked-FS exclusions are checked by the caller.
 */
static bool
bc250_safe_direct_candidate(nir_shader *mesh, bool parallel, unsigned max_slots, bool private_bary)
{
   if (!mesh || mesh->info.stage != MESA_SHADER_MESH || mesh->info.mesh.nv ||
       mesh->info.mesh.primitive_type != MESA_PRIM_TRIANGLES || mesh->info.task_payload_size ||
       !mesh->info.mesh.max_vertices_out || mesh->info.mesh.max_vertices_out > 256 ||
       !mesh->info.mesh.max_primitives_out || mesh->info.mesh.max_primitives_out > 256 ||
       mesh->info.workgroup_size_variable || mesh->info.outputs_read || mesh->info.outputs_written_16bit ||
       mesh->info.workgroup_size[0] * mesh->info.workgroup_size[1] * mesh->info.workgroup_size[2] > 256 ||
       (!private_bary && mesh->info.shared_size > 16 * 1024))
      return false;

   /* The first gate uses ordinary 32-bit scalar/vector/matrix varyings. Reject
    * narrow/wide and aggregate generic I/O until separately covered by the oracle. */
   nir_foreach_variable_with_modes(var, mesh, nir_var_shader_out) {
      if (var->data.location < VARYING_SLOT_VAR0)
         continue;
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!(glsl_type_is_scalar(type) || glsl_type_is_vector(type) || glsl_type_is_matrix(type)) ||
          glsl_get_bit_size(type) != 32)
         return false;
   }

   const unsigned v = mesh->info.mesh.max_vertices_out, p = mesh->info.mesh.max_primitives_out;
   parallel &= p <= 85; /* Larger small-V classes retain the serial specialization. */
   const unsigned bound = v <= 32 ? MIN2(v, 3 * p) : 3 * p;
   if (bound > 256)
      return false;

   const uint64_t special = VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_PRIMITIVE_COUNT;
   /* PrimitiveId is accepted only after owned-corner lowering has converted the
    * per-primitive value to flat per-vertex data. A raw per-primitive export is
    * still rejected by the check below. */
   const uint64_t primitive_id_vertex =
      (mesh->info.outputs_written & VARYING_BIT_PRIMITIVE_ID) &&
      !(mesh->info.per_primitive_outputs & VARYING_BIT_PRIMITIVE_ID) ? VARYING_BIT_PRIMITIVE_ID : 0;
   const uint64_t vertex = mesh->info.outputs_written & ~(special | primitive_id_vertex);
   const uint64_t allowed = VARYING_BIT_POS | primitive_id_vertex | (private_bary ? VARYING_BIT_PSIZ : 0) |
                            (UINT64_C(0xffffffff) << VARYING_SLOT_VAR0);
   /* Conservative LDS bound includes alignment, original outputs/indices,
    * latest[V], inverse[bound], remapped indices and count publication. Staying
    * below 30 KiB also prevents the general layout from spilling outputs. */
   unsigned vertex_stride = 16 * util_bitcount64(vertex);
   if (private_bary) {
      /* Expansion writes each scalar/vector as one constant-slot output.
       * SAFE_CORNERS enables the existing packed LDS layout. Bound its highest
       * component per location, including holes; no direct-read success is
       * assumed. Matrices/aggregate layouts keep the fallback. */
      uint8_t components[VARYING_SLOT_MAX] = {0};
      nir_foreach_shader_out_variable(var, mesh) {
         if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
            continue;
         const struct glsl_type *type = glsl_get_array_element(var->type);
         if (!glsl_type_is_vector_or_scalar(type) || glsl_get_bit_size(type) != 32 ||
             var->data.location_frac + glsl_get_vector_elements(type) > 4)
            return false;
         components[var->data.location] = MAX2(components[var->data.location],
            var->data.location_frac + glsl_get_vector_elements(type));
      }
      vertex_stride = 0;
      for (unsigned i = 0; i < VARYING_SLOT_MAX; i++)
         vertex_stride += 4 * components[i];
   }
   /* Private corners need only the inverse map and two published counts, not
    * latest[V] or remapped connectivity. 128 + 6P + bound also covers every
    * alignment, workgroup info and original index byte. */
   const unsigned lds_bound = align(mesh->info.shared_size, 16) + 128 + vertex_stride * v + 6 * p + bound +
      (private_bary ? 0 : parallel ? MAX2(4 * v, align(3 * p, 4) + 32) : 4 * v);
   if (private_bary && getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH SAFE BARY: LDS bound=%u shared=%u slots=%u V=%u P=%u\n",
              lds_bound, mesh->info.shared_size, util_bitcount64(vertex), v, p);
   if (lds_bound >= (private_bary ? 32 : 30) * 1024)
      return false;
   return (vertex & VARYING_BIT_POS) && !(vertex & ~allowed) &&
          !(mesh->info.per_primitive_outputs & ~special) && util_bitcount64(vertex) <= max_slots;
}

bool
radv_bc250_mesh_safe_direct_candidate(nir_shader *mesh, bool parallel)
{
   return bc250_safe_direct_candidate(mesh, parallel, 8, false);
}

bool
radv_bc250_mesh_safe_direct_candidate_slots(nir_shader *mesh, bool parallel, unsigned max_slots)
{
   return bc250_safe_direct_candidate(mesh, parallel, max_slots, false);
}

/* RADV_BC250_MESH_AMD (opt-in, keyed as compiler_info->key.bc250_mesh_amd):
 * run a Mesh workgroup as ONE unsplit NGG subgroup the way AMD's LLPC does on
 * GFX10.3 (PRIM_AMP_FACTOR = THDS_PER_SUBGRP = max(API threads, V, P), no
 * split, no private-vertex expansion, VGT_REUSE_OFF = 1). Only Mesh shaders
 * whose outputs need none of the base driver's rewrites are eligible: no Task shader, no
 * per-primitive generic/PrimitiveId/Layer/Viewport output, no CullPrimitive
 * and no multiview. Everything else keeps the base driver's route unchanged. The decision
 * is a pure function of the Mesh NIR as translated from SPIR-V, so pipeline
 * preparation and compilation agree.
 */
bool
radv_bc250_mesh_amd_route(bool enabled, bool fast, bool merged, nir_shader *mesh, bool has_task, bool multiview,
                          bool allow_clipcull)
{
   /* A shader rewritten by radv_bc250_merge_mesh must take the raw route: its
    * K*P >= 65 triangles, hole primitives and sink vertices are exported as
    * they are (no split, no private-vertex expansion). The merge planner
    * already required everything the checks below test. */
   if (merged)
      return true;
   /* RADV_BC250_MESH_FAST admits only triangle meshes declaring more than 64
    * primitives (the hardware-passed class, MESH_PERF/amdmode/GATES.md). */
   if (!enabled && fast && mesh)
      enabled = mesh->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
                mesh->info.mesh.max_primitives_out > 64;
   /* Clip/cull distances (left after RADV_BC250_MESH_CLIPCULL_CONST): the raw route would export
    * them as POS1, which is refused until hardware-proven (radv_bc250_mesh_pos_export_refusal);
    * the expanded route culls cull distances in the shader instead (RADV_BC250_MESH_CULLDIST_CULL).
    * allow_clipcull: RADV_BC250_MESH_ALLOW_POS1=1. */
   return enabled && mesh && mesh->info.stage == MESA_SHADER_MESH && !has_task && !multiview &&
          !mesh->info.mesh.nv && !mesh->info.task_payload_size &&
          !(mesh->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) &&
          (allow_clipcull || !(mesh->info.outputs_written & RADV_BC250_CLIPCULL_OUTPUTS)) &&
          !radv_bc250_mesh_needs_expansion(mesh);
}

/* RADV_BC250_MESH_AUTOCULL (opt-in, keyed as compiler_info->key.bc250_mesh_autocull):
 * The base driver's expanded Mesh shaders (every triangle owns 3 private vertices) cull their
 * triangles in the NGG epilogue like VS NGG culling and export only the
 * survivors, still as private-vertex triangles (ac_nir_lower_ngg_mesh.c,
 * ms_autocull_compact). Candidates are decided here from the final Mesh NIR and
 * the pipeline state; radv_shader_info.c additionally requires a lane per packed
 * vertex and applies the size policy (wave32, 24..32 triangles, <= 96 lanes, unless
 * RADV_BC250_MESH_AUTOCULL_ALL=1), and the lowering requires no Mesh scratch ring
 * and a layout that keeps the switch-off epilogue's LDS addresses. At draw time the
 * shader culls only when front or back face culling is enabled; otherwise it runs
 * the switch-off epilogue (one scalar test + branch).
 * Only expanded shaders qualify, so a raw shared-vertex shape is never produced;
 * merged and AMD-route shaders are raw and stay unchanged. Skipped: multiview
 * (layer per view), polygon mode other than a static FILL (small-primitive and
 * frustum culling assume filled triangles) and the viewport index (the frustum
 * and small-primitive tests use viewport 0). Clip and cull distances are
 * handled: the surviving triangles export the clip/cull distances of their
 * source vertices (the hardware still clips and culls with them), and the
 * epilogue also culls a triangle when one of its clip or cull distances is
 * negative at all three corners, like VS NGG culling (ms_autocull_accept).
 * Dynamic cull mode, front face, rasterizer discard, conservative rasterization
 * and sample locations need nothing here: they reach the shader through the
 * NGG culling settings user SGPRs. */
bool
radv_bc250_mesh_autocull_candidate(const nir_shader *mesh, bool expanded, bool merged, bool amd_mesh,
                                   const struct radv_graphics_state_key *gfx_state, const char **reason)
{
   const uint64_t unhandled = VARYING_BIT_VIEWPORT | VARYING_BIT_VIEWPORT_MASK;
   *reason = NULL;
   if (!mesh || mesh->info.stage != MESA_SHADER_MESH || mesh->info.mesh.nv)
      *reason = "not an EXT Mesh shader";
   else if (!expanded || merged || amd_mesh)
      *reason = "not expanded (raw output)";
   else if (mesh->info.mesh.primitive_type != MESA_PRIM_TRIANGLES)
      *reason = "not triangles";
   else if (gfx_state->has_multiview_view_index)
      *reason = "multiview";
   else if (gfx_state->rs.polygon_mode_unknown || gfx_state->rs.polygon_mode != VK_POLYGON_MODE_FILL)
      *reason = "polygon mode not static FILL";
   else if (mesh->info.outputs_written & unhandled)
      *reason = "viewport output";
   else if (!(mesh->info.outputs_written & VARYING_BIT_POS))
      *reason = "no position";
   else if (mesh->info.mesh.max_primitives_out == 0 || mesh->info.mesh.max_vertices_out > 256 ||
            mesh->info.mesh.max_vertices_out < 3 * mesh->info.mesh.max_primitives_out)
      *reason = "shape";
   return *reason == NULL;
}

/* RADV_BC250_MESH_CLIPCULL_CONST (default on; =0 keeps every store): the rule of
 * nir_opt_clip_cull_const (RADV runs it for VS/TES/GS only) for Mesh shaders, on the
 * variables right after SPIR-V -> NIR, before any BC250 rewrite (split, expansion, merge,
 * route decisions). A clip or cull distance whose every store is a constant >= 0 (not NaN,
 * not +Inf; -0.0 is not below 0) never clips and never culls, so its stores are deleted.
 * Written elements that also get another value keep all their stores. The two
 * compact per-vertex arrays (after nir_merge_clip_cull_distance_vars: ClipDistance at
 * CLIP_DIST0, CullDistance right after it) are handled separately:
 *  - only store_deref of var[vertex][constant element] with a scalar value is understood;
 *    any other access of the array (load, copy, dynamic element index) keeps it whole;
 *  - an array left without stores is removed, and its array size becomes 0 (a CullDistance
 *    array left alone moves to CLIP_DIST0), so outputs_written, the written components,
 *    clip_dist_mask / cull_dist_mask, PA_CL_VS_OUT_CNTL and SPI_SHADER_POS_FORMAT are those
 *    of the same shader without these stores.
 * Not applied when the fragment shader reads clip/cull distance inputs (the values are then
 * also varyings) or is not known (a pre-rasterization library). FINAL FANTASY VII
 * REBIRTH's CullDistance Mesh shaders all write the constant 0.0 (MESH_PERF/ff7hang). */
struct bc250_cc_array {
   nir_variable *var;
   unsigned len;
   uint32_t noop, normal;
   bool whole; /* another access: keep every store */
};

static bool
bc250_cc_noop_value(nir_def *value)
{
   if (value->num_components != 1 || value->bit_size != 32)
      return false;
   nir_scalar s = nir_scalar_resolved(value, 0);
   if (!nir_scalar_is_const(s))
      return false;
   const float d = nir_scalar_as_float(s);
   /* NaN is clipped/culled, +Inf becomes NaN after interpolation. */
   return !isnan(d) && !(d < 0.0f) && d != INFINITY;
}

/* The element of an understood store (var[vertex][constant element], scalar), or -1. */
static int
bc250_cc_store_element(const nir_intrinsic_instr *in, const struct bc250_cc_array *a)
{
   if (in->intrinsic != nir_intrinsic_store_deref || nir_intrinsic_write_mask(in) != 0x1 ||
       in->src[1].ssa->num_components != 1)
      return -1;
   nir_deref_instr *d = nir_instr_as_deref(nir_def_instr(in->src[0].ssa));
   if (d->deref_type != nir_deref_type_array || !nir_src_is_const(d->arr.index))
      return -1;
   nir_deref_instr *vtx = nir_deref_instr_parent(d);
   if (!vtx || vtx->deref_type != nir_deref_type_array)
      return -1;
   nir_deref_instr *root = nir_deref_instr_parent(vtx);
   if (!root || root->deref_type != nir_deref_type_var || root->var != a->var)
      return -1;
   const uint64_t e = nir_src_as_uint(d->arr.index);
   return e < a->len ? (int)e : -1;
}

static bool
bc250_cc_is_listed(nir_variable *var, void *data)
{
   const struct bc250_cc_array *arrays = data;
   return var == arrays[0].var || var == arrays[1].var;
}

bool
radv_bc250_mesh_clip_cull_const(nir_shader *mesh, const nir_shader *fs, bool fs_known)
{
   if (!mesh || mesh->info.stage != MESA_SHADER_MESH || mesh->info.mesh.nv ||
       !(mesh->info.outputs_written & (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 | VARYING_BIT_CULL_DIST0 |
                                       VARYING_BIT_CULL_DIST1)))
      return false;
   const bool trace = getenv("BC250_TRACE_COMPILE") != NULL;
   const char *reason = NULL;
   if (!fs_known) {
      reason = "fragment shader not known";
   } else if (fs) {
      nir_foreach_shader_in_variable(var, fs) {
         if (var->data.location == VARYING_SLOT_CLIP_DIST0 || var->data.location == VARYING_SLOT_CLIP_DIST1 ||
             var->data.location == VARYING_SLOT_CULL_DIST0 || var->data.location == VARYING_SLOT_CULL_DIST1)
            reason = "fragment shader reads clip/cull distances";
      }
   }

   /* arrays[0]: ClipDistance (merged index 0), arrays[1]: CullDistance (merged index
    * clip_distance_array_size). */
   struct bc250_cc_array arrays[2] = {{0}};
   const unsigned clip_size = mesh->info.clip_distance_array_size, cull_size = mesh->info.cull_distance_array_size;
   nir_foreach_shader_out_variable(var, mesh) {
      if (var->data.location != VARYING_SLOT_CLIP_DIST0 && var->data.location != VARYING_SLOT_CLIP_DIST1)
         continue;
      const struct glsl_type *element = glsl_type_is_array(var->type) ? glsl_get_array_element(var->type) : NULL;
      const unsigned base = (var->data.location - VARYING_SLOT_CLIP_DIST0) * 4 + var->data.location_frac;
      const unsigned which = clip_size && base == 0 ? 0 : 1;
      if (!reason && (!var->data.compact || var->data.per_primitive || !element || !glsl_type_is_array(element) ||
                      arrays[which].var))
         reason = "clip/cull distance variable layout";
      arrays[which].var = var;
      arrays[which].len = element && glsl_type_is_array(element) ? MIN2(glsl_get_length(element), 32) : 0;
   }
   if (!reason && !arrays[0].var && !arrays[1].var)
      reason = "no clip/cull distance array";

   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   if (!reason) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            for (unsigned i = 0; i < nir_intrinsic_infos[in->intrinsic].num_srcs; i++) {
               if (nir_def_instr(in->src[i].ssa)->type != nir_instr_type_deref)
                  continue;
               nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(nir_def_instr(in->src[i].ssa)));
               for (unsigned w = 0; w < 2; w++) {
                  struct bc250_cc_array *a = &arrays[w];
                  if (!a->var || var != a->var)
                     continue;
                  const int e = i == 0 ? bc250_cc_store_element(in, a) : -1;
                  if (e < 0)
                     a->whole = true;
                  else if (bc250_cc_noop_value(in->src[1].ssa))
                     a->noop |= BITFIELD_BIT(e);
                  else
                     a->normal |= BITFIELD_BIT(e);
               }
            }
         }
      }
   }

   uint32_t removed[2] = {0, 0};
   bool gone[2] = {false, false};
   if (!reason) {
      for (unsigned w = 0; w < 2; w++) {
         if (arrays[w].var && !arrays[w].whole)
            removed[w] = arrays[w].noop & ~arrays[w].normal;
      }
      if (!removed[0] && !removed[1])
         reason = "no constant non-negative distance";
   }
   if (reason) {
      if (trace)
         fprintf(stderr, "BC250 MESH CLIPCULL CONST: not applied clip=%u cull=%u reason=%s\n", clip_size, cull_size,
                 reason);
      return false;
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic != nir_intrinsic_store_deref)
            continue;
         for (unsigned w = 0; w < 2; w++) {
            if (!removed[w])
               continue;
            const int e = bc250_cc_store_element(in, &arrays[w]);
            if (e >= 0 && (removed[w] & BITFIELD_BIT(e))) {
               nir_instr_remove(instr);
               break;
            }
         }
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
   NIR_PASS(_, mesh, nir_opt_dce);
   NIR_PASS(_, mesh, nir_remove_dead_derefs);

   /* An array without stores left: remove it. */
   for (unsigned w = 0; w < 2; w++)
      gone[w] = arrays[w].var && (arrays[w].noop | arrays[w].normal) == removed[w];
   const nir_remove_dead_variables_options opts = {.can_remove_var = bc250_cc_is_listed,
                                                   .can_remove_var_data = arrays};
   NIR_PASS(_, mesh, nir_remove_dead_variables, nir_var_shader_out, &opts);
   if (gone[1])
      mesh->info.cull_distance_array_size = 0;
   if (gone[0]) {
      mesh->info.clip_distance_array_size = 0;
      if (arrays[1].var && !gone[1]) {
         arrays[1].var->data.location = VARYING_SLOT_CLIP_DIST0;
         arrays[1].var->data.location_frac = 0;
      }
   }
   const unsigned clip_after = mesh->info.clip_distance_array_size, cull_after = mesh->info.cull_distance_array_size;
   nir_shader_gather_info(mesh, impl);
   /* nir_gather_info clears both sizes only when no clip/cull output is left. */
   if (mesh->info.outputs_written & (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1)) {
      mesh->info.clip_distance_array_size = clip_after;
      mesh->info.cull_distance_array_size = cull_after;
   }
   if (trace)
      fprintf(stderr, "BC250 MESH CLIPCULL CONST: applied removed_clip=0x%x removed_cull=0x%x clip=%u->%u "
              "cull=%u->%u outputs_clipcull=%u\n", removed[0], removed[1], clip_size,
              mesh->info.clip_distance_array_size, cull_size, mesh->info.cull_distance_array_size,
              !!(mesh->info.outputs_written & (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1)));
   return true;
}

/* RADV_BC250_MESH_AUTOCULL_WIDE (compiler key bc250_mesh_autocull_wide, only read with
 * RADV_BC250_MESH_AUTOCULL=1): the size policy of radv_shader_info.c for the wider expanded
 * shapes. 0 / unset / off: the default policy only. 1 (on, true, yes, single): also every
 * candidate with >= 24 triangles whose primitives fit in one wave (Control's 64 triangles in
 * wave64 / 192 lanes, Hellblade 2's 43-triangle pieces in wave64), the one-wave compaction
 * that ran on hardware (MESH_PERF/autocull GATES.md AC1, AC3b k_half64). 2 (multiwave):
 * also primitives spanning several waves (e.g. 65..85 triangles in wave64, 33..85 in
 * wave32), whose cross-wave slot counts never ran on hardware. Anything else: 0. */
/* Shared size policy for expansion and private bary corners. */
bool
radv_bc250_autocull_size_policy(const struct radv_compiler_info *compiler_info,
                               unsigned prims, unsigned wave, unsigned lanes)
{
   const unsigned wide = compiler_info->key.bc250_mesh_autocull_wide;
   return compiler_info->key.bc250_mesh_autocull_all ||
          (wide && prims >= 24 && (prims <= wave || wide >= 2)) ||
          (wave == 32 && prims >= 24 && prims <= 32 && lanes <= 96);
}

unsigned
radv_bc250_autocull_wide_level(const char *option)
{
   if (!option)
      return 0;
   if (!strcmp(option, "1") || !strcasecmp(option, "on") || !strcasecmp(option, "true") ||
       !strcasecmp(option, "yes") || !strcasecmp(option, "single"))
      return 1;
   if (!strcmp(option, "2") || !strcasecmp(option, "multiwave"))
      return 2;
   return 0;
}

/* RADV_BC250_MESH_MERGE (opt-in, keyed as compiler_info->key.bc250_mesh_merge),
 * stage 1 of MESH_PERF/merge/DESIGN.md: K consecutive API Mesh workgroups of a
 * small triangle Mesh shader (P <= 64) run as ONE raw NGG subgroup of K*P >= 65
 * triangles on fast launch 0, the class that hardware ran at VS parity
 * (t_m1/t_s80/t_s65), while every raw 64-triangle subgroup hung.
 *
 * Case A only: the API workgroup (L <= 64 invocations) fits in one wave64.
 * Instance k of the merged subgroup owns lanes [kS, kS+S), vertices
 * [kV, kV+V) and primitives [kP, kP+P); S is the next power of two >= L when
 * the shader uses no subgroup operation (instances never straddle a wave),
 * otherwise S = 64 (one instance per wave, subgroup semantics unchanged).
 *
 * The rewritten shader:
 * - reads the hardware subgroup index g (= workgroup index) and computes
 *   linear = g*K + k; the grid SGPRs keep the application's (x,y,z), so
 *   instance k is valid when linear < x*y*z and a lane runs the API body when
 *   its instance is valid and its lane-in-instance a < L;
 * - replaces LocalInvocationIndex/ID by a (and its 3D split), WorkgroupID by
 *   the 3D split of linear, SubgroupID/NumSubgroups by 0/1, WorkgroupSize by
 *   the API size, offsets shared memory by k*slice;
 * - shrinks API workgroup barriers to subgroup scope (an instance lives in
 *   one wave; the same argument as handle_smaller_ms_api_workgroup);
 * - offsets output array indices by k*V / k*P and primitive vertex indices by
 *   k*V (an index that is exactly the API local index keeps the same-lane
 *   form when V (P) == S, so outputs can stay in registers);
 * - records the instance's clamped SetMeshOutputs(v, p) and, after the body,
 *   calls SetMeshOutputs(K*V+3, K*P) for every live subgroup (0, 0 for a
 *   subgroup without any valid instance, i.e. the GFX10 fully-culled path).
 *   Primitive slots an instance did not produce become degenerate "hole"
 *   triangles (K*V, K*V+1, K*V+2) on three sink vertices at clip position
 *   (2, 2, 2, 1): distinct, increasing indices, zero area, outside the view
 *   volume.
 * Direct draws launch ceil(x*y*z/K) subgroups (radv_emit_direct_mesh_draw_packet);
 * indirect draws are unchanged and the surplus subgroups exit empty.
 *
 * Stage 2 (plan->pp, see bc250_merge_pp_epilogue): shaders with per-primitive
 * outputs (generic attributes, PrimitiveId) or CullPrimitive. Every output is
 * staged in LDS and lane h re-emits physical vertex h and primitive h after a
 * workgroup barrier; per-primitive data rides on a private provoking vertex
 * per primitive (K*(V-1+P)+3 vertices, up to 256 lanes), CullPrimitive becomes
 * hole primitives. Wave32 shaders with subgroup operations merge with one
 * instance per wave32 (S = 32) in both stages.
 */
static bool
bc250_merge_refuse(struct radv_bc250_merge_plan *plan, const char *reason)
{
   plan->reason = reason;
   return false;
}

static bool
bc250_merge_is_subgroup_op(const nir_intrinsic_instr *in)
{
   if (in->intrinsic == nir_intrinsic_load_subgroup_id || in->intrinsic == nir_intrinsic_load_num_subgroups)
      return false;
   static const char *const keys[] = {
      "subgroup", "ballot", "vote", "reduce", "scan", "shuffle", "quad", "elect", "read_invocation",
      "read_first_invocation", "rotate", "swizzle", "first_invocation", "last_invocation", "as_uniform",
      "dpp", "permlane", "mbcnt", "write_invocation",
   };
   const char *name = nir_intrinsic_infos[in->intrinsic].name;
   for (unsigned i = 0; i < ARRAY_SIZE(keys); i++) {
      if (strstr(name, keys[i]))
         return true;
   }
   return false;
}

/* RADV_BC250_MESH_SAFE_PIECES_EXT: a wave32 Mesh shader can run in wave64 unchanged when no subgroup size
 * was required and nothing observes the subgroup: no subgroup operation, subgroup id, subgroup count,
 * subgroup size or invocation. */
bool
radv_bc250_mesh_wave64_promotable(const struct radv_shader_stage *stage)
{
   const nir_shader *nir = stage->nir;
   if (!nir || nir->info.stage != MESA_SHADER_MESH || nir->info.min_subgroup_size != 32 ||
       nir->info.max_subgroup_size != 32 || stage->key.subgroup_required_size)
      return false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic == nir_intrinsic_load_subgroup_id || in->intrinsic == nir_intrinsic_load_num_subgroups ||
                bc250_merge_is_subgroup_op(in))
               return false;
         }
      }
   }
   return true;
}

void
radv_bc250_mesh_set_wave(nir_shader *nir, unsigned wave)
{
   nir->info.min_subgroup_size = wave;
   nir->info.max_subgroup_size = wave;
   nir->info.api_subgroup_size = wave;
}

/* The outermost array deref of an output store, or NULL when the path is not
 * var -> array (vertex/primitive index) -> ... */
static nir_deref_instr *
bc250_merge_output_array(nir_deref_instr *deref, nir_variable **var)
{
   nir_deref_instr *child = NULL;
   while (deref && deref->deref_type != nir_deref_type_var) {
      if (deref->deref_type == nir_deref_type_cast)
         return NULL;
      child = deref;
      deref = nir_deref_instr_parent(deref);
   }
   if (!deref || !child || child->deref_type != nir_deref_type_array)
      return NULL;
   *var = deref->var;
   return child;
}

bool
radv_bc250_mesh_has_per_primitive_data(nir_shader *mesh)
{
   nir_foreach_variable_with_modes(var, mesh, nir_var_shader_out) {
      if (var->data.per_primitive && var->data.location != VARYING_SLOT_PRIMITIVE_INDICES)
         return true;
   }
   return false;
}

/* Stage 2 moves per-primitive data onto the primitive's provoking vertex and
 * the fragment shader reads it as a flat input. A fragment shader that reads
 * explicit per-vertex inputs (barycentrics) would see the physical corners,
 * so it is refused, as for the base driver's compact vertex map. */
const char *
radv_bc250_merge_fs_refusal(nir_shader *fs)
{
   if (!fs)
      return "per-primitive output without a fragment shader";
   nir_foreach_function_impl(impl, fs) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const char *name = nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].name;
            if (strstr(name, "barycentric") || strstr(name, "per_vertex_input"))
               return "fragment shader reads per-vertex inputs";
         }
      }
   }
   return NULL;
}

/* LDS bytes of one staged element, with room for the array's alignment. */
static unsigned
bc250_merge_staged_bytes(const struct glsl_type *element)
{
   unsigned size, alignment;
   glsl_get_explicit_type_for_size_align(element, bc250_shared_type, &size, &alignment);
   return ALIGN_POT(size, alignment);
}

bool
radv_bc250_mesh_merge_plan(nir_shader *mesh, const char *pp_refusal, bool provoking_last,
                           struct radv_bc250_merge_plan *plan)
{
   memset(plan, 0, sizeof(*plan));
   if (!mesh || mesh->info.stage != MESA_SHADER_MESH || mesh->info.mesh.nv)
      return bc250_merge_refuse(plan, "not an EXT Mesh shader");
   if (mesh->info.mesh.primitive_type != MESA_PRIM_TRIANGLES)
      return bc250_merge_refuse(plan, "not triangles");
   const unsigned V = mesh->info.mesh.max_vertices_out, P = mesh->info.mesh.max_primitives_out;
   if (!V || !P || P > 64)
      return bc250_merge_refuse(plan, "P > 64 (or empty)");
   if (mesh->info.task_payload_size)
      return bc250_merge_refuse(plan, "task payload");
   if (mesh->info.workgroup_size_variable)
      return bc250_merge_refuse(plan, "variable workgroup size");
   const unsigned L = mesh->info.workgroup_size[0] * mesh->info.workgroup_size[1] * mesh->info.workgroup_size[2];
   if (!L || L > 64)
      return bc250_merge_refuse(plan, "API workgroup larger than a wave64 (case B)");
   /* Stage 2: per-primitive data and CullPrimitive are staged and re-emitted
    * (radv_bc250_merge_mesh). Data needs a pipeline that allows it
    * (pp_refusal == NULL); CullPrimitive alone does not. */
   bool pp_data = false, pp_cull = false;
   nir_foreach_variable_with_modes(var, mesh, nir_var_shader_out) {
      if (!var->data.per_primitive || var->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
         continue;
      if (var->data.location == VARYING_SLOT_CULL_PRIMITIVE)
         pp_cull = true;
      else
         pp_data = true;
   }
   const bool pp = pp_data || pp_cull;
   if (!pp && (mesh->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE))
      return bc250_merge_refuse(plan, "CullPrimitive");
   if (!pp && (radv_bc250_mesh_needs_expansion(mesh) ||
               (mesh->info.per_primitive_outputs & mesh->info.outputs_written & ~VARYING_BIT_PRIMITIVE_INDICES)))
      return bc250_merge_refuse(plan, "per-primitive output");
   if (pp_data && pp_refusal)
      return bc250_merge_refuse(plan, pp_refusal);
   /* The staging arrays are placed by nir_lower_vars_to_explicit_types,
    * which cannot mix them with an explicit (block) shared layout. */
   if (pp_data && V < 2)
      return bc250_merge_refuse(plan, "per-primitive data with a single vertex");
   if (pp && mesh->info.shared_memory_explicit_layout)
      return bc250_merge_refuse(plan, "explicit shared memory layout (stage-2 staging)");
   if (mesh->info.derivative_group != DERIVATIVE_GROUP_NONE)
      return bc250_merge_refuse(plan, "derivative groups");
   if (mesh->info.zero_initialize_shared_memory)
      return bc250_merge_refuse(plan, "zero-initialized shared memory");
   if (mesh->info.outputs_read)
      return bc250_merge_refuse(plan, "reads its outputs");

   unsigned impls = 0;
   nir_foreach_function_impl(impl, mesh)
      impls++;
   if (impls != 1)
      return bc250_merge_refuse(plan, "more than one function");

   nir_variable *pos = NULL, *indices = NULL;
   unsigned vertex_slots = 0, vertex_bytes = 0, primitive_bytes = 0, outputs = 0;
   nir_foreach_variable_with_modes(var, mesh, nir_var_shader_out) {
      if (!glsl_type_is_array(var->type))
         return bc250_merge_refuse(plan, "non-arrayed output");
      const struct glsl_type *element = glsl_get_array_element(var->type);
      outputs++;
      if (var->data.per_primitive && var->data.location != VARYING_SLOT_PRIMITIVE_INDICES) {
         /* Stage 2 only (pp). Generic attributes and PrimitiveId ride on the
          * provoking vertex as flat attributes; Layer, Viewport and the
          * shading rate are consumed by the hardware from the position
          * exports and stay refused. */
         if (!pp || (var->data.location < VARYING_SLOT_VAR0 && var->data.location != VARYING_SLOT_PRIMITIVE_ID &&
                     var->data.location != VARYING_SLOT_CULL_PRIMITIVE))
            return bc250_merge_refuse(plan, "per-primitive built-in output (Layer/Viewport/shading rate)");
         if (glsl_get_length(var->type) != P)
            return bc250_merge_refuse(plan, "per-primitive array length != max_primitives");
         primitive_bytes += bc250_merge_staged_bytes(element);
         continue;
      }
      if (var->data.per_primitive) {
         if (glsl_get_length(var->type) != P || glsl_get_vector_elements(element) != 3 ||
             glsl_get_bit_size(element) != 32)
            return bc250_merge_refuse(plan, "unexpected primitive index array");
         indices = var;
         primitive_bytes += bc250_merge_staged_bytes(element);
         continue;
      }
      if (glsl_get_length(var->type) != V)
         return bc250_merge_refuse(plan, "per-vertex array length != max_vertices");
      if (var->data.location == VARYING_SLOT_POS) {
         if (!glsl_type_is_vector(element) || glsl_get_vector_elements(element) != 4 ||
             glsl_get_base_type(element) != GLSL_TYPE_FLOAT)
            return bc250_merge_refuse(plan, "unexpected position type");
         pos = var;
      }
      vertex_slots += var->data.compact ? DIV_ROUND_UP(glsl_get_aoa_size(element), 4)
                                        : glsl_count_attribute_slots(element, false);
      vertex_bytes += bc250_merge_staged_bytes(element);
   }
   if (pp && outputs > 64)
      return bc250_merge_refuse(plan, "more than 64 output variables");
   if (!pos || !indices)
      return bc250_merge_refuse(plan, "no position or primitive index output");

   bool subgroup_ops = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return bc250_merge_refuse(plan, "function call");
         if (instr->type == nir_instr_type_deref) {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (nir_deref_mode_may_be(deref, nir_var_mem_shared | nir_var_mem_task_payload))
               return bc250_merge_refuse(plan, "shared/payload deref");
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         const char *name = nir_intrinsic_infos[in->intrinsic].name;
         switch (in->intrinsic) {
         case nir_intrinsic_load_local_invocation_index:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_workgroup_index:
         case nir_intrinsic_load_workgroup_size:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_subgroup_id:
         case nir_intrinsic_load_num_subgroups:
         case nir_intrinsic_set_vertex_and_primitive_count:
         case nir_intrinsic_barrier:
         case nir_intrinsic_load_shared:
         case nir_intrinsic_store_shared:
         case nir_intrinsic_shared_atomic:
         case nir_intrinsic_shared_atomic_swap:
            continue;
         case nir_intrinsic_store_deref: {
            nir_deref_instr *deref = nir_src_as_deref(in->src[0]);
            if (!nir_deref_mode_is(deref, nir_var_shader_out))
               continue;
            nir_variable *var = NULL;
            if (!bc250_merge_output_array(deref, &var) || !(var->data.mode & nir_var_shader_out))
               return bc250_merge_refuse(plan, "unsupported output store");
            continue;
         }
         default:
            break;
         }
         if (bc250_merge_is_subgroup_op(in)) {
            subgroup_ops = true;
            continue;
         }
         for (unsigned i = 0; i < nir_intrinsic_infos[in->intrinsic].num_srcs; i++) {
            nir_deref_instr *deref = nir_src_as_deref(in->src[i]);
            if (deref && nir_deref_mode_may_be(deref, nir_var_shader_out))
               return bc250_merge_refuse(plan, "output access other than a store");
         }
         if (strstr(name, "shared") || strstr(name, "task_payload") || strstr(name, "output") ||
             strstr(name, "global_invocation") || strstr(name, "base_workgroup") ||
             strstr(name, "mesh_view") || strstr(name, "view_index") || strstr(name, "launch_mesh") ||
             strstr(name, "workgroup_id") || strstr(name, "vertex_id") || strstr(name, "instance_id") ||
             strstr(name, "first_vertex") || strstr(name, "base_vertex") || strstr(name, "draw_id_zero"))
            return bc250_merge_refuse(plan, name);
      }
   }

   /* Merged subgroups run in wave64 (the hardware-passed raw regime) unless
    * the shader's subgroup operations pin it to wave32. Without subgroup
    * operations the wave size is not observable, so a wave32 choice (small
    * API workgroups) is overridden. */
   unsigned S, wave = 64;
   if (subgroup_ops) {
      const bool wave64 = mesh->info.min_subgroup_size == 64 && mesh->info.max_subgroup_size == 64 &&
                          (!mesh->info.api_subgroup_size || mesh->info.api_subgroup_size == 64);
      const bool wave32 = mesh->info.min_subgroup_size == 32 && mesh->info.max_subgroup_size == 32 &&
                          (!mesh->info.api_subgroup_size || mesh->info.api_subgroup_size == 32);
      /* Wave32 shaders with subgroup operations (typically L <= 32 with
       * wave intrinsics, which RADV runs as one wave32) merge with one
       * instance per wave32 (S = 32): every subgroup operation still sees
       * exactly its API workgroup, and gl_SubgroupSize stays 32. */
      if (wave64) {
         S = 64;
      } else if (wave32 && L <= 32) {
         S = 32;
         wave = 32;
      } else {
         return bc250_merge_refuse(plan, "subgroup operations in a wave32 shader");
      }
   } else {
      S = util_next_power_of_two(L);
   }

   const unsigned K = DIV_ROUND_UP(65, P);
   /* Stage 2 with per-primitive data: per instance V-1 representative
    * vertices, then one private provoking vertex per primitive (the logical
    * vertex of the first primitive's provoking corner is represented by that
    * primitive's provoking vertex, so V-1 representatives suffice). */
   const unsigned C = pp_data ? V - 1 + P : V;
   const unsigned vm = pp ? K * C + 3 : K * V + 3;
   const unsigned lanes = MAX3(K * S, vm, K * P);
   const unsigned slice = ALIGN_POT(mesh->info.shared_size, 16);
   /* Worst case of ms_calculate_output_layout: every per-vertex output in LDS,
    * workgroup info and 8-bit primitive indices. Staying under 30 KiB keeps
    * all output storage out of the (hardware-unvalidated) Mesh scratch ring.
    * Stage 2 stages every output (K*V vertices, K*P primitives) plus the
    * per-instance counts; its re-emitted outputs are indexed by the lane and
    * stay in registers. */
   const unsigned lds = pp ? K * slice + 16 + K * V * vertex_bytes + K * P * primitive_bytes +
                                16 * outputs + K * 8 + 16
                           : K * slice + 16 + ALIGN_POT(vertex_slots * vm * 16, 16) + 16 + K * P * 3 + 16;
   plan->k = K;
   plan->s = S;
   plan->l = L;
   plan->v = V;
   plan->p = P;
   plan->vm = vm;
   plan->lanes = lanes;
   plan->lds = lds;
   plan->subgroup_ops = subgroup_ops;
   plan->wave = wave;
   plan->pp = pp;
   plan->provoking = pp_data;
   plan->provoking_last = pp_data && provoking_last;
   plan->cull = pp_cull;
   plan->c = pp ? C : V;
   if (K * P > 98)
      return bc250_merge_refuse(plan, "K*P > 98 (stage-1 triangle limit)");
   /* Stage 2 keeps one lane per physical vertex (up to 4 waves64). */
   if (vm > 256 || lanes > (pp ? 256 : 128))
      return bc250_merge_refuse(plan, pp ? "more than 256 physical vertices" : "more than 2 waves64 of lanes");
   if (lds >= 30 * 1024)
      return bc250_merge_refuse(plan, "LDS >= 30 KiB");
   return true;
}

/* Stage 2 of the merge (plan->pp): shaders with per-primitive outputs.
 *
 * GFX10.1 has no per-primitive attribute export, so per-primitive data rides
 * on a per-primitive provoking vertex: flat interpolation reads the provoking
 * vertex (corner 0, or corner 2 with a static "provoking vertex last" mode).
 * Every application output becomes LDS staging (K*V vertices, K*P primitives,
 * indexed as in stage 1), and after one workgroup barrier lane h re-emits
 * physical vertex h and primitive h, so the final outputs are indexed by the
 * lane and stay in registers (no LDS output layout, no index LDS).
 *
 * Physical layout of instance k: C = R + P slots from B = k*C, R = V-1.
 *   R representatives: representative t holds logical vertex
 *     u = t < u0 ? t : t+1, where u0 is the provoking-corner vertex of the
 *     instance's first primitive (that primitive's provoking vertex
 *     represents u0, so V-1 representatives suffice);
 *   P provoking vertices: provoking vertex j is a private copy of primitive
 *     j's provoking-corner vertex plus primitive j's per-primitive outputs
 *     (flat vertex attributes).
 *   The two sequences are zippered by relative position (representative t
 *   has key t/R, provoking vertex j key j/P, representative first on ties):
 *     slot(rep t)  = t + min(P, ceil(t*P/R))
 *     slot(prov j) = j + min(R, floor(j*R/P) + 1)
 *   so primitive j's provoking vertex sits next to the representatives of
 *   the logical vertices around u = j*R/P. For meshlets whose primitives use
 *   vertices roughly in order (every meshlet builder) the exported index
 *   stream keeps the application's own locality, stretched by about C/V:
 *   32/32 on an 8x4 lattice has a largest backward jump of 17 (raw: 8, a
 *   provoking-block layout: 40). Instance k only references [B, B+C); later
 *   instances only use higher slots.
 * Primitive j of instance k exports (corner pc -> slot(prov j), corner
 * c != pc -> u_c == u0 ? slot(prov 0) : slot(rep u_c - (u_c > u0))).
 * Lane h (slot x = h - B) inverts the zipper: j' = floor(x*P/C) and the
 * largest j in {j'-1, j', j'+1} with slot(prov j) <= x decide whether x is
 * a provoking vertex (exhaustively checked for V, P <= 64).
 * Without per-primitive data (CullPrimitive only), C = V and slot u is
 * logical vertex u (the stage-1 layout).
 * Slots whose instance, vertex or primitive does not exist hold a vertex at
 * the sink position; primitive slots that were not produced, or whose
 * CullPrimitive is set, become the stage-1 holes (K*C, K*C+1, K*C+2) on the
 * three sink vertices. Every live subgroup declares K*C+3 vertices and K*P
 * primitives (constant GS_ALLOC_REQ). */
static void
bc250_merge_pp_epilogue(nir_builder *b, nir_shader *mesh, nir_function_impl *impl,
                        const struct radv_bc250_merge_plan *plan, nir_def *h, nir_def *k,
                        nir_def *a, nir_def *g, nir_def *n, nir_variable *counts)
{
   const unsigned K = plan->k, S = plan->s, V = plan->v, P = plan->p, C = plan->c, vm = plan->vm;
   const unsigned R = plan->provoking ? V - 1 : V; /* representative slots per instance */
   const unsigned pc = plan->provoking_last ? 2 : 0;

   /* Staging: the application's outputs become LDS arrays, and every output
    * but CullPrimitive gets a lane-indexed replacement. */
   nir_variable *vars[64], *staged[64], *out[64];
   nir_variable *indices = NULL, *new_indices = NULL, *cull = NULL, *pos = NULL;
   unsigned count = 0;
   nir_foreach_variable_with_modes(var, mesh, nir_var_shader_out) {
      assert(count < ARRAY_SIZE(vars));
      vars[count++] = var;
   }
   for (unsigned i = 0; i < count; i++) {
      nir_variable *var = vars[i];
      const struct glsl_type *element = glsl_get_array_element(var->type);
      const bool prim = var->data.per_primitive;
      out[i] = NULL;
      if (var->data.location == VARYING_SLOT_CULL_PRIMITIVE) {
         cull = var;
      } else {
         out[i] = nir_variable_clone(var, mesh);
         if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES) {
            out[i]->type = glsl_array_type(element, K * P, 0);
            indices = var;
            new_indices = out[i];
         } else {
            out[i]->type = glsl_array_type(element, vm, 0);
            out[i]->data.per_primitive = false;
            if (prim)
               out[i]->data.interpolation = INTERP_MODE_FLAT;
            if (var->data.location == VARYING_SLOT_POS)
               pos = out[i];
         }
         nir_shader_add_variable(mesh, out[i]);
      }
      var->type = glsl_array_type(element, prim ? K * P : K * V, 0);
      var->data.mode = nir_var_mem_shared;
      var->data.compact = false;
      staged[i] = var;
   }
   assert(indices && new_indices && pos);
   nir_fixup_deref_modes(mesh);
   nir_fixup_deref_types(mesh);

   nir_variable *inst_counts = nir_variable_create(mesh, nir_var_mem_shared, glsl_array_type(glsl_uvec_type(2), K, 0),
                                                   "bc250_merge_instance_counts");

   /* Per-instance counts (as stage 1: clustered max, clamped, 0/0 if either
    * is 0), published by the instance's first lane. */
   nir_def *cnt = nir_load_var(b, counts);
   const unsigned cluster = S < plan->wave ? S : 0;
   nir_def *cv = nir_reduce(b, nir_channel(b, cnt, 0), .reduction_op = nir_op_umax, .cluster_size = cluster);
   nir_def *cp = nir_reduce(b, nir_channel(b, cnt, 1), .reduction_op = nir_op_umax, .cluster_size = cluster);
   nir_def *empty = nir_ieq_imm(b, nir_umin(b, cv, cp), 0);
   nir_def *produced = nir_bcsel(b, empty, nir_imm_int(b, 0), cp);
   nir_def *vproduced = nir_bcsel(b, empty, nir_imm_int(b, 0), cv);
   nir_push_if(b, nir_iand(b, nir_ieq_imm(b, a, 0), nir_ult_imm(b, k, K)));
   nir_store_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, inst_counts), k),
                   nir_vec2(b, vproduced, produced), 0x3);
   nir_pop_if(b, NULL);
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);

   nir_def *live = nir_ult(b, nir_imul_imm(b, g, K), n);
   nir_set_vertex_and_primitive_count(b, nir_bcsel(b, live, nir_imm_int(b, vm), nir_imm_int(b, 0)),
                                      nir_bcsel(b, live, nir_imm_int(b, K * P), nir_imm_int(b, 0)),
                                      nir_imm_int(b, 0));

   /* The logical vertex (instance-local, clamped to V-1) of corner c of the
    * staged primitive `slot` of instance kk. */
/* slot(prov j) = j + min(R, floor(j*R/P) + 1) */
#define BC250_PROV_SLOT(j) \
   nir_iadd(b, j, nir_umin(b, nir_iadd_imm(b, nir_udiv_imm(b, nir_imul_imm(b, j, R), P), 1), nir_imm_int(b, R)))
#define BC250_CORNER(tuple, c, kk) \
   nir_umin(b, nir_isub(b, nir_channel(b, tuple, c), nir_imul_imm(b, kk, V)), nir_imm_int(b, V - 1))
   nir_def *sink = nir_imm_vec4(b, 2.0, 2.0, 2.0, 1.0);

   /* Physical vertex h. */
   nir_push_if(b, nir_ult_imm(b, h, K * C));
   {
      nir_def *kk = nir_udiv_imm(b, h, C);
      nir_def *t = nir_isub(b, h, nir_imul_imm(b, kk, C));
      nir_def *ic = nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, inst_counts), kk));
      nir_def *vk = nir_channel(b, ic, 0), *pk = nir_channel(b, ic, 1);
      nir_def *u, *valid, *is_rep = NULL, *prim_slot = NULL;
      if (plan->provoking) {
         /* Invert the zipper (see above): best = the largest candidate j
          * with slot(prov j) <= t, or none. */
         nir_def *jg = nir_udiv_imm(b, nir_imul_imm(b, t, P), C);
         nir_def *best = nir_imm_int(b, -1);
         for (int d = -1; d <= 1; d++) {
            nir_def *j = nir_iadd_imm(b, jg, d);
            /* j < P also rejects jg - 1 = -1 (unsigned compare). */
            nir_def *ok = nir_iand(b, nir_ult_imm(b, j, P), nir_uge(b, t, BC250_PROV_SLOT(j)));
            best = nir_bcsel(b, ok, j, best);
         }
         nir_def *has = nir_ine_imm(b, best, -1);
         is_rep = nir_inot(b, nir_iand(b, has, nir_ieq(b, BC250_PROV_SLOT(best), t)));
         nir_def *j = best;
         nir_def *first = nir_imul_imm(b, kk, P);
         prim_slot = nir_bcsel(b, is_rep, first, nir_iadd(b, first, j));
         nir_def *first_tuple = nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, indices), first));
         nir_def *u0 = nir_bcsel(b, nir_ieq_imm(b, pk, 0), nir_imm_int(b, V), BC250_CORNER(first_tuple, pc, kk));
         /* Representative index: the slot minus the provoking vertices before it. */
         nir_def *rt = nir_isub(b, t, nir_bcsel(b, has, nir_iadd_imm(b, best, 1), nir_imm_int(b, 0)));
         nir_def *u_rep = nir_iadd(b, rt, nir_b2i32(b, nir_uge(b, rt, u0)));
         nir_def *tuple = nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, indices), prim_slot));
         nir_def *u_prov = BC250_CORNER(tuple, pc, kk);
         u = nir_bcsel(b, is_rep, u_rep, u_prov);
         valid = nir_bcsel(b, is_rep, nir_ult(b, u_rep, vk), nir_ult(b, j, pk));
      } else {
         u = t;
         valid = nir_ult(b, t, vk);
      }
      nir_def *logical = nir_iadd(b, nir_imul_imm(b, kk, V), u);
      nir_push_if(b, valid);
      for (unsigned i = 0; i < count; i++) {
         if (!out[i] || staged[i] == indices || staged[i]->data.per_primitive)
            continue;
         nir_copy_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, out[i]), h),
                        nir_build_deref_array(b, nir_build_deref_var(b, staged[i]), logical));
      }
      if (plan->provoking) {
         nir_push_if(b, nir_inot(b, is_rep));
         for (unsigned i = 0; i < count; i++) {
            if (!out[i] || staged[i] == indices || !staged[i]->data.per_primitive)
               continue;
            nir_copy_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, out[i]), h),
                           nir_build_deref_array(b, nir_build_deref_var(b, staged[i]), prim_slot));
         }
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      nir_store_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, pos), h), sink, 0xf);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
   /* The three sink vertices of hole primitives. */
   nir_push_if(b, nir_ult_imm(b, nir_iadd_imm(b, h, -(int)(K * C)), 3));
   nir_store_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, pos), h), sink, 0xf);
   nir_pop_if(b, NULL);

   /* Primitive h. */
   nir_push_if(b, nir_ult_imm(b, h, K * P));
   {
      nir_def *kk = nir_udiv_imm(b, h, P);
      nir_def *j = nir_isub(b, h, nir_imul_imm(b, kk, P));
      nir_def *base = nir_imul_imm(b, kk, C);
      nir_def *ic = nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, inst_counts), kk));
      nir_def *keep = nir_ult(b, j, nir_channel(b, ic, 1));
      if (cull)
         keep = nir_iand(b, keep, nir_inot(b, nir_load_deref(b,
            nir_build_deref_array(b, nir_build_deref_var(b, cull), h))));
      nir_def *tuple = nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, indices), h));
      nir_def *corners[3];
      if (plan->provoking) {
         nir_def *first_tuple = nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, indices),
                                                                        nir_imul_imm(b, kk, P)));
         nir_def *u0 = BC250_CORNER(first_tuple, pc, kk);
         for (unsigned c = 0; c < 3; c++) {
            if (c == pc) {
               corners[c] = nir_iadd(b, base, BC250_PROV_SLOT(j));
               continue;
            }
            nir_def *uc = BC250_CORNER(tuple, c, kk);
            nir_def *rt = nir_isub(b, uc, nir_b2i32(b, nir_ult(b, u0, uc)));
            /* slot(rep t) = t + min(P, ceil(t*P/R)); R >= 1 here (uc != u0). */
            nir_def *rep = nir_iadd(b, rt, nir_umin(b, nir_udiv_imm(b, nir_iadd_imm(b, nir_imul_imm(b, rt, P), R - 1), R),
                                                    nir_imm_int(b, P)));
            corners[c] = nir_bcsel(b, nir_ieq(b, uc, u0), nir_iadd_imm(b, base, R ? 1 : 0), nir_iadd(b, base, rep));
         }
      } else {
         for (unsigned c = 0; c < 3; c++)
            corners[c] = nir_iadd(b, base, BC250_CORNER(tuple, c, kk));
      }
      nir_def *hole = nir_imm_ivec3(b, K * C, K * C + 1, K * C + 2);
      nir_store_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, new_indices), h),
                      nir_bcsel(b, keep, nir_vec(b, corners, 3), hole), 0x7);
   }
   nir_pop_if(b, NULL);
#undef BC250_CORNER
#undef BC250_PROV_SLOT
}

bool
radv_bc250_merge_mesh(nir_shader *mesh, nir_shader *fs, const struct radv_bc250_merge_plan *plan)
{
   const unsigned K = plan->k, S = plan->s, L = plan->l, V = plan->v, P = plan->p;
   const unsigned vm = plan->vm;
   const unsigned lx = mesh->info.workgroup_size[0], ly = mesh->info.workgroup_size[1],
                  lz = mesh->info.workgroup_size[2];
   const unsigned slice = ALIGN_POT(mesh->info.shared_size, 16);
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);

   nir_variable *pos = NULL, *indices = NULL;
   nir_foreach_variable_with_modes(var, mesh, nir_var_shader_out) {
      if (var->data.location == VARYING_SLOT_POS && !var->data.per_primitive)
         pos = var;
      if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
         indices = var;
   }
   assert(pos && indices && K >= 2 && L <= S && S <= 64);

   /* The application's instructions, collected before any new code exists. */
   struct util_dynarray instrs;
   util_dynarray_init(&instrs, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            util_dynarray_append(&instrs, instr);
         } else if (instr->type == nir_instr_type_deref) {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            nir_deref_instr *parent = nir_deref_instr_parent(deref);
            if (deref->deref_type == nir_deref_type_array && parent &&
                parent->deref_type == nir_deref_type_var && (parent->var->data.mode & nir_var_shader_out))
               util_dynarray_append(&instrs, instr);
         }
      }
   }

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *h = nir_load_local_invocation_index(&b);
   nir_def *k = nir_ushr_imm(&b, h, util_logbase2(S));
   nir_def *a = nir_iand_imm(&b, h, S - 1);
   nir_def *g = nir_load_workgroup_index(&b);
   nir_def *grid;
   if (plan->dims) {
      /* Option A (RADV_BC250_MESH_MERGE_INDIRECT=a): a prepped indirect draw
       * launches ceil(N/K) groups and the CP writes the driver record
       * (ceil(N/K),1,1) into the grid SGPRs. The dims user SGPR then holds the
       * low 32 bits of the driver records ({ceil(N/K),1,1,0, X,Y,Z,0}, 32-byte
       * stride, indexed by DrawID); it is 0 for direct draws, option B draws
       * and DGC, whose grid SGPRs hold the application's grid. The branch is
       * uniform, so the load is one scalar load. */
      nir_def *ptr = nir_channel(&b, nir_load_user_data_amd(&b), 0);
      nir_def *draw_id = nir_load_draw_id(&b);
      nir_push_if(&b, nir_ine_imm(&b, ptr, 0));
      nir_def *lo = nir_iadd(&b, ptr, nir_iadd_imm(&b, nir_imul_imm(&b, draw_id, 32), 16));
      nir_def *prepped = nir_load_global(&b, 3, 32, nir_pack_64_2x32_split(&b, lo, nir_imm_int(&b, plan->address32_hi)),
                                         .align_mul = 16, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      nir_push_else(&b, NULL);
      nir_def *app = nir_load_num_workgroups(&b);
      nir_pop_if(&b, NULL);
      grid = nir_if_phi(&b, prepped, app);
   } else {
      grid = nir_load_num_workgroups(&b);
   }
   nir_def *nx = nir_channel(&b, grid, 0), *ny = nir_channel(&b, grid, 1);
   nir_def *n = nir_imul(&b, nir_imul(&b, nx, ny), nir_channel(&b, grid, 2));
   nir_def *linear = nir_iadd(&b, nir_imul_imm(&b, g, K), k);
   nir_def *instance_valid = nir_iand(&b, nir_ult_imm(&b, k, K), nir_ult(&b, linear, n));
   nir_def *lane_valid = nir_iand(&b, instance_valid, nir_ult_imm(&b, a, L));
   nir_variable *counts = nir_local_variable_create(impl, glsl_uvec_type(2), "bc250_merge_counts");
   nir_store_var(&b, counts, nir_imm_ivec2(&b, 0, 0), 0x3);

   nir_if *nif = nir_push_if(&b, lane_valid);
   nir_cf_reinsert(&body, b.cursor);
   b.cursor = nir_after_cf_list(&nif->then_list);
   nir_pop_if(&b, nif);
   nir_cursor epilogue = b.cursor;

   /* Intrinsics first, so output indices see the rewritten local index. */
   util_dynarray_foreach(&instrs, nir_instr *, it) {
      nir_instr *instr = *it;
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
      b.cursor = nir_before_instr(instr);
      nir_def *repl = NULL;
      switch (in->intrinsic) {
      case nir_intrinsic_load_local_invocation_index:
         repl = a;
         break;
      case nir_intrinsic_load_local_invocation_id:
         repl = nir_vec3(&b, nir_umod_imm(&b, a, lx), nir_umod_imm(&b, nir_udiv_imm(&b, a, lx), ly),
                         nir_udiv_imm(&b, a, lx * ly));
         break;
      case nir_intrinsic_load_workgroup_id:
         repl = nir_vec3(&b, nir_umod(&b, linear, nx), nir_umod(&b, nir_udiv(&b, linear, nx), ny),
                         nir_udiv(&b, linear, nir_imul(&b, nx, ny)));
         break;
      case nir_intrinsic_load_workgroup_index:
         repl = linear;
         break;
      case nir_intrinsic_load_num_workgroups:
         /* Without option A the grid SGPRs always hold the application's grid. */
         if (!plan->dims)
            continue;
         repl = grid;
         break;
      case nir_intrinsic_load_workgroup_size:
         repl = nir_imm_ivec3(&b, lx, ly, lz);
         break;
      case nir_intrinsic_load_subgroup_id:
         repl = nir_imm_int(&b, 0);
         break;
      case nir_intrinsic_load_num_subgroups:
         repl = nir_imm_int(&b, 1);
         break;
      case nir_intrinsic_set_vertex_and_primitive_count:
         nir_store_var(&b, counts,
                       nir_vec2(&b, nir_umin(&b, in->src[0].ssa, nir_imm_int(&b, V)),
                                nir_umin(&b, in->src[1].ssa, nir_imm_int(&b, P))), 0x3);
         nir_instr_remove(instr);
         continue;
      case nir_intrinsic_barrier:
         if (nir_intrinsic_execution_scope(in) == SCOPE_WORKGROUP)
            nir_intrinsic_set_execution_scope(in, SCOPE_SUBGROUP);
         if (nir_intrinsic_memory_scope(in) == SCOPE_WORKGROUP)
            nir_intrinsic_set_memory_scope(in, SCOPE_SUBGROUP);
         continue;
      case nir_intrinsic_load_shared:
      case nir_intrinsic_shared_atomic:
      case nir_intrinsic_shared_atomic_swap:
      case nir_intrinsic_store_shared: {
         const unsigned src = in->intrinsic == nir_intrinsic_store_shared ? 1 : 0;
         if (slice)
            nir_src_rewrite(&in->src[src], nir_iadd(&b, in->src[src].ssa, nir_imul_imm(&b, k, slice)));
         continue;
      }
      case nir_intrinsic_store_deref: {
         nir_variable *var = NULL;
         nir_deref_instr *deref = nir_src_as_deref(in->src[0]);
         if (nir_deref_mode_is(deref, nir_var_shader_out) && bc250_merge_output_array(deref, &var) &&
             var == indices) {
            nir_def *value = in->src[1].ssa;
            nir_src_rewrite(&in->src[1], nir_iadd(&b, value,
               nir_imul_imm(&b, nir_u2uN(&b, k, value->bit_size), V)));
         }
         continue;
      }
      default:
         continue;
      }
      nir_def_replace(&in->def, repl);
   }

   /* Output array indices: vertex i of instance k is kV+i, primitive j is kP+j. */
   util_dynarray_foreach(&instrs, nir_instr *, it) {
      nir_instr *instr = *it;
      if (instr->type != nir_instr_type_deref)
         continue;
      nir_deref_instr *arr = nir_instr_as_deref(instr);
      nir_variable *var = nir_deref_instr_parent(arr)->var;
      const unsigned stride = var->data.per_primitive ? P : V;
      nir_def *index = arr->arr.index.ssa;
      nir_scalar resolved = nir_scalar_resolved(index, 0);
      b.cursor = nir_before_instr(instr);
      nir_def *remapped;
      if (stride == S && resolved.def == a && resolved.comp == 0 && index->bit_size == 32)
         remapped = h; /* same lane form: outputs may stay in registers */
      else
         remapped = nir_iadd(&b, index, nir_imul_imm(&b, nir_u2uN(&b, k, index->bit_size), stride));
      nir_src_rewrite(&arr->arr.index, remapped);
   }
   util_dynarray_fini(&instrs);

   if (plan->pp) {
      b.cursor = epilogue;
      bc250_merge_pp_epilogue(&b, mesh, impl, plan, h, k, a, g, n, counts);
   } else {
   /* Resize the output arrays (K*V+3 vertices, K*P primitives) and the derefs
    * of the application's stores. */
   nir_foreach_variable_with_modes(var, mesh, nir_var_shader_out) {
      var->type = glsl_array_type(glsl_get_array_element(var->type), var->data.per_primitive ? K * P : vm,
                                  glsl_get_explicit_stride(var->type));
   }
   nir_fixup_deref_types(mesh);

   /* Epilogue: constant output counts, hole primitives and sink vertices. */
   b.cursor = epilogue;
   nir_def *cnt = nir_load_var(&b, counts);
   const unsigned cluster = S < plan->wave ? S : 0;
   nir_def *cv = nir_reduce(&b, nir_channel(&b, cnt, 0), .reduction_op = nir_op_umax, .cluster_size = cluster);
   nir_def *cp = nir_reduce(&b, nir_channel(&b, cnt, 1), .reduction_op = nir_op_umax, .cluster_size = cluster);
   nir_def *produced = nir_bcsel(&b, nir_ieq_imm(&b, nir_umin(&b, cv, cp), 0), nir_imm_int(&b, 0), cp);
   nir_def *live = nir_ult(&b, nir_imul_imm(&b, g, K), n);
   nir_set_vertex_and_primitive_count(&b, nir_bcsel(&b, live, nir_imm_int(&b, vm), nir_imm_int(&b, 0)),
                                      nir_bcsel(&b, live, nir_imm_int(&b, K * P), nir_imm_int(&b, 0)),
                                      nir_imm_int(&b, 0));
   nir_def *hole = nir_imm_ivec3(&b, K * V, K * V + 1, K * V + 2);
   nir_push_if(&b, nir_ult_imm(&b, k, K));
   if (P == S) {
      nir_push_if(&b, nir_uge(&b, a, produced));
      nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, indices), h), hole, 0x7);
      nir_pop_if(&b, NULL);
   } else {
      nir_variable *cursor = nir_local_variable_create(impl, glsl_uint_type(), "bc250_merge_hole");
      nir_store_var(&b, cursor, nir_iadd(&b, produced, a), 0x1);
      nir_push_loop(&b);
      {
         nir_def *slot = nir_load_var(&b, cursor);
         nir_break_if(&b, nir_uge_imm(&b, slot, P));
         nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, indices),
                                                   nir_iadd(&b, nir_imul_imm(&b, k, P), slot)), hole, 0x7);
         nir_store_var(&b, cursor, nir_iadd_imm(&b, slot, S), 0x1);
      }
      nir_pop_loop(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ult_imm(&b, nir_iadd_imm(&b, h, -(int)(K * V)), 3));
   nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, pos), h),
                   nir_imm_vec4(&b, 2.0, 2.0, 2.0, 1.0), 0xf);
   nir_pop_if(&b, NULL);
   }

   mesh->info.mesh.max_vertices_out = vm;
   mesh->info.mesh.max_primitives_out = K * P;
   mesh->info.workgroup_size[0] = plan->pp ? plan->lanes : MAX2(K * S, vm);
   mesh->info.workgroup_size[1] = 1;
   mesh->info.workgroup_size[2] = 1;
   mesh->info.shared_size = K * slice;
   if (!plan->subgroup_ops) {
      mesh->info.min_subgroup_size = 64;
      mesh->info.max_subgroup_size = 64;
      mesh->info.api_subgroup_size = 64;
   }
   nir_progress(true, impl, nir_metadata_none);
   if (plan->pp) {
      /* The staging arrays are laid out after the K API shared slices. */
      NIR_PASS(_, mesh, nir_split_var_copies);
      NIR_PASS(_, mesh, nir_lower_var_copies);
      NIR_PASS(_, mesh, nir_lower_vars_to_explicit_types, nir_var_mem_shared, bc250_shared_type);
      NIR_PASS(_, mesh, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   }
   NIR_PASS(_, mesh, nir_lower_vars_to_ssa);
   NIR_PASS(_, mesh, nir_opt_dce);
   nir_shader_gather_info(mesh, impl);
   if (plan->provoking && fs) {
      /* The fragment shader reads the per-primitive data as flat attributes
       * of the provoking vertex (same as the base driver's expansion epilogue). */
      nir_foreach_shader_in_variable(var, fs) {
         if (var->data.per_primitive &&
             (var->data.location >= VARYING_SLOT_VAR0 || var->data.location == VARYING_SLOT_PRIMITIVE_ID)) {
            var->data.per_primitive = false;
            var->data.interpolation = INTERP_MODE_FLAT;
         }
      }
      nir_shader_gather_info(fs, nir_shader_get_entrypoint(fs));
   }
   if (getenv("BC250_TRACE_MERGE_NIR")) {
      fprintf(stderr, "BC250 MERGED NIR BEGIN\n");
      nir_print_shader(mesh, stderr);
      fprintf(stderr, "BC250 MERGED NIR END\n");
   }
   return true;
}

/* Shared variables that no deref uses any more (already lowered to explicit
 * offsets inside [0, info.shared_size)). */
static unsigned
bc250_dead_shared_variables(nir_shader *shader)
{
   struct set *used = _mesa_pointer_set_create(NULL);
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref &&
                nir_instr_as_deref(instr)->deref_type == nir_deref_type_var)
               _mesa_set_add(used, nir_instr_as_deref(instr)->var);
         }
      }
   }
   unsigned dead = 0;
   nir_foreach_variable_with_modes(var, shader, nir_var_mem_shared)
      dead += !_mesa_set_search(used, var);
   _mesa_set_destroy(used, NULL);
   return dead;
}

void
radv_bc250_dump_mesh_nir(const char *dir, const nir_shader *nir, unsigned wave_size, unsigned hw_workgroup_size,
                         const unsigned char *blake3, const uint32_t index_staging[4], uint32_t compact_flags,
                         uint64_t pp_params, uint32_t clipcull)
{
   static uint32_t sequence;
   const uint32_t n = p_atomic_inc_return(&sequence);
   struct blob blob;
   blob_init(&blob);
   /* "MSR3": word 7 = RADV_BC250_MESH_COMPACT flags (bit 0 applied, bits 8..10 owned
    * corners, bits 16..23 W), words 8..9 = the parameter exports that carry a
    * per-primitive (flat) payload, word 10 = clip/cull distance exports (bits 0..3 exported
    * clip components, bits 4..7 exported cull components, bit 8 cull distances culled in the
    * shader and not exported), word 11 reserved. */
   const uint32_t header[12] = {0x3352534d /* "MSR3" */, wave_size, hw_workgroup_size, index_staging[0],
                                index_staging[1], index_staging[2], index_staging[3], compact_flags,
                                (uint32_t)pp_params, (uint32_t)(pp_params >> 32), clipcull, 0};
   blob_write_bytes(&blob, header, sizeof(header));
   nir_serialize(&blob, nir, false);
   char path[4096];
   snprintf(path, sizeof(path), "%s/%06u_%02x%02x%02x%02x%02x%02x%02x%02x.nir", dir, n, blake3[0], blake3[1],
            blake3[2], blake3[3], blake3[4], blake3[5], blake3[6], blake3[7]);
   FILE *f = fopen(path, "wb");
   if (f) {
      if (!blob.out_of_memory)
         fwrite(blob.data, 1, blob.size, f);
      fclose(f);
   }
   blob_finish(&blob);
}

/* RADV_BC250_MESH_DIRECT_READ staging parts (radv_bc250_expand_primitive_attributes).
 * An access of a staging variable: load_deref or store_deref of var[element] or
 * var[element][component]. */
struct bc250_staging_access {
   nir_intrinsic_instr *intr;
   nir_def *element;
   nir_def *component; /* NULL: the whole element */
};

/* Every access of var, or false when var has another use (a copy of the whole
 * array, a cast, an atomic, a deref passed elsewhere). */
static bool
bc250_staging_accesses(nir_function_impl *impl, nir_variable *var, struct util_dynarray *accesses)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *root = nir_instr_as_deref(instr);
         if (root->deref_type != nir_deref_type_var || root->var != var)
            continue;
         nir_foreach_use_including_if(use, &root->def) {
            if (nir_src_is_if(use) || nir_src_use_instr(use)->type != nir_instr_type_deref)
               return false;
            nir_deref_instr *element = nir_instr_as_deref(nir_src_use_instr(use));
            if (element->deref_type != nir_deref_type_array)
               return false;
            nir_foreach_use_including_if(euse, &element->def) {
               if (nir_src_is_if(euse))
                  return false;
               nir_instr *user = nir_src_use_instr(euse);
               nir_deref_instr *access = element;
               nir_def *component = NULL;
               if (user->type == nir_instr_type_deref) {
                  nir_deref_instr *comp = nir_instr_as_deref(user);
                  if (comp->deref_type != nir_deref_type_array || !glsl_type_is_vector(element->type))
                     return false;
                  nir_foreach_use_including_if(cuse, &comp->def) {
                     if (nir_src_is_if(cuse) || nir_src_use_instr(cuse)->type != nir_instr_type_intrinsic)
                        return false;
                     nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_src_use_instr(cuse));
                     if ((intr->intrinsic != nir_intrinsic_load_deref && intr->intrinsic != nir_intrinsic_store_deref) ||
                         cuse != &intr->src[0])
                        return false;
                     struct bc250_staging_access a = {intr, element->arr.index.ssa, comp->arr.index.ssa};
                     util_dynarray_append(accesses, a);
                  }
                  continue;
               }
               if (user->type != nir_instr_type_intrinsic)
                  return false;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(user);
               if ((intr->intrinsic != nir_intrinsic_load_deref && intr->intrinsic != nir_intrinsic_store_deref) ||
                   euse != &intr->src[0])
                  return false;
               struct bc250_staging_access a = {intr, access->arr.index.ssa, component};
               util_dynarray_append(accesses, a);
            }
         }
      }
   }
   return true;
}

static bool
bc250_instr_in_loop(const nir_instr *instr)
{
   for (const nir_cf_node *n = instr->block->cf_node.parent; n; n = n->parent) {
      if (n->type == nir_cf_node_loop)
         return true;
   }
   return false;
}

/* A workgroup-uniform value: defined (by the divergence analysis across
 * subgroups) as uniform in uniform control flow outside loops, or constants and
 * ALU of such values (a pure function of uniform values is uniform wherever it
 * is computed). */
static bool
bc250_uniform_value(nir_def *def, unsigned depth)
{
   nir_instr *instr = nir_def_instr(def);
   if (!def->divergent && !instr->block->divergent && !bc250_instr_in_loop(instr))
      return true;
   if (depth > 16)
      return false;
   if (instr->type == nir_instr_type_load_const)
      return true;
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      if (!bc250_uniform_value(alu->src[i].src.ssa, depth + 1))
         return false;
   }
   return true;
}

/* Uniform part: a staging variable whose accesses are all stores outside loops
 * of workgroup-uniform values (bc250_uniform_value), each storing its own
 * components (the whole element with a write mask, or one constant component),
 * holds the same value in every element written: whichever vertex or primitive
 * the expansion copies, it copies that value (or an element never written,
 * undefined). One cell of the element type replaces the array. Needs the
 * divergence analysis across subgroups. Returns the cell or NULL. */
static nir_variable *
bc250_uniform_staging(nir_shader *mesh, nir_function_impl *impl, nir_variable *var)
{
   const struct glsl_type *element = glsl_get_array_element(var->type);
   if (!glsl_type_is_vector_or_scalar(element) || glsl_get_bit_size(element) != 32)
      return NULL;

   struct util_dynarray accesses;
   util_dynarray_init(&accesses, NULL);
   bool ok = bc250_staging_accesses(impl, var, &accesses) && accesses.size;
   unsigned written = 0;
   util_dynarray_foreach(&accesses, struct bc250_staging_access, a) {
      if (!ok)
         break;
      unsigned mask;
      if (a->intr->intrinsic != nir_intrinsic_store_deref || bc250_instr_in_loop(&a->intr->instr) ||
          !bc250_uniform_value(a->intr->src[1].ssa, 0)) {
         ok = false;
      } else if (!a->component) {
         mask = nir_intrinsic_write_mask(a->intr);
      } else if (nir_src_is_const(nir_src_for_ssa(a->component)) &&
                 nir_src_as_uint(nir_src_for_ssa(a->component)) < glsl_get_vector_elements(element)) {
         mask = BITFIELD_BIT(nir_src_as_uint(nir_src_for_ssa(a->component)));
      } else {
         ok = false;
      }
      if (ok && (written & mask))
         ok = false;
      if (ok)
         written |= mask;
   }
   nir_variable *cell = NULL;
   if (ok) {
      cell = nir_variable_create(mesh, nir_var_mem_shared, element, "bc250_uniform_output");
      cell->data.location = var->data.location;
      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&accesses, struct bc250_staging_access, a) {
         b.cursor = nir_before_instr(&a->intr->instr);
         nir_deref_instr *d = nir_build_deref_var(&b, cell);
         if (a->component)
            d = nir_build_deref_array_imm(&b, d, nir_src_as_uint(nir_src_for_ssa(a->component)));
         nir_src_rewrite(&a->intr->src[0], &d->def);
      }
   }
   util_dynarray_fini(&accesses);
   return cell;
}

/* Index16 part: the staged primitive indices as a flat array of 16-bit values
 * (corners per primitive) instead of 32-bit vectors. The staged values are the
 * application's indices, which must be below the vertex count (<= 256): the low
 * 16 bits are the value. Loads zero-extend back to 32 bits; the exported
 * connectivity is unchanged. Returns the new array or NULL. */
static nir_variable *
bc250_narrow_index_staging(nir_shader *mesh, nir_function_impl *impl, nir_variable *indices, unsigned corners)
{
   const struct glsl_type *element = glsl_get_array_element(indices->type);
   if (!glsl_type_is_vector_or_scalar(element) || glsl_get_bit_size(element) != 32 ||
       glsl_get_vector_elements(element) != corners)
      return NULL;
   struct util_dynarray accesses;
   util_dynarray_init(&accesses, NULL);
   if (!bc250_staging_accesses(impl, indices, &accesses)) {
      util_dynarray_fini(&accesses);
      return NULL;
   }
   nir_variable *narrow = nir_variable_create(mesh, nir_var_mem_shared,
      glsl_array_type(glsl_uint16_t_type(), glsl_get_length(indices->type) * corners, 0), "bc250_indices16");
   nir_builder b = nir_builder_create(impl);
   util_dynarray_foreach(&accesses, struct bc250_staging_access, a) {
      b.cursor = nir_before_instr(&a->intr->instr);
      nir_def *first = nir_imul_imm(&b, a->element, corners);
      if (a->intr->intrinsic == nir_intrinsic_store_deref) {
         nir_def *value = a->intr->src[1].ssa;
         if (a->component) {
            nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, narrow), nir_iadd(&b, first, a->component)),
                            nir_u2u16(&b, value), 1);
         } else {
            u_foreach_bit(c, nir_intrinsic_write_mask(a->intr)) {
               nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, narrow), nir_iadd_imm(&b, first, c)),
                               nir_u2u16(&b, nir_channel(&b, value, c)), 1);
            }
         }
      } else {
         nir_def *result;
         if (a->component) {
            result = nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, narrow),
                                                                            nir_iadd(&b, first, a->component))));
         } else {
            nir_def *comps[NIR_MAX_VEC_COMPONENTS];
            for (unsigned c = 0; c < corners; c++)
               comps[c] = nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, narrow),
                                                                                nir_iadd_imm(&b, first, c))));
            result = nir_vec(&b, comps, corners);
         }
         nir_def_rewrite_uses(&a->intr->def, result);
      }
      nir_instr_remove(&a->intr->instr);
   }
   util_dynarray_fini(&accesses);
   return narrow;
}

/* The staged index tuple of primitive p (32-bit), and its corner c. */
static nir_def *
bc250_index_tuple(nir_builder *b, nir_variable *indices, nir_variable *narrow, unsigned corners, nir_def *p)
{
   if (!narrow)
      return nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, indices), p));
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   nir_def *first = nir_imul_imm(b, p, corners);
   for (unsigned c = 0; c < corners; c++)
      comps[c] = nir_u2u32(b, nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, narrow),
                                                                      nir_iadd_imm(b, first, c))));
   return nir_vec(b, comps, corners);
}

/* Corner part: load only the selected corner (one LDS load) instead of the
 * tuple and a select. */
static nir_def *
bc250_index_corner(nir_builder *b, nir_variable *indices, nir_variable *narrow, unsigned corners, nir_def *p,
                   nir_def *corner, bool corner_load)
{
   if (corners == 1)
      return bc250_index_tuple(b, indices, narrow, corners, p);
   if (!corner_load)
      return nir_vector_extract(b, bc250_index_tuple(b, indices, narrow, corners, p), corner);
   if (narrow)
      return nir_u2u32(b, nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_var(b, narrow),
                                                                 nir_iadd(b, nir_imul_imm(b, p, corners), corner))));
   return nir_load_deref(b, nir_build_deref_array(b, nir_build_deref_array(b, nir_build_deref_var(b, indices), p),
                                                  corner));
}

static bool
bc250_is_replaced_staging(nir_variable *var, void *data)
{
   return _mesa_set_search((struct set *)data, var) != NULL;
}

/* RADV_BC250_MESH_DIRECT_READ: "1" (or true/yes/on) = export from the staging
 * only; "full" (or all) = every part; or a comma-separated list of parts:
 * export, uniform, dead, index16, corner (RADV_BC250_DIRECT_READ_*). */
unsigned
radv_bc250_direct_read_parts(const char *option)
{
   static const struct {
      const char *name;
      unsigned bit;
   } parts[] = {
      {"export", RADV_BC250_DIRECT_READ_EXPORT},   {"uniform", RADV_BC250_DIRECT_READ_UNIFORM},
      {"dead", RADV_BC250_DIRECT_READ_DEAD},       {"index16", RADV_BC250_DIRECT_READ_INDEX16},
      {"corner", RADV_BC250_DIRECT_READ_CORNER},
   };
   if (!option || !*option || !strcmp(option, "0") || !strcasecmp(option, "false") || !strcasecmp(option, "no") ||
       !strcasecmp(option, "off"))
      return 0;
   if (!strcmp(option, "1") || !strcasecmp(option, "true") || !strcasecmp(option, "yes") || !strcasecmp(option, "on"))
      return RADV_BC250_DIRECT_READ_EXPORT;
   if (!strcasecmp(option, "full") || !strcasecmp(option, "all"))
      return RADV_BC250_DIRECT_READ_ALL;

   unsigned result = 0;
   const char *p = option;
   while (*p) {
      const size_t len = strcspn(p, ",");
      bool known = false;
      for (unsigned i = 0; i < ARRAY_SIZE(parts); i++) {
         if (strlen(parts[i].name) == len && !strncasecmp(p, parts[i].name, len)) {
            result |= parts[i].bit;
            known = true;
         }
      }
      if (!known && len)
         fprintf(stderr, "radv: RADV_BC250_MESH_DIRECT_READ: unknown part '%.*s' ignored\n", (int)len, p);
      p += len;
      if (*p == ',')
         p++;
   }
   return result;
}

/* The elements of a compact per-vertex array output (the merged ClipDistance/CullDistance
 * array) that the application writes: bit e for stores to var[v][e] with a constant e, every
 * element for a dynamic element index or a store of a whole element array. The expansion
 * copies only these to the expanded vertices, so the expanded shader writes, exports and
 * enables (PA_CL_VS_OUT_CNTL, from the written components) exactly the distances the
 * application shader does; an element it never writes stays unwritten instead of receiving the
 * undefined staging contents. */
static uint32_t
bc250_compact_written_elements(nir_function_impl *impl, nir_variable *var)
{
   const struct glsl_type *element = glsl_get_array_element(var->type);
   const unsigned len = glsl_type_is_array(element) ? glsl_get_length(element) : 1;
   const uint32_t all = BITFIELD_MASK(MIN2(len, 32));
   uint32_t mask = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic != nir_intrinsic_store_deref && in->intrinsic != nir_intrinsic_copy_deref)
            continue;
         nir_deref_instr *d = nir_src_as_deref(in->src[0]);
         if (nir_deref_instr_get_variable(d) != var)
            continue;
         if (d->deref_type == nir_deref_type_array && nir_deref_instr_parent(d)->deref_type == nir_deref_type_array &&
             nir_src_is_const(d->arr.index) && nir_src_as_uint(d->arr.index) < len)
            mask |= BITFIELD_BIT(nir_src_as_uint(d->arr.index));
         else
            return all;
      }
   }
   return mask;
}

bool
radv_bc250_expand_primitive_attributes(nir_shader *mesh, nir_shader *fs, bool compact_policy,
                                       bool reclaim_dead_shared, unsigned direct_read, bool *over_budget,
                                       bool *dead_shared, uint32_t index_staging[4], bool *compact_map,
                                       bool clamp_primitives)
{
   const uint64_t generic = mesh->info.per_primitive_outputs &
      ((UINT64_C(0xffffffff) << VARYING_SLOT_VAR0) | VARYING_BIT_PRIMITIVE_ID |
       VARYING_BIT_VIEWPORT | VARYING_BIT_LAYER);
   /* The staging variables below are laid out after info.shared_size by
    * nir_lower_vars_to_explicit_types, which an explicit (block) shared
    * layout cannot do (nir_assign_shared_var_locations: blocks only). */
   if (!fs || mesh->info.shared_memory_explicit_layout)
      return false;

   /* BC250 Mesh LDS fit retry (radv_graphics_pipeline_init). The shared
    * variables that exist here are already lowered to explicit offsets inside
    * [0, info.shared_size): the application's (radv_shader_spirv_to_nir) and
    * the split's (radv_bc250_split_mesh). No deref uses them any more, but
    * nir_lower_vars_to_explicit_types below would place every one of them
    * again after info.shared_size: a dead copy that costs LDS (3588 bytes for
    * a Hellblade 2 Nanite piece). Drop them. Only on a retry, so shaders that
    * already fit keep their exact layout. */
   const unsigned dead = bc250_dead_shared_variables(mesh);
   if (dead_shared)
      *dead_shared = dead > 0;
   /* RADV_BC250_MESH_DIRECT_READ dead part: always (the retry skips its same-pieces step). */
   if (direct_read & RADV_BC250_DIRECT_READ_DEAD)
      reclaim_dead_shared = true;
   if (reclaim_dead_shared && dead) {
      NIR_PASS(_, mesh, nir_remove_dead_variables, nir_var_mem_shared, NULL);
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 Mesh compact LDS: dead shared variables dropped=%u (live shared_size=%u)\n",
                 dead, mesh->info.shared_size);
   }

   unsigned corners = mesh->info.mesh.primitive_type == MESA_PRIM_TRIANGLES ? 3 :
                      mesh->info.mesh.primitive_type == MESA_PRIM_LINES ? 2 : 1;
   unsigned expanded = corners * mesh->info.mesh.max_primitives_out;
   const unsigned logical_vertices = mesh->info.mesh.max_vertices_out;
   nir_foreach_shader_out_variable(var, mesh) {
      const struct glsl_type *element = glsl_type_is_array(var->type) ? glsl_get_array_element(var->type) : NULL;
      if (!element || !glsl_type_is_vector_or_scalar(element) || glsl_get_bit_size(element) != 32 ||
          (var->data.location < VARYING_SLOT_VAR0 && var->data.location != VARYING_SLOT_POS &&
           var->data.location != VARYING_SLOT_PRIMITIVE_INDICES))
         compact_policy = false;
   }
   nir_foreach_function_impl(fimpl, fs) {
      nir_foreach_block(block, fimpl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               const char *name = nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].name;
               if (strstr(name, "barycentric") || strstr(name, "per_vertex_input"))
                  compact_policy = false;
            }
         }
      }
   }
   const bool compact = compact_policy && corners == 3 && generic &&
      mesh->info.mesh.max_primitives_out <= 63 && logical_vertices > 0 &&
      logical_vertices + mesh->info.mesh.max_primitives_out - 1 < expanded &&
      !(mesh->info.outputs_written & (VARYING_BIT_CULL_PRIMITIVE | VARYING_BIT_LAYER |
                                     VARYING_BIT_VIEWPORT | VARYING_BIT_PRIMITIVE_ID)) &&
      !(fs->info.inputs_read & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT | VARYING_BIT_PRIMITIVE_ID));
   if (compact_map)
      *compact_map = compact;
   if (compact) {
      expanded = logical_vertices + mesh->info.mesh.max_primitives_out - 1;
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 compact map: logical_vertices=%u primitives=%u physical_capacity=%u\n",
                 logical_vertices, mesh->info.mesh.max_primitives_out, expanded);
   }
   /* Opt-in BC250 expansion also gives ordinary shared-index meshes
    * private consecutive vertex slots. Do not broaden the unvalidated
    * null-primitive/culling path: preserve its previous eligibility.
    * The 256-vertex and shared-memory bounds below still apply. */
   if (!generic && mesh->info.mesh.max_vertices_out <= expanded &&
       (mesh->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE))
      return false;
   if (expanded > 256 || mesh->info.mesh.nv)
      return false;

   nir_variable *source[128], *output[128], *indices = NULL, *new_indices = NULL;
   unsigned count = 0;
   unsigned staging_bytes = ALIGN_POT(mesh->info.shared_size, 16) + 4;
   if (compact)
      staging_bytes += (logical_vertices + expanded + 1) * 4;
   nir_foreach_shader_out_variable(var, mesh) {
      if (count == ARRAY_SIZE(source) || !glsl_type_is_array(var->type))
         return false;
      /* Duplicated vertices carry identical routing and ID for each primitive.
       * CullPrimitive is handled natively by the NGG prim-exp argument (no
       * duplication needed); shading rate remains excluded. */
      if (var->data.per_primitive && var->data.location < VARYING_SLOT_VAR0 &&
          var->data.location != VARYING_SLOT_PRIMITIVE_INDICES &&
          var->data.location != VARYING_SLOT_PRIMITIVE_ID &&
          var->data.location != VARYING_SLOT_VIEWPORT &&
          var->data.location != VARYING_SLOT_LAYER &&
          var->data.location != VARYING_SLOT_CULL_PRIMITIVE)
         return false;
      unsigned size, alignment;
      glsl_get_explicit_type_for_size_align(var->type, bc250_shared_type, &size, &alignment);
      staging_bytes = ALIGN_POT(staging_bytes, alignment) + size;
      /* The staging arrays become API shared memory for the NGG Mesh
       * lowering, which guarantees 28 KiB of it (the Vulkan minimum
       * maxMeshSharedMemorySize) and keeps the total within the 32 KiB NGG
       * LDS window, moving its own output storage to the Mesh scratch ring
       * only when it runs out. That ring is not validated on GFX1013, so
       * radv_graphics_shaders_compile refuses expanded shaders that need it.
       * Hellblade 2's Nanite shaders stage 18.9 KiB (26.5 KiB total LDS). */
      if (staging_bytes > 28 * 1024) {
         if (over_budget)
            *over_budget = true;
         return false;
      }
      source[count++] = var;
      if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
         indices = var;
   }
   if (!indices)
      return false;

   /* Compact arrays (ClipDistance/CullDistance): the elements the application writes. */
   nir_function_impl *entry = nir_shader_get_entrypoint(mesh);
   uint32_t compact_written[ARRAY_SIZE(source)];
   for (unsigned i = 0; i < count; i++) {
      const struct glsl_type *element = glsl_get_array_element(source[i]->type);
      compact_written[i] = source[i]->data.compact && glsl_type_is_array(element) ?
                           bc250_compact_written_elements(entry, source[i]) : ~0u;
   }

   for (unsigned i = 0; i < count; i++) {
      nir_variable *var = source[i];
      output[i] = nir_variable_clone(var, mesh);
      if (var == indices) {
         new_indices = output[i];
      } else if (var->data.location != VARYING_SLOT_CULL_PRIMITIVE) {
         output[i]->type = glsl_array_type(glsl_get_array_element(var->type), expanded, 0);
         output[i]->data.per_primitive = false;
         if (var->data.per_primitive)
            output[i]->data.interpolation = INTERP_MODE_FLAT;
      }
      var->data.mode = nir_var_mem_shared;
      var->data.compact = false;
      nir_shader_add_variable(mesh, output[i]);
   }
   nir_fixup_deref_modes(mesh);

   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);

   /* RADV_BC250_MESH_DIRECT_READ staging parts: one cell per uniform output,
    * 16-bit primitive indices. The replaced arrays are removed below. */
   nir_variable *uniform_cell[ARRAY_SIZE(source)] = {0};
   nir_variable *narrow_indices = NULL;
   struct set *replaced = _mesa_pointer_set_create(NULL);
   if (direct_read & RADV_BC250_DIRECT_READ_UNIFORM) {
      nir_custom_divergence_analysis(mesh, nir_divergence_across_subgroups);
      for (unsigned i = 0; i < count; i++) {
         if (source[i] == indices || source[i]->data.location < VARYING_SLOT_VAR0)
            continue;
         uniform_cell[i] = bc250_uniform_staging(mesh, impl, source[i]);
         if (uniform_cell[i])
            _mesa_set_add(replaced, source[i]);
      }
   }
   if ((direct_read & RADV_BC250_DIRECT_READ_INDEX16) && logical_vertices <= 256) {
      narrow_indices = bc250_narrow_index_staging(mesh, impl, indices, corners);
      if (narrow_indices)
         _mesa_set_add(replaced, indices);
   }
   const bool corner_load = direct_read & RADV_BC250_DIRECT_READ_CORNER;
   nir_variable *counts = nir_variable_create(mesh, nir_var_mem_shared, glsl_uint_type(), "bc250_primitive_count");
   nir_builder b = nir_builder_create(impl);
   b.cursor = nir_before_cf_list(&impl->body);
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_store_var(&b, counts, nir_imm_int(&b, 0), 1);
   nir_pop_if(&b, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_barrier &&
             (nir_intrinsic_memory_modes(intr) & nir_var_shader_out))
            nir_intrinsic_set_memory_modes(intr, nir_intrinsic_memory_modes(intr) | nir_var_mem_shared);
         if (intr->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
         nir_def *empty = nir_ieq_imm(&b, intr->src[0].ssa, 0);
         nir_store_var(&b, counts, nir_bcsel(&b, empty, nir_imm_int(&b, 0), intr->src[1].ssa), 1);
         nir_pop_if(&b, NULL);
         nir_instr_remove(instr);
      }
   }

   b.cursor = nir_after_cf_list(&impl->body);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   nir_def *primitives = nir_load_var(&b, counts);
   /* RADV_BC250_MESH_COMPACT: a primitive count above the declared maximum (undefined in
    * the API) must not make the copy loop below write past the expanded records, into the
    * lowering's LDS (the compaction tables). */
   if (clamp_primitives)
      primitives = nir_umin(&b, primitives, nir_imm_int(&b, mesh->info.mesh.max_primitives_out));
   nir_def *vertices = nir_imul_imm(&b, primitives, corners);
   nir_variable *vertex_map = NULL;
   if (compact) {
      nir_variable *representatives = nir_variable_create(mesh, nir_var_mem_shared,
         glsl_array_type(glsl_uint_type(), logical_vertices, 0), "bc250_representatives");
      vertex_map = nir_variable_create(mesh, nir_var_mem_shared,
         glsl_array_type(glsl_uint_type(), expanded, 0), "bc250_logical_vertex_map");
      nir_variable *physical_count = nir_variable_create(mesh, nir_var_mem_shared,
         glsl_uint_type(), "bc250_physical_vertex_count");
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
      nir_variable *it = nir_local_variable_create(impl, glsl_uint_type(), "bc250_map_cursor");
      nir_variable *next = nir_local_variable_create(impl, glsl_uint_type(), "bc250_map_next");
      nir_store_var(&b, next, primitives, 1);
      nir_store_var(&b, it, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      nir_def *i = nir_load_var(&b, it);
      nir_push_if(&b, nir_uge_imm(&b, i, logical_vertices));
      nir_jump(&b, nir_jump_break); nir_pop_if(&b, NULL);
      nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, representatives), i),
                      nir_imm_int(&b, -1), 1);
      nir_store_var(&b, it, nir_iadd_imm(&b, i, 1), 1);
      nir_pop_loop(&b, NULL);
      /* Every primitive owns its first vertex, preserving flat payloads and
       * original corner order. Other corners may share a representative. */
      nir_store_var(&b, it, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      i = nir_load_var(&b, it);
      nir_push_if(&b, nir_uge(&b, i, primitives));
      nir_jump(&b, nir_jump_break); nir_pop_if(&b, NULL);
      nir_def *tuple = bc250_index_tuple(&b, indices, narrow_indices, corners, i);
      nir_def *first = nir_channel(&b, tuple, 0);
      nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, representatives), first), i, 1);
      nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, vertex_map), i), first, 1);
      nir_store_var(&b, it, nir_iadd_imm(&b, i, 1), 1);
      nir_pop_loop(&b, NULL);
      nir_store_var(&b, it, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      i = nir_load_var(&b, it);
      nir_push_if(&b, nir_uge(&b, i, primitives));
      nir_jump(&b, nir_jump_break); nir_pop_if(&b, NULL);
      tuple = bc250_index_tuple(&b, indices, narrow_indices, corners, i);
      nir_def *mapped_corners[3] = {i, NULL, NULL};
      for (unsigned c = 1; c < 3; c++) {
         nir_def *logical = nir_channel(&b, tuple, c);
         nir_deref_instr *rep = nir_build_deref_array(&b, nir_build_deref_var(&b, representatives), logical);
         nir_def *old = nir_load_deref(&b, rep);
         nir_push_if(&b, nir_ieq_imm(&b, old, -1));
         nir_def *fresh = nir_load_var(&b, next);
         nir_store_deref(&b, rep, fresh, 1);
         nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, vertex_map), fresh), logical, 1);
         nir_store_var(&b, next, nir_iadd_imm(&b, fresh, 1), 1);
         nir_pop_if(&b, NULL);
         mapped_corners[c] = nir_load_deref(&b, rep);
      }
      nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, new_indices), i),
                      nir_vec(&b, mapped_corners, 3), 7);
      nir_store_var(&b, it, nir_iadd_imm(&b, i, 1), 1);
      nir_pop_loop(&b, NULL);
      nir_store_var(&b, physical_count, nir_load_var(&b, next), 1);
      nir_pop_if(&b, NULL);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared | nir_var_shader_out);
      vertices = nir_load_var(&b, physical_count);
   }
   nir_set_vertex_and_primitive_count(&b, vertices, primitives, nir_imm_int(&b, 0));
   nir_variable *cursor = nir_local_variable_create(impl, glsl_uint_type(), "bc250_expanded_vertex");
   nir_store_var(&b, cursor, nir_load_local_invocation_index(&b), 1);
   nir_push_loop(&b);
   nir_def *v = nir_load_var(&b, cursor);
   nir_push_if(&b, nir_uge(&b, v, vertices));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);
   nir_def *primitive = compact ? nir_bcsel(&b, nir_ult(&b, v, primitives), v, nir_imm_int(&b, 0)) :
                                 nir_udiv_imm(&b, v, corners);
   nir_def *corner = nir_umod_imm(&b, v, corners);
   nir_def *old_vertex = compact ? nir_load_deref(&b,
      nir_build_deref_array(&b, nir_build_deref_var(&b, vertex_map), v)) :
      bc250_index_corner(&b, indices, narrow_indices, corners, primitive, corner, corner_load);
   for (unsigned i = 0; i < count; i++) {
      if (source[i] == indices)
         continue;
      if (source[i]->data.location == VARYING_SLOT_CULL_PRIMITIVE) {
         /* This flag belongs to the primitive, never to a duplicated vertex. */
         nir_push_if(&b, nir_ieq_imm(&b, corner, 0));
         nir_copy_deref(&b,
                       nir_build_deref_array(&b, nir_build_deref_var(&b, output[i]), primitive),
                       nir_build_deref_array(&b, nir_build_deref_var(&b, source[i]), primitive));
         nir_pop_if(&b, NULL);
         continue;
      }
      nir_def *src_index = source[i]->data.per_primitive ? primitive : old_vertex;
      const struct glsl_type *element = glsl_get_array_element(source[i]->type);
      if (compact_written[i] != ~0u &&
          compact_written[i] != BITFIELD_MASK(MIN2(glsl_get_length(element), 32))) {
         /* Only the ClipDistance/CullDistance elements the application writes. */
         u_foreach_bit(e, compact_written[i]) {
            nir_copy_deref(&b,
                          nir_build_deref_array_imm(&b, nir_build_deref_array(&b,
                             nir_build_deref_var(&b, output[i]), v), e),
                          nir_build_deref_array_imm(&b, nir_build_deref_array(&b,
                             nir_build_deref_var(&b, source[i]), src_index), e));
         }
         continue;
      }
      nir_copy_deref(&b,
                    nir_build_deref_array(&b, nir_build_deref_var(&b, output[i]), v),
                    uniform_cell[i] ? nir_build_deref_var(&b, uniform_cell[i]) :
                    nir_build_deref_array(&b, nir_build_deref_var(&b, source[i]), src_index));
   }
   if (!compact) {
   nir_push_if(&b, nir_ieq_imm(&b, corner, 0));
   nir_def *new_idx[3];
   for (unsigned c = 0; c < corners; c++)
      new_idx[c] = nir_iadd_imm(&b, v, c);
   nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, new_indices), primitive),
                   nir_vec(&b, new_idx, corners), BITFIELD_MASK(corners));
   nir_pop_if(&b, NULL);
   }
   nir_store_var(&b, cursor, nir_iadd_imm(&b, v, mesh->info.workgroup_size[0] *
                                      mesh->info.workgroup_size[1] *
                                      mesh->info.workgroup_size[2]), 1);
   nir_pop_loop(&b, NULL);
   mesh->info.mesh.max_vertices_out = expanded;
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, mesh, nir_split_var_copies);
   NIR_PASS(_, mesh, nir_lower_var_copies);
   if (replaced->entries) {
      /* The arrays the staging parts replaced have dead derefs only. */
      NIR_PASS(_, mesh, nir_opt_dce);
      NIR_PASS(_, mesh, nir_remove_dead_derefs);
      const nir_remove_dead_variables_options remove = {
         .can_remove_var = bc250_is_replaced_staging,
         .can_remove_var_data = replaced,
      };
      NIR_PASS(_, mesh, nir_remove_dead_variables, nir_var_mem_shared, &remove);
   }
   NIR_PASS(_, mesh, nir_lower_vars_to_explicit_types, nir_var_mem_shared, bc250_shared_type);
   if (getenv("BC250_TRACE_STAGING")) {
      /* The LDS staging layout: name@offset+bytes of every shared variable. */
      fprintf(stderr, "BC250 MESH STAGING:");
      nir_foreach_variable_with_modes(var, mesh, nir_var_mem_shared)
         fprintf(stderr, " %s@%u+%u", var->name ? var->name : "?", var->data.driver_location,
                 glsl_get_explicit_size(var->type, false));
      fprintf(stderr, " shared_size=%u\n", mesh->info.shared_size);
   }
   if (index_staging) {
      nir_variable *staged = narrow_indices ? narrow_indices : indices;
      index_staging[0] = staged->data.driver_location;
      index_staging[2] = glsl_get_bit_size(glsl_without_array(staged->type)) / 8;
      index_staging[1] = narrow_indices ? corners * index_staging[2] : glsl_get_explicit_stride(staged->type);
      index_staging[3] = logical_vertices;
   }
   if ((direct_read & (RADV_BC250_DIRECT_READ_UNIFORM | RADV_BC250_DIRECT_READ_INDEX16)) &&
       getenv("BC250_TRACE_COMPILE")) {
      unsigned uniform = 0;
      for (unsigned i = 0; i < count; i++)
         uniform += uniform_cell[i] != NULL;
      fprintf(stderr, "BC250 MESH DIRECT READ staging: uniform_outputs=%u index16=%u corner=%u shared_size=%u\n",
              uniform, narrow_indices != NULL, corner_load, mesh->info.shared_size);
   }
   _mesa_set_destroy(replaced, NULL);
   NIR_PASS(_, mesh, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   NIR_PASS(_, mesh, nir_lower_vars_to_ssa);
   nir_shader_gather_info(mesh, impl);
   nir_foreach_shader_in_variable(var, fs) {
      if (var->data.per_primitive &&
          (var->data.location >= VARYING_SLOT_VAR0 ||
           var->data.location == VARYING_SLOT_PRIMITIVE_ID ||
           var->data.location == VARYING_SLOT_VIEWPORT ||
           var->data.location == VARYING_SLOT_LAYER)) {
         var->data.per_primitive = false;
         var->data.interpolation = INTERP_MODE_FLAT;
      }
   }
   nir_shader_gather_info(fs, nir_shader_get_entrypoint(fs));
   /* Without the compact vertex map (*compact_map false), primitive p uses the private
    * vertices corners * p .. corners * p + corners - 1: the terminal loop above is the
    * only writer of the primitive indices (RADV_BC250_MESH_IMPLICIT_TRIS). */
   return true;
}

/* Conservative PS input/PARAM capacity and two unused generic reference slots. */
bool
radv_bc250_bary_ref_slots(const nir_shader *producer, const nir_shader *fs, int *raw_slot, int *flat_slot)
{
   const uint64_t param_slots = BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32) | VARYING_BIT_LAYER |
      VARYING_BIT_VIEWPORT | VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1;
   const unsigned inputs = util_bitcount64(fs->info.inputs_read) + util_bitcount(fs->info.inputs_read_16bit);
   const unsigned params = util_bitcount64(producer->info.outputs_written & param_slots) + 1 +
      util_bitcount(producer->info.outputs_written_16bit);
   const uint64_t used = producer->info.outputs_written | producer->info.outputs_read | fs->info.inputs_read;
   *raw_slot = *flat_slot = -1;
   if (inputs + 2 > 32 || params + 2 > 32)
      return false;
   for (int i = 31; i >= 0; i--) {
      const unsigned slot = VARYING_SLOT_VAR0 + i;
      if (used & BITFIELD64_BIT(slot))
         continue;
      if (*raw_slot < 0)
         *raw_slot = slot;
      else {
         *flat_slot = slot;
         return true;
      }
   }
   *raw_slot = -1;
   return false;
}

/* RADV_BC250_BARY_CORNER_ID: one unused generic slot for the private-corner number, with exact
 * PARAM and PS input counts. */
bool
radv_bc250_bary_cid_slot(const nir_shader *producer, const nir_shader *fs, int *slot)
{
   const uint64_t param_slots = BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32) | VARYING_BIT_LAYER |
      VARYING_BIT_VIEWPORT | VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1;
   const unsigned inputs = util_bitcount64(fs->info.inputs_read) + util_bitcount(fs->info.inputs_read_16bit);
   const unsigned params = util_bitcount64(producer->info.outputs_written & param_slots) +
      util_bitcount(producer->info.outputs_written_16bit);
   const uint64_t used = producer->info.outputs_written | producer->info.outputs_read | fs->info.inputs_read;
   *slot = -1;
   if (inputs + 1 > 32 || params + 1 > 32)
      return false;
   for (int i = 31; i >= 0; i--) {
      if (!(used & BITFIELD64_BIT(VARYING_SLOT_VAR0 + i))) {
         *slot = VARYING_SLOT_VAR0 + i;
         return true;
      }
   }
   return false;
}

bool
radv_bc250_mesh_fs_refused(const struct radv_compiler_info *compiler_info, nir_shader *fs)
{
   if (radv_bc250_merge_fs_refusal(fs))
      return true;
   if (compiler_info->key.bc250_mesh_safe_bary) {
      /* PerVertexKHR is still a variable decoration before lowering I/O. A
       * shader can use it without ever reading a barycentric weight intrinsic. */
      nir_foreach_shader_in_variable(var, fs) {
         if (var->data.per_vertex)
            return true;
      }
      return radv_nir_bc250_fs_needs_bary_rotation(fs);
   }
   return false;
}

bool
radv_bc250_mesh_private_bary(const struct radv_compiler_info *compiler_info, nir_shader *fs)
{
   /* Affect only interfaces refused by the shared-corner planners. The caller
    * must additionally prove private corners, reference capacity and LDS fit. */
   return compiler_info->key.bc250_mesh_safe_bary && compiler_info->key.bc250_mesh_safe_owned &&
          compiler_info->key.bc250_mesh_safe_corners && compiler_info->hw.bc250_barycentrics &&
          !compiler_info->hw.bc250_bary_no_ref && fs && radv_bc250_mesh_fs_refused(compiler_info, fs);
}

/* Strict 16-bit vectors require the opt-in ACO subdword extraction fix.
 * Keep the older admission policy byte-identical when it is disabled. */
static bool
bc250_type_has_16bit(const struct glsl_type *type)
{
   if (glsl_type_is_array(type))
      return bc250_type_has_16bit(glsl_get_array_element(type));
   if (glsl_type_is_struct(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++)
         if (bc250_type_has_16bit(glsl_get_struct_field(type, i)))
            return true;
      return false;
   }
   return (glsl_type_is_vector_or_scalar(type) || glsl_type_is_matrix(type)) && glsl_get_bit_size(type) == 16;
}

static bool
bc250_mesh_has_16bit_outputs(const nir_shader *mesh)
{
   nir_foreach_shader_out_variable(var, mesh)
      if (bc250_type_has_16bit(var->type))
         return true;
   return mesh->info.outputs_written_16bit != 0;
}

/* Every count store is compiler-visible here. A constant count within the
 * declaration needs no runtime clamp after staging; the initialized count is
 * zero. Dynamic or excessive counts keep the clamp. */
static unsigned
bc250_mesh_static_primitive_bound(nir_shader *mesh)
{
   unsigned bound = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(mesh)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count &&
             (!nir_src_is_const(in->src[1]) ||
              nir_src_as_uint(in->src[1]) > mesh->info.mesh.max_primitives_out))
            return UINT_MAX;
         if (in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count)
            bound = MAX2(bound, nir_src_as_uint(in->src[1]));
      }
   }
   return bound;
}

/* The tiny specialization uses the ordinary export layout, not a W31 map.
 * Bound full 16-byte slots, including aggregate and wide values. Packed LDS or
 * successful direct reads can only reduce this allocation. */
static bool
bc250_safe_bary_tiny_candidate(nir_shader *mesh, bool last)
{
   unsigned slots = 0;
   nir_foreach_shader_out_variable(var, mesh) {
      if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
         continue;
      if (!glsl_type_is_array(var->type) || var->data.per_primitive)
         return false;
      slots += glsl_count_attribute_slots(glsl_get_array_element(var->type), false);
   }
   const unsigned corners = mesa_vertices_per_prim(mesh->info.mesh.primitive_type);
   const unsigned vertices = corners * mesh->info.mesh.max_primitives_out;
   const unsigned threads = mesh->info.workgroup_size[0] * mesh->info.workgroup_size[1] *
                            mesh->info.workgroup_size[2];
   const unsigned attribute_copies = radv_bc250_owned_lds_direct ? 0 : 16 * slots * vertices;
   return vertices == mesh->info.mesh.max_vertices_out && vertices <= ((corners == 3 || last) ? 256 : 32) && threads <= 256 &&
      align(mesh->info.shared_size, 16) + 128 + attribute_copies +
         3 * mesh->info.mesh.max_primitives_out < 30 * 1024;
}

/* A single invocation with unconditional, constant CullPrimitive stores can
 * discard an already-culled suffix before private-corner expansion. Preserve
 * the surviving primitive numbers and all application work. Dynamic culling,
 * holes, output readback and control flow keep the existing fallback. */
static bool
bc250_fold_culled_suffix(nir_shader *mesh)
{
   if (mesh->info.workgroup_size[0] * mesh->info.workgroup_size[1] *
          mesh->info.workgroup_size[2] != 1 || mesh->info.outputs_read)
      return false;
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   nir_intrinsic_instr *count = NULL;
   nir_variable *cull = NULL;
   nir_intrinsic_instr *stores[256] = {0};
   bool culled[256] = {0};
   nir_foreach_shader_out_variable(var, mesh)
      if (var->data.location == VARYING_SLOT_CULL_PRIMITIVE)
         cull = var;
   if (!cull)
      return false;
   nir_foreach_block(block, impl) {
      if (block->cf_node.parent != &impl->cf_node)
         return false;
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
            if (count || !nir_src_is_const(in->src[1]))
               return false;
            count = in;
         } else if (in->intrinsic == nir_intrinsic_store_deref) {
            nir_deref_instr *deref = nir_src_as_deref(in->src[0]);
            if (nir_deref_instr_get_variable(deref) != cull)
               continue;
            if (deref->deref_type != nir_deref_type_array ||
                nir_deref_instr_parent(deref)->deref_type != nir_deref_type_var ||
                !nir_src_is_const(deref->arr.index) || !nir_src_is_const(in->src[1]))
               return false;
            unsigned p = nir_src_as_uint(deref->arr.index);
            if (p >= ARRAY_SIZE(stores) || stores[p])
               return false;
            stores[p] = in;
            culled[p] = nir_src_as_uint(in->src[1]) != 0;
         }
      }
   }
   const unsigned primitives = count ? nir_src_as_uint(count->src[1]) : 0;
   if (!primitives || primitives > ARRAY_SIZE(stores))
      return false;
   unsigned prefix = 0;
   while (prefix < primitives && stores[prefix] && !culled[prefix])
      prefix++;
   if (prefix == primitives)
      return false;
   for (unsigned p = prefix; p < primitives; p++)
      if (!stores[p] || !culled[p])
         return false;
   nir_builder b = nir_builder_create(impl);
   b.cursor = nir_before_instr(&count->instr);
   nir_src_rewrite(&count->src[1], nir_imm_int(&b, prefix));
   for (unsigned p = 0; p < ARRAY_SIZE(stores); p++)
      if (stores[p])
         nir_instr_remove(&stores[p]->instr);
   exec_node_remove(&cull->node);
   NIR_PASS(_, mesh, nir_remove_dead_derefs);
   nir_shader_gather_info(mesh, impl);
   nir_progress(true, impl, nir_metadata_control_flow);
   return true;
}

/* Stage 2: reuse the owned-corner representation with bounded safe exports.
 * SAFE_CORNERS uses the expansion's private corners and direct-read staging:
 * the local survivor ballot replaces ownership construction and W31 repair.
 * Otherwise the first corner of every primitive has a distinct logical vertex;
 * the W31 planner may duplicate it, but cannot merge different payloads.
 * Work on clones: any budget/type exclusion leaves the existing route intact. */
bool
radv_bc250_mesh_safe_owned(nir_shader **mesh_ptr, nir_shader **fs_ptr, bool corners, bool bary, bool tiny, bool last, bool io16, bool piece, bool mesh_queries,
                            unsigned direct_read, uint32_t index_staging[4],
                            uint64_t *pp_locations)
{
   nir_shader *original = *mesh_ptr, *fragment = *fs_ptr;
   /* RADV_BC250_MESH_SAFE_SPLIT_PIECES: no fragment shader (rasterizer discard) exports against an
    * empty fragment shader (no varyings read, positions only); none is returned to the caller. */
   const bool empty_fs = !fragment && original && debug_get_bool_option("RADV_BC250_MESH_SAFE_SPLIT_PIECES", false);
   if (empty_fs) {
      fragment = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, original->options, "bc250_empty_fs").shader;
      ralloc_steal(original, fragment);
   }
   if (!original || !fragment || original->info.mesh.nv ||
       (original->info.task_payload_size && !(tiny && last)) ||
       (original->info.mesh.primitive_type != MESA_PRIM_TRIANGLES &&
        (!tiny || (original->info.mesh.primitive_type != MESA_PRIM_LINES &&
                   original->info.mesh.primitive_type != MESA_PRIM_POINTS))) ||
       !original->info.mesh.max_vertices_out || original->info.mesh.max_vertices_out > 256 ||
       !original->info.mesh.max_primitives_out || original->info.mesh.max_primitives_out > 256 ||
       (!tiny && 3 * original->info.mesh.max_primitives_out > 192) ||
       (original->info.outputs_read && !(tiny && last)) || (!tiny && original->info.outputs_written_16bit) ||
       original->info.shared_memory_explicit_layout || original->info.workgroup_size_variable ||
       (!bary && radv_bc250_merge_fs_refusal(fragment)) ||
       (bary && (!corners || (!tiny && original->info.mesh.max_primitives_out < 24) ||
                 (!tiny && original->info.mesh.max_primitives_out > original->info.min_subgroup_size))))
      return false;
   if (tiny && bc250_mesh_has_16bit_outputs(original) && !(last && io16))
      return false;
   const uint64_t allowed = VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES |
                            VARYING_BIT_PRIMITIVE_COUNT | VARYING_BIT_PRIMITIVE_ID |
                            ((bary || tiny) ? VARYING_BIT_PSIZ : 0) |
                            ((tiny && last) ? (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT | VARYING_BIT_CULL_PRIMITIVE | RADV_BC250_CLIPCULL_OUTPUTS) : 0) | (UINT64_C(0xffffffff) << VARYING_SLOT_VAR0);
   if (original->info.outputs_written & ~allowed)
      return false;
   nir_shader *mesh = nir_shader_clone(NULL, original);
   nir_shader *fs = nir_shader_clone(NULL, fragment);
   /* Keep the API primitive count when mesh-generated-primitive queries are enabled. */
   if ((mesh->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) &&
       !(tiny && last && !mesh_queries && bc250_fold_culled_suffix(mesh))) {
      ralloc_free(mesh);
      ralloc_free(fs);
      return false;
   }
   NIR_PASS(_, fs, nir_remove_dead_variables, nir_var_shader_in, NULL);
   NIR_PASS(_, mesh, nir_remove_unused_varyings, fs);
   nir_shader_gather_info(mesh, nir_shader_get_entrypoint(mesh));
   bool clamp_primitives = !tiny;
   if (tiny) {
      NIR_PASS(_, mesh, nir_lower_vars_to_ssa);
      NIR_PASS(_, mesh, nir_opt_copy_prop);
      NIR_PASS(_, mesh, nir_opt_constant_folding);
      const unsigned bound = bc250_mesh_static_primitive_bound(mesh);
      const unsigned capacity = 256u / mesa_vertices_per_prim(mesh->info.mesh.primitive_type);
      if (last && piece && mesh->info.mesh.max_primitives_out <= capacity) {
         /* radv_bc250_split_mesh already bounded this child count. */
         clamp_primitives = false;
      } else if (last && bound == UINT_MAX && mesh->info.mesh.max_primitives_out <= capacity &&
          (mesh->info.mesh.primitive_type != MESA_PRIM_TRIANGLES || mesh->info.mesh.max_primitives_out == 1)) {
         /* The same private-corner expansion also proves a dynamic count:
          * cap it to the declaration, then export exactly N*pc vertices. */
         clamp_primitives = true;
      } else {
         if (bound > (last ? capacity : 23u)) {
            ralloc_free(mesh);
            ralloc_free(fs);
            return false;
         }
         mesh->info.mesh.max_primitives_out = MAX2(1, bound);
      }
   }
   int raw_slot, flat_slot;
   int cid_slot;
   if (bary && mesh->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
       !debug_get_bool_option("RADV_BC250_BARY_PRIVATE_ROT0", false) &&
       !radv_bc250_bary_ref_slots(mesh, fs, &raw_slot, &flat_slot) &&
       !(corners && debug_get_bool_option("RADV_BC250_BARY_CORNER_ID", false) &&
         radv_bc250_bary_cid_slot(mesh, fs, &cid_slot))) {
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH SAFE BARY: declined reason=no_two_free_reference_parameters\n");
      ralloc_free(mesh);
      ralloc_free(fs);
      return false;
   }
   bool compact = false, over_budget = false;
   uint32_t staged[4] = {0};
   const uint64_t payload = mesh->info.per_primitive_outputs & (UINT64_C(0xffffffff) << VARYING_SLOT_VAR0);
   const bool transformed = (corners || radv_bc250_mesh_needs_expansion(mesh)) &&
      radv_bc250_expand_primitive_attributes(mesh, fs, !corners, true, corners ? direct_read : 0, &over_budget,
                                             NULL, staged, &compact, clamp_primitives);
   /* The general planner fits 3*P <= 192; the ordinary private exporter
    * admits N*P <= 256. The conservative LDS proof rules out a scratch ring. Private
    * bary corners may exceed 16 KiB API staging: the full conservative 32 KiB
    * packed sum (without assuming direct read) is the relevant bound. Position
    * aliases for the two reference parameters add no LDS records. */
   if (!transformed || !(tiny ? bc250_safe_bary_tiny_candidate(mesh, last) :
                                  bc250_safe_direct_candidate(mesh, true, 8, bary))) {
      ralloc_free(mesh);
      ralloc_free(fs);
      return false;
   }
   nir_validate_shader(mesh, "BC250 safe owned corners");
   nir_validate_shader(fs, "BC250 safe owned fragment inputs");
   if (empty_fs) {
      ralloc_free(fs);
      fs = NULL;
   } else {
      ralloc_free(fragment);
   }
   ralloc_free(original);
   *mesh_ptr = mesh;
   *fs_ptr = fs;
   memcpy(index_staging, staged, sizeof(staged));
   *pp_locations = payload;
   return true;
}

/* Experimental replay splitting for a deliberately restricted shader subset.
 * This is not a general mesh-production fallback. Query advertising is disabled
 * when the option is active, since replay would otherwise inflate invocations.
 */
static bool
bc250_split_reject(const char *reason)
{
   fprintf(stderr, "BC250 mesh split rejected: %s\n", reason);
   return false;
}

/* Keep staging capacity at the declared limit. Only the number of replayed
 * pieces uses this proof; runtime output counts are never clamped to a guess. */
static unsigned
bc250_primitive_count_bound(nir_shader *mesh)
{
   struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
   unsigned bound = 0;
   bool found = false;
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
               continue;
            found = true;
            bound = MAX2(bound, nir_unsigned_upper_bound(mesh, ranges,
               nir_scalar_chase_movs(nir_get_scalar(in->src[1].ssa, 0))));
         }
      }
   }
   _mesa_hash_table_destroy(ranges, NULL);
   return found ? MIN2(bound, mesh->info.mesh.max_primitives_out) : mesh->info.mesh.max_primitives_out;
}

/* Sink only lane-local arithmetic whose entire use set is this consumer.
 * Do not move memory/subgroup operations, phi nodes, or calculations shared
 * with other outputs. Limit recursion for adversarially deep expression trees. */
static unsigned
bc250_sink_output_alu(nir_def *value, nir_instr *consumer, nir_block *source_block, unsigned depth)
{
   nir_instr *instr = nir_def_instr(value);
   if (depth == 64 || instr->type != nir_instr_type_alu || instr->block != source_block)
      return 0;
   nir_foreach_use_including_if(use, value) {
      if (nir_src_is_if(use) || nir_src_use_instr(use) != consumer)
         return 0;
   }
   nir_instr_move(nir_before_instr(consumer), instr);
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   unsigned moved = 1;
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
      moved += bc250_sink_output_alu(alu->src[i].src.ssa, instr, source_block, depth + 1);
   return moved;
}

/* Group output stores sharing an index into one guarded region. Pure ALU
 * whose every use is in that region can then execute only in the owning
 * slice. No loads, barriers, phis or subgroup operations are moved. */
static bool
bc250_group_output_region(nir_function_impl *impl, nir_variable **sources,
                         nir_variable **targets, unsigned num_targets,
                         nir_def *base, unsigned limit)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(start, block) {
         if (start->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(start)->intrinsic != nir_intrinsic_store_deref)
            continue;
         nir_intrinsic_instr *stores[128];
         nir_variable *destinations[128];
         unsigned count = 0;
         nir_def *index = NULL;
         bool active = false;
         nir_foreach_instr(instr, block) {
            active |= instr == start;
            if (!active)
               continue;
            if (instr->type == nir_instr_type_alu || instr->type == nir_instr_type_deref ||
                instr->type == nir_instr_type_load_const || instr->type == nir_instr_type_undef)
               continue;
            if (instr->type != nir_instr_type_intrinsic)
               break;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic != nir_intrinsic_store_deref || count == ARRAY_SIZE(stores))
               break;
            nir_deref_instr *d = nir_src_as_deref(in->src[0]);
            if (d->deref_type != nir_deref_type_array ||
                nir_deref_instr_parent(d)->deref_type != nir_deref_type_var)
               break;
            nir_variable *var = nir_deref_instr_get_variable(d), *target = NULL;
            for (unsigned i = 0; i < num_targets; i++)
               if (sources[i] == var) target = targets[i];
            if (!target || (index && index != d->arr.index.ssa))
               break;
            index = d->arr.index.ssa;
            stores[count] = in;
            destinations[count++] = target;
         }
         if (count < 2)
            continue;
         nir_builder b = nir_builder_create(impl);
         b.cursor = nir_before_instr(&stores[count - 1]->instr);
         nir_def *slot = nir_isub(&b, index, base);
         nir_push_if(&b, nir_ult_imm(&b, slot, limit));
         nir_block *region = NULL;
         for (unsigned i = 0; i < count; i++) {
            nir_store_deref(&b, nir_build_deref_array(&b,
               nir_build_deref_var(&b, destinations[i]), slot), stores[i]->src[1].ssa,
               nir_intrinsic_write_mask(stores[i]));
            region = nir_cursor_current_block(b.cursor);
         }
         nir_pop_if(&b, NULL);
         for (unsigned i = 0; i < count; i++)
            nir_instr_remove(&stores[i]->instr);
         unsigned moved = 0;
         nir_foreach_instr_reverse_safe(instr, nir_def_instr(slot)->block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_def *def = nir_instr_def(instr);
            bool eligible = !nir_def_is_unused(def);
            nir_foreach_use_including_if(use, def) {
               if (nir_src_is_if(use) || nir_src_use_instr(use)->block != region)
                  eligible = false;
            }
            if (eligible) {
               nir_instr_move(nir_before_block(region), instr);
               moved++;
            }
         }
         if (getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 output region: stores=%u ALU=%u\n", count, moved);
         return true;
      }
   }
   return false;
}

/* Byte offset of bc250_constants.split_piece (static_assert next to the struct). */
#define BC250_SPLIT_PIECE_OFFSET 68

/* The direct split's flat launch index: piece = index % pieces, API workgroup
 * = index / pieces. Direct draws and the split-argument setup launch pieces
 * workgroups per API workgroup, so it is WorkgroupIndex. With piece_select
 * (RADV_BC250_SPLIT_PREP_FREE) a prep-free split indirect draw launches the
 * application's own grid once per piece and passes split_piece = piece + 1;
 * 0 (every other draw of the pipeline) keeps WorkgroupIndex. */
static nir_def *
bc250_split_flat_index(nir_builder *b, unsigned pieces, bool piece_select)
{
   nir_def *index = nir_load_workgroup_index(b);
   if (!piece_select)
      return index;
   nir_def *select = nir_load_push_constant(b, 1, 32, nir_imm_int(b, BC250_SPLIT_PIECE_OFFSET), .range = 4);
   return nir_bcsel(b, nir_ieq_imm(b, select, 0), index,
                    nir_iadd(b, nir_imul_imm(b, index, pieces), nir_iadd_imm(b, select, -1)));
}

/* Conservative: any intrinsic that may read buffer, global or image memory
 * (tex instructions are checked by the caller). */
static bool
bc250_reads_external_memory(const nir_intrinsic_instr *in)
{
   const char *name = nir_intrinsic_infos[in->intrinsic].name;
   if (in->intrinsic == nir_intrinsic_load_deref)
      return nir_deref_mode_may_be(nir_src_as_deref(in->src[0]),
                                   nir_var_mem_ssbo | nir_var_mem_global | nir_var_mem_ubo | nir_var_image);
   return !strncmp(name, "load_ssbo", 9) || !strncmp(name, "load_ubo", 8) ||
          !strncmp(name, "load_global", 11) || !strncmp(name, "load_buffer", 11) ||
          (strstr(name, "image") && strstr(name, "load"));
}

static bool
bc250_split_primitive_output(nir_variable *var)
{
   return var->data.per_primitive || var->data.location == VARYING_SLOT_PRIMITIVE_INDICES;
}

static bool
bc250_split_reject_output(const char *what, const nir_variable *var)
{
   fprintf(stderr, "BC250 mesh split rejected: %s %s\n", what,
           gl_varying_slot_name_for_stage(var->data.location, MESA_SHADER_MESH));
   return false;
}

/* Per-vertex built-in outputs of an EXT Mesh shader (the others are
 * per-primitive). Every piece re-executes the whole body and writes all its
 * vertices, and the split only slices the primitive arrays, so per-vertex
 * outputs pass through unchanged like the position: ClipDistance and
 * CullDistance (one compact array at CLIP_DIST0/1 after
 * nir_merge_clip_cull_distance_vars, CULL_DIST0/1 before it) and PointSize.
 * The expansion copies them per expanded vertex like any other output. A
 * primitive whose three corners have the same cull distance negative is culled
 * by the hardware from the exported values, so the private copies keep that
 * decision exactly. */
static bool
bc250_split_vertex_builtin(gl_varying_slot location)
{
   switch (location) {
   case VARYING_SLOT_POS:
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CLIP_DIST1:
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_CULL_DIST1:
      return true;
   default:
      return false;
   }
}

/* Per-primitive built-in outputs the split slices like generic per-primitive
 * attributes: PrimitiveId, Layer and ViewportIndex, which the expansion turns
 * into flat per-vertex outputs of every private vertex (as for unsplit Mesh
 * shaders). A fragment shader that reads them is still refused above (the
 * fragment primitive system input checks). The primitive shading rate stays
 * refused: GFX10.1 has no VRS and the expansion does not handle it. */
static bool
bc250_split_primitive_builtin(gl_varying_slot location)
{
   return location == VARYING_SLOT_PRIMITIVE_ID || location == VARYING_SLOT_LAYER ||
          location == VARYING_SLOT_VIEWPORT;
}

/* RADV_BC250_MESH_NESTED_SLICE: the outermost array dereference directly under a
 * per-primitive output variable (its primitive index), or NULL. */
static nir_deref_instr *
bc250_slice_prim_deref(nir_deref_instr *d)
{
   nir_deref_instr *child = NULL;
   for (nir_deref_instr *it = d; it; it = nir_deref_instr_parent(it)) {
      if (it->deref_type == nir_deref_type_var)
         return child && child->deref_type == nir_deref_type_array ? child : NULL;
      if (it->deref_type != nir_deref_type_array && it->deref_type != nir_deref_type_struct)
         return NULL;
      child = it;
   }
   return NULL;
}

/* Rebuild d with the variable replaced by target and the primitive index by slot. */
static nir_deref_instr *
bc250_slice_rebuild_deref(nir_builder *b, nir_deref_instr *d, nir_variable *target, nir_def *slot)
{
   nir_deref_instr *parent = nir_deref_instr_parent(d);
   if (parent->deref_type == nir_deref_type_var)
      return nir_build_deref_array(b, nir_build_deref_var(b, target), slot);
   nir_deref_instr *p = bc250_slice_rebuild_deref(b, parent, target, slot);
   if (d->deref_type == nir_deref_type_struct)
      return nir_build_deref_struct(b, p, d->strct.index);
   return nir_build_deref_array(b, p, d->arr.index.ssa);
}

__thread bool radv_bc250_split_refused_retry;
__thread bool radv_bc250_split_piece_primid;
__thread bool radv_bc250_split_task_grid_fold;
__thread bool radv_bc250_owned_lds_direct;

/* RADV_BC250_MESH_PIECE_PRIMID: the Mesh shader writes PrimitiveId as one scalar 32-bit integer per
 * primitive. The split slices it like any other per-primitive output, so each piece exports the value
 * the application wrote. */
static bool
bc250_split_primid_sliceable(const nir_shader *mesh)
{
   if (!(mesh->info.outputs_written & mesh->info.per_primitive_outputs & VARYING_BIT_PRIMITIVE_ID))
      return false;
   unsigned found = 0;
   nir_foreach_shader_out_variable(var, mesh) {
      if (var->data.location != VARYING_SLOT_PRIMITIVE_ID)
         continue;
      const struct glsl_type *elem = glsl_type_is_array(var->type) ? glsl_get_array_element(var->type) : NULL;
      if (!var->data.per_primitive || !elem || !glsl_type_is_scalar(elem) || !glsl_type_is_32bit(elem) ||
          !glsl_type_is_integer(elem) || glsl_get_length(var->type) != mesh->info.mesh.max_primitives_out)
         return false;
      found++;
   }
   return found == 1;
}

bool
radv_bc250_split_mesh(nir_shader *mesh, nir_shader *task, nir_shader *fs, bool direct_split, bool piece_select,
                      bool balanced_slices, bool parallel_cull, bool output_regions, bool compact_lds, unsigned piece_ceiling, unsigned min_pieces, unsigned *pieces_out)
{
   /* piece_ceiling: the per-piece primitive ceiling, RADV_BC250_PERF_PIECE_PRIMS
    * (compiler key, default 63). Performance A/B: up to 85, the size the base
    * driver already runs unsplit with full expansion (3 x 85 = 255 vertices).
    * Fewer pieces re-execute the Mesh body fewer times (a 128-primitive
    * Nanite cluster: 2 instead of 3).
    * min_pieces: BC250 Mesh LDS fit retry (radv_graphics_pipeline_init): the
    * previous attempt's expanded pieces did not fit in LDS, lower the ceiling
    * until the proven bound needs at least this many (smaller) pieces. */
   const unsigned declared = mesh->info.mesh.max_primitives_out;
   const unsigned bound = bc250_primitive_count_bound(mesh);
   unsigned old_limit = MIN2(CLAMP(piece_ceiling, 1, 85), MAX2(1, declared));
   if (min_pieces > 1 && DIV_ROUND_UP(MAX2(1, bound), old_limit) < min_pieces)
      old_limit = MAX2(1, DIV_ROUND_UP(MAX2(1, bound), min_pieces));
   const unsigned pieces = DIV_ROUND_UP(MAX2(1, bound), old_limit);
   /* Keep the qualified child count and logical arrays; only reduce physical
    * child capacity from a proven bound. The 63-primitive ceiling remains. */
   const unsigned limit = balanced_slices ? DIV_ROUND_UP(MAX2(1, bound), pieces) : old_limit;
   if ((min_pieces > 1 || piece_ceiling != RADV_BC250_DEFAULT_PIECE_PRIMS) &&
       debug_get_bool_option("BC250_TRACE_COMPILE", false))
      fprintf(stderr, "BC250 split piece ceiling: ceiling=%u min_pieces=%u bound=%u pieces=%u primitives_per_piece=%u\n",
              piece_ceiling, min_pieces, bound, pieces, limit);
   /* RADV_BC250_MESH_SPLIT_ANY (default 0): also split line and point Mesh shaders and Mesh shaders
    * without a fragment shader (rasterizer discard) behind a Task stage. The pieces copy the index array element type
    * as it is, so lines/points keep their index width. Only shapes refused before are affected. */
   /* Task pipelines only (real or the internal mesh-only amplification producer): Mesh-only
    * shapes keep their existing direct routes, which the split would otherwise pre-empt. */
   const bool split_any = (task || radv_bc250_split_refused_retry) &&
                          debug_get_bool_option("RADV_BC250_MESH_SPLIT_ANY", false);
   const bool topo_ok = mesh->info.mesh.primitive_type == MESA_PRIM_TRIANGLES ||
      (split_any && (mesh->info.mesh.primitive_type == MESA_PRIM_LINES ||
                     mesh->info.mesh.primitive_type == MESA_PRIM_POINTS));
   if (!topo_ok || mesh->info.mesh.nv ||
       mesh->info.mesh.max_primitives_out > 256 || pieces < 1 || pieces > 5 || (!fs && !split_any))
      return bc250_split_reject("topology/capacity/stage");
   uint64_t primitive_inputs = VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT;
   if (radv_bc250_split_piece_primid && direct_split && !task && fs &&
       (fs->info.inputs_read & VARYING_BIT_PRIMITIVE_ID) && bc250_split_primid_sliceable(mesh)) {
      primitive_inputs &= ~VARYING_BIT_PRIMITIVE_ID;
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 MESH PIECE PRIMID: written PrimitiveId sliced with the pieces\n");
   }
   if (fs && fs->info.inputs_read & primitive_inputs)
      return bc250_split_reject("fragment primitive system input");

   /* SPIR-V PrimitiveId inputs are system values, not ordinary varyings. */
   if (fs && BITSET_TEST(fs->info.system_values_read, SYSTEM_VALUE_PRIMITIVE_ID))
      return bc250_split_reject("fragment primitive ID system value");
   if (fs) nir_foreach_variable_with_modes(var, fs, nir_var_system_value) {
      if (var->data.location == SYSTEM_VALUE_PRIMITIVE_ID)
         return bc250_split_reject("fragment primitive ID system variable");
   }
   if (fs) nir_foreach_function_impl(fimpl, fs) {
      nir_foreach_block(block, fimpl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_primitive_id)
               return bc250_split_reject("fragment primitive ID intrinsic");
         }
      }
   }

   nir_variable *old_indices = NULL;
   nir_variable *old_cull = NULL;
   nir_variable *old_attrs[128], *new_attrs[128];
   unsigned attr_count = 0;
   nir_foreach_shader_out_variable(var, mesh) {
      if (!glsl_type_is_array(var->type))
         return bc250_split_reject("non-array output");
      if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES) {
         old_indices = var;
      } else if (var->data.per_primitive && var->data.location == VARYING_SLOT_CULL_PRIMITIVE) {
         if (old_cull)
            return bc250_split_reject("duplicate cull output");
         old_cull = var;
      } else if (var->data.per_primitive) {
         if (attr_count == ARRAY_SIZE(old_attrs))
            return bc250_split_reject("primitive attribute capacity");
         if (var->data.location < VARYING_SLOT_VAR0 && !bc250_split_primitive_builtin(var->data.location))
            return bc250_split_reject_output("special primitive output", var);
         old_attrs[attr_count++] = var;
      } else if (var->data.location < VARYING_SLOT_VAR0 && !bc250_split_vertex_builtin(var->data.location)) {
         return bc250_split_reject_output("special vertex output", var);
      }
   }
   if (!old_indices)
      return bc250_split_reject("missing primitive indices");

   /* The application's shared memory arrives already lowered to explicit
    * offsets [0, info.shared_size) (radv_shader_spirv_to_nir). The split and
    * the expansion lay out their own shared variables with
    * nir_lower_vars_to_explicit_types, which starts at info.shared_size, so
    * the two regions are disjoint. An explicit (block) layout instead goes
    * through nir_assign_shared_var_locations, which only handles interface
    * blocks and cannot place the split's variables: refuse it.
    */
   if (mesh->info.shared_memory_explicit_layout)
      return bc250_split_reject("explicit shared memory layout");

   /* Check before rewriting. Reject external writes/atomics, non-repeatable
    * values, and system values whose reconstruction is not implemented here.
    *
    * Every piece re-executes the same deterministic body in its own
    * workgroup, so each piece leaves the same values in its own shared
    * memory: non-atomic shared loads and stores and shared barriers are
    * replica-safe. Shared atomics are not (a used result depends on the
    * hardware serialization, which differs between the pieces), nor are
    * atomic shared loads/stores, which observe an order between waves.
    * NumWorkGroups is rewritten to the application grid below.
    *
    * An external (buffer, global or image) atomic whose result is unused is
    * write-only: it runs in the first piece only, exactly once per original
    * workgroup, with the operands that piece computes like the original
    * body. That holds only while no piece can observe it: refuse any body
    * that also reads external memory (another piece may or may not see the
    * first piece's write; bindings may alias).
    */
   bool external_atomics = false, reads_external = false;
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               return bc250_split_reject("uninlined call");
            if (instr->type == nir_instr_type_tex)
               reads_external = true;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            const char *name = nir_intrinsic_infos[in->intrinsic].name;
            reads_external |= bc250_reads_external_memory(in);
            if (nir_intrinsic_has_atomic_op(in) && nir_intrinsic_writes_external_memory(in)) {
               if (nir_intrinsic_has_access(in) && (nir_intrinsic_access(in) & ACCESS_VOLATILE))
                  return bc250_split_reject("volatile access");
               if (!nir_def_is_unused(&in->def))
                  return bc250_split_reject("external atomic result used");
               external_atomics = true;
               continue;
            }
            if (strstr(name, "atomic") || strstr(name, "clock") ||
                strstr(name, "global_invocation_id") ||
                (strstr(name, "num_workgroups") && in->intrinsic != nir_intrinsic_load_num_workgroups) ||
                strstr(name, "base_workgroup_id"))
               return bc250_split_reject(name);
            if (nir_intrinsic_has_access(in) && (nir_intrinsic_access(in) & ACCESS_VOLATILE))
               return bc250_split_reject("volatile access");
            if (in->intrinsic == nir_intrinsic_load_shared || in->intrinsic == nir_intrinsic_store_shared) {
               if (nir_intrinsic_access(in) & ACCESS_ATOMIC)
                  return bc250_split_reject("atomic shared load/store");
               continue;
            }
            if (in->intrinsic == nir_intrinsic_store_deref || in->intrinsic == nir_intrinsic_copy_deref) {
               nir_deref_instr *dst = nir_src_as_deref(in->src[0]);
               if (dst->modes & ~(nir_var_shader_out | nir_var_mem_shared |
                                  nir_var_shader_temp | nir_var_function_temp))
                  return bc250_split_reject("external deref store");
               continue;
            }
            if (in->intrinsic == nir_intrinsic_barrier) {
               if (nir_intrinsic_memory_modes(in) & ~(nir_var_mem_shared | nir_var_shader_out))
                  return bc250_split_reject("external memory barrier");
               continue;
            }
            if (in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count)
               continue;
            if (!(nir_intrinsic_infos[in->intrinsic].flags & NIR_INTRINSIC_CAN_ELIMINATE))
               return bc250_split_reject(name);
         }
      }
   }
   if (external_atomics && reads_external)
      return bc250_split_reject("external atomic with external memory reads");

   /* Restrict task launches to proven bounded one-dimensional grids. Never
    * clamp a valid application's launch to make it fit this experiment.
    */
   if (!task && ((!direct_split && pieces != 1) || mesh->info.task_payload_size))
      return bc250_split_reject("mesh-only amplification/payload unsupported");
   /* RADV_BC250_TASK_GRID_FOLD: per pipeline, the Mesh records are (pieces, x, y) instead of
    * (x * pieces, y, z), so a launch count without a compile-time bound needs only the API bound
    * x <= 65535. The flat launch order is unchanged. */
   bool fold = false;
   if (task) {
      struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
      bool launch_ok = true, has_launch = false;
      nir_foreach_function_impl(impl, task) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
               if (in->intrinsic != nir_intrinsic_launch_mesh_workgroups)
                  continue;
               has_launch = true;
               /* A constant zero dimension launches no Mesh workgroups at all: nothing to bound. */
               bool empty = false;
               for (unsigned c = 0; c < 3; c++) {
                  nir_scalar dim = nir_scalar_chase_movs(nir_get_scalar(in->src[0].ssa, c));
                  empty |= nir_scalar_is_const(dim) && nir_scalar_as_uint(dim) == 0;
               }
               if (empty)
                  continue;
               for (unsigned c = 1; c < 3; c++) {
                  nir_scalar dim = nir_scalar_chase_movs(nir_get_scalar(in->src[0].ssa, c));
                  /* RADV_BC250_MESH_SPLIT_ANY: y/z pass through unchanged (only x is multiplied by
                   * the piece count), so they only need the per-dimension hardware bound. */
                  if (debug_get_bool_option("RADV_BC250_MESH_SPLIT_ANY", false))
                     launch_ok &= nir_unsigned_upper_bound(task, ranges, dim) <= 65535;
                  else
                     launch_ok &= nir_scalar_is_const(dim) && nir_scalar_as_uint(dim) == 1;
               }
               uint32_t upper = nir_unsigned_upper_bound(task, ranges, nir_scalar_chase_movs(nir_get_scalar(in->src[0].ssa, 0)));
               const bool x_ok = upper <= 65535 / pieces;
               nir_scalar zs = nir_scalar_chase_movs(nir_get_scalar(in->src[0].ssa, 2));
               if (!x_ok && radv_bc250_split_task_grid_fold && !direct_split &&
                   nir_scalar_is_const(zs) && nir_scalar_as_uint(zs) == 1)
                  fold = true;
               else
                  launch_ok &= x_ok;
            }
         }
      }
      _mesa_hash_table_destroy(ranges, NULL);
      if (!launch_ok || !has_launch) {
         nir_print_shader(task, stderr);
         return bc250_split_reject("unproven task grid bound");
      }
      if (fold && getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 TASK GRID FOLD: pieces=%u records=(pieces,x,y) x_bound=api65535\n", pieces);

   }

   NIR_PASS(_, mesh, nir_lower_returns);
   /* The application's shared variables no longer have derefs; drop them so
    * the explicit layout below does not place a second, unused copy of the
    * application region after info.shared_size. */
   NIR_PASS(_, mesh, nir_remove_dead_variables, nir_var_mem_shared, NULL);
   const unsigned application_shared = mesh->info.shared_size;
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   nir_builder b = nir_builder_create(impl);

   /* Rewrite only original loads. A separate raw ID for the private chunk is
    * inserted afterward, so it cannot accidentally get divided twice.
    */
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic == nir_intrinsic_load_num_workgroups) {
            /* The application grid. direct_split: the draw's recorded
             * dimensions (the native grid is reshaped). Otherwise x was
             * multiplied by pieces (task launch rewrite below; one piece
             * keeps the native grid). */
            b.cursor = nir_before_instr(instr);
            nir_def *grid;
            if (direct_split) {
               nir_def *input = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 32), .range = 8);
               nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 16), .range = 4);
               nir_def *offset = nir_imul(&b, nir_u2u64(&b, nir_load_draw_id(&b)), nir_u2u64(&b, stride));
               grid = nir_load_global(&b, 3, 32, nir_iadd(&b, input, offset), .align_mul = 4);
            } else if (fold) {
               nir_def *raw = nir_load_num_workgroups(&b);
               grid = nir_vec3(&b, nir_channel(&b, raw, 1), nir_channel(&b, raw, 2), nir_imm_int(&b, 1));
            } else {
               nir_def *raw = nir_load_num_workgroups(&b);
               grid = nir_vec3(&b, nir_udiv_imm(&b, nir_channel(&b, raw, 0), pieces),
                               nir_channel(&b, raw, 1), nir_channel(&b, raw, 2));
            }
            nir_def_rewrite_uses(&in->def, nir_trim_vector(&b, grid, in->def.num_components));
            nir_instr_remove(instr);
            continue;
         }
         if (in->intrinsic != nir_intrinsic_load_workgroup_id)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_def *id;
         if (direct_split) {
            /* The native grid is reshaped to keep every dimension <=65535.
             * Recover original row-major identity from the flat group index. */
            nir_def *linear = nir_udiv_imm(&b, bc250_split_flat_index(&b, pieces, piece_select), pieces);
            nir_def *input = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 32), .range = 8);
            nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 16), .range = 4);
            nir_def *offset = nir_imul(&b, nir_u2u64(&b, nir_load_draw_id(&b)), nir_u2u64(&b, stride));
            nir_def *dims = nir_load_global(&b, 3, 32, nir_iadd(&b, input, offset), .align_mul = 4);
            nir_def *x = nir_channel(&b, dims, 0), *y = nir_channel(&b, dims, 1);
            nir_def *xy = nir_imul(&b, x, y);
            nir_def *zid = nir_udiv(&b, linear, xy);
            nir_def *rest = nir_isub(&b, linear, nir_imul(&b, zid, xy));
            nir_def *yid = nir_udiv(&b, rest, x);
            id = nir_vec3(&b, nir_isub(&b, rest, nir_imul(&b, yid, x)), yid, zid);
         } else if (fold) {
            /* Record (pieces, X, Y): flat index L = piece + pieces * (x + X * y). */
            nir_def *a = nir_udiv_imm(&b, nir_load_workgroup_index(&b), pieces);
            nir_def *X = nir_channel(&b, nir_load_num_workgroups(&b), 1);
            nir_def *ay = nir_udiv(&b, a, X);
            id = nir_vec3(&b, nir_isub(&b, a, nir_imul(&b, ay, X)), ay, nir_imm_int(&b, 0));
         } else {
            nir_def *raw = nir_load_workgroup_id(&b);
            id = nir_vec3(&b, nir_udiv_imm(&b, nir_channel(&b, raw, 0), pieces),
                           nir_channel(&b, raw, 1), nir_channel(&b, raw, 2));
         }
         nir_def_rewrite_uses(&in->def, nir_trim_vector(&b, id, in->def.num_components));
         nir_instr_remove(instr);
      }
   }

   /* RADV_BC250_MESH_COMPACT_LDS: a primitive output written one component at a
    * time (Hellblade 2's Nanite shaders) is a store through a vector-component
    * deref, which the slice stores below cannot redirect, so the whole
    * 128-entry index and attribute arrays would be staged in shared memory and
    * copied. Turn those stores into write-masked vector stores first (same
    * values, same order). */
   if (compact_lds)
      NIR_PASS(_, mesh, nir_lower_array_deref_of_vec, nir_var_shader_out, bc250_split_primitive_output,
               nir_lower_direct_array_deref_of_vec_store);

   nir_variable *indices = nir_variable_clone(old_indices, mesh);
   indices->type = glsl_array_type(glsl_get_array_element(old_indices->type), limit, 0);
   nir_shader_add_variable(mesh, indices);
   old_indices->data.mode = nir_var_mem_shared;
   old_indices->data.compact = false;
   for (unsigned i = 0; i < attr_count; i++) {
      new_attrs[i] = nir_variable_clone(old_attrs[i], mesh);
      new_attrs[i]->type = glsl_array_type(glsl_get_array_element(old_attrs[i]->type), limit, 0);
      nir_shader_add_variable(mesh, new_attrs[i]);
      old_attrs[i]->data.mode = nir_var_mem_shared;
      old_attrs[i]->data.compact = false;
   }
   if (old_cull) {
      old_cull->data.mode = nir_var_mem_shared;
      old_cull->data.compact = false;
   }
   nir_fixup_deref_modes(mesh);
   nir_variable *count = nir_variable_create(mesh, nir_var_mem_shared, glsl_uint_type(), "bc250_split_count");
   nir_variable *vertex_count = old_cull ?
      nir_variable_create(mesh, nir_var_mem_shared, glsl_uint_type(), "bc250_split_vertex_count") : NULL;
   b.cursor = nir_before_cf_list(&impl->body);
   nir_def *raw_x = direct_split ? bc250_split_flat_index(&b, pieces, piece_select) :
                    fold ? nir_load_workgroup_index(&b) :
                                   nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *base = nir_imul_imm(&b, nir_umod_imm(&b, raw_x, pieces), limit);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_var(&b, count, nir_imm_int(&b, 0), 1);
   if (vertex_count)
      nir_store_var(&b, vertex_count, nir_imm_int(&b, 0), 1);
   nir_pop_if(&b, NULL);

   if (external_atomics) {
      /* Write-only external atomics run in the first piece (base 0) only. */
      nir_def *first_piece = nir_ieq_imm(&b, base, 0);
      struct util_dynarray atomics;
      util_dynarray_init(&atomics, NULL);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (nir_intrinsic_has_atomic_op(in) && nir_intrinsic_writes_external_memory(in))
               util_dynarray_append(&atomics, in);
         }
      }
      util_dynarray_foreach(&atomics, nir_intrinsic_instr *, it) {
         nir_instr *instr = &(*it)->instr;
         assert(nir_def_is_unused(&(*it)->def));
         b.cursor = nir_before_instr(instr);
         nir_instr_remove(instr);
         nir_push_if(&b, first_piece);
         nir_builder_instr_insert(&b, instr);
         nir_pop_if(&b, NULL);
      }
      if (debug_get_bool_option("BC250_TRACE_COMPILE", false))
         fprintf(stderr, "BC250 split external atomics: %u guarded to the first piece\n",
                 (unsigned)util_dynarray_num_elements(&atomics, nir_intrinsic_instr *));
      util_dynarray_fini(&atomics);
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic == nir_intrinsic_barrier &&
             (nir_intrinsic_memory_modes(in) & nir_var_shader_out))
            nir_intrinsic_set_memory_modes(in, nir_intrinsic_memory_modes(in) | nir_var_mem_shared);
         if (in->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_def *original = in->src[1].ssa;
         nir_def *remaining = nir_isub(&b, original, nir_umin(&b, original, base));
         nir_def *pc = nir_umin(&b, remaining, nir_imm_int(&b, limit));
         pc = nir_bcsel(&b, nir_ieq_imm(&b, in->src[0].ssa, 0), nir_imm_int(&b, 0), pc);
         nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
         nir_store_var(&b, count, pc, 1);
         if (vertex_count)
            nir_store_var(&b, vertex_count, in->src[0].ssa, 1);
         nir_pop_if(&b, NULL);
         if (old_cull)
            nir_instr_remove(instr);
         else
            nir_src_rewrite(&in->src[1], pc);
      }
   }
   /* Simple per-primitive stores can target this slice directly. Keep the
    * old staging/copy path for reads, copies, nested dereferences and culling. */
   bool slice_stores = !old_cull;
   /* RADV_BC250_MESH_NESTED_SLICE (default 0): stores into struct/array members of a primitive
    * (e.g. loc[prim].elements[i] = v) also target the slice, so the full-size per-primitive
    * arrays are never kept in LDS. */
   const bool nested_slice = debug_get_bool_option("RADV_BC250_MESH_NESTED_SLICE", false);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic != nir_intrinsic_store_deref &&
             in->intrinsic != nir_intrinsic_load_deref &&
             in->intrinsic != nir_intrinsic_copy_deref)
            continue;
         unsigned sources = in->intrinsic == nir_intrinsic_copy_deref ? 2 : 1;
         for (unsigned j = 0; j < sources; j++) {
            nir_deref_instr *d = nir_src_as_deref(in->src[j]);
            nir_variable *var = nir_deref_instr_get_variable(d);
            bool selected = var == old_indices;
            for (unsigned i = 0; i < attr_count; i++)
               selected |= var == old_attrs[i];
            const bool simple = d->deref_type == nir_deref_type_array &&
                                nir_deref_instr_parent(d)->deref_type == nir_deref_type_var;
            const bool nested = nested_slice && bc250_slice_prim_deref(d);
            if (selected && (in->intrinsic != nir_intrinsic_store_deref || !(simple || nested)))
               slice_stores = false;
         }
      }
   }
   if (slice_stores && output_regions) {
      nir_variable *sources[129], *targets[129];
      sources[0] = old_indices; targets[0] = indices;
      for (unsigned i = 0; i < attr_count; i++) {
         sources[i + 1] = old_attrs[i]; targets[i + 1] = new_attrs[i];
      }
      while (bc250_group_output_region(impl, sources, targets, attr_count + 1, base, limit)) {}
   }
   if (slice_stores) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_deref_instr *d = nir_src_as_deref(in->src[0]);
            nir_variable *var = nir_deref_instr_get_variable(d);
            nir_variable *target = var == old_indices ? indices : NULL;
            for (unsigned i = 0; i < attr_count; i++)
               if (var == old_attrs[i]) target = new_attrs[i];
            if (!target)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_deref_instr *prim_deref = bc250_slice_prim_deref(d);
            nir_def *slot = nir_isub(&b, prim_deref->arr.index.ssa, base);
            /* Unsigned subtraction rejects indices below base too. */
            nir_push_if(&b, nir_ult_imm(&b, slot, limit));
            nir_store_deref(&b, bc250_slice_rebuild_deref(&b, d, target, slot),
                            in->src[1].ssa, nir_intrinsic_write_mask(in));
            nir_instr *slice_store = nir_block_last_instr(nir_cursor_current_block(b.cursor));
            nir_def *value = in->src[1].ssa;
            nir_block *source_block = nir_def_instr(slot)->block;
            nir_pop_if(&b, NULL);
            nir_instr_remove(instr);
            unsigned sunk = bc250_sink_output_alu(value, slice_store, source_block, 0);
            if (sunk && debug_get_bool_option("BC250_TRACE_COMPILE", false))
               fprintf(stderr, "BC250 slice arithmetic sunk: %u instructions\n", sunk);
         }
      }
   }
   b.cursor = nir_after_cf_list(&impl->body);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   nir_def *pc = nir_load_var(&b, count);
   if (old_cull) {
      /* Consume cull flags in shared memory rather than exporting native null
       * primitives. Preserve source order and move every generic attribute with
       * its index tuple. Required expansion later moves these temporary output
       * writes to shared staging; actual exports follow the final output count.
       */
      const unsigned wg_size = mesh->info.workgroup_size[0] * mesh->info.workgroup_size[1] *
                               mesh->info.workgroup_size[2];
      if (parallel_cull && wg_size >= limit) {
         nir_variable *ranks = nir_variable_create(mesh, nir_var_mem_shared,
            glsl_array_type(glsl_uint_type(), limit, 0), "bc250_cull_prefix");
         nir_push_if(&b, nir_ult_imm(&b, lane, limit));
         nir_push_if(&b, nir_ult(&b, lane, pc));
         nir_def *flag = nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, old_cull),
                                                               nir_iadd(&b, base, lane)));
         nir_def *live = nir_b2i32(&b, nir_ieq_imm(&b, flag, 0));
         nir_push_else(&b, NULL);
         nir_def *zero = nir_imm_int(&b, 0);
         nir_pop_if(&b, NULL);
         live = nir_if_phi(&b, live, zero);
         nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, ranks), lane), live, 1);
         nir_pop_if(&b, NULL);
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
         for (unsigned step = 1; step < limit; step *= 2) {
            nir_push_if(&b, nir_ult_imm(&b, lane, limit));
            nir_def *own = nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, ranks), lane));
            nir_push_if(&b, nir_uge_imm(&b, lane, step));
            nir_def *previous = nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, ranks),
                                                                      nir_iadd_imm(&b, lane, -(int)step)));
            nir_push_else(&b, NULL);
            zero = nir_imm_int(&b, 0);
            nir_pop_if(&b, NULL);
            previous = nir_if_phi(&b, previous, zero);
            nir_def *sum = nir_iadd(&b, own, previous);
            nir_push_else(&b, NULL);
            zero = nir_imm_int(&b, 0);
            nir_pop_if(&b, NULL);
            sum = nir_if_phi(&b, sum, zero);
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                        .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
            nir_push_if(&b, nir_ult_imm(&b, lane, limit));
            nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, ranks), lane), sum, 1);
            nir_pop_if(&b, NULL);
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                        .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
         }
         nir_push_if(&b, nir_ult(&b, lane, pc));
         nir_def *original = nir_iadd(&b, base, lane);
         flag = nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, old_cull), original));
         nir_push_if(&b, nir_ieq_imm(&b, flag, 0));
         nir_def *dst = nir_iadd_imm(&b, nir_load_deref(&b,
            nir_build_deref_array(&b, nir_build_deref_var(&b, ranks), lane)), -1);
         nir_copy_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, indices), dst),
                       nir_build_deref_array(&b, nir_build_deref_var(&b, old_indices), original));
         for (unsigned i = 0; i < attr_count; i++)
            nir_copy_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, new_attrs[i]), dst),
                          nir_build_deref_array(&b, nir_build_deref_var(&b, old_attrs[i]), original));
         nir_pop_if(&b, NULL);
         nir_pop_if(&b, NULL);
         nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
         nir_store_var(&b, count, nir_load_deref(&b, nir_build_deref_array(&b,
            nir_build_deref_var(&b, ranks), nir_imm_int(&b, limit - 1))), 1);
         nir_pop_if(&b, NULL);
      } else {
      nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
      nir_variable *src_cursor = nir_local_variable_create(impl, glsl_uint_type(), "bc250_cull_source");
      nir_variable *dst_cursor = nir_local_variable_create(impl, glsl_uint_type(), "bc250_cull_destination");
      nir_store_var(&b, src_cursor, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, dst_cursor, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      nir_def *p = nir_load_var(&b, src_cursor);
      nir_push_if(&b, nir_uge(&b, p, pc));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_def *original = nir_iadd(&b, base, p);
      nir_def *culled = nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, old_cull), original));
      nir_push_if(&b, nir_ieq_imm(&b, culled, 0));
      nir_def *dst = nir_load_var(&b, dst_cursor);
      nir_copy_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, indices), dst),
                    nir_build_deref_array(&b, nir_build_deref_var(&b, old_indices), original));
      for (unsigned i = 0; i < attr_count; i++) {
         nir_copy_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, new_attrs[i]), dst),
                       nir_build_deref_array(&b, nir_build_deref_var(&b, old_attrs[i]), original));
      }
      nir_store_var(&b, dst_cursor, nir_iadd_imm(&b, dst, 1), 1);
      nir_pop_if(&b, NULL);
      nir_store_var(&b, src_cursor, nir_iadd_imm(&b, p, 1), 1);
      nir_pop_loop(&b, NULL);
      nir_store_var(&b, count, nir_load_var(&b, dst_cursor), 1);
      nir_pop_if(&b, NULL);
      }
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared | nir_var_shader_out);
      nir_set_vertex_and_primitive_count(&b, nir_load_var(&b, vertex_count), nir_load_var(&b, count), nir_undef(&b, 1, 32));
   } else if (!slice_stores) {
      nir_variable *cursor = nir_local_variable_create(impl, glsl_uint_type(), "bc250_split_primitive");
      nir_store_var(&b, cursor, lane, 1);
      nir_push_loop(&b);
      nir_def *p = nir_load_var(&b, cursor);
      nir_push_if(&b, nir_uge(&b, p, pc));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_copy_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, indices), p),
                    nir_build_deref_array(&b, nir_build_deref_var(&b, old_indices), nir_iadd(&b, base, p)));
      for (unsigned i = 0; i < attr_count; i++) {
         nir_copy_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, new_attrs[i]), p),
                       nir_build_deref_array(&b, nir_build_deref_var(&b, old_attrs[i]), nir_iadd(&b, base, p)));
      }
      nir_store_var(&b, cursor, nir_iadd_imm(&b, p, mesh->info.workgroup_size[0] *
                                          mesh->info.workgroup_size[1] * mesh->info.workgroup_size[2]), 1);
      nir_pop_loop(&b, NULL);
   }
   mesh->info.mesh.max_primitives_out = limit;
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, mesh, nir_split_var_copies);
   NIR_PASS(_, mesh, nir_lower_var_copies);
   if (slice_stores) {
      NIR_PASS(_, mesh, nir_remove_dead_derefs);
      NIR_PASS(_, mesh, nir_remove_dead_variables, nir_var_mem_shared, NULL);
   }
   NIR_PASS(_, mesh, nir_lower_vars_to_explicit_types, nir_var_mem_shared, bc250_shared_type);
   NIR_PASS(_, mesh, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   nir_foreach_variable_with_modes(var, mesh, nir_var_mem_shared)
      assert(var->data.driver_location >= application_shared);
   if (debug_get_bool_option("BC250_TRACE_COMPILE", false))
      fprintf(stderr, "BC250 split shared layout: application=[0,%u) split=[%u,%u)\n",
              application_shared, application_shared, mesh->info.shared_size);
   nir_shader_gather_info(mesh, impl);
   nir_validate_shader(mesh, "BC250 restricted mesh split");
   if (debug_get_bool_option("BC250_TRACE_SPLIT_NIR", false))
      nir_print_shader(mesh, stderr);

   if (task) {
      nir_foreach_function_impl(timpl, task) {
         nir_builder tb = nir_builder_create(timpl);
         nir_foreach_block(block, timpl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
               if (in->intrinsic != nir_intrinsic_launch_mesh_workgroups)
                  continue;
               tb.cursor = nir_before_instr(instr);
               nir_def *old = in->src[0].ssa;
               nir_def *dims = nir_vec3(&tb, nir_imul_imm(&tb, nir_channel(&tb, old, 0), pieces),
                                        nir_channel(&tb, old, 1), nir_channel(&tb, old, 2));
               if (fold) {
                  /* API: x <= 65535 and x * y <= 2^22. An out-of-spec launch is dropped, never clamped. */
                  nir_def *x = nir_channel(&tb, old, 0), *y = nir_channel(&tb, old, 1);
                  nir_def *total = nir_imul(&tb, nir_u2u64(&tb, x), nir_u2u64(&tb, y));
                  nir_def *valid = nir_iand(&tb, nir_ult_imm(&tb, x, 65536), nir_ule_imm(&tb, total, 1u << 22));
                  nir_def *empty = nir_ieq_imm(&tb, nir_channel(&tb, old, 2), 0);
                  dims = nir_vec3(&tb, nir_imm_int(&tb, pieces),
                                  nir_bcsel(&tb, nir_iand(&tb, valid, nir_inot(&tb, empty)), x, nir_imm_int(&tb, 0)), y);
               }
               nir_src_rewrite(&in->src[0], dims);
            }
         }
         nir_progress(true, timpl, nir_metadata_none);
      }
      nir_shader_gather_info(task, nir_shader_get_entrypoint(task));
      nir_validate_shader(task, "BC250 split task launch");
   }
   if (pieces_out)
      *pieces_out = pieces;
   fprintf(stderr, "BC250 split count proof: declared=%u bound=%u pieces=%u\n",
           declared, bound, pieces);
   fprintf(stderr, "BC250 mesh split applied: pieces=%u primitive_limit=%u generic_primitive_arrays=%u cull_compaction=%u\n", pieces, limit, attr_count, old_cull != NULL);
   return true;
}


/* Private constants point to per-draw storage. Application push constants are
 * copied intact to that storage and lowered to global reads in each stage. */
struct bc250_constants {
   uint64_t xyz;
   uint64_t payload;
   uint32_t stride;
   uint32_t application_draw_id;
   uint64_t application_constants;
   uint64_t input;
   uint64_t input_count;
   uint64_t compute_args;
   /* VA of the per-draw full-dispatch size entry (struct bc250_task_slot). */
   uint64_t task_slot;
   /* Global row-major index of this chunk's first task group. Direct chunks:
    * written by the CPU per dispatch. Indirect chunks: k * BC250_TASK_CHUNK,
    * known to the CPU because it emits one producer/mesh pair per chunk slot.
    */
   uint32_t chunk_base;
   /* RADV_BC250_SPLIT_PREP_FREE: piece + 1 of a prep-free split indirect
    * draw, 0 for every other draw (bc250_split_flat_index). */
   uint32_t split_piece;
};
static_assert(offsetof(struct bc250_constants, split_piece) == BC250_SPLIT_PIECE_OFFSET, "split piece select");
static_assert(sizeof(struct bc250_constants) == 72, "the constant block size is unchanged");

/* Full-dispatch size published once per draw (direct: CPU memcpy before the
 * first chunk; indirect: the setup shader after validating the record). The
 * producer unflattens its global task index against it to emulate the
 * application's WorkGroupID. */
struct bc250_task_slot {
   uint32_t full_x;
   uint32_t full_y;
   uint32_t full_z;
   uint32_t pad;
};

/* Proven per-dispatch task-group bound. Larger application dispatches are
 * split into sequential chunks of at most this many groups so that scratch
 * allocation stays bounded and each producer dispatch remains inside the range
 * already exercised by the test suites. */
#define BC250_TASK_CHUNK 4096u
/* Advertised total cap: the Vulkan spec minimum for task shaders, which also
 * covers the largest CTS dispatch (65535 x 8 x 8). Must match
 * maxTaskWorkGroupTotalCount in radv_physical_device.c. */
#define BC250_MAX_TASK_TOTAL 4194304u
/* Advertised per-dimension cap; must match maxTaskWorkGroupCount in
 * radv_physical_device.c. Indirect records above it are undefined input and
 * are dropped (zeroed) by the setup shader rather than clamped. */
#define BC250_MAX_TASK_DIM 65535u
/* ceil(BC250_MAX_TASK_TOTAL / BC250_TASK_CHUNK): the maximum number of chunks
 * one indirect record can require. The CPU emits this many producer/mesh pairs
 * per record; inactive chunk slots hold zero records, which are no-op
 * dispatches and count-0 draws. */
#define BC250_INDIRECT_CHUNKS 1024u

static nir_def *
bc250_pointer(nir_builder *b, unsigned offset)
{
   return nir_pack_64_2x32(b, nir_load_push_constant(b, 2, 32, nir_imm_int(b, offset),
                                                  .base = 0, .range = sizeof(struct bc250_constants)));
}

static nir_def *
bc250_task_index(nir_builder *b)
{
   /* Chunk-local flattened workgroup index. This indexes the shared record and
    * payload regions, which hold one chunk at a time; it must stay based on the
    * hardware values of the current dispatch, not the emulated full-dispatch
    * builtins. */
   nir_def *id = nir_load_workgroup_id(b);
   nir_def *size = nir_load_num_workgroups(b);
   return nir_iadd(b, nir_channel(b, id, 0),
                  nir_imul(b, nir_channel(b, size, 0),
                           nir_iadd(b, nir_channel(b, id, 1),
                                    nir_imul(b, nir_channel(b, size, 1), nir_channel(b, id, 2)))));
}

static nir_def *
bc250_payload_address(nir_builder *b, nir_def *offset, bool producer)
{
   nir_def *index = producer ? bc250_task_index(b) : nir_load_draw_id(b);
   nir_def *stride = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 16),
                                           .base = 0, .range = sizeof(struct bc250_constants));
   nir_def *byte_offset = nir_iadd(b, nir_imul(b, index, stride), offset);
   return nir_iadd(b, bc250_pointer(b, 8), nir_u2u64(b, byte_offset));
}

static bool
bc250_lower_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   const bool producer = *(bool *)data;
   b->cursor = nir_before_instr(&intrin->instr);
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_task_payload: {
      nir_def *offset = nir_iadd_imm(b, intrin->src[0].ssa, nir_intrinsic_base(intrin));
      nir_def *addr = bc250_payload_address(b, offset, producer);
      nir_def *value = nir_load_global(b, intrin->def.num_components, intrin->def.bit_size, addr,
                                      .align_mul = MIN2(nir_intrinsic_align_mul(intrin), 16),
                                      .align_offset = nir_intrinsic_align_offset(intrin) % MIN2(nir_intrinsic_align_mul(intrin), 16));
      nir_def_rewrite_uses(&intrin->def, value);
      nir_instr_remove(&intrin->instr);
      return true;
   }
   case nir_intrinsic_store_task_payload: {
      assert(producer);
      nir_def *offset = nir_iadd_imm(b, intrin->src[1].ssa, nir_intrinsic_base(intrin));
      nir_def *addr = bc250_payload_address(b, offset, true);
      nir_store_global(b, intrin->src[0].ssa, addr,
                       .write_mask = nir_intrinsic_write_mask(intrin),
                       .align_mul = MIN2(nir_intrinsic_align_mul(intrin), 16),
                       .align_offset = nir_intrinsic_align_offset(intrin) % MIN2(nir_intrinsic_align_mul(intrin), 16));
      nir_instr_remove(&intrin->instr);
      return true;
   }
   case nir_intrinsic_launch_mesh_workgroups: {
      assert(producer);
      nir_def *addr = nir_iadd(b, bc250_pointer(b, 0),
                              nir_u2u64(b, nir_imul_imm(b, bc250_task_index(b), 12)));
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      nir_store_global(b, intrin->src[0].ssa, addr, .write_mask = 7, .align_mul = 4);
      nir_pop_if(b, NULL);
      nir_instr_remove(&intrin->instr);
      return true;
   }
   case nir_intrinsic_load_workgroup_id: {
      if (!producer)
         return false;
      /* Emulate the full-dispatch WorkGroupID across chunked producer
       * dispatches. Chunks are flat row-major ranges of the original box: this
       * workgroup's global index is the push-constant chunk base plus its
       * hardware position within the current dispatch. Unflatten it against
       * the published full size to recover exactly the 3D id a native single
       * dispatch would assign (row-major: idx = x + X*(y + Y*z)). */
      nir_def *flat_base = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 64), .base = 0,
                                                  .range = sizeof(struct bc250_constants));
      nir_def *g = nir_iadd(b, flat_base, bc250_task_index(b));
      nir_def *full = nir_load_global(b, 3, 32, bc250_pointer(b, 8 * 7), .align_mul = 4);
      /* g < BC250_MAX_TASK_TOTAL (2^22) and full_x*full_y <= 65535^2 fits u32;
       * use u64 for the products to stay overflow-free regardless. */
      nir_def *xy = nir_imul(b, nir_u2u64(b, nir_channel(b, full, 0)),
                             nir_u2u64(b, nir_channel(b, full, 1)));
      /* This NIR generation has no unsigned-remainder opcode; compute
       * r = g - (g/d)*d. All values are non-negative u64. */
      nir_def *gz = nir_u2u64(b, g);
      nir_def *qz = nir_udiv(b, gz, xy);
      nir_def *rem = nir_isub(b, gz, nir_imul(b, qz, xy));
      nir_def *fx64 = nir_u2u64(b, nir_channel(b, full, 0));
      nir_def *qy = nir_udiv(b, rem, fx64);
      nir_def *tx64 = nir_isub(b, rem, nir_imul(b, qy, fx64));
      nir_def *tz = nir_u2u32(b, qz), *ty = nir_u2u32(b, qy), *tx = nir_u2u32(b, tx64);
      if (intrin->def.num_components == 1) {
         nir_def_rewrite_uses(&intrin->def, tx);
      } else if (intrin->def.num_components == 2) {
         nir_def *parts[2] = {tx, ty};
         nir_def_rewrite_uses(&intrin->def, nir_vec(b, parts, 2));
      } else {
         nir_def *parts[3] = {tx, ty, tz};
         nir_def_rewrite_uses(&intrin->def, nir_vec(b, parts, 3));
      }
      nir_instr_remove(&intrin->instr);
      return true;
   }
   case nir_intrinsic_load_num_workgroups: {
      if (!producer)
         return false;
      /* Emulate the full-dispatch NumWorkGroups from the published size. */
      nir_def_rewrite_uses(&intrin->def,
                           nir_load_global(b, intrin->def.num_components, 32, bc250_pointer(b, 8 * 7),
                                           .align_mul = 4));
      nir_instr_remove(&intrin->instr);
      return true;
   }
   case nir_intrinsic_load_draw_id:
      if (producer) {
         /* A single application draw has DrawID 0 in its task stage. */
         nir_def_rewrite_uses(&intrin->def, nir_imm_int(b, 0));
         nir_instr_remove(&intrin->instr);
         return true;
      }
      /* Emulate a continuous MESH DrawID across chunked mesh draws: the
       * hardware record index within this draw plus the chunk's global base. */
      {
         nir_def *hw = nir_load_draw_id(b);
         nir_def *base = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 8 * 8), .base = 0,
                                               .range = sizeof(struct bc250_constants));
         nir_def_rewrite_uses(&intrin->def, nir_iadd(b, hw, base));
         nir_instr_remove(&intrin->instr);
         return true;
      }
   case nir_intrinsic_barrier: {
      nir_variable_mode modes = nir_intrinsic_memory_modes(intrin);
      if (!(modes & nir_var_mem_task_payload))
         return false;
      nir_intrinsic_set_memory_modes(intrin, (modes & ~nir_var_mem_task_payload) | nir_var_mem_global);
      return true;
   }
   default:
      return false;
   }
}

static bool
bc250_lower_application_constants(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic == nir_intrinsic_load_draw_id && !data) {
      b->cursor = nir_before_instr(&intrin->instr);
      nir_def_rewrite_uses(&intrin->def,
         nir_load_push_constant(b, 1, 32, nir_imm_int(b, 20), .base = 0,
                                .range = sizeof(struct bc250_constants)));
      nir_instr_remove(&intrin->instr);
      return true;
   }
   if (intrin->intrinsic != nir_intrinsic_load_push_constant)
      return false;
   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *offset = nir_iadd_imm(b, intrin->src[0].ssa, nir_intrinsic_base(intrin));
   nir_def *addr = nir_iadd(b, bc250_pointer(b, 24), nir_u2u64(b, offset));
   nir_def *value = nir_load_global(b, intrin->def.num_components, intrin->def.bit_size, addr,
                                   .align_mul = MIN2(nir_intrinsic_align_mul(intrin), 256),
                                   .align_offset = nir_intrinsic_align_offset(intrin) %
                                                   MIN2(nir_intrinsic_align_mul(intrin), 256),
                                   .access = ACCESS_NON_WRITEABLE);
   nir_def_rewrite_uses(&intrin->def, value);
   nir_instr_remove(&intrin->instr);
   return true;
}

static nir_shader *
bc250_build_indirect_setup(struct radv_device *device)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &device->compiler_info.nir_options[MESA_SHADER_COMPUTE], "bc250_indirect_setup");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *draw_id = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 20),
      .base = 0, .range = sizeof(struct bc250_constants));
   nir_def *count_addr = bc250_pointer(&b, 40);
   nir_def *unlimited = nir_imm_int(&b, -1);
   nir_def *zero_record = nir_imm_ivec4(&b, 0, 0, 0, 0);
   /* Created before the if/phi below: phi nodes must be first in their block,
    * and building an immediate at the cursor would precede the phi. */
   nir_def *xyz_zero = nir_imm_ivec3(&b, 0, 0, 0);
   nir_push_if(&b, nir_ine_imm(&b, count_addr, 0));
   nir_def *count = nir_load_global(&b, 1, 32, count_addr, .align_mul = 4);
   nir_pop_if(&b, NULL);
   count = nir_if_phi(&b, count, unlimited);
   /* Validate the record against the advertised limits (each dimension <=
    * BC250_MAX_TASK_DIM and total <= BC250_MAX_TASK_TOTAL). Larger records are
    * undefined input per spec and are dropped (zeroed) rather than clamped.
    * Dimensions are checked before forming the product so malformed input
    * cannot overflow it; with all dimensions bounded the u64 product is exact.
    */
   /* Inactive records (draw_id >= count) are treated as zero-sized: the phi
    * below yields x=y=z=0 for them. Every store in this shader therefore runs
    * unconditionally and rewrites all chunk slots to zero; scratch is reused
    * across indirect records, so stale sub-records must never survive. */
   nir_push_if(&b, nir_ult(&b, draw_id, count));
   nir_def *xyz_in = nir_load_global(&b, 3, 32, bc250_pointer(&b, 32), .align_mul = 4);
   nir_pop_if(&b, NULL);
   nir_def *xyz = nir_if_phi(&b, xyz_in, xyz_zero);
   nir_def *x = nir_channel(&b, xyz, 0), *y = nir_channel(&b, xyz, 1), *z = nir_channel(&b, xyz, 2);
   nir_def *bounded = nir_iand(&b, nir_ule_imm(&b, x, BC250_MAX_TASK_DIM),
      nir_iand(&b, nir_ule_imm(&b, y, BC250_MAX_TASK_DIM), nir_ule_imm(&b, z, BC250_MAX_TASK_DIM)));
   nir_def *total = nir_imul(&b, nir_u2u64(&b, x), nir_imul(&b, nir_u2u64(&b, y), nir_u2u64(&b, z)));
   nir_def *valid = nir_iand(&b, bounded, nir_ule_imm(&b, total, BC250_MAX_TASK_TOTAL));
   /* Publish the full-dispatch size for this record's chunks at the per-draw
    * slot (constants.task_slot). */
   nir_def *full_size = nir_bcsel(&b, valid, nir_vec4(&b, x, y, z, nir_imm_int(&b, 0)),
                                  nir_imm_ivec4(&b, 0, 0, 0, 0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_global(&b, full_size, bc250_pointer(&b, 8 * 7), .align_mul = 16);
   nir_pop_if(&b, NULL);
   /* Split the record into flat row-major chunks of at most BC250_TASK_CHUNK
    * groups. Chunk k covers global task indices [k*CHUNK, min((k+1)*CHUNK,
    * total)) and is emitted as an (n_k,1,1) producer dispatch; sub-record k at
    * compute_args + 16*k holds (n_k,1,1,n_k), where the w component doubles as
    * the mesh draw count. Chunks beyond ceil(total/CHUNK) are written zero so
    * their dispatches and draws are no-ops. Stores are unconditional: scratch
    * is reused across records, so every slot must be rewritten each time. */
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_variable *kvar = nir_local_variable_create(impl, glsl_uint_type(), "bc250_chunk");
   /* Each invocation owns disjoint records; all 1024 slots are still
    * overwritten, including inactive ones from earlier indirect draws. */
   nir_store_var(&b, kvar, lane, 1);
   nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, kvar);
      /* While-loop exit. */
      nir_push_if(&b, nir_uge(&b, k, nir_imm_int(&b, BC250_INDIRECT_CHUNKS)));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      /* k < BC250_INDIRECT_CHUNKS and CHUNK is a power of two, so the product
       * fits u32 (max (K_MAX-1)*CHUNK < 2^22). */
      nir_def *base_k = nir_imul_imm(&b, k, BC250_TASK_CHUNK);
      nir_def *active = nir_iand(&b, valid, nir_ult(&b, nir_u2u64(&b, base_k), total));
      /* n_k = min(CHUNK, total - base_k). The subtraction underflows (wraps to
       * a huge u64) when the chunk is inactive; such chunks are stored zero by
       * the outer select, so only the clamped value matters for active ones. */
      nir_def *rem = nir_isub(&b, total, nir_u2u64(&b, base_k));
      nir_def *le_chunk = nir_ule_imm(&b, rem, BC250_TASK_CHUNK);
      nir_def *n = nir_u2u32(&b, nir_bcsel(&b, le_chunk, rem,
                                           nir_u2u64(&b, nir_imm_int(&b, BC250_TASK_CHUNK))));
      nir_store_global(&b, nir_bcsel(&b, active, nir_vec4(&b, n, nir_imm_int(&b, 1), nir_imm_int(&b, 1), n),
                                     zero_record),
         nir_iadd(&b, bc250_pointer(&b, 48), nir_u2u64(&b, nir_imul_imm(&b, k, 16))), .align_mul = 16);
      nir_store_var(&b, kvar, nir_iadd_imm(&b, k, 64), 1);
   }
   nir_pop_loop(&b, NULL);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_validate_shader(b.shader, "BC250 indirect setup");
   return b.shader;
}

/* One batched argument rewrite replaces synthetic per-group producers for
 * mesh-only splits. Original indirect DrawID and record ordering are retained. */
struct bc250_split_args {
   uint64_t input, output, count;
   uint32_t records, stride, pieces;
};

/* RADV_BC250_SPLIT_PREP_FREE admission, static part. A prep-free split
 * indirect draw launches every piece of every API workgroup, but in a
 * different order: all records' piece 0, then piece 1, ... instead of
 * workgroup by workgroup (Vulkan primitive order: all primitives of a mesh
 * workgroup before those of the next one). That is only invisible when the
 * result of the draw does not depend on the order of its primitives:
 *  - the fragment shader writes no output (no color, depth, stencil or sample
 *    mask export) and reads no framebuffer or interlock state, so attachments
 *    are only touched by depth/stencil tests (draw-time part:
 *    bc250_split_order_free_now requires depth writes and stencil off);
 *  - its only writes to memory are integer atomics that commute (add, min,
 *    max, and, or, xor) and whose returned value is unused, so the final
 *    memory contents depend only on the set of fragments, which with read-only
 *    depth/stencil does not depend on primitive order;
 *  - it reads no storage memory that may be written (ssbo, global, storage
 *    image loads without NON_WRITEABLE, atomics returning values).
 * Hellblade 2's Nanite fragment shaders (one 64-bit image atomic max, no
 * outputs) pass. Everything else keeps the per-draw setup. */
static bool
bc250_fs_order_independent(nir_shader *fs, const char **reason)
{
#define BC250_ORDER_REFUSE(r) do { if (reason) *reason = (r); return false; } while (0)
   if (fs->info.fs.uses_fbfetch_output || fs->info.fs.pixel_interlock_ordered ||
       fs->info.fs.pixel_interlock_unordered || fs->info.fs.sample_interlock_ordered ||
       fs->info.fs.sample_interlock_unordered)
      BC250_ORDER_REFUSE("framebuffer fetch or interlock");
   nir_foreach_variable_with_modes(var, fs, nir_var_shader_out)
      BC250_ORDER_REFUSE("fragment output");
   nir_foreach_function_impl(impl, fs) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            switch (in->intrinsic) {
            case nir_intrinsic_store_output:
            case nir_intrinsic_store_per_primitive_output:
            case nir_intrinsic_store_per_vertex_output:
            case nir_intrinsic_begin_invocation_interlock:
            case nir_intrinsic_end_invocation_interlock:
               BC250_ORDER_REFUSE("fragment output or interlock");
            case nir_intrinsic_store_deref:
               if (nir_deref_mode_may_be(nir_src_as_deref(in->src[0]), nir_var_shader_out))
                  BC250_ORDER_REFUSE("fragment output");
               break;
            default:
               break;
            }
            if (nir_intrinsic_writes_external_memory(in)) {
               if (!nir_intrinsic_has_atomic_op(in))
                  BC250_ORDER_REFUSE("memory store");
               switch (nir_intrinsic_atomic_op(in)) {
               case nir_atomic_op_iadd:
               case nir_atomic_op_imin:
               case nir_atomic_op_umin:
               case nir_atomic_op_imax:
               case nir_atomic_op_umax:
               case nir_atomic_op_iand:
               case nir_atomic_op_ior:
               case nir_atomic_op_ixor:
                  break;
               default:
                  BC250_ORDER_REFUSE("non-commuting atomic");
               }
               if (!nir_def_is_unused(&in->def))
                  BC250_ORDER_REFUSE("atomic result used");
               continue;
            }
            if (bc250_reads_external_memory(in)) {
               const bool ubo = in->intrinsic == nir_intrinsic_load_deref ?
                  !nir_deref_mode_may_be(nir_src_as_deref(in->src[0]),
                                         nir_var_mem_ssbo | nir_var_mem_global | nir_var_image) :
                  !strncmp(nir_intrinsic_infos[in->intrinsic].name, "load_ubo", 8);
               const bool readonly = nir_intrinsic_has_access(in) &&
                                     (nir_intrinsic_access(in) & ACCESS_NON_WRITEABLE);
               if (!ubo && !readonly)
                  BC250_ORDER_REFUSE("storage read");
            }
         }
      }
   }
#undef BC250_ORDER_REFUSE
   return true;
}

VkResult
radv_bc250_prepare_direct_split(struct radv_device *device, struct radv_graphics_pipeline *pipeline)
{
   VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct bc250_split_args)};
   const enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_BC250_SPLIT_ARGUMENTS;
   /* All shader inputs are push constants. Share the identical helper across
    * graphics pipelines, including cache hits and concurrent creation. Mark
    * ownership before allocation so partial failures use the same cleanup. */
   pipeline->bc250_shared_setup = true;
   VkResult result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device,
      NULL, &range, &key, sizeof(key), &pipeline->bc250_task_layout);
   if (result != VK_SUCCESS)
      return result;
   pipeline->bc250_setup_pipeline = vk_meta_lookup_pipeline(&device->meta_state.device,
                                                           &key, sizeof(key));
   if (pipeline->bc250_setup_pipeline != VK_NULL_HANDLE)
      return VK_SUCCESS;
   if (pipeline->base.create_flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)
      return VK_PIPELINE_COMPILE_REQUIRED;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &device->compiler_info.nir_options[MESA_SHADER_COMPUTE], "bc250_split_arguments");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *args = nir_load_push_constant(&b, 3, 64, nir_imm_int(&b, 0), .range = 24);
   nir_def *sizes = nir_load_push_constant(&b, 3, 32, nir_imm_int(&b, 24), .range = 12);
   nir_def *records = nir_channel(&b, sizes, 0);
   nir_def *i = nir_iadd(&b, nir_imul_imm(&b, nir_channel(&b, nir_load_workgroup_id(&b), 0), 64),
                        nir_load_local_invocation_index(&b));
   nir_def *zero = nir_imm_ivec3(&b, 0, 0, 0);
   nir_push_if(&b, nir_ult(&b, i, records));
   nir_def *count_addr = nir_channel(&b, args, 2);
   nir_push_if(&b, nir_ine_imm(&b, count_addr, 0));
   nir_def *count = nir_load_global(&b, 1, 32, count_addr, .align_mul = 4);
   nir_pop_if(&b, NULL);
   count = nir_if_phi(&b, count, records);
   nir_push_if(&b, nir_ult(&b, i, count));
   nir_def *addr = nir_iadd(&b, nir_channel(&b, args, 0),
      nir_imul(&b, nir_u2u64(&b, i), nir_u2u64(&b, nir_channel(&b, sizes, 1))));
   nir_def *xyz = nir_load_global(&b, 3, 32, addr, .align_mul = 4);
   nir_def *dx = nir_channel(&b, xyz, 0), *dy = nir_channel(&b, xyz, 1), *dz = nir_channel(&b, xyz, 2);
   nir_def *px = nir_iand(&b, nir_uge(&b, dy, dx), nir_uge(&b, dz, dx));
   nir_def *py = nir_iand(&b, nir_inot(&b, px), nir_uge(&b, dz, dy));
   nir_def *pz = nir_inot(&b, nir_ior(&b, px, py));
   nir_def *factor = nir_channel(&b, sizes, 2);
   nir_def *expanded = nir_vec3(&b,
      nir_imul(&b, dx, nir_bcsel(&b, px, factor, nir_imm_int(&b, 1))),
      nir_imul(&b, dy, nir_bcsel(&b, py, factor, nir_imm_int(&b, 1))),
      nir_imul(&b, dz, nir_bcsel(&b, pz, factor, nir_imm_int(&b, 1))));
   nir_pop_if(&b, NULL);
   expanded = nir_if_phi(&b, expanded, zero);
   nir_store_global(&b, expanded, nir_iadd(&b, nir_channel(&b, args, 1),
      nir_u2u64(&b, nir_imul_imm(&b, i, 16))), .align_mul = 16);
   nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_validate_shader(b.shader, "BC250 compact split argument setup");
   /* Test aid (tests/bc250-mesh/split-batch): the builder NIR of both setup shaders. */
   if (debug_get_bool_option("RADV_BC250_SPLIT_BATCH_PRINT_NIR", false))
      nir_print_shader(b.shader, stderr);
   VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = vk_shader_module_handle_from_nir(b.shader),
         .pName = "main"}, .layout = pipeline->bc250_task_layout};
   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device,
      &ci, &key, sizeof(key), &pipeline->bc250_setup_pipeline);
   ralloc_free(b.shader);
   return result;
}

/* Pipeline create flags the private Mesh/Task paths (direct split, internal
 * mesh-only amplifier, hybrid Task producer) do not handle:
 * - INDIRECT_BINDABLE: device-generated commands would bind the graphics
 *   pipeline and draw without the BC250 draw paths;
 * - LIBRARY: a library is linked later (possibly without link-time
 *   optimization) and never reaches this preparation;
 * - CAPTURE_DATA: pipeline binaries do not serialize the private compute
 *   pipelines or the split plan.
 * DESCRIPTOR_BUFFER is refused unless RADV_BC250_EXPOSE_FAST_BINDING=1. With a
 * descriptor buffer pipeline the shader code only differs in the set layouts it
 * was compiled against; set addresses reach the user SGPRs through
 * radv_descriptor_get_va (descriptor_buffers[] for sets bound by offset), and
 * the hybrid producer receives the graphics descriptor state, descriptor
 * buffer addresses and push descriptors included, as a whole-struct copy
 * (bc250_draw_task). The split argument/batch/merge helpers bind no
 * descriptors at all. */
static VkPipelineCreateFlags2
bc250_refused_create_flags(const struct radv_device *device)
{
   VkPipelineCreateFlags2 flags = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT |
                                  VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR |
                                  VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;
   if (!radv_device_physical(device)->bc250_fast_binding)
      flags |= VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT;
   return flags;
}

/* RADV_BC250_MESH_CLIPCULL_CONST (radv_bc250_mesh_clip_cull_const) on the Mesh shader before
 * radv_bc250_prepare_task rewrites it. The fragment shader decides whether the distances are
 * also varyings; it is translated here if the Mesh shader writes distances (the graphics
 * compile reuses that NIR, with its own options). A graphics pipeline without a fragment
 * stage has no consumer. */
static void
bc250_prepare_clip_cull_const(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                              const struct radv_graphics_pipeline_state *gfx_state, struct radv_shader_stage *stages)
{
   nir_shader *mesh = stages[MESA_SHADER_MESH].nir;
   if (!device->compiler_info.hw.bc250_mesh_cc_const || !mesh ||
       !(mesh->info.outputs_written & RADV_BC250_CLIPCULL_OUTPUTS))
      return;
   struct radv_shader_stage *fs = &stages[MESA_SHADER_FRAGMENT];
   if (fs->stage != MESA_SHADER_NONE && !fs->nir && !fs->layout.mapping) {
      struct radv_spirv_to_nir_options options = {
         .lower_view_index_to_zero = !gfx_state->key.gfx_state.has_multiview_view_index,
         .lower_view_index_to_device_index = fs->key.view_index_from_device_index,
      };
      fs->nir = radv_shader_spirv_to_nir_cached(&device->compiler_info, NULL, fs, &options, false);
   }
   const bool no_fs = fs->stage == MESA_SHADER_NONE && pipeline->base.type == RADV_PIPELINE_GRAPHICS &&
                      !(pipeline->active_stages & VK_SHADER_STAGE_FRAGMENT_BIT);
   radv_bc250_mesh_clip_cull_const(mesh, fs->nir, fs->nir || no_fs);
}

bool
radv_bc250_mesh_culldist_split(const struct radv_compiler_info *compiler_info, const nir_shader *mesh)
{
   return compiler_info->hw.bc250_mesh_culldist_cull && mesh && mesh->info.stage == MESA_SHADER_MESH &&
          mesh->info.mesh.primitive_type == MESA_PRIM_TRIANGLES && mesh->info.cull_distance_array_size &&
          (mesh->info.outputs_written & RADV_BC250_CLIPCULL_OUTPUTS) && mesh->info.mesh.max_primitives_out > 64;
}

/* Default-off ordered Mesh materialization. One API body per record, followed
 * by one 128-lane exporter or two independently compacted 32-primitive pieces.
 * The existing hybrid draw scheduler consumes records in API workgroup order.
 * No output from a later group can precede an earlier group's overflow piece.
 */
struct bc250_ordered_plan {
   unsigned indices, index_stride, scratch, threads;
};
static nir_def *
bc250_ordered_rank(nir_builder *b, nir_def *key, nir_def **words)
{
   nir_def *word = nir_ushr_imm(b, key, 5);
   nir_def *below = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), nir_iand_imm(b, key, 31)), -1);
   nir_def *rank = nir_imm_int(b, 0);
   for (unsigned i = 0; i < 8; i++)
      rank = nir_iadd(b, rank, nir_bit_count(b,
         nir_bcsel(b, nir_ult_imm(b, word, i), nir_imm_int(b, 0),
            nir_bcsel(b, nir_ieq_imm(b, word, i), nir_iand(b, words[i], below), words[i]))));
   return rank;
}
static void
bc250_ordered_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
}
/* Scratch: refs[192], masks[8], inverse[192], triangles[192], all u32. */
static nir_def *
bc250_ordered_plan(nir_builder *b, const struct bc250_ordered_plan *s,
                   nir_def *vc, nir_def *pc, unsigned first)
{
   const unsigned masks = s->scratch + 192 * 4;
   const unsigned inverse = masks + 32;
   const unsigned triangles = inverse + 192 * 4;
   nir_def *lane = nir_load_local_invocation_index(b), *zero = nir_imm_int(b, 0);
   nir_push_if(b, nir_ult_imm(b, lane, 8));
   nir_store_shared(b, zero, nir_imul_imm(b, lane, 4), .base = masks, .align_mul = 4);
   nir_pop_if(b, NULL);
   bc250_ordered_barrier(b);
   nir_def *limit = nir_iadd_imm(b, nir_umax_imm(b, vc, 1), -1);
   for (unsigned base = 0; base < 128; base += s->threads) {
      nir_def *v = nir_iadd_imm(b, lane, base);
      nir_push_if(b, nir_ult(b, v, vc));
      nir_variable *anchor = nir_local_variable_create(b->impl, glsl_uint_type(), "ordered_anchor");
      nir_variable *scan = nir_local_variable_create(b->impl, glsl_uint_type(), "ordered_scan");
      nir_store_var(b, anchor, nir_imm_int(b, -1), 1);
      nir_store_var(b, scan, zero, 1);
      nir_push_loop(b);
      nir_def *p = nir_load_var(b, scan);
      nir_break_if(b, nir_uge(b, p, pc));
      nir_def *key = nir_imul_imm(b, p, 3);
      nir_def *ix = nir_load_shared(b, 3, 32, nir_imul_imm(b, nir_iadd_imm(b, p, first), s->index_stride),
                                   .base = s->indices, .align_mul = 4);
      nir_def *eq[3];
      for (unsigned c = 0; c < 3; c++)
         eq[c] = nir_ieq(b, v, nir_umin(b, nir_channel(b, ix, c), limit));
      nir_push_if(b, nir_ior(b, eq[0], nir_ior(b, eq[1], eq[2])));
      nir_def *old = nir_load_var(b, anchor);
      nir_def *fresh = nir_ior(b, nir_ieq_imm(b, old, -1),
         nir_ugt_imm(b, nir_isub(b, nir_iadd_imm(b, key, 2), old), 31));
      nir_push_if(b, fresh);
      nir_def *a = nir_iadd(b, key, nir_bcsel(b, eq[0], zero,
                                      nir_bcsel(b, eq[1], nir_imm_int(b, 1), nir_imm_int(b, 2))));
      nir_store_var(b, anchor, a, 1);
      nir_shared_atomic(b, 32, nir_imul_imm(b, nir_ushr_imm(b, a, 5), 4),
                        nir_ishl(b, nir_imm_int(b, 1), nir_iand_imm(b, a, 31)),
                        .base = masks, .atomic_op = nir_atomic_op_ior);
      nir_pop_if(b, NULL);
      for (unsigned c = 0; c < 3; c++) {
         nir_push_if(b, eq[c]);
         nir_store_shared(b, nir_load_var(b, anchor), nir_imul_imm(b, nir_iadd_imm(b, key, c), 4),
                          .base = s->scratch, .align_mul = 4);
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
      nir_store_var(b, scan, nir_iadd_imm(b, p, 1), 1);
      nir_pop_loop(b, NULL);
      nir_pop_if(b, NULL);
   }
   bc250_ordered_barrier(b);
   nir_def *words[8], *total = zero;
   for (unsigned i = 0; i < 8; i++) {
      words[i] = nir_load_shared(b, 1, 32, zero, .base = masks + i * 4, .align_mul = 4);
      total = nir_iadd(b, total, nir_bit_count(b, words[i]));
   }
   nir_push_if(b, nir_ult(b, lane, pc));
   nir_def *key = nir_imul_imm(b, lane, 3);
   nir_def *a = nir_load_shared(b, 3, 32, nir_imul_imm(b, key, 4), .base = s->scratch, .align_mul = 4);
   nir_def *ix = nir_load_shared(b, 3, 32, nir_imul_imm(b, nir_iadd_imm(b, lane, first), s->index_stride),
                                .base = s->indices, .align_mul = 4);
   nir_def *slot[3];
   for (unsigned c = 0; c < 3; c++) {
      nir_def *ac = nir_channel(b, a, c);
      slot[c] = bc250_ordered_rank(b, ac, words);
      nir_push_if(b, nir_ieq(b, ac, nir_iadd_imm(b, key, c)));
      nir_store_shared(b, nir_umin(b, nir_channel(b, ix, c), limit), nir_imul_imm(b, slot[c], 4),
                       .base = inverse, .align_mul = 4);
      nir_pop_if(b, NULL);
   }
   nir_store_shared(b, nir_vec3(b, slot[0], slot[1], slot[2]), nir_imul_imm(b, key, 4),
                    .base = triangles, .align_mul = 4);
   nir_pop_if(b, NULL);
   bc250_ordered_barrier(b);
   return total;
}
/* Each piece: uvec4 header, 128 u32 inverse slots, 64 uvec3 triangles. */
#define BC250_ORDERED_PIECE_BYTES (16 + 128 * 4 + 64 * 12)
static void
bc250_ordered_store_piece(nir_builder *b, const struct bc250_ordered_plan *s,
                          nir_def *record, unsigned offset, nir_def *vc, nir_def *pc)
{
   const unsigned inverse = s->scratch + 192 * 4 + 32;
   const unsigned triangles = inverse + 192 * 4;
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *dst = nir_iadd_imm(b, record, offset);
   nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_store_global(b, nir_vec2(b, vc, pc), dst, .align_mul = 16);
   nir_pop_if(b, NULL);
   for (unsigned base = 0; base < 128; base += s->threads) {
      nir_def *v = nir_iadd_imm(b, lane, base);
      nir_push_if(b, nir_ult(b, v, vc));
      nir_def *src = nir_load_shared(b, 1, 32, nir_imul_imm(b, v, 4), .base = inverse, .align_mul = 4);
      nir_store_global(b, src, nir_iadd(b, nir_iadd_imm(b, dst, 16), nir_u2u64(b, nir_imul_imm(b, v, 4))),
                       .align_mul = 4);
      nir_pop_if(b, NULL);
   }
   nir_push_if(b, nir_ult(b, lane, pc));
   nir_def *tri = nir_load_shared(b, 3, 32, nir_imul_imm(b, lane, 12), .base = triangles, .align_mul = 4);
   nir_store_global(b, tri, nir_iadd(b, nir_iadd_imm(b, dst, 16 + 128 * 4),
                                   nir_u2u64(b, nir_imul_imm(b, lane, 12))), .align_mul = 4);
   nir_pop_if(b, NULL);
   bc250_ordered_barrier(b);
}
static bool
bc250_ordered_counts(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
      return false;
   b->cursor = nir_before_instr(&in->instr);
   nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
   nir_def *vc = nir_umin_imm(b, in->src[0].ssa, 128);
   nir_def *pc = nir_bcsel(b, nir_ieq_imm(b, vc, 0), nir_imm_int(b, 0), nir_umin_imm(b, in->src[1].ssa, 64));
   nir_store_var(b, data, nir_vec2(b, vc, pc), 3);
   nir_pop_if(b, NULL);
   nir_instr_remove(&in->instr);
   return true;
}
static bool
bc250_ordered_eligible(nir_shader *mesh, nir_shader *fs)
{
   if (!mesh || !fs)
      return false;
   const unsigned t = mesh->info.workgroup_size[0] * mesh->info.workgroup_size[1] * mesh->info.workgroup_size[2];
   if (mesh->info.min_subgroup_size != 64 || mesh->info.mesh.max_vertices_out != 128 || mesh->info.mesh.max_primitives_out != 64 ||
       (t != 64 && t != 128) || !radv_bc250_mesh_safe_direct_candidate(mesh, true) ||
       !fs || radv_bc250_merge_fs_refusal(fs))
      return false;
   unsigned n = 0;
   bool ix = false;
   nir_foreach_shader_out_variable(v, mesh) {
      if (!glsl_type_is_array(v->type))
         return false;
      const struct glsl_type *el = glsl_get_array_element(v->type);
      if ((!glsl_type_is_vector(el) && !glsl_type_is_scalar(el)) || glsl_get_bit_size(el) != 32)
         return false;
      if (v->data.location == VARYING_SLOT_PRIMITIVE_INDICES) {
         ix = glsl_get_vector_elements(el) == 3;
      } else if (++n > 16 || v->data.per_primitive) {
         return false;
      }
   }
   return ix && n && mesh->info.shared_size + (n * 128 + 64) * 16 + 4096 <= 65536;
}
static VkResult
bc250_prepare_ordered(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                       const struct radv_graphics_pipeline_state *gfx_state)
{
   struct radv_shader_stage *ms = &gfx_state->stages[MESA_SHADER_MESH];
   nir_shader *producer = nir_shader_clone(NULL, ms->nir);
   nir_shader *fs = gfx_state->stages[MESA_SHADER_FRAGMENT].nir;
   /* Preserve application DrawID and constants before private ABI loads exist. */
   NIR_PASS(_, producer, nir_shader_intrinsics_pass, bc250_lower_application_constants, nir_metadata_none, NULL);
   NIR_PASS(_, fs, nir_shader_intrinsics_pass, bc250_lower_application_constants, nir_metadata_none, NULL);
   bool yes = true;
   NIR_PASS(_, producer, nir_shader_intrinsics_pass, bc250_lower_intrinsic, nir_metadata_none, &yes);
   NIR_PASS(_, producer, nir_lower_returns);
   NIR_PASS(_, producer, nir_remove_dead_variables, nir_var_mem_shared, NULL);
   nir_variable *attrs[16], *indices = NULL;
   unsigned nattrs = 0;
   nir_foreach_shader_out_variable(v, producer) {
      if (v->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
         indices = v;
      else
         attrs[nattrs++] = v;
   }
   nir_builder eb = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &device->compiler_info.nir_options[MESA_SHADER_MESH], "bc250_ordered_export");
   eb.shader->info.min_subgroup_size = eb.shader->info.max_subgroup_size = 64;
   eb.shader->info.workgroup_size[0] = 128;
   eb.shader->info.workgroup_size[1] = eb.shader->info.workgroup_size[2] = 1;
   eb.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   eb.shader->info.mesh.max_vertices_out = 128;
   eb.shader->info.mesh.max_primitives_out = 64;
   nir_variable *exports[16];
   for (unsigned i = 0; i < nattrs; i++)
      exports[i] = nir_variable_clone(attrs[i], eb.shader);
   nir_variable *export_indices = nir_variable_clone(indices, eb.shader);
   for (unsigned i = 0; i < nattrs; i++)
      exec_list_push_tail(&eb.shader->variables, &exports[i]->node);
   exec_list_push_tail(&eb.shader->variables, &export_indices->node);
   nir_foreach_shader_out_variable(v, producer)
      v->data.mode = nir_var_mem_shared;
   NIR_PASS(_, producer, nir_fixup_deref_modes);
   nir_variable *counts = nir_variable_create(producer, nir_var_mem_shared, glsl_uvec2_type(), "ordered_counts");
   NIR_PASS(_, producer, nir_shader_intrinsics_pass, bc250_ordered_counts, nir_metadata_none, counts);
   NIR_PASS(_, producer, nir_lower_vars_to_explicit_types, nir_var_mem_shared, bc250_shared_type);
   NIR_PASS(_, producer, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   nir_function_impl *impl = nir_shader_get_entrypoint(producer);
   nir_builder b = nir_builder_create(impl);
   b.cursor = nir_before_cf_list(&impl->body);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_shared(&b, nir_imm_ivec2(&b, 0, 0), nir_imm_int(&b, 0),
                    .base = counts->data.driver_location, .align_mul = 4);
   nir_gds_atomic_add_amd(&b, 32, nir_imm_int(&b,
      producer->info.workgroup_size[0] * producer->info.workgroup_size[1] * producer->info.workgroup_size[2]),
      nir_imm_int(&b, RADV_SHADER_QUERY_MS_INVOCATION_OFFSET), nir_imm_int(&b, 0x100));
   nir_pop_if(&b, NULL);
   bc250_ordered_barrier(&b);
   b.cursor = nir_after_cf_list(&impl->body);
   bc250_ordered_barrier(&b);
   nir_def *cp = nir_load_shared(&b, 2, 32, nir_imm_int(&b, 0),
                                .base = counts->data.driver_location, .align_mul = 4);
   nir_def *vc = nir_channel(&b, cp, 0), *pc = nir_channel(&b, cp, 1);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_gds_atomic_add_amd(&b, 32, pc, nir_imm_int(&b, RADV_SHADER_QUERY_MS_PRIM_GEN_OFFSET),
                         nir_imm_int(&b, 0x100));
   nir_pop_if(&b, NULL);
   nir_def *record = bc250_payload_address(&b, nir_imm_int(&b, 0), true);
   unsigned attr_offset[16];
   unsigned stride = 0;
   const unsigned threads = producer->info.workgroup_size[0] * producer->info.workgroup_size[1] * producer->info.workgroup_size[2];
   for (unsigned i = 0; i < nattrs; i++) {
      attr_offset[i] = stride;
      unsigned nc = glsl_get_vector_elements(glsl_get_array_element(attrs[i]->type));
      unsigned as = glsl_get_explicit_stride(attrs[i]->type);
      for (unsigned first = 0; first < 128; first += threads) {
         nir_def *v = nir_iadd_imm(&b, lane, first);
         nir_push_if(&b, nir_ult(&b, v, vc));
         nir_def *val = nir_load_shared(&b, nc, 32, nir_imul_imm(&b, v, as),
                                       .base = attrs[i]->data.driver_location, .align_mul = 4);
         nir_store_global(&b, val, nir_iadd(&b, nir_iadd_imm(&b, record, stride),
                                          nir_u2u64(&b, nir_imul_imm(&b, v, 16))), .align_mul = 16);
         nir_pop_if(&b, NULL);
      }
      stride += 128 * 16;
   }
   unsigned piece_offset = stride;
   stride += 2 * BC250_ORDERED_PIECE_BYTES;
   struct bc250_ordered_plan plan = {indices->data.driver_location, glsl_get_explicit_stride(indices->type),
                                     align(producer->info.shared_size, 16), threads};
   producer->info.shared_size = plan.scratch + (192 * 3 + 8) * 4;
   assert(producer->info.shared_size <= 65536);
   nir_def *total = bc250_ordered_plan(&b, &plan, vc, pc, 0);
   nir_def *fits = nir_ule_imm(&b, total, 128);
   nir_push_if(&b, fits);
   bc250_ordered_store_piece(&b, &plan, record, piece_offset, total, pc);
   nir_push_else(&b, NULL);
   nir_def *half = nir_umin_imm(&b, pc, 32);
   nir_def *n = bc250_ordered_plan(&b, &plan, vc, half, 0);
   bc250_ordered_store_piece(&b, &plan, record, piece_offset, n, half);
   half = nir_isub(&b, pc, half);
   n = bc250_ordered_plan(&b, &plan, vc, half, 32);
   bc250_ordered_store_piece(&b, &plan, record, piece_offset + BC250_ORDERED_PIECE_BYTES, n, half);
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_def *pieces = nir_bcsel(&b, nir_ieq_imm(&b, pc, 0), nir_imm_int(&b, 0),
                              nir_bcsel(&b, fits, nir_imm_int(&b, 1), nir_imm_int(&b, 2)));
   nir_def *arg = nir_iadd(&b, bc250_pointer(&b, 0), nir_u2u64(&b, nir_imul_imm(&b, bc250_task_index(&b), 12)));
   nir_store_global(&b, nir_vec3(&b, pieces, nir_imm_int(&b, 1), nir_imm_int(&b, 1)), arg, .align_mul = 4);
   nir_pop_if(&b, NULL);
   producer->info.stage = MESA_SHADER_COMPUTE;
   producer->info.outputs_written = 0;
   producer->info.task_payload_size = 0;
   producer->info.name = ralloc_asprintf(producer, "bc250_ordered_producer:%u:%u", plan.indices, plan.index_stride);
   nir_shader_gather_info(producer, impl);
   nir_validate_shader(producer, "BC250 ordered producer");
   if (getenv("BC250_ORDERED_NIR_DUMP")) {
      nir_shader *test = nir_shader_clone(NULL, producer);
      NIR_PASS(_, test, nir_lower_vars_to_ssa);
      NIR_PASS(_, test, nir_opt_copy_prop);
      NIR_PASS(_, test, nir_opt_dce);
      nir_validate_shader(test, "ordered oracle producer SSA");
      test->info.stage = MESA_SHADER_MESH;
      test->info.mesh.max_vertices_out = 128;
      test->info.mesh.max_primitives_out = 64;
      test->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      const uint32_t ix[4] = {plan.indices, plan.index_stride, 4, 128};
      const uint8_t hash[32] = {0};
      radv_bc250_dump_mesh_nir(getenv("BC250_ORDERED_NIR_DUMP"), test, 64, threads, hash, ix, 0, 0, 0);
      ralloc_free(test);
   }

   /* The exporter contains no API body and cannot repeat its side effects. */
   nir_def *er = bc250_payload_address(&eb, nir_imm_int(&eb, 0), false);
   nir_def *piece = nir_iadd(&eb, nir_iadd_imm(&eb, er, piece_offset),
      nir_u2u64(&eb, nir_imul_imm(&eb, nir_channel(&eb, nir_load_workgroup_id(&eb), 0), BC250_ORDERED_PIECE_BYTES)));
   nir_def *ec = nir_load_global(&eb, 2, 32, piece, .align_mul = 16);
   nir_def *ev = nir_channel(&eb, ec, 0), *ep = nir_channel(&eb, ec, 1);
   nir_set_vertex_and_primitive_count(&eb, ev, ep, nir_imm_int(&eb, 0));
   nir_def *el = nir_load_local_invocation_index(&eb);
   nir_push_if(&eb, nir_ult(&eb, el, ev));
   nir_def *original = nir_load_global(&eb, 1, 32,
      nir_iadd(&eb, nir_iadd_imm(&eb, piece, 16), nir_u2u64(&eb, nir_imul_imm(&eb, el, 4))), .align_mul = 4);
   for (unsigned i = 0; i < nattrs; i++) {
      unsigned nc = glsl_get_vector_elements(glsl_get_array_element(exports[i]->type));
      nir_def *val = nir_load_global(&eb, nc, 32, nir_iadd(&eb, nir_iadd_imm(&eb, er, attr_offset[i]),
         nir_u2u64(&eb, nir_imul_imm(&eb, original, 16))), .align_mul = 16);
      nir_store_deref(&eb, nir_build_deref_array(&eb, nir_build_deref_var(&eb, exports[i]), el), val, BITFIELD_MASK(nc));
   }
   nir_pop_if(&eb, NULL);
   nir_push_if(&eb, nir_ult(&eb, el, ep));
   nir_def *tri = nir_load_global(&eb, 3, 32, nir_iadd(&eb, nir_iadd_imm(&eb, piece, 16 + 128 * 4),
      nir_u2u64(&eb, nir_imul_imm(&eb, el, 12))), .align_mul = 4);
   nir_store_deref(&eb, nir_build_deref_array(&eb, nir_build_deref_var(&eb, export_indices), el), tri, 7);
   nir_pop_if(&eb, NULL);
   nir_shader_gather_info(eb.shader, nir_shader_get_entrypoint(eb.shader));
   nir_validate_shader(eb.shader, "BC250 ordered exporter");
   if (getenv("BC250_ORDERED_PRINT_NIR")) {
      nir_print_shader(producer, stderr);
      nir_print_shader(eb.shader, stderr);
   }

   VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(struct bc250_constants)};
   VkDescriptorSetLayout sets[MAX_SETS];
   for (unsigned i = 0; i < gfx_state->layout.num_sets; i++)
      sets[i] = radv_descriptor_set_layout_to_handle(gfx_state->layout.set[i].layout);
   VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = gfx_state->layout.num_sets, .pSetLayouts = sets,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
   VkResult result = radv_CreatePipelineLayout(radv_device_to_handle(device), &li, NULL, &pipeline->bc250_task_layout);
   if (result == VK_SUCCESS) {
      VK_FROM_HANDLE(radv_pipeline_layout, pl, pipeline->bc250_task_layout);
      if (pl->dynamic_shader_stages & VK_SHADER_STAGE_MESH_BIT_EXT)
         pl->dynamic_shader_stages |= VK_SHADER_STAGE_COMPUTE_BIT;
      VkPipelineCreateFlags2CreateInfo flags = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
         .flags = pipeline->base.create_flags & VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT};
      if (ms->key.optimisations_disabled)
         flags.flags |= VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT;
      VkPipelineRobustnessCreateInfo robust = {.sType = VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO,
         .storageBuffers = ms->key.storage_robustness2 ? VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2 :
                                                       VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
         .uniformBuffers = ms->key.uniform_robustness2 ? VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2 :
                                                       VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS};
      VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
         .pNext = &robust, .requiredSubgroupSize = 64};
      VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = &flags,
         .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &subgroup,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = vk_shader_module_handle_from_nir(producer), .pName = "main"},
         .layout = pipeline->bc250_task_layout};
      result = radv_compute_pipeline_create(radv_device_to_handle(device), device->meta_state.cache, &ci, NULL,
                                            &pipeline->bc250_task_pipeline);
      if (result == VK_SUCCESS) {
         nir_shader *setup = bc250_build_indirect_setup(device);
         ci.stage.module = vk_shader_module_handle_from_nir(setup);
         result = radv_compute_pipeline_create(radv_device_to_handle(device), device->meta_state.cache, &ci, NULL,
                                               &pipeline->bc250_setup_pipeline);
         ralloc_free(setup);
      }
   }
   ralloc_free(producer);
   if (result != VK_SUCCESS) {
      ralloc_free(eb.shader);
      return result;
   }
   ralloc_free(ms->nir);
   ms->nir = eb.shader;
   ms->bc250_ordered_export = true;
   pipeline->bc250_ordered = true;
   pipeline->bc250_payload_stride = stride;
   if (getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH ORDERED: V128/P64 T=%u producer once, exporter=128 pieces=1-or-2 stride=%u\n", threads, stride);
   return VK_SUCCESS;
}

/* Preserve the existing ordered child-grid transport and count-only indirect
 * setup. Each child uses safe shared vertices or safe private owned corners.
 * Keep the existing unsplit <=85P classes and proven single-piece cases out:
 * neither may gain an indirect setup dispatch. A declined transformation only
 * touched clones and leaves the fallback intact. */
static bool
bc250_prepare_safe_pieces(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                           const struct radv_graphics_pipeline_state *gfx_state)
{
   struct radv_shader_stage *ms = &gfx_state->stages[MESA_SHADER_MESH];
   struct radv_shader_stage *fs = &gfx_state->stages[MESA_SHADER_FRAGMENT];
   nir_shader *original = ms->nir;
   if (original && getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH PIECES CHECK: V=%u P=%u wave=%u threads=%u outputs=%" PRIx64 " pp=%" PRIx64 " shared=%u\n",
              original->info.mesh.max_vertices_out, original->info.mesh.max_primitives_out,
              original->info.min_subgroup_size, original->info.workgroup_size[0] * original->info.workgroup_size[1] * original->info.workgroup_size[2],
              original->info.outputs_written, original->info.per_primitive_outputs, original->info.shared_size);
   const bool bary = radv_bc250_mesh_private_bary(&device->compiler_info, fs->nir);
   if (!original || !fs->nir || (original->info.min_subgroup_size != 64 &&
                                !device->compiler_info.key.bc250_mesh_safe_bary_last) ||
       original->info.mesh.max_primitives_out <= 85 || original->info.mesh.max_primitives_out > 256 ||
       original->info.workgroup_size[0] * original->info.workgroup_size[1] * original->info.workgroup_size[2] > 192 ||
       (!bary && radv_bc250_mesh_fs_refused(&device->compiler_info, fs->nir)))
      return false;
   nir_shader *mesh = nir_shader_clone(NULL, original);
   nir_shader *fragment = nir_shader_clone(NULL, fs->nir);
   if (device->compiler_info.hw.bc250_mesh_cc_const)
      radv_bc250_mesh_clip_cull_const(mesh, fragment, true);
   const unsigned original_p = mesh->info.mesh.max_primitives_out;
   const bool owned_corners = device->compiler_info.key.bc250_mesh_safe_owned &&
      device->compiler_info.key.bc250_mesh_safe_corners &&
      (bary || (radv_bc250_mesh_needs_expansion(original) &&
      !gfx_state->key.gfx_state.dynamic_provoking_vtx_mode &&
      !gfx_state->key.gfx_state.rs.provoking_vtx_last));
   if (owned_corners) {
      NIR_PASS(_, fragment, nir_remove_dead_variables, nir_var_shader_in, NULL);
      NIR_PASS(_, mesh, nir_remove_unused_varyings, fragment);
      nir_shader_gather_info(mesh, nir_shader_get_entrypoint(mesh));
   }
   mesh->info.mesh.max_primitives_out = 64;
   /* RADV_BC250_MESH_SAFE_PIECES_EXT: a subgroup-free wave32 shader takes the wave64 pieces, and
    * CullPrimitive, which the split consumes (surviving triangles are compacted), does not block the
    * shared-vertex check of the unsplit clone; the check after the split still applies. */
   const bool pieces_ext = device->compiler_info.bc250x.safe_pieces_ext && !owned_corners;
   const bool wave64_promote = pieces_ext && radv_bc250_mesh_wave64_promotable(ms);
   if (wave64_promote)
      radv_bc250_mesh_set_wave(mesh, 64);
   const uint64_t cull_bit = pieces_ext ? (mesh->info.per_primitive_outputs & VARYING_BIT_CULL_PRIMITIVE) : 0;
   const uint64_t saved_pp = mesh->info.per_primitive_outputs, saved_written = mesh->info.outputs_written;
   mesh->info.per_primitive_outputs &= ~cull_bit;
   mesh->info.outputs_written &= ~cull_bit;
   const bool plain = (original->info.min_subgroup_size == 64 || wave64_promote) &&
                      bc250_safe_direct_candidate(mesh, true, 16, false);
   mesh->info.per_primitive_outputs = saved_pp;
   mesh->info.outputs_written = saved_written;
   if (getenv("BC250_TRACE_COMPILE") && (wave64_promote || cull_bit))
      fprintf(stderr, "BC250 MESH SAFE PIECES EXT: wave64=%u cull_primitive=%u plain=%u\n",
              wave64_promote, cull_bit != 0, plain);
   const bool ordinary_piece = device->compiler_info.key.bc250_mesh_safe_bary_last &&
                               original->info.min_subgroup_size == 32 && owned_corners;
   mesh->info.mesh.max_primitives_out = original_p;
   unsigned pieces = 0;
   if (plain || owned_corners) {
      NIR_PASS(_, mesh, nir_shader_intrinsics_pass, bc250_lower_application_constants,
                 nir_metadata_control_flow, pipeline);
      NIR_PASS(_, fragment, nir_shader_intrinsics_pass, bc250_lower_application_constants,
                 nir_metadata_control_flow, pipeline);
   }
   uint32_t staged[4] = {0};
   uint64_t payload = 0;
   radv_bc250_split_piece_primid = device->compiler_info.hw.bc250_mesh_piece_primid && owned_corners;
   const bool piece_split = (plain || owned_corners) &&
      radv_bc250_split_mesh(mesh, NULL, fragment, true, false, true, false,
                            device->compiler_info.key.bc250_output_regions, true, 64,
                            ms->bc250_fit_min_pieces, &pieces);
   radv_bc250_split_piece_primid = false;
   if (!piece_split ||
       pieces < 2 || (owned_corners ?
          !radv_bc250_mesh_safe_owned(&mesh, &fragment, true, bary, ordinary_piece, ordinary_piece,
             device->compiler_info.key.bc250_bary_io16, ordinary_piece,
             device->compiler_info.key.mesh_shader_queries,
             device->compiler_info.key.bc250_mesh_direct_read & RADV_BC250_DIRECT_READ_ALL, staged, &payload) :
          !bc250_safe_direct_candidate(mesh, true, 16, false))) {
      ralloc_free(mesh);
      ralloc_free(fragment);
      return false;
   }
   if (getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH SAFE PIECES: V=%u P=%u pieces=%u piece_P=%u bound=%u ordered=1 count_setup=indirect_only\n",
              mesh->info.mesh.max_vertices_out, original_p, pieces, mesh->info.mesh.max_primitives_out,
              3 * mesh->info.mesh.max_primitives_out);
   ralloc_free(ms->nir);
   ralloc_free(fs->nir);
   ms->nir = mesh;
   fs->nir = fragment;
   ms->bc250_safe_fast = !owned_corners;
   ms->bc250_safe_owned = owned_corners;
   if (ordinary_piece)
      ms->bc250_safe_bary_tiny = ms->bc250_expanded = ms->bc250_private_tris = true;
   if (owned_corners) {
      /* RADV_BC250_MESH_PP_SHARE needs the real provoking corner (0x5: dynamic, not shared). */
      ms->bc250_compact_owned = !device->compiler_info.bc250x.pp_share ? 1 :
         gfx_state->key.gfx_state.dynamic_provoking_vtx_mode ? 0x5 :
         gfx_state->key.gfx_state.rs.provoking_vtx_last ? 0x4 : 0x1;
      ms->bc250_pp_locations = payload;
      memcpy(ms->bc250_index_staging, staged, sizeof(staged));
   }
   ms->bc250_split_mesh = true;
   ms->bc250_split_pieces = pieces;
   pipeline->bc250_direct_split_pieces = pieces;
   pipeline->bc250_split_order_free = false;
   return true;
}

/* Integer polynomial proof modulo 2^32. Unknown SSA scalars are independent
 * symbols, never assumed equal. This recognizes both N*p+c and an optimizer's
 * distributed form (e.g. p=8*i+k, vertex=16*i+2*k+c). */
struct bc250_affine {
   nir_scalar term[16];
   uint32_t coeff[16], constant;
   unsigned count;
};

static bool
bc250_affine_add(struct bc250_affine *out, nir_scalar value, uint32_t coefficient, unsigned *budget)
{
   if (!coefficient)
      return true;
   if (!*budget)
      return false;
   --*budget;
   value = nir_scalar_chase_movs(value);
   if (value.def->bit_size != 32)
      return false;
   if (nir_scalar_is_const(value)) {
      out->constant += coefficient * nir_scalar_as_uint(value);
      return true;
   }
   if (nir_scalar_is_alu(value)) {
      const nir_op op = nir_scalar_alu_op(value);
      if (op == nir_op_iadd || op == nir_op_isub) {
         return bc250_affine_add(out, nir_scalar_chase_alu_src(value, 0), coefficient, budget) &&
                bc250_affine_add(out, nir_scalar_chase_alu_src(value, 1),
                                 op == nir_op_isub ? -coefficient : coefficient, budget);
      }
      if (op == nir_op_imul || op == nir_op_ishl) {
         nir_scalar a = nir_scalar_chase_alu_src(value, 0);
         nir_scalar b = nir_scalar_chase_alu_src(value, 1);
         if (nir_scalar_is_const(b)) {
            uint32_t factor = nir_scalar_as_uint(b);
            if (op == nir_op_ishl) {
               if (factor >= 32)
                  return false;
               factor = 1u << factor;
            }
            return bc250_affine_add(out, a, coefficient * factor, budget);
         }
         if (op == nir_op_imul && nir_scalar_is_const(a))
            return bc250_affine_add(out, b, coefficient * nir_scalar_as_uint(a), budget);
      }
   }
   for (unsigned i = 0; i < out->count; i++) {
      if (out->term[i].def == value.def && out->term[i].comp == value.comp) {
         out->coeff[i] += coefficient;
         return true;
      }
   }
   if (out->count == ARRAY_SIZE(out->term))
      return false;
   out->term[out->count] = value;
   out->coeff[out->count++] = coefficient;
   return true;
}

static bool
bc250_private_index(nir_scalar value, nir_scalar index, unsigned corners, unsigned corner)
{
   struct bc250_affine difference = {.constant = -corner};
   unsigned budget = 256;
   if (!bc250_affine_add(&difference, value, 1, &budget) ||
       !bc250_affine_add(&difference, index, -corners, &budget) || difference.constant)
      return false;
   for (unsigned i = 0; i < difference.count; i++)
      if (difference.coeff[i])
         return false;
   return true;
}

bool
radv_bc250_prepare_bary_affine(const struct radv_compiler_info *ci, struct radv_shader_stage *ms,
                          nir_shader *fs)
{
   nir_shader *original = ms->nir;
   /* A shader with no output stores and only explicit empty count writes uses
    * the ordinary GFX10 fully-culled dummy. Do not shrink a declaration when
    * application output stores/loads could still need its original storage. */
   /* The shrink also serves the fallback routes (without SAFE_BARY/AFFINE): an empty shader then
    * fits the ordinary expansion instead of being refused for its declared size. */
   if (ci->key.bc250_mesh_safe_bary_last && original && !original->info.mesh.nv &&
       !original->info.outputs_read && !original->info.outputs_written &&
       !original->info.workgroup_size_variable &&
       original->info.workgroup_size[0] * original->info.workgroup_size[1] *
          original->info.workgroup_size[2] <= 256) {
      bool empty = true, count_written = false;
      nir_foreach_block(block, nir_shader_get_entrypoint(original)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
               continue;
            count_written = true;
            empty &= (nir_src_is_const(in->src[0]) && !nir_src_as_uint(in->src[0])) ||
                     (nir_src_is_const(in->src[1]) && !nir_src_as_uint(in->src[1]));
         }
      }
      /* RADV_BC250_MESH_SAFE_SPLIT_PIECES: never calling SetMeshOutputsEXT also outputs nothing
       * (VK_EXT_mesh_shader: the counts are then zero). */
      if (empty && (count_written || debug_get_bool_option("RADV_BC250_MESH_SAFE_SPLIT_PIECES", false))) {
         original->info.mesh.max_vertices_out = original->info.mesh.max_primitives_out = 1;
         if (!ci->key.bc250_mesh_safe_bary || !ci->key.bc250_mesh_safe_bary_affine) {
            /* Make both counts literally zero: the output is then nothing at all (no vertex, no
             * primitive), which needs no protected route. */
            nir_foreach_block(block, nir_shader_get_entrypoint(original)) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
                  if (in->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
                     continue;
                  nir_builder cb = nir_builder_at(nir_before_instr(instr));
                  nir_src_rewrite(&in->src[0], nir_imm_int(&cb, 0));
                  nir_src_rewrite(&in->src[1], nir_imm_int(&cb, 0));
               }
            }
            ms->bc250_empty_output = true;
            if (getenv("BC250_TRACE_COMPILE"))
               fprintf(stderr, "BC250 MESH EMPTY: no output stores; declaration shrunk for the fallback\n");
            return false;
         }
         ms->bc250_safe_fast = ms->bc250_safe_bary_affine = true;
         if (getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 MESH SAFE EMPTY: no output stores; every count is empty\n");
         return true;
      }
   }
   if (!ci->key.bc250_mesh_safe_bary || !ci->key.bc250_mesh_safe_bary_affine ||
       ci->key.bc250_mesh_autocull_all || !original || (!fs && !ci->key.bc250_mesh_safe_bary_last) ||
       original->info.mesh.nv ||
       (original->info.task_payload_size && !ci->key.bc250_mesh_safe_bary_last) ||
       original->info.outputs_read || original->info.workgroup_size_variable ||
       ((original->info.outputs_read_indirectly | original->info.outputs_written_indirectly) &
        VARYING_BIT_PRIMITIVE_INDICES) || !(original->info.outputs_written & VARYING_BIT_POS) ||
       !original->info.mesh.max_vertices_out || original->info.mesh.max_vertices_out > 256 ||
       original->info.workgroup_size[0] * original->info.workgroup_size[1] * original->info.workgroup_size[2] > 256 ||
       (original->info.per_primitive_outputs & ~VARYING_BIT_PRIMITIVE_INDICES) ||
       (original->info.outputs_written & ~(VARYING_BIT_POS | VARYING_BIT_PSIZ | VARYING_BIT_PRIMITIVE_INDICES |
                                          (UINT64_C(0xffffffff) << VARYING_SLOT_VAR0))))
      return false;
   if (original->info.mesh.primitive_type != MESA_PRIM_POINTS &&
       original->info.mesh.primitive_type != MESA_PRIM_LINES && original->info.mesh.primitive_type != MESA_PRIM_TRIANGLES)
      return false;
   if (bc250_mesh_has_16bit_outputs(original) &&
       !(ci->key.bc250_bary_io16 && ci->key.bc250_mesh_safe_bary_last))
      return false;
   const unsigned corners = mesa_vertices_per_prim(original->info.mesh.primitive_type);
   nir_shader *mesh = nir_shader_clone(NULL, original);
   NIR_PASS(_, mesh, nir_lower_vars_to_ssa);
   NIR_PASS(_, mesh, nir_opt_copy_prop);
   NIR_PASS(_, mesh, nir_opt_constant_folding);
   unsigned max_prims = 0, stores = 0, slots = 0;
   nir_foreach_shader_out_variable(var, mesh) {
      if (var->data.location != VARYING_SLOT_PRIMITIVE_INDICES) {
         if (!glsl_type_is_array(var->type))
            goto decline;
         slots += glsl_count_attribute_slots(glsl_get_array_element(var->type), false);
      }
   }
   nir_foreach_block(block, nir_shader_get_entrypoint(mesh)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
            if (!nir_src_is_const(in->src[0]) || !nir_src_is_const(in->src[1]))
               goto decline;
            const unsigned v = nir_src_as_uint(in->src[0]), p = nir_src_as_uint(in->src[1]);
            if (v && p && (v > original->info.mesh.max_vertices_out || p > original->info.mesh.max_primitives_out ||
                           p > 256 / corners || v != corners * p))
               goto decline;
            if (v && p)
               max_prims = MAX2(max_prims, p);
         }
         if (in->intrinsic != nir_intrinsic_store_deref && in->intrinsic != nir_intrinsic_copy_deref)
            continue;
         nir_deref_instr *deref = nir_src_as_deref(in->src[0]);
         if (nir_deref_instr_get_variable(deref)->data.location != VARYING_SLOT_PRIMITIVE_INDICES ||
             !(deref->modes & nir_var_shader_out))
            continue;
         if (in->intrinsic != nir_intrinsic_store_deref || deref->deref_type != nir_deref_type_array ||
             nir_deref_instr_parent(deref)->deref_type != nir_deref_type_var ||
             in->src[1].ssa->num_components != corners || nir_intrinsic_write_mask(in) != BITFIELD_MASK(corners))
            goto decline;
         nir_scalar index = nir_scalar_chase_movs(nir_get_scalar(deref->arr.index.ssa, 0));
         for (unsigned c = 0; c < corners; c++) {
            if (!bc250_private_index(nir_get_scalar(in->src[1].ssa, c), index, corners, c))
               goto decline;
         }
         stores++;
      }
   }
   /* Preserve the split culling policy. Large triangles are admitted only
    * without an FS, where the original route has no shader culler. Keep the
    * original vertex declaration (and therefore storage) even when only a
    * prefix is exported. Affine points may reuse the existing raw exporter's
    * scratch ring: this proof introduces no staging or new ring accesses. */
   if (!stores || !max_prims ||
       (ci->key.bc250_mesh_safe_bary_last ? mesh->info.mesh.max_vertices_out < corners * max_prims :
                                          mesh->info.mesh.max_vertices_out != corners * max_prims) ||
       (corners == 3 && max_prims >= 24 && (fs || !ci->key.bc250_mesh_safe_bary_last)) ||
       (!(ci->key.bc250_mesh_safe_bary_last && corners == 1) &&
        align(mesh->info.shared_size, 16) + 128 + 16 * slots * mesh->info.mesh.max_vertices_out +
           corners * max_prims >= 30 * 1024))
      goto decline;
   if (fs && corners == 3 && radv_bc250_mesh_fs_refused(ci, fs)) {
      int raw_slot, flat_slot;
      if (ci->hw.bc250_bary_no_ref || !radv_bc250_bary_ref_slots(mesh, fs, &raw_slot, &flat_slot))
         goto decline;
   }
   mesh->info.mesh.max_primitives_out = max_prims;
   ms->nir = mesh;
   ms->bc250_safe_fast = ms->bc250_safe_bary_affine = true;
   if (getenv("BC250_TRACE_COMPILE"))
      fprintf(stderr, "BC250 MESH SAFE BARY AFFINE: corners=%u count_bound=%u referenced=proved backjump_checked=%u\n",
              corners, max_prims, corners - 1);
   ralloc_free(original);
   return true;
decline:
   ralloc_free(mesh);
   return false;
}

VkResult
radv_bc250_prepare_task(struct radv_device *device,
                       struct radv_graphics_pipeline *pipeline,
                       const struct radv_graphics_pipeline_state *gfx_state)
{
   struct radv_shader_stage *stages = gfx_state->stages;
   if (device->compiler_info.key.bc250_mesh_safe_pieces &&
       !device->compiler_info.key.bc250_mesh_amd_size && !device->compiler_info.key.bc250_mesh_min2waves &&
       stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE &&
       stages[MESA_SHADER_MESH].stage == MESA_SHADER_MESH &&
       (stages[MESA_SHADER_MESH].key.subgroup_required_size != RADV_REQUIRED_WAVE32 ||
        device->compiler_info.key.bc250_mesh_safe_bary_last) &&
       stages[MESA_SHADER_FRAGMENT].stage == MESA_SHADER_FRAGMENT &&
       pipeline->base.type == RADV_PIPELINE_GRAPHICS &&
       !(pipeline->base.create_flags & bc250_refused_create_flags(device)) &&
       !gfx_state->key.gfx_state.has_multiview_view_index && !gfx_state->key.gfx_state.vrs_may_be_enabled) {
      for (unsigned st = MESA_SHADER_FRAGMENT; st <= MESA_SHADER_MESH; st++) {
         if (st != MESA_SHADER_FRAGMENT && st != MESA_SHADER_MESH)
            continue;
         if (!stages[st].nir) {
            struct radv_spirv_to_nir_options options = {.lower_view_index_to_zero = true};
            stages[st].nir = radv_shader_spirv_to_nir_cached(&device->compiler_info, NULL, &stages[st], &options, false);
         }
      }
      if (bc250_prepare_safe_pieces(device, pipeline, gfx_state))
         return radv_bc250_prepare_direct_split(device, pipeline);
   }
   if (device->compiler_info.key.bc250_mesh_safe_fast &&
       !device->compiler_info.key.bc250_mesh_amd_size && !device->compiler_info.key.bc250_mesh_min2waves &&
       stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE &&
       stages[MESA_SHADER_MESH].stage == MESA_SHADER_MESH &&
       stages[MESA_SHADER_FRAGMENT].stage == MESA_SHADER_FRAGMENT &&
       pipeline->base.type == RADV_PIPELINE_GRAPHICS &&
       !(pipeline->base.create_flags & bc250_refused_create_flags(device)) &&
       !gfx_state->key.gfx_state.has_multiview_view_index && !gfx_state->key.gfx_state.vrs_may_be_enabled) {
      for (unsigned st = MESA_SHADER_FRAGMENT; st <= MESA_SHADER_MESH; st++) {
         if (st != MESA_SHADER_FRAGMENT && st != MESA_SHADER_MESH)
            continue;
         if (!stages[st].nir) {
            struct radv_spirv_to_nir_options options = {.lower_view_index_to_zero = true};
            stages[st].nir = radv_shader_spirv_to_nir_cached(&device->compiler_info, NULL, &stages[st], &options, false);
         }
      }
      nir_shader *fast_mesh = stages[MESA_SHADER_MESH].nir;
      const unsigned fast_v = fast_mesh ? fast_mesh->info.mesh.max_vertices_out : 0;
      const unsigned fast_p = fast_mesh ? fast_mesh->info.mesh.max_primitives_out : 0;
      const unsigned fast_threads = fast_mesh ? fast_mesh->info.workgroup_size[0] *
         fast_mesh->info.workgroup_size[1] * fast_mesh->info.workgroup_size[2] : 0;
      if (radv_bc250_prepare_bary_affine(&device->compiler_info, &stages[MESA_SHADER_MESH], stages[MESA_SHADER_FRAGMENT].nir))
         return VK_SUCCESS;
      /* Reuse the constant-count proof before the legacy declaration-driven
       * split. A large declaration with a tiny proven count needs one bounded
       * private export, not replay pieces. No Task transport is changed here. */
      if (device->compiler_info.key.bc250_mesh_safe_bary &&
          device->compiler_info.key.bc250_mesh_safe_bary_tiny &&
          device->compiler_info.key.bc250_mesh_safe_owned && device->compiler_info.key.bc250_mesh_safe_corners &&
          !device->compiler_info.key.bc250_mesh_autocull_all && fast_mesh &&
          (fast_p > 85 || fast_mesh->info.mesh.primitive_type != MESA_PRIM_TRIANGLES) &&
          radv_bc250_mesh_safe_owned(&stages[MESA_SHADER_MESH].nir, &stages[MESA_SHADER_FRAGMENT].nir,
             true, radv_bc250_mesh_private_bary(&device->compiler_info, stages[MESA_SHADER_FRAGMENT].nir), true,
             device->compiler_info.key.bc250_mesh_safe_bary_last, device->compiler_info.key.bc250_bary_io16, false,
             device->compiler_info.key.mesh_shader_queries,
             device->compiler_info.key.bc250_mesh_direct_read & RADV_BC250_DIRECT_READ_ALL,
             stages[MESA_SHADER_MESH].bc250_index_staging, &stages[MESA_SHADER_MESH].bc250_pp_locations)) {
         struct radv_shader_stage *ms = &stages[MESA_SHADER_MESH];
         ms->bc250_safe_owned = ms->bc250_safe_bary_tiny = true;
         ms->bc250_expanded = ms->bc250_private_tris = true;
         ms->bc250_compact_owned = 1;
         if (getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 MESH SAFE BARY TINY: declared_V=%u declared_P=%u proven_V=%u proven_P=%u\n",
                    fast_v, fast_p, ms->nir->info.mesh.max_vertices_out, ms->nir->info.mesh.max_primitives_out);
         return VK_SUCCESS;
      }
      const unsigned fast_bound = fast_v <= 32 ? MIN2(fast_v, 3 * fast_p) : 3 * fast_p;
      const unsigned fast_launch = MAX4(fast_threads, fast_v, fast_p, fast_bound);
      /* SAFE_FAST checks connectivity in one wave. Wave64 covers P<=64; wave32
       * covers P<=32 when all declared vertices fit its 32-index domain. The
       * latest-copy bound determines the launch, not the API V. */
      const bool check_wave_fits = fast_mesh &&
         ((fast_mesh->info.min_subgroup_size == 64 && fast_p <= 64) ||
          (fast_mesh->info.min_subgroup_size == 32 && fast_p <= 32 && fast_v <= 32));
      const bool proven_plain = check_wave_fits && fast_launch <= 256 &&
         bc250_safe_direct_candidate(fast_mesh, true, 16, false) &&
         !radv_bc250_mesh_fs_refused(&device->compiler_info, stages[MESA_SHADER_FRAGMENT].nir);
      if (proven_plain || (!radv_bc250_mesh_fs_refused(&device->compiler_info, stages[MESA_SHADER_FRAGMENT].nir) &&
                          bc250_ordered_eligible(fast_mesh, stages[MESA_SHADER_FRAGMENT].nir))) {
         stages[MESA_SHADER_MESH].bc250_safe_fast = true;
         if (getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 MESH SAFE FAST: V=%u P=%u bound=%u launch=%u single-pass lossless\n",
                    fast_v, fast_p, fast_bound, fast_launch);
         return VK_SUCCESS;
      }
   }
   /* Autocull retains its separately validated direct route until it is integrated
    * into the ordered producer. */
   if (device->compiler_info.key.bc250_mesh_safe_ordered &&
       radv_device_physical(device)->bc250_hybrid_task &&
       !device->compiler_info.key.bc250_mesh_safe_autocull &&
       device->compiler_info.key.bc250_mesh_safe_direct &&
       device->compiler_info.key.bc250_mesh_safe_parallel &&
       !device->compiler_info.key.bc250_mesh_amd_size && !device->compiler_info.key.bc250_mesh_min2waves &&
       stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE &&
       stages[MESA_SHADER_MESH].stage == MESA_SHADER_MESH &&
       stages[MESA_SHADER_MESH].key.subgroup_required_size != RADV_REQUIRED_WAVE32 &&
       stages[MESA_SHADER_FRAGMENT].stage == MESA_SHADER_FRAGMENT &&
       pipeline->base.type == RADV_PIPELINE_GRAPHICS &&
       !(pipeline->base.create_flags & bc250_refused_create_flags(device)) &&
       !gfx_state->key.gfx_state.has_multiview_view_index && !gfx_state->key.gfx_state.vrs_may_be_enabled) {
      for (unsigned stage = MESA_SHADER_FRAGMENT; stage <= MESA_SHADER_MESH; stage++) {
         if (stage != MESA_SHADER_FRAGMENT && stage != MESA_SHADER_MESH)
            continue;
         if (!stages[stage].nir) {
            struct radv_spirv_to_nir_options options = {.lower_view_index_to_zero = true};
            stages[stage].nir = radv_shader_spirv_to_nir_cached(&device->compiler_info, NULL, &stages[stage], &options, false);
         }
      }
      if (bc250_ordered_eligible(stages[MESA_SHADER_MESH].nir, stages[MESA_SHADER_FRAGMENT].nir))
         return bc250_prepare_ordered(device, pipeline, gfx_state);
   }
   /* Native TASK cannot use the private compute producer's mesh replay ABI.
    * Preserve that path for shaders needing expansion/replay, while keeping
    * eligible pipelines on native task rings. Decide before shader-cache lookup.
    */
   if (device->compiler_info.key.bc250_native_task &&
       stages[MESA_SHADER_TASK].stage != MESA_SHADER_NONE) {
      struct radv_shader_stage *ms = &stages[MESA_SHADER_MESH];
      if (ms->stage == MESA_SHADER_NONE)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      if (!ms->nir) {
         struct radv_spirv_to_nir_options options = {
            .lower_view_index_to_zero = !gfx_state->key.gfx_state.has_multiview_view_index,
            .lower_view_index_to_device_index = ms->key.view_index_from_device_index,
         };
         ms->nir = radv_shader_spirv_to_nir_cached(&device->compiler_info, NULL, ms, &options, false);
      }
      if (!ms->nir)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      const bool native_eligible =
         ms->nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
         ms->nir->info.mesh.max_primitives_out <= 63 &&
         ms->nir->info.mesh.max_vertices_out <= 256 &&
         !(ms->nir->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) &&
         !radv_bc250_mesh_needs_expansion(ms->nir) &&
         !gfx_state->key.gfx_state.has_multiview_view_index;
      if (getenv("BC250_TRACE_NATIVE_TASK"))
         fprintf(stderr, "BC250 task route: %s\n", native_eligible ? "native" : "hybrid-required");
      if (native_eligible)
         return VK_SUCCESS;
      if (!radv_device_physical(device)->bc250_hybrid_task)
         return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   bool synthetic_task = false;
   if (stages[MESA_SHADER_TASK].stage == MESA_SHADER_NONE) {
      if (!device->compiler_info.key.bc250_split_mesh ||
          !device->compiler_info.key.bc250_expand_primitives ||
          stages[MESA_SHADER_MESH].stage == MESA_SHADER_NONE)
         return VK_SUCCESS;
      struct radv_shader_stage *ms = &stages[MESA_SHADER_MESH];
      if (!ms->nir) {
         /* Match the ordinary graphics loader even when this mesh is small
          * enough not to need an internal producer. */
         struct radv_spirv_to_nir_options options = {
            .lower_view_index_to_zero = !gfx_state->key.gfx_state.has_multiview_view_index,
            .lower_view_index_to_device_index = ms->key.view_index_from_device_index,
         };
         ms->nir = radv_shader_spirv_to_nir_cached(&device->compiler_info, NULL, ms, &options, false);
      }
      nir_shader *mesh_only = ms->nir;
      /* RADV_BC250_MESH_CLIPCULL_CONST before the route decision and the split. With
       * RADV_BC250_MESH_REFERENCE (CULLDIST=keep) a Mesh shader the raw route admits keeps its
       * clip/cull distances (POS1), as in the reference driver. */
      const bool reference_keep_cd = device->compiler_info.hw.bc250_mesh_reference_keep_cd;
      if (!reference_keep_cd ||
          !radv_bc250_mesh_amd_route(true, false, false, mesh_only, false,
                                     gfx_state->key.gfx_state.has_multiview_view_index, true))
         bc250_prepare_clip_cull_const(device, pipeline, gfx_state, stages);
      if (radv_bc250_mesh_amd_route(device->compiler_info.key.bc250_mesh_amd,
                                    device->compiler_info.key.bc250_mesh_fast, false, mesh_only, false,
                                    gfx_state->key.gfx_state.has_multiview_view_index,
                                    device->compiler_info.hw.bc250_mesh_allow_pos1 || reference_keep_cd)) {
         /* Native unsplit route: direct DRAW_INDEX_AUTO, indirect
          * DISPATCH_MESH_INDIRECT_MULTI, no setup helper, no producer. */
         if (getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 MESH AMD route: V=%u P=%u no split, no setup dispatch\n",
                    mesh_only->info.mesh.max_vertices_out, mesh_only->info.mesh.max_primitives_out);
         return VK_SUCCESS;
      }
      /* ms->bc250_fit_min_pieces > 1: an LDS fit retry splits a Mesh shader
       * that the policy runs unsplit (see radv_graphics_pipeline_init). */
      /* RADV_BC250_MESH_SPLIT_ANY refused-pipeline retry: Mesh-only lines/points take the
       * direct split (e.g. 255 lines exceed one 256-vertex expansion). */
      /* Only refused pipelines reach the retry: their unsplit form did not fit (lanes or LDS). */
      const bool lines_points_overflow = radv_bc250_split_refused_retry &&
         mesh_only->info.mesh.primitive_type != MESA_PRIM_TRIANGLES;
      if (!lines_points_overflow &&
          (mesh_only->info.mesh.primitive_type != MESA_PRIM_TRIANGLES ||
          !(mesh_only->info.mesh.max_primitives_out > 85 ||
            ((mesh_only->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) &&
             mesh_only->info.mesh.max_primitives_out > 63) ||
            radv_bc250_mesh_culldist_split(&device->compiler_info, mesh_only) ||
            ms->bc250_fit_min_pieces > 1)))
         return VK_SUCCESS;
      if (mesh_only->info.task_payload_size)
         return VK_ERROR_FEATURE_NOT_PRESENT;

      if (device->compiler_info.key.bc250_direct_split &&
          pipeline->base.type == RADV_PIPELINE_GRAPHICS &&
          !(pipeline->base.create_flags & bc250_refused_create_flags(device)) &&
          !gfx_state->key.gfx_state.has_multiview_view_index &&
          stages[MESA_SHADER_FRAGMENT].stage != MESA_SHADER_NONE) {
         struct radv_shader_stage *fs = &stages[MESA_SHADER_FRAGMENT];
         if (!fs->nir) {
            struct radv_spirv_to_nir_options options = {.lower_view_index_to_zero = true};
            fs->nir = radv_shader_spirv_to_nir_cached(&device->compiler_info, NULL, fs, &options, false);
         }
         if (device->compiler_info.key.bc250_single_piece && ms->bc250_fit_min_pieces <= 1 &&
             bc250_primitive_count_bound(mesh_only) <= 63) {
            /* Preserve the original dispatch, DrawID and application constants.
             * A single child uses base zero without the private argument ABI. */
            unsigned single = 0;
            if (!radv_bc250_split_mesh(mesh_only, NULL, fs->nir, false, false,
                                      device->compiler_info.key.bc250_balanced_slices, device->compiler_info.key.bc250_parallel_cull, device->compiler_info.key.bc250_output_regions,
                                      device->compiler_info.key.bc250_mesh_compact_lds, device->compiler_info.key.bc250_piece_prims, ms->bc250_fit_min_pieces, &single))
               return VK_ERROR_FEATURE_NOT_PRESENT;
            stages[MESA_SHADER_MESH].bc250_split_mesh = true;
            stages[MESA_SHADER_MESH].bc250_split_pieces = single;
            if (getenv("BC250_TRACE_COMPILE"))
               fprintf(stderr, "BC250 single piece: native dispatch, no setup helper\n");
            return VK_SUCCESS;
         }
         unsigned pieces = 0;
         /* Before bc250_lower_application_constants (it adds global loads). */
         const char *order_reason = NULL;
         const bool order_free_fs = device->compiler_info.key.bc250_split_prep_free &&
                                    bc250_fs_order_independent(fs->nir, &order_reason);
         /* Preserve application push constants while private draw pointers use
          * the graphics constant block; DrawID remains native. */
         NIR_PASS(_, mesh_only, nir_shader_intrinsics_pass, bc250_lower_application_constants,
                  nir_metadata_control_flow, pipeline);
         NIR_PASS(_, fs->nir, nir_shader_intrinsics_pass, bc250_lower_application_constants,
                  nir_metadata_control_flow, pipeline);
         /* RADV_BC250_MESH_PIECE_PRIMID also for the direct split fallback: the same slicing keeps the
          * application's PrimitiveId values. */
         radv_bc250_split_piece_primid = device->compiler_info.hw.bc250_mesh_piece_primid;
         const bool split_ok = radv_bc250_split_mesh(mesh_only, NULL, fs->nir, true,
                                    device->compiler_info.key.bc250_split_prep_free, device->compiler_info.key.bc250_balanced_slices, device->compiler_info.key.bc250_parallel_cull, device->compiler_info.key.bc250_output_regions,
                                    device->compiler_info.key.bc250_mesh_compact_lds, device->compiler_info.key.bc250_piece_prims, ms->bc250_fit_min_pieces, &pieces);
         radv_bc250_split_piece_primid = false;
         if (!split_ok)
            return VK_ERROR_FEATURE_NOT_PRESENT;
         stages[MESA_SHADER_MESH].bc250_split_mesh = true;
         stages[MESA_SHADER_MESH].bc250_split_pieces = pieces;
         pipeline->bc250_direct_split_pieces = pieces;
         pipeline->bc250_split_order_free = order_free_fs;
         if (device->compiler_info.key.bc250_split_prep_free && device->bc250_env.trace_compile)
            fprintf(stderr, "BC250 split prep-free: %s%s\n",
                    order_free_fs ? "order-independent fragment shader" : "per-draw setup only: ",
                    order_free_fs ? "" : order_reason);
         fprintf(stderr, "BC250 direct mesh-only split enabled: pieces=%u\n", pieces);
         return radv_bc250_prepare_direct_split(device, pipeline);
      }

      /* The internal producer launches (1,1,1) per API group, so the Mesh
       * grid no longer holds the application's NumWorkGroups; the payload
       * carries only the group identity. Refuse it on this path. */
      nir_foreach_function_impl(impl, mesh_only) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_num_workgroups) {
                  fprintf(stderr, "BC250 mesh split rejected: load_num_workgroups (internal amplifier)\n");
                  return VK_ERROR_FEATURE_NOT_PRESENT;
               }
            }
         }
      }

      /* One internal producer per original API mesh group. The payload carries
       * its original 3D identity; child WorkGroupID remains private to replay.
       * Never multiply API draw/indirect counts on the host or lose GroupID.
       */
      nir_builder tb = nir_builder_init_simple_shader(MESA_SHADER_TASK,
         &device->compiler_info.nir_options[MESA_SHADER_TASK], "bc250_mesh_only_amplifier");
      tb.shader->info.workgroup_size[0] = 1;
      tb.shader->info.workgroup_size[1] = 1;
      tb.shader->info.workgroup_size[2] = 1;
      tb.shader->info.task_payload_size = 16;
      nir_store_task_payload(&tb, nir_load_workgroup_id(&tb), nir_imm_int(&tb, 0),
                             .write_mask = 7, .align_mul = 4);
      nir_launch_mesh_workgroups(&tb, nir_imm_ivec3(&tb, 1, 1, 1), .range = 16);
      nir_shader_gather_info(tb.shader, nir_shader_get_entrypoint(tb.shader));
      nir_validate_shader(tb.shader, "BC250 internal mesh-only producer");
      stages[MESA_SHADER_TASK].stage = MESA_SHADER_TASK;
      stages[MESA_SHADER_TASK].nir = tb.shader;
      mesh_only->info.task_payload_size = 16;
      nir_foreach_function_impl(impl, mesh_only) {
         nir_builder mb = nir_builder_create(impl);
         nir_foreach_block(block, impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
               if (in->intrinsic != nir_intrinsic_load_workgroup_id)
                  continue;
               mb.cursor = nir_before_instr(instr);
               nir_def *original = nir_load_task_payload(&mb, in->def.num_components, 32,
                  nir_imm_int(&mb, 0), .align_mul = 4);
               nir_def_rewrite_uses(&in->def, original);
               nir_instr_remove(instr);
            }
         }
         nir_progress(true, impl, nir_metadata_none);
      }
      nir_shader_gather_info(mesh_only, nir_shader_get_entrypoint(mesh_only));
      synthetic_task = true;
      fprintf(stderr, "BC250 internal mesh-only amplification enabled\n");
   }
   if (!radv_device_physical(device)->bc250_hybrid_task ||
       stages[MESA_SHADER_MESH].stage == MESA_SHADER_NONE ||
       pipeline->base.type != RADV_PIPELINE_GRAPHICS ||
       (pipeline->base.create_flags & bc250_refused_create_flags(device)) ||
       gfx_state->key.gfx_state.has_multiview_view_index)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   /* Lower application constants before introducing any private constants. */
   for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
      if (stages[s].stage == MESA_SHADER_NONE)
         continue;
      if (stages[s].layout.mapping) {
         if (getenv("BC250_TRACE_COMPILE"))
            fprintf(stderr, "BC250 hybrid task refused: stage layout mapping\n");
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      if (!stages[s].nir) {
         struct radv_spirv_to_nir_options options = {
            .lower_view_index_to_zero = true,
         };
         stages[s].nir = radv_shader_spirv_to_nir_cached(&device->compiler_info, NULL, &stages[s], &options, false);
      }
      NIR_PASS(_, stages[s].nir, nir_shader_intrinsics_pass, bc250_lower_application_constants,
               nir_metadata_control_flow, NULL);
   }

   /* RADV_BC250_MESH_CLIPCULL_CONST before the split (a mesh-only shader already had it). */
   bc250_prepare_clip_cull_const(device, pipeline, gfx_state, stages);
   nir_shader *task = stages[MESA_SHADER_TASK].nir;
   nir_shader *mesh = stages[MESA_SHADER_MESH].nir;
   /* RADV_BC250_MESH_SPLIT_ANY refused-pipeline retry: the >85 primitive split exists for
    * 3-vertex expansion. Lines/points whose vertices fit one 256-lane group, without
    * CullPrimitive, run unsplit like their Mesh-only form. */
   const bool retry_unsplit = radv_bc250_split_refused_retry &&
      mesh->info.mesh.primitive_type != MESA_PRIM_TRIANGLES &&
      !(mesh->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) &&
      mesa_vertices_per_prim(mesh->info.mesh.primitive_type) * mesh->info.mesh.max_primitives_out <= 256 &&
      mesh->info.mesh.max_vertices_out <= 256 && stages[MESA_SHADER_MESH].bc250_fit_min_pieces <= 1;
   if (device->compiler_info.key.bc250_split_mesh && !retry_unsplit &&
       (mesh->info.mesh.max_primitives_out > 85 ||
        (mesh->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) ||
        (mesh->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
         radv_bc250_mesh_culldist_split(&device->compiler_info, mesh)) ||
        stages[MESA_SHADER_MESH].bc250_fit_min_pieces > 1)) {
      unsigned pieces = 0;
      radv_bc250_split_task_grid_fold = device->compiler_info.bc250x.task_grid_fold;
      const bool split_ok = device->compiler_info.key.bc250_expand_primitives &&
          radv_bc250_split_mesh(mesh, task, stages[MESA_SHADER_FRAGMENT].nir, false, false, device->compiler_info.key.bc250_balanced_slices, device->compiler_info.key.bc250_parallel_cull, device->compiler_info.key.bc250_output_regions,
                                 device->compiler_info.key.bc250_mesh_compact_lds, device->compiler_info.key.bc250_piece_prims, stages[MESA_SHADER_MESH].bc250_fit_min_pieces, &pieces);
      radv_bc250_split_task_grid_fold = false;
      if (!split_ok)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      stages[MESA_SHADER_MESH].bc250_split_mesh = true;
      stages[MESA_SHADER_MESH].bc250_split_pieces = pieces;
   }
   if (task->info.task_payload_size > 16384 || mesh->info.task_payload_size > 16384) {
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 hybrid task refused: payload over 16 KiB\n");
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   pipeline->bc250_payload_stride = MAX2(16, align(MAX2(task->info.task_payload_size,
                                                       mesh->info.task_payload_size), 16));

   nir_shader *producer = nir_shader_clone(NULL, task);
   nir_lower_task_shader_options lower = {
      .payload_to_shared_for_atomics = true,
      .payload_to_shared_for_small_types = true,
   };
   NIR_PASS(_, producer, nir_lower_task_shader, lower);
   nir_validate_shader(producer, "BC250 TASK normalized");
   if (producer->info.shared_size > 65536) {
      if (getenv("BC250_TRACE_COMPILE"))
         fprintf(stderr, "BC250 hybrid task refused: producer shared %u over 64 KiB\n", producer->info.shared_size);
      ralloc_free(producer);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   bool is_producer = true;
   NIR_PASS(_, producer, nir_shader_intrinsics_pass, bc250_lower_intrinsic,
            nir_metadata_none, &is_producer);
   NIR_PASS(_, producer, nir_remove_dead_variables, nir_var_mem_task_payload, NULL);
   /* Count API invocations, not workgroups. Exactly one
    * GDS atomic per workgroup (guarded to a single thread); the counter word is
    * snapshotted into the query result slots by radv_begin/end_pipeline_stat_query. */
   {
      nir_function_impl *producer_impl = nir_shader_get_entrypoint(producer);
      nir_builder pb = nir_builder_create(producer_impl);
      pb.cursor = nir_before_cf_list(&producer_impl->body);
      nir_def *lid = nir_load_local_invocation_index(&pb);
      /* Zero this workgroup's mesh record slot before any application code.
       * Task shaders may call EmitMeshTasksEXT conditionally (culling); the
       * chunk's mesh draw consumes every slot up to the dispatch size, so
       * non-emitting workgroups must leave a zero (no-op) record instead of
       * stale data from an earlier chunk or indirect record reusing scratch. */
      {
         nir_def *slot_addr = nir_iadd(&pb, bc250_pointer(&pb, 0),
                                       nir_u2u64(&pb, nir_imul_imm(&pb, bc250_task_index(&pb), 12)));
         nir_push_if(&pb, nir_ieq_imm(&pb, lid, 0));
         nir_store_global(&pb, nir_imm_ivec3(&pb, 0, 0, 0), slot_addr, .write_mask = 7, .align_mul = 4);
         nir_pop_if(&pb, NULL);
      }
      nir_if *if_lid0 = nir_push_if(&pb, nir_i2b(&pb, nir_ieq(&pb, lid, nir_imm_int(&pb, 0))));
      {
         const unsigned api_invocations = producer->info.workgroup_size[0] *
                                          producer->info.workgroup_size[1] *
                                          producer->info.workgroup_size[2];
         /* Internal producers are not application TASK invocations. */
         if (!synthetic_task)
            nir_gds_atomic_add_amd(&pb, 32, nir_imm_int(&pb, api_invocations),
                                   nir_imm_int(&pb, RADV_SHADER_QUERY_TS_INVOCATION_OFFSET),
                                   nir_imm_int(&pb, 0x100));
      }
      nir_pop_if(&pb, if_lid0);
   }
   producer->info.stage = MESA_SHADER_COMPUTE;
   producer->info.task_payload_size = 0;
   producer->info.name = ralloc_strdup(producer, "bc250_hybrid_task_producer");
   nir_shader_gather_info(producer, nir_shader_get_entrypoint(producer));
   nir_validate_shader(producer, "BC250 COMPUTE producer before serialization");

   /* Application DrawID was lowered before inserting the private DrawID used
    * to index per-task payload records.
    */
   is_producer = false;
   NIR_PASS(_, mesh, nir_shader_intrinsics_pass, bc250_lower_intrinsic,
            nir_metadata_none, &is_producer);
   mesh->info.task_payload_size = 0;
   nir_shader_gather_info(mesh, nir_shader_get_entrypoint(mesh));
   nir_validate_shader(mesh, "BC250 native MESH payload lowering");

   VkPushConstantRange range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct bc250_constants),
   };
   VkDescriptorSetLayout sets[MAX_SETS];
   for (unsigned i = 0; i < gfx_state->layout.num_sets; i++)
      sets[i] = radv_descriptor_set_layout_to_handle(gfx_state->layout.set[i].layout);
   VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = gfx_state->layout.num_sets,
      .pSetLayouts = sets,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &range,
   };
   VkResult result = radv_CreatePipelineLayout(radv_device_to_handle(device), &layout_info, NULL,
                                              &pipeline->bc250_task_layout);
   if (result == VK_SUCCESS) {
      VK_FROM_HANDLE(radv_pipeline_layout, producer_layout, pipeline->bc250_task_layout);
      if (producer_layout->dynamic_shader_stages & VK_SHADER_STAGE_TASK_BIT_EXT)
         producer_layout->dynamic_shader_stages |= VK_SHADER_STAGE_COMPUTE_BIT;
      /* A descriptor buffer graphics pipeline's set layouts are descriptor
       * buffer layouts: create the private compute pipelines the same way
       * (the flag does not enter the shader key). */
      VkPipelineCreateFlags2CreateInfo producer_flags = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
         .flags = pipeline->base.create_flags & VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
      };
      VkComputePipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .pNext = producer_flags.flags ? &producer_flags : NULL,
         .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = vk_shader_module_handle_from_nir(producer),
            .pName = "main",
         },
         .layout = pipeline->bc250_task_layout,
      };
      result = radv_compute_pipeline_create(radv_device_to_handle(device), device->meta_state.cache,
                                            &info, NULL, &pipeline->bc250_task_pipeline);
      if (result == VK_SUCCESS) {
         nir_shader *setup = bc250_build_indirect_setup(device);
         info.stage.module = vk_shader_module_handle_from_nir(setup);
         result = radv_compute_pipeline_create(radv_device_to_handle(device), device->meta_state.cache,
                                               &info, NULL, &pipeline->bc250_setup_pipeline);
         ralloc_free(setup);
      }
   }
   ralloc_free(producer);
   if (result != VK_SUCCESS)
      return result;

   /* Only the private compute pipeline owns executable producer code. The
    * graphics pipeline must never bind TASK or request native task rings.
    */
   ralloc_free(stages[MESA_SHADER_TASK].nir);
   stages[MESA_SHADER_TASK].nir = NULL;
   stages[MESA_SHADER_TASK].stage = MESA_SHADER_NONE;
   stages[MESA_SHADER_MESH].key.has_task_shader = false;
   stages[MESA_SHADER_MESH].bc250_task_replay = true;
   pipeline->active_stages &= ~VK_SHADER_STAGE_TASK_BIT_EXT;
   return VK_SUCCESS;
}

/* Emit one flat chunk: a producer dispatch of n consecutive task groups
 * (global row-major indices [flat_base, flat_base + n)) followed by the mesh
 * draw that consumes its records before the next chunk reuses scratch. The
 * published full size plus the push-constant chunk base restore the
 * application's WorkGroupID/NumWorkGroups semantics; application DrawID remains the original draw record index. */
static void
bc250_emit_chunk(struct radv_cmd_buffer *cmd_buffer, struct radv_graphics_pipeline *pipeline,
                 const struct bc250_constants *constants, uint64_t address, uint64_t flat_base,
                 uint32_t n)
{
   /* The preceding mesh draw can still be reading shared records/payloads.
    * Serialize readers before the next producer overwrites this allocation. */
   if (flat_base)
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_VS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH;

   struct bc250_constants cc = *constants;
   cc.chunk_base = (uint32_t)flat_base;

   radv_meta_begin(cmd_buffer);
   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline->bc250_task_pipeline);
   radv_meta_push_constants(cmd_buffer, pipeline->bc250_task_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(cc), &cc);
   struct radv_dispatch_info dispatch = {.blocks = {n, 1, 1}};
   bc250_chain_dispatch(cmd_buffer, &dispatch, "task_producer");
   radv_meta_end(cmd_buffer);

   cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                   RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_L2;
   if (pipeline->bc250_ordered) {
      /* Publish both shader-readable output records and CP-readable draw counts. */
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_WB_L2;
      radv_emit_cache_flush(cmd_buffer);
      ac_emit_cp_pfp_sync_me(cmd_buffer->cs->b, false);
   }
   uint8_t saved[sizeof(cc)];
   memcpy(saved, cmd_buffer->push_constants, sizeof(saved));
   memcpy(cmd_buffer->push_constants, &cc, sizeof(cc));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
   bc250_chain_consume(cmd_buffer, address, 0, cc.payload, n, 12, cc.chunk_base);
   VkDrawIndirect2InfoKHR draw = {
      .sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR,
      .addressRange = {.address = address, .size = (uint64_t)n * 12, .stride = 12},
      .drawCount = n,
   };
   cmd_buffer->bc250_inside_mesh_draw = true;
   radv_CmdDrawMeshTasksIndirect2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &draw);
   cmd_buffer->bc250_inside_mesh_draw = false;
   memcpy(cmd_buffer->push_constants, saved, sizeof(saved));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
}

/* Emit the application's dispatch as sequential flat chunks of at most
 * BC250_TASK_CHUNK workgroups each, in row-major order. Chunk k covers global
 * task indices [k*CHUNK, min((k+1)*CHUNK, total)). */
static void
bc250_emit_chunks(struct radv_cmd_buffer *cmd_buffer, struct radv_graphics_pipeline *pipeline,
                  const struct bc250_constants *constants, uint64_t address,
                  uint32_t full_x, uint32_t full_y, uint32_t full_z, uint64_t total)
{
   for (uint64_t base = 0; base < total; base += BC250_TASK_CHUNK) {
      uint32_t n = (uint32_t)MIN2(total - base, (uint64_t)BC250_TASK_CHUNK);
      bc250_emit_chunk(cmd_buffer, pipeline, constants, address, base, n);
   }
}

/* Emit one chunk of an indirect record: the producer grid is read from the
 * setup shader's sub-record k (zero for inactive chunks, which makes both the
 * dispatch and the count-0 mesh draw no-ops). */
static void
bc250_emit_indirect_chunk(struct radv_cmd_buffer *cmd_buffer, struct radv_graphics_pipeline *pipeline,
                          const struct bc250_constants *constants, uint64_t address, unsigned k, bool skip_suffix)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct ac_cmdbuf *cs = cmd_buffer->cs->b;
   uint32_t *conditional_buf = NULL;
   unsigned conditional_start = 0;
   if (k && skip_suffix) {
      /* Chunk zero always establishes the shared compute/graphics state.
       * Setup wrote the complete table before it; synchronize PFP once before
       * allowing COND_EXEC to inspect the GPU-written Boolean table. */
      radeon_check_space(device->ws, cs, 16384);
      if (k == 1) {
         cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH |
            RADV_CMD_FLAG_VS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH |
            RADV_CMD_FLAG_INV_L2 | RADV_CMD_FLAG_WB_L2;
         radv_emit_cache_flush(cmd_buffer);
         ac_emit_cp_pfp_sync_me(cs, false);
      }
      conditional_buf = cs->buf;
      conditional_start = cs->cdw;
      /* The Y dispatch dimension is exactly 1 for active chunks, 0 otherwise.
       * Use this Boolean, not the workgroup count: a constant predicate of
       * 4096 failed on BC250 while 1 passed the same full-range fixture. */
      ac_emit_cp_cond_exec(cs, pdev->info.gfx_level, constants->compute_args + (uint64_t)k * 16 + 4, 0);
   }
   if (k)
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_VS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH;

   uint64_t sub_record = constants->compute_args + (uint64_t)k * 16;
   struct bc250_constants cc = *constants;
   cc.chunk_base = (uint32_t)((uint64_t)k * BC250_TASK_CHUNK);

   radv_meta_begin(cmd_buffer);
   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline->bc250_task_pipeline);
   radv_meta_push_constants(cmd_buffer, pipeline->bc250_task_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(cc), &cc);
   struct radv_dispatch_info dispatch = {.indirect_va = sub_record};
   bc250_chain_dispatch(cmd_buffer, &dispatch, "task_producer");
   radv_meta_end(cmd_buffer);

   cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                   RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_L2;
   if (pipeline->bc250_ordered) {
      /* Publish both shader-readable output records and CP-readable draw counts. */
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_WB_L2;
      radv_emit_cache_flush(cmd_buffer);
      ac_emit_cp_pfp_sync_me(cmd_buffer->cs->b, false);
   }
   uint8_t saved[sizeof(cc)];
   memcpy(saved, cmd_buffer->push_constants, sizeof(saved));
   memcpy(cmd_buffer->push_constants, &cc, sizeof(cc));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
   bc250_chain_consume(cmd_buffer, address, sub_record + 12, cc.payload, BC250_TASK_CHUNK, 12, k);
   VkDrawIndirectCount2InfoKHR counted = {
      .sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_COUNT_2_INFO_KHR,
      .addressRange = {.address = address, .size = (uint64_t)BC250_TASK_CHUNK * 12, .stride = 12},
      .countAddressRange = {.address = sub_record + 12, .size = 4},
      .maxDrawCount = BC250_TASK_CHUNK,
   };
   cmd_buffer->bc250_inside_mesh_draw = true;
   radv_CmdDrawMeshTasksIndirectCount2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &counted);
   cmd_buffer->bc250_inside_mesh_draw = false;
   memcpy(cmd_buffer->push_constants, saved, sizeof(saved));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
   if (conditional_buf) {
      /* Never allow a conditional region to span IB chains. On unexpected
       * growth, reject recording rather than submitting an invalid region. */
      if (conditional_buf != cs->buf || cs->cdw - conditional_start - 5 >= 16384) {
         vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_UNKNOWN);
         return;
      }
      conditional_buf[conditional_start + 4] = cs->cdw - conditional_start - 5;
   }
}

/* Task payload scratch lives until command-buffer reset, but must not grow
 * the persistent general upload cache. Keep it on the existing transient list. */
static bool
bc250_alloc_scratch(struct radv_cmd_buffer *cmd_buffer, unsigned bytes,
                    uint64_t *address, void **mapped)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   /* Suballocation is restricted to one execution of a primary buffer. Keep
    * existing allocation semantics for secondary/repeated/simultaneous uses.
    * RADV_BC250_SCRATCH_REUSE=1 also suballocates for primaries that may be
    * submitted again (no ONE_TIME_SUBMIT; vkd3d-proton records all its command
    * lists that way): an arena range is, like a dedicated BO, exclusive to one
    * allocation and alive until the command buffer is reset (the page is bump
    * allocated and only freed at reset), and executions of a primary without
    * SIMULTANEOUS_USE never overlap, so every execution sees the same memory
    * with the same lifetime as with one BO per allocation. */
   const bool arena = bytes <= 65536 && cmd_buffer->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
      ((cmd_buffer->usage_flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) || device->bc250_env.scratch_reuse) &&
      !(cmd_buffer->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) &&
      !radv_bc250_chain_enabled(device) && device->bc250_env.transient_arena;
   if (arena && cmd_buffer->bc250_small_arena) {
      struct radv_cmd_buffer_upload *cur = cmd_buffer->bc250_small_arena;
      const unsigned offset = ALIGN_POT(cur->offset, 256);
      if (offset <= cur->size && bytes <= cur->size - offset) {
         *address = radv_buffer_get_va(cur->upload_bo) + offset;
         *mapped = cur->map + offset;
         cur->offset = offset + bytes;
         if (device->bc250_env.trace_compile)
            fprintf(stderr, "BC250 arena reuse: offset=%u bytes=%u capacity=%llu\n", offset, bytes,
                    (unsigned long long)cur->size);
         return true;
      }
   }
   const unsigned allocation_size = arena ? 65536 : bytes;
   struct radv_cmd_buffer_upload *up = calloc(1, sizeof(*up));
   if (!up) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }
   VkResult result = radv_bo_create(device, &cmd_buffer->vk.base, allocation_size, 4096,
      device->ws->cs_domain(device->ws),
      RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING |
      RADEON_FLAG_32BIT | RADEON_FLAG_GTT_WC,
      RADV_BO_PRIORITY_UPLOAD_BUFFER, 0, true, &up->upload_bo);
   if (result != VK_SUCCESS) {
      free(up);
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return false;
   }
   up->map = radv_buffer_map(device->ws, up->upload_bo);
   if (!up->map) {
      radv_bo_destroy(device, &cmd_buffer->vk.base, up->upload_bo);
      free(up);
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   up->size = allocation_size;
   up->offset = bytes;
   if (arena) {
      cmd_buffer->bc250_small_arena = up;
      if (device->bc250_env.trace_compile)
         fprintf(stderr, "BC250 arena page: bytes=%u requested=%u\n", allocation_size, bytes);
   }
   radv_cs_add_buffer(device->ws, cmd_buffer->cs->b, up->upload_bo);
   radv_rmv_log_command_buffer_bo_create(device, up->upload_bo, 0, allocation_size, 0);
   list_add(&up->list, &cmd_buffer->upload.list);
   *address = radv_buffer_get_va(up->upload_bo);
   *mapped = up->map;
   radv_bc250_chain_event(device, "SCRATCH_ALLOC",
      "cmd=%p epoch=%llu bo=%p va=%llx bytes=%u lifetime=UNTIL_CMD_RESET_OR_DESTROY",
      (void *)cmd_buffer, (unsigned long long)cmd_buffer->bc250_trace_epoch,
      (void *)up->upload_bo, (unsigned long long)*address, bytes);
   return true;
}

/* Small per-draw records a helper dispatch rewrites in every execution
 * (split indirect outputs, split batch lists). RADV_BC250_SCRATCH_REUSE=1:
 * from the command buffer's upload buffer (same heap and flags as
 * bc250_alloc_scratch's BOs, kept across resets, so no buffer object is
 * created per draw); each allocation is a distinct range that lives until the
 * command buffer is reset, exactly as a dedicated BO. Chain tracing keeps the
 * dedicated BOs (its output sampling looks them up on the upload list). */
static bool
bc250_alloc_records(struct radv_cmd_buffer *cmd_buffer, unsigned bytes, uint64_t *address, void **mapped)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   if (!device->bc250_env.scratch_reuse || radv_bc250_chain_enabled(device))
      return bc250_alloc_scratch(cmd_buffer, bytes, address, mapped);
   unsigned offset;
   if (!radv_cmd_buffer_upload_alloc_aligned(cmd_buffer, bytes, 256, &offset, mapped))
      return false;
   *address = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + offset;
   return true;
}

static void
bc250_draw_task(struct radv_cmd_buffer *cmd_buffer, uint32_t x, uint32_t y, uint32_t z,
                uint64_t indirect_va, uint64_t count_va, uint32_t draw_id, uint64_t *shared_address)
{
   struct radv_graphics_pipeline *pipeline = cmd_buffer->state.graphics_pipeline;
   assert(pipeline && pipeline->bc250_task_pipeline);
   bc250_chain_begin(cmd_buffer, "hybrid_task", indirect_va, count_va, 1, 0);
   /* Native TASK may already have allocated an ACE gang for this command
    * buffer. The private producer still executes on the graphics PM4 stream;
    * gang existence does not change radv_get_pm4_cs for a graphics queue.
    */
   assert(!radv_cmd_buffer_uses_mec(cmd_buffer));
   /* The producer shader unconditionally increments the GDS task-invocation
    * counter, so the queue must always have the shader query region. */
   cmd_buffer->queue_state.gds_needed = true;
   if (!indirect_va && (!x || !y || !z))
      return;

   /* A SIMULTANEOUS_USE command buffer can execute again (e.g. a secondary
    * recorded twice into one primary) while the Mesh waves of its previous
    * execution still read the records and payloads this draw's first
    * producer overwrites: they live at the same addresses in every execution.
    * The chunk loops only drain between their own chunks, so drain before the
    * first producer as well (LoneWolf Tree C fix M1). */
   if (cmd_buffer->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_VS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH;

   uint64_t total = (uint64_t)x * y * z;
   bool chunked = false;
   if (!indirect_va) {
      /* Direct draws may exceed one chunk; the dispatch is split into
       * sequential producer/mesh sequences that share bounded scratch. */
      if (total > BC250_MAX_TASK_TOTAL) {
         vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      if (total > BC250_TASK_CHUNK)
         chunked = true;
   }

   /* Scratch holds one chunk of records and payloads at a time plus the
    * published full-dispatch size. Indirect adds the setup shader's sub-record
    * table (BC250_INDIRECT_CHUNKS entries). Reusing it across chunks and
    * indirect records bounds memory by BC250_TASK_CHUNK rather than by the
    * dispatch total or indirect drawCount. */
   unsigned count = BC250_TASK_CHUNK;
   if (!indirect_va && !chunked)
      count = (unsigned)total;
   unsigned payload_offset = align(count * 12, 256);
   unsigned payload_bytes = count * pipeline->bc250_payload_stride;
   uint64_t ordered_payload = 0;
   if (pipeline->bc250_ordered) {
      /* Drain every prior consumer before reusing this command-buffer arena,
       * including a concurrent reuse of a SIMULTANEOUS_USE command buffer.
       * The arena contains GPU-written records only, never host constants. */
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_VS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH |
                                     RADV_CMD_FLAG_CS_PARTIAL_FLUSH;
      if (cmd_buffer->bc250_ordered_arena_size < payload_bytes) {
         void *unused;
         if (!bc250_alloc_scratch(cmd_buffer, payload_bytes, &cmd_buffer->bc250_ordered_arena, &unused))
            return;
         cmd_buffer->bc250_ordered_arena_size = payload_bytes;
      }
      ordered_payload = cmd_buffer->bc250_ordered_arena;
   }
   unsigned app_offset = align(payload_offset + (pipeline->bc250_ordered ? 0 : payload_bytes), 256);
   /* After the push-constant copy: the full-size slot, and for indirect draws
    * the setup shader's sub-record table after it. */
   unsigned slots_base = app_offset + MAX_PUSH_CONSTANTS_SIZE;
   unsigned bytes = slots_base + (unsigned)sizeof(struct bc250_task_slot) +
      (indirect_va ? BC250_INDIRECT_CHUNKS * 16u : 0);
   uint64_t address = shared_address ? *shared_address : 0;
   void *mapped = NULL;
   if (!address) {
      if (!bc250_alloc_scratch(cmd_buffer, bytes, &address, &mapped))
         return;
      memset(mapped, 0, bytes);
      memcpy((char *)mapped + app_offset, cmd_buffer->push_constants, MAX_PUSH_CONSTANTS_SIZE);
      if (shared_address)
         *shared_address = address;
   } else {
      /* Reuse scratch across records only after all previous consumers finish. */
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_VS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH |
                                      RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                      RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_L2;
   }

   if (cmd_buffer->bc250_trace_chain)
      radv_bc250_chain_event(radv_cmd_buffer_device(cmd_buffer), "SCRATCH_USE",
         "cmd=%p epoch=%llu chain=%llu va=%llx bytes=%u reused=%u draw_id=%u payload_offset=%u slots_offset=%u flush_requested=%llx",
         (void *)cmd_buffer, (unsigned long long)cmd_buffer->bc250_trace_epoch,
         (unsigned long long)cmd_buffer->bc250_trace_chain, (unsigned long long)address,
         bytes, mapped == NULL, draw_id, payload_offset, slots_base, (unsigned long long)cmd_buffer->state.flush_bits);
   struct bc250_constants constants = {
      .xyz = address,
      .payload = pipeline->bc250_ordered ? ordered_payload : address + payload_offset,
      .stride = pipeline->bc250_payload_stride,
      .application_constants = address + app_offset,
      .application_draw_id = draw_id,
      .input = indirect_va,
      .input_count = count_va,
   };

   struct radv_descriptor_state *compute_descriptors =
      radv_get_descriptors_state(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE);
   struct radv_descriptor_state saved_compute_descriptors = *compute_descriptors;
   *compute_descriptors = *radv_get_descriptors_state(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
   compute_descriptors->dirty |= compute_descriptors->valid;
   compute_descriptors->dirty_dynamic = true;
   compute_descriptors->dirty_heaps |= compute_descriptors->valid_heaps;

   if (indirect_va) {
      /* The setup shader validates the record against the advertised limits,
       * publishes the full-dispatch size, and splits it into flat sub-records.
       * The CPU then emits one producer/mesh pair per possible chunk slot;
       * inactive slots are zero records and therefore no-ops. */
      constants.compute_args = address + slots_base + (unsigned)sizeof(struct bc250_task_slot);
      constants.task_slot = address + slots_base;
      radv_meta_begin(cmd_buffer);
      radv_meta_bind_compute_pipeline(cmd_buffer, pipeline->bc250_setup_pipeline);
      radv_meta_push_constants(cmd_buffer, pipeline->bc250_task_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(constants), &constants);
      struct radv_dispatch_info setup_dispatch = {.blocks = {1, 1, 1}};
      bc250_chain_dispatch(cmd_buffer, &setup_dispatch, "indirect_task_setup");
      radv_meta_end(cmd_buffer);
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                      RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_L2;
      const bool skip_suffix = radv_cmd_buffer_device(cmd_buffer)->bc250_env.skip_inactive_chunks;
      for (unsigned k = 0; k < BC250_INDIRECT_CHUNKS; k++) {
         bc250_emit_indirect_chunk(cmd_buffer, pipeline, &constants, address, k, skip_suffix);
         if (vk_command_buffer_has_error(&cmd_buffer->vk))
            return;
      }
      if (skip_suffix) {
         /* Inactive chunks no longer drain the preceding active mesh draw. */
         cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_VS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH;
      }
   } else if (chunked) {
      /* CPU-side chunking: each flat range is dispatched directly and its mesh
       * records are consumed before the next chunk reuses the scratch. */
      assert(mapped);
      struct bc250_task_slot slot = {.full_x = x, .full_y = y, .full_z = z};
      memcpy((char *)mapped + slots_base, &slot, sizeof(slot));
      constants.task_slot = address + slots_base;
      bc250_emit_chunks(cmd_buffer, pipeline, &constants, address, x, y, z, total);
   } else {
      radv_meta_begin(cmd_buffer);
      assert(mapped);
      struct bc250_task_slot slot = {.full_x = x, .full_y = y, .full_z = z};
      memcpy((char *)mapped + slots_base, &slot, sizeof(slot));
      constants.task_slot = address + slots_base;
      radv_meta_bind_compute_pipeline(cmd_buffer, pipeline->bc250_task_pipeline);
      radv_meta_push_constants(cmd_buffer, pipeline->bc250_task_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(constants), &constants);
      struct radv_dispatch_info dispatch = {.blocks = {x, y, z}};
      bc250_chain_dispatch(cmd_buffer, &dispatch, "task_producer");
      radv_meta_end(cmd_buffer);

      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                      RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_L2;
      if (pipeline->bc250_ordered) {
         cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_WB_L2;
         radv_emit_cache_flush(cmd_buffer);
         ac_emit_cp_pfp_sync_me(cmd_buffer->cs->b, false);
      }
      uint8_t saved[sizeof(constants)];
      memcpy(saved, cmd_buffer->push_constants, sizeof(saved));
      memcpy(cmd_buffer->push_constants, &constants, sizeof(constants));
      cmd_buffer->push_constant_stages |= pipeline->active_stages;
      bc250_chain_consume(cmd_buffer, address, 0, constants.payload, count, 12, 0);
      VkDrawIndirect2InfoKHR draw = {
         .sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR,
         .addressRange = {.address = address, .size = (uint64_t)count * 12, .stride = 12},
         .drawCount = count,
      };
      cmd_buffer->bc250_inside_mesh_draw = true;
      radv_CmdDrawMeshTasksIndirect2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &draw);
      cmd_buffer->bc250_inside_mesh_draw = false;
      memcpy(cmd_buffer->push_constants, saved, sizeof(saved));
      cmd_buffer->push_constant_stages |= pipeline->active_stages;
   }

   *compute_descriptors = saved_compute_descriptors;
   compute_descriptors->dirty |= compute_descriptors->valid;
   compute_descriptors->dirty_dynamic = true;
   compute_descriptors->dirty_heaps |= compute_descriptors->valid_heaps;

   cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                   RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_L2;
}

void
radv_bc250_draw_task(struct radv_cmd_buffer *cmd_buffer, uint32_t x, uint32_t y, uint32_t z)
{
   bc250_draw_task(cmd_buffer, x, y, z, 0, 0, 0, NULL);
}

void
radv_bc250_draw_task_indirect(struct radv_cmd_buffer *cmd_buffer, uint64_t address,
                             uint32_t draw_count, uint32_t stride, uint64_t count_address)
{
   /* Mesh indirect records are bounded by maxDrawIndirectCount (4096),
    * independently of the EXT_multi_draw maxMultiDrawCount property. */
   if (draw_count > 4096) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint64_t scratch_address = 0;
   for (uint32_t i = 0; i < draw_count; i++) {
      bc250_draw_task(cmd_buffer, 0, 0, 0, address + (uint64_t)i * stride, count_address, i, &scratch_address);
      if (vk_command_buffer_has_error(&cmd_buffer->vk))
         return;
   }
}

/* RADV_BC250_SPLIT_BATCH_PREP (opt-in, runtime only, no compiler-key bit).
 *
 * The per-draw path below runs the split-argument setup shader once per split
 * indirect draw: meta begin/end around a small dispatch, then CS_PARTIAL_FLUSH
 * + INV_VCACHE + INV_SCACHE (+ INV_L2) before the draw. vkd3d-proton records
 * D3D12 compute on the graphics queue, so every CS_PARTIAL_FLUSH also drains
 * the application's compute work in flight.
 *
 * With the switch on, the first split indirect draw of a render pass instance
 * (of a primary command buffer, outside conditional rendering) opens a batch:
 * ONE indirect setup dispatch whose group count and work list live in a
 * driver scratch allocation that the CPU keeps appending to while later split
 * indirect draws of the same instance are recorded, followed by ONE
 * CS_PARTIAL_FLUSH + INV_VCACHE + INV_SCACHE (+ INV_L2 unless
 * RADV_BC250_PERF_NO_SPLIT_L2_INV=1). Every draw keeps the per-draw shader's
 * work split: draw d owns ceil(records/64) consecutive workgroups ("chunks")
 * of the batched dispatch, found through a chunk -> entry table, and each
 * workgroup rewrites records chunk*64 + lane exactly as workgroup "chunk" of
 * the per-draw dispatch would, into that draw's own output allocation (same
 * allocator, same size, same record layout). Each draw's
 * DISPATCH_MESH_INDIRECT_MULTI reads its own output with the unchanged push
 * constants and draw IDs.
 *
 * Why the setup may run early: inside one render pass instance an application
 * cannot make a GPU write to an indirect or count buffer visible to a later
 * DRAW_INDIRECT read. Transfer and dispatch commands are not allowed there,
 * and barriers inside a render pass instance are restricted to
 * framebuffer-space stages (dynamic rendering) or to subpass self-dependencies
 * whose source stages must not be logically later than their destination
 * stages (DRAW_INDIRECT is the first stage); vkCmdWaitEvents inside a render
 * pass must not wait on HOST. So the records every split draw of the instance
 * reads are final before its first split draw.
 *
 * The batch is closed (the next split draw opens a new one) at every
 * vkCmdBeginRendering/vkCmdEndRendering (so subpasses of a VkRenderPass, which
 * the common runtime records as separate rendering begin/end pairs, and each
 * suspended/resumed part are separate batches), at every barrier and event
 * wait, conditional rendering begin/end, vkCmdExecuteCommands, and command
 * buffer begin/reset. Secondary command buffers, draws under conditional
 * rendering and BC250_CHAIN_TRACE keep the per-draw path. A draw that does not
 * fit the open batch (BC250_SPLIT_BATCH_ENTRIES entries or
 * BC250_SPLIT_BATCH_CHUNKS chunks) opens a new batch at its own position.
 *
 * The list is CPU-written at record time into memory that stays mapped and
 * allocated until the command buffer is reset (bc250_alloc_scratch never
 * moves an allocation), so every entry, the chunk table and the final group
 * count are in memory before the command buffer can be submitted, and the
 * queue preamble invalidates the GPU caches at every submission. The
 * dispatch is emitted at the first draw's position, i.e. before all draws
 * that read its output. */
#define BC250_SPLIT_BATCH_ENTRIES 128u
#define BC250_SPLIT_BATCH_CHUNKS 1024u /* >= 64 = the chunks of one 4096-record draw */

struct bc250_split_batch_entry {
   uint64_t input, output, count;
   uint32_t records, stride, pieces, first_chunk, pad[2];
};
static_assert(sizeof(struct bc250_split_batch_entry) == 48, "entry layout used by the setup shader");
/* List layout: {group count x, 1, 1, 0}, entries, then one uint32 entry index per chunk. */
#define BC250_SPLIT_BATCH_HEADER_BYTES 16u
#define BC250_SPLIT_BATCH_TABLE_OFFSET \
   (BC250_SPLIT_BATCH_HEADER_BYTES + BC250_SPLIT_BATCH_ENTRIES * sizeof(struct bc250_split_batch_entry))
#define BC250_SPLIT_BATCH_BYTES (BC250_SPLIT_BATCH_TABLE_OFFSET + BC250_SPLIT_BATCH_CHUNKS * 4u)

void
radv_bc250_split_batch_close(struct radv_cmd_buffer *cmd_buffer)
{
   cmd_buffer->bc250_split_batch = NULL;
   cmd_buffer->bc250_split_batch_va = 0;
   cmd_buffer->bc250_split_batch_used = 0;
}

static bool
bc250_split_batch_trace(const struct radv_cmd_buffer *cmd_buffer)
{
   return radv_cmd_buffer_device(cmd_buffer)->bc250_env.split_batch_trace;
}

/* The batched form of the setup shader in radv_bc250_prepare_direct_split.
 * Workgroup w is chunk w - first_chunk of entry table[w]; from there on the
 * code is the per-draw shader's with that entry's arguments in place of the
 * push constants and the chunk in place of the workgroup ID. */
static VkResult
bc250_split_batch_pipeline(struct radv_device *device, VkPipeline *pipeline, VkPipelineLayout *layout)
{
   const enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_BC250_SPLIT_BATCH;
   VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 8};
   VkResult result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &range,
                                                 &key, sizeof(key), layout);
   if (result != VK_SUCCESS)
      return result;
   *pipeline = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (*pipeline != VK_NULL_HANDLE)
      return VK_SUCCESS;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &device->compiler_info.nir_options[MESA_SHADER_COMPUTE], "bc250_split_batch");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *list = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0), .range = 8);
   nir_def *w = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *e = nir_load_global(&b, 1, 32,
      nir_iadd(&b, nir_iadd_imm(&b, list, BC250_SPLIT_BATCH_TABLE_OFFSET), nir_u2u64(&b, nir_imul_imm(&b, w, 4))),
      .align_mul = 4);
   nir_def *entry = nir_iadd(&b, nir_iadd_imm(&b, list, BC250_SPLIT_BATCH_HEADER_BYTES),
                             nir_u2u64(&b, nir_imul_imm(&b, e, sizeof(struct bc250_split_batch_entry))));
   nir_def *w0 = nir_load_global(&b, 4, 32, entry, .align_mul = 16);
   nir_def *w1 = nir_load_global(&b, 4, 32, nir_iadd_imm(&b, entry, 16), .align_mul = 16);
   nir_def *w2 = nir_load_global(&b, 4, 32, nir_iadd_imm(&b, entry, 32), .align_mul = 16);
   nir_def *input = nir_pack_64_2x32_split(&b, nir_channel(&b, w0, 0), nir_channel(&b, w0, 1));
   nir_def *output = nir_pack_64_2x32_split(&b, nir_channel(&b, w0, 2), nir_channel(&b, w0, 3));
   nir_def *count_addr = nir_pack_64_2x32_split(&b, nir_channel(&b, w1, 0), nir_channel(&b, w1, 1));
   nir_def *records = nir_channel(&b, w1, 2);
   nir_def *stride = nir_channel(&b, w1, 3);
   nir_def *factor = nir_channel(&b, w2, 0);
   nir_def *chunk = nir_isub(&b, w, nir_channel(&b, w2, 1));
   nir_def *i = nir_iadd(&b, nir_imul_imm(&b, chunk, 64), nir_load_local_invocation_index(&b));
   nir_def *zero = nir_imm_ivec3(&b, 0, 0, 0);
   nir_push_if(&b, nir_ult(&b, i, records));
   nir_push_if(&b, nir_ine_imm(&b, count_addr, 0));
   nir_def *count = nir_load_global(&b, 1, 32, count_addr, .align_mul = 4);
   nir_pop_if(&b, NULL);
   count = nir_if_phi(&b, count, records);
   nir_push_if(&b, nir_ult(&b, i, count));
   nir_def *addr = nir_iadd(&b, input, nir_imul(&b, nir_u2u64(&b, i), nir_u2u64(&b, stride)));
   nir_def *xyz = nir_load_global(&b, 3, 32, addr, .align_mul = 4);
   nir_def *dx = nir_channel(&b, xyz, 0), *dy = nir_channel(&b, xyz, 1), *dz = nir_channel(&b, xyz, 2);
   nir_def *px = nir_iand(&b, nir_uge(&b, dy, dx), nir_uge(&b, dz, dx));
   nir_def *py = nir_iand(&b, nir_inot(&b, px), nir_uge(&b, dz, dy));
   nir_def *pz = nir_inot(&b, nir_ior(&b, px, py));
   nir_def *expanded = nir_vec3(&b,
      nir_imul(&b, dx, nir_bcsel(&b, px, factor, nir_imm_int(&b, 1))),
      nir_imul(&b, dy, nir_bcsel(&b, py, factor, nir_imm_int(&b, 1))),
      nir_imul(&b, dz, nir_bcsel(&b, pz, factor, nir_imm_int(&b, 1))));
   nir_pop_if(&b, NULL);
   expanded = nir_if_phi(&b, expanded, zero);
   nir_store_global(&b, expanded, nir_iadd(&b, output, nir_u2u64(&b, nir_imul_imm(&b, i, 16))), .align_mul = 16);
   nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_validate_shader(b.shader, "BC250 batched split argument setup");
   /* Test aid (tests/bc250-mesh/split-batch): the builder NIR of both setup shaders. */
   if (debug_get_bool_option("RADV_BC250_SPLIT_BATCH_PRINT_NIR", false))
      nir_print_shader(b.shader, stderr);
   VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = vk_shader_module_handle_from_nir(b.shader),
         .pName = "main"}, .layout = *layout};
   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &ci, &key, sizeof(key),
                                            pipeline);
   ralloc_free(b.shader);
   return result;
}

/* True when this split indirect draw may join (or open) a batch. */
static bool
bc250_split_batch_eligible(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   return device->bc250_split_batch_prep && cmd_buffer->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
          cmd_buffer->state.render.active && !cmd_buffer->state.cond_render.enabled &&
          !cmd_buffer->bc250_inside_mesh_draw && !radv_bc250_chain_enabled(device);
}

/* Appends one split indirect draw to the open batch, opening one (and
 * emitting its setup dispatch at the current position) when none is open or
 * the draw does not fit the open one. */
static bool
bc250_split_batch_append(struct radv_cmd_buffer *cmd_buffer, const struct bc250_split_args *args)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const uint32_t chunks = DIV_ROUND_UP(args->records, 64); /* the per-draw dispatch's workgroups */
   uint32_t *list = cmd_buffer->bc250_split_batch;
   if (!list || cmd_buffer->bc250_split_batch_used == BC250_SPLIT_BATCH_ENTRIES ||
       list[0] + chunks > BC250_SPLIT_BATCH_CHUNKS) {
      VkPipeline pipeline;
      VkPipelineLayout layout;
      VkResult result = bc250_split_batch_pipeline(device, &pipeline, &layout);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd_buffer->vk, result);
         return false;
      }
      uint64_t va;
      void *mapped;
      if (!bc250_alloc_records(cmd_buffer, BC250_SPLIT_BATCH_BYTES, &va, &mapped))
         return false;
      /* Group count {chunks, 1, 1}: completed by the appends below. */
      list = mapped;
      list[0] = 0;
      list[1] = 1;
      list[2] = 1;
      list[3] = 0;
      radv_meta_begin(cmd_buffer);
      radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);
      radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(va), &va);
      struct radv_dispatch_info dispatch = {.indirect_va = va};
      radv_compute_dispatch(cmd_buffer, &dispatch);
      radv_meta_end(cmd_buffer);
      /* The flush the per-draw path emits after each setup dispatch, once. */
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                      RADV_CMD_FLAG_INV_SCACHE;
      if (!device->bc250_env.no_split_l2_inv)
         cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_INV_L2;
      cmd_buffer->bc250_split_batch = list;
      cmd_buffer->bc250_split_batch_va = va;
      cmd_buffer->bc250_split_batch_used = 0;
      if (bc250_split_batch_trace(cmd_buffer))
         fprintf(stderr, "BC250_SPLIT_BATCH open list=0x%llx entries=%u chunks=%u\n", (unsigned long long)va,
                 BC250_SPLIT_BATCH_ENTRIES, BC250_SPLIT_BATCH_CHUNKS);
   }
   const uint32_t slot = cmd_buffer->bc250_split_batch_used;
   const uint32_t first_chunk = list[0];
   struct bc250_split_batch_entry entry = {.input = args->input, .output = args->output, .count = args->count,
      .records = args->records, .stride = args->stride, .pieces = args->pieces, .first_chunk = first_chunk};
   memcpy((char *)list + BC250_SPLIT_BATCH_HEADER_BYTES + slot * sizeof(entry), &entry, sizeof(entry));
   uint32_t *table = (uint32_t *)((char *)list + BC250_SPLIT_BATCH_TABLE_OFFSET);
   for (uint32_t c = 0; c < chunks; c++)
      table[first_chunk + c] = slot;
   cmd_buffer->bc250_split_batch_used = slot + 1;
   list[0] = first_chunk + chunks;
   if (bc250_split_batch_trace(cmd_buffer))
      fprintf(stderr, "BC250_SPLIT_BATCH entry list=0x%llx slot=%u first_chunk=%u chunks=%u input=0x%llx "
              "output=0x%llx count=0x%llx records=%u stride=%u pieces=%u\n",
              (unsigned long long)cmd_buffer->bc250_split_batch_va, slot, first_chunk, chunks,
              (unsigned long long)args->input, (unsigned long long)args->output, (unsigned long long)args->count,
              args->records, args->stride, args->pieces);
   return true;
}

static bool
bc250_queries_active(struct radv_cmd_buffer *cmd_buffer)
{
   return radv_get_num_pipeline_stat_queries(cmd_buffer) || cmd_buffer->state.active_pipeline_queries ||
          cmd_buffer->state.active_occlusion_queries || cmd_buffer->state.active_prims_gen_queries ||
          cmd_buffer->state.active_emulated_prims_gen_queries || cmd_buffer->state.active_emulated_prims_xfb_queries;
}

/* The per-draw split argument setup dispatch and its flush.
 * RADV_BC250_SPLIT_LEAN_SETUP=1: without an active query or bound shader
 * object, radv_meta_begin/radv_meta_end reduce to saving and restoring the
 * compute pipeline and the push constants (queries are only suspended when
 * active; shader objects only rebound when bound). The dispatch touches only
 * the compute pipeline and the first sizeof(args) push constant bytes, so the
 * lean path restores exactly those, with the same dirty state, and records the
 * same packets; otherwise the meta path runs. */
static void
bc250_split_setup_dispatch(struct radv_cmd_buffer *cmd_buffer, const struct radv_graphics_pipeline *pipeline,
                           const struct bc250_split_args *args)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_dispatch_info dispatch = {.blocks = {DIV_ROUND_UP(args->records, 64), 1, 1}};
   bool lean = device->bc250_env.split_lean_setup && !bc250_queries_active(cmd_buffer);
   for (unsigned i = 0; lean && i <= MESA_SHADER_MESH; i++)
      lean = !cmd_buffer->state.shader_objs[i];
   if (lean) {
      VkCommandBuffer handle = radv_cmd_buffer_to_handle(cmd_buffer);
      struct radv_compute_pipeline *old = cmd_buffer->state.compute_pipeline;
      uint8_t saved[sizeof(*args)];
      memcpy(saved, cmd_buffer->push_constants, sizeof(saved));
      radv_CmdBindPipeline(handle, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->bc250_setup_pipeline);
      memcpy(cmd_buffer->push_constants, args, sizeof(*args));
      cmd_buffer->push_constant_stages |= VK_SHADER_STAGE_COMPUTE_BIT;
      bc250_chain_dispatch(cmd_buffer, &dispatch, "split_arguments");
      if (old)
         radv_CmdBindPipeline(handle, VK_PIPELINE_BIND_POINT_COMPUTE, radv_pipeline_to_handle(&old->base));
      memcpy(cmd_buffer->push_constants, saved, sizeof(saved));
      cmd_buffer->push_constant_stages |= VK_SHADER_STAGE_COMPUTE_BIT;
   } else {
      radv_meta_begin(cmd_buffer);
      radv_meta_bind_compute_pipeline(cmd_buffer, pipeline->bc250_setup_pipeline);
      radv_meta_push_constants(cmd_buffer, pipeline->bc250_task_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(*args), args);
      bc250_chain_dispatch(cmd_buffer, &dispatch, "split_arguments");
      radv_meta_end(cmd_buffer);
   }
   cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                   RADV_CMD_FLAG_INV_SCACHE;
   /* Performance A/B (default unchanged): on GFX10 the CP reads the setup
    * shader's records through L2, which the compute writes already reached, so
    * RADV_BC250_PERF_NO_SPLIT_L2_INV=1 skips the per-draw L2 invalidation. */
   if (!device->bc250_env.no_split_l2_inv)
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_INV_L2;
}

/* RADV_BC250_SPLIT_PREP_FREE admission, draw-time part: no depth writes and
 * no stencil test on a present attachment, so the set of fragments that pass
 * the depth/stencil tests does not depend on primitive order (static part:
 * bc250_fs_order_independent). */
static bool
bc250_split_order_free_now(const struct radv_cmd_buffer *cmd_buffer)
{
   const struct vk_depth_stencil_state *ds = &cmd_buffer->state.dynamic.vk.ds;
   const VkImageAspectFlags aspects = cmd_buffer->state.render.ds_att_aspects;
   if ((aspects & VK_IMAGE_ASPECT_DEPTH_BIT) && ds->depth.test_enable && ds->depth.write_enable)
      return false;
   if ((aspects & VK_IMAGE_ASPECT_STENCIL_BIT) && ds->stencil.test_enable)
      return false;
   return true;
}

/* RADV_BC250_SPLIT_PREP_FREE=1: a split indirect draw without the argument
 * setup. Instead of one setup dispatch (plus its CS_PARTIAL_FLUSH and cache
 * invalidations) that rewrites every record (x,y,z) into a grid with pieces
 * times more workgroups, the application's records are drawn once per piece,
 * with the same indirect (count) packet, stride, draw count and count buffer,
 * and bc250_constants.split_piece = piece + 1: the shader (compiled with the
 * piece select, bc250_split_flat_index) then sees flat index
 * WorkgroupIndex * pieces + piece, i.e. the same API workgroup ID, piece,
 * DrawID and NumWorkGroups (read from the record) as under the setup. Same
 * pipeline and registers, fast launch unchanged; per piece the launch is the
 * application's grid instead of the multiplied one. Only the primitive order
 * changes, hence the admission (bc250_fs_order_independent,
 * bc250_split_order_free_now). */
static void
bc250_draw_split_prep_free(struct radv_cmd_buffer *cmd_buffer, struct radv_graphics_pipeline *pipeline,
                           uint64_t input, uint32_t records, uint32_t stride, uint64_t count)
{
   unsigned offset;
   void *app_map;
   if (!radv_cmd_buffer_upload_alloc(cmd_buffer, MAX_PUSH_CONSTANTS_SIZE, &offset, &app_map))
      return;
   memcpy(app_map, cmd_buffer->push_constants, MAX_PUSH_CONSTANTS_SIZE);
   struct bc250_constants cc = {.input = input, .stride = stride,
      .application_constants = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + offset};
   uint8_t saved[sizeof(cc)];
   memcpy(saved, cmd_buffer->push_constants, sizeof(saved));
   cmd_buffer->bc250_inside_mesh_draw = true;
   for (uint32_t piece = 0; piece < pipeline->bc250_direct_split_pieces; piece++) {
      cc.split_piece = piece + 1;
      memcpy(cmd_buffer->push_constants, &cc, sizeof(cc));
      cmd_buffer->push_constant_stages |= pipeline->active_stages;
      if (count) {
         VkDrawIndirectCount2InfoKHR draw = {.sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_COUNT_2_INFO_KHR,
            .addressRange = {.address = input, .size = (uint64_t)records * stride, .stride = stride},
            .countAddressRange = {.address = count, .size = 4}, .maxDrawCount = records};
         radv_CmdDrawMeshTasksIndirectCount2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &draw);
      } else {
         VkDrawIndirect2InfoKHR draw = {.sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR,
            .addressRange = {.address = input, .size = (uint64_t)records * stride, .stride = stride},
            .drawCount = records};
         radv_CmdDrawMeshTasksIndirect2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &draw);
      }
   }
   cmd_buffer->bc250_inside_mesh_draw = false;
   memcpy(cmd_buffer->push_constants, saved, sizeof(saved));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
}

void
radv_bc250_draw_split_indirect(struct radv_cmd_buffer *cmd_buffer, uint64_t input,
                               uint32_t records, uint32_t stride, uint64_t count)
{
   if (!records)
      return;
   if (records > 4096) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_graphics_pipeline *pipeline = cmd_buffer->state.graphics_pipeline;
   if (device->bc250_env.split_prep_free && pipeline->bc250_split_order_free &&
       bc250_split_order_free_now(cmd_buffer) && !radv_bc250_chain_enabled(device)) {
      bc250_draw_split_prep_free(cmd_buffer, pipeline, input, records, stride, count);
      return;
   }
   uint64_t output;
   void *mapped;
   if (!bc250_alloc_records(cmd_buffer, records * 16u, &output, &mapped))
      return;
   bc250_chain_begin(cmd_buffer, "split_indirect", input, count, records, stride);
   /* Diagnostic only: expose this already CPU-mapped helper allocation. No
    * additional GPU command, shader store, wait, or readback is introduced.
    * A sampler may race writes or release; it must not infer completion. */
   if (cmd_buffer->bc250_trace_chain && device->bc250_env.chain_sample_output) {
      memset(mapped, 0xa5, records * 16u);
      list_for_each_entry(struct radv_cmd_buffer_upload, sample_up, &cmd_buffer->upload.list, list) {
         if (sample_up->map == mapped) {
            sample_up->bc250_sample_chain = cmd_buffer->bc250_trace_chain;
            break;
         }
      }
      radv_bc250_chain_event(device, "OUTPUT_CPU_MAPPING",
         "cmd=%p epoch=%llu chain=%llu output=%llx cpu=%llx bytes=%u records=%u sentinel=a5a5a5a5 observation=RACY_NOT_COMPLETION",
         (void *)cmd_buffer, (unsigned long long)cmd_buffer->bc250_trace_epoch,
         (unsigned long long)cmd_buffer->bc250_trace_chain, (unsigned long long)output,
         (unsigned long long)(uintptr_t)mapped, records * 16u, records);
   }

   struct bc250_split_args args = {.input = input, .output = output, .count = count,
      .records = records, .stride = stride, .pieces = pipeline->bc250_direct_split_pieces};
   if (bc250_split_batch_eligible(cmd_buffer)) {
      /* RADV_BC250_SPLIT_BATCH_PREP: the batch's setup dispatch (emitted at
       * the first split draw of the render pass instance) writes this draw's
       * records; see bc250_split_batch_append. */
      if (!bc250_split_batch_append(cmd_buffer, &args))
         return;
   } else {
      bc250_split_setup_dispatch(cmd_buffer, pipeline, &args);
   }
   unsigned offset;
   void *app_map;
   if (!radv_cmd_buffer_upload_alloc(cmd_buffer, MAX_PUSH_CONSTANTS_SIZE, &offset, &app_map))
      return;
   memcpy(app_map, cmd_buffer->push_constants, MAX_PUSH_CONSTANTS_SIZE);
   struct bc250_constants cc = {.input = input, .stride = stride,
      .application_constants = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + offset};
   uint8_t saved[sizeof(cc)];
   memcpy(saved, cmd_buffer->push_constants, sizeof(saved));
   memcpy(cmd_buffer->push_constants, &cc, sizeof(cc));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
   cmd_buffer->bc250_inside_mesh_draw = true;
   bc250_chain_consume(cmd_buffer, output, count, 0, records, 16, 0);
   if (count) {
      VkDrawIndirectCount2InfoKHR draw = {.sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_COUNT_2_INFO_KHR,
         .addressRange = {.address = output, .size = records * 16u, .stride = 16},
         .countAddressRange = {.address = count, .size = 4}, .maxDrawCount = records};
      radv_CmdDrawMeshTasksIndirectCount2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &draw);
   } else {
      VkDrawIndirect2InfoKHR draw = {.sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR,
         .addressRange = {.address = output, .size = records * 16u, .stride = 16}, .drawCount = records};
      radv_CmdDrawMeshTasksIndirect2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &draw);
   }
   cmd_buffer->bc250_inside_mesh_draw = false;
   memcpy(cmd_buffer->push_constants, saved, sizeof(saved));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
}

void
radv_bc250_draw_split(struct radv_cmd_buffer *cmd_buffer, uint32_t x, uint32_t y, uint32_t z)
{
   bc250_chain_begin(cmd_buffer, "direct_split_cpu_dimensions", 0, 0, 1, 16);
   if (!x || !y || !z)
      return;
   struct radv_graphics_pipeline *pipeline = cmd_buffer->state.graphics_pipeline;
   unsigned offset;
   void *mapped;
   if (!radv_cmd_buffer_upload_alloc(cmd_buffer, MAX_PUSH_CONSTANTS_SIZE + 16, &offset, &mapped))
      return;
   memcpy(mapped, cmd_buffer->push_constants, MAX_PUSH_CONSTANTS_SIZE);
   uint32_t dims[4] = {x, y, z, 0};
   memcpy((char *)mapped + MAX_PUSH_CONSTANTS_SIZE, dims, sizeof(dims));
   uint64_t address = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + offset;
   if (cmd_buffer->bc250_trace_chain)
      radv_bc250_chain_event(radv_cmd_buffer_device(cmd_buffer), "CPU_DIMENSIONS_WRITE",
         "cmd=%p epoch=%llu chain=%llu va=%llx x=%u y=%u z=%u bytes=16 private_compute_producer=NONE external_resource_generation=UNKNOWN",
         (void *)cmd_buffer, (unsigned long long)cmd_buffer->bc250_trace_epoch,
         (unsigned long long)cmd_buffer->bc250_trace_chain,
         (unsigned long long)(address + MAX_PUSH_CONSTANTS_SIZE), x, y, z);
   struct bc250_constants cc = {.input = address + MAX_PUSH_CONSTANTS_SIZE, .stride = 16,
      .application_constants = address};
   uint8_t saved[sizeof(cc)];
   memcpy(saved, cmd_buffer->push_constants, sizeof(saved));
   memcpy(cmd_buffer->push_constants, &cc, sizeof(cc));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
   if (x <= y && x <= z) x *= pipeline->bc250_direct_split_pieces;
   else if (y <= z) y *= pipeline->bc250_direct_split_pieces;
   else z *= pipeline->bc250_direct_split_pieces;
   cmd_buffer->bc250_inside_mesh_draw = true;
   radv_CmdDrawMeshTasksEXT(radv_cmd_buffer_to_handle(cmd_buffer), x, y, z);
   cmd_buffer->bc250_inside_mesh_draw = false;
   memcpy(cmd_buffer->push_constants, saved, sizeof(saved));
   cmd_buffer->push_constant_stages |= pipeline->active_stages;
}

/* RADV_BC250_MESH_MERGE option A (MESH_PERF/merge/DESIGN.md "Option A
 * indirect"). A merged shader runs K API workgroups per hardware subgroup.
 * For an indirect draw, one prep dispatch per API call rewrites every active
 * application record i (x,y,z) into a driver record at output + 32*i:
 *    {ceil(x*y*z/K), 1, 1, 0,  x, y, z, 0}
 * DISPATCH_MESH_INDIRECT_MULTI reads the first three dwords (stride 32), so it
 * launches ceil(N/K) groups instead of N and writes (ceil(N/K),1,1) into the
 * grid SGPRs, plus DrawID = i. The shader reads the application's grid from
 * the second half of record DrawID, addressed by the dims user SGPR
 * (AC_UD_MS_BC250_DIMS_VA = low 32 bits of output, never 0). Records at or
 * beyond the count buffer's value are not written: the CP does not read them. */
#define BC250_MERGE_PREP_MAX_RECORDS 4096u
#define BC250_MERGE_RECORD_BYTES 32u

struct bc250_merge_prep_args {
   uint64_t input, output, count;
   uint32_t records, stride, k, pad;
};

static VkResult
bc250_merge_prep_pipeline(struct radv_device *device, VkPipeline *pipeline, VkPipelineLayout *layout)
{
   const enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_BC250_MERGE_PREP;
   VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct bc250_merge_prep_args)};
   VkResult result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &range,
                                                 &key, sizeof(key), layout);
   if (result != VK_SUCCESS)
      return result;
   *pipeline = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (*pipeline != VK_NULL_HANDLE)
      return VK_SUCCESS;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &device->compiler_info.nir_options[MESA_SHADER_COMPUTE], "bc250_merge_prep");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *args = nir_load_push_constant(&b, 3, 64, nir_imm_int(&b, 0), .range = 24);
   nir_def *sizes = nir_load_push_constant(&b, 3, 32, nir_imm_int(&b, 24), .range = 12);
   nir_def *records = nir_channel(&b, sizes, 0);
   nir_def *i = nir_iadd(&b, nir_imul_imm(&b, nir_channel(&b, nir_load_workgroup_id(&b), 0), 64),
                         nir_load_local_invocation_index(&b));
   nir_push_if(&b, nir_ult(&b, i, records));
   {
      nir_def *count_addr = nir_channel(&b, args, 2);
      nir_push_if(&b, nir_ine_imm(&b, count_addr, 0));
      nir_def *count = nir_load_global(&b, 1, 32, count_addr, .align_mul = 4);
      nir_pop_if(&b, NULL);
      count = nir_if_phi(&b, count, records);
      nir_push_if(&b, nir_ult(&b, i, count));
      {
         nir_def *in = nir_iadd(&b, nir_channel(&b, args, 0),
                                nir_imul(&b, nir_u2u64(&b, i), nir_u2u64(&b, nir_channel(&b, sizes, 1))));
         nir_def *xyz = nir_load_global(&b, 3, 32, in, .align_mul = 4);
         nir_def *x = nir_channel(&b, xyz, 0), *y = nir_channel(&b, xyz, 1), *z = nir_channel(&b, xyz, 2);
         /* The same u32 product as the shader's N = x*y*z. */
         nir_def *n = nir_imul(&b, nir_imul(&b, x, y), z);
         nir_def *k = nir_channel(&b, sizes, 2);
         nir_def *groups = nir_iadd(&b, nir_udiv(&b, n, k), nir_b2i32(&b, nir_ine_imm(&b, nir_umod(&b, n, k), 0)));
         nir_def *out = nir_iadd(&b, nir_channel(&b, args, 1), nir_u2u64(&b, nir_imul_imm(&b, i, BC250_MERGE_RECORD_BYTES)));
         nir_def *zero = nir_imm_int(&b, 0);
         nir_store_global(&b, nir_vec4(&b, groups, nir_imm_int(&b, 1), nir_imm_int(&b, 1), zero), out,
                          .align_mul = 16);
         nir_store_global(&b, nir_vec4(&b, x, y, z, zero), nir_iadd_imm(&b, out, 16), .align_mul = 16);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_validate_shader(b.shader, "BC250 merge prep");
   VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = vk_shader_module_handle_from_nir(b.shader),
         .pName = "main"}, .layout = *layout};
   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &ci, &key, sizeof(key),
                                            pipeline);
   ralloc_free(b.shader);
   return result;
}

void
radv_bc250_draw_merge_indirect(struct radv_cmd_buffer *cmd_buffer, unsigned merge_k, uint64_t input,
                               uint32_t records, uint32_t stride, uint64_t count)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   /* Very large maxDrawCount values keep option B (N groups, the surplus
    * exits empty) instead of reserving 32 bytes per possible record. */
   const bool prep = records && records <= BC250_MERGE_PREP_MAX_RECORDS;
   uint64_t output = 0;
   if (prep) {
      VkPipeline pipeline;
      VkPipelineLayout layout;
      VkResult result = bc250_merge_prep_pipeline(device, &pipeline, &layout);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd_buffer->vk, result);
         return;
      }
      unsigned offset;
      void *mapped;
      /* The upload BO is 32-bit addressable. One spare 64-byte line keeps
       * the low 32 bits nonzero (0 selects the grid SGPRs in the shader). */
      if (!radv_cmd_buffer_upload_alloc_aligned(cmd_buffer, records * BC250_MERGE_RECORD_BYTES + 64, 64, &offset,
                                                &mapped))
         return;
      output = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + offset;
      if ((uint32_t)output == 0)
         output += 64;
      bc250_chain_begin(cmd_buffer, "merge_indirect", input, count, records, stride);
      /* A SIMULTANEOUS_USE command buffer can run again while the previous
       * execution's draw still reads these records. */
      if (cmd_buffer->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)
         cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_VS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH;
      struct bc250_merge_prep_args args = {.input = input, .output = output, .count = count,
         .records = records, .stride = stride, .k = merge_k};
      radv_meta_begin(cmd_buffer);
      radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);
      radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args), &args);
      struct radv_dispatch_info dispatch = {.blocks = {DIV_ROUND_UP(records, 64), 1, 1}};
      bc250_chain_dispatch(cmd_buffer, &dispatch, "merge_prep");
      radv_meta_end(cmd_buffer);
      /* The CP must not read the records before the prep finished, and the
       * shader's scalar load must not hit a stale K$ line (the upload BO also
       * holds data that shaders read through SMEM). No L2 invalidation: on
       * GFX10 the compute stores land in L2 and the CP's indirect fetch and
       * the shader's scalar loads both read through L2, so the records are
       * visible after CS_PARTIAL_FLUSH (upstream's own compute-write ->
       * INDIRECT_COMMAND_READ barrier is CS_PARTIAL_FLUSH + INV_SCACHE on
       * GFX9+; the base driver's RADV_BC250_PERF_NO_SPLIT_L2_INV is the same argument).
       * RADV_BC250_MESH_MERGE_PREP_L2_INV=1 adds INV_VCACHE | INV_L2 for an A/B. */
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_SCACHE;
      if (radv_cmd_buffer_device(cmd_buffer)->bc250_env.merge_prep_l2_inv)
         cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_INV_VCACHE | RADV_CMD_FLAG_INV_L2;
   }
   cmd_buffer->bc250_merge_dims_va = (uint32_t)output;
   cmd_buffer->bc250_inside_mesh_draw = true;
   const uint64_t address = prep ? output : input;
   const uint32_t record_stride = prep ? BC250_MERGE_RECORD_BYTES : stride;
   if (count) {
      VkDrawIndirectCount2InfoKHR draw = {.sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_COUNT_2_INFO_KHR,
         .addressRange = {.address = address, .size = (uint64_t)records * record_stride, .stride = record_stride},
         .countAddressRange = {.address = count, .size = 4}, .maxDrawCount = records};
      radv_CmdDrawMeshTasksIndirectCount2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &draw);
   } else {
      VkDrawIndirect2InfoKHR draw = {.sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR,
         .addressRange = {.address = address, .size = (uint64_t)records * record_stride, .stride = record_stride},
         .drawCount = records};
      radv_CmdDrawMeshTasksIndirect2EXT(radv_cmd_buffer_to_handle(cmd_buffer), &draw);
   }
   cmd_buffer->bc250_inside_mesh_draw = false;
   cmd_buffer->bc250_merge_dims_va = 0;
}

/* RADV_DIRECTMESH=1: one switch for the BC-250 direct Mesh path. Before the instance reads any option,
 * fill in the BC-250 base settings and every validated direct-path switch that is not already set
 * (explicitly set variables always win). The values are the validated configuration (Mesh CTS
 * 3,558/3,558 on hardware, every pipeline direct). The switches themselves only act on GFX1013. */
void
radv_bc250_directmesh_env(void)
{
   const char *on = getenv("RADV_DIRECTMESH");
   if (!on || strcmp(on, "1"))
      return;
   static const char *const settings[][2] = {
      /* BC-250 base: Mesh through hybrid Task; split/expansion stays the automatic fallback. */
      {"RADV_BC250_NATIVE_TASK", "0"}, {"RADV_BC250_HYBRID_TASK", "1"},
      {"BC250_EXPERIMENTAL_COMPUTE_CU_MODE", "false"},
      {"BC250_BALANCED_SLICES", "true"}, {"BC250_SINGLE_PIECE", "true"}, {"BC250_CACHE_PLAN", "true"},
      {"BC250_TRANSIENT_ARENA", "true"}, {"BC250_PARALLEL_CULL", "true"}, {"BC250_COMPACT_VERTICES", "false"},
      /* BC250_EXPERIMENTAL_PRIVATE_GTT is not set: it moved every private allocation out of the VRAM
       * carveout into snooped GTT (a hang experiment, never shown to help). */
      {"BC250_OUTPUT_REGIONS", "true"},
      {"BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT", "false"}, {"RADV_BC250_MESH_ONCE", "false"},
      {"BC250_EXPERIMENTAL_DIRECT_SPLIT", "true"}, {"BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS", "true"},
      {"RADV_BC250_SPLIT_MESH", "true"}, {"RADV_BC250_EXPAND_PRIMITIVES", "true"},
      {"BC250_EXPERIMENTAL_CULL_COMPACT", "true"}, {"BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES", "true"},
      {"BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH", "0"}, {"BC250_CAPTURE_RAW_IBS", "false"},
      {"RADV_BC250_EXPOSE_FAST_BINDING", "1"}, {"RADV_BC250_SUBMIT_KNOWN_SIGNALS", "1"},
      {"RADV_BC250_LOCAL_BOS", "1"}, {"RADV_BC250_KEEP_IB_KB", "640"},
      {"RADV_BC250_PERF_PIECE_PRIMS", "64"}, {"RADV_BC250_SCRATCH_REUSE", "1"},
      {"RADV_BC250_MESH_DIRECT_READ", "full"}, {"RADV_BC250_MESH_AUTOCULL", "1"},
      {"RADV_BC250_MESH_AUTOCULL_WIDE", "1"},
      /* Direct Mesh path. */
      {"RADV_BC250_MESH_SAFE_FAST", "1"}, {"RADV_BC250_MESH_SAFE_PIECES", "1"},
      {"RADV_BC250_MESH_SAFE_OWNED", "1"}, {"RADV_BC250_MESH_SAFE_LOCAL", "1"},
      {"RADV_BC250_MESH_SAFE_CHECK", "1"}, {"RADV_BC250_MESH_SAFE_CORNERS", "1"},
      {"RADV_BC250_MESH_SAFE_AUTOCULL", "1"}, {"RADV_BC250_MESH_SAFE_PARALLEL", "1"},
      {"RADV_BC250_MESH_SAFE_BARY", "1"}, {"RADV_BC250_BARY_IO16", "1"},
      {"RADV_BC250_MESH_SAFE_BARY_TINY", "1"}, {"RADV_BC250_MESH_SAFE_BARY_AFFINE", "1"},
      {"RADV_BC250_MESH_SAFE_BARY_LAST", "1"}, {"RADV_BC250_BARY_CORNER_ID", "1"},
      {"RADV_BC250_MESH_ALLOW_POS1", "1"}, {"RADV_BC250_MESH_SPLIT_ANY", "1"},
      {"RADV_BC250_MESH_NESTED_SLICE", "1"}, {"RADV_BC250_MESH_SAFE_SPLIT_PIECES", "1"},
      {"RADV_BC250_MESH_SAFE_ADAPTIVE", "1"},
      /* Never the raw route: a Mesh pipeline without a protected route is refused. */
      {"RADV_BC250_MESH_FAIL_CLOSED", "1"},
      /* More shapes admitted to the safe routes (gates PP1, TG1, PX1). */
      {"RADV_BC250_MESH_DEAD_PAYLOAD", "1"}, {"RADV_BC250_MESH_PIECE_PRIMID", "1"},
      {"RADV_BC250_MESH_SAFE_PIECES_EXT", "1"}, {"RADV_BC250_TASK_GRID_FOLD", "1"},
      /* Renumbered referenced vertices after culling instead of private corners (gate CM1). */
      {"RADV_BC250_MESH_SAFE_COMPACT", "1"},
      /* Shared vertices on the owned route, private provoking corners (gates PS1, PS2). */
      {"RADV_BC250_MESH_PP_SHARE", "1"},
      /* Skip adaptive checks whose result is already known (gate LC1). */
      {"RADV_BC250_MESH_LEAN_CHECK", "1"},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(settings); i++)
      setenv(settings[i][0], settings[i][1], 0);
   /* Mesh shaders are exposed through RADV_PERFTEST=mesh on this driver line. */
   const char *perftest = getenv("RADV_PERFTEST");
   if (!perftest || !*perftest) {
      setenv("RADV_PERFTEST", "mesh,nircache", 1);
   } else if (!strstr(perftest, "mesh")) {
      char buf[256];
      snprintf(buf, sizeof(buf), "%s,mesh", perftest);
      setenv("RADV_PERFTEST", buf, 1);
   }
}

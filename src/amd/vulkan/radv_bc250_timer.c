/*
 * Copyright © 2026 BC-250 LoneWolf project
 * SPDX-License-Identifier: MIT
 */

/* BC250_MESH_TIMER: see radv_bc250_timer.h. */

#include "radv_bc250_timer.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "radv_buffer.h"
#include "ac_cmdbuf_cp.h"
#include "radv_cmd_buffer.h"
#include "radv_cs.h"
#include "radv_device.h"
#include "radv_physical_device.h"
#include "radv_query.h"
#include "radv_queue.h"
#include "radv_shader.h"
#include "util/os_time.h"
#include "util/u_debug.h"
#include "vk_queue.h"

#define SLOT_MASK (RADV_BC250_TIMER_SLOTS - 1)

/* ---- device ---------------------------------------------------------- */

static int timer_worker(void *arg);

void
radv_bc250_timer_init(struct radv_device *device)
{
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   const struct radv_physical_device *pdev = radv_device_physical(device);

   memset(t, 0, sizeof(*t));
   if (!debug_get_bool_option("BC250_MESH_TIMER", false))
      return;

   const VkResult result = radv_bo_create(device, NULL, (uint64_t)RADV_BC250_TIMER_SLOTS * 16, 4096, RADEON_DOMAIN_GTT,
                                          RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING,
                                          RADV_BO_PRIORITY_UPLOAD_BUFFER, 0, true, &t->bo);
   if (result != VK_SUCCESS) {
      fprintf(stderr, "BC250_MESH_TIMER: ring buffer allocation failed (%d), timer off\n", result);
      t->bo = NULL;
      return;
   }
   if (device->ws->buffer_make_resident(device->ws, t->bo, true) != VK_SUCCESS) {
      fprintf(stderr, "BC250_MESH_TIMER: ring buffer residency failed, timer off\n");
      radv_bo_destroy(device, NULL, t->bo);
      t->bo = NULL;
      return;
   }
   t->map = radv_buffer_map(device->ws, t->bo);
   if (!t->map) {
      fprintf(stderr, "BC250_MESH_TIMER: ring buffer map failed, timer off\n");
      device->ws->buffer_make_resident(device->ws, t->bo, false);
      radv_bo_destroy(device, NULL, t->bo);
      t->bo = NULL;
      return;
   }
   memset(t->map, 0, (size_t)RADV_BC250_TIMER_SLOTS * 16);
   t->va = radv_buffer_get_va(t->bo);

   t->classes = calloc(RADV_BC250_TIMER_MAX_CLASSES, sizeof(*t->classes));
   t->pending = calloc(RADV_BC250_TIMER_MAX_PENDING, sizeof(*t->pending));
   if (!t->classes || !t->pending) {
      free(t->classes);
      free(t->pending);
      t->classes = NULL;
      t->pending = NULL;
      device->ws->buffer_make_resident(device->ws, t->bo, false);
      radv_bo_destroy(device, NULL, t->bo);
      t->bo = NULL;
      return;
   }

   simple_mtx_init(&t->mtx, mtx_plain);
   t->sample = MAX2(debug_get_num_option("BC250_MESH_TIMER_SAMPLE", 1), 1);
   const char *interval = getenv("BC250_MESH_TIMER_INTERVAL");
   t->interval_s = interval ? atof(interval) : 5.0;
   if (t->interval_s < 0)
      t->interval_s = 0;
   t->fake = debug_get_bool_option("BC250_MESH_TIMER_FAKE", false);
   t->per_submit_lines = debug_get_bool_option("BC250_MESH_TIMER_SUBMITS", false);
   t->compute = debug_get_bool_option("BC250_MESH_TIMER_COMPUTE", true);
   t->safe_stats = debug_get_bool_option("RADV_BC250_MESH_SAFE_STATS", false);
   if (t->safe_stats) {
      const VkResult stats_result = radv_bo_create(device, NULL,
         (uint64_t)RADV_BC250_TIMER_SLOTS * RADV_BC250_SAFE_STATS_DWORDS * sizeof(uint32_t), 4096,
         RADEON_DOMAIN_GTT, RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING,
         RADV_BO_PRIORITY_UPLOAD_BUFFER, 0, true, &t->safe_stats_bo);
      if (stats_result == VK_SUCCESS &&
          device->ws->buffer_make_resident(device->ws, t->safe_stats_bo, true) == VK_SUCCESS) {
         t->safe_stats_map = radv_buffer_map(device->ws, t->safe_stats_bo);
         if (t->safe_stats_map) {
            memset(t->safe_stats_map, 0,
                   (size_t)RADV_BC250_TIMER_SLOTS * RADV_BC250_SAFE_STATS_DWORDS * sizeof(uint32_t));
            t->safe_stats_va = radv_buffer_get_va(t->safe_stats_bo);
         }
      }
      if (!t->safe_stats_map) {
         fprintf(stderr, "BC250_MESH_TIMER: SAFE_STATS buffer failed; counters off\n");
         if (t->safe_stats_bo) {
            device->ws->buffer_make_resident(device->ws, t->safe_stats_bo, false);
            radv_bo_destroy(device, NULL, t->safe_stats_bo);
            t->safe_stats_bo = NULL;
         }
         t->safe_stats = false;
      }
   }
   /* vkGetPhysicalDeviceProperties: timestampPeriod = 1e6 / clock_crystal_freq (kHz). */
   t->ns_per_tick = pdev->info.clock_crystal_freq ? 1000000.0 / pdev->info.clock_crystal_freq : 1.0;
   t->fake_clock = 1000;
   const char *skip = getenv("BC250_MESH_TIMER_SKIP");
   if (skip)
      snprintf(t->skip_spec, sizeof(t->skip_spec), "%s", skip);

   const char *path = getenv("BC250_MESH_TIMER_FILE");
   if (path && path[0]) {
      t->file = fopen(path, "a");
      if (!t->file)
         fprintf(stderr, "BC250_MESH_TIMER: cannot open %s\n", path);
   }
   t->start_ns = t->last_print_ns = os_time_get_nano();

   char header[512];
   snprintf(header, sizeof(header),
            "# BC250_MESH_TIMER v2 pid=%ld ns_per_tick=%.3f interval=%.3f sample=%u fake=%u bo_va=0x%" PRIx64
            " slots=%u compute=%u safe_stats=%u skip=\"%s\"\n",
            (long)getpid(), t->ns_per_tick, t->interval_s, t->sample, t->fake, t->va, RADV_BC250_TIMER_SLOTS,
            t->compute, t->safe_stats, t->skip_spec);
   fputs(header, stderr);
   if (t->file) {
      fputs(header, t->file);
      fflush(t->file);
      fsync(fileno(t->file));
   }
   t->enabled = true;
   t->device = device;
   if (debug_get_bool_option("BC250_MESH_TIMER_ASYNC", false)) {
      t->async = true;
      t->worker_on = true;
      if (thrd_create(&t->worker, timer_worker, t) != thrd_success) {
         fprintf(stderr, "BC250_MESH_TIMER: async worker thread failed, synchronous mode\n");
         t->async = false;
         t->worker_on = false;
      }
   }
}

static void radv_bc250_timer_print(struct radv_device *device, bool final);
struct radv_bc250_timer_pending;
static void fold_pending(struct radv_bc250_mesh_timer *t, struct radv_bc250_timer_pending *p);
static bool pending_complete(const struct radv_bc250_mesh_timer *t, const struct radv_bc250_timer_pending *p);

void
radv_bc250_timer_finish(struct radv_device *device)
{
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   if (!t->enabled)
      return;
   t->enabled = false;

   if (t->async) {
      /* The worker drains the queued records and writes them before it exits; then the final
       * fold / print below writes directly. */
      simple_mtx_lock(&t->mtx);
      t->worker_stop = true;
      simple_mtx_unlock(&t->mtx);
      thrd_join(t->worker, NULL);
      t->worker_on = false;
      free(t->wrecs);
      t->wrecs = NULL;
      free(t->srecs);
      t->srecs = NULL;
      free(t->abuf);
      t->abuf = NULL;
   }

   simple_mtx_lock(&t->mtx);
   /* Include every submission the GPU has already finished (the device is idle at destroy). */
   while (t->pending_count && pending_complete(t, &t->pending[t->pending_head])) {
      struct radv_bc250_timer_pending *p = &t->pending[t->pending_head];
      fold_pending(t, p);
      free(p->entries);
      p->entries = NULL;
      t->pending_head = (t->pending_head + 1) % RADV_BC250_TIMER_MAX_PENDING;
      t->pending_count--;
   }
   radv_bc250_timer_print(device, true);
   for (uint32_t i = 0; i < t->pending_count; i++)
      free(t->pending[(t->pending_head + i) % RADV_BC250_TIMER_MAX_PENDING].entries);
   t->pending_count = 0;
   if (t->file) {
      fclose(t->file);
      t->file = NULL;
   }
   simple_mtx_unlock(&t->mtx);
   simple_mtx_destroy(&t->mtx);

   free(t->classes);
   t->classes = NULL;
   free(t->pending);
   t->pending = NULL;
   device->ws->buffer_unmap(device->ws, t->bo, false);
   device->ws->buffer_make_resident(device->ws, t->bo, false);
   radv_bo_destroy(device, NULL, t->bo);
   t->bo = NULL;
   if (t->safe_stats_bo) {
      device->ws->buffer_unmap(device->ws, t->safe_stats_bo, false);
      device->ws->buffer_make_resident(device->ws, t->safe_stats_bo, false);
      radv_bo_destroy(device, NULL, t->safe_stats_bo);
      t->safe_stats_bo = NULL;
   }
}

/* ---- classes --------------------------------------------------------- */

static bool
skip_matches(const struct radv_bc250_mesh_timer *t, const struct radv_bc250_timer_class *c)
{
   if (!t->skip_spec[0])
      return false;
   char spec[sizeof(t->skip_spec)];
   memcpy(spec, t->skip_spec, sizeof(spec));
   char *save = NULL;
   for (char *tok = strtok_r(spec, ", ", &save); tok; tok = strtok_r(NULL, ", ", &save)) {
      if (!strcmp(tok, "all"))
         return true;
      if (!strcmp(tok, "split"))
         if (c->pieces)
            return true;
      if (!strcmp(tok, "expanded"))
         if (c->expanded)
            return true;
      if (!strcmp(tok, "raw"))
         if (!c->expanded && !c->pieces && !c->replay)
            return true;
      if (!strcmp(tok, "replay") || !strcmp(tok, "task"))
         if (c->replay)
            return true;
      if (!strcmp(tok, "merged"))
         if (c->merge_k > 1)
            return true;
      if (!strcmp(tok, "amd"))
         if (c->amd)
            return true;
      if (!strcmp(tok, "autocull"))
         if (c->autocull)
            return true;
      if (!strcmp(tok, "compact"))
         if (c->compact)
            return true;
      const size_t n = strlen(tok);
      if (n && n <= 16 && strspn(tok, "0123456789abcdefABCDEF") == n) {
         char lower[17];
         for (size_t i = 0; i <= n; i++)
            lower[i] = tok[i] >= 'A' && tok[i] <= 'F' ? tok[i] - 'A' + 'a' : tok[i];
         if (!strncmp(lower, c->hex, n))
            return true;
      }
   }
   return false;
}

/* Called with the mutex held. */
static uint32_t
class_lookup(struct radv_device *device, const struct radv_cmd_buffer *cmd_buffer, const struct radv_shader *ms)
{
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   const struct radv_graphics_pipeline *pipeline = cmd_buffer->state.graphics_pipeline;
   const uint8_t pieces = pipeline ? MIN2(pipeline->bc250_direct_split_pieces, 255) : 0;
   const bool replay = pipeline && pipeline->bc250_task_pipeline != VK_NULL_HANDLE;

   for (uint32_t i = 0; i < t->class_count; i++) {
      struct radv_bc250_timer_class *c = &t->classes[i];
      if (c->shader == ms && !memcmp(c->hash, ms->hash, sizeof(c->hash)) && c->pieces == pieces &&
          c->replay == replay && c->safe_direct == ms->info.ms.bc250_safe_direct)
         return i;
   }
   if (t->class_count == RADV_BC250_TIMER_MAX_CLASSES)
      return RADV_BC250_TIMER_MAX_CLASSES - 1; /* overflow bucket */

   struct radv_bc250_timer_class *c = &t->classes[t->class_count];
   memset(c, 0, sizeof(*c));
   c->shader = ms;
   memcpy(c->hash, ms->hash, sizeof(c->hash));
   for (unsigned i = 0; i < 8; i++)
      snprintf(&c->hex[2 * i], 3, "%02x", c->hash[i]);
   c->vertices = ms->info.ms.bc250_api_vertices;
   c->primitives = ms->info.ms.bc250_api_primitives;
   c->local_size = MIN2(ms->info.cs.block_size[0] * ms->info.cs.block_size[1] * ms->info.cs.block_size[2], 65535);
   c->workgroup = ms->info.workgroup_size;
   c->wave = ms->info.wave_size;
   c->pieces = pieces;
   c->merge_k = ms->info.ms.bc250_merge_k;
   c->expanded = ms->info.ms.bc250_expanded;
   c->replay = replay;
   c->amd = ms->info.ms.bc250_amd_mesh;
   c->autocull = ms->info.ms.bc250_autocull;
   c->compact = ms->info.ms.bc250_compact;
   c->safe_direct = ms->info.ms.bc250_safe_direct;
   if (c->replay)
      snprintf(c->route, sizeof(c->route), "task-replay%s", c->expanded ? "+expanded" : "");
   else if (c->pieces)
      snprintf(c->route, sizeof(c->route), "%s=%u%s", c->safe_direct ? "safe-direct-pieces" : "split",
               c->pieces, c->safe_direct ? "" : c->expanded ? "+expanded" : "");
   else if (c->merge_k > 1)
      snprintf(c->route, sizeof(c->route), "merged=%u", c->merge_k);
   else if (c->safe_direct)
      snprintf(c->route, sizeof(c->route), "safe-direct");
   else if (c->amd)
      snprintf(c->route, sizeof(c->route), "amd-raw");
   else
      snprintf(c->route, sizeof(c->route), "%s", c->expanded ? "expanded" : "raw");
   if (c->autocull)
      strncat(c->route, "+autocull", sizeof(c->route) - strlen(c->route) - 1);
   if (c->compact)
      strncat(c->route, "+cmp", sizeof(c->route) - strlen(c->route) - 1);
   c->cur.min = c->total.min = UINT64_MAX;
   c->skip = skip_matches(t, c);
   return t->class_count++;
}

static uint32_t
class_for_draw(struct radv_device *device, struct radv_cmd_buffer *cmd_buffer, const struct radv_shader *ms)
{
   struct radv_bc250_timer_cmd *tc = &cmd_buffer->bc250_timer;
   if (tc->last_ms == ms)
      return tc->last_class;
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   simple_mtx_lock(&t->mtx);
   const uint32_t id = class_lookup(device, cmd_buffer, ms);
   simple_mtx_unlock(&t->mtx);
   tc->last_ms = ms;
   tc->last_class = id;
   return id;
}

/* ---- command buffers ------------------------------------------------- */

static void
emit_timestamp(struct radv_cmd_buffer *cmd_buffer, uint64_t va)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   radeon_check_space(device->ws, cmd_buffer->cs->b, 16);
   radv_write_timestamp(cmd_buffer, va, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

static bool
push_entry(struct radv_cmd_buffer *cmd_buffer, uint64_t idx, uint32_t class_id, enum radv_bc250_timer_kind kind)
{
   struct radv_bc250_timer_cmd *tc = &cmd_buffer->bc250_timer;
   if (tc->count == tc->cap) {
      const uint32_t cap = tc->cap ? tc->cap * 2 : 256;
      struct radv_bc250_timer_entry *n = realloc(tc->entries, cap * sizeof(*n));
      if (!n)
         return false;
      tc->entries = n;
      tc->cap = cap;
   }
   tc->entries[tc->count++] = (struct radv_bc250_timer_entry){.idx = idx, .class_id = class_id, .kind = kind};
   return true;
}

void
radv_bc250_timer_cmd_reset(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_bc250_timer_cmd *tc = &cmd_buffer->bc250_timer;
   tc->count = 0;
   tc->has_cb = false;
   tc->last_ms = NULL;
   tc->draws = tc->vs_draws = tc->skipped = 0;
}

void
radv_bc250_timer_cmd_destroy(struct radv_cmd_buffer *cmd_buffer)
{
   free(cmd_buffer->bc250_timer.entries);
   cmd_buffer->bc250_timer.entries = NULL;
   cmd_buffer->bc250_timer.cap = cmd_buffer->bc250_timer.count = 0;
}

void
radv_bc250_timer_cmd_begin(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   radv_bc250_timer_cmd_reset(cmd_buffer);
   /* Graphics and (BC250_MESH_TIMER_COMPUTE, default on) compute queues: RELEASE_MEM with a
    * bottom-of-pipe timestamp is the packet vkCmdWriteTimestamp2 uses on both rings (ME and MEC). */
   if (cmd_buffer->qf != RADV_QUEUE_GENERAL && !(cmd_buffer->qf == RADV_QUEUE_COMPUTE && t->compute))
      return;
   radv_cs_add_buffer(device->ws, cmd_buffer->cs->b, t->bo);
   if (cmd_buffer->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return;
   const uint64_t idx = __atomic_fetch_add(&t->next, 1, __ATOMIC_RELAXED);
   cmd_buffer->bc250_timer.cb_idx = idx;
   cmd_buffer->bc250_timer.has_cb = true;
   emit_timestamp(cmd_buffer, t->va + (idx & SLOT_MASK) * 16);
}

void
radv_bc250_timer_cmd_end(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   if (!cmd_buffer->bc250_timer.has_cb)
      return;
   emit_timestamp(cmd_buffer, t->va + (cmd_buffer->bc250_timer.cb_idx & SLOT_MASK) * 16 + 8);
}

void
radv_bc250_timer_cmd_execute(struct radv_cmd_buffer *primary, struct radv_cmd_buffer *secondary)
{
   const struct radv_bc250_timer_cmd *s = &secondary->bc250_timer;
   for (uint32_t i = 0; i < s->count; i++)
      if (!push_entry(primary, s->entries[i].idx, s->entries[i].class_id, s->entries[i].kind))
         break;
   primary->bc250_timer.draws += s->draws;
   primary->bc250_timer.vs_draws += s->vs_draws;
   primary->bc250_timer.skipped += s->skipped;
}

/* ---- draws ----------------------------------------------------------- */

int32_t
radv_bc250_timer_draw_begin(struct radv_cmd_buffer *cmd_buffer, enum radv_bc250_timer_kind kind)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   const struct radv_shader *ms = radv_bc250_api_mesh_shader(cmd_buffer);
   if (!ms || cmd_buffer->qf != RADV_QUEUE_GENERAL)
      return RADV_BC250_TIMER_NOT_TIMED;

   if (t->safe_stats && ms->info.ms.bc250_safe_stats) {
      cmd_buffer->state.bc250_safe_stats_buf_va = 0;
      cmd_buffer->state.dirty |= RADV_CMD_DIRTY_NGG_STATE;
   }

   const uint32_t class_id = class_for_draw(device, cmd_buffer, ms);
   struct radv_bc250_timer_class *c = &t->classes[class_id];
   cmd_buffer->bc250_timer.draws++;
   __atomic_fetch_add(&c->cur.draws[kind], 1, __ATOMIC_RELAXED);
   __atomic_fetch_add(&c->total.draws[kind], 1, __ATOMIC_RELAXED);

   if (c->skip) {
      __atomic_fetch_add(&c->cur.skipped, 1, __ATOMIC_RELAXED);
      __atomic_fetch_add(&c->total.skipped, 1, __ATOMIC_RELAXED);
      cmd_buffer->bc250_timer.skipped++;
      if (!c->skip_printed) {
         simple_mtx_lock(&t->mtx);
         if (!c->skip_printed) {
            c->skip_printed = true;
            fprintf(stderr, "BC250_MESH_TIMER skipping hash=%s V=%u P=%u LS=%u W%u WG%u route=%s\n", c->hex,
                    c->vertices, c->primitives, c->local_size, c->wave, c->workgroup, c->route);
            if (t->file) {
               fprintf(t->file, "K hash=%s V=%u P=%u LS=%u W=%u WG=%u route=%s\n", c->hex, c->vertices, c->primitives,
                       c->local_size, c->wave, c->workgroup, c->route);
               fflush(t->file);
            }
         }
         simple_mtx_unlock(&t->mtx);
      }
      return RADV_BC250_TIMER_SKIP_DRAW;
   }

   if (t->sample > 1 && (__atomic_fetch_add(&t->sample_counter, 1, __ATOMIC_RELAXED) % t->sample) != 0)
      return RADV_BC250_TIMER_NOT_TIMED;

   const uint64_t idx = __atomic_fetch_add(&t->next, 1, __ATOMIC_RELAXED);
   if (!push_entry(cmd_buffer, idx, class_id, kind))
      return RADV_BC250_TIMER_NOT_TIMED;
   emit_timestamp(cmd_buffer, t->va + (idx & SLOT_MASK) * 16);
   if (t->safe_stats && ms->info.ms.bc250_safe_stats) {
      const uint64_t stats_offset = (idx & SLOT_MASK) * RADV_BC250_SAFE_STATS_DWORDS * sizeof(uint32_t);
      const uint64_t stats_va = t->safe_stats_va + stats_offset;
      uint32_t zero[RADV_BC250_SAFE_STATS_DWORDS] = {0};
      radv_cs_add_buffer(device->ws, cmd_buffer->cs->b, t->safe_stats_bo);
      ac_emit_cp_write_data(cmd_buffer->cs->b, V_371_MICRO_ENGINE, V_371_MEMORY,
                            stats_va, ARRAY_SIZE(zero), zero, false);
      cmd_buffer->state.bc250_safe_stats_buf_va = stats_va;
      cmd_buffer->state.dirty |= RADV_CMD_DIRTY_NGG_STATE;
   }
   return (int32_t)(cmd_buffer->bc250_timer.count - 1);
}

void
radv_bc250_timer_draw_end(struct radv_cmd_buffer *cmd_buffer, int32_t entry)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   if (entry < 0)
      return;
   const uint64_t idx = cmd_buffer->bc250_timer.entries[entry].idx;
   emit_timestamp(cmd_buffer, t->va + (idx & SLOT_MASK) * 16 + 8);
   if (t->safe_stats) {
      cmd_buffer->state.bc250_safe_stats_buf_va = 0;
      cmd_buffer->state.dirty |= RADV_CMD_DIRTY_NGG_STATE;
   }
}

/* ---- submissions and readback ---------------------------------------- */

static const char *
queue_kind(unsigned qf)
{
   switch (qf) {
   case RADV_QUEUE_GENERAL:
      return "gfx";
   case RADV_QUEUE_COMPUTE:
      return "ace";
   case RADV_QUEUE_TRANSFER:
      return "dma";
   case RADV_QUEUE_SPARSE:
      return "sparse";
   case RADV_QUEUE_VIDEO_DEC:
      return "vdec";
   case RADV_QUEUE_VIDEO_ENC:
      return "venc";
   default:
      return "other";
   }
}

static uint64_t
raw_clock_ns(void)
{
   struct timespec ts;
   if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts))
      return 0;
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void file_line(struct radv_bc250_mesh_timer *t, const char *line);

/* BC250_MESH_TIMER_ASYNC, mutex held: room for one more record (NULL: out of memory, dropped). */
static struct radv_bc250_timer_wrec *
wrec_push(struct radv_bc250_mesh_timer *t)
{
   if (t->wrec_count == t->wrec_cap) {
      const uint32_t cap = t->wrec_cap ? t->wrec_cap * 2 : 1024;
      struct radv_bc250_timer_wrec *n = realloc(t->wrecs, cap * sizeof(*n));
      if (!n)
         return NULL;
      t->wrecs = n;
      t->wrec_cap = cap;
   }
   struct radv_bc250_timer_wrec *r = &t->wrecs[t->wrec_count++];
   memset(r, 0, sizeof(*r));
   return r;
}

void
radv_bc250_timer_present(struct radv_device *device, const struct radv_queue *queue)
{
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   __atomic_fetch_add(&t->frames, 1, __ATOMIC_RELAXED);
   if (t->per_submit_lines && t->file && t->async) {
      const uint64_t now = os_time_get_nano();
      simple_mtx_lock(&t->mtx);
      struct radv_bc250_timer_wrec *r = wrec_push(t);
      if (r) {
         r->kind = 'P';
         r->seq = t->seq;
         r->wall_ns = now;
         r->fam = queue->vk.queue_family_index;
         r->idx = queue->vk.index_in_family;
      }
      simple_mtx_unlock(&t->mtx);
      return;
   }
   if (t->per_submit_lines && t->file) {
      /* P: present call (frame boundary on the CPU timeline); seq = the last submission before it. */
      const uint64_t now = os_time_get_nano();
      char line[128];
      simple_mtx_lock(&t->mtx);
      snprintf(line, sizeof(line), "P seq=%" PRIu64 " wall_ns=%" PRIu64 " q=%u.%u\n", t->seq, now,
               queue->vk.queue_family_index, queue->vk.index_in_family);
      file_line(t, line);
      simple_mtx_unlock(&t->mtx);
   }
}

static void
file_line(struct radv_bc250_mesh_timer *t, const char *line)
{
   if (!t->file)
      return;
   if (t->worker_on) {
      /* BC250_MESH_TIMER_ASYNC: only the worker formats (mutex held); it writes the buffer later. */
      const size_t n = strlen(line);
      if (t->abuf_len + n > t->abuf_cap) {
         size_t cap = t->abuf_cap ? t->abuf_cap : 1 << 16;
         while (cap < t->abuf_len + n)
            cap *= 2;
         char *b = realloc(t->abuf, cap);
         if (!b)
            return;
         t->abuf = b;
         t->abuf_cap = cap;
      }
      memcpy(t->abuf + t->abuf_len, line, n);
      t->abuf_len += n;
      return;
   }
   fputs(line, t->file);
}

static void
file_sync(struct radv_bc250_mesh_timer *t)
{
   if (!t->file)
      return;
   if (t->worker_on) {
      t->want_sync = true;
      return;
   }
   fflush(t->file);
   fsync(fileno(t->file));
}

/* Fold one completed submission into the accumulators (mutex held). */
static void
fold_pending(struct radv_bc250_mesh_timer *t, struct radv_bc250_timer_pending *p)
{
   const uint64_t next = __atomic_load_n(&t->next, __ATOMIC_RELAXED);
   uint64_t submit_ticks = 0, mesh_ticks = 0, timed = 0, gpu_t0 = 0, gpu_t1 = 0;

   if (p->has_cb) {
      if (next - p->cb_idx > RADV_BC250_TIMER_SLOTS) {
         t->cur.dropped_slots++;
         t->total.dropped_slots++;
      } else {
         const uint64_t t0 = t->map[(p->cb_idx & SLOT_MASK) * 2], t1 = t->map[(p->cb_idx & SLOT_MASK) * 2 + 1];
         if (t0 && t1 && t1 >= t0) {
            submit_ticks = t1 - t0;
            gpu_t0 = t0;
            gpu_t1 = t1;
         } else {
            t->cur.invalid++;
            t->total.invalid++;
         }
      }
   }

   for (uint32_t i = 0; i < p->count; i++) {
      const struct radv_bc250_timer_entry *e = &p->entries[i];
      if (next - e->idx > RADV_BC250_TIMER_SLOTS) {
         t->cur.dropped_slots++;
         t->total.dropped_slots++;
         continue;
      }
      const uint64_t t0 = t->map[(e->idx & SLOT_MASK) * 2], t1 = t->map[(e->idx & SLOT_MASK) * 2 + 1];
      if (!t0 || !t1 || t1 < t0) {
         t->cur.invalid++;
         t->total.invalid++;
         continue;
      }
      const uint64_t dt = t1 - t0;
      struct radv_bc250_timer_class *c = &t->classes[e->class_id];
      if (t->safe_stats) {
         const uint32_t *diag = t->safe_stats_map + (e->idx & SLOT_MASK) * RADV_BC250_SAFE_STATS_DWORDS;
         for (unsigned j = 0; j < RADV_BC250_SAFE_STATS_DWORDS; j++) {
            c->safe_stats_cur[j] += diag[j];
            c->safe_stats_total[j] += diag[j];
         }
      }
      struct radv_bc250_timer_stats *s[2] = {&c->cur, &c->total};
      for (unsigned k = 0; k < 2; k++) {
         s[k]->timed++;
         s[k]->sum += dt;
         s[k]->min = MIN2(s[k]->min, dt);
         s[k]->max = MAX2(s[k]->max, dt);
      }
      mesh_ticks += dt;
      timed++;
   }

   struct radv_bc250_timer_totals *tt[2] = {&t->cur, &t->total};
   for (unsigned k = 0; k < 2; k++) {
      if (p->qf == RADV_QUEUE_COMPUTE) {
         /* Compute-queue command buffers: own totals, the graphics numbers stay comparable with v1 logs. */
         tt[k]->ace_submits++;
         tt[k]->ace_ticks += submit_ticks;
         continue;
      }
      tt[k]->submits++;
      tt[k]->submit_ticks += submit_ticks;
      tt[k]->mesh_ticks += mesh_ticks;
      tt[k]->mesh_draws += p->mesh_draws;
      tt[k]->timed += timed;
      tt[k]->vs_draws += p->vs_draws;
      tt[k]->skipped += p->skipped;
   }

   if (t->per_submit_lines && t->file && t->worker_on) {
      /* BC250_MESH_TIMER_ASYNC: the worker formats the S line outside the mutex. */
      if (t->srec_count == t->srec_cap) {
         const uint32_t cap = t->srec_cap ? t->srec_cap * 2 : 4096;
         struct radv_bc250_timer_srec *n = realloc(t->srecs, cap * sizeof(*n));
         if (!n)
            return;
         t->srecs = n;
         t->srec_cap = cap;
      }
      t->srecs[t->srec_count++] = (struct radv_bc250_timer_srec){
         .seq = p->seq, .wall_ns = p->wall_ns, .submit_ticks = submit_ticks, .mesh_ticks = mesh_ticks,
         .timed = timed, .gpu_t0 = gpu_t0, .gpu_t1 = gpu_t1, .mesh_draws = p->mesh_draws, .vs_draws = p->vs_draws,
         .fam = p->vk_family, .idx = p->vk_index, .cb_i = p->cb_i, .cb_n = p->cb_n, .qf = p->qf};
   } else if (t->per_submit_lines && t->file) {
      char line[384];
      /* gpu_t0/gpu_t1: raw GPU clock ticks of the command buffer's begin/end timestamps; with the
       * T calibration lines they place every submission on the CPU timeline (idle-gap analysis).
       * q/qf/cb: the queue and the command buffer's position in its submission (W line, same seq). */
      snprintf(line, sizeof(line), "S seq=%" PRIu64 " wall_ns=%" PRIu64 " submit_ns=%.0f mesh_ns=%.0f mesh_draws=%u timed=%" PRIu64
               " vs_draws=%u gpu_t0=%" PRIu64 " gpu_t1=%" PRIu64 " q=%u.%u qf=%s cb=%u/%u\n",
               p->seq, p->wall_ns, submit_ticks * t->ns_per_tick, mesh_ticks * t->ns_per_tick, p->mesh_draws, timed,
               p->vs_draws, gpu_t0, gpu_t1, p->vk_family, p->vk_index, queue_kind(p->qf), p->cb_i, p->cb_n);
      file_line(t, line);
   }
}

static void
drop_oldest_pending(struct radv_bc250_mesh_timer *t)
{
   struct radv_bc250_timer_pending *p = &t->pending[t->pending_head];
   t->cur.dropped_submits++;
   t->total.dropped_submits++;
   free(p->entries);
   p->entries = NULL;
   t->pending_head = (t->pending_head + 1) % RADV_BC250_TIMER_MAX_PENDING;
   t->pending_count--;
}

static bool
pending_complete(const struct radv_bc250_mesh_timer *t, const struct radv_bc250_timer_pending *p)
{
   if (!p->has_cb)
      return true; /* nothing to wait for: only counters */
   const uint64_t next = __atomic_load_n(&t->next, __ATOMIC_RELAXED);
   if (next - p->cb_idx > RADV_BC250_TIMER_SLOTS)
      return true; /* overwritten: folded as dropped */
   return __atomic_load_n(&t->map[(p->cb_idx & SLOT_MASK) * 2 + 1], __ATOMIC_ACQUIRE) != 0;
}

/* Tests only (drm-shim): synthesise the GPU timestamps of a submission on the
 * CPU. Each timed draw lasts 100 * V + P ticks, the command buffer the sum
 * plus 1000 ticks; the end marker is written last. */
static void
fake_fill(struct radv_bc250_mesh_timer *t, const struct radv_bc250_timer_pending *p)
{
   /* The fake GPU runs each command buffer at its submission (clock = CLOCK_MONOTONIC / ns_per_tick,
    * never backwards), so the per-queue timeline tools also work on drm-shim logs. */
   uint64_t clk = MAX2(t->fake_clock, (uint64_t)(os_time_get_nano() / t->ns_per_tick));
   const uint64_t cb_t0 = clk;
   for (uint32_t i = 0; i < p->count; i++) {
      const struct radv_bc250_timer_entry *e = &p->entries[i];
      const struct radv_bc250_timer_class *c = &t->classes[e->class_id];
      const uint64_t dt = 100u * c->vertices + c->primitives;
      t->map[(e->idx & SLOT_MASK) * 2] = clk;
      t->map[(e->idx & SLOT_MASK) * 2 + 1] = clk + dt;
      clk += dt;
   }
   clk += 1000;
   t->fake_clock = clk;
   if (p->has_cb) {
      t->map[(p->cb_idx & SLOT_MASK) * 2] = cb_t0;
      __atomic_store_n(&t->map[(p->cb_idx & SLOT_MASK) * 2 + 1], clk, __ATOMIC_RELEASE);
   }
}

static void
radv_bc250_timer_print(struct radv_device *device, bool final)
{
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   const uint64_t now = os_time_get_nano();
   const double dt_s = MAX2((now - t->last_print_ns) / 1e9, 1e-9);
   const double elapsed_s = (now - t->start_ns) / 1e9;
   const double ns = t->ns_per_tick;
   const struct radv_bc250_timer_totals *tot = final ? &t->total : &t->cur;
   const double span_s = final ? MAX2(elapsed_s, 1e-9) : dt_s;

   /* Clock calibration (BC250_MESH_TIMER_SUBMITS only): the GPU clock read the same way as
    * vkGetCalibratedTimestampsKHR (RADEON_TIMESTAMP query), bracketed by CPU CLOCK_MONOTONIC reads. */
   if (t->per_submit_lines && t->file) {
      const uint64_t c0 = os_time_get_nano();
      const uint64_t g = t->fake ? (uint64_t)(c0 / t->ns_per_tick) : device->ws->query_value(device->ws, RADEON_TIMESTAMP);
      const uint64_t c1 = os_time_get_nano();
      const uint64_t r = raw_clock_ns();
      const uint64_t c2 = os_time_get_nano();
      char tl[200];
      /* raw_ns: CLOCK_MONOTONIC_RAW read right after (at cpu_raw_at_ns on CLOCK_MONOTONIC): maps
       * Wine QueryPerformanceCounter based traces (VKD3D_QUEUE_PROFILE_ABSOLUTE=1) onto this timeline. */
      snprintf(tl, sizeof(tl), "T cpu_ns=%" PRIu64 " gpu_tick=%" PRIu64 " err_ns=%" PRIu64 " raw_ns=%" PRIu64
               " cpu_raw_at_ns=%" PRIu64 "\n", c0 + (c1 - c0) / 2, g, (c1 - c0) / 2, r, c1 + (c2 - c1) / 2);
      file_line(t, tl);
   }
   const uint64_t frames = tot->frames;
   const double per_frame_div = frames ? (double)frames : (tot->submits ? (double)tot->submits : 1.0);
   const char *per_frame_unit = frames ? "frame" : "submit";
   char line[1024];

   snprintf(line, sizeof(line),
            "BC250_MESH_TIMER %s t=%.1fs dt=%.1fs frames=%" PRIu64 " submits=%" PRIu64 " submit_gpu_ms/s=%.1f mesh_gpu_ms/s=%.1f"
            " mesh_share_of_submit=%.1f%% mesh_draws/s=%.0f timed/s=%.0f vs_draws/s=%.0f skipped/s=%.0f"
            " dropped_slots=%" PRIu64 " dropped_submits=%" PRIu64 " invalid=%" PRIu64 " per_%s: submit_ms=%.3f mesh_ms=%.3f mesh_draws=%.1f"
            " ace_ms=%.3f\n",
            final ? "FINAL" : "interval", elapsed_s, span_s, frames, tot->submits, tot->submit_ticks * ns / 1e6 / span_s,
            tot->mesh_ticks * ns / 1e6 / span_s,
            tot->submit_ticks ? 100.0 * tot->mesh_ticks / tot->submit_ticks : 0.0, tot->mesh_draws / span_s,
            tot->timed / span_s, tot->vs_draws / span_s, tot->skipped / span_s, tot->dropped_slots,
            tot->dropped_submits, tot->invalid, per_frame_unit, tot->submit_ticks * ns / 1e6 / per_frame_div,
            tot->mesh_ticks * ns / 1e6 / per_frame_div, tot->mesh_draws / per_frame_div,
            tot->ace_ticks * ns / 1e6 / per_frame_div);
   fputs(line, stderr);
   snprintf(line, sizeof(line),
            "I %s t=%.3f dt=%.3f frames=%" PRIu64 " submits=%" PRIu64 " submit_ns=%.0f mesh_ns=%.0f mesh_draws=%" PRIu64
            " timed=%" PRIu64 " vs_draws=%" PRIu64 " skipped=%" PRIu64 " dropped_slots=%" PRIu64 " dropped_submits=%" PRIu64
            " invalid=%" PRIu64 " ace_submits=%" PRIu64 " ace_ns=%.0f\n",
            final ? "final" : "interval", elapsed_s, span_s, frames, tot->submits, tot->submit_ticks * ns,
            tot->mesh_ticks * ns, tot->mesh_draws, tot->timed, tot->vs_draws, tot->skipped, tot->dropped_slots,
            tot->dropped_submits, tot->invalid, tot->ace_submits, tot->ace_ticks * ns);
   file_line(t, line);

   /* Classes sorted by GPU time (insertion sort on an index array; few classes). */
   uint32_t order[RADV_BC250_TIMER_MAX_CLASSES];
   for (uint32_t i = 0; i < t->class_count; i++)
      order[i] = i;
   for (uint32_t i = 1; i < t->class_count; i++) {
      const uint32_t v = order[i];
      const uint64_t key = final ? t->classes[v].total.sum : t->classes[v].cur.sum;
      uint32_t j = i;
      while (j > 0 && (final ? t->classes[order[j - 1]].total.sum : t->classes[order[j - 1]].cur.sum) < key) {
         order[j] = order[j - 1];
         j--;
      }
      order[j] = v;
   }

   fprintf(stderr, "  %-16s %-26s %-24s %9s %9s %9s %6s %7s %8s %8s %8s\n", "hash", "class", "route", "draws/s",
           "timed/s", "gpu_ms/s", "mesh%", "submit%", "avg_us", "min_us", "max_us");
   for (uint32_t k = 0; k < t->class_count; k++) {
      struct radv_bc250_timer_class *c = &t->classes[order[k]];
      const struct radv_bc250_timer_stats *s = final ? &c->total : &c->cur;
      const uint64_t draws = s->draws[0] + s->draws[1] + s->draws[2];
      if (!draws && !s->timed && !s->skipped)
         continue;
      char cls[64];
      snprintf(cls, sizeof(cls), "V=%u P=%u LS=%u W%u WG%u", c->vertices, c->primitives, c->local_size, c->wave,
               c->workgroup);
      fprintf(stderr, "  %-16s %-26s %-24s %9.0f %9.0f %9.2f %6.1f %7.1f %8.1f %8.1f %8.1f%s\n", c->hex, cls, c->route,
              draws / span_s, s->timed / span_s, s->sum * ns / 1e6 / span_s,
              tot->mesh_ticks ? 100.0 * s->sum / tot->mesh_ticks : 0.0,
              tot->submit_ticks ? 100.0 * s->sum / tot->submit_ticks : 0.0,
              s->timed ? s->sum * ns / 1e3 / s->timed : 0.0, s->timed ? s->min * ns / 1e3 : 0.0,
              s->timed ? s->max * ns / 1e3 : 0.0, s->skipped ? " (SKIPPED)" : "");
      snprintf(line, sizeof(line),
               "C hash=%s V=%u P=%u LS=%u W=%u WG=%u route=%s draws=%" PRIu64 " direct=%" PRIu64 " indirect=%" PRIu64
               " count=%" PRIu64 " timed=%" PRIu64 " sum_ns=%.0f min_ns=%.0f max_ns=%.0f skipped=%" PRIu64 "\n",
               c->hex, c->vertices, c->primitives, c->local_size, c->wave, c->workgroup, c->route, draws, s->draws[0],
               s->draws[1], s->draws[2], s->timed, s->sum * ns, s->timed ? s->min * ns : 0.0, s->max * ns,
               s->skipped);
      file_line(t, line);
      if (t->safe_stats && c->safe_direct) {
         const uint64_t *d = final ? c->safe_stats_total : c->safe_stats_cur;
         snprintf(line, sizeof(line),
                  "D hash=%s clean=%" PRIu64 " r1_fail=%" PRIu64 " r2_fail=%" PRIu64
                  " repaired=%" PRIu64 " check_cycles=%" PRIu64 " planner_cycles=%" PRIu64
                  " export_cycles=%" PRIu64 "\n",
                  c->hex, d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
         fputs(line, stderr);
         file_line(t, line);
      }
      if (!final) {
         memset(&c->cur, 0, sizeof(c->cur));
         c->cur.min = UINT64_MAX;
         memset(c->safe_stats_cur, 0, sizeof(c->safe_stats_cur));
      }
   }
   file_sync(t);
   if (!final) {
      memset(&t->cur, 0, sizeof(t->cur));
      t->last_print_ns = now;
   }
}

/* "<sync>:<value>,..." (at most 8, then "+<n>"), "-" when empty. */
static void
sem_list(char *buf, size_t size, const struct vk_sync_wait *waits, const struct vk_sync_signal *signals, uint32_t count)
{
   size_t o = 0;
   buf[0] = '-';
   buf[1] = 0;
   for (uint32_t i = 0; i < count && i < 8; i++) {
      const void *sync = waits ? (const void *)waits[i].sync : (const void *)signals[i].sync;
      const uint64_t value = waits ? waits[i].wait_value : signals[i].signal_value;
      const int n = snprintf(buf + o, size - o, "%s%" PRIxPTR ":%" PRIu64, i ? "," : "", (uintptr_t)sync, value);
      if (n < 0 || (size_t)n >= size - o)
         return;
      o += n;
   }
   if (count > 8)
      snprintf(buf + o, size - o, ",+%u", count - 8);
}

void
radv_bc250_timer_submit(struct radv_device *device, const struct radv_queue *queue,
                        const struct vk_queue_submit *submission, uint64_t call_ns)
{
   struct radv_bc250_mesh_timer *t = &device->bc250_timer;
   const uint64_t now = os_time_get_nano();
   const unsigned qf = queue->state.qf;

   simple_mtx_lock(&t->mtx);
   t->seq++;
   if (t->per_submit_lines && t->file && t->async) {
      struct radv_bc250_timer_wrec *r = wrec_push(t);
      if (r) {
         r->kind = 'W';
         r->seq = t->seq;
         r->call_ns = call_ns;
         r->wall_ns = now;
         r->fam = queue->vk.queue_family_index;
         r->idx = queue->vk.index_in_family;
         r->qf = qf;
         r->thr = queue->vk.submit.mode == VK_QUEUE_SUBMIT_MODE_THREADED;
         r->ncb = submission->command_buffer_count;
         r->nw = submission->wait_count;
         r->nsig = submission->signal_count;
         for (uint32_t i = 0; i < r->nw && i < 8; i++) {
            r->wsync[i] = submission->waits[i].sync;
            r->wval[i] = submission->waits[i].wait_value;
         }
         for (uint32_t i = 0; i < r->nsig && i < 8; i++) {
            r->ssync[i] = submission->signals[i].sync;
            r->sval[i] = submission->signals[i].signal_value;
         }
      }
   } else if (t->per_submit_lines && t->file) {
      /* W: every vkQueueSubmit of every queue (also without command buffers), at submission time.
       * call_ns: the driver's submit entry, wall_ns: after the kernel submission returned.
       * w / s: wait / signal semaphores as <vk_sync address>:<value> (timeline value, 0 = binary);
       * thr: the queue runs the vk_queue submit thread (wait-before-signal seen). */
      char waits[256], signals[256], line[768];
      sem_list(waits, sizeof(waits), submission->waits, NULL, submission->wait_count);
      sem_list(signals, sizeof(signals), NULL, submission->signals, submission->signal_count);
      snprintf(line, sizeof(line),
               "W seq=%" PRIu64 " call_ns=%" PRIu64 " wall_ns=%" PRIu64 " q=%u.%u qf=%s ncb=%u nw=%u nsig=%u thr=%u w=%s s=%s\n",
               t->seq, call_ns, now, queue->vk.queue_family_index, queue->vk.index_in_family, queue_kind(qf),
               submission->command_buffer_count, submission->wait_count, submission->signal_count,
               queue->vk.submit.mode == VK_QUEUE_SUBMIT_MODE_THREADED, waits, signals);
      file_line(t, line);
   }
   for (uint32_t j = 0; j < submission->command_buffer_count; j++) {
      struct radv_cmd_buffer *cb = (struct radv_cmd_buffer *)submission->command_buffers[j];
      const struct radv_bc250_timer_cmd *tc = &cb->bc250_timer;
      if (cb->qf == RADV_QUEUE_COMPUTE ? !tc->has_cb : (cb->qf != RADV_QUEUE_GENERAL || (!tc->has_cb && !tc->count)))
         continue;
      if (t->pending_count == RADV_BC250_TIMER_MAX_PENDING)
         drop_oldest_pending(t);
      struct radv_bc250_timer_pending *p =
         &t->pending[(t->pending_head + t->pending_count) % RADV_BC250_TIMER_MAX_PENDING];
      memset(p, 0, sizeof(*p));
      p->cb_idx = tc->cb_idx;
      p->has_cb = tc->has_cb;
      p->count = tc->count;
      if (tc->count) {
         p->entries = malloc(tc->count * sizeof(*p->entries));
         if (!p->entries)
            p->count = 0;
         else
            memcpy(p->entries, tc->entries, tc->count * sizeof(*p->entries));
      }
      p->wall_ns = now;
      p->seq = t->seq;
      p->mesh_draws = tc->draws;
      p->vs_draws = tc->vs_draws;
      p->skipped = tc->skipped;
      p->vk_family = queue->vk.queue_family_index;
      p->vk_index = queue->vk.index_in_family;
      p->cb_i = j;
      p->cb_n = submission->command_buffer_count;
      p->qf = cb->qf;
      if (t->fake)
         fake_fill(t, p);
      else if (p->has_cb)
         __atomic_store_n(&t->map[(p->cb_idx & SLOT_MASK) * 2 + 1], 0, __ATOMIC_RELEASE); /* end marker */
      t->pending_count++;
   }

   if (t->async) {
      /* BC250_MESH_TIMER_ASYNC: the worker folds, prints and writes. */
      simple_mtx_unlock(&t->mtx);
      return;
   }

   /* Fold every completed submission in order (no wait: at most what the GPU has finished). */
   while (t->pending_count && pending_complete(t, &t->pending[t->pending_head])) {
      struct radv_bc250_timer_pending *p = &t->pending[t->pending_head];
      fold_pending(t, p);
      free(p->entries);
      p->entries = NULL;
      t->pending_head = (t->pending_head + 1) % RADV_BC250_TIMER_MAX_PENDING;
      t->pending_count--;
   }

   const uint64_t frames = __atomic_exchange_n(&t->frames, 0, __ATOMIC_RELAXED);
   t->cur.frames += frames;
   t->total.frames += frames;

   if (t->cur.submits && (now - t->last_print_ns) / 1e9 >= t->interval_s)
      radv_bc250_timer_print(device, false);
   simple_mtx_unlock(&t->mtx);
}

/* ---- BC250_MESH_TIMER_ASYNC worker ------------------------------------- */

/* "<sync>:<value>,..." from a record (the format of sem_list). */
static void
wrec_sem_list(char *buf, size_t size, const void *const *syncs, const uint64_t *values, uint32_t count)
{
   size_t o = 0;
   buf[0] = '-';
   buf[1] = 0;
   for (uint32_t i = 0; i < count && i < 8; i++) {
      const int n = snprintf(buf + o, size - o, "%s%" PRIxPTR ":%" PRIu64, i ? "," : "", (uintptr_t)syncs[i], values[i]);
      if (n < 0 || (size_t)n >= size - o)
         return;
      o += n;
   }
   if (count > 8)
      snprintf(buf + o, size - o, ",+%u", count - 8);
}

/* Worker text buffer (outside the mutex). */
struct timer_text {
   char *b;
   size_t len, cap;
};

static void
text_add(struct timer_text *x, const char *line)
{
   const size_t n = strlen(line);
   if (x->len + n > x->cap) {
      size_t cap = x->cap ? x->cap : 1 << 16;
      while (cap < x->len + n)
         cap *= 2;
      char *b = realloc(x->b, cap);
      if (!b)
         return;
      x->b = b;
      x->cap = cap;
   }
   memcpy(x->b + x->len, line, n);
   x->len += n;
}

static void
format_wrecs(struct timer_text *x, const struct radv_bc250_timer_wrec *recs, uint32_t count)
{
   for (uint32_t k = 0; k < count; k++) {
      const struct radv_bc250_timer_wrec *r = &recs[k];
      char line[768];
      if (r->kind == 'P') {
         snprintf(line, sizeof(line), "P seq=%" PRIu64 " wall_ns=%" PRIu64 " q=%u.%u\n", r->seq, r->wall_ns, r->fam,
                  r->idx);
      } else {
         char waits[256], signals[256];
         wrec_sem_list(waits, sizeof(waits), r->wsync, r->wval, r->nw);
         wrec_sem_list(signals, sizeof(signals), r->ssync, r->sval, r->nsig);
         snprintf(line, sizeof(line),
                  "W seq=%" PRIu64 " call_ns=%" PRIu64 " wall_ns=%" PRIu64 " q=%u.%u qf=%s ncb=%u nw=%u nsig=%u thr=%u w=%s s=%s\n",
                  r->seq, r->call_ns, r->wall_ns, r->fam, r->idx, queue_kind(r->qf), r->ncb, r->nw, r->nsig, r->thr,
                  waits, signals);
      }
      text_add(x, line);
   }
}

static void
format_srecs(struct timer_text *x, const struct radv_bc250_timer_srec *recs, uint32_t count, double ns_per_tick)
{
   for (uint32_t k = 0; k < count; k++) {
      const struct radv_bc250_timer_srec *r = &recs[k];
      char line[384];
      snprintf(line, sizeof(line), "S seq=%" PRIu64 " wall_ns=%" PRIu64 " submit_ns=%.0f mesh_ns=%.0f mesh_draws=%u timed=%" PRIu64
               " vs_draws=%u gpu_t0=%" PRIu64 " gpu_t1=%" PRIu64 " q=%u.%u qf=%s cb=%u/%u\n",
               r->seq, r->wall_ns, r->submit_ticks * ns_per_tick, r->mesh_ticks * ns_per_tick, r->mesh_draws, r->timed,
               r->vs_draws, r->gpu_t0, r->gpu_t1, r->fam, r->idx, queue_kind(r->qf), r->cb_i, r->cb_n);
      text_add(x, line);
   }
}

static int
timer_worker(void *arg)
{
   struct radv_bc250_mesh_timer *t = arg;
   struct radv_bc250_timer_wrec *wspare = NULL;
   struct radv_bc250_timer_srec *sspare = NULL;
   uint32_t wspare_cap = 0, sspare_cap = 0;
   char *aspare = NULL;
   size_t aspare_cap = 0;
   struct timer_text text = {0};
   for (;;) {
      os_time_sleep(1000);
      simple_mtx_lock(&t->mtx);
      const bool stop = t->worker_stop;
      /* take the queued W / P records */
      struct radv_bc250_timer_wrec *wrecs = t->wrecs;
      const uint32_t wcount = t->wrec_count, wcap = t->wrec_cap;
      t->wrecs = wspare;
      t->wrec_cap = wspare_cap;
      t->wrec_count = 0;
      /* fold what the GPU finished (S records, arithmetic only) */
      while (t->pending_count && pending_complete(t, &t->pending[t->pending_head])) {
         struct radv_bc250_timer_pending *p = &t->pending[t->pending_head];
         fold_pending(t, p);
         free(p->entries);
         p->entries = NULL;
         t->pending_head = (t->pending_head + 1) % RADV_BC250_TIMER_MAX_PENDING;
         t->pending_count--;
      }
      struct radv_bc250_timer_srec *srecs = t->srecs;
      const uint32_t scount = t->srec_count, scap = t->srec_cap;
      t->srecs = sspare;
      t->srec_cap = sspare_cap;
      t->srec_count = 0;
      const uint64_t frames = __atomic_exchange_n(&t->frames, 0, __ATOMIC_RELAXED);
      t->cur.frames += frames;
      t->total.frames += frames;
      const uint64_t now = os_time_get_nano();
      if (t->cur.submits && (now - t->last_print_ns) / 1e9 >= t->interval_s)
         radv_bc250_timer_print(t->device, false); /* every interval: its lines go to abuf */
      char *abuf = t->abuf;
      const size_t alen = t->abuf_len, acap = t->abuf_cap;
      t->abuf = aspare;
      t->abuf_cap = aspare_cap;
      t->abuf_len = 0;
      const bool sync = t->want_sync;
      t->want_sync = false;
      simple_mtx_unlock(&t->mtx);

      /* format and write outside the mutex: W / P lines, then S lines, then interval lines */
      text.len = 0;
      format_wrecs(&text, wrecs, wcount);
      format_srecs(&text, srecs, scount, t->ns_per_tick);
      if (t->file) {
         if (text.len)
            fwrite(text.b, 1, text.len, t->file);
         if (alen)
            fwrite(abuf, 1, alen, t->file);
         if (sync) {
            fflush(t->file);
            fsync(fileno(t->file));
         }
      }
      wspare = wrecs;
      wspare_cap = wcap;
      sspare = srecs;
      sspare_cap = scap;
      aspare = abuf;
      aspare_cap = acap;
      if (stop)
         break;
   }
   if (t->file)
      fflush(t->file);
   free(wspare);
   free(sspare);
   free(aspare);
   free(text.b);
   return 0;
}

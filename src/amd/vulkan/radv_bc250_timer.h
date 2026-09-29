/*
 * Copyright © 2026 BC-250 LoneWolf project
 * SPDX-License-Identifier: MIT
 */

/* BC250_MESH_TIMER: an in-driver GPU timer for Mesh draws (GFX1013 / BC-250).
 *
 * Runtime-only diagnostic (no compiler-key change). With BC250_MESH_TIMER=1
 * every top-level Mesh draw (vkCmdDrawMeshTasksEXT / Indirect2EXT /
 * IndirectCount2EXT, the outermost call, so the base driver's split pieces, hybrid-Task
 * producer + consumer chunks and prep dispatches are inside the interval) is
 * bracketed by two bottom-of-pipe GPU timestamps (RELEASE_MEM, the packet
 * vkCmdWriteTimestamp2 uses) written into a driver-owned ring buffer, and
 * every primary command buffer gets a begin/end pair. The CPU tags each
 * slot with the bound Mesh shader's class (hash, V/P, local size, wave,
 * workgroup, route) and folds completed submissions into per-class
 * accumulators, printed every BC250_MESH_TIMER_INTERVAL seconds (stderr and
 * BC250_MESH_TIMER_FILE). Per-draw intervals are inclusive of pipeline
 * overlap with neighbouring work, so sums per class are a relative measure.
 * With the switch off the driver records byte-identical command streams.
 *
 * Environment:
 *   BC250_MESH_TIMER=1              enable
 *   RADV_BC250_MESH_SAFE_STATS=1    with SAFE_FAST: per-draw check/repair counts and shader-clock deltas
 *   BC250_MESH_TIMER_INTERVAL=<s>   print period (default 5; 0 = every submit)
 *   BC250_MESH_TIMER_FILE=<path>    append the tables (fsync after each write)
 *   BC250_MESH_TIMER_SAMPLE=<n>     time every n-th Mesh draw (default 1)
 *   BC250_MESH_TIMER_SKIP=<list>    skip the matching Mesh draws entirely: "all",
 *                                   hash prefixes, or class words split / expanded /
 *                                   raw / replay / merged / amd / autocull / compact
 *   BC250_MESH_TIMER_SUBMITS=1      also log per-submission lines (async-compute timelines):
 *                                   "W" at vkQueueSubmit (every queue, also command-buffer-less
 *                                   semaphore submissions: queue family.index, kind, call / return
 *                                   CPU time, command buffers, wait / signal semaphores with
 *                                   values, vk_queue submit-thread flag), "S" per completed
 *                                   graphics or compute command buffer (GPU begin / end ticks,
 *                                   queue), "P" per present, "T" clock calibration (GPU tick,
 *                                   CLOCK_MONOTONIC and CLOCK_MONOTONIC_RAW for vkd3d traces)
 *   BC250_MESH_TIMER_COMPUTE=0      do not bracket compute-queue command buffers (default: every
 *                                   primary command buffer of the graphics and the compute queues
 *                                   gets the begin / end pair; Mesh draws exist on graphics only)
 *   BC250_MESH_TIMER_FAKE=1         tests only: synthesise the timestamps on the CPU
 *                                   at submit time (drm-shim runs never write them)
 *   BC250_MESH_TIMER_ASYNC=1        keep the submitting threads cheap: vkQueueSubmit / present
 *                                   only queue raw W / P records and the command buffers' pending
 *                                   entries; a worker thread (1 ms period) formats the lines, folds
 *                                   the completed command buffers (S lines), prints the intervals,
 *                                   writes the file and fsyncs outside the lock. Same lines as the
 *                                   default mode; interval prints follow the worker, not a submit.
 */

#ifndef RADV_BC250_TIMER_H
#define RADV_BC250_TIMER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "util/simple_mtx.h"
#include "c11/threads.h"

struct radv_device;
struct radv_queue;
struct radv_cmd_buffer;
struct radv_shader;
struct radeon_winsys_bo;
struct vk_queue_submit;

#define RADV_BC250_TIMER_SLOTS (64u * 1024u) /* 16 bytes each: t0, t1 */
#define RADV_BC250_TIMER_MAX_CLASSES 1024
#define RADV_BC250_TIMER_MAX_PENDING 4096 /* command buffers in flight (all queues) */
#define RADV_BC250_SAFE_STATS_DWORDS 7

enum radv_bc250_timer_kind {
   RADV_BC250_TIMER_DIRECT,
   RADV_BC250_TIMER_INDIRECT,
   RADV_BC250_TIMER_COUNT,
   RADV_BC250_TIMER_KINDS,
};

struct radv_bc250_timer_stats {
   uint64_t draws[RADV_BC250_TIMER_KINDS]; /* all top-level draws, timed or not */
   uint64_t timed;                          /* slots folded with a valid interval */
   uint64_t sum;                            /* ticks */
   uint64_t min, max;                       /* ticks */
   uint64_t skipped;                        /* BC250_MESH_TIMER_SKIP */
};

struct radv_bc250_timer_class {
   const struct radv_shader *shader;
   uint8_t hash[8];
   char hex[17];
   uint16_t vertices, primitives, local_size, workgroup;
   uint8_t wave, pieces, merge_k;
   bool expanded, replay, amd, autocull, compact, safe_direct;
   char route[32];
   bool skip, skip_printed;
   struct radv_bc250_timer_stats cur, total;
   uint64_t safe_stats_cur[RADV_BC250_SAFE_STATS_DWORDS];
   uint64_t safe_stats_total[RADV_BC250_SAFE_STATS_DWORDS];
};

struct radv_bc250_timer_entry {
   uint64_t idx; /* ring allocation index; slot = idx % SLOTS */
   uint16_t class_id;
   uint8_t kind;
};

struct radv_bc250_timer_pending {
   uint64_t cb_idx;
   bool has_cb;
   struct radv_bc250_timer_entry *entries;
   uint32_t count;
   uint64_t wall_ns, seq;
   uint32_t mesh_draws, vs_draws, skipped;
   uint16_t vk_family, vk_index; /* the queue (Vulkan family and index) */
   uint16_t cb_i, cb_n;          /* command buffer i of n in the submission */
   uint8_t qf;                   /* enum radv_queue_family */
};

/* BC250_MESH_TIMER_ASYNC: a W (submission) or P (present) record, formatted by the worker. */
struct radv_bc250_timer_wrec {
   uint64_t seq, call_ns, wall_ns;
   uint16_t fam, idx;
   uint8_t kind; /* 'W' or 'P' */
   uint8_t qf, thr;
   uint32_t ncb, nw, nsig;
   const void *wsync[8], *ssync[8];
   uint64_t wval[8], sval[8];
};

/* BC250_MESH_TIMER_ASYNC: a folded command buffer (S line), formatted by the worker. */
struct radv_bc250_timer_srec {
   uint64_t seq, wall_ns, submit_ticks, mesh_ticks, timed, gpu_t0, gpu_t1;
   uint32_t mesh_draws, vs_draws;
   uint16_t fam, idx, cb_i, cb_n;
   uint8_t qf;
};

struct radv_bc250_timer_totals {
   uint64_t submits, frames, submit_ticks, mesh_ticks, mesh_draws, timed, vs_draws, skipped;
   uint64_t dropped_slots, dropped_submits, invalid;
   uint64_t ace_submits, ace_ticks; /* compute-queue command buffers (not in submits / submit_ticks) */
};

struct radv_bc250_mesh_timer {
   bool enabled;
   bool fake;
   bool per_submit_lines;
   bool compute; /* bracket compute-queue command buffers too */
   bool safe_stats;
   uint32_t sample;
   double interval_s;
   double ns_per_tick;

   struct radeon_winsys_bo *bo;
   uint64_t va;
   uint64_t *map; /* SLOTS * 2 */
   struct radeon_winsys_bo *safe_stats_bo;
   uint64_t safe_stats_va;
   uint32_t *safe_stats_map; /* SLOTS * RADV_BC250_SAFE_STATS_DWORDS */
   uint64_t next;           /* atomic ring allocation index */
   uint64_t sample_counter; /* atomic */
   uint64_t fake_clock;

   simple_mtx_t mtx; /* classes, pending, totals, output */
   struct radv_bc250_timer_class *classes;
   uint32_t class_count;
   struct radv_bc250_timer_pending *pending; /* RADV_BC250_TIMER_MAX_PENDING */
   uint32_t pending_head, pending_count;
   uint64_t seq, frames;
   uint64_t start_ns, last_print_ns;
   struct radv_bc250_timer_totals cur, total;
   FILE *file;
   char skip_spec[256];

   /* BC250_MESH_TIMER_ASYNC (mtx protects everything but the worker's file writes) */
   bool async;
   bool worker_on;   /* file_line appends to abuf, file_sync sets want_sync */
   bool worker_stop;
   bool want_sync;
   thrd_t worker;
   struct radv_device *device;
   struct radv_bc250_timer_wrec *wrecs;
   uint32_t wrec_count, wrec_cap;
   struct radv_bc250_timer_srec *srecs;
   uint32_t srec_count, srec_cap;
   char *abuf;
   size_t abuf_len, abuf_cap;
};

/* Per command buffer. */
struct radv_bc250_timer_cmd {
   struct radv_bc250_timer_entry *entries;
   uint32_t count, cap;
   uint64_t cb_idx;
   bool has_cb;
   const struct radv_shader *last_ms;
   uint32_t last_class;
   uint32_t draws, vs_draws, skipped; /* top-level Mesh draws (skipped ones included), VS draws, skipped */
};

void radv_bc250_timer_init(struct radv_device *device);
void radv_bc250_timer_finish(struct radv_device *device);

void radv_bc250_timer_cmd_reset(struct radv_cmd_buffer *cmd_buffer);
void radv_bc250_timer_cmd_destroy(struct radv_cmd_buffer *cmd_buffer);
void radv_bc250_timer_cmd_begin(struct radv_cmd_buffer *cmd_buffer);
void radv_bc250_timer_cmd_end(struct radv_cmd_buffer *cmd_buffer);
void radv_bc250_timer_cmd_execute(struct radv_cmd_buffer *primary, struct radv_cmd_buffer *secondary);

/* Top-level Mesh draw bracket. Returns RADV_BC250_TIMER_SKIP_DRAW (the caller
 * records nothing), RADV_BC250_TIMER_NOT_TIMED (counted, no packets) or the
 * entry index to pass to radv_bc250_timer_draw_end. */
#define RADV_BC250_TIMER_SKIP_DRAW (-2)
#define RADV_BC250_TIMER_NOT_TIMED (-1)
int32_t radv_bc250_timer_draw_begin(struct radv_cmd_buffer *cmd_buffer, enum radv_bc250_timer_kind kind);
void radv_bc250_timer_draw_end(struct radv_cmd_buffer *cmd_buffer, int32_t entry);

/* call_ns: CLOCK_MONOTONIC when the driver's queue submit entry point was reached. */
void radv_bc250_timer_submit(struct radv_device *device, const struct radv_queue *queue,
                             const struct vk_queue_submit *submission, uint64_t call_ns);
void radv_bc250_timer_present(struct radv_device *device, const struct radv_queue *queue);

#endif /* RADV_BC250_TIMER_H */

/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_DEVICE_H
#define RADV_DEVICE_H

#include "ac_spm.h"
#include "ac_sqtt.h"

#include "util/mesa-blake3.h"
#include "util/u_queue.h"

#include "tools/radv_debug.h"
#include "tools/radv_debug_nir.h"
#include "tools/radv_rra.h"

#include "radv_pipeline.h"
#include "radv_queue.h"
#include "radv_radeon_winsys.h"
#include "radv_shader.h"
#include "radv_bc250_timer.h"

#include "vk_acceleration_structure.h"
#include "vk_device.h"
#include "vk_meta.h"
#include "vk_texcompress_astc.h"
#include "vk_texcompress_etc2.h"

#define RADV_NUM_HW_CTX (RADEON_CTX_PRIORITY_REALTIME + 1)

struct radv_image_view;
struct radv_cmd_stream;

enum radv_dispatch_table {
   RADV_DEVICE_DISPATCH_TABLE,
   RADV_ANNOTATE_DISPATCH_TABLE,
   RADV_APP_DISPATCH_TABLE,
   RADV_RGP_DISPATCH_TABLE,
   RADV_RRA_DISPATCH_TABLE,
   RADV_RMV_DISPATCH_TABLE,
   RADV_UTRACE_DISPATCH_TABLE,
   RADV_CTX_ROLL_DISPATCH_TABLE,
   RADV_DISPATCH_TABLE_COUNT,
};

struct radv_layer_dispatch_tables {
   struct vk_device_dispatch_table annotate;
   struct vk_device_dispatch_table app;
   struct vk_device_dispatch_table rgp;
   struct vk_device_dispatch_table rra;
   struct vk_device_dispatch_table rmv;
   struct vk_device_dispatch_table utrace;
   struct vk_device_dispatch_table ctx_roll;
};

enum radv_force_vrs {
   RADV_FORCE_VRS_1x1 = 0,
   RADV_FORCE_VRS_2x2,
   RADV_FORCE_VRS_2x1,
   RADV_FORCE_VRS_1x2,
};

struct radv_notifier {
   int fd;
   int watch;
   bool quit;
   thrd_t thread;
};

struct radv_meta_state {
   VkAllocationCallbacks alloc;

   VkPipelineCache cache;
   uint32_t initial_cache_entries;

   /*
    * For on-demand pipeline creation, makes sure that
    * only one thread tries to build a pipeline at the same time.
    */
   mtx_t mtx;

   struct {
      struct radix_sort_vk *radix_sort_64;
      struct radix_sort_vk *radix_sort_96;
      struct vk_acceleration_structure_build_ops build_ops;
      struct vk_acceleration_structure_build_args build_args;
   } accel_struct_build;

   struct vk_texcompress_etc2_state etc_decode;

   struct vk_texcompress_astc_state *astc_decode;

   struct vk_meta_device device;
};

struct radv_memory_trace_data {
   /* ID of the PTE update event in ftrace data */
   uint16_t ftrace_update_ptes_id;

   uint32_t num_cpus;
   int *pipe_fds;
};

struct radv_sqtt_timestamp {
   uint8_t *map;
   unsigned offset;
   uint64_t size;
   struct radeon_winsys_bo *bo;
   struct list_head list;
};

#define RADV_BORDER_COLOR_COUNT       4096
#define RADV_BORDER_COLOR_BUFFER_SIZE (sizeof(VkClearColorValue) * RADV_BORDER_COLOR_COUNT)

struct radv_device_border_color_data {
   bool used[RADV_BORDER_COLOR_COUNT];

   struct radeon_winsys_bo *bo;
   VkClearColorValue *colors_gpu_ptr;

   /* Mutex is required to guarantee vkCreateSampler thread safety
    * given that we are writing to a buffer and checking color occupation */
   mtx_t mutex;
};

struct radv_pso_cache_stats {
   uint32_t hits;
   uint32_t misses;
};

struct radv_shader_abort_data {
   uint32_t buffer_size;
   struct radv_backed_buffer buffer;
   VkDeviceAddress buffer_addr;
};

/* BC250 switches the draw, dispatch and submit paths consult, read once by
 * radv_bc250_device_env_init() right after vk_device_init() instead of by a
 * getenv() per call (getenv scans the whole environment; a Proton game has
 * ~150 variables). Same values as the per-call reads as long as the process
 * environment does not change after device creation. */
struct radv_bc250_device_env {
   bool gpl_source_link; /* RADV_BC250_GPL_SOURCE_LINK: conservative final-source linking. */
   bool gpl_binary_link; /* RADV_BC250_GPL_BINARY_LINK: complete executable libraries. */
   bool shader_object_plan; /* RADV_BC250_SHADER_OBJECT_PLAN: linked executable ownership. */
   bool pipeline_plan; /* RADV_BC250_PIPELINE_PLAN: opt-in portable cache plan. */
   bool chain_trace;          /* BC250_CHAIN_TRACE on GFX1013 (radv_bc250_chain_enabled) */
   bool chain_shader_only;    /* BC250_CHAIN_SHADER_ONLY */
   bool chain_arguments_only; /* BC250_CHAIN_ARGUMENTS_ONLY */
   bool chain_sample_output;  /* BC250_CHAIN_SAMPLE_OUTPUT */
   bool transient_arena;      /* BC250_TRANSIENT_ARENA */
   bool trace_compile;        /* BC250_TRACE_COMPILE is set (any value) */
   bool trace_regs;           /* BC250_TRACE_REGS is set */
   bool trace_usage;          /* BC250_TRACE_USAGE is set */
   bool omit_launch;          /* BC250_DIAGNOSTIC_OMIT_LAUNCH is set */
   bool post_mesh_vgt_flush;  /* BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH */
   bool skip_inactive_chunks; /* BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS */
   bool no_split_l2_inv;      /* RADV_BC250_PERF_NO_SPLIT_L2_INV */
   bool split_batch_trace;    /* RADV_BC250_SPLIT_BATCH_TRACE */
   bool merge_prep_l2_inv;    /* RADV_BC250_MESH_MERGE_PREP_L2_INV */
   bool native_task_direct;   /* BC250_NATIVE_TASK_DIRECT || RADV_BC250_NATIVE_TASK */
   bool native_task_indirect; /* BC250_NATIVE_TASK_INDIRECT || RADV_BC250_NATIVE_TASK */
   /* RADV_BC250_SCRATCH_REUSE=1: split indirect outputs and batch lists come
    * from the command buffer's upload buffer, other BC250 scratch from the
    * transient arena also without ONE_TIME_SUBMIT (bc250_alloc_scratch). */
   bool scratch_reuse;
   /* RADV_BC250_SPLIT_LEAN_SETUP=1: the per-draw split argument setup binds,
    * pushes and restores only what it touches (bc250_split_setup_dispatch). */
   bool split_lean_setup;
   /* RADV_BC250_SPLIT_PREP_FREE=1: order-independent split indirect draws are
    * drawn as one indirect draw per piece, without the setup dispatch
    * (bc250_draw_split_prep_free); the shaders carry the piece select. */
   bool split_prep_free;
   bool mesh_no_split; /* Coverage reporting only; never disables the fallback. */
   /* RADV_BC250_LOCAL_BOS=1: the device's winsys is created with RADV_PERFTEST=localbos: driver-internal
    * buffers (command buffer IBs, upload buffers, shader arenas, query pools, rings) become
    * VM_ALWAYS_VALID kernel objects and leave the per-submission BO list (radv_create_winsys). */
   bool local_bos;
   /* RADV_BC250_SUBMIT_PROFILE=1: per-queue submission cost (runtime, driver, BO list, CS ioctl,
    * timer, present) printed every 5 s (radv_bc250_submit_profile_*). */
   bool submit_profile;
};

#define RADV_BC250_MESH_VERT_GRP_CLAMP 1
#define RADV_BC250_MESH_VERT_GRP_OFF   2

struct radv_device {
   struct vk_device vk;
   struct radv_bc250_device_env bc250_env;
   /* RADV_BC250_ASYNC_COMPILE: background queue for the optimized Mesh/FS binaries of pipelines that were
    * created with an unoptimized ACO build first (radv_pipeline_graphics.c). */
   bool bc250_async;
   struct util_queue bc250_async_queue;
   /* DGC query-state certificates, protected by meta_state.mtx. */
   struct hash_table_u64 *bc250_dgc_query_states;
   uint64_t bc250_trace_device_id; /* Process-local diagnostic generation. */
   struct radv_bc250_mesh_timer bc250_timer; /* BC250_MESH_TIMER (radv_bc250_timer.h) */
   /* RADV_BC250_COMPUTE_QUEUE_PRIORITY (radv_queue.c): the kernel context priority of compute-family
    * queues created without a global priority; RADEON_CTX_PRIORITY_INVALID = off. */
   enum radeon_ctx_priority bc250_compute_priority;
   /* RADV_BC250_SUBMIT_PROFILE (radv_queue.c): print state; RADV_BC250_SUBMIT_PROFILE_FILE copy. */
   simple_mtx_t bc250_prof_mtx;
   uint64_t bc250_prof_start_ns, bc250_prof_last_ns;
   uint64_t bc250_prof_bo_last[8], bc250_prof_presents_last;
   FILE *bc250_prof_file;
   /* RADV_BC250_SPLIT_BATCH_PREP=1: one split-argument setup dispatch per
    * render pass instance instead of one per split indirect draw. */
   bool bc250_split_batch_prep;
   /* RADV_BC250_MESH_DEALLOC_DIST=N (GFX1013, 1..127): VGT_OUT_DEALLOC_CNTL.DEALLOC_DIST
    * while a Mesh shader is bound, 32 (the CLEAR_STATE value) again for other graphics
    * shaders and at the end of the command buffer. 0 = unset: the register is never written. */
   uint32_t bc250_mesh_dealloc_dist;
   /* RADV_BC250_MESH_VERT_GRP (GFX1013): GE_CNTL.VERT_GRP_SIZE of Mesh draws. Unset (0) keeps
    * ES_VERTS_PER_SUBGRP (1 for every BC250 Mesh route), which AMD's PAL lists as illegal on
    * GFX10.1 without fast launch. "clamp" applies the ordinary GFX10 rule used for the other NGG
    * stages (ES_VERTS_PER_SUBGRP - 5, 0 when that is not positive), "off" disables vertex
    * grouping (256, PAL's default on GFX10). */
   uint32_t bc250_mesh_vert_grp;

   struct radeon_winsys *ws;

   struct radv_layer_dispatch_tables layer_dispatch;

   struct radeon_winsys_ctx *hw_ctx[RADV_NUM_HW_CTX];
   struct radeon_winsys_ctx *hw_vcn_enc_ctx;

   struct radv_meta_state meta_state;

   struct radv_queue *queues[RADV_MAX_QUEUE_FAMILIES];
   struct radv_queue *queues_protected[RADV_MAX_QUEUE_FAMILIES];
   int queue_count[RADV_MAX_QUEUE_FAMILIES];
   int queue_count_protected[RADV_MAX_QUEUE_FAMILIES];

   bool pbb_allowed;
   uint32_t scratch_waves;
   uint32_t dispatch_initiator;
   uint32_t dispatch_initiator_task;

   /* MSAA sample locations.
    * The first index is the sample index.
    * The second index is the coordinate: X, Y. */
   float sample_locations_1x[1][2];
   float sample_locations_2x[2][2];
   float sample_locations_4x[4][2];
   float sample_locations_8x[8][2];

   /* GFX7 and later */
   uint32_t gfx_init_size_dw;
   struct radeon_winsys_bo *gfx_init;
   struct radeon_winsys_bo *zero_bo;

   struct radeon_winsys_bo *trace_bo;
   struct radv_trace_data *trace_data;

   /* Whether to keep shader debug info, for debugging. */
   bool keep_shader_info;

   /* Backup in-memory cache to be used if the app doesn't provide one */
   struct vk_pipeline_cache *mem_cache;

   struct list_head shader_arenas;
   struct hash_table_u64 *capture_replay_arena_vas;
   unsigned shader_arena_shift;
   uint8_t shader_free_list_mask;
   struct radv_shader_free_list shader_free_list;
   struct radv_shader_free_list capture_replay_free_list;
   struct list_head shader_block_obj_pool;
   mtx_t shader_arena_mutex;

   mtx_t shader_upload_hw_ctx_mutex;
   struct radeon_winsys_ctx *shader_upload_hw_ctx;
   VkSemaphore shader_upload_sem;
   uint64_t shader_upload_seq;
   struct list_head shader_dma_submissions;
   mtx_t shader_dma_submission_list_mutex;
   cnd_t shader_dma_submission_list_cond;

   /* Whether to DMA shaders to invisible VRAM or to upload directly through BAR. */
   bool shader_use_invisible_vram;

   /* Whether anisotropy is forced with RADV_TEX_ANISO (-1 is disabled). */
   int force_aniso;

   /* Always disable TRUNC_COORD. */
   bool disable_trunc_coord;

   struct radv_device_border_color_data border_color_data;

   /* Thread trace. */
   struct ac_sqtt sqtt;
   bool sqtt_enabled;
   bool sqtt_triggered;

   VkCommandBuffer sqtt_start_cmdbuf[2];
   VkCommandBuffer sqtt_stop_cmdbuf[2];

   uint64_t sqtt_size;
   struct radv_backed_buffer sqtt_buffer;
   struct radv_backed_buffer sqtt_staging_buffer;

   /* SQTT timestamps for queue events. */
   simple_mtx_t sqtt_timestamp_mtx;
   struct radv_sqtt_timestamp sqtt_timestamp;

   /* SQTT timed cmd buffers. */
   simple_mtx_t sqtt_command_pool_mtx;
   struct vk_command_pool *sqtt_command_pool[2];

   /* Whether to use a staging buffer for SQTT/SPM buffers. */
   bool rgp_use_staging_buffer;

   /* Count the number of submits for per-submit RGP captures. */
   uint32_t rgp_num_submits;

   /* Memory trace. */
   struct radv_memory_trace_data memory_trace;

   /* SPM. */
   struct ac_spm spm;
   struct ac_spm_user_config *spm_user_config;

   struct radv_backed_buffer spm_buffer;
   struct radv_backed_buffer spm_staging_buffer;

   /* Radeon Raytracing Analyzer trace. */
   struct radv_rra_trace_data rra_trace;

   FILE *ctx_roll_file;
   simple_mtx_t ctx_roll_mtx;

   /* Trap handler. */
   struct radv_shader *trap_handler_shader;
   struct radeon_winsys_bo *tma_bo; /* Trap Memory Address */
   uint32_t *tma_ptr;

   /* Overallocation. */
   bool overallocation_disallowed;
   uint64_t allocated_memory_size[VK_MAX_MEMORY_HEAPS];
   mtx_t overallocation_mutex;

   /* RADV_FORCE_VRS. */
   struct radv_notifier notifier;
   enum radv_force_vrs force_vrs;

   /* Depth image for VRS when not bound by the app. */
   struct {
      struct radv_image *image;
      struct radv_buffer *buffer; /* HTILE */
      struct radv_device_memory *mem;
   } vrs;

   /* Prime blit sdma queue */
   struct radv_queue *private_sdma_queue;

   struct radv_shader_part_cache vs_prologs;
   struct radv_shader_part *simple_vs_prologs[MAX_VERTEX_ATTRIBS];
   struct radv_shader_part *instance_rate_vs_prologs[816];

   struct radv_shader_part_cache ps_epilogs;

   simple_mtx_t trace_mtx;

   /* Whether per-vertex VRS is forced. */
   bool force_vrs_enabled;

   simple_mtx_t pstate_mtx;
   unsigned pstate_cnt;

   /* BO to contain some performance counter helpers:
    * - A lock for profiling cmdbuffers.
    * - a temporary fence for the end query synchronization.
    * - the pass to use for profiling. (as an array of bools)
    */
   struct radeon_winsys_bo *perf_counter_bo;

   /* Interleaved lock/unlock commandbuffers for perfcounter passes. */
   struct radv_cmd_stream **perf_counter_lock_cs;

   bool uses_shadow_regs;

   struct hash_table *rt_handles;
   simple_mtx_t rt_handles_mtx;

   struct radv_debug_nir debug_nir;

   blake3_hash cache_hash;

   /* Not NULL if a GPU hang report has been generated for VK_EXT_device_fault. */
   char *gpu_hang_report;

   /* PSO cache stats */
   simple_mtx_t pso_cache_stats_mtx;
   struct radv_pso_cache_stats pso_cache_stats[RADV_PIPELINE_TYPE_COUNT];

   simple_mtx_t blit_queue_mtx;

   struct radv_address_binding_tracker *addr_binding_tracker;

   struct radv_compiler_info compiler_info;

   struct radv_shader_abort_data shader_abort;

   struct {
      struct u_trace_context *context;
      simple_mtx_t lock;
   } utrace;
};

VK_DEFINE_HANDLE_CASTS(radv_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct radv_physical_device *
radv_device_physical(const struct radv_device *dev)
{
   return (struct radv_physical_device *)dev->vk.physical;
}

static inline bool
radv_uses_primitives_generated_query(const struct radv_device *device)
{
   return device->vk.enabled_features.primitivesGeneratedQuery ||
          device->vk.enabled_features.primitivesGeneratedQueryWithRasterizerDiscard ||
          device->vk.enabled_features.primitivesGeneratedQueryWithNonZeroStreams;
}

static inline bool
radv_uses_image_float32_atomics(const struct radv_device *device)
{
   return device->vk.enabled_features.shaderImageFloat32Atomics ||
          device->vk.enabled_features.sparseImageFloat32Atomics ||
          device->vk.enabled_features.shaderImageFloat32AtomicMinMax ||
          device->vk.enabled_features.sparseImageFloat32AtomicMinMax;
}

VkResult radv_device_init_vrs_state(struct radv_device *device);

unsigned radv_get_default_max_sample_dist(int log_samples);

void radv_emit_default_sample_locations(const struct radv_physical_device *pdev, struct radv_cmd_stream *cs,
                                        int nr_samples);

void radv_gfx11_set_db_render_control(const struct radv_device *device, unsigned num_samples,
                                      unsigned *db_render_control);

bool radv_device_set_pstate(struct radv_device *device, bool enable);

bool radv_device_acquire_performance_counters(struct radv_device *device);

void radv_device_release_performance_counters(struct radv_device *device);

bool radv_device_should_clear_vram(const struct radv_device *device);

VkResult radv_device_init_utrace(struct radv_device *device);

void radv_device_finish_utrace(struct radv_device *device);

#endif /* RADV_DEVICE_H */

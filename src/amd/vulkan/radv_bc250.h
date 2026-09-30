/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_H
#define RADV_BC250_H
#include "vulkan/vulkan_core.h"
#include <stdbool.h>
struct radv_device;
struct radv_graphics_pipeline;
struct radv_graphics_pipeline_state;
struct radv_cmd_buffer;
VkResult radv_bc250_prepare_task(struct radv_device *device,
                                struct radv_graphics_pipeline *pipeline,
                                const struct radv_graphics_pipeline_state *gfx_state);
void radv_bc250_draw_task(struct radv_cmd_buffer *cmd_buffer,
                         uint32_t x, uint32_t y, uint32_t z);
void radv_bc250_draw_task_indirect(struct radv_cmd_buffer *cmd_buffer, uint64_t address,
                                  uint32_t draw_count, uint32_t stride, uint64_t count_address);
void radv_bc250_draw_split(struct radv_cmd_buffer *cmd_buffer, uint32_t x, uint32_t y, uint32_t z);
void radv_bc250_draw_split_indirect(struct radv_cmd_buffer *cmd_buffer, uint64_t input,
                                   uint32_t records, uint32_t stride, uint64_t count);
/* RADV_BC250_SPLIT_BATCH_PREP: ends the open split-argument batch. Called at
 * every rendering begin/end, barrier, conditional rendering begin/end,
 * vkCmdExecuteCommands and command buffer begin/reset. */
void radv_bc250_split_batch_close(struct radv_cmd_buffer *cmd_buffer);
struct nir_shader;
struct radv_compiler_info;
struct radv_shader_stage;
/* RADV_BC250_MESH_DIRECT_READ parts (compiler key bc250_mesh_direct_read, see
 * radv_bc250_direct_read_parts). */
#define RADV_BC250_DIRECT_READ_EXPORT  (1u << 0) /* export from the staging (ac_nir_lower_ngg_mesh) */
#define RADV_BC250_DIRECT_READ_UNIFORM (1u << 1) /* one LDS cell per uniform output */
#define RADV_BC250_DIRECT_READ_DEAD    (1u << 2) /* dead shared-variable cleanup */
#define RADV_BC250_DIRECT_READ_INDEX16 (1u << 3) /* 16-bit primitive index staging */
#define RADV_BC250_DIRECT_READ_CORNER  (1u << 4) /* load only the selected triangle corner */
#define RADV_BC250_DIRECT_READ_ALL     0x1fu
#define RADV_BC250_MESH_SAFE_CHECK_KEY (1u << 5)
#define RADV_BC250_MESH_SAFE_LOCAL_KEY (1u << 6)
#define RADV_BC250_MESH_SAFE_STATS_KEY (1u << 7)
unsigned radv_bc250_direct_read_parts(const char *option);
/* BC250_MESH_NIR_DUMP: writes <dir>/<sequence>_<blake3>.nir (a small header and
 * nir_serialize of the lowered NGG Mesh shader) for the offline oracle. */
void radv_bc250_dump_mesh_nir(const char *dir, const struct nir_shader *nir, unsigned wave_size,
                              unsigned hw_workgroup_size, const unsigned char *blake3,
                              const uint32_t index_staging[4], uint32_t compact_flags, uint64_t pp_params,
                              uint32_t clipcull);
bool radv_bc250_expand_primitive_attributes(struct nir_shader *mesh, struct nir_shader *fs, bool compact_policy,
                                            bool reclaim_dead_shared, unsigned direct_read, bool *over_budget,
                                            bool *dead_shared, uint32_t index_staging[4], bool *compact_map,
                                            bool clamp_primitives);
bool radv_bc250_mesh_needs_expansion(struct nir_shader *mesh);
bool radv_bc250_prepare_bary_affine(const struct radv_compiler_info *ci, struct radv_shader_stage *ms,
                                   struct nir_shader *fs);
bool radv_bc250_mesh_safe_owned(struct nir_shader **mesh, struct nir_shader **fs, bool corners, bool bary, bool tiny, bool last, bool io16, bool piece, bool mesh_queries,
                              unsigned direct_read, uint32_t index_staging[4],
                                uint64_t *pp_locations);
bool radv_bc250_bary_ref_slots(const struct nir_shader *producer, const struct nir_shader *fs,
                              int *raw_slot, int *flat_slot);
bool radv_bc250_mesh_private_bary(const struct radv_compiler_info *compiler_info,
                                 struct nir_shader *fs);
bool radv_bc250_mesh_fs_refused(const struct radv_compiler_info *compiler_info, struct nir_shader *fs);
bool radv_bc250_autocull_size_policy(const struct radv_compiler_info *compiler_info,
                                   unsigned prims, unsigned wave, unsigned lanes);
bool radv_bc250_mesh_safe_direct_candidate(struct nir_shader *mesh, bool parallel);
bool radv_bc250_mesh_safe_direct_candidate_slots(struct nir_shader *mesh, bool parallel, unsigned max_slots);
/* RADV_BC250_MESH_CLIPCULL_CONST: removes the constant non-negative clip/cull distance stores
 * of a Mesh shader (variables, before the BC250 rewrites). fs: the linked fragment shader
 * (NULL: none); fs_known false: not known (nothing is removed). */
bool radv_bc250_mesh_clip_cull_const(struct nir_shader *mesh, const struct nir_shader *fs, bool fs_known);
struct radv_compiler_info;
struct radv_shader_info;
/* RADV_BC250_MESH_CULLDIST_CULL: split a triangle Mesh shader with cull distances and more than 64
 * primitives, so that its pieces cull them in the one-wave autocull class. */
bool radv_bc250_mesh_culldist_split(const struct radv_compiler_info *compiler_info, const struct nir_shader *mesh);
/* After the NGG lowering: why a BC250 Mesh pipeline is refused for its position exports, or NULL
 * (radv_shader.c). */
const char *radv_bc250_mesh_pos_export_refusal(const struct radv_compiler_info *compiler_info,
                                               const struct nir_shader *nir, const struct radv_shader_info *info);
/* The clip/cull distance outputs a Mesh shader writes (VARYING_BIT_CLIP_DIST0/1, CULL_DIST0/1). */
#define RADV_BC250_CLIPCULL_OUTPUTS (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 | VARYING_BIT_CULL_DIST0 | \
                                     VARYING_BIT_CULL_DIST1)
bool radv_bc250_mesh_amd_route(bool enabled, bool fast, bool merged, struct nir_shader *mesh, bool has_task, bool multiview,
                               bool allow_clipcull);
struct radv_graphics_state_key;
unsigned radv_bc250_autocull_wide_level(const char *option);
bool radv_bc250_mesh_autocull_candidate(const struct nir_shader *mesh, bool expanded, bool merged, bool amd_mesh,
                                        const struct radv_graphics_state_key *gfx_state, const char **reason);
/* RADV_BC250_MESH_MERGE plan: K API workgroups of L invocations (lane stride S),
 * V vertices and P primitives each, become one subgroup of vm vertices, K*P
 * primitives and `lanes` threads.
 * Stage 1 (pp = false): vm = K*V+3 (3 sink vertices for hole primitives).
 * Stage 2 (pp = true, per-primitive outputs or CullPrimitive): every output is
 * staged in LDS and re-emitted by lane h for physical vertex h; instance k owns
 * physical vertices [k*c, (k+1)*c), c = V-1+P when per-primitive data rides on
 * a private provoking vertex per primitive (provoking = true), else c = V
 * (CullPrimitive only); vm = K*c+3. */
struct radv_bc250_merge_plan {
   unsigned k, s, l, v, p, vm, lanes, lds;
   bool subgroup_ops;
   unsigned wave;       /* wave size of the merged shader (32 only for stage 2 with subgroup operations) */
   bool pp;             /* stage 2: staged outputs, per-lane re-emission */
   bool provoking;      /* stage 2: per-primitive data on a private provoking vertex */
   bool provoking_last; /* the provoking vertex is corner 2 (else corner 0) */
   bool cull;           /* CullPrimitive becomes hole primitives */
   unsigned c;          /* physical vertices per instance */
   /* Set by the caller: option A indirect (RADV_BC250_MESH_MERGE_INDIRECT=a).
    * The grid comes from the dims user SGPR when it is nonzero; address32_hi
    * completes its 32-bit address. */
   bool dims;
   uint32_t address32_hi;
   const char *reason; /* why the shader is not merged (planner returned false) */
};
/* pp_refusal: NULL when the pipeline allows stage 2 (per-primitive data on a
 * provoking vertex), otherwise why not; provoking_last: static provoking mode. */
bool radv_bc250_mesh_merge_plan(struct nir_shader *mesh, const char *pp_refusal, bool provoking_last,
                                struct radv_bc250_merge_plan *plan);
bool radv_bc250_merge_mesh(struct nir_shader *mesh, struct nir_shader *fs, const struct radv_bc250_merge_plan *plan);
/* True when the Mesh shader writes per-primitive data other than the indices. */
bool radv_bc250_mesh_has_per_primitive_data(struct nir_shader *mesh);
/* NULL when a fragment shader can read per-primitive inputs as flat inputs of
 * the provoking vertex, otherwise the reason. */
const char *radv_bc250_merge_fs_refusal(struct nir_shader *fs);
/* Default per-piece primitive ceiling of the Mesh split (RADV_BC250_PERF_PIECE_PRIMS). */
#define RADV_BC250_DEFAULT_PIECE_PRIMS 63
/* piece_select (direct_split only, RADV_BC250_SPLIT_PREP_FREE): the piece may also come from
 * bc250_constants.split_piece (radv_bc250.c, bc250_split_flat_index). */
/* Set only while radv_graphics_pipeline_init retries a pipeline every other route refused
 * (RADV_BC250_MESH_SPLIT_ANY): Mesh-only lines/points may then split. */
extern __thread bool radv_bc250_split_refused_retry;
struct radv_shader_stage;
bool radv_bc250_mesh_wave64_promotable(const struct radv_shader_stage *stage);
struct nir_shader;
void radv_bc250_mesh_set_wave(struct nir_shader *nir, unsigned wave);
/* RADV_BC250_MESH_PIECE_PRIMID: set only around the safe direct pieces split. */
extern __thread bool radv_bc250_split_piece_primid;
/* RADV_BC250_MESH_SAFE_PIECES_EXT: set only around the owned admission of a split piece with every
 * direct-read part on. The private-corner LDS estimate then leaves out the per-vertex attribute copies
 * (read directly at export); an actual overflow still spills to the scratch ring and is refused. */
extern __thread bool radv_bc250_owned_lds_direct;
/* RADV_BC250_TASK_GRID_FOLD: set only around the Task-route split. */
extern __thread bool radv_bc250_split_task_grid_fold;
void radv_bc250_directmesh_env(void);
bool radv_bc250_bary_cid_slot(const struct nir_shader *producer, const struct nir_shader *fs, int *slot);
bool radv_bc250_split_mesh(struct nir_shader *mesh, struct nir_shader *task, struct nir_shader *fs, bool direct_split,
                           bool piece_select, bool balanced_slices, bool parallel_cull, bool output_regions,
                           bool compact_lds, unsigned piece_ceiling, unsigned min_pieces, unsigned *pieces_out);
struct radv_shader;
struct radv_physical_device;
struct radv_pipeline_layout;
struct radv_bc250_pipeline_plan;
struct radv_shader_stage;
void radv_bc250_capture_pipeline_plan(const struct radv_device *device,
                                      struct radv_graphics_pipeline *pipeline,
                                      const struct radv_shader_stage *mesh_stage);
VkResult radv_bc250_restore_cached_plan(struct radv_device *device,
                                       struct radv_graphics_pipeline *pipeline,
                                       const struct radv_pipeline_layout *layout,
                                       const struct radv_bc250_pipeline_plan *plan,
                                       const struct radv_shader *mesh, const struct radv_shader *fragment,
                                       struct radv_shader *producer, struct radv_shader *setup);
/* Select the immutable helper owner without binding a pipeline for shader objects. */
struct radv_graphics_pipeline *radv_bc250_mesh_pipeline(const struct radv_cmd_buffer *cmd_buffer);

/* Reads the BC250 hot-path switches into device->bc250_env (radv_device.h). */
void radv_bc250_device_env_init(struct radv_device *device, const struct radv_physical_device *pdev);
bool radv_bc250_chain_enabled(const struct radv_device *device);
uint64_t radv_bc250_chain_event(const struct radv_device *device, const char *event, const char *format, ...);
void radv_bc250_chain_state(struct radv_cmd_buffer *cmd, const char *event, const struct radv_shader *shader);
void radv_bc250_chain_packets(struct radv_cmd_buffer *cmd);
VkResult radv_bc250_prepare_direct_split(struct radv_device *device, struct radv_graphics_pipeline *pipeline);
/* RADV_BC250_MESH_MERGE option A: one prep dispatch per indirect call writes
 * the driver records, then the merged draw launches ceil(N/K) groups. */
void radv_bc250_draw_merge_indirect(struct radv_cmd_buffer *cmd_buffer, unsigned merge_k, uint64_t input,
                                    uint32_t records, uint32_t stride, uint64_t count);

struct radv_shader;
void radv_bc250_report_mesh_route(const struct radv_device *device, const struct radv_shader *shader,
                                  const char *object, unsigned pieces, bool task, bool ordered);

#endif

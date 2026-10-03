/*
 * Copyright © 2021 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */


#ifndef AC_NIR_H
#define AC_NIR_H

#include "ac_hw_stage.h"
#include "ac_shader_args.h"
#include "ac_shader_util.h"
#include "nir_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AC_MULTIVIEW_MAX_VIEWS 8

enum
{
   /* SPI_PS_INPUT_CNTL_i.OFFSET[0:4] */
   AC_EXP_PARAM_OFFSET_0 = 0,
   AC_EXP_PARAM_OFFSET_31 = 31,
   /* SPI_PS_INPUT_CNTL_i.DEFAULT_VAL[0:1] */
   AC_EXP_PARAM_DEFAULT_VAL_0000 = 64,
   AC_EXP_PARAM_DEFAULT_VAL_0001,
   AC_EXP_PARAM_DEFAULT_VAL_1110,
   AC_EXP_PARAM_DEFAULT_VAL_1111,
   AC_EXP_PARAM_UNDEFINED = 255,
};

enum {
   AC_EXP_FLAG_COMPRESSED = (1 << 0),
   AC_EXP_FLAG_DONE       = (1 << 1),
   AC_EXP_FLAG_VALID_MASK = (1 << 2),
};

enum {
   /* Whether nir_tex_instr should treat the deref or handle as an image binding
    * (image_load lowered to tex, etc.).
    */
   AC_NIR_TEX_BACKEND_FLAG_IS_IMAGE = BITFIELD_BIT(0),
};

struct ac_nir_config {
   enum amd_gfx_level gfx_level;
   bool uses_aco;
};

/* Maps I/O semantics to the actual location used by the lowering pass. */
typedef unsigned (*ac_nir_map_io_driver_location)(unsigned semantic);

/* Forward declaration of nir_builder so we don't have to include nir_builder.h here */
struct nir_builder;
typedef struct nir_builder nir_builder;

struct nir_xfb_info;
typedef struct nir_xfb_info nir_xfb_info;

/* Executed by ac_nir_cull when the current primitive is accepted. */
typedef void (*ac_nir_cull_accepted)(nir_builder *b, void *state);

struct ac_compiler_info;

void
ac_nir_set_options(const struct ac_compiler_info *info, bool use_llvm,
                   nir_shader_compiler_options *options);

nir_def *
ac_nir_load_arg_at_offset(nir_builder *b, const struct ac_shader_args *ac_args,
                          struct ac_arg arg, unsigned relative_index, bool scalar_wg_div);

nir_def *
ac_nir_load_arg(nir_builder *b, const struct ac_shader_args *ac_args, struct ac_arg arg);

nir_def *
ac_nir_load_arg_upper_bound(nir_builder *b, const struct ac_shader_args *ac_args, struct ac_arg arg,
                            unsigned upper_bound);

void ac_nir_store_arg(nir_builder *b, const struct ac_shader_args *ac_args, struct ac_arg arg,
                      nir_def *val);

nir_def *
ac_nir_unpack_arg(nir_builder *b, const struct ac_shader_args *ac_args, struct ac_arg arg,
                  unsigned rshift, unsigned bitwidth);

nir_def *
ac_nir_unpack_arg_wg_div(nir_builder *b, const struct ac_shader_args *ac_args, struct ac_arg arg,
                         unsigned rshift, unsigned bitwidth);

nir_def *
ac_nir_load_smem(nir_builder *b, unsigned num_components, nir_def *addr, nir_def *offset,
                 unsigned align_mul, enum gl_access_qualifier access);

typedef struct {
   enum amd_gfx_level gfx_level;
   bool has_ls_vgpr_init_bug;
   const enum ac_hw_stage hw_stage;
   unsigned wave_size;
   unsigned workgroup_size;
   bool use_llvm;
   bool load_grid_size_from_user_sgpr;
} ac_nir_lower_intrinsics_to_args_options;

bool ac_nir_lower_intrinsics_to_args(nir_shader *shader, const struct ac_shader_args *ac_args,
                                     const ac_nir_lower_intrinsics_to_args_options *options);

nir_xfb_info *ac_nir_get_sorted_xfb_info(const nir_shader *nir);

bool ac_nir_optimize_outputs(nir_shader *nir, bool sprite_tex_disallowed,
                             int8_t slot_remap[NUM_TOTAL_VARYING_SLOTS],
                             uint8_t param_export_index[NUM_TOTAL_VARYING_SLOTS]);

typedef struct {
   /* Per-vertex slots and tess levels. */
   uint64_t vram_output_mask;
   uint64_t lds_output_mask;
   uint64_t vgpr_output_mask; /* Hold the output values in VGPRs until the end. */
   /* Generic per-patch slots. */
   uint32_t vram_patch_output_mask;
   uint32_t lds_patch_output_mask;
   uint32_t vgpr_patch_output_mask; /* Hold the output values in VGPRs until the end. */

   /* The highest index returned by map_io + 1. */
   uint8_t highest_remapped_vram_output;
   uint8_t highest_remapped_vram_patch_output;
} ac_nir_tess_io_info;

void
ac_nir_get_tess_io_info(const nir_shader *tcs, const nir_tcs_info *tcs_info, uint64_t tes_inputs_read,
                        uint32_t tes_patch_inputs_read, ac_nir_map_io_driver_location map_io,
                        bool remapped_outputs_include_tess_levels, ac_nir_tess_io_info *io_info);

bool
ac_nir_lower_ls_outputs_to_mem(nir_shader *ls,
                               ac_nir_map_io_driver_location map,
                               enum amd_gfx_level gfx_level,
                               bool tcs_in_out_eq,
                               uint64_t tcs_inputs_via_temp,
                               uint64_t tcs_inputs_via_lds);

bool
ac_nir_lower_hs_inputs_to_mem(nir_shader *shader,
                              ac_nir_map_io_driver_location map,
                              enum amd_gfx_level gfx_level,
                              bool tcs_in_out_eq,
                              uint64_t tcs_inputs_via_temp,
                              uint64_t tcs_inputs_via_lds);

bool
ac_nir_lower_hs_outputs_to_mem(nir_shader *shader, const nir_tcs_info *info,
                               const ac_nir_tess_io_info *io_info,
                               ac_nir_map_io_driver_location map,
                               enum amd_gfx_level gfx_level,
                               unsigned wave_size);

bool
ac_nir_lower_tes_inputs_to_mem(nir_shader *shader,
                               ac_nir_map_io_driver_location map);

void
ac_nir_compute_tess_wg_info(const struct ac_compiler_info *info,
                            const ac_nir_tess_io_info *io_info, unsigned tcs_vertices_out,
                            unsigned wave_size, bool tess_uses_primid,
                            unsigned num_tcs_input_cp, unsigned lds_input_vertex_size,
                            unsigned num_remapped_tess_level_outputs,
                            unsigned *num_patches_per_wg, unsigned *lds_size);

bool
ac_nir_lower_es_outputs_to_mem(nir_shader *shader,
                               ac_nir_map_io_driver_location map,
                               enum amd_gfx_level gfx_level,
                               unsigned esgs_itemsize,
                               uint64_t gs_inputs_read);

bool
ac_nir_lower_gs_inputs_to_mem(nir_shader *shader,
                              ac_nir_map_io_driver_location map,
                              enum amd_gfx_level gfx_level,
                              bool triangle_strip_adjacency_fix);

bool
ac_nir_lower_indirect_derefs_early(nir_shader *shader);

bool
ac_nir_lower_indirect_derefs(nir_shader *shader);

typedef struct {
   const struct ac_compiler_info *compiler_info;

   unsigned max_workgroup_size;
   unsigned wave_size;

   /* The mask of clip and cull distances that the shader should export.
    *
    * Clip/cull distance components that are missing in export_clipdist_mask are removed, improving
    * throughput by up to 50% (3 pos exports -> 2 pos exports). The caller shouldn't set no-op
    * components (>= 0) in export_clipdist_mask to remove those completely. No-op components
    * should be determined by nir_opt_clip_cull_const before this.
    *
    * If can_cull is true, the shader culls cull distances and they are not exported to increase
    * throughput by reducing the number of pos exports. cull_clipdist_mask must be set to include
    * all cull distances that are < 0. The best case scenario is 100% increase in throughput from
    * not exporting any cull distances (2 pos exports -> 1 pos export).
    */
   uint8_t export_clipdist_mask;

   const uint8_t *vs_output_param_offset; /* GFX11+ */
   bool has_param_exports;
   bool has_gen_prim_query;
   bool has_ms_gs_invocations_query;

   /* VS/GS */
   /* The mask of clip and cull distances that the shader should cull against.
    * If no clip and cull distance outputs are present, it will load clip planes and cull
    * either against CLIP_VERTEX or POS.
    */
   uint8_t cull_clipdist_mask;
   bool write_pos_to_clipvertex;
   bool can_cull; /* if true, cull distances are not exported because the shader culls against them */
   bool disable_streamout;
   bool has_xfb_prim_query;
   bool use_gfx12_xfb_intrinsic;
   bool has_gs_primitives_query;
   bool force_vrs;
   bool compact_primitives;
   /* Skip culling dependent on the viewport state, which is frustum culling and small prim
    * culling. Set this when the shader writes the viewport index.
    */
   bool skip_viewport_state_culling;
   /* Use the point-triangle intersection to cull small triangles. */
   bool use_point_tri_intersection;

   /* VS */
   unsigned num_vertices_per_primitive;
   bool early_prim_export;
   bool passthrough;
   bool use_edgeflags;
   bool export_primitive_id;
   bool export_primitive_id_per_prim;
   uint32_t instance_rate_inputs;

   /* MS: effective device compiler policy, included in the caller cache key. */
   bool bc250_cull_compact;
   bool bc250_pack_triangle_vertices;
   /* RADV_BC250_MESH_AUTOCULL: cull triangles in the Mesh epilogue (backface, frustum, small
    * primitive, driven by the runtime NGG culling settings) and compact the survivors into
    * private-vertex triangles (ac_nir_lower_ngg_mesh.c, ms_autocull_compact). */
   bool bc250_autocull;
   /* RADV_BC250_MESH_CULLDIST_CULL (with bc250_autocull): when autocull applies and the shader
    * exports cull distances, cull with them in the shader (ms_autocull_accept) on every draw
    * (no runtime switch-off branch) and do not export them, as VS NGG culling does
    * (ac_nir_export_position dont_export_cull_distances). Clip distances stay exported. */
   bool bc250_autocull_culldist;
   /* Output (may be NULL): set to whether the cull distances are culled in the shader and not
    * exported (the caller then leaves them out of SPI_SHADER_POS_FORMAT / PA_CL_VS_OUT_CNTL). */
   bool *bc250_culldist_culled;
   /* RADV_BC250_MESH_COMPACT_LDS: per-vertex outputs kept in LDS use only the
    * dwords the shader accesses (packed records) instead of one vec4 slot per
    * location (ac_nir_lower_ngg_mesh.c, ms_packed_vertex_components). */
   bool bc250_compact_lds;
   /* RADV_BC250_MESH_DIRECT_READ: the expanded Mesh shader's per-vertex outputs
    * whose only stores are the expansion loop's copies of immutable LDS staging
    * are exported by recomputing that copy for the exported vertex instead of
    * storing and reloading an expanded LDS record (ac_nir_lower_ngg_mesh.c,
    * ms_direct_read_analyze). GFX10 without attribute ring, no multiview. */
   bool bc250_direct_read;
   /* RADV_BC250_MESH_IMPLICIT_TRIS: the shader is a base-driver expanded Mesh shader
    * whose primitive p uses the private vertices N p .. N p + N - 1 (N vertices per
    * primitive): derive the connectivity instead of storing it to LDS and reloading
    * it (ac_nir_lower_ngg_mesh.c, ms_implicit_index). */
   bool bc250_implicit_tris;
   /* BC250 barycentrics: generic slots that carry a copy of the vertex position
    * (radv_nir_bc250_export_bary_ref). Their stores are dropped and the position
    * loaded for the exported vertex is exported to them, so they take no LDS. */
   uint64_t bc250_bary_ref_mask;
   /* RADV_BC250_MESH_COMPACT: the expanded Mesh shader exports shared vertices
    * renumbered GE-like in its epilogue (ac_nir_lower_ngg_mesh.c,
    * ms_compact_vertices): every exported vertex is referenced by an exported
    * primitive and no corner references a vertex more than
    * AC_NIR_BC250_COMPACT_W below the running maximum index, for any index
    * data. The launch shape (3P lanes) is unchanged. */
   bool bc250_compact;
   /* Proven-capacity plain Mesh: latest-copy, triangle-wide W31 closure. */
   bool bc250_safe_direct;
   bool bc250_safe_fast;
   bool bc250_safe_local;
   /* RADV_BC250_MESH_PP_DIRECT (bit 0) and the provoking corner (bits 1-2): per-primitive generic outputs on the
    * local route. Every provoking corner owns a fresh slot; slot k exports the per-primitive outputs of the
    * triangle that created it as flat per-vertex parameters. */
   uint8_t bc250_pp_direct;
   /* RADV_BC250_MESH_LDS_PLAN: the local slot planner (ms_safe_local_vertices) finds each corner's first
    * occurrence in its interval with an LDS table (atomic min of corner keys) instead of bit-plane ballots.
    * Same plan. */
   bool bc250_lds_plan;
   /* RADV_BC250_MESH_LDS_COVER: on the local route, the coverage words are OR-ed in LDS instead of
    * wave reductions, and hole removal ranks vertices from scalar word prefix sums and an LDS rank table
    * instead of per-corner popcounts of every word. Same counts, map and connectivity. */
   bool bc250_lds_cover;
   /* RADV_BC250_MESH_MERGED_CHECK: with the adaptive and renumbered exports, the survivor check runs once on the
    * ranks. When the survivors reference every vertex, each rank is the index itself, so the ranked check is the
    * shared check; otherwise the shared check fails and the ranked check is the renumbered check. */
   bool bc250_merged_check;
   bool bc250_safe_check;
   bool bc250_safe_corners;
   /* RADV_BC250_MESH_SAFE_ADAPTIVE (with bc250_safe_corners, no owned per-primitive data): a
    * workgroup whose API connectivity references every vertex with backjump <= 31 exports its shared vertices and
    * every primitive without shader culling; any other workgroup exports the private corners. */
   bool bc250_safe_adaptive;
   /* RADV_BC250_MESH_SAFE_COMPACT (with bc250_safe_adaptive): before the shared-plus-all and corner
    * cases, a workgroup whose survivors keep backjump <= 31 once the vertices they do not reference
    * are deleted exports only the referenced vertices, renumbered in increasing order. */
   bool bc250_safe_compact;
   /* RADV_BC250_MESH_PP_SHARE (owned private corners): share vertices inside 10-triangle windows by the
    * application's logical index (bc250_compact_index_staging); the provoking corner
    * (bc250_compact_owned_corners 0x1 or 0x4) always gets its own copy. */
   bool bc250_pp_share;
   /* RADV_BC250_MESH_LEAN_CHECK: skip adaptive checks whose result is already known. */
   bool bc250_lean_check;
   bool bc250_safe_bary;
   bool bc250_safe_stats;
   bool bc250_safe_parallel;
   bool bc250_safe_autocull;
   unsigned bc250_safe_direct_bound;
   /* Optional dump metadata for the CPU export oracle (offset, stride, bytes, V). */
   uint32_t *bc250_safe_direct_index_staging;
   /* Corners that own a private vertex carrying the primitive's flat
    * (per-primitive) payload: bit 0 = corner 0 (provoking first), bit 2 =
    * corner 2 (provoking last); both when the provoking mode is dynamic. */
   uint8_t bc250_compact_owned_corners;
   /* The application's staged primitive indices in LDS (offset, array stride,
    * component bytes, declared logical vertices), radv_bc250_expand_primitive_attributes. */
   uint32_t bc250_compact_index_staging[4];
   /* Output (may be NULL): set to whether the compaction was applied. */
   bool *bc250_compact_applied;
   /* RADV_BC250_MESH_REFERENCE (raw-route Mesh): the reference driver's epilogue: the output
    * counts of multi-wave workgroups always go through LDS (no workgroup-uniform shortcut), and
    * the GFX10 fully-culled dummy exports POS0 only. */
   bool bc250_reference;
   /* GFX1013: waves without API invocations follow the API barriers by an exact published barrier total
    * instead of the "API waves in flight reached zero" check, which can end one barrier early. */
   bool bc250_wave_sync;
   bool multiview;
   /* RADV_BC250_MESH_MULTIVIEW_VTX: the inserted multiview layer is a per-vertex output (GFX10.1 has no
    * layer field in the primitive export). */
   bool multiview_layer_per_vertex;
} ac_nir_lower_ngg_options;

/* RADV_BC250_MESH_COMPACT: the hardware-proven backjump window (MESH_PERF/rawroot:
 * every raw shape with all vertices referenced and backjump <= 8 passed on GFX1013).
 * Only raise it after the backjump ladder gates pass on hardware. */
#define AC_NIR_BC250_COMPACT_W 8
/* RADV_BC250_MESH_COMPACT: anchor levels of the renumbering (ms_compact_vertices). One
 * level references only the first use of a vertex; two reach the stream quality of a
 * sequential allocator on meshlet-like data (MESH_PERF/compact/layout_variants.py). */
#define AC_NIR_BC250_COMPACT_LEVELS 2

bool
ac_nir_lower_ngg_nogs(nir_shader *shader, const ac_nir_lower_ngg_options *options,
                      uint32_t *out_lds_vertex_size, uint8_t *out_lds_scratch_size);

bool
ac_nir_lower_ngg_gs(nir_shader *shader, const ac_nir_lower_ngg_options *options,
                    uint32_t *out_lds_vertex_size, uint8_t *out_lds_scratch_size);

bool
ac_nir_lower_ngg_mesh(nir_shader *shader, const ac_nir_lower_ngg_options *options,
                      bool *out_needs_scratch_ring);

bool
ac_nir_lower_task_outputs_to_mem(nir_shader *shader,
                                 bool has_query);

bool
ac_nir_lower_mesh_inputs_to_mem(nir_shader *shader, bool has_task_shader);

bool
ac_nir_lower_global_access(nir_shader *shader, enum amd_gfx_level gfx_level);

bool ac_nir_lower_resinfo(nir_shader *nir, enum amd_gfx_level gfx_level);
bool ac_nir_lower_image_opcodes(nir_shader *nir);

bool
ac_nir_lower_legacy_vs(nir_shader *nir,
                       enum amd_gfx_level gfx_level,
                       uint32_t export_clipdist_mask,
                       bool write_pos_to_clipvertex,
                       const uint8_t *param_offsets,
                       bool has_param_exports,
                       bool export_primitive_id,
                       bool disable_streamout,
                       bool force_vrs);

typedef struct {
   bool has_gen_prim_query;
   bool has_pipeline_stats_query;

   enum amd_gfx_level gfx_level;
   uint32_t export_clipdist_mask;
   bool write_pos_to_clipvertex;
   const uint8_t *param_offsets;
   bool has_param_exports;
   bool disable_streamout;
   bool force_vrs;
} ac_nir_lower_legacy_gs_options;

typedef struct {
   uint8_t num_components_per_stream[4];
} ac_nir_legacy_gs_info;

bool
ac_nir_lower_legacy_gs(nir_shader *nir, ac_nir_lower_legacy_gs_options *options,
                       nir_shader **gs_copy_shader, ac_nir_legacy_gs_info *out_info);

/* This is a pre-link pass. It should only eliminate code and do lowering that mostly doesn't
 * generate AMD-specific intrinsics.
 */
typedef struct {
   /* System values. */
   bool msaa_disabled; /* true if MSAA is disabled, false may mean that the state is unknown */
   bool load_sample_positions_always_loads_current_ones; /* TODO: unify with RADV or remove */
   bool dynamic_rasterization_samples; /* TODO: unify with RADV or remove */
   int force_front_face; /* 0 -> keep, 1 -> set to true, -1 -> set to false */
   bool sample_shading;

   /* barycentrics:
    *    msaa_disabled:
    *       * All barycentrics including at_sample but excluding at_offset are changed to
    *         barycentric_pixel
    *    sample_shading:
    *       * All barycentrics are changed to per-sample interpolation except at_offset/at_sample.
    *       * barycentric_at_sample(sample_id) is replaced by barycentric_sample.
    */

   /* fbfetch_output */
   bool fbfetch_is_1D;
   bool fbfetch_layered;
   bool fbfetch_msaa;
   bool fbfetch_apply_fmask;

   /* Inputs. */
   bool lower_color_inputs_to_load_color01;

   /* Outputs. */
   bool clamp_color;                /* GL only */
   bool alpha_test_alpha_to_one;    /* GL only, this only affects alpha test */
   enum compare_func alpha_func;    /* GL only */
   bool keep_alpha_for_mrtz;        /* this prevents killing alpha based on spi_shader_col_format_hint */
   unsigned spi_shader_col_format_hint; /* this only shrinks and eliminates output stores */
   bool kill_z;
   bool kill_stencil;
   bool kill_samplemask;
} ac_nir_lower_ps_early_options;

bool
ac_nir_lower_ps_early(nir_shader *nir, const ac_nir_lower_ps_early_options *options);

typedef enum {
   /* sample_mask_in is replaced with b2i32(inot(load_helper_invocation)).
    *
    * If fragmentShadingRateWithSampleMask == VK_FALSE, pass this flag to the pass even if VRS is
    * enabled.
    */
   ac_nir_lower_samplemask_1sample_no_vrs,

   /* sample_mask_in is replaced with:
    *    nir_load_use_sample_mask_in_amd ? load_sample_mask_in : b2i32(inot(load_helper_invocation))
    *
    * Sample shading can't use this.
    */
   ac_nir_lower_samplemask_unknown_states_no_sample_shading,

   /* ps_iter_samples == 0 means that the value is unknown (dependent on dynamic rasterization
    * samples), and the real value is one of: 1, 2, 4. That requires loading PS_ITER_MASK from
    * a user SGPR to compute sample_mask_in.
    *
    * If (ps_iter_samples)
    *    sample_mask_in is ANDed with (nir_imm_int(ac_get_ps_iter_mask(num_samples)) << sample_id);
    * else
    *    sample_mask_in is ANDed with (load_ps_iter_mask_amd << sample_id);
    *
    * This is only used with min_sample_shading < 1.
    */
   ac_nir_lower_samplemask_sample_shading_partial,

   /* sample_mask_in is replaced with b2i32(inot(load_helper_invocation)) << sample_id.
    *
    * This is only used with min_sample_shading == 1.
    */
   ac_nir_lower_samplemask_sample_shading_max,
} ac_nir_lower_sample_mask_in_behavior;

typedef struct {
   ac_nir_lower_sample_mask_in_behavior behavior;

   /* The number of sample shading samples. Only for ac_nir_lower_samplemask_sample_shading_partial.
    * Set to 0 if unknown.
    */
   unsigned ps_iter_samples;
} ac_nir_lower_sample_mask_in_options;

bool
ac_nir_lower_sample_mask_in(nir_shader *nir, const ac_nir_lower_sample_mask_in_options *options);

/* This is a post-link pass. It shouldn't eliminate any code and it shouldn't affect shader_info
 * (those should be done in the early pass).
 */
typedef struct {
   enum amd_gfx_level gfx_level;
   bool use_aco;

   /* System values. */
   bool bc_optimize_for_persp;
   bool bc_optimize_for_linear;

   /* Exports. */
   bool uses_discard;
   bool dcc_decompress_gfx11;
   bool alpha_to_coverage_via_mrtz;
   bool dual_src_blend;
   unsigned spi_shader_col_format;
   unsigned color_is_int8;
   unsigned color_is_int10;
   bool alpha_to_one;

   /* Vulkan only */
   unsigned enable_mrt_output_nan_fixup;
   bool no_color_export;
   bool no_depth_export;
} ac_nir_lower_ps_late_options;

bool
ac_nir_lower_ps_late(nir_shader *nir, const ac_nir_lower_ps_late_options *options);

typedef struct {
   enum amd_gfx_level gfx_level;

   /* If true, round the layer component of the coordinates source to the nearest
    * integer for all array ops. This is always done for cube array ops.
    */
   bool lower_array_layer_round_even;

   /* Fix derivatives of constants and FS inputs in control flow.
    *
    * Ignores interpolateAtSample()/interpolateAtOffset(), dynamically indexed input loads,
    * pervertexEXT input loads, textureGather() with implicit LOD and 16-bit derivatives and
    * texture samples with nir_tex_src_min_lod.
    *
    * The layer must also be a constant or FS input.
    */
   bool fix_derivs_in_divergent_cf;
   unsigned max_wqm_vgprs;
} ac_nir_lower_tex_coords_options;

bool
ac_nir_lower_tex_coords(nir_shader *nir, const ac_nir_lower_tex_coords_options *options);

typedef struct {
   enum amd_gfx_level gfx_level;
} ac_nir_lower_image_tex_options;

bool
ac_nir_lower_image_tex(nir_shader *nir, const ac_nir_lower_image_tex_options *options);

void
ac_nir_store_debug_log_amd(nir_builder *b, nir_def *uvec4);

unsigned
ac_nir_varying_expression_max_cost(nir_shader *producer, nir_shader *consumer);

bool
ac_nir_opt_shared_append(nir_shader *shader);

bool
ac_nir_opt_flip_if_for_mem_loads(nir_shader *shader);

bool
ac_nir_flag_smem_for_loads(nir_shader *shader, enum amd_gfx_level gfx_level, bool use_llvm);

bool
ac_nir_fixup_mem_access_gfx6(nir_shader *shader,
                             struct ac_shader_args *args,
                             const uint32_t padding_bytes,
                             const bool fixup_null_desc,
                             const bool fixup_robust_oob);

bool
ac_nir_lower_mem_access_bit_sizes(nir_shader *shader, enum amd_gfx_level gfx_level, bool use_llvm);

bool
ac_nir_optimize_uniform_atomics(nir_shader *nir);

unsigned
ac_nir_lower_bit_size_callback(const nir_instr *instr, void *data);

bool
ac_nir_might_lower_bit_size(const nir_shader *shader);

bool
ac_nir_mem_vectorize_callback(unsigned align_mul, unsigned align_offset, unsigned bit_size,
                              unsigned num_components, int64_t hole_size,
                              nir_intrinsic_instr *low, nir_intrinsic_instr *high, void *data);

bool
ac_nir_scalarize_overfetching_loads_callback(const nir_intrinsic_instr *intr, const void *data);

bool
ac_nir_store_may_be_subdword(const nir_intrinsic_instr *instr);

uint8_t
ac_nir_lower_phis_to_scalar_cb(const nir_instr *instr, const void *_);

bool
ac_nir_allow_offset_wrap_cb(nir_intrinsic_instr *instr, const void *data);

bool
ac_nir_op_supports_packed_math_16bit(const nir_alu_instr* alu);

uint8_t
ac_nir_opt_vectorize_cb(const nir_instr *instr, const void *data);

unsigned
ac_nir_get_io_driver_location(const nir_shader *nir, unsigned location, bool is_input);

bool
ac_nir_assign_fs_input_locations(nir_shader *nir);

bool
ac_nir_fixup_smem_loads_null_prt(nir_shader *shader, uint8_t address_prt_wa_control_bit);

#ifdef __cplusplus
}
#endif

#endif /* AC_NIR_H */

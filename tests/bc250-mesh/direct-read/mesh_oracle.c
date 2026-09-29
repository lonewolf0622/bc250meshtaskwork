/*
 * BC250 Mesh export oracle (offline, CPU only).
 *
 *   mesh_oracle [options] A.nir [B.nir]
 *
 * Loads lowered NGG Mesh shaders written by the driver with
 * BC250_MESH_NIR_DUMP=<dir> (radv_bc250_dump_mesh_nir: a 16-byte header with the
 * wave size and the hardware workgroup size, then nir_serialize) and executes one
 * hardware workgroup of each on the CPU, for many seeds. Every external input
 * (buffer, push constant, descriptor, image, workgroup and draw values) is a
 * deterministic hash of the seed and of the access (intrinsic, constant indices
 * and source values), so two variants of one pipeline that run the same
 * application code see the same inputs. With B, the results of A (reference)
 * and B (candidate) are compared:
 *   - GS_ALLOC_REQ (vertex and primitive counts) must be equal;
 *   - every lane must issue the same exports (target, channels, flags) in the
 *     same order, and every channel that is defined in A must be defined and
 *     bit-identical in B (values A computed from never-written LDS or undef are
 *     undefined, B may export anything there);
 *   - side effects (memory stores/atomics and query counters) must be equal;
 *   - B must not access LDS out of bounds, race or run away where A does not.
 * Seeds where A invokes undefined behaviour (counts above the declared maximum,
 * out-of-bounds LDS access, a data race, a runaway loop) are skipped and counted.
 *
 * Execution model: the whole workgroup runs in lock step (structured control
 * flow with per-lane masks), subgroup operations act on each wave, workgroup
 * barriers advance an epoch, and an LDS read of a byte that another wave wrote
 * in the same epoch is reported as a race. LDS starts undefined; undefined
 * values propagate through ALU (bcsel only through the selected source).
 *
 * options: --seeds N (default 64), --first S, --cull 0|1|2 (load_cull_any_enabled_amd:
 * off, on, both; default both), --query 0|1, --list (print the intrinsics), --print (print A),
 * -v (-v -v also prints lane 0's exports per seed, or each piece's counts and the first
 * corner's exports with --geometry).
 *
 * --autocull (RADV_BC250_MESH_AUTOCULL): A is the compile without autocull, B the compile with
 * it (same pipeline and other switches). Every seed also draws an NGG culling state (cull
 * front / back, front face CCW, small-primitive culling, viewport scale and offset, small
 * primitive precision), which both shaders read through the culling intrinsics
 * (load_cull_*_amd, load_cull_any_enabled_amd = cull front or back, as RADV lowers it for
 * Mesh). Without face culling B must equal A exactly (the runtime skip). With it, the
 * oracle decides for every primitive A exports (not a null primitive) whether it survives,
 * from A's exported positions, with a C transcription of ac_nir_cull_primitive for
 * triangles (all w <= 0 or NaN, face culling with w reflection and zero area, frustum,
 * small primitive; skip_viewport_state_culling and use_point_tri_intersection off, as RADV
 * compiles autocull shaders), each float operation evaluated with NIR's constant folding.
 * The survivors S (in A's order) define B's expected output: GS_ALLOC_REQ 3|S| / |S|
 * (GFX10 fully-culled workaround when S is empty: 1 / 1, lane 0 exports the degenerate
 * primitive and a NaN position, nothing else), lane j exports primitive (3j, 3j+1, 3j+2),
 * lane 3j+c exports exactly the vertex exports of A's lane of corner c of the j-th survivor,
 * no other exports, equal side effects. A runs once per seed; B runs under --states N
 * (default 8) culling states when A's workgroup has primitives, under one otherwise.
 * --mutate 1..4 breaks the reference decision on purpose (self-check: must report mismatches).
 * --bary-ref (BC250 barycentrics): B is the same pipeline with a fragment shader that reads
 * barycentrics, so its last pre-rasterization stage also exports the position to two extra
 * parameters (radv_nir_bc250_export_bary_ref, ac_nir_lower_ngg_mesh bc250_bary_ref_mask).
 * Two of B's parameters are those references (usually the two numbered right after A's
 * parameters; with clip/cull distances read by the fragment shader those come after the
 * references, so every pair is tried): a lane has none or exactly two, each defined channel
 * equals the lane's position export (bit-identical), and every vertex lane that exports
 * parameters in A has both. They are removed (the other parameters renumbered) before the
 * exact comparison of everything else.
 * --compact (RADV_BC250_MESH_COMPACT): A is the compile without the switch, B the compile with it
 * (plus --autocull when B also culls). Every seed uses one index generator (seed % 13, or --gen
 * NAME|N): the application's data, random, degenerate, strips with backjumps, unreferenced
 * vertices, few vertices, grid, fan, descending, top vertex only last, meshoptimizer-like
 * meshlets, invalid and huge invalid indices. On every B run (also when A is undefined, e.g.
 * invalid indices; LDS out of bounds then reads 0 and undefined bits are followed, as on the
 * hardware) the safety of B's export is checked: V'/P' within the lanes, the maxima and 256,
 * exactly lanes < V' export a position and lanes < P' a primitive, no null primitive, indices
 * < V', every vertex referenced and backjump <= W (--w N, default 8). Where A is defined,
 * B's primitives in order, each as its 3 corners' vertex exports, must equal A's live primitives
 * (with --autocull the survivors, clip/cull distance test included, or the fully-culled dummy),
 * bit-exact on every channel A defines, except per-primitive payload channels (channels whose
 * value differs between two of A's corners of the same logical vertex) at corners that do not
 * own the payload. --mutate 5
 * swaps two expected primitives (self-check). A candidate compile that did not apply the switch
 * (MSR3 header) is compared exactly. -v prints per-generator statistics.
 * --bary-rot (with --compact): also recovers the barycentric vertex order on every candidate
 * primitive as the BC250 fragment shader does (x, y, w of the raw corners against the flat
 * provoking corner, both provoking conventions, every cyclic rotation) and checks it; see
 * check_bary_rotation.
 * --cost (with --autocull) also prints, per culled share of the workgroup, the average work per
 * workgroup of A and B (instructions issued per wave and active lanes, total and by kind); the
 * summary line stays last.
 * Clip and cull distances: every mode compares them with the other position exports (POS1..3
 * after the misc vector, ac_nir_export_position); the summary counts the defined clip/cull
 * channels compared (clipcull_channels). With --autocull the reference decision also culls a
 * triangle when one of its exported clip/cull distances is negative (flt) at all three corners
 * (as VS NGG culling and ms_autocull_accept; also with --compact --autocull, culled_dist in both
 * summaries); --mutate 6 drops that test.
 * --geometry [--pieces-a N] [--pieces-b M] [--grid]: A and B are different compiles of one
 * Mesh shader whose workgroups do not match lane for lane (e.g. the raw, unsplit shader of
 * RADV_BC250_MESH_AMD=1 against the base driver's split pieces and private-vertex expansion, or
 * two split piece counts). Per seed, one API workgroup L (0..63) runs as N hardware workgroups of
 * A (workgroup index L*N + k, k = 0..N-1: the direct split's piece order) and M of B, with the
 * same inputs. The live primitives (not null, not the GFX10 fully-culled dummy) of all pieces,
 * in order, must be equal: same count, same primitive export apart from the vertex indices
 * (edge flags, null bit, second channel), and for every corner the vertex exports of the
 * referenced vertex (targets, masks, flags; channels defined in A bit-identical in B), i.e.
 * the position, the misc vector and the clip/cull distances per exported vertex, and with
 * --geometry-params also the parameters (only when A and B share the varying layout: the
 * linker optimizes the raw shader's varyings, e.g. packs a workgroup-uniform component
 * elsewhere, which it cannot do through the expansion's staging; a parameter of B may then
 * still export more channels than A's). Side
 * effects are not compared (a split repeats the application's side-effect-free body per piece).
 * --grid: NumWorkGroups reads (4096, 1, 1) and every 32-bit global load reads 4096 (the direct
 * split's dimension record), so both compiles derive WorkGroupID.x = L from the index; only for
 * shaders without their own global memory loads.
 * Clip/cull distance forms (MSR3 header word 10: exported clip and cull components, bit 8 cull
 * distances culled in the shader, RADV_BC250_MESH_CULLDIST_CULL): when B exports fewer of them than
 * A (RADV_BC250_MESH_CLIPCULL_CONST removed constant ones, or B culls its cull distances in the
 * shader), every mode compares B with A's exports minus the removed part (the clip part and/or the
 * cull part; packed again from the first clip/cull target, DONE on the last position export,
 * and the GFX10 fully-culled dummy with B's position vectors). A dump without word 10 (an older
 * build) can only be compared against a B without any clip/cull export. The culling decision of
 * --autocull / --compact --autocull still uses all of A's distances; a B that culls its cull
 * distances has no runtime skip (it culls without face culling too), so every culling state goes
 * through the reference decision. The fully-culled dummy exports every position vector B declares
 * (POS0 NaN, then zero vectors).
 */
#include "nir.h"
#include "nir_builder.h"
#include "nir_constant_expressions.h"
#include "nir_serialize.h"
#include "compiler/glsl_types.h"
#include "util/blob.h"
#include "util/hash_table.h"
#include "util/ralloc.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LANES 256
#define LDS_BYTES 65536
#define MAX_EXPORTS 40 /* 32 PARAMs, up to four POS exports, and PRIM */
#define STEP_LIMIT 8000000ull
#define RING_BYTES (1 << 20)
#define GRID_X 4096u /* --grid */
#define MAX_PIECES 8 /* --geometry */
static const uint32_t ring_desc[4] = {0x5c4a7c11, 0x0badf00d, 0x7e57ab1e, 0x0000ffff};

typedef struct {
   uint64_t w[MAX_LANES / 64];
} mask_t;

static inline bool m_test(const mask_t *m, unsigned l) { return (m->w[l / 64] >> (l % 64)) & 1; }
static inline void m_set(mask_t *m, unsigned l) { m->w[l / 64] |= 1ull << (l % 64); }
static inline bool m_any(mask_t m) { for (unsigned i = 0; i < MAX_LANES / 64; i++) if (m.w[i]) return true; return false; }
static inline mask_t m_and(mask_t a, mask_t b) { for (unsigned i = 0; i < MAX_LANES / 64; i++) a.w[i] &= b.w[i]; return a; }
static inline mask_t m_andn(mask_t a, mask_t b) { for (unsigned i = 0; i < MAX_LANES / 64; i++) a.w[i] &= ~b.w[i]; return a; }
static inline mask_t m_or(mask_t a, mask_t b) { for (unsigned i = 0; i < MAX_LANES / 64; i++) a.w[i] |= b.w[i]; return a; }
#define foreach_lane(l, m) for (unsigned l = 0; l < st->lanes; l++) if (m_test(&(m), l))

typedef struct {
   nir_const_value *v; /* lanes * comps */
   /* 0: defined, 1: wholly undefined. For packed 32-bit values, bits
    * 1/2 mark undefined low/high halfwords. Non-bitwise operations remain
    * conservative; LDS and pack/unpack retain the halfword distinction. */
   uint8_t *undef;
   unsigned comps;
} value_t;

enum { COST_ALL, COST_LDS_LOAD, COST_LDS_STORE, COST_EXP_POS, COST_EXP_PARAM, COST_EXP_PRIM, COST_BARRIER,
       COST_KINDS };

typedef struct {
   uint32_t target, mask, flags;
   uint32_t v[4];
   uint8_t undef[4];
} export_t;

typedef struct {
   /* Observable results. */
   bool alloc_done;
   uint32_t alloc_vtx, alloc_prm, alloc_count;
   unsigned nexp[MAX_LANES];
   export_t exp[MAX_LANES][MAX_EXPORTS];
   uint64_t side_hash; /* memory stores, atomics, query counters (in lane order) */
   unsigned side_count;
   /* Undefined behaviour / model failures. */
   bool oob, race, runaway, unsupported, exp_overflow, count_over_max, bad_index;
   bool oob_soft; /* --compact: an LDS access out of bounds (reads 0, store dropped) */
   /* --compact: the application's staged index of (primitive, corner) at the end of the run. */
   uint32_t logical[MAX_LANES][3];
   bool logical_defined[MAX_LANES][3];
   /* Work model (--cost): instructions issued per wave (a wave issues an instruction when at
    * least one of its lanes is active; phis, constants, undefs and movs not counted) and the
    * active lanes, in total and for LDS loads, LDS stores, position, parameter and primitive
    * exports and barriers. */
   uint64_t cost_wave[COST_KINDS], cost_lane[COST_KINDS];
   char why[256];
} result_t;

/* Clip/cull distance exports (ac_nir_export_position: POS0, the misc vector when the shader
 * writes a per-vertex PointSize, edge flag, Layer, ViewportIndex or shading rate, then the
 * packed clip and cull distances): the first such target of the reference shader (~0u when it
 * writes none), and the number of defined clip/cull channels the comparisons checked. */
static unsigned clip_target = ~0u;
static uint64_t clip_channels;

static unsigned
clip_first_target(const nir_shader *nir)
{
   const uint64_t written = nir->info.outputs_written & ~nir->info.per_primitive_outputs;
   if (!(written & (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 | VARYING_BIT_CULL_DIST0 |
                    VARYING_BIT_CULL_DIST1)))
      return ~0u;
   const bool misc = written & (VARYING_BIT_PSIZ | VARYING_BIT_EDGE | VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT |
                                VARYING_BIT_PRIMITIVE_SHADING_RATE);
   return 12 /* V_008DFC_SQ_EXP_POS */ + 1 + misc;
}

static inline bool
is_clip_export(unsigned target)
{
   return target >= clip_target && target < 16;
}

/* --autocull: the NGG culling state of one seed. */
typedef struct {
   bool on;
   bool front, back, ccw, small;
   float vp[4]; /* x scale, y scale, x offset, y offset */
   float prec;
} cullstate_t;

typedef struct {
   nir_shader *nir;
   nir_function_impl *impl;
   unsigned wave, lanes, api_lanes;
   unsigned safe_export_bound; /* Candidate physical copies, independent of API vertex maximum. */
   uint64_t seed;
   int cull;
   int query;
   cullstate_t cs;
   value_t *vals;
   nir_block **pred; /* per lane */
   mask_t halted;
   uint8_t lds[LDS_BYTES], lds_def[LDS_BYTES];
   uint16_t lds_wave[LDS_BYTES];
   uint32_t lds_epoch[LDS_BYTES];
   /* The Mesh scratch ring (outputs that do not fit in LDS), for this workgroup. */
   uint8_t ring[RING_BYTES], ring_def[RING_BYTES];
   uint32_t epoch;
   unsigned lds_size;
   /* Primitive index staging (offset, stride, component bytes, logical vertices), 0 if unknown. */
   uint32_t index_staging[4];
   unsigned generator_primitives; /* Shared input domain when a proven count shrinks P. */
   /* "MSR3" header: RADV_BC250_MESH_COMPACT flags (bit 0 applied, bits 8..10 owned corners,
    * bits 16..23 W) and the parameter exports that carry a per-primitive payload. */
   uint32_t compact_flags;
   uint64_t pp_params;
   /* "MSR3" word 10: clip/cull distance exports (CC_CLIP, CC_CULL, CC_CULLED); 0 in older dumps. */
   uint32_t clipcull;
   /* --compact: the index generator of the current seed (see gen_index). */
   unsigned gen;
   uint64_t steps;
   result_t *r;
   struct hash_table *unknown;
   bool list;
   /* --geometry: the hardware workgroup index of this run (see the header). */
   bool geo;
   uint32_t wg_index;
   /* --grid: NumWorkGroups and every 32-bit global load read GRID_X (see the header). */
   bool grid;
} state_t;

typedef struct {
   mask_t brk, cont;
} loop_state_t;

/* Undefined behaviour or a model failure: the run is not compared, stop it. */
static inline bool
aborted(const state_t *st)
{
   const result_t *r = st->r;
   return r->oob || r->race || r->runaway || (r->unsupported && !st->list) || r->exp_overflow;
}

static uint64_t
mix64(uint64_t x)
{
   x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
   x ^= x >> 27; x *= 0x94d049bb133111ebull;
   x ^= x >> 31;
   return x;
}

static uint64_t
hash_add(uint64_t h, uint64_t v)
{
   return mix64(h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2)));
}

static void
fail(state_t *st, const char *fmt, const char *arg)
{
   if (!st->r->why[0])
      snprintf(st->r->why, sizeof(st->r->why), fmt, arg);
}

static value_t *
val(state_t *st, nir_def *def)
{
   value_t *v = &st->vals[def->index];
   if (!v->v) {
      v->comps = def->num_components;
      v->v = calloc((size_t)st->lanes * def->num_components, sizeof(nir_const_value));
      v->undef = calloc((size_t)st->lanes * def->num_components, 1);
      memset(v->undef, 1, (size_t)st->lanes * def->num_components);
   }
   return v;
}

/* Mask only unknown halfwords; an undefined padding half must not hide a
 * mismatch in a live 16-bit component. */
static uint32_t
known_bits(uint8_t undef)
{
   if (undef & 1)
      return 0;
   return ((undef & 2) ? 0 : 0xffffu) | ((undef & 4) ? 0 : 0xffff0000u);
}

static bool
export_channel_differs(const export_t *a, const export_t *b, unsigned c)
{
   uint32_t known = known_bits(a->undef[c]);
   return (known & ~known_bits(b->undef[c])) || ((a->v[c] ^ b->v[c]) & known);
}

static inline nir_const_value *
cv(state_t *st, nir_def *def, unsigned lane, unsigned c)
{
   return &val(st, def)->v[lane * def->num_components + c];
}

static inline uint8_t *
cu(state_t *st, nir_def *def, unsigned lane, unsigned c)
{
   return &val(st, def)->undef[lane * def->num_components + c];
}

static uint64_t
cv_bits(nir_const_value v, unsigned bit_size)
{
   switch (bit_size) {
   case 1: return v.b;
   case 8: return v.u8;
   case 16: return v.u16;
   case 32: return v.u32;
   default: return v.u64;
   }
}

static nir_const_value
bits_cv(uint64_t x, unsigned bit_size)
{
   nir_const_value v;
   memset(&v, 0, sizeof(v));
   switch (bit_size) {
   case 1: v.b = x & 1; break;
   case 8: v.u8 = x; break;
   case 16: v.u16 = x; break;
   case 32: v.u32 = x; break;
   default: v.u64 = x; break;
   }
   return v;
}

static uint64_t
src_bits(state_t *st, nir_src *src, unsigned lane, unsigned c, bool *undef)
{
   if (undef && *cu(st, src->ssa, lane, c))
      *undef = true;
   return cv_bits(*cv(st, src->ssa, lane, c), src->ssa->bit_size);
}

static void
set_def(state_t *st, nir_def *def, unsigned lane, unsigned c, uint64_t bits, bool undef)
{
   *cv(st, def, lane, c) = bits_cv(bits, def->bit_size);
   *cu(st, def, lane, c) = undef;
}

/* An external input: a hash of the seed and the access, shaped so that some
 * values are small integers (counts, indices), some small floats, some raw. */
static uint64_t
external_value(state_t *st, uint64_t h, unsigned c, unsigned bit_size)
{
   h = hash_add(h, c);
   uint64_t mode = st->seed % 4;
   uint64_t pick = (h >> 60) & 3;
   uint64_t x = mix64(h);
   if (bit_size == 64)
      return x;
   if (mode == 0 || (mode == 3 && pick == 0))
      return x % 32;
   if (mode == 1 || (mode == 3 && pick == 1))
      return x % 300;
   if (mode == 2 || (mode == 3 && pick == 2)) {
      float f = ((double)(x % 20001) / 5000.0) - 2.0;
      uint32_t u;
      memcpy(&u, &f, 4);
      return bit_size == 16 ? _mesa_float_to_half(f) : u;
   }
   return x;
}

static uint64_t
access_hash(state_t *st, nir_intrinsic_instr *intr, unsigned lane)
{
   uint64_t h = hash_add(st->seed * 0x1000193ull + 7, intr->intrinsic);
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
   for (unsigned i = 0; i < info->num_indices; i++)
      h = hash_add(h, intr->const_index[i]);
   for (unsigned s = 0; s < info->num_srcs; s++) {
      for (unsigned c = 0; c < intr->src[s].ssa->num_components; c++)
         h = hash_add(h, src_bits(st, &intr->src[s], lane, c, NULL));
   }
   return h;
}

static void
note_unknown(state_t *st, const char *name)
{
   if (!_mesa_hash_table_search(st->unknown, name))
      _mesa_hash_table_insert(st->unknown, name, NULL);
}


/* ---- --compact index generators ----
 * The application's staged primitive index of (primitive p, corner c), written by the
 * application part (both compiles see the same function of the seed). 0 keeps the
 * application's own data (below the vertex count); the others are adversarial for the
 * RADV_BC250_MESH_COMPACT renumbering. Generators >= GEN_INVALID produce values the API
 * forbids (the reference is then undefined and only the candidate's safety is checked). */
enum {
   GEN_APP, GEN_RANDOM, GEN_DEGENERATE, GEN_STRIP_BACKJUMP, GEN_UNREFERENCED, GEN_FEW, GEN_GRID, GEN_FAN,
   GEN_DESCENDING, GEN_TOP_ONLY_LAST, GEN_MESHLET, GEN_INVALID, GEN_INVALID_HUGE, NGEN,
   GEN_FILE = NGEN, /* --indices FILE (not in the cycle) */
};
/* --indices FILE: fixed triangles ("tris a,b,c a,b,c ..." on one line), e.g. a bench shape's index data. */
static unsigned file_tris[MAX_LANES][3], file_count;
static const char *gen_name[NGEN + 1] = {"app", "random", "degenerate", "strip_backjump", "unreferenced", "few",
                                     "grid", "fan", "descending", "top_only_last", "meshlet", "invalid",
                                     "invalid_huge", "file"};

/* GEN_MESHLET: a meshoptimizer-like meshlet (MESH_PERF/rawroot/layout_model.py greedy_meshlets):
 * on a 48x48-quad grid mesh, start at a seed-chosen triangle and repeatedly add the adjacent
 * triangle that adds the fewest new vertices (ties: lowest id) while the vertex and primitive
 * limits allow; vertices are numbered in first-use order. Primitives beyond the meshlet repeat it. */
#define MESHLET_N 48
static unsigned meshlet_tris[MAX_LANES][3], meshlet_count;

static void
build_meshlet(uint64_t seed, unsigned vmax, unsigned pmax)
{
   enum { N = MESHLET_N, NV = (N + 1) * (N + 1), NT = 2 * N * N };
   static unsigned tri[NT][3];
   static bool built;
   if (!built) {
      for (unsigned y = 0; y < N; y++)
         for (unsigned x = 0; x < N; x++) {
            const unsigned a = y * (N + 1) + x, b = a + 1, c = a + N + 1, d = c + 1, t = 2 * (y * N + x);
            tri[t][0] = a; tri[t][1] = b; tri[t][2] = c;
            tri[t + 1][0] = b; tri[t + 1][1] = d; tri[t + 1][2] = c;
         }
      built = true;
   }
   static int local[NV];
   static bool used[NT];
   memset(local, -1, sizeof(local));
   memset(used, 0, sizeof(used));
   unsigned nverts = 0;
   meshlet_count = 0;
   unsigned t = mix64(seed * 0x3141ull) % NT;
   pmax = MIN2(pmax, MAX_LANES);
   while (meshlet_count < pmax) {
      used[t] = true;
      for (unsigned c = 0; c < 3; c++) {
         if (local[tri[t][c]] < 0)
            local[tri[t][c]] = nverts++;
         meshlet_tris[meshlet_count][c] = local[tri[t][c]];
      }
      meshlet_count++;
      /* The next triangle: adjacent to a meshlet vertex, fewest new vertices, lowest id. */
      int best = -1;
      unsigned best_new = 4;
      for (unsigned i = 0; i < NT; i++) {
         if (used[i])
            continue;
         unsigned fresh = 0;
         for (unsigned c = 0; c < 3; c++)
            fresh += local[tri[i][c]] < 0;
         if (fresh == 3 || nverts + fresh > vmax || fresh >= best_new)
            continue;
         best = i;
         best_new = fresh;
      }
      if (best < 0)
         break;
      t = best;
   }
}

static bool dummy_matches(const result_t *rb, unsigned lanes);

static bool sgpr_model;
static bool compact_mode;
static int fixed_gen = -1;

static uint64_t
gen_index(state_t *st, unsigned p, unsigned c, uint64_t x, unsigned bytes)
{
   const unsigned v = st->index_staging[3];
   const unsigned pmax = st->generator_primitives;
   const uint64_t h = mix64(st->seed * 0x9e37ull + p * 3 + c + 0x51ab);
   const uint64_t mask = bytes >= 8 ? ~0ull : (1ull << (8 * bytes)) - 1;
   if (!compact_mode)
      return (x & 0xffff) % v;
   switch (st->gen) {
   case GEN_APP: default: return (x & 0xffff) % v;
   case GEN_RANDOM: return h % v;
   case GEN_DEGENERATE: return mix64(st->seed * 7 + p) % v; /* (L, L, L) */
   case GEN_STRIP_BACKJUMP: /* a strip over the whole vertex range, every 5th primitive reaches back to 0 */
      return p % 5 == 4 && c == 0 ? 0 : (p + c) % v;
   case GEN_UNREFERENCED: /* only even vertices below v/2: odd and top vertices never referenced */
      return (h % MAX2(1, v / 4)) * 2;
   case GEN_FEW: return h % MIN2(v, 3); /* the same vertex many times */
   case GEN_GRID: { /* bench-like 8-wide grid (t_m1 order), wrapped at v */
      const unsigned q = p / 2, a = (q / 7) * 8 + q % 7;
      static const unsigned off[2][3] = {{0, 1, 8}, {1, 9, 8}};
      return (a + off[p & 1][c]) % v;
   }
   case GEN_FAN: return c == 0 ? 0 : (p + c) % v; /* (0, p+1, p+2): 0 in every primitive */
   case GEN_DESCENDING: return (v - 1) - (3 * p + c) % v;
   case GEN_TOP_ONLY_LAST: /* the top vertex only in the last primitive, far below otherwise */
      return p + 1 == pmax && c == 2 ? v - 1 : h % MAX2(1, v / 8);
   case GEN_MESHLET: return meshlet_count ? meshlet_tris[p % meshlet_count][c] % v : 0;
   case GEN_FILE: return file_count ? file_tris[p % file_count][c] : 0;
   case GEN_INVALID: return h & 1 ? h % v : (v + (h >> 8) % 64) & mask; /* half out of range */
   case GEN_INVALID_HUGE: return (h >> 3) & mask;
   }
}

/* ---- LDS ---- */

/* --compact: an LDS access out of bounds reads 0 (undefined) and drops the store, as the
 * hardware does, instead of stopping the run: garbage index data must still leave a safe
 * export (the candidate's safety check runs on every seed). */
static bool lds_oob_soft;

static bool
lds_range(state_t *st, uint64_t addr, unsigned bytes)
{
   if (addr + bytes > st->lds_size || addr + bytes > LDS_BYTES) {
      if (lds_oob_soft) {
         st->r->oob_soft = true;
         fail(st, "%s", " (LDS access out of bounds, soft)");
         return false;
      }
      st->r->oob = true;
      char where[96];
      snprintf(where, sizeof(where), " (LDS access out of bounds: %" PRIu64 "+%u of %u)", addr, bytes, st->lds_size);
      fail(st, "%s", where);
      return false;
   }
   return true;
}

static uint64_t
lds_read(state_t *st, unsigned lane, uint64_t addr, unsigned bytes, bool *undef)
{
   if (!lds_range(st, addr, bytes)) {
      *undef = true;
      return 0;
   }
   uint64_t x = 0;
   unsigned wave = lane / st->wave;
   for (unsigned i = 0; i < bytes; i++) {
      x |= (uint64_t)st->lds[addr + i] << (8 * i);
      if (!st->lds_def[addr + i])
         *undef = true;
      else if (st->lds_wave[addr + i] != wave && st->lds_epoch[addr + i] == st->epoch)
         st->r->race = true;
   }
   return x;
}

static void
lds_write(state_t *st, unsigned lane, uint64_t addr, unsigned bytes, uint64_t x, bool undef)
{
   if (!lds_range(st, addr, bytes))
      return;
   unsigned wave = lane / st->wave;
   for (unsigned i = 0; i < bytes; i++) {
      /* Two waves writing different data in one epoch: a race. */
      if (st->lds_def[addr + i] && st->lds_wave[addr + i] != wave && st->lds_epoch[addr + i] == st->epoch &&
          st->lds[addr + i] != (uint8_t)(x >> (8 * i)))
         st->r->race = true;
      st->lds[addr + i] = x >> (8 * i);
      st->lds_def[addr + i] = !undef;
      st->lds_wave[addr + i] = wave;
      st->lds_epoch[addr + i] = st->epoch;
   }
}

/* ---- ALU ---- */

static void
exec_alu(state_t *st, nir_alu_instr *alu, mask_t m)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   unsigned bit_size = 0;
   if (!nir_alu_type_get_type_size(info->output_type))
      bit_size = alu->def.bit_size;
   for (unsigned i = 0; i < info->num_inputs; i++) {
      if (!bit_size && !nir_alu_type_get_type_size(info->input_types[i]))
         bit_size = alu->src[i].src.ssa->bit_size;
   }
   if (!bit_size)
      bit_size = alu->def.bit_size;

   foreach_lane(l, m) {
      nir_const_value src[NIR_ALU_MAX_INPUTS][NIR_MAX_VEC_COMPONENTS];
      nir_const_value *srcs[NIR_ALU_MAX_INPUTS];
      uint8_t src_undef[NIR_ALU_MAX_INPUTS][NIR_MAX_VEC_COMPONENTS];
      memset(src, 0, sizeof(src));
      for (unsigned i = 0; i < info->num_inputs; i++) {
         for (unsigned j = 0; j < nir_ssa_alu_instr_src_components(alu, i); j++) {
            src[i][j] = *cv(st, alu->src[i].src.ssa, l, alu->src[i].swizzle[j]);
            src_undef[i][j] = *cu(st, alu->src[i].src.ssa, l, alu->src[i].swizzle[j]);
         }
         srcs[i] = src[i];
      }
      nir_const_value dest[NIR_MAX_VEC_COMPONENTS];
      memset(dest, 0, sizeof(dest));
      uint16_t poison = 0;
      nir_eval_const_opcode(alu->op, dest, &poison, alu->def.num_components, bit_size, srcs,
                            st->nir->info.float_controls_execution_mode);
      for (unsigned c = 0; c < alu->def.num_components; c++) {
         uint8_t undef = !!(poison & (1u << c));
         if (alu->op == nir_op_bcsel || alu->op == nir_op_b32csel || alu->op == nir_op_fcsel) {
            unsigned cc = info->input_sizes[0] ? c : c;
            bool cond_undef = src_undef[0][info->input_sizes[0] ? 0 : cc];
            bool cond;
            if (alu->op == nir_op_fcsel)
               cond = src[0][cc].f32 != 0.0f;
            else if (alu->src[0].src.ssa->bit_size == 1)
               cond = src[0][cc].b;
            else
               cond = cv_bits(src[0][cc], alu->src[0].src.ssa->bit_size) != 0;
            undef |= cond_undef ? 1 : src_undef[cond ? 1 : 2][c];
         } else if (alu->op == nir_op_mov) {
            undef |= src_undef[0][c];
         } else if (alu->op == nir_op_pack_32_2x16_split) {
            undef |= (src_undef[0][0] ? 2 : 0) | (src_undef[1][0] ? 4 : 0);
         } else if (alu->op == nir_op_unpack_32_2x16_split_x ||
                    alu->op == nir_op_unpack_32_2x16_split_y) {
            unsigned half = alu->op == nir_op_unpack_32_2x16_split_x ? 2 : 4;
            undef |= !!(src_undef[0][0] & (1 | half));
         } else if (nir_op_is_vec(alu->op)) {
            /* vecN: channel c is source c (the other sources do not matter). */
            undef |= src_undef[c][0];
         } else {
            for (unsigned i = 0; i < info->num_inputs; i++) {
               unsigned n = nir_ssa_alu_instr_src_components(alu, i);
               if (info->input_sizes[i]) {
                  for (unsigned j = 0; j < n; j++)
                     undef |= !!src_undef[i][j];
               } else if (c < n) {
                  undef |= !!src_undef[i][c];
               }
            }
         }
         *cv(st, &alu->def, l, c) = dest[c];
         *cu(st, &alu->def, l, c) = undef;
      }
   }
}

/* ---- intrinsics ---- */

static unsigned
wave_first(state_t *st, mask_t m, unsigned wave)
{
   for (unsigned l = wave * st->wave; l < (wave + 1) * st->wave && l < st->lanes; l++) {
      if (m_test(&m, l))
         return l;
   }
   return ~0u;
}

static uint64_t
wave_ballot(state_t *st, mask_t m, unsigned wave, nir_src *cond, bool *undef)
{
   uint64_t b = 0;
   for (unsigned l = wave * st->wave; l < (wave + 1) * st->wave && l < st->lanes; l++) {
      if (m_test(&m, l) && src_bits(st, cond, l, 0, undef))
         b |= 1ull << (l - wave * st->wave);
   }
   return b;
}

static uint64_t
reduce_op(nir_op op, uint64_t a, uint64_t b, unsigned bit_size, bool *ok)
{
   nir_const_value sa = bits_cv(a, bit_size), sb = bits_cv(b, bit_size), d;
   nir_const_value *srcs[2] = {&sa, &sb};
   switch (op) {
   case nir_op_iadd: case nir_op_imul: case nir_op_fadd: case nir_op_fmul: case nir_op_imin: case nir_op_umin:
   case nir_op_fmin: case nir_op_imax: case nir_op_umax: case nir_op_fmax: case nir_op_iand: case nir_op_ior:
   case nir_op_ixor:
      nir_eval_const_opcode(op, &d, NULL, 1, bit_size, srcs, 0);
      return cv_bits(d, bit_size);
   default:
      *ok = false;
      return 0;
   }
}

static uint64_t
reduce_identity(nir_op op, unsigned bit_size)
{
   uint64_t all = bit_size == 64 ? ~0ull : (1ull << bit_size) - 1;
   switch (op) {
   case nir_op_iadd: case nir_op_ior: case nir_op_ixor: case nir_op_umax: return 0;
   case nir_op_fadd: return 0;
   case nir_op_imul: return 1;
   case nir_op_fmul: return bit_size == 32 ? 0x3f800000 : bit_size == 16 ? 0x3c00 : 0x3ff0000000000000ull;
   case nir_op_umin: case nir_op_iand: return all;
   case nir_op_imin: return all >> 1;
   case nir_op_imax: return (all >> 1) + 1;
   case nir_op_fmin: return bit_size == 32 ? 0x7f800000 : bit_size == 16 ? 0x7c00 : 0x7ff0000000000000ull;
   case nir_op_fmax: return bit_size == 32 ? 0xff800000 : bit_size == 16 ? 0xfc00 : 0xfff0000000000000ull;
   default: return 0;
   }
}

static void
side_effect(state_t *st, nir_intrinsic_instr *intr, mask_t m)
{
   foreach_lane(l, m) {
      st->r->side_hash = hash_add(st->r->side_hash, access_hash(st, intr, l) ^ l);
      st->r->side_count++;
   }
}

static bool (*private_intrinsic)(state_t *, nir_intrinsic_instr *, mask_t);

static void
exec_intrinsic(state_t *st, nir_intrinsic_instr *intr, mask_t m)
{
   if (private_intrinsic && private_intrinsic(st, intr, m))
      return;
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
   nir_def *def = info->has_dest ? &intr->def : NULL;
   const unsigned nwaves = DIV_ROUND_UP(st->lanes, st->wave);

   if (st->list) {
      note_unknown(st, info->name);
   }

   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_invocation_index:
      foreach_lane(l, m) set_def(st, def, l, 0, l, false);
      return;
   case nir_intrinsic_load_local_invocation_id:
      foreach_lane(l, m) for (unsigned c = 0; c < 3; c++) set_def(st, def, l, c, c ? 0 : l, false);
      return;
   case nir_intrinsic_load_subgroup_id:
      foreach_lane(l, m) set_def(st, def, l, 0, l / st->wave, false);
      return;
   case nir_intrinsic_load_num_subgroups:
      foreach_lane(l, m) set_def(st, def, l, 0, nwaves, false);
      return;
   case nir_intrinsic_load_subgroup_invocation:
      foreach_lane(l, m) set_def(st, def, l, 0, l % st->wave, false);
      return;
   case nir_intrinsic_load_subgroup_size:
      foreach_lane(l, m) set_def(st, def, l, 0, st->wave, false);
      return;
   case nir_intrinsic_load_view_index:
      foreach_lane(l, m) set_def(st, def, l, 0, 0, false);
      return;
   case nir_intrinsic_load_cull_any_enabled_amd:
      foreach_lane(l, m) set_def(st, def, l, 0, st->cs.on ? (st->cs.front || st->cs.back) : st->cull, false);
      return;
   case nir_intrinsic_load_cull_front_face_enabled_amd:
   case nir_intrinsic_load_cull_back_face_enabled_amd:
   case nir_intrinsic_load_cull_ccw_amd:
   case nir_intrinsic_load_cull_small_triangles_enabled_amd:
      if (!st->cs.on)
         break;
      foreach_lane(l, m) set_def(st, def, l, 0,
                                 intr->intrinsic == nir_intrinsic_load_cull_front_face_enabled_amd ? st->cs.front :
                                 intr->intrinsic == nir_intrinsic_load_cull_back_face_enabled_amd ? st->cs.back :
                                 intr->intrinsic == nir_intrinsic_load_cull_ccw_amd ? st->cs.ccw : st->cs.small, false);
      return;
   case nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd:
   case nir_intrinsic_load_cull_small_triangle_precision_amd:
      if (!st->cs.on)
         break;
      foreach_lane(l, m) for (unsigned c = 0; c < def->num_components; c++) {
         float f = intr->intrinsic == nir_intrinsic_load_cull_small_triangle_precision_amd ? st->cs.prec : st->cs.vp[c];
         uint32_t u;
         memcpy(&u, &f, 4);
         set_def(st, def, l, c, u, false);
      }
      return;
   case nir_intrinsic_load_initial_edgeflags_amd:
      /* RADV: 0 for Mesh shaders. Modeled only with --autocull, --compact and --geometry
       * (primitive exports decoded). */
      if (!st->cs.on && !compact_mode && !st->geo)
         break;
      foreach_lane(l, m) set_def(st, def, l, 0, 0, false);
      return;
   case nir_intrinsic_load_prim_gen_query_enabled_amd:
   case nir_intrinsic_load_pipeline_stat_query_enabled_amd:
      foreach_lane(l, m) set_def(st, def, l, 0, st->query, false);
      return;
   case nir_intrinsic_load_vertex_id_zero_base:
   case nir_intrinsic_load_workgroup_index:
   case nir_intrinsic_load_workgroup_id: {
      if (st->geo) {
         /* --geometry: the run's hardware workgroup index (x only for an id). */
         foreach_lane(l, m) for (unsigned c = 0; c < def->num_components; c++)
            set_def(st, def, l, c, c ? 0 : st->wg_index, false);
         return;
      }
      /* The workgroup index: the same for every lane. */
      uint64_t h = hash_add(st->seed, 0x7767);
      foreach_lane(l, m) for (unsigned c = 0; c < def->num_components; c++)
         set_def(st, def, l, c, external_value(st, h, c, def->bit_size), false);
      return;
   }
   case nir_intrinsic_load_num_workgroups:
      if (!st->grid)
         break;
      foreach_lane(l, m) for (unsigned c = 0; c < def->num_components; c++)
         set_def(st, def, l, c, c ? 1 : GRID_X, false);
      return;
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_global_amd:
      if (!st->grid)
         break;
      if (def->bit_size != 32) {
         st->r->unsupported = true;
         fail(st, "%s", "--grid: a global load that is not 32-bit");
         return;
      }
      foreach_lane(l, m) for (unsigned c = 0; c < def->num_components; c++)
         set_def(st, def, l, c, GRID_X, false);
      return;
   case nir_intrinsic_barrier:
      if (nir_intrinsic_execution_scope(intr) >= SCOPE_WORKGROUP ||
          nir_intrinsic_memory_scope(intr) >= SCOPE_WORKGROUP)
         st->epoch++;
      return;
   case nir_intrinsic_load_shared: {
      unsigned bytes = def->bit_size / 8;
      foreach_lane(l, m) {
         bool u = false;
         uint64_t addr = src_bits(st, &intr->src[0], l, 0, &u) + nir_intrinsic_base(intr);
         if (u) {
            if (lds_oob_soft)
               st->r->oob_soft = true;
            else {
               st->r->oob = true;
               fail(st, "%s", " (LDS load at an undefined address)");
            }
         }
         for (unsigned c = 0; c < def->num_components; c++) {
            bool cu_ = u;
            uint64_t x = lds_read(st, l, (uint32_t)addr + c * bytes, bytes, &cu_);
            set_def(st, def, l, c, x, cu_);
            uint64_t a = (uint32_t)addr + c * bytes;
            if (!u && bytes == 4 && a + bytes <= st->lds_size) {
               *cu(st, def, l, c) =
                  ((!st->lds_def[a] || !st->lds_def[a + 1]) ? 2 : 0) |
                  ((!st->lds_def[a + 2] || !st->lds_def[a + 3]) ? 4 : 0);
            }
         }
      }
      return;
   }
   case nir_intrinsic_store_shared: {
      nir_src *data = &intr->src[0];
      unsigned bytes = data->ssa->bit_size / 8;
      unsigned wrmask = nir_intrinsic_write_mask(intr);
      foreach_lane(l, m) {
         bool u = false;
         uint64_t addr = src_bits(st, &intr->src[1], l, 0, &u) + nir_intrinsic_base(intr);
         if (u) {
            if (lds_oob_soft)
               st->r->oob_soft = true;
            else {
               st->r->oob = true;
               fail(st, "%s", " (LDS store at an undefined address)");
            }
         }
         u_foreach_bit(c, wrmask) {
            bool du = false;
            uint64_t x = src_bits(st, data, l, c, &du);
            uint32_t a = (uint32_t)addr + c * bytes;
            const uint32_t *ix = st->index_staging;
            /* Keep the application's primitive indices valid (below the vertex
             * count) so that workgroups export real geometry: the same function
             * of the low 16 bits in every variant, whatever the staging width. */
            if (ix[1] && ix[3] && bytes == ix[2] && a >= ix[0] &&
                a < ix[0] + st->nir->info.mesh.max_primitives_out * ix[1] && (a - ix[0]) % ix[1] < 3 * ix[2] &&
                (a - ix[0]) % ix[2] == 0)
               x = gen_index(st, (a - ix[0]) / ix[1], (a - ix[0]) % ix[1] / ix[2], x, bytes);
            uint8_t packed_undef = *cu(st, data->ssa, l, c);
            if (bytes == 4 && !(packed_undef & 1)) {
               lds_write(st, l, a, 2, x, packed_undef & 2);
               lds_write(st, l, a + 2, 2, x >> 16, packed_undef & 4);
            } else {
               lds_write(st, l, a, bytes, x, du);
            }
         }
      }
      return;
   }
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap: {
      unsigned bytes = def->bit_size / 8;
      nir_atomic_op op = nir_intrinsic_atomic_op(intr);
      foreach_lane(l, m) {
         bool u = false;
         uint64_t addr = (uint32_t)(src_bits(st, &intr->src[0], l, 0, &u) + nir_intrinsic_base(intr));
         bool ou = false;
         uint64_t old = lds_read(st, l, addr, bytes, &ou);
         uint64_t data = src_bits(st, &intr->src[1], l, 0, &u);
         uint64_t res;
         if (intr->intrinsic == nir_intrinsic_shared_atomic_swap) {
            uint64_t data2 = src_bits(st, &intr->src[2], l, 0, &u);
            res = old == data ? data2 : old;
         } else {
            nir_const_value a = bits_cv(old, def->bit_size), b = bits_cv(data, def->bit_size), d;
            nir_const_value *srcs[2] = {&a, &b};
            nir_op alu;
            switch (op) {
            case nir_atomic_op_iadd: alu = nir_op_iadd; break;
            case nir_atomic_op_imin: alu = nir_op_imin; break;
            case nir_atomic_op_umin: alu = nir_op_umin; break;
            case nir_atomic_op_imax: alu = nir_op_imax; break;
            case nir_atomic_op_umax: alu = nir_op_umax; break;
            case nir_atomic_op_iand: alu = nir_op_iand; break;
            case nir_atomic_op_ior: alu = nir_op_ior; break;
            case nir_atomic_op_ixor: alu = nir_op_ixor; break;
            case nir_atomic_op_xchg: alu = nir_op_mov; break;
            default: note_unknown(st, "shared_atomic(op)"); st->r->unsupported = true; return;
            }
            if (alu == nir_op_mov) {
               res = data;
            } else {
               nir_eval_const_opcode(alu, &d, NULL, 1, def->bit_size, srcs, 0);
               res = cv_bits(d, def->bit_size);
            }
         }
         /* Atomics are ordered, never a race. */
         uint32_t e = st->epoch;
         st->epoch = ~0u;
         lds_write(st, l, addr, bytes, res, u || ou);
         st->epoch = e;
         if (lds_range(st, addr, bytes))
            for (unsigned i = 0; i < bytes; i++) st->lds_epoch[addr + i] = 0;
         set_def(st, def, l, 0, old, ou);
      }
      return;
   }
   case nir_intrinsic_elect:
      for (unsigned w = 0; w < nwaves; w++) {
         unsigned first = wave_first(st, m, w);
         for (unsigned l = w * st->wave; l < (w + 1) * st->wave && l < st->lanes; l++)
            if (m_test(&m, l)) set_def(st, def, l, 0, l == first, false);
      }
      return;
   case nir_intrinsic_ballot:
      for (unsigned w = 0; w < nwaves; w++) {
         bool u = false;
         uint64_t b = wave_ballot(st, m, w, &intr->src[0], &u);
         for (unsigned l = w * st->wave; l < (w + 1) * st->wave && l < st->lanes; l++) {
            if (!m_test(&m, l))
               continue;
            for (unsigned c = 0; c < def->num_components; c++)
               set_def(st, def, l, c, def->bit_size == 64 ? b : (b >> (32 * c)) & 0xffffffffu, u);
         }
      }
      return;
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
      for (unsigned w = 0; w < nwaves; w++) {
         bool u = false;
         uint64_t b = wave_ballot(st, m, w, &intr->src[0], &u);
         mask_t act = {0};
         for (unsigned l = w * st->wave; l < (w + 1) * st->wave && l < st->lanes; l++)
            if (m_test(&m, l)) m_set(&act, l);
         uint64_t actb = 0;
         for (unsigned l = w * st->wave; l < (w + 1) * st->wave && l < st->lanes; l++)
            if (m_test(&act, l)) actb |= 1ull << (l - w * st->wave);
         bool res = intr->intrinsic == nir_intrinsic_vote_any ? b != 0 : b == actb;
         for (unsigned l = w * st->wave; l < (w + 1) * st->wave && l < st->lanes; l++)
            if (m_test(&m, l)) set_def(st, def, l, 0, res, u);
      }
      return;
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_shuffle:
      for (unsigned w = 0; w < nwaves; w++) {
         for (unsigned l = w * st->wave; l < (w + 1) * st->wave && l < st->lanes; l++) {
            if (!m_test(&m, l))
               continue;
            unsigned src_lane;
            if (intr->intrinsic == nir_intrinsic_read_first_invocation) {
               src_lane = wave_first(st, m, w);
            } else {
               bool u = false;
               src_lane = w * st->wave + (src_bits(st, &intr->src[1], l, 0, &u) % st->wave);
            }
            for (unsigned c = 0; c < def->num_components; c++) {
               bool u = false;
               uint64_t x = src_bits(st, &intr->src[0], src_lane, c, &u);
               set_def(st, def, l, c, x, u);
            }
         }
      }
      return;
   case nir_intrinsic_inverse_ballot:
      foreach_lane(l, m) {
         bool u = false;
         unsigned sub = l % st->wave;
         nir_src *src = &intr->src[0];
         uint64_t x;
         if (src->ssa->bit_size == 64)
            x = src_bits(st, src, l, 0, &u) >> sub;
         else
            x = src_bits(st, src, l, sub / 32, &u) >> (sub % 32);
         set_def(st, def, l, 0, x & 1, u);
      }
      return;
   case nir_intrinsic_mbcnt_amd:
      foreach_lane(l, m) {
         bool u = false;
         uint64_t mask = src_bits(st, &intr->src[0], l, 0, &u);
         uint64_t add = src_bits(st, &intr->src[1], l, 0, &u);
         unsigned sub = l % st->wave;
         uint64_t below = sub ? mask & ((1ull << sub) - 1) : 0;
         set_def(st, def, l, 0, (uint32_t)(util_bitcount64(below) + add), u);
      }
      return;
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan: {
      nir_op op = nir_intrinsic_reduction_op(intr);
      unsigned cluster = intr->intrinsic == nir_intrinsic_reduce ? nir_intrinsic_cluster_size(intr) : 0;
      if (!cluster || cluster > st->wave)
         cluster = st->wave;
      bool ok = true;
      for (unsigned c = 0; c < def->num_components; c++) {
         foreach_lane(l, m) {
            unsigned base = l - (l % cluster);
            uint64_t acc = reduce_identity(op, def->bit_size);
            bool u = false;
            for (unsigned k = base; k < base + cluster && k < st->lanes; k++) {
               if (!m_test(&m, k))
                  continue;
               if (intr->intrinsic == nir_intrinsic_inclusive_scan && k > l)
                  break;
               if (intr->intrinsic == nir_intrinsic_exclusive_scan && k >= l)
                  break;
               acc = reduce_op(op, acc, src_bits(st, &intr->src[0], k, c, &u), def->bit_size, &ok);
            }
            set_def(st, def, l, c, acc, u);
         }
      }
      if (!ok) {
         note_unknown(st, "reduce(op)");
         st->r->unsupported = true;
      }
      return;
   }
   case nir_intrinsic_export_amd:
      foreach_lane(l, m) {
         if (st->r->nexp[l] >= MAX_EXPORTS) {
            st->r->exp_overflow = true;
            continue;
         }
         export_t *e = &st->r->exp[l][st->r->nexp[l]++];
         e->target = nir_intrinsic_target(intr);
         e->mask = nir_intrinsic_enabled_channels(intr);
         e->flags = nir_intrinsic_flags(intr);
         for (unsigned c = 0; c < 4; c++) {
            e->v[c] = 0;
            e->undef[c] = 1;
            if (c < intr->src[0].ssa->num_components && (e->mask & (1u << c))) {
               bool u = false;
               e->v[c] = src_bits(st, &intr->src[0], l, c, &u);
               e->undef[c] = *cu(st, intr->src[0].ssa, l, c);
            }
         }
      }
      return;
   case nir_intrinsic_sendmsg_amd: {
      /* GS_ALLOC_REQ (the only message the Mesh lowering sends): wave 0 lane values. */
      unsigned lane = ~0u;
      foreach_lane(l, m) { lane = l; break; }
      if (lane != ~0u) {
         bool u = false;
         uint64_t x = src_bits(st, &intr->src[0], lane, 0, &u);
         st->r->alloc_count++;
         st->r->alloc_done = true;
         st->r->alloc_vtx = x & 0xfff;
         st->r->alloc_prm = (x >> 12) & 0xfff;
         if (u && lds_oob_soft)
            st->r->oob_soft = true; /* --compact: the hardware uses the bits (safety still checked) */
         else if (u)
            st->r->unsupported = true, fail(st, "%s", "undefined GS_ALLOC_REQ");
      }
      return;
   }
   case nir_intrinsic_load_ring_mesh_scratch_amd:
      foreach_lane(l, m) for (unsigned c = 0; c < 4; c++) set_def(st, def, l, c, ring_desc[c], false);
      return;
   case nir_intrinsic_load_ring_mesh_scratch_offset_amd:
      foreach_lane(l, m) set_def(st, def, l, 0, 0, false);
      return;
   case nir_intrinsic_load_buffer_amd:
   case nir_intrinsic_store_buffer_amd: {
      const bool store = intr->intrinsic == nir_intrinsic_store_buffer_amd;
      nir_src *desc = &intr->src[store ? 1 : 0];
      bool ring = true;
      foreach_lane(l, m) for (unsigned c = 0; c < 4; c++) ring &= src_bits(st, desc, l, c, NULL) == ring_desc[c];
      if (!ring)
         break; /* an application buffer: generic handling below */
      nir_src *data = store ? &intr->src[0] : NULL;
      unsigned bits = store ? data->ssa->bit_size : def->bit_size;
      unsigned comps = store ? data->ssa->num_components : def->num_components;
      unsigned wrmask = store ? nir_intrinsic_write_mask(intr) : BITFIELD_MASK(comps);
      foreach_lane(l, m) {
         bool u = false;
         uint64_t addr = src_bits(st, &intr->src[store ? 2 : 1], l, 0, &u) +
                         src_bits(st, &intr->src[store ? 3 : 2], l, 0, &u) + nir_intrinsic_base(intr);
         if (src_bits(st, &intr->src[store ? 4 : 3], l, 0, &u))
            st->r->unsupported = true, fail(st, "%s", "indexed scratch ring access");
         u_foreach_bit(c, wrmask) {
            uint64_t a = addr + c * (bits / 8);
            if (u || a + bits / 8 > RING_BYTES) {
               st->r->oob = true;
               continue;
            }
            if (store) {
               bool du = false;
               uint64_t x = src_bits(st, data, l, c, &du);
               for (unsigned i = 0; i < bits / 8; i++) {
                  st->ring[a + i] = x >> (8 * i);
                  st->ring_def[a + i] = !du;
               }
            } else {
               uint64_t x = 0;
               bool du = false;
               for (unsigned i = 0; i < bits / 8; i++) {
                  x |= (uint64_t)st->ring[a + i] << (8 * i);
                  du |= !st->ring_def[a + i];
               }
               set_def(st, def, l, c, x, du);
            }
         }
      }
      return;
   }
   case nir_intrinsic_atomic_add_gen_prim_count_amd:
   case nir_intrinsic_atomic_add_shader_invocation_count_amd:
      side_effect(st, intr, m);
      return;
   default:
      break;
   }

   /* Generic: pure loads return hashed inputs; everything else with side effects
    * is recorded. */
   if (def) {
      if (!(info->flags & NIR_INTRINSIC_CAN_ELIMINATE)) {
         /* Loads with side effects (atomics on memory): hashed result + recorded. */
         side_effect(st, intr, m);
      } else if (!strstr(info->name, "load") && !strstr(info->name, "image") && !strstr(info->name, "resource") &&
                 !strstr(info->name, "descriptor")) {
         note_unknown(st, info->name);
         st->r->unsupported = true;
         fail(st, "unsupported intrinsic %s", info->name);
      }
      foreach_lane(l, m) {
         uint64_t h = access_hash(st, intr, l);
         for (unsigned c = 0; c < def->num_components; c++)
            set_def(st, def, l, c, external_value(st, h, c, def->bit_size), false);
      }
   } else {
      if (strstr(info->name, "store") || strstr(info->name, "atomic")) {
         side_effect(st, intr, m);
      } else {
         note_unknown(st, info->name);
         st->r->unsupported = true;
         fail(st, "unsupported intrinsic %s", info->name);
      }
   }
}

static void
exec_tex(state_t *st, nir_tex_instr *tex, mask_t m)
{
   foreach_lane(l, m) {
      uint64_t h = hash_add(st->seed, 0x7e7 + tex->op);
      h = hash_add(h, tex->texture_index);
      h = hash_add(h, tex->sampler_index);
      for (unsigned s = 0; s < tex->num_srcs; s++)
         for (unsigned c = 0; c < tex->src[s].src.ssa->num_components; c++)
            h = hash_add(h, src_bits(st, &tex->src[s].src, l, c, NULL));
      for (unsigned c = 0; c < tex->def.num_components; c++)
         set_def(st, &tex->def, l, c, external_value(st, h, c, tex->def.bit_size), false);
   }
}

/* ---- control flow ---- */

static void exec_cf_list(state_t *st, struct exec_list *list, mask_t m, loop_state_t *ls);

/* ACO assigns uniform definitions to SGPRs. A write in any active lane of
 * a wave is visible to its inactive lanes as well. In particular, the count
 * phi after a sub-wave API workgroup merges a uniform value with undef, not
 * an undefined per-lane VGPR. Use ACO's ignore-undef divergence rule and
 * reject conflicting defined values instead of silently choosing one. */
static void
broadcast_sgpr(state_t *st, nir_def *def, mask_t active)
{
   if (!sgpr_model || !def || def->divergent)
      return;
   for (unsigned w = 0; w < st->lanes; w += st->wave) {
      for (unsigned c = 0; c < def->num_components; c++) {
         unsigned first = MAX_LANES;
         for (unsigned l = w; l < MIN2(w + st->wave, st->lanes); l++) {
            if (!m_test(&active, l) || *cu(st, def, l, c))
               continue;
            if (first == MAX_LANES)
               first = l;
            else if (cv_bits(*cv(st, def, first, c), def->bit_size) !=
                     cv_bits(*cv(st, def, l, c), def->bit_size)) {
               st->r->unsupported = true;
               fail(st, "%s", "conflicting SGPR values");
               return;
            }
         }
         if (first == MAX_LANES)
            continue;
         const nir_const_value value = *cv(st, def, first, c);
         for (unsigned l = w; l < MIN2(w + st->wave, st->lanes); l++) {
            *cv(st, def, l, c) = value;
            *cu(st, def, l, c) = 0;
         }
      }
   }
}

static void
exec_block(state_t *st, nir_block *block, mask_t m, loop_state_t *ls)
{
   if (!m_any(m))
      return;

   /* Phis: all read before any is written. */
   unsigned nphis = 0;
   nir_foreach_phi(phi, block) nphis++;
   if (nphis) {
      struct { nir_phi_instr *phi; nir_const_value *v; uint8_t *u; } tmp[nphis];
      unsigned i = 0;
      nir_foreach_phi(phi, block) {
         tmp[i].phi = phi;
         tmp[i].v = calloc((size_t)st->lanes * phi->def.num_components, sizeof(nir_const_value));
         tmp[i].u = calloc((size_t)st->lanes * phi->def.num_components, 1);
         foreach_lane(l, m) {
            nir_phi_src *src = NULL;
            nir_foreach_phi_src(ps, phi) {
               if (ps->pred == st->pred[l])
                  src = ps;
            }
            for (unsigned c = 0; c < phi->def.num_components; c++) {
               if (src) {
                  tmp[i].v[l * phi->def.num_components + c] = *cv(st, src->src.ssa, l, c);
                  tmp[i].u[l * phi->def.num_components + c] = *cu(st, src->src.ssa, l, c);
               } else {
                  tmp[i].u[l * phi->def.num_components + c] = 1;
               }
            }
         }
         i++;
      }
      for (i = 0; i < nphis; i++) {
         nir_phi_instr *phi = tmp[i].phi;
         foreach_lane(l, m) for (unsigned c = 0; c < phi->def.num_components; c++) {
            *cv(st, &phi->def, l, c) = tmp[i].v[l * phi->def.num_components + c];
            *cu(st, &phi->def, l, c) = tmp[i].u[l * phi->def.num_components + c];
         }
         free(tmp[i].v);
         free(tmp[i].u);
      }
   }

   nir_foreach_instr(instr, block) {
      unsigned active = 0;
      foreach_lane(l, m) active++;
      st->steps += active;
      if (instr->type == nir_instr_type_alu || instr->type == nir_instr_type_intrinsic ||
          instr->type == nir_instr_type_tex) {
         unsigned kind = COST_KINDS;
         if (instr->type == nir_instr_type_alu && nir_op_is_vec_or_mov(nir_instr_as_alu(instr)->op))
            kind = COST_KINDS + 1; /* register moves: not counted */
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_shared)
               kind = COST_LDS_LOAD;
            else if (intr->intrinsic == nir_intrinsic_store_shared)
               kind = COST_LDS_STORE;
            else if (intr->intrinsic == nir_intrinsic_barrier)
               kind = COST_BARRIER;
            else if (intr->intrinsic == nir_intrinsic_export_amd)
               kind = nir_intrinsic_target(intr) == 20 ? COST_EXP_PRIM :
                      nir_intrinsic_target(intr) >= 32 ? COST_EXP_PARAM : COST_EXP_POS;
         }
         if (kind <= COST_KINDS) {
            unsigned waves = 0;
            for (unsigned w = 0; w * st->wave < st->lanes; w++) {
               for (unsigned l = w * st->wave; l < (w + 1) * st->wave && l < st->lanes; l++) {
                  if (m_test(&m, l)) {
                     waves++;
                     break;
                  }
               }
            }
            st->r->cost_wave[COST_ALL] += waves;
            st->r->cost_lane[COST_ALL] += active;
            if (kind < COST_KINDS) {
               st->r->cost_wave[kind] += waves;
               st->r->cost_lane[kind] += active;
            }
         }
      }
      if (st->steps > STEP_LIMIT) {
         st->r->runaway = true;
         return;
      }
      switch (instr->type) {
      case nir_instr_type_phi:
         break;
      case nir_instr_type_alu:
         exec_alu(st, nir_instr_as_alu(instr), m);
         break;
      case nir_instr_type_load_const: {
         nir_load_const_instr *lc = nir_instr_as_load_const(instr);
         foreach_lane(l, m) for (unsigned c = 0; c < lc->def.num_components; c++) {
            *cv(st, &lc->def, l, c) = lc->value[c];
            *cu(st, &lc->def, l, c) = 0;
         }
         break;
      }
      case nir_instr_type_undef: {
         nir_undef_instr *u = nir_instr_as_undef(instr);
         foreach_lane(l, m) for (unsigned c = 0; c < u->def.num_components; c++)
            *cu(st, &u->def, l, c) = 1;
         break;
      }
      case nir_instr_type_intrinsic:
         exec_intrinsic(st, nir_instr_as_intrinsic(instr), m);
         break;
      case nir_instr_type_tex:
         exec_tex(st, nir_instr_as_tex(instr), m);
         break;
      case nir_instr_type_deref: {
         /* Resource derefs (textures, samplers) only feed texture and image
          * accesses: an identity value per lane for the input hash. */
         nir_deref_instr *d = nir_instr_as_deref(instr);
         foreach_lane(l, m) {
            uint64_t h = hash_add(0xde7ef, d->deref_type);
            if (d->deref_type == nir_deref_type_var) {
               h = hash_add(h, d->var->data.descriptor_set);
               h = hash_add(h, d->var->data.binding);
            } else {
               h = hash_add(h, src_bits(st, &d->parent, l, 0, NULL));
               if (d->deref_type == nir_deref_type_array || d->deref_type == nir_deref_type_ptr_as_array)
                  h = hash_add(h, src_bits(st, &d->arr.index, l, 0, NULL));
               else if (d->deref_type == nir_deref_type_struct)
                  h = hash_add(h, d->strct.index);
            }
            if (!(d->modes & (nir_var_uniform | nir_var_image))) {
               st->r->unsupported = true;
               fail(st, "%s", "memory deref");
            }
            set_def(st, &d->def, l, 0, h, false);
         }
         break;
      }
      case nir_instr_type_jump: {
         nir_jump_instr *j = nir_instr_as_jump(instr);
         if (j->type == nir_jump_break && ls)
            ls->brk = m_or(ls->brk, m);
         else if (j->type == nir_jump_continue && ls)
            ls->cont = m_or(ls->cont, m);
         else if (j->type == nir_jump_halt || j->type == nir_jump_return)
            st->halted = m_or(st->halted, m);
         else {
            st->r->unsupported = true;
            fail(st, "%s", "unsupported jump");
         }
         break;
      }
      default: {
         char type[16];
         snprintf(type, sizeof(type), "%d", instr->type);
         st->r->unsupported = true;
         fail(st, "unsupported instruction type %s", type);
         break;
      }
      }
      if (sgpr_model && (instr->type == nir_instr_type_phi || instr->type == nir_instr_type_alu ||
                         instr->type == nir_instr_type_intrinsic || instr->type == nir_instr_type_load_const))
         broadcast_sgpr(st, nir_instr_def(instr), m);
      if (aborted(st))
         return;
   }
   foreach_lane(l, m) st->pred[l] = block;
}

static void
exec_cf_list(state_t *st, struct exec_list *list, mask_t m, loop_state_t *ls)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      mask_t eff = m_andn(m, st->halted);
      if (ls)
         eff = m_andn(eff, m_or(ls->brk, ls->cont));
      if (!m_any(eff) || aborted(st))
         return;
      switch (node->type) {
      case nir_cf_node_block:
         exec_block(st, nir_cf_node_as_block(node), eff, ls);
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         mask_t t = {0}, e = {0};
         foreach_lane(l, eff) {
            bool u = false;
            if (src_bits(st, &nif->condition, l, 0, &u))
               m_set(&t, l);
            else
               m_set(&e, l);
            if (u && lds_oob_soft) {
               /* --compact: an undefined value still has bits on the hardware (the model keeps
                * them); follow them and mark the run undefined for the comparison only, so the
                * candidate's safety is still checked for garbage input. */
               st->r->oob_soft = true;
               char ssa[16];
               snprintf(ssa, sizeof(ssa), "%u", nif->condition.ssa->index);
               fail(st, " (undefined branch condition %%%s, soft)", ssa);
            } else if (u) {
               /* Branching on an undefined value: the program is undefined. */
               st->r->oob = true;
               char ssa[16];
               snprintf(ssa, sizeof(ssa), "%u", nif->condition.ssa->index);
               fail(st, " (undefined branch condition %%%s)", ssa);
            }
         }
         exec_cf_list(st, &nif->then_list, t, ls);
         exec_cf_list(st, &nif->else_list, e, ls);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         loop_state_t inner;
         memset(&inner, 0, sizeof(inner));
         mask_t active = eff;
         while (m_any(active) && !aborted(st)) {
            memset(&inner.cont, 0, sizeof(inner.cont));
            exec_cf_list(st, &loop->body, active, &inner);
            active = m_andn(m_andn(active, inner.brk), st->halted);
            if (nir_loop_has_continue_construct(loop)) {
               st->r->unsupported = true;
               fail(st, "%s", "continue construct");
            }
         }
         break;
      }
      default:
         st->r->unsupported = true;
         break;
      }
   }
}

static void
run(state_t *st, uint64_t seed, int cull, result_t *r)
{
   memset(r, 0, sizeof(*r));
   st->r = r;
   st->seed = seed;
   st->cull = cull;
   st->epoch = 1;
   st->steps = 0;
   memset(&st->halted, 0, sizeof(st->halted));
   if (compact_mode && st->gen == GEN_MESHLET && st->index_staging[3])
      build_meshlet(seed, st->index_staging[3], st->generator_primitives);
   memset(st->lds_def, 0, sizeof(st->lds_def));
   memset(st->lds_epoch, 0, sizeof(st->lds_epoch));
   memset(st->ring_def, 0, sizeof(st->ring_def));
   /* Undefined LDS still holds some bits: a seed-dependent pattern. */
   for (unsigned i = 0; i < LDS_BYTES; i++)
      st->lds[i] = mix64(seed * 131 + i) & 0xff;
   for (unsigned i = 0; i < st->impl->ssa_alloc; i++) {
      if (st->vals[i].v)
         memset(st->vals[i].undef, 1, (size_t)st->lanes * st->vals[i].comps);
   }
   for (unsigned l = 0; l < st->lanes; l++)
      st->pred[l] = NULL;
   mask_t all = {0};
   for (unsigned l = 0; l < st->lanes; l++)
      m_set(&all, l);
   exec_cf_list(st, &st->impl->body, all, NULL);
   if (r->alloc_done && (r->alloc_vtx > (st->safe_export_bound ? st->safe_export_bound : st->nir->info.mesh.max_vertices_out) ||
                         r->alloc_prm > st->nir->info.mesh.max_primitives_out))
      r->count_over_max = true;

   /* Application-level undefined behaviour: a primitive of the workgroup whose staged
    * index is undefined or not below the vertex count (the expansion then copies an
    * arbitrary LDS location). Checked on the final LDS: the staging is not written
    * after the application part. */
   const uint32_t *ix = st->index_staging;
   if (ix[1] && ix[2]) {
      for (unsigned p = 0; p < MAX_LANES && p < st->nir->info.mesh.max_primitives_out; p++) {
         for (unsigned c = 0; c < mesa_vertices_per_prim(st->nir->info.mesh.primitive_type); c++) {
            uint64_t addr = ix[0] + (uint64_t)p * ix[1] + c * ix[2], x = 0;
            bool defined = addr + ix[2] <= LDS_BYTES;
            for (unsigned i = 0; defined && i < ix[2]; i++) {
               defined &= st->lds_def[addr + i];
               x |= (uint64_t)st->lds[addr + i] << (8 * i);
            }
            r->logical[p][c] = x;
            r->logical_defined[p][c] = defined;
         }
      }
   }
   if (r->alloc_done && ix[1] && ix[2]) {
      unsigned prims = r->alloc_vtx ? r->alloc_prm : 0;
      if (r->alloc_vtx == 1 && r->alloc_prm == 1 && dummy_matches(r, st->lanes))
         prims = 0; /* GFX10 fully-culled workaround: no application primitive. */
      for (unsigned p = 0; p < prims && !r->bad_index; p++) {
         for (unsigned c = 0; c < mesa_vertices_per_prim(st->nir->info.mesh.primitive_type); c++) {
            uint64_t addr = ix[0] + (uint64_t)p * ix[1] + c * ix[2];
            uint64_t x = 0;
            bool defined = addr + ix[2] <= LDS_BYTES;
            for (unsigned i = 0; defined && i < ix[2]; i++) {
               defined &= st->lds_def[addr + i];
               x |= (uint64_t)st->lds[addr + i] << (8 * i);
            }
            if (!defined || x >= ix[3]) {
               r->bad_index = true;
               break;
            }
         }
      }
   }
}

static nir_shader *
load_shader(const char *path, unsigned *wave, unsigned *lanes, uint32_t index_staging[4], uint32_t *compact_flags,
            uint64_t *pp_params, uint32_t *clipcull)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      perror(path);
      exit(2);
   }
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   rewind(f);
   uint8_t *data = malloc(n);
   if (fread(data, 1, n, f) != (size_t)n)
      exit(2);
   fclose(f);
   uint32_t header[12] = {0};
   memcpy(header, data, 8 * sizeof(uint32_t));
   size_t header_size = 8 * sizeof(uint32_t);
   if (header[0] == 0x3352534d) { /* "MSR3" */
      memcpy(header, data, sizeof(header));
      header_size = sizeof(header);
   } else if (header[0] != 0x3252534d) {
      fprintf(stderr, "%s: not a BC250_MESH_NIR_DUMP file\n", path);
      exit(2);
   }
   *wave = header[1];
   *lanes = header[2];
   memcpy(index_staging, &header[3], 4 * sizeof(uint32_t));
   *compact_flags = header[7];
   *pp_params = header[8] | (uint64_t)header[9] << 32;
   *clipcull = header[10];
   struct blob_reader br;
   blob_reader_init(&br, data + header_size, n - header_size);
   static const nir_shader_compiler_options options = {0};
   nir_shader *nir = nir_deserialize(NULL, &options, &br);
   free(data);
   return nir;
}

static state_t *
make_state(const char *path, bool list)
{
   state_t *st = calloc(1, sizeof(*st));
   st->nir = load_shader(path, &st->wave, &st->lanes, st->index_staging, &st->compact_flags, &st->pp_params,
                         &st->clipcull);
   if (sgpr_model)
      nir_custom_divergence_analysis(st->nir, nir_divergence_ignore_undef_if_phi_srcs);
   st->impl = nir_shader_get_entrypoint(st->nir);
   nir_index_ssa_defs(st->impl);
   st->vals = calloc(st->impl->ssa_alloc, sizeof(value_t));
   st->pred = calloc(MAX_LANES, sizeof(nir_block *));
   st->lds_size = st->nir->info.shared_size;
   st->generator_primitives = st->nir->info.mesh.max_primitives_out;
   st->api_lanes = st->nir->info.workgroup_size[0] * st->nir->info.workgroup_size[1] * st->nir->info.workgroup_size[2];
   st->unknown = _mesa_hash_table_create(NULL, _mesa_hash_string, _mesa_key_string_equal);
   st->list = list;
   if (st->lanes > MAX_LANES || !st->wave) {
      fprintf(stderr, "%s: unsupported workgroup %u / wave %u\n", path, st->lanes, st->wave);
      exit(2);
   }
   return st;
}

static bool
ub(const result_t *r)
{
   return r->oob || r->race || r->runaway || r->unsupported || r->exp_overflow || r->count_over_max || r->bad_index ||
          r->oob_soft;
}

static void
flags_str(const result_t *r, char *buf, size_t n)
{
   snprintf(buf, n, "%s%s%s%s%s%s%s%s%s", r->oob ? " oob/undefined-branch" : "", r->oob_soft ? " lds-oob(soft)" : "",
            r->race ? " race" : "",
            r->runaway ? " runaway" : "", r->unsupported ? " unsupported" : "",
            r->exp_overflow ? " export-overflow" : "", r->count_over_max ? " count>max" : "",
            r->bad_index ? " invalid-index" : "", r->why);
}

/* The exact comparison of A and B (see the header). Returns true on a mismatch. */
static bool
compare_exact(const state_t *a, const result_t *ra, const result_t *rb, char *why, size_t n,
              uint64_t *defined_channels)
{
   if (ra->alloc_done != rb->alloc_done || ra->alloc_vtx != rb->alloc_vtx || ra->alloc_prm != rb->alloc_prm ||
       ra->alloc_count != rb->alloc_count) {
      snprintf(why, n, "alloc %u/%u (%u) vs %u/%u (%u)", ra->alloc_vtx, ra->alloc_prm, ra->alloc_count,
               rb->alloc_vtx, rb->alloc_prm, rb->alloc_count);
      return true;
   }
   if (ra->side_hash != rb->side_hash || ra->side_count != rb->side_count) {
      snprintf(why, n, "side effects differ (%u vs %u)", ra->side_count, rb->side_count);
      return true;
   }
   for (unsigned l = 0; l < a->lanes; l++) {
      if (ra->nexp[l] != rb->nexp[l]) {
         snprintf(why, n, "lane %u: %u vs %u exports", l, ra->nexp[l], rb->nexp[l]);
         return true;
      }
      for (unsigned k = 0; k < ra->nexp[l]; k++) {
         const export_t *x = &ra->exp[l][k], *y = &rb->exp[l][k];
         if (x->target != y->target || x->mask != y->mask || x->flags != y->flags) {
            snprintf(why, n, "lane %u export %u: target/mask/flags %u/%x/%x vs %u/%x/%x", l, k, x->target, x->mask,
                     x->flags, y->target, y->mask, y->flags);
            return true;
         }
         for (unsigned c = 0; c < 4; c++) {
            if (!(x->mask & (1u << c)) || !known_bits(x->undef[c]))
               continue;
            (*defined_channels)++;
            clip_channels += is_clip_export(x->target);
            if (export_channel_differs(x, y, c)) {
               snprintf(why, n, "lane %u export %u (target %u) channel %u: %08x vs %08x%s", l, k, x->target, c,
                        x->v[c], y->v[c], y->undef[c] ? " (undefined)" : "");
               return true;
            }
         }
      }
   }
   return false;
}

/* --bary-ref (see the header). The two reference parameters of B are ref0 < ref1 (parameter
 * numbers); B's other parameters are renumbered as if they were absent. Returns true on a
 * mismatch. */
#define EXP_PARAM 32 /* V_008DFC_SQ_EXP_PARAM */
static bool
strip_bary_ref(unsigned lanes, const result_t *ra, result_t *rb, unsigned ref0, unsigned ref1, char *why, size_t n,
               uint64_t *checked)
{
   for (unsigned l = 0; l < lanes; l++) {
      const export_t *pos = NULL;
      bool a_params = false;
      for (unsigned k = 0; k < rb->nexp[l]; k++) {
         if (rb->exp[l][k].target == 12 /* POS0 */)
            pos = &rb->exp[l][k];
      }
      for (unsigned k = 0; k < ra->nexp[l]; k++)
         a_params |= ra->exp[l][k].target >= EXP_PARAM;
      unsigned refs = 0, kept = 0;
      export_t tmp[MAX_EXPORTS];
      for (unsigned k = 0; k < rb->nexp[l]; k++) {
         const export_t *e = &rb->exp[l][k];
         if (e->target < EXP_PARAM || (e->target != EXP_PARAM + ref0 && e->target != EXP_PARAM + ref1)) {
            tmp[kept] = *e;
            if (e->target >= EXP_PARAM)
               tmp[kept].target -= (e->target > EXP_PARAM + ref0) + (e->target > EXP_PARAM + ref1);
            kept++;
            continue;
         }
         refs++;
         if (!pos) {
            snprintf(why, n, "lane %u: reference parameter %u without a position export", l, e->target - EXP_PARAM);
            return true;
         }
         if (e->mask != pos->mask) {
            snprintf(why, n, "lane %u: reference parameter %u mask %x, position mask %x", l, e->target - EXP_PARAM,
                     e->mask, pos->mask);
            return true;
         }
         for (unsigned c = 0; c < 4; c++) {
            if (!(pos->mask & (1u << c)) || pos->undef[c])
               continue;
            (*checked)++;
            if (e->undef[c] || e->v[c] != pos->v[c]) {
               snprintf(why, n, "lane %u: reference parameter %u channel %u %08x%s, position %08x", l,
                        e->target - EXP_PARAM, c, e->v[c], e->undef[c] ? " (undefined)" : "", pos->v[c]);
               return true;
            }
         }
      }
      if (refs != 0 && refs != 2) {
         snprintf(why, n, "lane %u: %u reference parameters", l, refs);
         return true;
      }
      if (a_params && refs != 2) {
         snprintf(why, n, "lane %u: exports parameters in A but %u reference parameters in B", l, refs);
         return true;
      }
      memcpy(rb->exp[l], tmp, kept * sizeof(export_t));
      rb->nexp[l] = kept;
   }
   return false;
}

/* --bary-ref: B = A + two reference parameters. Their numbers are not always the last two (the
 * parameter order follows the output slots: the references take two free generic slots, and
 * e.g. clip/cull distances read by the fragment shader come after them), so every pair of B's
 * parameter numbers is tried; B matches when one pair strips to exactly A. The mismatch reported
 * is that of the pair right after A's parameters. Returns true on a mismatch. */
static bool
compare_bary_ref(const state_t *a, const result_t *ra, const result_t *rb, char *why, size_t n, uint64_t *checked,
                 uint64_t *defined_channels)
{
   static result_t tmp;
   unsigned na = 0, nb = 0;
   for (unsigned l = 0; l < a->lanes; l++) {
      for (unsigned k = 0; k < ra->nexp[l]; k++)
         if (ra->exp[l][k].target >= EXP_PARAM)
            na = MAX2(na, ra->exp[l][k].target - EXP_PARAM + 1);
      for (unsigned k = 0; k < rb->nexp[l]; k++)
         if (rb->exp[l][k].target >= EXP_PARAM)
            nb = MAX2(nb, rb->exp[l][k].target - EXP_PARAM + 1);
   }
   char first_why[512] = "";
   bool tried_first = false;
   const uint64_t clip_before = clip_channels;
   /* The pair right after A's parameters first (the usual layout), then every other pair. */
   for (unsigned pass = 0; pass < 2; pass++) {
      for (unsigned r0 = 0; r0 + 1 < MAX2(nb, 2); r0++) {
         for (unsigned r1 = r0 + 1; r1 < MAX2(nb, 2); r1++) {
            const bool usual = r0 == na && r1 == na + 1;
            if ((pass == 0) != usual)
               continue;
            memcpy(&tmp, rb, sizeof(tmp));
            clip_channels = clip_before;
            char w[512] = "";
            uint64_t c = 0, d = 0;
            bool bad = strip_bary_ref(a->lanes, ra, &tmp, r0, r1, w, sizeof(w), &c);
            if (!bad)
               bad = compare_exact(a, ra, &tmp, w, sizeof(w), &d);
            if (!bad) {
               *checked += c;
               *defined_channels += d;
               return false;
            }
            if (!tried_first) {
               tried_first = true;
               snprintf(first_why, sizeof(first_why), "%s", w);
            }
         }
      }
   }
   clip_channels = clip_before;
   snprintf(why, n, "%s", first_why[0] ? first_why : "no reference parameter pair");
   return true;
}

/* ---- --autocull ---- */

#define EXP_POS 12  /* V_008DFC_SQ_EXP_POS */
#define EXP_PRIM 20 /* V_008DFC_SQ_EXP_PRIM */

/* The culling state of a seed: 15% no face culling (the runtime skip), 15% both faces
 * (rasterizer discard: everything culled), 35% back, 35% front; the rest independent. */
static void
make_cullstate(uint64_t seed, cullstate_t *cs)
{
   static const float scale[][2] = {{960.0f, -540.0f}, {960.0f, 540.0f}, {1.0f, 1.0f}, {4096.0f, 4096.0f},
                                    {0.25f, -0.25f}, {1e-3f, 1e-3f}, {64.0f, 16.0f}, {-32.0f, 32.0f}};
   static const float offset[][2] = {{960.0f, 540.0f}, {0.0f, 0.0f}, {0.5f, 0.5f}, {-3.25f, 7.5f}};
   static const float prec[] = {1.0f / 256.0f, 1.0f / 1024.0f, 1.0f / 16.0f, 0.5f};
   uint64_t h = mix64(seed * 0x51ed27ull + 0xc0110ull);
   unsigned faces = h % 20;
   cs->on = true;
   cs->front = faces >= 3 && (faces < 6 || faces >= 13);
   cs->back = faces >= 3 && faces < 13;
   cs->ccw = (h >> 8) & 1;
   cs->small = (h >> 9) & 1;
   unsigned si = (h >> 12) % 8, oi = (h >> 16) % 4;
   cs->vp[0] = scale[si][0];
   cs->vp[1] = scale[si][1];
   cs->vp[2] = offset[oi][0];
   cs->vp[3] = offset[oi][1];
   cs->prec = prec[(h >> 20) % 4];
}

static unsigned ref_exec_mode;
/* --mutate N (oracle self-check, must fail): 1 flips the front face in the reference decision,
 * 2 drops its small-primitive test, 3 its frustum test, 4 its w test, 5 its clip/cull distance
 * test. */
static unsigned ref_mutate;

/* One 32-bit float operation with NIR's constant folding (same rounding, denormal and NaN
 * rules as the interpreted shaders). */
static float
fop(nir_op op, float x, float y, float z)
{
   nir_const_value v[3], d;
   memset(v, 0, sizeof(v));
   v[0].f32 = x;
   v[1].f32 = y;
   v[2].f32 = z;
   nir_const_value *srcs[3] = {&v[0], &v[1], &v[2]};
   nir_eval_const_opcode(op, &d, NULL, 1, 32, srcs, ref_exec_mode);
   return d.f32;
}

static bool
fcmp(nir_op op, float x, float y)
{
   nir_const_value v[2], d;
   memset(v, 0, sizeof(v));
   v[0].f32 = x;
   v[1].f32 = y;
   nir_const_value *srcs[2] = {&v[0], &v[1]};
   nir_eval_const_opcode(op, &d, NULL, 1, 32, srcs, ref_exec_mode);
   return d.b;
}

enum { CULL_KEPT, CULL_W, CULL_FACE, CULL_FRUSTUM, CULL_SMALL, CULL_DIST, CULL_REASONS };

/* ac_nir_cull_primitive for a triangle (ac_nir_cull.c), skip_viewport_state_culling and
 * use_point_tri_intersection off, with the face culling tests reached only when front or
 * back face culling is enabled (the autocull epilogue's runtime branch). clip = the exported
 * clip-space positions. Returns CULL_KEPT or the first test that culls. */
static unsigned
ref_cull_triangle(const float clip[3][4], const cullstate_t *cs)
{
   float pos[3][2], w[3];
   bool all_w_nonpos = true, w_reflection = false, any_w_negative = false;
   for (unsigned v = 0; v < 3; v++) {
      w[v] = clip[v][3];
      pos[v][0] = fop(nir_op_fdiv, clip[v][0], w[v], 0);
      pos[v][1] = fop(nir_op_fdiv, clip[v][1], w[v], 0);
      const bool neg_w = fcmp(nir_op_flt, w[v], 0.0f);
      w_reflection ^= neg_w;
      any_w_negative |= neg_w;
      all_w_nonpos &= fcmp(nir_op_fgeu, 0.0f, w[v]);
   }
   if (all_w_nonpos && ref_mutate != 4)
      return CULL_W;

   const float t0 = fop(nir_op_fsub, pos[2][0], pos[0][0], 0), t1 = fop(nir_op_fsub, pos[1][1], pos[0][1], 0);
   const float t2 = fop(nir_op_fsub, pos[0][0], pos[1][0], 0), t3 = fop(nir_op_fsub, pos[0][1], pos[2][1], 0);
   float det = fop(nir_op_fsub, fop(nir_op_fmul, t0, t1, 0), fop(nir_op_fmul, t2, t3, 0), 0);
   if (w_reflection)
      det = fop(nir_op_fneg, det, 0, 0);
   const bool front_facing_ccw = fcmp(nir_op_flt, 0.0f, det);
   const bool zero_area = fcmp(nir_op_feq, det, 0.0f);
   const bool front_facing = (front_facing_ccw == cs->ccw) != (ref_mutate == 1);
   bool face_culled = (front_facing ? cs->front : cs->back) || zero_area;
   face_culled = face_culled && isfinite(det);
   if (face_culled)
      return CULL_FACE;

   float bmin[2], bmax[2];
   bool outside = false;
   for (unsigned c = 0; c < 2; c++) {
      bmin[c] = fop(nir_op_fmin, pos[0][c], fop(nir_op_fmin, pos[1][c], pos[2][c], 0), 0);
      bmax[c] = fop(nir_op_fmax, pos[0][c], fop(nir_op_fmax, pos[1][c], pos[2][c], 0), 0);
      outside |= fcmp(nir_op_flt, bmax[c], -1.0f) || fcmp(nir_op_flt, 1.0f, bmin[c]);
   }
   bool small = false;
   if (cs->small) {
      for (unsigned c = 0; c < 2; c++) {
         float mn = fop(nir_op_ffma_weak, bmin[c], cs->vp[c], cs->vp[2 + c]);
         float mx = fop(nir_op_ffma_weak, bmax[c], cs->vp[c], cs->vp[2 + c]);
         mn = fop(nir_op_fsub, mn, cs->prec, 0);
         mx = fop(nir_op_fadd, mx, cs->prec, 0);
         small |= fcmp(nir_op_feq, fop(nir_op_fround_even, mn, 0, 0), fop(nir_op_fround_even, mx, 0, 0));
      }
   }
   if (ref_mutate == 2)
      small = false;
   if (ref_mutate == 3)
      outside = false;
   if ((outside || small) && !any_w_negative)
      return outside ? CULL_FRUSTUM : CULL_SMALL;
   return CULL_KEPT;
}

/* --cost buckets: culled share of the workgroup's live primitives (0, 1-25%, 26-50%, 51-75%,
 * 76-99%, 100%), plus workgroups run without face culling (the runtime skip). */
enum { COST_BUCKETS = 7 };
static const char *cost_bucket_name[COST_BUCKETS] = {"culled0", "culled1-25", "culled26-50", "culled51-75",
                                                     "culled76-99", "culled100", "skip"};

typedef struct {
   uint64_t wg_empty, wg_none_culled, wg_partial, wg_all_culled, wg_skip;
   uint64_t prims, culled[CULL_REASONS], ref_undefined;
   uint64_t cost_wg[COST_BUCKETS], cost_prims[COST_BUCKETS], cost_kept[COST_BUCKETS];
   uint64_t cost_a_wave[COST_BUCKETS][COST_KINDS], cost_b_wave[COST_BUCKETS][COST_KINDS];
   uint64_t cost_a_lane[COST_BUCKETS][COST_KINDS], cost_b_lane[COST_BUCKETS][COST_KINDS];
} cull_stats_t;

static void
cost_add(cull_stats_t *stats, unsigned bucket, const result_t *ra, const result_t *rb, unsigned live, unsigned kept)
{
   stats->cost_wg[bucket]++;
   stats->cost_prims[bucket] += live;
   stats->cost_kept[bucket] += kept;
   for (unsigned k = 0; k < COST_KINDS; k++) {
      stats->cost_a_wave[bucket][k] += ra->cost_wave[k];
      stats->cost_b_wave[bucket][k] += rb->cost_wave[k];
      stats->cost_a_lane[bucket][k] += ra->cost_lane[k];
      stats->cost_b_lane[bucket][k] += rb->cost_lane[k];
   }
}

static const export_t *
find_export(const result_t *r, unsigned lane, bool prim)
{
   for (unsigned k = 0; k < r->nexp[lane]; k++) {
      if ((r->exp[lane][k].target == EXP_PRIM) == prim)
         return &r->exp[lane][k];
   }
   return NULL;
}

static unsigned
count_exports(const result_t *r, unsigned lane, bool prim)
{
   unsigned n = 0;
   for (unsigned k = 0; k < r->nexp[lane]; k++)
      n += (r->exp[lane][k].target == EXP_PRIM) == prim;
   return n;
}

/* ---- Clip/cull distance forms (see the header) ---- */
#define CC_CLIP(x) ((x) & 0xfu)
#define CC_CULL(x) (((x) >> 4) & 0xfu)
#define CC_CULLED(x) (((x) >> 8) & 1u)
#define EXP_FLAG_DONE 2u  /* AC_EXP_FLAG_DONE */
#define EXP_FLAG_VALID 4u /* AC_EXP_FLAG_VALID_MASK */
static bool strip_active;   /* B exports only part of A's clip/cull channels */
static bool strip_a_known;  /* A's header tells its clip / cull components */
static unsigned strip_a_clip;
static bool strip_keep_clip, strip_keep_cull;
static bool b_always_cull;  /* B culls its cull distances in the shader: no runtime skip */
static unsigned b_num_pos;  /* position vectors B declares (its fully-culled dummy exports them) */

/* The position vectors a dump declares: POS0, the misc vector, the packed clip/cull distances. */
static unsigned
num_pos_of(const state_t *st)
{
   const uint64_t written = st->nir->info.outputs_written & ~st->nir->info.per_primitive_outputs;
   const bool misc = written & (VARYING_BIT_PSIZ | VARYING_BIT_EDGE | VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT |
                                VARYING_BIT_PRIMITIVE_SHADING_RATE);
   const unsigned cc = CC_CLIP(st->clipcull) + CC_CULL(st->clipcull);
   return 1 + misc + (cc > 0) + (cc > 4);
}

static bool
is_fully_culled_dummy(const result_t *r, unsigned lane)
{
   if (lane || r->alloc_vtx != 1 || r->alloc_prm != 1)
      return false;
   const export_t *ve = find_export(r, 0, false);
   return ve && ve->target == EXP_POS && !ve->undef[0] && ve->v[0] == 0xffffffffu && !ve->undef[3] &&
          ve->v[1] == 0xffffffffu && ve->v[2] == 0xffffffffu && ve->v[3] == 0xffffffffu;
}

/* The GFX10 fully-culled workaround as B exports it: lane 0 exports the primitive (0, 0, 0) and
 * b_num_pos position vectors (POS0 NaN, then zeros, DONE on the last), other lanes nothing. */
static bool
dummy_matches(const result_t *rb, unsigned lanes)
{
   for (unsigned l = 1; l < lanes; l++) {
      if (rb->nexp[l])
         return false;
   }
   if (rb->nexp[0] != 1 + b_num_pos || count_exports(rb, 0, true) != 1)
      return false;
   const export_t *pe = find_export(rb, 0, true);
   if (pe->mask != 1 || pe->undef[0] || pe->v[0])
      return false;
   unsigned pos = 0;
   for (unsigned k = 0; k < rb->nexp[0]; k++) {
      const export_t *e = &rb->exp[0][k];
      if (e->target == EXP_PRIM)
         continue;
      const bool last = pos == b_num_pos - 1;
      const uint32_t flags = last ? EXP_FLAG_DONE : 0;
      const uint32_t v = pos ? 0 : 0xffffffffu;
      if (e->target != EXP_POS + pos || e->mask != 0xf ||
          (pos ? e->flags != flags : e->flags != (b_num_pos > 1 ? EXP_FLAG_VALID : EXP_FLAG_DONE)))
         return false;
      for (unsigned c = 0; c < 4; c++) {
         if (e->undef[c] || e->v[c] != v)
            return false;
      }
      pos++;
   }
   return pos == b_num_pos;
}

/* out = in with the clip/cull channels B does not export removed (see the header). */
static void
strip_dist(const result_t *in, result_t *out, unsigned lanes)
{
   memcpy(out, in, sizeof(*out));
   if (!strip_active)
      return;
   for (unsigned l = 0; l < lanes; l++) {
      if (!in->nexp[l])
         continue;
      export_t list[MAX_EXPORTS + 2];
      unsigned n = 0, nch = 0, kept = 0;
      int first_clip = -1;
      uint32_t ch_v[8], kv[8];
      uint8_t ch_u[8], ku[8];
      bool had_done = false;
      const bool dummy = is_fully_culled_dummy(in, l);
      for (unsigned k = 0; k < in->nexp[l]; k++) {
         const export_t *e = &in->exp[l][k];
         if (e->target >= EXP_POS && e->target < EXP_POS + 4)
            had_done |= (e->flags & EXP_FLAG_DONE) != 0;
         if (dummy && e->target > EXP_POS && e->target < EXP_POS + 4)
            continue; /* rebuilt below */
         if (!dummy && is_clip_export(e->target) && e->target < EXP_POS + 4) {
            if (first_clip < 0)
               first_clip = n++;
            for (unsigned c = 0; c < 4; c++) {
               if ((e->mask & (1u << c)) && nch < 8) {
                  ch_v[nch] = e->v[c];
                  ch_u[nch++] = e->undef[c];
               }
            }
            continue;
         }
         list[n++] = *e;
      }
      if (dummy) {
         /* B's fully-culled dummy (dummy_matches). */
         n = 0;
         for (unsigned k = 0; k < in->nexp[l]; k++) {
            if (in->exp[l][k].target == EXP_PRIM)
               list[n++] = in->exp[l][k];
         }
         for (unsigned p = 0; p < b_num_pos && n < MAX_EXPORTS; p++) {
            export_t e = {.target = EXP_POS + p, .mask = 0xf};
            e.flags = p == b_num_pos - 1 ? EXP_FLAG_DONE : 0;
            if (!p && b_num_pos > 1)
               e.flags = EXP_FLAG_VALID;
            for (unsigned c = 0; c < 4; c++)
               e.v[c] = p ? 0 : 0xffffffffu;
            list[n++] = e;
         }
      } else {
         for (unsigned i = 0; i < nch; i++) {
            const bool clip = strip_a_known && i < strip_a_clip;
            if (strip_a_known && (clip ? strip_keep_clip : strip_keep_cull)) {
               kv[kept] = ch_v[i];
               ku[kept++] = ch_u[i];
            }
         }
         if (first_clip >= 0) {
            export_t vec[2];
            unsigned nvec = 0;
            for (unsigned i = 0; i < kept; i += 4, nvec++) {
               memset(&vec[nvec], 0, sizeof(vec[nvec]));
               vec[nvec].target = clip_target + nvec;
               vec[nvec].mask = BITFIELD_MASK(MIN2(kept - i, 4));
               for (unsigned c = 0; c < 4; c++) {
                  vec[nvec].undef[c] = 1;
                  if (i + c < kept) {
                     vec[nvec].v[c] = kv[i + c];
                     vec[nvec].undef[c] = ku[i + c];
                  }
               }
            }
            memmove(&list[first_clip + nvec], &list[first_clip + 1], (n - first_clip - 1) * sizeof(export_t));
            memcpy(&list[first_clip], vec, nvec * sizeof(export_t));
            n = n - 1 + nvec;
         }
         /* DONE on the last position export. */
         int last_pos = -1;
         for (unsigned k = 0; k < n; k++) {
            if (list[k].target >= EXP_POS && list[k].target < EXP_POS + 4) {
               list[k].flags &= ~EXP_FLAG_DONE;
               last_pos = k;
            }
         }
         if (had_done && last_pos >= 0)
            list[last_pos].flags |= EXP_FLAG_DONE;
      }
      if (n > MAX_EXPORTS)
         n = MAX_EXPORTS;
      out->nexp[l] = n;
      memcpy(out->exp[l], list, n * sizeof(export_t));
   }
}

/* Sets up the stripping for A and B (see the header); returns false when B's clip/cull form
 * cannot be derived from A's. */
static bool
strip_setup(const state_t *a, const state_t *b)
{
   b_always_cull = CC_CULLED(b->clipcull);
   b_num_pos = num_pos_of(b);
   const unsigned bc = CC_CLIP(b->clipcull), bk = CC_CULL(b->clipcull);
   if (clip_target == ~0u)
      return bc + bk == 0;
   strip_a_known = a->clipcull != 0;
   const unsigned ac = CC_CLIP(a->clipcull), ak = CC_CULL(a->clipcull);
   strip_a_clip = ac;
   if (!strip_a_known) {
      /* An older dump: only "B exports none of them" can be derived. */
      strip_active = bc + bk == 0;
      strip_keep_clip = strip_keep_cull = false;
      return true; /* otherwise compared as it is */
   }
   if (ac == bc && ak == bk)
      return true;
   if ((bc != ac && bc) || (bk != ak && bk))
      return false;
   strip_active = true;
   strip_keep_clip = bc == ac;
   strip_keep_cull = bk == ak;
   return true;
}

/* B (autocull) against A (no autocull) under the culling state cs. Returns 1 on a mismatch,
 * 2 when B is undefined (see ub) although the reference is not, 0 when equal, -1 when the
 * reference culling decision is undefined (an exported position channel of a live primitive
 * is undefined in A, e.g. the application never wrote that vertex): the seed is skipped. */
/* Clip and cull distances (ms_autocull_accept, as VS NGG culling): a triangle is culled when one
 * exported distance is negative (flt) at all three corners. The exported distances are packed in
 * the same order for every vertex. Returns 0 (*culled set), -1 when a distance is undefined, 1 when
 * the corners export different numbers of distances (why is set). --mutate 6 drops the test. */
static int
ref_dist_culled(const state_t *a, const result_t *ra, const unsigned corner[3], unsigned prim, bool *culled,
                char *why, size_t n)
{
   uint32_t dist[3][8];
   unsigned ndist[3] = {0, 0, 0};
   *culled = false;
   for (unsigned c = 0; c < 3 && clip_target != ~0u; c++) {
      const unsigned vl = corner[c];
      if (vl >= ra->alloc_vtx || vl >= a->lanes)
         continue;
      for (unsigned k = 0; k < ra->nexp[vl]; k++) {
         const export_t *e = &ra->exp[vl][k];
         if (!is_clip_export(e->target))
            continue;
         for (unsigned ch = 0; ch < 4; ch++) {
            if (!(e->mask & (1u << ch)) || ndist[c] == 8)
               continue;
            if (e->undef[ch]) {
               snprintf(why, n, "primitive %u corner %u: clip/cull distance of vertex lane %u undefined", prim, c, vl);
               return -1;
            }
            dist[c][ndist[c]++] = e->v[ch];
         }
      }
   }
   if (ndist[0] != ndist[1] || ndist[0] != ndist[2]) {
      snprintf(why, n, "primitive %u: corners export %u/%u/%u clip/cull distances", prim, ndist[0], ndist[1],
               ndist[2]);
      return 1;
   }
   for (unsigned d = 0; d < ndist[0]; d++) {
      bool all_negative = true;
      for (unsigned c = 0; c < 3; c++) {
         float f;
         memcpy(&f, &dist[c][d], 4);
         all_negative &= fcmp(nir_op_flt, f, 0.0f);
      }
      *culled |= all_negative;
   }
   if (ref_mutate == 6)
      *culled = false;
   return 0;
}

static int
compare_autocull(const state_t *a, const result_t *ra, const result_t *rax, const result_t *rb, const cullstate_t *cs,
                 char *why, size_t n, uint64_t *defined_channels, cull_stats_t *stats)
{
   if (!cs->front && !cs->back && !b_always_cull) {
      if (ub(rb)) {
         char fb[512];
         flags_str(rb, fb, sizeof(fb));
         snprintf(why, n, "candidate:%s", fb);
         return 2;
      }
      stats->wg_skip++;
      const int r = compare_exact(a, rax, rb, why, n, defined_channels);
      if (!r && ra->alloc_prm && !(ra->alloc_vtx == 1 && ra->alloc_prm == 1))
         cost_add(stats, COST_BUCKETS - 1, ra, rb, ra->alloc_prm, ra->alloc_prm);
      return r;
   }

   /* A's live primitives, in export order: lane p exports primitive p. */
   unsigned live[MAX_LANES], corner[MAX_LANES][3], nlive = 0, survivors[MAX_LANES], ns = 0;
   const bool a_dummy = ra->alloc_vtx == 1 && ra->alloc_prm == 1;
   const unsigned a_prims = a_dummy ? 0 : ra->alloc_prm;
   for (unsigned p = 0; p < a_prims && p < a->lanes; p++) {
      const export_t *e = find_export(ra, p, true);
      if (!e || e->undef[0]) {
         snprintf(why, n, "primitive %u: %s primitive export", p, e ? "undefined" : "no");
         return -1;
      }
      if (e->v[0] & 0x80000000u)
         continue; /* null primitive (application CullPrimitive) */
      for (unsigned c = 0; c < 3; c++)
         corner[nlive][c] = (e->v[0] >> (10 * c)) & 0x1ff;
      live[nlive++] = p;
   }
   unsigned reasons[MAX_LANES];
   for (unsigned i = 0; i < nlive; i++) {
      bool dist_culled = false;
      const int dr = ref_dist_culled(a, ra, corner[i], live[i], &dist_culled, why, n);
      if (dr)
         return dr;
      if (dist_culled) {
         reasons[i] = CULL_DIST;
         continue;
      }
      float clip[3][4];
      for (unsigned c = 0; c < 3; c++) {
         const unsigned vl = corner[i][c];
         const export_t *e = vl < ra->alloc_vtx && vl < a->lanes ? find_export(ra, vl, false) : NULL;
         if (!e || e->target != EXP_POS || e->mask != 0xf) {
            snprintf(why, n, "primitive %u corner %u: vertex lane %u has no position export", live[i], c, vl);
            return -1;
         }
         for (unsigned k = 0; k < 4; k++) {
            if (e->undef[k]) {
               snprintf(why, n, "primitive %u corner %u: position channel %u of vertex lane %u undefined", live[i], c,
                        k, vl);
               return -1;
            }
            memcpy(&clip[c][k], &e->v[k], 4);
         }
      }
      reasons[i] = ref_cull_triangle(clip, cs);
   }
   /* The reference decision is defined: B must not be undefined anywhere. */
   if (ub(rb)) {
      char fb[512];
      flags_str(rb, fb, sizeof(fb));
      snprintf(why, n, "candidate:%s", fb);
      return 2;
   }
   if (ra->side_hash != rb->side_hash || ra->side_count != rb->side_count) {
      snprintf(why, n, "side effects differ (%u vs %u)", ra->side_count, rb->side_count);
      return 1;
   }
   for (unsigned i = 0; i < nlive; i++) {
      stats->culled[reasons[i]]++;
      if (reasons[i] == CULL_KEPT)
         survivors[ns++] = i;
   }
   stats->prims += nlive;
   if (nlive) {
      const unsigned culled = nlive - ns;
      const unsigned bucket = !culled ? 0 : culled == nlive ? 5 : 1 + MIN2(3, (4 * culled - 1) / nlive);
      cost_add(stats, bucket, ra, rb, nlive, ns);
   }
   if (!nlive)
      stats->wg_empty++;
   else if (ns == nlive)
      stats->wg_none_culled++;
   else if (ns)
      stats->wg_partial++;
   else
      stats->wg_all_culled++;

   const unsigned exp_vtx = ns ? 3 * ns : 1, exp_prm = ns ? ns : 1;
   if (!rb->alloc_done || rb->alloc_vtx != exp_vtx || rb->alloc_prm != exp_prm || rb->alloc_count != ra->alloc_count) {
      snprintf(why, n, "alloc %u/%u (%u) expected %u/%u (%u) [live %u survivors %u]", rb->alloc_vtx, rb->alloc_prm,
               rb->alloc_count, exp_vtx, exp_prm, ra->alloc_count, nlive, ns);
      return 1;
   }
   for (unsigned l = 0; l < a->lanes; l++) {
      const unsigned bprims = count_exports(rb, l, true), bverts = count_exports(rb, l, false);
      if (!ns) {
         /* GFX10 fully-culled workaround: lane 0 exports primitive (0, 0, 0) and a NaN position
          * (and zero vectors for B's other position exports). */
         (void)bprims;
         const bool ok = l || dummy_matches(rb, a->lanes);
         if (!ok) {
            snprintf(why, n, "all culled: lane %u exports do not match the fully-culled workaround", l);
            return 1;
         }
         continue;
      }
      /* Primitive slot l. */
      if (l < ns) {
         const export_t *pa = find_export(ra, live[survivors[l]], true), *pb = find_export(rb, l, true);
         const uint32_t expect = (3 * l) | (3 * l + 1) << 10 | (3 * l + 2) << 20;
         if (bprims != 1 || pb->mask != pa->mask || pb->flags != pa->flags || pb->undef[0] || pb->v[0] != expect) {
            snprintf(why, n, "lane %u: primitive export %08x (%u exports) expected %08x (survivor %u = primitive %u)",
                     l, pb ? pb->v[0] : 0, bprims, expect, l, live[survivors[l]]);
            return 1;
         }
      } else if (bprims) {
         snprintf(why, n, "lane %u: primitive export beyond the %u survivors", l, ns);
         return 1;
      }
      /* Vertex slot l = corner l % 3 of survivor l / 3: A's exports of that corner. */
      if (l < 3 * ns) {
         const unsigned al = corner[survivors[l / 3]][l % 3];
         const unsigned averts = count_exports(rax, al, false);
         if (averts != bverts) {
            snprintf(why, n, "lane %u: %u vertex exports, A lane %u has %u", l, bverts, al, averts);
            return 1;
         }
         unsigned ka = 0, kb = 0;
         for (unsigned i = 0; i < averts; i++, ka++, kb++) {
            while (rax->exp[al][ka].target == EXP_PRIM)
               ka++;
            while (rb->exp[l][kb].target == EXP_PRIM)
               kb++;
            const export_t *x = &rax->exp[al][ka], *y = &rb->exp[l][kb];
            if (x->target != y->target || x->mask != y->mask || x->flags != y->flags) {
               snprintf(why, n, "lane %u export %u: target/mask/flags %u/%x/%x vs A lane %u %u/%x/%x", l, i, y->target,
                        y->mask, y->flags, al, x->target, x->mask, x->flags);
               return 1;
            }
            for (unsigned c = 0; c < 4; c++) {
               if (!(x->mask & (1u << c)) || !known_bits(x->undef[c]))
                  continue;
               (*defined_channels)++;
               clip_channels += is_clip_export(x->target);
               if (export_channel_differs(x, y, c)) {
                  snprintf(why, n, "lane %u export %u (target %u) channel %u: %08x vs A lane %u %08x%s", l, i,
                           x->target, c, y->v[c], al, x->v[c], y->undef[c] ? " (undefined)" : "");
                  return 1;
               }
            }
         }
      } else if (bverts) {
         snprintf(why, n, "lane %u: vertex export beyond the %u survivor vertices", l, 3 * ns);
         return 1;
      }
   }
   return 0;
}

/* ---- --compact (RADV_BC250_MESH_COMPACT) ----
 * (a) Safety, on every candidate run, whatever the reference does (also for index data the
 *     API forbids): GS_ALLOC_REQ V'/P' within the lanes, the declared maximum and 256; exactly
 *     lanes < V' export one position (vertex) and lanes < P' one primitive; every primitive
 *     is non-null with indices < V'; every vertex < V' is referenced by a primitive;
 *     the backjump (running maximum index before a corner minus the corner's index, in
 *     export order) is <= W. The GFX10 fully-culled dummy (1/1, primitive (0,0,0)) passes.
 * (b) Equality, where the reference is defined: the candidate's primitives in export order,
 *     each as its 3 corners' complete vertex exports, equal the reference's (expanded) live
 *     primitives in order (with --autocull: the survivors of the CPU culling decision; the
 *     dummy when none), bit-exact on every channel the reference defines, except the
 *     per-primitive payload parameters (header pp_params) at corners that do not own it
 *     (header owned corners). Side effects equal. */
static unsigned compact_w = 8;
/* With --safe-direct, compare the same API group across expansion/direct lane
 * counts; require a <=256-slot candidate with its separate proven physical bound
 * and equal logical index domains. --safety-only checks its allocation/export contract without claiming
 * equality across different varying packing. For equality tests, compile both
 * pipelines with VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT; optimized D0 also
 * compares directly. Missing defined reference channels still fail. */
static bool safe_direct_mode;
static bool safe_owned_mode;
static bool all_corner_params;
static bool safety_only;

typedef struct {
   uint64_t runs, safety_runs, safety_fail, dummy, prims_b, verts_b, verts_a, params_a, params_b, bj_max;
   uint64_t gen_runs[NGEN + 1], gen_safety_fail[NGEN + 1], gen_compared[NGEN + 1], gen_tris[NGEN + 1],
      gen_verts[NGEN + 1];
   uint64_t v_hist_max, payload_skipped;
   /* --autocull: live reference primitives culled by the clip/cull distance test. */
   uint64_t culled_dist;
   /* --bary-rot: primitives whose barycentric vertex order was recovered (both provoking
    * conventions, every cyclic rotation), primitives without a defined position, and
    * primitives where a second corner matches the provoking corner (zero screen area). */
   uint64_t rot_prims, rot_undefined, rot_ambiguous;
   /* Work per compared workgroup (the --cost model): A and B instructions issued per wave and
    * active lanes, by kind. */
   uint64_t cost_wg, cost_a_wave[COST_KINDS], cost_b_wave[COST_KINDS], cost_a_lane[COST_KINDS],
      cost_b_lane[COST_KINDS];
} compact_stats_t;

static unsigned
prim_index(uint32_t v, unsigned c)
{
   return (v >> (10 * c)) & 0x1ff;
}

static bool
check_compact_safety(const state_t *b, const result_t *rb, char *why, size_t n, compact_stats_t *cs)
{
   const unsigned corners = mesa_vertices_per_prim(b->nir->info.mesh.primitive_type);
   cs->safety_runs++;
   if (!rb->alloc_done || rb->alloc_count != 1) {
      snprintf(why, n, "safety: GS_ALLOC_REQ issued %u times", rb->alloc_count);
      return true;
   }
   const unsigned V = rb->alloc_vtx, P = rb->alloc_prm;
   const unsigned logical_v = b->nir->info.mesh.max_vertices_out;
   const unsigned physical_bound = safe_direct_mode ?
      (logical_v <= 32 ? MIN2(logical_v, corners * b->nir->info.mesh.max_primitives_out) :
                        corners * b->nir->info.mesh.max_primitives_out) : logical_v;
   if (!V || !P || V > b->lanes || P > b->lanes || V > 256 || P > 256 || V > physical_bound ||
       P > b->nir->info.mesh.max_primitives_out) {
      snprintf(why, n, "safety: GS_ALLOC_REQ %u/%u (lanes %u, max %u/%u)", V, P, b->lanes,
               physical_bound, b->nir->info.mesh.max_primitives_out);
      return true;
   }
   bool referenced[MAX_LANES] = {0};
   bool provoking[MAX_LANES] = {0};
   int hi = -1;
   unsigned bj = 0;
   for (unsigned l = 0; l < b->lanes; l++) {
      const unsigned np = count_exports(rb, l, true), nv = count_exports(rb, l, false);
      unsigned npos0 = 0;
      for (unsigned k = 0; k < rb->nexp[l]; k++)
         npos0 += rb->exp[l][k].target == EXP_POS;
      if (np != (l < P) || npos0 != (l < V) || (l >= V && nv)) {
         snprintf(why, n, "safety: lane %u exports %u primitives / %u vertices (V'=%u P'=%u)", l, np, nv, V, P);
         return true;
      }
   }
   for (unsigned l = 0; l < P; l++) {
      const export_t *pe = find_export(rb, l, true);
      /* Undefined bits (garbage input) still reach the hardware: check the bits. */
      if (pe->v[0] & 0x80000000u) {
         snprintf(why, n, "safety: primitive %u null (culled)", l);
         return true;
      }
      if (safe_direct_mode) {
         for (unsigned c = 0; c < corners; c++)
            hi = MAX2(hi, (int)prim_index(pe->v[0], c));
      }
      if (safe_owned_mode) {
         const unsigned first = prim_index(pe->v[0], 0);
         if (first >= V || provoking[first]) {
            snprintf(why, n, "safety: payload provoking vertex shared by distinct primitives");
            return true;
         }
         provoking[first] = true;
      }
      for (unsigned c = 0; c < corners; c++) {
         const unsigned i = prim_index(pe->v[0], c);
         if (i >= V) {
            snprintf(why, n, "safety: primitive %u corner %u index %u >= V' %u", l, c, i, V);
            return true;
         }
         if (hi >= 0 && hi - (int)i > (int)bj)
            bj = hi - i;
         if ((int)i > hi)
            hi = i;
         referenced[i] = true;
      }
   }
   for (unsigned v = 0; v < V; v++) {
      if (!referenced[v]) {
         snprintf(why, n, "safety (all referenced): vertex %u of %u never referenced", v, V);
         return true;
      }
   }
   if (bj > compact_w) {
      snprintf(why, n, "safety (window): backjump %u > W %u", bj, compact_w);
      return true;
   }
   cs->bj_max = MAX2(cs->bj_max, bj);
   cs->v_hist_max = MAX2(cs->v_hist_max, V);
   return false;
}

static unsigned
count_params(const result_t *r, unsigned lane)
{
   unsigned k = 0;
   for (unsigned i = 0; i < r->nexp[lane]; i++)
      k += r->exp[lane][i].target >= 32 && r->exp[lane][i].target < 64;
   return k;
}

/* --bary-rot (with --compact): the fragment shader's barycentric vertex order recovery
 * (radv_nir_bc250_lower_bary_rotation) on the candidate's shared vertices. The parameter
 * cache holds a triangle's corners in a cyclic rotation r of the API order (raw slot i =
 * API corner (i + r) % 3); the fragment shader reads the position reference of raw slots 1
 * and 2 and, flat, of the provoking vertex (API corner 0, or 2 with the last-vertex
 * convention), and takes the raw slot whose x, y, w bits equal the flat value as the
 * provoking vertex (slot 0 when neither matches). The reference exports are the position
 * (--bary-ref), so the candidate's position exports stand in for them. For every exported
 * primitive, both conventions and every r, the recovered slot of API corner 0 must be
 * (3 - r) % 3, unless a second corner has the provoking corner's x, y, w bits: then both
 * project to one screen point and the primitive has no fragments (counted as ambiguous).
 * Returns true on a failure. */
static bool bary_rot;

static bool
check_bary_rotation(const result_t *rb, unsigned j, unsigned bl[3], char *why, size_t n, compact_stats_t *cs)
{
   uint32_t xyw[3][3];
   for (unsigned c = 0; c < 3; c++) {
      const export_t *e = find_export(rb, bl[c], false);
      if (!e || e->target != EXP_POS || (e->mask & 0xb) != 0xb || e->undef[0] || e->undef[1] || e->undef[3]) {
         cs->rot_undefined++;
         return false;
      }
      xyw[c][0] = e->v[0];
      xyw[c][1] = e->v[1];
      xyw[c][2] = e->v[3];
   }
#define XYW_EQ(a, b) (xyw[a][0] == xyw[b][0] && xyw[a][1] == xyw[b][1] && xyw[a][2] == xyw[b][2])
   for (unsigned last = 0; last < 2; last++) {
      const unsigned pc = last ? 2 : 0;
      if (XYW_EQ((pc + 1) % 3, pc) || XYW_EQ((pc + 2) % 3, pc)) {
         cs->rot_ambiguous++;
         return false;
      }
   }
   for (unsigned last = 0; last < 2; last++) {
      const unsigned pc = last ? 2 : 0;
      for (unsigned r = 0; r < 3; r++) {
         /* raw slot i holds API corner (i + r) % 3 */
         unsigned slot = XYW_EQ((1 + r) % 3, pc) ? 1 : XYW_EQ((2 + r) % 3, pc) ? 2 : 0;
         if (last)
            slot = slot == 2 ? 0 : slot + 1;
         if (slot != (3 - r) % 3) {
            snprintf(why, n, "barycentric rotation: primitive %u (%s vertex, rotation %u) recovered slot %u, expected %u",
                     j, last ? "last" : "first", r, slot, (3 - r) % 3);
            return true;
         }
      }
   }
#undef XYW_EQ
   cs->rot_prims++;
   return false;
}

/* Returns 1 on a mismatch, 0 when equal, -1 when the reference is undefined for this state. */
static int
compare_compact(const state_t *a, const state_t *b, const result_t *ra, const result_t *rax, const result_t *rb,
                const cullstate_t *cull, char *why, size_t n, uint64_t *defined_channels, compact_stats_t *cs)
{
   const unsigned corners = mesa_vertices_per_prim(b->nir->info.mesh.primitive_type);
   if (ra->side_hash != rb->side_hash || ra->side_count != rb->side_count) {
      snprintf(why, n, "side effects differ (%u vs %u)", ra->side_count, rb->side_count);
      return 1;
   }
   /* The reference's live primitives in export order. */
   unsigned live[MAX_LANES], corner[MAX_LANES][3], nlive = 0;
   const bool a_dummy = ra->alloc_vtx == 1 && ra->alloc_prm == 1 && dummy_matches(ra, a->lanes);
   const unsigned a_prims = a_dummy ? 0 : ra->alloc_prm;
   for (unsigned p = 0; p < a_prims && p < a->lanes; p++) {
      const export_t *e = find_export(ra, p, true);
      if (!e || e->undef[0]) {
         snprintf(why, n, "primitive %u: %s primitive export", p, e ? "undefined" : "no");
         return -1;
      }
      if (e->v[0] & 0x80000000u)
         continue;
      for (unsigned c = 0; c < corners; c++)
         corner[nlive][c] = prim_index(e->v[0], c);
      live[nlive++] = p;
   }
   /* Channels whose reference value depends on more than the logical vertex (the per-primitive
    * payload the expansion copies to every corner, possibly packed with per-vertex data): two
    * corners of the same logical vertex export different values. Only those are exempt from
    * the comparison at corners that do not own the payload. The staged index of the expanded
    * vertex a primitive's corner exports is the application's index of (primitive, corner). */
   static bool varies[64][4];
   memset(varies, 0, sizeof(varies));
   for (unsigned i = 0; i < nlive; i++) {
      for (unsigned c = 0; c < corners; c++) {
         const unsigned p = live[i];
         if (!ra->logical_defined[p][c])
            continue;
         for (unsigned j = 0; j <= i; j++) {
            for (unsigned c2 = 0; c2 < corners; c2++) {
               if ((j == i && c2 >= c) || !ra->logical_defined[live[j]][c2] ||
                   ra->logical[live[j]][c2] != ra->logical[p][c])
                  continue;
               const unsigned l1 = corner[i][c], l2 = corner[j][c2];
               for (unsigned k = 0; k < ra->nexp[l1] && k < ra->nexp[l2]; k++) {
                  const export_t *x = &ra->exp[l1][k], *y = &ra->exp[l2][k];
                  if (x->target != y->target || x->target < 32 || x->target >= 96)
                     continue;
                  for (unsigned ch = 0; ch < 4; ch++)
                     varies[x->target - 32][ch] |= !x->undef[ch] && !y->undef[ch] && x->v[ch] != y->v[ch];
               }
               goto next_corner; /* one earlier corner of the same vertex is enough (transitive) */
            }
         }
      next_corner:;
      }
   }
   unsigned expect[MAX_LANES], ns = 0;
   for (unsigned i = 0; i < nlive; i++) {
      if (cull && (cull->front || cull->back || b_always_cull)) {
         float clip[3][4];
         for (unsigned c = 0; c < corners; c++) {
            const unsigned vl = corner[i][c];
            const export_t *e = vl < ra->alloc_vtx && vl < a->lanes ? find_export(ra, vl, false) : NULL;
            if (!e || e->target != EXP_POS || e->mask != 0xf)
               return snprintf(why, n, "reference corner without position"), -1;
            for (unsigned k = 0; k < 4; k++) {
               if (e->undef[k])
                  return snprintf(why, n, "reference position undefined"), -1;
               memcpy(&clip[c][k], &e->v[k], 4);
            }
         }
         bool dist_culled = false;
         const int dr = ref_dist_culled(a, ra, corner[i], live[i], &dist_culled, why, n);
         if (dr)
            return dr;
         if (dist_culled) {
            cs->culled_dist++;
            continue;
         }
         if (ref_cull_triangle(clip, cull) != CULL_KEPT)
            continue;
      }
      expect[ns++] = i;
   }
   if (ref_mutate == 5 && ns > 1) {
      /* Self-check: expect the first two survivors swapped (must report a mismatch). */
      const unsigned t = expect[0];
      expect[0] = expect[1];
      expect[1] = t;
   }
   if (!ns) {
      const bool ok = rb->alloc_vtx == 1 && rb->alloc_prm == 1 && dummy_matches(rb, b->lanes);
      if (!ok) {
         snprintf(why, n, "no live primitive: candidate %u/%u is not the fully-culled dummy", rb->alloc_vtx,
                  rb->alloc_prm);
         return 1;
      }
      cs->dummy++;
      return 0;
   }
   if (rb->alloc_prm != ns) {
      snprintf(why, n, "candidate exports %u primitives, expected %u (live %u)", rb->alloc_prm, ns, nlive);
      return 1;
   }
   const unsigned owned = all_corner_params ? 7 : (b->compact_flags >> 8) & 0x7;
   for (unsigned j = 0; j < ns; j++) {
      const export_t *pb = find_export(rb, j, true), *pa = find_export(ra, live[expect[j]], true);
      if (!pb || pb->mask != pa->mask || pb->flags != pa->flags || pb->undef[0]) {
         snprintf(why, n, "primitive %u: export mask/flags differ or undefined connectivity", j);
         return 1;
      }
      for (unsigned c = 0; c < corners; c++) {
         const unsigned bl = prim_index(pb->v[0], c), al = corner[expect[j]][c];
         const unsigned na = count_exports(rax, al, false), nb = count_exports(rb, bl, false);
         if (na != nb) {
            snprintf(why, n, "primitive %u corner %u: vertex lane %u has %u exports, reference lane %u %u", j, c, bl,
                     nb, al, na);
            return 1;
         }
         unsigned ka = 0, kb = 0;
         for (unsigned i = 0; i < na; i++, ka++, kb++) {
            while (rax->exp[al][ka].target == EXP_PRIM)
               ka++;
            while (rb->exp[bl][kb].target == EXP_PRIM)
               kb++;
            const export_t *x = &rax->exp[al][ka], *y = &rb->exp[bl][kb];
            if (x->target != y->target || (!safe_direct_mode && x->mask != y->mask) || x->flags != y->flags) {
               snprintf(why, n, "primitive %u corner %u export %u: target/mask/flags %u/%x/%x vs reference %u/%x/%x", j,
                        c, i, y->target, y->mask, y->flags, x->target, x->mask, x->flags);
               return 1;
            }
            /* The per-primitive payload only counts at a corner that owns it. */
            const bool payload_param = x->target >= 32 && x->target < 96 && !(owned >> c & 1);
            for (unsigned ch = 0; ch < 4; ch++) {
               if (!(x->mask & (1u << ch)) || !known_bits(x->undef[ch]))
                  continue;
               if (payload_param && varies[x->target - 32][ch]) {
                  cs->payload_skipped++;
                  continue;
               }
               (*defined_channels)++;
               clip_channels += is_clip_export(x->target);
               if (!(y->mask & (1u << ch)) || export_channel_differs(x, y, ch)) {
                  snprintf(why, n, "primitive %u corner %u (lane %u vs reference lane %u) target %u channel %u: %08x vs "
                           "%08x%s", j, c, bl, al, x->target, ch, y->v[ch], x->v[ch], y->undef[ch] ? " (undefined)" : "");
                  return 1;
               }
            }
         }
      }
      if (bary_rot && corners == 3) {
         unsigned bl[3];
         for (unsigned c = 0; c < corners; c++)
            bl[c] = prim_index(pb->v[0], c);
         if (check_bary_rotation(rb, j, bl, why, n, cs))
            return 1;
      }
   }
   cs->cost_wg++;
   for (unsigned k = 0; k < COST_KINDS; k++) {
      cs->cost_a_wave[k] += ra->cost_wave[k];
      cs->cost_b_wave[k] += rb->cost_wave[k];
      cs->cost_a_lane[k] += ra->cost_lane[k];
      cs->cost_b_lane[k] += rb->cost_lane[k];
   }
   cs->prims_b += ns;
   cs->verts_b += rb->alloc_vtx;
   cs->verts_a += corners * ns;
   for (unsigned l = 0; l < b->lanes; l++) {
      cs->params_b += count_params(rb, l);
   }
   for (unsigned j = 0; j < ns; j++)
      for (unsigned c = 0; c < corners; c++)
         cs->params_a += count_params(ra, corner[expect[j]][c]);
   return 0;
}

/* --geometry (see the header): the live primitives of one API workgroup, over the runs of its
 * pieces, in order. */
typedef struct {
   const result_t *r;
   unsigned prim_lane;
   unsigned corner[3];
} geo_prim_t;

/* Returns 0, or -1 when a run's primitive export is missing or undefined (why is set). */
static int
geo_collect(const state_t *st, result_t *const *runs, unsigned nruns, geo_prim_t *out, unsigned *count, char *why,
            size_t n)
{
   *count = 0;
   for (unsigned k = 0; k < nruns; k++) {
      const result_t *r = runs[k];
      if (!r->alloc_done || !r->alloc_prm || !r->alloc_vtx)
         continue;
      if (r->alloc_vtx == 1 && r->alloc_prm == 1) {
         /* GFX10 fully-culled workaround (primitive 0,0,0 and a NaN position): no primitive. */
         const export_t *ve = find_export(r, 0, false);
         if (ve && ve->target == EXP_POS && !ve->undef[0] && ve->v[0] == 0xffffffffu && ve->v[1] == 0xffffffffu &&
             ve->v[2] == 0xffffffffu && ve->v[3] == 0xffffffffu)
            continue;
      }
      for (unsigned p = 0; p < r->alloc_prm && p < st->lanes; p++) {
         const export_t *e = find_export(r, p, true);
         if (!e || e->undef[0]) {
            snprintf(why, n, "piece %u primitive %u: %s primitive export", k, p, e ? "undefined" : "no");
            return -1;
         }
         if (e->v[0] & 0x80000000u)
            continue; /* null primitive */
         geo_prim_t *g = &out[(*count)++];
         g->r = r;
         g->prim_lane = p;
         for (unsigned c = 0; c < 3; c++)
            g->corner[c] = (e->v[0] >> (10 * c)) & 0x1ff;
      }
   }
   return 0;
}

/* --geometry-params: also compare the parameter exports (same varying layout in A and B). */
static bool geo_params;

/* The vertex exports (not the primitive export) of one lane, in order: the position exports
 * (position, misc vector, clip/cull distances) and, with --geometry-params, the parameters. */
static unsigned
geo_vertex_exports(const result_t *r, unsigned lane, const export_t **out)
{
   unsigned n = 0;
   for (unsigned k = 0; k < r->nexp[lane]; k++) {
      const unsigned t = r->exp[lane][k].target;
      if (t != EXP_PRIM && (geo_params || t < EXP_PARAM))
         out[n++] = &r->exp[lane][k];
   }
   return n;
}

static int geo_verbose;

/* Returns 1 on a mismatch, -1 when A's geometry is undefined (skipped), 0 when equal. */
static int
compare_geometry(const state_t *a, const state_t *b, result_t *const *ra, unsigned na, result_t *const *rb,
                 unsigned nb, char *why, size_t n, uint64_t *defined_channels, uint64_t *prims)
{
   static geo_prim_t pa[MAX_PIECES * MAX_LANES], pb[MAX_PIECES * MAX_LANES];
   unsigned ca, cb;
   if (geo_collect(a, ra, na, pa, &ca, why, n))
      return -1;
   if (geo_collect(b, rb, nb, pb, &cb, why, n))
      return 1;
   if (ca != cb) {
      snprintf(why, n, "%u live primitives vs %u", ca, cb);
      return 1;
   }
   for (unsigned i = 0; i < ca; i++) {
      const export_t *xa = find_export(pa[i].r, pa[i].prim_lane, true);
      const export_t *xb = find_export(pb[i].r, pb[i].prim_lane, true);
      /* Everything but the vertex indices: edge flags, the null bit, the second channel. */
      const uint32_t idx_bits = 0x1ffu | 0x1ffu << 10 | 0x1ffu << 20;
      if (xa->mask != xb->mask || xa->flags != xb->flags || (xa->v[0] & ~idx_bits) != (xb->v[0] & ~idx_bits) ||
          ((xa->mask & 2) && !xa->undef[1] && (xb->undef[1] || xa->v[1] != xb->v[1]))) {
         snprintf(why, n, "primitive %u: export %08x/%x/%x vs %08x/%x/%x", i, xa->v[0], xa->mask, xa->flags, xb->v[0],
                  xb->mask, xb->flags);
         return 1;
      }
      for (unsigned c = 0; c < 3; c++) {
         const export_t *ea[MAX_EXPORTS], *eb[MAX_EXPORTS];
         const unsigned va = pa[i].corner[c], vb = pb[i].corner[c];
         if (va >= pa[i].r->alloc_vtx || vb >= pb[i].r->alloc_vtx || va >= a->lanes || vb >= b->lanes) {
            snprintf(why, n, "primitive %u corner %u: vertex %u/%u beyond the vertex count", i, c, va, vb);
            return va >= pa[i].r->alloc_vtx || va >= a->lanes ? -1 : 1;
         }
         const unsigned n_a = geo_vertex_exports(pa[i].r, va, ea), n_b = geo_vertex_exports(pb[i].r, vb, eb);
         if (geo_verbose && i == 0 && c == 0) {
            for (unsigned k = 0; k < MAX2(n_a, n_b); k++)
               printf("  export %u: A %u/%x %08x %08x %08x %08x | B %u/%x %08x %08x %08x %08x\n", k,
                      k < n_a ? ea[k]->target : 0, k < n_a ? ea[k]->mask : 0, k < n_a ? ea[k]->v[0] : 0,
                      k < n_a ? ea[k]->v[1] : 0, k < n_a ? ea[k]->v[2] : 0, k < n_a ? ea[k]->v[3] : 0,
                      k < n_b ? eb[k]->target : 0, k < n_b ? eb[k]->mask : 0, k < n_b ? eb[k]->v[0] : 0,
                      k < n_b ? eb[k]->v[1] : 0, k < n_b ? eb[k]->v[2] : 0, k < n_b ? eb[k]->v[3] : 0);
         }
         if (n_a != n_b) {
            snprintf(why, n, "primitive %u corner %u: %u vs %u vertex exports", i, c, n_a, n_b);
            return 1;
         }
         for (unsigned k = 0; k < n_a; k++) {
            const export_t *x = ea[k], *y = eb[k];
            /* A parameter may carry more channels in B (the expansion copies whole output
             * variables; the linker trims the components the fragment shader never reads
             * from the raw compile): every channel of A must be in B. Position exports
             * (clip/cull distances included) must match exactly. */
            unsigned defined_mask = 0;
            for (unsigned ch = 0; ch < 4; ch++)
               if ((x->mask & (1u << ch)) && !x->undef[ch])
                  defined_mask |= 1u << ch;
            const bool mask_ok = x->target >= EXP_PARAM ?
               !((safe_direct_mode ? defined_mask : x->mask) & ~y->mask) : x->mask == y->mask;
            if (x->target != y->target || !mask_ok || x->flags != y->flags) {
               snprintf(why, n, "primitive %u corner %u export %u: target/mask/flags %u/%x/%x vs %u/%x/%x", i, c, k,
                        x->target, x->mask, x->flags, y->target, y->mask, y->flags);
               return 1;
            }
            for (unsigned ch = 0; ch < 4; ch++) {
               if (!(x->mask & (1u << ch)) || !known_bits(x->undef[ch]))
                  continue;
               (*defined_channels)++;
               clip_channels += is_clip_export(x->target);
               if (export_channel_differs(x, y, ch)) {
                  snprintf(why, n, "primitive %u corner %u export %u (target %u) channel %u: %08x vs %08x%s", i, c, k,
                           x->target, ch, x->v[ch], y->v[ch], y->undef[ch] ? " (undefined)" : "");
                  return 1;
               }
            }
         }
      }
   }
   *prims += ca;
   return 0;
}

int
main(int argc, char **argv)
{
   unsigned seeds = 64, first = 1;
   int cull_mode = 2, query = 0, verbose = 0;
   bool list = false, print = false, autocull = false, bary_ref = false, geometry = false, grid = false;
   unsigned pieces_a = 1, pieces_b = 1;
   uint64_t bary_ref_checked = 0;
   unsigned states = 8;
   bool cost = false;
   const char *files[2] = {NULL, NULL};
   unsigned nfiles = 0;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--seeds") && i + 1 < argc)
         seeds = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--first") && i + 1 < argc)
         first = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--cull") && i + 1 < argc)
         cull_mode = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--query") && i + 1 < argc)
         query = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--autocull"))
         autocull = true;
      else if (!strcmp(argv[i], "--bary-ref"))
         bary_ref = true;
      else if (!strcmp(argv[i], "--states") && i + 1 < argc)
         states = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--mutate") && i + 1 < argc)
         ref_mutate = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--cost"))
         cost = true;
      else if (!strcmp(argv[i], "--bary-rot"))
         bary_rot = true;
      else if (!strcmp(argv[i], "--sgpr"))
         sgpr_model = true;
      else if (!strcmp(argv[i], "--all-corners"))
         all_corner_params = true;
      else if (!strcmp(argv[i], "--safety-only"))
         safety_only = true;
      else if (!strcmp(argv[i], "--safe-owned"))
         safe_direct_mode = safe_owned_mode = true;
      else if (!strcmp(argv[i], "--safe-direct"))
         safe_direct_mode = true;
      else if (!strcmp(argv[i], "--compact"))
         compact_mode = lds_oob_soft = true;
      else if (!strcmp(argv[i], "--gen") && i + 1 < argc) {
         const char *g = argv[++i];
         fixed_gen = atoi(g);
         for (unsigned k = 0; k < NGEN; k++)
            if (!strcmp(g, gen_name[k]))
               fixed_gen = k;
      }
      else if (!strcmp(argv[i], "--indices") && i + 1 < argc) {
         FILE *f = fopen(argv[++i], "r");
         char word[64];
         if (!f || fscanf(f, "%63s", word) != 1 || strcmp(word, "tris")) {
            fprintf(stderr, "--indices: expected a line 'tris a,b,c ...'\n");
            return 2;
         }
         while (file_count < MAX_LANES && fscanf(f, "%u,%u,%u", &file_tris[file_count][0], &file_tris[file_count][1],
                                                 &file_tris[file_count][2]) == 3)
            file_count++;
         fclose(f);
         fixed_gen = GEN_FILE;
      } else if (!strcmp(argv[i], "--w") && i + 1 < argc)
         compact_w = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--geometry"))
         geometry = true;
      else if (!strcmp(argv[i], "--grid"))
         grid = true;
      else if (!strcmp(argv[i], "--geometry-params"))
         geo_params = true;
      else if (!strcmp(argv[i], "--pieces-a") && i + 1 < argc)
         pieces_a = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--pieces-b") && i + 1 < argc)
         pieces_b = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--list"))
         list = true;
      else if (!strcmp(argv[i], "--print"))
         print = true;
      else if (!strcmp(argv[i], "-v"))
         verbose++;
      else if (nfiles < 2)
         files[nfiles++] = argv[i];
   }
   if (!nfiles) {
      fprintf(stderr, "usage: mesh_oracle [--seeds N] [--first S] [--cull 0|1|2] [--query 0|1] [--autocull] [--list] "
                      "[--bary-ref] [--sgpr] [--compact [--safe-direct|--safe-owned] [--gen NAME|N] [--w N] [--bary-rot]] "
                      "[--geometry [--pieces-a N] [--pieces-b M] [--grid] [--geometry-params]] [--mutate N] [--cost] "
                      "[--print] [-v] "
                      "A.nir [B.nir]\n");
      return 2;
   }
   glsl_type_singleton_init_or_ref();

   state_t *a = make_state(files[0], list);
   if (print) {
      /* The dumped shader, with its LDS variables and the primitive index staging. */
      printf("wave=%u hw_workgroup=%u index_staging: offset=%u stride=%u bytes=%u vertices=%u\n", a->wave, a->lanes,
             a->index_staging[0], a->index_staging[1], a->index_staging[2], a->index_staging[3]);
      nir_print_shader(a->nir, stdout);
      return 0;
   }
   state_t *b = nfiles > 1 ? make_state(files[1], list) : NULL;
   if (b)
      a->generator_primitives = b->generator_primitives = MAX2(a->generator_primitives, b->generator_primitives);
   if (safety_only && !safe_direct_mode) {
      fprintf(stderr, "--safety-only requires --safe-direct\n");
      return 2;
   }
   const unsigned a_logical_vertices = a->index_staging[3] ? a->index_staging[3] : a->nir->info.mesh.max_vertices_out;
   const unsigned b_logical_vertices = b && b->index_staging[3] ? b->index_staging[3] :
                                       b ? b->nir->info.mesh.max_vertices_out : 0;
   if (safe_direct_mode && ((!compact_mode && !geometry) || !b || b->lanes > 256 ||
                            b->nir->info.mesh.max_vertices_out > 256 || (!safe_owned_mode && b->pp_params) ||
                            (a_logical_vertices != b_logical_vertices &&
                             !(fixed_gen == GEN_APP && !a->index_staging[1] && !b->index_staging[1])))) {
      fprintf(stderr, "--safe-direct requires a plain <=256-slot compact/geometry comparison with identical logical index domains\n");
      return 2;
   }
   /* Owned-corner lowering may turn built-in PrimitiveID into a flat
    * per-vertex value. That one bit then leaves pp_params; generic payload
    * locations must still match exactly. */
   if (safe_owned_mode && (!b->pp_params || ((b->compact_flags >> 8) & 7) != 1 ||
                            ((a->pp_params ^ b->pp_params) & ~UINT64_C(1)))) {
      fprintf(stderr, "--safe-owned requires matched flat payload parameters and owned first corners\n");
      return 2;
   }
   if (safe_direct_mode) {
      const unsigned v = b->nir->info.mesh.max_vertices_out;
      const unsigned p = b->nir->info.mesh.max_primitives_out;
      const unsigned corners = mesa_vertices_per_prim(b->nir->info.mesh.primitive_type);
      b->safe_export_bound = v <= 32 ? MIN2(v, corners * p) : corners * p;
      if (b->safe_export_bound > b->lanes || b->safe_export_bound > 256) {
         fprintf(stderr, "safe-direct bound exceeds physical launch\n");
         return 2;
      }
   }
   a->query = query;
   if (b)
      b->query = query;
   clip_target = clip_first_target(a->nir);
   if (b) {
      if (!strip_setup(a, b)) {
         fprintf(stderr, "clip/cull forms not derivable: A clip %u cull %u (header 0x%x), B clip %u cull %u\n",
                 CC_CLIP(a->clipcull), CC_CULL(a->clipcull), a->clipcull, CC_CLIP(b->clipcull), CC_CULL(b->clipcull));
         return 2;
      }
      if (strip_active || b_always_cull)
         printf("ORACLE NOTE clip/cull forms: A clip=%u cull=%u%s B clip=%u cull=%u culled_in_shader=%u "
                "strip=%s%s%s pos_b=%u\n", CC_CLIP(a->clipcull), CC_CULL(a->clipcull),
                strip_a_known ? "" : " (header without the form)", CC_CLIP(b->clipcull), CC_CULL(b->clipcull),
                b_always_cull, !strip_active ? "none" : strip_a_known ? "" : "all",
                strip_active && strip_a_known && !strip_keep_clip ? "clip" : "",
                strip_active && strip_a_known && !strip_keep_cull ? "cull" : "", b_num_pos);
      if (CC_CULLED(a->clipcull) && (autocull || compact_mode)) {
         fprintf(stderr, "A culls its cull distances in the shader (RADV_BC250_MESH_CULLDIST_CULL): not a reference\n");
         return 2;
      }
   }
   if (geometry) {
      /* Owned private corners can compare every parameter at every corner.
       * Keep compare_geometry's full equality check: unlike --compact alone,
       * this deliberately does not exempt non-provoking payload channels. */
      if (!b || autocull || bary_ref || (compact_mode && !safe_direct_mode) || !pieces_a || !pieces_b || pieces_a > MAX_PIECES || pieces_b > MAX_PIECES) {
         fprintf(stderr, "--geometry needs A and B, excludes --autocull/--bary-ref and non-safe --compact, 1 <= pieces <= %u\n", MAX_PIECES);
         return 2;
      }
      a->geo = b->geo = true;
      a->grid = b->grid = grid;
      result_t *ra_k[MAX_PIECES], *rb_k[MAX_PIECES];
      for (unsigned k = 0; k < MAX_PIECES; k++) {
         ra_k[k] = malloc(sizeof(result_t));
         rb_k[k] = malloc(sizeof(result_t));
      }
      unsigned compared = 0, skipped = 0, failures = 0, b_ub = 0;
      uint64_t defined_channels = 0, prims = 0;
      for (unsigned s = first; s < first + seeds; s++) {
         /* The API workgroup L of this seed: A runs its pieces_a hardware workgroups
          * L * pieces_a + k, B its pieces_b ones, with the same seed (the same inputs). */
         const uint32_t L = mix64(s * 0x9e37ull + 0x6e0) % 64;
         bool a_ub = false, bad_b = false;
         char fa[512] = "", fb[512] = "";
         for (unsigned k = 0; k < pieces_a; k++) {
            a->wg_index = L * pieces_a + k;
            run(a, s, 0, ra_k[k]);
            if (ub(ra_k[k]) && !a_ub) {
               a_ub = true;
               flags_str(ra_k[k], fa, sizeof(fa));
            }
         }
         if (a_ub) {
            skipped++;
            if (verbose)
               printf("seed=%u skipped (reference:%s)\n", s, fa);
            continue;
         }
         if (strip_active) {
            for (unsigned k = 0; k < pieces_a; k++) {
               static result_t tmp;
               strip_dist(ra_k[k], &tmp, a->lanes);
               memcpy(ra_k[k], &tmp, sizeof(tmp));
            }
         }
         for (unsigned k = 0; k < pieces_b; k++) {
            b->wg_index = L * pieces_b + k;
            run(b, s, 0, rb_k[k]);
            if (safe_direct_mode && !ub(rb_k[k]) && !bad_b) {
               compact_stats_t stats = {0};
               bad_b = check_compact_safety(b, rb_k[k], fb, sizeof(fb), &stats);
            }
            if (ub(rb_k[k]) && !bad_b) {
               bad_b = true;
               flags_str(rb_k[k], fb, sizeof(fb));
            }
         }
         geo_verbose = verbose > 1;
         if (verbose > 1) {
            for (unsigned k = 0; k < pieces_a + pieces_b; k++) {
               const result_t *rr = k < pieces_a ? ra_k[k] : rb_k[k - pieces_a];
               const export_t *pe = find_export(rr, 0, true), *ve = find_export(rr, 0, false);
               printf("seed=%u %c%u alloc=%u/%u prim0=%08x vtx0=%u:%08x,%08x,%08x,%08x\n", s, k < pieces_a ? 'A' : 'B',
                      k < pieces_a ? k : k - pieces_a, rr->alloc_vtx, rr->alloc_prm, pe ? pe->v[0] : 0,
                      ve ? ve->target : 0, ve ? ve->v[0] : 0, ve ? ve->v[1] : 0, ve ? ve->v[2] : 0, ve ? ve->v[3] : 0);
            }
         }
         char why[512] = "";
         int r = 0;
         if (bad_b) {
            r = 1;
            b_ub++;
            snprintf(why, sizeof(why), "candidate:%s", fb);
         } else {
            r = compare_geometry(a, b, ra_k, pieces_a, rb_k, pieces_b, why, sizeof(why), &defined_channels, &prims);
         }
         if (r < 0) {
            skipped++;
            if (verbose)
               printf("seed=%u skipped (reference geometry undefined: %s)\n", s, why);
            continue;
         }
         compared++;
         if (r) {
            failures++;
            printf("ORACLE MISMATCH seed=%u workgroup=%u %s\n", s, L, why);
         } else if (verbose) {
            printf("seed=%u workgroup=%u ok\n", s, L);
         }
      }
      printf("ORACLE GEOMETRY %s compared=%u skipped_reference_ub=%u candidate_ub=%u mismatches=%u pieces=%u/%u "
             "primitives=%" PRIu64 " defined_channels=%" PRIu64 " clipcull_channels=%" PRIu64 "\n",
             failures ? "FAIL" : "PASS", compared, skipped, b_ub, failures, pieces_a, pieces_b, prims,
             defined_channels, clip_channels);
      return failures ? 1 : 0;
   }
   if (b && ((!safe_direct_mode && a->lanes != b->lanes) || a->wave != b->wave ||
             (!safe_direct_mode && a->nir->info.mesh.max_primitives_out != b->nir->info.mesh.max_primitives_out))) {
      /* Different split pieces (e.g. one variant needed the LDS fit retry): not the same workgroup. */
      printf("ORACLE NOT-COMPARABLE lanes %u/%u wave %u/%u primitives %u/%u\n", a->lanes, b->lanes, a->wave, b->wave,
             a->nir->info.mesh.max_primitives_out, b->nir->info.mesh.max_primitives_out);
      return 3;
   }
   if (bary_ref && (!b || autocull || compact_mode)) {
      fprintf(stderr, "--bary-ref needs A and B and excludes --autocull and --compact\n");
      return 2;
   }
   if (bary_rot && !compact_mode) {
      fprintf(stderr, "--bary-rot needs --compact\n");
      return 2;
   }
   if (autocull && !b) {
      fprintf(stderr, "--autocull needs A (without autocull) and B (with autocull)\n");
      return 2;
   }
   if (autocull) {
      ref_exec_mode = b->nir->info.float_controls_execution_mode;
      /* A must be the compile without autocull (it runs once for all culling states). */
      nir_foreach_block(block, a->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_cull_any_enabled_amd) {
               fprintf(stderr, "--autocull: A reads the culling state (an autocull compile)\n");
               return 2;
            }
         }
      }
   }

   if (compact_mode && b && !(b->compact_flags & 1)) {
      /* The candidate compile did not apply the compaction: compare exactly (or as --autocull). */
      printf("ORACLE NOTE candidate compile did not apply RADV_BC250_MESH_COMPACT (header flags 0x%x): plain "
             "comparison\n", b->compact_flags);
      compact_mode = lds_oob_soft = false;
   }
   if (compact_mode) {
      /* --compact: A = reference (no compaction, no autocull), B = candidate. */
      if (!b) {
         fprintf(stderr, "--compact needs A and B\n");
         return 2;
      }
      if (autocull)
         ref_exec_mode = b->nir->info.float_controls_execution_mode;
      result_t *ra = malloc(sizeof(result_t)), *rb = malloc(sizeof(result_t)), *rax = malloc(sizeof(result_t));
      compact_stats_t cs;
      memset(&cs, 0, sizeof(cs));
      unsigned compared = 0, skipped = 0, failures = 0, safety_failures = 0, b_ub = 0, both_ub = 0;
      uint64_t defined_channels = 0;
      for (unsigned sd = first; sd < first + seeds; sd++) {
         const unsigned g = fixed_gen == GEN_FILE ? GEN_FILE : fixed_gen >= 0 ? (unsigned)fixed_gen % NGEN : sd % NGEN;
         a->gen = b->gen = g;
         const unsigned nst = autocull ? states : 2;
         for (unsigned k = 0; k < nst; k++) {
            const int cull = autocull ? 0 : k;
            if (!autocull && cull_mode != 2 && cull != cull_mode)
               continue;
            if (autocull) {
               make_cullstate(sd * 64 + k, &b->cs);
               a->cs = b->cs;
            }
            run(a, sd, cull, ra);
            run(b, sd, cull, rb);
            cs.runs++;
            cs.gen_runs[g]++;
            char why[512] = "", fa[512], fb[512];
            flags_str(ra, fa, sizeof(fa));
            flags_str(rb, fb, sizeof(fb));
            /* B's hard undefined behaviour (not LDS out of bounds, not invalid application data). */
            const bool b_hard = rb->oob || rb->race || rb->runaway || rb->unsupported || rb->exp_overflow;
            const bool a_hard = ra->oob || ra->race || ra->runaway || ra->unsupported || ra->exp_overflow;
            if (b_hard) {
               if (a_hard) {
                  /* The application part itself is undefined in the model (a race, a runaway loop, an
                   * unsupported intrinsic) in both compiles: nothing to check. */
                  both_ub++;
                  if (verbose)
                     printf("seed=%u gen=%s state=%u both undefined: reference:%s candidate:%s\n", sd, gen_name[g], k,
                            fa, fb);
                  continue;
               }
               failures++;
               b_ub++;
               printf("ORACLE MISMATCH seed=%u gen=%s state=%u candidate undefined:%s (reference:%s)\n", sd,
                      gen_name[g], k, fb, fa);
               continue;
            }
            if (getenv("ORACLE_DUMP_PRIMS")) {
               printf("seed=%u gen=%s alloc B %u/%u A %u/%u\n", sd, gen_name[g], rb->alloc_vtx, rb->alloc_prm,
                      ra->alloc_vtx, ra->alloc_prm);
               for (unsigned l = 0; l < b->lanes; l++) {
                  const export_t *pe = find_export(rb, l, true);
                  if (pe)
                     printf(" p%u=(%u,%u,%u)", l, prim_index(pe->v[0], 0), prim_index(pe->v[0], 1),
                            prim_index(pe->v[0], 2));
               }
               printf("\n");
            }
            if (check_compact_safety(b, rb, why, sizeof(why), &cs)) {
               failures++;
               safety_failures++;
               cs.gen_safety_fail[g]++;
               printf("ORACLE SAFETY-FAIL seed=%u gen=%s state=%u %s\n", sd, gen_name[g], k, why);
               continue;
            }
            if (safety_only)
               continue;
            if (ub(ra)) {
               skipped++;
               if (verbose)
                  printf("seed=%u gen=%s skipped equality (reference:%s)\n", sd, gen_name[g], fa);
               continue;
            }
            strip_dist(ra, rax, a->lanes);
            const int r = compare_compact(a, b, ra, rax, rb, autocull ? &b->cs : NULL, why, sizeof(why),
                                          &defined_channels, &cs);
            if (r >= 0 && ub(rb)) {
               /* B undefined (e.g. its culling test branched on a position the application never
                * wrote) although the reference, including its culling decision, is defined. */
               failures++;
               b_ub++;
               printf("ORACLE MISMATCH seed=%u gen=%s state=%u candidate undefined:%s\n", sd, gen_name[g], k, fb);
               continue;
            }
            if (r < 0) {
               skipped++;
               continue;
            }
            compared++;
            cs.gen_compared[g]++;
            if (!r && rb->alloc_prm && !(rb->alloc_vtx == 1 && rb->alloc_prm == 1)) {
               cs.gen_tris[g] += rb->alloc_prm;
               cs.gen_verts[g] += rb->alloc_vtx;
            }
            if (r > 0) {
               failures++;
               if (autocull)
                  printf("ORACLE MISMATCH seed=%u gen=%s state=%u cull=front%u/back%u/ccw%u/small%u %s\n", sd,
                         gen_name[g], k, b->cs.front, b->cs.back, b->cs.ccw, b->cs.small, why);
               else
                  printf("ORACLE MISMATCH seed=%u gen=%s cull=%d %s\n", sd, gen_name[g], cull, why);
            } else if (verbose) {
               printf("seed=%u gen=%s ok %u/%u -> %u/%u\n", sd, gen_name[g], ra->alloc_vtx, ra->alloc_prm,
                      rb->alloc_vtx, rb->alloc_prm);
            }
         }
      }
      if (cost && cs.cost_wg) {
         static const char *kinds[COST_KINDS] = {"all", "lds_load", "lds_store", "exp_pos", "exp_param", "exp_prim",
                                                 "barrier"};
         printf("ORACLE COST wg=%" PRIu64, cs.cost_wg);
         for (unsigned k = 0; k < COST_KINDS; k++)
            printf(" %s_wave=%.1f/%.1f %s_lane=%.1f/%.1f", kinds[k], (double)cs.cost_a_wave[k] / cs.cost_wg,
                   (double)cs.cost_b_wave[k] / cs.cost_wg, kinds[k], (double)cs.cost_a_lane[k] / cs.cost_wg,
                   (double)cs.cost_b_lane[k] / cs.cost_wg);
         printf("\n");
      }
      if (verbose) {
         for (unsigned g = 0; g <= NGEN; g++)
            printf("ORACLE GEN %s runs=%" PRIu64 " compared=%" PRIu64 " safety_fail=%" PRIu64 " verts_per_tri=%.3f\n",
                   gen_name[g], cs.gen_runs[g], cs.gen_compared[g], cs.gen_safety_fail[g],
                   cs.gen_tris[g] ? (double)cs.gen_verts[g] / cs.gen_tris[g] : 0.0);
      }
      if (bary_rot)
         printf("ORACLE BARY-ROT primitives=%" PRIu64 " ambiguous_no_fragments=%" PRIu64 " position_undefined=%" PRIu64
                "\n", cs.rot_prims, cs.rot_ambiguous, cs.rot_undefined);
      printf("ORACLE %s compared=%u skipped_reference_ub=%u candidate_ub=%u both_ub=%u mismatches=%u "
             "safety_checked=%" PRIu64 " safety_fail=%u defined_channels=%" PRIu64 " applied=%u W=%u owned=0x%x "
             "max_backjump=%" PRIu64 " max_V=%" PRIu64 " dummy=%" PRIu64 " tris=%" PRIu64 " verts_expanded=%" PRIu64
             " verts_compact=%" PRIu64 " verts_per_tri=%.3f params_expanded=%" PRIu64 " params_compact=%" PRIu64
             " payload_channels_at_unowned_corners=%" PRIu64 " clipcull_channels=%" PRIu64 " culled_dist=%" PRIu64
             "\n",
             failures ? "FAIL" : "PASS", compared, skipped, b_ub, both_ub, failures - safety_failures - b_ub,
             cs.safety_runs, safety_failures, defined_channels, b->compact_flags & 1, compact_w,
             (b->compact_flags >> 8) & 7, cs.bj_max, cs.v_hist_max, cs.dummy, cs.prims_b, cs.verts_a, cs.verts_b,
             cs.prims_b ? (double)cs.verts_b / cs.prims_b : 0.0, cs.params_a, cs.params_b, cs.payload_skipped,
             clip_channels, cs.culled_dist);
      return failures ? 1 : 0;
   }

   result_t *ra = malloc(sizeof(result_t)), *rb = malloc(sizeof(result_t)), *rax = malloc(sizeof(result_t));
   unsigned compared = 0, skipped = 0, failures = 0, b_ub = 0;
   uint64_t defined_channels = 0, exported_vertices = 0, alloc_vtx_total = 0, b_exported_vertices = 0;
   cull_stats_t stats;
   memset(&stats, 0, sizeof(stats));
   for (unsigned s = first; s < first + seeds; s++) {
      unsigned nstates = 1;
      for (unsigned k = 0; k < (autocull ? nstates : 2); k++) {
         const int cull = autocull ? 0 : k;
         if (autocull) {
            /* A (no culling code) runs once per seed; B runs under `states` culling states when
             * A's workgroup has primitives (one otherwise). */
            make_cullstate(s * 64 + k, &b->cs);
            if (k == 0) {
               a->cs = b->cs;
               run(a, s, 0, ra);
               nstates = ra->alloc_done && ra->alloc_prm && !(ra->alloc_vtx == 1 && ra->alloc_prm == 1) ? states : 1;
            }
         } else if (cull_mode != 2 && cull != cull_mode) {
            continue;
         }
         if (!autocull)
            run(a, s, cull, ra);
         if (!b) {
            char f[512];
            flags_str(ra, f, sizeof(f));
            if (verbose || ub(ra))
               printf("seed=%u cull=%d alloc=%u/%u exports(lane0)=%u%s\n", s, cull, ra->alloc_vtx, ra->alloc_prm,
                      ra->nexp[0], f);
            continue;
         }
         run(b, s, cull, rb);
         if (verbose > 1) {
            for (unsigned side = 0; side < 2; side++) {
               const result_t *rr = side ? rb : ra;
               for (unsigned k = 0; k < rr->nexp[0]; k++)
                  printf("seed=%u %c lane 0 export %u: target %u mask %x flags %x %08x %08x %08x %08x\n", s,
                         side ? 'B' : 'A', k, rr->exp[0][k].target, rr->exp[0][k].mask, rr->exp[0][k].flags,
                         rr->exp[0][k].v[0], rr->exp[0][k].v[1], rr->exp[0][k].v[2], rr->exp[0][k].v[3]);
            }
         }
         char fa[512], fb[512];
         flags_str(ra, fa, sizeof(fa));
         flags_str(rb, fb, sizeof(fb));
         if (ub(ra)) {
            skipped++;
            if (verbose)
               printf("seed=%u cull=%d skipped (reference:%s)\n", s, cull, fa);
            if (autocull)
               break;
            continue;
         }
         bool bad = false;
         char why[512] = "";
         strip_dist(ra, rax, a->lanes);
         if (autocull) {
            const int r = compare_autocull(a, ra, rax, rb, &b->cs, why, sizeof(why), &defined_channels, &stats);
            if (r < 0) {
               stats.ref_undefined++;
               skipped++;
               if (verbose)
                  printf("seed=%u skipped (reference culling decision undefined: %s)\n", s, why);
               continue;
            }
            bad = r > 0;
            b_ub += r == 2;
         } else if (ub(rb)) {
            bad = true;
            b_ub++;
            snprintf(why, sizeof(why), "candidate:%s", fb);
         } else {
            if (bary_ref)
               bad = compare_bary_ref(a, rax, rb, why, sizeof(why), &bary_ref_checked, &defined_channels);
            else
               bad = compare_exact(a, rax, rb, why, sizeof(why), &defined_channels);
         }
         compared++;
         alloc_vtx_total += ra->alloc_vtx;
         for (unsigned l = 0; l < a->lanes; l++) {
            exported_vertices += ra->nexp[l] > 0;
            b_exported_vertices += count_exports(rb, l, false) > 0;
         }
         if (bad) {
            failures++;
            if (autocull)
               printf("ORACLE MISMATCH seed=%u state=%u cull=front%u/back%u/ccw%u/small%u/vp%g,%g,%g,%g/prec%g %s\n", s,
                      k, b->cs.front, b->cs.back, b->cs.ccw, b->cs.small, b->cs.vp[0], b->cs.vp[1], b->cs.vp[2],
                      b->cs.vp[3], b->cs.prec, why);
            else
               printf("ORACLE MISMATCH seed=%u cull=%d %s\n", s, cull, why);
         } else if (verbose) {
            printf("seed=%u cull=%d ok alloc=%u/%u -> %u/%u\n", s, cull, ra->alloc_vtx, ra->alloc_prm, rb->alloc_vtx,
                   rb->alloc_prm);
         }
      }
   }

   if (list) {
      hash_table_foreach(a->unknown, e) printf("intrinsic %s\n", (const char *)e->key);
   }
   if (b && autocull) {
      if (cost) {
         static const char *kinds[COST_KINDS] = {"all", "lds_load", "lds_store", "exp_pos", "exp_param", "exp_prim",
                                                 "barrier"};
         for (unsigned bk = 0; bk < COST_BUCKETS; bk++) {
            if (!stats.cost_wg[bk])
               continue;
            printf("ORACLE COST %s wg=%" PRIu64 " live_prims/wg=%.1f kept/wg=%.1f", cost_bucket_name[bk],
                   stats.cost_wg[bk], (double)stats.cost_prims[bk] / stats.cost_wg[bk],
                   (double)stats.cost_kept[bk] / stats.cost_wg[bk]);
            for (unsigned k = 0; k < COST_KINDS; k++)
               printf(" %s_wave=%.1f/%.1f %s_lane=%.1f/%.1f", kinds[k],
                      (double)stats.cost_a_wave[bk][k] / stats.cost_wg[bk], (double)stats.cost_b_wave[bk][k] / stats.cost_wg[bk],
                      kinds[k], (double)stats.cost_a_lane[bk][k] / stats.cost_wg[bk],
                      (double)stats.cost_b_lane[bk][k] / stats.cost_wg[bk]);
            printf("\n");
         }
      }
      printf("ORACLE %s compared=%u skipped_reference_ub=%u candidate_ub=%u mismatches=%u defined_channels=%" PRIu64
             " wg_skip=%" PRIu64 " wg_empty=%" PRIu64 " wg_none_culled=%" PRIu64 " wg_partial=%" PRIu64
             " wg_all_culled=%" PRIu64 " prims=%" PRIu64 " kept=%" PRIu64 " culled_w=%" PRIu64 " culled_face=%" PRIu64
             " culled_frustum=%" PRIu64 " culled_small=%" PRIu64 " culled_dist=%" PRIu64 " ref_undefined=%" PRIu64
             " vertex_lanes_a=%" PRIu64 " vertex_lanes_b=%" PRIu64 " clipcull_channels=%" PRIu64 "\n",
             failures ? "FAIL" : "PASS", compared, skipped, b_ub, failures, defined_channels, stats.wg_skip,
             stats.wg_empty, stats.wg_none_culled, stats.wg_partial, stats.wg_all_culled, stats.prims,
             stats.culled[CULL_KEPT], stats.culled[CULL_W], stats.culled[CULL_FACE], stats.culled[CULL_FRUSTUM],
             stats.culled[CULL_SMALL], stats.culled[CULL_DIST], stats.ref_undefined, exported_vertices,
             b_exported_vertices, clip_channels);
   } else if (b) {
      if (bary_ref)
         printf("ORACLE BARY-REF reference_channels=%" PRIu64 "\n", bary_ref_checked);
      printf("ORACLE %s compared=%u skipped_reference_ub=%u candidate_ub=%u mismatches=%u defined_channels=%" PRIu64
             " exporting_lanes=%" PRIu64 " alloc_vertices=%" PRIu64 " clipcull_channels=%" PRIu64 "\n",
             failures || (bary_ref && !bary_ref_checked) ? "FAIL" : "PASS", compared, skipped, b_ub, failures,
             defined_channels, exported_vertices, alloc_vtx_total, clip_channels);
   }
   return failures || (b && bary_ref && !bary_ref_checked) ? 1 : 0;
}

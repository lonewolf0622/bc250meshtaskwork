# RADV_BC250_MESH_COMPACT offline regression

Shared-vertex renumbering of the base driver's expanded Mesh shaders. The suite runs only under the drm-shim and never
touches the real GPU: every Vulkan program runs in bwrap with a fresh `/dev` (no `/dev/dri`),
`libamdgpu_noop_drm_shim.so` preloaded and `AMDGPU_GPU_ID=gfx1013`, and `pipe.c` refuses any other device.

```sh
ICD=<build>/src/amd/vulkan/radeon_devenv_icd.x86_64.json BUILD=<build> \
OLDICD=<icd of the build before the switch> [SEEDS=120] [ONLY=<case regex>] [KEEP=<dir>] ./run.sh
```

## What the switch does
Without it, an expanded Mesh shader exports three private vertices per triangle (3P lanes, 3P exported vertices).
With `RADV_BC250_MESH_COMPACT=1` (compiler key `bc250_mesh_compact`) the NGG lowering's epilogue
(`ac_nir_lower_ngg_mesh.c`, `ms_compact_vertices`; after autocull when that is on) exports shared vertices, renumbered
so that the exported connectivity stays in the class GFX1013 ran safely with raw shared vertices, for any index data:

- all referenced: every exported vertex is referenced by an exported, non-null primitive;
- window: no corner references a vertex more than W = 8 below the highest index referenced before it
  (`AC_NIR_BC250_COMPACT_W`, printed by `BC250_TRACE_COMPILE=1` as `BC250 MESH COMPACT: applied W=8 D=9 ...`).

The corners of the exported primitives, in export order, are the keys. Two anchor levels (LDS atomic min per logical
vertex) give each key an anchor at most D = W + 1 keys earlier; a key that is its own anchor, has an index outside
the declared vertices, or owns its primitive's per-primitive payload (the provoking corner: 0 for first, 2 for last,
both when the provoking mode is dynamic) gets a new vertex slot; the others reference their anchor's slot. The
backjump of any reference is at most (key distance - 1) <= W. Each slot exports the expanded vertex that created it,
so every exported value is one the expanded shader exports; primitive order, corner order, winding and the provoking
vertex are unchanged. The launch shape (lanes, `GE_NGG_SUBGRP_CNTL`, `VGT_GS_MAX_VERT_OUT`, fast launch 0) is that of
the expanded shader: a workgroup whose keys are all fresh is exactly the expanded shape.

Refused (compiled exactly as with the switch off, trace `not applied reason=...`): CullPrimitive, per-primitive export
arguments, layer/viewport/shading-rate outputs, lines and points, multiview, the base driver's compact vertex map,
raw/merged/AMD routes, outputs on the Mesh scratch ring.

## Checks (run.sh)
51 pipelines from `cmp.mesh` (a copy of `../direct-read/dr.mesh` with `GRID` meshlet-like indices and `OVERCOUNT`
counts above the maximum) and `../piece-ceiling/wide.*`, each compiled with the switch off (reference) and on
(candidate, plus autocull where the case says so), NIR and ACO validation, a direct, an indirect and an
indirect-count draw submitted to the shim. Per case: equal pipeline results, submission, no validation error, the
expected `BC250 MESH COMPACT` trace, and `../direct-read/mesh_oracle --compact` (plus `--autocull`) PASS on every dumped
pair; with `OLDICD`, the switch-off shaders (unset and `=0`) byte-identical to the base build. Two oracle self-checks
(`--w 0`, `--mutate 5`) must fail.

The oracle (`mesh_oracle --compact`) cycles, seed by seed, through index generators: the application's data, random,
degenerate, strips with backjumps, unreferenced vertices, few vertices, grid, fan, descending, top vertex only in the
last primitive, meshoptimizer-like meshlets, invalid and huge invalid indices. On every candidate run (also where the
reference is undefined) it checks the safety of the export: V'/P' within the lanes, the maxima and 256, positions
exactly on lanes < V', primitives exactly on lanes < P', no null primitive, indices < V', both rules. Where the
reference is defined, the candidate's primitives in order, each as its three corners' vertex exports, must equal the
reference's live primitives (or the autocull survivors, or the fully-culled dummy) bit for bit, except the
per-primitive payload channels at corners that do not own the payload.

### With barycentrics and implicit triangles
Every oracle run also passes `--bary-rot`: for each candidate primitive the oracle recovers the barycentric vertex
order the way the BC250 fragment shader does (`radv_nir_bc250_lower_bary_rotation`: x, y, w of the raw corners
against the flat provoking corner), for both provoking conventions and every cyclic rotation of the parameter cache,
and checks that API vertex 0 is found. Shared vertices carry the position of their logical vertex, so a match fails
only when a second corner has the provoking corner's x, y, w bits: both corners then project to one screen point and
the primitive has no fragments (reported as `ambiguous`). The `bary*` cases compile `cmp.frag` with `-DBARY=1`
(reads `gl_BaryCoordEXT`), so the Mesh shader also exports the two position references on its shared vertices; the
candidate's exports, references included, must equal the reference's per corner. The `implicit*` cases add
`RADV_BC250_MESH_IMPLICIT_TRIS=1` (to both compiles, or to the candidate only): the compaction's connectivity replaces
the derived (3j, 3j+1, 3j+2) one, which only still feeds the autocull test; with the base driver's compact vertex map
neither switch applies.

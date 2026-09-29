# RADV_BC250_MESH_MERGE offline regression

Stages 1 and 2 of the BC-250 Mesh workgroup merge (`MESH_PERF/merge/DESIGN.md`).
- Stage 1: lonewolfMT `c350460`, build `git-c350460c2a`.
- Stage 2: `9168095`, build `git-9168095359`. It adds per-primitive outputs and wave32 subgroup-op shaders.

It runs only under the drm-shim and never touches the real GPU. `run.sh` starts every Vulkan program in bwrap:
- a fresh `/dev`, so there is no `/dev/dri`;
- `libamdgpu_noop_drm_shim.so` preloaded;
- `AMDGPU_GPU_ID=gfx1013`.

`merge.c` also refuses to run unless the shim is preloaded and the device is the shim's GFX1013.

```sh
ICD=/path/to/your-mesa-build/src/amd/vulkan/radeon_devenv_icd.x86_64.json ./run.sh   # KEEP=<dir> keeps the dumps
```

## What the switch does
`RADV_BC250_MESH_MERGE=1` sets the compiler key bit `bc250_mesh_merge` (default off). A mesh-only triangle Mesh shader with P <= 64 and an API workgroup of L <= 64 invocations is rewritten by `radv_bc250_merge_mesh`. The pass runs in `radv_bc250.c`, before the route decision.

**The rewrite:**
- K consecutive API workgroups run as one raw subgroup on fast launch 0, with K*V+3 vertices, K*P triangles and max(K*S, K*V+3, K*P) lanes. K is the smallest value with K*P >= 65.
- The lane stride S is the next power of two >= L when the shader has no subgroup operations; wave64 is then forced. Otherwise S = 64, which requires a wave64 shader.
- Primitive slots that an instance does not produce become degenerate triangles on 3 sink vertices at clip position (2,2,2,1).
- A subgroup without any valid instance outputs 0/0.

**Refusals:**
- The planner refuses when K*P > 98, lanes > 128, or the worst-case LDS is >= 30 KiB.
- It also refuses CullPrimitive, per-primitive outputs, output reads, subgroup operations in wave32 shaders (this includes shared variables that `nir_opt_shared_vars_to_subgroup` turned into subgroup operations), and unknown system values.
- The pipeline gate excludes Task/replay, split, multiview, mesh queries and graphics-pipeline-library NIR.

**Draws:** direct draws emit DRAW_INDEX_AUTO ceil(x*y*z/K), and the grid SGPRs keep the application's (x,y,z). Indirect draws: option A (below) since f5143a6; before it (and with `RADV_BC250_MESH_MERGE_INDIRECT=b`) N subgroups launch and the surplus exits empty.

## Option A indirect draws (lonewolfMT `f5143a6`, build `git-f5143a66c4`)
`RADV_BC250_MESH_MERGE_INDIRECT=a` (the default with the merge on) gives indirect and indirect-count draws of merged shaders one prep dispatch per API call:
- it writes a driver record `{ceil(x*y*z/K),1,1,0, x,y,z,0}` per active record (32-byte stride);
- DISPATCH_MESH_INDIRECT_MULTI reads those records, so ceil(N/K) groups launch instead of N;
- the shader gets the application's grid from the dims user SGPR (`AC_UD_MS_BC250_DIMS_VA`, the records' low 32 bits, 0 for direct draws and DGC) at `dims + 32*DrawID + 16`.

`=b` is option B, the stage 1/2 behaviour (N groups, the surplus exits empty).

`run.sh` runs every case three times: switch off, on (option A), and on with `RADV_BC250_MESH_MERGE_INDIRECT=b` (`OPTB` for check.py). For candidates, `check.py` checks option A per indirect call:
- exactly one prep DISPATCH_DIRECT of ceil(records/64) workgroups;
- the prep's push constants carry K, the record count and the application's count buffer address;
- EVENT_WRITE CS_PARTIAL_FLUSH and an ACQUIRE_MEM with GLK_INV (K$) and no GL2_INV/GL2_WB between the prep and the draw;
- SET_BASE on the driver records, not the application's buffer;
- the dims SGPR (SPI_SHADER_USER_DATA_GS_<DRAW_INDEX_LOC+1-0x8c>) = the records' low 32 bits;
- stride 32;
- COUNT, the count address and its enable, and XYZ_DIM_LOC equal to the switch-off packet;
- DRAW_INDEX_ENABLE set;
- the direct draw writes the dims SGPR as 0.

It also checks the NIR and ISA:
- `load_user_data_amd`, `load_draw_id` and `load_global` are present;
- one `load_num_workgroups` is left (every application read goes through the selected grid);
- an `s_load_dwordx4`/`x2` is present.

Option B: no setup dispatch, indirect packets byte-identical to the switch off, and no dims SGPR in the NIR.

New cases (MERGE_IND_RECORDS / MERGE_CNT in merge.c):

| case | shape | records (indirect / count of max) |
|---|---|---|
| ind1_d32 | d32 | 1 / 1 of 1 |
| ind1000_d32 | d32 | 1000 / 700 of 1000 (gl_DrawID) |
| ind1000_wg3d | wg3d_tail (3D grid 7x5x2, N=70 not a multiple of 3) | 1000 / 999 of 1000 |
| ind1000_hb32sg | HB2 regime (per-primitive, wave32) | 1000 / 3 of 1000 |
| ind1000_pp16 | K=5 | 1000 / 1000 of 1000 |

`indmodel.py` is a CPU model of the prep and the shader's index math, using the same u32 formulas. For every call it checks that each API workgroup id of every active record is run exactly once, by its own DrawID, and that gl_NumWorkGroups is the application's grid. It also checks that option A launches exactly ceil(N/K) groups with none empty, and that option B launches N. It covers:
- exhaustive 3D grids up to 6x4x3, including zero dimensions, for K = 2..5;
- 900 random multi-record calls with count below, equal to, above the maximum, and 0;
- the 1000-record layout;
- 1D grids up to 65536.

Result at f5143a6: **47/47** (41 + 5 option A cases + indirect_model).

Dumps were compared with a build of 977726b (the pre-option-A source), version string excepted:
- switch off: 45/45 cases byte-identical;
- option B: 45/45 byte-identical to the old switch-on run;
- non-candidates with option A: 13/13 byte-identical.

## Checks (check.py)
Every case runs with the switch on and off, with the the base driver launcher policy, post-Mesh VGT_FLUSH off, `NIR_DEBUG=validate` and `ACO_DEBUG=validateir,validatera`. The harness records a direct draw of the case's grid, a VS draw, an indirect draw (2 records: the grid and 64x1x1) and an indirect-count draw.

For merge candidates:
- the planner's K and S;
- raw route, `fl0=1`, no AMD size part, `merge_k`/`merge_s` in the shader info, no scratch ring, wave64, workgroup = lanes <= 128;
- VGT_SHADER_STAGES_EN 0x00012020 (GS_FAST_LAUNCH=0);
- GE_NGG_SUBGRP_CNTL AMP >= workgroup lanes and THDS 0; VGT_GS_MAX_VERT_OUT = lanes;
- DRAW_INDEX_AUTO count = ceil(N/K);
- DISPATCH_MESH_INDIRECT_MULTI packets byte-identical to the switch off;
- no setup dispatch;
- ACO: exactly 2 GS_ALLOC_REQ (the GFX10 fully-culled workaround path and the live path), with the K*P and K*V+3 literals;
- merged NIR: one `set_vertex_and_primitive_count`; no `load_workgroup_id`, `load_local_invocation_id`, `load_subgroup_id` or `load_num_subgroups` left; `load_workgroup_index` used; max_vertices_out/max_primitives_out = K*V+3 / K*P;
- the uploaded binary has LDS < 32 KiB and scratch 0.

For non-candidates: the switch-on run is byte-identical to the switch-off run (IB, NIR and ISA dumps), apart from the planner's trace line.

For stage-2 candidates (per-primitive outputs or CullPrimitive, `stage2=1` in the planner line), `check.py` also checks:
- the expected wave size (wave32 when subgroup operations pin the shader to wave32) and the merged vertex count K*(V-1+P)+3;
- physical vertices and lanes <= 256;
- every output deref in the merged NIR is indexed by `load_local_invocation_index` (re-emitted by its own lane, so no cross-lane output access);
- only the primitive indices stay per-primitive; each former per-primitive output is a FLAT per-vertex output;
- each store of a former per-primitive output sits under the zipper's "this slot is a provoking vertex" test (an `iand` of an `ieq`), i.e. it writes the provoking-vertex record only;
- the fragment shader has no per-primitive input left, and its `SPI_PS_INPUT_CNTL_n` equal those of the base driver's expansion (switch off): flat attributes.

`mapmodel.py` models the stage-2 vertex map on the CPU with the pass's integer formulas:
- It checks that the zipper is a permutation with a correct inverse for every V <= 64, P <= 64.
- 4000 random merged subgroups cover runtime counts, invalid instances, out-of-range indices and both provoking modes. For each one it checks that:
  - every kept primitive's corners hold the right logical vertices;
  - the provoking corner is private and carries the primitive's data;
  - holes use only the sink vertices.
- It prints the exported index pattern of lattice meshlets.

`api_barriers`: sh32 (two API workgroup barriers) has the same s_barrier count as d32 (none). The API barriers become subgroup scope.

## Result at 9168095 (stage 2): 41/41

At 9168095 the 10 stage-1 candidates and all non-candidates are byte-identical to the pre-stage-2 build (dump compare, version string and addresses excepted). The switch-off runs of the new cases are byte-identical too.

| case | shape (V/P/L), grid | K | S | wave | V'/P'/lanes | note |
|---|---|---|---|---|---|---|
| sg32, shc32 | 32/32/32 + subgroup op | 3 | 32 | 32 | 99/96/99 | was refused; one instance per wave32 |
| pp32 | 32/32/32, per-primitive vec4 | 3 | 32 | 64 | 192/96/192 | was refused (per-primitive output) |
| hb32 | HB2 small shape (vec4, flat uvec3, flat ivec4; per-primitive uvec4) | 3 | 32 | 64 | 192/96/192 | |
| hb32sg | hb32 + subgroup op (HB2 regime) | 3 | 32 | 32 | 192/96/192 | 6 waves32 |
| hb32sg_rt | + runtime counts + shared memory and barriers | 3 | 32 | 32 | 192/96/192 | |
| hb32_rt / hb32_tail | runtime counts / 3D grid 7x5x2 | 3 | 32 | 64 | 192/96/192 | |
| pp16 / pp24 / pp48 / pp20 | 16/16/16, 24/24/32, 48/48/64, 20/20/20 (3 per-primitive outputs) | 5/3/2/4 | 16/32/64/32 | 64 | 158/80, 144/72, 193/96, 159/80 | |
| ppmulti32 | vec4 + ivec2 + uint + gl_PrimitiveID | 3 | 32 | 64 | 192/96/192 | |
| pplast32 | provoking vertex LAST (static) | 3 | 32 | 64 | 192/96/192 | provoking corner 2 |
| ppclip32 | + gl_ClipDistance[2] (compact) | 3 | 32 | 64 | 192/96/192 | |
| cull32 | CullPrimitive only | 3 | 32 | 64 | 99/96/99 | was refused; culled = hole |
| ppcull32 | per-primitive + CullPrimitive + runtime counts | 3 | 32 | 64 | 192/96/192 | |
| pp64_32 / pp32_64 / pp64 | 64V/32P, 32V/64P, 64/64 per-primitive | - | - | - | refused: 288 > 256 physical vertices / K*P = 128 | identical |
| pplayer32 / ppdyn32 / pptask32 | per-primitive Layer / dynamic provoking mode / Task | - | - | - | refused / not planned | identical |
| pp124 / pp128 / c128 | 64/124, 128/128 per-primitive; 128/128 | - | - | - | the base driver split + expansion | identical |

`vertex_map` (mapmodel.py): ok.

## Result at c350460: 19/19

| case | shape (V/P/L), grid | K | S | V'/P'/lanes | direct count |
|---|---|---|---|---|---|
| d32 | 32/32/32, 100 | 3 | 32 | 99/96/99 | 34 (tail) |
| d32_rss32 | same, requiredSubgroupSize 32, 99 | 3 | 32 | 99/96/99 | 33 |
| d16 | 16/16/16, 100 | 5 | 16 | 83/80/83 | 20 |
| d20 | 20/20/20, 101 | 4 | 32 | 83/80/128 | 26 |
| d24 | 24/24/32, 100 | 3 | 32 | 75/72/96 | 34 |
| sh32 | 32/32/32 + shared + 2 barriers, 100 | 3 | 32 | 99/96/99 | 34 |
| rt32 | 32/32/32, runtime counts, 100 | 3 | 32 | 99/96/99 | 34 |
| wg3d | 32/32/(8x4), WorkGroupID.xyz, 5x4x3 | 3 | 32 | 99/96/99 | 20 |
| wg3d_tail | same + runtime counts, 7x5x2 | 3 | 32 | 99/96/99 | 24 |
| sg64 | 48/48/64 + subgroupAdd, 100 | 2 | 64 | 99/96/128 | 50 |
| sg32 | 32/32/32 + subgroupAdd | - | - | refused: wave32 subgroup ops | identical |
| shc32 | 32/32/32 + constant-index shared counter | - | - | refused: wave32 subgroup ops | identical |
| sg64_32 | 32/32/64 + subgroupAdd | - | - | refused: 192 lanes | identical |
| f64 | 64/64/64 | - | - | refused: K*P = 128 > 98 | identical |
| p124 | 64/124/32 (split) | - | - | not planned | identical |
| pp32 / cull32 | per-primitive output / CullPrimitive | - | - | refused | identical |
| task32 | Task + 32/32/32 | - | - | not planned | identical |

Other regressions at c350460 with the switch off:
- `amd-mode`: 28/28.
- `split-admission`: identical output to the previous build.
- meshbench (24 shapes, 180 tests, shim) against the pre-change build: byte-identical code, registers and IBs; only the version string differs.

Correctness has not been checked by execution. The shim executes nothing and there is no CPU pixel reference yet (see DESIGN.md). Only the stage-2 index/vertex map is modeled on the CPU (mapmodel.py). **NOT hardware-run.**

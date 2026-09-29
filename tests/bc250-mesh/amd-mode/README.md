# RADV_BC250_MESH_AMD regression (drm-shim only)

`RADV_BC250_MESH_AMD=1` (default off, GFX1013 only) runs some Mesh shaders the way AMD's LLPC programs Mesh on GFX10.3. Each workgroup runs as one NGG subgroup, with no split and no private-vertex expansion. The switch is part of the compiler cache key (`compiler_info->key.bc250_mesh_amd`), so shaders compiled with the other setting are never reused.

## Eligibility (`radv_bc250_mesh_amd_route`)

A Mesh shader takes the AMD route only if all of these hold:

- There is no Task shader: none from the application, none from the base driver's hybrid Task replay, and none from the base driver's synthetic mesh-only producer.
- It has no per-primitive generic, PrimitiveId, Layer or Viewport output (`radv_bc250_mesh_needs_expansion` is false).
- It writes no CullPrimitive.
- There is no multiview.
- `radv_bc250_prepare_task` has not already split it.

Every other Mesh pipeline takes the base driver's route unchanged.

## What changes on the AMD route

| | the base driver (switch off, or an ineligible shader) | AMD route |
|---|---|---|
| split / expansion | split when P > 85, 3 private vertices per triangle | none |
| direct draw | DRAW_INDEX_AUTO (count × pieces) | DRAW_INDEX_AUTO (API count) |
| indirect / indirect count | setup dispatch + DISPATCH_MESH_INDIRECT_MULTI | DISPATCH_MESH_INDIRECT_MULTI only |
| GE_NGG_SUBGRP_CNTL | PRIM_AMP_FACTOR = P (per piece), THDS_PER_SUBGRP = 0 | PRIM_AMP_FACTOR = THDS_PER_SUBGRP = T = max(API threads, V, P) ≤ 256 |
| workgroup (lanes) | max(V', P', API threads) | align(T, wave size) |
| VGT_GS_MAX_VERT_OUT | the base driver's max(V', P', API threads) | T (the base driver's formula without the split or expansion) |
| VGT_REUSE_OFF | never written on GFX10 | 1 while bound; 0 is restored for the next NGG pipeline and at the end of the command buffer |
| post-Mesh VGT_FLUSH (`BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=1`) | after every draw | skipped (the NGG→legacy flush is kept) |

In the table, V' and P' are the post-split or post-expansion counts. With `BC250_TRACE_COMPILE=1` (or `BC250_TRACE_REGS=1`), the driver prints `BC250 MESH AMD route: ...` and `BC250 MESH AMD: T= WG= WAVE= V= GE_NGG_SUBGRP_CNTL= VGT_GS_MAX_VERT_OUT=`.

## Run

`ICD=<build>/src/amd/vulkan/radeon_devenv_icd.x86_64.json ./run.sh` (set `KEEP=<dir>` to keep the dumps).

Every case runs in bwrap with a fresh `/dev` (no `/dev/dri`), the noop drm-shim and `AMDGPU_GPU_ID=gfx1013`. `amdmode` also refuses to run without the shim or on a device other than GFX1013. The runner uses the the base driver launcher policy with `BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=1`, `NIR_DEBUG=validate`, `ACO_DEBUG=validateir,validatera` and `RADV_DEBUG=dumpibs`.

One command buffer records, in order:

1. a Mesh direct draw (128,3,1);
2. a VS draw;
3. a Mesh indirect draw (2 records);
4. a Mesh indirect-count draw.

It is submitted once to the shim. `check.py` reads the IB. Each case runs at wave64 and wave32, with the switch off and on. For base-route cases, the dump with the switch on must be byte-identical to the dump with the switch off. The runner exits non-zero on any mismatch or validation error.

## Expected registers (build git-f718ba9059)

Values are the same at wave64 and wave32. "Mesh draws" counts the Mesh draw packets in the IB. With the switch on, the AMD cases have 3 (direct, indirect, indirect count), no setup dispatch, VGT_REUSE_OFF 1/0/0 (Mesh/VS/end of command buffer) and no post-Mesh VGT_FLUSH.

| case | V/P/threads | route (on) | GE_NGG_SUBGRP_CNTL on | VGT_GS_MAX_VERT_OUT on | GE_NGG_SUBGRP_CNTL off | VGT_GS_MAX_VERT_OUT off | off: Mesh draws / setup dispatches |
|---|---|---|---|---|---|---|---|
| a64_124_ws128 | 64/124/128 | amd | 0x00010080 | 0x80 | 0x0000003e | 0xba | 4 / 2 (2-piece split) |
| a64_124_ws32 | 64/124/32 | amd | 0x0000f87c | 0x7c | 0x0000003e | 0xba | 4 / 2 |
| v5_256_2_ws64 | 256/2/64 (README-v5) | amd | 0x00020100 | 0x100 | 0x00000002 | 0x40 | 3 / 0 (expanded to 6 vertices) |
| v5_shared_256_2_ws64 | as v5, plus shared memory and barriers | amd | 0x00020100 | 0x100 | 0x00000002 | 0x40 | 3 / 0 |
| c128_128_ws128 | 128/128/128 | amd | 0x00010080 | 0x80 | 0x0000002b | 0x81 | 4 / 2 (3-piece split) |
| f64_64_ws64 | 64/64/64 | amd | 0x00008040 | 0x40 | 0x00000040 | 0xc0 | 3 / 0 (expanded) |
| pp32_32_ws32 | per-primitive vec4 | base | identical to off | | 0x00000020 | 0x60 | 3 / 0 |
| pp64_124_ws128 | per-primitive, 124P | base | identical to off | | 0x0000003e | 0xba | 4 / 2 |
| cull32_32_ws32 | CullPrimitive | base | identical to off | | 0x00000020 | 0x60 | 3 / 0 |
| cull64_126_ws128 | CullPrimitive, 126P | base | identical to off | | 0x0000003f | 0xbd | 4 / 2 |

On the AMD route, GE_CNTL (0x201), VGT_GS_ONCHIP_CNTL (0x00400801) and GE_MAX_OUTPUT_PER_SUBGROUP (= V) are unchanged. The workgroup is align(T, wave), for example 128 lanes for T = 124. When the API workgroup is smaller than that, `ac_nir_lower_ngg_mesh` (`handle_smaller_ms_api_workgroup`) makes the extra waves consume the workgroup barriers. The v5_shared case covers this: at wave32 there are 2 API waves and 8 hardware waves.

This validation is offline only. Nothing has run on the GPU. The hardware gate records are kept outside this repository.

## Parts (G1 bisection)
G1 (`f_ls64` with `RADV_BC250_MESH_AMD=1`) hung the board on 2026-09-25. `RADV_BC250_MESH_AMD=1` combines three parts, and each can now be switched on alone (each is its own compiler-key bit):
- `RADV_BC250_MESH_AMD_ROUTE=1`: no split or expansion, native draws, the base driver sizing (AMP = P, THDS = 0, VGT_GS_MAX_VERT_OUT = workgroup).
- `RADV_BC250_MESH_AMD_SIZE=1`: PRIM_AMP = THDS = T and workgroup = align(T, wave), with T taken from the final NIR, so the the base driver expansion is kept.
- `RADV_BC250_MESH_AMD_REUSE_OFF=1`: VGT_REUSE_OFF = 1 while the Mesh is bound.

On their own, SIZE and REUSE_OFF apply to native mesh-only shaders that the base driver did not split: no Task, no replay, no multiview. With the route part on (the full switch), they apply only to AMD-route shaders, so base-route pipelines stay byte-identical. The `part` lines in `run.sh` check each part's registers on f64_64_ws64 (wave64).

## RADV_BC250_MESH_FL0 (fast launch 0)
`RADV_BC250_MESH_FL0=1` (default off, a compiler-key bit) keeps the base driver's compilation (split, expansion, lowering) and launches mesh-only, unsplit, non-multiview Mesh shaders with GS_FAST_LAUNCH=0, Tree C's carrier launch. With fast launch 0 the hardware launches PRIM_AMP_FACTOR lanes, so AMP (and the output bound) is raised to the workgroup size (Tree C's f45f746 rule). For example, f64_64_ws64 expanded goes to AMP 192 and SUBGRP 0xc0; THDS stays 0 and GE_CNTL stays 0x201. Direct draws are unchanged (DRAW_INDEX_AUTO x*y*z). Indirect draws of an FL0 Mesh shader are skipped with a warning, because DISPATCH_MESH_INDIRECT_MULTI has never run without fast launch. `run.sh` checks the registers and the GS_FAST_LAUNCH bit.

## Fast launch 0 is the default (2026-09-25)
GS_FAST_LAUNCH=0 is now the Mesh launch mode on GFX1013 for every Mesh pipeline the base driver produces: mesh-only, expanded, split pieces, hybrid-Task replay and multiview. Only a native hardware Task stage and the experimental AMD size part keep fast launch 1. `RADV_BC250_MESH_FL1=1` restores fast launch 1. The part table's baseline is therefore the fast-launch-0 register set: f64_64_ws64 expanded gets AMP 192 (0xc0). A `RADV_BC250_MESH_FL1=1` row checks the escape hatch (the base driver's 0x40), and the `launch` lines check the GS_FAST_LAUNCH bit both ways. Indirect Mesh draws are no longer skipped on fast launch 0; the base driver's DISPATCH_MESH_INDIRECT_MULTI passed on hardware (IND1/IND2).

## Mixed launch mode (2026-09-25, supersedes "fast launch 0 is the default")
Fast launch 0 for all Mesh pipelines cost Hellblade 2 about 10 FPS in-game (52 vs 62 with the same build), although isolated benchmarks showed no difference. The driver now picks the mode per shader, automatically: raw-route and merged shaders use fast launch 0 (the only mode they passed in), and everything else (the base driver expansion, split pieces, Task replay, multiview) uses the base driver's fast launch 1. `RADV_BC250_MESH_FL0_ALL=1` restores fast launch 0 for everything (experiments). The part table's baseline is therefore the base driver's register set again (f64 expanded: AMP 0x40), with a FL0_ALL row (0xc0). The `launch` lines check the GS_FAST_LAUNCH bit both ways.

## Fast launch 0 only (2026-09-25, supersedes "Mixed launch mode")
A clean in-game A/B showed fast launch 0 for every Mesh pipeline at the base driver's 62 FPS in Hellblade 2 (the earlier 52 FPS result was an invalid A/B: RADV_BC250_MESH_FAST forced fast launch 0 in both runs). Every meshbench A/B agreed. Fast launch 0 is therefore the only Mesh launch mode: RADV_BC250_MESH_FL0_ALL and RADV_BC250_MESH_FL1 are gone. The part table's baseline is the fast-launch-0 register set (f64 expanded: AMP 0xc0), and the `launch` line checks that every stages register has GS_FAST_LAUNCH = 0.

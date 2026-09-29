# RADV_BC250_MESH_AUTOCULL offline regression

Driver-side per-triangle culling of the base driver's expanded Mesh output (`MESH_PERF/autocull`). Driver: lonewolfMT-cull `ce898e3` (branch `lonewolf/mesh-cull`), build `git-ce898e3f17` (first version: `9ce1a76`).

It runs only under the drm-shim and never touches the real GPU. `run.sh` starts every Vulkan program in bwrap:
- a fresh `/dev`, so there is no `/dev/dri`;
- `libamdgpu_noop_drm_shim.so` preloaded;
- `AMDGPU_GPU_ID=gfx1013`.

`autocull.c` also refuses to run unless the shim is preloaded and the device is the shim's GFX1013.

```sh
ICD=/path/to/your-mesa-build/src/amd/vulkan/radeon_devenv_icd.x86_64.json \
OLDICD=/path/to/your-mesa-build/icd.json ./run.sh   # KEEP=<dir> keeps the dumps
```

`OLDICD` (optional) is a copy of the build before the switch existed (`e9e6dcc`); every switch-off run is then compared byte for byte with it.

## What the switch does
`RADV_BC250_MESH_AUTOCULL=1` sets the compiler key bit `bc250_mesh_autocull` (default off).

**Candidates** (`radv_bc250_mesh_autocull_candidate`, `radv_bc250.c`): Mesh shaders that the base driver expanded to private vertices (`radv_bc250_expand_primitive_attributes`, plain, split pieces and Task-replay consumers), triangles, no multiview, a static FILL polygon mode, no viewport-index output, and 3P <= the workgroup lanes (`radv_shader_info.c`). Merged (`RADV_BC250_MESH_MERGE`) and AMD-route (raw shared-vertex) shaders are never candidates, so autocull never produces a raw shape. The lowering also skips shaders whose outputs spill to the Mesh scratch ring (fallback = the exact switch-off lowering).

**Culling** (`ac_nir_lower_ngg_mesh.c`, `ms_autocull_compact`): after the finale barrier, lane p owns primitive p. It loads the three positions from LDS and runs `ac_nir_cull_primitive`, the VS NGG culling code: all w <= 0, backface, frustum (bbox outside [-1, 1], skipped when any w < 0), small primitive. Everything is driven by the NGG culling settings and viewport user SGPRs. The Mesh shader declares them like a culling VS, and `radv_emit_nggc_settings` fills them. Dynamic cull mode and front face, y-inverted viewports, rasterizer discard (cull all), conservative rasterization (settings 0: no culling) and sample locations (no small-primitive culling) therefore behave as on the VS path. An application-culled primitive (CullPrimitive) stays culled.

**Compaction:** survivors get slots in their original order (ballot + mbcnt, plus the counts of the lower waves when the primitives span several waves; one extra workgroup barrier, or two in that case). The slot -> primitive map goes to LDS. The export then uses the existing compaction + packed-triangle-vertices path (`compact_cull` / `pack_triangle_vertices`): slot j exports vertices 3j..3j+2, copies of the original corners in their original order. Every exported vertex is private and referenced (the expanded class), the provoking vertex and winding are unchanged, and per-primitive outputs follow their triangle. A fully culled workgroup allocates 0/0 (GFX10: the fully-culled workaround's dummy primitive).

**Clip and cull distances** (since the clip/cull distance support, `8c3fed4`): shaders writing them are candidates.
The packed survivor vertices load every per-vertex output, the distances included, from their source corner, so the
hardware still clips and culls with them; `ms_autocull_accept` also culls a triangle when one exported clip or cull
distance is negative at all three corners (VS NGG culling's test; it only removes triangles the hardware would drop).
`clip32` is now an applied case (it was a non-candidate before).

**Unchanged:** launch registers (GS_FAST_LAUNCH, GE_NGG_SUBGRP_CNTL, VGT_GS_MAX_VERT_OUT, GE_CNTL); primitives-generated queries count the triangles before culling.

## v2 (ce898e3): runtime skip and size policy
Hardware (GATES.md): nothing culled cost f_ls64 +10%; half the triangles back-facing gained only on k_half32 (-16%), not k_half16 (-3%) or k_half64 (+4%).

**Runtime skip.** The autocull shader holds both epilogues behind one workgroup-uniform branch on the NGG culling settings SGPR: `s_and_b32 <settings>, 3` + `s_cbranch_scc0` (3 = cull front | cull back; rasterizer discard sets both). Without face culling (cull mode NONE, conservative rasterization) it runs the switch-off epilogue: no culling test, no compaction, no packed-triangle map, no extra barrier. Frustum and small-primitive culling never run alone. The culling LDS layout only appends to the switch-off layout (`ms_autocull_layouts_compatible`; otherwise no autocull), so the skip side reads everything at the switch-off addresses.

**Size policy** (`radv_shader_info.c`). By default only wave32 shapes with 24..32 triangles and at most 96 lanes get autocull (the k_half32 / HB2 96V/32P class). Other candidates are refused (`BC250 MESH AUTOCULL policy: refused`) and compile exactly as with the switch off. `RADV_BC250_MESH_AUTOCULL_ALL=1`, used together with `RADV_BC250_MESH_AUTOCULL=1`, keeps every candidate (the 9ce1a76 policy), for experiments.

`skipcmp.py` checks the skip side in the final ACO program (`RADV_DEBUG=shaders`, "After lowering to hw instructions"; this build has no disassembler):
- the non-culling branch target, up to `s_endpgm`, must equal the switch-off program's epilogue instruction for instruction: opcodes, registers, immediates, LDS offsets, GS_ALLOC_REQ and barriers;
- the code before the branch must equal the switch-off code apart from SGPR numbering. The settings SGPR is live through the API body, so some scalar temporaries get other register numbers.

The only extra instructions are the test, the branch and 0-2 exec restores.

## Checks (check.py)
Every case runs with the switch on and off, with the the base driver launcher policy, post-Mesh VGT_FLUSH off, `NIR_DEBUG=validate` and `ACO_DEBUG=validateir,validatera`. The harness records a direct and an indirect Mesh draw, plus the extra draws of the dynamic cases.

For candidates:
- trace: candidate and applied;
- ACO ISA: `v_rcp_f32` (x/w), `v_mbcnt_lo` and `s_bcnt1` (compaction), primitive and position exports, 2 GS_ALLOC_REQ (fully-culled + live);
- IB: every Mesh draw has the expected NGG culling settings value in a SPI_SHADER_USER_DATA_GS register and the viewport SGPRs (128.0f); the settings value never appears with the switch off;
- launch registers identical to the switch off: VGT_SHADER_STAGES_EN with GS_FAST_LAUNCH=0, GE_NGG_SUBGRP_CNTL (AMP >= workgroup), VGT_GS_MAX_VERT_OUT, GE_CNTL;
- the uploaded binary has LDS < 32 KiB and scratch 0.

For `policy` cases (candidates refused by the size policy) and non-candidates: the switch-on run is byte-identical to the switch-off run (IB, NIR and ISA dumps), apart from the candidate trace line.

## Result at ce898e3: 39/39 (mv32 skipped: the device has no multiviewMeshShader)
Cases (run.sh):
- default policy, autocull applied: p32_back, p24_back, p32_front_cw, p32_none, p32_yflip, p32_dyncull, p32_dyndiscard, p32_cons, shared32, pp32, cull32, shared_pp_cull, cull_nosplit (app_cull), behind32, sg32, task32;
- refused by policy, identical to off: pol_p16 (P=16), pol_p23, pol_p48, pol_p64 (f_ls64's shape), pol_p85, pol_p32_lanes128 (wave64), pol_shared64, pol_split128 (43-triangle pieces);
- with `RADV_BC250_MESH_AUTOCULL_ALL=1`: p16_back, p48_back, p64_back, p64_none (AC1's case), p85_back, p85_w32, shared64_ls64, pp_cull_rt48, cull_nosplit48, split128, split128_cull;
- not candidates: clip32, vp32, line32, dynpoly32 (mv32 skipped).

Every applied case passes skipcmp: the skip side has 74-82 instructions, identical to the switch-off epilogue, and the skip overhead is 2-4 instructions.

Every switch-off run is byte-identical to the pre-change build (`e9e6dcc`) (39/39).

Other regressions at ce898e3 with the switch off, all with output identical to the pre-change build:
- `amd-mode`: 28 ok;
- `split-admission`: 16 ok;
- `mesh-merge`: 19/19.

Correctness has not been checked by execution. The shim executes nothing, and nothing checks the pixels. **NOT hardware-run.**

# RADV_BC250_MESH_AUTOCULL_WIDE offline regression

Driver-side triangle culling (`RADV_BC250_MESH_AUTOCULL=1`, see `../mesh-autocull/README.md`) for the wider expanded
Mesh shapes. It runs only under the drm-shim and never touches the real GPU: every Vulkan program runs in bwrap with a
fresh `/dev` (no `/dev/dri`), `libamdgpu_noop_drm_shim.so` preloaded and `AMDGPU_GPU_ID=gfx1013`, and the harness
(`../mesh-autocull/autocull.c`) refuses any other device.

```sh
ICD=<build>/src/amd/vulkan/radeon_devenv_icd.x86_64.json BUILD=<build> \
OLDICD=<icd of the build before the switch> [SEEDS=400] [ONLY=<case regex>] [KEEP=<dir>] ./run.sh
```

## What the switch does
The default autocull size policy only takes wave32 shapes with 24..32 triangles and at most 96 lanes (the measured-win
class). `RADV_BC250_MESH_AUTOCULL_WIDE` (only read together with `RADV_BC250_MESH_AUTOCULL=1`; compiler key byte
`bc250_mesh_autocull_wide`) adds the wider candidates with at least 24 triangles:

| value | admitted | hardware history of the compaction |
|---|---|---|
| unset / `0` | default policy only | - |
| `1` (`on`, `true`, `yes`, `single`) | + every candidate whose primitives fit in one wave (P <= wave size): 64 triangles in wave64 with 192 lanes (Control Resonant), 32-triangle pieces with workgroup 128, 43-triangle split pieces in wave64 (Hellblade 2 Nanite) | the one-wave compaction ran on hardware for 64 triangles in wave64 (MESH_PERF/autocull GATES.md AC1, AC3b) |
| `2` (`multiwave`) | + primitives spanning several waves: 65..85 triangles in wave64, 33..85 in wave32 | never ran on hardware (the cross-wave slot counts: 2 extra barriers) |

An admitted shape compiles exactly as with `RADV_BC250_MESH_AUTOCULL_ALL=1`. Nothing else changes: the lowering, the
runtime skip (no face culling: the switch-off epilogue), the launch registers (fast launch 0, `GE_NGG_SUBGRP_CNTL`,
`VGT_GS_MAX_VERT_OUT`, `GE_CNTL`) and the output class (3 private consecutive vertices per surviving triangle,
`GS_ALLOC_REQ` = survivors, the GFX10 fully-culled workaround for an empty workgroup). `BC250_TRACE_COMPILE=1` prints
`BC250 MESH AUTOCULL policy: wide P=.. wave=.. lanes=.. prim_waves=..` for every shape the switch admits.

## Checks (run.sh)
Every case compiles `wide.mesh` (`../mesh-autocull/shape.mesh` plus a per-workgroup culling pattern) with the base
driver's launcher policy (post-Mesh VGT flush off), `NIR_DEBUG=validate` and `ACO_DEBUG=validateir,validatera`,
records a direct and an indirect Mesh draw (and the dynamic-state draws of the case) and submits them:

- `on` (`RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=<level>`) against `off` (autocull off) with
  `../mesh-autocull/check.py`: applied cases have the culling and compaction code, the NGG culling settings and
  viewport SGPRs at every draw, the launch registers of the switch-off run, LDS < 32 KiB and no scratch, and the
  runtime skip side equals the switch-off epilogue (`skipcmp.py`; where ACO numbers a scalar register differently or
  moves scalar instructions around the settings test, `skipcmp.py --sgpr-modulo` must still match: reported as
  `runtime_skip=equal_modulo_sgpr_numbers`); refused (`policy`) and non-candidate cases are byte-identical to off;
- `nowide`: the switch without `RADV_BC250_MESH_AUTOCULL` is byte-identical to off;
- `base` / `old`: `RADV_BC250_MESH_AUTOCULL=1` without the switch is byte-identical to the build before the switch
  (`OLDICD`);
- `all`: an applied case compiles to the same dumps as `RADV_BC250_MESH_AUTOCULL_ALL=1`;
- `oracle`: the lowered NGG Mesh NIR (`BC250_MESH_NIR_DUMP`) with autocull off and on, for
  `RADV_BC250_MESH_DIRECT_READ` off, `1` and `full`, through `../direct-read/mesh_oracle --autocull`: for random inputs
  and random culling states (cull front/back, front face, small-primitive culling, viewport, precision) the survivors
  of a C transcription of `ac_nir_cull_primitive` applied to the reference's exported positions must be exactly the
  candidate's primitives, in the reference's order, each exporting (3j, 3j+1, 3j+2) whose vertex exports equal the
  reference's corners bit for bit, with `GS_ALLOC_REQ` 3S/S (1/1 plus the degenerate primitive and a NaN position when
  S = 0); without face culling the candidate must equal the reference exactly. At the end the suite requires every
  workgroup class (no face culling, none, partly and all culled) and every culling test (w, face, frustum, small
  primitive) to have occurred, and the oracle to report mismatches with each deliberately broken reference
  (`--mutate 1..4`).

Cases: level 1 applied: `w64_p64_*` (back, none, front/CW, y-flip, dynamic cull mode and front face, dynamic rasterizer
discard, conservative rasterization, runtime counts, subgroup operation, per-primitive output, application
CullPrimitive without the split), `w64_shared64` (Control's 64 shared vertices / 64 triangles, expanded), `w64_p24`,
`w128_p32` (Control's LDS-fit pieces), `split128`, `split128_cull` (43-triangle pieces), `task64`, `w32_p32_back` (the
default class); refused at level 1: 85 triangles in wave64, 48 in wave32, 16 and 23 triangles; level 2 applied:
`w64_p85_*`, `w32_p85_rt`, `w32_p48_*`, `w64_p64_l2`; refused at level 2: 16 triangles; not candidates: polygon mode LINE. `clip64` (a clip
distance) is applied since the clip/cull distance support (`8c3fed4`; survivors keep their corners' distances,
triangles with a distance negative at all three corners are culled too).

Correctness is checked by the CPU oracle on the lowered NIR, not by execution: the shim executes nothing and nothing
checks pixels. **NOT hardware-run.**

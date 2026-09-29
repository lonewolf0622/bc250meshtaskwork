# VK_KHR_fragment_shader_barycentric on GFX1013 (drm-shim only)

GFX1013 (GFX10.1) has neither `LOAD_PROVOKING_VTX` nor `PS_INPUT_CNTL.ROTATE_PC_PTR`, which the base
driver's GFX10.3+ barycentrics use. On hardware the parameter cache holds each triangle's vertices in a
per-primitive cyclic rotation of the API order, and a flat read returns the provoking vertex. The driver
recovers the rotation in the fragment shader (`radv_nir_bc250_lower_bary_rotation`): the last
pre-rasterization stage also exports its position to two free generic slots, the fragment shader reads
one per vertex (raw order) and the other flat, and the raw slot that matches (x, y, w) holds the provoking
vertex (API vertex 0 with the first-vertex convention, 2 with the last). Only pipelines whose fragment
shader reads barycentrics or `pervertexEXT` inputs are affected. Mesh shaders export the reference from
the position they already load for each exported vertex (`ac_nir_lower_ngg_mesh`,
`bc250_bary_ref_mask`): no LDS on any Mesh route.

`RADV_BC250_NO_BARYCENTRICS=1` hides the extension. `RADV_BC250_DIAG_BARY_NO_REF=1` (diagnostic) compiles
without the reference (rotation 0, wrong for rotated triangles); the suite uses it as the A side of the
oracle comparison.

Every Vulkan program runs inside bwrap with a fresh `/dev` (no `/dev/dri`), the noop amdgpu drm-shim and
`AMDGPU_GPU_ID=gfx1013`; `bary` and `pipe` refuse any other device.

```sh
ICD=<build>/src/amd/vulkan/radeon_devenv_icd.x86_64.json BUILD=<build> \
  [OLDICD=<icd of the build before, for the identity checks>] [SEEDS=64] [KEEP=<dir>] [ONLY=<regex>] ./run.sh
```

| Section | What | Checks |
|---|---|---|
| features | `bary features` | extension and feature listed by default and without hybrid Task, hidden with `RADV_BC250_NO_BARYCENTRICS=1` (Mesh stays listed by the driver); GPL / shader objects only without hybrid Task; a barycentric pipeline is refused when hidden |
| vertex | `bary.c`: VS, VS+GS, VS+TCS+TES; triangle list/strip/fan, lines, points; provoking first/last/dynamic; graphics pipeline libraries (vkd3d-proton style one library, per-part fast-linked, link-time optimized) and shader objects (linked, unlinked) without hybrid Task | plain / `BARY` / `PERVERTEX` fragment shader (`vary.frag`): pipeline and submission succeed, no NIR/ACO validation error, the expected `BC250 BARYCENTRICS` trace (reference producer and slots, or rotation 0 and why) |
| mesh | `../direct-read/pipe.c` with `../direct-read/dr.mesh`: expanded, per-primitive data, PrimitiveID, uniform, partial, clip distance, multi-store, zero output, loops, split pieces (3 and 2 with `RADV_BC250_PERF_PIECE_PRIMS=64`, batch prep), direct read, autocull (also with direct read, wide), compact LDS, implicit triangles, shared-vertex compaction (`RADV_BC250_MESH_COMPACT`, alone and with implicit triangles, direct read, autocull, split pieces, Task), compact vertex map, AMD route, raw fast route, merge, hybrid Task (plain and split), lines, points | results, trace; the Mesh LDS layout with the references equals the one without (and the plain pipeline's); CPU export oracle (`mesh_oracle --bary-ref`) on the lowered Mesh NIR: `BARY` with `RADV_BC250_DIAG_BARY_NO_REF=1` vs `BARY`, the same for `PERVERTEX`, and plain vs `BARY` — identical exports apart from two reference parameters per exported vertex, each equal to that vertex's position export; lines and points: no reference, byte-identical Mesh NIR |
| identical | with `OLDICD` | every plain pipeline of the vertex and mesh sections is byte-identical to the base build (`BC250_CAPTURE_POLICY_SHADERS` code and configuration); switches the base build does not have (`RADV_BC250_MESH_COMPACT`) are compared switched off |
| isa | with `KEEP` | `RADV_DEBUG=shaders` dumps of representative pipelines and per-shader code sizes (`$KEEP/isa/sizes.txt`) |

`bary.frag` is `../direct-read/dr.frag` with the barycentric variants: it reads every `dr.mesh` output
for the same `-D` options, so the Mesh shader is the same for all fragment shader variants (except where
the compact vertex map refuses barycentric fragment shaders).

Known limitation (by design, never a pipeline failure): a fragment shader compiled without its producer
(fast-linked fragment shader library, unlinked shader object; both hidden with hybrid Task) or a producer
without two free parameters uses rotation 0, exact only for primitives the hardware does not rotate.

# piece-ceiling: expanded Mesh LDS fit offline regression (drm-shim only)

An expanded Mesh shader (every triangle owns three private vertices) keeps its output staging and its expanded outputs
in LDS. When they do not fit, the NGG lowering would move outputs to the Mesh scratch ring, which is not validated on
GFX1013. Pipeline creation now compiles such a pipeline again instead of refusing it (`radv_bc250_fit_retry` in
`src/amd/vulkan/radv_pipeline_graphics.c`): first with the dead shared-variable copies dropped, then with one more,
smaller piece per retry (at most 5), in the same launch class (private vertices, fast launch 0, no scratch ring).
`RADV_BC250_MESH_COMPACT_LDS=1` (default off) gives every expanded shader the compact LDS layout.

    ICD=<radeon_devenv_icd json> [OLDICD=<icd of a build without the LDS fit>] [KEEP=<dir>] ./run.sh

- `pipe.c`: one Mesh (+Task) + fragment pipeline; records a direct, an indirect (2 records) and an indirect-count Mesh
  draw, submits them to the shim; refuses any device that is not the shim's GFX1013.
- `nanite.mesh`/`.frag`: Hellblade 2's Nanite interface (128 lanes, 256 V / 128 P, per-vertex vec4 / flat uvec3 /
  flat ivec4, a per-primitive uvec4 written one component at a time; the fragment shader reads tc0, tc2 and tc7.xy).
  `-DEXTRA=1` adds a per-vertex vec4 that fits neither 2 nor 3 pieces.
- `wide.mesh`/`.frag`/`.task`: Control's G-buffer interface (64 lanes, 64 V / 64 P, 8 generic outputs, 5 render
  targets); `-DFIT=1` 7 outputs (fits unsplit), `-DPRIMID=1` a fragment shader reading PrimitiveID (the split refuses
  it), `-DTASK=1` with an application Task stage (hybrid Task path).
- `run.sh`: 14 pipeline cases (default ceiling unchanged, `RADV_BC250_PERF_PIECE_PRIMS=64/85`, `COMPACT_LDS`, more
  pieces for the wider variant, the Control interface split into 2 pieces also with autocull, split batch prep and a
  Task stage, the fitting variant unchanged, the PrimitiveID refusal) checking the result, the pieces, the
  `BC250 MESH LDS` layout and the retries (`BC250_TRACE_COMPILE=1`), under NIR and ACO validation; plus the disk
  shader cache in two processes with and without `BC250_CACHE_PLAN` (a hit restores 2 pieces; without the plan the
  retried pipeline is compiled again). With `OLDICD`, every case the old build admits must be byte-identical
  (`BC250_CAPTURE_POLICY_SHADERS=1`).

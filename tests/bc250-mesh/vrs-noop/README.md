# No-op VK_KHR_fragment_shading_rate on GFX1013 (drm-shim only)

GFX10.1 has no VRS hardware (`PA_CL_VRS_CNTL`, `GE_VRS_RATE`, `DB_VRS_OVERRIDE_CNTL`, the HTILE VRS encoding and
`PA_CL_VS_OUT_CNTL.USE_VTX_VRS_RATE` exist only on GFX10.3+). vkd3d-proton reports DirectX 12 feature level 12_2 only
with VRS tier 2, so the driver exposes `VK_KHR_fragment_shading_rate` on GFX1013 (on by default,
`RADV_BC250_VRS_NOOP=0` hides it, `=1` forces it) with pipeline, primitive and attachment rates, 8x8 attachment
texels and non-trivial combiners, and always shades at 1x1:

- `radv_nir_bc250_vrs_noop` (right after spirv_to_nir): the `PrimitiveShadingRateKHR` output of VS/TES/GS/Mesh
  becomes a temporary and its stores go away; `ShadingRateKHR` reads in the fragment shader are 0 (1x1). No BC250
  Mesh route ever sees the output.
- `vrs_may_be_enabled` is never set in the graphics state key (rate state does not change the fragment shader and
  does not refuse the compact vertex map or merge stage 2).
- No VRS register or field is written on GFX10.1; rate attachments are accepted and ignored (no rate-to-HTILE copy).

Every Vulkan program runs inside bwrap with a fresh `/dev` (no `/dev/dri`), the noop amdgpu drm-shim and
`AMDGPU_GPU_ID=gfx1013`; `vrs` refuses any other device (the control section uses the shim's NAVI21 with
`VRS_CONTROL_DEVICE=NAVI21`).

```sh
ICD=<radeon_devenv_icd json or frozen icd> [OLDICD=<icd of the build before, for the identity checks>] \
  [KEEP=<dir>] [ONLY=<case regex>] [CONTROL=0] ./run.sh
```

| Section | What | Checks |
|---|---|---|
| features | `vrs features` | extension, the three rate features, 8x8 texels, non-trivial combiners, rates 2x2/2x1/1x2/1x1, R8_UINT attachment format feature, image format query with the usage and a device with every rate feature: by default and with `=1` (with and without hybrid Task); nothing with `=0` (and a rate pipeline is refused); `primitiveFragmentShadingRateMeshShader` stays 0; the shader object binary UUID differs between the two states |
| vertex | `vrs.c` VS and VS+GS pipelines: one pipeline, GPL (vkd3d-proton style library), linked shader objects; rate modes none, static, dynamic (`vkCmdSetFragmentShadingRateKHR`), attachment (+depth), attachment-nodepth, renderpass (VkRenderPass2) | with a rate-writing pre-rasterization stage and a rate-reading fragment shader (`WRITE_RATE` / `READ_RATE`) against the same pipeline with plain shaders: result and submission, no NIR/ACO validation error, the `BC250 VRS NOOP` trace (output stripped, read zeroed), byte-identical code and configuration (`BC250_CAPTURE_POLICY_SHADERS`), a byte-identical command stream (`RADV_DEBUG=dumpibs`), `check_ib.py` (no VRS register by name or offset, no VRS field); static/dynamic rate state = no rate state, bound = unbound rate attachment (same command stream and code) |
| mesh | the same with `../direct-read/dr.mesh -DVRS=1` (per-primitive `gl_PrimitiveShadingRateEXT`) and `dr.frag -DREAD_RATE=1` on every BC250 Mesh route: expanded, per-primitive data (with and without a redeclared block), PrimitiveID, uniform, partial, clip, multi-store, zero output, loops, split pieces (3, 2 with `PIECE_PRIMS=64`, batch prep), direct read, autocull (also direct read, wide), compact LDS, implicit triangles, shared-vertex compaction (`RADV_BC250_MESH_COMPACT`, also with implicit triangles, direct read, split pieces and autocull; compared switched off in the identity check against the older build), compact vertex map, AMD route, raw fast route, merge (also stage 2), hybrid Task (plain and split), the game launcher policy, lines, points; modes none, dynamic, attachment | the vertex checks, plus the same `BC250 MESH` route trace as the plain pipeline |
| cache | `nircache` with a disk cache | a NIR cache hit is still stripped (code = plain); the NIR compiled with the switch on is not reused with it off |
| identical | with `OLDICD` | with `=0`, every plain pipeline of the vertex and mesh sections is byte-identical (code and command stream) to the base build and `vrs features` prints the same line; by default (on) the plain pipelines are byte-identical too |
| control | the attachment workload on the shim's NAVI21 (GFX10.3, real VRS) | `check_ib.py` must report `PA_CL_VRS_CNTL`, `GE_VRS_RATE`, the HTILE VRS encoding, the extra rate-to-HTILE dispatch, and bound != unbound: the checks can fail |
| isa | with `KEEP` | `RADV_DEBUG=shaders,spirv` dumps of representative pipelines in `$KEEP/isa` and `$KEEP/isa/grep.txt`: the SPIR-V has the built-ins; the NIR/ISA has no named shading rate leftover, no rate position export (`exp ... pos1`) and no ancillary rate unpack (`v_bfe_u32 x, 2, 2` / `x, 4, 2`); the same rate shaders on the shim's NAVI21 (GFX10.3 control) have both |

`check_ib.py <dumpibs stderr> [dispatches]` can be used on any `RADV_DEBUG=dumpibs` output.

Conformance: the rates are accepted and ignored, so real-hardware CTS rate checks
(`dEQP-VK.fragment_shading_rate.*`) fail by design; `primitiveFragmentShadingRateMeshShader` is not exposed.

# clip-cull-distance: ClipDistance / CullDistance through the BC250 Mesh routes (drm-shim only)

Mesh shaders that write per-vertex `gl_ClipDistance` / `gl_CullDistance` used to be refused by the split
(`BC250 mesh split rejected: special vertex output`), so every such pipeline above the piece ceiling failed with
`VK_ERROR_FEATURE_NOT_PRESENT` (FINAL FANTASY VII REBIRTH's 128-triangle Mesh shaders write `CullDistance[1]`).
What the driver does now (`radv_bc250.c`, `ac_nir_lower_ngg_mesh.c`):

| route | clip/cull distances |
|---|---|
| split (direct mesh-only split, hybrid Task replay, LDS-fit pieces) | per-vertex built-ins pass through unchanged like the position (every piece re-executes the body and writes all its vertices; only primitive arrays are sliced): ClipDistance, CullDistance and PointSize are admitted |
| private-vertex expansion | the merged compact array (`CLIP_DIST0/1`) is copied per expanded vertex, only the elements the application writes (all of them for a dynamic index): the three copies of a triangle's corner carry the same values, so hardware clipping and culling decide as for the unexpanded shader, and the enabled distances (`PA_CL_VS_OUT_CNTL`, from the final shader's written components) are the same |
| `RADV_BC250_MESH_DIRECT_READ` | `CLIP_DIST0/1` are direct-read locations like the position |
| `RADV_BC250_MESH_AUTOCULL` | candidates now; survivors export their corners' distances, and a triangle with one exported clip or cull distance negative at all three corners is also culled (as VS NGG culling); the viewport index still makes a shader a non-candidate |
| `RADV_BC250_MESH_COMPACT` (shared-vertex compaction) | every per-vertex output, the distances included, is exported from the shared vertex's source corner; with autocull the survivors are decided with the same distance test |
| compact LDS, implicit triangles, barycentrics, raw AMD route | unchanged code paths, checked here |

**Default forms since b1f639e** (the FF7 hang, `MESH_PERF/ff7hang` "Fix"): no Mesh pipeline exports a clip/cull
distance position vector (`SPI_SHADER_POS_FORMAT` POS1/POS2, `PA_CL_VS_OUT_CNTL` CCDIST) unless
`RADV_BC250_MESH_ALLOW_POS1=1`:

| switch (default) | what |
|---|---|
| `RADV_BC250_MESH_CLIPCULL_CONST` (1) | constant non-negative distances (not NaN, not +Inf; `-0.0` counts as 0) are removed on the variables before every BC250 rewrite; an array left without stores is dropped with its array size. Not when the fragment shader reads the distances. |
| `RADV_BC250_MESH_CULLDIST_CULL` (1) | remaining cull distances are culled in the shader and not exported (autocull, also without `RADV_BC250_MESH_AUTOCULL`, on every draw: no runtime skip); one-wave size rule; > 64 triangles are split for it |
| `RADV_BC250_MESH_ALLOW_POS1` (0) | admit the clip/cull vector form (hardware gates only); otherwise a non-constant clip distance, or a cull distance autocull does not take (viewport index output, raw shapes), is refused with `BC250 Mesh pipeline refused: ...` |

The GFX10 fully-culled dummy vertex exports every declared position vector. The legacy variants of the suite run with
`RADV_BC250_MESH_ALLOW_POS1=1 RADV_BC250_MESH_CULLDIST_CULL=0` (the vector form a hardware gate would test, and the only
one with a raw reference); the `def` / `defgame` / `defshared` / `pos` / `legconst` variants check the defaults
(column 6 of the case table: `refuse`, `cull`, `none` with a byte-identical twin (column 7), `same`), see run.sh.
New fixtures: `CONSTCLIP` / `CONSTCULL` with `CVAL` (constant element stores), cull-only variants of the split,
Task, CullPrimitive, per-primitive, partial, special-value and PointSize cases, and `k80` (the culling split).

Other built-ins in the split: per-primitive PrimitiveId, Layer and ViewportIndex are sliced like generic
per-primitive attributes (the expansion already makes them flat per-vertex outputs; a fragment shader reading them
is still refused); the primitive shading rate stays refused (GFX10.1 has no VRS; the expansion does not handle it).

```sh
ICD=<build>/src/amd/vulkan/radeon_devenv_icd.x86_64.json BUILD=<build> \
  [OLDICD=<icd of the build before, for the identity checks>] [SEEDS=60] [KEEP=<dir>] [ONLY=<regex>] ./run.sh
```

Every Vulkan program runs inside bwrap with a fresh `/dev` (no `/dev/dri`), the noop amdgpu drm-shim and
`AMDGPU_GPU_ID=gfx1013` (`../direct-read/pipe.c` refuses any other device). Fixtures: `cc.mesh` (`-D` options in its
header: `NCLIP`/`NCULL` 0..8, dynamic index, sparse writes, NaN/Inf/-0.0, PointSize, per-primitive Layer /
ViewportIndex / generic / CullPrimitive, partial counts, push constants, Task payload), `cc.frag` (optionally reads
the clip/cull inputs and barycentrics), `cc.task`.

Per case (26, plus the autocull distance-culling and shared-vertex coverage checks and three self-checks): variants off, raw (`RADV_BC250_MESH_AMD=1`, the unsplit unexpanded shader), `PERF_PIECE_PRIMS=64`,
`DIRECT_READ=1`/`full`, `COMPACT_LDS`, `IMPLICIT_TRIS`, `RADV_BC250_MESH_COMPACT=1` alone and with autocull, autocull (with and without direct read), the game switch set
of `run-bc250.sh` with and without autocull, and a barycentric fragment shader; each must compile and submit without
NIR/ACO validation errors or split refusals, with the expected traces (pieces, direct read of the clip/cull slots,
autocull applied or refused for the viewport index). The CPU oracle (`../direct-read/mesh_oracle`) then compares:

- **geometry** (`--geometry --grid`): the raw shader against the split/expanded variants: the same live triangles
  in order and, per corner, identical position, misc vector and clip/cull distance exports;
- **pieces** (`--geometry --geometry-params`): 3 pieces against 2 pieces, parameters included;
- **exact**: off against direct read, compact LDS and implicit triangles;
- **autocull** (`--autocull`): the culling reference now includes the distance test; survivors must export exactly
  their corners' distances;
- **shared** (`--compact`, and `--compact --autocull` with autocull): the shared-vertex export's hang-safety rules on
  adversarial index data and every corner's exports, distances included (exact where the compaction is refused, e.g.
  CullPrimitive, per-primitive outputs, the viewport index); at least one case with distances must apply it;
- **bary** (`--bary-ref`).

Every oracle line of a case with distances must have compared clip/cull channels (`clipcull_channels`). Self-checks:
the geometry oracle fails against a shader with one wrong distance (`SKEW`), and `--autocull --mutate 6` (no distance
test) fails, alone and with `--compact`. With `OLDICD`, the cases without distances are byte-identical to that build in every variant.

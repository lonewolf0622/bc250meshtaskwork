# direct-read: RADV_BC250_MESH_DIRECT_READ offline regression and export oracle (drm-shim only)

The expanded Mesh shaders (every triangle owns three private vertices) stage the application's outputs in LDS and
end with a loop that copies them to the expanded vertices; the NGG lowering keeps those copies in a second, expanded
LDS layout (16 bytes per location per expanded vertex) and the epilogue reloads them. `RADV_BC250_MESH_DIRECT_READ`
(default off, compiler key) removes that second copy:

| part | where | what |
|---|---|---|
| `export` (`=1`) | `ac_nir_lower_ngg_mesh.c` (`ms_direct_read_analyze`) | a location whose value in the terminal expansion loop is a pure function of the loop's vertex index (constants, ALU, LDS loads in the loop) is exported by recomputing that function for the exported vertex; its stores and LDS record go away, its output metadata stays. Other locations keep their layout. |
| `uniform` | `radv_bc250.c` (`bc250_uniform_staging`) | a staged output only stored (outside loops) with workgroup-uniform values uses one LDS cell instead of one element per vertex or primitive. |
| `dead` | `radv_bc250.c` | the dead shared-variable copies are always dropped (as the LDS fit retry does). |
| `index16` | `radv_bc250.c` (`bc250_narrow_index_staging`) | the staged primitive indices are a flat 16-bit array (the Mesh shader has at most 256 vertices; loads zero-extend). |
| `corner` | `radv_bc250.c` (`bc250_index_corner`) | the expansion loop loads only the corner a vertex copies. |

`=full` enables all parts, `=<part>,<part>` any subset.

    ICD=<radeon_devenv_icd json> BUILD=<mesa build dir> [OLDICD=<icd of the base build>] [SEEDS=200] [KEEP=<dir>]
    [ONLY=<case regex>] ./run.sh

- `mesh_oracle.c` (built by `build_oracle.sh` against `BUILD`'s static NIR libraries, with the same preprocessor
  definitions as the NIR sources): executes the lowered NGG Mesh NIR that the driver dumps with
  `BC250_MESH_NIR_DUMP=<dir>` for one hardware workgroup on the CPU, for many seeds, and compares a reference and a
  candidate compile of the same pipeline: GS_ALLOC_REQ, every export of every lane (target, channels, flags; a channel
  defined in the reference must be bit-identical in the candidate), side effects. The model: lock-step lanes with
  structured control flow, per-wave subgroup operations, LDS with byte definedness, workgroup barriers as epochs and
  cross-wave race detection, the Mesh scratch ring, and external inputs (buffers, push constants, descriptors, images,
  workgroup ids) hashed from the seed and the access so both compiles see the same values. The application's staged
  primitive indices are kept below the vertex count so that workgroups export real geometry. Seeds where the reference
  is undefined (counts above the maximum, invalid indices, out-of-bounds LDS, races, runaway loops) are skipped; the
  candidate must not be undefined where the reference is not. `--print` prints a dump, `--list` its intrinsics.
  `--bary-ref` (used by `../barycentrics`): the candidate also exports the barycentric vertex-order reference, two
  extra parameters per exported vertex that must equal its position export; they are checked and removed before
  the exact comparison (every pair of parameter numbers is tried: clip/cull distances read by the fragment shader
  are numbered after the references). Clip/cull distances (the position exports after the misc vector) are compared
  like every other export; the summary counts them (`clipcull_channels`), and the `--autocull` reference also culls a
  triangle whose exported clip or cull distance is negative at all three corners, also with `--compact --autocull`
  (`--mutate 6` drops that test; `--mutate 5` is the `--compact` primitive swap).
  `--geometry [--pieces-a N] [--pieces-b M] [--grid] [--geometry-params]` (used by `../clip-cull-distance`) compares
  two compiles whose workgroups do not match lane for lane, e.g. the raw unsplit shader of `RADV_BC250_MESH_AMD=1`
  against split pieces and the private-vertex expansion, or two piece counts: one API workgroup runs as N hardware
  workgroups of A (index L*N + k) and M of B, and the live triangles of all pieces, in order, must have identical
  primitive exports (apart from the vertex indices) and, per corner, identical position, misc vector and clip/cull
  distance exports (and parameters with `--geometry-params`, when both share the varying layout). `--grid` makes
  NumWorkGroups and every 32-bit global load read 4096 so both derive the same WorkGroupID from the index (shaders
  without their own global loads).
- `dr.mesh`/`dr.frag`/`dr.task`: fixtures selected by `-D` options (see the header of `dr.mesh`); `pipe.c` creates
  one pipeline, records a direct, an indirect and an indirect-count draw and submits them to the shim.
- `run.sh`: 28 cases compiled with the switch off, `=1` and `=full` under NIR and ACO validation, with the launcher
  policy (post-Mesh VGT flush off): pipeline result and submission, the `BC250 MESH DIRECT READ` trace, and the
  oracle on every comparable pair of compile attempts. Cases: uniform and divergent flat outputs (uniform cells,
  two stores of different uniform values, a store in a loop), partial writes and zero counts, arrayed outputs,
  per-primitive outputs with and without CullPrimitive, PrimitiveID, a clip distance (exported directly like the position),
  repeated and cross-invocation stores after barriers with application shared memory, index components written
  separately and read back, empty workgroups, lines and points (PointSize: fallback), a Task stage, vertex and
  primitive loops, 256-vertex split pieces (default ceiling, `RADV_BC250_PERF_PIECE_PRIMS=64`,
  `RADV_BC250_SPLIT_BATCH_PREP=1`, `RADV_BC250_MESH_COMPACT_LDS=1`), 32-triangle shapes with autocull, the compact
  vertex map (`BC250_COMPACT_VERTICES=true`: locations derived from the primitive count keep their layout), and the
  Control-like interface that needs the LDS fit retry with the switch off (`../piece-ceiling/wide.*`; with the switch it
  fits unsplit, so the admitted attempts have different pieces and are not comparable; the first attempts, where the
  reference uses the Mesh scratch ring, are compared). Every part alone on two cases. With `OLDICD`, the switch-off
  shaders (variable unset and `=0`) of every admitted case are byte-identical to that build. An oracle self-check
  (two different shaders must not compare equal).
- Clip/cull distance forms (dump header word 10: exported clip / cull components, cull distances culled in the shader):
  when B exports fewer of them than A (`RADV_BC250_MESH_CLIPCULL_CONST` removed constants, or
  `RADV_BC250_MESH_CULLDIST_CULL` culls them in the shader), every mode compares B with A's exports minus the removed
  clip and/or cull part (packed again, DONE on the last position export); a B that culls its cull distances has no
  runtime skip, so every `--autocull` state goes through the reference decision (all of A's distances). The GFX10
  fully-culled dummy must export every position vector B declares. A dump of an older build (no word 10) only
  compares against a B without clip/cull exports (e.g. FF7 on the hang build vs the fix).

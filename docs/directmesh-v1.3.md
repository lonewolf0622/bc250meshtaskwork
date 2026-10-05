# DirectMesh v1.3

The driver can draw eligible Mesh-only triangle pipelines through a compute index pass and an ordinary indexed
vertex/fragment pipeline. This avoids repeated Mesh export checks and pieces for larger meshlets. Generic flat
per-primitive attributes are carried on a private provoking corner, using either static provoking-vertex mode.
The vertex and primitive slices retain their original invocation ownership. Ballot, shared-memory and subgroup
operations can remain in index work; a live cross-invocation dependency in vertex or primitive attribute work
keeps the protected Mesh route. Task pipelines keep their existing routes.

`RADV_DIRECTMESH=1` enables these additions. An explicit value, including `0`, always wins:

| Variable | Addition |
|---|---|
| `RADV_BC250_MESH_LDS_PLAN` | Plan shared output storage from proven output intervals |
| `RADV_BC250_MESH_LDS_COVER` | Avoid redundant shared output storage where coverage is proven |
| `RADV_BC250_MESH_MERGED_CHECK` | Merge compatible small Mesh workgroups |
| `RADV_BC250_MESH_PP_DIRECT` | Carry primitive attributes through the direct Mesh route |
| `RADV_BC250_MESH_PIECES_65` | Reduce pieces for eligible larger primitive declarations |
| `RADV_BC250_MESH_IDXPASS` | Compute index work followed by indexed drawing |

## Routing and storage

At most 32 declared vertices stay on Mesh, following the measured performance advantage of the single-wave route.
Single-record indirect calls also stay on Mesh: repeated calls cannot amortize the setup scan. Multi-record indirect
and indirect-count calls retain the index route. Dynamic provoking-vertex state, primitive builtins, unsupported
fragment inputs, task payloads, external memory side effects, multiview render passes and simultaneous-use
recordings keep Mesh, as do draws recorded while a pipeline-statistics or primitive-count query is active.
Declining this optional route never refuses the application pipeline.

Multiview: `RADV_DIRECTMESH=1` also sets `RADV_BC250_MESH_MULTIVIEW_VTX=1` (the view layer is a per-vertex export, as
GFX10.1 has no layer field in the primitive export) and `RADV_BC250_TASK_MULTIVIEW=1` (Task+Mesh draws run the Task
stage once and the Mesh stage once per view). Mesh pipelines with a view mask are then admitted on the protected
routes instead of being refused. A multiview pipeline that every protected route still refuses is compiled once more
with the owned and AMD routes allowed, so the largest meshlets with many per-vertex outputs are admitted too. The index
route stays off for multiview pipelines. Clears and other internal driver operations
between draws keep the index route in place for the following draws.

Index pages are private to each command buffer because recordings can remain pending or be submitted repeatedly.
A shared device reservation bounds their total with the indirect pools to 128 MiB, within the board's 512 MiB carve-out.
Each recording has at most 16 MiB of pages and an indirect pool of at most 16 MiB. When a command buffer is reset
or destroyed, its pages and pool return to a device spare list (at most 32 MiB) that later recordings take from, so
idle command buffers hold no index storage and steady-state frames create no buffers. Side-stream bodies (command
memory) are bounded to 8 MiB per recording; subsequent passes run inline. Insufficient optional storage returns to
Mesh. Pool overflow and records outside the
index encoding become an ordered Mesh suffix with original DrawIDs.

Debugging controls remain available: `RADV_BC250_IDX_POOL_BYTES` (4096 bytes through 16 MiB),
`RADV_BC250_IDX_DEBUG`, `BC250_IDX_TRACE`, and `RADV_BC250_MESH_IDXPASS_BATCH`.

## Offline evidence

PROVEN offline: all four recorded game corpora compile with NIR and ACO validation clean. With indexed drawing
switched off, 1,113 captured shader binaries are byte-identical to the previous implementation with the same five
Mesh additions enabled. Per-primitive pipelines are admitted for 22 Hellblade II captures; the 50 small captures
stay on Mesh. Index admission is 227/229 for Final Fantasy VII Rebirth, 33/43 for Control, 22/72 for Hellblade II,
and 0/2 for Crimson Desert. The remaining ballot captures still require ballot-dependent vertex work. Crimson's
other capture writes external memory and conservatively keeps Mesh.

PROVEN offline: independent VS reference shaders and both static provoking-vertex modes compile and record cleanly;
indices-only ballot work is admitted, ballot-dependent vertex work is declined, and exhausting the device storage
budget keeps recording valid through Mesh fallback. A clear between two draws keeps both on the index route, a
multiview pipeline is declined, an active statistics query keeps the draw on Mesh, and 80 recordings whose command
buffers are freed or reset reuse storage without fallback. Noop execution does not render pixels or measure GPU
speed.

## Hardware results (BC-250, 2000 MHz)

- Mesh CTS (`dEQP-VK.mesh_shader.*`): 1,902 passed, 0 failed; the remaining 26,142 cases are not supported
  (NV mesh extension and features this GPU does not expose). This includes every multiview case.
- Image checks against independent vertex-shader references: seven connectivity shapes, primitive attributes at 64
  and 128 vertices with both provoking modes, index-only ballots, a clear between two draws, an active statistics
  query, 80 reset command buffers, storage exhaustion, direct and indirect draws, lattice shapes, 1,000 calls, mixed
  Mesh and vertex-shader draws and indirect-pool overflow. All pass; no GPU faults.
- Speed at 2000 MHz, 1 million triangles per draw set. v1.3 against v1.2 (the same bench and board with v1.2's
  Mesh routes), and the indexed draws against v1.3's own Mesh route (which the other v1.3 additions made faster):

| Meshlet | Direct: vs v1.2 | Direct: vs v1.3 Mesh route | Indirect: vs v1.2 | Indirect: vs v1.3 Mesh route |
|---|---|---|---|---|
| 256 vertices | 213 vs 401 us (1.88x) | 213 vs 380 us (1.78x) | 216 vs 396 us (1.83x) | 216 vs 377 us (1.74x) |
| 128 vertices | 190 vs 337 us (1.77x) | 190 vs 304 us (1.60x) | 198 vs 335 us (1.69x) | 198 vs 293 us (1.48x) |
| 96 vertices | 200 vs 338 us (1.69x) | 200 vs 327 us (1.64x) | 212 vs 326 us (1.54x) | 212 vs 315 us (1.48x) |
| 64 vertices | 194 vs 208 us (1.07x) | 194 vs 200 us (1.03x) | 204 vs 210 us (1.03x) | 204 vs 201 us (0.98x) |
| 32 vertices | 158 vs 158 us | stays on Mesh | 158 vs 158 us | stays on Mesh |

  64-vertex meshlets are 3-7% faster than with v1.2, mostly from the Mesh-route additions; the indexed draws add
  little at that size. Mixed Mesh and vertex-shader draws gain like direct draws (1.60-1.79x over the v1.3 Mesh route).

Game frame rates depend on how much of a frame is Mesh drawing and on meshlet size; they were not measured for this
release. Compare a game with `RADV_BC250_MESH_IDXPASS=0` to see what the indexed draws change there.

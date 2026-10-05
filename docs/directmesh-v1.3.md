# DirectMesh v1.3 candidate

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

## Evidence and remaining validation

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

UNPROVEN for this candidate: hardware image identity, Mesh CTS, game FPS and performance with the bounded pools.
Earlier index-route hardware checks passed and common larger meshlets improved, but those measurements precede
these changes. The earlier Control measurement is inconclusive because the board configuration was subsequently
corrected. No hardware workloads were run while preparing this candidate.

## Test plan

Run the frozen image and indirect/overflow kits first, then the Mesh CTS kit. Images must match the independent VS
reference and be nonempty. Each hardware kit is pinned, requires GPU recovery disabled, runs once and stops on
failure. Stop on a hang and retain its logs; do not retry before investigating.

For Control and Hellblade II, use the same save, scene, graphics settings, camera route and corrected board setup.
Keep the CPU and GPU quiet. Warm each configuration, then measure three runs with v1.2, v1.3, and v1.3 plus
`RADV_BC250_MESH_IDXPASS=0`. Record median FPS, frame-time distribution, visible corruption and hangs. Compare
v1.3 against its index-disabled variant to isolate this change, and against v1.2 to evaluate the complete preset.
Hellblade II's existing standard-scene result is about 70 FPS; it is a reference to remeasure, not a candidate result.
Do not publish or install this candidate system-wide until these checks pass.

# draw-cost: CPU cost switches for split indirect Mesh draws (drm-shim only)

Three runtime switches (default off) and one default change for the recording cost of split Mesh draws
(Hellblade 2's Nanite draws: mesh-only, 256V/128P, split, drawn by vkd3d-proton's ExecuteIndirect as
vkCmdDrawMeshTasksIndirect(Count)EXT):

- `RADV_BC250_SPLIT_PREP_FREE=1` (compiler-key bit `bc250_split_prep_free`): a split indirect draw whose
  result does not depend on primitive order is recorded as one indirect draw per piece reading the
  application's records, with `bc250_constants.split_piece = piece + 1`, instead of a setup dispatch
  (plus CS_PARTIAL_FLUSH and cache invalidations) that multiplies the records. Admission: the fragment
  shader has no outputs, no framebuffer fetch or interlock, writes memory only through commuting integer
  atomics with unused results and reads no writable storage memory (`bc250_fs_order_independent`), and at
  the draw there are no depth writes and no stencil test on a present depth/stencil attachment
  (`bc250_split_order_free_now`). Everything else keeps the setup. Design: comments above
  `bc250_draw_split_prep_free` and `bc250_split_flat_index` in `src/amd/vulkan/radv_bc250.c`.
- `RADV_BC250_SCRATCH_REUSE=1`: split indirect records and split-batch lists come from the command
  buffer's upload buffer, and the other BC250 scratch (Task records/payloads) from the transient arena
  also without ONE_TIME_SUBMIT. Without it every split indirect draw of a command buffer begun without
  ONE_TIME_SUBMIT (every vkd3d-proton command list) created, mapped and later destroyed its own buffer
  object.
- `RADV_BC250_SPLIT_LEAN_SETUP=1`: the per-draw split setup saves and restores only the compute pipeline
  and the push constant bytes it overwrites instead of `radv_meta_begin/end`; with an active query or a
  bound shader object the meta path runs. Records the same packets.
- Default: the BC250 switches the draw, dispatch and submit paths used to read with `getenv()` on every
  call are read once at device creation (`radv_bc250_device_env_init`, `struct radv_bc250_device_env`).

    ICD=<radeon_devenv_icd json> [OLDICD=<icd of the build before these changes>] [KEEP=<dir>] ./run.sh

- `pf.c`: records a script (see its header): split indirect / indirect-count draws of a 256V/128P
  per-primitive shape (split=3; 2 with `RADV_BC250_PERF_PIECE_PRIMS=64`) with five fragment shaders
  (`pf.frag` KIND 0 "vis": no output, unused `atomicMax`, the order-independent Nanite pattern; 1 used
  atomic result; 2 `atomicExchange`; 3 plain store; `col`: color output), direct split draws, non-split
  Mesh and VS draws, dynamic depth test / depth write / stencil test with and without a D32S8 attachment,
  pipeline statistics queries, application compute pipelines and dispatches, push constants, secondary
  command buffers; begins the primary without usage flags (as vkd3d-proton) or with ONE_TIME_SUBMIT;
  submits twice; refuses to run without the noop drm-shim.
- `run.sh`: 14 cases, each recorded with no switch, each switch alone and all three, under
  `RADV_DEBUG=dumpibs` (the query case: `BC250_CAPTURE_RAW_IBS`, because queries crash this tree's
  packet printer). The lean run must be byte-identical to the off run. With `OLDICD`, the off run (also
  with the three switches set to 0) must be byte-identical to that build: this covers the default change.
- `check.py`: reuse vs off: the same packets and registers draw by draw except driver record addresses
  and user data SGPRs, and every setup writes its own record range. pf and all vs off: from the script it
  models which split draws are prep-free; those must be exactly `pieces` DISPATCH_MESH_INDIRECT_MULTI
  packets reading the application's records (base, stride, count, count buffer), without setup dispatch
  or CS_PARTIAL_FLUSH, differing from each other only in user data SGPRs; every other draw as in the off
  run (with `RADV_BC250_SPLIT_BATCH_PREP=1`: the batch pattern); register state equal to the off run's
  except user data SGPRs and the Mesh shader's `SPI_SHADER_PGM_RSRC*_GS` (observed: one more user SGPR,
  `USER_SGPR` 10 -> 11, for the inlined piece select).

# mesh-timer: BC250_MESH_TIMER offline regression (drm-shim only)

The in-driver GPU timer for Mesh draws (`src/amd/vulkan/radv_bc250_timer.[ch]`, environment
`BC250_MESH_TIMER*`; design and hardware steps in
the project's timer notes, kept outside this repository).

    ICD=<radeon_devenv_icd json> [OLDICD=<icd of the build before the timer>] [KEEP=<dir>] ./run.sh

- `timer.c`: records a direct Mesh draw, a VS draw, an indirect Mesh draw (2 records), an
  indirect-count Mesh draw (4 records, count 2) and a second direct Mesh draw in one primary command
  buffer (optionally in a secondary), submits twice; refuses to run without the noop drm-shim.
- `run.sh`: 22 cases. 17 Mesh cases (expanded, expanded+autocull, private 48P, split=2, split=3 Nanite, Task replay,
  MESH_FAST raw, secondary command buffer, SAMPLE=2, SKIP all / expanded / split / replay / autocull /
  amd / hash prefix / non-matching words), each with the timer on (fake CPU timestamps, print every
  submit, file) and off, under `RADV_DEBUG=dumpibs`. 5 async-compute cases (`ace.c`, below).
- `check.py`: one begin/end timestamp pair per command buffer, exactly one pair around every top-level
  Mesh draw (RELEASE_MEM BOTTOM_OF_PIPE_TS SEND_GPU_CLOCK_COUNTER into the ring), pieces / chunks /
  prep dispatches inside, the VS draw outside, no nesting, the rest of the packet stream equal to the
  off run (modulo address-valued fields, which the ring buffer shifts under the shim, and IB
  chaining), the class route, the file's I/C/K lines with the expected fake sums
  (100 V + P ticks per timed draw). With `OLDICD`, the off runs are byte-identical to that build,
  also with `BC250_MESH_TIMER=0` and the other timer variables set.
- `summarize.py <BC250_MESH_TIMER file>`: classes by GPU time, Mesh share, per-frame / per-submit /
  per-window estimate (the tool for the hardware runs).
- `gaps.py <file>`: graphics-queue idle gaps (single queue; ignores compute-queue S lines of v2 logs).

Async compute (timer v2, `BC250_MESH_TIMER_SUBMITS=1`):

- `ace.c` + `ace.comp`: one graphics queue and two compute-family queues. Per iteration:
  - compute queue 0: dispatch, signal T;
  - graphics: fill, wait T, signal T';
  - compute queue 1: dispatch, wait T', signal T'' and a binary semaphore;
  - graphics: fill, wait on the binary semaphore;
  - graphics: a submission without command buffers that only signals.
  Two iterations, then vkDeviceWaitIdle.
- `checkace.py`, per case:
  - Every GFX and COMPUTE main IB has exactly one begin/end timer pair (RELEASE_MEM BOTTOM_OF_PIPE_TS
    SEND_GPU_CLOCK_COUNTER, slot t0 / t0+8) around its work. With `BC250_MESH_TIMER_COMPUTE=0`, COMPUTE IBs have
    none.
  - The on IBs minus the timer packets equal the off IBs, ring by ring.
  - W lines: consecutive seq, the submission pattern, and waits naming the signalled semaphore values.
  - S lines per timed command buffer with q / qf / cb. T lines with raw_ns. I lines with ace_submits.
  - `queuetl.py` on the log finds the queues and splits the graphics idle.
- Cases:
  - `ace_compute`;
  - `ace_nocompute`;
  - `ace_queues1` (`RADV_BC250_COMPUTE_QUEUE_COUNT=1`: both compute submissions on queue 1.0);
  - `ace_prio_high` (`RADV_BC250_COMPUTE_QUEUE_PRIORITY=high`: the shim grants it, and the stderr note is
    checked);
  - `ace_prio_bad` (an unknown value is ignored and printed).
  With `OLDICD`, the default case's off run is byte-identical to that build, also with `BC250_MESH_TIMER=0` and
  the other timer variables set.
- `queuetl.py <file> [--from s --to s] [--hist] [--vkd3d-trace f] [--json f]`: per-queue timelines of v1 and v2
  logs. The graphics idle is split into:
  - app late (split further into until-the-awaited-signal-was-submitted and the rest);
  - driver + kernel submit;
  - waiting for another queue's work;
  - release latency after it;
  - other kernel waits.
  It also reports compute overlap and lateness, the GPU-side cross-queue wait latency, and vkd3d-proton queue
  profile blocking regions. Analysis and the hardware plan:
  the project's async-compute notes, kept outside this repository.

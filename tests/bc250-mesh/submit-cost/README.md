# submit-cost: submission-cost switches, offline regression and microbenchmark (drm-shim only)

Switches (all default off, GFX1013 only; `src/vulkan/runtime/vk_queue.c`, `vk_sync.[ch]`,
`src/amd/vulkan/radv_queue.c`, `radv_bc250_timer.c`, `winsys/amdgpu/radv_amdgpu_{cs,bo,winsys}.c`):

| switch | effect |
|---|---|
| `RADV_BC250_SUBMIT_KNOWN_SIGNALS=1` | The runtime tracks per non-shared `vk_sync` the timeline value / binary payload known to be attached in the kernel (a submission signalling it was accepted, or a host signal) and skips the zero-timeout `WAIT_PENDING` probe ioctl (`DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`) that submit mode THREADED_ON_DEMAND issues before every submission with waits. Unknown waits are still probed; wait-before-signal still moves the queue to its submit thread. |
| `RADV_BC250_LOCAL_BOS=1` | `RADV_PERFTEST=localbos` placement for this device: driver-internal buffers (IBs, upload buffers, shader arenas, query pools, rings, border colours) are `VM_ALWAYS_VALID` and leave the per-submission BO list; the winsys no longer lists local IB buffers; the user fence BO stays an ordinary BO. |
| `RADV_BC250_KEEP_IB_KB=<kb>` | At command stream reset keep the largest IB up to `<kb>` (the base driver keeps the smallest one, so every re-recorded large command buffer creates and frees IB buffers again). |
| `BC250_MESH_TIMER_ASYNC=1` | Timer submissions / presents only queue raw W / P records; a worker thread formats W / P / S lines, folds, prints, writes and fsyncs outside the timer mutex. Same lines. |
| `RADV_BC250_SUBMIT_PROFILE=1` (`RADV_BC250_SUBMIT_PROFILE_FILE=<path>`) | Every 5 s per queue: mean µs per vkQueueSubmit call split into runtime (`total_us`), probe, driver, RADV, BO list, **CS ioctl (the kernel)**, command-buffer-less submission, timer and post-submit resets; BO handles and IBs per CS; present time. `BC250_SUBMIT_PROFILE_BO`: BO and IB buffer creations per frame. |

Files:

- `ioctlspy.c`: `LD_PRELOAD` interposer placed before the drm-shim. Counts DRM ioctls per type, logs every
  `AMDGPU_CS` (IB sizes, BO list length, syncobj wait / signal entries with points and flags) and syncobj
  ioctl (`IOCTLSPY_LOG`), hands out unique syncobj handles (the shim returns 1 for all), and with
  `IOCTLSPY_SYNCOBJ=1` models syncobj availability so wait-before-signal behaves as on the kernel
  (probe returns `-ETIME`, blocking waits and a CS waiting on an unsubmitted point block until the
  signal arrives).
- `semtest.c` + `checksem.py`: 10 scenarios (cross-queue timeline chain, vkd3d serializing binary,
  binary consumed on another queue, host signal, zero-submit and command-buffer-less submissions,
  secondary command buffer, three VkSubmitInfo2 in one call, sparse bind, wait-before-signal from the
  host and from another queue). Checks the CS wait / signal entries in order, `WAIT_FOR_SUBMIT`
  flags, probes (all submissions with waits by default, only wait-before-signal ones with
  KNOWN_SIGNALS), binary resets, that waiting CSs reach the kernel only after their signal, that
  vkQueueSubmit does not block, and host-side results.
- `submitbench.c` + `bench.comp`: times vkQueueSubmit2 with vkd3d-proton-like submissions (48 graphics
  + 10 compute submissions per frame, 4 command buffers each, binary serializing semaphore wait +
  re-signal, timeline signal, cross-queue timeline waits, one fence-only submission per frame,
  command buffers re-recorded without ONE_TIME_SUBMIT, buffer-device-address memory objects, query
  pools, optional two submit threads, optional large command buffers).
- `run.sh`: the regression (14 checks). `bench.sh`: the per-switch microbenchmark matrix.

    ICD=<radeon_devenv_icd json> [OLDICD=<icd of the build before the switches>] [KEEP=<dir>] ./run.sh
    ICD=<icd> [OLDICD=<icd>] [REPS=5] [OUT=<file>] ./bench.sh

`run.sh` checks: semtest under off / each switch / all together (9), with `OLDICD` the switches-off
ioctl logs of semtest and submitbench identical to the old build (2), KEEP_IB fewer IB buffers,
timer async same W / S lines and cheaper submissions, profile lines and counts (known signals:
`probes=0.00 probes_skipped=1.00`), LOCAL_BOS empty CS BO lists.

Limits: the noop shim executes nothing, so timings are CPU-only (user space); the kernel's share of a
submission (the dominant part on hardware) is what `RADV_BC250_SUBMIT_PROFILE`'s `cs_ioctl_us`
measures on the real GPU. The shim cannot export sync files, so no scenario waits in a
command-buffer-less submission.

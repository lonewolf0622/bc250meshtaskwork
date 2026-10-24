# DirectMesh benchmark-qualified release

The release uses the unchanged, hardware-validated `c84115b` driver
(`01a7ecb52c133853528ef40e8621e4017190516180585d6745f880e15b1cbb71`).
The portable launcher enables shared setup for repeated indirect IDX calls and
trimmed command execution. The first call retains its ordinary setup; a second
call promotes the span to shared setup. Stored outputs remain an explicit
profile. Existing shader eligibility, Mesh fallbacks and Task/DGC routes are
unchanged. These defaults belong to the launcher; the library is not rebuilt.

The prior owner benchmark passed 624 exact route images and confirmed shared
setup gains in three rounds for V128/P128 indirect workloads. Existing IDX gained
1.083–1.085x against its own per-call setup; stored IDX gained 3.421–3.423x
against its own per-call setup. These ratios are not gains over Mesh or game FPS.
There were 15 stable per-round setup wins and no stable setup regressions.

The final connectivity-reuse benchmark passed all 120 route images exactly.
Independent raw-data auditing found 21 stable and three excluded comparisons,
zero stable wins, zero stable regressions and no confirmed three-round win.
Connectivity reuse is omitted from the release binary. The development source
and original frozen experiment remain available separately. No further owner
benchmark or game capture is required to close this release phase.

`launch.sh` defaults to the established indexed-route eligibility with shared
setup (`standard`). `--profile mesh` selects Mesh for lighter workloads that
favor it. `--profile stored` selects the measured heavy-work alternative.
These are explicit choices; no shader-cost heuristic is added. Explicit
environment values, including zero, override every profile default.
The launcher sets `VK_DRIVER_FILES` and the legacy `VK_ICD_FILENAMES` to the
bundled ICD when unset. It neither installs a driver nor edits game settings.
Rollback is restoring the user's saved launch options.

`test_launch.py` checks defaults, overrides, profile selection and exact argument
forwarding without Vulkan. `check_recording.py` compares all three profiles,
singleton/repeated calls and a disabled-sharing override to explicit tested
settings in 14 fresh-device noop launches, requiring identical shader programs
and command words. Actual GPU execution remains with the owner.

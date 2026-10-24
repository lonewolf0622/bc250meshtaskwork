# FF7 paired frame-time comparison

This runner measures selective indirect IDX setup using the same previously validated driver for
all three game launches. A1 and A2 disable shared setup; B1 enables it. Replay IDX remains enabled
and stored IDX remains disabled. This isolates batching from compiler changes or other IDX routes.
It does not install a driver, edit Steam configuration, start a game, or change GPU clocks/recovery.
The owner launches and plays the game.

Use a staged kit containing `game_time.py`, `manifest.json`, the validated driver and ICD, private
MangoHud configs, and a pinned `SHA256SUMS`. Run `bash run.sh`. The runner prints a complete Steam
launch option for each phase and waits for manual capture. It preserves the CSVs, filtered process
identity, accepted-submission IDX usage counters, and a summary under `results/`.

1. Close FF7 and other GPU work before starting. Note your current Steam launch options.
2. Choose one repeatable save and fixed camera position. Record your resolution, graphics settings
   and upscaler. Keep them fixed. Disable frame generation, dynamic resolution, VSync and FPS caps
   during this comparison so rendered frame time can reveal a driver difference.
3. Paste the complete generated A1 line into FF7's Steam launch options and launch it yourself.
   Load the chosen scene and warm up there for at least two minutes. Shader caches are retained.
4. Return to the terminal and press Enter. The runner checks the rendering process's mapped RADV
   library against the driver hash and verifies the phase's flags and private MangoHud config.
5. Return to FF7 and press **Left Shift + F2** once. Keep the same scene/camera for 90 seconds.
   MangoHud stops logging automatically; do not alt-tab or open menus while it is recording.
6. Answer the rendering check, exit FF7 completely, and follow the same steps for B1 and A2.
   Restore your original Steam launch options and game settings afterwards.

The private config uses `log_interval=0` to record every presented frame. The analyzer detects raw
CSV headers, validates a 90-second recording with monotonic timestamps and per-frame coverage,
and reports FPS as 1000 divided by mean frame time. It also reports median, p95 and p99 frame time,
long-frame counts, and clock/load/temperature diagnostics. It does not average instantaneous FPS
or derive frame percentiles from sparse sampling. The global MangoHud config is left untouched.

The comparison brackets B1 with both baselines. A candidate gain must exceed 5% and baseline drift,
with baseline drift at most 5%, no p99 deterioration above 10%, correct rendering reported by the
owner, and indirect IDX work observed during all captures. The usage counters describe accepted
submissions, not GPU completion; they do not count individual shared spans. If FF7 selects Mesh or
DGC instead, the summary will not attribute an apparent gain to shared setup. One scene is an
exploratory result; repeat it and sample other scenes before widening automatic route selection.

A phase's exclusive start marker preserves evidence after a stopped run. The runner may continue
between completed phases, but it never overwrites an incomplete phase or existing capture. On a
hang, crash or visual difference, stop and preserve the whole kit. Performance evidence is not an
independent image oracle for game resources and shaders.

MangoHud configuration and per-frame behavior follow its [configuration documentation](https://github.com/flightlessmango/MangoHud/blob/master/data/MangoHud.conf)
and [frame update code](https://github.com/flightlessmango/MangoHud/blob/master/src/overlay.cpp).
`python3 -B -m unittest discover -s tests/bc250-mesh/game-time` checks the analysis and mode guards
with CPU-only fixtures. No game or Vulkan workload is launched by those tests.

A kit with `comparison_kind=mesh_replay_stored` in its manifest uses five phases:
Mesh, replay IDX, stored IDX, replay IDX, Mesh. Shared setup is explicitly enabled
for both IDX routes and disabled for Mesh. Each route uses the same frozen driver.
The outer Mesh pair brackets all three IDX captures; the inner replay pair brackets
stored IDX. Both comparisons retain the same 5 percent drift/gain and 10 percent
p99 limits. Observed direct or indirect IDX calls are required for a candidate gain,
and Mesh controls must record no IDX calls. Usage counters do not prove that shared
spans executed. One scene remains exploratory even when all checks pass.

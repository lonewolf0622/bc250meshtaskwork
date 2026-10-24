`RADV_BC250_MESH_CORNER_ALIAS=1` lets the safe private-corner/adaptive Mesh
exporter reuse the API connectivity buffer as its corner source map. It is off
by default and does not change route admission or the worst-case export bound.

The checker proves that selected triangles are the complete original primitive
stream and that their indices are in range. It then skips the ballot, prefix
count, clamping and LDS map stores. Exporters select the original connectivity
buffer using a wave-uniform flag published alongside the existing counts. The
producer and publication barriers remain in place. Removed triangles, invalid
indices and compacted vertices keep the existing map. The owned per-primitive
and shared-corner planners are excluded.

This removes work but adds a uniform predicate/branch and address selection.
Performance is a hypothesis until the owner runs the frozen hardware kit.

The CPU NIR interpreter supports four new safe-direct variants:
`corners-alias`, `corners-alias-cull`, `corners-adaptive-alias` and
`corners-adaptive-alias-cull`. They compare lowered exports against the expanded
reference, cover runtime counts and empty outputs, and retain switch-off shader
and command-byte identity checks against the old frozen driver.

`corner_alias.py` uses independent VS references and all three exact flat corner
probes before timing six V64/P64 workloads and two excluded larger controls.
Each workload runs off/on/off in three reordered rounds with 500 ms sustained
warmup and 31 samples. A confirmed gain requires both full and settled medians
to improve by more than 5% in all three stable rounds, at most 5% bracket drift
and at most 10% central sample spread. Excluded timings and raw telemetry are
retained. Whole-pass timestamps cannot attribute time to a shader phase.

The frozen kit's `run.sh` is a one-shot owner GPU benchmark and requires no game
capture. CPU-only compilation/recording uses `--offline`, a fresh `/dev` in
bubblewrap, the gfx1013 noop shim, and an external `OUT` directory.

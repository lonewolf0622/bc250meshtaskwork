# Final driver comparison

Stage `final_run.py` and `audit_benchmark.py` alongside a `benchmark/` kit using
`setup_final.py`, a `game/` kit using the route comparison in `game_time.py`, the
validated driver, and pinned checksums. The root manifest identifies both the
driver and tools commits. The root checksum list includes the driver and both
component checksum lists. No installed driver or game configuration is changed
by these scripts; the owner applies and later restores the printed Steam options.

The entry command runs the benchmark once and audits all raw images, counters,
pool tables, timestamps and warmup telemetry. A completed benchmark can be skipped
when auditing saved evidence. A started but incomplete benchmark is never retried.
The default completes benchmark validation without asking for a game test.
`--game` explicitly selects the optional manual game session; `--compare` audits
completed benchmark evidence without starting GPU work. Use `--game --compare`
to audit previously completed game captures.

The standalone suite compares Mesh, replay IDX and stored IDX across eight shader
property workloads, each repeated three times. They include captured game geometry
classes, but their arithmetic and resources remain synthetic. All three repeats
must satisfy the strict focus criteria before a win is confirmed. The image policy
requires exact rotated corners and zero-step images. Heavy interpolated scored
images retain the existing one-unit RGB allowance with exact alpha/background;
the audit reports these separately from byte-exact images.

If requested with `--game`, the owner captures one FF7 scene in Mesh/replay/stored/replay/Mesh order.
All phases load the same frozen driver, and both indexed routes enable shared
setup. Mean frame time, tail frame times, bracket drift, expected route work and
visual checks determine candidate gains. One scene remains exploratory. Usage
counters do not establish the number of shared setup spans executed.

This validation does not widen automatic route selection, enable stored IDX or
shared setup by default, establish performance on all RDNA1 cards, or publish a
release. Run the CPU tests with
`python3 -B -m unittest discover -s tests/bc250-mesh/final-check`.

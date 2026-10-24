# Indexed Mesh offline checks

`RADV_BC250_MESH_IDXPASS_SETUP_BATCH=1` enables an experimental shared indirect setup within each side batch.
It remains off by default. One wave scans at most 32 consecutive API calls in order, carrying the private pool
allocation in registers and writing disjoint per-call tables. A single setup completion/cache/PFP wait precedes
their indirect producers; the final producer wait remains unchanged. Full lists, side-stream capacity and existing
barrier/state boundaries close the span. Inline paths retain per-call setup. The immutable list and count live
with the recording and are included in the existing 8 MiB side-stream upload bound.

Run `ICD=... SHIM=... OUT=... python3 setup_batch.py --offline` for the paired setup experiment, including
64/128/256-vertex profiles, both provoking conventions, direct controls, 32/33-call list boundaries, 512-call
recordings, uncounted indirect draws and small-pool overflow controls. Hardware kits require byte-exact flat
corner probes and smooth images against independent VS references, GPU pool/reset audits, nine timestamps
after two warmups, off/on/off setup brackets and Mesh timing brackets. These measurements do not change
automatic routing or establish game FPS. `setup_recording.py` covers clears, unfinished reset, 80 free/reset
cycles, query and single-record fallback, and inline operation in the DRM-free sandbox. CPU NIR tests compare
shared setup tables and pool carry against sequential setup execution across mixed record/output layouts,
overflow, invalid grids, empty/clamped counts, full lists and repeated submissions.

Build shader assets and the Vulkan harness with `./build.sh`. Run `ICD=... SHIM=... OUT=... ./offline.py`, pointing
at the candidate ICD and amdgpu noop DRM shim. The runner uses bubblewrap with a fresh `/dev`, so real GPU nodes
are unavailable. NIR, ACO and Vulkan validation are enabled.

The suite covers the seven existing connectivity shapes, flat primitive attributes at 64 and 128 vertices,
first and last provoking vertices, index-only ballots and rejection of ballot-dependent vertex work. Independent
VS shaders compute the same geometry and attributes without the Mesh converter. A memory-pressure check retains
160 independent command buffers and verifies automatic fallback after the 128 MiB reservation is exhausted.
Noop runs compile and record commands; they neither render images nor measure hardware performance.

Allocation cases cover mutually exclusive count sites, nested branches, dynamic uniform counts and a uniform
early return with no allocation. The independent VS references emit degenerate triangles past each workgroup's
logical primitive count. Their image comparisons still require a hardware run. The NIR contract unit tests also
reject sequential/repeated allocations, non-uniform branches, subgroup-only uniformity and output-before-count
paths; these rejected cases are deliberately tested as NIR, not executed as invalid Mesh shaders.

The allocation contract proves that each executing path selects at most one allocation, before output writes,
with control flow and count values uniform across the whole workgroup. It analyzes a private copy with lowered
returns. The original count sites keep their order; the compute producer's shared count and common epilogue
already provide the single final primitive count. Count values that need first-invocation selection remain
ineligible. Existing single-site admission and other route checks keep their current behavior.

With `build-radv-tests=true`, run `radv_tests --gtest_filter='radv_bc250_mesh_contract_test.*'` for the contract
unit tests. The shared structure records workgroup dimensions, output limits and declarations, subgroup
observability, invocation ownership, memory effects and task-payload use. Safe-direct, owned-corner, merge, split
and IDX admission consume these facts. Fast-route alternatives and diagnostic checks reuse the same snapshot.
Refresh it after rewriting the shader; allocation CFG proof is lazy and works on a private normalized copy.

Output dependency masks follow SSA values and enclosing branch conditions. Shared memory, writable storage,
temporary memory and loop dependencies remain explicitly unknown where the analysis cannot prove their source.
IDX keeps its final check of the optimized vertex program, so an unknown dependency does not become permission
to replay subgroup or shared-memory operations as vertex work. DGC consumes the admitted pipeline's compiled
route and cached plan; it does not reanalyze the transformed shader at command execution.

After building the assets, run `BASELINE_ICD=... CANDIDATE_ICD=... SHIM=... OUT=... python3 identity.py` to compare
both IDX settings against a baseline driver. This captures every shader upload on the CPU and checks its bytes
and resource configuration; GPU virtual placement is excluded. It uses the same DRM-free sandbox as the offline
suite. It checks compiled shaders, rather than rendered images or command timing.

The build products (`*.spv` and `index_gate`) are generated and excluded from source control. Frozen hardware kits
carry these products, the candidate driver, a SHA256 manifest, and an independent VS image comparison.

Run `python3 reference_check.py` after building the shader assets. This executes the compiled direct and
indirect VS references on CPU and compares positions and all varyings for every corner in 64 groups. It
checks the empty middle record's DrawID, the 16-by-4 grid's logical X coordinate, and disabled remapping.
Logical coordinates must be reconstructed before both connectivity and vertex attributes are calculated.
This validates the image oracle; it does not execute the driver or replace hardware image comparisons.

`INDEX_LAST_VERTEX=0` selects the first provoking vertex and `=1` selects the last. Other values are rejected.

Run `ICD=... SHIM=... OUT=... python3 benchmark.py --offline` to validate the standalone performance
fixtures with GPU nodes unavailable. It uses the same five gated shapes at 64 workgroups, with either one
draw or 16 draws in a render pass. Direct, multi-record indirect and indirect-count calls are covered;
the 128-vertex primitive-attribute fixture uses direct draws only. Each command buffer is resubmitted after
its fence completes. Timestamp queries enclose the render pass and all driver-internal producer work;
shader compilation, CPU recording and image readback are outside that interval. No clocks are changed.
Indirect IDX reservations are first audited after three submissions using a mapped debug pool: all records
must fit, with no fallback and exactly one recording's worth of pool usage. Those times are discarded;
scored trials use the normal VRAM pool with debug disabled.

A frozen benchmark kit provides the driver, shader assets, saved reference-image hashes and a one-shot
launcher. It takes two warmups and nine GPU-time samples per trial, in this order: Mesh, existing IDX
settings, stored IDX, existing settings again, Mesh again. Subgroup/shared-dependent outputs use Mesh
under the existing settings, and the report names the actual route. Both IDX routes must also match the
independent VS images. Mesh image hashes remain diagnostic because workgroup primitive order may vary.
The summary flags baseline drift above 10 percent. Timings describe these synthetic 256-by-256 workloads,
including attachment clear and repeated overdraw; they do not predict game FPS. The stored switch remains
off by default until its performance policy is measured and reviewed.

After `build.sh`, run `python3 heavy_reference_check.py` to execute the compiled heavy VS on CPU and
compare its positions, colors and primitive attributes against a separate integer oracle. It checks all
triangle corners, indirect grid boundaries, runtime iteration counts and alternate seeds. Unsigned
arithmetic wraps at 32 bits. The calculated hash affects the fragment color; it cannot be optimized away.

`benchmark.py --heavy` sweeps 0, 16, 64, 256 and 1024 runtime arithmetic iterations per vertex on 64-vertex
meshlets, 128-vertex meshlets with primitive attributes, and 64-vertex ballot-dependent outputs. Each uses
direct and indirect-count calls with one and 16 draws. The 128-vertex fixture has 64 invocations, so its
vertex loop computes two outputs per invocation. Primitive connectivity, positions and attachment size
stay fixed across the sweep. The iteration count is the third application push-constant word; compilation
does not specialize it. The loop uses dependent integer shifts, XORs, multiplication and addition. This is
an ALU diagnostic, not a skinning, texture-fetch or game simulation.

For the heavy suite, every timed workload first runs three exact corner probes on Mesh, existing IDX
settings and stored IDX. Each probe cyclically rotates the triangle so a different original corner becomes
the first provoking vertex. Flat vertex attributes bypass interpolation. The probe fragment recovers the
discrete hash bytes and primitive attributes and combines them with integer arithmetic. All nine probe
images must match independently compiled VS references byte for byte. Their times are discarded, and their
readbacks, hashes and results are retained separately in `corner_oracle_trials`.

After these probes pass, the original interpolated workload runs with a separate VS reference. Every route
must match alpha, coverage and background exactly; RGB may differ by at most one UNORM8 step per channel.
The tolerance is unavailable before the exact probes pass, and never applies to the probes themselves.
Every comparison records exactness, differing pixels and maximum channel error. A bounded difference does
not prove its cause; floating-point interpolation can differ slightly between pipelines. Earlier hardware
runs reproduced two one-step Mesh differences and one one-step existing IDX difference, with identical
coverage. This paired check retains an exact discrete-output oracle without making quantized interpolation
an exact-bit oracle. The lightweight stored-route gates keep their existing byte-exact requirements.

GPU reference hashes and readbacks are retained, and normal IDX pool
preflights still check resubmission and overflow. Run `ICD=... SHIM=... OUT=... python3 benchmark.py --offline
--heavy` to compile and record every case without GPU nodes. Frozen kits use the usual pinned, one-shot
launcher. `break-even.json` reports the first sampled iteration count with a stable gain over both baselines
exceeding 5 percent and the observed baseline drift; this is not an interpolated break-even point or an
automatic driver policy. The stored route remains off by default.

`--start-at WORKLOAD_TAG` selects the remaining sweep for a new frozen kit. It never resumes a used kit;
the original one-shot files and completed measurements remain intact.

`RADV_BC250_MESH_IDXPASS_STORE=1` selects an experimental stored-output IDX route. It requires IDX to be enabled
and remains off by default. The compute producer runs the Mesh body once with its original workgroup, subgroup
operations and shared memory, and stores vertex and generic primitive attributes alongside the index reservation.
A fetch-only vertex shader reads those outputs. Primitive attributes use the existing encoded provoking corner;
vertices are stored once rather than duplicated for each primitive. This implementation adapts geometry staging
concepts independently; it includes no AnyPS5 source code.

Stored IDX removes whole generic outputs that the fragment shader does not consume before assigning output
planes. A private fragment copy supplies input liveness; position, rasterization builtins, connectivity,
explicitly active outputs and transform-feedback outputs are retained. Partially consumed vectors and
aggregates retain their entire output plane. Removing every generic primitive attribute also removes the
private provoking-vertex encoding. `RADV_BC250_MESH_IDXPASS_PRUNE=0` disables this optimization for comparison;
the default is enabled within the opt-in stored route. Mesh and replay IDX keep their existing behavior.

`ICD=... SHIM=... OUT=... python3 prune.py --offline` audits 44 pruning workloads without GPU access.
The fixtures include unused vertex/primitive arrays, live primitive attributes, subgroup outputs, dead
primitive attributes and a control whose layout stays unchanged. Repeated draws of the wider layouts are
bounded to fit the existing 16 MiB reservation and pool, keeping timing trials on stored IDX.
Both first and last provoking vertices are covered for the primitive attribute fixtures.

The frozen hardware runner checks three exact flat corner probes against independent VS references with
pruning both off and on. It then verifies smooth images, allowing at most one UNORM8 RGB step after the exact
corner probes pass, with exact alpha and background. Indirect pool preflights verify reservation/reset tables
and discard their timings. Scored trials use the same driver with pruning off, on, and off again; each uses
nine timestamp samples after two warmups. Baseline drift above 10% excludes a timing result from the win
criterion, and a credible win must exceed both 5% and the measured drift. These timings do not select an
automatic route policy or establish game FPS.

The layout supports 32-bit scalar, vector, array, matrix and aggregate outputs. Clip and cull distance lengths
are preserved. Output reads, external writes, task payloads, fragment primitive builtins and non-owned output
stores keep their existing exclusions. Primitive culling and primitive builtins remain outside this prototype.
The existing small-meshlet and single-record indirect policies still select Mesh.

Direct allocations include all output bytes and reject oversized requests before narrowing to 32 bits. Indirect
setup reserves index and output space together; draw counts include indices only. Output data follows each
record's contiguous index section. The 16 MiB indirect pool and shared 128 MiB reservation remain bounded, with
Mesh fallback when records or allocations do not fit. Stored draws invalidate shader caches after producer
completion, and restore application push constants after the internal indexed draw. DGC keeps its existing Mesh
execution path.

Run `ICD=... SHIM=... OUT=... python3 stored.py` after building the assets for the stored-route offline suite.
It includes ballot-dependent and shared-dependent vertex outputs, both provoking corners, multiple/count indirect
calls with zero records and 2D grids, single-record Mesh policy, clear/query state transitions, storage reuse,
reservation pressure, oversized direct allocations and small indirect pools. Noop cannot execute GPU overflow
selection or render pixels; these checks validate compilation and recording only.

`radv_tests --gtest_filter='radv_bc250_idx_test.*'` executes the actual producer and fetch NIR on CPU and compares
indices and attributes with independent expected values. The indirect roundtrip includes nonzero record IDs and
firstIndex offsets. The setup oracle runs 64 CPU invocations with collective rendezvous for scans and reductions,
and checks exact-fit capacity, overflow suffixes, GPU count zero/clamping, pool carry and invalid grids across
multiple scan iterations. The sequential producer interpreter checks lane-local shared values and unconditional
full-wave ballots; shared barriers and arbitrary ballot predicates require hardware image comparisons.

Output pruning passed 44 hardware workloads with all 440 images byte-exact against independent VS references.
Paired pruning off/on/off measurements found 26 credible wins: 1.23–1.40 times for the wider vertex-output
fixtures and 1.05–1.13 times with live primitive attributes. Small dead-output fixtures and controls stayed
essentially unchanged. These are fixture results, not game FPS or a reason to enable stored IDX by default.

`ICD=... SHIM=... OUT=... python3 light.py --offline` audits a 50-workload route-cost sweep. Its generic
fixtures include a 64-thread control, 128-thread V128/P64 and V128/P128 shapes with dynamic allocation
counts and a cull distance, and a 128-thread V256/P128 shape with live primitive attributes. The dimensions
and features were selected from captured FF7 Rebirth and Hellblade II shader inventories; the arithmetic,
geometry and inputs are synthetic. Game descriptor contents, textures, dispatch frequencies and workloads
are not reproduced. The captured 32-thread V32/P32 class stays on Mesh under the existing policy.

Each profile varies groups per live record over 1, 8 and 64, and draws per command buffer over 1 and 4,
using direct or indirect-count commands with a zero middle record and a 2D final record at 64 groups.
Two additional primitive-attribute cases use the last provoking vertex. Mesh and replay IDX trials bracket
stored IDX. All five modes require three exact flat corner probes, and the smooth timing images must also
match the independent VS reference byte for byte. Indirect replay/stored preflights check reservation
and resubmission tables. Separate empty render-pass controls bracket the sweep and require an exactly
zero readback. All reservations fit the existing 16 MiB bounds.

The runner retains nine timestamp samples after two warmups, reports baseline drift and applies the existing
5 percent/drift threshold for credible gains. Diagnostic linear fits use total GPU time = intercept +
draws * per-draw cost + original API Mesh groups * per-group cost. They include attachment/pass and rasterization
costs, can have negative coefficients, and are not separate setup/synchronization/fetch phase timings or
automatic route policy. One CPU recording measurement per trial is also retained; tracing and scheduling
affect it, so it is diagnostic only. GPU hardware measurements and game FPS for these new profiles remain
unproven until measured. The sweep also reports replay IDX versus Mesh and whether stored IDX wins against
both alternatives. Both baseline pairs must stay within 10 percent drift for a replay-versus-Mesh win.
The runner verifies the driver's default trimmed IDX calls without setting the trim override, including
their bounds and synchronization before releasing submissions. Mesh modes must record no IDX batches.
The frozen runner pins checksums, requires a quiet GPU with recovery disabled,
creates a one-shot marker, and stops at the first failure.

IDX side-batch IB2 calls use their shortened execution length by default. `RADV_BC250_MESH_IDXPASS_BATCH_TRIM=0`
restores padded execution for comparison. Trimming uses the command
length, rounded to the normal GFX IB size alignment. The producer wait, stored-output cache invalidations
and PFP synchronization stay inside the call. Minimal alignment NOPs end at the execution boundary;
the remaining 32 KiB allocation stays initialized. The size word is patched during recording, before
submission, and its main-stream BO remains mapped when the main stream grows. Reset discards an open
batch without patching its call. The switch is enabled by default and applies
to both replay and stored IDX. Disabling batching still records the passes inline.

Run `ICD=... SHIM=... OUT=... python3 batch.py --offline` to audit the shortened calls with no GPU
nodes. The frozen hardware runner compares padded, trimmed and padded execution using one driver,
separately for replay and stored IDX. It retains the 50 shader-property workloads and adds direct
recordings with 64 and 4096 draws at one group per draw; the longest recording checks multiple side
batches and growth of the main command stream. All image comparisons, including smooth timed images,
must match independent VS references byte for byte. Three rotated flat corner probes and indirect
pool/reset preflights precede timings. Every scored trial resubmits the same command buffer with two
warmups and nine timestamp samples. The summary reports saved microseconds, speedup and baseline drift;
a credible gain must exceed both 5% and the drift, with drift no greater than 10%.

`ICD=... SHIM=... OUT=... python3 batch_recording.py` audits the unset default and both explicit settings across clear attachments,
reset/free cycles, query and reservation fallback, inline recording, and reset of an unfinished recording
followed by recording the same command buffer again. These checks use the noop shim and a fresh `/dev`.
They prove command recording and validation, not rendered pixels or performance.

The paired hardware gate passed all 52 workloads and 1,036 byte-exact route images. It found 47 credible
replay IDX gains among 49 stable comparisons and 48 stored IDX gains among 50 stable comparisons.
Median speedups across stable cases were 1.36 and 1.37 times respectively. Five comparisons exceeded
the baseline drift limit, including both 4,096-draw timing comparisons; those long recordings passed
the image and command-stream checks. A small stored draw improved from 38.56 to 21.76 microseconds.
These results support shorter execution as an IDX default, not an automatic choice of IDX over Mesh
or a measured game FPS gain. Stored IDX remains opt-in; the new route-cost sweep measures the alternatives
with trimming active by default before any route policy is widened.

`RADV_BC250_MESH_IDXPASS_SETUP_BATCH=1` enables experimental shared setup for consecutive indirect IDX
calls within a side batch. It remains off by default. Paired hardware measurements of the initial
implementation passed 35 workloads and 937 exact reference images, with 28 credible setup gains.
Repeated-call stored IDX improved by a median 1.18 times across stable comparisons, but scanning a
single-entry list added 1.00–1.34 microseconds. These gains compare setup implementations; lighter
fixtures still favor Mesh, and game FPS remains unproven.

The current implementation keeps the ordinary setup shader and wait for the first indirect call.
When a second call joins that span, recording replaces only the first setup and its wait with the
shared scan, preserving the following producers and any intervening direct producers. All changes
finish before submission. A one-call span executes the ordinary setup; spans with two or more calls
execute one shared scan and wait. Lists still hold at most 32 calls, count toward the existing 8 MiB
side-upload bound, and preserve the 16 MiB indirect pool and Mesh overflow fallback. The final
producer wait and stored-output cache invalidations remain unchanged. Inline recording uses the
ordinary setup, and reset discards the open span and its replacement offsets.

`ICD=... SHIM=... OUT=... python3 setup_batch.py --offline` audits 42 workloads with setup off/on/off
and Mesh controls. In addition to the shader-property fixtures, it exercises 2, 31, 32, 33, 34, 65,
128 and 512 calls, direct producers around indirect calls, both provoking corners, and small-pool
overflow. The frozen hardware runner requires exact independent reference images and pool preflights
before accepting timing results. The selective hardware gate passed all 42 workloads and 1,126 exact
route images. Of 81 stable setup comparisons, 35 showed credible gains and none showed credible
regressions. Repeated-call stored IDX improved by a median 1.184 times; the single-call change stayed
below 0.6 percent in the tested controls. These measurements compare setup implementations, not game FPS.

`ICD=... SHIM=... OUT=... python3 setup_packets.py` decodes the actual recorded side batches in a
fresh-device noop sandbox. It checks shader programs and resource registers, producer order, setup
wait placement, list boundaries, and identical packet layouts and shader code for single-call spans.
`setup_recording.py` adds 24 lifetime/fallback cases, including reset of an unfinished promoted span
and 80 reset/free cycles with two calls per recording. These checks prove recording and compiler
behavior, not rendered pixels or GPU speed. `BC250_IDX_CAPTURE_BATCH=1` prints CPU packet evidence;
it does not enable hardware tracing.

Run `ICD=... SHIM=... OUT=... python3 setup_work.py --offline` for a 33-workload vertex-work sweep.
It uses the same shader-property profiles at V64/P64 and V256/P128 with live primitive outputs,
eight groups per live record, and 0, 64, 256 or 1,024 integer mixing steps per vertex. Each step count
has 1, 16 and 32 indirect-count calls plus a 16-call direct control. An additional primitive-output
case uses the last provoking vertex. The inputs remain synthetic; captured game resources and
dispatch frequencies are not reproduced.

Both replay and stored IDX use setup off/on/off brackets with Mesh brackets. Three rotated flat
corner probes must match independent VS references exactly for every route and setup setting.
Zero-step scored images remain byte-exact. Nonzero-step interpolated scored images permit at most
one RGB UNORM8 unit only after all exact corner probes pass, with alpha and background byte-exact.
Indirect GPU preflights check allocation tables and resubmission reset. Timings retain nine samples
after two warmups. Reports distinguish gains over per-call setup from wins over Mesh, exclude noisy
baselines, and list sampled vertex-work counts beating Mesh without interpolating a crossover.
This suite does not change route policy or establish game FPS. Run
`python3 -B -m unittest test_setup_work` from this directory for CPU-only report checks.

The vertex-work hardware gate passed 33 workloads and 859 byte-exact route images; no scored image
needed its rounding allowance. It found 23 credible setup gains among 62 stable comparisons and no
credible setup regressions. A V256/P128 primitive-output workload with 256 integer steps and 16
indirect calls took 498.44 microseconds using stored IDX and shared setup, versus 759.58 for Mesh
and 2,119.74 for per-call stored setup. V64/P64 workloads still favored Mesh. Four setup comparisons
and 16 Mesh comparisons exceeded the drift limit. These are synthetic measurements, not game FPS.

`ICD=... SHIM=... OUT=... python3 setup_focus.py --offline` audits a focused repeat kit using the same
validated driver. Eight workloads cover the larger primitive-output winner and nearby 64-step,
32-call and last-provoking cases, a direct control, a light indirect control and the V64/P64
negative control. All eight run three times with different case orders; round two also reverses
replay/stored timing order. Exact corner probes and pool/reset preflights precede each repeat.

Hardware scored trials collect 31 timestamps after the harness's two warmup submissions. The report
checks both all 31 samples and the last 23, discarding the first eight measured submissions only for
the secondary settling check. Both checks must agree. Focus timings require setup and Mesh baseline
drift at most 5 percent and central p10-to-p90 sample spread at most 10 percent. A confirmed win must
pass in all three repeats, exceed 5 percent and baseline drift, and pass the original image policy.
Excluded timing results remain in the logs. No clocks or automatic route policies change. Run
`python3 -B -m unittest test_setup_work test_setup_focus` here for CPU report checks.

The focused hardware gate passed all 636 route images byte for byte. The larger primitive-output
direct stored fixture beat Mesh in all three stable rounds by 1.61 to 1.65 times. Indirect wins
were not confirmed: only 11 of 48 per-round comparisons passed the stricter drift/spread criteria.
The source of the timing instability remains unproven; route defaults stay unchanged.

`ICD=... SHIM=... OUT=... python3 setup_steady.py --offline` audits a smaller repeat with sustained
warmup. Four workloads retain the 256-step primitive-output indirect candidates at 16 and 32 calls,
the 16-call last-provoking case and the confirmed direct control. They run three reordered rounds
with the same 31/full and 23/settled agreement checks and strict focus thresholds.

`setup_final.py` combines the four sustained-warmup candidates with a light V64
control and V128/P64 and V128/P128 shader-property fixtures from the captured FF7
inventory. Eight workloads run three reordered rounds with sustained warmup,
telemetry and the same strict criteria. Arithmetic and resources remain synthetic.
It compares existing routes without changing defaults or automatic route selection.

`INDEX_WARMUP_MS=500` resubmits the same audited command buffer for at least 500 milliseconds before
the normal two warmups and measured samples. The launcher first verifies compilation, allocation
and synchronization, then releases hardware execution. Extra submissions and wall time are logged
separately. Noop mode logs that sustained warmup was skipped and retains its three ordinary
submissions. The option accepts 0 to 2,000 milliseconds and defaults to zero, so existing kits keep
their submission behavior. Cumulative accepted-submission usage counters are checked against the
number of submissions, including diagnostics emitted after many warmup submissions.

Scored hardware trials retain exclusive telemetry JSONL files with read-only 50-millisecond sysfs
clock, load and temperature snapshots. Warmup and measurement windows share the monotonic clock
with those snapshots. Short measured windows may contain no telemetry sample; observed ranges
cannot prove clocks were constant between reads or establish the cause of drift. No clock controls
or recovery settings are written. Whether sustained work stabilizes these timings is unproven
until the owner runs the kit. CPU checks are
`python3 -B -m unittest test_setup_work test_setup_focus test_setup_steady` from this directory.

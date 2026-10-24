# Automatic Mesh and stored-output selection

`RADV_DIRECTMESH=1` selects the route inside the driver. No stored-output profile
or launcher is required. The preset supplies `RADV_BC250_MESH_IDXPASS_STORE=auto`
and enables the validated shared indirect setup. Explicit numeric values remain
diagnostic overrides: `STORE=0` uses the legacy replay route, `STORE=1` forces
stored outputs for eligible shaders, and `IDXPASS=0` keeps Mesh.

Correctness admission is unchanged. Automatic candidates use the existing Mesh
contract, output pruning, stored producer/fetch pair, bounded memory pools and
overflow suffix fallback. Task, DGC, simultaneous-use recordings and counting
queries retain their existing Mesh behavior. Single-record indirect draws stay
Mesh. Modeled stored candidates with fewer than eight direct groups stay Mesh;
the established replay policy is preserved for unmodeled shaders.

The initial cost model is deliberately limited to the measured large-output
class. It requires declared and runtime counts of at least 192 vertices and 96
primitives, within the existing 256-element limits. It follows SSA dependencies
of live per-vertex outputs, counts arithmetic in canonical unsigned integer
loops and compiles loop/count expressions into at most 48 scalar nodes and eight
terms. Only constants, supported integer ALU and bounded push-constant loads are
accepted. No game names, shader hashes or fixed application constant offsets
are used.

At draw recording the driver evaluates that small expression program against
the current application constants. It selects stored outputs for the recognized
long-loop class (at least 256 iterations and 393,216 estimated vertex ALU units
for direct draws). Multi-record indirect draws use the owner-measured medium
class as well: at least 64 iterations and 98,304 estimated vertex ALU units.
Modeled small counts, short loops and invalid runtime estimates stay Mesh.
Unmodeled shaders retain the established replay route when it is eligible,
including its Mesh fallbacks. This preserves earlier indexed wins outside the
new cost model. Changing constants can change the route within one command buffer. There
is no per-draw shader compilation, descriptor readback, GPU timing probe, SQTT,
clock control or implicit application workload replay. Indirect GPU record data
is not read by the CPU; the existing GPU setup and overflow handling are used.

The owner comparison passed 314 route images byte for byte and confirmed three
automatic workload wins in all three stable repeats, about 1.52–2.14 times Mesh.
One medium indirect workload exposed an overly conservative threshold: stored
was 5.8–6.0 percent faster than the selected Mesh route in three stable repeats.
The indirect threshold was corrected using that measured control. Direct
selection is unchanged. The correction is checked by comparing shader uploads
and command words with the measured paths in a fresh-device noop sandbox.
Those checks establish recording equivalence, not a separately timed run of the
corrected binary. The original run and its three missed decisions are retained.

The heuristic does not prove the fastest route for every shader, indirect record
distribution or game. Unsupported expensive shaders retain the prior route.
GPU benchmark gains are synthetic measurements, not game FPS claims.

`automatic.py` prepares one combined comparison against forced Mesh and stored
routes. It tests 12 workload profiles in three reordered rounds, 31 samples and
500 ms sustained warmup. Additional correctness cases cover tiny direct grids,
single-record indirect draws, overflow, legacy replay and alternating light/heavy constants.
Independent VS images and three exact provoking-corner probes precede scoring.
Heavy interpolated pixels retain the existing one-unit RGB allowance only after
the exact probes, with exact alpha and background. Stability requires at most
5% Mesh bracket drift and 10% central spread; comparisons retain excluded data
and flag a stable loss exceeding 5% against the faster forced route. No game
capture is required. The frozen hardware kit is owner-only and one-shot.

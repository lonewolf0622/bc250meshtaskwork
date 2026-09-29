# Safe direct Mesh regression

RADV_BC250_MESH_SAFE_DIRECT is default off. GFX1013 plain triangle Mesh shaders use
latest-copy remapping with whole-triangle closure W=31. The static physical bound is
min(V,3P) for V<=32, otherwise 3P; only bounds <=256 and conservatively bounded LDS
are eligible. Existing split/Task/per-primitive/unsupported cases retain their route.

Run with ICD=<candidate frozen ICD> BUILD=<Mesa build> OLDICD=<49f8a47 baseline ICD>
KEEP=<evidence directory> SEEDS=64 bash run.sh. The complete runner executes inside
bwrap with fresh /dev and the amdgpu noop shim; it has no hardware mode. If OLDICD is
omitted, identity is checked only against the same candidate with its switch off.
ONLY is an optional case-name regex. SRC and the oracle source resolve from this tree.

The 31 cases cover direct admission, the 255/258-slot boundary, preserved fallback,
runtime counts/empty output, early return, shared barriers including T96/T128 extra waves,
queries with split disabled, T2, atomics, partial and arrayed
outputs. SPIR-V, Vulkan core and NIR/ACO validation must pass. Uploaded shader bytes,
configuration and captured IB bytes are compared with the switch unset and zero;
excluded cases also require on/off identity. The CPU NIR oracle checks actual lowered
exports for every-vertex-referenced and the triangle-wide backjump, physical bounds, primitive/corner attribute equality,
side effects and query accounting. It counts undefined reference cases separately.
Unoptimized builds avoid unrelated varying-packing differences in equality tests;
optimized candidates receive independent safety checks.

The larger project's independent closure model, adversarial NIR inputs, reduced
hardware-ladder preparation and new one-shot SD0 assets are under
the project's direct-path records (kept outside this repository). None of these offline checks measures GPU
performance or authorizes hardware execution.

T2 equality uses the geometry-equivalent T32 fixture because the baseline T2 expansion
has undefined extra-lane counts in the CPU model; it does not compare invocation statistics.

## Parallel and direct-autocull variants

VARIANT=parallel enables RADV_BC250_MESH_SAFE_PARALLEL in on/noon compilations;
VARIANT=autocull also enables RADV_BC250_MESH_SAFE_AUTOCULL and the CPU culling oracle.
Both switches default off. The parallel planner assigns independent logical-vertex
source-key chains, then ranks anchor bits; whole-triangle source-key distance <=31
implies physical distance <=31. It can duplicate more vertices than serial closure.
P>85 retains the serial specialization if otherwise admitted. Launch budgets remain
proven worst-case bounds; no statistical-budget launch or runtime overflow redraw exists.
Direct autocull additionally requires the parallel planner and static FILL mode; it
builds a survivor mask before any vertex copies and retains original primitive order.
The driver corpus/dependency tests additionally compare each new switch off against
its preceding frozen implementation. Ordinary suite OLDICD remains the original49f8a47
baseline with all new switches off. See MESH_PERF/direct/DIRECT_PERF.md for limits.

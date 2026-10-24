# DirectMesh on v1.3.1: automatic IDX and RDNA 1

This tree merges the DirectMesh development work with public v1.3.1
(`3977ab4`). Its BC-250 GTT storage fix and inline pending-restore checks are
preserved. The merge uses the byte-identical v1.3 trees at `ed58ea5` and `872468f`
as its common source base (tree `61e7d6c29b3e50ebd07c1e18b8fd9e9b6f1507af`).
The previous `74d1bc1` candidate did not include the v1.3.1 fixes and is superseded.

This source build activates DirectMesh automatically on BC-250 (GFX1013) and
RDNA 1 (Navi 10, Navi 12 and Navi 14). `RADV_DIRECTMESH=1` is no longer required.
`RADV_DIRECTMESH=0` disables DirectMesh. Existing individual environment options,
including explicit zeroes, still override defaults. Selecting a staged ICD is
still necessary when testing a driver that has not been installed.

DRM PCI metadata selects the preset before cached driver options are read. The
allowlist uses exact AMD display-device IDs from the Linux amdgpu PCI table;
family checks also gate compiler and command-buffer paths. An unrelated GPU
alone does not enable the preset. Defaults still use process environment settings,
so on a mixed-GPU system their environment values are process-wide; the Mesh
implementation remains gated per physical device.

The shared protected Mesh, split, merge, Task and indexed paths now accept RDNA 1.
Navi 14 retains Mesa's legacy path for ordinary vertex/geometry graphics. Its Mesh
stage alone uses NGG, preserving the existing zero late-allocation workaround.
BC-250 memory, submission, CU/slice topology, VRS emulation and ray instruction
exceptions remain limited to BC-250. RDNA 1 does not gain ray tracing capabilities.
The BC-250 per-primitive interpolation register exception is also retained only
for BC-250; the RDNA 1 fixtures use the protected per-vertex representations.

DXVK keeps graphics pipeline libraries, shader objects and pipeline binaries when
hybrid Task is implicit. An explicit `RADV_BC250_HYBRID_TASK=1` still enables Task.
The implicit Task default is not written into the environment, so it remains
implicit even if the Vulkan loader unloads/reloads the driver between instances.

## Validation and limits

The merged gate passed 110 driver unit tests and 291 fresh-device noop launches:
47 activation/override/feature probes (each creates two successive Vulkan
instances), 76 pairs of shader/command recordings, and 92 extended recordings.
On BC-250, all 19 automatic recordings match the `74d1bc1` candidate with
its explicit preset; changing storage placement preserves shaders and command words, including shader uploads and command words. Each RDNA 1
target matches its explicit preset on the same 19 workloads. Extended checks
cover primitive attributes and both provoking modes, shared memory, ballots,
branches, dynamic/empty output, query and multiview fallback, Task direct/indirect,
barycentrics, per-vertex reads, PrimitiveID and a conventional vertex reference.
Recorded Navi 14 NGG stages have zero late allocation. Both BC-250 PCI IDs from
v1.3.1 (`13fe` and `143f`) activate the preset. Allocation traces assert GTT with
write combining for BC-250 index/output pages and indirect pools, local VRAM
for discrete RDNA 1, and CPU-visible GTT for diagnostic pools on all targets.

These are CPU compiler, Vulkan validation and command-recording checks. They do
not render or measure GPU time. Navi 10 uses Mesa's captured shim profile;
Navi 12, Navi 14 and GFX1013 profiles use synthetic identities with Navi 10 topology.
RDNA 1 rendering, stability, conformance and performance remain unverified on
hardware. In particular, the automatic indexed cost thresholds were measured on
BC-250 and are not tuned separately for RDNA 1. Earlier BC-250 hardware results do
not establish RDNA 1 correctness or speed. Treat this as an RDNA 1 development
candidate, not a hardware-validated release for those cards.

The installed driver and existing frozen owner benchmark kits are unchanged.
The query recording test omits `dumpibs`: the existing optional GFX10 packet
pretty-printer asserts on a query counter enumeration. Vulkan validation and
route-usage checks still cover that recording.

Reproduction is in `tests/bc250-mesh/directmesh-defaults/README.md`. CPU evidence:
`/home/deck/mesh-test/directmesh-v131-rdna-evidence/`.

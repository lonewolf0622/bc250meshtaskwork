# Default activation and RDNA 1 CPU gate

All tools here require the noop DRM shim inside bubblewrap with a fresh `/dev`.
No GPU rendering, timing or conformance result is inferred from this gate.
`probe.c` creates and destroys two successive Vulkan instances to cover DSO reload.
`check.py` covers default activation, overrides, unrelated GPUs, DXVK, debug
inhibitors and automatic shader/packet identity against the explicit preset.
`extended.py` covers native Mesh, stored IDX, queries, multiview, Task,
barycentrics, PrimitiveID and ordinary vertex pipelines. Navi 14 keeps its
ordinary graphics NGG exclusion and NGG late allocation must remain zero.

Build `src/amd/vulkan/libvulkan_radeon.so` and `src/amd/vulkan/radv_tests` in an
existing configured Mesa build. Choose a fresh evidence directory. Example paths:

```sh
src=/home/deck/mesh-test/directmesh-mesh-contract
build=/home/deck/mesh-test/directmesh-mesh-contract-build
out=/home/deck/mesh-test/directmesh-v131-rdna-evidence
fixtures=/home/deck/mesh-test/directmesh-auto-final-evidence/fixtures
cd "$src"
python3 tests/bc250-mesh/directmesh-defaults/build_noop.py --build "$build" --out "$out/noop"
cc -O2 tests/bc250-mesh/directmesh-defaults/probe.c -o "$out/probe" -lvulkan
cc -O2 tests/bc250-mesh/index-pass/index_gate.c -o "$out/index_gate" -lvulkan
"$build/src/amd/vulkan/radv_tests" > "$out/unit.log"
python3 tests/bc250-mesh/directmesh-defaults/check.py --build "$build" --fixtures "$fixtures" --out "$out" --baseline-icd /home/deck/mesh-test/releases/directmesh-default-rdna1-74d1bc1/radeon_icd.json --baseline-label 74d1bc1
python3 tests/bc250-mesh/directmesh-defaults/extended.py --build "$build" --out "$out"
```

The captured Navi 10 table is unchanged. `build_noop.py` generates private,
explicitly synthetic GFX1013/Navi12/Navi14 entries with Navi 10 topology. This
exercises family-specific compilation, not real memory/CU configurations.
The shim now reports the selected PCI device ID to DRM enumeration, which is
necessary to exercise automatic discovery instead of forcing the preset.

Each run creates its log directory exclusively; use `--checks-name` with a new
name for subsequent CPU runs, preserving previous evidence. Frozen owner GPU
kits must never be retried or edited. These tools do not operate on them.

The merged gate also covers the v1.3.1 `143f` alias and asserts index-page/pool
allocation domains and CPU/write-combining flags, including diagnostic pools.

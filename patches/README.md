# DirectMesh development patches on v1.3.1

These patches include the v1.3.1 GTT storage fix and inline restore checks,
automatic stored-output IDX routing, shared Mesh analysis and the RDNA 1 port.
They supersede the patch in the stable v1.3.1 release for this development tree.

- `bc250-directmesh-mesa-26.2.1.patch`: driver changes against stock Mesa 26.2.1; apply with `git apply` or `patch -p1`.
- `bc250-directmesh-tests.patch`: additional test fixtures under `tests/bc250-mesh/`.
- `SHA256SUMS`: driver patch, test patch and repository README checksums.

Build and usage: see [README](../README.md). Current validation and limitations:
[DirectMesh on v1.3.1](../docs/directmesh-defaults.md). RDNA 1 has CPU validation
only; BC-250 hardware timings from earlier builds do not validate RDNA 1 speed.

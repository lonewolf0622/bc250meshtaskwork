# Bounded private bary exports

Default-off SAFE_BARY_TINY and SAFE_BARY_AFFINE use the existing private exporter.
This suite checks actual split-lowered NIR with every corner, references and static/dynamic
provoking modes. Affine negative controls reject changed connectivity and unknown counts.
Run run.sh inside the noop sandbox, with BUILD, ICD and KEEP. No GPU mode exists.

The count proof reduces export capacity, not the original output arrays: API
writes above the emitted count still have their original staging allocation.
The affine proof compares integer polynomials modulo 2^32 and treats unknown SSA
values as independent symbols. Both optimized and unoptimized variants run.
For this proof-only route, corpus oracles use the application index generator:
mutating indices after lowering would invalidate the compiler's premise.
The general tiny expansion also runs arbitrary-index adversarial generators.

`mesh_oracle --sgpr` models uniform wave registers for sub-wave point shaders,
using the same ignore-undef divergence rule as ACO. It rejects conflicting
defined values; ordinary oracle runs retain their existing execution model.

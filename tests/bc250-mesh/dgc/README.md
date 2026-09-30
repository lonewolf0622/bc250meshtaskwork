# Offline BC250 DGC proof

Run `run.sh` with `ICD`, `SHIM`, `BUILD`, and optionally `KEEP` set. The runner
creates a fresh `/dev` in bwrap and refuses a visible render node. It never
submits work to a GPU.

The six shader shapes cover ordinary Mesh, safe direct pieces, per-primitive
shared corners, hybrid Task, safe Task pieces, and unbounded folded Task.
Each runs private, PP_SHARE, and adaptive policies with direct and count tokens.
Adaptive probes explicitly preprocess in another command buffer, complete its
noop submission, destroy its command pool, then execute the recorded stream.
Push-constant and sequence-index tokens are checked on all non-shared fixtures.

The CPU interpreter executes the driver's serialized prepare NIR against bounded
memory mappings. It compares every active generated PM4 byte with the ordinary
indirect backend's captured sequence at the same final addresses, checks packet
boundaries and Task producer/visibility-barrier/consumer ordering, rejects native
Task packets, verifies DrawID/grid/count records and application constants, and
checks inactive NOP programs and output sentinels. Sixteen count combinations
per direct/count pair give 576 cases across 18 routes. This proves command
construction and token resolution; the noop shim cannot prove rendered pixels.

Extension and feature queries check the DGC switch unset, zero, and one, plus
native Task refusal. Hardware gates are maintained separately in the evidence
packet and must not be launched as part of this suite.

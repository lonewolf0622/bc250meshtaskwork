# Safe private-corner barycentrics

`RADV_BC250_MESH_SAFE_BARY=1` is default off and requires SAFE_OWNED and SAFE_CORNERS (plus SAFE_PIECES for large declarations). Shared-corner plans retain their refusal. Run `ICD=... BUILD=... KEEP=... ./run.sh` through the offline wrapper; it has no GPU mode.

The suite executes the actual split and direct lowered NIR with the CPU Mesh oracle: two reference PARAMs equal position (`--bary-ref` on split), then all corner exports and primitive order equal split (`--compact --safe-owned --all-corners --bary-rot --w 2`). It covers the retained 6/6 cost fallback, 32/32 with PointSize, 192/64, 256/128 ordered pieces, payload, autocull, direct read on/off, and static/dynamic first/last provoking modes. Reference undefined seeds are reported separately. These are export and rotation proofs, not rendered-image tests.

Negative cases retain split for shared corners, diagnostic missing references, exhausted parameter slots, and tiny classes with higher static cost than split. A strict PerVertexKHR-only input is checked independently of weight intrinsics. The GPU gates and saved-game/CTS audits are in MESH_PERF/direct/BARY_DIRECT.md.

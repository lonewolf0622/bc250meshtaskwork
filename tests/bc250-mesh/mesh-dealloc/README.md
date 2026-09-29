# RADV_BC250_MESH_DEALLOC_DIST regression (drm-shim only)

`RADV_BC250_MESH_DEALLOC_DIST=N` (GFX1013 only, default unset) programs
`VGT_OUT_DEALLOC_CNTL.DEALLOC_DIST = N` while a Mesh shader is bound. It is a hardware experiment:
on the raw shared-vertex route (`RADV_BC250_MESH_AMD_ROUTE=1`, all vertices referenced) a primitive
may reference a vertex at most 31 below the highest index referenced so far; 33 hangs. RADV never
writes this register on GFX10, so every draw runs with the CLEAR_STATE value 32 (the running
kernel's gfx10 clear-state table and Mesa's gfx10 table both hold 0x20).

## What the driver does

| | switch unset | `N` = 1..127 |
|---|---|---|
| Mesh shader emitted (`radv_emit_graphics_shaders`, any Mesh path: expansion, split pieces, raw/AMD, compaction, hybrid Task) | nothing | `VGT_OUT_DEALLOC_CNTL = N` (the register has only this 7-bit field) |
| other graphics shaders emitted after a Mesh shader in the same command buffer (including meta draws) | nothing | `32` |
| end of a graphics command buffer (primary or secondary) that wrote `N` | nothing | `32` |
| command buffer without Mesh draws | nothing | nothing |

The writes use a tracked context register slot (`AC_TRACKED_VGT_OUT_DEALLOC_CNTL`), so repeated
Mesh draws do not write again. Every command stream starts with the slot unknown; after
`vkCmdExecuteCommands` the primary takes the secondary's tracked value, as for every tracked register.
Values that do not fit the field (0, > 127) and anything that is not a plain decimal number are
ignored with a `radv/bc250: RADV_BC250_MESH_DEALLOC_DIST=... ignored` message; a valid value prints
`radv/bc250: Mesh draws use VGT_OUT_DEALLOC_CNTL.DEALLOC_DIST=N` once per device.

PAL for reference: it writes `VGT_OUT_DEALLOC_CNTL` once per queue context on every chip with a
hardware VS (GFX9-10.3, NGG included): `DEALLOC_DIST = 16` with its default VS half-pack threshold,
32 when half-pack is off. It has no Mesh- or NGG-specific value; LLPC does not program it.

## The suite

`ICD=<icd json> [OLDICD=<icd of the base build>] [KEEP=<dir>] ./run.sh`

`deallocmode.c` (derived from `../amd-mode/amdmode.c`, same shapes) records one of five modes and
submits it to the shim; the IB is dumped with `RADV_DEBUG=dumpibs,allbos` (executed secondaries
appear nested). `check.py` follows the register value through the IB (32 at the start of every IB):

- `mix`: Mesh direct, VS draw, Mesh indirect, Mesh indirect count, Mesh direct:
  `D127 M D32 V D127 M M M D32`.
- `meta`: Mesh, `vkCmdClearAttachments` on a partial rectangle (a meta draw), Mesh, VS:
  `D127 M D32 V D127 M D32 V`.
- `secondary`: a Mesh and a VS secondary executed in a render pass, then Mesh and VS inline:
  `[D127 M D32] [V] D127 M D32 V`.
- `reset`: a Mesh recording, `vkResetCommandBuffer`, a VS recording: `V`.
- `vsonly`: `V V`.

Cases: the base route (`base_f64`), the raw route (`raw_f64`), the full AMD mode (`amd_f64`,
VGT_REUSE_OFF switched in the same place), split pieces (`split_a124`), the compaction
(`compact_f64`) and a wave32 per-primitive shape (`pp32_w32`). Each runs unset and with 127:

- unset: no `VGT_OUT_DEALLOC_CNTL` write; with `OLDICD` the IB is byte-identical to the base build's.
- 127: every Mesh draw runs with 127, every other draw with 32, every IB (primary and nested
  secondary) ends with 32, no redundant write; the rest of the IB equals the unset run after removing
  the writes, the NOP padding (its payload is stale memory) and the executed secondaries' IB sizes.

Then the values on the raw route (`mix`): 64, 32 and 1 are used; 0, 128, 200, `abc`, `12x`, `-1`
and an empty value are ignored and the IB is byte-identical to the unset run.

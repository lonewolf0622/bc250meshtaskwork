# fast-binding: RADV_BC250_EXPOSE_FAST_BINDING offline regression (drm-shim only)

With hybrid Task on (`RADV_BC250_HYBRID_TASK=1`, as the game launchers set) the driver hides
`VK_EXT_descriptor_buffer`, and vkd3d-proton falls back to descriptor sets. `RADV_BC250_EXPOSE_FAST_BINDING=1`
(runtime switch, default off, no compiler-key bit) exposes descriptor buffers again and lets the direct
Mesh-only split, the internal Mesh-only amplifier and the hybrid Task producer accept pipelines created with
`VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT` (see `bc250_refused_create_flags` in
`src/amd/vulkan/radv_bc250.c`). Descriptor heap, DGC, GPL, shader objects and pipeline binaries stay hidden.

    ICD=<radeon_devenv_icd json> [OLDICD=<icd of the build before the switch>] [KEEP=<dir>] ./run.sh

- `fb.c`: records a vkd3d-proton-like binding model: set 0 = resource heap (mutable storage buffer /
  sampled image, variable count), set 1 = sampler heap, set 2 = static sampler (immutable; embedded with
  descriptor buffers), set 3 = root CBV (push descriptor), 16 bytes of root constants. Modes: `legacy`
  (descriptor sets, what vkd3d-proton records while descriptor buffers are hidden), `db` (descriptor-buffer
  set layouts and pipelines, `vkCmdBindDescriptorBuffersEXT` + `vkCmdSetDescriptorBufferOffsetsEXT` +
  `vkCmdBindDescriptorBufferEmbeddedSamplersEXT` + `vkCmdPushDescriptorSetKHR`), `legacydb` (descriptor
  sets, but the descriptor buffers are allocated as in `db`, so both runs have the same addresses). The
  compute bind point has its own heap copy / sets. Script tokens: small Mesh direct/indirect, a 256V/128P
  per-primitive Mesh (direct split) direct/indirect/indirect-count, hybrid Task + small or split Mesh
  direct/indirect/indirect-count, VS draw, application compute dispatch, rebind. Refuses to run without the
  noop drm-shim.
- `run.sh`: exposure (switch off: all six features hidden and `db` finds no descriptor buffer; on: only
  descriptor buffer + its four features, `maxDrawIndirectCount` 4096; on with native Task: hidden; without
  hybrid Task the switch changes nothing), the pre-existing refusal of a large Mesh-only descriptor buffer
  pipeline without hybrid Task when the switch is off, and 11 cases (all paths, compute restore, producer
  after compute, split batch prep on/off, internal amplifier, rebinds, autocull, the Hellblade 2 switches
  (batch prep, 64-triangle pieces, compact LDS), timer, hybrid Task off). Every case runs `legacydb` and
  `db` with the switch on under `RADV_DEBUG=dumpibs,shaders`, and `legacy` with the switch off and on (and
  with `OLDICD`) for byte-identical dumps.
- `check.py`: the `legacydb` and `db` dumps (IBs and NIR/ACO/disassembly) must be identical except values
  of `SPI_SHADER_USER_DATA_*` / `COMPUTE_USER_DATA_*` writes; those map one-to-one onto the descriptor
  buffer set addresses; application compute dispatches see the compute heap after hybrid Task draws;
  driver dispatches (Task producer, setup helpers) only get graphics set pointers; every hybrid Task draw
  gives the producer the graphics heap.

# split-batch: RADV_BC250_SPLIT_BATCH_PREP offline regression (drm-shim only)

`RADV_BC250_SPLIT_BATCH_PREP=1` (runtime switch, default off, no compiler-key bit) runs the base
driver's split-argument setup once per render pass instance instead of once per split indirect draw
(design and safety argument: the comment above `bc250_split_batch_pipeline` in
`src/amd/vulkan/radv_bc250.c`).

    ICD=<radeon_devenv_icd json> [OLDICD=<icd of the build before the switch>] [KEEP=<dir>] ./run.sh

- `sb.c`: records a script of render passes and draws (see its header): split indirect / indirect-count
  draws of an HB2-Nanite-like 256V/128P per-primitive shape (split=3) and an a_ls32-like 64V/124P shape
  (split=2), VS draws, non-split Mesh draws, direct split draws, several render passes, suspend/resume,
  VkRenderPass subpasses with an in-subpass barrier, secondary command buffers, conditional rendering;
  submits twice; refuses to run without the noop drm-shim.
- `run.sh`: 15 IB cases, each with the switch on (`RADV_BC250_SPLIT_BATCH_TRACE=1`) and off under
  `RADV_DEBUG=dumpibs`, plus the NIR model case. With `OLDICD`, the off runs (also with
  `RADV_BC250_SPLIT_BATCH_PREP=0` and the trace variable set) must be byte-identical to that build.
- `check.py`: models from the script where batches open (primary command buffer, inside a render pass
  instance, outside conditional rendering; closed by rendering begin/end, subpass changes, barriers,
  conditional rendering begin/end and vkCmdExecuteCommands; a new batch when 128 entries or 1024 setup
  workgroups are used) and checks per draw: one DISPATCH_INDIRECT (SET_BASE = the traced list) and a
  CS_PARTIAL_FLUSH before the draw that opens a batch, no dispatch and no CS_PARTIAL_FLUSH before a
  joining draw, one DISPATCH_DIRECT + CS_PARTIAL_FLUSH for per-draw setups (switch off, secondary,
  conditional rendering); the traced list entry is the draw's (slot, first setup workgroup, input,
  count buffer, records, stride, split pieces) and its DISPATCH_MESH_INDIRECT_MULTI reads that entry's
  output; draw by draw the same packets and graphics register state as the off run.
- `nirsim.py`: interprets the builder NIR of the per-draw and the batched setup shader (printed with
  `RADV_BC250_SPLIT_BATCH_PRINT_NIR=1`) on random draws (count buffer below / at / above the record
  count, zero and 32-bit-wrapping dimensions, 1..1000 records) and requires byte-identical driver
  records.

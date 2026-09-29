# Nanite staging regression (drm-shim only)

Hellblade 2's Nanite Mesh shader (128 invocations, 256 vertices, 128 triangles) declares a flat uvec3 output (`tc1`). Its fragment shaders declare that output but never read it. The split and expansion stage every declared output in at most 16 KiB of LDS, and this dead output raises the total from 15.9 KiB to 18.9 KiB. Most of HB2's Nanite material fragment shaders do read tc1, which needs 18.9 KiB of staging. Both cases failed pipeline creation with `BC250 mesh split rejected: required vertex expansion unavailable` (VK_ERROR_FEATURE_NOT_PRESENT), and the game aborted.

Expected results with the the base driver launcher policy, under the noop drm-shim, with /dev/dri hidden:

| Mesh | Fragment | Before the fix | After the fix |
|---|---|---|---|
| nanite.mesh | dead_tc1.frag (declares tc1, never reads it) | -8 | 0 (split into 3 pieces; exports pos, param0-2) |
| nanite.mesh | live_tc1.frag (reads tc1; HB2's material pipelines) | -8 | 0 (18.9 KiB staging; 26.5 KiB LDS total, no scratch) |
| wide3.mesh | wide3.frag | -8 | 0 (27.5 KiB LDS, no scratch) |
| wide4.mesh | wide4.frag | -8 | -8 (would spill to the Mesh scratch ring; fail-closed) |
| wide6.mesh | wide6.frag | -8 | -8 (staging above 28 KiB)

Build: `glslangValidator --target-env vulkan1.3 -o X.spv X.<stage>`, then `cc pipe.c -lvulkan -o pipe`, then `./pipe nanite.spv dead_tc1.spv`.
Admitted Mesh stage: 21.5 KiB LDS, no spill, no scratch; NIR/ACO validation is clean.

The fix has two parts. It drops Mesh outputs the fragment shader never reads (`radv_pipeline_graphics.c`). It also raises the staging cap from 16 KiB to the 28 KiB of API shared memory that the NGG Mesh lowering guarantees (`radv_bc250.c`), and fails closed when the lowering would need the Mesh scratch ring, which is not validated on GFX1013.

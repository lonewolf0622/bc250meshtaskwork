# M1 simultaneous-use regression (Tree C port, drm-shim only)

A SIMULTANEOUS_USE command buffer can execute again while the Mesh waves of its previous execution still read the Task records and payloads. The records and payloads sit at the same addresses in every execution. So the first producer of a Task draw must wait for earlier geometry work.

`simul 0|1` records one direct Task draw (1 = SIMULTANEOUS_USE) and submits it. Run it under the noop drm-shim with `RADV_DEBUG=dumpibs`. The flush events before the first DISPATCH_DIRECT must be:

| Build | simul 0 | simul 1 |
|---|---|---|
| before the port | none | none (race) |
| after the port | none (unchanged) | PS_PARTIAL_FLUSH (also covers VS) |

Build: `cc simul.c -lvulkan -o simul`. The shaders compile to t.spv, m.spv and f.spv with glslangValidator.

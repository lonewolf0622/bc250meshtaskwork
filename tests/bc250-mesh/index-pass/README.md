# Indexed Mesh offline checks

Build shader assets and the Vulkan harness with `./build.sh`. Run `ICD=... SHIM=... OUT=... ./offline.py`, pointing
at the candidate ICD and amdgpu noop DRM shim. The runner uses bubblewrap with a fresh `/dev`, so real GPU nodes
are unavailable. NIR, ACO and Vulkan validation are enabled.

The suite covers the seven existing connectivity shapes, flat primitive attributes at 64 and 128 vertices,
first and last provoking vertices, index-only ballots and rejection of ballot-dependent vertex work. Independent
VS shaders compute the same geometry and attributes without the Mesh converter. A memory-pressure check retains
160 independent command buffers and verifies automatic fallback after the 128 MiB reservation is exhausted.
Noop runs compile and record commands; they neither render images nor measure hardware performance.

The build products (`*.spv` and `index_gate`) are generated and excluded from source control. Frozen hardware kits
carry these products, the candidate driver, a SHA256 manifest, and an independent VS image comparison.

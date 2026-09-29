# bc250meshtaskwork: DirectMesh for the AMD BC-250

Mesh shader support for the AMD BC-250 (GFX1013, RDNA1-based) in Mesa's RADV Vulkan driver, with Mesh shaders drawn on
a **safe direct path**: no split/replay, one launch per Mesh workgroup, and built-in protection against the index
patterns that hang this chip.

**Download:** [patch against stock Mesa 26.2.1](https://github.com/lonewolf0622/bc250meshtaskwork/releases/download/directmesh-v1.0/bc250-directmesh-mesa-26.2.1.patch)
(also in [`patches/`](patches/) and on the [release page](https://github.com/lonewolf0622/bc250meshtaskwork/releases/tag/directmesh-v1.0)),
or build this repository directly (see Build).

Turn it on with one switch:

```
RADV_DIRECTMESH=1 %command%
```

## What you get

- `VK_EXT_mesh_shader` (Mesh + Task) on the BC-250, usable by D3D12 games through normal Proton / vkd3d-proton.
- `VK_KHR_fragment_shader_barycentric`, so vkd3d-proton keeps Mesh shaders enabled in UE5 games.
- **Direct Mesh path:** each Mesh workgroup is drawn in a single launch, per meshlet:
  - shared vertices when the meshlet's triangles, after culling, use every vertex with small index backjumps;
  - otherwise private triangle corners, which rule out the chip's hang patterns by construction.
- Automatic fallbacks: shapes the direct path cannot prove safe take the older split/expansion path. They never take an
  unprotected raw route.

## Tested

Vulkan CTS on real hardware (with `RADV_DIRECTMESH=1`):

| Test set | Result |
|---|---|
| `dEQP-VK.mesh_shader.ext.*` + fragment-barycentric Mesh cases that run on this device | 3,558 / 3,558 pass, 0 hangs, every pipeline direct |
| Mesh/Task stages in other groups (binding model, subgroups, SPIR-V, atomics, dynamic state, ...) | 10,036 pass, 0 fail, 0 hangs |

Games (Steam / Proton, D3D12):

| Game | Result |
|---|---|
| Final Fantasy VII Rebirth | runs, Mesh direct (its shaders are unchanged by the adaptive export) |
| Control | runs, Mesh direct; about 8% less Mesh GPU time per draw than private corners alone |
| Hellblade 2 | runs, all Mesh pipelines direct, including barycentric shaders (unchanged by the adaptive export) |
| Alan Wake 2 | runs, Mesh direct; about 10% less Mesh GPU time per draw than private corners alone |

## Build

```sh
git clone https://github.com/lonewolf0622/bc250meshtaskwork.git
cd bc250meshtaskwork
meson setup build -Dvulkan-drivers=amd -Dgallium-drivers= -Dplatforms=x11,wayland \
      -Dbuildtype=release -Dllvm=disabled -Dvideo-codecs=
ninja -C build src/amd/vulkan/libvulkan_radeon.so src/amd/vulkan/radeon_devenv_icd.x86_64.json
```

The driver is then `build/src/amd/vulkan/libvulkan_radeon.so`. The ICD file
`build/src/amd/vulkan/radeon_devenv_icd.x86_64.json` already points at it. This tree is Mesa 26.2.1 plus the BC-250
work; a patch against stock Mesa 26.2.1 is provided with the releases.

## Use

**Without replacing your system driver** (recommended to start), point the Vulkan loader at the build. Steam
launch options:

```
VK_DRIVER_FILES=/path/to/bc250meshtaskwork/build/src/amd/vulkan/radeon_devenv_icd.x86_64.json RADV_DIRECTMESH=1 %command%
```

Keep the build folder in place: the ICD file points at the library inside it.

**As your system driver** (after `meson install`, or through your distribution's packaging):

```
RADV_DIRECTMESH=1 %command%
```

`RADV_DIRECTMESH=1` enables Mesh shaders and every direct-path setting. Any `RADV_BC250_*` / `BC250_*` variable you
set yourself still takes priority, which is useful for testing. Without the switch the driver behaves as its base BC-250
configuration. The switch only affects the BC-250 (GFX1013).

Useful extras:

- `RADV_DEBUG=nomeshshader`: hide Mesh shaders, so the game uses its own non-Mesh path (for comparisons).
- `RADV_BC250_MESH_SAFE_ADAPTIVE=0`: private corners only (no shared-vertex export).

## Not included

- `VK_EXT_device_generated_commands`, graphics pipeline libraries and shader objects are hidden while Mesh uses the
  hybrid Task path. vkd3d-proton still handles `ExecuteIndirect` for Mesh draws. A small number of games that need
  state-changing `ExecuteIndirect` may not render fully.
- Mesh pipeline statistics queries, and multiview with Mesh shaders.

## If a game hangs

The BC-250 can lock up hard on a GPU hang. If you want to report one:

1. Set the kernel parameter `amdgpu.gpu_recovery=0` (the machine then stays reachable, e.g. over SSH).
2. Collect `dmesg` and, if possible, a `umr` wave dump.
3. Include the game and the launch options you used.

## Offline tests

`tests/bc250-mesh/` holds offline suites that run under Mesa's amdgpu noop drm-shim (`AMDGPU_GPU_ID=gfx1013`) without
touching the GPU. Each suite's README lists its variables (`ICD=`, `BUILD=`, `SHIM=`).

## Credits

- Luckiskind ([github.com/luckiskind](https://github.com/luckiskind)): [bc250-radv-r2](https://github.com/luckiskind/bc250-radv-r2), BC-250 RADV Mesh work referenced during this project.

## License

MIT, like Mesa. See `docs/license.rst`. Mesa 3D and RADV are MIT-licensed; AMD PAL and LLPC sources were used as a
programming reference for GFX10 NGG.

# bc250meshtaskwork: DirectMesh for the AMD BC-250

Mesh shader support for the AMD BC-250 (GFX1013, RDNA1-based) in Mesa's RADV Vulkan driver, with Mesh shaders drawn on
a **safe direct path**: no split/replay, one launch per Mesh workgroup, and built-in protection against the index
patterns that hang this chip.

**Download:** [patch against stock Mesa 26.2.1](https://github.com/lonewolf0622/bc250meshtaskwork/releases/download/directmesh-v1.2/bc250-directmesh-mesa-26.2.1.patch)
(also in [`patches/`](patches/) and on the [release page](https://github.com/lonewolf0622/bc250meshtaskwork/releases/tag/directmesh-v1.2)),
or build this repository directly (see Build).

Turn it on with one switch:

```
RADV_DIRECTMESH=1 %command%
```

**v1.3 candidate:** adds indexed draws for eligible Mesh-only shaders, including generic flat per-primitive
attributes, plus LDS planning, LDS coverage, merged checks, direct primitive attributes and 65-primitive pieces.
Offline compilation and validation are complete; v1.3 hardware correctness, CTS and game performance testing
must finish before release. The download above remains the tested v1.2 release.

## What you get

- `VK_EXT_mesh_shader` (Mesh + Task) on the BC-250, usable by D3D12 games through normal Proton / vkd3d-proton.
- `VK_KHR_fragment_shader_barycentric`, so vkd3d-proton keeps Mesh shaders enabled in UE5 games.
- `VK_EXT_device_generated_commands` with Mesh and Task draws, so D3D12 `ExecuteIndirect` works through vkd3d-proton
  (Crimson Desert needs it).
- Multiview with Mesh shaders, so vkd3d-proton keeps D3D12 view instancing when Mesh shaders are on.
- Indirect Task draws that scale: the rarely used extra Task chunks are recorded once per call and skipped with one
  check per draw, instead of being recorded for every draw.
- **Direct Mesh path:** each Mesh workgroup is drawn in a single launch. Per workgroup, after culling, the driver
  picks the cheapest export that it can prove safe:
  - shared vertices, when the surviving triangles use every vertex with small index backjumps;
  - renumbered vertices, when culling or a split left unused vertices: only the used ones are exported, in order;
  - shared vertices plus every triangle, when only a few triangles were culled;
  - otherwise private triangle corners, which rule out the chip's hang patterns by construction.
- **Automatic Mesh shader converter:** the driver rewrites any application Mesh (and Task) shader into a form the
  direct path accepts, with no per-game profiles. It splits large meshlets into pieces (Task pipelines included),
  keeps written `gl_PrimitiveID` values across pieces, runs subgroup-free wave32 shaders in wave64, drops task payload
  declarations a Mesh shader never reads, and folds Task launch counts that have no compile-time bound into their own
  grid dimension.
- **Fail closed:** a shape nothing can prove safe takes the older split/expansion path (also private corners) or, as a
  last resort, is refused at pipeline creation. It is never drawn on the unprotected raw route. Split Mesh shaders
  that write images run those writes once, in the first piece.

## Tested

Vulkan CTS on real hardware (with `RADV_DIRECTMESH=1`):

| Test set | Result |
|---|---|
| `dEQP-VK.mesh_shader.ext.*` + fragment-barycentric Mesh cases that run on this device | 3,558 / 3,558 pass, 0 hangs, every pipeline direct (v1.1) |
| Mesh/Task stages in other groups (binding model, subgroups, SPIR-V, atomics, dynamic state, ...) | 10,036 pass, 0 fail, 0 hangs |

v1.2 hardware image checks (each byte-identical against the proven route): device-generated Mesh and Task draws,
descriptor buffers with device-generated commands, graphics pipeline libraries and shader objects with Mesh, and many
indirect Task draws including one large enough to need the extra chunks (old path, new path and device-generated
commands all identical).

Each converter step also passed a one-shot hardware image check: the same scene drawn by the proven route and by the
new route must be byte-identical.

Public Mesh shader samples, compiled offline (Microsoft D3D12 Mesh shader samples, Khronos Vulkan-Samples, Sascha
Willems' Vulkan examples, niagara, NVIDIA vk_lod_clusters; 85 shader/size variants that build):

| Result | v1.0 | v1.1 |
|---|---|---|
| Direct | 33 | 83 |
| Older split/expansion path | 4 | 0 |
| Unprotected raw route | 2 | 0 |
| Refused | 46 | 2 |

The two refused variants: one fragment shader reads `gl_PrimitiveID` that its Mesh shader never writes (invalid
usage), and one debug-statistics variant updates global counters with atomics that a split would repeat.

Real meshlet data (meshoptimizer on 44 models) showed that culling usually leaves unused vertices in a meshlet; the
renumbered export keeps most of those workgroups on shared vertices instead of private corners.

Games (Steam / Proton, D3D12):

| Game | Result |
|---|---|
| Final Fantasy VII Rebirth | runs, Mesh direct (its shaders are unchanged by the adaptive export) |
| Control | runs, Mesh direct; about 8% less Mesh GPU time per draw than private corners alone |
| Hellblade 2 | runs, all Mesh pipelines direct, including barycentric shaders (unchanged by the adaptive export) |
| Alan Wake 2 | runs, Mesh direct; about 10% less Mesh GPU time per draw than private corners alone |
| Crimson Desert | runs with Mesh shaders on (v1.2), including Medium settings |

The game figures were measured with v1.0. With v1.1, Final Fantasy VII Rebirth and Hellblade 2 compile to the same
shaders, and Control was played again with the renumbered export.

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
- `RADV_BC250_MESH_SAFE_COMPACT=0`: no renumbered-vertex export.
- `RADV_BC250_MESH_IDXPASS=0`: keep the v1.3 Mesh routes while disabling indexed Mesh draws.

## Changes

- **v1.3 candidate:** eligible Mesh-only triangle pipelines can run their index work as compute and render with
  ordinary indexed draws. Primitive attributes use private provoking corners and flat fragment inputs. Vertex
  work that still requires another invocation, small meshlets (at most 32 vertices), single-record indirect calls,
  externally visible memory writes and simultaneous command buffers keep the protected Mesh route. Persistent
  index storage shares a 128 MiB device budget, with automatic Mesh fallback on allocation or indirect-pool overflow.
  The preset also enables LDS planning and coverage, merged checks, direct primitive attributes and 65-primitive
  pieces. All six additions can be switched off individually; see [release notes](docs/directmesh-v1.3.md).

- **v1.2:** `VK_EXT_device_generated_commands` for Mesh/Task, multiview with Mesh shaders, split Mesh shaders that write
  images, indirect Task draws recorded once per call (many indirect Task draws were very slow), vertex sharing for
  per-primitive data, leaner per-workgroup checks, wider NGG culling, graphics pipeline libraries kept for DXVK games.
- **v1.1:** automatic Mesh shader converter (renumbered-vertex export, Task pieces on the direct path, `gl_PrimitiveID`
  with pieces, wave64 promotion, unread task payloads, unbounded Task launch counts); fail-closed refusal of any Mesh
  pipeline without a protected route (in v1.0 two rare shapes could reach the raw route: Mesh-only lines above the
  private-corner capacity, and triangles with an explicit shared-memory layout and 65-85 primitives).
- **v1.0:** safe direct Mesh path, adaptive per-workgroup export, `RADV_DIRECTMESH=1`.

## Not included

- Mesh pipeline statistics queries.
- Per-primitive shading rate from Mesh shaders.

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

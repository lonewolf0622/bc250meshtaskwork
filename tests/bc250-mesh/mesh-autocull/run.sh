#!/usr/bin/env bash
# RADV_BC250_MESH_AUTOCULL offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; the harness also refuses any other device. Each case compiles
# a Mesh shape with the the base driver launcher policy (post-Mesh VGT_FLUSH off, as HB2 runs), NIR_DEBUG=validate
# and ACO_DEBUG=validateir,validatera, records a direct and an indirect Mesh draw (plus the dynamic
# state draws of the case), submits to the shim and checks the IB (RADV_DEBUG=dumpibs), the ACO IR
# (RADV_DEBUG=shaders) and the uploaded binary (see check.py), once with RADV_BC250_MESH_AUTOCULL=1
# and once with it off:
#   candidates:     autocull applied (trace), culling + compaction code in the ISA, the NGG culling
#                   settings user SGPR carries the expected value at every draw (dynamic cull mode,
#                   front face, rasterizer discard, y-flipped viewport, conservative rasterization),
#                   the viewport SGPRs are written, launch registers identical to off (fast launch 0,
#                   AMP >= workgroup), LDS < 32 KiB, no scratch;
#   non-candidates: on is byte-identical to off.
# With OLDICD=<icd of the build before the switch existed>, every off run is also compared with that
# build (byte-identical dumps, the switch-off guarantee).
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] [OLDICD=<icd>] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
# Baselines which already implement autocull must compare with the switch off.
OLD_AUTOCULL=${OLD_AUTOCULL:-1}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-mesh-autocull.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1
  BC250_CAPTURE_MESH_CODE=1 RADV_DEBUG=dumpibs,shaders,nocache)

shim() {
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_FL1 \
    -u RADV_BC250_MESH_MERGE -u RADV_BC250_MESH_AMD \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/autocull" "$here/autocull.c" -lvulkan || exit 2
glslangValidator -S task --target-env vulkan1.3 -o "$work/task.spv" "$here/shape.task" >/dev/null || exit 2

fail=0; passed=0; total=0
# name | glslang defines | grid x | harness options | expect (yes / no:<reason>) | settings per Mesh draw
#      | extra environment
# Settings (radv_get_nggc_settings, 1 sample): 0xf8000000 = small-primitive precision 2^-8, 8 = small
# primitives, 4 = front face CCW (after the y-flip correction), 2 = cull back, 1 = cull front.
while IFS='|' read -r name defs gx opts expect settings extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  pp=0; case "$defs" in *PERPRIM=1*) pp=1;; esac
  glslangValidator -S mesh --target-env vulkan1.3 $defs -o "$work/$name.mesh.spv" "$here/shape.mesh" >/dev/null || exit 2
  glslangValidator -S frag --target-env vulkan1.3 -DPERPRIM=$pp -o "$work/$name.frag.spv" "$here/shape.frag" >/dev/null || exit 2
  for sw in 0 1; do
    shim "$ICD" env $extra RADV_BC250_MESH_AUTOCULL=$sw ./autocull "$name.mesh.spv" "$name.frag.spv" $gx $opts \
      > "$work/$name.$sw.out" 2> "$work/$name.$sw.err"
  done
  total=$((total + 1))
  status=ok; detail=
  if grep -q '^UNSUPPORTED' "$work/$name.1.out"; then
    total=$((total - 1))
    printf '%-14s status=skipped %s\n' "$name" "$(head -1 "$work/$name.1.out")"
    continue
  fi
  grep -q '^SUBMIT_OK' "$work/$name.1.out" && grep -q '^SUBMIT_OK' "$work/$name.0.out" || { status=FAIL; detail="no-submit"; }
  res=$(python3 "$here/check.py" "$work/$name.1.err" "$work/$name.0.err" "$expect" "$settings") || status=FAIL
  if [ -n "$OLDICD" ]; then
    shim "$OLDICD" env $extra RADV_BC250_MESH_AUTOCULL=$OLD_AUTOCULL ./autocull "$name.mesh.spv" "$name.frag.spv" $gx $opts \
      > "$work/$name.old.out" 2> "$work/$name.old.err"
    if cmp -s "$work/$name.old.err" "$work/$name.0.err" && cmp -s "$work/$name.old.out" "$work/$name.0.out"; then
      res="$res off_identical_to_old_build=True"
    else
      res="$res off_identical_to_old_build=False"; status=FAIL
    fi
  fi
  [ "$status" = ok ] && passed=$((passed + 1)) || fail=1
  printf '%-14s status=%s %s\n   %s\n' "$name" "$status" "$detail" "$res"
done <<'CASES'
# Size policy (default): only wave32 shapes with 24..32 triangles (<= 96 lanes) get autocull; the
# other candidates are refused and compile exactly as with the switch off ("policy").
p32_back|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back|yes|0xf800000e
p24_back|-DWS=32 -DNV=72 -DNP=24 -DPRIVATE=1|10|cull=back|yes|0xf800000e
pol_p16|-DWS=32 -DNV=48 -DNP=16 -DPRIVATE=1|10|cull=back|policy|
pol_p23|-DWS=32 -DNV=69 -DNP=23 -DPRIVATE=1|10|cull=back|policy|
pol_p48|-DWS=32 -DNV=144 -DNP=48 -DPRIVATE=1|10|cull=back|policy|
pol_p64|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=none|policy|
pol_p85|-DWS=64 -DNV=255 -DNP=85 -DPRIVATE=1|10|cull=back|policy|
pol_p32_lanes128|-DWS=128 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back|policy|
pol_shared64|-DWS=64 -DNV=64 -DNP=64|10|cull=back|policy|
pol_split128|-DWS=128 -DNV=256 -DNP=128|10|cull=back|policy|
# RADV_BC250_MESH_AUTOCULL_ALL=1: every candidate (the wide policy of 9ce1a76), for experiments.
# private vertices (V = 3P): alternate winding, every 4th triangle off-screen
p16_back|-DWS=32 -DNV=48 -DNP=16 -DPRIVATE=1|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
# wave32, primitives in 2 waves (48 = 32 + 16): the multi-wave slot computation
p48_back|-DWS=32 -DNV=144 -DNP=48 -DPRIVATE=1|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
# wave64, 64 triangles, 192 lanes (f_ls64's expanded registers)
p64_back|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
# f_ls64's case (AC1): cull mode NONE, the runtime skip
p64_none|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=none|yes|0xf800000c|RADV_BC250_MESH_AUTOCULL_ALL=1
# 85 triangles, 255 vertices: the largest expanded shape
p85_back|-DWS=64 -DNV=255 -DNP=85 -DPRIVATE=1|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
# the same in wave32: 256 lanes = 8 waves, the primitives in 3 of them
p85_w32|-DWS=32 -DNV=255 -DNP=85 -DPRIVATE=1 -DRUNTIME=1|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
# default policy, 32 triangles in wave32
p32_front_cw|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=front ff=cw|yes|0xf8000009
# cull mode NONE: the runtime skip (no face culling: switch-off epilogue)
p32_none|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=none|yes|0xf800000c
# negative viewport height: the face orientation flips (CCW front becomes CW)
p32_yflip|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back yflip=1|yes|0xf800000a
# dynamic cull mode + front face: direct/indirect (NONE, CCW), then (BACK, CCW), (FRONT, CW), (NONE, CCW)
p32_dyncull|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=dyn|yes|0xf800000c,0xf800000c,0xf800000e,0xf8000009,0xf800000c
# dynamic rasterizer discard: off (direct, indirect), on (everything culled), off
p32_dyndiscard|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back discard=dyn|yes|0xf800000e,0xf800000e,0x00000003,0xf800000e
# conservative rasterization (overestimate): settings 0, the runtime skip
p32_cons|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back cons=1|yes|0x0
# shared indices on a circle: the base driver's expansion gives private vertices, then autocull
shared32|-DWS=32 -DNV=32 -DNP=32|10|cull=back|yes|0xf800000e
shared64_ls64|-DWS=64 -DNV=64 -DNP=64|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
# per-primitive attribute (moves with its triangle), application CullPrimitive (combined)
pp32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1 -DPERPRIM=1|10|cull=back|yes|0xf800000e
cull32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1 -DCULL=1|10|cull=back|yes|0xf800000e
pp_cull_rt48|-DWS=32 -DNV=144 -DNP=48 -DPRIVATE=1 -DPERPRIM=1 -DCULL=1 -DRUNTIME=1|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
shared_pp_cull|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=1 -DCULL=1 -DRUNTIME=1|10|cull=back|yes|0xf800000e
# the base driver's split consumes CullPrimitive (pp32/cull32 above run as split pieces). Without the split the
# expanded shader keeps CullPrimitive and autocull combines it with its own tests (app_cull=1).
cull_nosplit|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1 -DPERPRIM=1 -DCULL=1|10|cull=back|yes:app_cull|0xf800000e|RADV_BC250_SPLIT_MESH=false
cull_nosplit48|-DWS=32 -DNV=144 -DNP=48 -DPRIVATE=1 -DPERPRIM=1 -DCULL=1 -DRUNTIME=1|10|cull=back|yes:app_cull|0xf800000e|RADV_BC250_SPLIT_MESH=false RADV_BC250_MESH_AUTOCULL_ALL=1
# w < 0 corners, subgroup operation
behind32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1 -DBEHIND=1|10|cull=back|yes|0xf800000e
sg32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1 -DSUBGROUP=1|10|cull=back|yes|0xf800000e
# the base driver split (128 triangles > 85): the pieces are expanded and get autocull
split128|-DWS=128 -DNV=256 -DNP=128|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
split128_cull|-DWS=64 -DNV=128 -DNP=128 -DCULL=1 -DPERPRIM=1|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_AUTOCULL_ALL=1
# Task + Mesh (the base driver hybrid Task replay)
task32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back task=task.spv|yes|0xf800000e
# clip distance: a candidate (the survivors export their corners' distances; a triangle with the
# distance negative at all three corners is also culled)
clip32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1 -DCLIP=1|10|cull=back|yes|0xf800000e|RADV_BC250_MESH_ALLOW_POS1=1
# not candidates: byte-identical to the switch off
vp32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1 -DVIEWPORT=1|10|cull=back|no:viewport output|
line32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back poly=line|no:polygon mode|
dynpoly32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back poly=dyn|no:polygon mode|
mv32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back mv=1|no:multiview|
CASES
echo "mesh-autocull: $passed/$total passed"
exit $fail

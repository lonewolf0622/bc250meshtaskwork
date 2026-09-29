#!/usr/bin/env bash
# RADV_BC250_MESH_AMD offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; amdmode also refuses any other device. Each case compiles a
# Mesh shape with the the base driver launcher policy (POST_MESH_VGT_FLUSH=1) and NIR_DEBUG=validate
# ACO_DEBUG=validateir,validatera, records direct / indirect / indirect-count Mesh draws around a VS
# draw, submits to the shim and checks the IB (RADV_DEBUG=dumpibs):
#   switch on : route, GE_NGG_SUBGRP_CNTL, VGT_GS_MAX_VERT_OUT, VGT_REUSE_OFF, no setup dispatch,
#               no post-Mesh VGT_FLUSH (AMD route); base-route cases must be byte-identical to off.
#   switch off: the base driver's registers (see README.md), no VGT_REUSE_OFF write.
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-amd-mode.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=1
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1 RADV_DEBUG=dumpibs,nocache)

shim() {
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$ICD" VK_ICD_FILENAMES="$ICD" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/amdmode" "$here/amdmode.c" -lvulkan || exit 2
glslangValidator -S vert --target-env vulkan1.3 -o "$work/tri_vert.spv" "$here/tri.vert" >/dev/null || exit 2
glslangValidator -S frag --target-env vulkan1.3 -o "$work/tri_frag.spv" "$here/tri.frag" >/dev/null || exit 2

fail=0
# name | glslang defines | route with the switch on | GE_NGG_SUBGRP_CNTL on | VGT_GS_MAX_VERT_OUT on
while IFS='|' read -r name defs route subgrp maxvert; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  pp=0; case "$defs" in *PERPRIM=1*) pp=1;; esac
  glslangValidator -S mesh --target-env vulkan1.3 $defs -o "$work/$name.mesh.spv" "$here/shape.mesh" >/dev/null || exit 2
  glslangValidator -S frag --target-env vulkan1.3 -DPERPRIM=$pp -o "$work/$name.frag.spv" "$here/shape.frag" >/dev/null || exit 2
  for wave in 64 32; do
    for sw in 0 1; do
      shim env RADV_BC250_MESH_AMD=$sw ./amdmode "$name.mesh.spv" "$name.frag.spv" $wave \
        > "$work/$name.w$wave.$sw.out" 2> "$work/$name.w$wave.$sw.err"
    done
    on=$work/$name.w$wave.1; off=$work/$name.w$wave.0
    status=ok; detail=
    grep -q '^SUBMIT_OK' "$on.out" && grep -q '^SUBMIT_OK' "$off.out" || { status=FAIL; detail="no-submit"; }
    if cat "$on.err" "$off.err" | grep -qiE 'validation failed|NIR_VALIDATE|assertion|error:|^FAIL '; then status=FAIL; detail="$detail validation"; fi
    got=base; grep -q 'BC250 MESH AMD route' "$on.err" && got=amd
    [ "$got" = "$route" ] || { status=FAIL; detail="$detail route=$got"; }
    offregs=$(python3 "$here/check.py" "$off.err" base 0 0 | sed 's/ OK$//; s/ FAIL.*//')
    if [ "$route" = amd ]; then
      onregs=$(python3 "$here/check.py" "$on.err" amd "$subgrp" "$maxvert") || status=FAIL
      grep -q 'VGT_REUSE_OFF' "$off.err" && { status=FAIL; detail="$detail off-writes-reuse"; }
    else
      onregs="identical_to_off"
      cmp -s "$on.err" "$off.err" || { status=FAIL; onregs="DIFFERS_FROM_OFF"; }
    fi
    [ "$status" = ok ] || fail=1
    printf '%-18s wave=%s route=%s status=%s %s\n   on:  %s\n   off: %s\n' "$name" "$wave" "$got" "$status" "$detail" "$onregs" "$offregs"
  done
done <<'CASES'
# the 64V/124P meshoptimizer shape with 128 threads (bench a_ls128, regdump A)
a64_124_ws128|-DWS=128 -DNV=64 -DNP=124|amd|0x00010080|0x80
# the same with 32 threads (bench a_ls32): T = 124 is not a wave multiple
a64_124_ws32|-DWS=32 -DNV=64 -DNP=124|amd|0x0000f87c|0x7c
# the README-v5 shape: 64 invocations, 256 vertices (regdump D)
v5_256_2_ws64|-DWS=64 -DNV=256 -DNP=2|amd|0x00020100|0x100
# v5 with shared memory and workgroup barriers: 1-2 API waves, 4-8 hardware waves
v5_shared_256_2_ws64|-DWS=64 -DNV=256 -DNP=2 -DSHARED=1|amd|0x00020100|0x100
c128_128_ws128|-DWS=128 -DNV=128 -DNP=128|amd|0x00010080|0x80
f64_64_ws64|-DWS=64 -DNV=64 -DNP=64|amd|0x00008040|0x40
# ineligible: per-primitive generic output, CullPrimitive -> the base driver route, byte-identical to off
pp32_32_ws32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=1|base|-|-
pp64_124_ws128|-DWS=128 -DNV=64 -DNP=124 -DPERPRIM=1|base|-|-
cull32_32_ws32|-DWS=32 -DNV=32 -DNP=32 -DCULL=1|base|-|-
cull64_126_ws128|-DWS=128 -DNV=64 -DNP=126 -DCULL=1|base|-|-
CASES

# One part at a time (the G1 bisection switches), f64_64_ws64 at wave64, post-Mesh VGT_FLUSH=1 as
# above: the expected GE_NGG_SUBGRP_CNTL / VGT_GS_MAX_VERT_OUT / VGT_REUSE_OFF(mesh/vs/end) per part.
# The the base driver expansion gives this shape 192 private vertices (AMP 64, THDS 0).
glslangValidator -S mesh --target-env vulkan1.3 -DWS=64 -DNV=64 -DNP=64 -o "$work/parts.mesh.spv" "$here/shape.mesh" >/dev/null || exit 2
glslangValidator -S frag --target-env vulkan1.3 -DPERPRIM=0 -o "$work/parts.frag.spv" "$here/shape.frag" >/dev/null || exit 2
while IFS='|' read -r sw want; do
  [ -z "$sw" ] && continue
  shim env $sw ./amdmode parts.mesh.spv parts.frag.spv 64 > "$work/parts.$sw.out" 2> "$work/parts.$sw.err"
  got=$(python3 "$here/check.py" "$work/parts.$sw.err" base 0 0 | grep -oE 'GE_NGG_SUBGRP_CNTL=[^ ]+ VGT_GS_MAX_VERT_OUT=[^ ]+ VGT_REUSE_OFF\(mesh/vs/end\)=[^ ]+')
  status=ok
  grep -q '^SUBMIT_OK' "$work/parts.$sw.out" || status=FAIL
  grep -qiE 'validation failed|NIR_VALIDATE|assertion|error:' "$work/parts.$sw.err" && status=FAIL
  [ "$got" = "$want" ] || status=FAIL
  [ "$status" = ok ] || fail=1
  printf 'part %-32s status=%s\n   got:  %s\n   want: %s\n' "$sw" "$status" "$got" "$want"
done <<'PARTS'
RADV_BC250_MESH_AMD=0|GE_NGG_SUBGRP_CNTL=0x000000c0 VGT_GS_MAX_VERT_OUT=0x000000c0 VGT_REUSE_OFF(mesh/vs/end)=None/None/None
RADV_BC250_MESH_AMD_ROUTE=1|GE_NGG_SUBGRP_CNTL=0x00000040 VGT_GS_MAX_VERT_OUT=0x00000040 VGT_REUSE_OFF(mesh/vs/end)=None/None/None
RADV_BC250_MESH_AMD_SIZE=1|GE_NGG_SUBGRP_CNTL=0x000180c0 VGT_GS_MAX_VERT_OUT=0x000000c0 VGT_REUSE_OFF(mesh/vs/end)=None/None/None
RADV_BC250_MESH_AMD_REUSE_OFF=1|GE_NGG_SUBGRP_CNTL=0x000000c0 VGT_GS_MAX_VERT_OUT=0x000000c0 VGT_REUSE_OFF(mesh/vs/end)=1/0/0
RADV_BC250_MESH_AMD=1|GE_NGG_SUBGRP_CNTL=0x00008040 VGT_GS_MAX_VERT_OUT=0x00000040 VGT_REUSE_OFF(mesh/vs/end)=1/0/0
PARTS

# Fast launch 0 is the only Mesh launch mode: every VGT_SHADER_STAGES_EN in the IB (Mesh and VS)
# has GS_FAST_LAUNCH = 0.
for sw in RADV_BC250_MESH_AMD=0; do
  want="GS_FAST_LAUNCH = 0 "
  got=$(sed 's/\x1b\[[0-9;]*m//g' "$work/parts.$sw.err" | grep -oE 'GS_FAST_LAUNCH = [0-9]+' | sort -u | tr '\n' ' ')
  status=ok; [ "$got" = "$want" ] || { status=FAIL; fail=1; }
  printf 'launch %-24s status=%s got: %s\n' "default" "$status" "$got"
done
exit $fail

#!/usr/bin/env bash
# RADV_BC250_MESH_DEALLOC_DIST offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; deallocmode also refuses any other device. Each case builds one
# Mesh pipeline (the amd-mode test shapes) and one VS pipeline with the launcher policy (post-Mesh VGT
# flush off), NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1, records one of
# the deallocmode modes (mix, meta, secondary, reset, vsonly) and submits it to the shim. The IB
# (RADV_DEBUG=dumpibs,allbos: executed secondaries are printed nested) is checked by check.py:
#   switch unset: no VGT_OUT_DEALLOC_CNTL write; with OLDICD=<icd of the base build> the IB is
#                 byte-identical to that build's.
#   switch =127:  DEALLOC_DIST = 127 for every Mesh draw, 32 for every other draw, 32 at the end of
#                 every IB (primary and secondary), no redundant write, and the rest of the IB equal
#                 to the unset run.
# Then the value checks on the raw route: 64, 32 and 1 are used; 0, 128, 200, abc, 12x, -1 and an empty
# value are ignored with a message, and the IB is byte-identical to the unset run.
# usage: ICD=<radeon_devenv_icd json> [SHIM=..] [OLDICD=..] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
shapes=$here/../amd-mode
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-mesh-dealloc.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1 RADV_DEBUG=dumpibs,nocache,allbos)

shim() { # icd env...
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_DEALLOC_DIST \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}
ibonly() { sed 's/\x1b\[[0-9;]*m//g' "$1" | awk '/IB begin/{f=1} f{print} /IB end/{f=0}'; }

cc -O1 -Wall -o "$work/deallocmode" "$here/deallocmode.c" -lvulkan || exit 2
glslangValidator -S vert --target-env vulkan1.3 -o "$work/tri_vert.spv" "$shapes/tri.vert" >/dev/null || exit 2
glslangValidator -S frag --target-env vulkan1.3 -o "$work/tri_frag.spv" "$shapes/tri.frag" >/dev/null || exit 2

fail=0
bad() { fail=1; }
# name | glslang defines | wave | switches | route trace expected
while IFS='|' read -r name defs wave sw route; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  pp=0; case "$defs" in *PERPRIM=1*) pp=1;; esac
  glslangValidator -S mesh --target-env vulkan1.3 $defs -o "$work/$name.mesh.spv" "$shapes/shape.mesh" >/dev/null || exit 2
  glslangValidator -S frag --target-env vulkan1.3 -DPERPRIM=$pp -o "$work/$name.frag.spv" "$shapes/shape.frag" >/dev/null || exit 2
  for mode in mix meta secondary reset vsonly; do
    b=$work/$name.$mode
    shim "$ICD" $sw ./deallocmode "$name.mesh.spv" "$name.frag.spv" $wave $mode > "$b.off.out" 2> "$b.off.err"
    shim "$ICD" $sw RADV_BC250_MESH_DEALLOC_DIST=127 ./deallocmode "$name.mesh.spv" "$name.frag.spv" $wave $mode \
      > "$b.on.out" 2> "$b.on.err"
    status=ok; detail=
    grep -q '^SUBMIT_OK' "$b.off.out" && grep -q '^SUBMIT_OK' "$b.on.out" || { status=FAIL; detail="$detail no-submit"; }
    if cat "$b.off.err" "$b.on.err" | grep -qiE 'validation failed|NIR_VALIDATE|assertion|error:|^FAIL '; then
      status=FAIL; detail="$detail validation"; fi
    got=base; grep -q 'BC250 MESH AMD route' "$b.on.err" && got=amd
    grep -q 'BC250 MESH COMPACT: applied' "$b.on.err" && got=compact
    [ "$mode" = vsonly ] || [ "$got" = "$route" ] || { status=FAIL; detail="$detail route=$got"; }
    grep -q 'DEALLOC_DIST=127 (RADV_BC250_MESH_DEALLOC_DIST)' "$b.on.err" || { status=FAIL; detail="$detail no-enable-message"; }
    offres=$(python3 "$here/check.py" "$b.off.err" off) || status=FAIL
    onres=$(python3 "$here/check.py" "$b.on.err" 127 "$b.off.err") || status=FAIL
    old=-
    if [ -n "$OLDICD" ]; then
      shim "$OLDICD" $sw ./deallocmode "$name.mesh.spv" "$name.frag.spv" $wave $mode > "$b.old.out" 2> "$b.old.err"
      if cmp -s <(ibonly "$b.old.err") <(ibonly "$b.off.err"); then old=identical; else old=DIFFERENT; status=FAIL; fi
    fi
    [ "$status" = ok ] || bad
    printf '%-14s %-9s route=%s status=%s%s unset_vs_base_build=%s\n   unset: %s\n   127:   %s\n' \
      "$name" "$mode" "$got" "$status" "$detail" "$old" "$offres" "$onres"
  done
done <<'CASES'
# the base driver's expanded route (64V/64P: 192 private vertices)
base_f64|-DWS=64 -DNV=64 -DNP=64|64||base
# the raw route (shared vertices, no split or expansion): the HW-9 configuration
raw_f64|-DWS=64 -DNV=64 -DNP=64|64|RADV_BC250_MESH_AMD_ROUTE=1|amd
# the full AMD mode: VGT_REUSE_OFF is switched in the same place
amd_f64|-DWS=64 -DNV=64 -DNP=64|64|RADV_BC250_MESH_AMD=1|amd
# P > 85: split pieces on the base route
split_a124|-DWS=128 -DNV=64 -DNP=124|64||base
# the shared-vertex compaction
compact_f64|-DWS=64 -DNV=64 -DNP=64|64|RADV_BC250_MESH_COMPACT=1|compact
# per-primitive output, wave32
pp32_w32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=1|32||base
CASES

# Values (raw route, mix): valid values are used as given, invalid ones are ignored with a message.
name=raw_f64; b=$work/$name.mix
while IFS='|' read -r val kind; do
  [ -z "$kind" ] && continue
  v=$work/$name.mix.val_$(echo "$val" | tr -c 'a-z0-9' '_')
  shim "$ICD" RADV_BC250_MESH_AMD_ROUTE=1 RADV_BC250_MESH_DEALLOC_DIST="$val" ./deallocmode "$name.mesh.spv" "$name.frag.spv" 64 mix \
    > "$v.out" 2> "$v.err"
  status=ok
  grep -q '^SUBMIT_OK' "$v.out" || status=FAIL
  if [ "$kind" = valid ]; then
    res=$(python3 "$here/check.py" "$v.err" "$val" "$b.off.err") || status=FAIL
    grep -q "DEALLOC_DIST=$val (RADV_BC250_MESH_DEALLOC_DIST)" "$v.err" || status=FAIL
  else
    res=$(python3 "$here/check.py" "$v.err" off) || status=FAIL
    grep -q "RADV_BC250_MESH_DEALLOC_DIST=$val ignored" "$v.err" || [ -z "$val" ] || status=FAIL
    cmp -s <(ibonly "$b.off.err") <(ibonly "$v.err") && res="$res ib=identical_to_unset" || { status=FAIL; res="$res ib=DIFFERENT"; }
  fi
  [ "$status" = ok ] || bad
  printf 'value %-5s %-7s status=%s %s\n' "'$val'" "$kind" "$status" "$res"
done <<'VALUES'
64|valid
32|valid
1|valid
0|invalid
128|invalid
200|invalid
abc|invalid
12x|invalid
-1|invalid
|invalid
VALUES
exit $fail

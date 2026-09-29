#!/usr/bin/env bash
# BC250 Mesh LDS fit / piece ceiling regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; pipe also refuses any other device. Each case creates one
# Mesh (+Task) pipeline with the base driver launcher policy (post-Mesh VGT_FLUSH off, as the games
# run), NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1, records a direct,
# an indirect and an indirect-count Mesh draw and submits them to the shim.
#   nanite.mesh/.frag : Hellblade 2's Nanite interface (256V/128P, split into pieces).
#   wide.mesh/.frag   : Control's G-buffer interface (64V/64P, 8 generic outputs, 5 render targets).
# Checked: the pipeline result, the pieces, the expanded Mesh LDS layout (BC250 MESH LDS), the LDS fit
# retries (a pipeline that does not fit is compiled again with dead shared copies dropped, then with
# more, smaller pieces), the shader cache with and without the cached split plan, and that
# pipelines that already fit are compiled exactly as before (no retry, same LDS).
# With OLDICD=<icd of a build without the LDS fit>, the shaders of every case that the old build
# admits are byte-identical (BC250_CAPTURE_POLICY_SHADERS code and configuration).
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] [OLDICD=<icd>] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-piece-ceiling.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1)

shim() {
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_MERGE \
    -u RADV_BC250_MESH_AMD -u RADV_BC250_MESH_AUTOCULL -u RADV_BC250_PERF_PIECE_PRIMS -u RADV_BC250_MESH_COMPACT_LDS \
    -u RADV_BC250_SPLIT_BATCH_PREP -u BC250_MESH_TIMER -u RADV_DEBUG \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/pipe" "$here/pipe.c" -lvulkan || exit 2
G="glslangValidator --target-env vulkan1.3"
$G -S mesh -o "$work/nanite.mesh.spv" "$here/nanite.mesh" >/dev/null || exit 2
$G -S frag -o "$work/nanite.frag.spv" "$here/nanite.frag" >/dev/null || exit 2
$G -S mesh -DEXTRA=1 -o "$work/nanite_extra.mesh.spv" "$here/nanite.mesh" >/dev/null || exit 2
$G -S frag -DEXTRA=1 -o "$work/nanite_extra.frag.spv" "$here/nanite.frag" >/dev/null || exit 2
for v in wide:"" wide_fit:"-DFIT=1" wide_primid:"-DPRIMID=1" wide_task:"-DTASK=1"; do
  $G -S mesh ${v#*:} -o "$work/${v%%:*}.mesh.spv" "$here/wide.mesh" >/dev/null || exit 2
  $G -S frag ${v#*:} -o "$work/${v%%:*}.frag.spv" "$here/wide.frag" >/dev/null || exit 2
done
$G -S task -o "$work/wide.task.spv" "$here/wide.task" >/dev/null || exit 2

fail=0; total=0
check() { # name, output file, result, want result, patterns (';'-separated, '!' = must not appear)
  local name=$1 out=$2 got=$3 want=$4 pats=$5 status=ok detail=
  [ "$got" = "$want" ] || { status=FAIL; detail="result=$got"; }
  if [ "$want" = 0 ] && ! grep -q '^SUBMIT_OK' "$out"; then status=FAIL; detail="$detail no-submit"; fi
  if grep -qiE 'validation failed|NIR_VALIDATE|assert|error:|^FAIL ' "$out"; then status=FAIL; detail="$detail validation"; fi
  local IFS=';'
  for p in $pats; do
    [ -z "$p" ] && continue
    if [ "${p#!}" != "$p" ]; then
      grep -qF -- "${p#!}" "$out" && { status=FAIL; detail="$detail unexpected[${p#!}]"; }
    else
      grep -qF -- "$p" "$out" || { status=FAIL; detail="$detail missing[$p]"; }
    fi
  done
  total=$((total + 1)); [ $status = ok ] || fail=1
  local lds=$(grep -o 'V=[0-9]* P=[0-9]* api_shared=[0-9]*.*total=[0-9]*' "$out" | tail -1 | sed 's/ vtx_attr.*total=/ total=/')
  printf '%-22s result=%-3s want=%-3s %-4s %s %s\n' "$name" "$got" "$want" "$status" "${lds:-}" "$detail"
}

# name | mesh | frag | task (- none) | attachments | extra env | want result | patterns
while IFS='|' read -r name mesh frag task rts extra want pats; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  out=$work/$name.out
  t=-; [ "$task" != - ] && t=$task.task.spv
  shim "$ICD" env $extra ./pipe "$mesh.mesh.spv" "$frag.frag.spv" $t $rts > "$out" 2>&1
  got=$(sed -n 's/^PIPELINE_RESULT=//p' "$out" | tail -1)
  check "$name" "$out" "${got:-none}" "$want" "$pats"
  if [ -n "$OLDICD" ]; then
    shim "$OLDICD" env $extra BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh.mesh.spv" "$frag.frag.spv" $t $rts > "$out.old" 2>&1
    if grep -q '^PIPELINE_RESULT=0' "$out.old"; then
      shim "$ICD" env $extra BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh.mesh.spv" "$frag.frag.spv" $t $rts > "$out.new" 2>&1
      sig() { grep -E '^BC250POLICY(CODE)? ' "$1" | sed 's/ va=[0-9a-f]*//' | sort | sha1sum | cut -c1-16; }
      a=$(sig "$out.old"); b=$(sig "$out.new"); s=ok; [ "$a" = "$b" ] || { s=FAIL; fail=1; }
      total=$((total + 1)); printf '%-22s identical-to-old-build %-4s old=%s new=%s\n' "$name" "$s" "$a" "$b"
    fi
  fi
done <<'EOF'
# Hellblade 2 Nanite interface: default ceiling 63 = 3 pieces of 43, unchanged (no retry).
nanite_default|nanite|nanite|-|1||0|pieces=3;V=129 P=43 api_shared=20672;total=29073;!Mesh LDS fit
# RADV_BC250_PERF_PIECE_PRIMS=64: 2 pieces of 64 spill; the retry drops the split's dead copy (3588 bytes) and fits.
nanite_p64|nanite|nanite|-|1|RADV_BC250_PERF_PIECE_PRIMS=64|0|V=192 P=64 api_shared=21260;expanded Mesh rejected;min_pieces=2 reclaim=1;dead shared variables dropped=3;V=192 P=64 api_shared=17672;total=30176;admitted after 1 retries (pieces=2 reclaim=1);direct mesh-only split enabled: pieces=2
nanite_p85|nanite|nanite|-|1|RADV_BC250_PERF_PIECE_PRIMS=85|0|admitted after 1 retries (pieces=2 reclaim=1);total=30176
# RADV_BC250_MESH_COMPACT_LDS=1: every expanded shader gets the compact layout: the per-component
# primitive stores become slice stores (split region 4 bytes instead of 3588), dead copies dropped,
# packed per-vertex records (tc7 keeps the 2 dwords the fragment shader reads). No retry needed.
nanite_compact|nanite|nanite|-|1|RADV_BC250_MESH_COMPACT_LDS=1|0|pieces=3;split=[0,4);BC250 output region: stores=5;V=129 P=43 api_shared=13500 vtx_attr=4@13520 packed;total=20881;!Mesh LDS fit
nanite_compact_p64|nanite|nanite|-|1|RADV_BC250_MESH_COMPACT_LDS=1 RADV_BC250_PERF_PIECE_PRIMS=64|0|split=[0,4);vtx_attr=4@14112 packed;total=25056;!Mesh LDS fit
# One more per-vertex vec4: refused today at any ceiling; now more, smaller pieces.
nanite_extra|nanite_extra|nanite_extra|-|1||0|expanded Mesh rejected;BC250 Mesh LDS fit: admitted
nanite_extra_p64|nanite_extra|nanite_extra|-|1|RADV_BC250_PERF_PIECE_PRIMS=64|0|expanded Mesh rejected;BC250 Mesh LDS fit: admitted
# Control G-buffer interface: unsplit 64 triangles (192 expanded vertices) spill; now 2 pieces of 32.
wide|wide|wide|-|5||0|V=192 P=64 api_shared=7940;expanded Mesh rejected;min_pieces=2 reclaim=1;direct mesh-only split enabled: pieces=2;V=96 P=32;admitted after 1 retries (pieces=2 reclaim=1)
wide_autocull|wide|wide|-|5|RADV_BC250_MESH_AUTOCULL=1|0|direct mesh-only split enabled: pieces=2;admitted after 1 retries (pieces=2 reclaim=1)
wide_batch|wide|wide|-|5|RADV_BC250_SPLIT_BATCH_PREP=1|0|direct mesh-only split enabled: pieces=2;admitted after 1 retries (pieces=2 reclaim=1)
wide_task|wide_task|wide_task|wide|5||0|min_pieces=2;admitted after 1 retries (pieces=2 reclaim=1)
# RADV_BC250_MESH_COMPACT_LDS=1: the packed records (no vec4 slot per 1-3 component output) fit unsplit.
wide_compact|wide|wide|-|5|RADV_BC250_MESH_COMPACT_LDS=1|0|V=192 P=64 api_shared=7940 vtx_attr=8@7968 packed;total=29664;!Mesh LDS fit
# Control interface that fits: unsplit, unchanged.
wide_fit|wide_fit|wide_fit|-|5||0|V=192 P=64;!Mesh LDS fit;!split
# The fragment shader reads PrimitiveID: the split refuses it, so the pipeline stays refused.
wide_primid|wide_primid|wide_primid|-|5||-8|expanded Mesh rejected;min_pieces=2;split rejected: fragment primitive;refused after 1 retries
EOF

# Shader cache: the retried pipeline is found again with its split plan (BC250_CACHE_PLAN=true, the
# policy), and without the plan it is compiled again instead of being paired with the wrong pieces.
for plan in true false; do
  rm -rf "$work/cache"; mkdir -p "$work/cache"
  for run in 1 2; do
    out=$work/cache_$plan.$run.out
    shim "$ICD" env MESA_SHADER_CACHE_DISABLE=false MESA_SHADER_CACHE_DIR="$work/cache" BC250_CACHE_PLAN=$plan \
      ./pipe wide.mesh.spv wide.frag.spv - 5 > "$out" 2>&1
    got=$(sed -n 's/^PIPELINE_RESULT=//p' "$out" | tail -1)
    if [ $run = 1 ]; then pats="admitted after 1 retries (pieces=2 reclaim=1)"
    elif [ $plan = true ]; then pats="BC250 graphics cache hit: split_pieces=2;!Mesh LDS fit"
    else pats="admitted after 1 retries (pieces=2 reclaim=1);!cache hit"; fi
    check "cache_plan_${plan}_run$run" "$out" "${got:-none}" 0 "$pats"
  done
done
echo "cases=$total failed=$fail"
exit $fail

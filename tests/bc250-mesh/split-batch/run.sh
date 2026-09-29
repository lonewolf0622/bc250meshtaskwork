#!/usr/bin/env bash
# RADV_BC250_SPLIT_BATCH_PREP offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; the harness also refuses any other device. Each case records an
# sb.c script (render passes with split indirect / indirect-count draws of an HB2-Nanite-like
# 256V/128P per-primitive shape (split=3) and an a_ls32-like 64V/124P shape (split=2), mixed with VS
# draws, non-split Mesh draws and direct split draws; several render passes, suspend/resume,
# VkRenderPass subpasses with an in-subpass barrier, secondary command buffers, conditional rendering,
# more than 128 split draws or more than 1024 setup workgroups in one render pass), submits it twice with the base driver launcher policy
# (post-Mesh VGT_FLUSH off, as HB2 runs) under RADV_DEBUG=dumpibs, with the switch on
# (RADV_BC250_SPLIT_BATCH_PREP=1, RADV_BC250_SPLIT_BATCH_TRACE=1) and off, and check.py checks the
# packet streams (one setup dispatch + one CS_PARTIAL_FLUSH per batch, none for joined draws, each
# draw reads its own list entry's output, the entry and its setup workgroups are the draw's, same draws
# and graphics state as the off run).
# With OLDICD=<icd of the build before the switch existed>, the off runs (also with
# RADV_BC250_SPLIT_BATCH_PREP=0 and the trace variables set) are byte-identical to that build.
# The last case (nirsim.py) interprets the NIR of the per-draw and the batched setup shader on random
# inputs and requires identical driver records.
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] [OLDICD=<icd>] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-split-batch.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera)

shim() {
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_FL1 \
    -u RADV_BC250_MESH_MERGE -u RADV_BC250_MESH_AMD -u BC250_MESH_TIMER -u RADV_BC250_SPLIT_BATCH_PREP \
    -u RADV_BC250_SPLIT_BATCH_TRACE -u RADV_BC250_SPLIT_BATCH_PRINT_NIR -u RADV_BC250_PERF_NO_SPLIT_L2_INV \
    -u BC250_CHAIN_TRACE LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/sb" "$here/sb.c" -lvulkan || exit 2
glslangValidator -S vert --target-env vulkan1.3 -o "$work/tri_vert.spv" "$here/../mesh-merge/tri.vert" >/dev/null || exit 2
glslangValidator -S frag --target-env vulkan1.3 -o "$work/tri_frag.spv" "$here/../mesh-merge/tri.frag" >/dev/null || exit 2
mk() { # name, mesh defines, per-primitive
  glslangValidator -S mesh --target-env vulkan1.3 $2 -o "$work/$1.mesh.spv" "$here/../mesh-autocull/shape.mesh" >/dev/null || exit 2
  glslangValidator -S frag --target-env vulkan1.3 -DPERPRIM=$3 -o "$work/$1.frag.spv" "$here/../mesh-autocull/shape.frag" >/dev/null || exit 2
}
mk nanite "-DWS=128 -DNV=256 -DNP=128 -DPERPRIM=1" 1
mk als32 "-DWS=32 -DNV=64 -DNP=124" 0
mk small "-DWS=32 -DNV=32 -DNP=32" 0

fail=0; passed=0; total=0
# name | sb.c script | extra environment (both runs)
while IFS='|' read -r name script extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  total=$((total + 1))
  status=ok
  shim "$ICD" env $extra RADV_DEBUG=dumpibs,nocache RADV_BC250_SPLIT_BATCH_PREP=1 RADV_BC250_SPLIT_BATCH_TRACE=1 \
    ./sb "$script" > "$work/$name.on.out" 2> "$work/$name.on.err"
  shim "$ICD" env $extra RADV_DEBUG=dumpibs,nocache ./sb "$script" > "$work/$name.off.out" 2> "$work/$name.off.err"
  grep -q '^DONE' "$work/$name.on.out" && grep -q '^DONE' "$work/$name.off.out" || status=FAIL
  res=$(python3 "$here/check.py" "$script" "$work/$name.on.out" "$work/$name.on.err" \
        "$work/$name.off.out" "$work/$name.off.err") || status=FAIL
  if [ -n "$OLDICD" ]; then
    shim "$OLDICD" env $extra RADV_DEBUG=dumpibs,nocache ./sb "$script" > "$work/$name.old.out" 2> "$work/$name.old.err"
    shim "$ICD" env $extra RADV_DEBUG=dumpibs,nocache RADV_BC250_SPLIT_BATCH_PREP=0 RADV_BC250_SPLIT_BATCH_TRACE=1 \
      ./sb "$script" > "$work/$name.off0.out" 2> "$work/$name.off0.err"
    if cmp -s "$work/$name.old.err" "$work/$name.off.err" && cmp -s "$work/$name.old.out" "$work/$name.off.out" &&
       cmp -s "$work/$name.old.err" "$work/$name.off0.err" && cmp -s "$work/$name.old.out" "$work/$name.off0.out"; then
      res="$res off_identical_to_old_build=True"
    else
      res="$res off_identical_to_old_build=False"; status=FAIL
    fi
  fi
  [ "$status" = ok ] && passed=$((passed + 1)) || fail=1
  printf '%-14s status=%s\n   %s\n' "$name" "$status" "$res"
done <<'CASES'
# one split indirect draw in a render pass
one|B si E|
# two (indirect + indirect count)
two|B si sc E|
# 61 split indirect draws (both split shapes, indirect and indirect count) mixed with VS draws, small
# (non-split) Mesh direct/indirect draws and a direct split draw, one render pass
mix61|B v m si*10 mi ai*5 v sc*10 sd m ac*5 si*10 v ai*5 mi sc*6 s1*10 E|
# indirect count: count below, equal to, above the maximum (sc: count k%6 of max 4; ac: 5 of max 3)
counted|B sc*8 ac*4 E|
# HB2-like: one-record Nanite draws over three render passes
hb2like|B s1*20 v E B s1*20 m E B s1*21 E|
# several render passes per command buffer, one without split draws
multi|B si si E B v m E B sc ai si E|
# suspended / resumed parts of one render pass instance: one batch per part
suspend|Bs si si E Brs si v E Br sc E|
# VkRenderPass: one batch per subpass; a barrier inside subpass 0 closes the batch
legacy|R si si N ai RE|
legacy_barrier|R si P si sc N si RE|
# secondary command buffers keep the per-draw setup (their IB2 is not dumped); the primary batches
secondary|Bx X2 E B si E|
# conditional rendering: draws under it keep the per-draw setup, begin/end close the batch
cond|B si C si sc c si si E|
# more than 128 split draws in one render pass: a second batch opens at draw 129
overflow|B si*130 E|
# large indirect-count draws (1000 records = 16 setup workgroups each): the 1024-workgroup budget
# closes the batch after 64 of them; small draws in between
chunks|B sL si*3 sL*62 s1 sL*4 E|
# no split draw: nothing changes
nosplit|B m mi v sd E|
# with the L2 invalidation A/B switch
l2off|B si ai sc E|RADV_BC250_PERF_NO_SPLIT_L2_INV=1
CASES

# CPU model: interpret the NIR of both setup shaders on random inputs
total=$((total + 1))
shim "$ICD" env RADV_DEBUG=nocache RADV_BC250_SPLIT_BATCH_PREP=1 RADV_BC250_SPLIT_BATCH_PRINT_NIR=1 \
  ./sb "B si E" 1 > "$work/nir.out" 2> "$work/nir.err"
if res=$(python3 "$here/nirsim.py" "$work/nir.err" 8); then passed=$((passed + 1)); st=ok; else fail=1; st=FAIL; fi
printf '%-14s status=%s\n   %s\n' nirsim "$st" "$res"
echo "split-batch: $passed/$total passed"
exit $fail

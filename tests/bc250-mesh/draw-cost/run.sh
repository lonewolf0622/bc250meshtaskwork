#!/usr/bin/env bash
# Draw-cost switches offline regression (drm-shim only, never the real GPU):
#   RADV_BC250_SPLIT_PREP_FREE=1   order-independent split indirect draws without the argument setup
#   RADV_BC250_SCRATCH_REUSE=1     split records / batch lists from the upload buffer, arena without
#                                  ONE_TIME_SUBMIT
#   RADV_BC250_SPLIT_LEAN_SETUP=1  per-draw split setup without radv_meta_begin/end
#   (and the device-frozen hot-path switches, always on: covered by the OLDICD comparison)
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; the harness also refuses any other device. Each case records a pf.c
# script (split indirect / indirect-count draws of a 256V/128P per-primitive Nanite-like shape with an
# order-independent fragment shader (one unused atomicMax, no outputs, like Hellblade 2's) and with
# order-dependent ones (color output, used atomic result, exchange, store), depth/stencil states, queries,
# application compute, push constants, secondaries, non-split Mesh and VS draws), submits it twice with the
# base driver launcher policy under RADV_DEBUG=dumpibs, with each switch and all three, and:
#   lean   byte-identical to the off run (IBs and output);
#   reuse  check.py: same packets and registers except driver record addresses / user data;
#   pf/all check.py: prep-free draws are `pieces` indirect draws of the application's records without
#          setup or CS_PARTIAL_FLUSH, everything else as in the off run, same registers except user data
#          SGPRs and the Mesh shader's PGM_RSRC;
#   old    with OLDICD=<icd of the build before these switches>, the off run is byte-identical to it
#          (also with the three switches set to 0).
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] [OLDICD=<icd>] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-draw-cost.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT
for f in "$ICD" "$OLDICD"; do
  case "$f" in /tmp/*) echo "ICD/OLDICD must not live under /tmp: $f"; exit 2;; esac
done

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
    -u RADV_BC250_SPLIT_PREP_FREE -u RADV_BC250_SCRATCH_REUSE -u RADV_BC250_SPLIT_LEAN_SETUP \
    -u RADV_BC250_PERF_NO_SPLIT_L2_INV -u RADV_BC250_MESH_AUTOCULL -u RADV_BC250_PERF_PIECE_PRIMS \
    -u RADV_BC250_EXPOSE_FAST_BINDING -u RADV_BC250_MESH_COMPACT_LDS -u BC250_CHAIN_TRACE -u RADV_DEBUG \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/pf" "$here/pf.c" -lvulkan || exit 2
G() { glslangValidator --target-env vulkan1.3 "$@" >/dev/null || exit 2; }
G -S vert -o "$work/tri_vert.spv" "$here/../mesh-merge/tri.vert"
G -S frag -o "$work/tri_frag.spv" "$here/../mesh-merge/tri.frag"
G -S mesh -DWS=128 -DNV=256 -DNP=128 -DPERPRIM=1 -o "$work/nanite.mesh.spv" "$here/../mesh-autocull/shape.mesh"
G -S mesh -DWS=32 -DNV=32 -DNP=32 -o "$work/small.mesh.spv" "$here/../mesh-autocull/shape.mesh"
G -S frag -DPERPRIM=0 -o "$work/small.frag.spv" "$here/../mesh-autocull/shape.frag"
G -S frag -DPERPRIM=1 -o "$work/col.frag.spv" "$here/../mesh-autocull/shape.frag"
for k in 0:vis 1:used 2:exch 3:store; do G -S frag -DKIND=${k%%:*} -o "$work/${k##*:}.frag.spv" "$here/pf.frag"; done
G -S comp -o "$work/cs.comp.spv" "$here/cs.comp"

# Byte comparison of two runs' output and IB dump; the BC250_MESH_TIMER exit table (rates per wall-clock
# second) is left out.
same() {
  cmp -s "$work/$1.out" "$work/$2.out" &&
    cmp -s <(sed '/^BC250_MESH_TIMER FINAL/,$d' "$work/$1.err") <(sed '/^BC250_MESH_TIMER FINAL/,$d' "$work/$2.err")
}

fail=0; passed=0; total=0
SW_PF=RADV_BC250_SPLIT_PREP_FREE=1; SW_RE=RADV_BC250_SCRATCH_REUSE=1; SW_LE=RADV_BC250_SPLIT_LEAN_SETUP=1
# name | pf.c script | extra environment (every run)
while IFS='|' read -r name script extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  total=$((total + 1))
  status=ok
  batch=; case "$extra" in *SPLIT_BATCH_PREP=1*) batch=batch;; esac
  # RAW=1: queries crash this tree's RADV_DEBUG=dumpibs packet printer; those cases compare the raw
  # submitted dwords (BC250_CAPTURE_RAW_IBS) and check.py does not run.
  dump="RADV_DEBUG=dumpibs,nocache"; raw=; case "$extra" in *RAW=1*) raw=1; dump="RADV_DEBUG=nocache BC250_CAPTURE_RAW_IBS=true";; esac
  for run in off lean reuse pf all; do
    case $run in off) sw=;; lean) sw=$SW_LE;; reuse) sw=$SW_RE;; pf) sw=$SW_PF;; all) sw="$SW_PF $SW_RE $SW_LE";; esac
    shim "$ICD" env $extra $sw $dump ./pf "$script" > "$work/$name.$run.out" 2> "$work/$name.$run.err"
    grep -q '^DONE' "$work/$name.$run.out" || { status=FAIL; echo "   $run did not finish: $(grep -m1 FAIL "$work/$name.$run.out")"; }
  done
  lean=True
  same "$name.lean" "$name.off" || { lean=False; status=FAIL; }
  if [ -z "$raw" ]; then
    res=$(python3 "$here/check.py" "$script" "$work" "$name" $batch) || status=FAIL
  else
    res="raw_ib_capture=$(grep -c '^BC250RAW SUBMIT' "$work/$name.off.err")_submits"
  fi
  res="$res lean_identical=$lean"
  if [ -n "$OLDICD" ]; then
    shim "$OLDICD" env $extra $dump ./pf "$script" > "$work/$name.old.out" 2> "$work/$name.old.err"
    shim "$ICD" env $extra RADV_BC250_SPLIT_PREP_FREE=0 RADV_BC250_SCRATCH_REUSE=0 RADV_BC250_SPLIT_LEAN_SETUP=0 \
      $dump ./pf "$script" > "$work/$name.off0.out" 2> "$work/$name.off0.err"
    if same "$name.old" "$name.off" && same "$name.old" "$name.off0"; then
      res="$res off_identical_to_old_build=True"
    else
      res="$res off_identical_to_old_build=False"; status=FAIL
    fi
  fi
  [ "$status" = ok ] && passed=$((passed + 1)) || fail=1
  printf '%-12s status=%s\n   %s\n' "$name" "$status" "$res"
done <<'CASES'
# one prep-free split indirect draw
one|B si E|
# every fragment shader kind and draw kind in one render pass
mix|B si sc ci ui xi wi sd m mi v s1 cc E|
# depth/stencil attachment: depth test + write refuses, test only / write only / stencil
depth|BD si dt1 si dw1 si dt0 si dw0 dt1 si st1 si st0 si dt0 E|
# without a depth/stencil attachment the dynamic depth/stencil state does not matter
nodepth|B dt1 dw1 si st1 sc E|
# an active pipeline statistics query (the lean setup falls back to the meta path; prep-free is unaffected)
query|B Q si ci q si Q sc q E|RAW=1
# application compute pipeline bound and dispatched around split draws (the setup rebinds it)
compute|cb cd B si ci sc E cd B ci E cd|
# application push constants between split draws
push|B p si p ci p sc v p si E|
# secondary command buffers (their IB2 is not dumped) and a primary instance after them
secondary|Bx X2 E B si ci E|
# Hellblade 2 like: one-record draws over three render passes
hb2like|B s1*20 v E B s1*20 m E B s1*21 E|
# ONE_TIME_SUBMIT primaries (the transient arena was already on for them)
onetime|B si sc ci wi E|PF_ONE_TIME=1
# with RADV_BC250_SPLIT_BATCH_PREP: prep-free draws leave the batch, the others batch as before
batch|B si ci sc cc E B ci ci si E|RADV_BC250_SPLIT_BATCH_PREP=1
# 64-triangle pieces (2 pieces), the Hellblade 2 setting
piece64|B si sc ci s1 E|RADV_BC250_PERF_PIECE_PRIMS=64
# autocull and the timer on
autocull|B si sc ci E|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1
timer|B si sc ci E|BC250_MESH_TIMER=1 BC250_MESH_TIMER_FAKE=1 BC250_MESH_TIMER_FILE=/dev/null
CASES
echo "draw-cost: $passed/$total passed"
exit $fail

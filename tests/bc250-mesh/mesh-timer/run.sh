#!/usr/bin/env bash
# BC250_MESH_TIMER offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; the harness also refuses any other device. Each case compiles
# a Mesh shape with the the base driver launcher policy (post-Mesh VGT_FLUSH off, as HB2 runs), records (timer.c)
# a direct Mesh draw, a VS draw, an indirect Mesh draw (2 records), an indirect-count Mesh draw and a
# second direct Mesh draw in one command buffer, submits twice to the shim and checks the IB
# (RADV_DEBUG=dumpibs) with the timer on (BC250_MESH_TIMER=1, fake CPU timestamps, print every
# submit, file output) and off (see check.py):
#   on:   one begin/end timestamp pair per command buffer, exactly one timestamp before and one
#         after every top-level Mesh draw (RELEASE_MEM BOTTOM_OF_PIPE_TS SEND_GPU_CLOCK_COUNTER into
#         the timer ring), split pieces / Task-replay chunks / prep dispatches inside the pair, the
#         VS draw outside, nothing else changed (packet stream equal to off modulo addresses), the
#         class line's route, the file's I/C lines with the fake sums;
#   SKIP: the matching Mesh draws and their timestamps are gone, the VS draw and the command buffer
#         pair remain, "skipping" printed once, K line in the file;
#   SAMPLE=2: every other Mesh draw timed, all counted;
#   off:  no timer packets; with OLDICD=<icd of the build before the switch existed> the off runs
#         (and BC250_MESH_TIMER=0 with the other timer variables set) are byte-identical to that build.
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] [OLDICD=<icd>] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-mesh-timer.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1 RADV_DEBUG=dumpibs,nocache)

shim() {
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_FL1 \
    -u RADV_BC250_MESH_MERGE -u RADV_BC250_MESH_AMD -u BC250_MESH_TIMER -u BC250_MESH_TIMER_SKIP \
    -u BC250_MESH_TIMER_SAMPLE -u BC250_MESH_TIMER_FILE -u BC250_MESH_TIMER_SUBMITS -u BC250_MESH_TIMER_COMPUTE \
    -u RADV_BC250_COMPUTE_QUEUE_COUNT -u RADV_BC250_COMPUTE_QUEUE_PRIORITY \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/timer" "$here/timer.c" -lvulkan || exit 2
glslangValidator -S vert --target-env vulkan1.3 -o "$work/tri_vert.spv" "$here/../mesh-merge/tri.vert" >/dev/null || exit 2
glslangValidator -S frag --target-env vulkan1.3 -o "$work/tri_frag.spv" "$here/../mesh-merge/tri.frag" >/dev/null || exit 2
glslangValidator -S task --target-env vulkan1.3 -o "$work/task.spv" "$here/../mesh-autocull/shape.task" >/dev/null || exit 2

fail=0; passed=0; total=0
# name | glslang defines (mesh-autocull/shape.mesh) | grid x | harness options | expected brackets per
# command buffer | expected route | timer environment of the on run | mode (no/yes=skip/sample/secondary) | extra environment
while IFS='|' read -r name defs gx opts brackets route tenv skip extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  pp=0; case "$defs" in *PERPRIM=1*) pp=1;; esac
  glslangValidator -S mesh --target-env vulkan1.3 $defs -o "$work/$name.mesh.spv" "$here/../mesh-autocull/shape.mesh" >/dev/null || exit 2
  glslangValidator -S frag --target-env vulkan1.3 -DPERPRIM=$pp -o "$work/$name.frag.spv" "$here/../mesh-autocull/shape.frag" >/dev/null || exit 2
  rm -f "$work/$name.timer"
  # HASHPREFIX: the class hash of the p32 case (same shape, same build), first 6 hex digits
  case "$tenv" in *HASHPREFIX*) h=$(sed -n 's/^C hash=\([0-9a-f]\{6\}\).*/\1/p' "$work/p32.timer" | head -1); tenv=${tenv//HASHPREFIX/$h};; esac
  shim "$ICD" env $extra BC250_MESH_TIMER=1 BC250_MESH_TIMER_FAKE=1 BC250_MESH_TIMER_INTERVAL=0 \
    BC250_MESH_TIMER_FILE="$work/$name.timer" $tenv ./timer "$name.mesh.spv" "$name.frag.spv" $gx $opts \
    > "$work/$name.on.out" 2> "$work/$name.on.err"
  shim "$ICD" env $extra ./timer "$name.mesh.spv" "$name.frag.spv" $gx $opts > "$work/$name.off.out" 2> "$work/$name.off.err"
  total=$((total + 1))
  status=ok; detail=
  grep -q '^SUBMIT_OK 1' "$work/$name.on.out" && grep -q '^SUBMIT_OK 1' "$work/$name.off.out" &&
    grep -q '^DONE' "$work/$name.on.out" || { status=FAIL; detail="no-submit"; }
  sk=; [ "$skip" = yes ] && sk=skip; [ "$skip" = sample ] && sk=sample; [ "$skip" = secondary ] && sk=secondary
  res=$(python3 "$here/check.py" "$work/$name.on.err" "$work/$name.off.err" "$brackets" "$route" "$work/$name.timer" $sk) || status=FAIL
  if [ -n "$OLDICD" ]; then
    # the switch off: byte-identical to the old build, also with the other timer variables set
    shim "$OLDICD" env $extra ./timer "$name.mesh.spv" "$name.frag.spv" $gx $opts > "$work/$name.old.out" 2> "$work/$name.old.err"
    shim "$ICD" env $extra BC250_MESH_TIMER=0 BC250_MESH_TIMER_FAKE=1 BC250_MESH_TIMER_INTERVAL=0 \
      BC250_MESH_TIMER_FILE="$work/$name.timer0" $tenv ./timer "$name.mesh.spv" "$name.frag.spv" $gx $opts \
      > "$work/$name.off0.out" 2> "$work/$name.off0.err"
    if cmp -s "$work/$name.old.err" "$work/$name.off.err" && cmp -s "$work/$name.old.out" "$work/$name.off.out" &&
       cmp -s "$work/$name.old.err" "$work/$name.off0.err" && [ ! -e "$work/$name.timer0" ]; then
      res="$res off_identical_to_old_build=True"
    else
      res="$res off_identical_to_old_build=False"; status=FAIL
    fi
  fi
  [ "$status" = ok ] && passed=$((passed + 1)) || fail=1
  printf '%-16s status=%s %s\n   %s\n' "$name" "$status" "$detail" "$res"
done <<'CASES'
# the base driver's expanded route (HB2's small shape class: 32V/32P shared -> 96V/32P expanded, wave32, autocull)
p32|-DWS=32 -DNV=32 -DNP=32|10||4|expanded||no|
# the same with RADV_BC250_MESH_AUTOCULL=1 (route suffix)
p32_ac|-DWS=32 -DNV=32 -DNP=32|10||4|expanded+autocull||no|RADV_BC250_MESH_AUTOCULL=1
# private vertices, 48 triangles (2 waves, expanded, refused by the autocull policy)
p48|-DWS=32 -DNV=144 -DNP=48 -DPRIVATE=1|10||4|expanded||no|
# the base driver's split (a_ls32 class: 64V/124P > 85 -> pieces), the pieces expanded
split124|-DWS=32 -DNV=64 -DNP=124|10||4|split=2+expanded||no|
# the base driver's split of HB2's Nanite class (256V/128P LS128 + per-primitive attribute)
nanite|-DWS=128 -DNV=256 -DNP=128 -DPERPRIM=1|10||4|split=3+expanded||no|
# the base driver hybrid Task replay (producer dispatch + consumer draw inside the pair)
task32|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|task=task.spv|4|task-replay+expanded||no|
# raw route (RADV_BC250_MESH_FAST=1, 64V/98P > 64 triangles: no split, no expansion)
raw98|-DWS=64 -DNV=64 -DNP=98|10||4|amd-raw||no|RADV_BC250_MESH_FAST=1
# safe direct route: timer classing must separate the direct planner from amd-raw.
safe_direct|-DWS=32 -DNV=32 -DNP=32|10||4|safe-direct||no|RADV_BC250_MESH_SAFE_DIRECT=1 RADV_BC250_MESH_SAFE_PARALLEL=1
# secondary command buffer: the draws' pairs live in the secondary (IB2, not dumped), the begin/end pair in the primary
secondary|-DWS=32 -DNV=32 -DNP=32|10|secondary=1|4|expanded||secondary|
# BC250_MESH_TIMER_SAMPLE=2: every other Mesh draw timed, all four counted
sample2|-DWS=32 -DNV=32 -DNP=32|10||2|expanded|BC250_MESH_TIMER_SAMPLE=2|sample|
# BC250_MESH_TIMER_SKIP: all / class words / hash prefix; a non-matching word skips nothing
skip_all|-DWS=32 -DNV=32 -DNP=32|10||0|expanded|BC250_MESH_TIMER_SKIP=all|yes|
skip_expanded|-DWS=32 -DNV=32 -DNP=32|10||0|expanded|BC250_MESH_TIMER_SKIP=expanded|yes|
skip_split|-DWS=32 -DNV=64 -DNP=124|10||0|split=2+expanded|BC250_MESH_TIMER_SKIP=raw,split|yes|
skip_replay|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|task=task.spv|0|task-replay+expanded|BC250_MESH_TIMER_SKIP=replay|yes|
skip_autocull|-DWS=32 -DNV=32 -DNP=32|10||0|expanded+autocull|BC250_MESH_TIMER_SKIP=autocull|yes|RADV_BC250_MESH_AUTOCULL=1
skip_hash|-DWS=32 -DNV=32 -DNP=32|10||0|expanded|BC250_MESH_TIMER_SKIP=HASHPREFIX|yes|
skip_none|-DWS=32 -DNV=32 -DNP=32|10||4|expanded|BC250_MESH_TIMER_SKIP=raw,split,replay,merged,amd,autocull,0123456789ab|no|
skip_raw|-DWS=64 -DNV=64 -DNP=98|10||0|amd-raw|BC250_MESH_TIMER_SKIP=amd|yes|RADV_BC250_MESH_FAST=1
CASES

# ---- async compute (ace.c): graphics + two compute-family queues, timeline + binary semaphores -----------
# name | mode for checkace.py | expected compute queues | extra environment (on and off runs) | expected stderr text
glslangValidator -S comp --target-env vulkan1.3 -o "$work/ace.comp.spv" "$here/ace.comp" >/dev/null || exit 2
cc -O1 -Wall -o "$work/ace" "$here/ace.c" -lvulkan || exit 2
QUEUETL=${QUEUETL:-$here/queuetl.py}
while IFS='|' read -r name mode nq extra expect; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  rm -f "$work/$name.timer"
  tenv=; [ "$mode" = nocompute ] && tenv=BC250_MESH_TIMER_COMPUTE=0
  shim "$ICD" env $extra BC250_MESH_TIMER=1 BC250_MESH_TIMER_FAKE=1 BC250_MESH_TIMER_INTERVAL=0 BC250_MESH_TIMER_SUBMITS=1 \
    BC250_MESH_TIMER_FILE="$work/$name.timer" $tenv ./ace ace.comp.spv > "$work/$name.on.out" 2> "$work/$name.on.err"
  shim "$ICD" env $extra ./ace ace.comp.spv > "$work/$name.off.out" 2> "$work/$name.off.err"
  total=$((total + 1))
  status=ok
  res=$(python3 "$here/checkace.py" "$mode" "$work/$name.on.err" "$work/$name.off.err" "$work/$name.timer" \
        "$work/$name.on.out" "$QUEUETL" "$nq") || status=FAIL
  if [ -n "$expect" ]; then
    if grep -qF "$expect" "$work/$name.on.err" && grep -qF "$expect" "$work/$name.off.err"; then
      res="$res stderr_note=True"
    else
      res="$res stderr_note=False"; status=FAIL
    fi
  fi
  if [ -n "$OLDICD" ] && [ -z "$extra" ]; then
    shim "$OLDICD" ./ace ace.comp.spv > "$work/$name.old.out" 2> "$work/$name.old.err"
    shim "$ICD" env BC250_MESH_TIMER=0 BC250_MESH_TIMER_FAKE=1 BC250_MESH_TIMER_INTERVAL=0 BC250_MESH_TIMER_SUBMITS=1 \
      BC250_MESH_TIMER_COMPUTE=1 BC250_MESH_TIMER_FILE="$work/$name.timer0" ./ace ace.comp.spv \
      > "$work/$name.off0.out" 2> "$work/$name.off0.err"
    if cmp -s "$work/$name.old.err" "$work/$name.off.err" && cmp -s "$work/$name.old.out" "$work/$name.off.out" &&
       cmp -s "$work/$name.old.err" "$work/$name.off0.err" && [ ! -e "$work/$name.timer0" ]; then
      res="$res off_identical_to_old_build=True"
    else
      res="$res off_identical_to_old_build=False"; status=FAIL
    fi
  fi
  [ "$status" = ok ] && passed=$((passed + 1)) || fail=1
  printf '%-16s status=%s\n   %s\n' "$name" "$status" "$res"
done <<'ACE'
# timer on: GFX and COMPUTE command buffers bracketed, W/S/T/I lines, queuetl.py on the log
ace_compute|compute|2||
# BC250_MESH_TIMER_COMPUTE=0: compute command buffers untouched, W lines still for every queue
ace_nocompute|nocompute|2||
# RADV_BC250_COMPUTE_QUEUE_COUNT=1: the compute family exposes one queue (both compute submissions on 1.0)
ace_queues1|compute|1|RADV_BC250_COMPUTE_QUEUE_COUNT=1|
# RADV_BC250_COMPUTE_QUEUE_PRIORITY=high: compute queues get the high kernel context priority (the shim grants it)
ace_prio_high|compute|2|RADV_BC250_COMPUTE_QUEUE_PRIORITY=high|compute queues use kernel context priority 2
# an unknown value is ignored (printed), nothing else changes
ace_prio_bad|compute|2|RADV_BC250_COMPUTE_QUEUE_PRIORITY=fastest|RADV_BC250_COMPUTE_QUEUE_PRIORITY=fastest ignored
ACE
echo "mesh-timer: $passed/$total passed"
exit $fail

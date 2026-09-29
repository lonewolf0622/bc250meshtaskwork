#!/usr/bin/env bash
# submit-cost offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), ioctlspy.so preloaded before
# the noop amdgpu drm-shim, AMDGPU_GPU_ID=gfx1013; the programs refuse any other device.
#   1. semtest (IOCTLSPY_SYNCOBJ=1 syncobj model) under every switch and all of them together;
#      checksem.py checks the kernel requests (CS wait / signal entries, WAIT_FOR_SUBMIT flags, probe
#      ioctls, binary resets), wait-before-signal on the host and across queues, and host results.
#   2. With OLDICD: switches off, the ioctlspy logs of semtest and of a submitbench run (every CS chunk,
#      IB size, syncobj ioctl) are identical to that build's.
#   3. RADV_BC250_KEEP_IB_KB: submitbench with large command buffers creates fewer IB buffers per frame.
#   4. BC250_MESH_TIMER_ASYNC: the W and S lines equal the synchronous timer's (fake timestamps,
#      addresses and times masked), and the per-submission cost is lower.
#   5. RADV_BC250_SUBMIT_PROFILE: FINAL lines per queue with the expected call / CS counts, and the
#      BO line; RADV_BC250_SUBMIT_KNOWN_SIGNALS shows probes=0 probes_skipped=1.
#   6. RADV_BC250_LOCAL_BOS: no BO handles left in the CS BO lists of submitbench.
# usage: ICD=<radeon_devenv_icd json> [OLDICD=<icd>] [SHIM=<libamdgpu_noop_drm_shim.so>] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-submit-cost.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT
inc=$(cd "$here/../../../include" && pwd)

cc -O2 -Wall -shared -fPIC -I"$inc" -o "$work/ioctlspy.so" "$here/ioctlspy.c" -ldl -lpthread || exit 2
cc -O1 -Wall -o "$work/semtest" "$here/semtest.c" -lvulkan -ldl -lpthread || exit 2
cc -O2 -Wall -o "$work/submitbench" "$here/submitbench.c" -lvulkan -ldl -lpthread || exit 2
glslangValidator -S comp --target-env vulkan1.3 -o "$work/bench.comp.spv" "$here/bench.comp" >/dev/null || exit 2

SWITCHES=(-u RADV_BC250_SUBMIT_KNOWN_SIGNALS -u RADV_BC250_LOCAL_BOS -u RADV_BC250_KEEP_IB_KB -u RADV_BC250_SUBMIT_PROFILE
  -u RADV_BC250_SUBMIT_PROFILE_FILE -u BC250_MESH_TIMER -u BC250_MESH_TIMER_ASYNC -u BC250_MESH_TIMER_SUBMITS
  -u BC250_MESH_TIMER_FILE -u BC250_MESH_TIMER_FAKE -u BC250_MESH_TIMER_INTERVAL -u RADV_PERFTEST -u MESA_VK_ENABLE_SUBMIT_THREAD)
shim() {
  local icd=$1; shift
  timeout 120 nice -n 19 bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env "${SWITCHES[@]}" LD_PRELOAD="$work/ioctlspy.so $SHIM" AMDGPU_GPU_ID=gfx1013 \
    VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" MESA_SHADER_CACHE_DISABLE=true "$@"
}
fail=0; total=0; passed=0
report() { # name status detail
  total=$((total + 1)); [ "$2" = ok ] && passed=$((passed + 1)) || fail=1
  printf '%-28s status=%s %s\n' "$1" "$2" "${3:-}"
}

# 1. semantics per switch
ALL="RADV_BC250_SUBMIT_KNOWN_SIGNALS=1 RADV_BC250_LOCAL_BOS=1 RADV_BC250_KEEP_IB_KB=640 RADV_BC250_SUBMIT_PROFILE=1 BC250_MESH_TIMER=1 BC250_MESH_TIMER_ASYNC=1 BC250_MESH_TIMER_SUBMITS=1 BC250_MESH_TIMER_FAKE=1"
while IFS='|' read -r name mode extra envs; do
  [ -z "$name" ] && continue
  rm -f "$work/sem_$name.timer"
  shim "$ICD" env IOCTLSPY_SYNCOBJ=1 IOCTLSPY_LOG="$work/sem_$name.log" BC250_MESH_TIMER_FILE="$work/sem_$name.timer" $envs \
    ./semtest > "$work/sem_$name.out" 2>&1
  if python3 "$here/checksem.py" "$work/sem_$name.log" "$work/sem_$name.out" "$mode" $extra > "$work/sem_$name.check"; then
    report "sem_$name" ok "($(grep -c 'status=ok' "$work/sem_$name.check") scenarios)"
  else
    report "sem_$name" FAIL "$(grep -v 'status=ok' "$work/sem_$name.check" | head -3 | tr '\n' ' ')"
  fi
done <<CASES
off|probe||
known|known||RADV_BC250_SUBMIT_KNOWN_SIGNALS=1
local_bos|probe|--bos-below 1|RADV_BC250_LOCAL_BOS=1
keep_ib|probe||RADV_BC250_KEEP_IB_KB=640
profile|probe||RADV_BC250_SUBMIT_PROFILE=1
timer_async|probe||BC250_MESH_TIMER=1 BC250_MESH_TIMER_ASYNC=1 BC250_MESH_TIMER_SUBMITS=1 BC250_MESH_TIMER_FAKE=1
timer_sync|probe||BC250_MESH_TIMER=1 BC250_MESH_TIMER_SUBMITS=1 BC250_MESH_TIMER_FAKE=1
all|known|--bos-below 1|$ALL
CASES

# 2. off identical to the old build
if [ -n "$OLDICD" ]; then
  shim "$OLDICD" env IOCTLSPY_SYNCOBJ=1 IOCTLSPY_LOG="$work/sem_old.log" ./semtest > "$work/sem_old.out" 2>&1
  # identical line by line; inside the two wait-before-signal scenarios (threads) as multisets
  if python3 - "$work/sem_old.log" "$work/sem_off.log" <<'PY'
import sys
def parts(p):
    out, cur = [], None
    for l in open(p):
        if l.startswith('MARK begin wbs'):
            cur = []
            out.append(('seq', l))
            continue
        if l.startswith('MARK end wbs'):
            out.append(('set', tuple(sorted(cur))))
            cur = None
        if cur is not None:
            if not l.startswith('MARK delayed'):
                cur.append(l)
            continue
        out.append(('seq', l))
    return out
sys.exit(0 if parts(sys.argv[1]) == parts(sys.argv[2]) else 1)
PY
  then
    report off_semtest_identical_old ok
  else
    report off_semtest_identical_old FAIL "ioctl logs differ"
  fi
  shim "$OLDICD" env IOCTLSPY_LOG="$work/bench_old.log" ./submitbench bench.comp.spv frames=5 warmup=0 bigevery=8 > /dev/null 2>&1
  shim "$ICD" env IOCTLSPY_LOG="$work/bench_new.log" ./submitbench bench.comp.spv frames=5 warmup=0 bigevery=8 > /dev/null 2>&1
  if [ -s "$work/bench_old.log" ] && cmp -s "$work/bench_old.log" "$work/bench_new.log"; then
    report off_bench_identical_old ok "($(grep -c '^CS' "$work/bench_new.log") CS)"
  else
    report off_bench_identical_old FAIL "ioctl logs differ"
  fi
fi

# 3. KEEP_IB: fewer IB buffers per frame with large command buffers
for k in 0 640; do
  shim "$ICD" env RADV_BC250_SUBMIT_PROFILE=1 RADV_BC250_KEEP_IB_KB=$k ./submitbench bench.comp.spv frames=60 warmup=10 bigevery=8 \
    > "$work/keep$k.out" 2>&1
done
ib0=$(sed -n 's/.*PROFILE_BO FINAL.* ib_creates=\([0-9]*\) .*/\1/p' "$work/keep0.out")
ib1=$(sed -n 's/.*PROFILE_BO FINAL.* ib_creates=\([0-9]*\) .*/\1/p' "$work/keep640.out")
if grep -q '^DONE' "$work/keep640.out" && [ -n "$ib0" ] && [ -n "$ib1" ] && [ "$ib1" -lt $((ib0 / 4)) ]; then
  report keep_ib_fewer_ib_buffers ok "(ib_creates $ib0 -> $ib1 over 70 frames)"
else
  report keep_ib_fewer_ib_buffers FAIL "ib_creates $ib0 -> $ib1"
fi

# 4. timer async: same lines as the synchronous timer, cheaper submissions
for a in 0 1; do
  rm -f "$work/timer$a.log"
  shim "$ICD" env BC250_MESH_TIMER=1 BC250_MESH_TIMER_FAKE=1 BC250_MESH_TIMER_SUBMITS=1 BC250_MESH_TIMER_ASYNC=$a \
    BC250_MESH_TIMER_FILE="$work/timer$a.log" ./submitbench bench.comp.spv frames=100 > "$work/timer$a.out" 2>&1
done
norm() { sed -E 's/[0-9a-f]{12}:/X:/g; s/(call_ns|wall_ns|gpu_t0|gpu_t1)=[0-9]+ //g' "$1" | grep -E '^(W|S) ' | sort; }
nw=$(grep -c '^W ' "$work/timer1.log"); ns=$(grep -c '^S ' "$work/timer1.log")
m0=$(sed -n 's/^QUEUE gfx .* median_us=\([0-9.]*\) .*/\1/p' "$work/timer0.out"); m1=$(sed -n 's/^QUEUE gfx .* median_us=\([0-9.]*\) .*/\1/p' "$work/timer1.out")
if cmp -s <(norm "$work/timer0.log") <(norm "$work/timer1.log") && [ "$nw" -gt 0 ] && [ "$ns" -gt 0 ] &&
   python3 -c "import sys; sys.exit(0 if float('$m1') < float('$m0') else 1)"; then
  report timer_async_same_lines ok "($nw W, $ns S; gfx submit median ${m0} -> ${m1} us)"
else
  report timer_async_same_lines FAIL "W=$nw S=$ns median $m0 -> $m1"
fi

# 5. profile lines; known signals skip the probes
rm -f "$work/prof.txt"
shim "$ICD" env RADV_BC250_SUBMIT_PROFILE=1 RADV_BC250_SUBMIT_KNOWN_SIGNALS=1 RADV_BC250_SUBMIT_PROFILE_FILE="$work/prof.txt" \
  ./submitbench bench.comp.spv frames=50 warmup=0 > "$work/prof.out" 2>&1
g=$(grep 'BC250_SUBMIT_PROFILE FINAL.* qf=gfx' "$work/prof.txt"); a=$(grep 'BC250_SUBMIT_PROFILE FINAL.* qf=ace' "$work/prof.txt")
if [ -n "$g" ] && [ -n "$a" ] && grep -q 'PROFILE_BO FINAL' "$work/prof.txt" &&
   echo "$g" | grep -q ' calls=2450 ' && echo "$a" | grep -q ' calls=500 ' &&
   echo "$a" | grep -q ' probes=0.00 probes_skipped=1.00 ' && echo "$g" | grep -q 'cs_per_call=0.98 '; then
  report profile_lines ok "(gfx 2450 calls = 48 + 1 fence-only per frame, ace 500)"
else
  report profile_lines FAIL "$(echo "$g" | cut -c1-160)"
fi

# 6. LOCAL_BOS: empty CS BO lists
shim "$ICD" env RADV_BC250_LOCAL_BOS=1 ./submitbench bench.comp.spv frames=30 warmup=5 > "$work/local.out" 2>&1
shim "$ICD" ./submitbench bench.comp.spv frames=30 warmup=5 > "$work/nolocal.out" 2>&1
b1=$(sed -n 's/^QUEUE gfx .* bo_handles_per_cs=\([0-9.]*\) .*/\1/p' "$work/local.out"); b0=$(sed -n 's/^QUEUE gfx .* bo_handles_per_cs=\([0-9.]*\) .*/\1/p' "$work/nolocal.out")
if [ "$b1" = "0.0" ] && [ -n "$b0" ] && [ "$b0" != "0.0" ]; then
  report local_bos_empty_lists ok "(BO handles per CS $b0 -> $b1)"
else
  report local_bos_empty_lists FAIL "$b0 -> $b1"
fi

echo "submit-cost: $passed/$total passed"
exit $fail

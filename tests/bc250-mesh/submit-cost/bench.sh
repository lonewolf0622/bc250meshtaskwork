#!/usr/bin/env bash
# Microbenchmark matrix: CPU cost of one vkQueueSubmit2 per switch (drm-shim, so ioctls cost ~0 and
# only the user-space part is timed; ioctls per submission are counted separately by ioctlspy).
# usage: ICD=<icd> [OLDICD=<icd>] [REPS=5] [OUT=<file>] ./bench.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
OLDICD=${OLDICD:-}
SHIM=${SHIM:?set SHIM}
REPS=${REPS:-5}
OUT=${OUT:-/dev/stdout}
work=$(mktemp -d "${TMPDIR:-/tmp}/bc250-submit-bench.XXXXXX")
trap 'rm -rf "$work"' EXIT
inc=$(cd "$here/../../../include" && pwd)
cc -O2 -Wall -shared -fPIC -I"$inc" -o "$work/ioctlspy.so" "$here/ioctlspy.c" -ldl -lpthread || exit 2
cc -O2 -Wall -o "$work/submitbench" "$here/submitbench.c" -lvulkan -ldl -lpthread || exit 2
glslangValidator -S comp --target-env vulkan1.3 -o "$work/bench.comp.spv" "$here/bench.comp" >/dev/null || exit 2

run() { # icd env... -- args...
  local icd=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  timeout 300 nice -n 19 bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env LD_PRELOAD="$work/ioctlspy.so $SHIM" AMDGPU_GPU_ID=gfx1013 \
    VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" MESA_SHADER_CACHE_DISABLE=true "${envs[@]}" \
    ./submitbench bench.comp.spv "$@" 2>&1
}

# config name | icd (new/old) | environment
CONFIGS=(
  "old_build|old|"
  "off|new|"
  "known_signals|new|RADV_BC250_SUBMIT_KNOWN_SIGNALS=1"
  "local_bos|new|RADV_BC250_LOCAL_BOS=1"
  "keep_ib_640|new|RADV_BC250_KEEP_IB_KB=640"
  "profile|new|RADV_BC250_SUBMIT_PROFILE=1"
  "timer_sync|new|BC250_MESH_TIMER=1 BC250_MESH_TIMER_SUBMITS=1 BC250_MESH_TIMER_FAKE=1 BC250_MESH_TIMER_FILE=$work/t.log"
  "timer_async|new|BC250_MESH_TIMER=1 BC250_MESH_TIMER_SUBMITS=1 BC250_MESH_TIMER_FAKE=1 BC250_MESH_TIMER_ASYNC=1 BC250_MESH_TIMER_FILE=$work/t.log"
  "all_no_timer|new|RADV_BC250_SUBMIT_KNOWN_SIGNALS=1 RADV_BC250_LOCAL_BOS=1 RADV_BC250_KEEP_IB_KB=640"
)
WORKLOADS=(
  "vkd3d|frames=300"
  "threads|frames=300 threads=1"
  "bigcb|frames=100 bigevery=8"
)
{
echo "# submitbench matrix $(date -u +%FT%TZ) reps=$REPS icd=$ICD oldicd=${OLDICD:-none}"
echo "# per config and workload: median over reps of the mean us per vkQueueSubmit2 (gfx / ace), gfx median,"
echo "# DRM ioctls per gfx submission (cs, probe), BO handles per gfx CS, graphics record+reset ms per frame"
printf '%-14s %-8s %9s %9s %9s %7s %7s %7s %9s\n' config workload gfx_mean ace_mean gfx_med ioctls probes bos rec_ms
for w in "${WORKLOADS[@]}"; do
  wn=${w%%|*}; wargs=${w#*|}
  for c in "${CONFIGS[@]}"; do
    cn=${c%%|*}; rest=${c#*|}; which=${rest%%|*}; envs=${rest#*|}
    icd=$ICD; [ "$which" = old ] && { [ -n "$OLDICD" ] || continue; icd=$OLDICD; }
    rm -f "$work"/res.*
    for r in $(seq "$REPS"); do
      rm -f "$work/t.log"
      # shellcheck disable=SC2086
      run "$icd" X=1 $envs -- $wargs > "$work/res.$r"
    done
    python3 - "$work" "$REPS" "$cn" "$wn" <<'PY'
import re, statistics, sys, glob
work, reps, cn, wn = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
vals = {k: [] for k in ('gm', 'am', 'gmed', 'io', 'pr', 'bos', 'rec')}
for f in sorted(glob.glob(work + '/res.*')):
    t = open(f).read()
    g = re.search(r'^QUEUE gfx .*', t, re.M); a = re.search(r'^QUEUE ace .*', t, re.M); c = re.search(r'^CONFIG .*', t, re.M)
    if not g or not c:
        continue
    kv = lambda s: dict(re.findall(r'(\w+)=([\d.]+)', s.group(0)))
    G, A, Cf = kv(g), kv(a) if a else {}, kv(c)
    vals['gm'].append(float(G['mean_us'])); vals['gmed'].append(float(G['median_us']))
    vals['am'].append(float(A.get('mean_us', 'nan'))); vals['io'].append(float(G['ioctls']))
    vals['pr'].append(float(G['probe_or_wait'])); vals['bos'].append(float(G['bo_handles_per_cs']))
    vals['rec'].append(float(Cf['record_ms_per_frame']))
m = {k: (statistics.median(v) if v else float('nan')) for k, v in vals.items()}
print(f"{cn:14s} {wn:8s} {m['gm']:9.2f} {m['am']:9.2f} {m['gmed']:9.2f} {m['io']:7.2f} {m['pr']:7.2f} {m['bos']:7.1f} {m['rec']:9.3f}")
PY
  done
done
} > "$OUT"

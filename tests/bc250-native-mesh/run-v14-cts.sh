#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$src/../.." && pwd)
out=${BC250_TEST_RESULTS:-$(mktemp -d "$src/cts-results-v14-XXXXXX")}
mkdir -p "$out"
export VK_DRIVER_FILES="$out/icd.json" MESA_SHADER_CACHE_DISABLE=true RADV_BC250_HYBRID_TASK=1 RADV_BC250_EXPAND_PRIMITIVES=1
unset VK_ICD_FILENAMES RADV_DEBUG
export BC250_DRIVER_LIBRARY="$root/build/src/amd/vulkan/libvulkan_radeon.so"
python3 - <<'PY'
import os,json
from pathlib import Path
Path(os.environ['VK_DRIVER_FILES']).write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':os.environ['BC250_DRIVER_LIBRARY'],'api_version':'1.4.0'}}))
PY
cts=${BC250_CTS:?set BC250_CTS to the deqp-vk binary}
sha256sum "$BC250_DRIVER_LIBRARY" "$cts" >"$out/SHA256SUMS"
run() {
 local name="dEQP-VK.mesh_shader.ext.$1" code=0
 printf '%s\n' "$name" >"$out/last-test"
 timeout --kill-after=5s 90s "$cts" --deqp-case="$name" --deqp-log-filename="$out/$1.qpa" --deqp-watchdog=enable >"$out/$1.log" 2>&1 || code=$?
 tail -15 "$out/$1.log"
 echo "$code" >"$out/$1.exit"
 test "$code" = 0
 grep -qE 'Passed: +1/1' "$out/$1.log"
}
for variant in 'get.wait.draw.64bit' 'copy.wait.indirect_draw.32bit' 'get.wait.indirect_with_count_draw.64bit'; do
 run "query.all_stats_query.triangles.no_reset.$variant.with_availability.single_block.task_mesh.include_rp.single_view.only_primary"
done
run query.all_stats_query.triangles.host_reset.copy.wait.draw.64bit.with_availability.single_block.task_mesh.include_rp.single_view.with_secondary
for name in per_prim_block_output custom_attributes custom_attributes_and_task_shader; do run "misc.$name"; done
printf 'PASS v14 selected CTS cases\n' | tee "$out/result.txt"

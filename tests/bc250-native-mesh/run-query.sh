#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# BC250 pipeline-statistics query tests (v8 development).
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mesa_root=${BC250_MESA_ROOT:-$(cd "$source_dir/../.." && pwd)}
run_dir=${BC250_TEST_RESULTS:-$(mktemp -d "$source_dir/query-results-v14-XXXXXX")}
mkdir -p "$run_dir"
cd "$run_dir"
export BC250_DRIVER_LIBRARY="$mesa_root/build/src/amd/vulkan/libvulkan_radeon.so"
test -f "$BC250_DRIVER_LIBRARY"
python3 - <<'PY'
import json, os
from pathlib import Path
Path('icd.json').write_text(json.dumps({'file_format_version':'1.0.0','ICD':{
    'library_path':os.environ['BC250_DRIVER_LIBRARY'],'api_version':'1.4.0'}})+'\n')
PY
export VK_DRIVER_FILES="$run_dir/icd.json" MESA_SHADER_CACHE_DISABLE=true
unset VK_ICD_FILENAMES RADV_BC250_HYBRID_TASK RADV_BC250_EXPAND_PRIMITIVES RADV_DEBUG BC250_TRACE_REGS

cc -Wall -Wextra -O2 -g -I"$mesa_root/include" "$source_dir/query_test.c" -l:libvulkan.so.1 -o query-test
compile() {
  glslangValidator -V --target-env vulkan1.3 "$@"
  local output=${!#}
  spirv-val --target-env vulkan1.3 "$output"
  printf '%q ' "$@" >>compile-commands.txt; printf '\n' >>compile-commands.txt
}
compile "$source_dir/qvert.vert" -o qvert.spv
compile "$source_dir/qfrag.frag" -o qfrag.spv
compile "$source_dir/qtask.task" -o qtask.spv
compile "$source_dir/qtask_cond.task" -o qtask_cond.spv
compile "$source_dir/qmesh.mesh" -o qmesh.spv

# RADV_BC250_HYBRID_TASK=1 is required for the task+mesh pipeline to take the
# BC250 producer path (matching run-hybrid.sh).
export RADV_BC250_HYBRID_TASK=1

run() {
  local label=$1 mode=$2
  printf '%s' "$label" >last-test
  printf "preparing %s\n" "$label" >phase
  sync last-test
  mkdir -p "cases/$label"
  cp ./*.spv icd.json compile-commands.txt "cases/$label/"
  sha256sum ./*.spv "$BC250_DRIVER_LIBRARY" >"cases/$label/SHA256SUMS"
  local code=0
  printf "launching %s\n" "$label" >phase
  timeout --kill-after=5s 60s ./query-test "$mode" >"$label.log" 2>&1 || code=$?
  printf "%s\n" "$code" >"$label.exit"
  cat "$label.log"
  if test "$code" = 3; then echo "UNTESTED $label (feature not exposed)"; return 3; fi
  test "$code" = 0
  grep -q "^PASS $mode$" "$label.log"
}

# hw baseline must always run and pass (it validates the HW PIPELINESTAT path).
run query-hw hw
if test "${1:-all}" = all || test "${1:-all}" = combined; then run query-combined combined; fi
# probe is NOT part of "all": on BC250 it deterministically wedges the GPU
# (HW pipeline-statistics region around a task/mesh draw, with the device
# created without meshShaderQueries). Reproduced 2026-09-12 on v7 and v9.
# Run it explicitly only, with monitoring.
if test "${1:-all}" = probe; then run query-probe probe; fi
if test "${1:-all}" = all || test "${1:-all}" = mesh; then run query-mesh mesh; fi
if test "${1:-all}" = all || test "${1:-all}" = task; then run query-task task; fi
if test "${1:-all}" = all || test "${1:-all}" = both; then run query-both both; fi
if test "${1:-all}" = all || test "${1:-all}" = chunked; then run query-chunked chunked; fi
if test "${1:-all}" = all || test "${1:-all}" = indirect; then run query-indirect indirect; fi
# v10: indirect records above one 4096-group chunk are split into GPU-side
# chunks by the setup shader (previously dropped); conditional emission must
# not draw stale scratch between records.
if test "${1:-all}" = all || test "${1:-all}" = indirect-large; then run query-indirect-large indirect-large; fi
if test "${1:-all}" = all || test "${1:-all}" = indirect-cond; then run query-indirect-cond indirect-cond; fi
if test "${1:-all}" = all || test "${1:-all}" = zero; then run query-zero zero; fi
if test "${1:-all}" = all || test "${1:-all}" = seq; then run query-seq seq; fi
if test "${1:-all}" = all || test "${1:-all}" = primgen; then run query-primgen primgen; fi
# v11: vkResetQueryPoolState + resubmit semantics (mixed HW+TASK+MESH pool and
# standalone primgen pool), and prim-gen accounting across indirect sub-record
# expansion. (This Vulkan generation has no standalone TASK/MESH invocation
# query types; those counts are only queryable via PIPELINE_STATISTICS bits.)
if test "${1:-all}" = all || test "${1:-all}" = reset; then run query-reset reset; fi
if test "${1:-all}" = all || test "${1:-all}" = primgen-large; then run query-primgen-large primgen-large; fi
echo ALL_QUERY_TESTS_DONE

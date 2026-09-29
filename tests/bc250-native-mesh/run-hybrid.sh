#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mesa_root=${BC250_MESA_ROOT:-$(cd "$source_dir/../.." && pwd)}
run_dir=${BC250_TEST_RESULTS:-$(mktemp -d "$source_dir/build-results-v14-XXXXXX")}
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
cc -Wall -Wextra -O2 -g -I"$mesa_root/include" "$source_dir/mesh_smoke.c" -l:libvulkan.so.1 -o mesh-smoke
compile() {
  glslangValidator -V --target-env vulkan1.3 "$@"
  local output=${!#}
  spirv-val --target-env vulkan1.3 "$output"
  printf '%q ' "$@" >>compile-commands.txt; printf '\n' >>compile-commands.txt
}
run() {
  local label=$1 mode=${2:-$1}
  printf '%s\n' "$label" >last-test
  printf "preparing %s\n" "$label" >phase
  sync last-test
  mkdir -p "cases/$label"
  cp ./*.spv icd.json compile-commands.txt "cases/$label/"
  sha256sum ./*.spv "$BC250_DRIVER_LIBRARY" >"cases/$label/SHA256SUMS"
  printf "launching %s\n" "$label" >phase
  local code=0
  timeout --kill-after=5s 30s ./mesh-smoke "$mode" >"$label.log" 2>&1 || code=$?
  printf "%s\n" "$code" >"$label.exit"
  test "$code" = 0
  cat "$label.log"
  grep -q '^PASS' "$label.log"
  if test -f last-frame.ppm; then cp last-frame.ppm "$label.ppm"; fi
}
reject() {
  local mode=$1 pattern=$2 code=0
  timeout --kill-after=5s 30s ./mesh-smoke "$mode" >"$mode.log" 2>&1 || code=$?
  cat "$mode.log"
  test "$code" = 2
  grep -q "$pattern.*VkResult -8" "$mode.log"
  ! grep -q SUBMIT "$mode.log"
}
compile "$source_dir/smoke.frag" -o frag.spv
compile "$source_dir/smoke.vert" -o vert.spv
compile "$source_dir/smoke.mesh" -o mesh.spv
run vertex
for mode in direct indirect indirect-count indirect-two; do run "native-$mode" "$mode"; done
# v13: gl_CullPrimitiveEXT — culled primitives must not rasterize while sibling
# primitives render normally (NGG null-primitive bit in the prim-exp argument).
compile "$source_dir/cull.mesh" -o mesh.spv
compile "$source_dir/cullred.frag" -o frag.spv
run cull-two
# Restore the standard fixtures: cullred.frag is a constant-red shader;
# leaving it in place would paint every later TASK case red and hide
# multi.mesh's red/green/blue diagnostic colors.
compile "$source_dir/smoke.mesh" -o mesh.spv
compile "$source_dir/smoke.frag" -o frag.spv
compile "$source_dir/smoke.task" -o task.spv
compile "$source_dir/task-smoke.mesh" -o task-mesh.spv
code=0
./mesh-smoke task-materialize >task-disabled.log 2>&1 || code=$?
test "$code" = 3
grep -q 'taskShader=0' task-disabled.log
export RADV_BC250_HYBRID_TASK=1
run task
compile "$source_dir/multi.task" -o task.spv
compile "$source_dir/multi.mesh" -o task-mesh.spv
for mode in task-two task-y task-z; do run "$mode"; done
compile -DCASE=2 "$source_dir/multi.task" -o task.spv
run task-skip
compile -DCASE=3 "$source_dir/multi.task" -o task.spv
compile -DCASE=3 "$source_dir/multi.mesh" -o task-mesh.spv
run task-fanout
compile "$source_dir/atomic.task" -o task.spv
compile "$source_dir/multi.mesh" -o task-mesh.spv
run task-atomic task-two
run task-repeat
compile "$source_dir/multi.task" -o task.spv
for size in 32 128; do
  compile -DCASE=4 -DLOCAL_SIZE=$size "$source_dir/multi.mesh" -o task-mesh.spv
  run "task-empty-mesh-$size" task-skip
  run "vertex-after-empty-$size" vertex
done
# v7: direct dispatches above the 4096 chunk bound are split into sequential
# producer/mesh sequences. Verify payload integrity across chunk boundaries.
compile "$source_dir/smoke.task" -o task.spv
compile "$source_dir/task-smoke.mesh" -o task-mesh.spv
run task-limit
reject task-draw-limit vkEndCommandBuffer
for mode in task-indirect task-indirect-count; do run "$mode"; done
compile "$source_dir/drawid.task" -o task.spv
compile -DCHECK_DRAW_ID=1 "$source_dir/multi.mesh" -o task-mesh.spv
for mode in task-indirect-two task-count-replay task-indirect-zero task-count-zero; do run "$mode"; done
compile "$source_dir/application-data.task" -o task.spv
compile "$source_dir/task-smoke.mesh" -o task-mesh.spv
run task-push
compile -DAPP_DESCRIPTOR=1 "$source_dir/application-data.task" -o task.spv
run task-descriptor
compile "$source_dir/resources.task" -o task.spv
compile "$source_dir/resources.mesh" -o task-mesh.spv
compile "$source_dir/resources.frag" -o frag.spv
compile "$source_dir/restore.comp" -o restore.spv
run task-resources
compile -DGPU_INDIRECT=1 "$source_dir/resources.task" -o task.spv
compile -DGPU_INDIRECT=1 "$source_dir/resources.mesh" -o task-mesh.spv
compile "$source_dir/generate-indirect.comp" -o generate-indirect.spv
run task-resources-indirect
compile "$source_dir/multi.task" -o task.spv
compile "$source_dir/perprimitive.mesh" -o task-mesh.spv
compile "$source_dir/perprimitive.frag" -o frag.spv
reject task-two vkEndCommandBuffer
compile "$source_dir/smoke.frag" -o frag.spv
run vertex-after-rejected-perprimitive vertex
printf '%s\n' 'PASS: BC250 native mesh and experimental hybrid TASK suite' | tee result.txt

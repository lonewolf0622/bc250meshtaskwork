#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mesa_root=${BC250_MESA_ROOT:-$(cd "$source_dir/../.." && pwd)}
run_dir=${BC250_TEST_RESULTS:-$(mktemp -d "$source_dir/expand-results-v14-XXXXXX")}
mkdir -p "$run_dir"
cd "$run_dir"
export BC250_DRIVER_LIBRARY="$mesa_root/build/src/amd/vulkan/libvulkan_radeon.so"
python3 - <<'JSON'
import json, os
from pathlib import Path
Path('icd.json').write_text(json.dumps({'file_format_version':'1.0.0','ICD':{
    'library_path':os.environ['BC250_DRIVER_LIBRARY'],'api_version':'1.4.0'}})+'\n')
JSON
export VK_DRIVER_FILES="$run_dir/icd.json" MESA_SHADER_CACHE_DISABLE=true
export RADV_BC250_EXPAND_PRIMITIVES=1 RADV_BC250_HYBRID_TASK=1
unset RADV_DEBUG
cc -Wall -Wextra -O2 -g -I"$mesa_root/include" "$source_dir/mesh_smoke.c" -l:libvulkan.so.1 -o mesh-smoke
compile() {
  glslangValidator -V --target-env vulkan1.3 "$@"
  local output=${!#}
  spirv-val --target-env vulkan1.3 "$output"
  printf '%q ' "$@" >>compile-commands.txt; printf '\n' >>compile-commands.txt
}
run() {
 local label=$1 mode=$2
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
 cp last-frame.ppm "$label.ppm"
}
control() {
 compile "$source_dir/smoke.frag" -o frag.spv
 run "$1" vertex
}
compile "$source_dir/smoke.vert" -o vert.spv
control vertex-before
for size in 32 128; do
 for sparse in 0 1; do
  for groups in 1 2; do
   label="native-$size-$sparse-$groups"
   mode=native-colors
   if test "$groups" = 2; then mode=native-colors-two; fi
   compile -DLOCAL_SIZE=$size -DSPARSE=$sparse -DGROUPS=$groups "$source_dir/expand.mesh" -o mesh.spv
   compile "$source_dir/expand.frag" -o frag.spv
   run "$label" "$mode"
   # Use a compact hand-deindexed reference to avoid the unresolved original
   # 256-output sizing path; the image must be bit-identical.
   compile -DREFERENCE=1 -DLOCAL_SIZE=$size -DGROUPS=$groups "$source_dir/expand.mesh" -o mesh.spv
   compile -DREFERENCE=1 "$source_dir/expand.frag" -o frag.spv
   run "$label-reference" "$mode"
   cmp "$label.ppm" "$label-reference.ppm"
   control "$label-control"
  done
 done
done
compile "$source_dir/multi.task" -o task.spv
for size in 32 128; do
 compile -DLOCAL_SIZE=$size "$source_dir/perprimitive.mesh" -o task-mesh.spv
 compile "$source_dir/perprimitive.frag" -o frag.spv
 for mode in task-two task-y task-z task-repeat; do run "expanded-$size-$mode" "$mode"; done
 compile -DLOCAL_SIZE=$size -DCASE=4 "$source_dir/perprimitive.mesh" -o task-mesh.spv
 run "expanded-empty-$size" task-skip
 control "vertex-after-task-$size"
done
reject() {
 local label=$1 code=0
 timeout --kill-after=5s 30s ./mesh-smoke native-colors >"$label.log" 2>&1 || code=$?
 cat "$label.log"
 test "$code" = 2
 # Clean rejection with VkResult -8 and no submit, whether raised at pipeline
 # creation (required-but-skipped expansion) or command-buffer time.
 grep -qE '(vkEndCommandBuffer|vkCreateGraphicsPipelines).*VkResult -8' "$label.log"
 ! grep -q SUBMIT "$label.log"
}
# Same as reject(), but for pipelines rejected at CREATION (vkCreateGraphicsPipelines)
# rather than at command-buffer time. Used when expansion is required-but-skipped
# (e.g. over the staging budget): the driver now rejects early with VkResult -8 instead
# of leaving raw per-primitive output intrinsics that crash a later NIR pass.
reject_pipeline() {
 local label=$1 code=0
 timeout --kill-after=5s 30s ./mesh-smoke native-colors >"$label.log" 2>&1 || code=$?
 cat "$label.log"
 test "$code" = 2
 grep -q 'vkCreateGraphicsPipelines.*VkResult -8' "$label.log"
 ! grep -q SUBMIT "$label.log"
}
compile -DMAX_PRIMITIVES=86 "$source_dir/expand.mesh" -o mesh.spv
compile "$source_dir/expand.frag" -o frag.spv
reject expanded-limit
compile "$source_dir/expand.mesh" -o mesh.spv
export MESA_SHADER_CACHE_DISABLE=false MESA_SHADER_CACHE_DIR="$run_dir/cache-$(date +%s)"
run cache-enabled-first native-colors
run cache-enabled-second native-colors
cmp cache-enabled-first.ppm cache-enabled-second.ppm
RADV_BC250_EXPAND_PRIMITIVES=0 reject cache-disabled-guard
run cache-reenabled native-colors
cmp cache-enabled-first.ppm cache-reenabled.ppm
control vertex-after-cache
# --- v12: expansion coverage (layer routing, non-triangle topologies, mesh
# shared memory alongside staging, staging budget guard boundary) -----------
compile "$source_dir/layer.mesh" -o mesh.spv
compile "$source_dir/expand.frag" -o frag.spv
run layer-two layer-two
control vertex-after-layer

compile "$source_dir/lines.mesh" -o mesh.spv
compile "$source_dir/expand.frag" -o frag.spv
run lines-two lines-two
control vertex-after-lines

compile "$source_dir/points.mesh" -o mesh.spv
compile "$source_dir/expand.frag" -o frag.spv
run points-four points-four
control vertex-after-points

# Mesh shared memory (4 KiB scratch) used alongside the staging arrays.
compile -DLOCAL_SIZE=32 "$source_dir/sharedstaging.mesh" -o mesh.spv
compile "$source_dir/expand.frag" -o frag.spv
run shared-staging native-colors
control vertex-after-shared

# Staging budget guard: 16 KiB scratch pushes staging_bytes over the limit, so
# expansion is skipped and the pipeline must be rejected cleanly at creation
# (VkResult -8 / VK_ERROR_FEATURE_NOT_PRESENT, no submit) instead of crashing a
# later NIR pass on raw per-primitive output intrinsics.
compile -DLOCAL_SIZE=32 -DSCRATCH=4096 "$source_dir/sharedstaging.mesh" -o mesh.spv
reject_pipeline staging-limit

printf '%s\n' 'PASS: BC250 expanded primitive attribute suite' | tee result.txt

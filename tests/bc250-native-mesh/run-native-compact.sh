#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mesa_root=${BC250_MESA_ROOT:-$(cd "$source_dir/../.." && pwd)}
run_dir=${BC250_TEST_RESULTS:-$(mktemp -d "$source_dir/compact-results-v14-XXXXXX")}
mkdir -p "$run_dir"
cd "$run_dir"
export BC250_DRIVER_LIBRARY="$mesa_root/build/src/amd/vulkan/libvulkan_radeon.so"
python3 - <<'JSON'
import json, os
from pathlib import Path
Path('icd.json').write_text(json.dumps({'file_format_version':'1.0.0','ICD':{
 'library_path':os.environ['BC250_DRIVER_LIBRARY'],'api_version':'1.4.0'}})+'\n')
JSON
export VK_DRIVER_FILES="$run_dir/icd.json" MESA_SHADER_CACHE_DISABLE=true RADV_BC250_EXPAND_PRIMITIVES=1
unset RADV_DEBUG RADV_PERFTEST RADV_BC250_HYBRID_TASK
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
 timeout --kill-after=5s 30 ./mesh-smoke "$mode" >"$label.log" 2>&1 || code=$?
 printf "%s\n" "$code" >"$label.exit"
 test "$code" = 0
 cat "$label.log"
 grep -q '^PASS' "$label.log"
 cp last-frame.ppm "$label.ppm"
}
compile "$source_dir/smoke.vert" -o vert.spv
compile "$source_dir/smoke.frag" -o frag.spv
run vertex-before vertex
for size in 32 64 128; do
 for groups in 1 2; do
  label="compact-$size-$groups"
  mode=native-colors
  if test "$groups" = 2; then mode=native-colors-two; fi
  compile -DREFERENCE=1 -DLOCAL_SIZE=$size -DSPARSE=1 -DGROUPS=$groups "$source_dir/expand.mesh" -o mesh.spv
  compile -DREFERENCE=1 "$source_dir/expand.frag" -o frag.spv
  run "$label" "$mode"
  compile -DREFERENCE=1 -DLOCAL_SIZE=$size -DGROUPS=$groups "$source_dir/expand.mesh" -o mesh.spv
  run "$label-reference" "$mode"
  cmp "$label.ppm" "$label-reference.ppm"
  compile "$source_dir/smoke.frag" -o frag.spv
  run "$label-control" vertex
 done
done
printf '%s\n' 'PASS: BC250 native vertex-output compaction suite' | tee result.txt

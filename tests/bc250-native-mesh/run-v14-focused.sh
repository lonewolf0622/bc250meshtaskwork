#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$src/../.." && pwd)
out=${BC250_TEST_RESULTS:-$(mktemp -d "$src/focused-results-v14-XXXXXX")}
mkdir -p "$out"
export VK_DRIVER_FILES="$out/icd.json" MESA_SHADER_CACHE_DISABLE=true RADV_BC250_HYBRID_TASK=1
unset VK_ICD_FILENAMES RADV_DEBUG RADV_BC250_EXPAND_PRIMITIVES
export BC250_DRIVER_LIBRARY="$root/build/src/amd/vulkan/libvulkan_radeon.so"
python3 - <<'PY'
import os,json
from pathlib import Path
Path(os.environ['VK_DRIVER_FILES']).write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':os.environ['BC250_DRIVER_LIBRARY'],'api_version':'1.4.0'}}))
PY
compile() { glslangValidator -V --target-env vulkan1.3 "$@"; spirv-val --target-env vulkan1.3 "${!#}"; printf '%q ' "$@" >>compile-commands.txt; printf '\n' >>compile-commands.txt; }
case_begin() { mkdir "$out/$1"; cd "$out/$1"; cp "$out/icd.json" .; }
run() {
 sha256sum ./*.spv "$BC250_DRIVER_LIBRARY" >SHA256SUMS
 env | sort | rg '^(VK_|RADV_|MESA_|BC250_)' >environment.txt
 printf '%s\n' "$PWD" >"$out/last-test"
 local code=0
 timeout --kill-after=5s 60s "$@" >result.log 2>&1 || code=$?
 cat result.log
 echo "$code" >exit-code
 test "$code" = 0
 grep -q '^PASS' result.log
}
for shape in '32 1 1' '2 4 4' '1 4 6'; do
 read -r x y z <<<"$shape"
 case_begin "local-$x-$y-$z"
 cc -Wall -Wextra -O2 -g -I"$root/include" -DTEST_TASK_INVOCATIONS=$((x*y*z)) -DTEST_MESH_INVOCATIONS=$((x*y*z)) "$src/query_test.c" -l:libvulkan.so.1 -o query-test
 compile "$src/qtask.task" -DLOCAL_X=$x -DLOCAL_Y=$y -DLOCAL_Z=$z -o qtask.spv
 compile "$src/qmesh.mesh" -DLOCAL_X=$x -DLOCAL_Y=$y -DLOCAL_Z=$z -o qmesh.spv
 compile "$src/qfrag.frag" -o qfrag.spv
 run ./query-test both
done
for axis in 0 1 2; do
 for indirect in 0 1 2; do
  case_begin "boundary-$axis-$indirect"
  cc -Wall -Wextra -O2 -g -I"$root/include" -DTEST_BOUNDARY_AXIS=$axis -DTEST_BOUNDARY_INDIRECT=$indirect -DTEST_TASK_INVOCATIONS=32 -DTEST_MESH_INVOCATIONS=32 "$src/query_test.c" -l:libvulkan.so.1 -o query-test
  compile "$src/boundary.task" -o qtask.spv
  compile "$src/boundary.mesh" -DAXIS=$axis -o qmesh.spv
  compile "$src/qfrag.frag" -o qfrag.spv
  run ./query-test both
 done
done
case_begin cull-expanded
cc -Wall -Wextra -O2 -g -I"$root/include" "$src/mesh_smoke.c" -l:libvulkan.so.1 -o mesh-smoke
compile "$src/cull-expanded.mesh" -o mesh.spv
compile "$src/expand.frag" -o frag.spv
export RADV_BC250_EXPAND_PRIMITIVES=1
run ./mesh-smoke cull-two
printf 'PASS v14 focused suite\n' | tee "$out/result.txt"

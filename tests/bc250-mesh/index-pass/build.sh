#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
for f in *.mesh *.vert *.frag; do
  glslangValidator -V --target-env vulkan1.3 "$f" -o "$f.spv"
  spirv-val --target-env vulkan1.3 "$f.spv"
done
cc -O2 index_gate.c -lvulkan -o index_gate

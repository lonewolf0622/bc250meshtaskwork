#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
for f in *.mesh *.vert *.frag; do
  glslangValidator -V --target-env vulkan1.3 "$f" -o "$f.spv"
  spirv-val --target-env vulkan1.3 "$f.spv"
done
for variant in heavy64 heavy_pp128 heavy_ballot64; do
  defs=(-DVERTICES=64 -DPER_PRIMITIVE=0 -DUSE_BALLOT=0)
  if [ "$variant" = heavy_pp128 ]; then defs=(-DVERTICES=128 -DPER_PRIMITIVE=1 -DUSE_BALLOT=0); fi
  if [ "$variant" = heavy_ballot64 ]; then defs=(-DVERTICES=64 -DPER_PRIMITIVE=0 -DUSE_BALLOT=1); fi
  for stage in mesh vert; do
    glslangValidator -V --target-env vulkan1.3 "${defs[@]}" "heavy.$stage" -o "$variant.$stage.spv"
    spirv-val --target-env vulkan1.3 "$variant.$stage.spv"
    glslangValidator -V --target-env vulkan1.3 "${defs[@]}" -DFLAT_ORACLE=1 "heavy.$stage" -o "${variant}_flat.$stage.spv"
    spirv-val --target-env vulkan1.3 "${variant}_flat.$stage.spv"
  done
done
for variant in prune64 prune_pp128 prune_ballot64 prune_deadpp64; do
  defs=(-DVERTICES=64 -DPER_PRIMITIVE=0 -DUSE_BALLOT=0 -DDEAD_VERTEX_OUTPUTS=12 -DDEAD_PRIMITIVE_OUTPUTS=4)
  if [ "$variant" = prune_pp128 ]; then
    defs=(-DVERTICES=128 -DPER_PRIMITIVE=1 -DUSE_BALLOT=0 -DDEAD_VERTEX_OUTPUTS=4 -DDEAD_PRIMITIVE_OUTPUTS=2)
  elif [ "$variant" = prune_ballot64 ]; then defs[2]=-DUSE_BALLOT=1
  elif [ "$variant" = prune_deadpp64 ]; then
    defs=(-DVERTICES=64 -DPER_PRIMITIVE=1 -DUSE_BALLOT=0 -DDEAD_VERTEX_OUTPUTS=0 -DDEAD_PRIMITIVE_OUTPUTS=0)
  fi
  for stage in mesh vert; do
    glslangValidator -V --target-env vulkan1.3 "${defs[@]}" "heavy.$stage" -o "$variant.$stage.spv"
    spirv-val --target-env vulkan1.3 "$variant.$stage.spv"
    glslangValidator -V --target-env vulkan1.3 "${defs[@]}" -DFLAT_ORACLE=1 "heavy.$stage" -o "${variant}_flat.$stage.spv"
    spirv-val --target-env vulkan1.3 "${variant}_flat.$stage.spv"
  done
done
for variant in profile64 profile128p64 profile128p128 profile256pp128; do
  defs=(-DVERTICES=128 -DPRIMITIVES=128 -DTHREADS=128 -DPER_PRIMITIVE=0 -DCULL_DISTANCE=1)
  if [ "$variant" = profile64 ]; then defs=(-DVERTICES=64 -DPRIMITIVES=64 -DTHREADS=64 -DPER_PRIMITIVE=0 -DCULL_DISTANCE=0); fi
  if [ "$variant" = profile128p64 ]; then defs[1]=-DPRIMITIVES=64; fi
  if [ "$variant" = profile256pp128 ]; then
    defs=(-DVERTICES=256 -DPRIMITIVES=128 -DTHREADS=128 -DPER_PRIMITIVE=1 -DCULL_DISTANCE=0)
  fi
  for stage in mesh vert; do
    glslangValidator -V --target-env vulkan1.3 "${defs[@]}" "profile.$stage" -o "$variant.$stage.spv"
    spirv-val --target-env vulkan1.3 "$variant.$stage.spv"
    glslangValidator -V --target-env vulkan1.3 "${defs[@]}" -DFLAT_ORACLE=1 "profile.$stage" -o "${variant}_flat.$stage.spv"
    spirv-val --target-env vulkan1.3 "${variant}_flat.$stage.spv"
  done
done
for kind in plain pp pp_ref; do
  defs=(-DPER_PRIMITIVE=0 -DREFERENCE=0)
  if [ "$kind" = pp ]; then defs=(-DPER_PRIMITIVE=1 -DREFERENCE=0); fi
  if [ "$kind" = pp_ref ]; then defs=(-DPER_PRIMITIVE=1 -DREFERENCE=1); fi
  glslangValidator -V --target-env vulkan1.3 "${defs[@]}" heavy_flat.frag -o "heavy_flat_${kind}.frag.spv"
  spirv-val --target-env vulkan1.3 "heavy_flat_${kind}.frag.spv"
done
cc -O2 index_gate.c -lvulkan -o index_gate

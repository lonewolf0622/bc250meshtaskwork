#!/usr/bin/env bash
# Split admission regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with /dev/dri replaced by an empty
# tmpfs, the noop amdgpu drm-shim preloaded and AMDGPU_GPU_ID=gfx1013; pipe
# and mkpipe also refuse any device that is not the shim's GFX1013. Pipelines
# are created and a draw is recorded; nothing is ever submitted.
#
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] ./run.sh
# Prints one line per case and exits non-zero on any unexpected result.
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
work=$(mktemp -d "${TMPDIR:-/tmp}/bc250-split-admission.XXXXXX")
trap 'rm -rf "$work"' EXIT

# the base driver launcher policy (the same environment the game launcher sets).
POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=1
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1)

shim() {
  bwrap --dev-bind / / --tmpfs /dev/dri -- env -u RADV_DEBUG -u RADV_EXPERIMENTAL \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$ICD" VK_ICD_FILENAMES="$ICD" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -o "$work/pipe" "$here/pipe.c" -lvulkan || exit 2
cc -O1 -o "$work/mkpipe" "$here/mkpipe.c" -lvulkan || exit 2
for s in "$here"/*.mesh; do
  glslangValidator -S mesh --target-env vulkan1.3 -o "$work/$(basename "$s" .mesh).spv" "$s" >/dev/null || exit 2
done
glslangValidator -S task --target-env vulkan1.3 -o "$work/t.spv" "$here/t.task" >/dev/null || exit 2
glslangValidator -S frag --target-env vulkan1.3 -o "$work/f.spv" "$here/f.frag" >/dev/null || exit 2

fail=0
# name | mesh | task (- for none) | extra env | expected result | expected message (grep -F, or -)
while IFS='|' read -r name mesh task extra want msg; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  if [ "$task" = - ]; then
    out=$(cd "$work" && shim env $extra ./pipe "$mesh.spv" f.spv 2>&1)
    got=$(printf '%s\n' "$out" | sed -n 's/^PIPELINE_RESULT=//p' | tail -1)
  else
    out=$(cd "$work" && shim env $extra ./mkpipe "$mesh.spv" f.spv "$task.spv" 3 2>&1)
    got=$(printf '%s\n' "$out" | sed -n 's/^RESULT //p' | tail -1)
  fi
  why=$(printf '%s\n' "$out" | grep -o 'split rejected: .*' | head -1)
  layout=$(printf '%s\n' "$out" | grep -o 'split shared layout: .*' | head -1)
  status=ok
  [ "$got" = "$want" ] || status=UNEXPECTED
  if [ "$msg" != - ] && ! printf '%s\n' "$out" | grep -qF -- "$msg"; then status=UNEXPECTED; fi
  if printf '%s\n' "$out" | grep -qiE 'validation|assert|NIR_VALIDATE|error:'; then status=UNEXPECTED; fi
  [ "$status" = ok ] || fail=1
  printf '%-22s result=%-3s want=%-3s %s %s %s\n' "$name" "${got:-none}" "$want" "$status" "${why:-}" "${layout:-}"
done <<'EOF'
# control: plain split, no application shared memory
tri128|tri128|-||0|split applied: pieces=3
# class 1: application shared memory + barrier (isolated variants)
shared_cull64|shared_cull64|-||0|application=[0,256) split=[256,
shared_barrier256|shared_barrier256|-||0|application=[0,512) split=[512,
explicit_layout128|explicit_layout128|-||-8|split rejected: explicit shared memory layout
# class 2: NumWorkGroups
numwg128|numwg128|-||0|direct mesh-only split enabled: pieces=3
numwg_single|numwg_single|-||0|split applied: pieces=1
numwg_task|numwg_task|t||0|split applied: pieces=3
numwg128_amplifier|numwg128|-|BC250_EXPERIMENTAL_DIRECT_SPLIT=false|-8|load_num_workgroups (internal amplifier)
# class 3: write-only external atomics, first piece only
atomic_cull_shared64|atomic_cull_shared64|-||0|external atomics: 1 guarded to the first piece
atomic_used128|atomic_used128|-||-8|split rejected: external atomic result used
atomic_read128|atomic_read128|-||-8|split rejected: external atomic with external memory reads
# G1 probes (scratch_G1), unchanged sources
tri64_cull_shared|tri64_cull_shared|-||-8|split rejected: store_ssbo
tri256_barrier|tri256_barrier|-||-8|split rejected: store_ssbo
tri128_numwg|tri128_numwg|-||-8|split rejected: store_ssbo
tri128_atomic|tri128_atomic|-||0|external atomics: 1 guarded to the first piece
tri128_shatomic|tri128_shatomic|-||-8|split rejected: shared_atomic
EOF
exit $fail

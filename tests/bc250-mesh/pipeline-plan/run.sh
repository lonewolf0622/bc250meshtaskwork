#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
: "${ICD:?set ICD to the candidate ICD}"
SHIM=${SHIM:-/home/deck/mesh-test/mesa-bc250-claude-rdna1-auditfix-build/src/amd/drm-shim/libamdgpu_noop_drm_shim.so}
work=${KEEP:-$(mktemp -d)}
mkdir -p "$work"
work=$(realpath "$work")
exec nice -n 19 bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --unshare-pid \
  --bind "$work" "$work" --chdir "$work" --clearenv --setenv PATH /usr/bin --setenv HOME /tmp \
  --setenv LD_PRELOAD "$SHIM" --setenv AMDGPU_GPU_ID gfx1013 --setenv MESA_SHADER_CACHE_DISABLE 1 \
  --setenv PYTHONDONTWRITEBYTECODE 1 --setenv ICD "$ICD" /usr/bin/python3 "$here/run.py"

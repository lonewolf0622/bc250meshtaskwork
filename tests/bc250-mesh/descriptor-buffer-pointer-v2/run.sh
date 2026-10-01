#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
: "${ICD:?set ICD to the candidate ICD}"
: "${SHIM:?set SHIM to the gfx1013 noop drm-shim library}"
: "${POLICY:?set POLICY to the frozen policy JSON}"
work=${KEEP:-$(mktemp -d)}
mkdir -p "$work"
work=$(realpath "$work")
exec bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --unshare-pid \
 --bind "$work" "$work" --chdir "$work" --clearenv --setenv PATH /usr/bin --setenv HOME /tmp \
 --setenv ICD "$ICD" --setenv POLICY "$POLICY" --setenv LD_PRELOAD "$SHIM" \
 --setenv AMDGPU_GPU_ID gfx1013 --setenv PYTHONDONTWRITEBYTECODE 1 python3 "$here/run.py"

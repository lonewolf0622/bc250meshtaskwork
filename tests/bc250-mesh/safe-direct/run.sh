#!/usr/bin/env bash
# Whole regression runs inside bwrap/noop; no GPU mode.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
: "${BUILD:?set BUILD to the Mesa build used for the CPU oracle}"
: "${ICD:?set ICD to the candidate frozen ICD}"
SHIM=${SHIM:?set SHIM}
work=${KEEP:-$(mktemp -d)}
mkdir -p "$work"
work=$(realpath "$work")
[ -n "${KEEP:-}" ] || trap 'rm -rf "$work"' EXIT
exec bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --unshare-pid  --bind "$work" "$work" --chdir "$work" --clearenv  --setenv PATH /usr/bin --setenv HOME /tmp --setenv LD_PRELOAD "$SHIM"  --setenv AMDGPU_GPU_ID gfx1013 --setenv MESA_SHADER_CACHE_DISABLE 1  --setenv PYTHONDONTWRITEBYTECODE 1 --setenv BUILD "$BUILD" --setenv ICD "$ICD"  --setenv OLDICD "${OLDICD:-$ICD}" --setenv KEEP "$work"  --setenv VARIANT "${VARIANT:-serial}" --setenv SEEDS "${SEEDS:-64}" --setenv ONLY "${ONLY:-}"  /usr/bin/python3 "$here/run.py"

#!/usr/bin/env bash
# Portable release profile; the bundled, hardware-validated driver is unchanged.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
profile=standard
print_env=0
while (($#)); do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo 'Missing profile' >&2; exit 2; }
      profile=$2
      shift 2
      ;;
    --print-env) print_env=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
case "$profile" in
  standard)
    : "${RADV_BC250_MESH_IDXPASS=1}"
    : "${RADV_BC250_MESH_IDXPASS_STORE=0}"
    ;;
  mesh)
    : "${RADV_BC250_MESH_IDXPASS=0}"
    : "${RADV_BC250_MESH_IDXPASS_STORE=0}"
    ;;
  stored)
    : "${RADV_BC250_MESH_IDXPASS=1}"
    : "${RADV_BC250_MESH_IDXPASS_STORE=1}"
    ;;
  *) echo "Unknown profile: $profile (choose standard, mesh or stored)" >&2; exit 2 ;;
esac
[[ -f "$here/radeon_icd.json" ]] || { echo 'Missing bundled radeon_icd.json' >&2; exit 2; }
: "${VK_DRIVER_FILES=$here/radeon_icd.json}"
: "${VK_ICD_FILENAMES=$VK_DRIVER_FILES}"
: "${RADV_DIRECTMESH=1}"
: "${RADV_BC250_MESH_IDXPASS_SETUP_BATCH=1}"
: "${RADV_BC250_MESH_IDXPASS_BATCH_TRIM=1}"
export VK_DRIVER_FILES VK_ICD_FILENAMES RADV_DIRECTMESH RADV_BC250_MESH_IDXPASS
export RADV_BC250_MESH_IDXPASS_STORE RADV_BC250_MESH_IDXPASS_SETUP_BATCH RADV_BC250_MESH_IDXPASS_BATCH_TRIM
if ((print_env)); then
  for name in VK_DRIVER_FILES VK_ICD_FILENAMES RADV_DIRECTMESH RADV_BC250_MESH_IDXPASS \
              RADV_BC250_MESH_IDXPASS_STORE RADV_BC250_MESH_IDXPASS_SETUP_BATCH RADV_BC250_MESH_IDXPASS_BATCH_TRIM; do
    printf '%s=%q\n' "$name" "${!name}"
  done
  exit 0
fi
(($#)) || { echo 'Usage: bash launch.sh [--profile standard|mesh|stored] [--print-env] -- command [args...]' >&2; exit 2; }
exec "$@"

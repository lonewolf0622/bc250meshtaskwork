#!/usr/bin/env bash
# RADV_BC250_MESH_MERGE offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; the harness also refuses any other device. Each case compiles
# a Mesh shape with the the base driver launcher policy (post-Mesh VGT_FLUSH off, as HB2 runs), NIR_DEBUG=validate
# and ACO_DEBUG=validateir,validatera, records a direct Mesh draw of the case's grid, a VS draw, an
# indirect Mesh draw (2 records) and an indirect-count Mesh draw, submits to the shim and checks the
# IB (RADV_DEBUG=dumpibs), the ACO IR (RADV_DEBUG=shaders), the merged NIR and the uploaded binary
# (see check.py), once with RADV_BC250_MESH_MERGE=1 and once with it off:
#   merge candidates: planner K/S (and wave / physical vertices when given), raw route, fast launch
#                     0, AMP >= lanes, THDS 0, direct count ceil(N/K), indirect packets identical to
#                     off, 2 GS_ALLOC_REQ (fully-culled + live) with the K*P / merged vertex literals,
#                     one SetMeshOutputs, no workgroup id left, LDS < 32 KiB, no scratch;
#   stage 2 (per-primitive outputs / CullPrimitive): physical vertices <= 256, outputs re-emitted
#                     by the lane (no cross-lane output access), the per-primitive data stored to the
#                     provoking-vertex record only, no per-primitive output left in the Mesh NIR, the
#                     fragment shader's per-primitive inputs read flat;
#   non-candidates:   on is byte-identical to off (IB, NIR/ISA dumps).
# Also: API barriers of a merged shape add no s_barrier (compared with the same shape without them).
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-mesh-merge.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1 BC250_TRACE_MERGE_NIR=1
  BC250_CAPTURE_MESH_CODE=1 RADV_DEBUG=dumpibs,shaders,nocache)

shim() {
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_FL1 \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$ICD" VK_ICD_FILENAMES="$ICD" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/merge" "$here/merge.c" -lvulkan || exit 2
glslangValidator -S vert --target-env vulkan1.3 -o "$work/tri_vert.spv" "$here/tri.vert" >/dev/null || exit 2
glslangValidator -S frag --target-env vulkan1.3 -o "$work/tri_frag.spv" "$here/tri.frag" >/dev/null || exit 2
glslangValidator -S task --target-env vulkan1.3 -o "$work/task.spv" "$here/shape.task" >/dev/null || exit 2

fail=0; passed=0; total=0
declare -A barriers
# name | glslang defines | required subgroup size (0 = none) | grid x y z | expected K S [wave
# merged_V] (0 0 = not a candidate: on must be byte-identical to off) | task shader | extra env
while IFS='|' read -r name defs wave grid want task xenv; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  pp=$(sed -n 's/.*-DPERPRIM=\([0-9]\).*/\1/p' <<<"$defs"); pp=${pp:-0}
  glslangValidator -S mesh --target-env vulkan1.3 $defs -o "$work/$name.mesh.spv" "$here/shape.mesh" >/dev/null || exit 2
  glslangValidator -S frag --target-env vulkan1.3 -DPERPRIM=$pp -o "$work/$name.frag.spv" "$here/shape.frag" >/dev/null || exit 2
  for sw in 0 1 1b; do
    ind=a; [ "$sw" = 1b ] && ind=b
    shim env RADV_BC250_MESH_MERGE=${sw%b} RADV_BC250_MESH_MERGE_INDIRECT=$ind $xenv ./merge "$name.mesh.spv" \
      "$name.frag.spv" $wave $grid $task > "$work/$name.$sw.out" 2> "$work/$name.$sw.err"
  done
  total=$((total + 1))
  status=ok; detail=
  grep -q '^SUBMIT_OK' "$work/$name.1.out" && grep -q '^SUBMIT_OK' "$work/$name.0.out" &&
    grep -q '^SUBMIT_OK' "$work/$name.1b.out" || { status=FAIL; detail="no-submit"; }
  set -- $want
  res=$(OPTB="$work/$name.1b.err" python3 "$here/check.py" "$work/$name.1.err" "$work/$name.0.err" "$1" "$2" $grid \
    "${3:-64}" "${4:-0}") || status=FAIL
  b=$(sed -n 's/.* s_barrier=\([0-9]*\) .*/\1/p' <<<"$res"); barriers[$name]=$b
  [ "$status" = ok ] && passed=$((passed + 1)) || fail=1
  printf '%-14s grid=%-8s status=%s %s\n   %s\n' "$name" "${grid// /x}" "$status" "$detail" "$res"
done <<'CASES'
# dense S = L = 32: 32V/32P -> K=3, 99 vertices (3 sinks), 96 triangles, 99 lanes; N=100 has a tail
d32|-DWS=32 -DNV=32 -DNP=32|0|100 1 1|3 32|
# the same with requiredSubgroupSize 32: no subgroup operation, so wave64 is used anyway
d32_rss32|-DWS=32 -DNV=32 -DNP=32|32|99 1 1|3 32|
d16|-DWS=16 -DNV=16 -DNP=16|0|100 1 1|5 16|
# L = 20 does not divide 64: padded to S = 32 (lanes 20..31 of each instance idle), K=4, 128 lanes
d20|-DWS=20 -DNV=20 -DNP=20|0|101 1 1|4 32|
# V = P = 24 < S = 32: vertex/primitive indices are offset, outputs are cross-lane
d24|-DWS=32 -DNV=24 -DNP=24|0|100 1 1|3 32|
# shared memory (dynamic index) + 2 API barriers: slices at k*128 bytes, barriers shrink to subgroup
sh32|-DWS=32 -DNV=32 -DNP=32 -DSHARED=1|0|100 1 1|3 32|
# runtime counts below the declared ones: hole primitives on the sink vertices
rt32|-DWS=32 -DNV=32 -DNP=32 -DRUNTIME=1|0|100 1 1|3 32|
# 3D API workgroup (8x4) and gl_WorkGroupID.xyz / gl_NumWorkGroups on 3D grids (N=60 exact, N=70 tail)
wg3d|-DWS=8 -DWSY=4 -DNV=32 -DNP=32 -DWGID3D=1|0|5 4 3|3 32|
wg3d_tail|-DWS=8 -DWSY=4 -DNV=32 -DNP=32 -DWGID3D=1 -DRUNTIME=1|0|7 5 2|3 32|
# subgroup operation with a 64-lane API workgroup: one instance per wave (S=64), K=2, 99 vertices
sg64|-DWS=64 -DNV=48 -DNP=48 -DSUBGROUP=1|0|100 1 1|2 64|
# wave32 shaders with subgroup operations (L <= 32; RADV runs them as one wave32): one instance per
# wave32 (S=32), the merged shader stays wave32 (was refused before stage 2)
sg32|-DWS=32 -DNV=32 -DNP=32 -DSUBGROUP=1|0|100 1 1|3 32 32 99|
shc32|-DWS=32 -DNV=32 -DNP=32 -DSHAREDCONST=1|0|100 1 1|3 32 32 99|
# stage 2: per-primitive outputs ride on a private provoking vertex per primitive, K*(V-1+P)+3 physical
# vertices (32/32: 3*63+3 = 192), one lane per physical vertex
pp32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=1|0|100 1 1|3 32 64 192|
# Hellblade 2's small Nanite shape (per-vertex vec4 + flat uvec3 + flat ivec4, per-primitive uvec4)
hb32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2|0|100 1 1|3 32 64 192|
# the same with wave intrinsics (HB2 is wave32 with subgroup operations): S=32, wave32, 6 waves
hb32sg|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2 -DSUBGROUP=1|0|100 1 1|3 32 32 192|
hb32sg_rt|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2 -DSUBGROUP=1 -DRUNTIME=1 -DSHARED=1|0|100 1 1|3 32 32 192|
hb32_rt|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2 -DRUNTIME=1|0|100 1 1|3 32 64 192|
hb32_tail|-DWS=8 -DWSY=4 -DNV=32 -DNP=32 -DPERPRIM=2 -DWGID3D=1 -DRUNTIME=1|0|7 5 2|3 32 64 192|
pp16|-DWS=16 -DNV=16 -DNP=16 -DPERPRIM=2|0|100 1 1|5 16 64 158|
pp24|-DWS=32 -DNV=24 -DNP=24 -DPERPRIM=1|0|101 1 1|3 32 64 144|
pp48|-DWS=64 -DNV=48 -DNP=48 -DPERPRIM=2|0|100 1 1|2 64 64 193|
pp20|-DWS=20 -DNV=20 -DNP=20 -DPERPRIM=3|0|101 1 1|4 32 64 159|
# several per-primitive outputs incl. gl_PrimitiveID, and provoking vertex LAST (static)
ppmulti32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=3|0|100 1 1|3 32 64 192|
pplast32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2|0|100 1 1|3 32 64 192||MERGE_PROVOKING=last
# a compact per-vertex output array (gl_ClipDistance[2]) next to the per-primitive data
ppclip32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=1 -DCLIP=1|0|100 1 1|3 32 64 192||RADV_BC250_MESH_ALLOW_POS1=1
# CullPrimitive: culled primitives become holes (with per-primitive data: provoking layout; alone:
# the stage-1 vertex layout, 3*32+3 = 99)
cull32|-DWS=32 -DNV=32 -DNP=32 -DCULL=1|0|100 1 1|3 32 64 99|
ppcull32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2 -DCULL=1 -DRUNTIME=1|0|100 1 1|3 32 64 192|
# option A indirect (RADV_BC250_MESH_MERGE_INDIRECT=a): one indirect record (count 1 of max 1);
# 1000 records (gl_DrawID; count 700 of 1000); 3D grid records and N not a multiple of K (N=70, K=3);
# a per-primitive wave32 HB2 shape with 1000 records; K=5 with 1000 records
ind1_d32|-DWS=32 -DNV=32 -DNP=32|0|100 1 1|3 32||MERGE_IND_RECORDS=1 MERGE_CNT=1
ind1000_d32|-DWS=32 -DNV=32 -DNP=32|0|100 1 1|3 32||MERGE_IND_RECORDS=1000 MERGE_CNT=700
ind1000_wg3d|-DWS=8 -DWSY=4 -DNV=32 -DNP=32 -DWGID3D=1 -DRUNTIME=1|0|7 5 2|3 32||MERGE_IND_RECORDS=1000 MERGE_CNT=999
ind1000_hb32sg|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2 -DSUBGROUP=1|0|100 1 1|3 32 32 192||MERGE_IND_RECORDS=1000 MERGE_CNT=3
ind1000_pp16|-DWS=16 -DNV=16 -DNP=16 -DPERPRIM=2|0|100 1 1|5 16 64 158||MERGE_IND_RECORDS=1000 MERGE_CNT=1000
# not candidates (planner refuses or never runs): byte-identical to the switch off
sg64_32|-DWS=64 -DNV=32 -DNP=32 -DSUBGROUP=1|0|100 1 1|0 0|
f64|-DWS=64 -DNV=64 -DNP=64|0|100 1 1|0 0|
p124|-DWS=32 -DNV=64 -DNP=124|0|100 1 1|0 0|
task32|-DWS=32 -DNV=32 -DNP=32|0|100 1 1|0 0|task.spv
# stage-2 refusals: 64V/32P per-primitive (3*95+3 = 288 > 256 physical vertices), 32V/64P (K*P = 128),
# per-primitive gl_Layer, dynamic provoking vertex mode, Task
pp64_32|-DWS=32 -DNV=64 -DNP=32 -DPERPRIM=2|0|100 1 1|0 0|
pp32_64|-DWS=32 -DNV=32 -DNP=64 -DPERPRIM=2|0|100 1 1|0 0|
pplayer32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=4|0|100 1 1|0 0|
ppdyn32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2|0|100 1 1|0 0||MERGE_PROVOKING=dynamic
pptask32|-DWS=32 -DNV=32 -DNP=32 -DPERPRIM=2|0|100 1 1|0 0|task.spv
# common meshlet classes with per-primitive data that stay on the base driver's route: 64V/64P (K*P = 128),
# meshoptimizer 64V/124P and 128V/128P (P > 64, the base driver split + expansion); and 128V/128P without
pp64|-DWS=64 -DNV=64 -DNP=64 -DPERPRIM=2|0|100 1 1|0 0|
pp124|-DWS=32 -DNV=64 -DNP=124 -DPERPRIM=2|0|100 1 1|0 0|
pp128|-DWS=128 -DNV=128 -DNP=128 -DPERPRIM=2|0|100 1 1|0 0|
c128|-DWS=128 -DNV=128 -DNP=128|0|100 1 1|0 0|
CASES

# API barriers must not reach the hardware in case A: sh32 has the s_barrier count of d32.
total=$((total + 1))
if [ -n "${barriers[d32]:-}" ] && [ "${barriers[sh32]:-x}" = "${barriers[d32]}" ]; then
  passed=$((passed + 1)); printf 'api_barriers   status=ok s_barrier(sh32)=%s s_barrier(d32)=%s\n' "${barriers[sh32]}" "${barriers[d32]}"
else
  fail=1; printf 'api_barriers   status=FAIL s_barrier(sh32)=%s s_barrier(d32)=%s\n' "${barriers[sh32]:-}" "${barriers[d32]:-}"
fi
# option A indirect: CPU model of the prep + shader index math (every API workgroup exactly once)
total=$((total + 1))
if res=$(python3 "$here/indmodel.py"); then
  passed=$((passed + 1)); printf 'indirect_model status=ok %s\n' "$res"
else
  fail=1; printf 'indirect_model status=FAIL %s\n' "$res"
fi
echo "mesh-merge: $passed/$total passed"
exit $fail

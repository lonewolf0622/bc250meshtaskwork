#!/usr/bin/env bash
# BC250 VK_KHR_fragment_shader_barycentric regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; bary and pipe also refuse any other device. Driver policy: the
# base driver launcher policy (post-Mesh VGT_FLUSH off, hybrid Task on unless a case says otherwise),
# NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1, shader cache off.
#
#   features   the extension and the feature are listed by default, not with RADV_BC250_NO_BARYCENTRICS=1
#              (Mesh stays listed by the driver either way); GPL / shader objects only without hybrid Task.
#   vertex     bary.c pipelines (VS, VS+GS, VS+TCS+TES; triangle list/strip/fan, lines, points; provoking
#              first/last/dynamic; graphics pipeline libraries and shader objects without hybrid Task) with
#              a plain, a barycentric (BARY) and a per-vertex (PERVERTEX) fragment shader: pipeline and
#              submission succeed, no validation error, the expected BC250 BARYCENTRICS trace.
#   mesh       ../direct-read/pipe.c Mesh (+Task) pipelines (expanded, split pieces, direct read,
#              autocull, compact LDS, implicit triangles, shared-vertex compaction (RADV_BC250_MESH_COMPACT,
#              also combined with implicit triangles, direct read and autocull), AMD route, merge, raw fast, hybrid Task,
#              lines, points, per-primitive data) with the same three fragment shaders (bary.frag reads
#              every dr.mesh output, so the Mesh shader is the same for all three). Checked: results,
#              trace, the Mesh LDS layout of the barycentric pipelines equal to the plain one (the
#              reference slots take no LDS), and the CPU export oracle (../direct-read/mesh_oracle
#              --bary-ref, SEEDS seeds) on the lowered Mesh NIR: plain vs BARY and PERVERTEX with
#              RADV_BC250_DIAG_BARY_NO_REF=1 vs PERVERTEX export exactly the same, plus two reference
#              parameters per exported vertex equal to its position.
#   identical  with OLDICD (the base build, be1f621): every plain pipeline of the vertex and mesh
#              sections is byte-identical (BC250_CAPTURE_POLICY_SHADERS code and configuration).
#   isa        with KEEP, RADV_DEBUG=shaders dumps of representative pipelines in $KEEP/isa, and the
#              code size of each shader, plain vs BARY, in $KEEP/isa/sizes.txt.
# usage: ICD=<radeon_devenv_icd json> BUILD=<mesa build dir, for the oracle> [OLDICD=<base icd>] [SHIM=..]
#        [SEEDS=64] [KEEP=<dir>] [ONLY=<case regex>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?ICD=<radeon_devenv_icd json>}
BUILD=${BUILD:?BUILD=<mesa build dir> (for mesh_oracle)}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
SEEDS=${SEEDS:-64}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-bary.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1)

shim() {
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -i PATH=/usr/bin HOME=/tmp \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cd "$work" || exit 2
cc -O1 -Wall -o "$work/bary" "$here/bary.c" -lvulkan || exit 2
cc -O1 -Wall -o "$work/pipe" "$here/../direct-read/pipe.c" -lvulkan || exit 2
"$here/../direct-read/build_oracle.sh" "$work/mesh_oracle" >/dev/null || exit 2
G="glslangValidator --target-env vulkan1.3"
$G -S vert -o v.spv "$here/bary.vert" >/dev/null || exit 2
$G -S geom -o g.spv "$here/bary.geom" >/dev/null || exit 2
$G -S tesc -o tc.spv "$here/bary.tesc" >/dev/null || exit 2
$G -S tese -o te.spv "$here/bary.tese" >/dev/null || exit 2
$G -S frag -o fs_plain.spv "$here/vary.frag" >/dev/null || exit 2
$G -S frag -DBARY=1 -o fs_bary.spv "$here/vary.frag" >/dev/null || exit 2
$G -S frag -DPERVERTEX=1 -o fs_pervertex.spv "$here/vary.frag" >/dev/null || exit 2

fail=0; total=0; oracle_pairs=0
sig() { grep -E '^BC250POLICY(CODE)? ' "$1" | sed 's/ va=[0-9a-f]*//' | sort | sha1sum | cut -c1-16; }
bad_output() { grep -qiE 'validation failed|NIR_VALIDATE|assert|error:|^FAIL |Unhandled' "$1"; }
report() { # name variant status detail
  total=$((total + 1)); [ "$3" = ok ] || fail=1
  printf '%-22s %-14s %-4s%s\n' "$1" "$2" "$3" "$4"
}
check_patterns() { # file, ';'-separated patterns ('!' = must not appear, '~' = regex)
  local out=$1 pats=$2 bad=0 IFS=';'
  for p in $pats; do
    [ -z "$p" ] && continue
    case "$p" in
      '!'*) grep -qF -- "${p:1}" "$out" && { echo -n " unexpected[${p:1}]"; bad=1; } ;;
      '~'*) grep -qE -- "${p:1}" "$out" || { echo -n " missing[${p:1}]"; bad=1; } ;;
      *) grep -qF -- "$p" "$out" || { echo -n " missing[$p]"; bad=1; } ;;
    esac
  done
  return $bad
}

# ---- features ----
if [ -z "${ONLY:-}" ]; then
  while IFS='|' read -r name env want; do
    shim "$ICD" env $env ./bary features > "$work/features.$name.out" 2>&1
    got=$(grep '^EXT=' "$work/features.$name.out")
    s=ok; [ "$got" = "$want" ] || s=FAIL
    report "features" "$name" $s " $got"
  done <<'EOF'
default|A=1|EXT=1 FEATURE=1 TRISTRIP_INDEPENDENT=0 MESH=1 TASK=1 GPL=0 ESO=0 PROVOKING_LAST=1
opt-out|RADV_BC250_NO_BARYCENTRICS=1|EXT=0 FEATURE=0 TRISTRIP_INDEPENDENT=0 MESH=1 TASK=1 GPL=0 ESO=0 PROVOKING_LAST=1
no-hybrid|RADV_BC250_HYBRID_TASK=0|EXT=1 FEATURE=1 TRISTRIP_INDEPENDENT=0 MESH=1 TASK=0 GPL=1 ESO=1 PROVOKING_LAST=1
opt-out-no-hybrid|RADV_BC250_HYBRID_TASK=0 RADV_BC250_NO_BARYCENTRICS=1|EXT=0 FEATURE=0 TRISTRIP_INDEPENDENT=0 MESH=1 TASK=0 GPL=1 ESO=1 PROVOKING_LAST=1
EOF
  shim "$ICD" env RADV_BC250_NO_BARYCENTRICS=1 ./bary mono tri first v.spv fs_bary.spv > "$work/features.refused.out" 2>&1
  s=ok; grep -q '^NO_BARYCENTRICS' "$work/features.refused.out" || s=FAIL
  report "features" "opt-out-refuses" $s ""
fi

# ---- vertex ----
# name | env | mode topology provoking | shaders (without the FS) | expected trace for BARY | for PERVERTEX
vertex_cases() { cat <<'EOF'
vs_tri_first|A=1|mono tri first|v.spv|rotation from reference producer=MESA_SHADER_VERTEX raw=31 flat=30 raster_vertices=3 provoking=first|rotation from reference producer=MESA_SHADER_VERTEX
vs_tri_last|A=1|mono tri last|v.spv|rotation from reference producer=MESA_SHADER_VERTEX raw=31 flat=30 raster_vertices=3 provoking=last|provoking=last
vs_tri_dynamic|A=1|mono tri dynamic|v.spv|rotation from reference producer=MESA_SHADER_VERTEX raw=31 flat=30 raster_vertices=3 provoking=dynamic|provoking=dynamic
vs_strip_first|A=1|mono strip first|v.spv|rotation from reference producer=MESA_SHADER_VERTEX|rotation from reference
vs_strip_last|A=1|mono strip last|v.spv|provoking=last|provoking=last
vs_fan_dynamic|A=1|mono fan dynamic|v.spv|provoking=dynamic|provoking=dynamic
vs_line|A=1|mono line first|v.spv|!BARYCENTRICS|rotation 0 producer=MESA_SHADER_VERTEX raw=-1 flat=-1 raster_vertices=2 provoking=first reason=points or lines
vs_point|A=1|mono point dynamic|v.spv|!BARYCENTRICS|reason=points or lines
gs_tri_first|A=1|mono tri first|v.spv g.spv|rotation from reference producer=MESA_SHADER_GEOMETRY raw=31 flat=30 raster_vertices=3 provoking=first|producer=MESA_SHADER_GEOMETRY
gs_tri_dynamic|A=1|mono tri dynamic|v.spv g.spv|producer=MESA_SHADER_GEOMETRY raw=31 flat=30 raster_vertices=3 provoking=dynamic|provoking=dynamic
tes_tri_last|A=1|mono tri last|v.spv tc.spv te.spv|rotation from reference producer=MESA_SHADER_TESS_EVAL raw=31 flat=30 raster_vertices=3 provoking=last|producer=MESA_SHADER_TESS_EVAL
tes_tri_dynamic|A=1|mono tri dynamic|v.spv tc.spv te.spv|producer=MESA_SHADER_TESS_EVAL raw=31 flat=30 raster_vertices=3 provoking=dynamic|provoking=dynamic
gpl_vkd3d|RADV_BC250_HYBRID_TASK=0|gpl-vkd3d tri first|v.spv|rotation from reference producer=MESA_SHADER_VERTEX raw=31 flat=30|rotation from reference
gpl_vkd3d_dynamic|RADV_BC250_HYBRID_TASK=0|gpl-vkd3d strip dynamic|v.spv|provoking=dynamic|provoking=dynamic
gpl_lto|RADV_BC250_HYBRID_TASK=0|gpl-lto tri last|v.spv|rotation from reference producer=MESA_SHADER_VERTEX raw=31 flat=30 raster_vertices=3 provoking=last|rotation from reference
gpl_split|RADV_BC250_HYBRID_TASK=0|gpl-split tri first|v.spv|rotation 0 producer=none raw=-1 flat=-1 raster_vertices=0 provoking=first reason=no producer;!rotation from reference|reason=no producer;!rotation from reference
eso_linked|RADV_BC250_HYBRID_TASK=0|eso-linked tri first|v.spv|rotation from reference producer=MESA_SHADER_VERTEX raw=31 flat=30 raster_vertices=0 provoking=dynamic|rotation from reference
eso_linked_tes|RADV_BC250_HYBRID_TASK=0|eso-linked tri last|v.spv tc.spv te.spv|rotation from reference producer=MESA_SHADER_TESS_EVAL|rotation from reference
eso_unlinked|RADV_BC250_HYBRID_TASK=0|eso-unlinked tri first|v.spv|rotation 0 producer=none raw=-1 flat=-1 raster_vertices=0 provoking=dynamic reason=no producer|reason=no producer
EOF
}

while IFS='|' read -r name env args shaders want_bary want_pv; do
  [ -z "$name" ] && continue
  for fsv in plain bary pervertex; do
    out=$work/vertex.$name.$fsv.out
    shim "$ICD" env $env ./bary $args $shaders fs_$fsv.spv > "$out" 2>&1
    s=ok; d=
    grep -q '^PIPELINE_RESULT=0' "$out" || { s=FAIL; d="$d result=$(sed -n 's/^PIPELINE_RESULT=//p' "$out" | tail -1)"; }
    grep -q '^SUBMIT_OK' "$out" || { s=FAIL; d="$d no-submit"; }
    bad_output "$out" && { s=FAIL; d="$d validation"; }
    case $fsv in
      plain) pats="!BARYCENTRICS" ;;
      bary) pats=$want_bary ;;
      pervertex) pats=$want_pv ;;
    esac
    msg=$(check_patterns "$out" "$pats") || s=FAIL
    report "$name" "$fsv" $s "$d$msg"
  done
  if [ -n "$OLDICD" ]; then
    shim "$OLDICD" env $env BARY_ALLOW_MISSING=1 BC250_CAPTURE_POLICY_SHADERS=1 ./bary $args $shaders fs_plain.spv > "$work/vertex.$name.old" 2>&1
    shim "$ICD" env $env BC250_CAPTURE_POLICY_SHADERS=1 ./bary $args $shaders fs_plain.spv > "$work/vertex.$name.new" 2>&1
    a=$(sig "$work/vertex.$name.old"); b=$(sig "$work/vertex.$name.new"); s=ok
    { [ "$a" = "$b" ] && grep -q '^BC250POLICY ' "$work/vertex.$name.old"; } || s=FAIL
    report "$name" "identical" $s " base=$a new=$b"
  fi
done < <(vertex_cases | grep -E "^(${ONLY:-.*})\|")

# ---- mesh ----
# name | dr.mesh / bary.frag -D options | task (-/1) | extra env | expected trace for BARY and PERVERTEX |
#   plain: "same" (the barycentric pipelines have the plain pipeline's Mesh shader plus the two
#   references: same LDS layout, oracle plain vs BARY), "other" (the Mesh route differs, e.g. the compact
#   vertex map refuses barycentric fragment shaders), "none" (points and lines: no reference, the Mesh
#   shader of BARY must be byte-identical to the plain one, PERVERTEX to PERVERTEX without references)
mesh_cases() { cat <<'EOF'
basic||-||rotation from reference producer=MESA_SHADER_MESH raw=31 flat=30 raster_vertices=3 provoking=first|same
perprim|-DPERPRIM=1|-||rotation from reference producer=MESA_SHADER_MESH|same
primid|-DPERPRIM=1 -DPRIMID=1 -DNOCULL=1|-||rotation from reference producer=MESA_SHADER_MESH|same
uniform|-DUNIFORM=1|-||rotation from reference producer=MESA_SHADER_MESH|same
partial|-DPARTIAL=1|-||rotation from reference producer=MESA_SHADER_MESH|same
clip|-DCLIP=1|-|RADV_BC250_MESH_ALLOW_POS1=1|rotation from reference producer=MESA_SHADER_MESH|same
multistore|-DMULTISTORE=1|-||rotation from reference producer=MESA_SHADER_MESH|same
zero|-DZERO=1|-||rotation from reference producer=MESA_SHADER_MESH|same
loop|-DLANES=64 -DVERTS=128 -DPRIMS=96|-||rotation from reference producer=MESA_SHADER_MESH|same
nanite|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-||pieces=3;rotation from reference producer=MESA_SHADER_MESH|same
nanite_p64|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_PERF_PIECE_PRIMS=64|pieces=2;rotation from reference producer=MESA_SHADER_MESH|same
nanite_batch|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_SPLIT_BATCH_PREP=1|pieces=3;rotation from reference|same
direct_read|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_MESH_DIRECT_READ=full|DIRECT READ: applied;rotation from reference producer=MESA_SHADER_MESH|same
direct_read_small|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1|-|RADV_BC250_MESH_DIRECT_READ=full|DIRECT READ: applied;rotation from reference|same
autocull|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1|-|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1|MESH AUTOCULL: applied;rotation from reference producer=MESA_SHADER_MESH|same
autocull_direct|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1|-|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1 RADV_BC250_MESH_DIRECT_READ=full|MESH AUTOCULL: applied;DIRECT READ: applied;rotation from reference|same
autocull_wide|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=2|rotation from reference producer=MESA_SHADER_MESH|same
compact_lds|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_MESH_COMPACT_LDS=1|rotation from reference producer=MESA_SHADER_MESH|same
implicit_tris|-DLANES=32 -DVERTS=32 -DPRIMS=32|-|RADV_BC250_MESH_IMPLICIT_TRIS=1|MESH IMPLICIT TRIS: applied;rotation from reference|same
compact|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|RADV_BC250_MESH_COMPACT=1|MESH COMPACT: applied;rotation from reference producer=MESA_SHADER_MESH|same
compact_basic||-|RADV_BC250_MESH_COMPACT=1|MESH COMPACT: applied;rotation from reference producer=MESA_SHADER_MESH|same
compact_implicit|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|RADV_BC250_MESH_COMPACT=1 RADV_BC250_MESH_IMPLICIT_TRIS=1|MESH COMPACT: applied;MESH IMPLICIT TRIS: applied;rotation from reference|same
compact_nanite_direct|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1 -DNOCULL=1|-|RADV_BC250_MESH_COMPACT=1 RADV_BC250_MESH_IMPLICIT_TRIS=1 RADV_BC250_MESH_DIRECT_READ=full RADV_BC250_PERF_PIECE_PRIMS=64|MESH COMPACT: applied;DIRECT READ: applied;rotation from reference producer=MESA_SHADER_MESH|same
compact_autocull|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|RADV_BC250_MESH_COMPACT=1 RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1 RADV_BC250_MESH_DIRECT_READ=full|MESH COMPACT: applied;MESH AUTOCULL: applied;rotation from reference|same
compact_task|-DTASK=1|1|RADV_BC250_MESH_COMPACT=1|MESH COMPACT: applied;rotation from reference producer=MESA_SHADER_MESH|same
compact_map|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|BC250_COMPACT_VERTICES=true|!compact map;rotation from reference|other
amd_route|-DLANES=32 -DVERTS=32 -DPRIMS=32|-|RADV_BC250_MESH_AMD=1|rotation from reference producer=MESA_SHADER_MESH|same
fast_raw|-DLANES=128 -DVERTS=128 -DPRIMS=126|-|RADV_BC250_MESH_FAST=1|rotation from reference producer=MESA_SHADER_MESH|same
merge|-DLANES=32 -DVERTS=24 -DPRIMS=8|-|RADV_BC250_MESH_MERGE=1|rotation from reference producer=MESA_SHADER_MESH|same
task|-DTASK=1|1||rotation from reference producer=MESA_SHADER_MESH|same
task_nanite|-DTASK=1 -DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|1||rotation from reference producer=MESA_SHADER_MESH|same
lines|-DLINES=1|-||!rotation from reference|none
points|-DPOINTS=1|-||!rotation from reference|none
EOF
}

compile_mesh() { # defines task
  local defs=$1 task=$2 tag
  tag=$(echo "m $defs" | tr -c 'A-Za-z0-9\n' '_')
  mesh=$work/$tag.mesh.spv; tspv=-
  if [ ! -f "$mesh" ]; then
    $G -S mesh $defs -o "$mesh" "$here/../direct-read/dr.mesh" >/dev/null || return 1
    $G -S frag $defs -o "$work/$tag.plain.spv" "$here/bary.frag" >/dev/null || return 1
    $G -S frag $defs -DBARY=1 -o "$work/$tag.bary.spv" "$here/bary.frag" >/dev/null || return 1
    $G -S frag $defs -DPERVERTEX=1 -o "$work/$tag.pervertex.spv" "$here/bary.frag" >/dev/null || return 1
  fi
  fplain=$work/$tag.plain.spv; fbary=$work/$tag.bary.spv; fpv=$work/$tag.pervertex.spv
  if [ "$task" = 1 ]; then
    tspv=$work/dr.task.spv; [ -f "$tspv" ] || $G -S task -o "$tspv" "$here/../direct-read/dr.task" >/dev/null || return 1
  fi
}

oracle_pair() { # refdir canddir -> prints summary, returns 1 on failure
  local ref=$1 cand=$2 bad=0 n=0 line c r checked=0
  local refs=("$ref"/*.nir) cands=("$cand"/*.nir)
  [ -e "${refs[0]}" ] && [ -e "${cands[0]}" ] || { echo -n " no-dumps"; return 1; }
  [ ${#refs[@]} = ${#cands[@]} ] || { echo -n " dumps ${#refs[@]}/${#cands[@]}"; bad=1; }
  for c in "${cands[@]}"; do
    r=$ref/$(basename "$c")
    [ -f "$r" ] || { echo -n " unpaired[$(basename "$c")]"; bad=1; continue; }
    line=$("$work/mesh_oracle" --bary-ref --seeds "$SEEDS" "$r" "$c" | tail -2 | tr '\n' ' ')
    n=$((n + 1))
    case "$line" in
      *"ORACLE PASS"*) checked=$((checked + $(echo "$line" | sed -n 's/.*reference_channels=\([0-9]*\).*/\1/p'))) ;;
      *) echo -n " oracle[$(basename "$c"): $line]"; bad=1 ;;
    esac
  done
  echo "$n" >> "$work/oracle_pairs"
  echo -n " $(basename "$ref")->$(basename "$cand")=$n/ref_channels=$checked"
  return $bad
}

same_dumps() { # dir dir -> byte-identical dumped Mesh NIR
  local a=$1 b=$2 f
  local fa=("$a"/*.nir)
  [ -e "${fa[0]}" ] || { echo -n " no-dumps"; return 1; }
  for f in "${fa[@]}"; do
    cmp -s "$f" "$b/$(basename "$f")" || { echo -n " mesh-differs[$(basename "$f")]"; return 1; }
  done
  echo -n " mesh-identical=${#fa[@]}"
}

lds_sig() { grep -o 'MESH LDS: .*' "$1" | md5sum | cut -c1-12; }

: > "$work/oracle_pairs"
while IFS='|' read -r name defs task extra want plainrel; do
  [ -z "$name" ] && continue
  compile_mesh "$defs" "$task" || { report "$name" compile FAIL ""; continue; }
  for v in plain bary0 bary pervertex0 pervertex; do
    d=$work/dump/$name/$v; rm -rf "$d"; mkdir -p "$d"
    case $v in
      plain) f=$fplain; e= ;;
      bary0) f=$fbary; e=RADV_BC250_DIAG_BARY_NO_REF=1 ;;
      bary) f=$fbary; e= ;;
      pervertex0) f=$fpv; e=RADV_BC250_DIAG_BARY_NO_REF=1 ;;
      pervertex) f=$fpv; e= ;;
    esac
    shim "$ICD" env $extra $e BC250_MESH_NIR_DUMP="$d" ./pipe "$mesh" "$f" "$tspv" 1 > "$work/mesh.$name.$v.out" 2>&1
  done
  for v in plain bary pervertex; do
    out=$work/mesh.$name.$v.out; s=ok; det=
    for o in "$out" "$work/mesh.$name.${v}0.out"; do
      [ $v = plain ] && [ "$o" != "$out" ] && continue
      grep -q '^PIPELINE_RESULT=0' "$o" || { s=FAIL; det="$det result=$(sed -n 's/^PIPELINE_RESULT=//p' "$o" | tail -1)"; }
      grep -q '^SUBMIT_OK' "$o" || { s=FAIL; det="$det no-submit"; }
      bad_output "$o" && { s=FAIL; det="$det validation"; }
    done
    pats="!BARYCENTRICS"; [ $v != plain ] && pats=$want
    msg=$(check_patterns "$out" "$pats") || s=FAIL
    det="$det$msg"
    if [ $v != plain ]; then
      if [ "$plainrel" = none ]; then
        ref=plain; [ $v = pervertex ] && ref=pervertex0
        msg=$(same_dumps "$work/dump/$name/$ref" "$work/dump/$name/$v") || s=FAIL
        det="$det$msg"
      else
        # The references take no LDS: same layout with and without them.
        [ "$(lds_sig "$out")" = "$(lds_sig "$work/mesh.$name.${v}0.out")" ] || { s=FAIL; det="$det lds-differs-from-noref"; }
        msg=$(oracle_pair "$work/dump/$name/${v}0" "$work/dump/$name/$v") || s=FAIL
        det="$det$msg"
        if [ "$plainrel" = same ] && [ $v = bary ]; then
          [ "$(lds_sig "$out")" = "$(lds_sig "$work/mesh.$name.plain.out")" ] || { s=FAIL; det="$det lds-differs-from-plain"; }
          msg=$(oracle_pair "$work/dump/$name/plain" "$work/dump/$name/bary") || s=FAIL
          det="$det$msg"
        fi
      fi
    fi
    report "$name" "$v" $s "$det"
  done
  if [ -n "$OLDICD" ]; then
    # Switches the base build does not have (RADV_BC250_MESH_COMPACT) are compared switched off.
    idextra=${extra//RADV_BC250_MESH_COMPACT=1/}
    shim "$OLDICD" env $idextra BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$fplain" "$tspv" 1 > "$work/mesh.$name.old" 2>&1
    shim "$ICD" env $idextra BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$fplain" "$tspv" 1 > "$work/mesh.$name.new" 2>&1
    a=$(sig "$work/mesh.$name.old"); b=$(sig "$work/mesh.$name.new"); s=ok; why=
    { [ "$a" = "$b" ] && grep -q '^BC250POLICY ' "$work/mesh.$name.old"; } || s=FAIL
    # Two position vectors (clip distance, point size): the GFX10 fully-culled dummy vertex now exports both
    # (b1f639e), so these differ from the base build by design.
    if [ $s = FAIL ] && { [ "$name" = clip ] || [ "$name" = points ]; } && grep -q '^BC250POLICY ' "$work/mesh.$name.new" &&
       grep -q 'MESH POS EXPORTS: format=2 .*done_exports' "$work/mesh.$name.new" && ! grep -q done_mismatch "$work/mesh.$name.new"; then
      s=ok; why=" (expected: the fully-culled dummy exports both declared position vectors)"
    fi
    report "$name" "identical" $s " base=$a new=$b$why"
  fi
done < <(mesh_cases | grep -E "^(${ONLY:-.*})\|")
oracle_pairs=$(awk '{s+=$1} END {print s+0}' "$work/oracle_pairs")

# ---- isa (KEEP only) ----
if [ -n "${KEEP:-}" ] && [ -z "${ONLY:-}" ]; then
  mkdir -p "$work/isa"
  : > "$work/isa/sizes.txt"
  isa() { # name kind(vertex/mesh) fsvariant args...
    local name=$1 kind=$2 v=$3; shift 3
    shim "$ICD" env RADV_DEBUG=shaders BC250_CAPTURE_POLICY_SHADERS=1 "$@" > "$work/isa/$name.$v.txt" 2>&1
    grep '^BC250POLICY ' "$work/isa/$name.$v.txt" | sed "s/^/$name $v /; s/ va=[0-9a-f]*//" >> "$work/isa/sizes.txt"
  }
  for v in plain bary pervertex; do
    isa vs_tri_first vertex $v ./bary mono tri first v.spv fs_$v.spv
    isa vs_tri_dynamic vertex $v ./bary mono tri dynamic v.spv fs_$v.spv
    isa gs_tri_first vertex $v ./bary mono tri first v.spv g.spv fs_$v.spv
    isa tes_tri_last vertex $v ./bary mono tri last v.spv tc.spv te.spv fs_$v.spv
  done
  compile_mesh "" - && for v in plain bary pervertex; do
    f=$fplain; [ $v = bary ] && f=$fbary; [ $v = pervertex ] && f=$fpv
    isa mesh_basic mesh $v ./pipe "$mesh" "$f" - 1
  done
  compile_mesh "-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1" - && for v in plain bary; do
    f=$fplain; [ $v = bary ] && f=$fbary
    isa mesh_nanite_direct mesh $v env RADV_BC250_MESH_DIRECT_READ=full ./pipe "$mesh" "$f" - 1
  done
  echo "isa dumps in $work/isa"
fi

echo "barycentrics: $total checks, oracle pairs $oracle_pairs, $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail

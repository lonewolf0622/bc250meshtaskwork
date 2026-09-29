#!/usr/bin/env bash
# BC250 no-op VRS regression (GFX1013: VK_KHR_fragment_shading_rate on by default, shading always 1x1;
# RADV_BC250_VRS_NOOP=0 hides it). drm-shim only, never the real GPU.
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim preloaded
# and AMDGPU_GPU_ID=gfx1013; vrs refuses any other device (the control section uses the shim's NAVI21).
# Driver policy: the base driver launcher policy (post-Mesh VGT_FLUSH off, hybrid Task on unless a case says
# otherwise), NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1, shader cache off.
#
#   features   VK_KHR_fragment_shading_rate with pipeline/primitive/attachment rates, 8x8 texels and
#              non-trivial combiners by default and with RADV_BC250_VRS_NOOP=1 (with and without hybrid
#              Task); with RADV_BC250_VRS_NOOP=0 nothing is listed and a rate pipeline is refused. The shader
#              object binary UUID differs between the two states.
#   vertex     vrs.c VS (+GS) pipelines, one pipeline / GPL (vkd3d-proton style) / linked shader objects,
#              rate modes none, static, dynamic, attachment (+depth), attachment-nodepth, renderpass:
#              the pre-rasterization stage writes PrimitiveShadingRateKHR and the fragment shader reads
#              ShadingRateKHR (WRITE_RATE / READ_RATE). Checked against the same pipeline with plain shaders:
#              result and submission, no validation error, the BC250 VRS NOOP trace (output stripped, read
#              zeroed), byte-identical shader code and configuration (BC250_CAPTURE_POLICY_SHADERS), a
#              byte-identical command stream (RADV_DEBUG=dumpibs), and check_ib.py: no VRS register (by name
#              or offset) and no VRS field in any IB. Rate state vs none and bound vs unbound attachment
#              (+ pipeline flag): identical command streams (and code).
#   mesh       vrs.c Mesh (+Task) pipelines with ../direct-read/dr.mesh (-DVRS=1: per-primitive
#              gl_PrimitiveShadingRateEXT) and dr.frag (-DREAD_RATE=1), on every BC250 Mesh route (expanded,
#              split pieces, direct read, autocull, compact LDS, implicit triangles, compact vertex map, AMD
#              route, raw fast route, merge, hybrid Task plain and split, lines, points), rate modes dynamic and
#              attachment: the same checks against the plain Mesh pipeline (none), plus the same BC250 MESH
#              route trace.
#   cache      nircache with a disk cache: a NIR cache hit is still stripped, and the switch-on NIR is not
#              reused with the switch off.
#   identical  with OLDICD (the base build): with RADV_BC250_VRS_NOOP=0 the plain pipelines of the vertex and
#              mesh sections are byte-identical (code and command stream) to the base build and `vrs features`
#              prints the same line; with the default (on) the plain pipelines are byte-identical too.
#   control    the same attachment workload on the shim's NAVI21 (GFX10.3, real VRS): check_ib.py must see
#              the VRS registers and the bound/unbound command streams must differ (the checks can fail).
#   isa        with KEEP, RADV_DEBUG=shaders,spirv dumps of representative pipelines in $KEEP/isa and
#              $KEEP/isa/grep.txt: the SPIR-V has the built-ins, the NIR/ISA has no named shading rate
#              leftover, no rate position export (pos1) and no ancillary rate unpack; the same rate shaders
#              on the shim's NAVI21 (GFX10.3 control) have both.
# usage: ICD=<radeon_devenv_icd json> [OLDICD=<base icd>] [SHIM=..] [KEEP=<dir>] [ONLY=<case regex>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?ICD=<radeon_devenv_icd json>}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-vrs.XXXXXX")}
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
ON=              # the default: on
OFF=RADV_BC250_VRS_NOOP=0

shim() { # icd env...
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -i PATH=/usr/bin HOME=/tmp \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cd "$work" || exit 2
cc -O1 -Wall -o "$work/vrs" "$here/vrs.c" -lvulkan || exit 2
G() { glslangValidator --target-env vulkan1.3 "$@" >/dev/null; }
G -S vert -o v.spv "$here/vrs.vert" || exit 2
G -S vert -DWRITE_RATE=1 -o vw.spv "$here/vrs.vert" || exit 2
G -S geom -o g.spv "$here/vrs.geom" || exit 2
G -S geom -DWRITE_RATE=1 -o gw.spv "$here/vrs.geom" || exit 2
G -S frag -o f.spv "$here/vrs.frag" || exit 2
G -S frag -DREAD_RATE=1 -o fr.spv "$here/vrs.frag" || exit 2

fail=0; total=0
sig() { grep -E '^BC250POLICY(CODE)? ' "$1" | sed 's/ va=[0-9a-f]*//' | sort | sha1sum | cut -c1-16; }
ibsig() { sed 's/\x1b\[[0-9;]*m//g' "$1" | sed -n '/IB begin/,/IB end/p' | md5sum | cut -c1-16; }
routes() { grep -E '^BC250 MESH' "$1" | sort -u | md5sum | cut -c1-12; }
bad_output() { grep -qiE 'validation failed|NIR_VALIDATE|assert|error:|^FAIL |Unhandled' "$1"; }
report() { # name variant status detail
  total=$((total + 1)); [ "$3" = ok ] || fail=1
  printf '%-26s %-22s %-4s%s\n' "$1" "$2" "$3" "$4"
}
ran_ok() { # out -> detail on failure
  local d=
  grep -q '^PIPELINE_RESULT=0' "$1" || d="$d result=$(sed -n 's/^PIPELINE_RESULT=//p' "$1" | tail -1)"
  grep -q '^SUBMIT_OK' "$1" || d="$d no-submit"
  bad_output "$1" && d="$d validation"
  echo -n "$d"
  [ -z "$d" ]
}
run() { # out icd env... : one capture run (code + IB)
  local out=$1 icd=$2; shift 2
  shim "$icd" env BC250_CAPTURE_POLICY_SHADERS=1 RADV_DEBUG=dumpibs "$@" > "$out" 2>&1
}

# ---- features ----
if [ -z "${ONLY:-}" ]; then
  ONLINE='EXT=1 PIPELINE=1 PRIMITIVE=1 ATTACHMENT=1 NONTRIVIAL=1 TEXEL=8x8-8x8 MAXSIZE=2x2 RATES=2x2:7,2x1:7,1x2:7,1x1:ff MESH_FSR=0 R8_FSR=1 IMG_FSR=0 DEVICE=0 MSAA2=1'
  OFFLINE='EXT=0 PIPELINE=0 PRIMITIVE=0 ATTACHMENT=0 NONTRIVIAL=0 TEXEL=0x0-0x0 MAXSIZE=0x0 RATES=- MESH_FSR=0 R8_FSR=1 IMG_FSR=-1 DEVICE=-1 MSAA2=1'
  while IFS='|' read -r name env want; do
    shim "$ICD" env $env ./vrs features > "$work/features.$name.out" 2>&1
    got=$(grep '^EXT=' "$work/features.$name.out" | sed 's/ SBUUID=.*//')
    s=ok; [ "$got" = "$want" ] || s=FAIL
    report features "$name" $s " $got"
  done <<EOF
default|A=1|$ONLINE MESH=1 TASK=1 GPL=0 ESO=0
on|RADV_BC250_VRS_NOOP=1|$ONLINE MESH=1 TASK=1 GPL=0 ESO=0
default-no-hybrid|RADV_BC250_HYBRID_TASK=0|$ONLINE MESH=1 TASK=0 GPL=1 ESO=1
off|$OFF|$OFFLINE MESH=1 TASK=1 GPL=0 ESO=0
off-no-hybrid|$OFF RADV_BC250_HYBRID_TASK=0|$OFFLINE MESH=1 TASK=0 GPL=1 ESO=1
EOF
  # Shader object binaries of the two switch states are not interchangeable.
  a=$(sed -n 's/.* SBUUID=//p' "$work/features.off.out"); b=$(sed -n 's/.* SBUUID=//p' "$work/features.default.out"); s=ok
  { [ -n "$a" ] && [ "$a" != "$b" ] && [ "$b" = "$(sed -n 's/.* SBUUID=//p' "$work/features.on.out")" ]; } || s=FAIL
  report features shaderBinaryUUID $s " off=$a on=$b"
  shim "$ICD" env $OFF ./vrs mono dynamic vw.spv fr.spv > "$work/features.refused.out" 2>&1
  s=ok; grep -q '^NO_FSR' "$work/features.refused.out" || s=FAIL
  report features off-refuses $s ""
fi

# Checks a rate run against its plain reference: result, trace, code, command stream, IB content.
# name variant out ref want_trace(;-separated, !=must not appear)
compare() {
  local name=$1 var=$2 out=$3 ref=$4 pats=$5 s=ok d IFS=';' p
  d=$(ran_ok "$out") || s=FAIL
  for p in $pats; do
    [ -z "$p" ] && continue
    case "$p" in
      '!'*) grep -qF -- "${p:1}" "$out" && { d="$d unexpected[${p:1}]"; s=FAIL; } ;;
      *) grep -qF -- "$p" "$out" || { d="$d missing[$p]"; s=FAIL; } ;;
    esac
  done
  unset IFS
  local a b
  a=$(sig "$ref"); b=$(sig "$out")
  { [ "$a" = "$b" ] && grep -q '^BC250POLICYCODE ' "$out"; } || { d="$d code(plain=$a rate=$b)"; s=FAIL; }
  a=$(ibsig "$ref"); b=$(ibsig "$out")
  [ "$a" = "$b" ] || { d="$d ib(plain=$a rate=$b)"; s=FAIL; }
  local ck; ck=$(python3 "$here/check_ib.py" "$out") || { s=FAIL; }
  d="$d code=$(sig "$out" | cut -c1-8) ib=$(ibsig "$out" | cut -c1-8) [$(echo "$ck" | sed 's/^ibs=[0-9]* //; s/ unknown_offsets=0//')]"
  report "$name" "$var" $s "$d"
}

# ---- vertex ----
# name | env | api | modes | plain shaders (without FS) | rate shaders (without FS) | expected rate trace
vertex_cases() { cat <<'EOF'
vs|A=1|mono|none static dynamic attachment attachment-nodepth renderpass|v.spv|vw.spv|stage=vertex stripped_shading_rate_outputs=1
vs_gs|A=1|mono|none dynamic attachment|v.spv g.spv|v.spv gw.spv|stage=geometry stripped_shading_rate_outputs=1
vs_nohybrid|RADV_BC250_HYBRID_TASK=0|mono|dynamic attachment renderpass|v.spv|vw.spv|stage=vertex stripped_shading_rate_outputs=1
gpl|RADV_BC250_HYBRID_TASK=0|gpl|static dynamic attachment|v.spv|vw.spv|stage=vertex stripped_shading_rate_outputs=1
gpl_gs|RADV_BC250_HYBRID_TASK=0|gpl|dynamic|v.spv g.spv|v.spv gw.spv|stage=geometry stripped_shading_rate_outputs=1
eso|RADV_BC250_HYBRID_TASK=0|eso|none dynamic attachment-nodepth|v.spv|vw.spv|stage=vertex stripped_shading_rate_outputs=1
eso_gs|RADV_BC250_HYBRID_TASK=0|eso|dynamic attachment|v.spv g.spv|v.spv gw.spv|stage=geometry stripped_shading_rate_outputs=1
EOF
}

while IFS='|' read -r name env api modes plain rate want; do
  [ -z "$name" ] && continue
  for m in $modes; do
    run "$work/vertex.$name.$m.plain" "$ICD" $ON $env ./vrs $api $m $plain f.spv
    run "$work/vertex.$name.$m.rate" "$ICD" $ON $env ./vrs $api $m $rate fr.spv
    compare "$name" "$m" "$work/vertex.$name.$m.rate" "$work/vertex.$name.$m.plain" \
      "$want;stage=fragment stripped_shading_rate_outputs=0 zeroed_shading_rate_reads=1;RATE_CALLS=$(case $m in dynamic|attachment*) echo 2;; *) echo 0;; esac)"
    case $m in
      attachment*|renderpass)
        run "$work/vertex.$name.$m-unbound.rate" "$ICD" $ON $env ./vrs $api $m-unbound $rate fr.spv
        s=ok; d=
        d=$(ran_ok "$work/vertex.$name.$m-unbound.rate") || s=FAIL
        a=$(ibsig "$work/vertex.$name.$m.rate"); b=$(ibsig "$work/vertex.$name.$m-unbound.rate")
        [ "$a" = "$b" ] || { s=FAIL; d="$d bound=$a unbound=$b"; }
        [ "$(sig "$work/vertex.$name.$m.rate")" = "$(sig "$work/vertex.$name.$m-unbound.rate")" ] || { s=FAIL; d="$d code"; }
        report "$name" "$m=unbound" $s "$d ib=${a:0:8}"
        ;;
    esac
  done
  # Rate state alone writes nothing: static / dynamic rates = no rate state.
  if [ -f "$work/vertex.$name.none.rate" ]; then
    for m in static dynamic; do
      [ -f "$work/vertex.$name.$m.rate" ] || continue
      a=$(ibsig "$work/vertex.$name.none.rate"); b=$(ibsig "$work/vertex.$name.$m.rate"); s=ok
      [ "$a" = "$b" ] || s=FAIL
      report "$name" "$m=none" $s " none=${a:0:8} $m=${b:0:8}"
    done
  fi
done < <(vertex_cases | grep -E "^(${ONLY:-.*})\|")

# ---- mesh ----
# name | dr.mesh / dr.frag -D options | task (-/1) | extra env | expected trace (plain and rate)
mesh_cases() { cat <<'EOF'
basic||-||
perprim|-DPERPRIM=1|-||
primid|-DPERPRIM=1 -DPRIMID=1 -DNOCULL=1|-||
perprim_nocull|-DPERPRIM=1 -DNOCULL=1|-||
uniform|-DUNIFORM=1|-||
partial|-DPARTIAL=1|-||
clip|-DCLIP=1|-|RADV_BC250_MESH_ALLOW_POS1=1|
multistore|-DMULTISTORE=1|-||
zero|-DZERO=1|-||
loop|-DLANES=64 -DVERTS=128 -DPRIMS=96|-||
nanite|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-||pieces=3
nanite_p64|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_PERF_PIECE_PRIMS=64|pieces=2
nanite_batch|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_SPLIT_BATCH_PREP=1|pieces=3
direct_read|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_MESH_DIRECT_READ=full|DIRECT READ: applied
direct_read_small|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1|-|RADV_BC250_MESH_DIRECT_READ=full|DIRECT READ: applied
autocull|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1|-|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1|MESH AUTOCULL: applied
autocull_direct|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1|-|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1 RADV_BC250_MESH_DIRECT_READ=full|MESH AUTOCULL: applied;DIRECT READ: applied
autocull_wide|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=2|
compact_lds|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_MESH_COMPACT_LDS=1|
implicit_tris|-DLANES=32 -DVERTS=32 -DPRIMS=32|-|RADV_BC250_MESH_IMPLICIT_TRIS=1|MESH IMPLICIT TRIS: applied
compact|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|RADV_BC250_MESH_COMPACT=1|MESH COMPACT: applied
compact_implicit_direct|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1 -DNOCULL=1|-|RADV_BC250_MESH_COMPACT=1 RADV_BC250_MESH_IMPLICIT_TRIS=1 RADV_BC250_MESH_DIRECT_READ=full RADV_BC250_PERF_PIECE_PRIMS=64|MESH COMPACT: applied;MESH IMPLICIT TRIS: applied;DIRECT READ: applied
compact_autocull|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|RADV_BC250_MESH_COMPACT=1 RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1 RADV_BC250_MESH_DIRECT_READ=full|MESH COMPACT: applied;MESH AUTOCULL: applied
compact_map|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|BC250_COMPACT_VERTICES=true|
amd_route|-DLANES=32 -DVERTS=32 -DPRIMS=32|-|RADV_BC250_MESH_AMD=1|
fast_raw|-DLANES=128 -DVERTS=128 -DPRIMS=126|-|RADV_BC250_MESH_FAST=1|
merge|-DLANES=32 -DVERTS=24 -DPRIMS=8|-|RADV_BC250_MESH_MERGE=1|
merge_perprim|-DLANES=32 -DVERTS=24 -DPRIMS=8 -DPERPRIM=1 -DNOCULL=1|-|RADV_BC250_MESH_MERGE=1|
task|-DTASK=1|1||
task_nanite|-DTASK=1 -DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|1||
game_policy|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|RADV_BC250_EXPOSE_FAST_BINDING=1 RADV_BC250_PERF_PIECE_PRIMS=64 RADV_BC250_SCRATCH_REUSE=1 RADV_BC250_MESH_DIRECT_READ=full RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=1|
lines|-DLINES=1|-||
points|-DPOINTS=1|-||
EOF
}

compile_mesh() { # defines task
  local defs=$1 task=$2
  tag=$(echo "m $defs" | tr -c 'A-Za-z0-9\n' '_')
  if [ ! -f "$work/$tag.rate.mesh.spv" ]; then
    G -S mesh $defs -o "$work/$tag.plain.mesh.spv" "$here/../direct-read/dr.mesh" || return 1
    G -S mesh $defs -DVRS=1 -o "$work/$tag.rate.mesh.spv" "$here/../direct-read/dr.mesh" || return 1
    G -S frag $defs -o "$work/$tag.plain.frag.spv" "$here/../direct-read/dr.frag" || return 1
    G -S frag $defs -DREAD_RATE=1 -o "$work/$tag.rate.frag.spv" "$here/../direct-read/dr.frag" || return 1
  fi
  tspv=-
  if [ "$task" = 1 ]; then
    tspv=$work/dr.task.spv; [ -f "$tspv" ] || G -S task -o "$tspv" "$here/../direct-read/dr.task" || return 1
  fi
}

while IFS='|' read -r name defs task extra want; do
  [ -z "$name" ] && continue
  compile_mesh "$defs" "$task" || { report "$name" compile FAIL ""; continue; }
  P=$work/$tag.plain; R=$work/$tag.rate
  for m in none dynamic attachment; do
    run "$work/mesh.$name.$m.plain" "$ICD" $ON $extra ./vrs mesh $m "$P.mesh.spv" "$P.frag.spv" "$tspv"
    run "$work/mesh.$name.$m.rate" "$ICD" $ON $extra ./vrs mesh $m "$R.mesh.spv" "$R.frag.spv" "$tspv"
    compare "$name" "$m" "$work/mesh.$name.$m.rate" "$work/mesh.$name.$m.plain" \
      "stage=mesh stripped_shading_rate_outputs=1;stage=fragment stripped_shading_rate_outputs=0 zeroed_shading_rate_reads=1;$want"
    a=$(routes "$work/mesh.$name.$m.plain"); b=$(routes "$work/mesh.$name.$m.rate"); s=ok
    { [ "$a" = "$b" ] && grep -q '^BC250 MESH' "$work/mesh.$name.$m.rate"; } || s=FAIL
    report "$name" "$m=routes" $s " $(grep -cE '^BC250 MESH' "$work/mesh.$name.$m.rate") trace lines, plain=$a rate=$b"
  done
  a=$(ibsig "$work/mesh.$name.none.rate"); b=$(ibsig "$work/mesh.$name.dynamic.rate"); s=ok
  [ "$a" = "$b" ] || s=FAIL
  report "$name" "dynamic=none" $s " none=${a:0:8} dynamic=${b:0:8}"
  run "$work/mesh.$name.attachment-unbound.rate" "$ICD" $ON $extra ./vrs mesh attachment-unbound "$R.mesh.spv" "$R.frag.spv" "$tspv"
  a=$(ibsig "$work/mesh.$name.attachment.rate"); b=$(ibsig "$work/mesh.$name.attachment-unbound.rate"); s=ok; d=
  d=$(ran_ok "$work/mesh.$name.attachment-unbound.rate") || s=FAIL
  [ "$a" = "$b" ] || s=FAIL
  [ "$(sig "$work/mesh.$name.attachment.rate")" = "$(sig "$work/mesh.$name.attachment-unbound.rate")" ] || { s=FAIL; d="$d code"; }
  report "$name" "attachment=unbound" $s "$d bound=${a:0:8} unbound=${b:0:8}"
done < <(mesh_cases | grep -E "^(${ONLY:-.*})\|")

# ---- cache ----
# The policy uses nircache. With a disk cache: (1) the switch on compiles vw.spv (NIR cache miss), (2) a new
# pipeline with vw.spv hits the NIR cache and is still stripped (code = the plain pipeline), (3) the switch on
# compiles v.spv, (4) the switch off must not reuse that NIR (the key differs when the switch is on).
if [ -z "${ONLY:-}" ]; then
  rm -rf "$work/diskcache"; mkdir -p "$work/diskcache"
  DC=(MESA_SHADER_CACHE_DISABLE=false MESA_SHADER_CACHE_DIR="$work/diskcache")
  run "$work/cache.1" "$ICD" $ON "${DC[@]}" ./vrs mono dynamic vw.spv fr.spv
  run "$work/cache.2" "$ICD" $ON "${DC[@]}" ./vrs mono dynamic vw.spv f.spv
  run "$work/cache.3" "$ICD" $ON "${DC[@]}" ./vrs mono none v.spv f.spv
  run "$work/cache.4" "$ICD" $OFF "${DC[@]}" VRS_ALLOW_MISSING=1 ./vrs mono none v.spv f.spv
  s=ok; d=
  for i in 1 2 3 4; do d="$d$(ran_ok "$work/cache.$i")" || s=FAIL; done
  grep -q 'BC250 original NIR: stage=0 cache=miss' "$work/cache.1" || { s=FAIL; d="$d 1:not-miss"; }
  grep -q 'BC250 original NIR: stage=0 cache=hit' "$work/cache.2" || { s=FAIL; d="$d 2:not-hit"; }
  grep -q 'BC250 original NIR: stage=0 cache=miss' "$work/cache.4" || { s=FAIL; d="$d 4:not-miss"; }
  [ "$(sig "$work/cache.2")" = "$(sig "$work/vertex.vs.dynamic.plain")" ] || { s=FAIL; d="$d 2:code"; }
  [ "$(ibsig "$work/cache.2")" = "$(ibsig "$work/vertex.vs.dynamic.plain")" ] || { s=FAIL; d="$d 2:ib"; }
  [ "$(sig "$work/cache.3")" = "$(sig "$work/cache.4")" ] || { s=FAIL; d="$d 3/4:code"; }
  report cache nircache-disk $s "$d"
fi

# ---- identical (switch off vs the base build) ----
if [ -n "$OLDICD" ]; then
  if [ -z "${ONLY:-}" ]; then
    shim "$OLDICD" env ./vrs features > "$work/features.old.out" 2>&1
    a=$(grep '^EXT=' "$work/features.old.out" | sed 's/ SBUUID=.*//'); b=$(grep '^EXT=' "$work/features.off.out" | sed 's/ SBUUID=.*//'); s=ok
    # A base build that already has the no-op VRS (its default exposes it) is compared with this build's default.
    case "$a" in EXT=1*) b=$(grep '^EXT=' "$work/features.default.out" | sed 's/ SBUUID=.*//') ;; esac
    [ -n "$a" ] && [ "$a" = "$b" ] || s=FAIL
    report identical features $s " $a"
  fi
  while IFS='|' read -r name env api modes plain rate want; do
    [ -z "$name" ] && continue
    [ "$api" = mono ] || [ "$api" = gpl ] || [ "$api" = eso ] || continue
    run "$work/ident.$name.old" "$OLDICD" $env VRS_ALLOW_MISSING=1 ./vrs $api none $plain f.spv
    run "$work/ident.$name.new" "$ICD" $OFF $env VRS_ALLOW_MISSING=1 ./vrs $api none $plain f.spv
    run "$work/ident.$name.on" "$ICD" $ON $env ./vrs $api none $plain f.spv
    s=ok; d=$(ran_ok "$work/ident.$name.old") || s=FAIL; d="$d$(ran_ok "$work/ident.$name.new")" || s=FAIL
    d="$d$(ran_ok "$work/ident.$name.on")" || s=FAIL
    [ "$(sig "$work/ident.$name.old")" = "$(sig "$work/ident.$name.new")" ] || { s=FAIL; d="$d code"; }
    [ "$(ibsig "$work/ident.$name.old")" = "$(ibsig "$work/ident.$name.new")" ] || { s=FAIL; d="$d ib"; }
    # The default (on), plain shaders: same code and command stream as off.
    [ "$(sig "$work/ident.$name.on")" = "$(sig "$work/ident.$name.new")" ] || { s=FAIL; d="$d on-code"; }
    [ "$(ibsig "$work/ident.$name.on")" = "$(ibsig "$work/ident.$name.new")" ] || { s=FAIL; d="$d on-ib"; }
    report "$name" identical $s "$d code=$(sig "$work/ident.$name.new" | cut -c1-8) ib=$(ibsig "$work/ident.$name.new" | cut -c1-8)"
  done < <(vertex_cases | grep -E "^(${ONLY:-.*})\|" | awk -F'|' '!seen[$2 $3 $5]++')
  while IFS='|' read -r name defs task extra want; do
    [ -z "$name" ] && continue
    compile_mesh "$defs" "$task" || continue
    P=$work/$tag.plain
    # Switches the base build does not have (RADV_BC250_MESH_COMPACT) are compared switched off.
    idextra=${extra//RADV_BC250_MESH_COMPACT=1/}
    run "$work/ident.m.$name.old" "$OLDICD" $idextra VRS_ALLOW_MISSING=1 ./vrs mesh none "$P.mesh.spv" "$P.frag.spv" "$tspv"
    run "$work/ident.m.$name.new" "$ICD" $OFF $idextra VRS_ALLOW_MISSING=1 ./vrs mesh none "$P.mesh.spv" "$P.frag.spv" "$tspv"
    s=ok; d=$(ran_ok "$work/ident.m.$name.old") || s=FAIL; d="$d$(ran_ok "$work/ident.m.$name.new")" || s=FAIL
    if { [ "$name" = clip ] && grep -q 'RADV_BC250_MESH_ALLOW_POS1=1' <<< "$extra"; } || [ "$name" = points ]; then
      # A clip distance or a point size (two position vectors): the GFX10 fully-culled dummy vertex now exports both (b1f639e),
      # so code and command stream differ from the base build by design; the default-on check below still holds.
      d="$d (base: expected dummy change)"
    else
      [ "$(sig "$work/ident.m.$name.old")" = "$(sig "$work/ident.m.$name.new")" ] || { s=FAIL; d="$d code"; }
      [ "$(ibsig "$work/ident.m.$name.old")" = "$(ibsig "$work/ident.m.$name.new")" ] || { s=FAIL; d="$d ib"; }
    fi
    # The default (on), plain shaders: same code and command stream as off.
    if [ "$idextra" = "$extra" ]; then
      [ "$(sig "$work/mesh.$name.none.plain")" = "$(sig "$work/ident.m.$name.new")" ] || { s=FAIL; d="$d on-code"; }
      [ "$(ibsig "$work/mesh.$name.none.plain")" = "$(ibsig "$work/ident.m.$name.new")" ] || { s=FAIL; d="$d on-ib"; }
    fi
    report "$name" identical $s "$d code=$(sig "$work/ident.m.$name.new" | cut -c1-8) ib=$(ibsig "$work/ident.m.$name.new" | cut -c1-8)"
  done < <(mesh_cases | grep -E "^(${ONLY:-.*})\|")
fi

# ---- control (GFX10.3: the checks see real VRS state) ----
if [ -z "${ONLY:-}" ] && [ "${CONTROL:-1}" = 1 ]; then
  for m in attachment attachment-unbound attachment-nodepth attachment-nodepth-unbound; do
    bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
      --unshare-pid --die-with-parent -- env -i PATH=/usr/bin HOME=/tmp LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=navi21 \
      VK_DRIVER_FILES="$ICD" VK_ICD_FILENAMES="$ICD" MESA_SHADER_CACHE_DISABLE=true VRS_CONTROL_DEVICE=NAVI21 \
      RADV_DEBUG=dumpibs ./vrs mono $m vw.spv fr.spv > "$work/control.$m" 2>&1
  done
  for m in attachment attachment-nodepth; do
    s=ok; d=$(ran_ok "$work/control.$m") || s=FAIL
    ck=$(python3 "$here/check_ib.py" "$work/control.$m") && { s=FAIL; d="$d checks-did-not-fire"; }
    echo "$ck" | grep -q 'reg=PA_CL_VRS_CNTL' || { s=FAIL; d="$d no-PA_CL_VRS_CNTL"; }
    echo "$ck" | grep -q 'reg=GE_VRS_RATE' || { s=FAIL; d="$d no-GE_VRS_RATE"; }
    a=$(ibsig "$work/control.$m"); b=$(ibsig "$work/control.$m-unbound")
    [ "$a" != "$b" ] || { s=FAIL; d="$d bound=unbound"; }
    report control-navi21 "$m" $s "$d [$(echo "$ck" | sed 's/.*vrs_writes=/vrs_writes=/')] dispatches bound/unbound=$(python3 "$here/check_ib.py" "$work/control.$m" | sed -n 's/.*gfx_dispatches=\([0-9]*\).*/\1/p')/$(python3 "$here/check_ib.py" "$work/control.$m-unbound" | sed -n 's/.*gfx_dispatches=\([0-9]*\).*/\1/p')"
  done
fi

# ---- isa (KEEP only) ----
if [ -n "${KEEP:-}" ] && [ -z "${ONLY:-}" ]; then
  mkdir -p "$work/isa"
  isa() { # name args...
    local name=$1; shift
    shim "$ICD" env $ON RADV_DEBUG=shaders,spirv "$@" > "$work/isa/$name.txt" 2>&1
  }
  isa vs_plain ./vrs mono dynamic v.spv f.spv
  isa vs_rate ./vrs mono dynamic vw.spv fr.spv
  isa gs_rate ./vrs mono dynamic v.spv gw.spv fr.spv
  compile_mesh "-DPERPRIM=1" - && {
    isa mesh_plain ./vrs mesh dynamic "$work/$tag.plain.mesh.spv" "$work/$tag.plain.frag.spv" -
    isa mesh_rate ./vrs mesh dynamic "$work/$tag.rate.mesh.spv" "$work/$tag.rate.frag.spv" -
  }
  compile_mesh "-DTASK=1 -DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1" 1 && \
    isa task_nanite_rate env RADV_BC250_MESH_DIRECT_READ=full ./vrs mesh attachment "$work/$tag.rate.mesh.spv" "$work/$tag.rate.frag.spv" "$tspv"
  # Control: the same rate shaders on the shim's NAVI21 (GFX10.3) keep the output and the read.
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -i PATH=/usr/bin HOME=/tmp LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=navi21 \
    VK_DRIVER_FILES="$ICD" VK_ICD_FILENAMES="$ICD" MESA_SHADER_CACHE_DISABLE=true VRS_CONTROL_DEVICE=NAVI21 \
    RADV_DEBUG=shaders,spirv ./vrs mono dynamic vw.spv fr.spv > "$work/isa/control_navi21_vs_rate.txt" 2>&1
  : > "$work/isa/grep.txt"
  for f in "$work"/isa/*.txt; do
    [ "$(basename "$f")" = grep.txt ] && continue
    # SPIR-V: the shading rate built-ins the application uses. NIR/ISA (every line but the SPIR-V
    # disassembly and the trace): named shading rate leftovers; the second position export that carries
    # the per-vertex rate (exp ... pos1, VS/GS); the fragment shader's rate unpack from the ancillary
    # VGPR (v_bfe_u32 x, 2, 2 / x, 4, 2).
    echo "$(basename "$f" .txt): spirv_builtins=$(grep -cE 'BuiltIn (PrimitiveShadingRateKHR|ShadingRateKHR)' "$f") nir_isa_shading_rate=$(grep -v 'BC250 VRS NOOP' "$f" | grep -vE '(^|[[:space:]=])Op[A-Z]' | grep -ciE 'shading_rate') rate_pos_export=$(grep -cE 'exp .* pos1$' "$f") ancillary_rate_unpack=$(grep -cE 'v_bfe_u32 .*, [24], 2$' "$f") vrs_noop_trace=$(grep -c 'BC250 VRS NOOP' "$f")" >> "$work/isa/grep.txt"
  done
  s=ok
  grep -E '^(vs|gs|mesh|task_nanite)_rate: ' "$work/isa/grep.txt" | \
    grep -qv 'nir_isa_shading_rate=0 rate_pos_export=0 ancillary_rate_unpack=0 ' && s=FAIL
  grep -E '^control_navi21_vs_rate: ' "$work/isa/grep.txt" | grep -qE 'rate_pos_export=0 |ancillary_rate_unpack=0 ' && s=FAIL
  report isa grep $s " $(tr '\n' ';' < "$work/isa/grep.txt")"
  echo "isa dumps in $work/isa"
fi

echo "vrs-noop: $total checks, $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail

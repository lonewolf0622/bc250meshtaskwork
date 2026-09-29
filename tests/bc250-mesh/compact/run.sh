#!/usr/bin/env bash
# RADV_BC250_MESH_COMPACT regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; pipe also refuses any other device. Each case creates one Mesh
# (+Task) pipeline with the launcher policy (post-Mesh VGT flush off), NIR_DEBUG=validate
# ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1, records a direct, an indirect and an
# indirect-count Mesh draw and submits them to the shim. It is compiled twice with the lowered NGG Mesh
# NIR dumped (BC250_MESH_NIR_DUMP): the reference (switch off; the case's common switches, e.g.
# RADV_BC250_MESH_DIRECT_READ) and the candidate (RADV_BC250_MESH_COMPACT=1 plus the case's candidate
# switches, e.g. RADV_BC250_MESH_AUTOCULL). The CPU oracle (../direct-read/mesh_oracle.c --compact,
# with --autocull when the candidate culls) then runs every comparable dump pair for SEEDS seeds,
# cycling through its index generators (application data, random, degenerate, strips with backjumps,
# unreferenced vertices, few vertices, grid, fan, descending, top vertex only last, invalid indices):
#   safety on every candidate run (every exported vertex referenced, backjump <= W, V'/P' within
#   the lanes and the maxima, no null primitive, GS_ALLOC_REQ consistent with the exports), and
#   equality of the exported triangles (every corner's exports bit-exact, per-primitive payload at the
#   owned corners) with the reference wherever the reference is defined.
#
# Checked per case: pipeline result and submission equal to the reference, no NIR/ACO validation error,
# the expected BC250 MESH COMPACT trace, oracle PASS for every pair. With OLDICD=<icd of the base build>,
# the switch-off shaders (variable unset and =0) of every admitted case are byte-identical to that build.
# Every oracle run also recovers the barycentric vertex order of each candidate primitive as the BC250
# fragment shader does (--bary-rot, both provoking conventions, every cyclic rotation); the bary* cases
# read gl_BaryCoordEXT, so their Mesh shaders also export the two position references, and the
# implicit* cases combine the switch with RADV_BC250_MESH_IMPLICIT_TRIS (the compact map wins).
# Oracle self-checks: --w 0 and --mutate 5 must fail on an applied pair.
# usage: ICD=<radeon_devenv_icd json> BUILD=<mesa build dir, for the oracle> [SHIM=..] [OLDICD=..] [SEEDS=120]
#        [KEEP=<dir>] [ONLY=<case regex>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
BUILD=${BUILD:?BUILD=<mesa build dir> (for mesh_oracle)}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
SEEDS=${SEEDS:-120}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-compact.XXXXXX")}
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
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_MERGE \
    -u RADV_BC250_MESH_AMD -u RADV_BC250_MESH_AUTOCULL -u RADV_BC250_MESH_AUTOCULL_WIDE -u RADV_BC250_MESH_AUTOCULL_ALL \
    -u RADV_BC250_PERF_PIECE_PRIMS -u RADV_BC250_MESH_COMPACT_LDS -u RADV_BC250_MESH_COMPACT \
    -u RADV_BC250_SPLIT_BATCH_PREP -u RADV_BC250_MESH_DIRECT_READ -u BC250_MESH_TIMER -u RADV_DEBUG \
    -u BC250_MESH_NIR_DUMP -u PIPE_PROVOKING -u RADV_BC250_MESH_IMPLICIT_TRIS -u RADV_BC250_NO_BARYCENTRICS \
    -u RADV_BC250_DIAG_BARY_NO_REF \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cd "$work" || exit 2
cc -O1 -Wall -o "$work/pipe" "$here/pipe.c" -lvulkan || exit 2
"$here/../direct-read/build_oracle.sh" "$work/mesh_oracle" >/dev/null || exit 2
G="glslangValidator --target-env vulkan1.3"

fail=0; total=0; oracle_pairs=0; applied_total=0
# name | shader (cmp or wide) | -D options | task (-/1/wide) | attachments | common env | candidate env |
#   want result | candidate patterns (';'-separated, '!' = must not appear, '~' = extended regex)
cases() { cat <<'EOF'
basic|cmp||-|1|||0|MESH COMPACT: applied;owned=0x0;W=8 D=9
grid|cmp|-DGRID=1|-|1|||0|MESH COMPACT: applied
perprim|cmp|-DPERPRIM=1 -DNOCULL=1|-|1|||0|MESH COMPACT: applied;owned=0x1
primid|cmp|-DPERPRIM=1 -DPRIMID=1 -DNOCULL=1|-|1|||0|MESH COMPACT: applied;owned=0x1
perprim_last|cmp|-DPERPRIM=1 -DNOCULL=1|-|1|PIPE_PROVOKING=last||0|MESH COMPACT: applied;owned=0x4;provoking=last
perprim_dynamic|cmp|-DPERPRIM=1 -DNOCULL=1|-|1|PIPE_PROVOKING=dynamic||0|MESH COMPACT: applied;owned=0x5;provoking=dynamic
cullprim|cmp|-DPERPRIM=1|-|1|||0|MESH COMPACT
partial|cmp|-DPARTIAL=1|-|1|||0|MESH COMPACT: applied
zero|cmp|-DZERO=1|-|1|||0|MESH COMPACT: applied
overcount|cmp|-DOVERCOUNT=1|-|1|||0|MESH COMPACT: applied
uniform|cmp|-DUNIFORM=1|-|1|||0|MESH COMPACT: applied
arrayed|cmp|-DARRAYED=1|-|1|||0|MESH COMPACT: applied
clip|cmp|-DCLIP=1|-|1|RADV_BC250_MESH_ALLOW_POS1=1||0|MESH COMPACT: applied
multistore|cmp|-DMULTISTORE=1|-|1|||0|MESH COMPACT: applied
idxcomp|cmp|-DIDXCOMP=1|-|1|||0|MESH COMPACT: applied
task|cmp|-DTASK=1|1|1|||0|MESH COMPACT: applied
small32pp|cmp|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|1|||0|MESH COMPACT: applied;owned=0x1;wave=32
v3p|cmp|-DLANES=32 -DVERTS=96 -DPRIMS=32|-|1|||0|MESH COMPACT: applied
p1|cmp|-DLANES=32 -DVERTS=3 -DPRIMS=1|-|1|||0|MESH COMPACT: applied;P=1
loop96|cmp|-DLANES=64 -DVERTS=128 -DPRIMS=96|-|1|||0|MESH COMPACT: applied
nanite|cmp|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1 -DNOCULL=1|-|1|||0|MESH COMPACT: applied;table=256
nanite_p64|cmp|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1 -DNOCULL=1 -DGRID=1|-|1|RADV_BC250_PERF_PIECE_PRIMS=64||0|MESH COMPACT: applied;P=64
lines|cmp|-DLINES=1|-|1|||0|!MESH COMPACT: applied
points|cmp|-DPOINTS=1|-|1|||0|!MESH COMPACT: applied
cmap|cmp|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|1|BC250_COMPACT_VERTICES=true||0|compact_map=1;!MESH COMPACT: applied
dr1|cmp||-|1|RADV_BC250_MESH_DIRECT_READ=1||0|MESH COMPACT: applied;direct=1
drfull|cmp|-DGRID=1|-|1|RADV_BC250_MESH_DIRECT_READ=full||0|MESH COMPACT: applied;direct=1
drfull_small32pp|cmp|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|1|RADV_BC250_MESH_DIRECT_READ=full||0|MESH COMPACT: applied;direct=1;owned=0x1
drfull_nanite|cmp|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1 -DNOCULL=1|-|1|RADV_BC250_MESH_DIRECT_READ=full RADV_BC250_PERF_PIECE_PRIMS=64||0|MESH COMPACT: applied;direct=1
compact_lds|cmp||-|1|RADV_BC250_MESH_COMPACT_LDS=1||0|MESH COMPACT: applied
autocull_all|cmp|-DGRID=1|-|1||RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1|0|MESH COMPACT: applied;autocull=1
autocull_small32pp|cmp|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|1||RADV_BC250_MESH_AUTOCULL=1|0|MESH COMPACT: applied;autocull=1;owned=0x1
autocull_wide|cmp||-|1||RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=1|0|MESH COMPACT: applied;autocull=1
autocull_wide_drfull|cmp|-DGRID=1|-|1|RADV_BC250_MESH_DIRECT_READ=full|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=1|0|MESH COMPACT: applied;autocull=1;direct=1
autocull_nanite_p64|cmp|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1 -DNOCULL=1 -DGRID=1|-|1|RADV_BC250_PERF_PIECE_PRIMS=64 RADV_BC250_MESH_DIRECT_READ=full|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=1|0|MESH COMPACT: applied
wide|wide||-|5|||0|MESH COMPACT
wide_drfull|wide||-|5|RADV_BC250_MESH_DIRECT_READ=full||0|MESH COMPACT: applied
wide_task|wide|-DTASK=1|wide|5|RADV_BC250_MESH_DIRECT_READ=full||0|MESH COMPACT: applied
implicit|cmp|-DGRID=1|-|1|RADV_BC250_MESH_IMPLICIT_TRIS=1||0|MESH COMPACT: applied;MESH IMPLICIT TRIS: applied
implicit_cand|cmp|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|1||RADV_BC250_MESH_IMPLICIT_TRIS=1|0|MESH COMPACT: applied;MESH IMPLICIT TRIS: applied;owned=0x1
implicit_drfull_autocull|cmp|-DGRID=1|-|1|RADV_BC250_MESH_IMPLICIT_TRIS=1 RADV_BC250_MESH_DIRECT_READ=full|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=1|0|MESH COMPACT: applied;MESH IMPLICIT TRIS: applied;autocull=1;direct=1
implicit_nanite_p64|cmp|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1 -DNOCULL=1 -DGRID=1|-|1|RADV_BC250_PERF_PIECE_PRIMS=64 RADV_BC250_MESH_DIRECT_READ=full RADV_BC250_MESH_IMPLICIT_TRIS=1||0|MESH COMPACT: applied;MESH IMPLICIT TRIS: applied
implicit_cmap|cmp|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|1|BC250_COMPACT_VERTICES=true RADV_BC250_MESH_IMPLICIT_TRIS=1||0|compact_map=1;!MESH COMPACT: applied;!MESH IMPLICIT TRIS: applied
bary|cmp|-DBARY=1|-|1|||0|MESH COMPACT: applied;rotation from reference producer=MESA_SHADER_MESH
bary_grid_implicit|cmp|-DBARY=1 -DGRID=1|-|1|RADV_BC250_MESH_IMPLICIT_TRIS=1||0|MESH COMPACT: applied;MESH IMPLICIT TRIS: applied;rotation from reference producer=MESA_SHADER_MESH
bary_perprim|cmp|-DBARY=1 -DPERPRIM=1 -DNOCULL=1|-|1|||0|MESH COMPACT: applied;owned=0x1;rotation from reference producer=MESA_SHADER_MESH raw=31 flat=30 raster_vertices=3 provoking=first
bary_perprim_last|cmp|-DBARY=1 -DPERPRIM=1 -DNOCULL=1|-|1|PIPE_PROVOKING=last||0|MESH COMPACT: applied;owned=0x4;rotation from reference producer=MESA_SHADER_MESH raw=31 flat=30 raster_vertices=3 provoking=last
bary_perprim_dynamic|cmp|-DBARY=1 -DPERPRIM=1 -DNOCULL=1|-|1|PIPE_PROVOKING=dynamic||0|MESH COMPACT: applied;owned=0x5;provoking=dynamic;rotation from reference producer=MESA_SHADER_MESH raw=31 flat=30 raster_vertices=3 provoking=dynamic
bary_small32pp_drfull_autocull|cmp|-DBARY=1 -DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|1|RADV_BC250_MESH_DIRECT_READ=full RADV_BC250_MESH_IMPLICIT_TRIS=1|RADV_BC250_MESH_AUTOCULL=1|0|MESH COMPACT: applied;autocull=1;direct=1;owned=0x1;rotation from reference
bary_nanite_p64|cmp|-DBARY=1 -DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1 -DNOCULL=1 -DGRID=1|-|1|RADV_BC250_PERF_PIECE_PRIMS=64 RADV_BC250_MESH_DIRECT_READ=full||0|MESH COMPACT: applied;P=64;rotation from reference producer=MESA_SHADER_MESH
bary_task|cmp|-DBARY=1 -DTASK=1|1|1|||0|MESH COMPACT: applied;rotation from reference producer=MESA_SHADER_MESH
EOF
}

check_patterns() { # file, patterns -> prints missing/unexpected, returns 1 on failure
  local out=$1 pats=$2 bad=0 IFS=';'
  for p in $pats; do
    [ -z "$p" ] && continue
    case "$p" in
      '!~'*) grep -qE -- "${p:2}" "$out" && { echo -n " unexpected[${p:2}]"; bad=1; } ;;
      '!'*) grep -qF -- "${p:1}" "$out" && { echo -n " unexpected[${p:1}]"; bad=1; } ;;
      '~'*) grep -qE -- "${p:1}" "$out" || { echo -n " missing[${p:1}]"; bad=1; } ;;
      *) grep -qF -- "$p" "$out" || { echo -n " missing[$p]"; bad=1; } ;;
    esac
  done
  return $bad
}

compile_case() { # shader (cmp: this directory, wide: ../piece-ceiling) defines task
  local shader=$1 defs=$2 task=$3 dir=$here tag
  [ "$shader" = wide ] && dir=$here/../piece-ceiling
  tag=$(echo "$shader $defs" | tr -c 'A-Za-z0-9\n' '_')
  mesh=$work/$tag.mesh.spv; frag=$work/$tag.frag.spv; tspv=-
  if [ ! -f "$mesh" ]; then
    $G -S mesh $defs -o "$mesh" "$dir/$shader.mesh" >/dev/null || return 1
    $G -S frag $defs -o "$frag" "$dir/$shader.frag" >/dev/null || return 1
  fi
  case "$task" in
    1) tspv=$work/cmp.task.spv; [ -f "$tspv" ] || $G -S task -o "$tspv" "$here/cmp.task" >/dev/null || return 1 ;;
    wide) tspv=$work/wide.task.spv; [ -f "$tspv" ] || $G -S task -o "$tspv" "$here/../piece-ceiling/wide.task" >/dev/null || return 1 ;;
  esac
}

oracle_dir() { # name oracle-flags -> prints summary, returns 1 on failure
  local name=$1 flags=$2 bad=0 n=0 line c r applied=0 oout rot rotp=0 rota=0
  local ref=$work/dump/$name/ref cand=$work/dump/$name/cand
  local refs=("$ref"/*.nir) cands=("$cand"/*.nir)
  [ -e "${refs[0]}" ] && [ -e "${cands[0]}" ] || { echo -n " no-dumps"; return 1; }
  pairs=()
  for c in "${cands[@]}"; do
    r=$ref/$(basename "$c")
    [ -f "$r" ] && pairs+=("$r|$c")
  done
  [ "$(basename "${refs[-1]}")" != "$(basename "${cands[-1]}")" ] && pairs+=("${refs[-1]}|${cands[-1]}")
  for pr in "${pairs[@]}"; do
    r=${pr%%|*}; c=${pr##*|}
    oout=$("$work/mesh_oracle" --compact --bary-rot $flags --seeds "$SEEDS" "$r" "$c")
    line=$(echo "$oout" | tail -1)
    rot=$(echo "$oout" | sed -n 's/^ORACLE BARY-ROT primitives=\([0-9]*\) ambiguous_no_fragments=\([0-9]*\).*/\1 \2/p')
    [ -n "$rot" ] && { rotp=$((rotp + ${rot% *})); rota=$((rota + ${rot#* })); }
    n=$((n + 1))
    case "$line" in
      "ORACLE PASS"*) ;;
      *) echo -n " oracle[$(basename "$c"): ${line:0:300}]"; bad=1 ;;
    esac
    case "$line" in *" applied=1 "*) applied=$((applied + 1)) ;; esac
    stats=$(echo "$line" | grep -o 'max_backjump=[0-9]* .*verts_per_tri=[0-9.]*' | sed 's/ dummy=[0-9]*//; s/ tris=[0-9]*//; s/ verts_expanded=[0-9]*//; s/ verts_compact=[0-9]*//')
  done
  echo -n " oracle=$n applied_pairs=$applied ${stats:-} bary_rot=$rotp/ambiguous=$rota"
  return $bad
}

while IFS='|' read -r name shader defs task rts common cand want pats; do
  [ -z "$name" ] && continue
  compile_case "$shader" "$defs" "$task" || { echo "$name: shader compile failed"; fail=1; continue; }
  for v in ref cand; do
    d=$work/dump/$name/$v; rm -rf "$d"; mkdir -p "$d"
    if [ $v = ref ]; then
      shim "$ICD" env $common BC250_MESH_NIR_DUMP="$d" ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.$v.out" 2>&1
    else
      shim "$ICD" env $common $cand RADV_BC250_MESH_COMPACT=1 BC250_MESH_NIR_DUMP="$d" \
        ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.$v.out" 2>&1
    fi
  done
  status=ok; detail=
  for v in ref cand; do
    out=$work/$name.$v.out
    got=$(sed -n 's/^PIPELINE_RESULT=//p' "$out" | tail -1)
    [ "${got:-none}" = "$want" ] || { status=FAIL; detail="$detail $v:result=${got:-none}"; }
    if [ "$want" = 0 ] && ! grep -q '^SUBMIT_OK' "$out"; then status=FAIL; detail="$detail $v:no-submit"; fi
    if grep -qiE 'validation failed|NIR_VALIDATE|assert|error:|^FAIL ' "$out"; then status=FAIL; detail="$detail $v:validation"; fi
  done
  grep -q 'MESH COMPACT' "$work/$name.ref.out" && { status=FAIL; detail="$detail ref:compact-trace"; }
  msg=$(check_patterns "$work/$name.cand.out" "$pats") || status=FAIL
  detail="$detail$msg"
  flags=""
  case "$cand" in *AUTOCULL=1*) flags="--autocull --states 4" ;; esac
  if [ "$want" = 0 ]; then
    msg=$(oracle_dir "$name" "$flags") || status=FAIL
    detail="$detail$msg"
    n=$(echo "$msg" | sed -n 's/.* oracle=\([0-9]*\) applied_pairs=\([0-9]*\).*/\1/p')
    a=$(echo "$msg" | sed -n 's/.* oracle=\([0-9]*\) applied_pairs=\([0-9]*\).*/\2/p')
    oracle_pairs=$((oracle_pairs + ${n:-0})); applied_total=$((applied_total + ${a:-0}))
  fi
  tr=$(grep -o 'MESH COMPACT: [a-z ]*applied.*' "$work/$name.cand.out" | tail -1 | sed 's/MESH COMPACT: //; s/ W=8 D=9//')
  total=$((total + 1)); [ $status = ok ] || fail=1
  printf '%-22s %-4s %s |%s\n' "$name" "$status" "${tr:-no trace}" "$detail"
  if [ -n "$OLDICD" ] && [ "$want" = 0 ]; then
    sig() { grep -E '^BC250POLICY(CODE)? ' "$1" | sed 's/ va=[0-9a-f]*//' | sort | sha1sum | cut -c1-16; }
    shim "$OLDICD" env $common BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.old" 2>&1
    shim "$ICD" env $common BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.new" 2>&1
    shim "$ICD" env $common RADV_BC250_MESH_COMPACT=0 BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.new0" 2>&1
    a=$(sig "$work/$name.old"); b=$(sig "$work/$name.new"); c=$(sig "$work/$name.new0"); s=ok
    why=
    if [ "$b" = "$c" ] && grep -q '^BC250POLICY ' "$work/$name.new" && { [ "$name" = clip ] || [ "$name" = points ]; } &&
       [ "$a" != "$b" ]; then
      # Two position vectors (clip distance, point size): the GFX10 fully-culled dummy vertex now exports both
      # (b1f639e), so these differ from the base build by design.
      why=" (expected: the fully-culled dummy exports both declared position vectors)"
    elif ! { [ "$a" = "$b" ] && [ "$a" = "$c" ] && grep -q '^BC250POLICY ' "$work/$name.old"; }; then
      s=FAIL; fail=1
    fi
    total=$((total + 1)); printf '%-22s %-4s identical-to-base base=%s off=%s zero=%s%s\n' "$name" "$s" "$a" "$b" "$c" "$why"
  fi
done < <(cases | grep -E "^(${ONLY:-.*})\|")

# Oracle self-checks on the basic pair: a wrong window (W=0) and a wrong expected order must fail.
if [ -z "${ONLY:-}" ]; then
  r=("$work/dump/basic/ref/"*.nir); c=("$work/dump/basic/cand/"*.nir)
  for f in "--w 0" "--mutate 5"; do
    line=$("$work/mesh_oracle" --compact $f --seeds 24 "${r[0]}" "${c[0]}" | tail -1)
    total=$((total + 1))
    case "$line" in
      "ORACLE FAIL"*) printf '%-22s ok   (%s)\n' "self-check $f" "${line:0:100}" ;;
      *) printf '%-22s FAIL (%s)\n' "self-check $f" "${line:0:100}"; fail=1 ;;
    esac
  done
fi
echo "cases=$total failed=$fail oracle_pairs=$oracle_pairs applied_pairs=$applied_total seeds=$SEEDS${ONLY:+ (ONLY=$ONLY)}"
exit $fail

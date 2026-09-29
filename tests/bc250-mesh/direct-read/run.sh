#!/usr/bin/env bash
# RADV_BC250_MESH_DIRECT_READ regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; pipe also refuses any other device. Each case creates one
# Mesh (+Task) pipeline with the base driver launcher policy (post-Mesh VGT_FLUSH off, as the games
# run), NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1, records a direct, an
# indirect and an indirect-count Mesh draw and submits them to the shim. It is compiled three times:
# switch off (reference), RADV_BC250_MESH_DIRECT_READ=1 and =full (candidates), with the lowered NGG
# Mesh NIR dumped (BC250_MESH_NIR_DUMP). The CPU export oracle (mesh_oracle.c) then executes the
# reference and each candidate workgroup for SEEDS random inputs and compares every export
# (GS_ALLOC_REQ, positions, parameters, primitives, side effects).
#
# Checked per case: pipeline result and submission equal to the reference, no NIR/ACO validation
# error, the expected BC250 MESH DIRECT READ trace (applied / not applied, uniform cells, 16-bit
# indices), and oracle PASS for every comparable pair of compile attempts (a candidate that fits where
# the reference needed the LDS fit retry has other pieces: that pair is NOT-COMPARABLE, allowed only
# where the table says so; its first attempt, which the reference compiled with the Mesh scratch
# ring, is compared instead). Each part alone is checked on two cases. With OLDICD=<icd of the base
# build>, the switch-off shaders of every case are byte-identical to the base build
# (BC250_CAPTURE_POLICY_SHADERS code and configuration).
# usage: ICD=<radeon_devenv_icd json> BUILD=<mesa build dir, for the oracle> [SHIM=..] [OLDICD=..] [SEEDS=200]
#        [KEEP=<dir>] [ONLY=<case regex>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
BUILD=${BUILD:?BUILD=<mesa build dir> (for mesh_oracle)}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
SEEDS=${SEEDS:-200}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-direct-read.XXXXXX")}
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
    -u RADV_BC250_MESH_AMD -u RADV_BC250_MESH_AUTOCULL -u RADV_BC250_PERF_PIECE_PRIMS -u RADV_BC250_MESH_COMPACT_LDS \
    -u RADV_BC250_SPLIT_BATCH_PREP -u RADV_BC250_MESH_DIRECT_READ -u BC250_MESH_TIMER -u RADV_DEBUG \
    -u BC250_MESH_NIR_DUMP \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cd "$work" || exit 2
cc -O1 -Wall -o "$work/pipe" "$here/pipe.c" -lvulkan || exit 2
"$here/build_oracle.sh" "$work/mesh_oracle" >/dev/null || exit 2
G="glslangValidator --target-env vulkan1.3"

fail=0; total=0; oracle_pairs=0
# name | shader (dr or wide) | -D options | task (-/1) | attachments | extra env | want result |
#   patterns for =1 | patterns for =full | allow NOT-COMPARABLE (0/1)
# Patterns: ';'-separated substrings, '!' = must not appear, '~' = extended regex.
cases() { cat <<'EOF'
basic|dr||-|1||0|DIRECT READ: applied;~\(([0-9]+) of \1 per-vertex\)|DIRECT READ: applied;index16=1 corner=1;uniform_outputs=0|0
uniform|dr|-DUNIFORM=1|-|1||0|DIRECT READ: applied|uniform_outputs=3;index16=1|0
partial|dr|-DPARTIAL=1|-|1||0|DIRECT READ: applied|DIRECT READ: applied;index16=1|0
arrayed|dr|-DARRAYED=1|-|1||0|DIRECT READ: applied;~\(([0-9]+) of \1 per-vertex\)|DIRECT READ: applied|0
perprim|dr|-DPERPRIM=1|-|1||0|DIRECT READ: applied|DIRECT READ: applied;index16=1|0
primid|dr|-DPERPRIM=1 -DPRIMID=1 -DNOCULL=1|-|1||0|DIRECT READ: applied|DIRECT READ: applied|0
primid_cull|dr|-DPERPRIM=1 -DPRIMID=1|-|1||-8|||0
clip|dr|-DCLIP=1|-|1|RADV_BC250_MESH_ALLOW_POS1=1|0|DIRECT READ: applied;~\(([0-9]+) of \1 per-vertex\)|DIRECT READ: applied|0
multistore|dr|-DMULTISTORE=1|-|1||0|DIRECT READ: applied|DIRECT READ: applied|0
idxcomp|dr|-DIDXCOMP=1|-|1||0|DIRECT READ: applied|index16=1|0
zero|dr|-DZERO=1|-|1||0|DIRECT READ: applied|DIRECT READ: applied|0
lines|dr|-DLINES=1|-|1||0||index16=1|0
points|dr|-DPOINTS=1|-|1||0|DIRECT READ: applied;!~\(([0-9]+) of \1 per-vertex\)|index16=1|0
task|dr|-DTASK=1|1|1||0|DIRECT READ: applied|DIRECT READ: applied|0
loop|dr|-DLANES=64 -DVERTS=128 -DPRIMS=96|-|1||0|DIRECT READ: applied|DIRECT READ: applied|0
uniform_loop|dr|-DUNIFORM=1 -DLANES=32 -DVERTS=64 -DPRIMS=64|-|1||0|DIRECT READ: applied|uniform_outputs=0|0
nanite|dr|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|1||0|pieces=3;DIRECT READ: applied|pieces=3;index16=1|0
nanite_p64|dr|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|1|RADV_BC250_PERF_PIECE_PRIMS=64|0|pieces=2;!Mesh LDS fit|pieces=2;!Mesh LDS fit|0
nanite_batch|dr|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|1|RADV_BC250_SPLIT_BATCH_PREP=1|0|pieces=3|pieces=3|0
nanite_compact|dr|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DPERPRIM=1|-|1|RADV_BC250_MESH_COMPACT_LDS=1|0|DIRECT READ: applied|DIRECT READ: applied|0
small|dr|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1|-|1||0|DIRECT READ: applied|DIRECT READ: applied|0
small_autocull|dr|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1|-|1|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1|0|MESH AUTOCULL: applied;DIRECT READ: applied|MESH AUTOCULL: applied;DIRECT READ: applied|0
basic_autocull|dr||-|1|RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1|0|DIRECT READ: applied|DIRECT READ: applied|0
small_cmap|dr|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DPERPRIM=1 -DNOCULL=1|-|1|BC250_COMPACT_VERTICES=true|0|compact map;DIRECT READ: applied|compact map;DIRECT READ: applied|0
basic_compact|dr||-|1|RADV_BC250_MESH_COMPACT_LDS=1|0|DIRECT READ: applied|DIRECT READ: applied|0
wide|wide||-|5||0|DIRECT READ: applied;!Mesh LDS fit|uniform_outputs=0;!Mesh LDS fit|1
wide_task|wide|-DTASK=1|wide|5||0|DIRECT READ: applied|DIRECT READ: applied|1
wide_fit|wide|-DFIT=1|-|5||0|DIRECT READ: applied|DIRECT READ: applied|0
EOF
}

# Every part alone, on two cases (patterns and oracle).
parts() { cat <<'EOF'
uniform|export
uniform|uniform
uniform|dead
uniform|index16
uniform|corner
uniform|index16,corner
nanite|export
nanite|uniform
nanite|dead
nanite|index16
nanite|corner
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

compile_case() { # shader (dr: this directory, wide: ../piece-ceiling) defines task (-, 1: dr.task, wide: wide.task)
  local shader=$1 defs=$2 task=$3 dir=$here tag
  [ "$shader" = wide ] && dir=$here/../piece-ceiling
  tag=$(echo "$shader $defs" | tr -c 'A-Za-z0-9\n' '_')
  mesh=$work/$tag.mesh.spv; frag=$work/$tag.frag.spv; tspv=-
  if [ ! -f "$mesh" ]; then
    $G -S mesh $defs -o "$mesh" "$dir/$shader.mesh" >/dev/null || return 1
    $G -S frag $defs -o "$frag" "$dir/$shader.frag" >/dev/null || return 1
  fi
  case "$task" in
    1) tspv=$work/dr.task.spv; [ -f "$tspv" ] || $G -S task -o "$tspv" "$here/dr.task" >/dev/null || return 1 ;;
    wide) tspv=$work/wide.task.spv; [ -f "$tspv" ] || $G -S task -o "$tspv" "$here/../piece-ceiling/wide.task" >/dev/null || return 1 ;;
  esac
}

run_variant() { # name variant extra... -> $work/<name>.<variant>.out, dump dir
  local name=$1 variant=$2; shift 2
  local d=$work/dump/$name/$variant
  rm -rf "$d"; mkdir -p "$d"
  if [ "$variant" = off ]; then
    shim "$ICD" env "$@" BC250_MESH_NIR_DUMP="$d" ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.$variant.out" 2>&1
  else
    shim "$ICD" env "$@" RADV_BC250_MESH_DIRECT_READ="$variant" BC250_MESH_NIR_DUMP="$d" \
      ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.$variant.out" 2>&1
  fi
}

oracle_dir() { # name variant allow_nc -> prints summary, returns 1 on failure
  local name=$1 variant=$2 allow=$3 bad=0 n=0 nc=0 line c r
  local ref=$work/dump/$name/off cand=$work/dump/$name/$variant
  local refs=("$ref"/*.nir) cands=("$cand"/*.nir)
  [ -e "${refs[0]}" ] && [ -e "${cands[0]}" ] || { echo -n " no-dumps"; return 1; }
  for c in "${cands[@]}"; do
    r=$ref/$(basename "$c")
    [ -f "$r" ] || continue
    line=$("$work/mesh_oracle" --seeds "$SEEDS" "$r" "$c" | tail -1)
    n=$((n + 1))
    case "$line" in
      "ORACLE PASS"*) ;;
      "ORACLE NOT-COMPARABLE"*) nc=$((nc + 1)); [ "$allow" = 1 ] || { echo -n " not-comparable[$(basename "$c")]"; bad=1; } ;;
      *) echo -n " oracle[$(basename "$c"): $line]"; bad=1 ;;
    esac
  done
  if [ "$(basename "${refs[-1]}")" != "$(basename "${cands[-1]}")" ]; then
    line=$("$work/mesh_oracle" --seeds "$SEEDS" "${refs[-1]}" "${cands[-1]}" | tail -1)
    n=$((n + 1))
    case "$line" in
      "ORACLE PASS"*) ;;
      "ORACLE NOT-COMPARABLE"*) nc=$((nc + 1)); [ "$allow" = 1 ] || { echo -n " not-comparable[admitted]"; bad=1; } ;;
      *) echo -n " oracle[admitted: $line]"; bad=1 ;;
    esac
  fi
  echo -n " oracle=$((n - nc))/$n"
  [ $nc -gt 0 ] && echo -n "(+$nc not comparable)"
  return $bad
}

one() { # name variant want pats allow
  local name=$1 variant=$2 want=$3 pats=$4 allow=$5 status=ok detail=
  local out=$work/$name.$variant.out got ref_got
  got=$(sed -n 's/^PIPELINE_RESULT=//p' "$out" | tail -1)
  [ "${got:-none}" = "$want" ] || { status=FAIL; detail="$detail result=${got:-none}"; }
  if [ "$want" = 0 ] && ! grep -q '^SUBMIT_OK' "$out"; then status=FAIL; detail="$detail no-submit"; fi
  if grep -qiE 'validation failed|NIR_VALIDATE|assert|error:|^FAIL ' "$out"; then status=FAIL; detail="$detail validation"; fi
  local msg
  msg=$(check_patterns "$out" "$pats") || { status=FAIL; }
  detail="$detail$msg"
  if [ "$variant" != off ] && [ "$want" = 0 ]; then
    msg=$(oracle_dir "$name" "$variant" "$allow") || status=FAIL
    detail="$detail$msg"
    local pairs=$(echo "$msg" | sed -n 's/.* oracle=[0-9]*\/\([0-9]*\).*/\1/p')
    oracle_pairs=$((oracle_pairs + ${pairs:-0}))
  fi
  local lds=$(grep -o 'MESH LDS: V=[0-9]* P=[0-9]*.*total=[0-9]*' "$out" | tail -1 | sed 's/MESH LDS: //; s/ vtx_attr=\([0-9]*\)@[0-9]*/ vtx_attr=\1/; s/ prm_attr.*total=/ total=/')
  total=$((total + 1)); [ $status = ok ] || fail=1
  printf '%-15s %-16s %-4s %s%s\n' "$name" "$variant" "$status" "${lds:-}" "$detail"
}

while IFS='|' read -r name shader defs task rts extra want pats1 patsfull allow; do
  [ -z "$name" ] && continue
  compile_case "$shader" "$defs" "$task" || { echo "$name: shader compile failed"; fail=1; continue; }
  run_variant "$name" off $extra
  run_variant "$name" 1 $extra
  run_variant "$name" full $extra
  one "$name" off "$want" "!DIRECT READ" 0
  one "$name" 1 "$want" "$pats1" "$allow"
  one "$name" full "$want" "$patsfull" "$allow"
  if [ -n "$OLDICD" ] && [ "$want" = 0 ]; then
    sig() { grep -E '^BC250POLICY(CODE)? ' "$1" | sed 's/ va=[0-9a-f]*//' | sort | sha1sum | cut -c1-16; }
    shim "$OLDICD" env $extra BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.old" 2>&1
    shim "$ICD" env $extra BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.new" 2>&1
    shim "$ICD" env $extra RADV_BC250_MESH_DIRECT_READ=0 BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" "$rts" > "$work/$name.new0" 2>&1
    a=$(sig "$work/$name.old"); b=$(sig "$work/$name.new"); c=$(sig "$work/$name.new0"); s=ok; why=
    if { [ "$name" = clip ] || [ "$name" = points ]; } && [ "$a" != "$b" ] && [ "$b" = "$c" ] &&
       grep -q '^BC250POLICY ' "$work/$name.new" && grep -q 'MESH POS EXPORTS: format=2 ' "$work/$name.new"; then
      # A clip distance (admitted with RADV_BC250_MESH_ALLOW_POS1=1) or a point size: two position vectors; the GFX10
      # fully-culled dummy vertex now exports both (b1f639e), a difference to the base build by design.
      why=" (expected: the fully-culled dummy exports both declared position vectors)"
    elif ! { [ "$a" = "$b" ] && [ "$a" = "$c" ] && grep -q '^BC250POLICY ' "$work/$name.old"; }; then
      s=FAIL; fail=1
    fi
    total=$((total + 1)); printf '%-15s %-16s %-4s base=%s off=%s zero=%s%s\n' "$name" "identical-to-base" "$s" "$a" "$b" "$c" "$why"
  fi
done < <(cases | grep -E "^(${ONLY:-.*})\|")

while IFS='|' read -r name part; do
  [ -z "$name" ] && continue
  [ -n "${ONLY:-}" ] && continue
  IFS='|' read -r _ shader defs task rts extra want _ _ allow < <(cases | grep "^$name|")
  compile_case "$shader" "$defs" "$task" || { fail=1; continue; }
  run_variant "$name" "$part" $extra
  pats="!unknown part"
  case "$part" in
    export) pats="$pats;DIRECT READ: applied;!DIRECT READ staging" ;;
    uniform) pats="$pats;!DIRECT READ: applied;index16=0" ;;
    dead) pats="$pats;!DIRECT READ: applied;!DIRECT READ staging" ;;
    index16) pats="$pats;index16=1 corner=0;uniform_outputs=0" ;;
    corner) pats="$pats;!DIRECT READ: applied;!DIRECT READ staging" ;;
    index16,corner) pats="$pats;index16=1 corner=1" ;;
  esac
  [ "$name,$part" = "uniform,uniform" ] && pats="$pats;uniform_outputs=3"
  one "$name" "$part" "$want" "$pats" "$allow"
done < <(parts)

# Oracle self-check: two different shaders with the same launch shape must not compare equal.
[ -n "${ONLY:-}" ] && { echo "cases=$total failed=$fail oracle_pairs=$oracle_pairs seeds=$SEEDS (ONLY=$ONLY)"; exit $fail; }
a=("$work/dump/basic/off/"*.nir); b=("$work/dump/multistore/off/"*.nir)
line=$("$work/mesh_oracle" --seeds 20 "${a[0]}" "${b[0]}" | tail -1)
total=$((total + 1))
case "$line" in
  "ORACLE FAIL"*) printf '%-15s %-16s ok   (%s)\n' oracle self-check "${line%% compared*}" ;;
  *) printf '%-15s %-16s FAIL (%s)\n' oracle self-check "$line"; fail=1 ;;
esac

echo "cases=$total failed=$fail oracle_pairs=$oracle_pairs seeds=$SEEDS"
exit $fail

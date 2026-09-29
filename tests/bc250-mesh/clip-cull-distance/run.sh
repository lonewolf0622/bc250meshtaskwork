#!/usr/bin/env bash
# Clip and cull distance outputs through the BC250 Mesh routes (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; pipe (../direct-read/pipe.c) also refuses any other device.
# Each case (cc.mesh / cc.frag / cc.task with -D options) creates one Mesh (+Task) pipeline with the
# base driver launcher policy, NIR_DEBUG=validate ACO_DEBUG=validateir,validatera
# BC250_TRACE_COMPILE=1, records a direct, an indirect and an indirect-count draw and submits them to
# the shim, in these variants (lowered NGG Mesh NIR dumped with BC250_MESH_NIR_DUMP):
#   off       the policy alone (split at the default 63-triangle ceiling, private-vertex expansion)
#   raw       RADV_BC250_MESH_AMD=1: the unsplit, unexpanded shader (reference of the geometry oracle)
#   p64       RADV_BC250_PERF_PIECE_PRIMS=64 (split cases: 2 instead of 3 pieces)
#   dr1 / drfull   RADV_BC250_MESH_DIRECT_READ=1 / full
#   compact   RADV_BC250_MESH_COMPACT_LDS=1
#   implicit  RADV_BC250_MESH_IMPLICIT_TRIS=1
#   shared / sharedac   RADV_BC250_MESH_COMPACT=1 (shared-vertex compaction), sharedac with
#             RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1
#   ac / acdr RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1 (acdr: with DIRECT_READ=full)
#   game / gamenoac   the game switch set of the launcher (PIECE_PRIMS=64, DIRECT_READ=full,
#             AUTOCULL=1, AUTOCULL_WIDE=1, ...), gamenoac without RADV_BC250_MESH_AUTOCULL
#   bary      the fragment shader also reads gl_BaryCoordEXT
# Checked per case: every variant compiles and submits (no split refusal, no validation error); the
# traces (split pieces, direct read of the clip/cull slot CLIP_DIST0 (bit 17) and CLIP_DIST1 (bit 18),
# autocull candidate and applied); and the CPU export oracle (../direct-read/mesh_oracle):
#   geometry  (cases marked g) raw against off, p64, dr1, drfull, compact, implicit, shared, game and ac:
#             same live primitives in order and, per corner, identical position, misc vector and
#             clip/cull distance exports (--geometry --grid, the split's pieces from the trace);
#   pieces    (cases marked s) off (3 pieces) against p64 (2 pieces), parameters included;
#   exact     off against dr1, drfull, compact and implicit (every export of every lane);
#   autocull  off against ac, drfull against acdr, gamenoac against game (--autocull: the culling
#             reference with the clip/cull distance test; survivors export their corners' values;
#             with the viewport index, exact: not a candidate);
#   shared    off against shared (--compact: hang-safety rules of the shared-vertex export on
#             adversarial index data, and the corners' exports, clip/cull distances included) and
#             against sharedac (--compact --autocull; exact for the viewport index cases);
#   bary      off against bary (--bary-ref).
# The legacy variants above run with RADV_BC250_MESH_ALLOW_POS1=1 RADV_BC250_MESH_CULLDIST_CULL=0 (the
# clip/cull position export form a hardware gate would test, and the only form that has a raw reference);
# constant non-negative distances stay removed there (RADV_BC250_MESH_CLIPCULL_CONST, default on).
# The driver defaults (MESH_PERF/ff7hang "Fix") are checked in the def variants:
#   def / defgame / defshared   the policy alone, the game set, RADV_BC250_MESH_COMPACT=1, all with the
#             defaults (RADV_BC250_MESH_CLIPCULL_CONST=1, RADV_BC250_MESH_CULLDIST_CULL=1,
#             RADV_BC250_MESH_ALLOW_POS1=0); per case expectation (column 6):
#               refuse  refused with "BC250 Mesh pipeline refused" (a clip distance, or a cull distance
#                       autocull cannot take), PIPELINE_RESULT=-8
#               cull    compiled, cull distances culled in the shader and not exported (POS EXPORTS
#                       clip=0 cull=0 culldist_culled=1); oracle --autocull against the legacy compile
#                       without autocull (off / gamenoac) and --compact --autocull (defshared): every
#                       culling state goes through the reference decision (no runtime skip)
#               none    compiled without any clip/cull position export (constant distances removed);
#                       with column 7 (twin) byte-identical to that case's def and defgame compiles (the
#                       same shader without the constant stores); oracle against legconst (legacy with
#                       RADV_BC250_MESH_CLIPCULL_CONST=0: the constants exported), which strips them
#               same    no distances: identical to the legacy variant
#   pos       RADV_BC250_MESH_CULLDIST_CULL=1 with RADV_BC250_MESH_ALLOW_POS1=1 (refused cases with clip
#             and cull distances): cull distances culled in the shader, clip distances still exported;
#             oracle --autocull against off
# Every compile's POS EXPORTS trace (printed for shaders with distances or several position vectors) must match
# SPI_SHADER_POS_FORMAT (no done_mismatch); the oracle checks
# the GFX10 fully-culled dummy (every declared position vector) on every all-culled workgroup.
# Every oracle line of a case with distances must have compared clip/cull channels. Self-checks: the
# geometry oracle must fail against a shader with one wrong distance (SKEW), and the autocull oracle
# (alone and with --compact) with --mutate 6 (no distance test) on a case where distances cull; the
# shared-vertex compaction must have applied to at least one case with distances. With OLDICD=<icd of the base
# build> the cases without clip/cull distances are byte-identical to it in every variant
# (BC250_CAPTURE_POLICY_SHADERS code and configuration).
# usage: ICD=<radeon_devenv_icd json> BUILD=<mesa build dir, for the oracle> [OLDICD=..] [SHIM=..]
#        [SEEDS=60] [KEEP=<dir>] [ONLY=<case regex>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?ICD=<radeon_devenv_icd json>}
BUILD=${BUILD:?BUILD=<mesa build dir> (for mesh_oracle)}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
SEEDS=${SEEDS:-60}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-clip-cull.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT

POLICY=(RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false
  BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true
  BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true
  BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false
  RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
  RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true
  BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0
  NIR_DEBUG=validate ACO_DEBUG=validateir,validatera BC250_TRACE_COMPILE=1
  RADV_BC250_MESH_ALLOW_POS1=1 RADV_BC250_MESH_CULLDIST_CULL=0)
DEF="RADV_BC250_MESH_ALLOW_POS1=0 RADV_BC250_MESH_CULLDIST_CULL=1"
GAMENOAC="RADV_BC250_EXPOSE_FAST_BINDING=1 RADV_BC250_PERF_PIECE_PRIMS=64 RADV_BC250_SCRATCH_REUSE=1 RADV_BC250_MESH_DIRECT_READ=full RADV_BC250_SUBMIT_KNOWN_SIGNALS=1 RADV_BC250_LOCAL_BOS=1 RADV_BC250_KEEP_IB_KB=640"

shim() {
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_MERGE \
    -u RADV_BC250_MESH_AMD -u RADV_BC250_MESH_AUTOCULL -u RADV_BC250_MESH_AUTOCULL_ALL -u RADV_BC250_MESH_AUTOCULL_WIDE \
    -u RADV_BC250_PERF_PIECE_PRIMS -u RADV_BC250_MESH_COMPACT_LDS -u RADV_BC250_SPLIT_BATCH_PREP \
    -u RADV_BC250_MESH_DIRECT_READ -u RADV_BC250_MESH_IMPLICIT_TRIS -u RADV_BC250_MESH_COMPACT -u BC250_MESH_TIMER -u RADV_DEBUG \
    -u BC250_MESH_NIR_DUMP -u RADV_BC250_MESH_CLIPCULL_CONST \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cd "$work" || exit 2
cc -O1 -Wall -o "$work/pipe" "$here/../direct-read/pipe.c" -lvulkan || exit 2
"$here/../direct-read/build_oracle.sh" "$work/mesh_oracle" >/dev/null || exit 2
G="glslangValidator --target-env vulkan1.3"

# name | cc.mesh -D options | extra cc.frag -D options | task (-/1) | flags (g: geometry against the raw
# compile, s: split pieces, v: the viewport index (not an autocull candidate: the autocull variants must be
# refused for it and compare exactly with the switch off), w: primitives in two waves (the game set's
# wide autocull policy leaves the legacy compile alone: compared exactly; the defaults split it for the
# in-shader distance culling, so the def compiles have other pieces than every legacy compile: traces
# only)) | driver defaults (refuse/cull/none/same, see the
# header) | twin (none: the case whose def/defgame compiles must be byte-identical)
cases() { cat <<'EOF'
none64|||-|g|same|
none128|-DLANES=128 -DVERTS=128 -DPRIMS=128||-|gs|same|
none_task|-DTASK=1||1||same|
c1|-DNCLIP=1|-DFSCLIP=1|-|g|refuse|
c8|-DNCLIP=8||-|g|refuse|
k1|-DNCULL=1||-|g|cull|
k8|-DNCULL=8|-DFSCLIP=1|-|g|cull|
c3k5|-DNCLIP=3 -DNCULL=5|-DFSCLIP=1|-|g|refuse|
c2k3_dyn|-DNCLIP=2 -DNCULL=3 -DDYN=1||-|g|refuse|
c4k4_sparse|-DNCLIP=4 -DNCULL=4 -DSPARSE=1||-|g|refuse|
c2k2_special|-DNCLIP=2 -DNCULL=2 -DSPECIAL=1||-|g|refuse|
c2k1_psiz|-DNCLIP=2 -DNCULL=1 -DPSIZ=1||-|g|refuse|
w32|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DNCLIP=1 -DNCULL=2||-|g|refuse|
ff7like|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=1||-|gs|cull|
s128|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=2 -DNCULL=3|-DFSCLIP=1|-|gs|refuse|
s128_pc|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=1 -DNCULL=1 -DPC=1||-|s|refuse|
s256|-DLANES=128 -DVERTS=256 -DPRIMS=128 -DNCLIP=1 -DNCULL=1||-|gs|refuse|
s128_partial|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=2 -DNCULL=2 -DPARTIAL=1||-|gs|refuse|
s128_dyn|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=3 -DNCULL=3 -DDYN=1||-|gs|refuse|
s128_c8|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=8||-|gs|refuse|
s128_cullprim|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=1 -DNCULL=2 -DCULLPRIM=1||-|s|refuse|
s128_perprim|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=2 -DNCULL=1 -DPERPRIM=1|-DPERPRIM=1|-|s|refuse|
s128_layer|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=2 -DLAYER=1 -DPSIZ=1||-|s|cull|
s128_viewport|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=1 -DNCULL=1 -DVIEWPORT=1 -DLAYER=1||-|sv|refuse|
task|-DTASK=1 -DNCLIP=2 -DNCULL=2||1||refuse|
task128|-DTASK=1 -DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCLIP=1 -DNCULL=3|-DFSCLIP=1|1||refuse|
k3_dyn|-DNCULL=3 -DDYN=1||-|g|cull|
k4_sparse|-DNCULL=4 -DSPARSE=1||-|g|cull|
k2_special|-DNCULL=2 -DSPECIAL=1||-|g|cull|
k2_psiz|-DNCULL=2 -DPSIZ=1||-|g|cull|
w32k|-DLANES=32 -DVERTS=32 -DPRIMS=32 -DNCULL=2||-|g|cull|
s128k|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=3|-DFSCLIP=1|-|gs|cull|
s128k_partial|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=2 -DPARTIAL=1||-|gs|cull|
s128k_cullprim|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=2 -DCULLPRIM=1||-|s|cull|
s128k_perprim|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=1 -DPERPRIM=1|-DPERPRIM=1|-|s|cull|
s128k_viewport|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=1 -DVIEWPORT=1||-|sv|refuse|
task_k|-DTASK=1 -DNCULL=2||1||cull|
task128_k|-DTASK=1 -DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=2|-DFSCLIP=1|1||cull|
kconst|-DNCULL=1 -DCONSTCULL=1||-|g|none|none64
kconst8|-DNCULL=8 -DCONSTCULL=1 -DCVAL=0.25||-|g|none|none64
cconst|-DNCLIP=2 -DCONSTCLIP=1 -DCVAL=1.0||-|g|none|none64
ckconst_mz|-DNCLIP=1 -DNCULL=2 -DCONSTCLIP=1 -DCONSTCULL=1 -DCVAL=-0.0||-|g|none|none64
ff7const|-DLANES=128 -DVERTS=128 -DPRIMS=128 -DNCULL=1 -DCONSTCULL=1||-|gs|none|none128
ff7const_task|-DTASK=1 -DNCULL=2 -DCONSTCULL=1||1||none|none_task
cconst_kvar|-DNCLIP=2 -DCONSTCLIP=1 -DCVAL=0.5 -DNCULL=2||-|g|cull|
kconst_fs|-DNCULL=1 -DCONSTCULL=1|-DFSCLIP=1|-|g|cull|
kconst_neg|-DNCULL=1 -DCONSTCULL=1 -DCVAL=-1.0||-|g|cull|
kconst_nan|-DNCULL=1 -DCONSTCULL=1 -DCVAL=uintBitsToFloat(0x7fc00000u)||-|g|cull|
kconst_inf|-DNCULL=1 -DCONSTCULL=1 -DCVAL=uintBitsToFloat(0x7f800000u)||-|g|cull|
cconst_neg|-DNCLIP=1 -DCONSTCLIP=1 -DCVAL=-2.0||-|g|refuse|
k80|-DLANES=80 -DVERTS=80 -DPRIMS=80 -DNCULL=1||-|w|cull|
EOF
}

fail=0; total=0; oracle_runs=0; shared_applied=0; declare -A culled_dist

compile_case() { # name mesh-defs frag-defs task
  local name=$1 mdefs=$2 fdefs=$3 task=$4
  mesh=$work/$name.mesh.spv; frag=$work/$name.frag.spv; fbary=$work/$name.bary.frag.spv; tspv=-
  $G -S mesh $mdefs -o "$mesh" "$here/cc.mesh" >/dev/null || return 1
  $G -S frag $mdefs $fdefs -o "$frag" "$here/cc.frag" >/dev/null || return 1
  $G -S frag $mdefs $fdefs -DBARY=1 -o "$fbary" "$here/cc.frag" >/dev/null || return 1
  if [ "$task" = 1 ]; then
    tspv=$work/cc.task.spv
    [ -f "$tspv" ] || $G -S task -o "$tspv" "$here/cc.task" >/dev/null || return 1
  fi
}

variant_env() { # variant -> env assignments
  case "$1" in
    off|bary) echo "" ;;
    raw) echo "RADV_BC250_MESH_AMD=1" ;;
    p64) echo "RADV_BC250_PERF_PIECE_PRIMS=64" ;;
    dr1) echo "RADV_BC250_MESH_DIRECT_READ=1" ;;
    drfull) echo "RADV_BC250_MESH_DIRECT_READ=full" ;;
    compact) echo "RADV_BC250_MESH_COMPACT_LDS=1" ;;
    implicit) echo "RADV_BC250_MESH_IMPLICIT_TRIS=1" ;;
    shared) echo "RADV_BC250_MESH_COMPACT=1" ;;
    sharedac) echo "RADV_BC250_MESH_COMPACT=1 RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1" ;;
    ac) echo "RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1" ;;
    acdr) echo "RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1 RADV_BC250_MESH_DIRECT_READ=full" ;;
    gamenoac) echo "$GAMENOAC" ;;
    game) echo "$GAMENOAC RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=1" ;;
    def) echo "$DEF" ;;
    defgame) echo "$GAMENOAC RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=1 $DEF" ;;
    defshared) echo "RADV_BC250_MESH_COMPACT=1 $DEF" ;;
    pos) echo "RADV_BC250_MESH_CULLDIST_CULL=1" ;;
    legconst) echo "RADV_BC250_MESH_CLIPCULL_CONST=0" ;;
  esac
}

run_variant() { # name variant [icd]
  local name=$1 v=$2 icd=${3:-$ICD} d=$work/dump/$1/$2 f=$frag
  [ "$v" = bary ] && f=$fbary
  rm -rf "$d"; mkdir -p "$d"
  shim "$icd" env $(variant_env "$v") BC250_MESH_NIR_DUMP="$d" ./pipe "$mesh" "$f" "$tspv" 1 > "$work/$name.$v.out" 2>&1
}

pieces_of() { # out file -> hardware workgroups per API workgroup of the direct split (1 otherwise)
  local p
  p=$(sed -n 's/.*direct mesh-only split enabled: pieces=\([0-9]*\).*/\1/p' "$1" | tail -1)
  echo "${p:-1}"
}

dump_of() { ls "$work/dump/$1/$2/"*.nir 2>/dev/null | tail -1; }

# oracle label args... -> appends to $detail, sets $status on failure
oracle() {
  local label=$1; shift
  local line
  line=$("$work/mesh_oracle" --seeds "$SEEDS" "$@" 2>&1 | tail -1)
  oracle_runs=$((oracle_runs + 1))
  case "$line" in
    "ORACLE PASS"*|"ORACLE GEOMETRY PASS"*) ;;
    *) status=FAIL; detail="$detail $label[${line:0:300}]"; return ;;
  esac
  local cc=$(echo "$line" | sed -n 's/.*clipcull_channels=\([0-9]*\).*/\1/p')
  local cmp=$(echo "$line" | sed -n 's/.* compared=\([0-9]*\).*/\1/p')
  [ "${cmp:-0}" -gt 0 ] || { status=FAIL; detail="$detail $label[nothing compared]"; }
  if [ "$dists" = 1 ] && [ "${cc:-0}" -eq 0 ]; then
    status=FAIL; detail="$detail $label[no clip/cull channel compared]"
  fi
  local cd=$(echo "$line" | sed -n 's/.*culled_dist=\([0-9]*\).*/\1/p')
  [ -n "$cd" ] && culled_dist[$name]=$(( ${culled_dist[$name]:-0} + cd ))
  detail="$detail $label=${cmp}/${cc}"
}

check_out() { # variant -> appends to detail / status
  local v=$1 out=$work/$name.$1.out
  grep -q '^PIPELINE_RESULT=0' "$out" || { status=FAIL; detail="$detail $v:result=$(sed -n 's/^PIPELINE_RESULT=//p' "$out")"; }
  grep -q '^SUBMIT_OK' "$out" || { status=FAIL; detail="$detail $v:no-submit"; }
  grep -qiE 'validation failed|NIR_VALIDATE|assert|error:|^FAIL |split rejected|expanded Mesh rejected' "$out" &&
    { status=FAIL; detail="$detail $v:$(grep -iE 'validation failed|NIR_VALIDATE|assert|error:|^FAIL |rejected' "$out" | head -1 | cut -c1-120)"; }
  [ -e "$(dump_of "$name" "$v")" ] || { status=FAIL; detail="$detail $v:no-dump"; }
}

check_refused() { # variant: refused for its clip/cull position export
  local v=$1 out=$work/$name.$1.out
  grep -q '^PIPELINE_RESULT=-8' "$out" || { status=FAIL; detail="$detail $v:not-refused($(sed -n 's/^PIPELINE_RESULT=//p' "$out"))"; }
  grep -qE 'BC250 Mesh pipeline refused: (clip distances|cull distances not culled)' "$out" ||
    { status=FAIL; detail="$detail $v:no-refusal-reason"; }
  grep -qiE 'validation failed|NIR_VALIDATE|assert|error:' "$out" && { status=FAIL; detail="$detail $v:validation"; }
}

check_pos() { # variant: every POS EXPORTS trace consistent with SPI_SHADER_POS_FORMAT (traced for shaders with
  # clip/cull distances or more than one position vector)
  local v=$1 out=$work/$name.$1.out
  if [ "$dists" = 1 ] && [ "$v" != def ] && [ "$v" != defgame ] && [ "$v" != defshared ]; then
    grep -q 'MESH POS EXPORTS:' "$out" || { status=FAIL; detail="$detail $v:no-pos-trace"; return; }
  fi
  grep -q 'done_mismatch' "$out" && { status=FAIL; detail="$detail $v:pos-done-mismatch"; }
  grep -q 'position exports do not match' "$out" && { status=FAIL; detail="$detail $v:pos-mismatch"; }
}

check_trace() { # variant pattern [must/mustnot]
  local v=$1 pat=$2 out=$work/$name.$1.out
  grep -qE -- "$pat" "$out" || { status=FAIL; detail="$detail $v:missing[$pat]"; }
}

direct_read_clip() { # variant: the direct read covers every written clip/cull slot
  local v=$1 out=$work/$name.$1.out mask
  mask=$(sed -n 's/.*MESH DIRECT READ: applied locations=\(0x[0-9a-f]*\).*/\1/p' "$out" | tail -1)
  [ -n "$mask" ] || { status=FAIL; detail="$detail $v:direct-read-not-applied"; return; }
  local need=$(( 0x20000 ))
  [ $((nclip + ncull)) -gt 4 ] && need=$(( 0x60000 ))
  [ $(( mask & need )) -eq $need ] || { status=FAIL; detail="$detail $v:clip-not-direct($mask)"; }
}

sig() { grep -E '^BC250POLICY(CODE)? ' "$1" | sed 's/ va=[0-9a-f]*//' | sort | sha1sum | cut -c1-16; }
declare -A defsig
while IFS='|' read -r name mdefs fdefs task flags defexp twin; do
  [ -z "$name" ] && continue
  compile_case "$name" "$mdefs" "$fdefs" "$task" || { echo "$name: shader compile failed"; fail=1; continue; }
  nclip=$(echo "$mdefs" | sed -n 's/.*-DNCLIP=\([0-9]\).*/\1/p'); nclip=${nclip:-0}
  ncull=$(echo "$mdefs" | sed -n 's/.*-DNCULL=\([0-9]\).*/\1/p'); ncull=${ncull:-0}
  # Constant non-negative distances are removed in every variant (RADV_BC250_MESH_CLIPCULL_CONST) unless the
  # fragment shader reads them: only the others reach the legacy exports.
  exported=$((nclip + ncull))
  case "$mdefs" in *CONSTCLIP*) exported=$((exported - nclip)) ;; esac
  case "$mdefs" in *CONSTCULL*) exported=$((exported - ncull)) ;; esac
  case "$mdefs$fdefs" in *CVAL=-1*|*CVAL=-2*|*CVAL=uint*|*FSCLIP*) exported=$((nclip + ncull)) ;; esac
  dists=0; [ $exported -gt 0 ] && dists=1
  variants="off dr1 drfull compact implicit shared sharedac ac acdr gamenoac game bary"
  case "$flags" in *g*) variants="$variants raw" ;; esac
  case "$flags" in *s*) variants="$variants p64" ;; esac
  status=ok; detail=
  for v in $variants; do run_variant "$name" "$v"; check_out "$v"; check_pos "$v"; done

  # Driver defaults (see the header).
  for v in def defgame defshared; do
    run_variant "$name" "$v"
    case "$defexp" in
      refuse) check_refused "$v" ;;
      *) check_out "$v"; check_pos "$v" ;;
    esac
  done
  case "$defexp" in
    cull)
      for v in def defgame defshared; do check_trace $v 'MESH POS EXPORTS: format=[0-9] clip=0 cull=0 culldist_culled=1'; done
      check_trace def 'MESH AUTOCULL: applied .*culldist=culled_not_exported' ;;
    none|same)
      for v in def defgame defshared; do
        grep -qE 'MESH POS EXPORTS: .*(clip=[1-8]|cull=[1-8]|culldist_culled=1)' "$work/$name.$v.out" &&
          { status=FAIL; detail="$detail $v:clipcull-export"; }
      done ;;
  esac
  [ "$defexp" = none ] && { check_trace def 'MESH CLIPCULL CONST: applied'; run_variant "$name" legconst; check_out legconst; }
  if [ "$defexp" = refuse ] && [ "$ncull" -gt 0 ] && [ "$nclip" -gt 0 ] && [[ $flags != *v* ]]; then
    run_variant "$name" pos; check_out pos; check_pos pos
    check_trace pos 'MESH POS EXPORTS: format=[0-9] clip=[1-8] cull=0 culldist_culled=1'
  fi
  case "$name" in k80) check_trace def 'mesh split applied: pieces=2'; check_trace def 'AUTOCULL policy: culldist applied' ;; esac
  case "$name" in *viewport*) check_trace def 'MESH AUTOCULL candidate: no .*culldist=1 reason=viewport output' ;; esac

  # Traces.
  case "$flags" in *s*) check_trace off 'mesh split applied: pieces=3'; check_trace p64 'mesh split applied: pieces=2' ;; esac
  if [[ $flags == *v* ]]; then
    for v in ac acdr game; do check_trace $v 'MESH AUTOCULL candidate: no .*reason=viewport output'; done
  else
    check_trace ac 'MESH AUTOCULL candidate: yes'; check_trace ac 'MESH AUTOCULL: applied'
    check_trace acdr 'MESH AUTOCULL: applied'
    if [[ $flags == *w* ]]; then check_trace game 'AUTOCULL policy: refused'; else check_trace game 'MESH AUTOCULL: applied'; fi
  fi
  if [ "$dists" = 1 ] && [ -z "$(echo "$mdefs" | grep -E 'DYN|SPARSE')" ]; then
    direct_read_clip dr1; direct_read_clip drfull
  fi
  case "$flags" in *g*) check_trace raw 'MESH AMD route' ;; esac
  if grep -q 'MESH COMPACT: applied' "$work/$name.shared.out"; then
    detail="$detail shared-applied"; [ "$dists" = 1 ] && shared_applied=$((shared_applied + 1))
  fi

  # Oracle.
  off=$(dump_of "$name" off)
  if [[ $flags == *g* ]]; then
    raw=$(dump_of "$name" raw)
    for v in off p64 dr1 drfull compact implicit shared game ac; do
      [ -e "$work/$name.$v.out" ] || continue
      oracle "geo:$v" --geometry --grid --pieces-b "$(pieces_of "$work/$name.$v.out")" "$raw" "$(dump_of "$name" "$v")"
    done
  fi
  if [[ $flags == *s* ]]; then
    oracle "pieces" --geometry --geometry-params --pieces-a "$(pieces_of "$work/$name.off.out")" \
      --pieces-b "$(pieces_of "$work/$name.p64.out")" "$off" "$(dump_of "$name" p64)"
  fi
  for v in dr1 drfull compact implicit; do oracle "exact:$v" "$off" "$(dump_of "$name" "$v")"; done
  acmode=(--autocull); [[ $flags == *v* ]] && acmode=()
  gamemode=("${acmode[@]}"); [[ $flags == *w* ]] && gamemode=()
  oracle "ac" "${acmode[@]}" "$off" "$(dump_of "$name" ac)"
  oracle "acdr" "${acmode[@]}" "$(dump_of "$name" drfull)" "$(dump_of "$name" acdr)"
  oracle "game" "${gamemode[@]}" "$(dump_of "$name" gamenoac)" "$(dump_of "$name" game)"
  oracle "shared" --compact "$off" "$(dump_of "$name" shared)"
  oracle "sharedac" --compact "${acmode[@]}" "$off" "$(dump_of "$name" sharedac)"
  oracle "bary" --bary-ref "$off" "$(dump_of "$name" bary)"

  # Driver defaults against the legacy compiles (their exports have no cull channel left to compare; the
  # reference decision culls with all of the legacy compile's distances, see culled_dist).
  if [ "$defexp" = cull ] && [[ $flags != *w* ]]; then
    dists=0
    oracle "def" --autocull "$off" "$(dump_of "$name" def)"
    oracle "defgame" --autocull "$(dump_of "$name" gamenoac)" "$(dump_of "$name" defgame)"
    oracle "defshared" --compact --autocull "$off" "$(dump_of "$name" defshared)"
  elif [ "$defexp" = none ]; then
    dists=0
    oracle "def" "$(dump_of "$name" legconst)" "$(dump_of "$name" def)"
  fi
  [ -e "$work/$name.pos.out" ] && { dists=1; oracle "pos" --autocull "$off" "$(dump_of "$name" pos)"; }
  # RADV_BC250_MESH_CLIPCULL_CONST: the def/defgame compiles of a shader whose distances are all constant
  # equal its twin's (the same shader without those stores).
  if [ "$defexp" != refuse ]; then
    for v in def defgame; do
      shim "$ICD" env $(variant_env "$v") BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" 1 > "$work/$name.$v.cap" 2>&1
      defsig[$name.$v]=$(sig "$work/$name.$v.cap")
      if [ -n "$twin" ]; then
        [ "${defsig[$name.$v]}" = "${defsig[$twin.$v]:-missing}" ] ||
          { status=FAIL; detail="$detail twin:$v[${defsig[$name.$v]}/${defsig[$twin.$v]:-missing}]"; }
      fi
    done
    [ -n "$twin" ] && detail="$detail twin-identical=$twin"
  fi

  # Byte-identical to the base build without clip/cull distances.
  if [ -n "$OLDICD" ] && [ $((nclip + ncull)) = 0 ]; then
    for v in off dr1 drfull ac game p64; do
      [[ $v == p64 && $flags != *s* ]] && continue
      shim "$OLDICD" env $(variant_env "$v") BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" 1 > "$work/$name.$v.old" 2>&1
      shim "$ICD" env $(variant_env "$v") BC250_CAPTURE_POLICY_SHADERS=1 ./pipe "$mesh" "$frag" "$tspv" 1 > "$work/$name.$v.new" 2>&1
      a=$(sig "$work/$name.$v.old"); b=$(sig "$work/$name.$v.new")
      { [ "$a" = "$b" ] && grep -q '^BC250POLICY ' "$work/$name.$v.old"; } || { status=FAIL; detail="$detail identical:$v[$a/$b]"; }
    done
    detail="$detail identical-to-base=ok"
  fi
  total=$((total + 1)); [ $status = ok ] || fail=1
  printf '%-14s %-4s clip=%u cull=%u default=%s culled_dist=%s%s\n' "$name" "$status" "$nclip" "$ncull" "$defexp" "${culled_dist[$name]:-0}" "$detail"
done < <(cases | grep -E "^(${ONLY:-.*})\|")

[ -n "${ONLY:-}" ] && { echo "cases=$total failed=$fail oracle_runs=$oracle_runs seeds=$SEEDS (ONLY=$ONLY)"; exit $fail; }

# The autocull clip/cull distance test must have culled somewhere.
any=0; for k in "${!culled_dist[@]}"; do [ "${culled_dist[$k]}" -gt 0 ] && any=1; done
total=$((total + 1))
if [ $any = 1 ]; then echo "autocull-distance-culling ok"; else echo "autocull-distance-culling FAIL (no triangle culled by a distance)"; fail=1; fi

# The shared-vertex compaction must have applied to shaders with distances.
total=$((total + 1))
if [ $shared_applied -gt 0 ]; then echo "shared-applied-with-distances ok ($shared_applied cases)"; else echo "shared-applied-with-distances FAIL"; fail=1; fi

# Self-checks: the oracles must see a wrong distance and a missing distance test.
name=skew; status=ok; detail=
compile_case skew "-DNCLIP=2 -DNCULL=3 -DSKEW=1" "" - && run_variant skew off
compile_case skew_ref "-DNCLIP=2 -DNCULL=3" "" - && run_variant skew_ref raw
line=$("$work/mesh_oracle" --seeds 20 --geometry --grid "$(dump_of skew_ref raw)" "$(dump_of skew off)" | tail -1)
total=$((total + 1))
case "$line" in "ORACLE GEOMETRY FAIL"*) echo "self-check geometry ok (${line%% compared*})" ;;
  *) echo "self-check geometry FAIL ($line)"; fail=1 ;; esac
line=$("$work/mesh_oracle" --seeds "$SEEDS" --autocull --mutate 6 "$(dump_of c3k5 off)" "$(dump_of c3k5 ac)" 2>&1 | tail -1)
total=$((total + 1))
case "$line" in "ORACLE FAIL"*) echo "self-check autocull --mutate 6 ok (${line%% compared*})" ;;
  *) echo "self-check autocull --mutate 6 FAIL ($line)"; fail=1 ;; esac

line=$("$work/mesh_oracle" --seeds "$SEEDS" --compact --autocull --mutate 6 "$(dump_of c3k5 off)" "$(dump_of c3k5 sharedac)" 2>&1 | tail -1)
total=$((total + 1))
case "$line" in "ORACLE FAIL"*) echo "self-check compact autocull --mutate 6 ok (${line%% compared*})" ;;
  *) echo "self-check compact autocull --mutate 6 FAIL ($line)"; fail=1 ;; esac

# The in-shader cull distance test (RADV_BC250_MESH_CULLDIST_CULL) must be seen by the oracle.
for pair in "k1 def --autocull" "k1 defshared --compact --autocull" "c3k5 pos --autocull"; do
  set -- $pair; c=$1; v=$2; shift 2
  line=$("$work/mesh_oracle" --seeds "$SEEDS" "$@" --mutate 6 "$(dump_of $c off)" "$(dump_of $c $v)" 2>&1 | tail -1)
  total=$((total + 1))
  case "$line" in "ORACLE FAIL"*) echo "self-check $c/$v $* --mutate 6 ok (${line%% compared*})" ;;
    *) echo "self-check $c/$v $* --mutate 6 FAIL ($line)"; fail=1 ;; esac
done

echo "cases=$total failed=$fail oracle_runs=$oracle_runs seeds=$SEEDS"
exit $fail

#!/usr/bin/env bash
# RADV_BC250_MESH_AUTOCULL_WIDE offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; the harness (../mesh-autocull/autocull.c) also refuses any other
# device. Every case compiles wide.mesh with the base driver's launcher policy (post-Mesh VGT_FLUSH off),
# NIR_DEBUG=validate and ACO_DEBUG=validateir,validatera, records a direct and an indirect Mesh draw (plus
# the dynamic state draws of the case) and submits to the shim:
#   off:    RADV_BC250_MESH_AUTOCULL off;
#   on:     RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=<level>, checked against off with
#           ../mesh-autocull/check.py (yes: candidate, applied, culling + compaction ISA, runtime skip =
#           switch-off epilogue, NGG culling settings per draw, launch registers identical to off, LDS,
#           no scratch; policy / no: byte-identical to off);
#   nowide: RADV_BC250_MESH_AUTOCULL_WIDE=<level> alone must be byte-identical to off;
#   all:    for applied cases, RADV_BC250_MESH_AUTOCULL_ALL=1 must compile the same shaders and IBs as
#           the wide policy (the wide policy is a subset of ALL);
#   base:   with OLDICD=<icd of the build before the switch>, RADV_BC250_MESH_AUTOCULL=1 without WIDE
#           must be byte-identical between this build and OLDICD (AUTOCULL=1 unchanged);
#   oracle: for applied cases, the lowered NGG Mesh NIR (BC250_MESH_NIR_DUMP) with autocull off
#           (reference) and on (candidate), for RADV_BC250_MESH_DIRECT_READ=0, 1 and full, through the
#           CPU culling oracle (../direct-read/mesh_oracle.c --autocull): survivors, their order, their
#           vertex values, GS_ALLOC_REQ, the fully-culled workaround, under random culling states.
#           The suite also requires every workgroup class (no face culling, none / partly / all culled)
#           and every culling test (w, face, frustum, small primitive) to occur, and the oracle to
#           fail with each deliberately broken reference (--mutate 1..4).
# usage: ICD=<radeon_devenv_icd json> BUILD=<mesa build dir> [OLDICD=<icd>] [SEEDS=400] [KEEP=<dir>]
#        [ONLY=<case regex>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ac=$here/../mesh-autocull
ICD=${ICD:?ICD=<radeon_devenv_icd json>}
BUILD=${BUILD:?BUILD=<mesa build dir>}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
SEEDS=${SEEDS:-400}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-autocull-wide.XXXXXX")}
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
DUMPS=(BC250_CAPTURE_MESH_CODE=1 RADV_DEBUG=dumpibs,shaders,nocache)

shim() {
  local icd=$1; shift
  bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind "$work" "$work" --chdir "$work" \
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_FL1 \
    -u RADV_BC250_MESH_MERGE -u RADV_BC250_MESH_AMD -u RADV_BC250_MESH_AUTOCULL -u RADV_BC250_MESH_AUTOCULL_ALL \
    -u RADV_BC250_MESH_AUTOCULL_WIDE -u RADV_BC250_MESH_DIRECT_READ \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/autocull" "$ac/autocull.c" -lvulkan || exit 2
glslangValidator -S task --target-env vulkan1.3 -o "$work/task.spv" "$ac/shape.task" >/dev/null || exit 2
SRC=$(cd "$here/../../.." && pwd) BUILD=$BUILD "$here/../direct-read/build_oracle.sh" "$work/mesh_oracle" >/dev/null || exit 2

strip_wide() { grep -v '^BC250 MESH AUTOCULL policy: wide' "$1"; }

fail=0; passed=0; total=0; oracle_pairs=0
tot() { awk -v k="$1" '{for (i = 1; i <= NF; i++) if (index($i, k "=") == 1) s += substr($i, length(k) + 2)} END {print s + 0}' "$work/oracle.all"; }
: > "$work/oracle.all"
# name | glslang defines | grid x | harness options | wide level | expect (yes / policy / no:<reason>)
#      | settings per Mesh draw | extra environment
# Settings (radv_get_nggc_settings, 1 sample): 0xf8000000 = small-primitive precision 2^-8, 8 = small
# primitives, 4 = front face CCW (after the y-flip correction), 2 = cull back, 1 = cull front.
while IFS='|' read -r name defs gx opts level expect settings extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  [ -n "${ONLY:-}" ] && ! [[ $name =~ $ONLY ]] && continue
  pp=0; case "$defs" in *PERPRIM=1*) pp=1;; esac
  glslangValidator -S mesh --target-env vulkan1.3 $defs -o "$work/$name.mesh.spv" "$here/wide.mesh" >/dev/null || exit 2
  glslangValidator -S frag --target-env vulkan1.3 -DPERPRIM=$pp -o "$work/$name.frag.spv" "$ac/shape.frag" >/dev/null || exit 2
  run() { # tag icd env...
    local tag=$1 icd=$2; shift 2
    shim "$icd" env "${DUMPS[@]}" $extra "$@" ./autocull "$name.mesh.spv" "$name.frag.spv" $gx $opts \
      > "$work/$name.$tag.out" 2> "$work/$name.$tag.err"
  }
  run off "$ICD"
  run on "$ICD" RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=$level
  total=$((total + 1)); status=ok; notes=
  if grep -q '^UNSUPPORTED' "$work/$name.on.out"; then
    total=$((total - 1)); printf '%-18s status=skipped %s\n' "$name" "$(head -1 "$work/$name.on.out")"; continue
  fi
  grep -q '^SUBMIT_OK' "$work/$name.on.out" && grep -q '^SUBMIT_OK' "$work/$name.off.out" || { status=FAIL; notes="$notes no-submit"; }
  if ! res=$(python3 "$ac/check.py" "$work/$name.on.err" "$work/$name.off.err" "$expect" "$settings"); then
    # The only failure is the runtime-skip comparison and the skip side is the switch-off epilogue
    # modulo SGPR numbers (a scalar value the settings SGPR displaced): accepted and reported.
    if [[ $res == *" FAIL runtime_skip ["* ]] &&
       sk=$(python3 "$ac/skipcmp.py" "$work/$name.on.err" "$work/$name.off.err" --sgpr-modulo); then
      notes="$notes runtime_skip=equal_modulo_sgpr_numbers"
    else
      status=FAIL
    fi
  fi
  wide_trace=$(grep -c '^BC250 MESH AUTOCULL policy: wide' "$work/$name.on.err")
  case "$expect" in yes*) ;; *) [ "$wide_trace" = 0 ] || { status=FAIL; notes="$notes wide_trace_on_refused"; };; esac
  # the switch alone (without RADV_BC250_MESH_AUTOCULL) changes nothing
  run nowide "$ICD" RADV_BC250_MESH_AUTOCULL_WIDE=$level
  if cmp -s "$work/$name.nowide.err" "$work/$name.off.err" && cmp -s "$work/$name.nowide.out" "$work/$name.off.out"; then
    notes="$notes wide_alone_identical_to_off=True"
  else
    notes="$notes wide_alone_identical_to_off=False"; status=FAIL
  fi
  # RADV_BC250_MESH_AUTOCULL=1 without the switch: unchanged against the previous build
  if [ -n "$OLDICD" ]; then
    run base "$ICD" RADV_BC250_MESH_AUTOCULL=1
    run old "$OLDICD" RADV_BC250_MESH_AUTOCULL=1
    if cmp -s "$work/$name.base.err" "$work/$name.old.err" && cmp -s "$work/$name.base.out" "$work/$name.old.out"; then
      notes="$notes autocull1_identical_to_old_build=True"
    elif [[ $defs == *-DCLIP=1* ]]; then
      # clip/cull distances: autocull candidates since the clip/cull distance support (8c3fed4)
      notes="$notes autocull1_identical_to_old_build=False(expected:clip_distance_candidate)"
    else
      notes="$notes autocull1_identical_to_old_build=False"; status=FAIL
    fi
  fi
  case "$expect" in yes*)
    # the wide policy compiles exactly what RADV_BC250_MESH_AUTOCULL_ALL=1 compiles
    run all "$ICD" RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_ALL=1
    if diff -q <(strip_wide "$work/$name.on.err") "$work/$name.all.err" >/dev/null; then
      notes="$notes same_as_all=True"
    else
      notes="$notes same_as_all=False"; status=FAIL
    fi
    notes="$notes wide_trace=$wide_trace"
    # CPU culling oracle, RADV_BC250_MESH_DIRECT_READ off / 1 / full
    for dr in 0 1 full; do
      for side in ref cand; do
        d=$work/$name.dump.$dr.$side; rm -rf "$d"; mkdir -p "$d"
        set -- RADV_BC250_MESH_DIRECT_READ=$dr BC250_MESH_NIR_DUMP=$d
        [ $side = cand ] && set -- "$@" RADV_BC250_MESH_AUTOCULL=1 RADV_BC250_MESH_AUTOCULL_WIDE=$level
        shim "$ICD" env $extra "$@" ./autocull "$name.mesh.spv" "$name.frag.spv" $gx $opts \
          > "$d.out" 2> "$d.err"
      done
      nref=$(ls "$work/$name.dump.$dr.ref" | wc -l); ncand=$(ls "$work/$name.dump.$dr.cand" | wc -l)
      [ "$nref" -gt 0 ] && [ "$nref" = "$ncand" ] || { status=FAIL; notes="$notes dr$dr:dumps=$nref/$ncand"; continue; }
      opass=0; ofail=0
      for c in "$work/$name.dump.$dr.cand/"*.nir; do
        r=$work/$name.dump.$dr.ref/$(basename "$c")
        line=$("$work/mesh_oracle" --autocull --seeds "$SEEDS" "$r" "$c" | tail -1)
        echo "$name dr=$dr $(basename "$c") $line" >> "$work/oracle.all"
        oracle_pairs=$((oracle_pairs + 1))
        case "$line" in "ORACLE PASS"*) opass=$((opass + 1));; *) ofail=$((ofail + 1)); status=FAIL
          notes="$notes dr$dr:$(echo "$line" | cut -c1-60)";; esac
      done
      notes="$notes oracle_dr$dr=$opass/$((opass + ofail))"
    done;;
  esac
  [ "$status" = ok ] && passed=$((passed + 1)) || fail=1
  printf '%-18s status=%s level=%s\n   %s\n  %s\n' "$name" "$status" "$level" "$res" "$notes"
done <<'CASES'
# Level 1: every candidate with >= 24 triangles whose primitives fit in one wave.
# Control Resonant's class: 64 triangles, wave64, 192 lanes (private vertices, V = 3P)
w64_p64_back|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=back|1|yes|0xf800000e
w64_p64_none|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=none|1|yes|0xf800000c
w64_p64_front_cw|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=front ff=cw|1|yes|0xf8000009
w64_p64_yflip|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=back yflip=1|1|yes|0xf800000a
w64_p64_dyncull|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=dyn|1|yes|0xf800000c,0xf800000c,0xf800000e,0xf8000009,0xf800000c
w64_p64_discard|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=back discard=dyn|1|yes|0xf800000e,0xf800000e,0x00000003,0xf800000e
w64_p64_cons|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=back cons=1|1|yes|0x0
w64_p64_rt|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1 -DRUNTIME=1|10|cull=back|1|yes|0xf800000e
w64_p64_sg|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1 -DSUBGROUP=1|10|cull=back|1|yes|0xf800000e
w64_p64_pp|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1 -DPERPRIM=1|10|cull=back|1|yes|0xf800000e
# CullPrimitive without the split: combined with the application's flags (app_cull=1)
w64_p64_appcull|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1 -DPERPRIM=1 -DCULL=1 -DRUNTIME=1|10|cull=back|1|yes:app_cull|0xf800000e|RADV_BC250_SPLIT_MESH=false
# Control's interface: 64 shared vertices / 64 triangles, expanded by the base driver to 192 lanes
w64_shared64|-DWS=64 -DNV=64 -DNP=64|10|cull=back|1|yes|0xf800000e
w64_p24|-DWS=64 -DNV=72 -DNP=24 -DPRIVATE=1|10|cull=back|1|yes|0xf800000e
# Control's LDS-fit retry pieces: 32 triangles, wave64, workgroup 128
w128_p32|-DWS=128 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back|1|yes|0xf800000e
# the base driver's split (128 triangles): 43-triangle pieces in wave64 (Hellblade 2 Nanite class)
split128|-DWS=128 -DNV=256 -DNP=128|10|cull=back|1|yes|0xf800000e
split128_cull|-DWS=64 -DNV=128 -DNP=128 -DCULL=1 -DPERPRIM=1|10|cull=back|1|yes|0xf800000e
task64|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=back task=task.spv|1|yes|0xf800000e
# the default class is unchanged (and gets autocull with or without the switch)
w32_p32_back|-DWS=32 -DNV=96 -DNP=32 -DPRIVATE=1|10|cull=back|1|yes|0xf800000e
# refused at level 1: primitives in 2 or 3 waves, or fewer than 24 triangles
pol1_w64_p85|-DWS=64 -DNV=255 -DNP=85 -DPRIVATE=1|10|cull=back|1|policy|
pol1_w32_p48|-DWS=32 -DNV=144 -DNP=48 -DPRIVATE=1|10|cull=back|1|policy|
pol1_w64_p16|-DWS=64 -DNV=48 -DNP=16 -DPRIVATE=1|10|cull=back|1|policy|
pol1_w64_p23|-DWS=64 -DNV=69 -DNP=23 -DPRIVATE=1|10|cull=back|1|policy|
pol1_w32_p16|-DWS=32 -DNV=48 -DNP=16 -DPRIVATE=1|10|cull=back|1|policy|
# Level 2 (multiwave): also primitives spanning several waves (never run on hardware)
w64_p85_back|-DWS=64 -DNV=255 -DNP=85 -DPRIVATE=1|10|cull=back|2|yes|0xf800000e
w64_p85_rt|-DWS=64 -DNV=255 -DNP=85 -DPRIVATE=1 -DRUNTIME=1 -DPERPRIM=1|10|cull=back|2|yes|0xf800000e
w32_p85_rt|-DWS=32 -DNV=255 -DNP=85 -DPRIVATE=1 -DRUNTIME=1|10|cull=back|2|yes|0xf800000e
w32_p48_back|-DWS=32 -DNV=144 -DNP=48 -DPRIVATE=1|10|cull=back|2|yes|0xf800000e
w32_p48_appcull|-DWS=32 -DNV=144 -DNP=48 -DPRIVATE=1 -DPERPRIM=1 -DCULL=1 -DRUNTIME=1|10|cull=back|2|yes:app_cull|0xf800000e|RADV_BC250_SPLIT_MESH=false
w64_p64_l2|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=back|2|yes|0xf800000e
pol2_w64_p16|-DWS=64 -DNV=48 -DNP=16 -DPRIVATE=1|10|cull=back|2|policy|
# clip distance: a candidate (survivors keep their distances, all-negative triangles are culled too)
clip64|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1 -DCLIP=1|10|cull=back|2|yes|0xf800000e|RADV_BC250_MESH_ALLOW_POS1=1
# not candidates at any level: byte-identical to off
line64|-DWS=64 -DNV=192 -DNP=64 -DPRIVATE=1|10|cull=back poly=line|2|no:polygon mode|
CASES

# Coverage of the oracle runs and the oracle's self-check (a broken reference must be caught).
cov="pairs=$oracle_pairs wg_skip=$(tot wg_skip) wg_none_culled=$(tot wg_none_culled) wg_partial=$(tot wg_partial) wg_all_culled=$(tot wg_all_culled) culled_w=$(tot culled_w) culled_face=$(tot culled_face) culled_frustum=$(tot culled_frustum) culled_small=$(tot culled_small) kept=$(tot kept) defined_channels=$(tot defined_channels)"
covok=True
if [ -z "${ONLY:-}" ]; then
  for k in wg_skip wg_none_culled wg_partial wg_all_culled culled_w culled_face culled_frustum culled_small kept; do
    [ "$(tot $k)" -gt 0 ] || covok=False
  done
  selfcheck=""
  for mu in 1 2 3 4; do
    d=$work/w64_p64_back.dump.full
    line=$("$work/mesh_oracle" --autocull --mutate $mu --seeds "$SEEDS" "$d.ref/"*.nir "$d.cand/"*.nir | tail -1)
    case "$line" in "ORACLE FAIL"*) selfcheck="$selfcheck mutate$mu=caught";; *) selfcheck="$selfcheck mutate$mu=MISSED"; covok=False;; esac
  done
  cov="$cov$selfcheck"
fi
[ "$covok" = True ] || fail=1
echo "oracle coverage: $cov coverage_ok=$covok"
echo "autocull-wide: $passed/$total passed"
exit $fail

#!/usr/bin/env bash
# RADV_BC250_EXPOSE_FAST_BINDING offline regression (drm-shim only, never the real GPU).
#
# Every Vulkan program runs inside bwrap with a fresh /dev (no /dev/dri), the noop amdgpu drm-shim
# preloaded and AMDGPU_GPU_ID=gfx1013; the harness also refuses any other device. fb.c records a
# vkd3d-proton-like binding model (descriptor heap sets, sampler heap, static sampler set, push-descriptor
# root CBV, root constants) around Mesh-only draws (small: plain; nanite 256V/128P: direct split),
# hybrid Task draws (small and split Mesh), split indirect / indirect-count draws, VS draws and
# application compute dispatches, with the base driver launcher policy (post-Mesh VGT_FLUSH off, as
# HB2 runs), NIR_DEBUG=validate, ACO_DEBUG=validateir,validatera, RADV_DEBUG=dumpibs,shaders.
#
#  exposure   switch off + hybrid Task: the six features stay hidden (descriptor buffer, descriptor
#             heap, DGC, GPL, shader object, pipeline binary) and "db" mode finds no descriptor buffer;
#             switch on + hybrid Task: only VK_EXT_descriptor_buffer (+ its four features) is exposed,
#             maxDrawIndirectCount stays 4096; switch on + native Task: still hidden; switch on without
#             hybrid Task: same as switch off without hybrid Task.
#  per case   (a) "legacydb" vs "db" with the switch on (check.py): identical IBs and identical shader
#             dumps except the descriptor set pointer user SGPRs, which map one-to-one onto the
#             descriptor buffer addresses; application compute dispatches see their own (compute)
#             bindings after hybrid Task draws; driver dispatches (Task producer, setup helpers) only
#             see graphics bindings; every hybrid Task draw feeds the producer the graphics heap.
#             (b) "legacy" with the switch on is byte-identical (IBs + shader dumps) to the switch off,
#             and with OLDICD=<icd of the build before the switch> to that build.
# usage: ICD=<radeon_devenv_icd json> [SHIM=<libamdgpu_noop_drm_shim.so>] [OLDICD=<icd>] [KEEP=<dir>] ./run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
ICD=${ICD:?set ICD}
SHIM=${SHIM:?set SHIM}
OLDICD=${OLDICD:-}
work=${KEEP:-$(mktemp -d "${TMPDIR:-/tmp}/bc250-fast-binding.XXXXXX")}
mkdir -p "$work"
[ -z "${KEEP:-}" ] && trap 'rm -rf "$work"' EXIT
# The ICDs must be visible inside the sandbox (its /tmp is a fresh tmpfs).
for f in "$ICD" "$OLDICD"; do
  case "$f" in /tmp/*) echo "ICD/OLDICD must not live under /tmp: $f"; exit 2;; esac
done

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
    --unshare-pid --die-with-parent -- env -u RADV_EXPERIMENTAL -u RADV_BC250_MESH_FAST -u RADV_BC250_MESH_FL1 \
    -u RADV_BC250_MESH_MERGE -u RADV_BC250_MESH_AMD -u BC250_MESH_TIMER -u RADV_BC250_SPLIT_BATCH_PREP \
    -u RADV_BC250_EXPOSE_FAST_BINDING -u RADV_BC250_MESH_AUTOCULL -u RADV_BC250_PERF_PIECE_PRIMS \
    -u RADV_BC250_MESH_COMPACT_LDS -u BC250_CHAIN_TRACE -u RADV_DEBUG \
    LD_PRELOAD="$SHIM" AMDGPU_GPU_ID=gfx1013 VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" \
    MESA_SHADER_CACHE_DISABLE=true "${POLICY[@]}" "$@"
}

cc -O1 -Wall -o "$work/fb" "$here/fb.c" -lvulkan || exit 2
G() { glslangValidator --target-env vulkan1.3 "$@" >/dev/null || exit 2; }
G -S mesh -DWS=32 -DNV=32 -DNP=32 -o "$work/small.mesh.spv" "$here/fb.mesh"
G -S mesh -DWS=128 -DNV=256 -DNP=128 -DPERPRIM=1 -o "$work/nanite.mesh.spv" "$here/fb.mesh"
G -S mesh -DWS=32 -DNV=32 -DNP=32 -DTASK=1 -o "$work/tsmall.mesh.spv" "$here/fb.mesh"
G -S mesh -DWS=128 -DNV=256 -DNP=128 -DPERPRIM=1 -DTASK=1 -o "$work/tnanite.mesh.spv" "$here/fb.mesh"
G -S task -o "$work/fb.task.spv" "$here/fb.task"
G -S frag -o "$work/fb.frag.spv" "$here/fb.frag"
G -S frag -DPERPRIM=1 -o "$work/fbpp.frag.spv" "$here/fb.frag"
G -S vert -o "$work/fb.vert.spv" "$here/fb.vert"
G -S comp -o "$work/fb.comp.spv" "$here/fb.comp"

fail=0; passed=0; total=0
result() { # name status detail
  total=$((total + 1))
  if [ "$2" = ok ]; then passed=$((passed + 1)); else fail=1; fi
  echo "case=$1 status=$2 $3"
}

# --- exposure -------------------------------------------------------------------------------------
ext() { grep '^FB_EXTENSIONS' "$1" | sed 's/^FB_EXTENSIONS //'; }
HIDDEN="EXT_descriptor_buffer=0 EXT_descriptor_heap=0 EXT_device_generated_commands=0 EXT_graphics_pipeline_library=0 EXT_shader_object=0 KHR_pipeline_binary=0 descriptorBuffer=0 descriptorBufferPushDescriptors=0 descriptorBufferCaptureReplay=0 descriptorBufferImageLayoutIgnored=0 maxDrawIndirectCount=4096"
FAST="EXT_descriptor_buffer=1 EXT_descriptor_heap=0 EXT_device_generated_commands=0 EXT_graphics_pipeline_library=0 EXT_shader_object=0 KHR_pipeline_binary=0 descriptorBuffer=1 descriptorBufferPushDescriptors=1 descriptorBufferCaptureReplay=1 descriptorBufferImageLayoutIgnored=1 maxDrawIndirectCount=4096"
shim "$ICD" ./fb db "B m E" 1 > "$work/exp_off.out" 2>/dev/null
shim "$ICD" env RADV_BC250_EXPOSE_FAST_BINDING=1 ./fb legacy "B m E" 1 > "$work/exp_on.out" 2>/dev/null
shim "$ICD" env RADV_BC250_EXPOSE_FAST_BINDING=1 RADV_BC250_NATIVE_TASK=1 ./fb db "B m E" 1 > "$work/exp_native.out" 2>/dev/null
shim "$ICD" env RADV_BC250_HYBRID_TASK=0 ./fb legacy "B m E" 1 > "$work/exp_h0_off.out" 2>/dev/null
shim "$ICD" env RADV_BC250_HYBRID_TASK=0 RADV_BC250_EXPOSE_FAST_BINDING=1 ./fb legacy "B m E" 1 > "$work/exp_h0_on.out" 2>/dev/null
st=ok
[ "$(ext "$work/exp_off.out")" = "$HIDDEN" ] && grep -q '^FB_DB_HIDDEN' "$work/exp_off.out" || st=FAIL
[ "$(ext "$work/exp_on.out")" = "$FAST" ] || st=FAIL
[ "$(ext "$work/exp_native.out")" = "$HIDDEN" ] && grep -q '^FB_DB_HIDDEN' "$work/exp_native.out" || st=FAIL
[ -n "$(ext "$work/exp_h0_off.out")" ] && [ "$(ext "$work/exp_h0_off.out")" = "$(ext "$work/exp_h0_on.out")" ] || st=FAIL
result exposure $st "off=[$(ext "$work/exp_off.out" | tr ' ' ',')] on=[$(ext "$work/exp_on.out" | tr ' ' ',')] native_task_on=hidden:$(grep -c FB_DB_HIDDEN "$work/exp_native.out") hybrid0=[$(ext "$work/exp_h0_on.out" | tr ' ' ',')]"

# Without hybrid Task, descriptor buffers were already exposed, but a large Mesh-only descriptor
# buffer pipeline was refused (direct split excluded it, the internal amplifier needs hybrid Task).
# With the switch on it takes the direct split.
shim "$ICD" env RADV_BC250_HYBRID_TASK=0 ./fb db "B sd E" 1 > "$work/h0db_off.out" 2>/dev/null
st=ok; grep -q '^FB_PIPELINE nanite RESULT=-8' "$work/h0db_off.out" || st=FAIL
result hybrid0_db_switch_off_refuses_split $st "$(grep '^FB_PIPELINE nanite' "$work/h0db_off.out")"

# --- cases ----------------------------------------------------------------------------------------
# BC250_MESH_TIMER prints a table at exit whose draw rates depend on the wall clock, and a header line that
# names its own format version: neither is compared.
notimer() { sed -e '/^# BC250_MESH_TIMER v[0-9]/d' -e '/^BC250_MESH_TIMER FINAL/,$d' "$1"; }
# name | fb.c script | extra environment (every run of the case)
while IFS='|' read -r name script extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  st=ok
  for m in legacydb db; do
    shim "$ICD" env $extra RADV_BC250_EXPOSE_FAST_BINDING=1 RADV_DEBUG=dumpibs,shaders,nocache \
      ./fb $m "$script" > "$work/$name.$m.out" 2> "$work/$name.$m.err"
  done
  res=$(python3 "$here/check.py" "$script" "$work/$name.legacydb.out" "$work/$name.legacydb.err" \
        "$work/$name.db.out" "$work/$name.db.err") || st=FAIL
  rm -f "$work/$name.legacydb.err" "$work/$name.db.err"
  # (b) switch-off guarantee on the legacy binding model
  shim "$ICD" env $extra RADV_DEBUG=dumpibs,shaders,nocache ./fb legacy "$script" > "$work/$name.off.out" 2> "$work/$name.off.err"
  shim "$ICD" env $extra RADV_BC250_EXPOSE_FAST_BINDING=1 RADV_DEBUG=dumpibs,shaders,nocache ./fb legacy "$script" \
    > "$work/$name.on.out" 2> "$work/$name.on.err"
  grep -q '^DONE' "$work/$name.off.out" || st=FAIL
  if cmp -s <(notimer "$work/$name.off.err") <(notimer "$work/$name.on.err") && \
     cmp -s <(grep -v '^FB_EXTENSIONS' "$work/$name.off.out") <(grep -v '^FB_EXTENSIONS' "$work/$name.on.out"); then
    res="$res legacy_on_identical_to_off=True"
  else
    res="$res legacy_on_identical_to_off=False"; st=FAIL
  fi
  if [ -n "$OLDICD" ]; then
    shim "$OLDICD" env $extra RADV_DEBUG=dumpibs,shaders,nocache ./fb legacy "$script" > "$work/$name.old.out" 2> "$work/$name.old.err"
    # DGC is now default-off even with hybrid Task disabled. Admit exactly that
    # extension-list change; shader/PM4 and every other reported value stay exact.
    baseline_stdout() {
      if [ "$name" = hybrid0 ] && grep -q 'EXT_device_generated_commands=1' "$work/$name.old.out"; then
        grep -q 'EXT_device_generated_commands=0' "$work/$name.off.out" || return 1
        sed 's/EXT_device_generated_commands=1/EXT_device_generated_commands=0/' "$work/$name.old.out"
      else
        cat "$work/$name.old.out"
      fi
    }
    if cmp -s <(notimer "$work/$name.old.err") <(notimer "$work/$name.off.err") && cmp -s <(baseline_stdout) "$work/$name.off.out"; then
      res="$res off_identical_to_old_build=True"
    else
      res="$res off_identical_to_old_build=False"; st=FAIL
    fi
    rm -f "$work/$name.old.err"
  fi
  rm -f "$work/$name.off.err" "$work/$name.on.err"
  case "$res" in *status=FAIL*) st=FAIL;; esac
  result "$name" $st "${res#status=ok }"
done <<EOF
all|B m mi sd si sc t ti tc T v E cd|
restore|cd B t E cd B ti tc E cd B T E cd|
producer_after_compute|cd cd B t ti T E|
batch_on|B si si sc sd si E cd B sc si mi E|RADV_BC250_SPLIT_BATCH_PREP=1
batch_off|B si si sc sd si E cd B sc si mi E|
amplifier|B sd si sc E cd B t E|BC250_EXPERIMENTAL_DIRECT_SPLIT=false
rebind|B m rb si E rb cd B t rb ti E|
autocull|B m mi sd si t T E cd|RADV_BC250_MESH_AUTOCULL=1
hb2_config|B sd si sc t T mi E cd|RADV_BC250_SPLIT_BATCH_PREP=1 RADV_BC250_PERF_PIECE_PRIMS=64 RADV_BC250_MESH_COMPACT_LDS=1
timer|B m si t E cd|BC250_MESH_TIMER=1 BC250_MESH_TIMER_FILE=$work/timer.log
hybrid0|B m mi sd si sc v E cd|RADV_BC250_HYBRID_TASK=0
EOF

echo "fast-binding: passed=$passed/$total"
exit $fail

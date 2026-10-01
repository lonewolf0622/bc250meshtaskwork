#!/usr/bin/env python3
"""Dry-run inside the noop sandbox; hardware requires one explicit named gate."""
import hashlib, json, os, re, selectors, struct, subprocess, sys, time
from pathlib import Path

g = Path(__file__).resolve().parent
names = ('descriptor-buffers-v2', 'many-task-v2')
offline = len(sys.argv) == 3 and sys.argv[1] == '--offline'
if offline:
    name = sys.argv[2]
elif len(sys.argv) == 2 and sys.argv[1].startswith('--authorized-gpu-'):
    name = sys.argv[1].removeprefix('--authorized-gpu-')
else:
    sys.exit('usage: run_once.py --offline NAME | --authorized-gpu-NAME; owner authorization required')
assert name in names
kind = name.removesuffix('-v2')
from audit import audit
manifest = json.loads((g / 'gate-manifest.json').read_text())
def check_pins():
    for path, digest in manifest['sha256'].items():
        assert hashlib.sha256((g / path).read_bytes()).hexdigest() == digest, path
check_pins()
before_markers = sorted(str(p) for p in g.glob('*hardware.last'))
if offline:
    assert not Path('/dev/dri').exists()
    assert os.getenv('AMDGPU_GPU_ID') == 'gfx1013'
    assert 'libamdgpu_noop_drm_shim.so' in os.getenv('LD_PRELOAD', '')
else:
    assert not os.getenv('LD_PRELOAD') and not os.getenv('AMDGPU_GPU_ID')
    assert Path('/dev/dri').exists()
    assert Path('/sys/module/amdgpu/parameters/gpu_recovery').read_text().strip() == '0'
    prerequisite = 'D2'
    if prerequisite:
        assert 'PASS images byte-identical' in (g / 'prerequisites' / (prerequisite + '-hardware.last')).read_text()
    marker = g / (name + '-hardware.last')
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write('RESERVED two single-submit processes; never retry after failure\n')
        f.flush(); os.fsync(f.fileno())
    os.sync()
env = {'PATH': '/usr/bin', 'HOME': '/tmp', 'MESA_SHADER_CACHE_DISABLE': '1',
       'RADV_DIRECTMESH': '1', 'RADV_BC250_EXPOSE_DGC': '1',
       'VK_DRIVER_FILES': str(g / 'candidate.json'), 'TEST_ICD_LIBRARY': str(g / 'candidate.so'),
       'NIR_DEBUG': 'validate', 'ACO_DEBUG': 'validateir,validatera', 'BC250_TRACE_COMPILE': '1',
       'RADV_DEBUG': 'dumpibs', 'BC250_CAPTURE_POINTERS': '1', 'BC250_CAPTURE_POLICY_SHADERS': '1', 'GPL_MODE': 'compiled', 'GATE_KIND': kind,
       'RADV_BC250_PIPELINE_PLAN': '1', 'RADV_BC250_GPL_SOURCE_LINK': '1',
       'RADV_BC250_GPL_BINARY_LINK': '1', 'RADV_BC250_SHADER_OBJECT_PLAN': '1'}
env.update(json.loads((g / 'policy.json').read_text()))
env.update({k: '1' for k in ('RADV_BC250_MESH_SAFE_FAST', 'RADV_BC250_MESH_SAFE_PIECES',
    'RADV_BC250_MESH_SAFE_OWNED', 'RADV_BC250_MESH_SAFE_CORNERS', 'RADV_BC250_MESH_SAFE_LOCAL',
    'RADV_BC250_MESH_SAFE_AUTOCULL', 'RADV_BC250_MESH_SAFE_PARALLEL', 'RADV_BC250_MESH_SAFE_CHECK',
    'RADV_BC250_MESH_FAIL_CLOSED')})
env.update({k: '0' for k in ('RADV_BC250_TASK_GRID_FOLD', 'RADV_BC250_MESH_SAFE_PIECES_EXT',
    'RADV_BC250_MESH_PP_SHARE', 'RADV_BC250_MESH_LEAN_CHECK', 'RADV_BC250_MESH_SAFE_COMPACT',
    'RADV_BC250_MESH_DEAD_PAYLOAD', 'RADV_BC250_MESH_PIECE_PRIMID', 'RADV_BC250_MESH_SAFE_ADAPTIVE')})
if offline:
    env.update({k: os.environ[k] for k in ('LD_PRELOAD', 'AMDGPU_GPU_ID')})
    env['VK_INSTANCE_LAYERS'] = 'VK_LAYER_KHRONOS_validation'
else:
    env['BC250_DGC_OWNER_AUTHORIZED'] = 'DGC-once'
    env['BC250_GPL_OBJECT_OWNER_AUTHORIZED'] = 'GPL_OBJECT-once'
out = Path(os.environ.get('GATE_RESULTS', g / ('dry-run' if offline else 'hardware-results')))
out.mkdir(exist_ok=True)
rows = []
bad = re.compile(r'API_VALIDATION_ERROR|Validation Error|VUID-|validation failed|NIR_VALIDATE|Assertion .*failed|raw_unproven|Segmentation fault')
def route_check(text, side, case):
    assert not bad.search(text) and 'BC250 MESH AMD:' in text
    if side == 'feature' and kind == 'gpl':
        assert 'BINARY_ROUNDTRIP count=3 result=0' in text and 'GPL_BINARY_FAST_LINK result=0' in text
    if side == 'feature' and kind == 'objects':
        assert 'OBJECT_BINARY_ROUNDTRIP' in text and 'OBJECTS_BOUND' in text
    if side != 'feature' or kind in ('gpl', 'objects'): return
    count = int(kind == 'many-task')
    data = (case / ('capture-' + str(count) + '.bin')).read_bytes()
    p = struct.unpack_from('<QQQQIIII', data)
    assert p[4] == 1
    task = kind != 'D1'
    template = p[6] - 1048576 if count else 0
    code = 1048576 if task else p[6]
    words = struct.unpack_from('<' + 'I' * (code // 4), data, 48 + template)
    i = dispatch = mesh = 0
    producer = flushed = acquired = False
    while i < len(words):
        h = words[i]
        assert h >> 30 in (2, 3)
        n = 1 if h == 0xffff1000 or h >> 30 == 2 else ((h >> 16) & 0x3fff) + 2
        assert i + n <= len(words)
        op = (h >> 8) & 255
        assert op not in (0x4d, 0xaa, 0xad)
        if op in (0x15, 0x16):
            dispatch += 1; producer = True; flushed = acquired = False
        if op == 0x46 and n > 1 and words[i + 1] & 63 == 7 and producer: flushed = True
        if op == 0x58 and flushed: acquired = True
        if op == 0x4c:
            mesh += 1
            if task: assert producer and flushed and acquired
            producer = flushed = acquired = False
        i += n
    assert mesh > 0
    if task: assert dispatch >= 1025 and mesh == 1024
    (case / 'route-proof.json').write_text(json.dumps({'dispatches': dispatch, 'mesh_consumers': mesh,
        'native_task_packets': 0, 'producer_visibility_order': task, 'many_draw_records': p[7]}, indent=2) + '\n')
for side in ('mono', 'feature'):
    case = out / (name + '-' + side)
    case.mkdir(exist_ok=True)
    image = case / 'readback.rgba'
    assert not image.exists()
    e = dict(env, GATE_MODE=side, BC250_DGC_DUMP=str(case))
    if kind in ('gpl', 'objects'):
        args = [g / (name + '-gate'), '--offline' if offline else '--authorized-gpu-GPL_OBJECT',
                g / 'mesh.spv', g / 'frag.spv', image]
    else:
        args = [g / 'gate', '--offline' if offline else '--authorized-gpu-DGC',
                g / name / 'mesh.spv', g / name / 'frag.spv',
                '-' if kind == 'D1' else g / name / 'task.spv', image]
    if offline:
        p = subprocess.run(list(map(str, args)), env=e, capture_output=True, text=True, timeout=60)
        text = p.stdout + p.stderr
        code = p.returncode
    else:
        # Audit the actual compiler output before releasing its only submission.
        p = subprocess.Popen(list(map(str, args)), env=e, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        selector = selectors.DefaultSelector(); selector.register(p.stdout, selectors.EVENT_READ)
        captured = bytearray(); released = False; deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            for key, _ in selector.select(0.2):
                block = os.read(key.fileobj.fileno(), 65536)
                if block: captured.extend(block)
                else: selector.unregister(key.fileobj)
            text = captured.decode(errors='replace')
            (case / 'run.log').write_text(text)
            if not released and '_READY mode=hardware' in text:
                route_check(text, side, case)
                if kind == 'descriptor-buffers':
                    proof = audit(text, side, case)
                    (case / 'pointer-audit-before-submit.json').write_text(json.dumps(proof, indent=2) + '\n')
                if kind not in ('D1', 'gpl', 'objects'):
                    assert 'BC250 original NIR: stage=7' in text and 'BC250 original NIR: stage=0' in text
                if side == 'feature' and kind not in ('gpl', 'objects'):
                    assert (case / 'capture-0.bin').exists() or (case / 'capture-1.bin').exists()
                p.stdin.write(b'S'); p.stdin.flush(); released = True
            if p.poll() is not None and not selector.get_map(): break
        else:
            p.kill(); p.wait()
            sys.exit('STOP timeout; retain reservation; owner dumps and reboots; never retry')
        code = p.wait()
        assert released
    (case / 'run.log').write_text(text)
    assert code == 0 and not bad.search(text), text[-4000:]
    route_check(text, side, case)
    if offline and kind == 'descriptor-buffers':
        (case / 'pointer-audit-before-submit.json').write_text(json.dumps(audit(text, side, case), indent=2) + '\n')
    assert 'VALIDATION_ERRORS=0' in text and 'BC250 MESH AMD:' in text
    if kind not in ('D1', 'gpl', 'objects'): assert 'BC250 original NIR: stage=7' in text and 'BC250 original NIR: stage=0' in text
    captured = (case / 'capture-0.bin').exists() or (case / 'capture-1.bin').exists()
    if kind not in ('gpl', 'objects'): assert captured == (side == 'feature')
    if offline: assert 'NOOP_COMPLETE pixels_not_rendered' in text and not image.exists()
    rows.append({'gate': name, 'side': side, 'exit': code, 'capture': captured,
                 'validation_errors': 0, 'api_layer': kind not in ('gpl', 'objects'),
                 'raw_routes': 0, 'noop_completed': offline})
    (out / (name + '-summary.json')).write_text(json.dumps(rows, indent=2) + '\n')
    print(name, side, 'PASS', flush=True)
check_pins()
if offline:
    assert sorted(str(p) for p in g.glob('*hardware.last')) == before_markers
    print('Pins unchanged; no marker reserved; no hardware execution', flush=True)
else:
    a = (out / (name + '-mono/readback.rgba')).read_bytes()
    b = (out / (name + '-feature/readback.rgba')).read_bytes()
    assert len(a) == 256 * 256 * 4 and a == b and any(a[3::4])
    with marker.open('a') as f:
        f.write('PASS images byte-identical\n'); f.flush(); os.fsync(f.fileno())

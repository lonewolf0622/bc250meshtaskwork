"""Serialized cache hits must retain private Task producers and split helpers."""
import json
import os
from pathlib import Path
import re
import subprocess

assert not Path('/dev/dri').exists()
assert os.getenv('AMDGPU_GPU_ID') == 'gfx1013'
src = Path(__file__).resolve().parents[3]
out = Path.cwd()
env = dict(os.environ, VK_DRIVER_FILES=os.environ['ICD'], VK_ICD_FILENAMES=os.environ['ICD'],
           RADV_BC250_PIPELINE_PLAN='1', BC250_TRACE_COMPILE='1',
           NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera')
policy = '''RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 RADV_PERFTEST=mesh,nircache
BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true
BC250_TRANSIENT_ARENA=true BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false
BC250_OUTPUT_REGIONS=true BC250_EXPERIMENTAL_PRIVATE_GTT=true
BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true
RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true
BC250_EXPERIMENTAL_CULL_COMPACT=true BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true
BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0'''
env.update(x.split('=', 1) for x in policy.split())
for flag in '''SAFE_FAST SAFE_PIECES SAFE_OWNED SAFE_LOCAL SAFE_CHECK SAFE_CORNERS
SAFE_AUTOCULL SAFE_PARALLEL SAFE_BARY SAFE_BARY_TINY SAFE_BARY_AFFINE SAFE_BARY_LAST
ALLOW_POS1 SPLIT_ANY NESTED_SLICE SAFE_SPLIT_PIECES'''.split():
    env['RADV_BC250_MESH_' + flag] = '1'
env.update(RADV_BC250_BARY_IO16='1', RADV_BC250_BARY_CORNER_ID='1')


def run(args, name):
    p = subprocess.run(list(map(str, args)), env=env, capture_output=True, text=True)
    text = p.stdout + p.stderr
    (out / (name + '.log')).write_text(text)
    assert p.returncode == 0, (name, p.returncode, text[-4000:])
    return text


run(['cc', '-O1', '-Wall', '-o', 'cache-pipe', src / 'tests/bc250-mesh/pipeline-plan/pipe.c', '-lvulkan'], 'pipe-build')
run(['cc', '-O1', '-Wall', '-o', 'binary-pipe', src / 'tests/bc250-mesh/pipeline-plan/binary-pipe.c', '-ldl'], 'binary-build')
env['TEST_ICD_LIBRARY'] = json.loads(Path(os.environ['ICD']).read_text())['ICD']['library_path']
run(['cc', '-Wall', '-Werror', '-I' + str(src / 'src/amd/vulkan'), '-o', 'validate-plan',
     src / 'tests/bc250-mesh/pipeline-plan/validate.c'], 'validate-build')
run([out / 'validate-plan'], 'validate')
rows = []
for name, defs in [('plain', []), ('pieces', ['-DVERTS=128', '-DPRIMS=128', '-DLANES=128']),
                   ('task', ['-DTASK=1']), ('task-pieces', ['-DTASK=1', '-DVERTS=128', '-DPRIMS=128', '-DLANES=128']),
                   ('bary', ['-DBARY=1']), ('task-bary', ['-DTASK=1', '-DBARY=1'])]:
    case = out / name
    case.mkdir(exist_ok=True)
    defs = list({x.split('=')[0]: x for x in ['-DVERTS=32', '-DPRIMS=32', '-DLANES=32'] + defs}.values())
    for stage in ['mesh', 'frag'] + (['task'] if name.startswith('task') else []):
        run(['glslangValidator', '--target-env', 'vulkan1.3', '-S', stage, *defs, '-o', case / (stage + '.spv'),
             src / ('tests/bc250-mesh/compact/cmp.' + stage)], name + '-' + stage)
    text = run([out / 'cache-pipe', case / 'mesh.spv', case / 'frag.spv',
                case / 'task.spv' if name.startswith('task') else '-', '1'], name)
    assert 'CACHE_ROUNDTRIP' in text and 'hit=1' in text and 'SUBMIT_OK' in text, (name, text[-4000:])
    assert not re.search(r'validation failed|NIR_VALIDATE|Assertion .*failed', text)
    binary = run([out / 'binary-pipe', case / 'mesh.spv', case / 'frag.spv',
                  case / 'task.spv' if name.startswith('task') else '-', '1'], name + '-binary')
    assert 'BINARY_ROUNDTRIP count=3 result=0' in binary and 'SUBMIT_OK' in binary, (name, binary[-4000:])
    rows.append(dict(case=name, serialized_cache_hit=True, binary_roundtrip=True, submit_ok=True))
    print(name, 'PASS', flush=True)
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

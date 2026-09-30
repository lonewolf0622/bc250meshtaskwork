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
           NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera', BC250_CAPTURE_POLICY_SHADERS='1')
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
hw_flags = ['RADV_BC250_MESH_SAFE_COMPACT', 'RADV_BC250_MESH_DEAD_PAYLOAD', 'RADV_BC250_MESH_PIECE_PRIMID']
hw_policy = os.environ.get('PLAN_HW_POLICY', 'off')
assert hw_policy in ('off', 'on')
env.update({flag: '1' if hw_policy == 'on' else '0' for flag in hw_flags})
variant = os.environ.get('PLAN_VARIANT', 'default')
assert variant in ('default', 'adaptive', 'corner-id')
if variant == 'adaptive':
    env['RADV_BC250_MESH_SAFE_ADAPTIVE'] = '1'
if variant == 'corner-id':
    env.update(RADV_BC250_MESH_SAFE_FAST='0', RADV_BC250_MESH_SAFE_PIECES='0',
               RADV_BC250_MESH_SAFE_BARY_AFFINE='0', RADV_BC250_BARY_CORNER_ID_FORCE='1')



def run(args, name):
    p = subprocess.run(list(map(str, args)), env=env, capture_output=True, text=True)
    text = p.stdout + p.stderr
    (out / (name + '.log')).write_text(text)
    assert p.returncode == 0, (name, p.returncode, text[-4000:])
    return text


run(['cc', '-O1', '-Wall', '-o', 'cache-pipe', src / 'tests/bc250-mesh/pipeline-plan/pipe.c', '-lvulkan'], 'pipe-build')
run(['cc', '-O1', '-Wall', '-o', 'binary-pipe', src / 'tests/bc250-mesh/pipeline-plan/binary-pipe.c', '-ldl'], 'binary-build')
run(['cc', '-O1', '-Wall', '-o', 'gpl-pipe', src / 'tests/bc250-mesh/pipeline-plan/gpl-pipe.c', '-ldl'], 'gpl-build')
run(['cc', '-O1', '-Wall', '-o', 'object-pipe', src / 'tests/bc250-mesh/pipeline-plan/object-pipe.c', '-ldl'], 'object-build')
env['TEST_ICD_LIBRARY'] = json.loads(Path(os.environ['ICD']).read_text())['ICD']['library_path']
run(['cc', '-Wall', '-Werror', '-I' + str(src / 'src/amd/vulkan'), '-o', 'validate-plan',
     src / 'tests/bc250-mesh/pipeline-plan/validate.c'], 'validate-build')
run([out / 'validate-plan'], 'validate')
run(['glslangValidator', '--target-env', 'vulkan1.3', '-S', 'frag', '-o', out / 'object-foreign.spv',
     src / 'tests/bc250-mesh/pipeline-plan/object-foreign.frag'], 'object-foreign-build')
run(['glslangValidator', '--target-env', 'vulkan1.3', '-S', 'task', '-o', out / 'object-foreign-task.spv',
     src / 'tests/bc250-mesh/pipeline-plan/object-foreign.task'], 'object-foreign-task-build')
run(['glslangValidator', '--target-env', 'vulkan1.3', '-S', 'mesh', '-DVERTS=3', '-DPRIMS=1', '-DLANES=32',
     '-o', out / 'swap.mesh.spv', src / 'tests/bc250-mesh/compact/cmp.mesh'], 'swap-mesh')
env['PLAN_SWAP_MESH'] = str(out / 'swap.mesh.spv')
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
    if variant == 'corner-id' and name == 'bary':
        assert 'rotation from corner id' in text, text[-4000:]
    assert 'CACHE_ROUNDTRIP' in text and 'hit=1' in text and 'SUBMIT_OK' in text, (name, text[-4000:])
    assert not re.search(r'validation failed|NIR_VALIDATE|Assertion .*failed', text)
    binary = run([out / 'binary-pipe', case / 'mesh.spv', case / 'frag.spv',
                  case / 'task.spv' if name.startswith('task') else '-', '1'], name + '-binary')
    assert 'BINARY_FOREIGN_PLAN_REFUSED' in binary and 'BINARY_ROUNDTRIP count=3 result=0' in binary and 'SUBMIT_OK' in binary, (name, binary[-4000:])
    swap_mesh = env.pop('PLAN_SWAP_MESH')
    env.update(PLAN_BINARY_BAD_STATE='1', PLAN_BINARY_EXPECT_REFUSAL='1')
    bad_state = subprocess.run([str(out / 'binary-pipe'), str(case / 'mesh.spv'), str(case / 'frag.spv'),
                               str(case / 'task.spv') if name.startswith('task') else '-', '1'],
                              env=env, capture_output=True, text=True)
    (out / (name + '-binary-bad-state.log')).write_text(bad_state.stdout + bad_state.stderr)
    assert bad_state.returncode == 1 and 'BINARY_POLICY_PLAN_REFUSED' in bad_state.stdout and 'SUBMIT_OK' not in bad_state.stdout
    env.pop('PLAN_BINARY_BAD_STATE')
    env.pop('PLAN_BINARY_EXPECT_REFUSAL')
    env['RADV_BC250_GPL_SOURCE_LINK'] = '1'
    for mode in ('fast', 'lto', 'nested'):
        env['GPL_MODE'] = mode
        linked = run([out / 'gpl-pipe', case / 'mesh.spv', case / 'frag.spv',
                      case / 'task.spv' if name.startswith('task') else '-', '1'], name + '-gpl-' + mode)
        assert 'GPL_SOURCE_LINK mode=' + mode + ' result=0' in linked and 'SUBMIT_OK' in linked
        assert 'route=raw_unproven' not in linked
        def shader_records(log):
            return sorted({re.sub(r' va=[0-9a-f]+', '', line) for line in log.splitlines()
                           if re.match(r'^BC250POLICY(CODE)? ', line)})
        assert shader_records(text) and shader_records(text) == shader_records(linked), (name, mode, 'shader identity')
        env['GPL_FINAL_BINARY'] = '1'
        imported = run([out / 'gpl-pipe', case / 'mesh.spv', case / 'frag.spv',
                        case / 'task.spv' if name.startswith('task') else '-', '1'], name + '-gpl-' + mode + '-binary')
        env.pop('GPL_FINAL_BINARY')
        assert 'BINARY_ROUNDTRIP count=3 result=0' in imported and 'SUBMIT_OK' in imported
        assert shader_records(text) == shader_records(imported), (name, mode, 'binary shader identity')

    env['RADV_BC250_GPL_BINARY_LINK'] = '1'
    env['GPL_MODE'] = 'compiled'
    linked = run([out / 'gpl-pipe', case / 'mesh.spv', case / 'frag.spv',
                  case / 'task.spv' if name.startswith('task') else '-', '1'], name + '-gpl-compiled')
    env.pop('RADV_BC250_GPL_BINARY_LINK')
    assert 'GPL_BINARY_FAST_LINK result=0' in linked and 'GPL_BINARY_LTO_REFUSED' in linked and 'SUBMIT_OK' in linked
    assert shader_records(text) == shader_records(linked), (name, 'compiled library shader identity')

    env['PLAN_SWAP_MESH'] = swap_mesh
    env.pop('RADV_BC250_GPL_SOURCE_LINK')
    env.pop('GPL_MODE')
    env['RADV_BC250_SHADER_OBJECT_PLAN'] = '1'
    object_args_case = [out / 'object-pipe', case / 'mesh.spv', case / 'frag.spv',
                        case / 'task.spv' if name.startswith('task') else '-', '1']
    objects = run(object_args_case, name + '-objects')
    assert 'OBJECT_LINK result=0' in objects and 'OBJECT_BINARY_ROUNDTRIP' in objects
    assert 'OBJECT_CORRUPT_TRUNCATED_LAYOUT_REFUSED' in objects and 'OBJECTS_BOUND' in objects and 'SUBMIT_OK' in objects
    assert 'raw_unproven' not in objects
    env['OBJECT_BAD_STATE'] = '1'
    refused = subprocess.run(list(map(str, object_args_case)), env=env, capture_output=True, text=True)
    (out / (name + '-objects-bad-state.log')).write_text(refused.stdout + refused.stderr)
    assert refused.returncode == 1 and 'vkEndCommandBuffer(cb) = -8' in refused.stdout and 'SUBMIT_OK' not in refused.stdout
    env.pop('OBJECT_BAD_STATE')
    if name in ('plain', 'bary'):
        env['OBJECT_FOREIGN_FS'] = str(out / 'object-foreign.spv')
        refused = subprocess.run(list(map(str, object_args_case)), env=env, capture_output=True, text=True)
        (out / (name + '-objects-foreign.log')).write_text(refused.stdout + refused.stderr)
        assert refused.returncode == 1 and 'OBJECT_FOREIGN_PAIR_BOUND' in refused.stdout and 'vkEndCommandBuffer(cb) = -8' in refused.stdout
        env.pop('OBJECT_FOREIGN_FS')
    if name.startswith('task'):
        env['OBJECT_FOREIGN_TASK'] = str(out / 'object-foreign-task.spv')
        refused = subprocess.run(list(map(str, object_args_case)), env=env, capture_output=True, text=True)
        (out / (name + '-objects-foreign-task.log')).write_text(refused.stdout + refused.stderr)
        assert refused.returncode == 1 and 'OBJECT_FOREIGN_TASK_BOUND' in refused.stdout and 'vkEndCommandBuffer(cb) = -8' in refused.stdout
        env.pop('OBJECT_FOREIGN_TASK')
    env.pop('RADV_BC250_SHADER_OBJECT_PLAN')
    rows.append(dict(case=name, serialized_cache_hit=True, binary_roundtrip=True, submit_ok=True,
                     gpl_source_links=['fast_request', 'lto', 'nested']))
    print(name, 'PASS', flush=True)
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

policy_rows = []
case = out / 'plain'
args = [str(out / 'binary-pipe'), str(case / 'mesh.spv'), str(case / 'frag.spv'), '-', '1']
base_env = dict(env)
base_env.pop('PLAN_SWAP_MESH', None)
base_env.update({flag: '0' for flag in hw_flags})
policies = {'off': {flag: '0' for flag in hw_flags}, 'on': {flag: '1' for flag in hw_flags}}
policies.update({flag: {name: str(int(name == flag)) for name in hw_flags} for flag in hw_flags})
for name, flags in policies.items():
    file = out / ('policy-' + name + '.bin')
    e = dict(base_env, **flags, PLAN_BINARY_WRITE=str(file))
    result = subprocess.run(args, env=e, capture_output=True, text=True)
    (out / ('policy-write-' + name + '.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (name, result.stdout, result.stderr[-4000:])
for source, target in [('off', name) for name in policies if name != 'off'] + [(name, 'off') for name in policies if name != 'off']:
    e = dict(base_env, **policies[target], PLAN_BINARY_READ=str(out / ('policy-' + source + '.bin')),
             PLAN_BINARY_EXPECT_REFUSAL='1')
    result = subprocess.run(args, env=e, capture_output=True, text=True)
    log = result.stdout + result.stderr
    (out / ('policy-' + source + '-to-' + target + '.log')).write_text(log)
    assert result.returncode == 1 and 'BINARY_POLICY_PLAN_REFUSED' in log and 'SUBMIT_OK' not in log, log[-4000:]
    policy_rows.append(dict(source=source, target=target, refused=True))
(out / 'policy-summary.json').write_text(json.dumps(policy_rows, indent=2) + '\n')
print('hardware-policy cross-import refusals', len(policy_rows), 'PASS', flush=True)

object_args = [str(out / 'object-pipe'), str(case / 'mesh.spv'), str(case / 'frag.spv'), '-', '1']
object_env = dict(base_env, RADV_BC250_SHADER_OBJECT_PLAN='1')
for name, flags in policies.items():
    e = dict(object_env, **flags, OBJECT_BINARY_WRITE=str(out / ('object-policy-' + name + '.bin')))
    result = subprocess.run(object_args, env=e, capture_output=True, text=True)
    (out / ('object-policy-write-' + name + '.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and 'SUBMIT_OK' in result.stdout, (name, result.returncode, result.stderr[-2000:])
object_rows = []
for source, target in [('off', name) for name in policies if name != 'off'] + [(name, 'off') for name in policies if name != 'off']:
    e = dict(object_env, **policies[target], OBJECT_BINARY_READ=str(out / ('object-policy-' + source + '.bin')),
             OBJECT_EXPECT_REFUSAL='1')
    result = subprocess.run(object_args, env=e, capture_output=True, text=True)
    (out / ('object-policy-' + source + '-to-' + target + '.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 1 and 'OBJECT_BINARY_POLICY_REFUSED' in result.stdout and 'SUBMIT_OK' not in result.stdout
    object_rows.append(dict(source=source, target=target, refused=True))
e = dict(object_env, RADV_BC250_SHADER_OBJECT_PLAN='0', **policies['off'],
         OBJECT_BINARY_READ=str(out / 'object-policy-off.bin'), OBJECT_EXPECT_REFUSAL='1')
result = subprocess.run(object_args, env=e, capture_output=True, text=True)
(out / 'object-feature-off-refusal.log').write_text(result.stdout + result.stderr)
assert result.returncode == 1 and 'OBJECT_BINARY_POLICY_REFUSED' in result.stdout and 'SUBMIT_OK' not in result.stdout
(out / 'object-policy-summary.json').write_text(json.dumps(object_rows, indent=2) + '\n')
print('object policy cross-import refusals', len(object_rows), 'PASS', flush=True)

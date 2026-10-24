#!/usr/bin/env python3
"""CPU-only default activation and RDNA1 compiler/recording gate. Never renders."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--fixtures', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--baseline-icd', type=Path)
p.add_argument('--baseline-label', default='793a7c8')
p.add_argument('--checks-name', default='checks')
a = p.parse_args()
build, fixtures, out = a.build.resolve(), a.fixtures.resolve(), a.out.resolve()
out.mkdir(parents=True, exist_ok=True)
logs = out / a.checks_name
logs.mkdir(exist_ok=False)
icd = build / 'src/amd/vulkan/radeon_devenv_icd.x86_64.json'
shim = out / 'noop/libamdgpu_noop_drm_shim.so'
sys.path.insert(0, str(fixtures))
import automatic
rows = []

def launch(tag, gpu, command, extra=None, driver=icd):
    env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(shim), AMDGPU_GPU_ID=gpu,
               VK_DRIVER_FILES=str(driver), MESA_SHADER_CACHE_DISABLE='1',
               NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera')
    env.update(extra or {})
    cmd = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
           '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--', 'env', '-i',
           *(f'{k}={v}' for k, v in env.items()), *map(str, command)]
    process = subprocess.run(cmd, capture_output=True, timeout=60)
    log = process.stdout.decode() + process.stderr.decode()
    (logs / (tag + '.log')).write_text(log)
    assert process.returncode == 0 and not re.search(r'(?i)validation failed|ACO ERROR|Assertion|Segmentation|VALIDATION:.*Error', log), tag
    return log

for gpu in ('gfx1013', 'gfx1013_143f', 'navi10', 'navi12', 'navi14', 'renoir', 'navi21'):
    targeted = gpu in ('gfx1013', 'gfx1013_143f', 'navi10', 'navi12', 'navi14')
    options = [('default', {}, None), ('explicit', dict(RADV_DIRECTMESH='1'), None),
               ('disabled', dict(RADV_DIRECTMESH='0'), None),
               ('store-off', dict(RADV_BC250_MESH_IDXPASS_STORE='0'), None),
               ('task-off', dict(RADV_BC250_HYBRID_TASK='0'), None),
               ('dxvk', {}, 'DXVK'), ('dxvk-explicit-task', dict(RADV_BC250_HYBRID_TASK='1'), 'DXVK'),
               ('nongg', dict(RADV_DEBUG='nongg'), None), ('nomesh', dict(RADV_DEBUG='nomeshshader'), None)]
    if not targeted:
        options = options[:1]
    for label, extra, engine in options:
        tag = gpu + '-probe-' + label
        log = launch(tag, gpu, [out / 'probe', *([engine] if engine else [])], extra)
        probes = [dict(re.findall(r'(\w+)=(\S+)', line)) for line in log.splitlines() if line.startswith('PROBE ')]
        assert len(probes) == 2, tag
        for result in probes:
            if targeted:
                assert result['mesh'] == ('0' if label in ('disabled', 'nongg', 'nomesh') else '1'), tag
                assert result['preset'] == ('unset' if label == 'disabled' else '1'), tag
                assert result['store'] == ('unset' if label == 'disabled' else '0' if label == 'store-off' else 'auto'), tag
                if label == 'dxvk':
                    assert result['task'] == '0' and result['gpl'] == '1', tag
                if label == 'dxvk-explicit-task':
                    assert result['task'] == '1' and result['gpl'] == '0', tag
                if label == 'task-off':
                    assert result['task'] == '0' and result['gpl'] == '1', tag
                if not gpu.startswith('gfx1013'):
                    assert result['rt'] == '0', tag
            else:
                assert result['preset'] == result['store'] == 'unset' and result['gpl'] == '1', tag
        rows.append(dict(case=tag, kind='enumeration', instances=2))
        print('PASS', tag, flush=True)

cases = automatic.CASES + automatic.SAFETY
for gpu in ('gfx1013', 'navi10', 'navi12', 'navi14'):
    for case in cases:
        extra = dict(BC250_TRACE_COMPILE='1', BC250_TRACE_USAGE='1', BC250_IDX_TRACE='1',
                     BC250_IDX_CAPTURE_BATCH='1', BC250_CAPTURE_POLICY_SHADERS='1', RADV_DEBUG='dumpibs',
                     INDEX_PRIMITIVES=str(case['primitives']), INDEX_GROUPS=str(case['groups']),
                     INDEX_BENCH_DRAWS=str(case['draws']), INDEX_BENCH_SAMPLES='1', INDEX_VERTEX_STEPS=str(case['steps']))
        if case['draw'] != 'direct':
            extra['INDEX_INDIRECT'] = case['draw']
        if case.get('alternate_steps'):
            extra['INDEX_ALTERNATE_STEPS'] = '1'
        if 'pool_bytes' in case:
            extra['RADV_BC250_IDX_POOL_BYTES'] = str(case['pool_bytes'])
        snapshots = []
        tag = gpu + '-' + case['tag']
        for label in ('default', 'explicit'):
            options = dict(extra)
            if label == 'explicit':
                options['RADV_DIRECTMESH'] = '1'
            driver = a.baseline_icd.resolve() if label == 'explicit' and gpu == 'gfx1013' and a.baseline_icd else icd
            log = launch(tag + '-' + label, gpu,
                         [out / 'index_gate', '--offline', fixtures / (case['name'] + '.mesh.spv'),
                          fixtures / (case['fragment'] + '.frag.spv'), logs / (tag + '-' + label + '.rgba')], options, driver)
            assert 'NOOP_COMPLETE pixels_not_rendered' in log and 'VALIDATION_ERRORS=0' in log, tag
            if driver == icd:
                allocations = re.findall(r'BC250 IDX STORAGE: (page|pool) domain=(GTT|VRAM) cpu=(\d) wc=(\d) bytes=(\d+)', log)
                for kind, domain, cpu, wc, size in allocations:
                    expected = 'GTT' if gpu == 'gfx1013' else 'VRAM'
                    assert domain == expected and cpu == '0' and wc == str(int(domain == 'GTT')), (tag, allocations)
                if 'MESH AUTO: direct stored' in log or 'MESH AUTO: indirect stored' in log or case.get('auto_replay'):
                    assert allocations, 'missing indexed storage evidence: ' + tag
            shaders = re.findall(r'^BC250POLICYCODE ([0-9a-f]+)$', log, re.M)
            words = re.findall(r'^([0-9a-f]{8}) ', log, re.M)
            assert shaders and words, tag
            snapshots.append(dict(shaders=shaders, words=words))
        assert snapshots[0] == snapshots[1], 'changed shader or command words: ' + tag
        rows.append(dict(case=tag, kind='recording-pair', shader_identity=True, packet_identity=True,
                         baseline=a.baseline_label if gpu == 'gfx1013' and a.baseline_icd else 'explicit preset',
                         digest=hashlib.sha256(json.dumps(snapshots[0], sort_keys=True).encode()).hexdigest()))
        print('PASS', tag, flush=True)
result = dict(status='PASS', GPU_runs=0, rows=rows, launches=sum(2 if r['kind']=='recording-pair' else 1 for r in rows),
              scope='Enumeration and compiler/command-recording checks only. RDNA1 hardware rendering and performance unverified.',
              fixtures='Captured Navi10; synthetic GFX1013/Navi12/Navi14 identities with Navi10 topology.')
(out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print('PASS', result['launches'], 'noop launches; no GPU execution')

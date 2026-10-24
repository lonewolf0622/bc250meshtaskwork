#!/usr/bin/env python3
"""Compile Task, per-primitive, barycentric and fallback cases without a GPU."""
import argparse
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--checks-name', default='extended')
a = p.parse_args()
build, out = a.build.resolve(), a.out.resolve()
root = Path(__file__).resolve().parent.parent
work = out / a.checks_name
work.mkdir(exist_ok=False)

def compile(source, stage, name=None, options=()):
    result = work / ((name or source.stem) + '.' + stage + '.spv')
    subprocess.run(['glslangValidator', '-S', stage, '--target-env', 'vulkan1.3', *options,
                    '-o', str(result), str(source)], check=True, stdout=subprocess.DEVNULL)
    return result

shaders = {}
for name in ('j64', 'pp64', 'pp128', 'shared_vertex64', 'ballot_vertex64', 'branch64', 'nested64', 'dynamic64', 'empty64'):
    shaders[name] = compile(root / 'index-pass' / (name + '.mesh'), 'mesh')
for name in ('plain', 'pp'):
    shaders[name] = compile(root / 'index-pass' / (name + '.frag'), 'frag')
shaders['reference'] = compile(root / 'index-pass/j64.vert', 'vert')
shaders['task'] = compile(root / 'split-admission/t.task', 'task')
shaders['taskmesh'] = compile(root / 'split-admission/numwg_task.mesh', 'mesh')
shaders['taskfrag'] = compile(root / 'split-admission/f.frag', 'frag')
for name, opts in [('bary', ('-DBARY=1',)), ('pervertex', ('-DPERVERTEX=1',))]:
    shaders[name] = compile(root / 'barycentrics/bary.frag', 'frag', name, opts)
opts = ('-DPERPRIM=1', '-DPRIMID=1', '-DNOCULL=1')
shaders['primidmesh'] = compile(root / 'direct-read/dr.mesh', 'mesh', 'primid', opts)
shaders['primidfrag'] = compile(root / 'barycentrics/bary.frag', 'frag', 'primid', opts)
shaders['barymesh'] = compile(root / 'direct-read/dr.mesh', 'mesh', 'bary')
configs = []
for name in ('pp64', 'pp128'):
    for last in (0, 1):
        for route in ('mesh', 'stored'):
            configs.append((f'{name}-{last}-{route}', name, 'pp', None,
                            dict(INDEX_LAST_VERTEX=str(last), RADV_BC250_MESH_IDXPASS='0' if route=='mesh' else '1',
                                 RADV_BC250_MESH_IDXPASS_STORE='1')))
for name in ('shared_vertex64', 'ballot_vertex64', 'branch64', 'nested64', 'dynamic64', 'empty64'):
    configs.append((name, name, 'plain', None, {}))
for name, options in [('query', dict(INDEX_QUERY='1')), ('multiview', dict(INDEX_MULTIVIEW='1')),
                       ('reference', dict(INDEX_REFERENCE='1'))]:
    configs.append((name, 'reference' if name=='reference' else 'j64', 'plain', None, options))
configs.append(('debug-pool', 'j64', 'plain', None,
                dict(INDEX_INDIRECT='count', RADV_BC250_IDX_DEBUG='1')))
for mode in ('direct', 'count'):
    options = {} if mode == 'direct' else dict(INDEX_INDIRECT='count')
    configs.append(('task-' + mode, 'taskmesh', 'taskfrag', 'task', options))
for frag in ('bary', 'pervertex'):
    configs.append((frag, 'barymesh', frag, None, dict(RADV_BC250_MESH_IDXPASS='0')))
configs.append(('primid', 'primidmesh', 'primidfrag', None, dict(RADV_BC250_MESH_IDXPASS='0')))
rows = []
for gpu in ('gfx1013', 'navi10', 'navi12', 'navi14'):
    for label, mesh, frag, task, extra in configs:
        tag = gpu + '-' + label
        env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(out/'noop/libamdgpu_noop_drm_shim.so'),
                   AMDGPU_GPU_ID=gpu, VK_DRIVER_FILES=str(build/'src/amd/vulkan/radeon_devenv_icd.x86_64.json'),
                   MESA_SHADER_CACHE_DISABLE='1', NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera',
                   BC250_TRACE_COMPILE='1', BC250_TRACE_USAGE='1', BC250_IDX_TRACE='1', RADV_DEBUG='dumpibs', INDEX_PRIMITIVES='64')
        env.update(extra)
        # The existing packet pretty-printer asserts on a GFX10 query counter
        # enum. Query recording itself is checked by validation and route usage.
        if label == 'query':
            env.pop('RADV_DEBUG')
        command = [out/'index_gate', '--offline', shaders[mesh], shaders[frag], work/(tag+'.rgba')]
        if task:
            command.append(shaders[task])
        cmd = ['bwrap', '--ro-bind','/','/', '--dev','/dev', '--proc','/proc', '--tmpfs','/tmp',
               '--bind',str(out),str(out), '--unshare-pid','--die-with-parent','--','env','-i',
               *(f'{k}={v}' for k,v in env.items()), *map(str,command)]
        r = subprocess.run(cmd,capture_output=True,timeout=60)
        log = r.stdout.decode()+r.stderr.decode()
        (work/(tag+'.log')).write_text(log)
        assert r.returncode == 0 and 'VALIDATION_ERRORS=0' in log and 'NOOP_COMPLETE pixels_not_rendered' in log, tag
        assert not re.search(r'(?i)validation failed|ACO ERROR|Assertion|Segmentation|VALIDATION:.*Error',log),tag
        allocations = re.findall(r'BC250 IDX STORAGE: (page|pool) domain=(GTT|VRAM) cpu=(\d) wc=(\d) bytes=(\d+)', log)
        for kind, domain, cpu, wc, size in allocations:
            expected_cpu = label == 'debug-pool' and kind == 'pool'
            expected_domain = 'GTT' if expected_cpu or gpu == 'gfx1013' else 'VRAM'
            assert cpu == str(int(expected_cpu)) and domain == expected_domain, (tag, allocations)
            assert wc == str(int(not expected_cpu and domain == 'GTT')), (tag, allocations)
        if label == 'debug-pool':
            assert any(k == 'pool' and d == 'GTT' and c == '1' for k,d,c,w,n in allocations), tag
        if label in ('query','multiview'):
            assert 'idx_direct=0 ' in log, tag
        if gpu == 'navi14' and label not in ('reference', 'query'):
            late = re.findall(r'SPI_SHADER_LATE_ALLOC_GS = (\d+)',log)
            assert set(late) <= {'0'}, 'Navi14 late allocation: '+tag
            assert late or re.search(r'idx_direct=[1-9]', log), 'missing Navi14 stage evidence: '+tag
        rows.append(dict(case=tag,validation_errors=0))
        print('PASS',tag,flush=True)
(out/'extended-summary.json').write_text(json.dumps(dict(status='PASS',launches=len(rows),GPU_runs=0,cases=rows),indent=2)+'\n')
print('PASS',len(rows),'extended noop launches; no GPU execution')

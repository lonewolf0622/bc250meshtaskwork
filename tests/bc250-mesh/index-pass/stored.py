#!/usr/bin/env python3
"""Compile and record stored-output IDX in a sandbox without GPU nodes."""
import json
import os
import re
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
out = Path(os.environ['OUT']).resolve()
out.mkdir(parents=True, exist_ok=True)
icd = Path(os.environ['ICD']).resolve()
shim = Path(os.environ['SHIM']).resolve()
rows = []

def run(name, mode='stored', extra=None, expected=1, fragment='plain'):
    extra = extra or {}
    tag = '-'.join([name, mode] + [f'{k}={v}' for k, v in extra.items()])
    env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(shim), AMDGPU_GPU_ID='gfx1013',
               VK_DRIVER_FILES=str(icd), MESA_SHADER_CACHE_DISABLE='1', RADV_DIRECTMESH='1',
               RADV_BC250_MESH_IDXPASS='0' if mode == 'mesh' else '1',
               RADV_BC250_MESH_IDXPASS_STORE='1' if mode == 'stored' else '0',
               NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera', BC250_TRACE_COMPILE='1',
               BC250_TRACE_USAGE='1', BC250_IDX_TRACE='1', INDEX_PRIMITIVES='64')
    if mode == 'reference':
        env['INDEX_REFERENCE'] = '1'
    env.update(extra)
    cmd = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
           '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--', 'env', '-i',
           *(f'{k}={v}' for k, v in env.items()), str(here / 'index_gate'), '--offline',
           str(here / f'{name}{"_ind" if mode == "reference" and "INDEX_INDIRECT" in extra else ""}.{"vert" if mode == "reference" else "mesh"}.spv'),
           str(here / f'{fragment}.frag.spv'), str(out / f'{tag}.rgba')]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    log = r.stdout.decode(errors='replace')
    (out / f'{tag}.log').write_text(log)
    assert r.returncode == 0 and 'VALIDATION_ERRORS=0' in log, (tag, r.returncode)
    assert not re.search(r'validation failed|ACO ERROR|VALIDATION:.*Error|Assertion', log), tag
    usage = re.findall(r'idx_direct=(\d+) idx_indirect_calls=(\d+)', log)
    assert usage, tag
    direct, indirect = map(int, usage[-1])
    assert direct + indirect == expected, (tag, direct, indirect, expected)
    if mode == 'stored':
        assert 'stored outputs bytes/group=' in log, tag
    rows.append(dict(case=tag, direct=direct, indirect=indirect, validation_errors=0))
    print(f'PASS {tag} direct={direct} indirect={indirect}', flush=True)
    return log

for name in ('j64', 'ballot_vertex64', 'shared_vertex64', 'branch64', 'nested64', 'dynamic64', 'empty64'):
    for mode in ('stored', 'mesh', 'reference'):
        run(name, mode, expected=int(mode == 'stored'))
    for draw in ('multi', 'count', 'single'):
        run(name, extra={'INDEX_INDIRECT': draw}, expected=int(draw != 'single'))
    run(name, extra={'INDEX_CLEAR_BETWEEN': '1'}, expected=2)
    run(name, extra={'RADV_BC250_MESH_IDXPASS_BATCH': '0'})
for name in ('pp64', 'pp128'):
    for last in ('0', '1'):
        for draw in ('multi', 'count'):
            run(name, extra={'INDEX_LAST_VERTEX': last, 'INDEX_INDIRECT': draw})
        run(name, extra={'INDEX_LAST_VERTEX': last})
# Stored and replay shaders must also preserve descriptor/push state around clear and query fallback.
run('ballot_vertex64', extra={'INDEX_QUERY': '1'}, expected=0)
log = run('j64', extra={'INDEX_GRID': '65535,32,1'}, expected=0)
assert 'direct Mesh fallback (storage unavailable)' in log
for mode in ('free', 'reset'):
    log = run('ballot_vertex64', extra={'INDEX_CYCLE': mode, 'INDEX_RECORDINGS': '80'}, expected=1)
    assert 'storage unavailable' not in log
log = run('ballot_vertex64', extra={'INDEX_RECORDINGS': '160'}, expected=0)
assert log.count('direct Mesh fallback (storage unavailable)') >= 16
# GPU overflow decisions cannot execute under noop. These cases verify compilation and command recording;
# output bounds and overflow pixels still require the one-shot hardware gate.
run('ballot_vertex64', extra={'INDEX_INDIRECT': 'count', 'RADV_BC250_IDX_POOL_BYTES': '4096'})

for name in ('j64', 'ballot_vertex64', 'shared_vertex64', 'pp64'):
    for draw in ('multi', 'count', 'single'):
        run(name, mode='reference', extra={'INDEX_INDIRECT': draw}, expected=0,
            fragment='pp_ref' if name == 'pp64' else 'plain')
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

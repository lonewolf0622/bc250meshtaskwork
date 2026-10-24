#!/usr/bin/env python3
"""Audit indirect setup lifetime and fallback recording without GPU nodes."""
import json
import os
from pathlib import Path
import re
import subprocess

import benchmark as bench

out = Path(os.environ['OUT']).resolve()
out.mkdir(parents=True, exist_ok=True)
icd = Path(os.environ['ICD']).resolve()
shim = Path(os.environ['SHIM']).resolve()
configs = [('clear', dict(INDEX_CLEAR_BETWEEN='1'), 2, 2),
           ('discard', dict(INDEX_DISCARD_RECORDING='1'), 2, 1),
           ('cycle-free', dict(INDEX_CYCLE='free', INDEX_RECORDINGS='80'), 80, 1),
           ('cycle-reset', dict(INDEX_CYCLE='reset', INDEX_RECORDINGS='80'), 80, 1),
           ('query', dict(INDEX_QUERY='1'), 0, 0),
           ('inline', dict(RADV_BC250_MESH_IDXPASS_BATCH='0'), 1, 1),
           ('single-policy', dict(INDEX_INDIRECT='single'), 0, 0)]
configs += [(name + '-pair', dict(extra, INDEX_REPEAT_INDIRECT='2'), total * 2, last * 2)
            for name, extra, total, last in configs if name in ('clear', 'discard', 'cycle-free', 'cycle-reset', 'inline')]
rows = []
for stored in (0, 1):
    for name, extra, total_calls, last_calls in configs:
        tag = f'{name}-stored{stored}'
        env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(shim), AMDGPU_GPU_ID='gfx1013',
                   VK_DRIVER_FILES=str(icd), MESA_SHADER_CACHE_DISABLE='1', RADV_DIRECTMESH='1',
                   RADV_BC250_MESH_IDXPASS='1', RADV_BC250_MESH_IDXPASS_STORE=str(stored),
                   RADV_BC250_MESH_IDXPASS_SETUP_BATCH='1', BC250_TRACE_COMPILE='1', BC250_TRACE_USAGE='1',
                   NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera', BC250_IDX_TRACE='1',
                   INDEX_INDIRECT='count', INDEX_PRIMITIVES='64')
        env.update(extra)
        command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                   '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--',
                   'env', '-i', *(f'{k}={v}' for k, v in env.items()), str(bench.HERE / 'index_gate'), '--offline',
                   str(bench.HERE / 'j64.mesh.spv'), str(bench.HERE / 'plain.frag.spv'), str(out / (tag + '.rgba'))]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        log = result.stdout.decode(errors='replace')
        (out / (tag + '.log')).write_text(log)
        assert not result.returncode and not re.search(bench.BAD, log), tag
        assert 'VALIDATION_ERRORS=0' in log and 'NOOP_COMPLETE pixels_not_rendered' in log, tag
        usage = re.findall(r'idx_direct=(\d+) idx_indirect_calls=(\d+)', log)
        assert usage and tuple(map(int, usage[-1])) == (0, last_calls), tag
        calls = re.findall(r'BC250 IDX SETUP: call shared=(\d+) dispatch=(\d+) wait=(\d+) reset=(\d+)', log)
        assert len(calls) == total_calls, tag
        shared = not name.startswith('inline')
        assert all(int(c[0]) == shared and c[1] == c[2] for c in calls), tag
        batches = bench.audit_batches(log, 1)
        spans = re.findall(r'BC250 IDX SETUP: close calls=(\d+) dispatches=(\d+) waits=(\d+) bytes=(\d+) shared=(\d+)', log)
        assert len(spans) == len(batches), tag
        paired = name.endswith('-pair')
        assert all(int(s[4]) == (int(s[0]) > 1) for s in spans), tag
        promotions = re.findall(r'BC250 IDX SETUP: promote .*sync_dw=(\d+) tail=0x([0-9a-f]+)', log)
        assert len(promotions) == sum(int(s[4]) for s in spans) + int(name == 'discard-pair'), tag
        assert all(int(sync) > 2 and int(tail, 16) == 0xc0004200 for sync, tail in promotions), tag
        if name.startswith('discard'):
            assert 'INDEX_DISCARDED_OPEN_RECORDING' in log and len(spans) == 1 and int(spans[0][0]) == (2 if paired else 1), tag
            assert [c[3] for c in calls] == (['1', '0'] * 2 if paired else ['1'] * 2), tag
        else:
            assert sum(int(s[0]) for s in spans) == (total_calls if shared else 0), tag
        if name.startswith('cycle'):
            assert len(spans) == 80 and [c[3] for c in calls] == (['1', '0'] * 80 if paired else ['1'] * 80) and 'storage unavailable' not in log, tag
        if name.startswith('clear'):
            assert [c[3] for c in calls] == (['1', '0', '0', '0'] if paired else ['1', '0']), tag
        rows.append(dict(case=tag, calls=total_calls, completed_spans=len(spans), validation_errors=0))
        print('PASS', tag, flush=True)
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print('PASS', len(rows), 'indirect setup lifetime cases', flush=True)

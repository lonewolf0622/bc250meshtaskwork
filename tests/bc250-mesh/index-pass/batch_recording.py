#!/usr/bin/env python3
"""Audit IDX batch lifetime paths with real Vulkan recording and no GPU nodes."""
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
CONFIGS = [('clear', dict(INDEX_CLEAR_BETWEEN='1'), 2, True),
           ('discard', dict(INDEX_DISCARD_RECORDING='1'), 1, True),
           ('cycle-free', dict(INDEX_CYCLE='free', INDEX_RECORDINGS='80'), 1, True),
           ('cycle-reset', dict(INDEX_CYCLE='reset', INDEX_RECORDINGS='80'), 1, True),
           ('query', dict(INDEX_QUERY='1'), 0, False),
           ('inline', dict(RADV_BC250_MESH_IDXPASS_BATCH='0'), 1, False),
           ('reservation-fallback', dict(INDEX_GRID='65535,32,1'), 0, False)]
rows = []
for stored in (0, 1):
    for trim in ('default', 0, 1):
        expected_trim = 1 if trim == 'default' else trim
        for name, extra, expected, batched in CONFIGS:
            tag = f'{name}-stored{stored}-trim{trim}'
            env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(shim), AMDGPU_GPU_ID='gfx1013',
                       VK_DRIVER_FILES=str(icd), MESA_SHADER_CACHE_DISABLE='1', RADV_DIRECTMESH='1',
                       RADV_BC250_MESH_IDXPASS='1', RADV_BC250_MESH_IDXPASS_STORE=str(stored),
                       NIR_DEBUG='validate',
                       ACO_DEBUG='validateir,validatera', BC250_TRACE_COMPILE='1', BC250_TRACE_USAGE='1',
                       BC250_IDX_TRACE='1', INDEX_PRIMITIVES='64', **extra)
            if trim != 'default':
                env['RADV_BC250_MESH_IDXPASS_BATCH_TRIM'] = str(trim)
            command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                       '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--',
                       'env', '-i', *(f'{k}={v}' for k, v in env.items()), str(bench.HERE / 'index_gate'),
                       '--offline', str(bench.HERE / 'j64.mesh.spv'), str(bench.HERE / 'plain.frag.spv'),
                       str(out / (tag + '.rgba'))]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            log = result.stdout.decode(errors='replace')
            (out / (tag + '.log')).write_text(log)
            assert result.returncode == 0 and not re.search(bench.BAD, log), tag
            assert 'VALIDATION_ERRORS=0' in log and 'NOOP_COMPLETE pixels_not_rendered' in log, tag
            usage = re.findall(r'idx_direct=(\d+) idx_indirect_calls=(\d+)', log)
            assert usage and tuple(map(int, usage[-1])) == (expected, 0), tag
            batches = bench.audit_batches(log, expected_trim)
            assert bool(batches) == batched, tag
            if name.startswith('cycle'):
                assert len(batches) == 80 and 'storage unavailable' not in log, tag
            if name == 'discard':
                assert 'INDEX_DISCARDED_OPEN_RECORDING' in log and len(batches) == 1, tag
            rows.append(dict(case=tag, stored=stored, trim=expected_trim,
                             trim_override_set=trim != 'default', batches=batches, validation_errors=0))
            print('PASS', tag, flush=True)
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print('PASS', len(rows), 'batch lifetime recordings', flush=True)

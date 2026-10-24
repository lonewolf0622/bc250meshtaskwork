#!/usr/bin/env python3
"""Compare CPU-side shader uploads from two ICDs in a DRM-free noop sandbox."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
out = Path(os.environ['OUT']).resolve()
out.mkdir(parents=True, exist_ok=True)
shim = Path(os.environ['SHIM']).resolve()
icds = [(tag, Path(os.environ[key]).resolve()) for tag, key in
        [('baseline', 'BASELINE_ICD'), ('candidate', 'CANDIDATE_ICD')]]
rows = []
for name in ('l32', 's64', 'h64', 'j64', 'm124', 'l128', 'n256', 'pp64', 'pp128',
             'ballot64', 'ballot_vertex64', 'shared_vertex64', 'branch64', 'nested64', 'dynamic64', 'empty64'):
    for mode in ('off', 'on', 'stored'):
        snapshots = []
        for tag, icd in icds:
            env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(shim), AMDGPU_GPU_ID='gfx1013',
                       VK_DRIVER_FILES=str(icd), MESA_SHADER_CACHE_DISABLE='1', RADV_DIRECTMESH='1',
                       RADV_BC250_MESH_IDXPASS='0' if mode == 'off' else '1',
                       RADV_BC250_MESH_IDXPASS_STORE='1' if mode == 'stored' else '0',
                       NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera', BC250_CAPTURE_POLICY_SHADERS='1',
                       INDEX_PRIMITIVES=str(32 if name == 'l32' else 124 if name == 'm124' else
                                            128 if name in ('l128', 'n256', 'pp128') else 64))
            frag = 'pp' if name.startswith('pp') else 'plain'
            case = f'{name}-{mode}-{tag}'
            command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                       '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--',
                       'env', '-i', *(f'{k}={v}' for k, v in env.items()), str(here / 'index_gate'), '--offline',
                       str(here / f'{name}.mesh.spv'), str(here / f'{frag}.frag.spv'), str(out / f'{case}.rgba')]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            log = result.stdout.decode(errors='replace')
            (out / f'{case}.log').write_text(log)
            assert not result.returncode and 'VALIDATION_ERRORS=0' in log, (case, result.returncode)
            assert not re.search(r'validation failed|ACO ERROR|VALIDATION:.*Error|Assertion', log), case
            headers = re.findall(r'^BC250POLICY (.*)$', log, re.M)
            code = [bytes.fromhex(hexcode) for hexcode in re.findall(r'^BC250POLICYCODE ([0-9a-f]+)$', log, re.M)]
            assert headers and len(headers) == len(code), case
            # GPU virtual placement is allocator state, rather than shader semantics.
            headers = [re.sub(r' va=[0-9a-f]+', '', header) for header in headers]
            snapshots.append(list(zip(headers, code)))
        assert snapshots[0] == snapshots[1], f'{name}-{mode}: uploaded shader bytes or configuration changed'
        rows.append(dict(case=name, mode=mode, byte_identical=True,
                         shaders=[dict(configuration=header, sha256=hashlib.sha256(code).hexdigest())
                                  for header, code in snapshots[1]]))
        print(f'PASS {name} IDX={mode}: {len(snapshots[1])} identical shader uploads', flush=True)
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

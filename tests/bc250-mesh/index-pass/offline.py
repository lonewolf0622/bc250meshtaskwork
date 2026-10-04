#!/usr/bin/env python3
"""Compile and record index-route cases in a DRM-free noop sandbox."""
import os, re, subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
icd = Path(os.environ['ICD']).resolve()
shim = Path(os.environ['SHIM']).resolve()
out = Path(os.environ.get('OUT', str(here / 'results'))).resolve()
out.mkdir(parents=True, exist_ok=True)
for name in ('l32', 's64', 'h64', 'j64', 'm124', 'l128', 'n256', 'pp64', 'pp128', 'ballot64', 'ballot_vertex64'):
    for mode in ('off', 'on', 'reference'):
        if name == 'ballot_vertex64' and mode == 'reference':
            continue
        for last in ((0, 1) if name.startswith('pp') else (0,)):
            env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(shim), AMDGPU_GPU_ID='gfx1013',
                       VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd), MESA_SHADER_CACHE_DISABLE='1',
                       RADV_DIRECTMESH='1', RADV_BC250_MESH_IDXPASS='0' if mode == 'off' else '1',
                       NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera', BC250_TRACE_COMPILE='1',
                       INDEX_PRIMITIVES='124' if name == 'm124' else ('128' if name in ('l128', 'n256', 'pp128') else ('32' if name == 'l32' else '64')))
            if mode == 'reference': env['INDEX_REFERENCE'] = '1'
            if last: env['INDEX_LAST_VERTEX'] = '1'
            suffix = 'vert' if mode == 'reference' else 'mesh'
            frag = ('pp_ref' if mode == 'reference' else 'pp') if name.startswith('pp') else 'plain'
            tag = f'{name}-{mode}-{last}'
            cmd = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                   '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--',
                   'env', '-i', *(f'{k}={v}' for k, v in env.items()), str(here / 'index_gate'), '--offline',
                   str(here / f'{name}.{suffix}.spv'), str(here / f'{frag}.frag.spv'), str(out / f'{tag}.rgba')]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            log = r.stdout.decode(errors='replace')
            (out / f'{tag}.log').write_text(log)
            applied = 'MESH IDXPASS: applied' in log
            expected = mode == 'on' and name not in ('l32', 'ballot_vertex64')
            if r.returncode or re.search(r'validation failed|ACO ERROR|VALIDATION:.*Error|Assertion', log) or applied != expected:
                raise SystemExit(f'FAIL {tag} rc={r.returncode} applied={applied} expected={expected}; see {out}')
            print(f'PASS {tag} applied={applied}', flush=True)
# Retain enough independent recordings to exhaust the shared reservation budget.
# Only the final buffer is submitted to the noop device; no GPU nodes are exposed.
env.update(RADV_BC250_MESH_IDXPASS='1', INDEX_RECORDINGS='80', BC250_IDX_TRACE='1')
env.pop('INDEX_LAST_VERTEX', None)
env.pop('INDEX_REFERENCE', None)
cmd = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
       '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--', 'env', '-i',
       *(f'{k}={v}' for k, v in env.items()), str(here / 'index_gate'), '--offline',
       str(here / 'j64.mesh.spv'), str(here / 'plain.frag.spv'), str(out / 'pressure.rgba')]
r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
log = r.stdout.decode(errors='replace')
(out / 'pressure.log').write_text(log)
if r.returncode or log.count('direct Mesh fallback (storage unavailable)') < 16 or 'VALIDATION_ERRORS=0' not in log:
    raise SystemExit(f'FAIL memory pressure rc={r.returncode}; see {out}')
print('PASS memory pressure: bounded storage falls back without command buffer errors', flush=True)

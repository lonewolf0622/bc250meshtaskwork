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

def run(tag, mesh, extra):
    e = dict(env)
    for k in ('INDEX_RECORDINGS', 'BC250_IDX_TRACE', 'INDEX_LAST_VERTEX', 'INDEX_REFERENCE'):
        e.pop(k, None)
    e.update(RADV_BC250_MESH_IDXPASS='1', INDEX_PRIMITIVES='64', BC250_TRACE_USAGE='1', **extra)
    cmd = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
           '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--', 'env', '-i',
           *(f'{k}={v}' for k, v in e.items()), str(here / 'index_gate'), '--offline',
           str(here / mesh), str(here / 'plain.frag.spv'), str(out / f'{tag}.rgba')]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    log = r.stdout.decode(errors='replace')
    (out / f'{tag}.log').write_text(log)
    if r.returncode or re.search(r'validation failed|ACO ERROR|VALIDATION:.*Error|Assertion', log) or \
       'VALIDATION_ERRORS=0' not in log:
        raise SystemExit(f'FAIL {tag} rc={r.returncode}; see {out}')
    usage = re.findall(r'idx_direct=(\d+)', log)
    return log, int(usage[-1]) if usage else 0

# A meta operation (vkCmdClearAttachments) between two draws: both draws keep the index route.
log, n = run('clear-between', 'j64.mesh.spv', dict(INDEX_CLEAR_BETWEEN='1'))
if n != 2:
    raise SystemExit(f'FAIL clear-between: idx_direct={n}, expected 2 (draw after the clear lost the route)')
print('PASS clear-between: both draws on the index route', flush=True)
# Pipeline-statistics query active: the draw keeps the Mesh route.
log, n = run('query', 'j64.mesh.spv', dict(INDEX_QUERY='1'))
if n != 0 or 'MESH IDXPASS: applied' not in log:
    raise SystemExit(f'FAIL query: idx_direct={n}, expected 0 with an applied pipeline')
print('PASS query: Mesh route while a counting query is active', flush=True)
# Multiview render pass: the index route is declined at pipeline creation.
# The Mesh route needs the per-vertex layer export for this shape on GFX10.1.
log, n = run('multiview', 'j64.mesh.spv', dict(INDEX_MULTIVIEW='1', RADV_BC250_MESH_MULTIVIEW_VTX='1'))
if 'MESH IDXPASS: declined (multiview)' not in log or 'MESH IDXPASS: applied' in log or n:
    raise SystemExit('FAIL multiview: index route not declined')
print('PASS multiview: declined', flush=True)
# Recycled storage: 80 recordings submitted one after another, each in a new command buffer that is then freed or
# reset and left idle; returned storage is reused, so none runs out of the 64 MiB budget.
for mode in ('free', 'reset'):
    log, n = run(f'cycle-{mode}', 'j64.mesh.spv', dict(INDEX_CYCLE=mode, INDEX_RECORDINGS='80', BC250_IDX_TRACE='1'))
    if 'storage unavailable' in log or 'idx_direct=1 ' not in log or log.count('INDEX_READY') != 1:
        raise SystemExit(f'FAIL cycle-{mode}: storage fallback or no index route; see {out}')
    print(f'PASS cycle-{mode}: 80 recordings reuse storage', flush=True)

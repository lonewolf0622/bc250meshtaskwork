#!/usr/bin/env python3
"""Compare release profiles to their hardware-tested settings in fresh noop devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--fixtures', required=True, type=Path)
    parser.add_argument('--shim', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root, fixtures, out = (p.resolve() for p in (args.release, args.fixtures, args.out))
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(fixtures))
    import benchmark as bench
    checks = []
    cases = [(p, n, 1) for p in ('standard', 'mesh', 'stored') for n in (1, 2)] + [('standard', 2, 0)]
    for profile, draws, shared in cases:
        snapshots = []
        for wrapped in (False, True):
            tag = f'{profile}-draws{draws}-shared{shared}-wrapped{int(wrapped)}'
            icd = str(root / 'radeon_icd.json')
            env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(args.shim.resolve()),
                       AMDGPU_GPU_ID='gfx1013', MESA_SHADER_CACHE_DISABLE='1',
                       BC250_IDX_TRACE='1', BC250_IDX_CAPTURE_BATCH='1', BC250_TRACE_COMPILE='1',
                       BC250_CAPTURE_POLICY_SHADERS='1', RADV_DEBUG='dumpibs',
                       NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera',
                       INDEX_INDIRECT='count', INDEX_PRIMITIVES='64', INDEX_GROUPS='1',
                       INDEX_BENCH_DRAWS=str(draws), INDEX_BENCH_SAMPLES='1')
            if not shared:
                env['RADV_BC250_MESH_IDXPASS_SETUP_BATCH'] = '0'
            if not wrapped:
                env.update(VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd, RADV_DIRECTMESH='1',
                           RADV_BC250_MESH_IDXPASS='0' if profile == 'mesh' else '1',
                           RADV_BC250_MESH_IDXPASS_STORE='1' if profile == 'stored' else '0',
                           RADV_BC250_MESH_IDXPASS_SETUP_BATCH=str(shared),
                           RADV_BC250_MESH_IDXPASS_BATCH_TRIM='1')
            command = [str(fixtures / 'index_gate'), '--offline', str(fixtures / 'profile64.mesh.spv'),
                       str(fixtures / 'plain.frag.spv'), str(out / (tag + '.rgba'))]
            if wrapped:
                command = ['bash', str(root / 'launch.sh'), '--profile', profile, '--', *command]
            command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                       '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--', 'env', '-i',
                       *(f'{k}={v}' for k, v in env.items()), *command]
            result = subprocess.run(command, capture_output=True, timeout=60)
            log = result.stdout.decode() + result.stderr.decode()
            (out / (tag + '.log')).write_text(log)
            assert result.returncode == 0 and not re.search(bench.BAD, log), tag
            assert 'INDEX_READY mode=noop' in log and 'NOOP_COMPLETE pixels_not_rendered' in log, tag
            assert 'VALIDATION_ERRORS=0' in log, tag
            code = re.findall(r'^BC250POLICYCODE ([0-9a-f]+)$', log, re.M)
            words = re.findall(r'^([0-9a-f]{8}) ', log, re.M)
            assert code and words, 'missing shader/packet evidence: ' + tag
            snapshot = dict(shaders=code, command_words=words)
            if profile != 'mesh':
                mode = 'stored' if profile == 'stored' else 'existing-before'
                case = dict(name='profile64', draw='count', draws=draws, setup_batch=shared)
                setup = bench.audit_setups(log, case, mode)
                assert setup['dispatches'] == setup['waits'] == (1 if shared else draws), tag
                if shared:
                    assert bool(setup['spans'][0]['shared']) == (draws > 1), tag
                else:
                    assert not setup['spans'], tag
                snapshot['setups'] = setup
            else:
                assert 'MESH IDXPASS: applied' not in log, tag
            snapshots.append(snapshot)
        assert snapshots[0] == snapshots[1], 'release profile changes tested recording: ' + tag
        checks.append(dict(profile=profile, draws=draws, shared=shared,
                           shader_and_packet_identity=True,
                           packet_sha256=hashlib.sha256(json.dumps(snapshots[0], sort_keys=True).encode()).hexdigest()))
    (out / 'summary.json').write_text(json.dumps(dict(status='PASS', launches=14, GPU_runs=0, cases=checks), indent=2) + '\n')
    print('PASS 14 fresh-device noop launches; release settings match tested shaders and packets')


if __name__ == '__main__':
    main()

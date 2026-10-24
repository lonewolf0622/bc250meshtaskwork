#!/usr/bin/env python3
"""Compare an updated selector with the measured driver's chosen command paths."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measured-kit', type=Path, required=True)
    parser.add_argument('--candidate-icd', type=Path, required=True)
    parser.add_argument('--shim', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    kit, out = args.measured_kit.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(kit))
    import automatic as measured
    assert Path(measured.__file__).resolve().parent == kit
    checks = []
    for case in measured.CASES + measured.SAFETY:
        correction = (case['name'] == 'profile256pp128' and case['draw'] == 'count' and case['steps'] == 64)
        snapshots = []
        for label, icd in [('measured', kit / 'radeon_icd.json'), ('candidate', args.candidate_icd.resolve())]:
            tag = case['tag'] + '-' + label
            env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(args.shim.resolve()),
                       AMDGPU_GPU_ID='gfx1013', VK_DRIVER_FILES=str(icd),
                       MESA_SHADER_CACHE_DISABLE='1', RADV_DIRECTMESH='1', BC250_TRACE_COMPILE='1',
                       BC250_IDX_TRACE='1', BC250_IDX_CAPTURE_BATCH='1', BC250_CAPTURE_POLICY_SHADERS='1',
                       RADV_DEBUG='dumpibs', NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera',
                       INDEX_PRIMITIVES=str(case['primitives']), INDEX_GROUPS=str(case['groups']),
                       INDEX_BENCH_DRAWS=str(case['draws']), INDEX_BENCH_SAMPLES='1',
                       INDEX_VERTEX_STEPS=str(case['steps']))
            if label == 'measured' and correction:
                env['RADV_BC250_MESH_IDXPASS_STORE'] = '1'
            if case['draw'] != 'direct':
                env['INDEX_INDIRECT'] = case['draw']
            if case.get('alternate_steps'):
                env['INDEX_ALTERNATE_STEPS'] = '1'
            if 'pool_bytes' in case:
                env['RADV_BC250_IDX_POOL_BYTES'] = str(case['pool_bytes'])
            command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                       '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--', 'env', '-i',
                       *(f'{k}={v}' for k, v in env.items()), str(kit / 'index_gate'), '--offline',
                       str(kit / (case['name'] + '.mesh.spv')), str(kit / (case['fragment'] + '.frag.spv')),
                       str(out / (tag + '.rgba'))]
            process = subprocess.run(command, capture_output=True, timeout=60)
            log = process.stdout.decode() + process.stderr.decode()
            (out / (tag + '.log')).write_text(log)
            assert process.returncode == 0 and 'NOOP_COMPLETE pixels_not_rendered' in log, tag
            assert 'VALIDATION_ERRORS=0' in log and not re.search(r'(?i)validation failed|ACO ERROR|Assertion|Segmentation', log), tag
            code = re.findall(r'^BC250POLICYCODE ([0-9a-f]+)$', log, re.M)
            packets = re.findall(r'^([0-9a-f]{8}) ', log, re.M)
            assert code and packets, tag
            snapshots.append(dict(shaders=code, packets=packets))
            if label == 'candidate' and correction:
                assert log.count('BC250 MESH AUTO: indirect stored') == case['draws'], tag
        assert snapshots[0] == snapshots[1], 'selector changed measured GPU path: ' + case['tag']
        checks.append(dict(case=case['tag'], measured_path='forced stored' if correction else 'automatic',
                           shader_identity=True, packet_identity=True,
                           digest=hashlib.sha256(json.dumps(snapshots[0], sort_keys=True).encode()).hexdigest()))
    result = dict(status='PASS', comparisons=len(checks), launches=2 * len(checks), GPU_runs=0, cases=checks,
                  scope='Byte-identical noop shader uploads and recorded command words; '
                        'not a new hardware timing measurement of the corrected binary')
    (out / 'recording-equivalence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'PASS {len(checks)} measured-path recording comparisons; no GPU execution')


if __name__ == '__main__':
    main()

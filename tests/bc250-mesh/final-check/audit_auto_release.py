#!/usr/bin/env python3
"""Re-audit the packaged owner data and CPU recording proof without Vulkan."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parent
    subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=root, check=True)
    sys.path.insert(0, str(root / 'tools'))
    from audit_automatic import audit
    measured, fingerprints = audit(root / 'evidence/measured-kit')
    assert measured == json.loads((root / 'evidence/owner-audit/automatic-audit.json').read_text())
    expected = dict(line.split('  ', 1)[::-1] for line in
                    (root / 'evidence/owner-audit/automatic-artifacts.sha256').read_text().splitlines())
    assert fingerprints == expected
    proof = root / 'evidence/recording-equivalence'
    records = json.loads((proof / 'recording-equivalence.json').read_text())
    assert records['status'] == 'PASS' and records['comparisons'] == 19 and records['GPU_runs'] == 0
    corrections = []
    for case in records['cases']:
        snapshots = []
        for label in ('measured', 'candidate'):
            log = (proof / (case['case'] + '-' + label + '.log')).read_text()
            assert 'NOOP_COMPLETE pixels_not_rendered' in log and 'VALIDATION_ERRORS=0' in log
            code = re.findall(r'^BC250POLICYCODE ([0-9a-f]+)$', log, re.M)
            packets = re.findall(r'^([0-9a-f]{8}) ', log, re.M)
            assert code and packets
            snapshots.append(dict(shaders=code, packets=packets))
        assert snapshots[0] == snapshots[1], case['case']
        assert hashlib.sha256(json.dumps(snapshots[0], sort_keys=True).encode()).hexdigest() == case['digest']
        if case['measured_path'] == 'forced stored':
            corrections.append(case['case'])
    assert corrections == ['profile256pp128-count-steps64-groups8-draws32']
    print(json.dumps(dict(status='PASS measured data and corrected recording proof',
                          exact_owner_route_images=measured['exact_route_images'],
                          measured_confirmed_automatic_wins=len(measured['confirmed_wins']),
                          original_policy_missed_comparisons=measured['stable_missed_route_comparisons'],
                          corrected_cases=corrections, recording_equivalence_cases=19,
                          corrected_binary_separately_timed=False, GPU_runs=0), indent=2))


if __name__ == '__main__':
    main()

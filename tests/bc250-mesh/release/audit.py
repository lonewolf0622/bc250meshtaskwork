#!/usr/bin/env python3
"""Verify the sealed release and re-audit both saved owner runs without Vulkan."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=root, check=True)
sys.path.insert(0, str(root / 'tools/corner'))
from audit_corner_alias import audit

corner, artifacts = audit(root / 'evidence/corner/raw')
expected = dict(line.strip().split('  ', 1)[::-1] for line in
                (root / 'evidence/corner/corner-artifacts.sha256').read_text().splitlines())
assert artifacts == expected, 'corner evidence fingerprint mismatch'
assert corner == json.loads((root / 'evidence/corner/corner-audit.json').read_text())
with tempfile.TemporaryDirectory(prefix='directmesh-release-audit-') as temp:
    work = Path(temp)
    prior = root / 'evidence/prior'
    shutil.copy2(prior / 'manifest.json', work / 'manifest.json')
    shutil.copy2(prior / 'audit_benchmark.py', work / 'audit_benchmark.py')
    (work / 'benchmark').symlink_to(prior / 'benchmark', target_is_directory=True)
    with (work / 'audit.log').open('w') as log:
        process = subprocess.run([sys.executable, '-B', str(work / 'audit_benchmark.py')],
                                 stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError((work / 'audit.log').read_text()[-5000:])
    result = json.loads((work / 'results/hardware-validation.json').read_text())
    assert result == json.loads((prior / 'results/hardware-validation.json').read_text())
    assert (work / 'results/hardware-artifacts.sha256').read_bytes() == (
        prior / 'results/hardware-artifacts.sha256').read_bytes()
print(json.dumps(dict(status='PASS', exact_route_images=result['exact_route_images'] + corner['exact_route_images'],
                      prior_shared_setup_confirmations=len(result['confirmed_setup_wins']),
                      corner_confirmed_wins=corner['confirmed_wins'], shipping_driver='c84115b',
                      setup_default='enabled by launcher for repeated IDX calls', stored_default=False,
                      corner_reuse_shipped=False, GPU_runs=0), indent=2))

"""Audit saved setup-batching GPU results without executing GPU programs."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
kit = root / 'benchmark'
out = root / 'results'
out.mkdir(exist_ok=True)
manifest = json.loads((root / 'manifest.json').read_text())
sys.path.insert(0, str(kit))
import benchmark as bench
import light
import setup_batch as gate
import setup_final as work
import setup_focus as focus

subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=kit, check=True)
s = json.loads((kit / 'summary.json').read_text())
assert not s['offline'] and s['cases'] == len(work.CASES) == 24
assert not any(s[k] for k in ('setup_batch_default', 'stored_IDX_default', 'automatic_route_policy_changed'))
assert 'PASS final repeats; retain telemetry and excluded timings' in (kit / 'hw.progress').read_text()
counts = dict(references=28, exact_corner_oracles=360, indirect_pool_preflights=72,
              trials=192, comparisons=48)
assert {k: len(s[k]) for k in counts} == counts
rows = {}
for category in ('exact_corner_oracles', 'indirect_pool_preflights', 'trials'):
    for row in s[category]:
        key = (row['case'], row['mode'], row['preflight'])
        assert key not in rows
        rows[key] = row
comparisons = {(r['case'], r['route']): r for r in s['comparisons']}
assert len(comparisons) == 48
artifacts, refs, used_rows = {}, {}, set()
submissions = samples = checks = launches = batches_total = 0
exact_checks = 0
warmup_submissions = 0
telemetry_rows = []


def remember(path):
    data = path.read_bytes()
    artifacts[path.name] = hashlib.sha256(data).hexdigest()
    return data


def logfile(tag, expected_submits):
    global submissions, launches
    log = remember(kit / (tag + '.log')).decode()
    assert not re.search(bench.BAD, log), tag
    assert 'VALIDATION_ERRORS=0' in log and 'INDEX_READY mode=hardware' in log, tag
    assert 'INDEX_FENCE_COMPLETE readback_saved' in log, tag
    assert len(re.findall(r'^INDEX_SUBMIT sample=', log, re.M)) == expected_submits, tag
    submissions += expected_submits
    launches += 1
    return log


for ref in s['references']:
    key = ref['reference_key']
    assert key not in refs
    tag = key + '-reference'
    logfile(tag, 1)
    data = remember(kit / (tag + '.rgba'))
    assert len(data) == 256 * 256 * 4 and any(data[3::4]), tag
    assert hashlib.sha256(data).hexdigest() == ref['image_sha256'], tag
    refs[key] = data


def verify(case, mode, reference_key, preflight=False):
    global checks, samples, batches_total, warmup_submissions, exact_checks
    key = (case['tag'], mode, preflight)
    assert key not in used_rows
    used_rows.add(key)
    row = rows[key]
    tag = case['tag'] + '-' + mode + ('-preflight' if preflight else '')
    n = 1 if case.get('flat_oracle') or preflight else 31
    log = logfile(tag, n + 2)
    bench.compile_audit(case, mode, log)
    batches = bench.audit_batches(log, 1)
    assert bool(batches) == bench.indexed(case, mode) and batches == row['batches'], tag
    batches_total += len(batches)
    assert row['batch_trim'] == (1 if batches else None) and not row['batch_trim_explicit'], tag
    assert row['actual_route'] == bench.route(case, mode), tag
    assert row['setup_batch'] == case['setup_batch'], tag
    assert bench.audit_setups(log, case, mode) == row['setups'], tag
    expected_bytes = light.PROFILES[case['name']]['output_bytes'] if mode == 'stored' else 0
    output_match = re.search(r'stored outputs bytes/group=(\d+)', log)
    assert row['output_bytes'] == expected_bytes, tag
    assert (int(output_match.group(1)) if output_match else 0) == expected_bytes, tag
    records = 1 if case['draw'] == 'direct' else 2
    assert f'INDEX_WORKLOAD groups={case["groups"]} live_records={records} empty=0' in log, tag
    usage = bench.audit_usage(log, case, mode)
    if n == 31:
        assert [list(v) for v in usage] == row['cumulative_usage_samples'], tag
        path = kit / (tag + '-telemetry.jsonl')
        raw_telemetry = remember(path)
        audited = bench.audit_warmup(log, 500, path)
        assert all(row[k] == v for k, v in audited.items()), tag
        snapshots = [json.loads(line) for line in raw_telemetry.decode().splitlines()]
        assert snapshots and all(a['monotonic_ns'] <= b['monotonic_ns'] for a, b in zip(snapshots, snapshots[1:])), tag
        warmup_submissions += audited['warmup_extra_submissions']
        assert all(v[0] <= audited['warmup_extra_submissions'] + n + 2 for v in usage), tag
        start = int(re.search(r'INDEX_WARMUP_BEGIN requested_ms=500 monotonic_ns=(\d+)', log)[1])
        end = int(re.search(r'INDEX_WARMUP_COMPLETE submissions=\d+ wall_ms=[\d.]+ monotonic_ns=(\d+)', log)[1])
        assert abs((end - start) / 1e6 - audited['warmup_wall_ms']) < .000002, tag
        telemetry_rows.append(dict(tag=tag, **audited))
    else:
        assert 'INDEX_WARMUP_BEGIN' not in log and 'warmup_extra_submissions' not in row, tag
    measured = re.findall(r'INDEX_BENCH_SAMPLE sample=(\d+) warmup=(\d+) gpu_us=([\d.]+) draws=(\d+)', log)
    assert len(measured) == n + 2, tag
    times = []
    for i, (sample, warm, us, draws) in enumerate(measured):
        assert int(sample) == i and int(warm) == (i < 2) and int(draws) == case['draws'], tag
        assert float(us) > 0 and math.isfinite(float(us)), tag
        if i >= 2:
            times.append(float(us))
    assert times == row['samples_us'], tag
    assert statistics.median(times) == row['median_us'] and min(times) == row['min_us'] and max(times) == row['max_us'], tag
    assert row['timestamp_config'] == re.search(r'^INDEX_BENCH_CONFIG .+$', log, re.M).group(0), tag
    provoking = 'last' if case.get('last') else 'first'
    assert f'samples={n} warmup=2 provoking={provoking} timestamp_bits=' in log, tag
    pixels = remember(kit / (tag + '.rgba'))
    check = bench.image_acceptance(case, mode, pixels, refs[reference_key])
    assert check['accepted'] and check == row['reference_comparison'], tag
    assert hashlib.sha256(pixels).hexdigest() == row['image_sha256'], tag
    assert row['image_matches_saved_reference'] == check['exact'], tag
    exact_checks += int(check['exact'])
    checks += 1
    if preflight:
        tables = re.findall(r'BC250 IDX DEBUG: call=(\d+) n=(\d+) groups=(\d+).*fallback_count=(\d+) '
                            r'drawn=(\d+) indices=(\d+) fallback_records=(\d+).* pool=(\d+)', log)
        expected_tables, pool_words = bench.expected_pool_tables(case, expected_bytes)
        assert list(map(lambda values: tuple(map(int, values)), tables)) == expected_tables[:8], tag
        assert row['verified_pool_tables'] == [list(t) for t in expected_tables[:8]], tag
        assert row['verified_pool_words_after_resubmission'] == pool_words, tag
    if n == 31:
        samples += n
    return row


for case in work.CASES:
    for witness in bench.flat_cases(case):
        reference_key = case['reference_key'] + '-flat-corner' + str(witness['oracle_corner'])
        verify(gate.variant(witness, 'mesh', 0), 'mesh-before', reference_key)
        for mode in ('existing-before', 'stored'):
            for enabled in (0, 1):
                verify(gate.variant(witness, 'setup' + str(enabled), enabled), mode, reference_key)
    case = dict(case, verified_flat_oracle=bool(case.get('allow_interpolated_rounding')))
    if case['draw'] != 'direct':
        for mode in ('existing-before', 'stored'):
            for enabled in (0, 1):
                verify(gate.variant(case, 'pool-setup' + str(enabled), enabled), mode, case['reference_key'], True)
    mesh = [verify(gate.variant(case, phase, 0), phase, case['reference_key'])
            for phase in ('mesh-before', 'mesh-after')]
    for mode in ('existing-before', 'stored'):
        pair = [verify(gate.variant(case, phase, enabled), mode, case['reference_key'])
                for phase, enabled in (('before', 0), ('batched', 1), ('after', 0))]
        assert pair[0]['setups'] == pair[2]['setups'], case['tag']
        assert gate.compare(case, bench.route(case, mode), pair, mesh) == comparisons[(case['tag'], bench.route(case, mode))]
assert used_rows == set(rows)
assert (checks, samples, submissions, launches) == (624, 5952, 7660, 652)
assert json.loads((kit / 'oracles.partial.json').read_text()) == s['exact_corner_oracles']
assert json.loads((kit / 'preflights.partial.json').read_text()) == s['indirect_pool_preflights']
assert json.loads((kit / 'comparisons.partial.json').read_text()) == s['comparisons']
assert json.loads((kit / 'results.partial.json').read_text()) == s['trials'][:-1]
assert s['focus_rounds'] == 3 and s['distinct_workloads'] == 8 and s['benchmark_samples'] == 31
assert focus.analyze(s, work.CASES) == s['focus'], 'focus consensus differs from audited timestamps'
for case in work.CASES:
    if case['draw'] == 'count' and case['draws'] == 32:
        for mode in ('existing-before', 'stored'):
            row = rows[(case['tag'] + '-batched', mode, False)]
            assert [(span['calls'], span['shared']) for span in row['setups']['spans']] == [(32, True)]
            assert row['setups']['dispatches'] == row['setups']['waits'] == 1
# Independently recompute the strict decisions using the audited raw samples.
def window_decision(pair, mesh, discard):
    medians = [statistics.median(r['samples_us'][discard:]) for r in pair + mesh]
    b, t, a, mb, ma = medians
    drift = 100 * abs(a - b) / min(a, b)
    md = 100 * abs(ma - mb) / min(ma, mb)
    gain = ((mb + ma) / 2) / t
    setup_gain = ((b + a) / 2) / t
    return drift, md, gain, setup_gain
independent_rounds = []
for case in work.CASES:
    mesh = [rows[(case['tag'] + '-' + phase, phase, False)] for phase in ('mesh-before', 'mesh-after')]
    for mode in ('existing-before', 'stored'):
        pair = [rows[(case['tag'] + '-' + phase, mode, False)] for phase in ('before', 'batched', 'after')]
        windows = [window_decision(pair, mesh, discard) for discard in (0, 8)]
        spreads = []
        for r in pair + mesh:
            values = sorted(r['samples_us'])
            assert len(values) == 31
            spreads.append(100 * (values[27] / values[3] - 1))
        stable = max(v for w in windows for v in w[:2]) <= 5 and max(spreads) <= 10
        winning = stable and all(w[2] > 1 + max(.05, w[0] / 100, w[1] / 100) for w in windows)
        setup_winning = stable and all(w[3] > 1 + max(.05, w[0] / 100) for w in windows)
        reported = next(r for r in s['focus']['rounds'] if r['focus_key'] == case['focus_key']
                        and r['repeat'] == case['repeat'] and r['route'] == bench.route(case, mode))
        assert (stable, winning, setup_winning) == (reported['stable'], reported['win_vs_mesh'], reported['setup_win'])
        independent_rounds.append(dict(case=case['focus_key'], repeat=case['repeat'], route=bench.route(case, mode),
                                       stable=stable, win=winning, setup_win=setup_winning))
for aggregate in s['focus']['aggregate']:
    matching = [r for r in independent_rounds if (r['case'], r['route']) == (aggregate['case'], aggregate['route'])]
    assert len(matching) == 3
    assert aggregate['confirmed_mesh_win'] == all(r['win'] for r in matching)
    assert aggregate['confirmed_setup_win'] == all(r['setup_win'] for r in matching)
result = dict(kit=kit.name, source_commit=manifest['driver_commit'],
    tools_commit=manifest['tools_commit'], hardware_status='PASS',
    checksums='PASS', hardware_started=(kit / 'hw.started').read_text().strip(),
    distinct_workloads=8, workloads=24, independent_references=28, exact_corner_probes=360,
    indirect_pool_preflights=72, timed_trials=192, accepted_route_images=checks,
    exact_route_images=exact_checks, bounded_only_images=checks - exact_checks,
    measured_timestamp_samples=samples, program_launches=launches, ordinary_GPU_submissions=submissions, warmup_extra_GPU_submissions=warmup_submissions,
    GPU_submissions=submissions + warmup_submissions, telemetry_files=len(telemetry_rows),
    warmup_and_measurement_audit=telemetry_rows,
    audited_IDX_batches=batches_total, comparisons=48,
    focus_stable_rounds=sum(r['stable'] for r in s['focus']['rounds']),
    excluded_focus_rounds=[r for r in s['focus']['rounds'] if not r['stable']],
    confirmed_mesh_wins=[r for r in s['focus']['aggregate'] if r['confirmed_mesh_win']],
    confirmed_setup_wins=[r for r in s['focus']['aggregate'] if r['confirmed_setup_win']],
    focus_aggregate=s['focus']['aggregate'], independent_focus_math='PASS 48 decisions and 16 three-round aggregates',
    setup_batch_default=False, stored_IDX_default=False, batch_trim_default=True,
    automatic_route_policy_changed=False, GPU_runs_by_agent=0, game_FPS='UNPROVEN',
    policy='Correctness passed; use only confirmed three-round wins. Retain excluded timings and keep automatic route policy unchanged. Synthetic workload evidence does not establish game FPS.')
for name in ('summary.json', 'oracles.partial.json', 'preflights.partial.json', 'results.partial.json',
             'comparisons.partial.json', 'hw.started', 'hw.progress', 'SHA256SUMS'):
    remember(kit / name)
(out / 'hardware-validation.json').write_text(json.dumps(result, indent=2) + '\n')
(out / 'hardware-summary.json').write_text(json.dumps(s, indent=2) + '\n')
(out / 'hardware-artifacts.sha256').write_text(''.join(digest + '  ' + name + '\n' for name, digest in sorted(artifacts.items())))
print(json.dumps(result, indent=2))

#!/usr/bin/env python3
"""Independently audit saved automatic-routing data; never executes Vulkan."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys


def independent_score(case, repeat, rows):
    windows = []
    for skip in (0, 8):
        a, forced, selected, z = [statistics.median(r['samples_us'][skip:]) for r in rows]
        baseline = (a + z) / 2
        windows.append(dict(mesh_us=baseline, stored_us=forced, automatic_us=selected,
                            mesh_drift_percent=(max(a, z) / min(a, z) - 1) * 100,
                            speedup_vs_mesh=baseline / selected,
                            slowdown_vs_fastest=selected / min(baseline, forced)))
    spreads = []
    for row in rows:
        ordered = sorted(row['samples_us'])
        assert len(ordered) == 31
        spreads.append((ordered[27] / ordered[3] - 1) * 100)
    spread = max(spreads)
    stable = max(w['mesh_drift_percent'] for w in windows) <= 5 and spread <= 10
    return dict(case=case['tag'], repeat=repeat, full=windows[0], settled=windows[1],
                stable=stable, max_spread_percent=spread,
                win=stable and all(w['speedup_vs_mesh'] > 1.05 for w in windows),
                regression=stable and any(w['slowdown_vs_fastest'] > 1.05 for w in windows))


def audit(kit):
    kit = kit.resolve()
    sys.path.insert(0, str(kit))
    import automatic as gate
    import benchmark as bench
    assert Path(gate.__file__).resolve().parent == kit and Path(bench.__file__).resolve().parent == kit
    data = kit / 'results'
    s = json.loads((data / 'summary.json').read_text())
    assert not s['offline'] and s['automatic_default']
    assert {key: len(s[key]) for key in ('references', 'oracles', 'trials', 'scores')} == dict(
        references=70, oracles=170, trials=144, scores=36)
    assert 'PASS automatic comparison; retain all raw evidence' in (data / 'hw.progress').read_text()
    refs = {r['reference_key']: r for r in s['references']}
    rows = {(r['case'], r['mode']): r for name in ('oracles', 'trials') for r in s[name]}
    assert len(refs) == 70 and len(rows) == 314
    artifacts, used, used_refs, images = {}, set(), set(), {}
    counts = dict(program_launches=0, ordinary_submissions=0, warmup_submissions=0,
                  measured_timestamp_samples=0, exact_route_images=0, bounded_only_images=0)

    def remember(path):
        raw = path.read_bytes()
        artifacts[str(path.relative_to(data))] = hashlib.sha256(raw).hexdigest()
        return raw

    def logfile(tag, n):
        text = remember(data / (tag + '.log')).decode()
        assert not re.search(bench.BAD, text), tag
        assert 'INDEX_READY mode=hardware' in text and 'VALIDATION_ERRORS=0' in text, tag
        assert 'INDEX_FENCE_COMPLETE readback_saved' in text, tag
        assert len(re.findall(r'^INDEX_SUBMIT sample=', text, re.M)) == n, tag
        counts['program_launches'] += 1
        counts['ordinary_submissions'] += n
        return text

    def reference(case):
        key = case['reference_key']
        if key not in images:
            used_refs.add(key)
            text = logfile(key + '-reference', 1)
            assert 'MESH IDXPASS: applied' not in text
            raw = remember(data / (key + '-reference.rgba'))
            assert len(raw) == 256 * 256 * 4 and any(raw[3::4]), key
            assert hashlib.sha256(raw).hexdigest() == refs[key]['image_sha256'], key
            images[key] = raw
        return images[key]

    def verify(case, phase):
        if phase == 'automatic':
            current, mode = gate.auto_variant(case)
        else:
            current = dict(case, tag=case['tag'] + '-' + phase, setup_batch=1)
            mode = 'stored' if phase == 'forced' else 'mesh-' + phase
        key = current['tag'], mode
        assert key not in used, key
        used.add(key)
        row = rows[key]
        assert row['phase'] == phase and row['actual_route'] == bench.route(current, mode), key
        n = 1 if case.get('flat_oracle') else case['benchmark_samples']
        tag = current['tag'] + '-' + mode
        text = logfile(tag, n + 2)
        bench.compile_audit(current, mode, text)
        batches = bench.audit_batches(text, 1)
        assert batches == row['batches'] and bool(batches) == bench.indexed(current, mode), tag
        assert row['batch_trim'] == (1 if batches else None) and not row['batch_trim_explicit'], tag
        assert row['setup_batch'] == 1 and bench.audit_setups(text, current, mode) == row['setups'], tag
        assert row['draws'] == case['draws'] and row['groups_per_record'] == case['groups'], tag
        records = 1 if case['draw'] in ('direct', 'single') else 2
        assert f'INDEX_WORKLOAD groups={case["groups"]} live_records={records} empty=0' in text, tag
        expected_bytes = 0
        if mode == 'stored':
            expected_bytes = int(re.search(r'stored outputs bytes/group=(\d+)', text)[1])
        assert row['output_bytes'] == expected_bytes, tag
        usage = bench.audit_usage(text, current, mode)
        samples = re.findall(r'INDEX_BENCH_SAMPLE sample=(\d+) warmup=(\d+) gpu_us=([\d.]+) draws=(\d+)', text)
        assert len(samples) == n + 2, tag
        values = []
        for i, (sample, warm, us, draws) in enumerate(samples):
            assert (int(sample), int(warm), int(draws)) == (i, int(i < 2), case['draws']), tag
            value = float(us)
            assert value > 0 and math.isfinite(value), tag
            if i >= 2:
                values.append(value)
        assert values == row['samples_us'], tag
        assert (statistics.median(values), min(values), max(values)) == (
            row['median_us'], row['min_us'], row['max_us']), tag
        assert row['timestamp_config'] == re.search(r'^INDEX_BENCH_CONFIG .+$', text, re.M)[0], tag
        counts['measured_timestamp_samples'] += n
        pixels = remember(data / (tag + '.rgba'))
        check = bench.image_acceptance(current, mode, pixels, reference(case))
        assert check['accepted'] and check == row['reference_comparison'], tag
        assert hashlib.sha256(pixels).hexdigest() == row['image_sha256'], tag
        assert row['image_matches_saved_reference'] == check['exact'], tag
        counts['exact_route_images' if check['exact'] else 'bounded_only_images'] += 1
        if n == 31:
            path = data / (tag + '-telemetry.jsonl')
            raw = remember(path)
            warmup = bench.audit_warmup(text, 500, path)
            assert all(row[k] == v for k, v in warmup.items()), tag
            assert [list(u) for u in usage] == row['cumulative_usage_samples'], tag
            snapshots = [json.loads(line) for line in raw.decode().splitlines()]
            assert snapshots and all(a['monotonic_ns'] <= b['monotonic_ns'] for a, b in zip(snapshots, snapshots[1:])), tag
            assert all(u[0] <= warmup['warmup_extra_submissions'] + n + 2 for u in usage), tag
            counts['warmup_submissions'] += warmup['warmup_extra_submissions']
        else:
            assert 'INDEX_WARMUP_BEGIN' not in text and 'warmup_extra_submissions' not in row, tag
        return row

    for case in gate.CASES + gate.SAFETY:
        if case.get('no_flat'):
            continue
        for witness in bench.flat_cases(case):
            phases = ('before', 'automatic') if case['draw'] == 'single' else ('before', 'forced', 'automatic')
            for phase in phases:
                verify(witness, phase)
    for case in gate.SAFETY:
        phases = ('before', 'automatic') if case['draw'] == 'single' else ('before', 'forced', 'automatic')
        current = dict(case, verified_flat_oracle=case['steps'] > 0 and not case.get('no_flat'))
        for phase in phases:
            verify(current, phase)
    scores = []
    for repeat in (1, 2, 3):
        order = gate.CASES if repeat == 1 else (list(reversed(gate.CASES)) if repeat == 2 else gate.CASES[4:] + gate.CASES[:4])
        for case in order:
            current = dict(case, tag=case['tag'] + f'-round{repeat}', verified_flat_oracle=case['steps'] > 0)
            phase_rows = [verify(current, phase) for phase in ('before', 'forced', 'automatic', 'after')]
            scores.append(independent_score(case, repeat, phase_rows))
    assert scores == s['scores'], 'decisions differ from independently audited raw timestamps'
    aggregates = []
    for case in gate.CASES:
        selected = [r for r in scores if r['case'] == case['tag']]
        aggregates.append(dict(case=case['tag'], stable_rounds=sum(r['stable'] for r in selected),
                               confirmed_win=all(r['win'] for r in selected),
                               regression_rounds=sum(r['regression'] for r in selected)))
    assert aggregates == s['aggregate']
    assert used == set(rows) and used_refs == set(refs)
    for name, category in [('oracles.partial.json', 'oracles'), ('trials.partial.json', 'trials'), ('scores.partial.json', 'scores')]:
        assert json.loads(remember(data / name)) == s[category]
    for name in ('summary.json', 'hw.started', 'hw.progress'):
        remember(data / name)
    assert (counts['program_launches'], counts['ordinary_submissions'], counts['measured_timestamp_samples']) == (384, 5332, 4634)
    result = dict(status='PASS data integrity and correctness', **counts, independent_references=70,
                  stable_comparisons=sum(r['stable'] for r in scores), excluded_comparisons=sum(not r['stable'] for r in scores),
                  stable_winning_comparisons=sum(r['win'] for r in scores),
                  stable_missed_route_comparisons=sum(r['regression'] for r in scores),
                  confirmed_wins=[r for r in aggregates if r['confirmed_win']],
                  missed_cases=[r for r in aggregates if r['regression_rounds']], aggregate=aggregates,
                  scores=scores, validation_errors=0, GPU_runs_by_auditor=0)
    return result, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kit', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.kit.resolve() not in (args.out.resolve(), *args.out.resolve().parents), 'keep frozen evidence unchanged'
    subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=args.kit, check=True)
    result, artifacts = audit(args.kit)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'automatic-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'automatic-artifacts.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in sorted(artifacts.items())))
    print(json.dumps({k: v for k, v in result.items() if k not in ('scores', 'aggregate')}, indent=2))


if __name__ == '__main__':
    main()

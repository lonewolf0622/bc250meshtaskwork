#!/usr/bin/env python3
"""Read-only audit of completed corner reuse results; never launches Vulkan."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

import benchmark as bench
import corner_alias as gate


def measured_samples(log, count, draws):
    found = re.findall(r'INDEX_BENCH_SAMPLE sample=(\d+) warmup=(\d+) gpu_us=([\d.]+) draws=(\d+)', log)
    assert len(found) == count + 2, 'incomplete timestamp samples'
    values = []
    for i, (sample, warm, us, actual_draws) in enumerate(found):
        assert (int(sample), int(warm), int(actual_draws)) == (i, int(i < 2), draws), 'sample metadata'
        value = float(us)
        assert math.isfinite(value) and value > 0, 'invalid timestamp'
        if i >= 2:
            values.append(value)
    return values


def independent_score(case, repeat, rows):
    readings = []
    for start in (0, 8):
        a, b, c = [statistics.median(row['samples_us'][start:]) for row in rows]
        readings.append(dict(baseline_us=(a + c) / 2, candidate_us=b,
                             drift_percent=(max(a, c) / min(a, c) - 1) * 100,
                             speedup=(a + c) / (2 * b)))
    spreads = []
    for row in rows:
        values = sorted(row['samples_us'])
        spreads.append((values[int((len(values) - 1) * .9)] /
                        values[int((len(values) - 1) * .1)] - 1) * 100)
    spread = max(spreads)
    stable = max(r['drift_percent'] for r in readings) <= 5 and spread <= 10
    return dict(case=case['tag'], repeat=repeat, applicable=case['expect_corner_alias'],
                full=readings[0], settled=readings[1], max_spread_percent=spread, stable=stable,
                win=case['expect_corner_alias'] and stable and all(r['speedup'] > 1.05 for r in readings),
                regression=stable and any(r['speedup'] < 1 / 1.05 for r in readings))


def audit(data):
    data = data.resolve()
    summary = json.loads((data / 'summary.json').read_text())
    assert summary['offline'] is False and not summary['default_enabled']
    assert {k: len(summary[k]) for k in ('references', 'oracles', 'trials', 'scores')} == dict(
        references=32, oracles=48, trials=72, scores=24)
    assert (data / 'hw.started').is_file()
    refs = {r['reference_key']: r for r in summary['references']}
    rows = {(r['case'], r['mode']): r for k in ('oracles', 'trials') for r in summary[k]}
    assert len(refs) == 32 and len(rows) == 120
    used, used_refs, artifacts, reference_bytes = set(), set(), {}, {}
    counts = dict(launches=0, ordinary_submissions=0, warmup_submissions=0, timestamp_samples=0)

    def remember(path):
        raw = path.read_bytes()
        artifacts[str(path.relative_to(data))] = hashlib.sha256(raw).hexdigest()
        return raw

    def log_for(tag, submissions):
        text = remember(data / (tag + '.log')).decode()
        assert not re.search(bench.BAD, text), tag
        assert 'INDEX_READY mode=hardware' in text and 'VALIDATION_ERRORS=0' in text, tag
        assert 'INDEX_FENCE_COMPLETE readback_saved' in text, tag
        assert len(re.findall(r'^INDEX_SUBMIT sample=', text, re.M)) == submissions, tag
        counts['launches'] += 1
        counts['ordinary_submissions'] += submissions
        return text

    def reference(case):
        key = case['reference_key']
        if key not in reference_bytes:
            row = refs[key]
            used_refs.add(key)
            text = log_for(key + '-reference', 1)
            assert 'MESH IDXPASS: applied' not in text
            pixels = remember(data / (key + '-reference.rgba'))
            assert len(pixels) == 256 * 256 * 4 and any(pixels[3::4]), key
            assert hashlib.sha256(pixels).hexdigest() == row['image_sha256'], key
            reference_bytes[key] = pixels
        return reference_bytes[key]

    def verify(case, phase, enabled):
        current = dict(case, tag=case['tag'] + '-' + phase, corner_alias=enabled)
        mode = 'mesh-' + phase
        key = current['tag'], mode
        assert key not in used, key
        used.add(key)
        row = rows[key]
        n = 1 if case.get('flat_oracle') else 31
        tag = current['tag'] + '-' + mode
        text = log_for(tag, n + 2)
        bench.compile_audit(current, mode, text)
        assert row['actual_route'] == 'Mesh' and row['output_bytes'] == 0 and not row['batches'], tag
        assert row['draws'] == case['draws'] and row['groups_per_record'] == case['groups'], tag
        records = 1 if case['draw'] == 'direct' else 2
        assert f'INDEX_WORKLOAD groups={case["groups"]} live_records={records} empty=0' in text, tag
        usage = bench.audit_usage(text, current, mode)
        values = measured_samples(text, n, case['draws'])
        assert values == row['samples_us'], tag
        assert (statistics.median(values), min(values), max(values)) == (
            row['median_us'], row['min_us'], row['max_us']), tag
        assert row['timestamp_config'] == re.search(r'^INDEX_BENCH_CONFIG .+$', text, re.M)[0], tag
        assert f'samples={n} warmup=2 provoking=first timestamp_bits=' in text, tag
        counts['timestamp_samples'] += n
        pixels = remember(data / (tag + '.rgba'))
        expected = reference(case)
        assert pixels == expected, 'image differs from independent VS: ' + tag
        assert hashlib.sha256(pixels).hexdigest() == row['image_sha256'], tag
        assert row['image_matches_saved_reference'] is True, tag
        assert bench.compare_image(pixels, expected) == row['reference_comparison'], tag
        if n == 31:
            telemetry = data / (tag + '-telemetry.jsonl')
            raw = remember(telemetry)
            checked = bench.audit_warmup(text, 500, telemetry)
            assert all(row[k] == v for k, v in checked.items()), tag
            assert [list(u) for u in usage] == row['cumulative_usage_samples'], tag
            snapshots = [json.loads(line) for line in raw.decode().splitlines()]
            assert snapshots and all(a['monotonic_ns'] <= b['monotonic_ns'] for a, b in zip(snapshots, snapshots[1:])), tag
            assert all(u[0] <= checked['warmup_extra_submissions'] + n + 2 for u in usage), tag
            counts['warmup_submissions'] += checked['warmup_extra_submissions']
        else:
            assert 'INDEX_WARMUP_BEGIN' not in text, tag
        return row

    for case in gate.CASES:
        for witness in bench.flat_cases(case):
            for phase, enabled in gate.PHASES[:2]:
                verify(witness, phase, enabled)
    scores = []
    for repeat in (1, 2, 3):
        order = gate.CASES if repeat == 1 else (list(reversed(gate.CASES)) if repeat == 2
                                              else gate.CASES[3:] + gate.CASES[:3])
        for case in order:
            current = dict(case, tag=case['tag'] + f'-round{repeat}')
            triplet = [verify(current, phase, enabled) for phase, enabled in gate.PHASES]
            scores.append(independent_score(case, repeat, triplet))
    assert scores == summary['scores'], 'reported decisions differ from raw timestamps'
    assert gate.aggregate(scores) == summary['aggregate']
    assert used == set(rows) and used_refs == set(refs)
    for name, category in [('oracles.partial.json', 'oracles'), ('trials.partial.json', 'trials'),
                           ('scores.partial.json', 'scores')]:
        assert json.loads(remember(data / name)) == summary[category]
    remember(data / 'summary.json')
    remember(data / 'hw.started')
    result = dict(status='PASS', **counts, independent_references=32, exact_corner_images=48,
                  exact_timed_images=72, exact_route_images=120, stable_comparisons=sum(r['stable'] for r in scores),
                  excluded_comparisons=sum(not r['stable'] for r in scores),
                  stable_wins=sum(r['win'] for r in scores), stable_regressions=sum(r['regression'] for r in scores),
                  confirmed_wins=sum(r['confirmed_win'] for r in summary['aggregate']),
                  aggregate=summary['aggregate'], validation_errors=0, GPU_runs_by_auditor=0,
                  decision='Do not enable connectivity reuse by default unless a repeated win is confirmed')
    assert (counts['launches'], counts['ordinary_submissions'], counts['timestamp_samples']) == (152, 2552, 2280)
    return result, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frozen-kit', type=Path)
    args = parser.parse_args()
    assert args.results.resolve() not in (args.out.resolve(), *args.out.resolve().parents), 'keep raw evidence unchanged'
    if args.frozen_kit:
        subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=args.frozen_kit, check=True)
    result, artifacts = audit(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'corner-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'corner-artifacts.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in sorted(artifacts.items())))
    print(json.dumps({k: v for k, v in result.items() if k != 'aggregate'}, indent=2))


if __name__ == '__main__':
    main()

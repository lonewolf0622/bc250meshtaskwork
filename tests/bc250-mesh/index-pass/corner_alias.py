#!/usr/bin/env python3
"""Bracket opt-in connectivity reuse against the existing Mesh exporter."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import benchmark as bench
import light
from setup_focus import spread_percent


def make_cases():
    cases = []
    for draw in ('direct', 'count'):
        for groups, draws in ((1, 1), (8, 32), (64, 4)):
            base = next(c for c in light.CASES if c['name'] == 'profile64' and
                        c['draw'] == draw and c['groups'] == groups and c['draws'] == 1)
            key = f'profile64-{draw}-groups{groups}-draws{draws}'
            cases.append(dict(base, draws=draws, tag=key, reference_key=key,
                              expect_corner_alias=True))
    for name in ('profile128p64', 'profile256pp128'):
        base = next(c for c in light.CASES if c['name'] == name and c['draw'] == 'count' and
                    c['groups'] == 8 and c['draws'] == 1)
        key = f'{name}-count-groups8-draws16-control'
        cases.append(dict(base, draws=16, tag=key, reference_key=key, expect_corner_alias=False))
    return [dict(c, benchmark_samples=31, warmup_ms=500) for c in cases]


CASES = make_cases()
PHASES = (('before', 0), ('alias', 1), ('after', 0))


def score(case, repeat, rows):
    assert len(rows) == 3 and all(len(r['samples_us']) == 31 for r in rows), 'incomplete samples'
    readings = []
    for first in (0, 8):
        before, candidate, after = [statistics.median(r['samples_us'][first:]) for r in rows]
        drift = (max(before, after) / min(before, after) - 1) * 100
        readings.append(dict(baseline_us=(before + after) / 2, candidate_us=candidate,
                             drift_percent=drift, speedup=(before + after) / (2 * candidate)))
    spread = max(spread_percent(r['samples_us']) for r in rows)
    stable = max(r['drift_percent'] for r in readings) <= 5 and spread <= 10
    return dict(case=case['tag'], repeat=repeat, applicable=case['expect_corner_alias'],
                full=readings[0], settled=readings[1], max_spread_percent=spread, stable=stable,
                win=case['expect_corner_alias'] and stable and all(r['speedup'] > 1.05 for r in readings),
                regression=stable and any(r['speedup'] < 1 / 1.05 for r in readings))


def aggregate(scores):
    result = []
    for key in sorted({r['case'] for r in scores}):
        rows = [r for r in scores if r['case'] == key]
        assert sorted(r['repeat'] for r in rows) == [1, 2, 3], 'incomplete rounds'
        result.append(dict(case=key, applicable=rows[0]['applicable'],
                           stable_rounds=sum(r['stable'] for r in rows),
                           winning_rounds=sum(r['win'] for r in rows),
                           regression_rounds=sum(r['regression'] for r in rows),
                           confirmed_win=all(r['win'] for r in rows),
                           full_speedups=[r['full']['speedup'] for r in rows]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    out = Path(os.environ['OUT']).resolve() if args.offline else bench.HERE / 'results'
    out.mkdir(parents=True, exist_ok=True)
    icd = Path(os.environ['ICD']).resolve() if args.offline else bench.HERE / 'radeon_icd.json'
    shim = Path(os.environ['SHIM']).resolve() if args.offline else None
    if not args.offline:
        subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=bench.HERE, check=True)
        assert not (out / 'hw.started').exists(), 'already started; preserve logs and do not retry'
        bench.idle_check()
        with (out / 'hw.started').open('x') as f:
            f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
            f.flush()
            os.fsync(f.fileno())
    references, oracles, trials, scores, cache = [], [], [], [], {}

    def reference(case):
        key = case['reference_key']
        if key not in cache:
            row = bench.run_reference(case, out, icd, shim)
            references.append(row)
            cache[key] = dict(reference=row.get('image_sha256'), mesh=row.get('image_sha256'))
        return {key: cache[key]}

    def run(case, phase, enabled):
        probe = dict(case, tag=case['tag'] + '-' + phase, corner_alias=enabled)
        return bench.run(probe, 'mesh-' + phase, out, icd, shim, references=reference(case))

    try:
        for case in CASES:
            for witness in bench.flat_cases(case):
                for phase, enabled in PHASES[:2]:
                    oracles.append(run(witness, phase, enabled))
                (out / 'oracles.partial.json').write_text(json.dumps(oracles, indent=2) + '\n')
        for repeat in (1, 2, 3):
            order = CASES if repeat == 1 else (list(reversed(CASES)) if repeat == 2 else CASES[3:] + CASES[:3])
            for case in order:
                current = dict(case, tag=case['tag'] + f'-round{repeat}')
                rows = []
                for phase, enabled in PHASES:
                    if not args.offline:
                        bench.progress(out, current['tag'] + ' ' + phase)
                    row = run(current, phase, enabled)
                    rows.append(row)
                    trials.append(row)
                    (out / 'trials.partial.json').write_text(json.dumps(trials, indent=2) + '\n')
                if not args.offline:
                    result = score(case, repeat, rows)
                    scores.append(result)
                    (out / 'scores.partial.json').write_text(json.dumps(scores, indent=2) + '\n')
                    print(f'{current["tag"]}: {result["full"]["speedup"]:.3f}x, '
                          f'stable={result["stable"]}, win={result["win"]}', flush=True)
        summary = dict(offline=args.offline, references=references, oracles=oracles, trials=trials,
                       scores=scores, aggregate=[] if args.offline else aggregate(scores),
                       default_enabled=False, clock_configuration_changed=False,
                       timing_scope='Whole draw/pass GPU timestamps; no separate shader phase timings',
                       acceptance='Exact independent VS and three flat corner probes; three reordered rounds, '
                                  '31/full and 23/settled samples, <=5% baseline drift, <=10% spread, >5% gain')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        for row in summary['aggregate']:
            print(f'{row["case"]}: stable={row["stable_rounds"]}/3, wins={row["winning_rounds"]}/3, '
                  f'confirmed={row["confirmed_win"]}', flush=True)
        if not args.offline:
            bench.progress(out, 'PASS exact images and corner alias comparison; retain excluded timings')
        print('PASS corner alias compilation/recording' if args.offline else 'PASS corner alias benchmark', flush=True)
    except BaseException:
        if not args.offline:
            bench.progress(out, 'STOP: preserve evidence and do not retry this kit')
        raise


if __name__ == '__main__':
    main()

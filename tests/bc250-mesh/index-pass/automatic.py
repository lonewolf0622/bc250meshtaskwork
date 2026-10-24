#!/usr/bin/env python3
"""Compare automatic Mesh/stored selection with both forced routes."""
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


def fixture(name, draw, steps, draws, groups=8, **extra):
    profile = light.PROFILES[name]
    tag = f'{name}-{draw}-steps{steps}-groups{groups}-draws{draws}' + extra.pop('suffix', '')
    return dict(name=name, geometry=profile['geometry'], fragment=profile['fragment'], draw=draw,
                primitives=profile['geometry'][1], groups=groups, draws=draws, steps=steps,
                tag=tag, reference_key=tag, require_mesh_reference=True, expected_batch_trim=1,
                benchmark_samples=31, warmup_ms=500, existing_fallback=False,
                auto_candidate=name == 'profile256pp128', **extra)


CASES = [fixture('profile64', draw, steps, 32 if draw == 'count' else 16)
         for draw in ('direct', 'count') for steps in (0, 1024)]
CASES += [fixture('profile128p64', 'count', steps, 32) for steps in (0, 256)]
CASES += [fixture('profile128p128', 'count', 256, 32)]
CASES += [fixture('profile256pp128', 'count', steps, 32) for steps in (0, 64, 256, 1024)]
CASES += [fixture('profile256pp128', 'direct', 256, 16)]
SAFETY = [fixture('profile256pp128', 'direct', 256, 4, groups=1),
          fixture('profile256pp128', 'single', 256, 4),
          fixture('profile256pp128', 'count', 256, 4, pool_bytes=16384, suffix='-overflow')]
SAFETY += [fixture('profile256pp128', draw, 256, 4, alternate_steps=True, suffix='-alternating')
           for draw in ('direct', 'count')]
SAFETY = [dict(c, benchmark_samples=1, warmup_ms=0) for c in SAFETY]
SAFETY += [dict(name='j64', geometry=(64, 64, 64), fragment='plain', draw=draw,
                primitives=64, groups=8, draws=4, steps=0, tag='j64-' + draw + '-legacy',
                reference_key='j64-' + draw + '-legacy', require_mesh_reference=True,
                expected_batch_trim=1, benchmark_samples=1, warmup_ms=0,
                existing_fallback=False, auto_candidate=False, auto_replay=True, no_flat=True)
           for draw in ('direct', 'count')]


def auto_variant(case):
    minimum_steps = 256 if case['draw'] == 'direct' else 64
    expected = (case['auto_candidate'] and case['steps'] >= minimum_steps and case['draw'] != 'single' and
                (case['draw'] != 'direct' or case['groups'] >= 8))
    c = dict(case, tag=case['tag'] + '-automatic', automatic=True, setup_batch=1)
    if case.get('auto_replay'):
        return c, 'existing-before'
    if expected and case.get('alternate_steps'):
        c['expected_idx_calls'] = case['draws'] // 2
    return c, 'stored' if expected else 'mesh-auto'


def score(case, repeat, rows):
    readings = []
    for start in (0, 8):
        before, forced, automatic, after = [statistics.median(row['samples_us'][start:]) for row in rows]
        mesh = (before + after) / 2
        readings.append(dict(mesh_us=mesh, stored_us=forced, automatic_us=automatic,
                             mesh_drift_percent=(max(before, after) / min(before, after) - 1) * 100,
                             speedup_vs_mesh=mesh / automatic,
                             slowdown_vs_fastest=automatic / min(mesh, forced)))
    spread = max(spread_percent(row['samples_us']) for row in rows)
    stable = max(r['mesh_drift_percent'] for r in readings) <= 5 and spread <= 10
    return dict(case=case['tag'], repeat=repeat, full=readings[0], settled=readings[1],
                stable=stable, max_spread_percent=spread,
                win=stable and all(r['speedup_vs_mesh'] > 1.05 for r in readings),
                regression=stable and any(r['slowdown_vs_fastest'] > 1.05 for r in readings))


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
        assert not (out / 'hw.started').exists(), 'already started; preserve evidence and do not retry'
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

    def run(case, phase):
        if phase == 'automatic':
            variant, mode = auto_variant(case)
        else:
            mode = 'stored' if phase == 'forced' else 'mesh-' + phase
            variant = dict(case, tag=case['tag'] + '-' + phase, setup_batch=1)
        row = bench.run(variant, mode, out, icd, shim, references=reference(case))
        row['phase'] = phase
        return row

    try:
        for case in CASES + SAFETY:
            if case.get('no_flat'):
                continue
            for witness in bench.flat_cases(case):
                phases = ('before', 'automatic') if case['draw'] == 'single' else ('before', 'forced', 'automatic')
                for phase in phases:
                    oracles.append(run(witness, phase))
                (out / 'oracles.partial.json').write_text(json.dumps(oracles, indent=2) + '\n')
        for case in SAFETY:
            phases = ('before', 'automatic') if case['draw'] == 'single' else ('before', 'forced', 'automatic')
            for phase in phases:
                oracles.append(run(dict(case, verified_flat_oracle=case['steps'] > 0 and not case.get('no_flat')), phase))
            (out / 'oracles.partial.json').write_text(json.dumps(oracles, indent=2) + '\n')
        for repeat in (1, 2, 3):
            order = CASES if repeat == 1 else (list(reversed(CASES)) if repeat == 2 else CASES[4:] + CASES[:4])
            for case in order:
                current = dict(case, tag=case['tag'] + f'-round{repeat}', verified_flat_oracle=case['steps'] > 0)
                rows = []
                for phase in ('before', 'forced', 'automatic', 'after'):
                    if not args.offline:
                        bench.progress(out, current['tag'] + ' ' + phase)
                    row = run(current, phase)
                    trials.append(row)
                    rows.append(row)
                    (out / 'trials.partial.json').write_text(json.dumps(trials, indent=2) + '\n')
                if not args.offline:
                    result = score(case, repeat, rows)
                    scores.append(result)
                    print(f'{current["tag"]}: auto vs Mesh={result["full"]["speedup_vs_mesh"]:.3f}x, '
                          f'stable={result["stable"]}, regression={result["regression"]}', flush=True)
                    (out / 'scores.partial.json').write_text(json.dumps(scores, indent=2) + '\n')
        aggregate = []
        if not args.offline:
            for case in CASES:
                rows = [r for r in scores if r['case'] == case['tag']]
                assert len(rows) == 3
                aggregate.append(dict(case=case['tag'], stable_rounds=sum(r['stable'] for r in rows),
                                      confirmed_win=all(r['win'] for r in rows),
                                      regression_rounds=sum(r['regression'] for r in rows)))
        summary = dict(offline=args.offline, automatic_default=True, references=references, oracles=oracles,
                       trials=trials, scores=scores, aggregate=aggregate,
                       scope='One driver switch; generic shader expressions and current push constants; '
                             'modeled light work stays Mesh; unmodeled shaders retain the established route; '
                             'no game identity or GPU-buffer CPU readback',
                       performance='Heuristic performance remains unproven until owner measurements; '
                                   'compare both routes and preserve unstable timings')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('PASS automatic compilation/recording' if args.offline else 'PASS automatic route benchmark', flush=True)
        if not args.offline:
            bench.progress(out, 'PASS automatic comparison; retain all raw evidence')
    except BaseException:
        if not args.offline:
            bench.progress(out, 'STOP: preserve evidence; do not retry this kit')
        raise


if __name__ == '__main__':
    main()

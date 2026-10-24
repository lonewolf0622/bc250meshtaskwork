#!/usr/bin/env python3
"""Repeat selected larger meshlets and controls before considering route changes."""
import json
import os
from pathlib import Path
import statistics

import benchmark as bench
import setup_batch
import setup_work

ROUNDS = 3
SAMPLES = 31
SETTLE_SAMPLES = 8


def make_cases():
    selected = []
    choices = [('profile256pp128', 'count', 64, 16),
               ('profile256pp128', 'count', 256, 16),
               ('profile256pp128', 'count', 256, 32),
               ('profile256pp128', 'direct', 256, 16),
               ('profile256pp128', 'count', 0, 16),
               ('profile64', 'count', 1024, 32)]
    for name, draw, steps, draws in choices:
        selected.append(dict(next(c for c in setup_work.CASES if c['name'] == name and c['draw'] == draw
                                  and c['steps'] == steps and c['draws'] == draws and not c.get('last'))))
    for case in selected[1:3]:
        selected.append(dict(case, last=1, tag=case['tag'] + '-last1',
                             reference_key=case['reference_key'] + '-last1'))
    cases = []
    for repeat in range(ROUNDS):
        order = list(range(len(selected)))
        if repeat == 1:
            order.reverse()
        elif repeat == 2:
            order = order[3:] + order[:3]
        for i in order:
            case = selected[i]
            cases.append(dict(case, tag=case['tag'] + f'-round{repeat + 1}', focus_key=case['tag'],
                              repeat=repeat + 1, benchmark_samples=SAMPLES,
                              route_order=('stored', 'existing-before') if repeat == 1
                                          else ('existing-before', 'stored')))
    return cases


CASES = make_cases()


def spread_percent(samples):
    """Use central quantiles so a single scheduling outlier is diagnostic only."""
    ordered = sorted(samples)
    lo = ordered[int((len(ordered) - 1) * .1)]
    hi = ordered[int((len(ordered) - 1) * .9)]
    return (hi / lo - 1) * 100


def score(case, route, pair, mesh):
    full = setup_batch.compare(case, route, pair, mesh)
    settled = []
    for row in pair + mesh:
        assert len(row['samples_us']) == SAMPLES, 'incomplete focus timestamps'
        values = row['samples_us'][SETTLE_SAMPLES:]
        settled.append(dict(row, median_us=statistics.median(values)))
    late = setup_batch.compare(case, route, settled[:3], settled[3:])
    spread = max(spread_percent(row['samples_us']) for row in pair + mesh)
    stable = max(full['baseline_drift_percent'], full['mesh_drift_percent'],
                 late['baseline_drift_percent'], late['mesh_drift_percent']) <= 5 and spread <= 10
    return dict(focus_key=case['focus_key'], repeat=case['repeat'], route=route, full=full, settled=late,
                max_central_sample_spread_percent=spread, stable=stable,
                win_vs_mesh=stable and full['credible_win_vs_mesh'] and late['credible_win_vs_mesh'],
                setup_win=stable and full['credible_win'] and late['credible_win'],
                setup_regression=stable and (full['credible_regression'] or late['credible_regression']))


def analyze(summary, cases=CASES):
    rows = {(row['case'], row['mode']): row for row in summary['trials']}
    assert len(rows) == len(cases) * 8, 'missing or duplicated focus trials'
    scores = []
    for case in cases:
        mesh = [rows[(case['tag'] + '-' + phase, phase)] for phase in ('mesh-before', 'mesh-after')]
        for mode in ('existing-before', 'stored'):
            pair = [rows[(case['tag'] + '-' + phase, mode)] for phase in ('before', 'batched', 'after')]
            scores.append(score(case, bench.route(case, mode), pair, mesh))
    groups = {}
    for row in scores:
        groups.setdefault((row['focus_key'], row['route']), []).append(row)
    aggregate = []
    for (key, route), values in sorted(groups.items()):
        assert sorted(row['repeat'] for row in values) == list(range(1, ROUNDS + 1))
        aggregate.append(dict(case=key, route=route, stable_rounds=sum(row['stable'] for row in values),
                              mesh_winning_rounds=sum(row['win_vs_mesh'] for row in values),
                              setup_winning_rounds=sum(row['setup_win'] for row in values),
                              setup_regression_rounds=sum(row['setup_regression'] for row in values),
                              confirmed_mesh_win=all(row['win_vs_mesh'] for row in values),
                              confirmed_setup_win=all(row['setup_win'] for row in values),
                              full_speedups_vs_mesh=[row['full']['speedup_vs_mesh'] for row in values],
                              settled_speedups_vs_mesh=[row['settled']['speedup_vs_mesh'] for row in values],
                              interpretation='All three rounds must be stable and win; synthetic workload only'))
    return dict(rounds=scores, aggregate=aggregate, baseline_drift_limit_percent=5,
                central_p10_p90_spread_limit_percent=10, measured_samples=SAMPLES,
                settled_samples=SAMPLES - SETTLE_SAMPLES, settled_discard=SETTLE_SAMPLES,
                requires_full_and_settled_agreement=True, requires_all_rounds=True)


def main():
    summary = setup_batch.main(CASES, 'focused repeated workloads',
                               'PASS image and recording checks; focus timing analysis follows')
    out = Path(os.environ['OUT']).resolve() if summary['offline'] else bench.HERE
    try:
        summary.update(focus=None if summary['offline'] else analyze(summary), focus_rounds=ROUNDS,
                       benchmark_samples=SAMPLES, distinct_workloads=len(CASES) // ROUNDS,
                       image_policy='Same exact-corner and bounded-scored policy as setup_work; no policy widening',
                       timing_scope='Three differently ordered repeats; replay/stored order reversed in round two; '
                                    'off/on/off setup and Mesh brackets, 31 samples and last-23 cross-check')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if not summary['offline']:
            for row in summary['focus']['aggregate']:
                print(f'{row["case"]} {row["route"]}: stable={row["stable_rounds"]}/{ROUNDS}, '
                      f'Mesh wins={row["mesh_winning_rounds"]}/{ROUNDS}, '
                      f'confirmed={row["confirmed_mesh_win"]}', flush=True)
            bench.progress(out, 'PASS focused repeats; keep logs including excluded timings')
    except BaseException:
        if not summary['offline']:
            bench.progress(out, 'STOP focus analysis: preserve evidence; do not retry')
        raise


if __name__ == '__main__':
    main()

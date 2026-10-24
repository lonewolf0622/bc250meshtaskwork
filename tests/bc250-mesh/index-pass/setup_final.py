#!/usr/bin/env python3
"""Validate the frozen driver across larger winners and lighter controls."""
import json
import os
from pathlib import Path

import benchmark as bench
import light
import setup_batch
import setup_focus
import setup_steady


def make_cases():
    selected = [dict(case) for case in setup_steady.CASES if case['repeat'] == 1]
    for name, draw, steps, draws in [('profile64', 'count', 0, 32),
                                    ('profile128p64', 'count', 256, 32),
                                    ('profile128p128', 'count', 256, 32),
                                    ('profile128p128', 'direct', 256, 16)]:
        base = next(c for c in light.CASES if c['name'] == name and c['draw'] == draw and
                    c['groups'] == 8 and c['draws'] == 1)
        key = f'{name}-{draw}-groups8-steps{steps}'
        selected.append(dict(base, steps=steps, draws=draws, tag=key + f'-draws{draws}',
                             reference_key=key, allow_interpolated_rounding=steps > 0))
    cases = []
    for repeat in range(3):
        order = list(range(len(selected)))
        if repeat == 1:
            order.reverse()
        elif repeat == 2:
            order = order[3:] + order[:3]
        for i in order:
            case = selected[i]
            key = case.get('focus_key', case['tag'])
            cases.append(dict(case, tag=key + f'-round{repeat + 1}', focus_key=key, repeat=repeat + 1,
                              benchmark_samples=31, warmup_ms=500,
                              route_order=('stored', 'existing-before') if repeat == 1
                                          else ('existing-before', 'stored')))
    return cases


CASES = make_cases()


def main():
    summary = setup_batch.main(CASES, 'final repeated workloads',
                               'PASS image and recording checks; final timing analysis follows')
    out = Path(os.environ['OUT']).resolve() if summary['offline'] else bench.HERE
    try:
        summary.update(focus=None if summary['offline'] else setup_focus.analyze(summary, CASES),
                       distinct_workloads=8, focus_rounds=3, benchmark_samples=31,
                       sustained_warmup_ms=500, clock_configuration_changed=False,
                       diagnostics='50 ms read-only sysfs snapshots; no clock control',
                       timing_scope='Three reordered rounds, Mesh and setup brackets; full 31 and settled 23 '
                                    'sample agreement, 500 ms sustained workload warmup')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if not summary['offline']:
            for row in summary['focus']['aggregate']:
                print(f'{row["case"]} {row["route"]}: stable={row["stable_rounds"]}/3, '
                      f'Mesh wins={row["mesh_winning_rounds"]}/3, confirmed={row["confirmed_mesh_win"]}', flush=True)
            bench.progress(out, 'PASS final repeats; retain telemetry and excluded timings')
    except BaseException:
        if not summary['offline']:
            bench.progress(out, 'STOP final analysis: preserve evidence; do not retry')
        raise


if __name__ == '__main__':
    main()

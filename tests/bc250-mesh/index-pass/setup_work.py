#!/usr/bin/env python3
"""Measure selective setup across sampled vertex-work and draw-call counts."""
import json
import os
from pathlib import Path

import benchmark as bench
import light
import setup_batch


def make_cases():
    cases = []
    for name in ('profile64', 'profile256pp128'):
        base = next(c for c in light.CASES if c['name'] == name and c['groups'] == 8
                    and c['draw'] == 'count' and c['draws'] == 1 and not c.get('last'))
        for steps in (0, 64, 256, 1024):
            for draw, draws in (('count', 1), ('count', 16), ('count', 32), ('direct', 16)):
                key = f'{name}-{draw}-groups8-steps{steps}'
                cases.append(dict(base, draw=draw, draws=draws, steps=steps,
                                  tag=key + f'-draws{draws}', reference_key=key,
                                  allow_interpolated_rounding=steps > 0))
    last = next(c for c in cases if c['name'] == 'profile256pp128' and c['draw'] == 'count'
                and c['steps'] == 256 and c['draws'] == 16)
    cases.append(dict(last, last=1, tag=last['tag'] + '-last1',
                      reference_key=last['reference_key'] + '-last1'))
    return cases


CASES = make_cases()


def sampled_wins(comparisons):
    """Report measured points only; do not infer thresholds between samples."""
    groups = {}
    for row in comparisons:
        key = (row['profile'], row['draw'], row['groups'], row['draws'], row['last'], row['route'])
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row['steps'])
        wins = [r['steps'] for r in rows if r['credible_win_vs_mesh']]
        result.append(dict(profile=key[0], draw=key[1], groups=key[2], draws=key[3], last=key[4],
                           route=key[5], measured_steps=[r['steps'] for r in rows],
                           stable_steps=[r['steps'] for r in rows if r['stable']
                                         and r['mesh_drift_percent'] <= 10],
                           steps_beating_mesh=wins, first_sampled_steps_beating_mesh=min(wins) if wins else None))
    return result


def main():
    summary = setup_batch.main(CASES, 'setup and vertex-work workloads',
                               'PASS exact corner probes, bounded scored images and paired setup timings')
    summary.update(sampled_wins=sampled_wins(summary['comparisons']),
                   vertex_work_steps=[0, 64, 256, 1024],
                   image_policy='Exact rotated flat corners and all zero-step images; nonzero-step scored '
                                'images permit at most one RGB UNORM8 unit with exact alpha/background '
                                'only after exact corner probes pass',
                   interpolation_between_samples=False)
    out = Path(os.environ['OUT']).resolve() if summary['offline'] else bench.HERE
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for row in summary['sampled_wins']:
        print(f'{row["profile"]} {row["draw"]} draws={row["draws"]} last={row["last"]} '
              f'{row["route"]}: sampled steps beating Mesh={row["steps_beating_mesh"]}', flush=True)


if __name__ == '__main__':
    main()

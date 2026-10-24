#!/usr/bin/env python3
"""Measure fixed and per-group route costs for shader-property fixtures."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import benchmark as bench

PROFILES = {
    'profile64': dict(geometry=(64, 64, 64), output_bytes=2048, fragment='plain'),
    'profile128p64': dict(geometry=(128, 64, 128), output_bytes=4608, fragment='plain'),
    'profile128p128': dict(geometry=(128, 128, 128), output_bytes=4608, fragment='plain'),
    'profile256pp128': dict(geometry=(256, 128, 128), output_bytes=10240, fragment='pp'),
}
CASES = [dict(name=name, geometry=profile['geometry'], fragment=profile['fragment'],
              primitives=profile['geometry'][1], draw=draw, groups=groups, draws=draws, steps=0,
              tag=f'{name}-{draw}-groups{groups}-draws{draws}',
              reference_key=f'{name}-{draw}-groups{groups}',
              existing_fallback=False, require_mesh_reference=True, expected_batch_trim=1)
         for name, profile in PROFILES.items() for draw in ('direct', 'count')
         for groups in (1, 8, 64) for draws in (1, 4)]
CASES += [dict(case, tag=case['tag'] + '-last1', reference_key=case['reference_key'] + '-last1', last=1)
          for case in CASES if case['name'] == 'profile256pp128' and case['groups'] == 8 and case['draws'] == 1]


def run(case, mode, out, icd, shim, references, preflight=False):
    if mode == 'stored':
        case = dict(case, expected_output_bytes=PROFILES[case['name']]['output_bytes'])
    return bench.run(case, mode, out, icd, shim, preflight=preflight, references=references)


def compare(case, rows):
    mesh_before, replay_before, stored, replay_after, mesh_after = [r['median_us'] for r in rows]
    result = dict(case=case['tag'], profile=case['name'], draw=case['draw'], groups=case['groups'],
                  draws=case['draws'], last=case.get('last', 0), stored_us=stored)
    for route, before, after in [('mesh', mesh_before, mesh_after), ('replay', replay_before, replay_after)]:
        drift = (max(before, after) / min(before, after) - 1) * 100
        baseline = (before + after) / 2
        speedup = baseline / stored
        result[route + '_us'] = baseline
        result[route + '_baseline_drift_percent'] = drift
        result['speedup_vs_' + route] = speedup
        result['credible_win_vs_' + route] = drift <= 10 and speedup > 1 + max(.05, drift / 100)
    drift = max(result['mesh_baseline_drift_percent'], result['replay_baseline_drift_percent'])
    result['speedup_replay_vs_mesh'] = result['mesh_us'] / result['replay_us']
    result['credible_win_replay_vs_mesh'] = drift <= 10 and result['speedup_replay_vs_mesh'] > 1 + max(.05, drift / 100)
    result['credible_win_stored_vs_both'] = result['credible_win_vs_mesh'] and result['credible_win_vs_replay']
    return result


def fit_cost(rows, route):
    """Unconstrained diagnostic fit; coefficients are not phase timings."""
    x = [(1.0, float(r['draws']), float(r['draws'] * r['groups'] * (2 if r['draw'] == 'count' else 1))) for r in rows]
    y = [r[route + '_us'] for r in rows]
    matrix = [[sum(a[i] * a[j] for a in x) for j in range(3)] + [sum(a[i] * b for a, b in zip(x, y))]
              for i in range(3)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda n: abs(matrix[n][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        scale = matrix[i][i]
        assert abs(scale) > 1e-9, 'insufficient independent draw/group samples'
        matrix[i] = [v / scale for v in matrix[i]]
        for j in range(3):
            if j != i:
                scale = matrix[j][i]
                matrix[j] = [a - scale * b for a, b in zip(matrix[j], matrix[i])]
    coefficients = [matrix[i][3] for i in range(3)]
    residuals = [b - sum(c * a for c, a in zip(coefficients, row)) for row, b in zip(x, y)]
    return dict(points=len(rows), intercept_us=coefficients[0], per_draw_us=coefficients[1],
                per_mesh_group_us=coefficients[2], rmse_us=math.sqrt(sum(r * r for r in residuals) / len(rows)),
                all_baselines_stable=all(max(r['mesh_baseline_drift_percent'], r['replay_baseline_drift_percent']) <= 10 for r in rows),
                interpretation='Diagnostic linear fit of total GPU time per original API Mesh group, including internal expansions and clear/pass cost; not separate phase measurements or an automatic routing rule')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    out = Path(os.environ['OUT']).resolve() if args.offline else bench.HERE
    out.mkdir(parents=True, exist_ok=True)
    icd = Path(os.environ['ICD']).resolve() if args.offline else bench.HERE / 'radeon_icd.json'
    shim = Path(os.environ['SHIM']).resolve() if args.offline else None
    if not args.offline:
        assert not (out / 'hw.started').exists(), 'already started; preserve logs and do not retry'
        subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=bench.HERE, check=True)
        bench.idle_check()
        with (out / 'hw.started').open('x') as f:
            f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
            f.flush()
            os.fsync(f.fileno())
    references, oracles, trials, preflights, comparisons, empty = [], [], [], [], [], []
    reference_cache = {}

    def reference(case):
        key = case['reference_key']
        if key not in reference_cache:
            ref = bench.run_reference(case, out, icd, shim)
            references.append(ref)
            reference_cache[key] = dict(reference=ref.get('image_sha256'), mesh=ref.get('image_sha256'))
        return {key: reference_cache[key]}

    def empty_pass(phase):
        case = dict(CASES[0], tag='empty-' + phase, empty=True, groups=64, draws=1)
        empty.append(bench.run(case, 'mesh-' + phase, out, icd, shim))

    try:
        empty_pass('before')
        for case in CASES:
            for witness in bench.flat_cases(case):
                witness['reference_key'] = case['reference_key'] + '-flat-corner' + str(witness['oracle_corner'])
                hashes = reference(witness)
                if not args.offline:
                    bench.progress(out, witness['tag'] + ' exact corner oracles')
                for mode in bench.MODES:
                    oracles.append(run(witness, mode, out, icd, shim, hashes))
                    (out / 'oracles.partial.json').write_text(json.dumps(oracles, indent=2) + '\n')
            # Keep the route-policy evidence byte-exact, including interpolated timed images.
            case = dict(case, verified_flat_oracle=False)
            hashes = reference(case)
            if case['draw'] == 'count':
                for mode in ('existing-before', 'stored'):
                    preflights.append(run(case, mode, out, icd, shim, hashes, preflight=True))
            paired = []
            for mode in bench.MODES:
                if not args.offline:
                    bench.progress(out, case['tag'] + ' ' + mode)
                row = run(case, mode, out, icd, shim, hashes)
                trials.append(row)
                paired.append(row)
                (out / 'results.partial.json').write_text(json.dumps(trials, indent=2) + '\n')
            if args.offline:
                print('PASS', case['tag'], 'Mesh / replay IDX / stored IDX', flush=True)
            else:
                result = compare(case, paired)
                comparisons.append(result)
                (out / 'comparisons.partial.json').write_text(json.dumps(comparisons, indent=2) + '\n')
                print(f'{case["tag"]}: stored={result["stored_us"]:.2f} us, '
                      f'vs Mesh={result["speedup_vs_mesh"]:.2f}x, vs replay={result["speedup_vs_replay"]:.2f}x'
                      + f', replay vs Mesh={result["speedup_replay_vs_mesh"]:.2f}x'
                      + (' [baseline drift >10%]' if max(result['mesh_baseline_drift_percent'], result['replay_baseline_drift_percent']) > 10 else ''), flush=True)
        empty_pass('after')
        models = []
        if not args.offline:
            for name in PROFILES:
                for draw in ('direct', 'count'):
                    rows = [r for r in comparisons if r['profile'] == name and r['draw'] == draw and not r['last']]
                    models.append(dict(profile=name, draw=draw, routes={mode: fit_cost(rows, mode) for mode in ('mesh', 'replay', 'stored')}))
        summary = dict(offline=args.offline, cases=len(CASES), references=references, exact_corner_oracles=oracles,
                       indirect_pool_preflights=preflights, trials=trials, comparisons=comparisons,
                       empty_pass_controls=empty, cost_models=models,
                       batch_trim_default_expected=True, batch_trim_override_set=False,
                       automatic_route_policy_changed=False,
                       game_FPS='UNPROVEN; synthetic shader-property fixtures only')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if not args.offline:
            bench.progress(out, 'PASS correctness and route cost sweep')
        print('PASS', len(CASES), 'shader-property route cost workloads', flush=True)
    except BaseException:
        if not args.offline:
            bench.progress(out, 'STOP: retain logs; do not retry this kit')
        raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bracket an opt-in shared indirect setup against the established setup path."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import benchmark as bench
import light

CASES = [dict(case) for case in light.CASES if case['draw'] == 'count']
BASE = next(c for c in CASES if c['name'] == 'profile64' and c['groups'] == c['draws'] == 1)
CASES += [dict(BASE, draws=n, tag=f'profile64-count-groups1-draws{n}') for n in (16, 32, 33, 128, 512)]
CASES += [dict(BASE, draws=n, tag=f'profile64-count-groups1-draws{n}') for n in (2, 31, 34, 65)]
CASES += [dict(BASE, draws=n, direct_around=True, tag=f'profile64-count-groups1-draws{n}-direct-around',
               reference_key='profile64-count-groups1-direct-around') for n in (1, 2, 33)]
CASES += [dict(BASE, draw='multi', draws=16, tag='profile64-multi-groups1-draws16',
               reference_key='profile64-multi-groups1')]
CASES += [dict(c) for c in light.CASES if c['draw'] == 'direct' and c['groups'] == 1 and c['draws'] == 4
          and c['name'] in ('profile64', 'profile256pp128')]
CASES += [dict(c, pool_bytes=size, tag=c['tag'] + '-pool' + str(size))
          for c in CASES[:25] if c['draws'] == 4 and c['groups'] == 1
          if c['name'] in ('profile64', 'profile256pp128')
          for size in (4096 if c['name'] == 'profile64' else 16384,)]


def variant(case, phase, enabled):
    return dict(case, tag=case['tag'] + '-' + phase, setup_batch=enabled)


def compare(case, route, rows, mesh_rows):
    before, batched, after = (r['median_us'] for r in rows)
    baseline = (before + after) / 2
    drift = (max(before, after) / min(before, after) - 1) * 100
    mesh_before, mesh_after = (r['median_us'] for r in mesh_rows)
    mesh_drift = (max(mesh_before, mesh_after) / min(mesh_before, mesh_after) - 1) * 100
    mesh = (mesh_before + mesh_after) / 2
    speedup = baseline / batched
    return dict(case=case['tag'], profile=case['name'], draw=case['draw'], groups=case['groups'],
                draws=case['draws'], steps=case.get('steps', 0), last=case.get('last', 0), pool_bytes=case.get('pool_bytes'), route=route,
                baseline_us=baseline, batched_us=batched, saved_us=baseline - batched, speedup=speedup,
                baseline_drift_percent=drift, stable=drift <= 10,
                credible_win=drift <= 10 and speedup > 1 + max(.05, drift / 100),
                credible_regression=drift <= 10 and 1 / speedup > 1 + max(.05, drift / 100),
                mesh_us=mesh, mesh_drift_percent=mesh_drift, speedup_vs_mesh=mesh / batched,
                credible_win_vs_mesh=max(drift, mesh_drift) <= 10 and
                    mesh / batched > 1 + max(.05, drift / 100, mesh_drift / 100),
                setup_dispatches_before=rows[0]['setups']['dispatches'],
                setup_dispatches_batched=rows[1]['setups']['dispatches'],
                setup_waits_batched=rows[1]['setups']['waits'],
                scope='Mixed IDX/Mesh overflow control' if 'pool_bytes' in case else 'Paired setup experiment')


def main(cases=None, workload_label='indirect setup workloads',
         completion_label='PASS exact images and paired indirect setup timings'):
    cases = CASES if cases is None else cases
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
    references, oracles, preflights, trials, comparisons = [], [], [], [], []
    cache = {}

    def reference(case):
        key = case['reference_key']
        if key not in cache:
            row = bench.run_reference(case, out, icd, shim)
            references.append(row)
            cache[key] = dict(reference=row.get('image_sha256'), mesh=row.get('image_sha256'))
        return {key: cache[key]}

    def run(case, mode, hashes, preflight=False):
        return light.run(case, mode, out, icd, shim, hashes, preflight=preflight)

    try:
        for case in cases:
            route_order = case.get('route_order', ('existing-before', 'stored'))
            assert len(route_order) == 2 and set(route_order) == {'existing-before', 'stored'}, 'invalid route order'
            if not args.offline:
                bench.progress(out, case['tag'] + ' exact corner probes')
            for witness in bench.flat_cases(case):
                witness['reference_key'] = case['reference_key'] + '-flat-corner' + str(witness['oracle_corner'])
                hashes = reference(witness)
                oracles.append(run(variant(witness, 'mesh', 0), 'mesh-before', hashes))
                for mode in ('existing-before', 'stored'):
                    for enabled in (0, 1):
                        oracles.append(run(variant(witness, 'setup' + str(enabled), enabled), mode, hashes))
                (out / 'oracles.partial.json').write_text(json.dumps(oracles, indent=2) + '\n')
            # Only scored images explicitly marked for interpolation rounding may
            # use that allowance, after all exact rotated corner probes passed.
            case = dict(case, verified_flat_oracle=bool(case.get('allow_interpolated_rounding')))
            hashes = reference(case)
            if case['draw'] != 'direct':
                for mode in ('existing-before', 'stored'):
                    for enabled in (0, 1):
                        preflights.append(run(variant(case, 'pool-setup' + str(enabled), enabled), mode, hashes, True))
                (out / 'preflights.partial.json').write_text(json.dumps(preflights, indent=2) + '\n')
            mesh_before = run(variant(case, 'mesh-before', 0), 'mesh-before', hashes)
            trials.append(mesh_before)
            pairs = []
            for mode in route_order:
                paired = []
                for phase, enabled in (('before', 0), ('batched', 1), ('after', 0)):
                    if not args.offline:
                        bench.progress(out, case['tag'] + ' ' + mode + ' ' + phase)
                    row = run(variant(case, phase, enabled), mode, hashes)
                    row['phase'] = phase
                    paired.append(row)
                    trials.append(row)
                    (out / 'results.partial.json').write_text(json.dumps(trials, indent=2) + '\n')
                assert paired[0]['setups'] == paired[2]['setups'], 'baseline setup recording changed'
                if case['draw'] != 'direct':
                    assert 0 < paired[1]['setups']['dispatches'] <= case['draws']
                    if case['draws'] > 32:
                        assert len(paired[1]['setups']['spans']) > 1, 'setup list boundary was not exercised'
                pairs.append((mode, paired))
            mesh_after = run(variant(case, 'mesh-after', 0), 'mesh-after', hashes)
            trials.append(mesh_after)
            if not args.offline:
                for mode, paired in pairs:
                    result = compare(case, bench.route(case, mode), paired, (mesh_before, mesh_after))
                    comparisons.append(result)
                    print(f'{case["tag"]} {result["route"]}: {result["batched_us"]:.2f} us, '
                          f'{result["speedup"]:.3f}x vs per-call setup, {result["speedup_vs_mesh"]:.3f}x vs Mesh'
                          + ('' if result['stable'] else ' [baseline drift >10%]'), flush=True)
                (out / 'comparisons.partial.json').write_text(json.dumps(comparisons, indent=2) + '\n')
            else:
                print('PASS', case['tag'], 'setup off/on/off and Mesh controls', flush=True)
        summary = dict(offline=args.offline, cases=len(cases), references=references,
                       exact_corner_oracles=oracles, indirect_pool_preflights=preflights,
                       trials=trials, comparisons=comparisons, setup_batch_default=False,
                       stored_IDX_default=False, automatic_route_policy_changed=False,
                       timing_scope='Same-driver per-call/shared/per-call setup with Mesh bracket controls',
                       game_FPS='UNPROVEN; synthetic shader-property fixtures only')
        summary['setup_batch_selection'] = 'ordinary setup for one call; shared scan for two or more per span'
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if not args.offline:
            bench.progress(out, completion_label)
        print('PASS', len(cases), workload_label, flush=True)
    except BaseException:
        if not args.offline:
            bench.progress(out, 'STOP: retain logs; do not retry this kit')
        raise
    return summary


if __name__ == '__main__':
    main()

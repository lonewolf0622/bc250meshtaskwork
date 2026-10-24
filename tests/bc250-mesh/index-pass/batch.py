#!/usr/bin/env python3
"""Compare padded and trimmed IDX batch execution using the same driver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import benchmark as bench
import light

# Keep the previous sweep and add long recordings that close multiple batches.
CASES = list(light.CASES)
CASES += [dict(light.CASES[0], draws=draws, tag=f'profile64-direct-groups1-draws{draws}')
          for draws in (64, 4096)]


def variant(case, phase, enabled):
    return dict(case, tag=case['tag'] + '-' + phase, batch_trim=enabled)


def compare(case, route, rows):
    before, trimmed, after = (r['median_us'] for r in rows)
    drift = (max(before, after) / min(before, after) - 1) * 100
    baseline = (before + after) / 2
    speedup = baseline / trimmed
    return dict(case=case['tag'], profile=case['name'], draw=case['draw'], groups=case['groups'],
                draws=case['draws'], last=case.get('last', 0), route=route, padded_us=baseline,
                trimmed_us=trimmed, saved_us=baseline-trimmed, speedup=speedup,
                baseline_drift_percent=drift, stable=drift <= 10,
                credible_win=drift <= 10 and speedup > 1 + max(.05, drift / 100))


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
    references, oracles, preflights, trials, comparisons = [], [], [], [], []
    reference_cache = {}

    def reference(case):
        key = case['reference_key']
        if key not in reference_cache:
            ref = bench.run_reference(case, out, icd, shim)
            references.append(ref)
            reference_cache[key] = dict(reference=ref.get('image_sha256'), mesh=ref.get('image_sha256'))
        return {key: reference_cache[key]}

    def run(case, mode, hashes, preflight=False):
        return light.run(case, mode, out, icd, shim, hashes, preflight=preflight)

    try:
        for case in CASES:
            for witness in bench.flat_cases(case):
                witness['reference_key'] = case['reference_key'] + '-flat-corner' + str(witness['oracle_corner'])
                hashes = reference(witness)
                for mode in ('existing-before', 'stored'):
                    for enabled in (0, 1):
                        trial = variant(witness, 'trim' + str(enabled), enabled)
                        oracles.append(run(trial, mode, hashes))
                        (out / 'oracles.partial.json').write_text(json.dumps(oracles, indent=2) + '\n')
            # Keep all comparisons byte-exact, including the smooth timing workload.
            case = dict(case, verified_flat_oracle=False)
            hashes = reference(case)
            for mode in ('existing-before', 'stored'):
                if case['draw'] == 'count':
                    for enabled in (0, 1):
                        trial = variant(case, 'pool-trim' + str(enabled), enabled)
                        preflights.append(run(trial, mode, hashes, preflight=True))
                paired = []
                for phase, enabled in (('before', 0), ('trimmed', 1), ('after', 0)):
                    if not args.offline:
                        bench.progress(out, case['tag'] + ' ' + mode + ' ' + phase)
                    row = run(variant(case, phase, enabled), mode, hashes)
                    row['phase'] = phase
                    trials.append(row)
                    paired.append(row)
                    (out / 'results.partial.json').write_text(json.dumps(trials, indent=2) + '\n')
                # The producer command count and packet flags must stay identical.
                signatures = [[(b['used_dwords'], b['capacity_dwords'], b['control_flags'],
                                b['synchronization_dwords'], b['pfp_sync_header'], b['main_stream_grew'],
                                b['alignment_dwords'])
                               for b in r['batches']] for r in paired]
                assert signatures[0] == signatures[1] == signatures[2], 'batch body length changed'
                assert all(b['executed_dwords'] < b['capacity_dwords'] for b in paired[1]['batches'])
                if case['draws'] == 4096:
                    assert len(paired[1]['batches']) > 1, 'long recording did not close multiple batches'
                    assert any(b['main_stream_grew'] for b in paired[1]['batches']), 'no main-stream growth'
                if not args.offline:
                    entry = compare(case, bench.route(case, mode), paired)
                    comparisons.append(entry)
                    (out / 'comparisons.partial.json').write_text(json.dumps(comparisons, indent=2) + '\n')
                    print(f'{case["tag"]} {entry["route"]}: trimmed={entry["trimmed_us"]:.2f} us, '
                          f'saved={entry["saved_us"]:.2f} us, {entry["speedup"]:.3f}x'
                          + ('' if entry['stable'] else ' [baseline drift >10%]'), flush=True)
            if args.offline:
                print('PASS', case['tag'], 'replay/stored padded/trimmed/padded', flush=True)
        summary = dict(offline=args.offline, cases=len(CASES), references=references,
                       exact_corner_oracles=oracles, indirect_pool_preflights=preflights,
                       trials=trials, comparisons=comparisons, timing_scope='Shorter IB2 execution with minimal GFX alignment NOPs; allocation and synchronization unchanged',
                       game_FPS='UNPROVEN; synthetic shader-property fixtures only')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if not args.offline:
            bench.progress(out, 'PASS exact images and paired batch-length timings')
        print('PASS', len(CASES), 'batch-length workloads', flush=True)
    except BaseException:
        if not args.offline:
            bench.progress(out, 'STOP: retain logs; do not retry this kit')
        raise


if __name__ == '__main__':
    main()

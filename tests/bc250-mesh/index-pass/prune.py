#!/usr/bin/env python3
"""Paired stored IDX output-pruning correctness and timing gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import benchmark as bench

LAYOUTS = {'heavy64': (2048, 2048), 'prune64': (18432, 2048),
           'prune_pp128': (18432, 6144), 'prune_ballot64': (18432, 2048),
           'prune_deadpp64': (3072, 2048)}
CASES = [dict(name=name, draw=draw, draws=draws, steps=steps,
              tag=f'{name}-{draw}-steps{steps}-draws{draws}',
              reference_key=f'{name}-{draw}-steps{steps}-draws{draws}',
              primitives=128 if name == 'prune_pp128' else 64,
              fragment='pp' if name == 'prune_pp128' else 'plain', require_mesh_reference=True)
         for name in LAYOUTS for draw in ('direct', 'count') for steps in (0, 64)
         for draws in (1, 16 if name in ('heavy64', 'prune_deadpp64') else 4)]
CASES += [dict(case, tag=case['tag'] + '-last1', reference_key=case['reference_key'] + '-last1', last=1)
          for case in CASES if case['name'] in ('prune_pp128', 'prune_deadpp64') and case['steps'] == 0 and case['draws'] == 1]


def variant(case, phase, enabled):
    return dict(case, tag=case['tag'] + '-' + phase, prune=enabled,
                expected_output_bytes=LAYOUTS[case['name']][enabled])


def summarize(case, rows):
    before, pruned, after = (row['median_us'] for row in rows)
    drift = (max(before, after) / min(before, after) - 1) * 100
    speedup = (before + after) / 2 / pruned
    old, new = LAYOUTS[case['name']]
    return dict(case=case['tag'], baseline_us=(before + after) / 2, pruned_us=pruned,
                speedup=speedup, baseline_drift_percent=drift, stable=drift <= 10,
                credible_win=drift <= 10 and speedup > 1 + max(.05, drift / 100),
                bytes_before=old, bytes_after=new, record_reduction_percent=(1 - new / old) * 100)


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
    rows, references, oracles, preflights, comparisons = [], [], [], [], []
    try:
        for case in CASES:
            for witness in bench.flat_cases(case):
                if not args.offline:
                    bench.progress(out, witness['tag'] + ' exact corner oracle')
                ref = bench.run_reference(witness, out, icd, shim)
                hashes = {witness['reference_key']: dict(reference=ref.get('image_sha256'), mesh=ref.get('image_sha256'))}
                references.append(ref)
                for enabled in (0, 1):
                    trial = variant(witness, 'prune' + str(enabled), enabled)
                    oracles.append(bench.run(trial, 'stored', out, icd, shim, references=hashes))
                    (out / 'oracles.partial.json').write_text(json.dumps(oracles, indent=2) + '\n')
            case = dict(case, verified_flat_oracle=not args.offline)
            ref = bench.run_reference(case, out, icd, shim)
            references.append(ref)
            hashes = {case['reference_key']: dict(reference=ref.get('image_sha256'), mesh=ref.get('image_sha256'))}
            if case['draw'] == 'count':
                for enabled in (0, 1):
                    trial = variant(case, 'pool-prune' + str(enabled), enabled)
                    preflights.append(bench.run(trial, 'stored', out, icd, shim, preflight=True, references=hashes))
            paired = []
            for phase, enabled in (('before', 0), ('pruned', 1), ('after', 0)):
                if not args.offline:
                    bench.progress(out, case['tag'] + ' ' + phase)
                trial = variant(case, phase, enabled)
                row = bench.run(trial, 'stored', out, icd, shim, references=hashes)
                row['phase'] = phase
                rows.append(row)
                paired.append(row)
                (out / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
            if args.offline:
                print('PASS', case['tag'], 'layouts', LAYOUTS[case['name']], flush=True)
            else:
                entry = summarize(case, paired)
                comparisons.append(entry)
                (out / 'comparisons.partial.json').write_text(json.dumps(comparisons, indent=2) + '\n')
                print(f'{case["tag"]}: {entry["bytes_before"]}->{entry["bytes_after"]} bytes/group, '
                      f'pruned={entry["pruned_us"]:.2f} us, {entry["speedup"]:.2f}x vs stored baseline'
                      + ('' if entry['stable'] else ' [baseline drift >10%]'), flush=True)
        (out / 'summary.json').write_text(json.dumps(dict(offline=args.offline, cases=len(CASES),
            references=references, exact_corner_oracles=oracles, indirect_pool_preflights=preflights,
            trials=rows, comparisons=comparisons), indent=2) + '\n')
        if not args.offline:
            bench.progress(out, 'PASS correctness and paired timing gate')
        print('PASS', len(CASES), 'output-pruning workloads', flush=True)
    except BaseException:
        if not args.offline:
            bench.progress(out, 'STOP: retain logs; do not retry this kit')
        raise


if __name__ == '__main__':
    main()

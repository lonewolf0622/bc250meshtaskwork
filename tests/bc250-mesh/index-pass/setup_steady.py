#!/usr/bin/env python3
"""Repeat indirect candidates with sustained warmup and read-only diagnostics."""
import json
import os
from pathlib import Path

import benchmark as bench
import setup_batch
import setup_focus


CASES = [dict(case, warmup_ms=500) for case in setup_focus.CASES
         if case['focus_key'] in ('profile256pp128-count-groups8-steps256-draws16',
                                  'profile256pp128-count-groups8-steps256-draws32',
                                  'profile256pp128-count-groups8-steps256-draws16-last1',
                                  'profile256pp128-direct-groups8-steps256-draws16')]


def main():
    summary = setup_batch.main(CASES, 'sustained-warmup repeated workloads',
                               'PASS image and recording checks; steady timing analysis follows')
    out = Path(os.environ['OUT']).resolve() if summary['offline'] else bench.HERE
    try:
        summary.update(focus=None if summary['offline'] else setup_focus.analyze(summary, CASES),
                       distinct_workloads=4, focus_rounds=3, benchmark_samples=31, sustained_warmup_ms=500,
                       clock_configuration_changed=False, diagnostics='50 ms read-only sysfs snapshots; no clock control',
                       timing_scope='Same focused consensus with 500 ms workload resubmission before each '
                                    'scored trial, followed by two warmups and 31 timestamps')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if not summary['offline']:
            for row in summary['focus']['aggregate']:
                print(f'{row["case"]} {row["route"]}: stable={row["stable_rounds"]}/3, '
                      f'Mesh wins={row["mesh_winning_rounds"]}/3, confirmed={row["confirmed_mesh_win"]}', flush=True)
            bench.progress(out, 'PASS sustained-warmup repeats; retain telemetry and excluded timings')
    except BaseException:
        if not summary['offline']:
            bench.progress(out, 'STOP steady analysis: preserve evidence; do not retry')
        raise


if __name__ == '__main__':
    main()

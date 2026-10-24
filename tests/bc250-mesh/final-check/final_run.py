#!/usr/bin/env python3
"""Finish frozen benchmark validation, with an optional owner-run game comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def completed_benchmark(folder):
    marker = folder / 'hw.started'
    summary = folder / 'summary.json'
    progress = folder / 'hw.progress'
    if not marker.exists():
        if summary.exists() or progress.exists():
            raise ValueError('unexpected benchmark outputs; preserve this kit')
        return False
    if not summary.exists() or not progress.exists() or 'PASS final repeats; retain telemetry and excluded timings' not in progress.read_text():
        raise ValueError('benchmark started without completion; preserve evidence and do not retry')
    result = json.loads(summary.read_text())
    if result['offline'] or result['cases'] != 24:
        raise ValueError('benchmark completion does not match this kit')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--benchmark-only', action='store_true', help='finish benchmark validation (the default)')
    modes.add_argument('--game', action='store_true', help='also guide the optional FF7 comparison')
    parser.add_argument('--compare', action='store_true', help='audit completed captures without launching GPU work')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=root, check=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    if args.game and not (root / 'game' / 'game_time.py').exists():
        raise ValueError('this package does not contain the optional game comparison')
    bench = root / 'benchmark'
    complete = completed_benchmark(bench)
    if args.compare and not complete:
        raise ValueError('no completed benchmark to audit')
    if not complete:
        subprocess.run(['bash', 'run.sh'], cwd=bench, check=True)
    with (root / 'results' / 'benchmark-audit.log').open('w') as log:
        subprocess.run([sys.executable, '-B', 'audit_benchmark.py'], cwd=root, stdout=log, check=True)
    benchmark = json.loads((root / 'results' / 'hardware-validation.json').read_text())
    print(f'Benchmark audit PASS: {benchmark["exact_route_images"]} exact images, '
          f'{benchmark["bounded_only_images"]} accepted with the bounded RGB check; '
          f'{benchmark["focus_stable_rounds"]}/48 stable route comparisons.', flush=True)
    game = None
    if args.game:
        command = [sys.executable, '-B', 'game_time.py'] + (['--compare'] if args.compare else [])
        subprocess.run(command, cwd=root / 'game', check=True)
        game = json.loads((root / 'game' / 'results' / 'summary.json').read_text())
    result = dict(driver_commit=manifest['driver_commit'], driver_sha256=manifest['driver_sha256'],
                  benchmark_status='PASS', exact_benchmark_images=benchmark['exact_route_images'],
                  confirmed_synthetic_wins=benchmark['confirmed_mesh_wins'],
                  confirmed_setup_wins=benchmark['confirmed_setup_wins'],
                  bounded_only_images=benchmark['bounded_only_images'],
                  stable_comparisons=benchmark['focus_stable_rounds'],
                  excluded_comparisons=len(benchmark['excluded_focus_rounds']),
                  FF7_comparison=game['comparison'] if game else None,
                  game_status='COMPLETE one scene' if game else manifest.get('game_frame_time_status', 'NOT_RUN optional'),
                  automatic_route_policy_changed=False,
                  stored_IDX_default=False, setup_batch_default=False,
                  validation_scope='Benchmarks and offline captured-shader compilation; game FPS unproven',
                  installed_driver_changed=False, GPU_runs_by_agent=0, published=False)
    path = root / 'results' / ('game-final-summary.json' if game else 'final-summary.json')
    text = json.dumps(result, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('final summary differs; preserve existing evidence')
    if not path.exists():
        path.write_text(text)
    names = ('hardware-validation.json', 'hardware-summary.json', 'hardware-artifacts.sha256',
             'benchmark-audit.log', path.name)
    (root / 'results' / 'evidence.sha256').write_text(''.join(
        hashlib.sha256((root / 'results' / name).read_bytes()).hexdigest() + '  ' + name + '\n' for name in names))
    print('Final validation saved in results/' + path.name + '.')
    if game:
        print('Restore the original Steam launch options.')
    print('Stored IDX and shared setup remain optional; no driver was installed or published.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError, KeyboardInterrupt, EOFError) as error:
        print('STOP:', error, '\nPreserve this kit and its logs.', file=sys.stderr)
        sys.exit(1)

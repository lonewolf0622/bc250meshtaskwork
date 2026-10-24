"""CPU-only checks that completed workloads cannot be rerun after a stop."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import final_run


class FinalRunnerTests(unittest.TestCase):
    def test_only_explicit_completion_can_skip_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(final_run.completed_benchmark(root))
            (root / 'hw.started').write_text('owner started')
            with self.assertRaisesRegex(ValueError, 'do not retry'):
                final_run.completed_benchmark(root)
            (root / 'summary.json').write_text(json.dumps(dict(offline=False, cases=24)))
            (root / 'hw.progress').write_text('STOP')
            with self.assertRaisesRegex(ValueError, 'do not retry'):
                final_run.completed_benchmark(root)
            (root / 'hw.progress').write_text('PASS final repeats; retain telemetry and excluded timings')
            self.assertTrue(final_run.completed_benchmark(root))
            (root / 'summary.json').write_text(json.dumps(dict(offline=True, cases=24)))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                final_run.completed_benchmark(root)

    def test_unmarked_outputs_are_not_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                final_run.completed_benchmark(root)

    def test_compare_mode_never_starts_an_unrun_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'benchmark').mkdir()
            (root / 'manifest.json').write_text('{}')
            with patch.object(final_run, '__file__', str(root / 'final_run.py')), \
                 patch.object(sys, 'argv', ['final_run.py', '--compare']), \
                 patch.object(final_run.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'no completed benchmark'):
                    final_run.main()
                self.assertEqual(len(run.call_args_list), 1)
                self.assertEqual(run.call_args_list[0].args[0][0], 'sha256sum')

    def test_completed_benchmark_finishes_without_game_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bench = root / 'benchmark'
            bench.mkdir()
            (bench / 'hw.started').write_text('owner completed')
            (bench / 'hw.progress').write_text('PASS final repeats; retain telemetry and excluded timings')
            (bench / 'summary.json').write_text(json.dumps(dict(offline=False, cases=24)))
            (root / 'manifest.json').write_text(json.dumps(dict(driver_commit='known driver',
                  driver_sha256='a' * 64, game_frame_time_status='SKIPPED by owner')))
            results = root / 'results'
            results.mkdir()
            audit = dict(exact_route_images=624, bounded_only_images=0, focus_stable_rounds=26,
                         confirmed_mesh_wins=[], confirmed_setup_wins=[], excluded_focus_rounds=[{}] * 22)
            (results / 'hardware-validation.json').write_text(json.dumps(audit))
            (results / 'hardware-summary.json').write_text('{}')
            (results / 'hardware-artifacts.sha256').write_text('test evidence')
            with patch.object(final_run, '__file__', str(root / 'final_run.py')), \
                 patch.object(sys, 'argv', ['final_run.py']), \
                 patch.object(final_run.subprocess, 'run') as run:
                final_run.main()
                self.assertEqual(len(run.call_args_list), 2)
                self.assertEqual(run.call_args_list[1].args[0][-1], 'audit_benchmark.py')
            summary = json.loads((results / 'final-summary.json').read_text())
            self.assertEqual(summary['benchmark_status'], 'PASS')
            self.assertEqual(summary['game_status'], 'SKIPPED by owner')
            self.assertIsNone(summary['FF7_comparison'])
            self.assertEqual(summary['excluded_comparisons'], 22)


if __name__ == '__main__':
    unittest.main()

"""CPU-only checks that reject misleading game performance evidence."""
import csv
import contextlib
import io
import json
import math
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import game_time as g


class GameEvidenceTests(unittest.TestCase):
    def fixture(self, directory, times, elapsed=None, versioned=False):
        path = Path(directory) / 'capture.csv'
        if elapsed is None:
            elapsed, total = [], 0
            for value in times:
                total += round(value * 1e6)
                elapsed.append(total)
        with path.open('w') as f:
            if versioned:
                f.write('v1\n0.8.4\n---SYSTEM INFO---\n')
            f.write('os,cpu,gpu,ram,kernel,driver,cpuscheduler\nLinux,CPU,GPU,1,k,d,s\n')
            w = csv.writer(f)
            w.writerow(['fps', 'frametime', 'gpu_core_clock', 'elapsed'])
            for t, ns in zip(times, elapsed):
                # Deliberately bogus instantaneous FPS: analysis must use frame times.
                w.writerow([10000, t, 1500, ns])
        return path

    def test_weighted_fps_and_tail_in_versioned_log(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.fixture(d, [10] * 4500 + [20] * 2250, versioned=True)
            result = g.frame_metrics(path)
            self.assertAlmostEqual(result['fps'], 75)
            self.assertEqual(result['median_frame_ms'], 10)
            self.assertEqual(result['p99_frame_ms'], 20)
            self.assertEqual(result['gpu_core_clock_mean'], 1500)

    def test_reject_sampled_logs_that_cannot_support_frame_percentiles(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.fixture(d, [10] * 900, [i * 100000000 for i in range(900)])
            with self.assertRaisesRegex(ValueError, 'coverage'):
                g.frame_metrics(path)

    def test_reject_short_nan_and_nonmonotonic_logs(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, '90-second'):
                g.frame_metrics(self.fixture(d, [10] * 3000))
            values = [10] * 9000
            values[500] = math.nan
            with self.assertRaisesRegex(ValueError, 'invalid frame'):
                g.frame_metrics(self.fixture(d, values, [i * 10000000 for i in range(9000)]))
            values = [i * 10000000 for i in range(9000)]
            values[400] = values[399]
            with self.assertRaisesRegex(ValueError, 'monotonic'):
                g.frame_metrics(self.fixture(d, [10] * 9000, values))

    def rows(self, baseline=20, batched=16, after=20, tail=25, work=True):
        return {p: dict(visual_check='pass', metrics=dict(mean_frame_ms=t, fps=1000/t,
                    p99_frame_ms=tail if p == 'B1' else 25, IDX_indirect_calls_during_capture=int(work)))
                for p, t in zip(g.PHASES, (baseline, batched, after))}

    def test_gain_requires_stability_tail_visual_check_and_exercised_route(self):
        self.assertTrue(g.comparison(self.rows())['candidate_gain'])
        self.assertFalse(g.comparison(self.rows(after=24))['candidate_gain'])
        self.assertFalse(g.comparison(self.rows(tail=50))['candidate_gain'])
        self.assertFalse(g.comparison(self.rows(work=False))['candidate_gain'])
        rows = self.rows()
        rows['B1']['visual_check'] = 'fail'
        self.assertFalse(g.comparison(rows)['candidate_gain'])

    def test_no_gain_for_noise_and_flag_stable_regression(self):
        self.assertFalse(g.comparison(self.rows(batched=19.5))['candidate_gain'])
        self.assertTrue(g.comparison(self.rows(batched=24))['candidate_regression'])

    def test_wrong_mode_and_traced_run_fail_environment_check(self):
        root = Path('/a/test')
        env = g.phase_env(root, 'A1')
        raw = '\0'.join(k + '=' + v for k, v in env.items())
        self.assertEqual(g.check_environment(raw, root, 'A1')['RADV_BC250_MESH_IDXPASS_SETUP_BATCH'], '0')
        with self.assertRaisesRegex(ValueError, 'launch options'):
            g.check_environment(raw, root, 'B1')
        with self.assertRaisesRegex(ValueError, 'tracing'):
            g.check_environment(raw + '\0BC250_IDX_TRACE=1', root, 'A1')

    def test_start_marker_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'started.json'
            g.save(path, dict(phase='A1'))
            with self.assertRaises(FileExistsError):
                g.save(path, dict(phase='B1'))
            self.assertEqual(json.loads(path.read_text())['phase'], 'A1')

    def test_mapped_driver_must_match_and_environment_is_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            process = root / '42'
            library = process / 'root/usr/lib/libvulkan_radeon.so'
            library.parent.mkdir(parents=True)
            library.write_bytes(b'known test driver')
            (process / 'maps').write_text('1000-2000 r-xp 0 00:01 123 /usr/lib/libvulkan_radeon.so\n'
                                         '2000-3000 r-xp 0 00:01 124 /usr/lib/libMangoHud.so\n')
            env = dict(g.phase_env(root, 'B1'), PRIVATE_TOKEN='do-not-save')
            (process / 'environ').write_bytes('\0'.join(k + '=' + v for k, v in env.items()).encode())
            with patch.object(g, 'game_pids', return_value=[process]):
                identity = g.verify_game(root, 'B1', dict(driver_sha256=g.digest(library)))
                self.assertEqual(identity['pid'], 42)
                self.assertNotIn('PRIVATE_TOKEN', identity['environment'])
                with self.assertRaisesRegex(ValueError, 'different RADV'):
                    g.verify_game(root, 'B1', dict(driver_sha256='0' * 64))

    def route_rows(self):
        times = (20, 18, 14, 18, 20)
        return {p: dict(visual_check='pass', metrics=dict(mean_frame_ms=t, fps=1000 / t,
                    p99_frame_ms=25, IDX_direct_calls_during_capture=int(p[0] != 'M'),
                    IDX_indirect_calls_during_capture=0)) for p, t in zip(g.ROUTE_PHASES, times)}

    def test_route_plan_scores_both_brackets_and_direct_work(self):
        comparisons = g.comparison(self.route_rows())['comparisons']
        self.assertEqual(len(comparisons), 4)
        self.assertTrue(all(r['candidate_gain'] for r in comparisons))
        self.assertAlmostEqual(comparisons[-1]['speedup'], 18 / 14)

    def test_route_gain_rejects_missing_work_mesh_leak_drift_and_tail(self):
        for phase, key, value in [('S1', 'IDX_direct_calls_during_capture', 0),
                                  ('M1', 'IDX_direct_calls_during_capture', 1),
                                  ('M2', 'mean_frame_ms', 24),
                                  ('S1', 'p99_frame_ms', 50)]:
            rows = self.route_rows()
            rows[phase]['metrics'][key] = value
            self.assertFalse(g.route_comparison(rows)['comparisons'][1]['candidate_gain'])
        rows = self.route_rows()
        rows['R2']['metrics']['mean_frame_ms'] = 25
        self.assertFalse(g.route_comparison(rows)['comparisons'][-1]['candidate_gain'])
        rows = self.route_rows()
        rows['R1']['visual_check'] = 'fail'
        self.assertFalse(any(r['candidate_gain'] for r in g.route_comparison(rows)['comparisons']))

    def test_route_flags_are_explicit_and_checked(self):
        root = Path('/a/test')
        for phase in g.ROUTE_PHASES:
            env = g.phase_env(root, phase)
            self.assertEqual(env['RADV_BC250_MESH_IDXPASS'], str(int(phase[0] != 'M')))
            self.assertEqual(env['RADV_BC250_MESH_IDXPASS_STORE'], str(int(phase[0] == 'S')))
            self.assertEqual(env['RADV_BC250_MESH_IDXPASS_SETUP_BATCH'], str(int(phase[0] != 'M')))
            raw = '\0'.join(k + '=' + v for k, v in env.items())
            g.check_environment(raw, root, phase)
        with self.assertRaisesRegex(ValueError, 'launch options'):
            g.check_environment(raw, root, 'M1')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            g.phase_env(root, 'invalid')

    def test_complete_route_analysis_and_usage_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(comparison_kind='mesh_replay_stored', driver_sha256='a' * 64)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(g, '__file__', str(root / 'game_time.py')), patch.object(g, 'PHASES', g.SETUP_PHASES):
                with patch.object(sys, 'argv', ['game_time.py', '--prepare']), contextlib.redirect_stdout(io.StringIO()):
                    g.main()
                for phase, duration in zip(g.ROUTE_PHASES, (20, 18, 14, 18, 20)):
                    folder = root / 'results' / phase
                    csv_path = self.fixture(folder, [duration] * (int(90000 / duration) + 1))
                    metrics = g.frame_metrics(csv_path)
                    after = [0, 0] if phase[0] == 'M' else [3000, 0]
                    log = folder / 'driver.capture.log'
                    log.write_text(f'BC250_USAGE submits=300 idx_direct={after[0]} idx_indirect_calls=0\n')
                    metrics.update(IDX_usage_before=[0, 0], IDX_usage_after=after,
                                   IDX_direct_calls_during_capture=after[0], IDX_indirect_calls_during_capture=0)
                    identity = dict(pid=42, environment=g.phase_env(root, phase),
                                    drivers=[dict(sha256=manifest['driver_sha256'])])
                    g.save(folder / 'capture.started.json', dict(phase=phase, game=identity, IDX_usage_before=[0, 0]))
                    g.save(folder / 'capture.finished.json', dict(phase=phase, metrics=metrics,
                           game_before=identity, game_after=identity, driver_log_sha256=g.digest(log), visual_check='pass'))
                (root / 'SHA256SUMS').write_text(g.digest(root / 'manifest.json') + '  manifest.json\n')
                with patch.object(sys, 'argv', ['game_time.py', '--compare']), contextlib.redirect_stdout(io.StringIO()):
                    g.main()
                summary = json.loads((root / 'results' / 'summary.json').read_text())
                self.assertTrue(all(r['candidate_gain'] for r in summary['comparison']['comparisons']))
                path = root / 'results' / 'S1' / 'capture.finished.json'
                row = json.loads(path.read_text())
                row['metrics']['IDX_direct_calls_during_capture'] += 1
                path.write_text(json.dumps(row))
                with patch.object(sys, 'argv', ['game_time.py', '--compare']), self.assertRaisesRegex(ValueError, 'usage evidence'):
                    g.main()


if __name__ == '__main__':
    unittest.main()

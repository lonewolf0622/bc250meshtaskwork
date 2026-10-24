"""CPU-only sustained-warmup guards and read-only telemetry fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import benchmark
import setup_steady
import telemetry


class SteadyTests(unittest.TestCase):
    def test_captured_warmup_format_and_missing_measurement_snapshot(self):
        # The three-field completion line previously exposed a positional-group bug.
        text = ('INDEX_WARMUP_BEGIN requested_ms=500 monotonic_ns=25917526684568\n'
                'INDEX_WARMUP_COMPLETE submissions=2157 wall_ms=500.089423 monotonic_ns=25918026773991\n'
                'INDEX_MEASURE_BEGIN monotonic_ns=25918026795181\n'
                'INDEX_MEASURE_COMPLETE monotonic_ns=25918034683495\n')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'telemetry.jsonl'
            path.write_text(json.dumps(dict(monotonic_ns=25917700000000, cards=[])) + '\n')
            result = benchmark.audit_warmup(text, 500, path)
            self.assertEqual(result['warmup_extra_submissions'], 2157)
            self.assertEqual(result['warmup_wall_ms'], 500.089423)
            self.assertEqual(result['warmup_telemetry']['samples_in_window'], 1)
            self.assertEqual(result['measured_telemetry']['samples_in_window'], 0)
            for bad in (text.replace('requested_ms=500','requested_ms=400'),
                        text.replace('wall_ms=500.089423','wall_ms=499.9'),
                        text.replace('submissions=2157','submissions=0'),
                        text.replace('25918026773991','25917500000000'),
                        text.replace('INDEX_MEASURE_COMPLETE','MISSING')):
                with self.assertRaises(AssertionError):benchmark.audit_warmup(bad, 500, path)

    def test_duration_and_probe_guards(self):
        self.assertEqual(benchmark.warmup_duration({}), 0)
        self.assertEqual(benchmark.warmup_duration(dict(warmup_ms=500)), 500)
        self.assertEqual(benchmark.warmup_duration(dict(warmup_ms=500), preflight=True), 0)
        self.assertEqual(benchmark.warmup_duration(dict(warmup_ms=500, flat_oracle=True)), 0)
        for ms in (-1, 2001, '500', 1.5, True):
            with self.assertRaises(AssertionError):benchmark.warmup_duration(dict(warmup_ms=ms))

    def test_workloads_keep_repeats_and_last_provoking(self):
        cases = setup_steady.CASES
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({c['focus_key'] for c in cases}), 4)
        self.assertEqual(len({c['tag'] for c in cases}), 12)
        self.assertTrue(all(c['warmup_ms'] == 500 and c['benchmark_samples'] == 31 for c in cases))
        self.assertEqual(sum(c.get('last', 0) for c in cases), 3)
        for key in {c['focus_key'] for c in cases}:
            self.assertEqual([c['repeat'] for c in cases if c['focus_key'] == key], [1, 2, 3])

    def test_cumulative_usage_after_many_warmup_submissions(self):
        case = setup_steady.CASES[0]
        n = case['draws']
        text = f'BC250_USAGE submits=1 idx_direct=0 idx_indirect_calls={n}\n'
        text += f'BC250_USAGE submits=300 idx_direct=0 idx_indirect_calls={300*n}\n'
        self.assertEqual(benchmark.audit_usage(text, case, 'stored'), [(1, 0, n), (300, 0, 300*n)])
        with self.assertRaises(AssertionError):benchmark.audit_usage(text.replace(str(300*n), str(n)), case, 'stored')
        self.assertEqual(benchmark.audit_usage('BC250_USAGE submits=1 idx_direct=0 idx_indirect_calls=0',
                                              case, 'mesh-before'), [(1, 0, 0)])

    def test_sysfs_errors_and_connector_filter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            device = root / 'card1/device'
            hwmon = device / 'hwmon/hwmon2'
            hwmon.mkdir(parents=True)
            (device / 'gpu_busy_percent').write_text('68\n')
            (hwmon / 'freq1_input').write_text('500000000\n')
            (root / 'card1-DP-1/device').mkdir(parents=True)
            (root / 'card1-DP-1/device/gpu_busy_percent').write_text('99')
            found = telemetry.devices(root)
            self.assertEqual(len(found), 1)
            snap = telemetry.snapshot(found)['cards'][0]
            self.assertEqual(snap['values']['clock_hz'], 500000000)
            (hwmon / 'freq1_input').write_text('unavailable')
            snap = telemetry.snapshot(found)['cards'][0]
            self.assertIn('clock_hz', snap['unavailable'])
            self.assertNotIn('clock_hz', snap['values'])

    def test_windows_report_missing_short_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'telemetry.jsonl'
            rows = [dict(monotonic_ns=n, cards=[dict(card='card1', values=dict(clock_hz=hz, busy_percent=70),
                                                    unavailable=[])]) for n,hz in ((100,500000000),(200,2230000000))]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            observed = telemetry.summarize(path, 100, 200)
            self.assertEqual(observed['clock_hz_range'], [500000000,2230000000])
            short = telemetry.summarize(path, 110, 190)
            self.assertEqual(short['samples_in_window'], 0)
            self.assertIsNone(short['clock_hz_range'])
            rows.reverse()
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(AssertionError):telemetry.summarize(path, 100, 200)

    def test_recorder_exclusive_file_and_final_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'telemetry.jsonl'
            with telemetry.Recorder(path, found=[]):pass
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertGreaterEqual(len(rows), 2)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                with telemetry.Recorder(path, found=[]):pass
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':unittest.main()

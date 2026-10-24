"""CPU-only checks of repeat consensus, drift and settling-sensitive results."""
import statistics
import unittest

import benchmark
import setup_focus as focus


def row(value, values=None):
    samples = values if values is not None else [value] * focus.SAMPLES
    return dict(samples_us=samples, median_us=statistics.median(samples),
                setups=dict(dispatches=1, waits=1))


class FocusTests(unittest.TestCase):
    def test_timestamp_limits_and_short_checks(self):
        self.assertEqual(benchmark.sample_count({}), 9)
        self.assertEqual(benchmark.sample_count(dict(benchmark_samples=31)), 31)
        for options in (dict(offline=True), dict(preflight=True)):
            self.assertEqual(benchmark.sample_count(dict(benchmark_samples=31), **options), 1)
        self.assertEqual(benchmark.sample_count(dict(benchmark_samples=31, flat_oracle=True)), 1)
        for count in (0, 32, 1.5, '31', True):
            with self.assertRaises(AssertionError):
                benchmark.sample_count(dict(benchmark_samples=count))

    def test_repeats_are_unique_and_change_order(self):
        cases = focus.CASES
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({c['tag'] for c in cases}), 24)
        keys = {c['focus_key'] for c in cases}
        self.assertEqual(len(keys), 8)
        for key in keys:
            selected = [c for c in cases if c['focus_key'] == key]
            self.assertEqual([c['repeat'] for c in selected], [1, 2, 3])
            self.assertEqual(len({c['reference_key'] for c in selected}), 1)
            self.assertTrue(all(c['benchmark_samples'] == 31 for c in selected))
        first = [c['focus_key'] for c in cases[:8]]
        self.assertEqual([c['focus_key'] for c in cases[8:16]], first[::-1])
        self.assertNotEqual([c['focus_key'] for c in cases[16:]], first)
        self.assertTrue(all(c['route_order'][0] == 'stored' for c in cases[8:16]))

    def test_drift_and_sustained_noise_exclude_win(self):
        case = focus.CASES[0]
        pair = [row(200), row(100), row(200)]
        self.assertTrue(focus.score(case, 'stored IDX', pair, [row(150), row(150)])['win_vs_mesh'])
        self.assertFalse(focus.score(case, 'stored IDX', pair, [row(150), row(160)])['stable'])
        noisy = [90] * 10 + [100] * 11 + [120] * 10
        self.assertFalse(focus.score(case, 'stored IDX', [row(200), row(100, noisy), row(200)],
                                     [row(150), row(150)])['stable'])

    def test_full_and_settled_must_agree(self):
        values = [101] * 8 + [101] * 11 + [107] * 12
        report = focus.score(focus.CASES[0], 'stored IDX', [row(200), row(100), row(200)],
                             [row(101, values), row(101, values)])
        self.assertTrue(report['stable'])
        self.assertFalse(report['full']['credible_win_vs_mesh'])
        self.assertTrue(report['settled']['credible_win_vs_mesh'])
        self.assertFalse(report['win_vs_mesh'])

    def test_all_rounds_required_and_no_missing_trials(self):
        trials = []
        for case in focus.CASES:
            for mode, phase, value in [('mesh-before', 'mesh-before', 150),
                                       ('mesh-after', 'mesh-after', 150)]:
                trials.append(dict(row(value), case=case['tag'] + '-' + phase, mode=mode))
            for mode in ('existing-before', 'stored'):
                for phase, value in (('before', 200), ('batched', 100), ('after', 200)):
                    trials.append(dict(row(value), case=case['tag'] + '-' + phase, mode=mode))
        summary = dict(trials=trials)
        self.assertTrue(all(r['confirmed_mesh_win'] for r in focus.analyze(summary)['aggregate']))
        case = next(c for c in focus.CASES if c['focus_key'] == focus.CASES[0]['focus_key'] and c['repeat'] == 2)
        trial = next(r for r in trials if r['case'] == case['tag'] + '-mesh-after')
        trial.update(row(170))
        result = next(r for r in focus.analyze(summary)['aggregate']
                      if r['case'] == case['focus_key'] and r['route'] == 'stored IDX')
        self.assertEqual(result['stable_rounds'], 2)
        self.assertFalse(result['confirmed_mesh_win'])
        with self.assertRaises(AssertionError):
            focus.analyze(dict(trials=trials[:-1]))


if __name__ == '__main__':
    unittest.main()

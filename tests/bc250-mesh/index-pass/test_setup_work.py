"""CPU checks for the work sweep and conservative measured-point reports."""
import unittest

import benchmark
import setup_batch
import setup_work


class WorkTests(unittest.TestCase):
    def test_case_identity_and_reference_sharing(self):
        cases = setup_work.CASES
        self.assertEqual(len(cases), 33)
        self.assertEqual(len({c['tag'] for c in cases}), 33)
        self.assertEqual(len({c['reference_key'] for c in cases}), 17)
        for a in cases:
            self.assertEqual(a['allow_interpolated_rounding'], a['steps'] > 0)
            for b in cases:
                if a['reference_key'] == b['reference_key']:
                    self.assertEqual((a['name'], a['draw'], a['groups'], a['steps'], a.get('last', 0)),
                                     (b['name'], b['draw'], b['groups'], b['steps'], b.get('last', 0)))

    def comparison(self, times=(100, 70, 100), mesh=(90, 90), steps=256):
        case = dict(setup_work.CASES[0], steps=steps)
        setups = dict(dispatches=1, waits=1)
        return setup_batch.compare(case, 'stored IDX',
                                   [dict(median_us=t, setups=setups) for t in times],
                                   [dict(median_us=t) for t in mesh])

    def test_gains_and_regressions(self):
        row = self.comparison()
        self.assertTrue(row['credible_win'])
        self.assertTrue(row['credible_win_vs_mesh'])
        self.assertFalse(row['credible_regression'])
        self.assertEqual(row['steps'], 256)
        row = self.comparison(times=(70, 100, 70))
        self.assertTrue(row['credible_regression'])
        self.assertFalse(row['credible_win'])

    def test_drift_excludes_wins(self):
        self.assertFalse(self.comparison(times=(100, 60, 120))['credible_win_vs_mesh'])
        self.assertFalse(self.comparison(mesh=(90, 120))['credible_win_vs_mesh'])
        self.assertFalse(self.comparison(times=(100, 60, 120))['credible_regression'])

    def test_sampled_wins_do_not_interpolate_or_hide_later_losses(self):
        rows = [self.comparison(steps=64), self.comparison(times=(100, 110, 100), steps=256)]
        report = setup_work.sampled_wins(rows)[0]
        self.assertEqual(report['steps_beating_mesh'], [64])
        self.assertEqual(report['first_sampled_steps_beating_mesh'], 64)
        self.assertEqual(report['measured_steps'], [64, 256])
        self.assertIsNone(setup_work.sampled_wins(rows[1:])[0]['first_sampled_steps_beating_mesh'])

    def test_rounding_requires_proven_corners_and_never_applies_to_corner_probe(self):
        ref = bytes((0, 0, 0, 0)) + bytes((10, 20, 30, 255)) * (256 * 256 - 1)
        image = bytearray(ref)
        image[4] += 1
        self.assertFalse(benchmark.image_acceptance({}, 'stored', image, ref)['accepted'])
        self.assertTrue(benchmark.image_acceptance(dict(verified_flat_oracle=True), 'stored', image, ref)['accepted'])
        self.assertFalse(benchmark.image_acceptance(dict(verified_flat_oracle=True, flat_oracle=True),
                                                     'stored', image, ref)['accepted'])

    def test_rounding_rejects_alpha_background_and_larger_errors(self):
        ref = bytes((0, 0, 0, 0)) + bytes((10, 20, 30, 255)) * (256 * 256 - 1)
        for offset, delta in ((4, 2), (7, -1), (0, 1)):
            image = bytearray(ref)
            image[offset] += delta
            self.assertFalse(benchmark.image_acceptance(dict(verified_flat_oracle=True),
                                                         'stored', image, ref)['accepted'])


if __name__ == '__main__':
    unittest.main()

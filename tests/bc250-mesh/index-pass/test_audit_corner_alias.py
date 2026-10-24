import unittest

from audit_corner_alias import independent_score, measured_samples
import corner_alias


class AuditTests(unittest.TestCase):
    def test_missing_and_reordered_samples_fail(self):
        text = '\n'.join(f'INDEX_BENCH_SAMPLE sample={i} warmup={int(i < 2)} gpu_us=10.00 draws=1'
                         for i in range(3))
        self.assertEqual(measured_samples(text, 1, 1), [10])
        with self.assertRaisesRegex(AssertionError, 'incomplete'):
            measured_samples(text, 31, 1)
        with self.assertRaisesRegex(AssertionError, 'metadata'):
            measured_samples(text.replace('sample=2', 'sample=1'), 1, 1)

    def test_unstable_apparent_gain_is_rejected(self):
        rows = [dict(samples_us=[v] * 31) for v in (100, 80, 140)]
        score = independent_score(corner_alias.CASES[0], 1, rows)
        self.assertGreater(score['full']['speedup'], 1.05)
        self.assertFalse(score['stable'])
        self.assertFalse(score['win'])

    def test_excluded_control_cannot_win_feature(self):
        rows = [dict(samples_us=[v] * 31) for v in (100, 80, 100)]
        score = independent_score(corner_alias.CASES[-1], 1, rows)
        self.assertTrue(score['stable'])
        self.assertFalse(score['win'])


if __name__ == '__main__':
    unittest.main()

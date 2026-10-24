import unittest

from audit_automatic import independent_score


class AutomaticAuditTests(unittest.TestCase):
    def test_missed_faster_route_is_retained_as_regression(self):
        case = dict(tag='medium')
        rows = [dict(samples_us=[v] * 31) for v in (155, 146, 155, 155)]
        result = independent_score(case, 1, rows)
        self.assertTrue(result['stable'])
        self.assertTrue(result['regression'])
        self.assertFalse(result['win'])

    def test_matching_measured_stored_route_wins(self):
        rows = [dict(samples_us=[v] * 31) for v in (358, 226, 226, 358)]
        result = independent_score(dict(tag='heavy'), 1, rows)
        self.assertTrue(result['win'])
        self.assertFalse(result['regression'])

    def test_drift_excludes_apparent_gain(self):
        rows = [dict(samples_us=[v] * 31) for v in (100, 60, 60, 120)]
        result = independent_score(dict(tag='drift'), 1, rows)
        self.assertFalse(result['stable'])
        self.assertFalse(result['win'])


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
import unittest

import benchmark as bench
import corner_alias as alias


class CornerAliasChecks(unittest.TestCase):
    def rows(self, before=100, candidate=90, after=100):
        return [dict(samples_us=[value] * 31) for value in (before, candidate, after)]

    def test_stable_gain_and_complete_rounds(self):
        case = alias.CASES[0]
        scores = [alias.score(case, r, self.rows()) for r in (1, 2, 3)]
        self.assertTrue(alias.aggregate(scores)[0]['confirmed_win'])
        with self.assertRaisesRegex(AssertionError, 'incomplete rounds'):
            alias.aggregate(scores[:2])

    def test_drift_and_sample_spread_exclude_gain(self):
        case = alias.CASES[0]
        self.assertFalse(alias.score(case, 1, self.rows(after=106))['win'])
        rows = self.rows()
        rows[1]['samples_us'] = [70] * 5 + [90] * 21 + [110] * 5
        self.assertFalse(alias.score(case, 1, rows)['stable'])

    def test_regressions_are_retained_and_incomplete_samples_fail(self):
        case = alias.CASES[-1]
        self.assertFalse(alias.score(case, 1, self.rows())['win'])
        self.assertTrue(alias.score(case, 1, self.rows(candidate=110))['regression'])
        rows = self.rows()
        rows[0]['samples_us'].pop()
        with self.assertRaisesRegex(AssertionError, 'incomplete samples'):
            alias.score(case, 1, rows)

    def test_compile_audit_checks_both_admission_and_exclusion(self):
        case = dict(alias.CASES[0], corner_alias=1)
        bench.compile_audit(case, 'mesh-alias', 'BC250 MESH CORNER ALIAS: applied')
        with self.assertRaisesRegex(AssertionError, 'alias admission'):
            bench.compile_audit(case, 'mesh-alias', '')
        control = dict(alias.CASES[-1], corner_alias=1)
        bench.compile_audit(control, 'mesh-alias', '')
        with self.assertRaisesRegex(AssertionError, 'alias admission'):
            bench.compile_audit(control, 'mesh-alias', 'BC250 MESH CORNER ALIAS: applied')


if __name__ == '__main__':
    unittest.main()

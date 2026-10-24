import unittest

import automatic
import benchmark


class AutomaticTests(unittest.TestCase):
    def test_heavy_and_light_use_current_workload(self):
        for name, steps, expected in [('profile64', 1024, 'mesh-auto'),
                                      ('profile256pp128', 0, 'mesh-auto'),
                                      ('profile256pp128', 64, 'stored'),
                                      ('profile256pp128', 256, 'stored')]:
            case, mode = automatic.auto_variant(automatic.fixture(name, 'count', steps, 32))
            self.assertEqual(mode, expected)
            self.assertTrue(case['automatic'])
        _, mode = automatic.auto_variant(automatic.fixture('profile256pp128', 'direct', 64, 16))
        self.assertEqual(mode, 'mesh-auto')

    def test_dispatch_guards_and_alternating_calls(self):
        for draw, groups in [('direct', 1), ('single', 8)]:
            _, mode = automatic.auto_variant(automatic.fixture('profile256pp128', draw, 256, 4, groups=groups))
            self.assertEqual(mode, 'mesh-auto')
        c, mode = automatic.auto_variant(automatic.fixture('profile256pp128', 'count', 256, 4, alternate_steps=True))
        self.assertEqual(c['expected_idx_calls'], 2)
        self.assertEqual(mode, 'stored')
        benchmark.audit_usage('BC250_USAGE submits=1 idx_direct=0 idx_indirect_calls=2', c, mode)
        for legacy in [c for c in automatic.SAFETY if c.get('auto_replay')]:
            _, mode = automatic.auto_variant(legacy)
            self.assertEqual(mode, 'existing-before')

    def test_wrong_recorded_decision_fails_before_gpu_release(self):
        c, mode = automatic.auto_variant(automatic.fixture('profile256pp128', 'count', 0, 4))
        log = 'MESH IDXPASS: applied V=256 P=128 threads=128\nstored outputs bytes/group=10240\n'
        benchmark.compile_audit(c, mode, log)
        with self.assertRaisesRegex(AssertionError, 'automatic route'):
            benchmark.compile_audit(c, mode, log + 'BC250 MESH AUTO: indirect stored\n')

    def test_stable_choice_must_track_faster_route(self):
        case = automatic.CASES[-1]
        good = [dict(samples_us=[v] * 31) for v in (100, 60, 60, 100)]
        self.assertTrue(automatic.score(case, 1, good)['win'])
        bad = [dict(samples_us=[v] * 31) for v in (100, 60, 100, 100)]
        self.assertTrue(automatic.score(case, 1, bad)['regression'])


if __name__ == '__main__':
    unittest.main()

"""CPU checks for final workload coverage and repeated ordering."""
import unittest

import setup_final


class FinalCasesTests(unittest.TestCase):
    def test_coverage_and_repeat_order(self):
        cases = setup_final.CASES
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({c['tag'] for c in cases}), 24)
        self.assertEqual(len({c['focus_key'] for c in cases}), 8)
        self.assertEqual({c['name'] for c in cases},
                         {'profile64', 'profile128p64', 'profile128p128', 'profile256pp128'})
        self.assertEqual(sum(c.get('last', 0) for c in cases), 3)
        for key in {c['focus_key'] for c in cases}:
            repeated = [c for c in cases if c['focus_key'] == key]
            self.assertEqual([c['repeat'] for c in repeated], [1, 2, 3])
            self.assertTrue(all(c['warmup_ms'] == 500 and c['benchmark_samples'] == 31 for c in repeated))
        self.assertTrue(all(c['route_order'] == ('stored', 'existing-before')
                            for c in cases if c['repeat'] == 2))


if __name__ == '__main__':
    unittest.main()

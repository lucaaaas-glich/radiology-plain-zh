from collections import Counter
import unittest

from scripts.prepare_p1_datasets import allocate, choose_arms, common_support_summary, representative, stratum


class SelectionTests(unittest.TestCase):
    def test_unsupported_strata_are_disclosed_and_excluded_from_both_arms(self):
        rows = [{'id': str(i), 'reference_zh': 'x'} for i in range(4)]
        risks = {str(i): {'features': {'source': 'a' if i == 0 else 'b'},
                           'screen_status': 'risk_flagged' if i == 0 else 'no_known_rule_flag'}
                 for i in range(4)}
        cfg = {'requested_per_arm': 2, 'minimum_per_arm': 1, 'seed': 1,
               'matching_features': ['source'], 'target_character_bin_boundaries': []}
        arms, _ = choose_arms(rows, risks, cfg)
        summary = common_support_summary(rows, risks, cfg)
        self.assertEqual(summary['strata_without_no_known_flag_candidates'], 1)
        self.assertEqual(summary['raw_reports_in_unsupported_strata'], 1)
        self.assertTrue(all(r['id'] != '0' for arm in arms.values() for r in arm))

    def test_capped_allocation_keeps_total_and_available_capacity(self):
        raw, clean = {('a',): 90, ('b',): 10}, {('a',): 2, ('b',): 10}
        counts = allocate(raw, clean, 10)
        self.assertEqual(sum(counts.values()), 10)
        self.assertEqual(counts[('a',)], 2)
        self.assertEqual(counts[('b',)], 8)

    def test_capacity_shortfall_is_explicit(self):
        with self.assertRaises(ValueError):
            allocate({('a',): 10}, {('a',): 2}, 3)

    def test_one_representative_per_group_deterministic_without_risk_labels(self):
        rows = [{'id': str(i)} for i in range(10)]
        groups = {str(i): str(i // 2) for i in range(10)}
        a = representative(rows, groups, 42)
        b = representative(list(reversed(rows)), groups, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)
        self.assertEqual(len({groups[r['id']] for r in a}), 5)

    def test_matched_arms_keep_strata_and_original_targets(self):
        rows = [{'id': str(i), 'reference_zh': 'original target ' + str(i)} for i in range(40)]
        risks = {r['id']: {'features': {'source': 'first' if int(r['id']) < 30 else 'second'},
                           'screen_status': 'risk_flagged' if int(r['id']) % 3 == 0 else 'no_known_rule_flag'} for r in rows}
        cfg = {'requested_per_arm': 20, 'minimum_per_arm': 10, 'seed': 7,
               'matching_features': ['source', 'target_length_bin'],
               'target_character_bin_boundaries': [80]}
        arms, _ = choose_arms(rows, risks, cfg)
        self.assertEqual(len(arms['S0']), 20)
        self.assertEqual(Counter(stratum(r, risks[r['id']], cfg) for r in arms['S0']),
                         Counter(stratum(r, risks[r['id']], cfg) for r in arms['S1']))
        self.assertTrue(all(risks[r['id']]['screen_status'] == 'no_known_rule_flag' for r in arms['S1']))
        originals = {r['id']: r for r in rows}
        self.assertTrue(all(r == originals[r['id']] for arm in arms.values() for r in arm))

    def test_insufficient_clean_rows_never_padded_or_duplicated(self):
        rows = [{'id': 'a', 'reference_zh': 'x'}]
        risks = {'a': {'features': {'source': 'one'}, 'screen_status': 'no_known_rule_flag'}}
        cfg = {'requested_per_arm': 5, 'minimum_per_arm': 3, 'seed': 1,
               'matching_features': ['source'], 'target_character_bin_boundaries': []}
        with self.assertRaisesRegex(ValueError, 'Not enough'):
            choose_arms(rows, risks, cfg)


if __name__ == '__main__':
    unittest.main()

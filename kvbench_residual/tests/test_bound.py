import unittest
import numpy as np
from kvbench_residual.fixtures import Case, Layout, build_case, selective_attention, make_dominant_sink


class BoundTests(unittest.TestCase):
    def test_full_reconstruction_and_global_budget(self):
        for group in (64, 128):
            for window in (0, group+9):
                data = build_case(Case('math', (3*group+11, 0), Layout(group), window=window, hq=4, hk=1))
                ref = selective_attention(data, 'parent')['output']
                np.testing.assert_allclose(selective_attention(data, 'repacked_full')['output'], ref, atol=0, rtol=0)
                for policy in ('bound_k','bound_v','bound_kv'):
                    for splits in (1, 4):
                        r = selective_attention(data, policy, .02, splits=splits)
                        actual = np.linalg.norm(r['output']-ref, axis=-1)
                        expected_bound = .02*np.repeat(r['value_reference'], 4, axis=1)
                        self.assertTrue(np.all(actual <= r['error_bound'] + 1e-11))
                        self.assertTrue(np.all(r['error_bound'] <= expected_bound + 1e-11))

    def test_attention_significance_exercises_true_skip(self):
        data = make_dominant_sink(build_case(Case('dominant', (1024,), Layout(), hq=4, hk=1)))
        ref = selective_attention(data, 'parent')['output']
        for policy in ('bound_k','bound_v','bound_kv'):
            r = selective_attention(data, policy)
            self.assertGreater(sum(r['counts'][k] for k in ('key_skipped_chunks','value_skipped_chunks')), 0)
            self.assertTrue(np.all(np.linalg.norm(r['output']-ref, axis=-1) <= r['error_bound']+1e-11))

    def test_gqa_union_keeps_residual_if_one_head_needs_it(self):
        data = make_dominant_sink(build_case(Case('union', (1024,), Layout(), hq=4, hk=1)))
        all_aligned = selective_attention(data, 'bound_kv')
        data['query'][0, 3, 0] = -1
        one_disagrees = selective_attention(data, 'bound_kv')
        self.assertGreater(all_aligned['counts']['key_skipped_chunks'], one_disagrees['counts']['key_skipped_chunks'])
        self.assertGreater(all_aligned['counts']['value_skipped_chunks'], one_disagrees['counts']['value_skipped_chunks'])

    def test_zero_budget_reads_all_and_anchor_keeps_all_tokens(self):
        data = build_case(Case('zero', (513,), Layout(), hq=4, hk=1))
        full = selective_attention(data, 'parent')
        r = selective_attention(data, 'bound_kv', 0)
        self.assertEqual(r['counts']['key_skipped_chunks'], 0)
        self.assertEqual(r['counts']['value_skipped_chunks'], 0)
        np.testing.assert_array_equal(r['output'], full['output'])
        anchor = selective_attention(data, 'anchor')
        self.assertEqual(anchor['counts']['quantized_chunks'], anchor['counts']['key_skipped_chunks'])
        self.assertEqual(anchor['counts']['quantized_chunks'], anchor['counts']['value_skipped_chunks'])
        self.assertEqual(sum(x['tokens'] for x in anchor['decisions']), sum(x['tokens'] for x in full['decisions']))


if __name__ == '__main__':
    unittest.main()

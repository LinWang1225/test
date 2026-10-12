import ast
import os
from pathlib import Path
import unittest

from kvbench_residual.transform import (get_expected_pins, transform_sources,
    read_sources, POLICIES, DECODE_PATH, STORE_PATH, TORCH_DECODE_PATH)


class TransformTests(unittest.TestCase):
    def test_unknown_source_and_policy_fail_closed(self):
        with self.assertRaises(ValueError):
            transform_sources({}, 'bound_kv')
        with self.assertRaises(ValueError):
            transform_sources({}, 'made_up')

    @unittest.skipUnless(os.getenv('AR_SOURCE_ROOT'), 'Set AR_SOURCE_ROOT to the pinned unmodified vllm directory')
    def test_all_consumers_writers_and_actual_demand_branches(self):
        original = read_sources(os.environ['AR_SOURCE_ROOT'])
        for policy in POLICIES:
            result = transform_sources(original, policy)
            self.assertEqual(set(result), set(get_expected_pins()))
            for text in result.values():
                ast.parse(text)
            self.assertIn('AR_POLICY_NAME = '+repr(policy), result[DECODE_PATH])
            if policy == 'parent':
                self.assertEqual(result[STORE_PATH], original[STORE_PATH])
                self.assertEqual(result[TORCH_DECODE_PATH], original[TORCH_DECODE_PATH])
                continue
            self.assertIn('pack_planes_torch', result[STORE_PATH])
            self.assertIn('unpack_planes_torch', result[TORCH_DECODE_PATH])
            tree = ast.parse(result[DECODE_PATH])
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name in ('_kvarn_fused_decode_kernel','_kvarn_fused_decode_stage1'):
                    low_load_branches = []
                    for child in ast.walk(node):
                        if isinstance(child, ast.If) and isinstance(child.test, ast.Name) and child.test.id in ('ar_skip_k','ar_skip_v'):
                            body_text = '\n'.join(ast.unparse(x) for x in child.body)
                            other_text = '\n'.join(ast.unparse(x) for x in child.orelse)
                            if 'ar_kl_byte = tl.load' in other_text or 'ar_vl_byte = tl.load' in other_text:
                                self.assertNotIn('tl.load', body_text)
                                low_load_branches.append(child.test.id)
                    self.assertIn('ar_skip_k', low_load_branches)
                    self.assertIn('ar_skip_v', low_load_branches)
                if node.name in ('_kvarn_dequant_blocks_kernel','_kvarn_build_packed_kv_kernel','_kvarn_fused_verify_stage1'):
                    txt = ast.unparse(node)
                    self.assertIn('ar_k_high', txt)
                    self.assertIn('ar_k_low', txt)
                    self.assertIn('ar_v_high', txt)
                    self.assertIn('ar_v_low', txt)


if __name__ == '__main__':
    unittest.main()

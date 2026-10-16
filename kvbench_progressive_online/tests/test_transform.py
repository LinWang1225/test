from __future__ import annotations
import unittest
from kvbench_progressive_online.kernel import ADAPTIVE,FULL
from kvbench_progressive_online.transform import (_full_consumer,_fused,DEFAULT_THRESHOLDS,
    transform_writer,transform_torch_decode)


class SourceGateTest(unittest.TestCase):
    def test_progressive_gpu_bit_loads_are_conditional(self):
        self.assertLess(ADAPTIVE.index('ar_kh_b = tl.load'),ADAPTIVE.index('if ar_risk_k2 <= PROG_K_2'))
        self.assertIn('ar_k1_b = tl.load',ADAPTIVE)
        self.assertIn('ar_k0_b = tl.load',ADAPTIVE)
        self.assertIn('ar_v1_b = tl.load',ADAPTIVE)
        self.assertIn('ar_v0_b = tl.load',ADAPTIVE)
        self.assertIn('ar_prefix_lower = _ar_chunk_mass',ADAPTIVE)
        self.assertIn('ar_v0_b = tl.load',FULL)
    def test_strict_full_consumer(self):
        old=('    ar_k_high = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)\n'
             '    ar_k_low = tl.load(KV_cache_ptr + k_addrs + D * GROUP // 4).to(tl.int32)\n'
             '    q_K = (4 * ((ar_k_high >> ((g_offs[None, :] % 4) * 2)) & 3) + ((ar_k_low >> ((g_offs[None, :] % 4) * 2)) & 3)).to(tl.float32)\n')
        result=_full_consumer(old,'k','g_offs')
        self.assertIn('ar_k_bit_addr',result)
        self.assertIn('d_offs[:, None] * (GROUP // 8)',result)
        self.assertIn('ar_k_bit0',result)
        with self.assertRaises(ValueError):_full_consumer(old,'v','g_offs')
    def test_fused_strict_and_adaptive_thresholds(self):
        chunk_start='                # Separate 2-bit planes occupy disjoint halves of each code field.'
        chunk_end='                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]'
        body='def f():\n    AR_POLICY: tl.constexpr = 1\n    b = tl.program_id(0)\n                if AR_POLICY >= 3:\n                    pass\n'+chunk_start+'\n'+chunk_end+'\n'
        prog=_fused(body,'p211_adaptive',DEFAULT_THRESHOLDS)
        self.assertIn('PROG_K_2: tl.constexpr = 0.08',prog)
        self.assertIn('if ar_risk_k2 <= PROG_K_2:',prog)
        self.assertIn('if True:',prog)
        full=_fused(body,'p211_full',DEFAULT_THRESHOLDS)
        self.assertIn('ar_k0_b = tl.load',full)
        with self.assertRaises(ValueError):_fused('def f():\n  pass\n','p211_adaptive',DEFAULT_THRESHOLDS)
    def test_store_imports_exact(self):
        s=('from kvbench_residual.representation import pack_planes_torch\nreturn pack_planes_torch(q)\n')*2
        t=transform_writer(s)
        self.assertEqual(t.count('pack_211_torch(q)'),2)
        d='from kvbench_residual.representation import unpack_planes_torch\nreturn unpack_planes_torch(packed, original_last_dim)\n'
        t=transform_torch_decode(d)
        self.assertIn('return unpack_211_torch(packed, original_last_dim)',t)


if __name__=='__main__':unittest.main()

"""Run in the user's repository to check the unchanged v1 CPU oracle API.

Skipped if kvbench_residual is absent. It is never replaced by a fabricated
implementation in order to claim compatibility with the user's source tree.
"""
import importlib.util
import unittest
import numpy as np
from kvbench_residual_diag.math_ops import build_group,bound_policy

HAS_SOURCE=importlib.util.find_spec('kvbench_residual') is not None

@unittest.skipUnless(HAS_SOURCE,'Existing kvbench_residual package is required for source-bridge tests')
class SourceBridgeTests(unittest.TestCase):
    def test_existing_v1_cpu_oracle(self):
        from kvbench_residual.fixtures import Case,Layout,build_case,selective_attention
        for length in (129,385,1024):
            data=build_case(Case('source_bridge',(length,),Layout()))
            for bn in (16,32,64):
                groups=[build_group(data,h,bn) for h in range(8)]
                for sp in (1,4):
                    for policy in ('parent','repacked_full','anchor','bound_k','bound_v','bound_kv'):
                        ref=selective_attention(data,policy,.02,bn,sp)
                        ours=[bound_policy(g,policy,.02,sp) for g in groups]
                        output=np.concatenate([x['output'] for x in ours],axis=0)[None,:,:]
                        np.testing.assert_allclose(output,ref['output'],rtol=1e-10,atol=1e-11)
                        for key in ('quantized_chunks','key_skipped_chunks','value_skipped_chunks'):
                            self.assertEqual(sum(x[key] for x in ours),ref['counts'][key])
    def test_existing_sliding_window(self):
        from kvbench_residual.fixtures import Case,Layout,build_case,selective_attention
        data=build_case(Case('sliding',(721,),Layout(),window=333))
        groups=[build_group(data,h,32) for h in range(8)]
        for policy in ('anchor','bound_kv'):
            ref=selective_attention(data,policy,.1,32,4)
            ours=[bound_policy(g,policy,.1,4) for g in groups]
            np.testing.assert_allclose(np.concatenate([r['output'] for r in ours],axis=0)[None,:,:],ref['output'],rtol=1e-10,atol=1e-11)

if __name__=='__main__':unittest.main()

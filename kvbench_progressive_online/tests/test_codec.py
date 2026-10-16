from __future__ import annotations
import unittest
import numpy as np
from kvbench_progressive_online.codec import pack_numpy,unpack_numpy,pack_211_torch,unpack_211_torch


class ProgressiveCodecTest(unittest.TestCase):
    def test_full_int4_exact(self):
        for seed in (0,7,31):
            q=np.random.default_rng(seed).integers(0,16,(3,128,128),dtype=np.uint8)
            p=pack_numpy(q)
            self.assertEqual(p.shape,(3,128,64))
            np.testing.assert_array_equal(unpack_numpy(p,128,4),q)
    def test_midpoints_and_refinement(self):
        q=np.broadcast_to(np.arange(16,dtype=np.uint8),(128,16))
        p=pack_numpy(q)
        d2=unpack_numpy(p,16,2)
        d3=unpack_numpy(p,16,3)
        d4=unpack_numpy(p,16,4)
        np.testing.assert_allclose(d2,4*(q//4)+1.5)
        np.testing.assert_allclose(d3,4*(q//4)+2*((q//2)%2)+0.5)
        np.testing.assert_array_equal(d4,q)
    def test_physical_triton_bit_addressing(self):
        # Verify the exact addressing formulas used by fused GPU code for
        # both K=[D,G] and V=[G,D] complete 4-bit reconstruction.
        rng=np.random.default_rng(23)
        for rows,cols in ((128,128),(64,128),(128,64)):
            original=rng.integers(0,16,(rows,cols),dtype=np.uint8)
            packed=pack_numpy(original).ravel()
            count=rows*cols
            for row in (0,1,rows//2,rows-1):
                for col in (0,1,6,7,8,31,cols-1):
                    high_addr=row*(cols//4)+col//4
                    bit_addr=count//4+row*(cols//8)+col//8
                    high=(int(packed[high_addr])>>(2*(col%4)))&3
                    bit1=(int(packed[bit_addr])>>(col%8))&1
                    bit0=(int(packed[bit_addr+count//8])>>(col%8))&1
                    self.assertEqual(4*high+2*bit1+bit0,int(original[row,col]))

    def test_bad_inputs(self):
        for arr in [np.array([1,2,3]),np.full((2,8),16)]:
            with self.assertRaises(ValueError):pack_numpy(arr)
    def test_torch_matches_numpy(self):
        try:import torch
        except ImportError:self.skipTest('torch unavailable')
        for rows,cols in ((128,128),(3,16)):
            q=np.random.default_rng(cols).integers(0,16,(rows,cols),dtype=np.uint8)
            p=pack_211_torch(torch.from_numpy(q))
            np.testing.assert_array_equal(p.numpy(),pack_numpy(q))
            np.testing.assert_array_equal(unpack_211_torch(p,cols).numpy(),q)


if __name__=='__main__':unittest.main()

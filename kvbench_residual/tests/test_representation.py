import unittest
import numpy as np

from kvbench_residual.representation import (pack_planes_numpy, unpack_planes_numpy,
    anchor_and_residual_codes, repack_parent_cache_numpy, parent_unpack_numpy)
from kvbench_residual.fixtures import Case, Layout, build_case


class RepresentationTests(unittest.TestCase):
    def test_all_codes_roundtrip_and_independent_global_halves(self):
        q = np.tile(np.arange(16, dtype=np.uint8), (3, 8, 8))
        p = pack_planes_numpy(q)
        self.assertEqual(p.nbytes, q.size//2)
        np.testing.assert_array_equal(unpack_planes_numpy(p, q.shape[-1]), q)
        flat = p.reshape(q.shape[0], -1)
        changed = q.copy()
        changed ^= np.uint8(3)
        other = pack_planes_numpy(changed).reshape(flat.shape)
        np.testing.assert_array_equal(flat[:, :flat.shape[-1]//2], other[:, :flat.shape[-1]//2])
        self.assertFalse(np.array_equal(flat[:, flat.shape[-1]//2:], other[:, flat.shape[-1]//2:]))

    def test_midpoint_residual_is_exact_and_bounded(self):
        q = np.arange(16)
        a, r = anchor_and_residual_codes(q)
        np.testing.assert_array_equal(a+r, q)
        self.assertEqual(float(np.abs(r).max()), 1.5)
        self.assertEqual(len(np.unique(a)), 4)

    def test_cache_bytes_metadata_and_codes_preserved(self):
        data = build_case(Case("pack", (257,), Layout(), hq=4, hk=1))
        native, cfg = data["cache"], data["case"].layout
        planar = repack_parent_cache_numpy(native, cfg)
        self.assertEqual(native.shape, planar.shape)
        self.assertEqual(native.nbytes, planar.nbytes)
        excluded = np.zeros(cfg.tile_bytes, bool)
        for first, last, rows, cols in ((cfg.offsets['K_PACKED_OFFSET'],cfg.offsets['K_S_COL_OFFSET'],128,128),
                                       (cfg.offsets['V_PACKED_OFFSET'],cfg.offsets['V_S_COL_OFFSET'],128,128)):
            excluded[first:last] = True
            shape = (*native.shape[:-1], rows, cols//2)
            actual = unpack_planes_numpy(planar[..., first:last].reshape(shape), cols)
            expected = parent_unpack_numpy(native[..., first:last].reshape(shape), cols)
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(native[..., ~excluded], planar[..., ~excluded])

    def test_invalid_representation_is_rejected(self):
        for q in (np.array([1,2]), np.ones((2,3)), np.full((2,4), 16), np.full((2,4), -1)):
            with self.assertRaises(ValueError):
                pack_planes_numpy(q)


if __name__ == '__main__':
    unittest.main()

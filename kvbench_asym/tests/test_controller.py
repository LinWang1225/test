import unittest

import numpy as np

from kvbench_asym.controller import ControllerConfig, estimate_error, run_controller
from kvbench_asym.analyze import group_rows
from kvbench_asym.tests.helpers import group


def chunks():
    return [
        dict(block=0, quantized=True, k_bytes=8, v_bytes=8, lse_a=np.array([3.0]),
             epsilon_k=np.array([.1]), eta_v=.2),
        dict(block=1, quantized=True, k_bytes=8, v_bytes=8, lse_a=np.array([3.0]),
             epsilon_k=np.array([.8]), eta_v=.4),
    ]


class ControllerTests(unittest.TestCase):
    def test_anchor_only_uses_only_anchor_metadata(self):
        value = estimate_error(chunks(), np.zeros(2, bool), np.zeros(2, bool))
        self.assertGreater(value, 0)

    def test_stops_when_estimate_reaches_tolerance(self):
        result = run_controller(chunks(), ControllerConfig(tolerance=.03))
        self.assertEqual(result.stop_reason, "estimated_tolerance")
        self.assertLessEqual(result.estimated_error, .03)
        self.assertGreater(result.rho, 0)

    def test_selects_highest_gain_block(self):
        result = run_controller(chunks(), ControllerConfig(tolerance=.15, max_steps=1))
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.history[0]["block"], 1)

    def test_zero_tolerance_exhausts_residual(self):
        result = run_controller(chunks(), ControllerConfig(tolerance=0))
        self.assertIn(result.stop_reason, ("estimated_tolerance", "residual_exhausted"))
        self.assertEqual(result.rho, 1.0)

    def test_invalid_margin_rejected(self):
        with self.assertRaises(ValueError):
            ControllerConfig(tolerance=.1, safety_margin=.9)

    def test_analysis_records_estimate_and_true_error(self):
        rows, packed, n = group_rows(
            group(), 16, ["1.0"], [".5"], ["staged_mass"], [42], {},
            controller_tolerances=(.01,), controller_margin=1.2)
        row = next(item for item in rows if item["kind"] == "controller")
        self.assertEqual(row["method"], "query_aware_bound")
        self.assertIn("controller_estimated_error", row)
        self.assertIn("relative_l2_max_head", row)
        self.assertGreaterEqual(row["rho_q"], 0)
        self.assertEqual(np.unpackbits(packed, axis=-1)[..., :n].shape, (len(rows), 2, n))
        self.assertTrue(all("controller_estimated_error" in item for item in rows))


if __name__ == "__main__":
    unittest.main()

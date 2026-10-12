"""Fixed, preregistered policies for the initial K4V4 residual experiment."""
from __future__ import annotations

POLICIES = ("parent", "repacked_full", "anchor", "bound_k", "bound_v", "bound_kv")
DEFAULT_EPSILON = 0.02
POLICY_DESCRIPTIONS = {
    "parent": "Original KVarN K4V4 packed layout/arithmetic with common empty-mask safeguards",
    "repacked_full": "Independent high-2/low-2 bit planes; always read all residual planes",
    "anchor": "All tokens contribute high-2-bit K/V anchors; skip all residual planes in quantized decode",
    "bound_k": "Select K residual blocks under the K error budget; read every V residual",
    "bound_v": "Read every K residual; select V residual blocks under the V error budget",
    "bound_kv": "Select K and V residual blocks with half the total budget allocated to each",
}


def validate_epsilon(value):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("--epsilon must be finite, positive and at most 1")
    return float(value)


def method_name(policy):
    if policy not in POLICIES:
        raise ValueError("Unknown residual policy: " + str(policy))
    return "ar_k4v4_" + policy

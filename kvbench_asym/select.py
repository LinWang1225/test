"""Reference-free masks with EXACT equal logical payload bytes.

Inputs contain only anchor logits, selected key logits, public geometry and
costs. Never use full attention output, unread V, best-prefix, or error oracle.
Physical DRAM transactions and scheduler costs are not inferred by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
import numpy as np


@dataclass(frozen=True)
class SelectionView:
    quantized: np.ndarray
    eligible: np.ndarray
    costs: np.ndarray
    lse_anchor: np.ndarray
    lse_full_key: np.ndarray
    packet_bytes: int


def from_group(group, block_n):
    quant = group.quantized
    costs = group.costs
    packet = block_n * group.full.shape[-1] // 4
    eligible = np.array([c['quantized'] and c['tokens'] == block_n and
                         c['k_bytes'] == packet and c['v_bytes'] == packet for c in group.chunks], bool)
    # Partial boundary packets are explicitly FULL READ, not silently discarded.
    return SelectionView(quant, eligible, costs,
                         np.stack([c['lse_a'] for c in group.chunks]),
                         np.stack([c['lse_f'] for c in group.chunks]), packet)


def quota(n, total, share):
    """total=rho_K+rho_V in [0,2]; share=K priority within feasible range.

    All shares at the same total receive the SAME integer action count.
    Odd symmetric counts round K upward; exact achieved K/V rates are reported.
    """
    total, share = Decimal(str(total)), Decimal(str(share))
    if not total.is_finite() or not share.is_finite() or not 0 <= total <= 2 or not 0 <= share <= 1 or n < 0:
        raise ValueError('Invalid total/share/packet count')
    m = int((total * n).to_integral_value(rounding=ROUND_FLOOR))
    lo, hi = max(0, m-n), min(m, n)
    nk = int((Decimal(lo) + share*(hi-lo)).to_integral_value(rounding=ROUND_HALF_UP))
    return nk, m-nk, m


def _mass(lse):
    if lse.ndim != 2 or not np.isfinite(lse).all():
        raise ValueError('Invalid log-masses')
    x = np.exp(lse-lse.max(axis=0, keepdims=True))
    return x / x.sum(axis=0, keepdims=True)


def _order(score, ids):
    # Public chunk order breaks ties. No residual-dependent stopping rule.
    return ids[np.argsort(-score[ids], kind='stable')]


def choose(view, nk, nv, method, seed=42):
    ids = np.flatnonzero(view.eligible)
    if not (0 <= nk <= len(ids) and 0 <= nv <= len(ids)):
        raise ValueError('Quota exceeds eligible packets')
    k, v = np.ones(len(view.quantized), bool), np.ones(len(view.quantized), bool)
    k[ids] = False; v[ids] = False
    if method == 'random':
        rng = np.random.default_rng(seed)
        ko, vo = rng.permutation(ids), rng.permutation(ids)
        k[ko[:nk]] = True; v[vo[:nv]] = True
    elif method in ('anchor_mass', 'staged_mass'):
        anchor_mass = _mass(view.lse_anchor).max(axis=1)
        ko = _order(anchor_mass, ids)
        k[ko[:nk]] = True
        if method == 'staged_mass':
            # Only already selected K residuals may affect V's ranking.
            used_lse = np.where(k[:, None], view.lse_full_key, view.lse_anchor)
            value_mass = _mass(used_lse).max(axis=1)
        else:
            value_mass = anchor_mass
        vo = _order(value_mass, ids)
        v[vo[:nv]] = True
    else:
        raise ValueError('Unknown selector: ' + method)
    if int(k[ids].sum()) != nk or int(v[ids].sum()) != nv:
        raise AssertionError('Selection did not spend its exact quota')
    if not np.all(k[~view.eligible]) or not np.all(v[~view.eligible]):
        raise AssertionError('Protected partial packets or FP16 pools were changed')
    return k, v


def accounting(view, k, v):
    costs = view.costs
    kb = int((costs[:, 0] * k).sum())
    vb = int((costs[:, 1] * v).sum())
    anchor = int(costs.sum())
    baseline = 2*anchor
    return dict(selected_key_packets=int(k[view.eligible].sum()),
                selected_value_packets=int(v[view.eligible].sum()),
                eligible_residual_read_bytes=int((k[view.eligible].sum()+v[view.eligible].sum())*view.packet_bytes),
                key_residual_bytes=kb, value_residual_bytes=vb,
                anchor_payload_bytes=anchor, logical_payload_read_bytes=anchor+kb+vb,
                full_planar_payload_bytes=baseline,
                payload_reduction_fraction=(1-(anchor+kb+vb)/baseline) if baseline else None,
                key_residual_read_fraction=kb/int(costs[:, 0].sum()) if costs[:, 0].sum() else None,
                value_residual_read_fraction=vb/int(costs[:, 1].sum()) if costs[:, 1].sum() else None,
                controllable_packets=int(view.eligible.sum()),
                protected_quantized_packets=int((view.quantized & ~view.eligible).sum()),
                eligible_for_equal_payload=bool(view.eligible.any()))

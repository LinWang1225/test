"""Offline Query-aware Anchor/Residual controller.

The controller only consumes Anchor logits and per-chunk metadata.  It does
not inspect full-key logits, full values, or the reference output.  Its error
estimate is a conservative, calibrated proxy; callers must compare it with
the true Full-KV error after the decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    tolerance: float
    safety_margin: float = 1.0
    max_steps: int | None = None
    block_n: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        if not np.isfinite(self.safety_margin) or self.safety_margin < 1:
            raise ValueError("safety_margin must be finite and at least 1")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be non-negative or None")
        if self.block_n is not None and self.block_n < 1:
            raise ValueError("block_n must be positive or None")


@dataclass(frozen=True)
class ControllerResult:
    read_k: np.ndarray
    read_v: np.ndarray
    estimated_error: float
    residual_bytes: int
    total_residual_bytes: int
    steps: int
    stop_reason: str
    history: tuple[dict[str, Any], ...]

    @property
    def rho(self) -> float | None:
        if not self.total_residual_bytes:
            return None
        return self.residual_bytes / self.total_residual_bytes


def _anchor_mass(chunks: list[dict[str, Any]]) -> np.ndarray:
    masses = []
    for chunk in chunks:
        lse = np.asarray(chunk["lse_a"], dtype=np.float64)
        if lse.ndim != 1 or not np.isfinite(lse).all():
            raise ValueError("Each chunk must contain finite 1-D anchor logits")
        masses.append(lse)
    if not masses:
        return np.empty((0,), dtype=np.float64)
    stacked = np.stack(masses)
    shifted = stacked - np.max(stacked, axis=0, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=0, keepdims=True)


def _chunk_bounds(chunk: dict[str, Any]) -> tuple[float, float]:
    key = float(np.max(np.asarray(chunk.get("epsilon_k", 0.0), dtype=np.float64)))
    value = float(chunk.get("eta_v", 0.0))
    if not np.isfinite(key) or key < 0 or not np.isfinite(value) or value < 0:
        raise ValueError("Chunk error metadata must be finite and non-negative")
    return key, value


def estimate_error(
    chunks: list[dict[str, Any]],
    read_k: np.ndarray,
    read_v: np.ndarray,
    safety_margin: float = 1.0,
) -> float:
    """Estimate remaining absolute attention error from Anchor metadata."""
    if not np.isfinite(safety_margin) or safety_margin < 1:
        raise ValueError("safety_margin must be finite and at least 1")
    if len(read_k) != len(chunks) or len(read_v) != len(chunks):
        raise ValueError("Masks and chunks must have equal lengths")
    masses = _anchor_mass(chunks)
    key = np.zeros(len(chunks), dtype=np.float64)
    value = np.zeros(len(chunks), dtype=np.float64)
    for i, chunk in enumerate(chunks):
        key_bound, value_bound = _chunk_bounds(chunk)
        key[i] = float(np.max(masses[i] * key_bound))
        value[i] = float(np.max(masses[i] * value_bound))
    return float((key[~np.asarray(read_k, dtype=bool)].sum()
                  + value[~np.asarray(read_v, dtype=bool)].sum()) * safety_margin)


def _cost(chunk: dict[str, Any], side: str) -> int:
    value = chunk["k_bytes"] if side == "k" else chunk["v_bytes"]
    value = int(value)
    if value < 0:
        raise ValueError("Residual byte costs must be non-negative")
    return value


def run_controller(chunks: list[dict[str, Any]], config: ControllerConfig) -> ControllerResult:
    """Select residual sides by estimated error reduction per byte.

    A step is block-oriented: all available K or V residual packets belonging
    to one logical block are considered as one action.  The implementation
    keeps the masks at chunk granularity so partial/protected chunks remain
    explicit and auditable.
    """
    n = len(chunks)
    eligible = np.array([
        bool(c.get("quantized", False))
        and bool(c.get("residual_eligible", True))
        and (config.block_n is None or int(c.get("tokens", config.block_n)) == config.block_n)
        for c in chunks
    ], dtype=bool)
    read_k = ~eligible
    read_v = read_k.copy()
    total = int(sum(_cost(c, "k") + _cost(c, "v") for c, ok in zip(chunks, eligible) if ok))
    masses = _anchor_mass(chunks)
    key_contrib = np.array([
        float(np.max(masses[i] * _chunk_bounds(chunk)[0])) for i, chunk in enumerate(chunks)
    ])
    value_contrib = np.array([
        float(np.max(masses[i] * _chunk_bounds(chunk)[1])) for i, chunk in enumerate(chunks)
    ])

    def estimate() -> float:
        return float((key_contrib[~read_k].sum() + value_contrib[~read_v].sum())
                     * config.safety_margin)

    history: list[dict[str, Any]] = []
    steps = 0
    current = estimate()
    if current <= config.tolerance:
        return ControllerResult(read_k, read_v, current, 0, total, 0, "estimated_tolerance", tuple(history))

    actions: list[tuple[int, str, np.ndarray, int]] = []
    for block in sorted({c.get("block") for c, ok in zip(chunks, eligible) if ok}):
        indices = np.array([i for i, c in enumerate(chunks) if eligible[i] and c.get("block") == block], dtype=int)
        for side in ("k", "v"):
            cost = int(sum(_cost(chunks[i], side) for i in indices))
            if cost:
                actions.append((int(block), side, indices, cost))

    while actions and (config.max_steps is None or steps < config.max_steps):
        before = estimate()
        candidates: list[tuple[float, int, str, np.ndarray, int, float]] = []
        for block, side, indices, cost in actions:
            trial_k, trial_v = read_k.copy(), read_v.copy()
            if side == "k":
                trial_k[indices] = True
            else:
                trial_v[indices] = True
            removed = (key_contrib[indices].sum() if side == "k" else value_contrib[indices].sum())
            after = before - float(removed * config.safety_margin)
            candidates.append(((before - after) / cost, block, side, indices, cost, after))
        gain, block, side, indices, cost, after = max(candidates, key=lambda item: (item[0], -item[4]))
        actions = [action for action in actions if not (action[0] == block and action[1] == side)]
        if gain <= 0:
            return ControllerResult(
                read_k, read_v, before, int(sum(_cost(c, "k") * read_k[i] + _cost(c, "v") * read_v[i] for i, c in enumerate(chunks))),
                total, steps, "no_positive_gain", tuple(history),
            )
        if side == "k":
            read_k[indices] = True
        else:
            read_v[indices] = True
        steps += 1
        history.append(dict(step=steps, block=block, side=side, bytes=cost, gain=gain, estimated_error=after))
        current = after
        if current <= config.tolerance:
            break

    reason = "estimated_tolerance" if current <= config.tolerance else (
        "max_steps" if config.max_steps is not None and steps >= config.max_steps else "residual_exhausted"
    )
    spent = int(sum(_cost(c, "k") * read_k[i] + _cost(c, "v") * read_v[i] for i, c in enumerate(chunks)))
    return ControllerResult(read_k, read_v, current, spent, total, steps, reason, tuple(history))

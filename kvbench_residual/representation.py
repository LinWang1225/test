# SPDX-License-Identifier: Apache-2.0
"""Independent high/low 2-bit planes of the *same* parent INT4 code.

The last two dimensions are one tile (rows, columns). Each tile contains
all its high-plane bytes first, followed by all its low-plane bytes. This
is deliberately not high/low interleaving within each row/cache line.
The outward tensor shape and total byte count match KVarN's INT4 record.
No second KV copy and no error-summary sidecar are allocated.
"""
from __future__ import annotations

import numpy as np


def _pack_numpy(code, bits):
    q = np.asarray(code, dtype=np.uint8)
    pack = 8 // bits
    if q.ndim < 2 or q.shape[-1] % pack:
        raise ValueError("A tile needs at least two dimensions and a divisible width")
    q = q.reshape(*q.shape[:-1], q.shape[-1] // pack, pack)
    out = np.zeros(q.shape[:-1], dtype=np.uint8)
    for j in range(pack):
        out |= (q[..., j] & ((1 << bits) - 1)) << (j * bits)
    return out


def pack_planes_numpy(code):
    q = np.asarray(code)
    if q.ndim < 2 or q.shape[-1] % 4 or np.any(q < 0) or np.any(q > 15):
        raise ValueError("INT4 tiles must have width divisible by four and codes 0..15")
    q = q.astype(np.uint8)
    high, low = _pack_numpy(q >> 2, 2), _pack_numpy(q & 3, 2)
    tiles = (*q.shape[:-2], -1)
    return np.concatenate((high.reshape(tiles), low.reshape(tiles)), axis=-1).reshape(
        *q.shape[:-1], q.shape[-1] // 2)


def unpack_planes_numpy(packed, original_last_dim):
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim < 2 or original_last_dim % 4 or p.shape[-1] * 2 != original_last_dim:
        raise ValueError("Invalid INT4 plane shape")
    rows = p.shape[-2]
    flat = p.reshape(*p.shape[:-2], -1)
    half = flat.shape[-1] // 2
    plane_shape = (*p.shape[:-2], rows, original_last_dim // 4)
    high, low = flat[..., :half].reshape(plane_shape), flat[..., half:].reshape(plane_shape)
    shifts = (np.arange(4) * 2).astype(np.uint8)
    high = ((high[..., None] >> shifts) & 3).reshape(*p.shape[:-1], original_last_dim)
    low = ((low[..., None] >> shifts) & 3).reshape(*p.shape[:-1], original_last_dim)
    return (high << 2) | low


def parent_unpack_numpy(packed, original_last_dim):
    p = np.asarray(packed, dtype=np.uint8)
    return ((p[..., None] >> np.array([0, 4], dtype=np.uint8)) & 15).reshape(
        *p.shape[:-1], original_last_dim)


def repack_parent_cache_numpy(cache, layout):
    """Repack only K/V code fields, preserving every scale/padding byte."""
    if layout.key_bits != 4 or layout.value_bits != 4:
        raise ValueError("The v1 experiment supports K4V4 only")
    out = np.array(cache, copy=True)
    d, g, o = layout.head_dim, layout.group, layout.offsets
    for start, end, rows, columns in (
        (o["K_PACKED_OFFSET"], o["K_S_COL_OFFSET"], d, g),
        (o["V_PACKED_OFFSET"], o["V_S_COL_OFFSET"], g, d),
    ):
        q = parent_unpack_numpy(out[..., start:end].reshape(*out.shape[:-1], rows, columns // 2), columns)
        out[..., start:end] = pack_planes_numpy(q).reshape(*out.shape[:-1], -1)
    return out


def pack_planes_torch(code):
    """Drop-in INT4 packer. Called at parent quantization/write time only."""
    import torch
    if code.ndim < 2 or code.shape[-1] % 4:
        raise ValueError("INT4 plane tiles require width divisible by four")
    q = code.to(torch.uint8) & 15
    shape = q.shape
    def plane(values):
        r = values.reshape(*shape[:-1], shape[-1] // 4, 4)
        packed = r[..., 0] | (r[..., 1] << 2) | (r[..., 2] << 4) | (r[..., 3] << 6)
        return packed.reshape(*shape[:-2], -1)
    return torch.cat((plane(q >> 2), plane(q & 3)), dim=-1).reshape(*shape[:-1], shape[-1] // 2)


def unpack_planes_torch(packed, original_last_dim):
    import torch
    if packed.ndim < 2 or original_last_dim % 4 or packed.shape[-1] * 2 != original_last_dim:
        raise ValueError("Invalid INT4 plane shape")
    rows = packed.shape[-2]
    flat = packed.reshape(*packed.shape[:-2], -1)
    half = flat.shape[-1] // 2
    target = (*packed.shape[:-2], rows, original_last_dim // 4)
    hi, lo = flat[..., :half].reshape(target), flat[..., half:].reshape(target)
    result = torch.empty(*packed.shape[:-1], original_last_dim,
                         dtype=torch.uint8, device=packed.device)
    for j in range(4):
        result[..., j::4] = (((hi >> (2*j)) & 3) << 2) | ((lo >> (2*j)) & 3)
    return result


def anchor_and_residual_codes(code):
    q = np.asarray(code, dtype=np.float64)
    anchor = 4 * (q.astype(np.int64) >> 2) + 1.5
    return anchor, q - anchor

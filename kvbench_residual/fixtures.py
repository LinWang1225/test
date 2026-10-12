# SPDX-License-Identifier: Apache-2.0
"""Deterministic native KVarN fixtures; independent from any previous patch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Layout:
    group: int = 128
    value_bits: int = 4
    head_dim: int = 128
    key_bits: int = 4

    def __post_init__(self):
        if self.group not in (64, 128) or self.value_bits != 4:
            raise ValueError("Residual v1 fixtures support K4/V4 with group 64/128")
        if self.head_dim != 128 or self.key_bits != 4:
            raise ValueError("This first experimental gate targets head_dim=128, K4")

    @property
    def offsets(self):
        d, g = self.head_dim, self.group
        kp = d * g // 2
        vp = d * g * self.value_bits // 8
        return {"K_PACKED_OFFSET": 0, "K_S_COL_OFFSET": kp,
                "K_ZP_OFFSET": kp + 2*d, "K_S_ROW_OFFSET": kp + 4*d,
                "V_PACKED_OFFSET": kp + 4*d + 2*g,
                "V_S_COL_OFFSET": kp + 4*d + 2*g + vp,
                "V_S_ROW_OFFSET": kp + 6*d + 2*g + vp,
                "V_ZP_OFFSET": kp + 6*d + 4*g + vp}

    @property
    def tile_bytes(self):
        return self.offsets["V_ZP_OFFSET"] + self.group * 2


@dataclass(frozen=True)
class Case:
    name: str
    seq_lens: tuple[int, ...]
    layout: Layout = Layout()
    window: int = 0
    seed: int = 723
    hq: int = 32
    hk: int = 8
    req_rows: tuple[int, ...] | None = None


def _metadata(values, cache, offset):
    values = np.ascontiguousarray(values, dtype=np.float16)
    cache[..., offset:offset + values.shape[-1]*2] = values.view(np.uint8)


def build_case(case: Case) -> dict:
    """Use actual parent packing, arbitrary non-contiguous pages, sink and tail.

    Values are synthetic bounded tensors, not model calibration data. All
    policies receive the exact same bytes and fp16 pool. No store kernel is
    replaced or used to select favourable quantization errors.
    """
    if not case.seq_lens or min(case.seq_lens) < 0 or case.hq % case.hk:
        raise ValueError("Invalid fixture shape")
    rng = np.random.default_rng(case.seed)
    cfg = case.layout
    d, g, hk, b = cfg.head_dim, cfg.group, case.hk, len(case.seq_lens)
    req_rows = np.arange(b, dtype=np.int32) if case.req_rows is None else np.asarray(case.req_rows, np.int32)
    if len(req_rows) != b or np.any(req_rows < 0):
        raise ValueError("Invalid indirect request-row mapping")
    requests = int(req_rows.max())+1
    request_lengths = [max((length for row, length in zip(req_rows, case.seq_lens) if row == r), default=0)
                       for r in range(requests)]
    max_blocks = max(1, math.ceil(max(case.seq_lens) / g))
    nblocks = requests * max_blocks
    cache = rng.integers(0, 256, (nblocks, hk, cfg.tile_bytes), dtype=np.uint8)
    off = cfg.offsets
    shape_d, shape_g = (nblocks, hk, d), (nblocks, hk, g)
    sk = rng.uniform(.03, .18, shape_d).astype(np.float16)
    _metadata(sk, cache, off["K_S_COL_OFFSET"])
    _metadata(-7.5 * sk.astype(np.float32) + rng.normal(0, .03, shape_d), cache, off["K_ZP_OFFSET"])
    _metadata(np.exp(rng.normal(0, .25, shape_g)), cache, off["K_S_ROW_OFFSET"])
    _metadata(np.exp(rng.normal(0, .25, shape_d)), cache, off["V_S_COL_OFFSET"])
    rv = rng.uniform(.08, .3, shape_g).astype(np.float16)
    _metadata(rv, cache, off["V_S_ROW_OFFSET"])
    vcenter = ((1 << cfg.value_bits) - 1) / 2
    _metadata(-vcenter * rv.astype(np.float32) + rng.normal(0, .02, shape_g), cache, off["V_ZP_OFFSET"])
    query = rng.normal(0, .8, (b, case.hq, d)).astype(np.float16)
    # Physical pages deliberately do not encode request or logical order.
    pages = rng.permutation(nblocks).astype(np.int32).reshape(requests, max_blocks)
    bt = np.full_like(pages, -1)
    b2s = np.full(nblocks, -1, dtype=np.int32)
    pool_pages = []
    for i, length in enumerate(request_lengths):
        count = math.ceil(length/g)
        bt[i, :count] = pages[i, :count]
        if count:
            pool_pages.append(int(pages[i, 0]))
        if count > 1 and length % g:
            pool_pages.append(int(pages[i, count-1]))
    for slot, bid in enumerate(pool_pages):
        b2s[bid] = slot
    pool_shape = (max(1, len(pool_pages)), g, hk, d)
    pool_k = rng.normal(0, .6, pool_shape).astype(np.float16)
    pool_v = rng.normal(0, .3, pool_shape).astype(np.float16)
    return {"case": case, "query": query, "cache": cache, "block_table": bt,
            "seq_lens": np.array(case.seq_lens, np.int32), "block_to_slot": b2s,
            "pool_k": pool_k, "pool_v": pool_v, "scale": 1/math.sqrt(d), "req_rows": req_rows}


def _half(tile, offset, count):
    return np.ascontiguousarray(tile[offset:offset+2*count]).view(np.float16).astype(np.float64)


def unpack_block(data, bid: int, hk: int):
    cfg = data["case"].layout
    d, g, bits = cfg.head_dim, cfg.group, cfg.value_bits
    tile, o = data["cache"][bid, hk], cfg.offsets
    kp = tile[o["K_PACKED_OFFSET"]:o["K_S_COL_OFFSET"]].reshape(d, g//2)
    kq = ((kp[..., None] >> (np.arange(2)*4)) & 15).reshape(d, g).astype(np.float64)
    pack = 8//bits
    vp = tile[o["V_PACKED_OFFSET"]:o["V_S_COL_OFFSET"]].reshape(g, d//pack)
    vq = ((vp[..., None] >> (np.arange(pack)*bits)) & ((1 << bits)-1)).reshape(g, d).astype(np.float64)
    return {"kq": kq, "vq": vq,
            "sk": _half(tile, o["K_S_COL_OFFSET"], d),
            "zk": _half(tile, o["K_ZP_OFFSET"], d),
            "rk": _half(tile, o["K_S_ROW_OFFSET"], g),
            "sv": _half(tile, o["V_S_COL_OFFSET"], d),
            "rv": _half(tile, o["V_S_ROW_OFFSET"], g),
            "zv": _half(tile, o["V_ZP_OFFSET"], g)}


def case_metadata(data):
    return {**asdict(data["case"]), "cache_bytes": int(data["cache"].nbytes),
            "pool_bytes": int(data["pool_k"].nbytes+data["pool_v"].nbytes),
            "pool_blocks": int((data["block_to_slot"] >= 0).sum()),
            "physical_pages": int(data["cache"].shape[0])}


def selective_attention(data, policy="repacked_full", epsilon=0.02, block_n=32, splits=1):
    """Float64 selection oracle and real-arithmetic budget check.

    CPU reads all reference codes to evaluate actual error; the GPU kernel
    instead loads low planes only in demand branches. The oracle is not a
    simulation of TF32/FMA/exp rounding or a performance implementation.
    """
    from .transform import POLICIES
    if policy not in POLICIES or block_n not in (16, 32, 64) or splits < 1:
        raise ValueError("Invalid policy/chunk/splits")
    c, cfg = data["case"], data["case"].layout
    g, d, ratio = cfg.group, cfg.head_dim, c.hq // c.hk
    output = np.zeros((len(c.seq_lens), c.hq, d), np.float64)
    vrefs = np.zeros((len(c.seq_lens), c.hk), np.float64)
    certified = np.zeros((len(c.seq_lens), c.hq), np.float64)
    counts = {"quantized_chunks": 0, "key_skipped_chunks": 0, "value_skipped_chunks": 0}
    decisions = []
    tau_k = epsilon if policy == "bound_k" else epsilon/2 if policy == "bound_kv" else 0
    tau_v = epsilon if policy == "bound_v" else epsilon/2 if policy == "bound_kv" else 0
    def mass(score, err, prefix):
        mx = score.max(axis=1)
        lse = mx + np.log(np.exp(score-mx[:, None]).sum(axis=1))
        lower = np.logaddexp(prefix, lse-err)
        upper = np.exp(np.minimum(0, lse+err-lower))
        return upper, lower
    for b, length in enumerate(c.seq_lens):
        if length <= 0:
            continue
        first = max(0, length-c.window) if c.window else 0
        n_attend = length-first
        nblocks = math.ceil(length/g)
        blocks_per_split = math.ceil(nblocks/splits)
        for hk in range(c.hk):
            q = data["query"][b, hk*ratio:(hk+1)*ratio].astype(np.float64)
            scores_all, values_all = [], []
            vref = 0.0
            kbound_terms, vbound_terms = [], []
            for sp in range(splits):
                prefix = np.full(ratio, -np.inf)
                for k in range(sp*blocks_per_split, min((sp+1)*blocks_per_split, nblocks)):
                    bid = int(data["block_table"][data["req_rows"][b], k])
                    slot = int(data["block_to_slot"][bid])
                    u = unpack_block(data, bid, hk) if slot < 0 else None
                    for begin in range(0, g, block_n):
                        cols = np.arange(begin, min(begin+block_n, g))
                        cols = cols[(k*g+cols >= first) & (k*g+cols < length)]
                        if not len(cols):
                            continue
                        skip_k = skip_v = False
                        err = np.zeros(ratio)
                        if slot >= 0:
                            K = data["pool_k"][slot, cols, hk].astype(np.float64).T
                            V = data["pool_v"][slot, cols, hk].astype(np.float64)
                            score = q @ K * data["scale"]
                            vref = max(vref, float(np.linalg.norm(V, axis=1).max()))
                            if policy.startswith("bound_"):
                                _, prefix = mass(score, err, prefix)
                        else:
                            Kcode, Vcode = u["kq"][:, cols], u["vq"][cols]
                            rk, rv, zv = u["rk"][cols], u["rv"][cols], u["zv"][cols]
                            norm_col = float(np.linalg.norm(u["sv"]))
                            vrange = float(np.maximum(np.abs(zv), np.abs(zv+15*rv)).max())
                            B = norm_col*vrange
                            vref = max(vref, B)
                            residual_v = 1.5*float(np.abs(rv).max())*norm_col
                            if policy in ("anchor", "bound_k", "bound_kv"):
                                Kanchor = 4*(Kcode.astype(np.int64) >> 2)+1.5
                                K = (Kanchor*u["sk"][:, None]+u["zk"][:, None])*rk[None, :]
                                score = q @ K * data["scale"]
                                err = 1.5*np.abs(rk).max()*np.sum(np.abs(q)*np.abs(u["sk"])[None, :], axis=1)*abs(data["scale"])
                                err = err*1.000001+1e-7
                                upper, _ = mass(score, err, prefix)
                                skip_k = policy == "anchor" or bool(np.all(2*upper*err <= tau_k*len(cols)/n_attend))
                                if skip_k and policy != "anchor":
                                    kbound_terms.append(2*upper*err)
                            if not skip_k:
                                K = (Kcode*u["sk"][:, None]+u["zk"][:, None])*rk[None, :]
                                score = q @ K * data["scale"]
                                err = np.zeros(ratio)
                            if policy.startswith("bound_"):
                                upper, prefix = mass(score, err, prefix)
                            if policy == "anchor":
                                skip_v = True
                            elif policy in ("bound_v", "bound_kv"):
                                rel = residual_v/B if B > 0 else 0.0
                                skip_v = bool(np.all(upper*rel*1.000001 <= tau_v*len(cols)/n_attend))
                                if skip_v:
                                    vbound_terms.append(upper*residual_v)
                            if skip_v:
                                Vcode = 4*(Vcode.astype(np.int64) >> 2)+1.5
                            V = (Vcode*rv[:, None]+zv[:, None])*u["sv"][None, :]
                            counts["quantized_chunks"] += 1
                            counts["key_skipped_chunks"] += int(skip_k)
                            counts["value_skipped_chunks"] += int(skip_v)
                            decisions.append({"request": b, "kv_head": hk, "split": sp, "block": k,
                                              "chunk": begin, "tokens": int(len(cols)),
                                              "skip_k": skip_k, "skip_v": skip_v})
                        scores_all.append(score)
                        values_all.append(V)
            score = np.concatenate(scores_all, axis=1)
            p = np.exp(score-score.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            output[b, hk*ratio:(hk+1)*ratio] = p @ np.concatenate(values_all, axis=0)
            vrefs[b, hk] = vref
            kb = np.sum(kbound_terms, axis=0) if kbound_terms else np.zeros(ratio)
            vb = np.sum(vbound_terms, axis=0) if vbound_terms else np.zeros(ratio)
            certified[b, hk*ratio:(hk+1)*ratio] = kb*vref+vb
    return {"output": output, "value_reference": vrefs, "error_bound": certified,
            "counts": counts, "decisions": decisions}


def make_dominant_sink(data):
    """Diagnostic fixture only: exercises skip/read branches at tight budgets."""
    data["query"].fill(0)
    data["query"][..., 0] = 1
    for b, length in enumerate(data["case"].seq_lens):
        if length:
            bid = int(data["block_table"][data["req_rows"][b], 0])
            slot = int(data["block_to_slot"][bid])
            if slot >= 0:
                data["pool_k"][slot].fill(0)
                data["pool_k"][slot, ..., 0] = 180
    return data


def make_dominant_chunks(data):
    """Synthetic branch stress: each quantized block starts with large logits.

    This is never presented as measured model activation statistics. Unlike a
    sole sink, the construction exercises selection even when splits reset
    their prefix lower bound on different blocks.
    """
    make_dominant_sink(data)
    cfg, cache = data["case"].layout, data["cache"]
    g, d, o = cfg.group, cfg.head_dim, cfg.offsets
    for bid, slot in enumerate(data["block_to_slot"]):
        if slot >= 0:
            continue
        for hk in range(data["case"].hk):
            tile = cache[bid, hk]
            # First channel: 32 large keys, all remaining keys small.
            tile[o['K_PACKED_OFFSET']:o['K_PACKED_OFFSET']+g//2] = 0
            tile[o['K_PACKED_OFFSET']:o['K_PACKED_OFFSET']+16] = 255
            tile[o['K_S_COL_OFFSET']:o['K_S_COL_OFFSET']+2] = np.array([20], np.float16).view(np.uint8)
            tile[o['K_ZP_OFFSET']:o['K_ZP_OFFSET']+2] = np.array([-150], np.float16).view(np.uint8)
            tile[o['K_S_ROW_OFFSET']:o['K_S_ROW_OFFSET']+2*g] = np.ones(g, np.float16).view(np.uint8)
    return data


class GpuCase:
    """Allocate native/planar fixtures once; no allocation in captured launch."""
    def __init__(self, data, device="cuda:0", splits=32, block_n=32, num_warps=4, num_stages=1):
        import torch
        from .representation import repack_parent_cache_numpy
        self.torch, self.data = torch, data
        self.t = {key: torch.from_numpy(data[key]).to(device) for key in
                  ("query", "cache", "block_table", "seq_lens", "block_to_slot", "pool_k", "pool_v", "req_rows")}
        self.planar = torch.from_numpy(repack_parent_cache_numpy(data["cache"], data["case"].layout)).to(device)
        c = data["case"]
        self.b, self.hq, self.hk, self.d = len(c.seq_lens), c.hq, c.hk, c.layout.head_dim
        self.splits, self.block_n, self.num_warps, self.num_stages = splits, block_n, num_warps, num_stages
        if splits <= 0 or splits & (splits-1):
            raise ValueError("Split count must be a positive power of two")
        n = self.b*self.hq
        self.out = torch.zeros((n, self.d), dtype=torch.float16, device=device)
        self.mid_o = torch.zeros((n, splits, self.d), dtype=torch.float32, device=device)
        self.mid_lse = torch.full((n, splits), -float("inf"), dtype=torch.float32, device=device)
        self.stats = torch.zeros((self.b*self.hk*splits, 12), dtype=torch.int64, device=device)

    def launch(self, module, mode="split", autotune=False, collect_stats=False):
        t, c = self.t, self.data["case"]
        q, sl, bt, pk, pv = [t[x] for x in ("query", "seq_lens", "block_table", "pool_k", "pool_v")]
        kv = t["cache"] if module.AR_POLICY_NAME == "parent" else self.planar
        ratio = self.hq//self.hk
        common = dict(MAX_BLOCKS_PER_REQ=bt.shape[1], D=self.d, GROUP=c.layout.group,
                      Q_PER_KV=ratio, Q_PER_KV_PAD=1 << (ratio-1).bit_length(),
                      SLIDING_WINDOW=c.window, K_BITS=4, V_BITS=4,
                      NUM_BLOCKS_LOOKUP=kv.shape[0], VQ_INDIRECT=c.req_rows is not None, **c.layout.offsets)
        kwargs = {} if autotune else dict(BLOCK_N=self.block_n, num_warps=self.num_warps, num_stages=self.num_stages)
        if collect_stats:
            kwargs.update(AR_STATS_PTR=self.stats, AR_COLLECT_STATS=True)
        def kernel(name):
            obj = getattr(module, name)
            return obj if autotune or not hasattr(obj, "configs") else obj.fn
        args = (q, t["req_rows"], bt, sl, t["block_to_slot"], kv, pk, pv)
        strides = (self.hq*self.d, self.d, bt.stride(0), kv.stride(0), kv.stride(1),
                   pk.stride(0), pk.stride(1), pk.stride(2))
        if mode == "single":
            kernel("_kvarn_fused_decode_kernel")[(self.b, self.hk)](
                *args, self.out, self.data["scale"], *strides,
                self.hq*self.d, self.d, **common, **kwargs)
        elif mode == "split":
            kernel("_kvarn_fused_decode_stage1")[(self.b, self.hk, self.splits)](
                *args, self.mid_o, self.mid_lse, self.data["scale"], *strides,
                self.mid_o.stride(0), self.mid_o.stride(1), self.mid_lse.stride(0),
                NUM_KV_SPLITS=self.splits, HQ=self.hq, **common, **kwargs)
            module._kvarn_fused_decode_stage2[(self.b*self.hq,)](
                self.mid_o, self.mid_lse, self.out,
                self.mid_o.stride(0), self.mid_o.stride(1), self.mid_lse.stride(0), self.out.stride(0),
                D=self.d, NUM_KV_SPLITS=self.splits, num_warps=2)
        else:
            raise ValueError(mode)
        return self.out.view(self.b, self.hq, self.d)

    def collect_stats(self, module, mode="split"):
        from .kernel import STATS_FIELDS
        self.stats.zero_()
        self.launch(module, mode=mode, autotune=False, collect_stats=True)
        self.torch.cuda.synchronize(self.out.device)
        rows = self.b*self.hk if mode == "single" else self.b*self.hk*self.splits
        summed = self.stats[:rows].sum(dim=0).cpu().tolist()
        result = {field: int(value) for field, value in zip(STATS_FIELDS, summed)}
        code = sum(result[k] for k in ("key_anchor_code_bytes", "key_residual_code_bytes",
                                      "value_anchor_code_bytes", "value_residual_code_bytes"))
        native = result["native_quantized_code_bytes"]
        result.update(logical_quantized_code_bytes=code,
                      logical_code_reduction_fraction=(1-code/native) if native else None,
                      scope="Instrumented GPU branch counts; logical unique packed addresses per chunk, not DRAM transactions")
        return result

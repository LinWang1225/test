# SPDX-License-Identifier: Apache-2.0
"""Triton source inserted into the pinned KVarN attention kernels.

No GPU package is imported by this module: runtime preparation remains a
CPU-only operation. The generated module is compiled by the real server's
Triton. Selection occurs before the residual tl.load and uses a uniform
branch for the whole GQA head group. All tokens retain Anchor contributions.
"""

JIT_HELPERS = r'''
@triton.jit
def _ar_chunk_mass(scores, epsilon, prefix_lower, valid):
    # Real-arithmetic interval bound. See TECH_CONTRACT.md for numerical scope.
    high = tl.max(scores, axis=1)
    origin = tl.where(high == -float("inf"), 0.0, high)
    mass = tl.sum(tl.exp(scores - origin[:, None]), axis=1)
    log_mass = tl.where(mass > 0, origin + tl.log(tl.maximum(mass, 1.0e-30)),
                        -float("inf"))
    current_lower = log_mass - epsilon
    m = tl.maximum(prefix_lower, current_lower)
    safe_m = tl.where(m == -float("inf"), 0.0, m)
    lower_mass = tl.exp(prefix_lower - safe_m) + tl.exp(current_lower - safe_m)
    total_lower = tl.where(lower_mass > 0,
                           safe_m + tl.log(tl.maximum(lower_mass, 1.0e-30)),
                           -float("inf"))
    upper = tl.exp(tl.minimum(0.0, log_mass + epsilon - total_lower))
    upper = tl.where(valid > 0, upper, 0.0)
    return upper, total_lower


'''


QUANTIZED_CHUNK = r'''
                # Separate 2-bit planes occupy disjoint halves of each code field.
                s_row_K = tl.load(ku16 + (K_S_ROW_OFFSET // 2) + cols,
                                  mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                s_row_V = tl.load(ku16 + (V_S_ROW_OFFSET // 2) + cols,
                                  mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                zp_V = tl.load(ku16 + (V_ZP_OFFSET // 2) + cols,
                               mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                ar_ka = tile_base + K_PACKED_OFFSET + d_offs[:, None] * (GROUP // 4) + cols[None, :] // 4
                ar_kh_byte = tl.load(KV_cache_ptr + ar_ka, mask=cmask[None, :], other=0).to(tl.int32)
                ar_kh = ((ar_kh_byte >> ((cols[None, :] % 4) * 2)) & 3).to(tl.float32)
                ar_skip_k = False
                ar_skip_v = False
                ar_eps = tl.zeros([Q_PER_KV_PAD], tl.float32)
                ar_count = tl.sum(cmask.to(tl.int32), axis=0)
                ar_fraction = ar_count.to(tl.float32) / tl.maximum(ar_attention_tokens, 1)

                if AR_POLICY == 2 or AR_POLICY == 3 or AR_POLICY == 5:
                    q_K = 4.0 * ar_kh + 1.5
                    K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                    ar_scores = tl.where(cmask[None, :], tl.dot(q, K_dg) * scale, -float("inf"))
                    # |q R_K| <= 1.5 max_t|row_scale_t| sum_d|q_d col_scale_d|.
                    ar_eps = 1.5 * tl.max(tl.abs(s_row_K), axis=0) * tl.sum(
                        tl.abs(q) * tl.abs(s_col_K)[None, :], axis=1) * tl.abs(scale)
                    # Small outward margin reduces rounding-boundary sensitivity;
                    # it is not an interval-arithmetic proof for GPU dot/exp.
                    ar_eps = ar_eps * 1.000001 + 1.0e-7
                    ar_upper, ar_candidate_lower = _ar_chunk_mass(ar_scores, ar_eps, ar_prefix_lower, ar_count)
                    if AR_POLICY == 2:
                        ar_skip_k = True
                    else:
                        ar_ok = (2.0 * ar_upper * ar_eps <= AR_TAU_K * ar_fraction)
                        ar_skip_k = tl.sum((qmask & ~ar_ok).to(tl.int32), axis=0) == 0
                    if ar_skip_k:
                        scores = ar_scores
                    else:
                        # This tl.load is *inside* the demand branch; masking
                        # already-loaded full-precision codes is not used.
                        ar_kl_byte = tl.load(KV_cache_ptr + ar_ka + D * GROUP // 4,
                                             mask=cmask[None, :], other=0).to(tl.int32)
                        ar_kl = ((ar_kl_byte >> ((cols[None, :] % 4) * 2)) & 3).to(tl.float32)
                        q_K = 4.0 * ar_kh + ar_kl
                        K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                        scores = tl.where(cmask[None, :], tl.dot(q, K_dg) * scale, -float("inf"))
                        ar_eps = tl.zeros([Q_PER_KV_PAD], tl.float32)
                else:
                    ar_kl_byte = tl.load(KV_cache_ptr + ar_ka + D * GROUP // 4,
                                         mask=cmask[None, :], other=0).to(tl.int32)
                    ar_kl = ((ar_kl_byte >> ((cols[None, :] % 4) * 2)) & 3).to(tl.float32)
                    q_K = 4.0 * ar_kh + ar_kl
                    K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                    scores = tl.where(cmask[None, :], tl.dot(q, K_dg) * scale, -float("inf"))

                if AR_POLICY >= 3:
                    ar_upper, ar_next_lower = _ar_chunk_mass(scores, ar_eps, ar_prefix_lower, ar_count)
                    ar_prefix_lower = ar_next_lower
                if AR_POLICY == 2:
                    ar_skip_v = True
                elif AR_POLICY == 4 or AR_POLICY == 5:
                    # Both residual and full-range V norms share ||s_col_V||_2.
                    # Cancelling this positive factor avoids a sqrt/reduction.
                    ar_v_range = tl.maximum(tl.abs(zp_V), tl.abs(zp_V + 15.0 * s_row_V))
                    ar_v_range = tl.max(tl.where(cmask, ar_v_range, 0.0), axis=0)
                    ar_v_res = 1.5 * tl.max(tl.abs(s_row_V), axis=0)
                    ar_ratio = tl.where(ar_v_range > 0, ar_v_res / tl.maximum(ar_v_range, 1.0e-30), 0.0)
                    ar_ratio = ar_ratio * 1.000001
                    ar_ok_v = ar_upper * ar_ratio <= AR_TAU_V * ar_fraction
                    ar_skip_v = tl.sum((qmask & ~ar_ok_v).to(tl.int32), axis=0) == 0

                ar_va = tile_base + V_PACKED_OFFSET + cols[:, None] * (D // 4) + d_offs[None, :] // 4
                ar_vh_byte = tl.load(KV_cache_ptr + ar_va, mask=cmask[:, None], other=0).to(tl.int32)
                ar_vh = ((ar_vh_byte >> ((d_offs[None, :] % 4) * 2)) & 3).to(tl.float32)
                if ar_skip_v:
                    q_V = 4.0 * ar_vh + 1.5
                else:
                    ar_vl_byte = tl.load(KV_cache_ptr + ar_va + D * GROUP // 4,
                                         mask=cmask[:, None], other=0).to(tl.int32)
                    ar_vl = ((ar_vl_byte >> ((d_offs[None, :] % 4) * 2)) & 3).to(tl.float32)
                    q_V = 4.0 * ar_vh + ar_vl
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]
'''


STATS_CHUNK = r'''
            if AR_COLLECT_STATS:
                ar_n = tl.sum(cmask.to(tl.int32), axis=0)
                if pool_slot >= 0:
                    ar_stat_pool += ar_n * D * 4
                else:
                    ar_first = tl.min(tl.where(cmask, cols, GROUP), axis=0)
                    ar_last = tl.max(tl.where(cmask, cols, -1), axis=0)
                    ar_k2 = tl.where(ar_n > 0, (ar_last // 4 - ar_first // 4 + 1) * D, 0)
                    ar_k4 = tl.where(ar_n > 0, (ar_last // 2 - ar_first // 2 + 1) * D, 0)
                    ar_stat_quant += ar_n
                    if AR_POLICY == 0:
                        # Parent packed loads have no token mask; account for
                        # their entire chunk even at a sliding-window boundary.
                        ar_stat_ka += BLOCK_N * D // 2
                        ar_stat_va += BLOCK_N * D // 2
                        ar_stat_scale += BLOCK_N * 6
                    else:
                        ar_stat_ka += ar_k2
                        ar_stat_kr += tl.where(ar_skip_k, 0, ar_k2)
                        ar_stat_va += ar_n * D // 4
                        ar_stat_vr += tl.where(ar_skip_v, 0, ar_n * D // 4)
                        ar_stat_scale += ar_n * 6
                    ar_stat_base += BLOCK_N * D
                    ar_stat_kskip += tl.where((ar_n > 0) & ar_skip_k, 1, 0)
                    ar_stat_vskip += tl.where((ar_n > 0) & ar_skip_v, 1, 0)
                    ar_stat_chunks += (ar_n > 0).to(tl.int32)
                ar_stat_allchunks += (ar_n > 0).to(tl.int32)
'''

STATS_INITIALIZE = '''    ar_stat_quant = 0
    ar_stat_ka = 0
    ar_stat_kr = 0
    ar_stat_va = 0
    ar_stat_vr = 0
    ar_stat_pool = 0
    ar_stat_scale = 0
    ar_stat_base = 0
    ar_stat_kskip = 0
    ar_stat_vskip = 0
    ar_stat_chunks = 0
    ar_stat_allchunks = 0
'''

STATS_FIELDS = ("quantized_tokens", "key_anchor_code_bytes", "key_residual_code_bytes",
                "value_anchor_code_bytes", "value_residual_code_bytes", "pool_kv_bytes",
                "scale_logical_bytes", "native_quantized_code_bytes", "key_skipped_chunks",
                "value_skipped_chunks", "quantized_chunks", "all_chunks")


def stats_finish(single):
    index = "(b * tl.num_programs(1) + hk)" if single else "((b * tl.num_programs(1) + hk) * NUM_KV_SPLITS + split)"
    variables = ("quant", "ka", "kr", "va", "vr", "pool", "scale", "base", "kskip", "vskip", "chunks", "allchunks")
    return "\n    if AR_COLLECT_STATS:\n" + "".join(
        f"        tl.store(AR_STATS_PTR + {index} * 12 + {i}, ar_stat_{name})\n"
        for i, name in enumerate(variables))

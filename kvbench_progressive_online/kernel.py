# SPDX-License-Identifier: Apache-2.0
"""Triton snippets for genuine fused progressive decode (no Python per-token loop).

K and V code layout is high2|bit1|bit0, each plane a contiguous tile region.
The adaptive branch is heuristic: approximate, prefix-relative chunk mass and
query/scale RMS are NOT a certified output-error bound. There is intentionally
no predeclared byte budget; read-depth is selected anew per decode chunk.
"""

FULL = r'''
                # ProgKV high2 + bit1 + bit0; all K/V source bits read.
                s_row_K = tl.load(ku16 + (K_S_ROW_OFFSET // 2) + cols,
                                  mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                s_row_V = tl.load(ku16 + (V_S_ROW_OFFSET // 2) + cols,
                                  mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                zp_V = tl.load(ku16 + (V_ZP_OFFSET // 2) + cols,
                               mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                tl.static_assert(K_BITS == 4 and V_BITS == 4)
                tl.static_assert(D % 8 == 0 and GROUP % 8 == 0)
                ar_skip_k = False
                ar_skip_v = False
                ar_eps = tl.zeros([Q_PER_KV_PAD], tl.float32)
                ar_count = tl.sum(cmask.to(tl.int32), axis=0)
                ar_k_hi_addr = tile_base + K_PACKED_OFFSET + d_offs[:, None] * (GROUP // 4) + cols[None, :] // 4
                ar_k_bit_addr = tile_base + K_PACKED_OFFSET + D * GROUP // 4 + d_offs[:, None] * (GROUP // 8) + cols[None, :] // 8
                ar_kh_b = tl.load(KV_cache_ptr + ar_k_hi_addr, mask=cmask[None, :], other=0).to(tl.int32)
                ar_k1_b = tl.load(KV_cache_ptr + ar_k_bit_addr, mask=cmask[None, :], other=0).to(tl.int32)
                ar_k0_b = tl.load(KV_cache_ptr + ar_k_bit_addr + D * GROUP // 8, mask=cmask[None, :], other=0).to(tl.int32)
                ar_kh = ((ar_kh_b >> (2 * (cols[None, :] % 4))) & 3).to(tl.float32)
                ar_k1 = ((ar_k1_b >> (cols[None, :] % 8)) & 1).to(tl.float32)
                ar_k0 = ((ar_k0_b >> (cols[None, :] % 8)) & 1).to(tl.float32)
                q_K = 4.0 * ar_kh + 2.0 * ar_k1 + ar_k0
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                scores = tl.where(cmask[None, :], tl.dot(q, K_dg) * scale, -float('inf'))
                ar_v_hi_addr = tile_base + V_PACKED_OFFSET + cols[:, None] * (D // 4) + d_offs[None, :] // 4
                ar_v_bit_addr = tile_base + V_PACKED_OFFSET + D * GROUP // 4 + cols[:, None] * (D // 8) + d_offs[None, :] // 8
                ar_vh_b = tl.load(KV_cache_ptr + ar_v_hi_addr, mask=cmask[:, None], other=0).to(tl.int32)
                ar_v1_b = tl.load(KV_cache_ptr + ar_v_bit_addr, mask=cmask[:, None], other=0).to(tl.int32)
                ar_v0_b = tl.load(KV_cache_ptr + ar_v_bit_addr + D * GROUP // 8, mask=cmask[:, None], other=0).to(tl.int32)
                ar_vh = ((ar_vh_b >> (2 * (d_offs[None, :] % 4))) & 3).to(tl.float32)
                ar_v1 = ((ar_v1_b >> (d_offs[None, :] % 8)) & 1).to(tl.float32)
                ar_v0 = ((ar_v0_b >> (d_offs[None, :] % 8)) & 1).to(tl.float32)
                q_V = 4.0 * ar_vh + 2.0 * ar_v1 + ar_v0
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]
'''

# PROG_K_2 / PROG_K_3 / PROG_V_2 / PROG_V_3 are compile-time constants
# placed at the top of each JIT kernel by transform.py. Unlike ReadKV, there
# is NO global B and NO second full-K scan; cost depends on per-chunk decisions.
ADAPTIVE = r'''
                s_row_K = tl.load(ku16 + (K_S_ROW_OFFSET // 2) + cols,
                                  mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                s_row_V = tl.load(ku16 + (V_S_ROW_OFFSET // 2) + cols,
                                  mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                zp_V = tl.load(ku16 + (V_ZP_OFFSET // 2) + cols,
                               mask=cmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
                tl.static_assert(K_BITS == 4 and V_BITS == 4)
                tl.static_assert(D % 8 == 0 and GROUP % 8 == 0)
                ar_skip_k = False
                ar_skip_v = False
                ar_eps = tl.zeros([Q_PER_KV_PAD], tl.float32)
                ar_count = tl.sum(cmask.to(tl.int32), axis=0)
                ar_k_hi_addr = tile_base + K_PACKED_OFFSET + d_offs[:, None] * (GROUP // 4) + cols[None, :] // 4
                ar_k_bit_addr = tile_base + K_PACKED_OFFSET + D * GROUP // 4 + d_offs[:, None] * (GROUP // 8) + cols[None, :] // 8
                ar_kh_b = tl.load(KV_cache_ptr + ar_k_hi_addr, mask=cmask[None, :], other=0).to(tl.int32)
                ar_kh = ((ar_kh_b >> (2 * (cols[None, :] % 4))) & 3).to(tl.float32)
                q_K = 4.0 * ar_kh + 1.5
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                ar_scores_2 = tl.where(cmask[None, :], tl.dot(q, K_dg) * scale, -float('inf'))
                # Low-cost RMS-style QUERY SENSITIVITY (not a worst-case bound).
                ar_qscale = q.to(tl.float32) * s_col_K[None, :]
                ar_rms_k = tl.sqrt(tl.sum(ar_qscale * ar_qscale, axis=1) + 1.0e-12)
                ar_rms_k = ar_rms_k * tl.max(tl.abs(s_row_K), axis=0) * tl.abs(scale)
                ar_mass_2, _ = _ar_chunk_mass(ar_scores_2, ar_eps, ar_prefix_lower, ar_count)
                ar_risk_k2 = tl.max(tl.where(qmask, ar_mass_2 * (1.5 * ar_rms_k), 0.0), axis=0)
                if ar_risk_k2 <= PROG_K_2:
                    scores = ar_scores_2
                    ar_skip_k = True
                else:
                    ar_k1_b = tl.load(KV_cache_ptr + ar_k_bit_addr,
                                     mask=cmask[None, :], other=0).to(tl.int32)
                    ar_k1 = ((ar_k1_b >> (cols[None, :] % 8)) & 1).to(tl.float32)
                    q_K = 4.0 * ar_kh + 2.0 * ar_k1 + 0.5
                    K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                    ar_scores_3 = tl.where(cmask[None, :], tl.dot(q, K_dg) * scale, -float('inf'))
                    ar_mass_3, _ = _ar_chunk_mass(ar_scores_3, ar_eps, ar_prefix_lower, ar_count)
                    ar_risk_k3 = tl.max(tl.where(qmask, ar_mass_3 * (0.5 * ar_rms_k), 0.0), axis=0)
                    if ar_risk_k3 <= PROG_K_3:
                        scores = ar_scores_3
                    else:
                        ar_k0_b = tl.load(KV_cache_ptr + ar_k_bit_addr + D * GROUP // 8,
                                         mask=cmask[None, :], other=0).to(tl.int32)
                        ar_k0 = ((ar_k0_b >> (cols[None, :] % 8)) & 1).to(tl.float32)
                        q_K = 4.0 * ar_kh + 2.0 * ar_k1 + ar_k0
                        K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                        scores = tl.where(cmask[None, :], tl.dot(q, K_dg) * scale, -float('inf'))
                # Value depth uses already-computed selected K logits. This is
                # an online, streaming estimation, NOT normalized full attention.
                ar_mass_v, _ = _ar_chunk_mass(scores, ar_eps, ar_prefix_lower, ar_count)
                ar_mass_v = tl.max(tl.where(qmask, ar_mass_v, 0.0), axis=0)
                ar_v_range = tl.maximum(tl.abs(zp_V), tl.abs(zp_V + 15.0 * s_row_V))
                ar_v_range = tl.max(tl.where(cmask, ar_v_range, 0.0), axis=0)
                ar_v_s = tl.max(tl.abs(s_row_V), axis=0)
                ar_v_ratio = tl.where(ar_v_range > 0, ar_v_s / tl.maximum(ar_v_range, 1.0e-30), 0.0)
                ar_v_hi_addr = tile_base + V_PACKED_OFFSET + cols[:, None] * (D // 4) + d_offs[None, :] // 4
                ar_v_bit_addr = tile_base + V_PACKED_OFFSET + D * GROUP // 4 + cols[:, None] * (D // 8) + d_offs[None, :] // 8
                ar_vh_b = tl.load(KV_cache_ptr + ar_v_hi_addr, mask=cmask[:, None], other=0).to(tl.int32)
                ar_vh = ((ar_vh_b >> (2 * (d_offs[None, :] % 4))) & 3).to(tl.float32)
                if ar_mass_v * (1.5 * ar_v_ratio) <= PROG_V_2:
                    q_V = 4.0 * ar_vh + 1.5
                    ar_skip_v = True
                else:
                    ar_v1_b = tl.load(KV_cache_ptr + ar_v_bit_addr,
                                     mask=cmask[:, None], other=0).to(tl.int32)
                    ar_v1 = ((ar_v1_b >> (d_offs[None, :] % 8)) & 1).to(tl.float32)
                    if ar_mass_v * (0.5 * ar_v_ratio) <= PROG_V_3:
                        q_V = 4.0 * ar_vh + 2.0 * ar_v1 + 0.5
                    else:
                        ar_v0_b = tl.load(KV_cache_ptr + ar_v_bit_addr + D * GROUP // 8,
                                         mask=cmask[:, None], other=0).to(tl.int32)
                        ar_v0 = ((ar_v0_b >> (d_offs[None, :] % 8)) & 1).to(tl.float32)
                        q_V = 4.0 * ar_vh + 2.0 * ar_v1 + ar_v0
                # Advance the single-pass prefix estimate AFTER this chunk.
                ar_unused, ar_prefix_lower = _ar_chunk_mass(scores, ar_eps, ar_prefix_lower, ar_count)
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]
'''

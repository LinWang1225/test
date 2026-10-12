# SPDX-License-Identifier: Apache-2.0
"""Hash-pinned transformations of every KVarN packed-cache consumer/writer."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys
import tempfile

from .kernel import JIT_HELPERS, QUANTIZED_CHUNK, STATS_CHUNK, STATS_INITIALIZE, stats_finish

PARENT_COMMIT = "7586257f1c632e63187bfacbbe21ccb51540f7b3"
PARENT_SHA256 = "dbc73619e34f564438343bc13f6ca07d9ad2980a016a3fe23230e0c29e9d130a"
DECODE_PATH = "v1/attention/ops/triton_kvarn_decode.py"
STORE_PATH = "v1/attention/ops/kvarn_store.py"
TORCH_DECODE_PATH = "v1/attention/ops/kvarn_decode.py"
POLICIES = ("parent", "repacked_full", "anchor", "bound_k", "bound_v", "bound_kv")
COMMON_FIXES = ("finite origin for fully masked online-softmax chunks",
                "zero padded single-stage rows", "zero empty split combinations",
                "define legacy dequant helper's missing K_BITS/V_BITS as explicit INT4 defaults")
_PINS = {
    DECODE_PATH: PARENT_SHA256,
    STORE_PATH: "7bc98ba130ad84c6e9bf482d03fd9faf3d6595ce55b7a0b6ecf09f40ccf0ed5d",
    TORCH_DECODE_PATH: "cfb2bf1a70883b0df6d4fbdde13c82afeb98415bd54bf8a2aad0b7425f355653",
    "v1/attention/backends/kvarn_attn.py": "1d31d9839fb57da096c8f8a1f42a485470c0b1b4f9990059adbdd7e376450575",
    "model_executor/layers/quantization/kvarn/config.py": "ece82a7721276a5c11fc3c56fb5d88dda39580f310cdff32bfa1aee857e7b7d3",
    "v1/attention/ops/triton_kvarn_sinkhorn.py": "122e7b2dc206413877b0d622ce3892ec65d3ec27b00625a02d9726bf1832b8a3",
}


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_expected_pins():
    return dict(_PINS)


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned source anchor occurs {text.count(old)} times: {old[:100]!r}")
    return text.replace(old, new, 1)


def _functions(text, replacements):
    lines = text.splitlines(keepends=True)
    edits = []
    found = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in replacements:
            start, end = node.lineno - 1, node.end_lineno
            old = "".join(lines[start:end])
            edits.append((start, end, replacements[node.name](old)))
            found.add(node.name)
    if found != set(replacements):
        raise ValueError(f"Missing pinned functions: {set(replacements) - found}")
    for start, end, new in sorted(edits, reverse=True):
        lines[start:end] = [new]
    result = "".join(lines)
    ast.parse(result)
    return result


def _safe_online(body):
    body = _replace_once(body, "            p = tl.exp(scores - m_new[:, None])",
                         "            m_safe = tl.where(m_new == -float('inf'), 0.0, m_new)\n"
                         "            p = tl.exp(scores - m_safe[:, None])")
    return _replace_once(body, "            alpha = tl.exp(m_i - m_new)",
                         "            alpha = tl.exp(m_i - m_safe)")


def _combine(body):
    body = _replace_once(body, "    w = tl.exp(lse - g)",
                         "    g_safe = tl.where(g == -float('inf'), 0.0, g)\n"
                         "    w = tl.exp(lse - g_safe)")
    return _replace_once(body, "out = tl.sum(w[:, None] * O, axis=0) / denom",
                         "out = tl.sum(w[:, None] * O, axis=0) / tl.where(denom > 0, denom, 1.0)")


def _consumer(body, cols):
    """Every full-read Triton materializer/verify consumer decodes both planes."""
    for side in ("k", "v"):
        # This span is a single simple assignment followed by load and unpack.
        pattern = rf"(?m)^( +){side}_addrs = \([\s\S]*?^\1q_{side.upper()} = .*?$"
        matches = list(re.finditer(pattern, body))
        if len(matches) != 1:
            raise ValueError(f"Expected one {side} packed consumer span, got {len(matches)}")
        match = matches[0]
        indent = match.group(1)
        if side == "k":
            address = f"tile_base + K_PACKED_OFFSET + d_offs[:, None] * (GROUP // 4) + {cols}[None, :] // 4"
            shift = f"(({cols}[None, :] % 4) * 2)"
        else:
            address = f"tile_base + V_PACKED_OFFSET + {cols}[:, None] * (D // 4) + d_offs[None, :] // 4"
            shift = "((d_offs[None, :] % 4) * 2)"
        code = [f"{side}_addrs = {address}",
                f"ar_{side}_high = tl.load(KV_cache_ptr + {side}_addrs).to(tl.int32)",
                f"ar_{side}_low = tl.load(KV_cache_ptr + {side}_addrs + D * GROUP // 4).to(tl.int32)",
                f"q_{side.upper()} = (4 * ((ar_{side}_high >> {shift}) & 3) + ((ar_{side}_low >> {shift}) & 3)).to(tl.float32)"]
        body = body[:match.start()] + "\n".join(indent + line for line in code) + body[match.end():]
    return body


def _dequant_defaults(body):
    return _replace_once(body, "    V_ZP_OFFSET: tl.constexpr,\n):",
                         "    V_ZP_OFFSET: tl.constexpr,\n"
                         "    K_BITS: tl.constexpr = 4,\n"
                         "    V_BITS: tl.constexpr = 4,\n):")


def _decode(body, policy, epsilon, single):
    number = POLICIES.index(policy)
    tau_k = epsilon if policy == "bound_k" else epsilon / 2 if policy == "bound_kv" else 0.0
    tau_v = epsilon if policy == "bound_v" else epsilon / 2 if policy == "bound_kv" else 0.0
    body = _replace_once(body, "    VQ_INDIRECT: tl.constexpr,\n):",
                         "    VQ_INDIRECT: tl.constexpr,\n"
                         "    AR_STATS_PTR=None,\n"
                         "    AR_COLLECT_STATS: tl.constexpr = False,\n):")
    # Policy/budgets are source constants, not overrideable launch kwargs.
    body = _replace_once(body, "    b = tl.program_id(0)",
                         f"    AR_POLICY: tl.constexpr = {number}\n"
                         f"    AR_TAU_K: tl.constexpr = {tau_k!r}\n"
                         f"    AR_TAU_V: tl.constexpr = {tau_v!r}\n"
                         "    b = tl.program_id(0)")
    body = _replace_once(body, "    acc = tl.zeros([Q_PER_KV_PAD, D], dtype=tl.float32)",
                         "    acc = tl.zeros([Q_PER_KV_PAD, D], dtype=tl.float32)\n"
                         "    ar_prefix_lower = tl.full([Q_PER_KV_PAD], -float('inf'), tl.float32)\n"
                         "    ar_attention_tokens = seq_len\n"
                         "    if SLIDING_WINDOW > 0:\n"
                         "        ar_attention_tokens = tl.minimum(seq_len, SLIDING_WINDOW)\n" + STATS_INITIALIZE.rstrip())
    # Reject non-v1 dtype explicitly when a generated plane kernel is reached.
    if policy != "parent":
        body = _replace_once(body, "    b = tl.program_id(0)",
                             "    tl.static_assert(K_BITS == 4 and V_BITS == 4, 'AR v1 supports K4V4 only')\n"
                             "    b = tl.program_id(0)")
    # Counts have no effect when AR_COLLECT_STATS=False (formal timed kernels).
    body = _replace_once(body, "        for c0 in range(0, GROUP, BLOCK_N):",
                         "        if AR_COLLECT_STATS:\n"
                         "            ar_stat_scale += 6 * D\n"
                         "        for c0 in range(0, GROUP, BLOCK_N):")
    pool_marker = "            if pool_slot >= 0:\n"
    body = _replace_once(body, pool_marker,
                         "            ar_skip_k = False\n            ar_skip_v = False\n" + pool_marker)
    if policy != "parent":
        # The quantified branch is replaced as a whole; original pool arithmetic remains.
        start = body.index("            else:\n", body.index(pool_marker)) + len("            else:\n")
        end = body.index("\n            # scores[", start) if "\n            # scores[" in body[start:] else body.index("\n            scores = tl.dot(q, K_dg)", start)
        body = body[:start] + QUANTIZED_CHUNK.lstrip("\n").rstrip() + body[end:]
        # Define scores in BOTH sides of the same branch for Triton's SSA merge.
        body, pool_count = re.subn(r"(?m)^(                K_dg = tl\.trans\(Kc\).*)$",
                      r"\1\n"
                      "                scores = tl.where(cmask[None, :], tl.dot(q, K_dg) * scale, -float('inf'))\n"
                      "                if AR_POLICY >= 3:\n"
                      "                    ar_count = tl.sum(cmask.to(tl.int32), axis=0)\n"
                      "                    ar_pool_eps = tl.zeros([Q_PER_KV_PAD], tl.float32)\n"
                      "                    ar_pool_upper, ar_prefix_lower = _ar_chunk_mass(scores, ar_pool_eps, ar_prefix_lower, ar_count)",
                      body, count=1)
        if pool_count != 1:
            raise ValueError("Missing pool-side score definition")
        body = re.sub(r"(?m)^            scores = tl\.dot\(q, K_dg\).*\n            scores = tl\.where\(cmask\[None, :\], scores \* scale, -float\(\"inf\"\)\).*?$",
                      "            # scores were formed inside the pool/quantized branches.", body, count=1)
    body = _replace_once(body, "            m_new = tl.maximum(m_i, tl.max(scores, axis=1))",
                         STATS_CHUNK.lstrip("\n").rstrip() + "\n            m_new = tl.maximum(m_i, tl.max(scores, axis=1))")
    body = _safe_online(body)
    if single:
        body = _replace_once(body, "    if seq_len <= 0:\n        return\n",
                             "    if seq_len <= 0:\n"
                             "        pad_d = tl.arange(0, D)\n"
                             "        tl.store(Out_ptr + b * stride_o_b + (hq0 + qh)[:, None] * stride_o_h + pad_d[None, :],\n"
                             "                 0.0, mask=qmask[:, None])\n"
                             "        return\n")
        body = _replace_once(body, "out = (acc / l_i[:, None]).to(tl.float16)",
                             "out = (acc / tl.where(l_i > 0, l_i, 1.0)[:, None]).to(tl.float16)")
    body += stats_finish(single)
    ast.parse(body)
    return body


def _transform_decode(text, policy, epsilon):
    replacements = {
        "_kvarn_fused_decode_kernel": lambda body: _decode(body, policy, epsilon, True),
        "_kvarn_fused_decode_stage1": lambda body: _decode(body, policy, epsilon, False),
        "_kvarn_fused_decode_stage2": _combine,
        "_kvarn_fused_verify_stage1": _safe_online,
        "_kvarn_dequant_blocks_kernel": _dequant_defaults,
    }
    if policy != "parent":
        replacements["_kvarn_dequant_blocks_kernel"] = lambda body: _consumer(_dequant_defaults(body), "g_offs")
        replacements["_kvarn_build_packed_kv_kernel"] = lambda body: _consumer(body, "g_offs")
        replacements["_kvarn_fused_verify_stage1"] = lambda body: _safe_online(_consumer(body, "cols"))
    text = _functions(text, replacements)
    text = _replace_once(text, "@triton.jit\ndef _kvarn_dequant_blocks_kernel(",
                         JIT_HELPERS + "@triton.jit\ndef _kvarn_dequant_blocks_kernel(")
    text += (f"\nAR_POLICY_NAME = {policy!r}\nAR_EPSILON = {epsilon!r}\n"
             f"AR_PARENT_SHA256 = {PARENT_SHA256!r}\nAR_LAYOUT = {'native-int4' if policy == 'parent' else 'tile-high2-low2'!r}\n")
    return text


def transform_sources(sources, policy, epsilon=0.02):
    if policy not in POLICIES or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Unknown policy or invalid epsilon")
    if set(sources) != set(_PINS):
        raise ValueError(f"Expected all six reviewed source files, got {sorted(sources)}")
    for path, expected in _PINS.items():
        if sha_text(sources[path]) != expected:
            raise ValueError(f"Unreviewed parent source {path}: {sha_text(sources[path])}; expected {expected}")
    output = dict(sources)
    output[DECODE_PATH] = _transform_decode(sources[DECODE_PATH], policy, epsilon)
    if policy != "parent":
        output[STORE_PATH] = _functions(sources[STORE_PATH], {
            "_pack_4bit": lambda _: "def _pack_4bit(q: torch.Tensor) -> torch.Tensor:\n"
                                   "    from kvbench_residual.representation import pack_planes_torch\n"
                                   "    return pack_planes_torch(q)\n",
            "_pack_lowbit": lambda body: _replace_once(body, "    pack = 8 // bits\n",
                                   "    if bits == 4:\n"
                                   "        from kvbench_residual.representation import pack_planes_torch\n"
                                   "        return pack_planes_torch(q)\n"
                                   "    pack = 8 // bits\n"),
        })
        output[TORCH_DECODE_PATH] = _functions(sources[TORCH_DECODE_PATH], {
            "_unpack_4bit": lambda _: "def _unpack_4bit(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:\n"
                                   "    from kvbench_residual.representation import unpack_planes_torch\n"
                                   "    return unpack_planes_torch(packed, original_last_dim)\n",
        })
    for text in output.values():
        ast.parse(text)
    return output


def read_sources(source_root):
    root = Path(source_root).resolve()
    return {p: (root / p).read_text(encoding="utf-8") for p in _PINS}


def generated_receipt(sources, epsilon=0.02):
    return {policy: {p: sha_text(text) for p, text in transform_sources(sources, policy, epsilon).items()}
            for policy in POLICIES}


def write_overlay(vllm_root, policy, epsilon=0.02):
    root = Path(vllm_root).resolve()
    sources = read_sources(root)
    output = transform_sources(sources, policy, epsilon)
    for path, text in output.items():
        if text != sources[path]:
            (root / path).write_text(text, encoding="utf-8")
    return {"policy": policy, "epsilon": epsilon, "parent_pins": get_expected_pins(),
            "generated_sha256": {p: sha_text(text) for p, text in output.items()},
            "common_fixes": COMMON_FIXES}


def load_variant(source_root, policy, epsilon=0.02):
    """Load generated decode source from a unique real file for Triton inspection."""
    source = transform_sources(read_sources(source_root), policy, epsilon)[DECODE_PATH]
    digest = sha_text(source)
    name = f"kvbench_residual_generated_{policy}_{digest[:12]}"
    if name in sys.modules:
        return sys.modules[name]
    directory = Path(tempfile.mkdtemp(prefix="kvbench_residual_"))
    path = directory / (policy + ".py")
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module

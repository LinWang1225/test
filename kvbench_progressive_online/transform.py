# SPDX-License-Identifier: Apache-2.0
"""Strict source-to-source transformation on the pinned KVarN AR full-reader.

NOT a monkeypatch to the live package. Independent private vLLM runtime only.
All five packed consumers of the original full 4-bit code must be adapted:
(1) full-read dequant helper, (2) packed materializer, (3) dedicated verify,
(4) fused decode single, (5) fused decode split.
The store (torch writer) and torch unpack are also adapted.
"""
from __future__ import annotations
import ast
import re

from .kernel import FULL, ADAPTIVE

DECODE='v1/attention/ops/triton_kvarn_decode.py'
STORE='v1/attention/ops/kvarn_store.py'
TORCH_DECODE='v1/attention/ops/kvarn_decode.py'
ARMS=('parent','p211_full','p211_adaptive')
DEFAULT_THRESHOLDS={'k2':0.08,'k3':0.04,'v2':0.12,'v3':0.08}


def _fn_rewrite(source,updates):
    lines=source.splitlines(keepends=True)
    edits=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in updates:
            old=''.join(lines[node.lineno-1:node.end_lineno])
            edits.append((node.lineno-1,node.end_lineno,updates[node.name](old)))
    missing=set(updates)-{node.name for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    if missing:raise ValueError('Missing KVarN pinned functions: '+str(sorted(missing)))
    for begin,end,text in sorted(edits,reverse=True):lines[begin:end]=[text]
    result=''.join(lines)
    ast.parse(result)
    return result


def _full_consumer(body,side,cols):
    # Exactly the strings emitted by kvbench_residual.transform._consumer.
    high=side.upper()
    pattern=(rf'(?m)^(?P<ind> +)ar_{side}_high = tl.load\(KV_cache_ptr \+ {side}_addrs\).to\(tl.int32\)\n'
             rf'(?P=ind)ar_{side}_low = tl.load\(KV_cache_ptr \+ {side}_addrs \+ D \* GROUP // 4\).to\(tl.int32\)\n'
             rf'(?P=ind)q_{high} = .*?\.to\(tl.float32\)')
    found=list(re.finditer(pattern,body))
    if len(found)!=1:
        raise ValueError(f'Expected one exact {side} full-consumer span, got {len(found)}')
    m=found[0]; ind=m.group('ind')
    if side=='k':
        addr=f'tile_base + K_PACKED_OFFSET + D * GROUP // 4 + d_offs[:, None] * (GROUP // 8) + {cols}[None, :] // 8'
        shift=f'{cols}[None, :] % 8'
        highshift=f'2 * ({cols}[None, :] % 4)'
    else:
        addr=f'tile_base + V_PACKED_OFFSET + D * GROUP // 4 + {cols}[:, None] * (D // 8) + d_offs[None, :] // 8'
        shift='d_offs[None, :] % 8'
        highshift='2 * (d_offs[None, :] % 4)'
    repl=[f'ar_{side}_bit_addr = {addr}',
          f'ar_{side}_high = tl.load(KV_cache_ptr + {side}_addrs).to(tl.int32)',
          f'ar_{side}_bit1 = tl.load(KV_cache_ptr + ar_{side}_bit_addr).to(tl.int32)',
          f'ar_{side}_bit0 = tl.load(KV_cache_ptr + ar_{side}_bit_addr + D * GROUP // 8).to(tl.int32)',
          f'q_{high} = (4 * ((ar_{side}_high >> ({highshift})) & 3) + 2 * ((ar_{side}_bit1 >> ({shift})) & 1) + ((ar_{side}_bit0 >> ({shift})) & 1)).to(tl.float32)']
    return body[:m.start()]+('\n'.join(ind+x for x in repl))+body[m.end():]


def _fused(body,arm,thresholds):
    first='                # Separate 2-bit planes occupy disjoint halves of each code field.'
    last='                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]'
    if body.count(first)!=1 or body.count(last)!=1:
        raise ValueError('Expected exact original AR quantized-chunk landmarks')
    a=body.index(first);b=body.index(last,a)+len(last)
    body=body[:a]+((ADAPTIVE if arm=='p211_adaptive' else FULL).strip('\n'))+body[b:]
    if arm=='p211_adaptive':
        # Pool chunks must update the same single-pass selected-K prefix LSE.
        marker='                if AR_POLICY >= 3:'
        if body.count(marker)!=1:
            raise ValueError(f'Expected exactly one pool-prefix marker, got {body.count(marker)}')
        body=body.replace(marker,'                if True:',1)
        local='    AR_POLICY: tl.constexpr = 1\n'
        if body.count(local)!=1:raise ValueError('Unable to pin compile-time adaptive constants')
        constants=''.join(f'    PROG_{k.upper()}: tl.constexpr = {thresholds[k]!r}\n' for k in ('k2','k3','v2','v3'))
        # Triton local compile-time parameters are inserted before the program IDs.
        constants=constants.replace('PROG_K2','PROG_K_2').replace('PROG_K3','PROG_K_3').replace('PROG_V2','PROG_V_2').replace('PROG_V3','PROG_V_3')
        body=body.replace(local,local+constants,1)
    return body


def transform_dequant(original,arm,thresholds=None):
    if arm not in ('p211_full','p211_adaptive'):raise ValueError('Unexpected arm')
    thresholds=DEFAULT_THRESHOLDS if thresholds is None else thresholds
    for k in DEFAULT_THRESHOLDS:
        if k not in thresholds or not 0<=float(thresholds[k])<10:raise ValueError('Invalid risk threshold '+k)
    marker='\n# AR-KV provenance hook:'
    if marker not in original:raise ValueError('AR source must include its provenance hook')
    source=original
    sources={
        '_kvarn_dequant_blocks_kernel':lambda s:_full_consumer(_full_consumer(s,'k','g_offs'),'v','g_offs'),
        '_kvarn_build_packed_kv_kernel':lambda s:_full_consumer(_full_consumer(s,'k','g_offs'),'v','g_offs'),
        '_kvarn_fused_verify_stage1':lambda s:_full_consumer(_full_consumer(s,'k','cols'),'v','cols'),
        '_kvarn_fused_decode_kernel':lambda s:_fused(s,arm,thresholds),
        '_kvarn_fused_decode_stage1':lambda s:_fused(s,arm,thresholds),
    }
    source=_fn_rewrite(source,sources)
    # AR policy and hash stay bound to this private runtime through the updated
    # runtime.json; do NOT omit the provenance hook appended by the AR builder.
    return source


def transform_writer(original):
    old='from kvbench_residual.representation import pack_planes_torch'
    if original.count(old)!=2:raise ValueError('Expected two KVarN store packer imports')
    result=original.replace(old,'from kvbench_progressive_online.codec import pack_211_torch')
    if result.count('return pack_planes_torch(q)')!=2:
        raise ValueError('Expected exactly two 4-bit store uses')
    result=result.replace('return pack_planes_torch(q)','return pack_211_torch(q)')
    ast.parse(result)
    return result


def transform_torch_decode(original):
    old='from kvbench_residual.representation import unpack_planes_torch'
    if original.count(old)!=1:raise ValueError('Expected KVarN unpack source')
    result=original.replace(old,'from kvbench_progressive_online.codec import unpack_211_torch')
    if result.count('return unpack_planes_torch(packed, original_last_dim)')!=1:
        raise ValueError('Expected KVarN torch unpack call')
    result=result.replace('return unpack_planes_torch(packed, original_last_dim)',
                          'return unpack_211_torch(packed, original_last_dim)')
    ast.parse(result)
    return result

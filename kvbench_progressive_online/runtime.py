# SPDX-License-Identifier: Apache-2.0
"""Build hash-checked, independent on-line vLLM copies from pinned existing AR bridge.

Does not mutate installed KVarN, reuse old results, or bypass the validated
native hash pins. Compatible with kvbench_residual.serve/trace/bootstrap.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import shutil
from kvbench_residual.runtime import build_runtime,verify_runtime,sha_file,atomic_json,text_sha,trace_footer
from .transform import ARMS,DECODE,STORE,TORCH_DECODE,DEFAULT_THRESHOLDS,transform_dequant,transform_writer,transform_torch_decode


def build(out,arm,thresholds=None):
    if arm not in ARMS:raise ValueError('Unknown progressive arm')
    thresholds=DEFAULT_THRESHOLDS if thresholds is None else dict(thresholds)
    parent_policy='parent' if arm=='parent' else 'repacked_full'
    out=Path(out).resolve()
    original=build_runtime(out,parent_policy,epsilon=.02)
    if arm=='parent':
        original['progressive']={'arm':arm,'layout':'original KVarN K4V4','compiler_source':'original reviewed parent'}
        atomic_json(out/'runtime.json',original)
        verify_runtime(out,parent_policy,full=True,epsilon=.02)
        return original
    package=Path(__file__).parent
    shutil.copytree(package,out/'kvbench_progressive_online',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','tests'))
    for rel,fn in ((STORE,transform_writer),(TORCH_DECODE,transform_torch_decode),
                   (DECODE,lambda s:transform_dequant(s,arm,thresholds))):
        file=out/'vllm'/rel
        new=fn(file.read_text(encoding='utf-8'))
        ast.parse(new)
        file.write_text(new,encoding='utf-8')
        original['generated_sha256'][rel]=text_sha(new[:-len(trace_footer(parent_policy))]) if rel==DECODE else text_sha(new)
        original['executed_files_sha256'][rel]=sha_file(file)
    original['transformed_source_sha256']=original['generated_sha256'][DECODE]
    original['executed_source_sha256']=original['executed_files_sha256'][DECODE]
    original['representation']='2+1+1 separately loadable planes (high2|bit1|bit0), native INT4 full-read identical'
    original['progressive']={
        'schema':'kvbench.progressive.online.v1',
        'arm':arm,'thresholds':thresholds if arm=='p211_adaptive' else None,
        'selection_scope':'per KV head and token chunk; GQA query heads share depth',
        'read_precision_levels':[2,3,4],
        'quality_guarantee':'heuristic; NO deterministic or statistical attention-output error guarantee',
        'physical_memory':'same 4-bit K/V code payload plus unchanged original KVarN metadata/pool',
        'formal_attention_traffic_bytes':None,
        'implementation_note':'All native packed consumers and producer updated in pinned private copy'}
    original['files_sha256']={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*'))
                              if p.is_file() and not p.is_symlink() and p.name!='runtime.json'}
    atomic_json(out/'runtime.json',original)
    verify_runtime(out,parent_policy,full=True,epsilon=.02)
    return original


def verify(root,arm):
    parent_policy='parent' if arm=='parent' else 'repacked_full'
    r=verify_runtime(root,parent_policy,full=True,epsilon=.02)
    if r.get('progressive',{}).get('arm')!=arm:raise ValueError('Incorrect progressive policy runtime')
    return r

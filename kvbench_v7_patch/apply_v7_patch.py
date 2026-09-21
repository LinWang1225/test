#!/usr/bin/env python3
from pathlib import Path
import shutil, sys, time

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
target = root / "run_real.py"
if not target.is_file():
    raise SystemExit(f"找不到 {target}；请把本脚本指向 kvbench_real_v6 目录。")

text = target.read_text(encoding="utf-8")
backup = target.with_name(f"run_real.py.before_v7_{time.strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(target, backup)

def replace_once(old: str, new: str, label: str):
    global text
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: 期望唯一匹配，实际 {n}；停止，未覆盖原文件。备份：{backup}")
    text = text.replace(old, new, 1)

replace_once(
"""    p.add_argument('--max-preemptions',type=int,default=-1,help='-1: only record; otherwise pilot eligibility bound')
    p.add_argument('--resume',action='store_true');p.add_argument('--continue-on-error',action='store_true')
""",
"""    p.add_argument('--max-preemptions',type=int,default=-1,help='-1: only record; otherwise pilot eligibility bound')
    p.add_argument('--require-native-usage',action='store_true',
                   help='默认不因流式响应缺少 server usage 淘汰 pilot；开启后才要求原生 usage 完整')
    p.add_argument('--resume',action='store_true');p.add_argument('--continue-on-error',action='store_true')
""",
"add --require-native-usage")

replace_once(
"""    keys=['model','methods','datasets','gpu_util','kv_bytes','max_len','max_seqs','batch_tokens','eager',
          'math_max_tokens','human_max_tokens','output_limit_policy','thinking','temperature','top_p','top_k','seed','data_seed',
          'data_hashes','prefix_sha256','knobs']
""",
"""    keys=['model','methods','datasets','gpu_util','kv_bytes','max_len','max_seqs','batch_tokens','eager',
          'math_max_tokens','human_max_tokens','output_limit_policy','thinking','temperature','top_p','top_k','seed','data_seed',
          'require_native_usage','data_hashes','prefix_sha256','knobs']
""",
"compatibility")

replace_once(
"""                    p=dict(id=f'r{rep:02d}_{method}_{dataset}_apc{apc}_C{c}',repeat=rep,method=method,
                        dataset=dataset,apc=apc,concurrency=c,number=n,max_tokens=cap,request_max_tokens=cap,
                        output_limit_policy=('explicit_request_cap' if cap is not None else 'server_context_window_only'),
""",
"""                    p=dict(id=f'r{rep:02d}_{method}_{dataset}_apc{apc}_C{c}',repeat=rep,method=method,
                        dataset=dataset,apc=apc,concurrency=c,number=n,max_tokens=cap,request_max_tokens=cap,
                        max_tokens_transport=('explicit_int' if cap is not None else 'json_null_sentinel'),
                        output_limit_policy=('explicit_request_cap' if cap is not None else 'server_context_window_only'),
""",
"record null sentinel")

replace_once(
"""        # Natural generation protocol: by default omit max_tokens/min_tokens/ignore_eos/stop.
        # serve.sh already uses --generation-config vllm, so absent max_tokens falls back to
        # the remaining server context window (subject to platform limits).
        if p.get('max_tokens') is not None:
            req['max_tokens']=p['max_tokens']
""",
"""        # EvalScope perf v1.12.0 has a CLI default max_tokens=2048.  For complete
        # line_by_line JSON bodies it fills only MISSING keys (setdefault semantics).
        # Therefore an explicit JSON null is a transport sentinel, NOT an output cap:
        # it blocks EvalScope's 2048 default while vLLM receives max_tokens=None and
        # resolves the real ceiling from max_model_len - prompt_len / platform limits.
        if p.get('max_tokens') is not None:
            req['max_tokens']=p['max_tokens']
        else:
            req['max_tokens']=None
""",
"natural null sentinel")

replace_once(
"""    if stat.get('token_usage_missing',0):reasons.append('missing_native_usage')
""",
"""    if cfg.get('require_native_usage') and stat.get('token_usage_missing',0):
        reasons.append('missing_native_usage')
""",
"native usage eligibility")

replace_once(
"""    for p in plan:print(f"  {p['id']} n={p['number']} max_tokens={'unset' if p['max_tokens'] is None else p['max_tokens']} APC={p['apc']}")
""",
"""    for p in plan:
        mt='null-sentinel(no cap)' if p['max_tokens'] is None else str(p['max_tokens'])
        print(f"  {p['id']} n={p['number']} max_tokens={mt} APC={p['apc']}")
""",
"plan label")

target.write_text(text, encoding="utf-8")
print(f"Patched: {target}")
print(f"Backup : {backup}")
print("v7 natural-generation patch applied.")

#!/usr/bin/env python3
"""MATH-500/HumanEval -> natural generation with EvalScope perf -> lossless output export.
Uses existing serve.sh; does not install dependencies or execute generated code.
"""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,itertools,json,os,signal,subprocess,sys,time
from pathlib import Path
from typing import Any
from engine import (ServerManager,Telemetry,log,run_text,sha,port_busy,ensure_gpu_idle,identity,http_text)
from metrics import atomic_json,native_metrics,extract_server_log,sampled_telemetry
from outputs import export_run

DTYPES={'fp16':'auto','tq4':'turboquant_4bit_nc','tq3':'turboquant_3bit_nc',
        'kvarn':'kvarn_k4v2_g128','kvarn44':'kvarn_k4v4_g128','kvarn64':'kvarn_k4v2_g64'}
KNOBS=('CUDA_HOME','TRITON_PTXAS_PATH','LD_LIBRARY_PATH','KVARN_SINKHORN_ITERS','KVARN_SINK_TOKENS',
       'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS','VLLM_ATTENTION_BACKEND')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stage',choices=['pilot','full'],default='pilot')
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',choices=['math_500','humaneval'],default=['math_500','humaneval'])
    p.add_argument('--methods',nargs='+',choices=list(DTYPES),default=['fp16','tq4','kvarn'])
    p.add_argument('--concurrency',nargs='+',type=int,help='pilot 默认 2 4 8；full 默认 4 或采用 recommendations')
    p.add_argument('--recommendations',type=Path,help='先前 pilot/recommendations.json；只选择并发，不修改输出上限/精度')
    p.add_argument('--selection',choices=['common','per-method'],default='common')
    p.add_argument('--pilot-size',type=int,default=32);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--apc',nargs='+',type=int,choices=[0,1],default=[0])
    p.add_argument('--math-max-tokens',type=int,default=None,help='可选调试上限；默认不发送 max_tokens，由服务端上下文窗口决定')
    p.add_argument('--human-max-tokens',type=int,default=None,help='可选调试上限；默认不发送 max_tokens，由服务端上下文窗口决定')
    p.add_argument('--thinking',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--temperature',type=float,default=0.6);p.add_argument('--top-p',type=float,default=0.95)
    p.add_argument('--top-k',type=int,default=20);p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--data-seed',type=int,default=42)
    p.add_argument('--shared-prefix-file',type=Path,help='可选：额外 system 文本；此时明确标记 augmented，不是原版 benchmark')
    p.add_argument('--gpu',default='0');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--gpu-util',type=float,default=0.85);p.add_argument('--kv-bytes',type=int)
    p.add_argument('--max-len',type=int,default=32768);p.add_argument('--max-seqs',type=int,default=16)
    p.add_argument('--batch-tokens',type=int,default=1024)
    p.add_argument('--eager',action='store_true')
    p.add_argument('--warmup-requests',type=int,default=2,help='独立非匹配前缀的真实题预热，不纳入统计')
    p.add_argument('--startup-timeout',type=int,default=1800);p.add_argument('--request-timeout',type=int,default=3600)
    p.add_argument('--point-timeout',type=int,default=86400)
    p.add_argument('--sample-interval',type=float,default=2);p.add_argument('--settle-seconds',type=float,default=6)
    p.add_argument('--max-p95-ttft-ms',type=float);p.add_argument('--max-p95-tpot-ms',type=float)
    p.add_argument('--max-pilot-cap-fraction',type=float,default=None,help='可选：仅在显式设置请求级 max_tokens 时用于筛查；默认不用于并发选择')
    p.add_argument('--max-preemptions',type=int,default=-1,help='-1: only record; otherwise pilot eligibility bound')
    p.add_argument('--resume',action='store_true');p.add_argument('--continue-on-error',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def stable_order(rows,seed):
    return sorted(rows,key=lambda x:hashlib.sha256(f'{seed}:{x["case_id"]}'.encode()).digest())


def load_data(cfg):
    base=Path(cfg['data_dir']);manifest=json.loads((base/'manifest.json').read_text())
    rows={};provenance={}
    for name in cfg['datasets']:
        m=manifest['datasets'][name];path=base/m['file']
        if sha(path)!=m['sha256']:raise ValueError('冻结数据文件被修改：'+str(path))
        data=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(data)!=m['count'] or len({r['case_id'] for r in data})!=len(data):raise ValueError('题目数量或题号不一致')
        rows[name]=stable_order(data,cfg['data_seed']);provenance[name]=m
    return rows,provenance


def compatibility(cfg):
    keys=['model','methods','datasets','gpu_util','kv_bytes','max_len','max_seqs','batch_tokens','eager',
          'math_max_tokens','human_max_tokens','output_limit_policy','thinking','temperature','top_p','top_k','seed','data_seed',
          'data_hashes','prefix_sha256','knobs']
    return {k:cfg.get(k) for k in keys}


def plan_points(cfg,rows):
    rec=json.loads(Path(cfg['recommendations']).read_text()) if cfg.get('recommendations') else None
    if rec and rec['compatibility']!=compatibility(cfg):
        raise ValueError('pilot 与当前模型/数据/缓存预算/引擎/采样配置不匹配，请重新 pilot 或明确 --concurrency。')
    points=[];methods=cfg['methods']
    for rep in range(1,cfg['repeats']+1):
        offset=(rep-1)%len(methods)
        for method in methods[offset:]+methods[:offset]:
            for dataset,apc in itertools.product(cfg['datasets'],cfg['apc']):
                n=min(cfg['pilot_size'],len(rows[dataset])) if cfg['stage']=='pilot' else len(rows[dataset])
                levels=cfg['concurrency']
                if rec:
                    chosen=rec['choices'].get(f'{dataset}:apc{apc}',{})
                    c=chosen.get('common') if cfg['selection']=='common' else chosen.get('per_method',{}).get(method)
                    if not c:raise ValueError(f'{dataset} APC={apc} 没有可用 pilot 建议；检查失败/截断/SLO 原因。')
                    levels=[int(c)]
                for c in levels:
                    if c>n:raise ValueError(f'并发 {c} 大于本点题数 {n}；调大 pilot-size 或减小并发。')
                    if c>cfg['max_seqs']:raise ValueError('并发超过 --max-seqs，拒绝把引擎排队误作实测有效并发上限。')
                    cap=cfg['math_max_tokens'] if dataset=='math_500' else cfg['human_max_tokens']
                    p=dict(id=f'r{rep:02d}_{method}_{dataset}_apc{apc}_C{c}',repeat=rep,method=method,
                        dataset=dataset,apc=apc,concurrency=c,number=n,max_tokens=cap,request_max_tokens=cap,
                        output_limit_policy=('explicit_request_cap' if cap is not None else 'server_context_window_only'),
                        kv_dtype=DTYPES[method],kv_cache_memory_bytes_requested=cfg.get('kv_bytes'),
                        kv_pool_budget_source='explicit_bytes' if cfg.get('kv_bytes') else 'gpu_memory_utilization',
                        gpu_memory_utilization_requested=cfg['gpu_util'],block_size=64 if method=='kvarn64' else 128,
                        max_model_len=cfg['max_len'],max_num_seqs=cfg['max_seqs'],max_num_batched_tokens=cfg['batch_tokens'],
                        enable_prefix_caching=bool(apc),cache_start_protocol='no-target-preload' if apc else 'APC-off',
                        evalscope_random_prefix_length=None,independent_prefix_cache_bytes=None,
                        prefix_capacity_control='shared_vllm_KV_pool_no_independent_quota',
                        input_variant='augmented-system-prefix' if cfg['prefix_sha256'] else 'native-evalscope-prompt',
                        shared_prefix_file=cfg.get('shared_prefix_file'),shared_prefix_sha256=cfg['prefix_sha256'],
                        thinking=cfg['thinking'],temperature=cfg['temperature'],top_p=cfg['top_p'],top_k=cfg['top_k'],
                        seed_base=cfg['seed'],ignore_eos=False,min_tokens=None,stop=None,eager=cfg['eager'])
                    points.append(p)
    return points


def build_requests(data,p,cfg,tokenizer=None,warm=False):
    selected=data[:p['number']] if not warm else data[-cfg['warmup_requests']:]
    out=[]
    for i,row in enumerate(selected):
        messages=[dict(m) for m in row['messages']]
        if warm:
            # Unique leading system text prevents target-prefix preloading. Never inject test answers.
            messages.insert(0,{'role':'system','content':f'Independent warm-up {p["id"]} {i}. Solve the user task.'})
        elif cfg.get('prefix_text'):
            messages.insert(0,{'role':'system','content':cfg['prefix_text']})
        seed=int(hashlib.sha256(f'{cfg["seed"]}:{p["repeat"]}:{row["case_id"]}'.encode()).hexdigest()[:8],16)%(2**31-1)
        req={'model':'qwen-kvbench','messages':messages,'stream':True,'stream_options':{'include_usage':True},
             'n':1,'repetition_penalty':1.0,'frequency_penalty':0.0,'presence_penalty':0.0,
             'temperature':cfg['temperature'],'top_p':cfg['top_p'],'top_k':cfg['top_k'],'min_p':0.0,
             'seed':seed,'chat_template_kwargs':{'enable_thinking':cfg['thinking']}}
        # Natural generation protocol: by default omit max_tokens/min_tokens/ignore_eos/stop.
        # serve.sh already uses --generation-config vllm, so absent max_tokens falls back to
        # the remaining server context window (subject to platform limits).
        if p.get('max_tokens') is not None:
            req['max_tokens']=p['max_tokens']
        tok_count=None;prefix_count=None
        if tokenizer is not None:
            tok_count=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                                        enable_thinking=cfg['thinking']))
            if tok_count>=cfg['max_len']:
                raise ValueError(f'{row["case_id"]}: prompt {tok_count} >= window {cfg["max_len"]}; 不截断题目。')
            if p.get('max_tokens') is not None and tok_count+p['max_tokens']>cfg['max_len']:
                raise ValueError(f'{row["case_id"]}: prompt {tok_count} + explicit cap {p["max_tokens"]} > window {cfg["max_len"]}; 不截断题目。')
            if cfg.get('prefix_text') and not warm:
                prefix_count=len(tokenizer.encode(cfg['prefix_text'],add_special_tokens=False))
        out.append({**row,'request':req,'input_tokens_local':tok_count,'shared_prefix_tokens_local':prefix_count})
    return out


def eligible(stat,cfg):
    reasons=[]
    if not stat.get('measurement_complete'):reasons.append('incomplete_or_failed_requests')
    if stat.get('missing_finish_reason',0):reasons.append('missing_finish_reason')
    if stat.get('token_usage_missing',0):reasons.append('missing_native_usage')
    # A context-window 'length' termination is a model/workload outcome, not a concurrency failure.
    # Only enforce a cap fraction when the user explicitly configured a request-level cap threshold.
    if cfg.get('max_pilot_cap_fraction') is not None and (stat.get('length_capped_fraction') or 0)>cfg['max_pilot_cap_fraction']:
        reasons.append('too_many_explicit_length_capped_requests')
    for option,field in [('max_p95_ttft_ms','ttft_ms_p95'),('max_p95_tpot_ms','tpot_ms_p95')]:
        limit=cfg.get(option)
        if limit is not None and (stat.get(field) is None or stat[field]>limit):reasons.append(option)
    if cfg['max_preemptions']>=0 and (stat.get('preemptions_delta') is None or stat['preemptions_delta']>cfg['max_preemptions']):
        reasons.append('preemption_bound_or_missing_counter')
    return not reasons,reasons


def collect(root):
    records=[]
    for point in sorted((root/'points').glob('*')):
        states=sorted(point.glob('attempt*/state.json'))
        if not states:continue
        state=json.loads(states[-1].read_text());d=states[-1].parent
        row={**state['point'],'status':state['status'],'attempt_path':str(d),'error':state.get('error')}
        for f in ['stats.json','scores.summary.json']:
            if (d/f).exists():
                row.update({k:v for k,v in json.loads((d/f).read_text()).items() if not isinstance(v,(list,dict))})
        records.append(row)
    atomic_json(root/'summary.json',records)
    if records:
        keys=list(dict.fromkeys(k for r in records for k in r))
        with (root/'summary.csv.tmp').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
        (root/'summary.csv.tmp').replace(root/'summary.csv')
    return records


def recommend(root,cfg,records,info):
    choices={}
    for dataset,apc in itertools.product(cfg['datasets'],cfg['apc']):
        per={};safe={};evidence=[]
        for method in cfg['methods']:
            candidates=[r for r in records if r['dataset']==dataset and r['apc']==apc and r['method']==method
                        and r.get('pilot_eligible') and r.get('status')=='ok']
            # A level is eligible only when every planned repeat finished and passed screening.
            levels=[]
            for c in cfg['concurrency']:
                subset=[r for r in candidates if r['concurrency']==c]
                if len(subset)==cfg['repeats']:
                    levels.append((sum(r['rps'] for r in subset)/len(subset),c))
            safe[method]={c for _,c in levels}
            per[method]=sorted(levels,key=lambda x:(-x[0],x[1]))[0][1] if levels else None
            evidence.extend({'method':method,'concurrency':c,'mean_rps':r} for r,c in levels)
        common=set.intersection(*(safe[m] for m in cfg['methods'])) if safe else set()
        best_min=min(v for v in per.values() if v) if all(per.values()) else None
        common_c=max((c for c in common if best_min is not None and c<=best_min),default=min(common) if common else None)
        choices[f'{dataset}:apc{apc}']={'per_method':per,'common':common_c,'eligible_levels':{m:sorted(v) for m,v in safe.items()},
                                      'evidence':evidence,'warning':'Pilot-only empirical choice, not a proven capacity or latency guarantee.'}
    atomic_json(root/'recommendations.json',{'compatibility':compatibility(cfg),'source_identity':info,'choices':choices,
        'selection_objective':'successful requests/s subject to request completeness/SLO criteria; length/context-window terminations are recorded but do not disqualify concurrency by default; accuracy is graded offline',
        'slo_configured':cfg['max_p95_ttft_ms'] is not None or cfg['max_p95_tpot_ms'] is not None})


class NaturalRunner(ServerManager):
    def __init__(self,cfg,rows):
        super().__init__(cfg);self.rows=rows
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)

    def command(self,p,out,data,warm=False):
        return ['evalscope','perf','--url',self.base+'/v1/chat/completions','--api','openai','--api-key','EMPTY',
            '--model','qwen-kvbench','--dataset','line_by_line','--dataset-path',str(data),
            '--tokenizer-path',self.cfg['model'],'--parallel',str(min(p['concurrency'],self.cfg['warmup_requests']) if warm else p['concurrency']),
            '--number',str(self.cfg['warmup_requests'] if warm else p['number']),
            '--stream','--no-test-connection','--warmup-num','0','--seed',str(self.cfg['data_seed']),
            '--total-timeout',str(self.cfg['request_timeout']),'--outputs-dir',str(out)]

    def one(self,p,dest):
        state={'point':p,'status':'running','started_unix':time.time()};atomic_json(dest/'state.json',state)
        telemetry=None
        try:
            # APC points need a fresh service: no hidden warm cache from preceding data/method/C.
            if p['apc']:self.stop()
            self.start(p)
            state['server_folder']=str(self.server_folder);atomic_json(dest/'state.json',state)
            requests=build_requests(self.rows[p['dataset']],p,self.cfg,self.tokenizer)
            warm=build_requests(self.rows[p['dataset']],p,self.cfg,self.tokenizer,True)
            for name,data in [('measure',requests),('warmup',warm)]:
                with (dest/f'{name}.requests.jsonl').open('w',encoding='utf-8') as f:
                    for r in data:f.write(json.dumps(r['request'],ensure_ascii=False)+'\n')
                atomic_json(dest/f'{name}.manifest.json',data)
            self.invoke(self.command(p,dest/'warmup',dest/'warmup.requests.jsonl',True),dest/'warmup.log')
            warmdest=dest/'warmup_export';warmdest.mkdir()
            warmstat=export_run(dest/'warmup',warmdest,warm,self.cfg['thinking'])
            if not warmstat['measurement_complete']:raise RuntimeError('预热请求失败；未进入正式测量。')
            time.sleep(self.cfg['settle_seconds'])
            before=http_text(self.base+'/metrics');(dest/'metrics.before.txt').write_text(before)
            launcher=self.server_folder/'launcher.log'
            offset=launcher.stat().st_size
            state.update(measurement_start_unix=time.time());atomic_json(dest/'state.json',state)
            telemetry=Telemetry(dest,self.base,self.cfg['gpu'],self.cfg['sample_interval']);telemetry.start()
            error=None
            try:self.invoke(self.command(p,dest/'measured',dest/'measure.requests.jsonl'),dest/'measurement.log')
            except Exception as exc:error=exc
            telemetry.stop();telemetry=None
            state['measurement_end_unix']=time.time();atomic_json(dest/'state.json',state)
            time.sleep(self.cfg['settle_seconds'])
            try:after=http_text(self.base+'/metrics')
            except OSError:after=''
            (dest/'metrics.after.txt').write_text(after)
            with launcher.open('rb') as f:f.seek(offset);segment=f.read().decode('utf-8',errors='replace')
            (dest/'server.measurement.log').write_text(segment)
            startup=extract_server_log(launcher.read_text(errors='replace'))
            native=extract_server_log(segment)
            atomic_json(dest/'server_capacity.json',startup)
            atomic_json(dest/'native_prefix_stats.json',native)
            stat=export_run(dest/'measured',dest,requests,self.cfg['thinking'])
            stat.update(native_metrics(before,after));stat.update(sampled_telemetry(dest/'telemetry.jsonl'))
            stat.update(kv_capacity_tokens_single_gpu_log=startup['kv_capacity_tokens_single_gpu_log'],
                available_kv_memory_native_value=(startup['available_kv_memory_log_values'][-1]['value'] if startup['available_kv_memory_log_values'] else None),
                available_kv_memory_native_unit=(startup['available_kv_memory_log_values'][-1]['unit'] if startup['available_kv_memory_log_values'] else None),
                num_gpu_blocks_log=(startup['num_gpu_blocks_log_values'][-1] if startup['num_gpu_blocks_log_values'] else None),
                prefix_native_last_percent=native['prefix_native_last_percent'],prefix_native_last_is_run_aggregate=False,
                server_folder=str(self.server_folder),
                native_prefix_stats_path=str(dest/'native_prefix_stats.json'),
                cache_config_path=str(dest/'server_capacity.json'),
                request_manifest_path=str(dest/'measure.manifest.json'),
                shared_prefix_tokens_local=requests[0].get('shared_prefix_tokens_local') if requests else None)
            ok,reasons=eligible(stat,self.cfg);stat['pilot_eligible']=ok;stat['pilot_ineligible_reasons']=reasons
            atomic_json(dest/'stats.json',stat)
            if error:raise error
            if not stat['measurement_complete']:raise RuntimeError('请求失败/缺失/重复/无法导出；完整证据保留，不将此点标为完成。')
            state.update(status='ok',ended_unix=time.time());atomic_json(dest/'state.json',state)
            limit_label='context/length终止' if p.get('max_tokens') is None else '达到显式上限'
            log(f"完成 {p['id']}：{stat['succeeded']} 请求，RPS={stat['rps']:.4f}，输出均值={stat['output_tokens_avg']:.1f}，{limit_label}={stat['length_capped']}")
        except BaseException as exc:
            state.update(status='aborted' if isinstance(exc,KeyboardInterrupt) else 'failed',ended_unix=time.time(),error=f'{type(exc).__name__}: {exc}')
            atomic_json(dest/'state.json',state);raise
        finally:
            if telemetry:telemetry.stop()
            collect(self.root)


def configure(a):
    cfg=vars(a).copy()
    for k in ['data_dir','run_dir','recommendations','shared_prefix_file']:
        if cfg.get(k):cfg[k]=str(Path(cfg[k]).expanduser().resolve())
    cfg.update(root=os.environ.get('KVBENCH_ROOT',str(Path.home()/'kvbench-3090-cu126')),
        kit=os.environ.get('KIT',''),model=os.environ.get('MODEL',''),
        serve_env=os.environ.get('KVBENCH_SERVE_ENV','kvarn-serve'),eval_env=os.environ.get('KVBENCH_EVAL_ENV','evalscope-client'),
        preset='real',knobs={k:os.environ.get(k,'') for k in KNOBS})
    for k in ['kit','model']:cfg[k]=str(Path(cfg[k]).expanduser().resolve()) if cfg[k] else ''
    if not cfg['kit'] or not Path(cfg['kit'],'serve.sh').is_file():raise ValueError('请设置 KIT 为已有 serve.sh 所在目录。')
    serve_text=Path(cfg['kit'],'serve.sh').read_text(encoding='utf-8',errors='replace')
    if '--generation-config vllm' not in serve_text:
        raise ValueError('默认无请求级 max_tokens 需要服务端使用 --generation-config vllm，避免模型 generation_config.json 注入隐藏的 max_new_tokens 上限。请先修正 KIT/serve.sh。')
    if not cfg['model'] or not Path(cfg['model'],'config.json').is_file():raise ValueError('请设置 MODEL 为固定的本地 snapshot。')
    if cfg['recommendations'] and (cfg['stage']!='full' or cfg['concurrency']):raise ValueError('recommendations 仅用于 full，且不能与显式 concurrency 同用。')
    if not cfg['concurrency']:cfg['concurrency']=[2,4,8] if cfg['stage']=='pilot' else [4]
    for key in ['pilot_size','repeats','max_len','max_seqs','batch_tokens','warmup_requests','request_timeout','point_timeout','startup_timeout']:
        if cfg[key]<=0:raise ValueError(key+' 必须为正')
    for key in ['math_max_tokens','human_max_tokens']:
        if cfg.get(key) is not None and cfg[key]<=0:raise ValueError(key+' 若设置则必须为正')
    for key in ['concurrency','methods','datasets','apc']:
        if len(cfg[key])!=len(set(cfg[key])):raise ValueError(key+' 不允许重复')
    if cfg['concurrency']!=sorted(cfg['concurrency']):raise ValueError('concurrency 必须从小到大列出。')
    if min(cfg['concurrency'])<1 or cfg['sample_interval']<0.5 or cfg['settle_seconds']<0:raise ValueError('无效并发/采样间隔')
    if cfg.get('kv_bytes') is not None and cfg['kv_bytes']<=0:raise ValueError('kv-bytes 必须为正')
    if not 0<cfg['gpu_util']<1 or not 1<=cfg['port']<=65535 or ',' in cfg['gpu']:raise ValueError('无效 GPU/端口/利用率')
    if cfg.get('max_pilot_cap_fraction') is not None and not 0<=cfg['max_pilot_cap_fraction']<=1:
        raise ValueError('max-pilot-cap-fraction 必须在 [0,1]')
    if cfg['temperature']<0 or not 0<cfg['top_p']<=1:raise ValueError('无效采样参数')
    cfg['output_limit_policy']='explicit_request_cap' if (cfg.get('math_max_tokens') is not None or cfg.get('human_max_tokens') is not None) else 'server_context_window_only'
    cfg['prefix_text']=Path(cfg['shared_prefix_file']).read_text(encoding='utf-8') if cfg['shared_prefix_file'] else None
    cfg['prefix_sha256']=hashlib.sha256(cfg['prefix_text'].encode()).hexdigest() if cfg['prefix_text'] else None
    return cfg


def main():
    cfg=configure(parser().parse_args());rows,provenance=load_data(cfg)
    cfg['data_hashes']={k:v['sha256'] for k,v in provenance.items()}
    plan=plan_points(cfg,rows)
    log(f"{cfg['stage']}：{len(plan)} 个配置点，自然停止，结果 {cfg['run_dir']}")
    for p in plan:print(f"  {p['id']} n={p['number']} max_tokens={'unset' if p['max_tokens'] is None else p['max_tokens']} APC={p['apc']}")
    if cfg['dry_run']:return 0
    if os.environ.get('CONDA_DEFAULT_ENV')!=cfg['eval_env']:raise ValueError('使用 run_real.sh 或激活指定 EvalScope Conda 环境。')
    if port_busy(cfg['port']):raise ValueError('端口已有服务，请手动停止冒烟服务。')
    ensure_gpu_idle(cfg['gpu'])
    help_text=run_text(['evalscope','perf','--help'])
    for flag in ['--no-test-connection','--warmup-num','--dataset-path','--total-timeout','--outputs-dir']:
        if flag not in help_text:raise ValueError('当前 EvalScope 缺少 '+flag+'；本包不自动升级环境。')
    root=Path(cfg['run_dir']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lockdir=Path(cfg['root'])/'locks';lockdir.mkdir(parents=True,exist_ok=True)
    glock=(lockdir/f'gpu_{hashlib.sha256(cfg["gpu"].encode()).hexdigest()[:12]}.lock').open('a')
    fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=identity(cfg)
    if cfg['recommendations']:
        rec=json.loads(Path(cfg['recommendations']).read_text())
        old=rec.get('source_identity',{})
        for k in ['gpu','source','serve_packages','eval_packages','model_config_tokenizer_sha256','weight_file_sizes_mtime_ns']:
            if old.get(k)!=info.get(k):raise ValueError('pilot 环境标识不同：'+k)
    relevant={k:v for k,v in cfg.items() if k not in ('run_dir','resume','continue_on_error','dry_run')}
    fp=hashlib.sha256(json.dumps({'cfg':relevant,'identity':info},sort_keys=True).encode()).hexdigest()
    manifest=root/'suite.json'
    if manifest.exists():
        if not cfg['resume']:raise ValueError('结果目录已存在，请使用新目录或 --resume。')
        if json.loads(manifest.read_text())['fingerprint']!=fp:raise ValueError('续跑配置/文件/环境改变，拒绝混入旧结果。')
    else:
        if cfg['resume']:raise ValueError('没有可续跑的 suite.json。')
        atomic_json(manifest,{'fingerprint':fp,'config':cfg,'identity':info,'datasets':provenance,'plan':plan})
        (root/'evalscope-perf-help.txt').write_text(help_text)
    (root/'controller.pid').write_text(str(os.getpid()))
    runner=NaturalRunner(cfg,rows);failed=0;blocked=set()
    try:
        for p in plan:
            group=(p['repeat'],p['method'],p['dataset'],p['apc'])
            if cfg['stage']=='pilot' and group in blocked:
                log('上一级失败，跳过同组更高并发：'+p['id']);continue
            point=root/'points'/p['id'];point.mkdir(parents=True,exist_ok=True)
            states=sorted(point.glob('attempt*/state.json'))
            if states and json.loads(states[-1].read_text()).get('status')=='ok':
                log('跳过完成点：'+p['id']);continue
            dest=point/f'attempt{len(states)+1:03d}';dest.mkdir()
            try:runner.one(p,dest)
            except KeyboardInterrupt:return 130
            except Exception as exc:
                failed+=1;log('失败：'+str(exc));runner.stop();blocked.add(group)
                if not cfg['continue_on_error']:return 1
                if port_busy(cfg['port']):raise RuntimeError('失败后端口仍占用，停止以防误测。')
                ensure_gpu_idle(cfg['gpu'])
    finally:
        runner.stop()
        records=collect(root)
        if cfg['stage']=='pilot':recommend(root,cfg,records,info)
        (root/'controller.pid').unlink(missing_ok=True)
        lock.close();glock.close()
    log('生成与性能测量结束；对保存输出评分请另运行 score_saved.py，不再进行模型推理。')
    return 1 if failed else 0


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError) as exc:
        log(f'ERROR: {exc}');raise SystemExit(2)

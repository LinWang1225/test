# SPDX-License-Identifier: Apache-2.0
"""All LongBench tasks with exactly 20 matched cases per task, three REAL vLLM K4V4 GPU arms.

This is NOT a simulator: starts one GPU-backed vLLM service at a time, invokes
existing frozen-input SSE benchmark+LongBench scorer, measures actual TPOT,
TTFT, E2E, output throughput and GPU telemetry. No result is fabricated.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

from kvbench_longcontext.common import checked_data,read_jsonl,write_jsonl,atomic_json,sha_file,digest
from .runtime import build,verify
from .transform import ARMS,DEFAULT_THRESHOLDS

DEFAULT_UPSTREAM='results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json'


def _read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def _write_csv(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def freeze_cases(source,out,tasks,n):
    """No new tokenization or truncation. Immutable exact V3 cohort subsets."""
    manifest,datasets=checked_data(source)
    missing=[task for task in tasks if task not in datasets]
    if missing:raise ValueError('Requested task(s) absent from frozen V3 LongBench: '+', '.join(missing))
    selected={}
    for task in tasks:
        rows=datasets[task]
        if len(rows)<n:raise ValueError(f'{task} has only {len(rows)} eligible frozen cases, required {n}')
        selected[task]=rows[:n]     # already deterministic hash-shuffled by V3 data_seed
        if len({r['case_id'] for r in selected[task]})!=n:raise ValueError('Duplicate case IDs in '+task)
    out.mkdir(parents=True,exist_ok=False)
    m=json.loads(json.dumps(manifest))
    m['datasets']={task:dict(manifest['datasets'][task]) for task in tasks}
    m['settings']['sample_size']=n
    m['subset_provenance']={'source_manifest_sha256':sha_file(Path(source)/'manifest.json'),
         'source_dir':str(Path(source).resolve()),'selection':'first N in frozen hash-shuffled V3 order',
         'source_selected_count':{task:len(datasets[task]) for task in tasks},
         'selected_count':{task:n for task in tasks},
         'selected_case_ids':{task:[r['case_id'] for r in selected[task]] for task in tasks}}
    for task in tasks:
        rows=selected[task]
        srcinfo=m['datasets'][task]
        data_path=out/(task+'.jsonl')
        write_jsonl(data_path,rows)
        srcinfo.update(file=data_path.name,sha256=sha_file(data_path),selected_count=n,
                       selected_case_ids=[r['case_id'] for r in rows],
                       truncated_count=sum(bool(r['was_truncated']) for r in rows),
                       cohort=f'locked V3 subset of {n}, no new truncation')
        for k in ('raw_file',):
            rel=srcinfo.get(k)
            if rel:
                sourcefile=Path(source)/rel
                if not sourcefile.exists() or sha_file(sourcefile)!=srcinfo['raw_sha256']:
                    raise ValueError('Original LongBench raw source is missing/corrupt')
                destination=out/rel;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(sourcefile,destination)
    atomic_json(out/'manifest.json',m)
    checked,_=checked_data(out)
    if any(checked['datasets'][task]['selected_count']!=n for task in tasks):
        raise AssertionError('Frozen subset invalid')
    return selected


def _client_python(cfg):
    if os.environ.get('PROG_CLIENT_PYTHON'):
        candidate=Path(os.environ['PROG_CLIENT_PYTHON']).resolve()
    else:
        # Both preinstalled environments live under the same Miniconda envs/.
        envs=Path(sys.prefix).parent
        candidate=(envs/cfg['client_env']/'bin/python').resolve()
    if not candidate.is_file():
        raise ValueError('Client Python not found; set PROG_CLIENT_PYTHON to the existing evalscope-client/bin/python')
    return candidate


def _gpu_idle(gpu):
    from kvbench_longcontext.serve import run_text
    x=run_text(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'])
    if x.get('returncode')!=0:raise RuntimeError('GPU query failed: '+str(x))
    if x['stdout'] and 'No running processes found' not in x['stdout']:
        raise RuntimeError('GPU has other compute processes. Refusing to interrupt another experiment: '+x['stdout'])


def _run_logged(executable,args,path,env,cwd):
    path.parent.mkdir(parents=True,exist_ok=True)
    print('EXEC '+str(path)+': '+' '.join(map(str,[executable,*args])),flush=True)
    with path.open('w',encoding='utf-8') as f:
        r=subprocess.run([str(executable),*map(str,args)],env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    return r.returncode


def _start_server(args,arm,runtime,server_dir,logfile,env):
    parent_policy='parent' if arm=='parent' else 'repacked_full'
    cmd=[sys.executable,'-m','kvbench_residual.serve','--model',str(args.model),
         '--method','kvarn44','--policy',parent_policy,'--runtime',str(runtime),
         '--epsilon','0.02','--out',str(server_dir),'--port',str(args.port),
         '--gpu',str(args.gpu),'--max-model-len',str(args.max_model_len),
         '--max-seqs',str(args.max_seqs),'--batch-tokens',str(args.batch_tokens),
         '--gpu-util',str(args.gpu_util),'--server-seed',str(args.server_seed),
         '--startup-timeout',str(args.startup_timeout)]
    if args.eager:cmd.append('--eager')
    logfile.parent.mkdir(parents=True,exist_ok=True)
    logfile_handle=logfile.open('w',encoding='utf-8')
    proc=subprocess.Popen(cmd,cwd=args.repo,env=env,stdout=logfile_handle,
                          stderr=subprocess.STDOUT,start_new_session=True)
    launch=server_dir/'launch.json'
    deadline=time.monotonic()+args.startup_timeout+180
    try:
        while time.monotonic()<deadline:
            if proc.poll() is not None:
                raise RuntimeError(f'{arm} server launcher exited before ready; see {logfile}, {server_dir}/server.log')
            if launch.is_file():
                record=_read(launch)
                if record.get('native_verified') is True:
                    if record['config']['kv_dtype']!='kvarn_k4v4_g128':
                        raise RuntimeError('Actual GPU cache dtype is not original K4V4')
                    if record['residual']['executed_source_sha256']!=_read(runtime/'runtime.json')['executed_source_sha256']:
                        raise RuntimeError('Fused GPU decode source does not match runtime manifest')
                    clone=json.loads(json.dumps(record))
                    clone['config']['method']=arm
                    clone['progressive_online']={'arm':arm,'runtime':str(runtime),
                        'runtime_sha256':sha_file(runtime/'runtime.json'),
                        'server_launch_original_sha256':sha_file(launch),
                        'selection':_read(runtime/'runtime.json')['progressive']}
                    user_manifest=server_dir/'progressive.launch.json'
                    atomic_json(user_manifest,clone)
                    return proc,logfile_handle,user_manifest
            time.sleep(1)
        raise TimeoutError('Timed out waiting for native KVarN vLLM startup: '+arm)
    except BaseException:
        _stop_server(proc)
        logfile_handle.close()
        raise


def _stop_server(proc):
    if proc is None:return
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=65)
        except subprocess.TimeoutExpired:
            # Only own launcher process group. It is responsible for cleaning
            # its separately created vLLM engine group.
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=15)
                raise RuntimeError('Own server did not cleanly terminate; halt subsequent arms')


def _summary(out,arms,tasks):
    summary=[];scored={}
    for arm in arms:
        concurrency=_read(out/"settings.json")["concurrency"]
        for task in tasks:
            folder=out/"runs"/arm/f"r01_{task}_C{concurrency}"
            perf=folder/'performance.summary.json';quality=folder/'scores.summary.json'
            if not perf.is_file() or not quality.is_file():continue
            p,q=_read(perf),_read(quality)
            s={'arm':arm,'dataset':task,'expected':p.get('expected_tasks'),
               'measurement_complete':p.get('measurement_complete'),
               'scoring_complete':q.get('evaluation_complete'),
               'valid':p.get('valid_responses'),'task_score_0_100':q.get('task_score_0_100'),
               'tpot_ms_mean':(p.get('tpot_s') or {}).get('mean')*1000 if (p.get('tpot_s') or {}).get('mean') is not None else None,
               'tpot_ms_p95':(p.get('tpot_s') or {}).get('p95')*1000 if (p.get('tpot_s') or {}).get('p95') is not None else None,
               'ttft_ms_mean':(p.get('ttft_s') or {}).get('mean')*1000 if (p.get('ttft_s') or {}).get('mean') is not None else None,
               'e2e_s_mean':(p.get('e2e_s') or {}).get('mean'),
               'output_tokens_per_s':p.get('output_tokens_per_s'),
               'output_length_mean':(p.get('completion_tokens') or {}).get('mean'),
               'length_capped':p.get('length_capped'),
               'preemptions_delta':p.get('preemptions_delta'),
               'native_kv_capacity_tokens':p.get('native_kv_capacity_tokens')}
            telemetry=folder/'telemetry.jsonl'
            if telemetry.is_file():
                used=[]
                for line in telemetry.read_text(encoding='utf-8').splitlines():
                    try:
                        v=json.loads(line).get('gpu',{})
                        if v.get('returncode')==0:
                            used.append(float(v['stdout'].split(',')[0].strip()))
                    except (ValueError,KeyError,IndexError,TypeError):pass
                s['sampled_gpu_used_mib_max']=max(used) if used else None
            summary.append(s)
            if (folder/'scores.jsonl').exists():
                scored.setdefault(arm,{}).update({r['case_id']:r for r in read_jsonl(folder/'scores.jsonl')})
    if summary:
        baselines=[s for s in summary if s['arm']=='parent']
        if baselines:
            baseline=baselines[0]
            for r in summary:
                if r['tpot_ms_mean'] and baseline['tpot_ms_mean']:
                    r['tpot_improvement_pct_vs_parent']=100*(1-r['tpot_ms_mean']/baseline['tpot_ms_mean'])
                if r['output_tokens_per_s'] and baseline['output_tokens_per_s']:
                    r['throughput_gain_pct_vs_parent']=100*(r['output_tokens_per_s']/baseline['output_tokens_per_s']-1)
                if r['task_score_0_100'] is not None and baseline['task_score_0_100'] is not None:
                    r['task_score_delta_pp_vs_parent']=r['task_score_0_100']-baseline['task_score_0_100']
        _write_csv(out/'comparison.csv',summary)
        atomic_json(out/'comparison.json',summary)
    paired=[]
    if 'parent' in scored:
        for arm in arms:
            if arm=='parent' or arm not in scored:continue
            for case_id,base in scored['parent'].items():
                v=scored[arm].get(case_id)
                paired.append({'arm':arm,'case_id':case_id,'baseline_scored':base['scored'],
                               'candidate_scored':v['scored'] if v else False,
                               'baseline_score':base['score'],
                               'candidate_score':v['score'] if v else None,
                               'delta_score':v['score']-base['score'] if v and v['score'] is not None and base['score'] is not None else None})
    _write_csv(out/'paired_quality.csv',paired)
    lines=['# Qwen3-4B · LongBench 每个数据集 ×20：Progressive-KV 在线实验','',
           '性能来自 vLLM 正式在线 SSE 请求与 GPU Attention；不是离线 Oracle。',
           'TPOT 采用已有客户端定义 `(E2E-TTFT)/(completion_tokens-1)`，包含服务调度与传输开销。',
           '不提供 DRAM 字节实测或逐 Chunk GPU 计数；所有缓存统一为原生 K4V4 容量。','',
           '| 方法 | 评分/100 | TPOT ms | P95 TPOT ms | TTFT ms | tok/s | 输出长度 | GPU峰值MiB | 得分Δpp | TPOT改善 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x,pct=False):return 'NA' if x is None else f'{x:.2f}{"%" if pct else ""}'
    for s in summary:
        lines.append('| '+s['arm']+' | '+fmt(s['task_score_0_100'])+' | '+fmt(s['tpot_ms_mean'])+
                     ' | '+fmt(s['tpot_ms_p95'])+' | '+fmt(s['ttft_ms_mean'])+
                     ' | '+fmt(s['output_tokens_per_s'])+' | '+fmt(s['output_length_mean'])+
                     ' | '+fmt(s.get('sampled_gpu_used_mib_max'))+
                     ' | '+fmt(s.get('task_score_delta_pp_vs_parent'))+
                     ' | '+fmt(s.get('tpot_improvement_pct_vs_parent'),True)+' |')
    lines+=['','**重要限制：** Adaptive Controller 的风险评分是在线启发式，不是经过校准的误差上界；',
            '当前阶段是每个 LongBench 数据集 20 条请求的工程可行性/收益筛选，不具有统计非劣保证。',
            '`p211_full` 用于验证新的表示布局是否引入额外延迟；`p211_adaptive` 同时改变读取行为。',
            '两组 K4V4 完整码字与原始 native INT4 等价，但实时 FP16 kernel 舍入与采样可能导致输出轨迹不同。','']
    (out/'readout.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-suite',type=Path,default=Path(DEFAULT_UPSTREAM))
    p.add_argument('--out',type=Path)
    p.add_argument('--task',default='all',
                   help='LongBench dataset to run; default runs every dataset with --count cases each')
    p.add_argument('--count',type=int,default=20)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--gpu',default='0')
    p.add_argument('--port',type=int,default=18181)
    p.add_argument('--eager',action='store_true',help='Disable CUDA graphs; not the primary main-run setting')
    p.add_argument('--risk-k2',type=float,default=.08)
    p.add_argument('--risk-k3',type=float,default=.04)
    p.add_argument('--risk-v2',type=float,default=.12)
    p.add_argument('--risk-v3',type=float,default=.08)
    p.add_argument('--startup-timeout',type=int,default=1800)
    p.add_argument('--gpu-util',type=float,default=.85)
    p.add_argument('--plan',action='store_true',help='No GPU or server; show exact 20-case plan')
    a=p.parse_args(argv)
    if a.count!=20: p.error('This experiment freezes exactly 20 cases; changing count requires a separate protocol')
    if not (1<=a.concurrency<=a.count):p.error('Invalid concurrency')
    if len(set(a.arms))!=len(a.arms):p.error('Duplicate arms')
    if not a.gpu.isdigit() or not 1024<=a.port<=65535:p.error('Invalid GPU or port')
    if not (0<a.gpu_util<1):p.error('Invalid gpu memory util')
    thresholds={'k2':a.risk_k2,'k3':a.risk_k3,'v2':a.risk_v2,'v3':a.risk_v3}
    if any(not 0<=x<10 for x in thresholds.values()):p.error('Invalid thresholds')
    a.repo=Path(__file__).resolve().parents[1]
    suite_path=a.upstream_suite.resolve(strict=True)
    suite=_read(suite_path)
    if suite.get('schema')!='kvbench.longcontext.full_suite.v1' or suite.get('exit_code')!=0:
        raise ValueError('Pass the successful frozen V3 LongBench suite.json')
    cfg=suite['settings']
    a.model=Path(cfg['model']).resolve(strict=True)
    a.max_model_len=cfg['max_model_len'];a.max_seqs=cfg['max_seqs']
    a.batch_tokens=cfg['batch_tokens'];a.server_seed=cfg['server_seed']
    src=Path(cfg['data']).resolve(strict=True)
    _,datasets=checked_data(src)
    tasks=list(datasets) if a.task=='all' else [a.task]
    missing=[task for task in tasks if task not in datasets]
    if missing:raise ValueError('Requested task(s) absent in original frozen cohort: '+', '.join(missing))
    insufficient=[f'{task} ({len(datasets[task])} available)' for task in tasks if len(datasets[task])<a.count]
    if insufficient:raise ValueError('Not enough selected cases: '+', '.join(insufficient))
    subset={task:datasets[task][:a.count] for task in tasks}
    info={'schema':'kvbench.progressive.online.plan.v1','task':a.task,'count':a.count,
          'concurrency':a.concurrency,'arms':a.arms,'risk_thresholds':thresholds,
          'tasks':tasks,
          'source_suite':str(suite_path),'source_suite_sha256':sha_file(suite_path),
          'source_data_sha256':sha_file(src/'manifest.json'),
          'model':str(a.model),'model_config_sha256':sha_file(a.model/'config.json'),
          'case_ids':{task:[r['case_id'] for r in subset[task]] for task in tasks},
          'precision':'kvarn_k4v4_g128','cache_capacity':'all arms same original 4-bit code capacity',
          'generation':'same V3 sampling=0.6/top_p=0.95/top_k=20, thinking, max_tokens=null',
          'timing':'online, native vLLM GPU decode Attention (not offline replay)'}
    if a.plan:
        print(json.dumps(info,ensure_ascii=False,indent=2));return
    if not a.out:
        a.out=a.repo/'results/progressive_online'/('longbench20_each_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    a.out=a.out.expanduser().resolve()
    if a.out.exists():raise ValueError('Output must be a new directory, never overwrite an existing run')
    if Path(sys.prefix).name!=cfg['server_env']:
        raise ValueError('Activate original '+cfg['server_env']+' server environment')
    client=_client_python(cfg)
    _gpu_idle(a.gpu)
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',a.port))==0:
            raise RuntimeError('Target port is already occupied; will not stop existing services')
    a.out.mkdir(parents=True)
    atomic_json(a.out/'plan.json',info)
    atomic_json(a.out/'settings.json',{'concurrency':a.concurrency,'gpu':a.gpu,'eager':a.eager,'port':a.port})
    selected=freeze_cases(src,a.out/'data',tasks,a.count)
    if {task:[r['case_id'] for r in selected[task]] for task in tasks}!=info['case_ids']:
        raise AssertionError('Frozen request cohort changed during preparation')
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env.update(PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    state={'status':'running','arms':{arm:'pending' for arm in a.arms},'completed_arms':0}
    atomic_json(a.out/'status.json',state)
    try:
        for arm in a.arms:
            state['arms'][arm]='building_runtime';atomic_json(a.out/'status.json',state)
            print('=== '+arm+' building private vLLM runtime ===',flush=True)
            runtime=a.out/'runtimes'/arm
            build(runtime,arm,thresholds)
            verify(runtime,arm)
            proc=log=None
            try:
                state['arms'][arm]='starting_server';atomic_json(a.out/'status.json',state)
                proc,log,manifest=_start_server(a,arm,runtime,a.out/'servers'/arm,
                                               a.out/'logs'/(arm+'.server_launcher.log'),env)
                state['arms'][arm]='generating_gpu';atomic_json(a.out/'status.json',state)
                run=a.out/'runs'/arm
                runargs=['-m','kvbench_longcontext.run','--data-dir',a.out/'data',
                    '--out',run,'--server-manifest',manifest,'--concurrency',str(a.concurrency),
                    '--repeats','1','--temperature','0.6','--top-p','0.95','--top-k','20',
                    '--min-p','0','--seed',str(cfg['request_seed']),
                    '--request-timeout','0','--warmup','1','--warmup-per-point',str(min(2,a.concurrency)),
                    '--gpu',a.gpu]
                rc=_run_logged(client,runargs,a.out/'logs'/(arm+'.run.log'),env,a.repo)
                if rc:raise RuntimeError(f'{arm} GPU run exited {rc}; see logs/{arm}.run.log')
            finally:
                try:_stop_server(proc)
                finally:
                    if log is not None:log.close()
            if not (a.out/'runs'/arm/'run.json').is_file():
                raise RuntimeError(arm+' did not produce a formal run manifest')
            state['arms'][arm]='scoring';atomic_json(a.out/'status.json',state)
            for module in ('scoring','report'):
                prog=['-m','kvbench_longcontext.'+module,
                      '--run-dir',a.out/'runs'/arm,'--data-dir',a.out/'data'] if module=='scoring' else [
                      '-m','kvbench_longcontext.report','--run-dir',a.out/'runs'/arm,
                      '--out',a.out/'runs'/arm/'readout']
                rc=_run_logged(client,prog,a.out/'logs'/(arm+'.'+module+'.log'),env,a.repo)
                if rc:raise RuntimeError(f'{arm} {module} failed: {rc}, inspect log')
            for task in tasks:
                point=a.out/'runs'/arm/f'r01_{task}_C{a.concurrency}'
                if _read(point/'performance.summary.json')['measurement_complete'] is not True:
                    raise RuntimeError(arm+' has incomplete measured requests for '+task)
                if _read(point/'scores.summary.json')['evaluation_complete'] is not True:
                    raise RuntimeError(arm+' has incomplete scored responses for '+task)
            state['arms'][arm]='completed';state['completed_arms']+=1
            atomic_json(a.out/'status.json',state)
            _summary(a.out,a.arms,tasks)
        state['status']='completed';atomic_json(a.out/'status.json',state)
    except BaseException as e:
        state.update(status='failed',error=repr(e),traceback=traceback.format_exc())
        atomic_json(a.out/'status.json',state)
        _summary(a.out,a.arms,tasks)
        raise
    print('Complete. Results: '+str(a.out/'readout.md'),flush=True)


if __name__=='__main__':main()

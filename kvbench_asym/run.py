"""All-record equal-payload K/V study on isolated native-Parent decode traces.

A request is generated once, then all asymmetric masks use that SAME Q/KV.
Analysis is CPU-only while our GPU server is idle. Fresh temporary snapshots
are pruned only after a committed case result, unless --keep-traces is set.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import traceback
from types import SimpleNamespace
import uuid
from .common import (DEFAULT_SUITE, atomic_json, canonical, digest, implementation_hashes,
                     inside, jsonl, load_upstream, read_json, sha_file, write_csv)
from .inputs import LONG_TASKS, inventory, prepare
from .select import quota


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-suite',type=Path,default=Path(DEFAULT_SUITE))
    p.add_argument('--datasets',nargs='+',choices=('longbench_all','longbench_frozen','math_500','humaneval'),default=['longbench_all'])
    p.add_argument('--longbench-source',type=Path)
    p.add_argument('--real-data',type=Path,default=Path('data/real_v5'))
    p.add_argument('--out',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--overflow',choices=('head-tail','error'),default='head-tail')
    p.add_argument('--case-limit',type=int,default=0,help='Explicit pilot limit across all datasets; 0 means ALL')
    p.add_argument('--tasks', nargs='+', choices=LONG_TASKS,
                   help='Explicit LongBench task subset; requires --datasets longbench_all')
    p.add_argument('--cases-per-task', type=int, default=0,
                   help='Select this many records from each selected task; 0 means ALL')
    p.add_argument('--layers',nargs='+',type=int,help='Cache ordinals, default four across local architecture')
    p.add_argument('--decode-calls',nargs='+',type=int,default=[0,15,63])
    p.add_argument('--diagnostic-max-tokens',type=int,default=96)
    p.add_argument('--block-n',type=int,choices=(16,32,64),default=32)
    p.add_argument('--controller-tolerances',nargs='+',type=float,default=[],
                   help='Enable offline Query-aware controller rows for these error tolerances')
    p.add_argument('--controller-margin',type=float,default=1.0)
    p.add_argument('--controller-max-steps',type=int)
    p.add_argument('--totals',nargs='+',default=['1.0','1.1'],help='rho_K+rho_V, each in [0,2]')
    p.add_argument('--shares',nargs='+',default=['0','.25','.5','.75','1'],help='K priority within feasible range; include 0.5')
    p.add_argument('--methods',nargs='+',choices=('anchor_mass','staged_mass','random'),default=['anchor_mass','staged_mass','random'])
    p.add_argument('--mask-seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--gpu',default='0');p.add_argument('--port',type=int,default=18080)
    p.add_argument('--startup-timeout',type=int,default=1800)
    p.add_argument('--scratch-gib',type=float,default=4,help='Hard per-request snapshot upper bound')
    p.add_argument('--keep-traces',action='store_true',help='Retain new raw NPZ files (can use hundreds of GiB)')
    p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args(argv)
    import math
    if (a.case_limit<0 or a.cases_per_task<0 or
            (a.tasks and a.cases_per_task == 0) or
            (a.case_limit and a.cases_per_task) or
            a.bootstrap<0 or a.diagnostic_max_tokens<2 or
            a.startup_timeout<1 or any(x<0 for x in a.controller_tolerances) or
            not math.isfinite(a.controller_margin) or a.controller_margin<1 or
            (a.controller_max_steps is not None and a.controller_max_steps<0)):
        p.error('Invalid limits')
    if not math.isfinite(a.scratch_gib) or a.scratch_gib<=0:
        p.error('scratch-gib must be positive and finite')
    if not a.gpu.isdigit() or not 1<=a.port<=65535:
        p.error('Use a single numeric GPU and a valid port')
    if len(set(a.decode_calls)) != len(a.decode_calls) or not a.decode_calls or any(x<0 or x>=a.diagnostic_max_tokens-1 for x in a.decode_calls):
        p.error('Distinct decode calls must be in [0, diagnostic-max-tokens-2]')
    if len(set(a.methods))!=len(a.methods) or len(set(a.mask_seeds))!=len(a.mask_seeds) or any(x<0 for x in a.mask_seeds):
        p.error('Duplicate selectors/seeds or negative seed')
    for total in a.totals:
        for share in a.shares:quota(10,total,share)
    if .5 not in [float(x) for x in a.shares] or len(set(map(float,a.shares)))!=len(a.shares) or len(set(map(float,a.totals)))!=len(a.totals):
        p.error('Distinct shares/totals and the symmetric share 0.5 are required')
    if a.resume and not a.out:
        p.error('--resume requires the exact existing --out directory')
    return a


def case_valid(root, base):
    p=base/'done.json'
    if not p.exists():return False
    d=read_json(p)
    if d.get('status')!='completed' or d.get('schema')!='kvbench.asym.case.v1':
        raise ValueError('Invalid case receipt: '+str(p))
    at=inside(root,d['attempt'])
    for rel,expected in d['file_hashes'].items():
        q=inside(at,rel)
        if not q.is_file() or sha_file(q)!=expected:
            raise ValueError('Completed case artifact changed: '+str(q))
    return True


def prune_new_arrays(attempt):
    """Only NPZ files named and hashed by THIS attempt's trace metadata."""
    for p in sorted((attempt/'traces').glob('trace_*.json')):
        m=read_json(p)
        q=inside(p.parent,m['arrays_file'])
        if q.parent!=p.parent or q.suffix!='.npz':
            raise ValueError('Unsafe temporary array path')
        if q.exists():
            if sha_file(q)!=m['arrays_sha256']:
                raise ValueError('Temporary snapshot changed before pruning')
            q.unlink()
    atomic_json(attempt/'temporary_arrays_pruned.json',dict(pruned=True,
        note='Fresh snapshots removed after analysis commit; metadata/hashes/masks retained. Re-analysis requires --keep-traces.'))


def validate_capture_coverage(records,row,result,layers,calls):
    by_point={}
    for m in records:
        if m['request']['case_id']!=row['case_id'] or m['request']['dataset']!=row['dataset']:
            raise ValueError('Trace belongs to a different request')
        key=(m['cache_ordinal'],m['decode_call_index'])
        if key in by_point:raise ValueError('Duplicate trace for one layer/decode position')
        if key[0] not in layers or key[1] not in calls:
            raise ValueError('Unplanned capture point')
        by_point[key]=m
    cov=[]
    # completion token 0 comes from prefill; decode call k produces token k+1.
    for layer in layers:
        for step in calls:
            observed=(layer,step) in by_point
            should_exist=result['completion_tokens']>=step+2
            if should_exist and not observed:
                raise RuntimeError(f'Missing expected capture at cache ordinal {layer}, decode {step}; not a natural-EOS omission')
            cov.append(dict(dataset=row['dataset'],case_id=row['case_id'],cache_ordinal=layer,
                            decode_call_index=step,captured=observed,expected_given_output=should_exist,
                            reason='' if observed else 'generation_ended_before_capture_point',
                            completion_tokens=result['completion_tokens'],finish_reason=result['finish_reason']))
    return cov


def main(argv=None):
    a=parse(argv)
    suite_path,upstream=load_upstream(a.upstream_suite);s=upstream['settings']
    model=Path(s['model']).resolve(strict=True)
    cfg=read_json(model/'config.json')
    if (cfg.get('model_type'),cfg.get('num_attention_heads'),cfg.get('num_key_value_heads'),cfg.get('head_dim',cfg.get('hidden_size',0)//32)) != ('qwen3',32,8,128):
        raise ValueError('This source-pinned capture currently supports Qwen3-4B / K4V4 G128 only')
    nlayers=cfg.get('num_hidden_layers')
    if not isinstance(nlayers,int) or nlayers<1:raise ValueError('Invalid model layer count')
    layers=a.layers if a.layers is not None else sorted(set((0,nlayers//3,2*nlayers//3,nlayers-1)))
    if not layers or len(set(layers))!=len(layers) or any(x<0 or x>=nlayers for x in layers):
        raise ValueError('Invalid cache ordinals')
    if a.diagnostic_max_tokens>=s['max_model_len']:
        raise ValueError('Diagnostic prefix leaves no prompt space')
    entries=inventory(upstream,a.datasets,a.longbench_source,a.real_data)
    if a.tasks:
        if a.datasets != ['longbench_all']:
            raise ValueError('--tasks requires exactly --datasets longbench_all')
        selected_tasks=set(a.tasks)
        entries=[entry for entry in entries if entry['dataset'] in selected_tasks]
    from .runtime import check_dependencies
    deps=check_dependencies()
    config={k:v for k,v in vars(a).items() if k not in ('out','resume','plan','prepare_only','longbench_source','real_data','upstream_suite')}
    config.update(layers=layers,upstream_sha256=sha_file(suite_path),model=str(model),
                  model_config_sha256=sha_file(model/'config.json'),entries=entries,
                  dependency_hashes=deps,implementation_hashes=implementation_hashes())
    # Tokenizer state is part of resume identity; metadata files only, no weights loaded.
    from kvbench_longcontext.common import tokenizer_identity
    config['tokenizer']=tokenizer_identity(model)
    total=sum(e['count'] for e in entries)
    selected=(min(total,a.case_limit) if a.case_limit else
              sum(min(entry['count'],a.cases_per_task) for entry in entries)
              if a.cases_per_task else total)
    max_points=selected*len(layers)*len(a.decode_calls)
    # Upper bound at deployment window, without assuming compression of snapshots.
    per_trace=((s['max_model_len']+127)//128)*(8*17920+128*8*128*4)+65536
    plan=dict(schema='kvbench.asym.plan.v1',config=config,source_cases=total,selected_cases=selected,
              source_tasks=len(entries),planned_max_traces=max_points,all_kv_heads=8,
              tokenized=False,overflow=a.overflow,raw_storage_upper_bound_gib=per_trace*max_points/1024**3,
              protocol='C1 eager native Parent, sampled decode positions, fixed diagnostic prefix, no forced EOS override',
              scope='All-record sensitivity study, NOT an official full-generation quality/performance leaderboard',
              equal_flow='Exact selected logical KV code payload; not DRAM or total implementation traffic')
    if a.plan:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    out=(a.out or Path('results/asym_equal_bytes')/(datetime.now().strftime('full_%Y%m%d_%H%M%S')+f'_{os.getpid()}')).resolve()
    if a.resume:
        old=read_json(out/'plan.json')
        if canonical(old['config'])!=canonical(config):
            raise ValueError('Resume configuration/source/tokenizer/package hashes differ. Use the same arguments or a new output.')
    else:
        out.mkdir(parents=True,exist_ok=False);atomic_json(out/'plan.json',plan)
    state=dict(schema='kvbench.asym.run.v1',status='preparing',out=str(out),planned_cases=selected,
               gpu_capture_executed=False,selectors_in_generation=False,measured_dram_bytes=None,task_scores=None,tpot=None)
    atomic_json(out/'status.json',state)
    proc=None; control=out/'trace_control'
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        if (out/'inputs/manifest.json').exists():
            data_manifest=read_json(out/'inputs/manifest.json')
            if sha_file(out/'inputs/cases.jsonl')!=data_manifest['cases_sha256']:
                raise ValueError('Frozen prepared inputs changed')
        else:
            if (out/'inputs').exists():
                # A previous prepare crash is retained, never consumed as complete.
                (out/'inputs').rename(out/('incomplete_inputs_'+uuid.uuid4().hex[:8]))
            data_manifest=prepare(out/'inputs',entries,model,s['max_model_len'],a.diagnostic_max_tokens,
                                  s.get('thinking',True),a.overflow,a.case_limit,
                                  cases_per_task=a.cases_per_task)
        if data_manifest['cases']!=selected:raise ValueError('Prepared case count differs from plan')
        if a.prepare_only:
            state.update(status='prepared');atomic_json(out/'status.json',state);print(str(out));return
        with (out/'selftest.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent/'tests'),'-v'],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        pending=[]; completed=0
        # Hold only indices/keys here, not all long prompt strings.
        for row in jsonl(out/'inputs/cases.jsonl'):
            base=out/'cases'/row['case_key']
            if case_valid(out,base):
                completed+=1
                if not a.keep_traces:prune_new_arrays(inside(out,read_json(base/'done.json')['attempt']))
            else:pending.append(row['case_key'])
        from .report import report
        if not pending:
            report(out,a.bootstrap);state.update(status='completed',completed_cases=completed)
            atomic_json(out/'status.json',state);print(str(out));return
        from kvbench_longcontext.serve import launch_command,run_text
        from kvbench_residual.serve import stop_owned_group
        from kvbench_residual_diag.run import ready,request_one
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(('127.0.0.1',a.port))==0:raise RuntimeError('Port occupied; no existing process is stopped')
        gpu=run_text(['nvidia-smi','-i',a.gpu,'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'])
        if gpu.get('returncode')!=0:raise RuntimeError('Cannot inspect target GPU: '+str(gpu))
        if gpu['stdout'] and 'No running processes found' not in gpu['stdout']:
            raise RuntimeError('GPU already has compute processes; this experiment does not stop them: '+gpu['stdout'])
        invocation=uuid.uuid4().hex[:10]
        control.mkdir(exist_ok=True);(control/'arm.json').unlink(missing_ok=True)
        atomic_json(control/'capture_config.json',dict(layer_ordinals=layers,decode_calls=a.decode_calls,min_context=1,
                     max_traces=len(layers)*len(a.decode_calls),max_bytes=int(a.scratch_gib*1024**3)))
        from .runtime import build
        runtime=out/('runtime_'+invocation)
        runtime_manifest=build(runtime)
        server=out/('server_'+invocation);server.mkdir();dispatch=server/'dispatch'
        name='asym-parent-'+invocation
        la=SimpleNamespace(method='kvarn44',model=model,port=a.port,max_model_len=s['max_model_len'],max_seqs=1,
                           batch_tokens=s['batch_tokens'],gpu_util=s['gpu_util'],server_seed=s['server_seed'],kv_bytes=None,eager=True)
        cli=launch_command(la,name)
        cmd=[sys.executable,'-m','kvbench_residual.bootstrap','--runtime',str(runtime),'--policy','parent','--trace-dir',str(dispatch),'--',*cli[3:]]
        env=os.environ.copy();env.pop('PYTHONHOME',None)
        env.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONNOUSERSITE='1',PYTHONPATH=str(runtime),AR_RUNTIME_ROOT=str(runtime),
                   AR_TRACE_DIR=str(dispatch),AR_POLICY='parent',AR_DIAG_TRACE_ROOT=str(control),KVARN_FUSED_DECODE='1',
                   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRITON_CACHE_DIR=str(out/'jit_cache/triton'),
                   TORCHINDUCTOR_CACHE_DIR=str(out/'jit_cache/torchinductor'),VLLM_CACHE_ROOT=str(out/'jit_cache/vllm'))
        native=dict(kv_dtype='kvarn_k4v4_g128',block_size=128,model_name=name,max_model_len=s['max_model_len'])
        launch=dict(command=cmd,config=native,runtime_manifest_sha256=sha_file(runtime/'runtime.json'),gpu=gpu,
                    source_hashes=runtime_manifest['executed_files_sha256'],environment={k:env.get(k) for k in
                    ('CUDA_VISIBLE_DEVICES','KVARN_SINK_TOKENS','KVARN_POOL_SLOTS','KVARN_FAST_FLUSH','KVARN_SPLIT_K','KVARN_NUM_KV_SPLITS')})
        from .analyze import analyze_paths
        pending_set=set(pending)
        with (server/'server.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launch['pid']=proc.pid;atomic_json(server/'launch.json',launch)
            base_url=f'http://127.0.0.1:{a.port}'
            launch['native']=ready(proc,base_url,native,a.startup_timeout)
            first=next(r for r in jsonl(out/'inputs/cases.jsonl') if r['case_key'] in pending_set)
            warm=request_one(base_url,name,first,s,min(8,a.diagnostic_max_tokens),server/'warmup.json.gz')
            atomic_json(server/'warmup.json',warm)
            from kvbench_residual.trace import summarize
            receipts=summarize(dispatch,'parent',runtime_manifest['executed_source_sha256'],.02)
            if not receipts['source_verified'] or not receipts['fused_launch_observed'] or not list(control.glob('hook_*.json')):
                raise RuntimeError('The isolated eager native Parent/capture hook was not verified')
            launch['dispatch']=receipts;atomic_json(server/'launch.json',launch)
            state['gpu_capture_executed']=True
            for row in jsonl(out/'inputs/cases.jsonl'):
                if row['case_key'] not in pending_set:continue
                if proc.poll() is not None:raise RuntimeError('Owned model server exited')
                base=out/'cases'/row['case_key'];base.mkdir(parents=True,exist_ok=True)
                attempt=base/('attempt_'+uuid.uuid4().hex[:12]);attempt.mkdir();trace=attempt/'traces';trace.mkdir()
                atomic_json(attempt/'input.json',row)
                estimate=((row['input_tokens']+a.diagnostic_max_tokens+127)//128)*(8*17920+128*8*128*4)+65536
                estimate*=len(layers)*len(a.decode_calls)
                if estimate>int(a.scratch_gib*1024**3):raise RuntimeError('Per-request snapshot bound exceeds --scratch-gib')
                if shutil.disk_usage(out).free<estimate+256*1024**2:
                    raise RuntimeError('Insufficient disk for the next complete case; free space and --resume')
                state.update(status='capturing',current_case=row['case_key'],completed_cases=completed)
                atomic_json(out/'status.json',state)
                arm=dict(epoch=uuid.uuid4().hex,dataset=row['dataset'],case_id=row['case_id'],
                         input_tokens=row['input_tokens'],prompt_sha256=digest(row['prompt']),sample_index=row['sample_index'],
                         trace_dir=str(trace),protocol='equal_payload_C1_eager_parent',source_manifest_sha256=sha_file(out/'inputs/manifest.json'))
                atomic_json(control/'arm.json',arm)
                try:
                    result=request_one(base_url,name,row,s,a.diagnostic_max_tokens,attempt/'response.json.gz')
                finally:(control/'arm.json').unlink(missing_ok=True)
                atomic_json(attempt/'request.json',result)
                if list(trace.glob('capture_error_*.json')):raise RuntimeError('Capture worker error; inspect current attempt')
                paths=sorted(trace.glob('trace_*.json'));records=[read_json(p) for p in paths]
                cov=validate_capture_coverage(records,row,result,layers,a.decode_calls)
                write_csv(attempt/'capture_coverage.csv',cov)
                state['status']='analyzing';atomic_json(out/'status.json',state)
                analyze_paths(paths,attempt/'analysis',a.totals,a.shares,a.methods,a.mask_seeds,a.block_n,
                              dict(case_key=row['case_key'],family=row['family'],was_truncated=row['was_truncated']),
                              controller_tolerances=a.controller_tolerances,
                              controller_margin=a.controller_margin,
                              controller_max_steps=a.controller_max_steps)
                # Verify self-contained analysis files before committing the case.
                hashes={str(p.relative_to(attempt)):sha_file(p) for p in attempt.rglob('*') if p.is_file() and 'traces' not in p.relative_to(attempt).parts}
                done=dict(schema='kvbench.asym.case.v1',status='completed',dataset=row['dataset'],case_id=row['case_id'],
                          case_key=row['case_key'],family=row['family'],attempt=str(attempt.relative_to(out)),
                          traces=len(paths),expected_capture_points=len(cov),missing_capture_points=sum(not r['captured'] for r in cov),
                          completion_tokens=result['completion_tokens'],finish_reason=result['finish_reason'],was_truncated=row['was_truncated'],
                          file_hashes=hashes,raw_arrays_retained=a.keep_traces)
                atomic_json(base/'done.json',done)
                if not a.keep_traces:prune_new_arrays(attempt)
                completed+=1
                state.update(status='case_completed',completed_cases=completed);atomic_json(out/'status.json',state)
                print(f"[{completed}/{selected}] {row['dataset']} {row['case_id']}: traces={len(paths)}, missing={done['missing_capture_points']}",flush=True)
        stop_owned_group(proc);proc=None
        state['status']='reporting';atomic_json(out/'status.json',state)
        report(out,a.bootstrap)
        state.update(status='completed',completed_cases=completed,all_source_records_selected=data_manifest['all_source_records_selected'])
        atomic_json(out/'status.json',state);print('Completed: '+str(out),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        atomic_json(out/'status.json',state);raise
    finally:
        (control/'arm.json').unlink(missing_ok=True)
        if proc is not None:
            from kvbench_residual.serve import stop_owned_group
            stop_owned_group(proc)


if __name__=='__main__':main()

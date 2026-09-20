#!/usr/bin/env python3
"""Score saved generations only. No network inference and no local execution of model-generated Python.
MATH: installed EvalScope extraction/numeric Accuracy; HumanEval: explicitly requested isolated Docker.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,multiprocessing as mp,subprocess,sys,time,uuid
from pathlib import Path
from metrics import atomic_json


def math_score(text,reference):
    from evalscope.metrics.math.parser import extract_answer
    from evalscope.metrics.nlp.metrics import Accuracy
    prediction=extract_answer(text) if text else ''
    value=Accuracy(numeric=True).apply([prediction],[str(reference)])[0] if prediction else 0.0
    return {'score':bool(value),'extracted_prediction':prediction,'grading_status':'graded'}


CONTAINER_WRAPPER=r'''
import json,subprocess,sys
request=json.load(sys.stdin)
try:
 p=subprocess.run([sys.executable,'-I','-B','-c',request['program']],capture_output=True,text=True,timeout=request['timeout'])
 result={'passed':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout[-8192:],'stderr':p.stderr[-8192:]}
except subprocess.TimeoutExpired:
 result={'passed':False,'timeout':True}
print(json.dumps(result))
'''


def human_score(row,image,timeout):
    from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter
    text=row.get('final_answer_for_scoring','');problem=row.get('original_record')
    if not isinstance(problem,dict) or not {'prompt','test','entry_point'}<=problem.keys():
        return {'score':None,'grading_status':'missing_original_problem'}
    if not text.strip():return {'score':False,'grading_status':'graded','reason':'no_final_answer','extracted_prediction':''}
    completion=HumanevalAdapter._postprocess(text)
    program=problem['prompt']+completion+'\n'+problem['test']+'\n'+f'check({problem["entry_point"]})'
    name='kvbench-grade-'+uuid.uuid4().hex
    cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only',
        '--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=64','--memory=512m','--cpus=1',
        '--user=65534:65534','--tmpfs=/tmp:rw,nosuid,nodev,size=64m','--workdir=/tmp','-i',image,
        'python','-I','-B','-c',CONTAINER_WRAPPER]
    try:
        res=subprocess.run(cmd,input=json.dumps({'program':program,'timeout':timeout}),text=True,
                           capture_output=True,timeout=timeout+60)
        if res.returncode:
            return {'score':None,'grading_status':'sandbox_error','stderr':res.stderr[-8192:],
                    'returncode':res.returncode,'extracted_prediction':completion}
        result=json.loads(res.stdout)
        return {'score':bool(result['passed']),'grading_status':'graded','execution_result':result,
                'extracted_prediction':completion}
    except (subprocess.TimeoutExpired,json.JSONDecodeError,OSError) as exc:
        # Only our uniquely named container is eligible for cleanup.
        try:subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
        except (OSError,subprocess.SubprocessError):pass
        return {'score':None,'grading_status':'sandbox_error','error':str(exc),'extracted_prediction':completion}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',choices=['math_500','humaneval'],default=['math_500','humaneval'])
    p.add_argument('--human-sandbox',choices=['none','docker'],default='none')
    p.add_argument('--docker-image',default='python:3.11-slim',help='Must already exist locally; image ID is recorded')
    p.add_argument('--human-timeout',type=int,default=4);p.add_argument('--math-timeout',type=int,default=30)
    p.add_argument('--force',action='store_true',help='Re-score same saved outputs; old scores are backed up')
    a=p.parse_args()
    if a.human_timeout<=0 or a.math_timeout<=0:raise ValueError('Timeout must be positive')
    image_id=None
    if a.human_sandbox=='docker' and 'humaneval' in a.datasets:
        q=subprocess.run(['docker','image','inspect',a.docker_image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20)
        if q.returncode:raise RuntimeError('Docker/image 不可用；不会回退到宿主执行。请先准备隔离环境。\n'+q.stderr)
        image_id=q.stdout.strip()
    protocol={'evalscope_version':importlib.metadata.version('evalscope'),'score_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'math_rule':'evalscope extract_answer + Accuracy(numeric=True)','math_timeout':a.math_timeout,
        'human_rule':'EvalScope _postprocess + original HumanEval prompt/test in isolated container',
        'human_sandbox':a.human_sandbox,'docker_image_requested':a.docker_image,'docker_image_id':image_id,
        'human_test_timeout_s':a.human_timeout,'cpu_limit':1,'memory_limit':'512m','network':'none','n_per_case':1,
        'inference_rerun':False,'length_cap_samples':'retained; scored using their saved final answer'}
    pool=None
    try:
        for point in sorted((a.suite/'points').glob('*')):
            states=sorted(point.glob('attempt*/state.json'))
            if not states:continue
            d=states[-1].parent;state=json.loads(states[-1].read_text());dataset=state['point']['dataset']
            outputs=d/'requests.jsonl'
            if dataset not in a.datasets or not outputs.exists():continue
            if (d/'scores.summary.json').exists():
                if not a.force:
                    print('跳过已有评分:',d);continue
                stamp=str(time.time_ns())
                for f in ['scores.jsonl','scores.summary.json','grading_protocol.json']:
                    if (d/f).exists():(d/f).rename(d/(f+'.backup_'+stamp))
            rows=[json.loads(x) for x in outputs.read_text(encoding='utf-8').splitlines() if x.strip()]
            atomic_json(d/'grading_protocol.json',{**protocol,'outputs_sha256':hashlib.sha256(outputs.read_bytes()).hexdigest()})
            results=[]
            with (d/'scores.jsonl').open('w',encoding='utf-8') as f:
                for row in rows:
                    if not row.get('http_success'):
                        result={'score':False,'grading_status':'generation_failed'}
                    elif dataset=='humaneval':
                        result=human_score(row,image_id,a.human_timeout) if a.human_sandbox=='docker' else {'score':None,'grading_status':'not_scored_no_sandbox'}
                    else:
                        if pool is None:pool=mp.get_context('spawn').Pool(1)
                        job=pool.apply_async(math_score,(row.get('final_answer_for_scoring',''),row['reference']))
                        try:result=job.get(timeout=a.math_timeout)
                        except mp.TimeoutError:
                            pool.terminate();pool.join();pool=None
                            result={'score':None,'grading_status':'grader_timeout'}
                        except Exception as exc:result={'score':None,'grading_status':'grader_error','error':str(exc)}
                    record={'case_id':row['case_id'],'dataset':dataset,'finish_reason':row.get('finish_reason'),
                            'length_capped':row['is_length_capped'],**result}
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();results.append(record)
            n=state['point']['number'];correct=sum(r.get('score') is True for r in results)
            unscored=sum(r.get('score') is None for r in results)
            complete=len(rows)==n and unscored==0 and state['status']=='ok'
            summary={'quality_scoring_complete':complete,'quality_total_tasks':n,'quality_scored_tasks':len(results)-unscored,
                'quality_unscored_tasks':unscored,'quality_correct_tasks':correct,
                'accuracy_all_tasks':correct/n if complete else None,
                'humaneval_pass_at_1':correct/n if complete and dataset=='humaneval' else None,
                'known_correct_fraction_lower_bound':correct/n if n else None,
                'quality_metric_note':'n=1 per task; HTTP success is not accuracy; grader errors remain unknown'}
            # Only pair quality and timing from this exact saved generation run.
            perf=json.loads((d/'stats.json').read_text()) if (d/'stats.json').exists() else {}
            wall=perf.get('request_window_s')
            summary['correct_tasks_per_second']=correct/wall if complete and wall else None
            atomic_json(d/'scores.summary.json',summary)
            print(state['point']['id'],summary)
    finally:
        if pool is not None:pool.close();pool.join()
    from run_real import collect
    collect(a.suite)
    print(a.suite/'summary.csv')

if __name__=='__main__':main()

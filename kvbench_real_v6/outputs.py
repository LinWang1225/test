#!/usr/bin/env python3
"""Export every measured response without executing pickle globals or generated code."""
from __future__ import annotations
import base64, csv, gzip, hashlib, io, json, pickle, sqlite3
from pathlib import Path
from contextlib import closing
from typing import Any
from metrics import atomic_json,describe,finite


def key(body:dict)->str:
    return hashlib.sha256(json.dumps({'messages':body.get('messages'), 'seed':body.get('seed')},
        sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


class NoGlobalsUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise ValueError(f'Refused pickle global {module}.{name}')
    def persistent_load(self,pid):raise ValueError('Persistent pickle objects are not supported')


def decode_saved(value):
    if not value:return []
    if isinstance(value,list):return value
    try: parsed=json.loads(value)
    except (ValueError,TypeError):
        parsed=NoGlobalsUnpickler(io.BytesIO(base64.b64decode(value,validate=True))).load()
    if not isinstance(parsed,list):raise ValueError('Unknown response_messages schema; original DB preserved')
    return parsed


def join_response(chunks:list, thinking:bool)->dict:
    content=[];reasoning=[];finish=None;stop_reason=None;usage=None;ids=[];errors=[]
    reasoning_channel=False
    for item in chunks:
        if isinstance(item,str):
            try:item=json.loads(item)
            except ValueError:errors.append(item);continue
        if not isinstance(item,dict):errors.append(repr(item));continue
        if item.get('error') is not None:errors.append(item['error'])
        if item.get('id') and item['id'] not in ids:ids.append(item['id'])
        if isinstance(item.get('usage'),dict):usage=item['usage']
        for choice in item.get('choices') or []:
            if choice.get('index',0)!=0:raise ValueError('Only n=1 is supported for paired task-level experiments')
            msg=choice.get('delta',choice.get('message',{}))
            if not isinstance(msg,dict):continue
            if isinstance(msg.get('content'),str):content.append(msg['content'])
            rv=msg.get('reasoning_content',msg.get('reasoning'))
            if isinstance(rv,str):reasoning_channel=True;reasoning.append(rv)
            if choice.get('finish_reason') is not None:finish=choice['finish_reason']
            if choice.get('stop_reason') is not None:stop_reason=choice['stop_reason']
    raw=''.join(content);rs=''.join(reasoning)
    if '</think>' in raw:
        before,answer=raw.rsplit('</think>',1)
        text_reason=before.split('<think>',1)[-1]
        final=answer.strip();extraction='closed_think_delimiter'
        if not rs:rs=text_reason
    elif reasoning_channel:
        final=raw.strip();extraction='server_reasoning_channel'
    elif thinking:
        final='';rs=raw;extraction='no_closed_think_or_server_reasoning_channel'
    else:
        final=raw.strip();extraction='non_thinking_content'
    return {'raw_content':raw,'reasoning_content':rs,'final_answer_for_scoring':final,
            'answer_extraction':extraction,'finish_reason':finish,'stop_reason':stop_reason,
            'usage':usage,'response_ids':ids,'response_errors':errors}


def export_run(measured:Path,dest:Path,expected:list[dict],thinking:bool)->dict[str,Any]:
    dbs=list(measured.rglob('benchmark_data.db'))
    if len(dbs)!=1:raise ValueError(f'需要一份 EvalScope 数据库，得到 {len(dbs)} 份；未猜测路径。')
    with closing(sqlite3.connect(dbs[0].resolve().as_uri()+'?mode=ro',uri=True)) as con:
        con.row_factory=sqlite3.Row
        cols={r[1] for r in con.execute('PRAGMA table_info(result)')}
        needed={'request','response_messages','success','start_time','completed_time','latency',
                'first_chunk_latency','time_per_output_token','completion_tokens','prompt_tokens'}
        if not needed<=cols:raise ValueError(f'EvalScope DB schema changed: missing {needed-cols}')
        rows=[dict(r) for r in con.execute('SELECT * FROM result')]
    index={key(r['request']):r for r in expected}
    if len(index)!=len(expected):raise ValueError('Duplicate request identity in manifest')
    rawdir=dest/'raw_responses';rawdir.mkdir(exist_ok=True)
    exported=[];seen=[];decode_errors=[];parameter_mismatches=[]
    for i,r in enumerate(rows):
        try:body=json.loads(r['request'])
        except (ValueError,TypeError):body={}
        fk=key(body);ref=index.get(fk);seen.append(fk)
        if ref:
            changed={k:{'expected':v,'actual':body.get(k)} for k,v in ref['request'].items() if body.get(k)!=v}
            if changed:parameter_mismatches.append({'case_id':ref['case_id'],'changed':changed})
        response_path=rawdir/f'{i:05d}_{fk[:12]}.json.gz'
        try:
            chunks=decode_saved(r.get('response_messages'))
            with gzip.open(response_path,'wt',encoding='utf-8') as f:json.dump(chunks,f,ensure_ascii=False)
            joined=join_response(chunks,thinking)
        except Exception as exc:
            decode_errors.append({'row':i,'error':str(exc)})
            # Preserve undecoded payload verbatim; never silently omit a failing output.
            response_path=rawdir/f'{i:05d}_{fk[:12]}.undecoded.txt'
            response_path.write_text(str(r.get('response_messages')),encoding='utf-8')
            joined={'finish_reason':None,'final_answer_for_scoring':'','decode_error':str(exc)}
        usage=joined.get('usage') or {}
        rec={'row':i,'dataset':ref.get('dataset') if ref else None,'case_id':ref.get('case_id') if ref else None,
             'request_identity':fk,'request':body,'reference':ref.get('reference') if ref else None,
             'original_record':ref.get('original_record') if ref else None,
             'http_success':r['success']==1,'request_id':r.get('request_id'),
             'raw_response_file':str(response_path.relative_to(dest)),
             'start_time_raw':r['start_time'],'completed_time_raw':r['completed_time'],
             'e2e_s':r['latency'],'ttft_s':r['first_chunk_latency'],'tpot_s':r['time_per_output_token'],
             'prompt_tokens':r['prompt_tokens'],'completion_tokens':r['completion_tokens'],
             'token_count_source':'server_usage' if {'prompt_tokens','completion_tokens'}<=usage.keys() else 'evalscope_fallback_or_missing',
             'client_chunk_intervals_s':json.loads(r.get('inter_token_latencies') or '[]'),**joined}
        rec['is_length_capped']=joined.get('finish_reason')=='length'  # compatibility alias
        rec['is_length_limit_termination']=joined.get('finish_reason')=='length'
        rec['natural_stop_reported']=joined.get('finish_reason')=='stop'
        exported.append(rec)
    outpath=dest/'requests.jsonl'
    with outpath.open('w',encoding='utf-8') as f:
        for row in exported:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    matched=set(seen)&index.keys();missing=[r['case_id'] for k,r in index.items() if k not in matched]
    good=[r for r in exported if r['http_success']]
    starts=[x for r in exported if (x:=finite(r['start_time_raw'])) is not None]
    ends=[x for r in exported if (x:=finite(r['completed_time_raw'])) is not None]
    wall=max(ends)-min(starts) if starts and ends else 0
    capped=sum(r['is_length_capped'] for r in good)
    stat={'requests_expected':len(expected),'requests_recorded':len(rows),'succeeded':len(good),
          'failed':len(rows)-len(good),'missing_case_ids':missing,
          'unmatched_rows':sum(k not in index for k in seen),'duplicate_rows':len(seen)-len(set(seen)),
          'response_decode_errors':decode_errors,'request_parameter_mismatches':parameter_mismatches,'request_window_s':wall,
          'rps':len(good)/wall if wall>0 else None,
          'output_tok_s':sum(r['completion_tokens'] or 0 for r in good)/wall if wall>0 else None,
          'length_capped':capped,'length_capped_fraction':capped/len(expected) if expected else None,
          'length_limit_terminations':capped,'length_limit_fraction':capped/len(expected) if expected else None,
          'natural_stops':sum(r['natural_stop_reported'] for r in good),
          'missing_finish_reason':sum(r.get('finish_reason') is None for r in good),
          'token_usage_missing':sum(r['token_count_source']!='server_usage' for r in good),
          'missing_final_answer':sum(not r.get('final_answer_for_scoring','').strip() for r in good),
          'request_multiset_sha256':hashlib.sha256('\n'.join(sorted(seen)).encode()).hexdigest(),
          'outputs_jsonl':str(outpath),'latency_population':'successful measured requests',
          'quantile_method':'nearest-rank; censored length-cap samples retained'}
    for col,prefix,scale in [('e2e_s','e2e_s',1),('ttft_s','ttft_ms',1000),('tpot_s','tpot_ms',1000),
                             ('prompt_tokens','input_tokens',1),('completion_tokens','output_tokens',1)]:
        stat.update(describe([x*scale for r in good if (x:=finite(r[col])) is not None],prefix))
    stat['measurement_complete']=bool(len(good)==len(expected)==len(rows) and not missing and not decode_errors
        and stat['unmatched_rows']==0 and stat['duplicate_rows']==0 and not parameter_mismatches and wall>0)
    atomic_json(dest/'output_export.json',stat)
    fields=['dataset','case_id','http_success','finish_reason','is_length_capped','is_length_limit_termination','prompt_tokens','completion_tokens',
            'e2e_s','ttft_s','tpot_s','token_count_source','answer_extraction','raw_response_file']
    with (dest/'requests.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(exported)
    return stat

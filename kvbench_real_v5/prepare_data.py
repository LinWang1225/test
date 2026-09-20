#!/usr/bin/env python3
"""Freeze original MATH-500 / HumanEval records and the installed EvalScope prompt templates.
No inference, prompt padding or reference-answer injection is performed.
"""
from __future__ import annotations
import argparse, hashlib, importlib, importlib.metadata, json
from pathlib import Path
from metrics import atomic_json

NAMES = {'math_500': (500, 'HuggingFaceH4/MATH-500', 'AI-ModelScope/MATH-500'),
         'humaneval': (164, 'openai/openai_humaneval', 'opencompass/humaneval')}

def hash_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local_rows(path):
    p=Path(path)
    if p.suffix=='.json':
        data=json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(data,list): raise ValueError('本地 JSON 必须是原始记录的数组。')
        return data
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def prepare(name, source, local, revision):
    expected,hf,ms=NAMES[name]
    importlib.import_module(f'evalscope.benchmarks.{name}.{name}_adapter')
    from evalscope.api.registry import BENCHMARK_REGISTRY
    template=BENCHMARK_REGISTRY[name].prompt_template
    provenance={'source':source,'revision_requested':revision,'revision_resolved':None}
    if source=='local':
        if not local: raise ValueError(f'缺少 {name} 本地原始 JSONL 路径。')
        records=local_rows(local);provenance.update(file=str(Path(local).resolve()),sha256=hash_file(local))
    elif source=='huggingface':
        from huggingface_hub import HfApi
        from datasets import load_dataset
        rev=HfApi().dataset_info(hf,revision=revision).sha
        records=list(load_dataset(hf,split='test',revision=rev))
        provenance.update(dataset_id=hf,revision_resolved=rev)
    else:
        from modelscope.msdatasets import MsDataset
        kwargs={'split':'test'}
        if revision: kwargs['version']=revision
        if name=='humaneval':kwargs['subset_name']='openai_humaneval'
        records=list(MsDataset.load(ms,**kwargs))
        provenance.update(dataset_id=ms,note='Resolved remote revision may be unavailable; frozen local bytes are authoritative.')
    rows=[];seen=set()
    for i,r in enumerate(records):
        if name=='math_500':
            cid=str(r.get('unique_id',r.get('id',f'math_500/test/{i}')))
            question=r['problem'];reference=r['answer']
        else:
            cid=str(r['task_id']);question=r['prompt'];reference=r['canonical_solution']
        if cid in seen: raise ValueError(f'重复题号：{cid}')
        seen.add(cid)
        rows.append({'dataset':name,'case_id':cid,'source_index':i,
                     'messages':[{'role':'user','content':template.format(question=question)}],
                     'reference':reference,'original_record':dict(r)})
    if len(rows)!=expected:
        raise ValueError(f'{name} 需要原始全集 {expected} 条，读取到 {len(rows)}。筛查子集由控制器固定抽取，不在这里裁剪。')
    return rows,{**provenance,'count':len(rows),'template':template,
                 'template_sha256':hashlib.sha256(template.encode()).hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',choices=['modelscope','huggingface','local'],default='modelscope')
    p.add_argument('--datasets',nargs='+',choices=list(NAMES),default=list(NAMES))
    p.add_argument('--math-file');p.add_argument('--human-file');p.add_argument('--revision')
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('输出目录非空，拒绝覆盖冻结数据。请换一个目录。')
    # Fetch/validate all records before writing the authoritative manifest.
    prepared={n:prepare(n,a.source,a.math_file if n=='math_500' else a.human_file,a.revision) for n in a.datasets}
    a.out.mkdir(parents=True,exist_ok=True)
    meta={'evalscope_version':importlib.metadata.version('evalscope'),'datasets':{}}
    for name,(rows,info) in prepared.items():
        path=a.out/f'{name}.jsonl'
        with path.open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        meta['datasets'][name]={**info,'file':path.name,'sha256':hash_file(path)}
        print(name,len(rows),path)
    atomic_json(a.out/'manifest.json',meta)
    print('已冻结题目和当前 EvalScope 模板；未运行模型。')

if __name__=='__main__':main()

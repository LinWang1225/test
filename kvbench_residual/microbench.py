# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape AR attention timings, writer overhead and logical GPU load counts."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from .transform import (POLICIES, PARENT_SHA256, STORE_PATH, COMMON_FIXES,
                        get_expected_pins, read_sources, generated_receipt, load_variant)
from .fixtures import Case, Layout, GpuCase, build_case, case_metadata, make_dominant_chunks
from .representation import pack_planes_torch
from .validate import atomic_json, _torch_functions


def timing(torch, state, module, mode, args):
    for _ in range(args.warmup):
        state.launch(module, mode, args.autotune)
    torch.cuda.synchronize(args.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.graph_repeats):
            state.launch(module, mode, args.autotune)
    graph.replay(); torch.cuda.synchronize(args.device)
    samples = []
    for _ in range(args.trials):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record(); end.synchronize()
        samples.append(begin.elapsed_time(end)*1000/(args.iterations*args.graph_repeats))
    return {"latency_us_median": statistics.median(samples), "latency_us_min": min(samples),
            "latency_us_max": max(samples), "latency_us_samples": samples}


def writer_timings(torch, sources, args):
    """Actual parent INT4 packer versus actual new plane packer; no claim of
    measuring Sinkhorn, rotation, full flush or prefill here."""
    parent = _torch_functions(sources[STORE_PATH], {'_pack_lowbit'})
    code = torch.arange(64*128*128, dtype=torch.int32, device=args.device).reshape(64,128,128) % 16
    records=[]
    for name, fn in (('native_int4',lambda:parent._pack_lowbit(code,4)),('tile_high2_low2',lambda:pack_planes_torch(code))):
        hold=fn(); torch.cuda.synchronize(args.device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hold=fn()
        graph.replay(); torch.cuda.synchronize(args.device)
        samples=[]
        for _ in range(args.trials):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(20):
                graph.replay()
            end.record(); end.synchronize()
            samples.append(begin.elapsed_time(end)*1000/20)
        records.append({'policy':name,'tiles':64,'rows':128,'columns':128,
                        'output_bytes':int(hold.numel()*hold.element_size()),
                        'median_us':statistics.median(samples),'samples_us':samples,
                        'scope':'Packing only after identical INT4 quantization; end-to-end store/flush cost remains in formal trials'})
        del graph,hold
    return records


def write_tables(out, result):
    out=Path(out)
    fields=('batch','context_tokens','group','value_bits','mode','policy','fixture_kind','BLOCK_N','splits',
            'latency_us_median','latency_us_min','latency_us_max','speedup_vs_parent','speedup_vs_repacked_full',
            'logical_code_bytes','native_code_bytes','logical_code_reduction_fraction',
            'key_skipped_chunks','value_skipped_chunks','quantized_chunks','cache_bytes','pool_bytes','autotune')
    with (out/'benchmark.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(result.get('rows',[]))
    lines=['# Anchor–Residual 固定形状 GPU 结果','',
        '时间包括 Anchor 打分、误差界判断、按需 Residual 读取、attention 与 split 合并；不含模型其他层、Q/output rotation、KV store、prefill或HTTP。',
        '时间核不带统计；logical code bytes 来自另一次带计数的GPU分支执行，不是DRAM计数。真正DRAM流量请用 profile_dram。',
        '默认使用随机合成KV；dominant为显式选择的分支压力测试，不能当作模型注意力分布证据。',
        '反复读取同一缓存可能命中L2，短形状尤其如此。native/planar两份仅存在于验证fixture，正式服务只保留其中一份缓存。',
        '速度比>1表示更快；减少的逻辑码字节不能直接等同TPOT或实际显存流量减少。','',
        '| B | L | Mode | Policy | µs | parent/本组 | full平面/本组 | 逻辑码减少 |',
        '|---:|---:|---|---|---:|---:|---:|---:|']
    for r in result.get('rows',[]):
        def num(key,digits=3):
            val=r.get(key)
            return f'{val:.{digits}f}' if val is not None else 'NA'
        reduction=r.get('logical_code_reduction_fraction')
        reduction='NA' if reduction is None else f'{100*reduction:.2f}%'
        lines.append(f"| {r['batch']} | {r['context_tokens']} | {r['mode']} | {r['policy']} | {num('latency_us_median')} | {num('speedup_vs_parent')} | {num('speedup_vs_repacked_full')} | {reduction} |")
    if result.get('writer_packing'):
        lines+=['','## 写缓存时的独立打包成本','', '| Packer | 64 tiles median µs | 输出字节 |','|---|---:|---:|']
        for r in result['writer_packing']:
            lines.append(f"| {r['policy']} | {r['median_us']:.3f} | {r['output_bytes']} |")
        lines+=['','此项仅隔离 packer，完整 prefill/flush 开销由全量服务实验承担，不能从正式耗时中扣除。']
    lines+=['',f"状态：{result['status']}；GPU执行：{result.get('gpu_executed',False)}。慢策略同样保留，不做速度筛选。"]
    (out/'readout.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def profile_case(torch, args, sources, hashes):
    """Only profiled interval is one warmed attention call (1 or 2 kernels)."""
    out=Path(args.out)
    receipt={'schema':'kvbench.residual.profile_case.v1','status':'running','gpu_executed':False,
             'policy':args.policy,'epsilon':args.epsilon,'source_root':str(Path(args.source_root).resolve()),
             'parent_pins':get_expected_pins(),'generated_sha256':hashes,
             'batch':args.batch,'length':args.length,'mode':args.mode,'block_n':args.block_n,
             'splits':args.splits if args.mode=='split' else 1,'profiled_launches':2 if args.mode=='split' else 1,
             'scope':'One fixed-shape attention call including bound/selection/loads and split combine; excludes rotations, pack/store and model other layers',
             'fixture_kind':args.fixture}
    atomic_json(out/'profile_case.json',receipt)
    try:
        data=build_case(Case('ncu_profile',tuple([args.length]*args.batch),Layout(),seed=913))
        if args.fixture=='dominant':
            make_dominant_chunks(data)
        state=GpuCase(data,args.device,args.splits,args.block_n,args.num_warps,args.num_stages)
        module=load_variant(args.source_root,args.policy,args.epsilon)
        for _ in range(args.warmup):
            state.launch(module,args.mode,False)
        torch.cuda.synchronize(args.device)
        # Nsight Compute is started with --profile-from-start off.
        torch.cuda.profiler.start()
        try:
            state.launch(module,args.mode,False)
            torch.cuda.synchronize(args.device)
        finally:
            torch.cuda.profiler.stop()
        receipt.update(status='completed',gpu_executed=True,passed=True)
        atomic_json(out/'profile_case.json',receipt)
    except BaseException as exc:
        receipt.update(status='failed',passed=False,error=str(exc),traceback=traceback.format_exc())
        atomic_json(out/'profile_case.json',receipt)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--epsilon',type=float,default=.02)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batches',nargs='+',type=int,default=[1,8])
    parser.add_argument('--lengths',nargs='+',type=int,default=[1024,4096,8192,16384,32768])
    parser.add_argument('--modes',nargs='+',choices=['single','split'],default=['single','split'])
    parser.add_argument('--groups',nargs='+',type=int,choices=[128],default=[128])
    parser.add_argument('--value-bits',nargs='+',type=int,choices=[4],default=[4])
    parser.add_argument('--max-model-len',type=int,default=32768)
    parser.add_argument('--splits',type=int,default=32)
    parser.add_argument('--block-n',type=int,choices=[16,32,64],default=32)
    parser.add_argument('--num-warps',type=int,choices=[2,4],default=4)
    parser.add_argument('--num-stages',type=int,choices=[1,2],default=1)
    parser.add_argument('--autotune',action='store_true')
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--graph-repeats',type=int,default=16)
    parser.add_argument('--iterations',type=int,default=10)
    parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--fixture',choices=['random','dominant'],default='random')
    parser.add_argument('--profile-case',action='store_true')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--length',type=int,default=8192)
    parser.add_argument('--policy',choices=POLICIES,default='bound_kv')
    parser.add_argument('--mode',choices=['single','split'],default='split')
    args=parser.parse_args()
    counts=args.batches+args.lengths+[args.batch,args.length,args.warmup,args.graph_repeats,args.iterations,args.trials,args.splits]
    if min(counts)<=0 or args.splits&(args.splits-1) or not math.isfinite(args.epsilon) or args.epsilon<0:
        parser.error('Positive shapes/counts, power-of-two splits and finite nonnegative epsilon required')
    if max(args.lengths+[args.length])>args.max_model_len:
        parser.error('Context exceeds maximum model window')
    out=Path(args.out); started=time.time()
    result={'schema':'kvbench.residual.microbenchmark.v1','status':'running','passed':False,'gpu_executed':False,
            'source_root':str(Path(args.source_root).resolve()),'parent_sha256':PARENT_SHA256,
            'parent_pins':get_expected_pins(),'epsilon':args.epsilon,'settings':vars(args),'rows':[],
            'common_fixes':list(COMMON_FIXES),'speed_gate':'none',
            'scope':'Fixed-shape attention including selection; writer packer measured separately; no generation',
            'traffic_scope':'Logical addresses from separate GPU instrumentation, not DRAM counters'}
    atomic_json(out/'benchmark.json',result)
    try:
        sources=read_sources(args.source_root)
        result['generated_sha256']=generated_receipt(sources,args.epsilon)
        import torch
        from vllm.triton_utils import triton
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; CPU timings cannot substitute for a GPU experiment')
        torch.cuda.set_device(args.device)
        props=torch.cuda.get_device_properties(args.device)
        if props.major<8:
            raise RuntimeError('AR v1 requires SM80 or newer')
        result['device']={'name':props.name,'capability':[props.major,props.minor],
            'sm_count':props.multi_processor_count,'total_memory':props.total_memory,
            'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,
            'triton_version':getattr(triton,'__version__',None),
            'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES'),
            'NVIDIA_TF32_OVERRIDE':os.getenv('NVIDIA_TF32_OVERRIDE'),'TRITON_F32_DEFAULT':os.getenv('TRITON_F32_DEFAULT')}
        if args.profile_case:
            profile_case(torch,args,sources,result['generated_sha256'])
            # This is not a complete default microbenchmark receipt.
            result.update(status='profile_case_only',gpu_executed=True,passed=False)
            atomic_json(out/'benchmark.json',result)
            return
        modules={p:load_variant(args.source_root,p,args.epsilon) for p in POLICIES}
        for i,(batch,length) in enumerate(itertools.product(args.batches,args.lengths)):
            data=build_case(Case('microbenchmark',tuple([length]*batch),Layout(),seed=817+i))
            if args.fixture=='dominant':
                make_dominant_chunks(data)
            state=GpuCase(data,args.device,args.splits,args.block_n,args.num_warps,args.num_stages)
            order=POLICIES[i%len(POLICIES):]+POLICIES[:i%len(POLICIES)]
            for mode in args.modes:
                rows=[]
                for policy in order:
                    timed=timing(torch,state,modules[policy],mode,args)
                    # The untimed instrumented launch uses the timed config,
                    # including the chosen chunk size when autotune is requested.
                    saved=(state.block_n,state.num_warps,state.num_stages)
                    if args.autotune:
                        obj=getattr(modules[policy], '_kvarn_fused_decode_kernel' if mode=='single' else '_kvarn_fused_decode_stage1')
                        cfg=obj.best_config
                        state.block_n=int(cfg.kwargs['BLOCK_N']); state.num_warps=cfg.num_warps; state.num_stages=cfg.num_stages
                    loads=state.collect_stats(modules[policy],mode)
                    block_n=state.block_n
                    state.block_n,state.num_warps,state.num_stages=saved
                    row={'batch':batch,'context_tokens':length,'group':128,'value_bits':4,'mode':mode,'policy':policy,
                         'fixture_kind':args.fixture,'BLOCK_N':block_n,'splits':args.splits if mode=='split' else 1,
                         'autotune':args.autotune,'cache_bytes':int(data['cache'].nbytes),
                         'pool_bytes':int(data['pool_k'].nbytes+data['pool_v'].nbytes),'fixture':case_metadata(data),
                         'logical_code_bytes':loads['logical_quantized_code_bytes'],
                         'native_code_bytes':loads['native_quantized_code_bytes'],
                         'logical_code_reduction_fraction':loads['logical_code_reduction_fraction'],
                         'key_skipped_chunks':loads['key_skipped_chunks'],'value_skipped_chunks':loads['value_skipped_chunks'],
                         'quantized_chunks':loads['quantized_chunks'],'logical_read_details':loads,**timed}
                    rows.append(row); result['gpu_executed']=True
                    print(f"B{batch} L{length} {mode} {policy}: {timed['latency_us_median']:.3f} us, logical code reduction={loads['logical_code_reduction_fraction']}",flush=True)
                native=next(r['latency_us_median'] for r in rows if r['policy']=='parent')
                full=next(r['latency_us_median'] for r in rows if r['policy']=='repacked_full')
                for row in rows:
                    row.update(speedup_vs_parent=native/row['latency_us_median'],speedup_vs_repacked_full=full/row['latency_us_median'])
                result['rows'].extend(rows)
                atomic_json(out/'benchmark.json',result); write_tables(out,result)
            del state,data
        result['writer_packing']=writer_timings(torch,sources,args)
        result.update(status='completed',passed=True,elapsed_s=time.time()-started)
        atomic_json(out/'benchmark.json',result); write_tables(out,result)
    except BaseException as exc:
        result.update(status='failed',passed=False,error=str(exc),traceback=traceback.format_exc(),elapsed_s=time.time()-started)
        atomic_json(out/'benchmark.json',result); write_tables(out,result)
        raise


if __name__=='__main__':
    main()

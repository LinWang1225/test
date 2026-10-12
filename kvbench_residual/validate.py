# SPDX-License-Identifier: Apache-2.0
"""CPU proof checks and mandatory CUDA implementation gate for AR-KV v1."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback
import types

import numpy as np

from .transform import (POLICIES, PARENT_SHA256, DECODE_PATH, STORE_PATH, TORCH_DECODE_PATH,
                        COMMON_FIXES, get_expected_pins, read_sources, transform_sources,
                        generated_receipt, load_variant, sha_text)
from .fixtures import (Case, Layout, GpuCase, build_case, case_metadata, selective_attention,
                       make_dominant_chunks, unpack_block)
from .representation import pack_planes_numpy, unpack_planes_numpy, parent_unpack_numpy


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def error_metrics(actual, reference):
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {"finite": False, "max_abs": None, "relative_l2": None, "rms": None}
    diff = actual-reference
    return {"finite": True, "max_abs": float(np.max(np.abs(diff), initial=0)),
            "relative_l2": float(np.linalg.norm(diff.ravel())/max(np.linalg.norm(reference.ravel()), 1e-12)),
            "rms": float(np.sqrt(np.mean(diff*diff)))}


def within(metrics, relative=0.02, absolute=0.02):
    return metrics["finite"] and metrics["relative_l2"] <= relative and metrics["max_abs"] <= absolute


def budget_diagnostics(measured, parent_measured, own_reference, full_reference, epsilon):
    """Show the raw GPU error against the declared *real* budget, separately
    from both implementations' numerical deviations. No empirical deviation
    is promoted to a theorem allowance or silently added to the budget."""
    actual=np.asarray(measured,np.float64)
    parent=np.asarray(parent_measured,np.float64)
    vrefs=own_reference['value_reference']
    ratio=actual.shape[1]//vrefs.shape[1]
    budget=epsilon*np.repeat(vrefs,ratio,axis=1)
    observed=np.linalg.norm(actual-parent,axis=-1)
    candidate_noise=np.linalg.norm(actual-own_reference['output'],axis=-1)
    parent_noise=np.linalg.norm(parent-full_reference,axis=-1)
    excess=np.maximum(observed-budget,0)
    if not all(np.isfinite(x).all() for x in (budget,observed,candidate_noise,parent_noise,excess)):
        return {'finite':False,'meaning':'Nonfinite GPU result; numerical gate fails'}
    return {'finite':True,'epsilon':epsilon,'gpu_error_l2_per_head':observed.tolist(),
            'real_budget_l2_per_head':budget.tolist(),'strict_excess_l2_per_head':excess.tolist(),
            'strict_exceedance_count':int(np.count_nonzero(observed>budget)),
            'max_gpu_error_l2':float(observed.max(initial=0)),
            'max_strict_budget_excess_l2':float(excess.max(initial=0)),
            'candidate_vs_float64_l2_per_head':candidate_noise.tolist(),
            'parent_vs_float64_l2_per_head':parent_noise.tolist(),
            'meaning':'Strict comparison to epsilon*V_ref in rotated attention space, without a floating-point allowance. Numerical deviations are shown separately; this is a diagnostic, not a formal GPU certificate.'}


def cpu_checks(sources, epsilon=0.02):
    hashes = generated_receipt(sources, epsilon)
    checks = []
    # Same code, not a replacement quantizer.
    codes = np.tile(np.arange(16, dtype=np.uint8), (3, 8, 8))
    recovered = unpack_planes_numpy(pack_planes_numpy(codes), codes.shape[-1])
    if not np.array_equal(codes, recovered):
        raise RuntimeError("Exact INT4 plane reconstruction failed")
    for group in (64, 128):
        for window in (0, group+9):
            for diagnostic in (False, True):
                data = build_case(Case("cpu_diagnostic" if diagnostic else "cpu_random",
                                      (3*group+11, 0), Layout(group), window=window, hq=4, hk=1))
                if diagnostic:
                    make_dominant_chunks(data)
                full = selective_attention(data, "parent")["output"]
                for policy in POLICIES:
                    for splits in (1, 4):
                        measured = selective_attention(data, policy, epsilon, splits=splits)
                        difference = np.linalg.norm(measured["output"]-full, axis=-1)
                        cert = measured["error_bound"]
                        vref = np.repeat(measured["value_reference"], 4, axis=1)
                        bounded = policy.startswith("bound_")
                        passed = bool(np.isfinite(measured["output"]).all())
                        if policy in ("parent", "repacked_full"):
                            passed &= bool(np.max(difference, initial=0) < 1e-11)
                        if bounded:
                            passed &= bool(np.all(difference <= cert+1e-10) and np.all(cert <= epsilon*vref+1e-10))
                        checks.append({"case": case_metadata(data), "policy": policy, "splits": splits,
                                       "max_attention_error_l2": float(difference.max(initial=0)),
                                       "max_declared_real_bound": float(cert.max(initial=0)) if bounded else None,
                                       "counts": measured["counts"], "passed": passed})
    if not all(x["passed"] for x in checks):
        raise RuntimeError("CPU representation/real-arithmetic budget check failed")
    return {"passed": True, "checks": checks, "generated_sha256": {p: h[DECODE_PATH] for p, h in hashes.items()},
            "all_files_sha256": hashes,
            "meaning": "Float64 representation and real-arithmetic budget checks, not a GPU floating-point certificate"}


def gpu_cases():
    layout = Layout(128)
    yield Case("pool_only", (1,), layout, hq=8, hk=2)
    yield Case("ragged_padded", (127, 129, 509, 0), layout, hq=8, hk=2)
    yield Case("exact_blocks", (256, 384), layout, hq=8, hk=2)
    yield Case("window_empty_splits", (653, 257, 0), layout, window=69, hq=8, hk=2)
    yield Case("long_random", (4096, 8191), layout, hq=4, hk=1)
    yield Case("indirect_verify", (385, 129, 386, 130, 0), layout,
               req_rows=(1, 0, 1, 0, 1), hq=8, hk=2)
    yield Case("dominant_chunks", (1024,), layout, hq=8, hk=2)
    yield Case("actual_head_geometry", (1536, 1023), layout, hq=32, hk=8)
    yield Case("group64_smoke", (193, 320), Layout(64), hq=8, hk=2)


def _torch_functions(source, names):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError("Missing pinned torch pack/unpack function")
    module = types.ModuleType("ar_gpu_validation_torch_functions")
    exec("import os\nimport torch\n", module.__dict__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<ar-verified-function-extract>", "exec"), module.__dict__)
    return module


def gpu_writer_checks(torch, sources, args):
    """Exercise both tile packers and the actual batched RTN writer functions."""
    names = {"_rtn_range", "_pack_4bit", "_pack_lowbit", "kvarn_store_tile_k_batch_from_sinkhorn",
             "kvarn_store_tile_v_batch_from_sinkhorn"}
    native = _torch_functions(sources[STORE_PATH], names)
    transformed = transform_sources(sources, "repacked_full", args.epsilon)
    planar = _torch_functions(transformed[STORE_PATH], names)
    unpack = _torch_functions(transformed[TORCH_DECODE_PATH], {"_unpack_4bit", "_unpack_lowbit"})
    generator = torch.Generator(device=args.device).manual_seed(82591)
    records = []
    for group in (64, 128):
        for rows, columns in ((128, group), (group, 128)):
            code = torch.randint(0, 16, (3, rows, columns), device=args.device, dtype=torch.int32, generator=generator)
            for packer in ("_pack_lowbit", "_pack_4bit"):
                p = getattr(planar, packer)(code, 4) if packer.endswith("lowbit") else getattr(planar, packer)(code)
                expected = pack_planes_numpy(code.cpu().numpy())
                restored = unpack._unpack_lowbit(p, columns, 4)
                passed = bool(np.array_equal(p.cpu().numpy(), expected) and torch.equal(restored, code.to(torch.uint8)))
                records.append({"kind": packer, "group": group, "rows": rows, "passed": passed})
        for side in ("k", "v"):
            rows, columns = (128, group) if side == "k" else (group, 128)
            balanced = torch.randn((3, rows, columns), device=args.device, generator=generator)
            col = torch.rand((3, columns), device=args.device, generator=generator)+0.5
            row = torch.rand((3, rows), device=args.device, generator=generator)+0.5
            name = f"kvarn_store_tile_{side}_batch_from_sinkhorn"
            a, b = getattr(native, name)(balanced, col, row, 4), getattr(planar, name)(balanced, col, row, 4)
            expected = parent_unpack_numpy(a['q_packed_uint8'].cpu().numpy(), columns)
            restored = unpack._unpack_lowbit(b['q_packed_uint8'], columns, 4).cpu().numpy()
            passed = bool(np.array_equal(expected, restored) and
                          all(torch.equal(a[key], b[key]) for key in a if key != 'q_packed_uint8'))
            records.append({"kind": name, "group": group, "passed": passed})
    torch.cuda.synchronize(args.device)
    if not all(r["passed"] for r in records):
        raise RuntimeError("GPU native-code/plane writer compatibility failed")
    return records


def gpu_consumer_checks(torch, modules, args):
    """Cover prefill materialization, legacy dequant, and dedicated full verify."""
    records = []
    for group in (64, 128):
        data = build_case(Case("full_read_consumers", (group+3, 3*group-1), Layout(group), hq=8, hk=2))
        state = GpuCase(data, args.device, splits=4)
        t, c = state.t, data['case']
        bt, sl, pk, pv = t['block_table'], t['seq_lens'], t['pool_k'], t['pool_v']
        cu = torch.tensor([0, c.seq_lens[0], sum(c.seq_lens)], device=args.device, dtype=torch.int32)
        shape = (sum(c.seq_lens), c.hk, c.layout.head_dim)
        ko = torch.zeros(shape, device=args.device, dtype=torch.float16)
        vo = torch.zeros_like(ko)
        outputs = {}
        for policy in POLICIES:
            mod = modules[policy]
            kv = t['cache'] if policy == 'parent' else state.planar
            mod._kvarn_build_packed_kv_kernel[(len(c.seq_lens)*bt.shape[1], c.hk)](
                bt, sl, cu, t['block_to_slot'], kv, pk, pv, ko, vo,
                bt.stride(0), kv.stride(0), kv.stride(1), pk.stride(0), pk.stride(1), pk.stride(2),
                ko.stride(0), ko.stride(1), MAX_BLOCKS_PER_REQ=bt.shape[1], D=128, GROUP=group,
                K_BITS=4, V_BITS=4, NUM_BLOCKS_LOOKUP=kv.shape[0], **c.layout.offsets,
                num_warps=4, num_stages=1)
            torch.cuda.synchronize(args.device)
            value = np.stack((ko.cpu().numpy(), vo.cpu().numpy()))
            if policy == 'parent':
                outputs['materialize'] = value.copy()
            err = error_metrics(value, outputs['materialize'])
            records.append({"kind": "build_packed_kv", "policy": policy, "group": group,
                            **err, "passed": within(err, .001, .002)})
        # The old standalone helper had missing bit-width symbols; all new
        # controls supply explicit INT4 defaults before exercising it here.
        bids = torch.tensor([0, 1], device=args.device, dtype=torch.int32)
        dst = torch.tensor([0, group], device=args.device, dtype=torch.int32)
        ko = torch.zeros((2*group, c.hk, 128), device=args.device, dtype=torch.float16)
        vo = torch.zeros_like(ko)
        for policy in POLICIES:
            kv = t['cache'] if policy == 'parent' else state.planar
            modules[policy]._kvarn_dequant_blocks_kernel[(2,c.hk)](
                kv,bids,dst,ko,vo,kv.stride(0),kv.stride(1),ko.stride(0),ko.stride(1),
                D=128,GROUP=group,**c.layout.offsets,num_warps=4)
            torch.cuda.synchronize(args.device)
            value = np.stack((ko.cpu().numpy(),vo.cpu().numpy()))
            if policy == 'parent':
                outputs['legacy'] = value.copy()
            err = error_metrics(value,outputs['legacy'])
            records.append({"kind":"dequant_blocks","policy":policy,"group":group,
                            **err,"passed":within(err,.001,.002)})
        del state, data
    # Uniform-QLEN verification uses a dedicated full-read kernel. Correct
    # source format here matters even though default full trials disable spec.
    data = build_case(Case('dedicated_verify', (511,512), Layout(), hq=8, hk=2, req_rows=(0,0)))
    state = GpuCase(data,args.device,splits=4)
    t,c = state.t,data['case']
    q,bt,sl,pk,pv = [t[k] for k in ('query','block_table','seq_lens','pool_k','pool_v')]
    native_output = None
    for policy in POLICIES:
        mod = modules[policy]
        kv = t['cache'] if policy=='parent' else state.planar
        obj = mod._kvarn_fused_verify_stage1
        obj = obj.fn if hasattr(obj,'configs') else obj
        obj[(1,c.hk,4)](q,bt,sl,t['block_to_slot'],kv,pk,pv,state.mid_o,state.mid_lse,data['scale'],
             c.hq*128,128,bt.stride(0),kv.stride(0),kv.stride(1),pk.stride(0),pk.stride(1),pk.stride(2),
             state.mid_o.stride(0),state.mid_o.stride(1),state.mid_lse.stride(0),
             MAX_BLOCKS_PER_REQ=bt.shape[1],D=128,GROUP=128,BLOCK_N=32,QLEN=2,Q_PER_KV=4,Q_PER_KV_PAD=4,
             HQ=c.hq,NUM_KV_SPLITS=4,SLIDING_WINDOW=0,K_BITS=4,V_BITS=4,NUM_BLOCKS_LOOKUP=kv.shape[0],
             **c.layout.offsets,num_warps=4,num_stages=1)
        mod._kvarn_fused_decode_stage2[(2*c.hq,)](state.mid_o,state.mid_lse,state.out,
             state.mid_o.stride(0),state.mid_o.stride(1),state.mid_lse.stride(0),state.out.stride(0),
             D=128,NUM_KV_SPLITS=4,num_warps=2)
        torch.cuda.synchronize(args.device)
        value=state.out.cpu().numpy().copy()
        if policy=='parent':
            native_output=value
        err=error_metrics(value,native_output)
        records.append({'kind':'dedicated_verify','policy':policy,'group':128,**err,'passed':within(err,.001,.002)})
    if not all(r['passed'] for r in records):
        raise RuntimeError('GPU packed-cache full-read consumer compatibility failed')
    return records


def gpu_checks(args, sources, result, path):
    import torch
    from vllm.triton_utils import triton
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU checks cannot authorize full experiments")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if props.major < 8:
        raise RuntimeError("AR v1 requires SM80 or newer")
    result['device']={'name':props.name,'capability':[props.major,props.minor],
        'sm_count':props.multi_processor_count,'total_memory':props.total_memory,
        'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,
        'triton_version':getattr(triton,'__version__',None),
        'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES'),
        'NVIDIA_TF32_OVERRIDE':os.getenv('NVIDIA_TF32_OVERRIDE'),
        'TRITON_F32_DEFAULT':os.getenv('TRITON_F32_DEFAULT')}
    from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
    for group in (64,128):
        actual=KVarNConfig.from_cache_dtype(f'kvarn_k4v4_g{group}',128)
        expected=Layout(group)
        if actual.tile_bytes_aligned != expected.tile_bytes or any(
                getattr(actual,k.lower())!=v for k,v in expected.offsets.items()):
            raise RuntimeError('Installed KVarN layout differs from reviewed fixture')
    result['writer_checks']=gpu_writer_checks(torch,sources,args)
    result['gpu_executed']=True
    atomic_json(path,result)
    modules={p:load_variant(args.source_root,p,args.epsilon) for p in POLICIES}
    result['consumer_checks']=gpu_consumer_checks(torch,modules,args)
    checks=result['gpu_checks']=[]
    def run_case(data,state,mode,autotune=False):
        parent=None
        full_reference=selective_attention(data,'parent')['output']
        for policy in POLICIES:
            measured=state.launch(modules[policy],mode,autotune).detach().cpu().numpy().copy()
            if policy=='parent':
                parent=measured
            # Autotune can choose a different BLOCK_N and therefore a different
            # admissible selection. Check it against its measured selected config.
            block_n=state.block_n
            if autotune:
                obj=getattr(modules[policy], '_kvarn_fused_decode_kernel' if mode=='single' else '_kvarn_fused_decode_stage1')
                config=getattr(obj,'best_config',None)
                if config is None:
                    raise RuntimeError('Autotuner did not expose its selected configuration')
                block_n=int(config.kwargs['BLOCK_N'])
            ref=selective_attention(data,policy,args.epsilon,block_n=block_n,
                                    splits=1 if mode=='single' else state.splits)
            err=error_metrics(measured,ref['output'])
            parent_err=error_metrics(measured,parent)
            passed=within(err,args.max_relative_l2,args.max_abs_error)
            if policy in ('parent','repacked_full'):
                passed &= within(parent_err,.002,.002)
            row={'case':case_metadata(data),'policy':policy,'mode':mode,'splits':1 if mode=='single' else state.splits,
                 'BLOCK_N':block_n,'autotune':autotune,'vs_float64_own_policy':err,
                 'vs_parent':parent_err,'cpu_selection_counts':ref['counts'],'passed':bool(passed)}
            if policy.startswith('bound_'):
                row['gpu_vs_real_budget']=budget_diagnostics(measured,parent,ref,full_reference,args.epsilon)
            if not autotune:
                row['gpu_logical_reads']=state.collect_stats(modules[policy],mode)
                actual_chunks=row['gpu_logical_reads']['quantized_chunks']
                # Instrumentation is validated on patterns away from thresholds.
                if data['case'].name=='dominant_chunks':
                    for key in ('key_skipped_chunks','value_skipped_chunks','quantized_chunks'):
                        row['passed'] &= row['gpu_logical_reads'][key]==ref['counts'][key]
                if policy=='anchor':
                    row['passed'] &= row['gpu_logical_reads']['key_skipped_chunks']==actual_chunks
                    row['passed'] &= row['gpu_logical_reads']['value_skipped_chunks']==actual_chunks
            checks.append(row)
            atomic_json(path,result)
            print(f"{data['case'].name} {mode} {policy} BN{block_n}: {'PASS' if row['passed'] else 'FAIL'}",flush=True)
    for case in gpu_cases():
        data=build_case(case)
        if case.name=='dominant_chunks':
            make_dominant_chunks(data)
        bns=[16,32,64] if case.name in ('ragged_padded','window_empty_splits','indirect_verify','dominant_chunks') else [32]
        for bn in bns:
            state=GpuCase(data,args.device,splits=32 if case.layout.group==128 else 64,block_n=bn)
            for mode in ('single','split'):
                run_case(data,state,mode)
            del state
        del data
    data=build_case(Case('autotune',tuple([1024]*8),Layout(),hq=32,hk=8))
    state=GpuCase(data,args.device,splits=32)
    for mode in ('single','split'):
        run_case(data,state,mode,True)
    del state,data
    result['autotune_executed']=True
    # Replay includes a query change that reverses dominant attention and
    # exercises runtime demand branches, rather than capturing a static mask.
    data=make_dominant_chunks(build_case(Case('cuda_graph_query_update',(1024,),Layout(),hq=8,hk=2)))
    original=data['query'].copy()
    updated=original.copy(); updated[...,3::4,0]=-1
    state=GpuCase(data,args.device,splits=32)
    queries={name:torch.from_numpy(value).to(args.device) for name,value in (('initial',original),('updated',updated))}
    graph_parents={}
    for policy in POLICIES:
        state.t['query'].copy_(queries['initial'])
        state.launch(modules[policy],'split')
        torch.cuda.synchronize(args.device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state.launch(modules[policy],'split')
        for revision,array in (('initial',original),('updated',updated)):
            state.t['query'].copy_(queries[revision])
            graph.replay(); torch.cuda.synchronize(args.device)
            measured=state.out.view(state.b,state.hq,state.d).cpu().numpy().copy()
            data['query']=array
            reference=selective_attention(data,policy,args.epsilon,splits=32)
            err=error_metrics(measured,reference['output'])
            if policy=='parent':
                graph_parents[revision]=measured
            graph_row={'case':case_metadata(data),'policy':policy,'mode':'split','splits':32,'BLOCK_N':32,
                           'autotune':False,'cuda_graph_replay':True,'input_revision':revision,
                           'vs_float64_own_policy':err,'passed':within(err,args.max_relative_l2,args.max_abs_error)}
            if policy.startswith('bound_'):
                full_reference=selective_attention(data,'parent')['output']
                graph_row['gpu_vs_real_budget']=budget_diagnostics(measured,graph_parents[revision],reference,full_reference,args.epsilon)
            checks.append(graph_row)
            atomic_json(path,result)
        del graph
    # Instrumented graph independently confirms query-dependent load decisions.
    # This separate diagnostic always uses .02, because epsilon=0 or very large
    # valid user budgets need not change masks. The six main graph tests above
    # still use the actual declared experiment epsilon.
    diagnostic_epsilon=.02
    diagnostic_module=load_variant(args.source_root,'bound_kv',diagnostic_epsilon)
    state.t['query'].copy_(queries['initial']); state.stats.zero_()
    state.launch(diagnostic_module,'split',collect_stats=True)
    torch.cuda.synchronize(args.device)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.launch(diagnostic_module,'split',collect_stats=True)
    observed={}
    for revision in ('initial','updated'):
        state.t['query'].copy_(queries[revision]); graph.replay(); torch.cuda.synchronize(args.device)
        observed[revision]=[int(x) for x in state.stats.sum(dim=0).cpu().tolist()]
    if observed['initial'][8]<=observed['updated'][8] or observed['initial'][9]<=observed['updated'][9]:
        raise RuntimeError('CUDA graph did not change residual demand when query changed')
    result['graph_branch_counts']={'epsilon':diagnostic_epsilon,'policy':'bound_kv','counts':observed,
        'generated_sha256':{p:sha_text(text) for p,text in transform_sources(sources,'bound_kv',diagnostic_epsilon).items()},
        'scope':'Fixed-budget synthetic branch diagnostic; main six-policy graph results retain requested epsilon'}
    result.update(cuda_graph_replay_executed=True,graph_updated_input_checked=True,
                  graph_updated_demand_checked=True,packed_consumers_checked=True,packed_writers_checked=True)
    if not all(row['passed'] for row in checks):
        raise RuntimeError('CUDA numerical/representation gate failed; full experiments remain blocked')
    result['tested_policies']=sorted(POLICIES)
    result['tested_value_bits']=[4]
    result['tested_groups']=[64,128]
    result['tested_modes']=['single','split']
    result['tested_block_ns']=[16,32,64]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True,help='Original unmodified installed vllm package directory')
    parser.add_argument('--epsilon',type=float,default=.02)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--max-relative-l2',type=float,default=.02)
    parser.add_argument('--max-abs-error',type=float,default=.02)
    args=parser.parse_args()
    if not math.isfinite(args.epsilon) or args.epsilon<0 or args.max_relative_l2<=0 or args.max_abs_error<=0:
        parser.error('Invalid numerical budget or thresholds')
    started=time.time(); path=Path(args.out)/'validation.json'
    result={'schema':'kvbench.residual.validation.v1','status':'running','passed':False,'gpu_executed':False,
            'source_root':str(Path(args.source_root).resolve()),'source':str(Path(args.source_root).resolve()/DECODE_PATH),
            'parent_sha256':PARENT_SHA256,'parent_pins':get_expected_pins(),'epsilon':args.epsilon,
            'python':platform.python_version(),'common_fixes':list(COMMON_FIXES),
            'tested_policies':[],'tested_value_bits':[],
            'limits':{'relative_l2':args.max_relative_l2,'max_abs_error':args.max_abs_error},
            'scope':'Synthetic complete-code fixtures and real-arithmetic bound oracle; not model accuracy or a formal GPU floating-point certificate',
            'attention_error_reference':'The same complete KVarN K4V4 representation; all policies preserve all token contributions'}
    atomic_json(path,result)
    try:
        sources=read_sources(args.source_root)
        result['generated_sha256']=generated_receipt(sources,args.epsilon)
        result['cpu']=cpu_checks(sources,args.epsilon)
        if args.cpu_only:
            result.update(status='cpu_only',passed=False,cpu_passed=True)
        else:
            gpu_checks(args,sources,result,path)
            result.update(status='passed',passed=True)
        result['elapsed_s']=time.time()-started
        atomic_json(path,result)
        print(f"Validation status={result['status']}; {path}",flush=True)
    except BaseException as exc:
        result.update(status='failed',passed=False,error=str(exc),traceback=traceback.format_exc(),elapsed_s=time.time()-started)
        atomic_json(path,result)
        raise


if __name__=='__main__':
    main()

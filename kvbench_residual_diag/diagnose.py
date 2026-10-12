"""Experiment 1: actual GPU branch counts on real saved decode inputs.

The original sampling run is not instrumented for timing. CPU explanations and
GPU branch counts are explicitly different fields, with mismatch rates reported.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import traceback
import numpy as np
from .common import atomic_json, load_trace, new_output, output_metrics, trace_paths, write_csv
from .math_ops import build_group, bound_policy


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-root",type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--cpu-only",action="store_true")
    p.add_argument("--epsilons",nargs="+",type=float,default=[.02])
    p.add_argument("--policies",nargs="+",choices=["bound_k","bound_v","bound_kv"],default=["bound_k","bound_v","bound_kv"])
    p.add_argument("--block-n",type=int,choices=[16,32,64],help="Default: recorded parent BLOCK_N")
    p.add_argument("--splits",nargs="+",type=int,help="Default: recorded parent split count; use 1 32 for ablation")
    p.add_argument("--trace-limit",type=int,default=0)
    a=p.parse_args(argv)
    if any(not np.isfinite(e) or e<0 for e in a.epsilons): p.error("Finite nonnegative epsilons required")
    if a.splits and any(x<1 or x&(x-1) for x in a.splits): p.error("Splits must be powers of two")
    paths=trace_paths(a.traces,a.trace_limit)
    out=new_output(a.out)
    state={"schema":"kvbench.residual.real_diagnosis.v1","status":"running", "gpu_executed":False,
           "source_kind":"real_vllm_decode","trace_count":len(paths),"timings_available":False,
           "dram_bytes":None,"scope":"GPU replay branch counts are logical address bytes, NOT DRAM. CPU bounds are float64 explanations, NOT task quality."}
    atomic_json(out/"status.json",state)
    rows=[]; validations=[]
    try:
        modules={}
        if not a.cpu_only:
            import torch
            if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; --cpu-only is an explicit partial alternative, not a GPU result")
            from kvbench_residual.fixtures import GpuCase
            from kvbench_residual.transform import load_variant
            if a.source_root is None:
                spec=importlib.util.find_spec("vllm")
                if spec is None or not spec.origin: raise RuntimeError("Use the existing kvarn-serve environment")
                a.source_root=Path(spec.origin).parent
            parent=load_variant(a.source_root,"parent",.02)
        with (out/"chunk_decisions.jsonl").open("w",encoding="utf-8") as detail_file:
            for ti,path in enumerate(paths):
                data,meta=load_trace(path)
                bn=a.block_n or meta["block_n"]
                split_values=a.splits or [meta["splits"]]
                groups=[build_group(data,h,bn) for h in range(8)]
                full=np.concatenate([g.full for g in groups],axis=0)[None,:,:]
                live=output_metrics(full,data["reference_gpu_rot"])
                validation={"trace":path.name,"live_gpu_vs_float64":live}
                if live["relative_l2_fro"] is None or live["relative_l2_fro"]>.02 or live["max_abs_element"]>.02:
                    raise RuntimeError(f"Full quantized reference mismatch on {path.name}: {live}; do not interpret selective results")
                if not a.cpu_only:
                    # Verify original parent geometry against its original rotated output.
                    original=GpuCase(data,a.device,meta["splits"],meta["block_n"],meta["num_warps"],meta["num_stages"])
                    original.launch(parent,mode=meta["mode"],autotune=False)
                    torch.cuda.synchronize()
                    check=output_metrics(original.out.view(1,32,128).cpu().numpy(),data["reference_gpu_rot"])
                    validation["replayed_parent_vs_live_gpu"]=check
                    if check["relative_l2_fro"] is None or check["relative_l2_fro"]>.02 or check["max_abs_element"]>.02:
                        raise RuntimeError(f"GPU parent replay mismatch: {path.name}: {check}")
                    del original
                validations.append(validation)
                atomic_json(out/"validation.json",validations)
                for splits in split_values:
                    mode="single" if splits==1 else "split"
                    gpu=None
                    if not a.cpu_only:
                        gpu=GpuCase(data,a.device,splits,bn,meta["num_warps"],meta["num_stages"])
                    for epsilon in a.epsilons:
                        for policy in a.policies:
                            results=[bound_policy(g,policy,epsilon,splits) for g in groups]
                            predicted=np.concatenate([r["output"] for r in results],axis=0)[None,:,:]
                            row={"trace":path.name,"dataset":meta["request"]["dataset"],
                                 "case_id":meta["request"]["case_id"],"cache_ordinal":meta["cache_ordinal"],
                                 "decode_call_index":meta["decode_call_index"],"context_tokens":meta["context_tokens"],
                                 "block_n":bn,"splits":splits,"policy":policy,"epsilon":epsilon,
                                 "quantized_chunks":sum(r["quantized_chunks"] for r in results),
                                 "cpu_key_skipped_chunks":sum(r["key_skipped_chunks"] for r in results),
                                 "cpu_value_skipped_chunks":sum(r["value_skipped_chunks"] for r in results),
                                 "cpu_logical_code_bytes":sum(r["logical_code_bytes"] for r in results),
                                 "native_code_bytes":sum(r["native_code_bytes"] for r in results)}
                            row.update({"cpu_"+k:v for k,v in output_metrics(predicted,full).items()})
                            all_details=[d for r in results for d in r["details"]]
                            for side in ("k","v"):
                                row[side+"_gqa_blocked_chunks"]=sum(d[side+"_gqa_prevented_skip"] is True for d in all_details)
                                v=[d[side+"_threshold_ratio_max"] for d in all_details if d[side+"_threshold_ratio_max"] is not None]
                                row[side+"_threshold_ratio_p50"]=float(np.median(v)) if v else None
                                row[side+"_threshold_ratio_p95"]=float(np.quantile(v,.95)) if v else None
                            for d in all_details:
                                detail_file.write(json.dumps({"trace":path.name,"policy":policy,"epsilon":epsilon,"splits":splits,"block_n":bn,**d},allow_nan=False)+"\n")
                            if gpu is not None:
                                key=(policy,epsilon)
                                if key not in modules: modules[key]=load_variant(a.source_root,policy,epsilon)
                                gpu.stats.zero_()
                                gpu.launch(modules[key],mode=mode,autotune=False,collect_stats=True)
                                torch.cuda.synchronize()
                                from kvbench_residual.kernel import STATS_FIELDS
                                nr=8 if mode=="single" else 8*splits
                                stat_array=gpu.stats[:nr].cpu().numpy()
                                stat={k:int(v) for k,v in zip(STATS_FIELDS,stat_array.sum(axis=0))}
                                code=sum(stat[x] for x in ("key_anchor_code_bytes","key_residual_code_bytes","value_anchor_code_bytes","value_residual_code_bytes"))
                                row.update({"gpu_executed":True,"gpu_key_skipped_chunks":stat["key_skipped_chunks"],
                                    "gpu_value_skipped_chunks":stat["value_skipped_chunks"],"gpu_logical_code_bytes":code,
                                    "gpu_logical_code_reduction_fraction":1-code/stat["native_quantized_code_bytes"] if stat["native_quantized_code_bytes"] else None,
                                    "gpu_scale_logical_bytes":stat["scale_logical_bytes"],"gpu_fp16_pool_logical_bytes":stat["pool_kv_bytes"],
                                    "cpu_gpu_count_match":stat["key_skipped_chunks"]==row["cpu_key_skipped_chunks"] and stat["value_skipped_chunks"]==row["cpu_value_skipped_chunks"]})
                                actual=gpu.out.view(1,32,128).cpu().numpy()
                                row.update({"gpu_vs_full_"+k:v for k,v in output_metrics(actual,full).items()})
                                policy_check=output_metrics(actual,predicted)
                                row.update({"gpu_vs_cpu_policy_"+k:v for k,v in policy_check.items()})
                                row["gpu_policy_numerics_passed"]=(policy_check["relative_l2_fro"] is not None
                                    and policy_check["relative_l2_fro"]<=.02 and policy_check["max_abs_element"]<=.02)
                                # Save per-GQA-group/split counters, not only the aggregate.
                                safe_e=str(epsilon).replace(".","p")
                                np.save(out/f"counts_{ti:04d}_{policy}_e{safe_e}_s{splits}_n{bn}.npy",stat_array,allow_pickle=False)
                                state["gpu_executed"]=True
                            else:
                                row.update(gpu_executed=False,gpu_key_skipped_chunks=None,gpu_value_skipped_chunks=None,
                                           gpu_logical_code_bytes=None,gpu_logical_code_reduction_fraction=None,cpu_gpu_count_match=None)
                            rows.append(row)
                            write_csv(out/"summary.csv",rows)
                            if row.get("gpu_policy_numerics_passed") is False:
                                raise RuntimeError("Selective GPU kernel differs from the CPU policy reference; inspect summary.csv")
                    if gpu is not None: del gpu
                print(f"diagnose {ti+1}/{len(paths)}: {path.name}",flush=True)
        state.update(status="completed_cpu_only" if a.cpu_only else "completed",rows=len(rows),
                     cpu_gpu_count_mismatch_rows=sum(r.get("cpu_gpu_count_match") is False for r in rows))
        atomic_json(out/"status.json",state)
        lines=["# 实验一：真实 decode 跳读诊断","",f"状态：{state['status']}；真实 trace：{len(paths)}；结果行：{len(rows)}。",
               "","这些是 Parent 轨迹上相同 Query/KV 的反事实重放，不是 bound 策略自然生成全程的跳读率。",
               "没有 TPOT、吞吐或 DRAM 实测。GPU 字节数来自实际分支逻辑地址计数。",
               "","## 读数方法","",
               "`summary.csv`：先看 gpu_executed，再看 gpu_key/value_skipped_chunks 与 gpu_logical_code_reduction_fraction。",
               "`chunk_decisions.jsonl`：threshold_ratio > 1 表示该头/组没有通过预算；查看 bound/actual、mass 上界和 GQA 阻断。",
               "`counts_*.npy`：列名顺序是 kvbench_residual.kernel.STATS_FIELDS，保存每个 KV head/split 的 GPU 计数。",
               "`validation.json`：完整量化 float64 输出与真实 GPU 输出、Parent 重放对照。CPU/GPU 分支不同必须检查边界，不能悄悄覆盖计数。",
               "","建议用 --splits 1 32 在相同 trace 比较 split 保守性的影响；这是数学/分支消融，不是调度性能结论。"]
        (out/"readout.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    except BaseException as exc:
        state.update(status="failed",error=repr(exc),traceback=traceback.format_exc(),completed_rows=len(rows))
        atomic_json(out/"status.json",state)
        if rows: write_csv(out/"summary.csv",rows)
        raise


if __name__=="__main__": main()

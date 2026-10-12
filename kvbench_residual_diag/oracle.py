"""Experiment 2: offline counterfactual error-versus-read-byte curves.

Greedy residual-aware selection is an achievable reference, NOT a global
optimum. V-only also receives a valid lower bound; tiny groups can be solved
exhaustively. No task accuracy, DRAM savings or speed is inferred from the curves.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
from .common import atomic_json,load_trace,new_output,output_metrics,trace_paths,write_csv
from .math_ops import (build_group,evaluate,action_setup,select_order,apply_budget,
                       v_lower_bound,exact_small,bound_policy,objective)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--modes",nargs="+",choices=["k","v","kv"],default=["v","kv"])
    p.add_argument("--fractions",nargs="+",type=float,default=[0,.1,.25,.5,.75,1])
    p.add_argument("--kv-heads",nargs="+",type=int,default=[0,3,7])
    p.add_argument("--trace-limit",type=int,default=12,help="0=all; positive=evenly spread acquisition indices")
    p.add_argument("--block-n",type=int,choices=[16,32,64],help="Default recorded BLOCK_N")
    p.add_argument("--random-seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--exact-max-actions",type=int,default=14)
    p.add_argument("--bound-epsilon",type=float,default=.02)
    a=p.parse_args(argv)
    if any(not np.isfinite(f) or not 0<=f<=1 for f in a.fractions): p.error("Read fractions must be in [0,1]")
    if not a.kv_heads or len(set(a.kv_heads))!=len(a.kv_heads) or any(h<0 or h>7 for h in a.kv_heads): p.error("KV heads must be 0..7")
    if not 0<=a.exact_max_actions<=20: p.error("Exhaustive solver is limited to 20 actions")
    if not np.isfinite(a.bound_epsilon) or a.bound_epsilon<0: p.error("Invalid bound epsilon")
    paths=trace_paths(a.traces,a.trace_limit)
    out=new_output(a.out)
    (out/"masks").mkdir()
    rows=[];coverage=[]
    state={"schema":"kvbench.residual.real_oracle.v1","status":"running", "trace_count":len(paths),
           "trace_selection":"evenly_spaced_acquisition_indices" if a.trace_limit else "all",
           "kv_heads":a.kv_heads,"modes":a.modes,"gpu_executed":False,
           "scope":"Float64 attention-output fidelity on fixed real Parent Q/KV. Greedy is feasible, not optimal. No model accuracy/speed or DRAM claim."}
    atomic_json(out/"status.json",state)
    try:
        for ti,path in enumerate(paths):
            data,meta=load_trace(path)
            bn=a.block_n or meta["block_n"]
            for hk in a.kv_heads:
                group=build_group(data,hk,bn)
                # Group-specific original full reference verification.
                live=data["reference_gpu_rot"][0,hk*4:(hk+1)*4]
                vm=output_metrics(group.full,live)
                if vm["relative_l2_fro"] is None or vm["relative_l2_fro"]>.02 or vm["max_abs_element"]>.02:
                    raise RuntimeError(f"Full reference mismatch on {path.name}, KV head {hk}: {vm}")
                for mode in a.modes:
                    _,_,actions=action_setup(group,mode)
                    controllable=int(sum(group.costs[i,s] for i,s in actions))
                    coverage.append({"trace":path.name,"kv_head":hk,"mode":mode,"actions":len(actions),
                                     "controllable_residual_bytes":controllable,"exact_eligible":len(actions)<=a.exact_max_actions})
                    methods=[("attention",select_order(group,mode,"attention")),("offline_greedy",select_order(group,mode,"greedy"))]
                    methods += [(f"random_seed_{seed}",select_order(group,mode,"random",seed)) for seed in a.random_seeds]
                    for fraction in sorted(set(a.fractions)):
                        budget=int(np.floor(controllable*fraction+1e-9))
                        lb=v_lower_bound(group,budget) if mode=="v" else None
                        exact=exact_small(group,mode,budget,a.exact_max_actions)
                        candidates=[]
                        for name,order in methods:
                            k,v,spent=apply_budget(group,mode,order,budget)
                            candidates.append((name,k,v,spent))
                        if exact is not None:
                            _,k,v,spent=exact
                            candidates.append(("exhaustive_exact_small",k,v,spent))
                        # Minimum across feasible methods is still only an upper bound on optimal error.
                        best=min(candidates,key=lambda x:objective(evaluate(group,x[1],x[2]),group.full))
                        candidates.append(("best_feasible_envelope",best[1],best[2],best[3]))
                        for name,k,v,spent in candidates:
                            row=make_row(group,meta,path.name,hk,mode,name,k,v,spent,controllable)
                            row.update(requested_fraction=fraction,budget_bytes=budget,
                                       certified_v_error_lower_bound=lb,
                                       exact_optimum_relative_error=exact[0] if exact is not None else None,
                                       exact_solver_status="solved" if exact is not None else "not_attempted_too_many_actions",
                                       selected_envelope_method=best[0] if name=="best_feasible_envelope" else None)
                            metrics=output_metrics(evaluate(group,k,v),group.full,group.vref)
                            row.update(metrics)
                            row["selection_objective_norm_floor"]=1e-12
                            row["selection_objective_value"]=objective(evaluate(group,k,v),group.full)
                            if lb is not None and row["relative_l2_max_head"] is not None and lb>row["relative_l2_max_head"]+1e-9:
                                raise AssertionError("Lower bound exceeds a feasible solution")
                            filename=f"t{ti:04d}_h{hk}_{mode}_{name}_b{budget}.npz"
                            np.savez(out/"masks"/filename,read_k=k,read_v=v,
                                     logical_blocks=np.array([c["block"] for c in group.chunks]),
                                     chunk_offsets=np.array([c["chunk"] for c in group.chunks]),
                                     quantized=group.quantized)
                            row["mask_file"]="masks/"+filename
                            rows.append(row)
                    # Compare the actual v1 bound at its own measured logical read point.
                    policy={"k":"bound_k","v":"bound_v","kv":"bound_kv"}[mode]
                    b=bound_policy(group,policy,a.bound_epsilon,meta["splits"])
                    k,v=b["read_k"],b["read_v"]
                    spent=int(sum(group.costs[i,s]*(k[i] if s==0 else v[i]) for i,s in actions))
                    row=make_row(group,meta,path.name,hk,mode,"current_bound_cpu",k,v,spent,controllable)
                    row.update(output_metrics(b["output"],group.full,group.vref))
                    row.update(epsilon=a.bound_epsilon,splits=meta["splits"],requested_fraction=None,
                               budget_bytes=None,certified_v_error_lower_bound=None,
                               exact_optimum_relative_error=None,exact_solver_status="not_applicable",mask_file=None)
                    rows.append(row)
                    write_csv(out/"curves.csv",rows)
            print(f"oracle {ti+1}/{len(paths)}: {path.name}",flush=True)
        write_csv(out/"coverage.csv",coverage)
        atomic_json(out/"coverage.json",coverage)
        summary=summarize(rows)
        write_csv(out/"aggregate.csv",summary)
        state.update(status="completed",rows=len(rows),groups=len(coverage),
                     exact_eligible_groups=sum(c["exact_eligible"] for c in coverage))
        atomic_json(out/"status.json",state)
        lines=["# 实验二：真实 trace 的读取量—attention 输出误差","",
            f"Trace：{len(paths)}；KV heads：{a.kv_heads}；结果行：{len(rows)}。默认只是显式子样本，不代表所有层/请求。",
            "","`curves.csv` 保留每个 trace/GQA 组，`aggregate.csv` 为组等权统计，不是模型总体准确率。",
            "K-only 固定 V 全读；V-only 固定 K 全读；KV 联合决定两类低位读取。所有 token 的 Anchor 始终保留，sink/tail 始终完整。",
            "`controllable_read_fraction` 仅指本实验可控制的 Residual；`all_residual_read_fraction` 包含固定全读的另一侧。",
            "`logical_code_reduction_fraction` 只统计量化码；scale/FP16 pool 不变且未含其中。不是 DRAM 节省。",
            "","## Oracle 的准确含义","",
            "offline_greedy 使用完整 Residual 做离线一步前瞻，不可部署，也不保证最优；记录 best-feasible-prefix，以处理残差抵消导致的非单调。",
            "best_feasible_envelope 是各可行算法在相同预算下的最佳误差，因而是最优误差的上界，不是信息论下界。",
            "V-only 的 certified_v_error_lower_bound 使用三角不等式和分数背包松弛；为 0 时通常没有判别力。",
            "exhaustive_exact_small 只在动作数不超过阈值的完整 GQA 组枚举所有子集。不裁短序列伪装原序列最优。",
            "K/KV 的大规模精确下界本补丁没有实现；相应字段为 NA。",
            "","## 判读","",
            "可行方案在低读取比例达到低误差：说明存在可实现选择空间，但还须设计廉价在线控制器。",
            "有效下界仍明显高于目标误差：可排除该 V-only 读取预算。",
            "贪心表现差、下界为 0：证据不足，不能据此否定方法。",
            "误差相对完整量化 attention 输出计算；不等同 FP16 模型误差、任务得分、生成长度或 TPOT。",
            "","| 模式 | 策略 | 预算比例 | 组数 | 平均实际可控残差读取 | 平均码字节减少 | 最大头相对误差中位数 |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for r in summary:
            if r["method"] not in ("best_feasible_envelope","current_bound_cpu"): continue
            def fmt(x): return "NA" if x is None else f"{x:.5f}"
            lines.append(f"| {r['mode']} | {r['method']} | {fmt(r['requested_fraction'])} | {r['groups']} | {fmt(r['mean_controllable_read_fraction'])} | {fmt(r['mean_logical_code_reduction_fraction'])} | {fmt(r['median_relative_l2_max_head'])} |")
        (out/"readout.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    except BaseException as exc:
        state.update(status="failed",error=repr(exc),traceback=traceback.format_exc(),completed_rows=len(rows))
        atomic_json(out/"status.json",state)
        if rows: write_csv(out/"curves.csv",rows)
        raise


def make_row(group,meta,trace,hk,mode,name,k,v,spent,controllable):
    costs=group.costs
    all_full=int(costs.sum())
    all_read=int((costs[:,0]*k+costs[:,1]*v).sum())
    return {"trace":trace,"dataset":meta["request"]["dataset"],"case_id":meta["request"]["case_id"],
        "cache_ordinal":meta["cache_ordinal"],"decode_call_index":meta["decode_call_index"],
        "context_tokens":meta["context_tokens"],"kv_head":hk,"gqa_heads":group.full.shape[0],
        "mode":mode,"method":name,"controllable_residual_bytes":controllable,"read_controllable_bytes":spent,
        "controllable_read_fraction":spent/controllable if controllable else None,
        "all_residual_bytes":all_full,"read_all_residual_bytes":all_read,
        "all_residual_read_fraction":all_read/all_full if all_full else None,
        "logical_code_bytes":all_full+all_read,"full_planar_code_bytes":2*all_full,
        "logical_code_reduction_fraction":1-(all_full+all_read)/(2*all_full) if all_full else None,
        "quantized_chunks":int(group.quantized.sum()),"vref":group.vref}


def summarize(rows):
    groups={}
    for r in rows:
        key=(r["mode"],r["method"],r.get("requested_fraction"))
        groups.setdefault(key,[]).append(r)
    result=[]
    for (mode,method,fraction),rs in groups.items():
        def vals(k): return [r[k] for r in rs if r.get(k) is not None]
        def mean(k):
            x=vals(k);return float(np.mean(x)) if x else None
        x=vals("relative_l2_max_head")
        result.append({"mode":mode,"method":method,"requested_fraction":fraction,"groups":len(rs),
                       "mean_controllable_read_fraction":mean("controllable_read_fraction"),
                       "mean_logical_code_reduction_fraction":mean("logical_code_reduction_fraction"),
                       "median_relative_l2_max_head":float(np.median(x)) if x else None,
                       "p95_relative_l2_max_head":float(np.quantile(x,.95)) if x else None,
                       "mean_abs_l2_max":mean("abs_l2_max"),
                       "mean_certified_v_error_lower_bound":mean("certified_v_error_lower_bound")})
    return result


if __name__=="__main__": main()

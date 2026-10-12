"""Capture real Parent decode inputs, stop our server, run both diagnostic experiments.

Reuse frozen LongBench inputs and the existing server environment. The default
C1/eager/96-token prefix is a diagnostic protocol, not the prior C8 quality run.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import uuid
import numpy as np
from .common import atomic_json,new_output,sha_file,trace_paths,write_csv

DEFAULT_SUITE="results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json"


def choose_samples(datasets,tasks,count,min_context,max_tokens,window):
    selected=[]
    for task in tasks:
        if task not in datasets: raise ValueError("Task absent from frozen data: "+task)
        eligible=[r for r in datasets[task] if r["input_tokens"]>=min_context and r["input_tokens"]+max_tokens<=window]
        eligible=sorted(eligible,key=lambda r:(r["input_tokens"],str(r["case_id"])))
        if len(eligible)<count:
            raise ValueError(f"Only {len(eligible)} eligible {task} cases, requested {count}; no clipping/substitution")
        indices=np.linspace(0,len(eligible)-1,count+2)[1:-1].round().astype(int)
        if len(set(indices))!=count: raise ValueError("Sampling indices are not unique")
        selected.extend(eligible[i] for i in indices)
    return selected


def ready(proc,base,config,timeout):
    from kvbench_longcontext.serve import http_text,verify_cache,verify_model
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if proc.poll() is not None: raise RuntimeError("Diagnostic server exited; inspect server/server.log")
        try:
            models=json.loads(http_text(base+"/v1/models"))
            metrics=http_text(base+"/metrics")
        except (OSError,ValueError):
            time.sleep(1);continue
        return {"model":verify_model(models,config),"cache":verify_cache(metrics,config)}
    raise TimeoutError("Diagnostic server startup timed out; inspect server/server.log")


def request_one(base,model,row,settings,cap,dest):
    from kvbench_longcontext.run import make_request,stream_request
    row=dict(row)
    row["max_tokens"]=cap
    request=make_request(row,model,thinking=settings.get("thinking",True),
                         temperature=settings.get("temperature",.6),top_p=settings.get("top_p",.95),
                         seed=settings.get("request_seed",20261009),top_k=settings.get("top_k",20),
                         min_p=settings.get("min_p",0.0))
    result=stream_request(base,row,request,dest,timeout=0,thinking=settings.get("thinking",True),
                          native_max_model_len=settings["max_model_len"])
    if not result["protocol_valid"]:
        raise RuntimeError("Diagnostic request failed native protocol checks: "+str({k:result.get(k) for k in ("case_id","error","response_complete","prompt_token_match")}))
    # Intentionally remove wall-clock TPOT/E2E fields; captures invalidate them.
    keep=("dataset","case_id","finish_reason","completion_tokens","prompt_tokens","response_complete","protocol_valid","usage","request","raw_chunks_file","raw_chunks_sha256")
    return {k:result[k] for k in keep}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-suite",type=Path,default=Path(DEFAULT_SUITE))
    p.add_argument("--out",type=Path)
    p.add_argument("--tasks",nargs="+",default=["qasper","hotpotqa","passage_retrieval_en"])
    p.add_argument("--per-task",type=int,default=2)
    p.add_argument("--decode-calls",nargs="+",type=int,default=[0,15,63])
    p.add_argument("--layer-ordinals",nargs="+",type=int)
    p.add_argument("--diagnostic-max-tokens",type=int,default=96)
    p.add_argument("--min-context",type=int,default=1024)
    p.add_argument("--max-traces",type=int,default=256)
    p.add_argument("--max-gib",type=float,default=8)
    p.add_argument("--gpu",default="0")
    p.add_argument("--port",type=int,default=18080)
    p.add_argument("--startup-timeout",type=int,default=1800)
    p.add_argument("--epsilons",nargs="+",type=float,default=[.02])
    p.add_argument("--oracle-trace-limit",type=int,default=12)
    p.add_argument("--oracle-modes",nargs="+",choices=["k","v","kv"],default=["v","kv"])
    p.add_argument("--capture-only",action="store_true")
    p.add_argument("--plan",action="store_true")
    a=p.parse_args(argv)
    if min(a.per_task,a.diagnostic_max_tokens,a.min_context,a.max_traces,a.startup_timeout)<1: p.error("Counts must be positive")
    if any(x<0 or x>=a.diagnostic_max_tokens-1 for x in a.decode_calls): p.error("Decode-call indices need room for the initial prefill token")
    if not a.decode_calls or len(set(a.decode_calls))!=len(a.decode_calls): p.error("Distinct decode-call indices required")
    if not np.isfinite(a.max_gib) or a.max_gib<=0: p.error("max-gib must be positive finite")
    if not a.gpu.isdigit() or not 1<=a.port<=65535: p.error("Use one numeric GPU and a valid port")
    if len(set(a.tasks))!=len(a.tasks): p.error("Duplicate tasks")
    if a.oracle_trace_limit<0 or any(not np.isfinite(e) or e<0 for e in a.epsilons): p.error("Invalid analysis setting")
    suite_path=a.upstream_suite.resolve(strict=True)
    upstream=json.loads(suite_path.read_text())
    if upstream.get("schema")!="kvbench.longcontext.full_suite.v1" or upstream.get("fatal_error"):
        raise ValueError("Use the successful frozen V3 suite.json, not the residual suite")
    if not upstream.get("methods") or any(m.get("state")!="completed" for m in upstream["methods"]):
        raise ValueError("Upstream suite is not complete; diagnostic capture does not wait for/stop it")
    s=upstream["settings"]
    from kvbench_longcontext.common import checked_data,tokenizer_identity,digest,package_versions
    from kvbench_residual.serve import validate_architecture
    data_dir=Path(s["data"]).resolve(strict=True)
    model=Path(s["model"]).resolve(strict=True)
    arch=validate_architecture(model)
    manifest,datasets=checked_data(data_dir)
    if manifest["settings"]["max_model_len"]!=s["max_model_len"]: raise ValueError("Frozen context window mismatch")
    samples=choose_samples(datasets,a.tasks,a.per_task,a.min_context,a.diagnostic_max_tokens,s["max_model_len"])
    nlayers=arch.get("num_hidden_layers")
    if not isinstance(nlayers,int) or nlayers<1: raise ValueError("Cannot infer diagnostic layer-ordinal sampling range")
    layers=a.layer_ordinals or sorted(set([0,nlayers//3,2*nlayers//3,nlayers-1]))
    if any(x<0 or x>=nlayers for x in layers) or len(set(layers))!=len(layers): raise ValueError("Invalid layer ordinal sampling")
    planned=len(samples)*len(layers)*len(a.decode_calls)
    if planned>a.max_traces: raise ValueError("Planned captures exceed --max-traces")
    plan={"schema":"kvbench.residual.real_diagnostics_plan.v1","upstream_suite":str(suite_path),
          "upstream_sha256":sha_file(suite_path),"data_manifest_sha256":sha_file(data_dir/"manifest.json"),
          "model":str(model),"architecture":arch,"tasks":a.tasks,"samples":[{k:r[k] for k in ("dataset","case_id","input_tokens")} for r in samples],
          "layer_cache_ordinals":layers,"decode_call_indices":a.decode_calls,"planned_upper_bound_traces":planned,
          "diagnostic_max_tokens":a.diagnostic_max_tokens,"batch":1,"eager":True,"port":a.port,"gpu":a.gpu,
          "generation_note":"Same sampling/thinking as upstream, but C1 eager and limited diagnostic prefixes. No task score, TPOT or natural full-length conclusion.",
          "oracle_note":"Default uses at most 12 evenly spaced traces and KV heads 0,3,7; see coverage.csv. Offline greedy is not a global optimum."}
    if a.plan:
        print(json.dumps(plan,indent=2,ensure_ascii=False));return
    from kvbench_longcontext.serve import launch_command,run_text
    from kvbench_residual.serve import stop_owned_group
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1",a.port))==0: raise RuntimeError("Port occupied; no existing process is stopped")
    gpu_info=run_text(["nvidia-smi","-i",a.gpu,"--query-compute-apps=pid,process_name","--format=csv,noheader,nounits"])
    if gpu_info.get("returncode")!=0: raise RuntimeError("Cannot inspect target GPU: "+str(gpu_info))
    if gpu_info["stdout"] and "No running processes found" not in gpu_info["stdout"]:
        raise RuntimeError("Target GPU already has compute processes; finish them before diagnostics: "+gpu_info["stdout"])
    out=new_output(a.out or Path("results/residual_diag")/(datetime.now().strftime("run_%Y%m%d_%H%M%S")+f"_{os.getpid()}"))
    atomic_json(out/"plan.json",plan)
    state={"status":"preparing","output":str(out),"experiments":["real_skip_diagnosis","offline_oracle"],"gpu":a.gpu}
    atomic_json(out/"status.json",state)
    proc=None
    trace=out/"traces";trace.mkdir()
    def interrupted(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        # Run the source-bridge checks against the user's actual unchanged v1 package.
        with (out/"cpu_preflight.log").open("w",encoding="utf-8") as testlog:
            subprocess.run([sys.executable,"-m","unittest","discover","-s",
                            str(Path(__file__).parent/"tests"),"-v"],
                           stdout=testlog,stderr=subprocess.STDOUT,check=True)
        from .runtime import build
        runtime=out/"runtime_parent_capture"
        runtime_manifest=build(runtime)
        atomic_json(trace/"capture_config.json",{"layer_ordinals":layers,"decode_calls":a.decode_calls,
                    "min_context":a.min_context,"max_traces":a.max_traces,"max_bytes":int(a.max_gib*1024**3)})
        server=out/"server";server.mkdir()
        dispatch=server/"dispatch"
        name="ardiag-parent-"+uuid.uuid4().hex[:8]
        launch_args=SimpleNamespace(method="kvarn44",model=model,port=a.port,max_model_len=s["max_model_len"],
            max_seqs=1,batch_tokens=s["batch_tokens"],gpu_util=s["gpu_util"],server_seed=s["server_seed"],kv_bytes=None,eager=True)
        cli=launch_command(launch_args,name)
        command=[sys.executable,"-m","kvbench_residual.bootstrap","--runtime",str(runtime),"--policy","parent","--trace-dir",str(dispatch),"--",*cli[3:]]
        env=os.environ.copy();env.pop("PYTHONHOME",None)
        env.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONNOUSERSITE="1",PYTHONPATH=str(runtime),
            AR_RUNTIME_ROOT=str(runtime),AR_TRACE_DIR=str(dispatch),AR_POLICY="parent",AR_DIAG_TRACE_ROOT=str(trace),
            KVARN_FUSED_DECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            TRITON_CACHE_DIR=str(server/"cache/triton"),TORCHINDUCTOR_CACHE_DIR=str(server/"cache/torchinductor"),VLLM_CACHE_ROOT=str(server/"cache/vllm"))
        config={"kv_dtype":"kvarn_k4v4_g128","block_size":128,"model_name":name,"max_model_len":s["max_model_len"]}
        launch_record={"command":command,"config":config,"python":sys.executable,"tokenizer":tokenizer_identity(model),
                       "packages":package_versions(["vllm","torch","triton","numpy","transformers"]),
                       "runtime_manifest_sha256":sha_file(runtime/"runtime.json"),
                       "gpu":run_text(["nvidia-smi","-i",a.gpu,"--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader,nounits"]),
                       "environment":{k:env.get(k) for k in ("KVARN_SINK_TOKENS","KVARN_SINKHORN_ITERS","KVARN_POOL_SLOTS","KVARN_FAST_FLUSH","KVARN_SPLIT_K","KVARN_NUM_KV_SPLITS","NVIDIA_TF32_OVERRIDE","TRITON_F32_DEFAULT","VLLM_WORKER_MULTIPROC_METHOD")}}
        atomic_json(server/"launch.json",launch_record)
        with (server/"server.log").open("w",encoding="utf-8") as log:
            proc=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launch_record["pid"]=proc.pid
            base=f"http://127.0.0.1:{a.port}"
            launch_record["native"]=ready(proc,base,config,a.startup_timeout)
            atomic_json(server/"launch.json",launch_record)
            # Eager mode may not execute decode during startup; verify dispatch AFTER an unarmed warmup.
            warm=request_one(base,name,samples[0],s,8,server/"warmup.json.gz")
            atomic_json(server/"warmup.json",warm)
            from kvbench_residual.trace import summarize
            receipts=summarize(dispatch,"parent",runtime_manifest["executed_source_sha256"],.02)
            if not receipts["source_verified"] or not receipts["fused_launch_observed"] or not list(trace.glob("hook_*.json")):
                raise RuntimeError("Eager warmup did not confirm the expected native decode/capture hook")
            atomic_json(server/"dispatch_verified.json",receipts)
            state["status"]="capturing";atomic_json(out/"status.json",state)
            requests=[]
            for j,row in enumerate(samples):
                arm={"epoch":uuid.uuid4().hex,"dataset":row["dataset"],"case_id":row["case_id"],
                     "prompt_sha256":digest(row["prompt"]),"input_tokens":row["input_tokens"],"sample_index":j,
                     "protocol":"diagnostic_C1_eager_parent","source_manifest_sha256":plan["data_manifest_sha256"]}
                atomic_json(trace/"arm.json",arm)
                try:
                    result=request_one(base,name,row,s,a.diagnostic_max_tokens,server/f"sample_{j:03d}.json.gz")
                finally:
                    (trace/"arm.json").unlink(missing_ok=True)
                result["capture_epoch"]=arm["epoch"]
                requests.append(result)
                atomic_json(out/"diagnostic_requests.json",requests)
                errors=list(trace.glob("capture_error_*.json"))
                if errors: raise RuntimeError("Capture worker reported errors: "+str(errors))
                print(f"captured request {j+1}/{len(samples)}: {row['dataset']} {row['case_id']}",flush=True)
        # Model server is fully stopped BEFORE any GPU replay or offline analysis.
        stop_owned_group(proc);proc=None
        records=[json.loads(x.read_text()) for x in trace_paths(trace)]
        coverage=[]
        for request in requests:
            seen={(m["cache_ordinal"],m["decode_call_index"]) for m in records if m["request"]["epoch"]==request["capture_epoch"]}
            for layer in layers:
                for step in a.decode_calls:
                    coverage.append({"dataset":request["dataset"],"case_id":request["case_id"],"cache_ordinal":layer,
                                     "decode_call_index":step,"captured":(layer,step) in seen,
                                     "completion_tokens":request["completion_tokens"],"finish_reason":request["finish_reason"],
                                     "note":"Missing later steps may be natural EOS; no forced continuation" if (layer,step) not in seen else ""})
        write_csv(out/"capture_coverage.csv",coverage)
        state.update(status="captured",actual_traces=len(records),planned_traces=planned,
                     missing_capture_points=sum(not x["captured"] for x in coverage))
        atomic_json(out/"status.json",state)
        if a.capture_only:
            print(str(out),flush=True);return
        # Run as children to guarantee isolated CUDA contexts and fail-fast exit propagation.
        analysis_env=os.environ.copy();analysis_env["CUDA_VISIBLE_DEVICES"]=a.gpu
        analysis_env["PYTHONNOUSERSITE"]="1"
        for key in ("AR_DIAG_TRACE_ROOT","AR_RUNTIME_ROOT","AR_TRACE_DIR","AR_POLICY"):
            analysis_env.pop(key,None)
        stages=[("experiment1",[sys.executable,"-m","kvbench_residual_diag.diagnose","--traces",str(trace),"--out",str(out/"experiment1"),"--epsilons",*[str(x) for x in a.epsilons]]),
                ("experiment2",[sys.executable,"-m","kvbench_residual_diag.oracle","--traces",str(trace),"--out",str(out/"experiment2"),"--trace-limit",str(a.oracle_trace_limit),"--modes",*a.oracle_modes])]
        for label,cmd in stages:
            state["status"]=label;atomic_json(out/"status.json",state)
            with (out/(label+".log")).open("w",encoding="utf-8") as log:
                subprocess.run(cmd,env=analysis_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state["status"]="completed";atomic_json(out/"status.json",state)
        print("Diagnostics completed: "+str(out),flush=True)
    except BaseException as exc:
        state.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
        atomic_json(out/"status.json",state)
        raise
    finally:
        (trace/"arm.json").unlink(missing_ok=True)
        if proc is not None:
            stop_owned_group(proc)


if __name__=="__main__": main()

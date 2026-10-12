"""Private eager-server hook. Never imported into an unmodified production runtime.

A client arm file excludes warmup. The hook observes the actual rotated query,
original native packed pages, FP16 sink/tail and the final rotated GPU output.
Only active pages are copied. D2H synchronization makes timing unusable.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
from .common import ARRAYS, SCHEMA, atomic_json, sha_file


class Capture:
    def __init__(self, root, source):
        self.root = Path(root).resolve()
        self.source = str(Path(source).resolve())
        self.source_sha = sha_file(source)
        self.epoch = None
        self.layers = {}
        self.calls = {}
        self.pending = None
        self.saved = 0
        self.bytes = 0
        self.config = json.loads((self.root / "capture_config.json").read_text())
        atomic_json(self.root / f"hook_{os.getpid()}.json", {
            "source": self.source, "sha256": self.source_sha, "pid": os.getpid(),
            "installed": True, "scope": "eager diagnostic only; no timings"})

    def arm(self):
        path = self.root / "arm.json"
        if not path.exists():
            return None
        arm = json.loads(path.read_text())
        if arm["epoch"] != self.epoch:
            self.epoch = arm["epoch"]
            self.layers, self.calls = {}, {}
            self.pending = None
        return arm

    def choose(self, args, kwargs, inner, mode):
        arm = self.arm()
        if arm is None:
            return None
        import torch
        q, _, bt, sl, _, kv = args[:6]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Real trace capture requires --eager; CUDA graph capture is not allowed")
        if bool(kwargs.get("VQ_INDIRECT", False)):
            raise RuntimeError("Speculative/indirect verify cannot be labelled as direct decode")
        # The real driver passes the persistent [B*Hq,D] rotation buffer.
        # GpuCase uses a [B,Hq,D] view; both represent the same C1 layout.
        if tuple(q.shape) not in ((32,128),(1,32,128)) or q.dtype != torch.float16:
            raise RuntimeError(f"Armed capture expects C1 rotated FP16 [32,128] or [1,32,128], got {q.shape}/{q.dtype}")
        if kv.dtype != torch.uint8 or kv.ndim != 3 or kv.shape[1] != 8:
            raise RuntimeError("Unsupported actual KV cache")
        if (kwargs.get("D"), kwargs.get("GROUP"), kwargs.get("K_BITS"), kwargs.get("V_BITS")) != (128,128,4,4):
            raise RuntimeError("Expected pinned K4V4/G128 launch")
        identity = int(kv.data_ptr())
        ordinal = self.layers.setdefault(identity, len(self.layers))
        step = self.calls.get(identity, 0)
        self.calls[identity] = step + 1
        if ordinal not in self.config["layer_ordinals"] or step not in self.config["decode_calls"]:
            return None
        length = int(sl[0].item())
        if length < self.config["min_context"]:
            return None
        if self.saved >= self.config["max_traces"]:
            raise RuntimeError("Trace count cap reached; increase --max-traces explicitly")
        cfg = getattr(inner, "best_config", None)
        bn = kwargs.get("BLOCK_N", None if cfg is None else cfg.kwargs.get("BLOCK_N"))
        if bn not in (16,32,64):
            raise RuntimeError("Cannot resolve actual selected BLOCK_N; refusing guessed geometry")
        return {"arm": arm, "ordinal": ordinal, "step": step, "length": length,
                "mode": mode, "block_n": int(bn),
                "splits": int(kwargs.get("NUM_KV_SPLITS", 1)),
                "num_warps": int(kwargs.get("num_warps", getattr(cfg, "num_warps", 4))),
                "num_stages": int(kwargs.get("num_stages", getattr(cfg, "num_stages", 1))),
                "maxnreg": kwargs.get("maxnreg", getattr(cfg, "maxnreg", None)),
                "autotune_config": str(cfg), "kwargs": kwargs, "args": args}

    def save(self, item, output):
        import torch
        from kvbench_residual.fixtures import Layout
        q, _, bt, sl, b2s, kv, pk, pv = item["args"][:8]
        layout = Layout()
        if kv.shape[2] != layout.tile_bytes:
            raise RuntimeError("Unexpected cache tile width")
        for k,v in layout.offsets.items():
            if item["kwargs"].get(k) != v:
                raise RuntimeError("Actual packed offset differs: " + k)
        n = (item["length"] + 127)//128
        ids = bt[0,:n].detach().cpu().numpy().astype(np.int64)
        if len(ids) != n or np.any(ids < 0) or np.any(ids >= kv.shape[0]):
            raise RuntimeError("Incomplete active block table")
        unique, reverse = np.unique(ids, return_inverse=True)
        ix = torch.as_tensor(unique, device=kv.device, dtype=torch.long)
        slots = b2s.index_select(0, ix).detach().cpu().numpy().astype(np.int64)
        active_slots = np.unique(slots[slots >= 0])
        if np.any(slots < -1) or np.any(active_slots >= pk.shape[0]):
            raise RuntimeError("Invalid live pool map")
        estimated = (len(unique)*kv.shape[1]*kv.shape[2] + max(1,len(active_slots))*128*8*128*4 + 65536)
        if self.bytes+estimated > self.config["max_bytes"]:
            raise RuntimeError("Trace byte cap reached before D2H copy; increase --max-gib explicitly")
        native = kv.index_select(0, ix).detach().cpu().numpy()
        # Pool-backed packed fields can be uninitialised; never consume them as codes.
        native[slots >= 0] = 0
        compact_slots = np.full(len(unique), -1, np.int32)
        for j,s in enumerate(active_slots):
            compact_slots[slots == s] = j
        if len(active_slots):
            pi = torch.as_tensor(active_slots, device=pk.device, dtype=torch.long)
            poolk = pk.index_select(0, pi).detach().cpu().numpy()
            poolv = pv.index_select(0, pi).detach().cpu().numpy()
        else:
            poolk = np.zeros((1,128,8,128), np.float16)
            poolv = np.zeros_like(poolk)
        table = np.full((1, bt.shape[1]), -1, np.int32)
        table[0,:n] = reverse
        arrays = {"query": q.detach().reshape(1,32,128).cpu().numpy().copy(), "cache": native,
                  "block_table": table, "seq_lens": np.array([item["length"]],np.int32),
                  "block_to_slot": compact_slots, "pool_k": poolk, "pool_v": poolv,
                  "req_rows": np.array([0],np.int32),
                  "reference_gpu_rot": output.detach().reshape(1,32,128).cpu().numpy().copy()}
        if any(not np.isfinite(arrays[k]).all() for k in ("query","pool_k","pool_v","reference_gpu_rot")):
            # Entire selected pool page is copied, so unused pool tail can be nonfinite;
            # sanitize only unused token positions below, never valid activations.
            for logical,bid in enumerate(table[0,:n]):
                slot = compact_slots[bid]
                if slot >= 0 and logical == n-1:
                    valid = item["length"] - logical*128
                    arrays["pool_k"][slot,valid:] = 0
                    arrays["pool_v"][slot,valid:] = 0
            if any(not np.isfinite(arrays[k]).all() for k in ("query","pool_k","pool_v","reference_gpu_rot")):
                raise RuntimeError("Nonfinite valid live activation")
        if set(arrays) != set(ARRAYS):
            raise AssertionError("Trace schema implementation error")
        stamp = time.time_ns()
        stem = f"trace_{stamp}_{os.getpid()}_c{item['ordinal']:03d}_d{item['step']:04d}"
        tmp = self.root/(stem+".tmp.npz")
        np.savez(tmp, **arrays)
        dest = self.root/(stem+".npz")
        tmp.replace(dest)
        kw = item["kwargs"]
        scale = float(item["args"][9 if item["mode"] == "single" else 10])
        meta = {"schema": SCHEMA, "source_kind": "real_vllm_decode", "arrays_file": dest.name,
                "arrays_sha256": sha_file(dest), "uncompressed_bytes": int(sum(x.nbytes for x in arrays.values())),
                "layout": asdict(layout), "hq":32, "hk":8, "scale":scale,
                "window":int(kw.get("SLIDING_WINDOW",0)), "mode":item["mode"],
                "block_n":item["block_n"], "splits":item["splits"], "num_warps":item["num_warps"],
                "num_stages":item["num_stages"], "maxnreg":item["maxnreg"],
                "autotune_config":item["autotune_config"], "request":item["arm"],
                "cache_ordinal":item["ordinal"], "decode_call_index":item["step"],
                "context_tokens":item["length"], "kernel_source":self.source,
                "kernel_sha256":self.source_sha, "original_cache_shape":list(kv.shape),
                "active_pages":len(unique), "active_pool_pages":len(active_slots), "original_query_shape":list(q.shape),
                "eager":True, "batch":1, "capture_pid":os.getpid(),
                "scope":"Parent trajectory; real rotated GPU Q/KV/output. No runtime speed or DRAM claim. Cache ordinal is acquisition order, not a verified model layer name."}
        atomic_json(self.root/(stem+".json"), meta)
        self.saved += 1
        self.bytes += meta["uncompressed_bytes"]
        atomic_json(self.root/f"capture_progress_{os.getpid()}.json", {"traces":self.saved,"bytes":self.bytes})


class ObservedLaunch:
    def __init__(self, inner, name, capture):
        self.inner, self.name, self.capture = inner,name,capture
    def __getattr__(self,name):
        return getattr(self.inner,name)
    def __getitem__(self,grid):
        fn = self.inner[grid]
        def invoke(*args,**kwargs):
            result = fn(*args,**kwargs)
            c = self.capture
            try:
                if self.name.endswith("stage2"):
                    if c.pending is not None:
                        item,c.pending = c.pending,None
                        c.save(item,args[2])
                else:
                    mode = "split" if self.name.endswith("stage1") else "single"
                    item = c.choose(args,kwargs,self.inner,mode)
                    if item is not None:
                        if mode == "single": c.save(item,args[8])
                        else: c.pending = item
            except BaseException as exc:
                atomic_json(c.root/f"capture_error_{os.getpid()}.json", {"error":repr(exc),"kernel":self.name})
                raise
            return result
        return invoke


def install(namespace, source):
    root = os.environ.get("AR_DIAG_TRACE_ROOT")
    if not root:
        raise RuntimeError("Diagnostic runtime requires AR_DIAG_TRACE_ROOT")
    if namespace.get("AR_POLICY_NAME") != "parent":
        raise RuntimeError("Capture native Parent trajectories only")
    cap = Capture(root,source)
    for name in ("_kvarn_fused_decode_kernel","_kvarn_fused_decode_stage1","_kvarn_fused_decode_stage2"):
        if name not in namespace:
            raise RuntimeError("Missing pinned kernel: "+name)
        namespace[name] = ObservedLaunch(namespace[name],name,cap)

#!/usr/bin/env python3
"""
Patch kvbench_real_v6 to add precision-sweep method aliases without editing serve.sh.

New aliases:
  fp8       -> fp8_e4m3 KV, dynamic KV scales; reuses serve.sh fp16 base
  tq84      -> turboquant_k8v4; reuses serve.sh tq4 base
  tq34      -> turboquant_k3v4_nc; reuses serve.sh tq4 base
  kvarn4464 -> kvarn_k4v4_g64; reuses serve.sh kvarn64 base

Existing aliases remain:
  fp16      -> auto
  tq4       -> turboquant_4bit_nc (K4V4)
  tq3       -> turboquant_3bit_nc (K3V3)
  kvarn     -> kvarn_k4v2_g128
  kvarn44   -> kvarn_k4v4_g128
  kvarn64   -> kvarn_k4v2_g64

The existing serve.sh passthrough used for --enforce-eager is reused. The base
method chooses the right block size and general launch settings, then extra CLI
arguments override --kv-cache-dtype at the end of the vLLM command.
"""
from __future__ import annotations
from pathlib import Path
import shutil, sys, time

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
run_real = root / "run_real.py"
engine = root / "engine.py"
profile = root / "profile_kernels.py"

for f in (run_real, engine):
    if not f.is_file():
        raise SystemExit(f"Missing required file: {f}")

stamp = time.strftime("%Y%m%d_%H%M%S")
for f in (run_real, engine, profile):
    if f.is_file():
        shutil.copy2(f, f.with_name(f.name + f".before_precision_v8_{stamp}"))

def patch_once(path: Path, old: str, new: str, label: str):
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"{label}: already patched")
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            f"{label}: expected one match in {path}, got {n}; backups were kept."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"{label}: patched {path}")

# 1) Extend run_real method registry.
old = """DTYPES={'fp16':'auto','tq4':'turboquant_4bit_nc','tq3':'turboquant_3bit_nc',
        'kvarn':'kvarn_k4v2_g128','kvarn44':'kvarn_k4v4_g128','kvarn64':'kvarn_k4v2_g64'}
"""
new = """DTYPES={
    'fp16':'auto',
    'fp8':'fp8_e4m3',
    'tq84':'turboquant_k8v4',
    'tq4':'turboquant_4bit_nc',
    'tq34':'turboquant_k3v4_nc',
    'tq3':'turboquant_3bit_nc',
    'kvarn':'kvarn_k4v2_g128',
    'kvarn44':'kvarn_k4v4_g128',
    'kvarn64':'kvarn_k4v2_g64',
    'kvarn4464':'kvarn_k4v4_g64',
}

# New methods reuse already-tested serve.sh launch cases. Extra args are passed
# through serve.sh after its fixed args, so the last --kv-cache-dtype wins.
# This avoids modifying the frozen service launcher itself.
SERVE_BASE={
    'fp8':'fp16',
    'tq84':'tq4',
    'tq34':'tq4',
    'kvarn4464':'kvarn64',
}
SERVE_EXTRA={
    # Qwen3-4B does not ship calibrated KV scales in this benchmark snapshot.
    # Without this flag standard FP8 would otherwise fall back to scale=1.0.
    'fp8':['--kv-cache-dtype','fp8_e4m3','--calculate-kv-scales'],
    'tq84':['--kv-cache-dtype','turboquant_k8v4'],
    'tq34':['--kv-cache-dtype','turboquant_k3v4_nc'],
    'kvarn4464':['--kv-cache-dtype','kvarn_k4v4_g64'],
}
"""
patch_once(run_real, old, new, "run_real method registry")

old = """                    p=dict(id=f'r{rep:02d}_{method}_{dataset}_apc{apc}_C{c}',repeat=rep,method=method,
                        dataset=dataset,apc=apc,concurrency=c,number=n,max_tokens=cap,request_max_tokens=cap,
                        max_tokens_transport=('explicit_int' if cap is not None else 'json_null_sentinel'),
"""
new = """                    p=dict(id=f'r{rep:02d}_{method}_{dataset}_apc{apc}_C{c}',repeat=rep,method=method,
                        dataset=dataset,apc=apc,concurrency=c,number=n,max_tokens=cap,request_max_tokens=cap,
                        serve_base_method=SERVE_BASE.get(method,method),
                        serve_extra_args=SERVE_EXTRA.get(method,[]),
                        max_tokens_transport=('explicit_int' if cap is not None else 'json_null_sentinel'),
"""
patch_once(run_real, old, new, "run_real plan metadata")

# 2) Make engine call serve.sh with the base alias plus precision override.
old = """        args = ['bash', str(Path(self.cfg['kit']) / 'serve.sh'), p['method']]
        if self.cfg['eager']:
            args.append('--enforce-eager')
"""
new = """        base_method=p.get('serve_base_method',p['method'])
        extra_args=list(p.get('serve_extra_args') or [])
        args = ['bash', str(Path(self.cfg['kit']) / 'serve.sh'), base_method, *extra_args]
        if self.cfg['eager']:
            args.append('--enforce-eager')
"""
patch_once(engine, old, new, "engine serve passthrough")

# 3) Extend profile_kernels.py if it is present.
if profile.is_file():
    old = """    ap.add_argument("--method", required=True, choices=["fp16","tq4","tq3","kvarn","kvarn44","kvarn64"])
"""
    new = """    ap.add_argument("--method", required=True, choices=[
        "fp16","fp8","tq84","tq4","tq34","tq3",
        "kvarn","kvarn44","kvarn64","kvarn4464"
    ])
"""
    patch_once(profile, old, new, "profile method choices")

    old = """    cmd = conda_command(serve_env, [
        "bash", str(kit / "serve.sh"), args.method,
        "--profiler-config", profiler_cfg,
    ])
"""
    new = """    _BASE={
        "fp8":"fp16","tq84":"tq4","tq34":"tq4","kvarn4464":"kvarn64",
    }
    _EXTRA={
        "fp8":["--kv-cache-dtype","fp8_e4m3","--calculate-kv-scales"],
        "tq84":["--kv-cache-dtype","turboquant_k8v4"],
        "tq34":["--kv-cache-dtype","turboquant_k3v4_nc"],
        "kvarn4464":["--kv-cache-dtype","kvarn_k4v4_g64"],
    }
    cmd = conda_command(serve_env, [
        "bash", str(kit / "serve.sh"), _BASE.get(args.method,args.method),
        *_EXTRA.get(args.method,[]),
        "--profiler-config", profiler_cfg,
    ])
"""
    patch_once(profile, old, new, "profile serve passthrough")

print("Precision-sweep patch applied.")
print("Backups suffix:", f".before_precision_v8_{stamp}")

"""Small standard-library helpers; no model or CUDA imports."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path

DEFAULT_SUITE = 'results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.tmp.{os.getpid()}')
    with t.open('w', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())
    t.replace(p)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    keys = fields or list(dict.fromkeys(k for r in rows for k in r))
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    with t.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    t.replace(p)


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_upstream(path):
    p = Path(path).expanduser().resolve(strict=True)
    u = read_json(p)
    if (u.get('schema') != 'kvbench.longcontext.full_suite.v1' or u.get('fatal_error')
            or not u.get('methods') or any(m.get('state') != 'completed' for m in u['methods'])):
        raise ValueError('Use a completed LongContext V3 suite.json, not a residual suite.')
    s = u['settings']
    for key in ('model', 'data', 'kit', 'server_env', 'max_model_len', 'batch_tokens', 'gpu_util', 'server_seed'):
        if key not in s:
            raise ValueError('Missing upstream setting: ' + key)
    return p, u


def implementation_hashes():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob('*'))
            if p.is_file() and (p.suffix in ('.py', '.json', '.sh')) and '__pycache__' not in p.parts}


def inside(root, value):
    root = Path(root).resolve()
    p = (root / value).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Path escapes run directory')
    return p

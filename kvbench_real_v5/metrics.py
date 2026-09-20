"""Statistics based on recorded native data; never infer prefix cache hits from prompt similarity."""
from __future__ import annotations
import json, math, re, statistics
from pathlib import Path
from typing import Any
SAMPLE_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>\S+)(?:\s+\S+)?$')


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def quantile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    seq = sorted(values)
    return seq[max(0, math.ceil(len(seq) * percent / 100) - 1)]

def describe(values: list[float], prefix: str) -> dict[str, Any]:
    return {prefix + '_n': len(values), prefix + '_avg': statistics.mean(values) if values else None,
            **{prefix + f'_p{p}': quantile(values, p) for p in (50, 95, 99)},
            prefix + '_max': max(values) if values else None}

def finite(x: Any) -> float | None:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (ValueError, TypeError):
        return None

def parse_prometheus(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        m = SAMPLE_RE.match(line)
        if m and (value := finite(m['value'])) is not None:
            out[m['name'] + (m['labels'] or '')] = value
    return out

def sampled_telemetry(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    out: dict[str, Any] = {'telemetry_samples': len(rows),
        'telemetry_scope': 'measured EvalScope CLI window; includes client setup/report, excludes explicit warmup'}
    for col in ('memory_used_mib', 'gpu_util_pct', 'temperature_c', 'power_w', 'sm_clock_mhz', 'memory_clock_mhz'):
        vals = [x for r in rows if (x := finite(r.get('gpu', {}).get(col))) is not None]
        out['sampled_' + col + '_max'] = max(vals) if vals else None
        out['sampled_' + col + '_avg'] = statistics.mean(vals) if vals else None
    for label, names in {
        'kv_usage_fraction': ['vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc'],
        'requests_waiting': ['vllm:num_requests_waiting'],
        'requests_running': ['vllm:num_requests_running'],
    }.items():
        vals = []
        for r in rows:
            gauges = r.get('gauges', {})
            v = next((gauges[n] for n in names if n in gauges), None)
            if finite(v) is not None:
                vals.append(float(v))
        out['sampled_' + label + '_max'] = max(vals) if vals else None
    return out


def native_metrics(before: str, after: str) -> dict[str, Any]:
    """Store vLLM's native cache counters/gauges; do NOT estimate/recompute a hit ratio."""
    a,b=parse_prometheus(before),parse_prometheus(after)
    names=('prefix_cache','cache_config','kv_cache_usage','gpu_cache_usage')
    selected=lambda data:{k:v for k,v in data.items() if any(n in k for n in names)}
    deltas={};resets=[]
    for k in a.keys() & b.keys():
        name=k.split('{',1)[0]
        if 'preemption' in name and name.endswith('_total'):
            if b[k]<a[k]:resets.append(k)
            else:deltas[name]=deltas.get(name,0)+(b[k]-a[k])
    def single(data,names):
        for name in names:
            vals=[v for k,v in data.items() if k.split('{',1)[0]==name]
            if vals:return vals[0] if len(vals)==1 else None
        return None
    flat={}
    for phase,data in [('before',a),('after',b)]:
        for part in ['queries','hits']:
            flat[f'native_prefix_{part}_{phase}']=single(data,[f'vllm:prefix_cache_{part}_total',f'vllm:prefix_cache_{part}',f'vllm:gpu_prefix_cache_{part}_total'])
    info=[]
    for line in after.splitlines():
        if 'cache_config_info{' not in line or line.startswith('#'):continue
        labels={k:json.loads('"'+v+'"') for k,v in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"',line)}
        info.append(labels)
    return {**flat,'native_cache_before':selected(a),'native_cache_after':selected(b),
            'native_cache_config_labels':info,
            'preemptions_delta':sum(deltas.values()) if deltas and not resets else None,
            'counter_resets':resets,
            'prefix_hit_ratio_source':'vLLM log lines; no client-side prompt-similarity or ratio estimate'}


def extract_server_log(text: str) -> dict[str, Any]:
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    evidence=[line for line in text.splitlines() if re.search(
        r'KV cache (?:size|memory)|Available KV|Maximum concurrency|num_gpu_blocks|cache_config_info|Prefix cache hit rate',line,re.I)]
    capacity=[];available=[];blocks=[];rate=[]
    for line in evidence:
        m=re.search(r'GPU KV cache size:\s*([\d,]+)\s*tokens',line,re.I)
        if m:capacity.append(int(m[1].replace(',','')))
        m=re.search(r'Available KV cache memory:\s*([\d.]+)\s*(GiB|GB|MiB|MB)',line,re.I)
        if m:available.append({'value':float(m[1]),'unit':m[2],'line':line})
        m=re.search(r'num_gpu_blocks(?:[\s:=]+)(\d+)',line)
        if m:blocks.append(int(m[1]))
        m=re.search(r'(?<!External )Prefix cache hit rate:\s*([\d.]+)%',line)
        if m:rate.append({'rate_percent':float(m[1]),'native_log_line':line})
    return {'native_log_evidence':evidence,'kv_capacity_tokens_log_values':capacity,
            'available_kv_memory_log_values':available,'num_gpu_blocks_log_values':blocks,
            'native_prefix_rates':rate,
            'kv_capacity_tokens_single_gpu_log':capacity[-1] if capacity else None,
            'prefix_native_last_percent':rate[-1]['rate_percent'] if rate else None,
            'prefix_native_last_is_run_aggregate':False}

"""All-record inputs. No implicit sampling, dataset substitution, or network I/O."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
from .common import atomic_json, canonical, digest, jsonl, read_json, sha_file, write_csv

LONG_TASKS = ('narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh',
              'hotpotqa', '2wikimqa', 'musique', 'dureader', 'gov_report', 'qmsum',
              'multi_news', 'vcsum', 'trec', 'triviaqa', 'samsum', 'lsht',
              'passage_count', 'passage_retrieval_en', 'passage_retrieval_zh', 'lcc', 'repobench-p')
RAW_COMPLETION = {'trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p'}
SOURCE_ZIP_SHA256 = 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'


def source_bytes(source, name):
    p = Path(source)
    if p.is_dir():
        found = [q for q in (p/(name+'.jsonl'), p/'data'/(name+'.jsonl')) if q.is_file()]
        if len(found) != 1:
            raise ValueError(f'Need exactly one {name}.jsonl under {p}; do not replace LongBench-21 with six tasks.')
        return found[0].read_bytes()
    if not zipfile.is_zipfile(p):
        raise ValueError('LongBench source must be the local data.zip or JSONL directory')
    with zipfile.ZipFile(p) as z:
        found = [q for q in z.namelist() if q in (name+'.jsonl', 'data/'+name+'.jsonl')]
        if len(found) != 1:
            raise ValueError('Missing/ambiguous LongBench dataset: ' + name)
        return z.read(found[0])


def parse_records(raw, dataset):
    records = [json.loads(x) for x in raw.decode('utf-8-sig').splitlines() if x.strip()]
    if not records or any(not isinstance(x, dict) for x in records):
        raise ValueError('Empty or invalid dataset: ' + dataset)
    return records


def inventory(suite, datasets, long_source=None, real_data='data/real_v5'):
    if len(datasets) != len(set(datasets)):
        raise ValueError('Duplicate datasets')
    allowed = {'longbench_all', 'longbench_frozen', 'math_500', 'humaneval'}
    if set(datasets)-allowed or {'longbench_all', 'longbench_frozen'} <= set(datasets):
        raise ValueError('Select longbench_all OR longbench_frozen, plus optional math_500/humaneval')
    s = suite['settings']; result = []
    if 'longbench_all' in datasets:
        source = Path(long_source or s['source']).expanduser().resolve(strict=True)
        zip_sha = sha_file(source) if source.is_file() else None
        for name in LONG_TASKS:
            raw = source_bytes(source, name)
            records = parse_records(raw, name)
            result.append(dict(dataset=name, family='longbench_all', source=str(source), count=len(records),
                               source_sha256=hashlib.sha256(raw).hexdigest(), zip_sha256=zip_sha,
                               matches_known_official_zip=zip_sha == SOURCE_ZIP_SHA256))
    if 'longbench_frozen' in datasets:
        folder = Path(s['data']).resolve(strict=True)
        m = read_json(folder/'manifest.json')
        for name, info in m['datasets'].items():
            file = folder/info['file']
            if file.parent.resolve() != folder or sha_file(file) != info['sha256']:
                raise ValueError('Frozen LongBench hash/path mismatch')
            rows = list(jsonl(file))
            if len(rows) != info['selected_count'] or [r['case_id'] for r in rows] != info['selected_case_ids']:
                raise ValueError('Frozen LongBench count/order mismatch')
            result.append(dict(dataset=name, family='longbench_frozen', source=str(file), count=len(rows),
                               source_sha256=sha_file(file), manifest_sha256=sha_file(folder/'manifest.json')))
    for name in ('math_500', 'humaneval'):
        if name not in datasets:
            continue
        folder = Path(real_data).expanduser().resolve(strict=True)
        m = read_json(folder/'manifest.json')['datasets'][name]
        file = folder/m['file']; expected_count = 500 if name == 'math_500' else 164
        if file.parent.resolve() != folder or sha_file(file) != m['sha256']:
            raise ValueError(name + ' frozen data hash/path mismatch')
        rows = list(jsonl(file))
        if len(rows) != expected_count or m['count'] != expected_count:
            raise ValueError(f'{name}: expected all {expected_count} records, got {len(rows)}')
        result.append(dict(dataset=name, family=name, source=str(file), count=len(rows),
                           source_sha256=sha_file(file), manifest_sha256=sha_file(folder/'manifest.json')))
    return result


def iter_source(entry):
    name = entry['dataset']
    if entry['family'] != 'longbench_all' and sha_file(entry['source']) != entry['source_sha256']:
        raise ValueError('Frozen source changed since planning: ' + name)
    if entry['family'] == 'longbench_all':
        raw = source_bytes(entry['source'], name)
        if hashlib.sha256(raw).hexdigest() != entry['source_sha256']:
            raise ValueError('Source changed since planning: ' + name)
        prompts = read_json(Path(__file__).with_name('longbench_prompts.json'))
        for i, r in enumerate(parse_records(raw, name)):
            if not isinstance(r.get('context'), str) or not isinstance(r.get('input'), str):
                raise ValueError(f'{name}/{i}: missing official context/input strings')
            yield dict(dataset=name, family='longbench_all', source_index=i,
                       case_id=name+'/'+str(r.get('_id', r.get('id', i))),
                       prompt=prompts[name].format(context=r['context'], input=r['input']),
                       request_kind='completion' if name in RAW_COMPLETION else 'chat',
                       source_record_sha256=digest(r))
    elif entry['family'] == 'longbench_frozen':
        for i, r in enumerate(jsonl(entry['source'])):
            yield dict(dataset=name, family=entry['family'], source_index=i, case_id=r['case_id'],
                       prompt=r['prompt'], request_kind=r['request_kind'], source_record_sha256=digest(r),
                       previous_input_tokens=r['input_tokens'])
    else:
        for i, r in enumerate(jsonl(entry['source'])):
            messages = r['messages']
            if len(messages) != 1 or messages[0].get('role') != 'user' or not isinstance(messages[0].get('content'), str):
                raise ValueError('Expected the frozen single-user real_v5 prompt')
            # Never use reference, original_record.solution or tests in the prompt.
            yield dict(dataset=name, family=entry['family'], source_index=i, case_id=r['case_id'],
                       prompt=messages[0]['content'], request_kind='chat', source_record_sha256=digest(r))


def count_input(tokenizer, prompt, kind, thinking):
    if kind == 'completion':
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    return len(tokenizer.apply_chat_template([{'role':'user', 'content':prompt}], tokenize=True,
                 add_generation_prompt=True, enable_thinking=thinking))


def fit(tokenizer, prompt, kind, thinking, limit, overflow):
    original = count_input(tokenizer, prompt, kind, thinking)
    if original <= limit:
        return prompt, original, original, False
    if overflow == 'error':
        raise ValueError(f'Prompt length {original} exceeds {limit}; use explicit --overflow head-tail or a larger supported context')
    if overflow != 'head-tail':
        raise ValueError('Only error/head-tail overflow policies are supported; no hidden row skipping')
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    keep = min(len(ids)-1, limit)
    while keep > 0:
        left, right = (keep+1)//2, keep//2
        candidate = tokenizer.decode(ids[:left], skip_special_tokens=False)
        if right:
            candidate += tokenizer.decode(ids[-right:], skip_special_tokens=False)
        actual = count_input(tokenizer, candidate, kind, thinking)
        if actual <= limit:
            return candidate, original, actual, True
        keep -= max(1, actual-limit)
    raise ValueError('Even the chat template cannot fit the configured window')


def prepare(out, entries, model, window, cap, thinking, overflow, case_limit=0,
            tokenizer=None, cases_per_task=0):
    """Write one frozen row for EVERY selected source record, preserving order."""
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    seen = set(); counts = []; n = 0
    with (out/'cases.jsonl').open('w', encoding='utf-8') as stream:
        for entry in entries:
            selected = clipped = 0
            for row in iter_source(entry):
                if case_limit and n >= case_limit:
                    break
                if cases_per_task and selected >= cases_per_task:
                    break
                key = (row['dataset'], row['case_id'])
                if key in seen:
                    raise ValueError('Duplicate source case: ' + str(key))
                seen.add(key)
                prompt, orig, effective, was_clipped = fit(tokenizer, row['prompt'], row['request_kind'], thinking, window-cap, overflow)
                if 'previous_input_tokens' in row and orig != row['previous_input_tokens']:
                    raise ValueError('Frozen source tokenization differs from current tokenizer/settings')
                row.update(prompt=prompt, original_input_tokens=orig, input_tokens=effective,
                           was_truncated=was_clipped, truncation_policy=overflow,
                           prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           sample_index=n, max_model_len=window, max_tokens=cap,
                           remaining_context_tokens=window-effective,
                           generation_budget_policy='diagnostic-prefix', generation_budget_tokens=cap)
                row['case_key'] = f'{n:06d}_'+digest([row['dataset'], row['case_id']])[:12]
                stream.write(canonical(row)+'\n')
                selected += 1; clipped += int(was_clipped); n += 1
            counts.append(dict(dataset=entry['dataset'], family=entry['family'], source_count=entry['count'],
                               selected_count=selected, clipped_count=clipped,
                               excluded_count=entry['count']-selected,
                               exclusion_reason=(
                                   'explicit_cases_per_task_limit' if cases_per_task and selected < entry['count']
                                   else 'explicit_global_case_limit' if selected < entry['count'] else '')))
            print(f"prepared {entry['dataset']}: {selected}/{entry['count']}, clipped={clipped}", flush=True)
    if n == 0:
        raise ValueError('No input cases')
    write_csv(out/'coverage.csv', counts)
    m = dict(schema='kvbench.asym.inputs.v1', cases=n, source_cases=sum(e['count'] for e in entries),
             all_source_records_selected=n == sum(e['count'] for e in entries),
             case_limit=case_limit, cases_per_task=cases_per_task,
             cases_sha256=sha_file(out/'cases.jsonl'), entries=entries,
             clipping_policy=overflow, coverage=counts,
             note='All-record diagnostic prefix evaluation. Not official task scores or unchanged full-context evaluation.')
    atomic_json(out/'manifest.json', m)
    return m

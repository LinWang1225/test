"""Request-clustered paired reports. Layers/heads are NOT independent samples."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
from pathlib import Path
import numpy as np
from .common import atomic_json, digest, inside, read_json, sha_file, write_csv


def committed_cases(root, verify=True):
    root = Path(root).resolve()
    for p in sorted((root/'cases').glob('*/done.json')):
        d = read_json(p)
        if d.get('schema') != 'kvbench.asym.case.v1' or d.get('status') != 'completed':
            raise ValueError('Invalid case receipt: '+str(p))
        attempt = inside(root, d['attempt'])
        if verify:
            for rel, expected in d['file_hashes'].items():
                q = inside(attempt, rel)
                if not q.is_file() or sha_file(q) != expected:
                    raise ValueError('Committed result changed: '+str(q))
        yield d, attempt


def number(row, key):
    v = row.get(key)
    return float(v) if v not in (None, '') else None


def stat(values, q=None):
    a = [x for x in values if x is not None and np.isfinite(x)]
    if not a:
        return None
    return float(np.quantile(a,q)) if q is not None else float(np.mean(a))


def paired_bootstrap(values, repeats, seed):
    x = np.asarray(values,dtype=np.float64)
    if len(x) < 2 or repeats < 1:
        return None, None
    rng = np.random.default_rng(seed)
    means = []
    for begin in range(0,repeats,64):
        indices = rng.integers(0,len(x),(min(64,repeats-begin),len(x)))
        means.extend(x[indices].mean(axis=1).tolist())
    lo, hi = np.quantile(means,[.025,.975])
    return float(lo),float(hi)


def case_summary(rows):
    by_profile = defaultdict(list)
    pairs = defaultdict(list)
    by_stratum = defaultdict(dict)
    for r in rows:
        if r['kind'] != 'paired':
            continue
        # No-quantized-history records remain in coverage, not as perfect zeros.
        if r['eligible_for_equal_payload'] != 'True':
            continue
        key = (r['dataset'],r['method'],int(r['mask_seed']),float(r['total_residual_rate']),float(r['k_priority']))
        by_profile[key].append(r)
        sk = (r['trace'],int(r['kv_head']),r['method'],int(r['mask_seed']),float(r['total_residual_rate']))
        by_stratum[sk][float(r['k_priority'])] = r
    for sk, values in by_stratum.items():
        base = values.get(.5)
        if base is None:
            raise ValueError('Missing symmetric paired control')
        for share, row in values.items():
            if int(row['logical_payload_read_bytes']) != int(base['logical_payload_read_bytes']):
                raise ValueError('Unequal payload in claimed paired comparison')
            a, b = number(row,'relative_l2_max_head'),number(base,'relative_l2_max_head')
            if a is not None and b is not None:
                key=(row['dataset'],row['method'],int(row['mask_seed']),float(row['total_residual_rate']),share)
                pairs[key].append(a-b)
    summaries=[]
    for key, rs in by_profile.items():
        ds, method, seed, total, share=key
        summaries.append(dict(dataset=ds,method=method,mask_seed=seed,total_residual_rate=total,k_priority=share,
            informative_groups=len(rs),defined_relative_groups=sum(number(r,'relative_l2_max_head') is not None for r in rs),
            mean_relative_l2_max_head=stat([number(r,'relative_l2_max_head') for r in rs]),
            p95_relative_l2_max_head=stat([number(r,'relative_l2_max_head') for r in rs],.95),
            maximum_relative_l2_max_head=max([number(r,'relative_l2_max_head') for r in rs if number(r,'relative_l2_max_head') is not None],default=None),
            mean_abs_l2_max=stat([number(r,'abs_l2_max') for r in rs]),
            mean_payload_reduction=stat([number(r,'payload_reduction_fraction') for r in rs]),
            mean_key_read_fraction=stat([number(r,'key_residual_read_fraction') for r in rs]),
            mean_value_read_fraction=stat([number(r,'value_residual_read_fraction') for r in rs]),
            paired_groups=len(pairs[key]),paired_mean_error_difference_vs_symmetric=stat(pairs[key])))
    return summaries


def controller_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get('kind') != 'controller':
            continue
        key = (row['dataset'], float(row['controller_tolerance']), float(row['controller_margin']))
        grouped[key].append(row)
    out = []
    for (dataset, tolerance, margin), values in grouped.items():
        true_errors = [number(row, 'relative_l2_max_head') for row in values]
        estimated = [number(row, 'controller_estimated_error') for row in values]
        rhos = [number(row, 'rho_q') for row in values]
        underread = [
            number(row, 'relative_l2_max_head') > tolerance
            for row in values
            if number(row, 'relative_l2_max_head') is not None
        ]
        out.append(dict(
            dataset=dataset,
            controller_tolerance=tolerance,
            controller_margin=margin,
            traces=len(values),
            mean_true_error=stat(true_errors),
            p95_true_error=stat(true_errors, .95),
            max_true_error=max((x for x in true_errors if x is not None), default=None),
            mean_estimated_error=stat(estimated),
            mean_rho=stat(rhos),
            p95_rho=stat(rhos, .95),
            underread_rate=(sum(underread) / len(underread)) if underread else None,
            mean_controller_steps=stat([number(row, 'controller_steps') for row in values]),
        ))
    return out


def report(root, bootstrap=1000):
    root=Path(root).resolve(); reports=root/'comparison';reports.mkdir(exist_ok=True)
    all_case=[];all_controller=[];coverage=[]
    for done, attempt in committed_cases(root):
        path=attempt/'analysis/rows.csv.gz'
        if done['traces']:
            with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
                rows = list(csv.DictReader(f))
                summaries=case_summary(rows)
                controller=controller_summary(rows)
        else:
            summaries=[];controller=[]
        for s in summaries:
            s.update(case_id=done['case_id'],case_key=done['case_key'],family=done['family'])
        all_case.extend(summaries)
        for s in controller:
            s.update(case_id=done['case_id'],case_key=done['case_key'],family=done['family'])
        all_controller.extend(controller)
        coverage.append({k:done[k] for k in ('dataset','case_id','case_key','family','traces','missing_capture_points',
                                            'expected_capture_points','completion_tokens','finish_reason','was_truncated')})
        coverage[-1]['informative_groups']=max((s['informative_groups'] for s in summaries if s['k_priority']==.5),default=0)
    write_csv(reports/'case_level.csv',all_case)
    write_csv(reports/'controller_case_level.csv',all_controller)
    write_csv(reports/'case_coverage.csv',coverage)
    groups=defaultdict(list)
    for s in all_case:
        key=tuple(s[k] for k in ('family','dataset','method','mask_seed','total_residual_rate','k_priority'))
        groups[key].append(s)
    aggregates=[];paired=[]
    for key,rs in groups.items():
        family,ds,method,seed,total,share=key
        common=dict(family=family,dataset=ds,method=method,mask_seed=seed,total_residual_rate=total,k_priority=share)
        aggregates.append(dict(**common,cases=len(rs),informative_groups=sum(s['informative_groups'] for s in rs),
            mean_case_mean_error=stat([s['mean_relative_l2_max_head'] for s in rs]),
            p95_case_mean_error=stat([s['mean_relative_l2_max_head'] for s in rs],.95),
            mean_case_p95_error=stat([s['p95_relative_l2_max_head'] for s in rs]),
            mean_case_payload_reduction=stat([s['mean_payload_reduction'] for s in rs])))
        diffs=[s['paired_mean_error_difference_vs_symmetric'] for s in rs if s['paired_mean_error_difference_vs_symmetric'] is not None]
        lo,hi=paired_bootstrap(diffs,bootstrap,int(digest(key)[:8],16))
        paired.append(dict(**common,paired_cases=len(diffs),mean_error_difference=stat(diffs),
                           bootstrap_ci95_low=lo,bootstrap_ci95_high=hi,bootstrap_repeats=bootstrap,
                           uncertainty_unit='request; correlated layers/heads averaged within request'))
    write_csv(reports/'aggregate.csv',aggregates)
    write_csv(reports/'paired_vs_symmetric.csv',paired)
    controller_groups=defaultdict(list)
    for row in all_controller:
        key=tuple(row[k] for k in ('family','dataset','controller_tolerance','controller_margin'))
        controller_groups[key].append(row)
    controller_aggregate=[]
    for key, rows in controller_groups.items():
        family, dataset, tolerance, margin=key
        controller_aggregate.append(dict(
            family=family, dataset=dataset, controller_tolerance=tolerance,
            controller_margin=margin, cases=len(rows),
            mean_case_true_error=stat([number(r,'mean_true_error') for r in rows]),
            p95_case_true_error=stat([number(r,'p95_true_error') for r in rows], .95),
            max_case_true_error=max((number(r,'max_true_error') for r in rows
                                     if number(r,'max_true_error') is not None), default=None),
            mean_case_rho=stat([number(r,'mean_rho') for r in rows]),
            mean_case_underread_rate=stat([number(r,'underread_rate') for r in rows]),
        ))
    write_csv(reports/'controller_aggregate.csv',controller_aggregate)
    m=read_json(root/'inputs/manifest.json') if (root/'inputs/manifest.json').exists() else {}
    state=dict(schema='kvbench.asym.report.v1',committed_cases=len(coverage),planned_cases=m.get('cases'),
               all_planned_cases_completed=len(coverage)==m.get('cases'),all_source_records_selected=m.get('all_source_records_selected'),
               total_captured_traces=sum(r['traces'] for r in coverage),
               missing_capture_points=sum(r['missing_capture_points'] for r in coverage),
               controller_cases=len(all_controller),
               files={p.name:sha_file(p) for p in reports.glob('*.csv')})
    atomic_json(reports/'report.json',state)
    lines=['# 等逻辑载荷 K/V 非对称对照','',
           f"完成请求：{len(coverage)} / {m.get('cases','NA')}；真实 trace：{state['total_captured_traces']}。",'',
           '先检查 inputs/coverage.csv 的全部子任务、原始题数和截断数；再检查 case_coverage.csv 的自然 EOS/缺失采集点。',
           'case_level.csv 先在每道题内平均层/head/位置；aggregate.csv 再在任务内对题等权统计。',
           'paired_vs_symmetric.csv：误差差值 < 0 表示比同流量对称分配更好。95% bootstrap 按请求重采样；是探索性区间，未作多重比较校正。',
           'mask_seed 是随机读取掩码种子，不是三个独立的生成随机种子。',
           '无可控量化历史的短 trace 不作为零误差成功样本进入非对称比较。',
           'K priority=0 为 V 优先，0.5 为对称，1 为 K 优先；整数取整可能使对称组相差一个 packet。',
           '所有配对组均校验相同逻辑码字节；这不等于 DRAM 事务相同，不含排序/中间张量流量，也不报告 TPOT。',
           '这是 Parent 轨迹上的 float64 注意力反事实；没有把选择策略写入生成过程，不提供任务准确率。','']
    lines[-1:-1]=[
        'controller_case_level.csv/controller_aggregate.csv 记录 Query-aware 估计、真实误差、误停率和 rho(q)。',
        '误差估计是 Anchor+metadata 的离线代理；underread_rate 以真实 Full-KV 误差超过 tolerance 定义。',
    ]
    (reports/'readout.md').write_text('\n'.join(lines),encoding='utf-8')
    return state


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args(argv)
    if a.bootstrap<0: p.error('bootstrap cannot be negative')
    print(report(a.run,a.bootstrap))


if __name__=='__main__':main()

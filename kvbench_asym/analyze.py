"""All supplied traces/heads, fixed-budget paired masks, no GPU speed claims."""
from __future__ import annotations
import argparse
import csv
import gzip
from pathlib import Path
import numpy as np
from .common import atomic_json, sha_file
from .select import from_group, quota, choose, accounting
from .controller import ControllerConfig, run_controller


def variants(methods, seeds):
    out = [(m, 0) for m in methods if m != 'random']
    if 'random' in methods:
        out += [('random', int(s)) for s in seeds]
    if not out:
        raise ValueError('At least one selection method is required')
    return out


def group_rows(group, block_n, totals, shares, methods, seeds, identity, fixed_aux=0,
               controller_tolerances=(), controller_margin=1.0, controller_max_steps=None):
    from kvbench_residual_diag.math_ops import evaluate
    from kvbench_residual_diag.common import output_metrics
    view = from_group(group, block_n)
    n = int(view.eligible.sum())
    rows, masks = [], []
    full_masks = np.ones(group.n, bool)
    anchor_masks = ~view.eligible
    specs = [('control_full', 2.0, .5, 'control', 0, full_masks, full_masks),
             ('control_anchor', 0.0, .5, 'control', 0, anchor_masks, anchor_masks)]
    for method, seed in variants(methods, seeds):
        # Stable per-trace/head seed, independent of allocation and reference output.
        effective_seed = (seed + int(identity.get('seed_offset', 0))) % (2**32)
        for total in totals:
            for share in shares:
                nk, nv, m = quota(n, total, share)
                k, v = choose(view, nk, nv, method, effective_seed)
                specs.append(('paired', float(total), float(share), method, seed, k, v))
    equal = {}
    for index, (kind, total, share, method, seed, k, v) in enumerate(specs):
        acc = accounting(view, k, v)
        if kind == 'paired':
            nk, nv, m = quota(n, total, share)
            expected = 2*int(view.costs[view.quantized & ~view.eligible].sum())
            # Above protected bytes include both high and low fields. Eligible
            # anchors are always two packets per chunk, plus m residual packets.
            expected += (2*n+m)*view.packet_bytes
            if acc['logical_payload_read_bytes'] != expected:
                raise AssertionError('Byte accounting does not match the exact quota')
            key = (method, seed, total)
            if key in equal and equal[key] != expected:
                raise AssertionError('Asymmetric profiles do NOT have equal payload')
            equal[key] = expected
        out = evaluate(group, k, v)
        metric = output_metrics(out, group.full, group.vref)
        row = {k:v for k,v in identity.items() if k != 'seed_offset'}
        row.update(kind=kind, total_residual_rate=total, k_priority=share,
                   method=method, mask_seed=seed, mask_row=index, block_n=block_n,
                   **acc, common_scale_and_pool_bytes=fixed_aux,
                   logical_payload_plus_common_bytes=acc['logical_payload_read_bytes']+fixed_aux,
                   equal_payload_checked=kind == 'paired',
                   controller_tolerance=None, controller_margin=None,
                   controller_estimated_error=None, controller_steps=None,
                   controller_stop_reason=None, rho_q=None,
                   residual_read_bytes=None, total_residual_bytes=None,
                   **metric)
        rows.append(row); masks.append(np.stack((k, v)))
    for tolerance in controller_tolerances:
        config = ControllerConfig(float(tolerance), float(controller_margin), controller_max_steps, block_n)
        result = run_controller(group.chunks, config)
        out = evaluate(group, result.read_k, result.read_v)
        metric = output_metrics(out, group.full, group.vref)
        row = {k: v for k, v in identity.items() if k != 'seed_offset'}
        row.update(kind='controller', total_residual_rate=None, k_priority=None,
                   method='query_aware_bound', mask_seed=None, mask_row=len(rows),
                   block_n=block_n, controller_tolerance=float(tolerance),
                   controller_margin=float(controller_margin),
                   controller_estimated_error=result.estimated_error,
                   controller_steps=result.steps, controller_stop_reason=result.stop_reason,
                   rho_q=result.rho, residual_read_bytes=result.residual_bytes,
                   total_residual_bytes=result.total_residual_bytes,
                   equal_payload_checked=False, **accounting(from_group(group, block_n),
                   result.read_k, result.read_v), **metric)
        rows.append(row)
        masks.append(np.stack((result.read_k, result.read_v)))
    return rows, np.packbits(np.stack(masks), axis=-1), group.n


def analyze_paths(paths, out, totals=('1.0','1.1'), shares=('0','.25','.5','.75','1'),
                  methods=('anchor_mass','staged_mass','random'), seeds=(42,43,44),
                  block_n=32, row_identity=None, controller_tolerances=(),
                  controller_margin=1.0, controller_max_steps=None):
    """CPU float64 counterfactuals. No selectors are injected into generation."""
    from kvbench_residual_diag.common import load_trace, output_metrics
    from kvbench_residual_diag.math_ops import build_group
    from .common import digest
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    paths = list(paths); validations = []; mask_arrays = {}; rows_count = 0
    writer = None; traces_metadata = []
    with gzip.open(out/'rows.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        for ti, path in enumerate(paths):
            data, meta = load_trace(path)
            traces_metadata.append(meta)
            fulls = []
            for hk in range(8):
                group = build_group(data, hk, block_n)
                fulls.append(group.full)
                qtokens = sum(c['tokens'] for c in group.chunks if c['quantized'])
                ptokens = sum(c['tokens'] for c in group.chunks if not c['quantized'])
                blocks = len({c['block'] for c in group.chunks})
                aux = 6*128*blocks+6*qtokens+4*128*ptokens
                identity = dict(dataset=meta['request']['dataset'], case_id=meta['request']['case_id'],
                                trace=Path(path).name, trace_arrays_sha256=meta['arrays_sha256'],
                                cache_ordinal=meta['cache_ordinal'], decode_call_index=meta['decode_call_index'],
                                context_tokens=meta['context_tokens'], kv_head=hk,
                                quantized_tokens=qtokens, fp16_pool_tokens=ptokens,
                                captured_block_n=meta['block_n'], captured_splits=meta['splits'])
                if row_identity:
                    identity.update(row_identity)
                identity['seed_offset'] = int(digest([identity['dataset'], identity['case_id'],
                            identity['cache_ordinal'], identity['decode_call_index'], hk])[:8], 16)
                rows, masks, nchunks = group_rows(
                    group, block_n, totals, shares, methods, seeds, identity, aux,
                    controller_tolerances, controller_margin, controller_max_steps)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows(rows); rows_count += len(rows)
                mask_arrays[f't{ti:03d}_h{hk}_packed'] = masks
                mask_arrays[f't{ti:03d}_h{hk}_nchunks'] = np.array(nchunks, np.int64)
                # Retain chunk geometry and mask order for audit without raw Q/KV.
                mask_arrays[f't{ti:03d}_h{hk}_geometry'] = np.asarray(
                    [[c['block'],c['chunk'],c['tokens'],c['quantized'],c['k_bytes'],c['v_bytes']] for c in group.chunks],np.int64)
            full = np.concatenate(fulls, axis=0)[None, :, :]
            metrics = output_metrics(full, data['reference_gpu_rot'])
            passed = (metrics['relative_l2_fro'] is not None and metrics['relative_l2_fro'] <= .02
                      and metrics['max_abs_element'] <= .02)
            validations.append(dict(trace=Path(path).name, passed=passed, **metrics))
            atomic_json(out/'validation.json', validations)
            if not passed:
                raise RuntimeError('Full-quantized/live-GPU reference mismatch; do not interpret this case: '+str(metrics))
            print(f'  analyzed trace {ti+1}/{len(paths)}: {Path(path).name}', flush=True)
    atomic_json(out/'validation.json', validations)
    np.savez_compressed(out/'masks.npz', **mask_arrays)
    atomic_json(out/'trace_metadata.json', traces_metadata)
    receipt = dict(schema='kvbench.asym.analysis.v1', status='completed', traces=len(paths), rows=rows_count,
                   all_captured_heads_used=True, selectors_use_full_output=False, oracle_stop=False,
                   gpu_selective_replay_executed=False, formal_task_score=None, tpot=None, measured_dram_bytes=None,
                   scope='Parent-trajectory CPU counterfactual; exact selected logical payload, NOT physical DRAM or online throughput',
                   file_hashes={p.name:sha_file(p) for p in out.iterdir() if p.is_file()})
    atomic_json(out/'receipt.json', receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--totals', nargs='+', default=['1.0','1.1'])
    p.add_argument('--shares', nargs='+', default=['0','.25','.5','.75','1'])
    p.add_argument('--block-n', type=int, choices=(16,32,64), default=32)
    p.add_argument('--controller-tolerances', nargs='+', type=float, default=[])
    p.add_argument('--controller-margin', type=float, default=1.0)
    p.add_argument('--controller-max-steps', type=int)
    a = p.parse_args(argv)
    paths = sorted(a.traces.glob('trace_*.json'))
    if not paths:
        raise ValueError('No original trace JSON files; raw arrays must still exist')
    analyze_paths(paths, a.out, a.totals, a.shares, block_n=a.block_n,
                  controller_tolerances=a.controller_tolerances,
                  controller_margin=a.controller_margin,
                  controller_max_steps=a.controller_max_steps)


if __name__ == '__main__':
    main()

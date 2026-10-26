#!/usr/bin/env python
"""CPU-only official scoring of eight complete attention-matched main cells.
No legacy-file fallback. All attempts, exact full-split IDs, common completion,
input hashes, source fingerprints and actual visual budgets are validated.
The original operational main table and its score artifacts stay untouched.
"""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cvm24h_score as SC

BENCHMARKS = {'textvqa': ('textvqa_val.jsonl', 5000, 100), 'ocrbench': ('ocrbench.jsonl', 1000, 1000)}
MODES = {'none': 'None', 'pre': 'RBM', 'fastv': 'FastV-k3', 'hilo': 'Hi-Lo'}
PAIRS = [('pre', 'fastv'), ('pre', 'hilo'), ('pre', 'none'),
         ('fastv', 'hilo'), ('fastv', 'none'), ('hilo', 'none')]
RESAMPLES = 20000
EXPECTED_PROTOCOL = {'attention': {'vision_config': 'sdpa', 'text_config': 'eager'},
                     'dtype': 'bfloat16', 'max_pixels': 4000000, 'max_new_tokens': 32,
                     'seed': 0, 'mrope': 'native', 'do_sample': False,
                     'model': 'Qwen/Qwen3-VL-8B-Instruct'}
WEIGHT_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def scaled_pair(pair, scale):
    factor = scale / 100
    return {'n_paired': pair['n_paired'], 'n_paired_answered': pair['n_paired_answered'],
            'mean_a': pair['mean_a'] * scale, 'mean_b': pair['mean_b'] * scale,
            'delta': pair['delta_pp'] * factor, 'ci95': [v * factor for v in pair['ci95_pp']],
            'sign_flip_p_two_sided': pair['sign_flip_p_two_sided'],
            'resamples': pair['resamples'], 'seed': pair['seed'], 'denominator': pair['denominator']}


def pformat(value):
    return '<0.0001' if value < 0.0001 else f'{value:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', type=Path,
                        default=ROOT / 'results/cvm2027_supplement_20261026/matched_main')
    args = parser.parse_args()
    os.chdir(ROOT)
    paths = {(bench, mode): args.prediction_dir / f'{mode}_{bench}.json'
             for bench in BENCHMARKS for mode in MODES}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit('Matched main incomplete; no report overwritten. Missing: ' + ', '.join(missing))
    sources, data, records, split_meta, expected_ids, split_hashes = {}, {}, {}, {}, {}, {}
    protocol = None
    shared_sources = None
    for bench, (split_name, expected_n, scale) in BENCHMARKS.items():
        split_path = ROOT / 'eval/full_splits' / split_name
        split_raw = split_path.read_bytes()
        reference = [json.loads(line) for line in split_raw.splitlines()]
        expected_ids[bench] = [str(row['id']) for row in reference]
        assert len(reference) == len(set(expected_ids[bench])) == expected_n
        split_meta[bench] = {str(row['id']): row for row in reference}
        split_hashes[bench] = {'path': str(split_path), 'sha256': sha(split_raw)}
        data[bench], records[bench], sources[bench] = {}, {}, {}
        for mode in MODES:
            path = paths[bench, mode]
            raw = path.read_bytes()
            cell = json.loads(raw)
            if not cell.get('complete'):
                raise SystemExit(f'{path.name} is a checkpoint/partial; no report overwritten')
            assert cell['expected_n'] == cell['n'] == expected_n, (path, 'expected count')
            assert cell['n_attempted'] == expected_n and cell['n_pending'] == 0, (path, 'unattempted IDs')
            assert not cell.get('protocol_failure'), (path, 'failed protocol')
            assert cell['keep_fraction'] == (1.0 if mode == 'none' else .25)
            rows_list = cell['per_sample']
            assert len(rows_list) == expected_n, (path, 'dropped attempted rows')
            assert [str(row['id']) for row in rows_list] == expected_ids[bench], (path, 'full-split ID/order drift')
            assert cell.get('benchmark', bench) == bench and cell.get('mode', mode) == mode
            current_protocol = cell['protocol']
            for key, expected in EXPECTED_PROTOCOL.items():
                assert current_protocol.get(key) == expected, (path, key, current_protocol.get(key), expected)
            assert WEIGHT_REVISION in json.dumps(current_protocol), (path, 'weight snapshot revision missing')
            if protocol is None:
                protocol = current_protocol
            assert current_protocol == protocol, (path, 'protocol differs')
            assert cell['source_hashes'], (path, 'missing source fingerprints')
            cfg = cell['configuration']
            assert cfg['fastv_k'] == 3 and cfg['hilo_prune_layer'] == 2
            if mode == 'hilo':
                for key, expected in {'hilo_keep_ratio': .25, 'hilo_candidate_ratio': .3,
                                      'hilo_use_fusion': True, 'hilo_fusion_strength': .3,
                                      'hilo_top_k_ratio': .1, 'hilo_attention_threshold': .05}.items():
                    assert cfg[key] == expected, (path, key, 'frozen Hi-Lo parameter changed')
            blind = cell['blind_p3_check']
            assert len(blind['warmup']) == 3 and blind['expected_formal_overlap'] == 100
            assert len(blind['formal_overlap']) == len({r['id'] for r in blind['formal_overlap']}) == 100
            for check in blind['warmup'] + blind['formal_overlap']:
                assert all(value for key, value in check.items() if key != 'id'), (path, check, 'P3 overlap parity failed')
            reference_path = Path(blind['source'])
            assert reference_path.exists() and sha(reference_path.read_bytes()) == blind['sha256'], (path, 'P3 reference changed')
            source = cell['source_hashes']
            if shared_sources is None:
                shared_sources = source
            assert source == shared_sources, (path, 'shared source fingerprints differ')
            # Only identities of shared inputs/dependencies are required equal;
            # the official Hi-Lo model remains a different adapted model class.
            for key in set(source) & set(shared_sources):
                if any(word in key.lower() for word in ('processor', 'weight', 'helper', 'timing_v2')):
                    assert source[key] == shared_sources[key], (path, key, 'source fingerprint differs')
            for row in rows_list:
                sid = str(row['id'])
                assert row['gt'] == split_meta[bench][sid]['gt'], (path, sid, 'GT drift')
                assert row.get('question') == split_meta[bench][sid]['question'], (path, sid, 'prompt/question drift')
                if row.get('error'):
                    row['skipped'] = True
                else:
                    assert row.get('answer') is not None, (path, sid, 'unmarked missing answer')
                if row.get('skipped'):
                    assert row.get('error'), (path, sid, 'failure reason missing')
            rows = {str(row['id']): row for row in rows_list}
            failures = sum(bool(row.get('skipped')) for row in rows_list)
            assert cell['n_error'] == failures and cell['n_scored'] == expected_n - failures, (path, 'failure counts')
            records[bench][mode], data[bench][mode] = rows, cell
            sources[bench][mode] = {'path': str(path.resolve()), 'sha256': sha(raw),
                                     'source_hashes': source,
                                     'runner_checks': {key: value for key, value in cell.items()
                                                       if any(word in key.lower() for word in ('parity', 'overlap', 'warmup', 'consistency', 'p3_check'))}}

    validation = {'complete': True, 'cell_count': 8, 'protocol_equal_all_eight': True,
                  'full_split_ids_and_order_match': True, 'input_and_budget': {}}
    for bench, (split_name, expected_n, scale) in BENCHMARKS.items():
        rows = records[bench]
        # Validate every successful attempt, even if another method failed it.
        for mode, rr in rows.items():
            for sid, row in rr.items():
                if row.get('skipped'):
                    continue
                n, k = row['n_image_full'], row['n_image_kept']
                assert n > 0 and k == (n if mode == 'none' else max(1, round(.25 * n))), (bench, mode, sid, 'invalid actual budget')
                assert len(row['input_ids_sha256']) == 64 and row['n_prompt_full'] >= n
                assert row['actual_k_verified'], (bench, mode, sid, 'unchecked budget')
                kept = row['kept_image_ids']
                if mode == 'none':
                    assert row['kept_all'] and (kept is None or kept == list(range(n))), (bench, mode, sid, 'nonidentity None')
                else:
                    assert len(kept) == k and kept == sorted(set(kept)) and min(kept) >= 0 and max(kept) < n, (bench, mode, sid, 'invalid kept unit IDs')
        common = [sid for sid in expected_ids[bench] if all(not rows[m][sid].get('skipped') for m in MODES)]
        assert common, (bench, 'no common answered IDs')
        for sid in common:
            n_values = {rows[m][sid]['n_image_full'] for m in MODES}
            assert len(n_values) == 1, (bench, sid, 'original visual input counts differ')
            n = int(next(iter(n_values)))
            assert n > 0 and rows['none'][sid]['n_image_kept'] == n
            expected_k = max(1, round(.25 * n))
            assert all(rows[m][sid]['n_image_kept'] == expected_k for m in ('pre', 'fastv', 'hilo')), (bench, sid, 'actual K mismatch')
            hashes = {rows[m][sid]['input_ids_sha256'] for m in MODES}
            assert len(hashes) == 1 and len(next(iter(hashes))) == 64, (bench, sid, 'input IDs differ')
            assert len({rows[m][sid]['n_prompt_full'] for m in MODES}) == 1, (bench, sid, 'full prompt lengths differ')
            assert len({rows[m][sid]['image'] for m in MODES}) == 1, (bench, sid, 'image path differs')
        validation['input_and_budget'][bench] = {
            'common_answered_n': len(common), 'checks_per_common_id': ['same original N', 'same original input-ID SHA256', 'same full prompt length', 'same image source', 'None K=N', 'RBM/FastV/Hi-Lo K=round(.25N)'],
            'mismatches': 0,
            'mean_original_n': statistics.mean(rows['none'][sid]['n_image_full'] for sid in common),
            'mean_kept_k': statistics.mean(rows['pre'][sid]['n_image_kept'] for sid in common)}

    result = {'schema_version': 1, 'complete': True, 'cell_count': 8,
              'scope': 'Attention-matched Qwen3-VL full TextVQA-5000 and OCRBench-1000; compressed retention .25 and uncompressed None',
              'protocol': protocol, 'validation': validation, 'sources': sources, 'split_sources': split_hashes,
              'score_sources': {
                  'textvqa': {'commit': SC.EX.REFERENCE_COMMIT, 'url': SC.EX.REFERENCE_URL,
                              'sha256': SC.EX.REFERENCE_SHA256,
                              'official_annotations_sha256': '4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d'},
                  'ocrbench': {'commit': SC.COMMIT, 'url': SC.OCR_URL, 'sha256': SC.OCR_SHA}},
              'statistics': {'resamples': RESAMPLES, 'seed_rule': '20261500 + benchmark_index*100 + pair_index*2; common-answered uses +1',
                             'pair_order': PAIRS, 'failures': 'zero in all-attempted primary score; all IDs retained',
                             'common_answered': 'pair-specific sensitivity; all-four-method intersection also reported',
                             'p_value': 'two-sided paired sign-flip test; not evidence of equivalence'},
              'limitation': 'Attention, preprocessing, weights and final budgets match. Hi-Lo still uses the official adapted model implementation; no-compression and P3 overlap checks bound this implementation difference.',
              'benchmarks': {}}
    md = ['# Attention-matched modern-baseline main comparison (2026-10-26)', '', result['scope']+'.', '',
          'This is a new protocol: shared vision SDPA / decoder eager, bf16, native positions, four-million-pixel cap, greedy decoding up to 32 new tokens. Both full benchmark splits include all attempted IDs and assign zero to failures. Historical operational comparisons are retained separately. The old 50/75% curves use their original attention protocol and are not relabelled as matched curves.', '',
          '| Benchmark | None@100% | RBM@25% | FastV-k3@25% | Hi-Lo@25% | Answered None / RBM / FastV / Hi-Lo |',
          '|---|---:|---:|---:|---:|---|']
    scored = {}
    for bench, (split_name, expected_n, scale) in BENCHMARKS.items():
        scored[bench] = {mode: SC.SC[bench](rows) for mode, rows in records[bench].items()}
        common = [sid for sid in expected_ids[bench] if all(not records[bench][m][sid].get('skipped') for m in MODES)]
        output = {'n_attempted': expected_n, 'unit': 'accuracy (%)' if bench == 'textvqa' else 'points per 1000',
                  'cells': {}, 'pairs': {}, 'common_all_methods_ids': common}
        for mode in MODES:
            rows, scores = records[bench][mode], scored[bench][mode]
            failures = {sid: rows[sid]['error'] for sid in rows if rows[sid].get('skipped')}
            answered = [sid for sid in rows if sid not in failures]
            output['cells'][mode] = {
                'label': MODES[mode], 'keep_fraction': 1.0 if mode == 'none' else .25,
                'n_attempted': expected_n, 'n_answered': len(answered), 'n_failed': len(failures),
                'attempted_score': sum(scores.values()) / expected_n * scale,
                'answered_score': sum(scores[sid] for sid in answered) / max(len(answered), 1) * scale,
                'common_all_methods_score': sum(scores[sid] for sid in common) / len(common) * scale,
                'failure_ids': sorted(failures), 'failure_reasons': failures,
                'actual_k': {'n': len(answered),
                             'mean_original_n': statistics.mean(rows[sid]['n_image_full'] for sid in answered) if answered else None,
                             'mean_kept_k': statistics.mean(rows[sid]['n_image_kept'] for sid in answered) if answered else None}}
        cells = output['cells']
        md.append(f'| {bench} n={expected_n} | ' + ' | '.join(f'{cells[m]["attempted_score"]:.2f}' for m in MODES)
                  + ' | ' + ' / '.join(str(cells[m]['n_answered']) for m in MODES) + ' |')
        result['benchmarks'][bench] = output
    for bi, (bench, (split_name, expected_n, scale)) in enumerate(BENCHMARKS.items()):
        md += ['', f'## {bench}: all six planned pairs', '',
               '| A − B | Attempted Δ [95% CI] | p | Common-answered Δ [95% CI] | Common n |',
               '|---|---:|---:|---:|---:|']
        for pi, (a, b) in enumerate(PAIRS):
            seed = 20261500 + bi * 100 + pi * 2
            pa = SC.EX.paired(scored[bench][a], records[bench][a], scored[bench][b], records[bench][b], RESAMPLES, seed, include_skips=True)
            pb = SC.EX.paired(scored[bench][a], records[bench][a], scored[bench][b], records[bench][b], RESAMPLES, seed + 1, include_skips=False)
            av, bv = scaled_pair(pa, scale), scaled_pair(pb, scale)
            skip_a = {sid for sid, row in records[bench][a].items() if row.get('skipped')}
            skip_b = {sid for sid, row in records[bench][b].items() if row.get('skipped')}
            result['benchmarks'][bench]['pairs'][f'{a}_minus_{b}'] = {
                'label_a': MODES[a], 'label_b': MODES[b], 'attempted': av, 'common_answered': bv,
                'failure_ids': {'a_only': sorted(skip_a - skip_b), 'b_only': sorted(skip_b - skip_a), 'both': sorted(skip_a & skip_b)}}
            md.append(f'| {MODES[a]} − {MODES[b]} | {av["delta"]:+.2f} [{av["ci95"][0]:+.2f}, {av["ci95"][1]:+.2f}] | {pformat(av["sign_flip_p_two_sided"])} | {bv["delta"]:+.2f} [{bv["ci95"][0]:+.2f}, {bv["ci95"][1]:+.2f}] | {bv["n_paired"]} |')
        v = validation['input_and_budget'][bench]
        md += ['', f'Input hashes, original visual N, full prompt lengths and image source match on all {v["common_answered_n"]} commonly answered IDs. All compressed methods retain exactly round(.25N) per ID; mean N={v["mean_original_n"]:.2f}, K={v["mean_kept_k"]:.2f}. Failure IDs and reasons are retained in the JSON.']
    md += ['', 'All paired intervals use 20,000 resamples of paired question scores and fixed seeds. Common-answered sensitivity does not drop failures from the primary denominator. ' + result['limitation'], '',
           'Reproduce from the worktree root:', '', '```bash',
           '/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/cvm24h_score_matched.py', '```', '',
           'Original predictions are read only. The JSON records every input SHA256, pinned scoring references, configuration, failure IDs, all twelve planned pair comparisons and validation evidence.']
    stem = ROOT / 'reports/cvm2027_matched_main_20261026'
    write_atomic(stem.with_suffix('.json'), json.dumps(result, indent=2) + '\n')
    write_atomic(stem.with_suffix('.md'), '\n'.join(md) + '\n')
    print(json.dumps({'complete': True, 'cells': 8, 'protocol': protocol,
                      'scores': {bench: {m: c['attempted_score'] for m, c in b['cells'].items()}
                                 for bench, b in result['benchmarks'].items()}}, indent=2))


if __name__ == '__main__':
    main()

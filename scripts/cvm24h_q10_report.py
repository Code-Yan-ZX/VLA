#!/usr/bin/env python
"""CPU-only Q10 closeout and optional Hi-Lo75 runtime planning.
Validates all existing full-5000 extension predictions and scores every complete
cell with the pinned official scorer. Never launches GPU work or a watcher.
The pre-fixed 1,000-question curve remains the main budget comparison.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results/cvm2027_supplement_20261026'
LABELS = {'rb': 'RBM', 'fv': 'FastV-k3', 'hl': 'Hi-Lo'}
PREFIX = {'rb': 'pre', 'fv': 'fastv', 'hl': 'hilo'}
KEEP = ('0.50', '0.75')
TZ = ZoneInfo('Asia/Shanghai')
DEFAULT_COMPUTE_DEADLINE = datetime(2026, 10, 26, 18, 30, tzinfo=TZ)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_text(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def china_timestamp(value):
    """Interpret an ISO timestamp without an offset as China local time."""
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'Invalid ISO timestamp: {value}') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ)
    return parsed.astimezone(TZ)


def runtime_budget(deadline=DEFAULT_COMPUTE_DEADLINE):
    source = D / 'plan/timing_hilo_corrected.json'
    raw = source.read_bytes()
    timing = json.loads(raw)
    means, median_means = {}, {}
    for keep in (.5, .75):
        rows = [r for r in timing['results'][f'hilo_{keep}']['requests'] if r['benchmark'] == 'textvqa']
        assert len(rows) == 100 and all(r['status'] == 'ok' for r in rows)
        means[str(keep)] = statistics.mean(rep['total_s'] for r in rows for rep in r['repeats_raw'])
        median_means[str(keep)] = statistics.mean(r['total_s'] for r in rows)
    previous_path = D / 'predictions/q10_hilo_textvqa5k_keep0.50.json'
    previous_raw = previous_path.read_bytes()
    previous = json.loads(previous_raw)
    ratio = means['0.75'] / means['0.5']
    old_scaled = previous['wall_s'] * ratio
    direct = means['0.75'] * 5000
    load_buffer = max(120., previous['load_s'])
    stop_buffer = 60
    required = math.ceil(max(old_scaled, direct) * 1.25 + load_buffer + stop_buffer)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=TZ)
    deadline = deadline.astimezone(TZ)
    return {
        'quality_scores_not_used': True,
        'P3_textvqa_raw_request_mean_total_s': means,
        'P3_textvqa_mean_of_request_medians_total_s': median_means,
        'P3_source': str(source), 'P3_sha256': sha(raw),
        'old_hilo50_source': str(previous_path), 'old_hilo50_sha256': sha(previous_raw),
        'old_hilo50_full5000_wall_s': previous['wall_s'], 'old_hilo50_load_s': previous['load_s'],
        'ratio_P3_mean75_over_mean50': ratio, 'projection_old50_scaled_s': old_scaled,
        'projection_direct_P3_mean75_times5000_s': direct,
        'policy': 'max(old full50 wall × P3 mean75/mean50, P3 mean75 × 5000) × 1.25 + max(120s, observed old load) + 60s stop buffer',
        'contingency_multiplier': 1.25, 'load_buffer_s': load_buffer, 'stop_buffer_s': stop_buffer,
        'required_conservative_seconds': required, 'hard_gpu_stop': deadline.isoformat(),
        'latest_start': (deadline - timedelta(seconds=required)).isoformat(),
        'limitation': 'Timing estimate, not a guaranteed bound: the fixed P3 sample excludes five earlier TextVQA failures and uses decoder eager, while old Q10 Hi-Lo uses SDPA. The 25% margin and fixed buffers cover some scheduling uncertainty.',
        'launch_rule': f'Only root may decide after all eight matched-main cells complete. Start only when remaining time to {deadline.isoformat()} is at least this budget; otherwise record the optional cell as skipped for time. No score-dependent decision.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compute-deadline', type=china_timestamp, default=DEFAULT_COMPUTE_DEADLINE,
                        help='Hard stop as an ISO timestamp; an omitted offset means China time (default: 2026-10-26T18:30:00+08:00). CPU planning only; never starts a task')
    parser.add_argument('--decision-at', type=china_timestamp, help='Optional ISO China-time timestamp for a CPU time-availability check; never starts a task')
    args = parser.parse_args()
    reproduce_command = ['/home/dell/miniconda3/envs/qwen3vl_clean/bin/python', 'scripts/cvm24h_q10_report.py',
                         '--compute-deadline', args.compute_deadline.isoformat()]
    if args.decision_at:
        reproduce_command += ['--decision-at', args.decision_at.isoformat()]
    wanted_rows = [json.loads(line) for line in (ROOT / 'eval/full_splits/textvqa_val.jsonl').read_text().splitlines()]
    wanted = [str(r['id']) for r in wanted_rows]
    gt_by_id = {str(r['id']): r['gt'] for r in wanted_rows}
    none_path = D / 'predictions/p1_none_textvqa_full.json'
    none_raw = none_path.read_bytes()
    none = json.loads(none_raw)
    none_rows = {str(r['id']): r for r in none['per_sample']}
    result = {
        'scope': 'Optional Q10 full-5000 TextVQA 50/75% extension, original mixed attention protocol',
        'planned_extension_cells': 6, 'completed_extension_cells': 0, 'full_curve_complete': False,
        'protocol_note': 'Original None/RBM/FastV eager and official Hi-Lo SDPA. These runs do not form the new vision-SDPA/text-eager matched main. Fixed TV1000 curves remain the complete pre-fixed comparison even if this optional full grid finishes.',
        'none_reference': {'path': str(none_path), 'sha256': sha(none_raw)},
        'cells': {}, 'missing_cells': [], 'runtime_budget': runtime_budget(args.compute_deadline),
        'reproduce_command': shlex.join(reproduce_command)}
    for mi, method in enumerate(LABELS):
        for ki, keep in enumerate(KEEP):
            key = f'{method}_{keep}'
            prediction_path = D / f'predictions/q10_{PREFIX[method]}_textvqa5k_keep{keep}.json'
            if not prediction_path.exists():
                result['missing_cells'].append({'method': LABELS[method], 'keep': float(keep), 'path': str(prediction_path), 'reason': 'No complete prediction file; no score inferred'})
                continue
            raw = prediction_path.read_bytes()
            try:
                prediction = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                result['missing_cells'].append({'method': LABELS[method], 'keep': float(keep),
                    'path': str(prediction_path), 'sha256': sha(raw),
                    'reason': 'Prediction JSON is incomplete or unreadable; preserved without scoring'})
                continue
            rr = prediction['per_sample']
            if prediction['n'] < 5000 or len(rr) < 5000:
                result['missing_cells'].append({'method': LABELS[method], 'keep': float(keep),
                    'path': str(prediction_path), 'sha256': sha(raw),
                    'reason': f'Only {len(rr)} prediction rows; incomplete file preserved without scoring'})
                continue
            assert prediction['n'] == len(rr) == len({str(r['id']) for r in rr}) == 5000
            assert [str(r['id']) for r in rr] == wanted
            assert all(r['gt'] == gt_by_id[str(r['id'])] for r in rr)
            successful = [r for r in rr if not r.get('skipped') and not r.get('error')]
            if method == 'hl':
                assert prediction['keep_ratio'] == float(keep) and prediction['attn'] == 'sdpa'
                assert all(r['n_image_kept'] == max(1, round(float(keep) * r['n_image_full'])) for r in successful)
            common = [r for r in successful if not none_rows[str(r['id'])].get('skipped')]
            assert all(r['n_image_kept'] == max(1, round(float(keep) * none_rows[str(r['id'])]['n_image_full'])) for r in common)
            score_path = D / f'plan/scores/full_q10_{method}_{keep}.json'
            score = json.loads(score_path.read_text()) if score_path.exists() else None
            if not score or score['sha_a'] != sha(raw) or score['sha_b'] != sha(none_raw):
                # Existing score artifacts with identical input SHA256s retain their original seeds and results.
                command = [sys.executable, str(ROOT / 'scripts/cvm24h_score.py'), 'textvqa', str(prediction_path), str(none_path),
                           '--seed-int', str(80 + mi * 2 + ki), '--out', str(score_path)]
                with score_path.with_suffix('.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
                score = json.loads(score_path.read_text())
            assert score['A']['n'] == score['B']['n'] == 5000 and score['resamples'] == 20000
            log_path = D / f'logs/{prediction_path.stem}.log'
            log_raw = log_path.read_bytes() if log_path.exists() else b''
            result['cells'][key] = {
                'method': LABELS[method], 'keep': float(keep), 'n_attempted': 5000,
                'n_answered': len(successful), 'official_score_percent': score['A']['attempted_score'] * 100,
                'prediction_source': str(prediction_path), 'prediction_sha256': sha(raw),
                'score_source': str(score_path), 'attempted_pair': score['paired_attempted_all'],
                'common_pair': score['paired_common_answered_sensitivity'],
                'failure_ids': [str(r['id']) for r in rr if r.get('skipped') or r.get('error')],
                'actual_budget_common_n': len(common), 'actual_budget_mismatches': 0,
                'raw_log': str(log_path), 'raw_log_sha256': sha(log_raw) if log_raw else None}
    result['completed_extension_cells'] = len(result['cells'])
    result['full_curve_complete'] = len(result['cells']) == 6
    if args.decision_at:
        decision = args.decision_at
        latest = datetime.fromisoformat(result['runtime_budget']['latest_start'])
        result['runtime_budget']['decision_at'] = decision.isoformat()
        result['runtime_budget']['time_budget_available'] = decision <= latest
        result['runtime_budget']['automatic_gpu_launch'] = False
    count = result['completed_extension_cells']
    md = [f'# Optional Q10 TextVQA full-split extension: {count}/6 cells complete (2026-10-26)', '',
          ('The optional full grid is complete. It is appended independently; the pre-fixed TV1000 curve and its figures remain unchanged.' if result['full_curve_complete'] else
           '**This optional extension is incomplete.** Every complete cell is reported; missing cells are not inferred from subset scores. The pre-fixed TV1000 curve remains the complete three-method budget comparison.'), '',
          result['protocol_note'], '',
          'Every complete cell has all 5,000 attempted IDs. Official ten-answer TextVQA scores use failures as zero and 20,000 paired bootstrap resamples against the old None reference. All source hashes, failure IDs and budgets are validated.', '',
          '| Method | Keep | Official score (%) | Answered / attempted | Δ vs None (pp) [95% CI] | Common-answered Δ (pp) [95% CI] |',
          '|---|---:|---:|---:|---:|---:|']
    for cell in result['cells'].values():
        a, b = cell['attempted_pair'], cell['common_pair']
        md.append(f'| {cell["method"]} | {cell["keep"]:.2f} | {cell["official_score_percent"]:.2f} | {cell["n_answered"]} / 5000 | {a["delta_pp"]:+.2f} [{a["ci95_pp"][0]:+.2f}, {a["ci95_pp"][1]:+.2f}] | {b["delta_pp"]:+.2f} [{b["ci95_pp"][0]:+.2f}, {b["ci95_pp"][1]:+.2f}] |')
    for cell in result['missing_cells']:
        md.append(f'\nMissing optional cell: {cell["method"]} keep={cell["keep"]:.2f}; {cell["reason"]}.')
    budget = result['runtime_budget']
    md += ['', '## Time-only contingency plan for the final optional Hi-Lo75 cell', '',
           f'P3 TextVQA mean total request time over all raw repeats is {budget["P3_textvqa_raw_request_mean_total_s"]["0.5"]:.6f}s at 50% and {budget["P3_textvqa_raw_request_mean_total_s"]["0.75"]:.6f}s at 75%. The completed old full-5000 Hi-Lo50 run took {budget["old_hilo50_full5000_wall_s"]:.1f}s, plus {budget["old_hilo50_load_s"]:.1f}s loading.', '',
           f'Using `{budget["policy"]}` gives a conservative reservation of **{budget["required_conservative_seconds"]} seconds**. To respect the hard GPU stop at **{args.compute_deadline.strftime("%Y-%m-%d %H:%M:%S")} China**, the latest start is **{datetime.fromisoformat(budget["latest_start"]).strftime("%Y-%m-%d %H:%M:%S")} China**.', '',
           budget['launch_rule'], '', budget['limitation'], '',
           'Reproduce or refresh this report without GPU work:', '', '```bash', result['reproduce_command'], '```', '',
           'This script never starts an evaluation or watcher. It computes or refreshes official scores only for existing complete prediction files and preserves unchanged score inputs/results.']
    if args.decision_at:
        state = ('already complete; no optional GPU action is required' if result['full_curve_complete'] else
                 'sufficient under the reserved budget' if budget['time_budget_available'] else
                 'insufficient; skip the optional cell under the fixed time rule')
        md += ['', f'Time check at {budget["decision_at"]}: **{state}**. Root must also confirm all eight matched-main cells have completed; this CPU check launches nothing.']
    stem = ROOT / 'reports/cvm2027_q10_extension_20261026'
    atomic_text(stem.with_suffix('.json'), json.dumps(result, indent=2) + '\n')
    atomic_text(stem.with_suffix('.md'), '\n'.join(md) + '\n')
    print(json.dumps({'completed': count, 'planned': 6, 'full_curve_complete': result['full_curve_complete'],
                      'required_seconds': budget['required_conservative_seconds'], 'latest_start': budget['latest_start']}, indent=2))
    # The sole closeout entry refreshes standalone full-grid artifacts when ready;
    # the builder keeps the pre-fixed TV1000 main figures unchanged.
    subprocess.run([sys.executable, str(ROOT / 'scripts/cvm24h_build_tables.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env bash
# Root-invoked, single-cell closeout only. No watcher and no automatic queue.
# --check-only prints a CPU preflight decision and writes no launch/log files.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=${CVM_PYTHON:-/home/dell/miniconda3/envs/qwen3vl_clean/bin/python}
exec "$P" - "$@" <<'PY'
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
import cvm24h_q10_report as Q10
import cvm24h_validate_timing_v2 as VT

TZ = ZoneInfo('Asia/Shanghai')
HARD_LIMIT = datetime(2026, 10, 26, 19, 0, tzinfo=TZ)
REQUIRED = 3445
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
UPSTREAM = Path('/tmp/Hi-Lo_Prune')
COMMIT = '29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6'
D = ROOT / 'results/cvm2027_supplement_20261026'
TARGET = D / 'predictions/q10_hilo_textvqa5k_keep0.75.json'
SPLIT = ROOT / 'eval/full_splits/textvqa_val.jsonl'
FROZEN = {
    'scripts/cvm24h_hilo_eval.py': '0614000b66e9ab6bb4190490e92b8a06d9839c3935a586e2d7a49b24bb4c9067',
    'src/v3_premerger/baselines_hf.py': 'cf12c4a747d85ab5ac7b67e5c9b4462b117d1e9a8d03f3ebfb4f17ae3269e7ca',
    'scripts/cvm24h_matched_quality.py': '67ae3554264dbd780ab98609623dec9111bd9b88f9136d446bbb7c75ae4e5834',
    'scripts/cvm24h_timing_v2.py': '25b526d6cadc809095f28199bfdde754bb9af2dc15d954ab05a05f8f0a3faa1c',
    'eval/full_splits/textvqa_val.jsonl': '8f50c15f2a560ed22ae5a918c4fa409df54d80078a84fe6b89d216e6d858f886',
    str(UPSTREAM / 'hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py'): 'abbf1286a277b356e4da5e169314ae257b043d83ecc6b66720dbe40ee2160cad',
    str(UPSTREAM / 'hilo_prune/selection.py'): '3732d070670aef31ef6ec41eb3043c2e6e93c602a81067140b0ce1653d4b5b39',
    str(UPSTREAM / 'hilo_prune/patches.py'): '8d4b96cac1e0b3eeb26e80930ee153ab3465e6af789788044af7c1deddb27e50',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def remaining(deadline, now=None):
    return deadline.timestamp() - (time.time() if now is None else now)


def enough_time(deadline, now=None):
    return remaining(deadline, now) >= REQUIRED


def frozen_sources():
    hashes = {}
    for name, expected in FROZEN.items():
        hashes[name] = sha(ROOT / name)
        require(hashes[name] == expected, f'frozen source changed: {name}')
    hashes['scripts/cvm24h_run_optional_hilo75.sh'] = sha(ROOT / 'scripts/cvm24h_run_optional_hilo75.sh')
    hashes['scripts/cvm24h_q10_report.py'] = sha(ROOT / 'scripts/cvm24h_q10_report.py')
    head = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == COMMIT, 'Hi-Lo upstream commit changed')
    return {'sha256': hashes, 'hilo_commit': head}


def completed_prerequisites():
    report_path = ROOT / 'reports/cvm2027_matched_main_20261026.json'
    require(report_path.exists(), 'matched main has no complete scored eight-cell report yet')
    report = read(report_path)
    require(report.get('complete') and report.get('cell_count') == 8, 'matched main report is incomplete')
    validation = report['validation']
    require(validation['complete'] and validation['cell_count'] == 8 and
            validation['protocol_equal_all_eight'] and validation['full_split_ids_and_order_match'],
            'matched main validation failed')
    sources = {}
    shared = None
    for bench, n, filename in [('textvqa', 5000, 'textvqa_val.jsonl'), ('ocrbench', 1000, 'ocrbench.jsonl')]:
        split = ROOT / 'eval/full_splits' / filename
        require(report['split_sources'][bench]['sha256'] == sha(split), f'{bench} split SHA changed')
        ids = [str(json.loads(line)['id']) for line in split.read_text().splitlines() if line.strip()]
        for mode in ('none', 'pre', 'fastv', 'hilo'):
            path = D / 'matched_main' / f'{mode}_{bench}.json'
            source = report['sources'][bench][mode]
            require(Path(source['path']).resolve() == path.resolve() and source['sha256'] == sha(path),
                    f'stale matched report source: {mode}/{bench}')
            cell = read(path)
            require(cell['complete'] and cell['expected_n'] == cell['n'] == cell['n_attempted'] == n and
                    cell['n_pending'] == 0 and not cell.get('protocol_failure'), f'partial matched cell: {mode}/{bench}')
            require(cell['n_error'] == 0 and cell['n_scored'] == n and
                    all(row['attempted'] and row['status'] == 'ok' and not row.get('error') and not row.get('skipped')
                        for row in cell['per_sample']), f'matched cell has failures: {mode}/{bench}')
            require([str(row['id']) for row in cell['per_sample']] == ids, f'matched IDs changed: {mode}/{bench}')
            require(cell['protocol'] == report['protocol'] and
                    cell['protocol']['attention'] == {'vision_config': 'sdpa', 'text_config': 'eager'} and
                    cell['protocol']['weight_revision'] == REVISION, 'matched shared protocol mismatch')
            if shared is None:
                shared = cell['source_hashes']
            require(cell['source_hashes'] == shared == source['source_hashes'], 'matched source fingerprints differ')
            sources[f'{mode}_{bench}'] = source['sha256']
    source_map = {'runner_sha256': 'scripts/cvm24h_matched_quality.py',
                  'helper_sha256': 'src/v3_premerger/baselines_hf.py',
                  'v2_sha256': 'scripts/cvm24h_timing_v2.py',
                  'hilo_model_sha256': str(UPSTREAM / 'hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py'),
                  'hilo_selection_sha256': str(UPSTREAM / 'hilo_prune/selection.py'),
                  'hilo_patches_sha256': str(UPSTREAM / 'hilo_prune/patches.py')}
    for key, path in source_map.items():
        require(shared[key] == sha(ROOT / path), f'matched current-source SHA changed: {key}')
    p3_paths = [D / 'plan/timing_harness_corrected.json', D / 'plan/timing_hilo_corrected.json']
    p3 = VT.validate(p3_paths)
    require(p3['complete'] and len(p3['cells']) == 10, 'corrected P3 is not ten complete validated cells')
    p3_report_path = ROOT / 'reports/cvm2027_corrected_quality_cost.json'
    p3_report = read(p3_report_path)
    for engine, path in zip(('harness', 'hilo'), p3_paths):
        data = read(path)
        require(data['script_sha256'] == sha(ROOT / 'scripts/cvm24h_timing_v2.py'), 'P3 frozen-source SHA changed')
        require(p3_report['provenance'][engine]['sha256'] == sha(path), 'P3 score source is stale')
        require(data['timing_ids_sha256'] == sha(D / 'plan/p3_timing_ids.json'), 'P3 fixed request IDs changed')
    return {'matched_report_sha256': sha(report_path), 'matched_prediction_sha256': sources,
            'p3_report_sha256': sha(p3_report_path), 'p3_raw_sha256': {p.name: sha(p) for p in p3_paths},
            'matched_cells': 8, 'P3_cells': 10}, shared


def cache_pin(shared):
    from huggingface_hub.constants import HF_HUB_CACHE
    cache = Path(HF_HUB_CACHE) / 'models--Qwen--Qwen3-VL-8B-Instruct'
    require((cache / 'refs/main').read_text().strip() == REVISION, 'offline model refs/main is not the pinned revision')
    snapshot = cache / 'snapshots' / REVISION
    hashes = {}
    for filename, expected in shared['processor_config_sha256'].items():
        path = snapshot / filename
        hashes[filename] = sha(path) if path.exists() else None
        require(hashes[filename] == expected, f'processor/config cache changed: {filename}')
    index = snapshot / 'model.safetensors.index.json'
    require(sha(index) == shared['weight_index_sha256'], 'weight index changed')
    for shard in shared['weight_shards']:
        path = snapshot / shard['file']
        require(path.stat().st_size == shard['bytes'] and path.resolve().name == shard['cache_blob_sha256'],
                f'content-addressed weight cache changed: {shard["file"]}')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers')}
    require(versions == {'torch': '2.10.0', 'transformers': '4.57.6'} or
            versions == {'torch': '2.10.0+cu128', 'transformers': '4.57.6'}, 'package version changed')
    return {'cache': str(cache), 'revision': REVISION, 'processor_config_sha256': hashes,
            'generation_config_sha256': sha(snapshot / 'generation_config.json'),
            'weight_index_sha256': sha(index), 'weight_shards': shared['weight_shards'],
            'weight_hash_policy': shared['weight_hash_policy'], 'package_versions': versions}


def gpu_idle():
    query = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
    result = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10)
    require(not result.stdout.strip(), f'GPU compute processes are still active: {result.stdout.strip()}')
    return {'query': query, 'stdout': result.stdout, 'checked_at': datetime.now(TZ).isoformat()}


def main():
    parser = argparse.ArgumentParser(description='Root-invoked optional old-protocol Hi-Lo75; never watches or queues other work.')
    parser.add_argument('--check-only', action='store_true', help='Read-only CPU gate check; no launch, fingerprint or log files written')
    parser.add_argument('--compute-deadline', type=Q10.china_timestamp, default=HARD_LIMIT,
                        help='Optional earlier ISO hard stop; default and latest allowed is 2026-10-26T19:00:00+08:00')
    args = parser.parse_args()
    require(args.compute_deadline <= HARD_LIMIT, 'optional hard stop may not exceed 19:00 China')
    tag = datetime.now(TZ).strftime('%Y%m%dT%H%M%S%f') + f'_{os.getpid()}'
    record_path = D / 'plan' / f'optional_hilo75_launch_{tag}.json'
    log_path = D / 'logs' / f'q10_hilo_textvqa5k_keep0.75_completion_{tag}.log'
    report_log = D / 'logs' / f'q10_hilo_textvqa5k_keep0.75_closeout_{tag}.log'
    record = {'status': 'preflight', 'created_at': datetime.now(TZ).isoformat(), 'check_only': args.check_only,
              'quality_scores_used_for_launch_decision': False, 'required_conservative_seconds': REQUIRED,
              'hard_gpu_stop': args.compute_deadline.isoformat(),
              'TERM_at': (args.compute_deadline - timedelta(seconds=30)).isoformat(), 'KILL_after_TERM_s': 25,
              'target': str(TARGET), 'completion_log': str(log_path), 'closeout_log': str(report_log),
              'configuration': {'model': 'Qwen/Qwen3-VL-8B-Instruct', 'benchmark': 'textvqa', 'n': 5000,
                                'keep_fraction': .75, 'candidate_fraction': .75, 'candidate_rule': 'official default 0.3 clamped to >= keep',
                                'prune_layer': 2, 'fusion': True, 'fusion_strength': .3, 'top_k_ratio': .1,
                                'attention_threshold': .05, 'fusion_layers': [0, 1], 'attention': 'sdpa',
                                'dtype': 'bfloat16', 'mrope': 'native', 'max_pixels': 4000000,
                                'seed': 0, 'do_sample': False, 'max_new_tokens': 32, 'bs': 1},
              'original_main_deadline_unchanged': '2026-10-26T18:30:00+08:00'}

    def save():
        if not args.check_only:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = record_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(record, indent=2) + '\n')
            tmp.replace(record_path)

    try:
        require(not TARGET.exists(), 'target prediction already exists; refuse overwrite, root must inspect it')
        require(enough_time(args.compute_deadline), 'insufficient time for the fixed 3445-second reservation')
        record['source_fingerprints'] = frozen_sources()
        record['prerequisites'], shared = completed_prerequisites()
        record['model_cache'] = cache_pin(shared)
        record['runtime_estimate'] = Q10.runtime_budget(args.compute_deadline)
        require(record['runtime_estimate']['required_conservative_seconds'] == REQUIRED, 'timing estimate changed; root must inspect the reservation')
        record['gpu_idle_first_check'] = gpu_idle()
        if args.check_only:
            record.update(status='check_only_pass', reason='all CPU/read-only gates pass; launch explicitly disabled')
            print(json.dumps(record, indent=2))
            return 0
        # One optional helper at a time. The main queue is already finished above;
        # a second process query under this lock guards against a concurrent job.
        lock = open(f'/tmp/cvm24h_optional_hilo75_{os.getuid()}.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not TARGET.exists(), 'target appeared after preflight; refuse overwrite')
        record['gpu_idle_final_check'] = gpu_idle()
        require(enough_time(args.compute_deadline), 'insufficient time after preflight for the fixed reservation')
        record['remaining_at_final_check_s'] = remaining(args.compute_deadline)
        command = [sys.executable, '-u', 'scripts/cvm24h_hilo_eval.py', '--model', 'Qwen/Qwen3-VL-8B-Instruct',
                   '--benchmark', 'textvqa', '--subset', 'eval/full_splits/textvqa_val.jsonl', '--n', '5000',
                   '--keep-frac', '0.75', '--prune-layer', '2', '--max-pixels', '4000000', '--seed', '0',
                   '--max-tokens', '32', '--attn', 'sdpa', '--out', str(TARGET)]
        soft_seconds = args.compute_deadline.timestamp() - 30 - time.time()
        require(soft_seconds > 0, 'TERM deadline passed during preflight')
        command = ['timeout', '--signal=TERM', '--kill-after=25s', f'{soft_seconds:.3f}s'] + command
        record.update(status='launching', started_at=datetime.now(TZ).isoformat(), command=command)
        save()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record.update(status='interrupted', gpu_returncode=process.returncode)
                save()
                returncode = process.returncode
        record.update(status='gpu_finished' if returncode == 0 else 'gpu_failed_or_timed_out',
                      gpu_returncode=returncode, gpu_finished_at=datetime.now(TZ).isoformat())
        save()
        # Always refresh the 5/6 or 6/6 closeout, including a timeout/nonzero exit.
        with report_log.open('x') as log:
            result = subprocess.run([sys.executable, 'scripts/cvm24h_q10_report.py', '--compute-deadline',
                                     args.compute_deadline.isoformat(), '--decision-at', record['started_at']],
                                    stdout=log, stderr=subprocess.STDOUT)
        record['closeout_returncode'] = result.returncode
        if result.returncode == 0:
            q10 = read(ROOT / 'reports/cvm2027_q10_extension_20261026.json')
            record['completed_extension_cells'] = q10['completed_extension_cells']
            record['full_curve_complete'] = q10['full_curve_complete']
        succeeded = returncode == result.returncode == 0 and record.get('full_curve_complete', False)
        if succeeded:
            record['status'] = 'completed'
        elif returncode == 0:
            record['status'] = 'closeout_failed_or_incomplete'
        record['finished_at'] = datetime.now(TZ).isoformat()
        save()
        print(json.dumps(record, indent=2))
        return 0 if succeeded else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='skipped' if record['status'] == 'preflight' else 'failed',
                      reason=f'{type(exc).__name__}: {exc}', stopped_at=datetime.now(TZ).isoformat())
        save()
        print(json.dumps(record, indent=2))
        return 0 if record['status'] == 'skipped' else 1


if __name__ == '__main__':
    raise SystemExit(main())
PY

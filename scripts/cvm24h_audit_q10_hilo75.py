#!/usr/bin/env python
"""CPU-only independent closeout audit for the original-protocol Hi-Lo75 cell.
Requires complete predictions and the canonical CPU closeout already present.
Reads legacy TV1000/None references; never uses the new matched25 predictions.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import numpy as np

from cvm24h_audit_matched import pair_bootstrap, near

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results/cvm2027_supplement_20261026'
CACHE = ROOT / 'tmp/cvm_cpu_rescore_20261025'
TARGET = D / 'predictions/q10_hilo_textvqa5k_keep0.75.json'
SUBSET = D / 'predictions/c25_hilo_textvqa1k_keep0.75.json'
NONE = D / 'predictions/p1_none_textvqa_full.json'
PIN = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def official_function():
    normalizer = CACHE / 'vqa_eval_metric_reference.py'
    task = CACHE / 'textvqa_utils_reference.py'
    assert sha(normalizer) == '4e4154c6c7e4657e6c44aa08e184ce5ae0e872665744c6cb684cf645e4647ce0'
    assert sha(task) == '4d1e10d2a72136e2fb11486827f29d955f4782b51e0cafe7228b1d53da14d405'
    spec = importlib.util.spec_from_file_location('q10_upstream_normalizer', normalizer)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    nodes = [n for n in ast.parse(task.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'textvqa_process_results']
    assert len(nodes) == 1
    ns = {'EvalAIAnswerProcessor': module.EvalAIAnswerProcessor, 'statistics': statistics}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(task), 'exec'), ns)
    return ns['textvqa_process_results']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/cvm2027_q10_hilo75_audit_20261026.json')
    args = parser.parse_args()
    full, subset, none = read(TARGET), read(SUBSET), read(NONE)
    rows = {str(r['id']): r for r in full['per_sample']}
    subrows = {str(r['id']): r for r in subset['per_sample']}
    nonerows = {str(r['id']): r for r in none['per_sample']}
    reference_path = ROOT / 'eval/full_splits/textvqa_val.jsonl'
    reference = [json.loads(line) for line in reference_path.read_text().splitlines() if line.strip()]
    ids = [str(r['id']) for r in reference]
    assert full['n'] == full['n_scored'] == len(full['per_sample']) == len(rows) == 5000
    assert full['n_error'] == full['n_skipped'] == 0
    assert [str(r['id']) for r in full['per_sample']] == ids
    assert [str(r['id']) for r in none['per_sample']] == ids and len(nonerows) == 5000
    parameters = ['model','mode','engine','benchmark','seed','max_pixels','max_tokens','attn','keep_ratio',
                  'candidate_ratio','candidate_rule','prune_layer','use_fusion','fusion_params','patches']
    for key in parameters:
        assert full[key] == subset[key], (key, 'old fixed subset configuration differs')
    expected = {'model':'Qwen/Qwen3-VL-8B-Instruct','benchmark':'textvqa','seed':0,'max_pixels':4000000,
                'max_tokens':32,'attn':'sdpa','keep_ratio':.75,'candidate_ratio':.75,'prune_layer':2,
                'use_fusion':True,'fusion_params':{'strength':.3,'top_k_ratio':.1,'attention_threshold':.05,'layers':[0,1]}}
    for key, value in expected.items():assert full[key] == value, key
    assert full['patches']['hilo_commit'] == '29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6'
    for row, source in zip(full['per_sample'], reference):
        assert row['question'] == source['question'] and row['gt'] == source['gt']
        assert not row.get('error') and not row.get('skipped') and row.get('answer') is not None
        assert row['n_image_full'] > 0 and row['n_image_kept'] == max(1, round(.75 * row['n_image_full']))
    id_path = D / 'plan/p2_textvqa_1000_ids.txt'
    selected = [line.strip() for line in id_path.read_text().splitlines() if line.strip()]
    subset_path = D / 'plan/p2_textvqa_1000.jsonl'
    subset_meta = read(D / 'plan/p2_subsets_meta.json')
    assert len(selected) == len(set(selected)) == len(subrows) == subset['n_scored'] == 1000
    assert set(selected) == set(subrows) <= set(rows)
    assert [str(r['id']) for r in subset['per_sample']] == [str(json.loads(line)['id']) for line in subset_path.read_text().splitlines() if line.strip()]
    assert hashlib.sha256('\n'.join(selected).encode()).hexdigest()[:16] == subset_meta['textvqa_1000']['sha256']
    differences = []
    for sid, old in subrows.items():
        new = rows[sid]
        assert all(new[key] == old[key] for key in ['question','gt','n_image_full','n_image_kept'])
        assert not old.get('error') and not old.get('skipped')
        if new['answer'] != old['answer']:
            differences.append({'id':sid,'old_answer':old['answer'],'new_answer':new['answer']})
    common = sorted(sid for sid in rows if not nonerows[sid].get('skipped') and not nonerows[sid].get('error'))
    for sid in common:assert rows[sid]['n_image_full'] == nonerows[sid]['n_image_full']
    launches = []
    for path in (D / 'plan').glob('optional_hilo75_launch_*.json'):
        launch = read(path)
        if launch.get('target') == str(TARGET) and launch.get('status') == 'completed':launches.append((path, launch))
    assert len(launches) == 1, 'Root closeout must finish once before the independent audit'
    launch_path, launch = launches[0]
    assert launch['gpu_returncode'] == launch['closeout_returncode'] == 0 and launch['full_curve_complete']
    assert launch['completed_extension_cells'] == 6 and not launch['quality_scores_used_for_launch_decision']
    assert launch['required_conservative_seconds'] == 3445
    for name, expected_sha in launch['source_fingerprints']['sha256'].items():
        assert sha(ROOT / name) == expected_sha, ('launch source changed', name)
    model = launch['model_cache']; assert model['revision'] == PIN
    cache = Path(model['cache']); snapshot = cache / 'snapshots' / PIN
    assert (cache / 'refs/main').read_text().strip() == PIN
    for name, expected_sha in model['processor_config_sha256'].items():
        path = snapshot / name
        assert (sha(path) if path.exists() else None) == expected_sha
    assert sha(snapshot / 'model.safetensors.index.json') == model['weight_index_sha256']
    assert sha(snapshot / 'generation_config.json') == model['generation_config_sha256']
    for shard in model['weight_shards']:
        path = snapshot / shard['file']
        assert path.stat().st_size == shard['bytes'] and path.resolve().name == shard['cache_blob_sha256']
    log_path = Path(launch['completion_log'])
    assert 'n=5000 ok=5000 err=0' in log_path.read_text().splitlines()[-1]
    ann_path = CACHE / 'TextVQA_0.5.1_val.json'
    assert sha(ann_path) == '4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d'
    official = {str(r['question_id']):r['answers'] for r in read(ann_path)['data']}
    score = official_function()

    def scores(rr):
        output = {}
        for sid, row in rr.items():
            assert row['gt'] == ';'.join(answer.strip() for answer in official[sid])
            output[sid] = (0. if row.get('skipped') or row.get('error') else
                           float(score({'answers':list(official[sid]),'question_id':sid},[row.get('answer','')])['exact_match']))
        return output

    full_scores, none_scores = scores(rows), scores(nonerows)
    score_path = D / 'plan/scores/full_q10_hl_0.75.json'
    canonical = read(score_path)
    assert canonical['sha_a'] == sha(TARGET) and canonical['sha_b'] == sha(NONE) and canonical['resamples'] == 20000
    assert canonical['A']['n'] == canonical['B']['n'] == 5000
    assert canonical['A']['skipped'] == 0 and canonical['B']['skipped'] == len(rows) - len(common) == 5
    near(canonical['A']['attempted_score'], statistics.mean(full_scores.values()))
    near(canonical['B']['attempted_score'], statistics.mean(none_scores.values()))
    pair_results = {}
    for key, selected_ids, seed in [('paired_attempted_all',sorted(rows),20261285),
                                    ('paired_common_answered_sensitivity',common,20261286)]:
        values = np.array([full_scores[sid] - none_scores[sid] for sid in selected_ids])
        delta, ci, p = pair_bootstrap(values, seed)
        published = canonical[key]
        assert published['seed'] == seed and published['resamples'] == 20000 and published['n_paired'] == len(selected_ids)
        near(delta * 100, published['delta_pp'])
        for x, y in zip(ci, published['ci95_pp']):near(x * 100, y)
        near(p, published['sign_flip_p_two_sided'])
        near(statistics.mean(full_scores[sid] for sid in selected_ids), published['mean_a'])
        near(statistics.mean(none_scores[sid] for sid in selected_ids), published['mean_b'])
        pair_results[key] = {'n':len(selected_ids),'seed':seed,'resamples':20000,'delta_pp':delta * 100,
                             'ci95_pp':[v * 100 for v in ci],'sign_flip_p_two_sided':p,'matches_official':True}
    report_path = ROOT / 'reports/cvm2027_q10_extension_20261026.json'
    report = read(report_path)
    assert report['full_curve_complete'] and report['completed_extension_cells'] == 6
    cell = report['cells']['hl_0.75']
    assert cell['prediction_sha256'] == sha(TARGET)
    near(cell['official_score_percent'], statistics.mean(full_scores.values()) * 100)
    result = {'status':'PASS' if not differences else 'PASS_WITH_SUBSET_ANSWER_DIFFERENCES',
              'audited_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'GPU_work':False,'canonical_results_changed':False,
              'scope':'Original mixed-attention optional Q10 extension; not the new matched main',
              'attempted':5000,'answered':5000,'failures':0,'exact_full_ID_order':'PASS','actual_K_and_original_N':'PASS',
              'mean_original_N':statistics.mean(r['n_image_full'] for r in rows.values()),
              'mean_actual_K':statistics.mean(r['n_image_kept'] for r in rows.values()),
              'fixed_subset_overlap':{'n':1000,'same_original_N_and_K':True,'identical_answers':1000-len(differences),
                                      'different_answers':differences,'fixed_metadata_prefix_sha256':subset_meta['textvqa_1000']['sha256']},
              'official_score_percent':statistics.mean(full_scores.values())*100,
              'old_None_score_percent':statistics.mean(none_scores.values())*100,
              'old_None_failure_ids':sorted(set(rows)-set(common)), 'pairs':pair_results,
              'launch_source_and_current_model_cache_fingerprints':'PASS','launch_completed_before_hard_stop':datetime.fromisoformat(launch['gpu_finished_at'])<datetime.fromisoformat(launch['hard_gpu_stop']),
              'protocol_configuration':{key:full[key] for key in parameters},
              'source_sha256':{str(path):sha(path) for path in [TARGET,SUBSET,NONE,id_path,subset_path,D/'plan/p2_subsets_meta.json',
                                                              launch_path,log_path,score_path,report_path,ann_path]},
              'audit_script':str(Path(__file__).resolve()),'audit_sha256':sha(__file__),
              'bootstrap_audit_helper_sha256':sha(ROOT/'scripts/cvm24h_audit_matched.py'),
              'reproduce_command':'/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/cvm24h_audit_q10_hilo75.py',
              'limitations':'Legacy TV1000 records configuration/commit/version, not original weight-file SHA256s. The new full run has a launch-side cache/source pin; overlap answers provide an additional reproducibility check. These mixed-attention results do not extend the shared-attention main or its cost measurements.'}
    assert result['launch_completed_before_hard_stop']
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    p=pair_results['paired_attempted_all'];c=pair_results['paired_common_answered_sensitivity']
    md=['# Independent optional Hi-Lo75 closeout audit (2026-10-26)','',
        f'**{result["status"]}.** Full 5,000 IDs, all answers and exact 75% visual budgets verified; no failed attempts. Launch-source and current model-cache fingerprints match. GPU inference finished before the 19:00 hard stop.','',
        f'The original pre-fixed TV1000 Hi-Lo75 prediction has {1000-len(differences)}/1000 byte-identical answer strings on the same original N and retained K. Differences, if any, are preserved in the JSON. The fixed 1k comparison is retained independently.','',
        f'Pinned upstream TextVQA task execution gives **{result["official_score_percent"]:.3f}%**, against the old None reference at {result["old_None_score_percent"]:.3f}% (five historical failures scored zero). Independent 20,000-resample primary delta: {p["delta_pp"]:+.3f} pp [{p["ci95_pp"][0]:+.3f}, {p["ci95_pp"][1]:+.3f}], seed {p["seed"]}. Common-answered sensitivity: {c["delta_pp"]:+.3f} pp [{c["ci95_pp"][0]:+.3f}, {c["ci95_pp"][1]:+.3f}], n={c["n"]}, seed {c["seed"]}. Both CIs, means and sign-flip p-values match canonical results within 1e-12 tolerance.','',
        result['limitations'],'','Reproduce from the worktree root (CPU only):','','```bash',result['reproduce_command'],'```','',
        'Original predictions, canonical score reports, figures and manuscript are read only.']
    args.out.with_suffix('.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({key:result[key] for key in ['status','official_score_percent','old_None_score_percent','fixed_subset_overlap','pairs']},indent=2))


if __name__ == '__main__':main()

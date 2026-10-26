#!/usr/bin/env python
"""Independent CPU audit; never imports a model, runner, scorer or builder."""
import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results/cvm2027_supplement_20261026'
CACHE = ROOT / 'tmp/cvm_cpu_rescore_20261025'
MODES = ['none', 'pre', 'fastv', 'hilo']
BENCHES = [('textvqa', 5000, 100, 'textvqa_val.jsonl'), ('ocrbench', 1000, 1000, 'ocrbench.jsonl')]
PAIRS = [('pre', 'fastv'), ('pre', 'hilo'), ('pre', 'none'), ('fastv', 'hilo'), ('fastv', 'none'), ('hilo', 'none')]
PINS = {
    'vqa_eval_metric_reference.py': '4e4154c6c7e4657e6c44aa08e184ce5ae0e872665744c6cb684cf645e4647ce0',
    'textvqa_utils_reference.py': '4d1e10d2a72136e2fb11486827f29d955f4782b51e0cafe7228b1d53da14d405',
    'TextVQA_0.5.1_val.json': '4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d',
}
SOURCE_FILES = {
    'runner_sha256': ROOT / 'scripts/cvm24h_matched_quality.py',
    'helper_sha256': ROOT / 'src/v3_premerger/baselines_hf.py',
    'v2_sha256': ROOT / 'scripts/cvm24h_timing_v2.py',
    'hilo_model_sha256': Path('/tmp/Hi-Lo_Prune/hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py'),
    'hilo_selection_sha256': Path('/tmp/Hi-Lo_Prune/hilo_prune/selection.py'),
    'hilo_patches_sha256': Path('/tmp/Hi-Lo_Prune/hilo_prune/patches.py'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def metadata():
    for name, pin in PINS.items():
        assert sha(CACHE / name) == pin, name
    official = {str(row['question_id']): row for row in read(CACHE / 'TextVQA_0.5.1_val.json')['data']}
    cells, hashes, summaries, missing = {}, {}, {}, []
    protocol = sources = None
    p3_checks = 0
    for bench, n, scale, splitfile in BENCHES:
        splitpath = ROOT / 'eval/full_splits' / splitfile
        ref = [json.loads(line) for line in splitpath.read_text().splitlines() if line.strip()]
        ids = [str(row['id']) for row in ref]
        assert len(ids) == len(set(ids)) == n
        for mode in MODES:
            tag = f'{mode}_{bench}'
            path = D / 'matched_main' / f'{tag}.json'
            cell = read(path)
            if not cell['complete']:
                missing.append({'tag': tag, 'attempted': cell['n_attempted'], 'pending': cell['n_pending']})
                continue
            assert cell['n'] == cell['expected_n'] == cell['n_attempted'] == cell['n_scored'] == n
            assert cell['n_error'] == cell['n_pending'] == 0 and not cell.get('protocol_failure')
            assert cell['mode'] == mode and cell['benchmark'] == bench
            rows = cell['per_sample']
            assert [str(row['id']) for row in rows] == ids
            assert cell['subset_sha256'] == sha(splitpath)
            assert cell['ids_sha256'] == hashlib.sha256('\n'.join(ids).encode()).hexdigest()
            assert cell['keep_fraction'] == (1. if mode == 'none' else .25)
            if protocol is None:
                protocol, sources = cell['protocol'], cell['source_hashes']
            assert cell['protocol'] == protocol and cell['source_hashes'] == sources
            assert protocol['attention'] == {'vision_config': 'sdpa', 'text_config': 'eager'}
            assert protocol['weight_revision'] == '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
            for key, value in {'model':'Qwen/Qwen3-VL-8B-Instruct','dtype':'bfloat16','max_pixels':4000000,
                               'max_new_tokens':32,'seed':0,'mrope':'native','do_sample':False,'bs':1}.items():
                assert protocol[key] == value
            for key, source in SOURCE_FILES.items():
                assert sources[key] == sha(source), (tag, key)
            cfg = cell['configuration']
            assert cfg['fastv_k'] == 3 and cfg['hilo_prune_layer'] == 2
            if mode == 'hilo':
                for key, expected in {'hilo_keep_ratio':.25,'hilo_candidate_ratio':.3,'hilo_use_fusion':True,
                                      'hilo_fusion_strength':.3,'hilo_top_k_ratio':.1,'hilo_attention_threshold':.05,
                                      'hilo_fusion_layers':[0,1]}.items():
                    assert cfg[key] == expected
            for row, source in zip(rows, ref):
                sid = str(row['id'])
                assert row['attempted'] and row['status'] == 'ok' and not row['error'] and not row['skipped']
                assert row['question'] == source['question'] and row['gt'] == source['gt']
                assert row['dataset'] == source.get('dataset') and row['question_type'] == source.get('question_type')
                assert str(row['question_id']) == sid
                if bench == 'textvqa':
                    assert row['answers'] == official[sid]['answers'] and len(row['answers']) == 10
                    assert row['image_id'] == official[sid]['image_id']
                    assert row['image_id_source'] == 'official TextVQA_0.5.1_val.json question_id lookup'
                else:
                    assert row['image_id'] == source.get('image_id', sid)
                    assert row['image_id_source'] == ('preserved OCRBench row image_id' if source.get('image_id') is not None else 'preserved OCRBench row index alias')
                original, kept = row['n_image_full'], row['n_image_kept']
                assert isinstance(original, int) and original > 0
                assert kept == (original if mode == 'none' else max(1, round(original * .25)))
                assert row['actual_k_verified'] and row['actual_k_expected'] == kept
                grid = row['image_grid_thw']
                assert len(grid) == 1 and len(grid[0]) == 3 and math.prod(grid[0]) == 4 * original
                assert all(isinstance(v, int) and v > 0 for v in grid[0])
                assert row['prompt_token_ids'] == row['n_prompt_full'] - original + kept
                assert len(row['input_ids_sha256']) == 64
                assert row['out_len'] == len(row['generated_token_ids']) <= 32
                if mode == 'none':
                    assert row['kept_all'] and row['kept_image_ids'] in (None, list(range(original)))
                else:
                    chosen = row['kept_image_ids']
                    assert not row['kept_all'] and len(chosen) == kept and chosen == sorted(set(chosen))
                    assert 0 <= min(chosen) <= max(chosen) < original
            blind = cell['blind_p3_check']
            p3path = D / 'plan' / ('timing_hilo_corrected.json' if mode == 'hilo' else 'timing_harness_corrected.json')
            assert Path(blind['source']).resolve() == p3path.resolve() and blind['sha256'] == sha(p3path)
            p3 = read(p3path)
            assert p3['script_sha256'] == sources['v2_sha256']
            block = p3['results']['none' if mode == 'none' else f'{mode}_0.25']
            references = {str(r['id']):r['repeats_raw'][0] for r in block['requests'] if r['benchmark'] == bench}
            assert len(references) == blind['expected_formal_overlap'] == len(blind['formal_overlap']) == 100
            assert {str(r['id']) for r in blind['formal_overlap']} == set(references)
            assert len(blind['warmup']) == 3
            for check in blind['warmup'] + blind['formal_overlap']:
                assert all(v for k, v in check.items() if k != 'id')
            for row in rows:
                sid = str(row['id'])
                if sid in references:
                    reference = references[sid]
                    for key in ['answer','generated_token_ids','n_image_full','n_image_kept']:
                        assert row[key] == reference[key], (tag, sid, key, 'independent saved P3 parity')
                    p3_checks += 1
            cells[tag], hashes[tag] = cell, sha(path)
            summaries[tag] = {'attempted':n,'failed':0,'prediction_sha256':hashes[tag],
                              'mean_original_N':statistics.mean(r['n_image_full'] for r in rows),
                              'mean_kept_K':statistics.mean(r['n_image_kept'] for r in rows),
                              'P3_saved_first_repeat_checks':100,'warmup_evidence_checks':3}
    common_checks = {}
    for bench, n, _, _ in BENCHES:
        available = [cells[f'{m}_{bench}'] for m in MODES if f'{m}_{bench}' in cells]
        for rows in zip(*(c['per_sample'] for c in available)):
            for key in ['id','image','n_image_full','input_ids_sha256','n_prompt_full','image_grid_thw']:
                assert all(r[key] == rows[0][key] for r in rows)
        common_checks[bench] = {'available_methods':len(available),'paired_metadata_rows':n,'mismatches':0}
    # Check current local processor/index bytes and immutable shard identities;
    # no model imports or multi-GB weight reads are needed.
    from huggingface_hub.constants import HF_HUB_CACHE
    model_cache = Path(HF_HUB_CACHE) / 'models--Qwen--Qwen3-VL-8B-Instruct'
    revision = protocol['weight_revision']
    assert (model_cache / 'refs/main').read_text().strip() == revision
    snapshot = model_cache / 'snapshots' / revision
    for filename, expected in sources['processor_config_sha256'].items():
        path = snapshot / filename
        assert (sha(path) if path.exists() else None) == expected
    assert sha(snapshot / 'model.safetensors.index.json') == sources['weight_index_sha256']
    for shard in sources['weight_shards']:
        path = snapshot / shard['file']
        assert path.stat().st_size == shard['bytes'] and path.resolve().name == shard['cache_blob_sha256']
    return {'complete_cells':len(cells),'pending_cells':missing,'cells':summaries,'shared_protocol':protocol,
            'source_hashes':sources,'paired_input_checks':common_checks,'independent_P3_parity_rows':p3_checks,
            'current_processor_index_and_weight_cache_pin':'PASS',
            'no_scores_computed':True}, cells, official


def upstream_scores(cells, official):
    spec = importlib.util.spec_from_file_location('audit_upstream_vqa', CACHE / 'vqa_eval_metric_reference.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tvpath = CACHE / 'textvqa_utils_reference.py'
    ns = {'EvalAIAnswerProcessor':module.EvalAIAnswerProcessor,'statistics':statistics}
    nodes = [n for n in ast.parse(tvpath.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='textvqa_process_results']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(tvpath),'exec'),ns)
    ocrpath = CACHE / 'ocrbench_utils_reference.py'
    # The pinned hash is declared in the repository's original OCR audit; parse
    # its literal without importing any local scoring wrapper.
    original = ROOT / 'scripts/analyze_cvm_ocrbench_official.py'
    pin = None
    for n in ast.parse(original.read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SHA256' for t in n.targets):
            pin = ast.literal_eval(n.value)
    assert pin and sha(ocrpath)==pin
    ocrns = {}
    nodes = [n for n in ast.parse(ocrpath.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_fold_fullwidth_ascii','ocrbench_process_results')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ocrpath),'exec'),ocrns)
    scored = {}
    for bench, _, _, _ in BENCHES:
        for mode in MODES:
            tag=f'{mode}_{bench}'; out={}
            for row in cells[tag]['per_sample']:
                sid=str(row['id'])
                if row.get('skipped') or row.get('error'):
                    value=0.
                elif bench=='textvqa':
                    doc={'answers':list(official[sid]['answers']),'question_id':sid}
                    value=float(ns['textvqa_process_results'](doc,[row['answer']])['exact_match'])
                else:
                    doc={'answer':str(row['gt']).split(';'),'dataset':row['dataset'],'question_type':row['question_type']}
                    value=float(ocrns['ocrbench_process_results'](doc,[row['answer']])['ocrbench_accuracy']['score'])
                out[sid]=value
            scored[tag]=out
    return scored, {'TextVQA_normalizer_sha256':PINS['vqa_eval_metric_reference.py'],
                    'TextVQA_task_sha256':PINS['textvqa_utils_reference.py'],'OCRBench_task_sha256':pin,
                    'method':'Direct pinned upstream task functions, independently AST-extracted; local scoring wrappers not imported'}


def pair_bootstrap(values, seed):
    # Independently reproduce the published RNG ordering: 128 bootstrap rows,
    # then 128 sign-flip rows for each batch, using the explicit fixed seed.
    rng=np.random.default_rng(seed); samples=[]; flips=[]
    for start in range(0,20000,128):
        count=min(128,20000-start)
        draw=rng.integers(0,len(values),(count,len(values)))
        samples.extend(np.mean(values[draw],axis=1))
        signs=rng.integers(0,2,(count,len(values)),dtype=np.int8)*2-1
        flips.extend(np.mean(signs*values,axis=1))
    observed=float(np.mean(values));ci=np.quantile(samples,[.025,.975])
    p=(1+sum(abs(x)>=abs(observed)-1e-12 for x in flips))/20001
    return observed, [float(x) for x in ci], p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'reports/cvm2027_matched_main_audit_20261026.json')
    args=parser.parse_args()
    meta,cells,official=metadata()
    result={'audit':'independent CPU matched-main audit','audited_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'metadata':meta,'audit_script':str(Path(__file__).resolve()),'audit_code_sha256':sha(__file__),
            'reproduce_command':'/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/cvm24h_audit_matched.py',
            'GPU_work':False,'source_results_changed':False}
    if not args.metadata_only:
        assert meta['complete_cells']==8 and not meta['pending_cells'], 'Audit waits for eight complete cells'
        source=ROOT/'reports/cvm2027_matched_main_20261026.json'
        published=read(source)
        assert published['complete'] and published['cell_count']==8
        result['official_report_sha256']=sha(source)
        scored, metric_pins=upstream_scores(cells,official)
        result['metric_verification']=metric_pins
        result['cell_scores']={};result['pairs']={}
        for bi,(bench,n,scale,_) in enumerate(BENCHES):
            for mode in MODES:
                tag=f'{mode}_{bench}';summary=published['benchmarks'][bench]['cells'][mode]
                assert published['sources'][bench][mode]['sha256']==meta['cells'][tag]['prediction_sha256']
                value=statistics.mean(scored[tag].values())*scale
                for key in ['attempted_score','answered_score','common_all_methods_score']:
                    near(value,summary[key])
                assert summary['n_attempted']==summary['n_answered']==n and summary['n_failed']==0 and not summary['failure_ids']
                result['cell_scores'][tag]={'independent_upstream_score':value,'official_difference':value-summary['attempted_score'],'n_attempted':n,'failures':0}
            for pi,(a,b) in enumerate(PAIRS):
                pairkey=f'{a}_minus_{b}';publishedpair=published['benchmarks'][bench]['pairs'][pairkey]
                ids=sorted(scored[f'{a}_{bench}'])
                values=np.array([scored[f'{a}_{bench}'][sid]-scored[f'{b}_{bench}'][sid] for sid in ids])
                for common, key in [(False,'attempted'),(True,'common_answered')]:
                    seed=20261500+bi*100+pi*2+int(common)
                    observed,ci,p=pair_bootstrap(values,seed)
                    actual=publishedpair[key]
                    assert actual['seed']==seed and actual['resamples']==20000 and actual['n_paired']==actual['n_paired_answered']==n
                    near(actual['delta'],observed*scale)
                    for x,y in zip(actual['ci95'],ci):near(x,y*scale)
                    near(actual['sign_flip_p_two_sided'],p)
                    near(actual['mean_a'],statistics.mean(scored[f'{a}_{bench}'].values())*scale)
                    near(actual['mean_b'],statistics.mean(scored[f'{b}_{bench}'].values())*scale)
                    result['pairs'][f'{bench}_{pairkey}_{key}']={'delta':observed*scale,'ci95':[x*scale for x in ci],
                                                              'p':p,'seed':seed,'resamples':20000,'n':n,'matches_official':True}
        meta['no_scores_computed']=False
        result.update(status='PASS',complete=True,planned_pairs_verified=12,primary_plus_common_intervals_verified=24,
                      limitation='Question-paired 95% CIs do not prove equivalence or isolate algorithm differences beyond the declared implementation controls.')
    else:
        result.update(status='METADATA_PASS',complete=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    if not args.metadata_only:
        md=['# Independent matched-main CPU audit (2026-10-26)','',
            '**PASS.** All eight full cells and all twelve planned pairs independently verified; each primary and common-answered interval uses 20,000 resamples with its prescribed seed (24 intervals total).','',
            'No GPU work, source prediction changes, manuscript edits, or figure rebuilds. The audit directly executes the pinned upstream TextVQA and OCRBench task functions; it does not call the main scorer or its metric wrappers. Bootstrap and sign-flip calculations are independently implemented.','',
            '| Cell | Independent official score | Attempts | Failures |','|---|---:|---:|---:|']
        for tag,cell in result['cell_scores'].items():md.append(f'| {tag} | {cell["independent_upstream_score"]:.4f} | {cell["n_attempted"]} | {cell["failures"]} |')
        md+=['','TextVQA units are percentage points; OCRBench scores use points per 1000. Every official cell score, pair mean/delta/95% CI/p-value, fixed seed, count and common-answered result agrees within 1e-12 relative/absolute tolerance.','',
             f'All IDs, official metadata, per-request visual N/K, kept indices, image grids, original input hashes, shared protocol and current dependency SHA256s pass. Saved first-repeat P3 parity is independently rechecked on {meta["independent_P3_parity_rows"]} overlapping requests. The three warmup comparisons per cell are verified as recorded evidence; no GPU rerun is claimed.','',
             'Source hashes and all 24 reproduced interval values are recorded in the JSON. '+result['limitation'],'',
             'Reproduce from the worktree root (CPU only):','','```bash',result['reproduce_command'],'```','',
             f'Archived audit script: `{Path(__file__).resolve()}`; SHA256 `{result["audit_code_sha256"]}`.']
        args.out.with_suffix('.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'status':result['status'],'complete_cells':meta['complete_cells'],'pending':meta['pending_cells'],
                      'P3_parity':meta['independent_P3_parity_rows'],'intervals_verified':result.get('primary_plus_common_intervals_verified',0)},indent=2))


if __name__=='__main__':main()

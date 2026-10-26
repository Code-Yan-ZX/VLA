#!/usr/bin/env python
"""Official quality and paired CIs for the SAME corrected 200-request workload.
CPU only. Saves per-benchmark score/latency summaries and a two-task figure.
Scores the first repeat's answer; repeat consistency is checked, not optimized.
"""
import hashlib
import json
import statistics
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                           'axes.labelsize': 12.5, 'axes.titlesize': 13,
                           'xtick.labelsize': 11, 'ytick.labelsize': 11})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cvm24h_score as SC
from cvm24h_validate_timing_v2 import validate as validate_timing

Q = ROOT / 'results/cvm2027_supplement_20261026/plan'
FIG = ROOT / 'reports/figures_cvm2027_supplement'
SCALE = {'textvqa': 100, 'ocrbench': 1000}
plan = json.loads((Q/'p3_timing_ids.json').read_text())
meta = {bench: {str(x['id']): x for x in map(json.loads, (ROOT/'eval/full_splits'/filename).read_text().splitlines())}
        for bench, filename in [('textvqa', 'textvqa_val.jsonl'), ('ocrbench', 'ocrbench.jsonl')]}


def quantile(v, q):
    v = sorted(v); z = (len(v)-1)*q; lo = int(z); hi = min(lo+1,len(v)-1)
    return v[lo] + (v[hi]-v[lo])*(z-lo)


def percentiles(rows, field):
    v = [r[field] for r in rows if r.get('status') == 'ok' and r.get(field) is not None]
    return {'p50': statistics.median(v), 'p95': quantile(v, .95)} if v else None


timing_paths = [Q / f'timing_{engine}_corrected.json' for engine in ('harness', 'hilo')]
if not all(path.exists() for path in timing_paths):
    raise SystemExit('Both corrected timing files are required; existing complete score artifacts preserved')
validation = validate_timing(timing_paths)
expected_tags = {'none'} | {f'{m}_{k}' for m in ('pre', 'fastv', 'hilo') for k in (.25, .5, .75)}
if not validation['complete'] or set(validation['cells']) != expected_tags:
    raise SystemExit('All ten corrected timing cells must pass validation; existing complete score artifacts preserved')

configs = {}
provenance = {}
for engine in ('harness', 'hilo'):
    path = Q/f'timing_{engine}_corrected.json'
    if not path.exists():
        continue
    raw = path.read_bytes(); data = json.loads(raw)
    assert data['schema_version'] == 2 and data['n_requests'] == 200 and data['repeats'] == 3
    provenance[engine] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'attention': data.get('attention'), 'protocol_note': data.get('protocol_note')}
    for tag, block in data['results'].items():
        assert block['n_attempted'] == 200 and block['actual_k_verified'], tag
        assert all(r.get('answers_consistent', True) for r in block['requests']), f'Nonrepeatable greedy answers {tag}'
        configs[tag] = {}
        for bench in ('textvqa', 'ocrbench'):
            timed = [r for r in block['requests'] if r['benchmark'] == bench]
            assert len(timed) == 100 and {r['id'] for r in timed} == set(map(str,plan[f'{bench}_100']))
            rows = {}
            for r in timed:
                sid = str(r['id']); reference = meta[bench][sid]
                row = {'id': sid, 'gt': reference['gt'], 'answer': r.get('answer', ''),
                       'skipped': r['status'] != 'ok', 'n_image_full': r.get('n_image_full'),
                       'n_image_kept': r.get('n_image_kept')}
                rows[sid] = row
            scores = SC.SC[bench](rows)
            configs[tag][bench] = {'rows': rows, 'scores': scores, 'keep': block['keep_fraction'],
                                   'n': len(rows), 'n_success': sum(not r['skipped'] for r in rows.values()),
                                   'official_score': sum(scores.values()) / len(rows) * SCALE[bench],
                                   'total_s': percentiles(timed,'total_s'), 'ttft_s': percentiles(timed,'ttft_s'),
                                   'peak_alloc_GB': max((rep['peak_alloc_GB'] for r in timed if r.get('status')=='ok' for rep in r['repeats_raw']), default=None),
                                   'peak_reserved_GB': max((rep['peak_reserved_GB'] for r in timed if r.get('status')=='ok' for rep in r['repeats_raw']), default=None),
                                   'memory_unit': 'GiB',
                                   'segments_s': {field: percentiles(timed, field) for field in
                                                  ('processor_cpu_s', 'transfer_s', 'vision_s', 'selection_s', 'fusion_s', 'prefill_s', 'decode_s')}}

if 'none' not in configs:
    raise SystemExit('Corrected None timing not yet complete; no quality-cost result emitted')

result = {'scope': 'Fixed TextVQA-100 and OCRBench-100 requests; subset, same protocol as corrected timing',
          'quality_rule': 'First repeat answer, all attempts denominator, failures zero; official pinned scoring; no validation-score tuning',
          'provenance': provenance, 'validation': validation, 'configs': {}, 'quality95': {},
          'note': 'Subset results are not full-split scores. Repeats are timing replicates, not independent quality samples.'}
md = ['# Quality and cost on the corrected fixed 200-request workload', '', result['scope']+'.', '',
      'Shared vision SDPA / text eager, bf16, native positions, greedy <=32 new tokens. Quality uses the first repeated answer; all three repetitions must agree. Latencies include processor and transfer from an already decoded RGB image; disk/decode/loading/warmup excluded.', '',
      '| Config | Task subset | Official score | Δ vs None [95% CI] | p50 / p95 total (s) | p50 / p95 TTFT (s) | Answered / attempted |',
      '|---|---|---:|---:|---:|---:|---:|']
for idx, (tag, benches) in enumerate(configs.items()):
    result['configs'][tag] = {}
    for bi, (bench, cell) in enumerate(benches.items()):
        reference = configs['none'][bench]
        paired = SC.EX.paired(cell['scores'],cell['rows'],reference['scores'],reference['rows'],20000,20261300+idx*2+bi,include_skips=True)
        out = {k:v for k,v in cell.items() if k not in ('rows','scores')}
        out['paired_vs_none'] = paired
        result['configs'][tag][bench] = out
        mult = SCALE[bench]/100
        ci = [x*mult for x in paired['ci95_pp']]
        total, ttft = cell['total_s'],cell['ttft_s']
        md.append(f'| {tag} | {bench} n=100 | {cell["official_score"]:.2f} | {paired["delta_pp"]*mult:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {total["p50"]:.3f} / {total["p95"]:.3f} | {ttft["p50"]:.3f} / {ttft["p95"]:.3f} | {cell["n_success"]} / 100 |')
md += ['', '## Cheapest measured point meeting 95% of same-protocol None', '']
for bench in ('textvqa','ocrbench'):
    ref = result['configs']['none'][bench]['official_score']; target = .95*ref
    result['quality95'][bench] = {'none':ref,'target':target,'methods':{}}
    for method in ('pre','fastv','hilo'):
        points = [(tag,values[bench]) for tag,values in result['configs'].items() if tag.startswith(method+'_')]
        passing = [(tag,c) for tag,c in points if c['official_score'] >= target]
        winner = min(passing,key=lambda pair:pair[1]['total_s']['p50']) if passing else None
        result['quality95'][bench]['methods'][method] = {'measured_tags':[tag for tag,_ in points],
                    'cheapest_p50_tag':winner[0] if winner else None,
                    'cheapest_p50_s':winner[1]['total_s']['p50'] if winner else None}
        md.append(f'- {bench} {method}: '+(f'{winner[0]}, p50={winner[1]["total_s"]["p50"]:.3f}s' if winner else 'not met at completed measured points')+f' (target={target:.2f}).')

md += ['', '## Request segments (p50 seconds) and raw peak memory', '',
       'Each task uses its fixed 100 requests; three repeats are reduced to one median per request before the p50. Selection/fusion can overlap prefill and must not be added again. Total and TTFT come from direct same-request timestamps. Memory uses the maximum over all raw repeats, in GiB.', '',
       '| Config | Task | CPU processor | Transfer | Full vision | Selection | Fusion | Prefill | TTFT | Decode | Peak allocated / reserved (GiB) |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for tag, benches in result['configs'].items():
    for bench, cell in benches.items():
        segments = cell['segments_s']
        def seg(field):
            value = segments.get(field)
            return f'{value["p50"]:.4f}' if value else '—'
        fields = ['processor_cpu_s', 'transfer_s', 'vision_s', 'selection_s', 'fusion_s', 'prefill_s']
        md.append(f'| {tag} | {bench} | ' + ' | '.join(seg(field) for field in fields)
                  + f' | {cell["ttft_s"]["p50"]:.4f} | {seg("decode_s")} | {cell["peak_alloc_GB"]:.2f} / {cell["peak_reserved_GB"]:.2f} |')

summary_path = ROOT/'reports/cvm2027_corrected_quality_cost.json'
summary_path.write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'reports/cvm2027_corrected_quality_cost.md').write_text('\n'.join(md)+'\n')
from cvm24h_plot_quality_cost import plot_quality_cost
plot_quality_cost(result, FIG)
print('Corrected quality-cost summaries written:',len(configs),'configs')

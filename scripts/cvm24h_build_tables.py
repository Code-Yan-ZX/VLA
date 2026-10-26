#!/usr/bin/env python
"""Build reviewable tables and plots from pinned official scores.
Keep subset curves even after the optional full TextVQA grid completes.
Legacy timing is excluded: its gradient/deepstack/config bugs were found in audit.
"""
import hashlib
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                           'axes.labelsize': 12.5, 'axes.titlesize': 13,
                           'xtick.labelsize': 11, 'ytick.labelsize': 11})
from cvm24h_validate_timing_v2 import validate as validate_timing

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / 'results/cvm2027_supplement_20261026/plan'
S = Q / 'scores'
REPORT = ROOT / 'reports'
FIG = REPORT / 'figures_cvm2027_supplement'
FIG.mkdir(parents=True, exist_ok=True)
KEEP = ('0.25', '0.50', '0.75')
METHODS = {'rb': 'RBM', 'fv': 'FastV-k3', 'hl': 'Hi-Lo'}
COLORS = {'rb': '#1f77b4', 'fv': '#d62728', 'hl': '#2ca02c'}
SCALE = {'textvqa': 100, 'ocrbench': 1000, 'docvqa': 100}

def load(name):
    p = S / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None

def curve(bench, method, keep, full=False):
    if full:
        return load(f'p1_tv_{method}_none' if keep == '0.25' else f'full_q10_{method}_{keep}')
    return load(f'curve_{"tv" if bench == "textvqa" else "oc"}_{method}_{keep}')

def vals(d, common=False):
    if common:
        p = d['paired_common_answered_sensitivity']
        return p['mean_a'], p['mean_b'], p['n_paired']
    return d['A']['attempted_score'], d['B']['attempted_score'], d['A']['n']

def pformat(v):
    return '<0.0001' if v < 0.0001 else f'{v:.4f}'

def save_figure(fig, stem):
    for suffix in ('svg', 'pdf'):
        path = FIG / f'{stem}.{suffix}'
        fig.savefig(path)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')

def summary(d, common=False):
    key = 'paired_common_answered_sensitivity' if common else 'paired_attempted_all'
    p = d[key]; scale = SCALE[d['bench']] / 100
    return {'a': vals(d, common)[0] * SCALE[d['bench']],
            'b': vals(d, common)[1] * SCALE[d['bench']], 'n': vals(d, common)[2],
            'delta': p['delta_pp'] * scale,
            'ci95': [x * scale for x in p['ci95_pp']],
            'p': p['sign_flip_p_two_sided']}

md = ['# CVM2027 supplement — audited tables (2026-10-26)', '',
      'Scores use all attempted IDs, with failures scored zero. Hi-Lo uses the official vendored adaptation with SDPA; RBM/FastV use the eager harness. Common-answered analyses control the failure set, but do not remove implementation differences.', '',
      '## Main comparison at 25% retention', '',
      '| Benchmark | None@100% | RBM | FastV-k3 | Hi-Lo | Attempted / answered |',
      '|---|---:|---:|---:|---:|---|']
summary_json = {'protocol_caveat': md[2], 'p1': {}, 'curves': {}, 'quality95': [], 'timing': {}}
for bench, prefix in (('textvqa', 'p1_tv'), ('ocrbench', 'p1_oc')):
    pairs = {m: load(f'{prefix}_{m}_none') for m in METHODS}
    assert all(pairs.values()), f'Missing P1 pair: {prefix}'
    ref = pairs['rb']['B']; scale = SCALE[bench]
    cell = [f'{ref["attempted_score"] * scale:.2f}']
    cell += [f'{pairs[m]["A"]["attempted_score"] * scale:.2f}' for m in METHODS]
    counts = ' / '.join(f'{pairs[m]["A"]["n"] - pairs[m]["A"]["skipped"]}' for m in METHODS)
    md.append(f'| {bench} | ' + ' | '.join(cell) + f' | {ref["n"]}; {counts} |')
    summary_json['p1'][bench] = {m: summary(d) for m, d in pairs.items()}
md += ['', 'Answered counts are shown in RBM/FastV/Hi-Lo order; None answers 4,995 TextVQA and 921 OCRBench requests.', '',
       '| Pair A − B | Task | Attempted Δ [95% CI] | Common-answered Δ [95% CI] | Common n |',
       '|---|---|---:|---:|---:|']
for bench, prefix in (('textvqa', 'p1_tv'), ('ocrbench', 'p1_oc')):
    for pair in ('rb_fv', 'rb_none', 'fv_none', 'hl_none', 'hl_rb', 'hl_fv'):
        d = load(f'{prefix}_{pair}')
        if d:
            a, b = summary(d), summary(d, True)
            md.append(f'| {pair} | {bench} | {a["delta"]:+.2f} [{a["ci95"][0]:+.2f}, {a["ci95"][1]:+.2f}] | {b["delta"]:+.2f} [{b["ci95"][0]:+.2f}, {b["ci95"][1]:+.2f}] | {b["n"]} |')

# A replacement main table never changes the retained operational results.
matched_path = REPORT / 'cvm2027_matched_main_20261026.json'
if matched_path.exists():
    matched = json.loads(matched_path.read_text())
    assert matched['complete'] and matched['cell_count'] == 8 and matched['validation']['complete']
    for bench in ('textvqa', 'ocrbench'):
        for source in matched['sources'][bench].values():
            path = Path(source['path'])
            assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], 'Matched main summary is stale'
    summary_json['matched_main'] = matched
    md += ['', '## New attention-matched main comparison at 25% retention', '',
           'Shared vision SDPA / text eager for all four methods. Both tasks use complete splits and the same inputs; the official Hi-Lo model remains an adapted implementation. The preceding operational table and the following original budget curves keep their original protocols.', '',
           '| Benchmark | None@100% | RBM@25% | FastV-k3@25% | Hi-Lo@25% |',
           '|---|---:|---:|---:|---:|']
    for bench in ('textvqa', 'ocrbench'):
        cells = matched['benchmarks'][bench]['cells']
        md.append(f'| {bench} | ' + ' | '.join(f'{cells[m]["attempted_score"]:.2f}' for m in ('none', 'pre', 'fastv', 'hilo')) + ' |')
    md += ['', 'All twelve planned pair comparisons, 20,000-resample confidence intervals, failure IDs and per-ID input/budget checks are in `cvm2027_matched_main_20261026.md` and its JSON.']

full_ready = all(curve('textvqa', m, k, True) and curve('textvqa', m, k, True)['A']['n'] == 5000
                 for m in METHODS for k in KEEP)
summary_json['textvqa_full_grid_complete'] = full_ready
curve_specs = [('textvqa', False, False, 'TextVQA fixed 1,000-question subset'),
               ('ocrbench', False, False, 'OCRBench full 1,000 attempts'),
               ('ocrbench', False, True, 'OCRBench common 921 answered IDs, normalized ×1,000')]
if full_ready:
    curve_specs.insert(0, ('textvqa', True, False, 'TextVQA full 5,000-question grid'))
else:
    md += ['', 'Optional full TextVQA grid: partial; subset curves remain the complete comparison. Completed extension cells are retained in scores/full_q10_*.json.']
for bench, full, common, label in curve_specs:
    key = f'{bench}_{"full" if full or bench == "ocrbench" else "subset"}_{"common" if common else "attempted"}'
    summary_json['curves'][key] = {}
    md += ['', f'## {label}', '', '| Keep | RBM | FastV-k3 | Hi-Lo | None |', '|---|---:|---:|---:|---:|']
    for k in KEEP:
        ds = {m: curve(bench, m, k, full) for m in METHODS}
        assert all(ds.values()), (bench, k)
        row = {m: summary(d, common) for m, d in ds.items()}
        assert len({v['n'] for v in row.values()}) == 1
        assert max(v['b'] for v in row.values()) - min(v['b'] for v in row.values()) < 1e-10
        summary_json['curves'][key][k] = row
        md.append(f'| {k} | ' + ' | '.join(f'{row[m]["a"]:.2f}' for m in METHODS) + f' | {row["rb"]["b"]:.2f} |')

md += ['', '## DocVQA fixed 500-question subset, unified 2M pixel cap', '',
       '| Method | ANLS (%) | Δ vs None [95% CI] | p |', '|---|---:|---:|---:|']
for m, source in (('rb', 'pre'), ('fv', 'fastv'), ('hl', 'hilo')):
    d = load(f'doc_{source}'); v = summary(d)
    md.append(f'| {METHODS[m]}@25% | {v["a"]:.2f} | {v["delta"]:+.2f} [{v["ci95"][0]:+.2f}, {v["ci95"][1]:+.2f}] | {pformat(v["p"])} |')
md.append(f'| None@100% | {v["b"]:.2f} | — | — |')
for name in ('rb_hl', 'rb_fv'):
    v = summary(load(f'doc_{name}'))
    md.append(f'\n{name}: Δ={v["delta"]:+.2f}pp, CI[{v["ci95"][0]:+.2f}, {v["ci95"][1]:+.2f}], p={pformat(v["p"])}.')

md += ['', '## 95% of None quality target', '',
       'This table gives the minimum **tested retention** meeting the target. It does not assume retention orders measured latency. The Hi-Lo OCRBench attempted result includes its higher completion rate.', '',
       '| Scope | None | 95% target | RBM keep | FastV keep | Hi-Lo keep |', '|---|---:|---:|---:|---:|---:|']
for bench, full, common, label in curve_specs:
    ds = {m: {k: curve(bench, m, k, full) for k in KEEP} for m in METHODS}
    ref = vals(ds['rb']['0.25'], common)[1]
    targets = {m: next((k for k in KEEP if vals(ds[m][k], common)[0] >= .95 * ref), 'not met') for m in METHODS}
    summary_json['quality95'].append({'scope': label, 'none': ref * SCALE[bench], 'target': .95 * ref * SCALE[bench], 'minimum_tested_keep': targets})
    md.append(f'| {label} | {ref*SCALE[bench]:.2f} | {.95*ref*SCALE[bench]:.2f} | ' + ' | '.join(targets.values()) + ' |')

# Inspect source prediction rows rather than misusing the pre-mode n_image_full field.
row_cache = {}
def rows(source):
    p = Path(source)
    if not p.is_absolute(): p = ROOT / p
    if p not in row_cache:
        data = json.loads(p.read_text()); row_cache[p] = {str(r['id']): r for r in data['per_sample']}
    return row_cache[p]
md += ['', '## Actual visual token budgets', '',
       'Statistics use the same answered IDs across the three methods at each budget. Hi-Lo fusion boosts attention before removal; its fusion candidates are not additional retained tokens. RBM retains K before decoder layer 0; FastV-k3 processes N through layers 0–3, then K; Hi-Lo processes N through layers 0–2, then K. All methods execute the full vision encoder and merger.', '',
       '| Scope | Keep | Common n | Mean input N | Mean K (RBM / FastV / Hi-Lo) | K equal on every common ID |', '|---|---:|---:|---:|---:|---|']
actual = {}
for bench in ('textvqa', 'ocrbench'):
    full = False  # Keep the pre-fixed TV1000 scope regardless of optional full-grid completion.
    actual[bench] = {}
    for k in KEEP:
        ds = {m: curve(bench, m, k, full) for m in METHODS}
        rr = {m: rows(ds[m]['A']['source']) for m in METHODS}
        reference = rows(ds['rb']['B']['source'])
        common = set(reference)
        for r in rr.values(): common &= set(r)
        common = sorted(i for i in common if all(not r[i].get('skipped') and not r[i].get('error') for r in [reference] + list(rr.values())))
        mismatch = [i for i in common if len({r[i]['n_image_kept'] for r in rr.values()}) != 1]
        formula_bad = [i for i in common if any(r[i]['n_image_kept'] != max(1, round(reference[i]['n_image_full']*float(k))) for r in rr.values())]
        assert not mismatch and not formula_bad, (bench, k, mismatch[:3], formula_bad[:3])
        mean_n = statistics.mean(reference[i]['n_image_full'] for i in common)
        mean_k = {m: statistics.mean(rr[m][i]['n_image_kept'] for i in common) for m in METHODS}
        actual[bench][k] = {'common_n': len(common), 'mean_input_n': mean_n, 'mean_k': mean_k, 'mismatch_count': len(mismatch)}
        md.append(f'| {bench} {"full" if full or bench == "ocrbench" else "1k subset"} | {k} | {len(common)} | {mean_n:.2f} | ' + ' / '.join(f'{mean_k[m]:.2f}' for m in METHODS) + ' | yes |')
summary_json['actual_k'] = actual

md += ['', '## Timing validation', '',
       'The original timing artifacts are excluded from comparison: the harness enabled gradient recording, FastV omitted deepstack features, and the Hi-Lo model reused its initial retention configuration at later budgets. Corrected timing requires a new artifact with explicit inference, parity, budget, and boundary validation.']
# Only a complete, validated ten-cell workload can enter the summary.
timing_paths = [Q / f'timing_{engine}_corrected.json' for engine in ('harness', 'hilo')]
timing_validation = None
if all(p.exists() for p in timing_paths):
    try:
        timing_validation = validate_timing(timing_paths)
    except (AssertionError, KeyError) as error:
        md.append(f'\nCorrected timing validation pending: `{error}`.')
expected_timing = {'none'} | {f'{m}_{k}' for m in ('pre', 'fastv', 'hilo') for k in (.25, .5, .75)}
if timing_validation and timing_validation['complete'] and set(timing_validation['cells']) == expected_timing:
    summary_json['timing_validation'] = timing_validation
    for p in timing_paths:
        data = json.loads(p.read_text())
        summary_json['timing'][p.name] = {'available': True, 'validated': True, 'results': {k: {x: v for x, v in block.items() if x != 'requests'} for k, block in data['results'].items()}}
        md.append(f'\nValidated corrected timing available: `{p.name}`; quality-cost statistics use its own saved answers.')
else:
    md.append('\nCorrected timing summary withheld until all ten planned cells pass validation.')
(REPORT / 'cvm2027_supplement_tables.md').write_text('\n'.join(md) + '\n')
(REPORT / 'cvm2027_supplement_summary.json').write_text(json.dumps(summary_json, indent=2) + '\n')

# Curves: show the attempted OCRBench view alongside its fixed-ID sensitivity.
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))
for ax, (bench, full, common, title) in zip(axes, [
    ('textvqa', False, False, 'TextVQA subset 1,000'),
    ('ocrbench', False, False, 'OCRBench: all 1,000 attempts'),
    ('ocrbench', False, True, 'OCRBench: same 921 answered')]):
    ref = vals(curve(bench, 'rb', '0.25', full), common)[1] * SCALE[bench]
    ax.axhline(ref, ls='--', c='black', lw=1, label='None')
    ax.axhline(.95*ref, ls=':', c='gray', lw=1, label='95% target')
    for m, label in METHODS.items():
        ax.plot([25, 50, 75], [vals(curve(bench,m,k,full), common)[0]*SCALE[bench] for k in KEEP], 'o-', c=COLORS[m], label=label)
    ax.set(xlabel='Retained visual tokens (%)', ylabel='Official score (%)' if bench == 'textvqa' else 'Score normalized to 1,000', title=title)
    ax.set_xticks([25, 50, 75])
    ax.margins(x=.07, y=.10)
handles, labels = axes[0].get_legend_handles_labels()
lookup = dict(zip(labels, handles))
order = ['None', 'RBM', 'FastV-k3', 'Hi-Lo', '95% target']
fig.legend([lookup[label] for label in order], order, loc='lower center',
           bbox_to_anchor=(.5, .015), ncol=5, fontsize=10.5, frameon=False)
fig.tight_layout(rect=(0, .16, 1, 1))
save_figure(fig, 'budget_curves')
plt.close(fig)

# Optional full-grid results are a separate artifact, never a silent swap.
if full_ready:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ref = vals(curve('textvqa', 'rb', '0.25', True))[1] * 100
    ax.axhline(ref, ls='--', c='black', lw=1, label='None')
    ax.axhline(.95 * ref, ls=':', c='gray', lw=1, label='95% target')
    for method, label in METHODS.items():
        ax.plot([25, 50, 75], [vals(curve('textvqa', method, k, True))[0] * 100 for k in KEEP],
                'o-', c=COLORS[method], label=label)
    ax.set(xlabel='Retained visual tokens (%)', ylabel='Official score (%)',
           title='Optional TextVQA full 5,000 grid (original mixed protocol)')
    ax.set_xticks([25, 50, 75])
    ax.margins(x=.07, y=.10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    save_figure(fig, 'textvqa_full_budget_curves')
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, bench in zip(axes, ('textvqa', 'ocrbench')):
    full = False  # The main actual-K plot uses the same fixed TV1000 IDs.
    for m, label in METHODS.items():
        ax.plot([actual[bench][k]['mean_k'][m] for k in KEEP], [vals(curve(bench,m,k,full), bench=='ocrbench')[0]*SCALE[bench] for k in KEEP], 'o-', c=COLORS[m], label=label)
    ax.set(xlabel='Mean actual retained K', ylabel='Official score (%)' if bench=='textvqa' else 'Common-answered score ×1,000', title=bench)
    ax.legend(fontsize=10.5)
fig.tight_layout()
save_figure(fig, 'score_vs_actual_k')
plt.close(fig)
# Remove the invalid legacy latency plot so it cannot be consumed by the paper.
legacy = FIG / 'score_vs_latency.svg'
cost_summary_path = REPORT / 'cvm2027_corrected_quality_cost.json'
if cost_summary_path.exists():
    from cvm24h_plot_quality_cost import plot_quality_cost
    cost_summary = json.loads(cost_summary_path.read_text())
    for source in cost_summary['provenance'].values():
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256'], 'Corrected cost summary is stale'
    plot_quality_cost(cost_summary, FIG)
elif legacy.exists():
    legacy.unlink()
print(f'Audited tables and curves written; optional full TextVQA ready={full_ready}')

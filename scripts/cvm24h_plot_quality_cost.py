#!/usr/bin/env python
"""Render corrected quality-cost figures from an existing audited summary.
CPU plot only: never reads model weights or changes scores/timing/CI artifacts.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                           'axes.labelsize': 12.5, 'axes.titlesize': 13,
                           'xtick.labelsize': 11, 'ytick.labelsize': 11})
ROOT = Path(__file__).resolve().parents[1]


def plot_quality_cost(result, figure_dir):
    assert result['validation']['complete'] and len(result['configs']) == 10
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    colors = {'pre': '#1f77b4', 'fastv': '#d62728', 'hilo': '#2ca02c'}
    markers = {'pre': 'o', 'fastv': 's', 'hilo': '^'}
    # Offsets separate neighbouring retention labels; no labels cover a legend.
    offsets = {
        'textvqa': {'pre_0.25': (-9, -16), 'pre_0.5': (-30, 5), 'pre_0.75': (-25, -18),
                    'fastv_0.25': (-16, -17), 'fastv_0.5': (8, -13), 'fastv_0.75': (7, 6),
                    'hilo_0.25': (7, -10), 'hilo_0.5': (8, -5), 'hilo_0.75': (8, -3)},
        'ocrbench': {'pre_0.25': (-15, -16), 'pre_0.5': (-26, 8), 'pre_0.75': (-25, -20),
                     'fastv_0.25': (7, -15), 'fastv_0.5': (7, -5), 'fastv_0.75': (9, 10),
                     'hilo_0.25': (7, -12), 'hilo_0.5': (-26, -15), 'hilo_0.75': (8, 3)}}
    for ax, bench in zip(axes, ('textvqa', 'ocrbench')):
        ref = result['configs']['none'][bench]
        ax.scatter(ref['total_s']['p50'], ref['official_score'], c='black', marker='*', s=90, label='None', zorder=5)
        ax.axhline(.95 * ref['official_score'], ls=':', c='gray', lw=1, label='95% quality target')
        for method, label in [('pre', 'RBM'), ('fastv', 'FastV-k3'), ('hilo', 'Hi-Lo')]:
            pts = sorted([(tag, values[bench]) for tag, values in result['configs'].items()
                          if tag.startswith(method + '_')], key=lambda pair: pair[1]['keep'])
            ax.plot([cell['total_s']['p50'] for _, cell in pts],
                    [cell['official_score'] for _, cell in pts],
                    marker=markers[method], color=colors[method], label=label)
            for tag, cell in pts:
                ax.annotate(f'{cell["keep"]*100:.0f}%',
                            (cell['total_s']['p50'], cell['official_score']),
                            xytext=offsets[bench][tag], textcoords='offset points',
                            fontsize=10.5, color=colors[method])
        ax.set(xlabel='p50 total request latency (s)',
               ylabel='Official score (%)' if bench == 'textvqa' else 'Official score normalized ×1,000',
               title=f'{bench}: fixed 100-question subset')
        ax.margins(x=.15, y=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    lookup = dict(zip(labels, handles))
    order = ['None', 'RBM', 'FastV-k3', 'Hi-Lo', '95% quality target']
    fig.legend([lookup[label] for label in order], order, loc='lower center',
               bbox_to_anchor=(.5, .015), ncol=5, fontsize=10.5, frameon=False)
    fig.tight_layout(rect=(0, .15, 1, 1))
    for suffix in ('svg', 'pdf'):
        path = figure_dir / f'score_vs_latency.{suffix}'
        fig.savefig(path)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', type=Path, default=ROOT / 'reports/cvm2027_corrected_quality_cost.json')
    ap.add_argument('--figure-dir', type=Path, default=ROOT / 'reports/figures_cvm2027_supplement')
    a = ap.parse_args()
    plot_quality_cost(json.loads(a.summary.read_text()), a.figure_dir)
    print('Quality-cost figures regenerated from saved summary; scores and CIs unchanged')


if __name__ == '__main__':
    main()

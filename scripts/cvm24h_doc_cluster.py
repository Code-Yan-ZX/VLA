#!/usr/bin/env python
"""CPU-only DocVQA image-cluster bootstrap; existing primary CIs stay untouched.
Reads only metadata columns from original validation parquet shards, never the
image column. Reuses saved predictions and the pinned official ANLS helper.
Run from any directory with the existing qwen3vl_clean Python environment.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cvm24h_score as SC
import analyze_cvm_docvqa_official as DOC

COLUMNS = ['questionId', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'question', 'answers']
RESAMPLES = 20000
METADATA_SHA256 = '32d44390dbf3bef851e09497a93ad74ab428f6e3f3bbc5ec21cc93f866be1930'
MAPPING_SHA256 = '0b32bff9473aebb0fdc00750bcebe71d8d3c8e24f60fde7ebaa43576560434ec'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata-dir', type=Path,
                    default=Path('/media/disk2/YZX/research/vla/runs/data/staging/docvqa'))
    a = ap.parse_args()
    os.chdir(ROOT)  # Scoring wrapper resolves its pinned reference cache here.
    data_dir = ROOT / 'results/cvm2027_supplement_20261026'
    score_dir = data_dir / 'plan/scores'
    report_dir = ROOT / 'reports'
    metadata, files = [], []
    for path in sorted(a.metadata_dir.glob('validation-*.parquet')):
        records = pq.read_table(path, columns=COLUMNS).to_pylist()
        metadata.extend(records)
        files.append({'path': str(path), 'size_bytes': path.stat().st_size,
                      'rows': len(records), 'columns_read': COLUMNS,
                      'metadata_projection_sha256': digest(records)})
    reference = {str(row['questionId']): row for row in metadata}
    assert len(reference) == len(metadata) == 5349
    assert digest(metadata) == METADATA_SHA256, 'Original annotation metadata changed'
    by_doc, by_page = defaultdict(set), defaultdict(set)
    for row in metadata:
        page = (row['ucsf_document_id'], row['ucsf_document_page_no'])
        by_doc[row['docId']].add(page)
        by_page[page].add(row['docId'])
    assert all(len(v) == 1 for v in by_doc.values()) and all(len(v) == 1 for v in by_page.values())
    ids_path = data_dir / 'plan/p2_docvqa_500_ids.txt'
    ids = sorted(ids_path.read_text().split())
    assert len(ids) == len(set(ids)) == 500
    mapping, clusters = [], defaultdict(list)
    for sid in ids:
        row = reference[sid]
        assert row['ucsf_document_id'] and row['ucsf_document_page_no']
        mapping.append({'question_id': sid, 'doc_id': row['docId'],
                        'ucsf_document_id': row['ucsf_document_id'], 'page_no': row['ucsf_document_page_no']})
        clusters[(row['ucsf_document_id'], row['ucsf_document_page_no'])].append(sid)
    cluster_ids = sorted(clusters)
    assert digest(mapping) == MAPPING_SHA256, 'Selected question-to-page mapping changed'
    counts = np.array([len(clusters[c]) for c in cluster_ids])
    n_clusters = len(cluster_ids)
    rows, scores, prediction_sources = {}, {}, {}
    for method, filename in [('rbm', 'doc500_pre25.json'), ('hilo', 'doc500_hilo25.json'), ('fastv', 'doc500_fastv25.json')]:
        path = data_dir / 'predictions' / filename
        raw = path.read_bytes()
        records = json.loads(raw)['per_sample']
        rows[method] = {str(row['id']): row for row in records}
        assert len(rows[method]) == len(records) == 500 and set(rows[method]) == set(ids)
        assert not any(row.get('skipped') or row.get('error') for row in records)
        for sid, row in rows[method].items():
            annotation = reference[sid]
            assert row['gt'] == ';'.join(str(v).strip() for v in annotation['answers'] if str(v).strip())
            assert row['question'] == annotation['question'].strip() + '\nAnswer the question using a single word or phrase.'
        scores[method] = SC.docvqa_scores(rows[method])
        prediction_sources[method] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    output = {
        'scope': 'DocVQA pre-fixed 500-question subset, 25% retention, unified 2M pixel cap',
        'purpose': 'Optional image/page-cluster bootstrap sensitivity; existing question-level primary results and saved GT are unchanged',
        'mapping_source': {
            'dataset': 'lmms-lab/DocVQA, DocVQA validation', 'files': files,
            'dataset_revision_note': 'Original staging files do not preserve a verified Hub revision; canonical metadata-projection digests identify the exact local annotation content.',
            'metadata_projection_sha256': digest(metadata),
            'all_questions': len(metadata), 'all_doc_ids': len(by_doc), 'all_ucsf_pages': len(by_page),
            'docid_page_bijection_verified': True, 'selected_mapping_sha256': digest(mapping),
            'ids_sha256': hashlib.sha256(ids_path.read_bytes()).hexdigest(),
            'metadata_only_no_image_bytes_read': True},
        'scoring_reference': {'commit': DOC.COMMIT, 'url': DOC.URL, 'sha256': DOC.SHA256},
        'cluster_definition': ['ucsf_document_id', 'ucsf_document_page_no'],
        'selected_questions': 500, 'selected_image_clusters': n_clusters,
        'cluster_size_histogram': dict(sorted(Counter(map(int, counts)).items())),
        'bootstrap': {
            'resamples': RESAMPLES,
            'sampling': 'Draw C image/page clusters uniformly with replacement; include every sampled cluster question, preserving multiplicities; each replicate uses its own question-count denominator',
            'estimand': 'question-weighted mean ANLS difference on the selected questions',
            'interval': 'paired percentile 95%; no new p-value or equivalence claim',
            'seed_rule': '20261400 for RBM-HiLo, 20261401 for RBM-FastV'},
        'prediction_sources': prediction_sources, 'pairs': {},
        'reproduce_command': '/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/cvm24h_doc_cluster.py'}
    for i, (method, label) in enumerate([('hilo', 'rb_hl'), ('fastv', 'rb_fv')]):
        delta = {sid: scores['rbm'][sid] - scores[method][sid] for sid in ids}
        observed = float(np.mean(list(delta.values())))
        sums = np.array([sum(delta[sid] for sid in clusters[c]) for c in cluster_ids])
        rng, replicates = np.random.default_rng(20261400 + i), []
        for start in range(0, RESAMPLES, 128):
            draw = rng.integers(0, n_clusters, (min(128, RESAMPLES - start), n_clusters))
            replicates.extend((sums[draw].sum(axis=1) / counts[draw].sum(axis=1)).tolist())
        lo, hi = np.quantile(replicates, [.025, .975])
        primary = json.loads((score_dir / f'doc_{label}.json').read_text())['paired_attempted_all']
        assert abs(observed * 100 - primary['delta_pp']) < 1e-10
        output['pairs'][label] = {
            'n_questions': 500, 'n_clusters': n_clusters, 'delta_pp': observed * 100,
            'cluster_ci95_pp': [float(lo) * 100, float(hi) * 100], 'cluster_bootstrap_seed': 20261400 + i,
            'question_ci95_pp_unchanged': primary['ci95_pp'], 'question_bootstrap_seed': primary['seed'],
            'question_resamples': primary['resamples'], 'rbm_score': sum(scores['rbm'].values()) / 500,
            'comparison_score': sum(scores[method].values()) / 500}
    base = report_dir / 'cvm2027_docvqa_cluster_sensitivity_20261026'
    base.with_suffix('.json').write_text(json.dumps(output, indent=2) + '\n')
    md = ['# DocVQA image-cluster sensitivity (2026-10-26)', '',
          f'The frozen 500-question DocVQA subset contains **{n_clusters} document-page images**. Original parquet metadata maps questionId to docId and the tuple (ucsf_document_id, ucsf_document_page_no); those identifiers form a verified bijection on all 5,349 validation questions. QID-based filenames were not used as cluster IDs. Only metadata columns were read, without image decoding or content hashing.', '',
          'Cluster sizes are 348 single-question pages, 57 two-question pages, 10 three-question pages, and 2 four-question pages. All 500 saved questions and joined GT strings match the original metadata. The existing saved predictions are rescored with the same pinned ANLS function; primary scores and question-level CIs remain unchanged.', '',
          'For each of 20,000 paired bootstrap replicates, sample 417 images with replacement, carry all their questions with multiplicities, and divide by that replicate\'s question count. Fixed seeds are 20261400 and 20261401.', '',
          '| Pair | ANLS difference (pp) | Primary question-bootstrap 95% CI | Image-cluster 95% CI |',
          '|---|---:|---:|---:|']
    for label, title in [('rb_hl', 'RBM − Hi-Lo'), ('rb_fv', 'RBM − FastV-k3')]:
        p = output['pairs'][label]
        q, c = p['question_ci95_pp_unchanged'], p['cluster_ci95_pp']
        md.append(f'| {title} | {p["delta_pp"]:+.2f} | [{q[0]:+.2f}, {q[1]:+.2f}] | [{c[0]:+.2f}, {c[1]:+.2f}] |')
    md += ['', 'Both intervals remain above zero. This subset sensitivity does not remove the Hi-Lo attention/model-code confound or establish full-split generalization. No new equivalence or p-value claim is made.', '',
           'Reproduce from the worktree root:', '', '```bash', output['reproduce_command'], '```', '',
           f'ANLS source: lmms-eval commit `{DOC.COMMIT}`, `lmms_eval/api/metrics.py`, SHA256 `{DOC.SHA256}`.', '',
           f'Original staging metadata has no independently verified Hub revision. Its canonical metadata-column projection SHA256 is `{output["mapping_source"]["metadata_projection_sha256"]}`; the selected question-to-page mapping SHA256 is `{output["mapping_source"]["selected_mapping_sha256"]}`. The JSON records per-shard metadata digests, sizes, prediction SHA256s, seeds, and unrounded results.']
    base.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({'clusters': n_clusters, 'pairs': output['pairs']}, indent=2))


if __name__ == '__main__':
    main()

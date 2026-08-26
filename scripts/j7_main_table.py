import json, glob, sys
sys.path.insert(0, 'src/v3_premerger')
import official_scorers as S
meta = {}
for line in open('eval/full_splits/ocrbench.jsonl'):
    o = json.loads(line)
    meta[str(o['id'])] = (o.get('question_type', ''), o.get('category', ''))
rows = []
for f in sorted(glob.glob('runs/full_matrix/j7_*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    ps = d.get('per_sample') or []
    b = d.get('benchmark'); n = len(ps)
    if not n:
        continue
    preds = [str(p.get('answer', '')) for p in ps]
    gts = [str(p.get('gt', '')) for p in ps]
    off = fin = None
    if b == 'textvqa':
        off = sum(S.score_textvqa_vqaacc(a, g) for a, g in zip(preds, gts)) / n
    elif b == 'docvqa':
        off = sum(S.score_docvqa_anls(a, g) for a, g in zip(preds, gts)) / n
    elif b == 'gqa':
        off = S.score_gqa_batch(preds, gts)['acc']
    elif b == 'ocrbench':
        items = []
        for a, g, p in zip(preds, gts, ps):
            qt, cat = meta.get(str(p.get('id')), ('', ''))
            items.append((a, g, qt, cat))
        r = S.score_ocrbench_batch(items)
        off = r['acc']; fin = r.get('final_score')
    rows.append((f.split('/')[-1], d.get('model_family'), b, d.get('mode'),
                 d.get('r'), off, fin, d.get('mean_ptid_len')))
M = {'qwen3vl': 'Qwen3-VL-8B', 'qwen2vl': 'Qwen2.5-VL-7B'}
Rr = {0.0: 'none', 0.75: '@25%', 0.875: '@12.5%'}
Bn = ['textvqa', 'docvqa', 'ocrbench', 'gqa']
for fam in ['qwen3vl', 'qwen2vl']:
    print('\n=== %s (official metrics, full split) ===' % M[fam])
    print('%9s | %6s | %6s | %6s | %10s | %8s | %8s' %
          ('bench', 'none', 'pre@25', 'post@25', 'd-pre-post', 'pre@12.5', 'post@12.5'))
    for bench in Bn:
        def get(mode, r):
            for fn, fl, bb, m2, r2, off, fin, pt in rows:
                if fl == fam and bb == bench and m2 == mode and r2 == r:
                    return off
            return None
        none = get('none', 0.0); pre75 = get('pre', 0.75); post75 = get('post', 0.75)
        pre875 = get('pre', 0.875); post875 = get('post', 0.875)
        def fmt(x):
            return '%.3f' % x if isinstance(x, float) and x > 0 else ('%.3f' % x if isinstance(x, float) else '  N/A')
        d75 = '%+.1fpp' % ((pre75 - post75) * 100) if (isinstance(pre75, float) and isinstance(post75, float) and pre75 > 0 and post75 > 0) else '?'
        print('%9s | %6s | %6s | %6s | %10s | %8s | %8s' %
              (bench, fmt(none), fmt(pre75), fmt(post75), d75, fmt(pre875), fmt(post875)))
print('\nOCRBench Final/1000 (per cell):')
for fam in ['qwen3vl', 'qwen2vl']:
    for mode in ['none', 'pre', 'post']:
        for r in [0.75, 0.875] if mode != 'none' else [0.0]:
            for fn, fl, bb, m2, r2, off, fin, pt in rows:
                if fl == fam and bb == 'ocrbench' and m2 == mode and r2 == r:
                    if fin is not None:
                        print('  %14s %-5s@%s: Final=%4s/1000 (%.1f%%)  acc=%.3f' %
                              (M[fam], mode, Rr[r], fin, fin / 10.0, off))
json.dump([{'file': r[0], 'model': r[1], 'bench': r[2], 'mode': r[3], 'r': r[4],
            'official': r[5], 'final_ocrbench': r[6], 'ptid': r[7]} for r in rows],
          open('runs/full_matrix/j7_main_table.json', 'w'), indent=1, ensure_ascii=False)
print('\nwrote runs/full_matrix/j7_main_table.json')
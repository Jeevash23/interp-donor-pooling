"""Recompute reviewer-relevant September diagnostics without changing old files."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parents[1] / 'submission-study'
MODELS = ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'pythia-410m', 'tinyllama-1.1b', 'qwen3-0.6b']


def main():
    rows = json.loads((OLD / 'data/test.json').read_text())
    task = np.array([r['task'] for r in rows])
    block = np.array([r['seed_block'] for r in rows])
    rng = np.random.default_rng(20261002)
    draws = np.concatenate([rng.choice(np.flatnonzero((task == t) & (block == b)), (20000, 32))
                            for t in ['ioi', 'sva', 'kv'] for b in range(3)], axis=1)
    output = {'status': 'retrospective diagnostics, not new confirmatory outcomes', 'models': {}}
    for model in MODELS:
        dr = OLD / 'results' / (model + '-float32')
        design = json.loads((dr / 'design.json').read_text())
        metrics, endpoints, dispersion, baseline = [], [], [], []
        for row in rows:
            with np.load(dr / 'test' / (row['id'] + '.npz')) as z:
                metrics.append(z['metrics']); endpoints.append(z['endpoints'])
                dispersion.append(z['dispersion']); baseline.append(z['baseline'])
        M, E, S, B = map(np.stack, [metrics, endpoints, dispersion, baseline])
        comps = {}
        for i, bank in enumerate(design['banks']):
            a = M[:, :, bank['mean'], 0]
            r = M[:, :, bank['indices'], 0].mean(2)
            b, c = E[:, :, i, 0, 0], E[:, :, i, 1, 0]
            comps[bank['name']] = {'D': a-r, 'P': a-b, 'N': b-c}
            assert np.max(np.abs((a-r)-((a-b)+(b-c)))) < 1e-4
        narrow = [comps[d + '/narrow/full'] for d in ['restore', 'destroy']]
        pn = np.stack([abs(x['P'][:, :6]) - abs(x['N'][:, :6]) for x in narrow], -1)
        g = pn.mean((1, 2)); sample = g[draws].mean(1)
        sva = task == 'sva'
        # These disjoint cells comprise the original equal-task primary mean.
        sva_end_part = pn[sva][:, [1, 3, 5]].mean() / 6
        remainder = np.concatenate([pn[:, [0, 2, 4]].ravel(), pn[~sva][:, [1, 3, 5]].ravel()])
        gap = np.stack([abs(x['D'][:, :6]) for x in narrow], -1)
        shared = np.concatenate([gap[:, [0, 2, 4]].ravel(), gap[~sva][:, [1, 3, 5]].ravel()])
        x, y = [], []
        for t in ['ioi', 'sva', 'kv']:
            for site in range(6):
                for di, direction in enumerate(['restore', 'destroy']):
                    for bi, bank in enumerate(['narrow', 'broad']):
                        ix = task == t
                        x.append(np.log((S[ix, site, 2*bi+di, 0] / S[ix, site, 2*bi+di, 1]).mean()))
                        y.append(np.log(abs(comps[direction + '/' + bank + '/full']['D'][ix, site]).mean()))
        results = {
            'old_primary': float(g.mean()),
            'old_primary_99.1667_ci': np.quantile(sample, [.05/12, 1-.05/12]).tolist(),
            'sva_end_share_of_old_primary': float(sva_end_part / g.mean()),
            'old_primary_excluding_sva_end': float(remainder.mean()),
            'absolute_gap_shared_token_cells': float(shared.mean()),
            'absolute_gap_sva_end': float(gap[sva][:, [1, 3, 5]].mean()),
            'condition_mean_log_gap_log_normalized_dispersion_correlation': float(np.corrcoef(x, y)[0, 1]),
            'terminal_end_abs_N': float(np.mean([abs(z['N'][:, 7]).mean() for z in narrow])),
            'clean_pair_accuracy': {t: float(B[task == t, 0, 3].mean()) for t in ['ioi','sva','kv']},
            'clean_unrestricted_top1_accuracy': {t: float(B[task == t, 0, 2].mean()) for t in ['ioi','sva','kv']},
        }
        if (dr / 'heads').is_dir() and any((dr / 'heads').glob('*.npz')):
            heads = []
            for row in rows:
                if row['task'] == 'ioi':
                    with np.load(dr / 'heads' / (row['id'] + '.npz')) as z: heads.append(z['metrics'])
            h = np.stack(heads)
            d = (h[:, :, 8, 0] - h[:, :, :8, 0].mean(2)).mean(0)
            results['head_max_aggregate_gap_difference'] = float(np.ptp(d))
        output['models'][model] = results
    tracked = [OLD / 'data/test.json', OLD / 'data/freeze.json', OLD / 'output/pdf/donor-pooling-boundary-audit.pdf']
    output['previous_input_sha256'] = {str(p.relative_to(OLD)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    dest = ROOT / 'results/previous-study-diagnostics.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

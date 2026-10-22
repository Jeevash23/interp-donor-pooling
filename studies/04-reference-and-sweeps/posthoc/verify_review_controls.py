"""Independent check of the post hoc review controls (results/posthoc/review-controls.json).

Imports nothing from the study's code. It reads the raw arrays and recomputes with separate code:
exact quantities exactly (mean-ablation counts and errors, the token rule, the IOI point estimates, the Study 1
slopes, every AUROC, the rank correlations and the surface-feature baseline, refitted with a different solver),
and Monte Carlo quantities (sampled donors, split halves, bootstrap intervals) by redrawing with other seeds and
comparing within tolerances.
Output: results/posthoc/verification-review-controls.json (exit code 1 if anything differs).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT.parents[0] / "03-grouping-and-circuit"
SAVED = ROOT / "results/posthoc/review-controls.json"
PANELS = {"sva-gpt2-small": ("sva", 64), "sva-qwen3-0.6b": ("sva", 32), "ioi-gpt2-small": ("ioi", 64)}
TAU, EXACT = 0.05, 1e-9
failures, performed = [], []


def read(p):
    return json.loads(Path(p).read_text())


def check(name, ok, detail=""):
    performed.append(name)
    if not ok:
        failures.append(f"{name}: {detail}")
    return bool(ok)


def close(a, b, tol=EXACT):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------- sweeps
def load_panel(panel):
    task, n = PANELS[panel]
    folder = ROOT / "results/sweep" / panel / "test"
    rows = read(ROOT / ("data/sva-test.json" if task == "sva" else "data/ioi-sweep.json"))
    rows = [r for r in rows if (folder / f"{r['id']}.npz").exists()][:n]
    m, clean, corrupt, strata = [], [], [], []
    for r in rows:
        with np.load(folder / f"{r['id']}.npz") as z:
            if task == "sva":
                m.append(z["margins"]); clean.append(z["baseline"][0]); corrupt.append(z["baseline"][1])
                strata.append(f"{r['template']}-{r['class']}")
            else:
                m.append(z["margins"][None, :, -14:]); clean.append(z["baseline"][0]); corrupt.append(float(np.mean(z["donor_margins"])))
                strata.append(f"{r['template']}-{r['order']}")
    m = np.stack(m)
    clean, corrupt = np.array(clean), np.array(corrupt)
    base = (np.stack([corrupt, clean], 1)[:, :, None, None, None] if task == "sva" else clean[:, None, None, None, None])
    live = np.abs(m - base).max(axis=(0, -1)) > 1e-4
    return task, rows, m, clean, corrupt, strata, live


def effect(task, gm, c, k):
    """gm: margins averaged over groups, [D, L, P]."""
    out = np.empty_like(gm)
    for d in range(gm.shape[0]):
        if task == "sva" and d == 0:
            out[d] = (gm[d] - k) / (c - k)
        else:
            out[d] = (c - gm[d]) / (c - k)
    return out


def verify_panel(panel, saved):
    task, rows, m, clean, corrupt, strata, live = load_panel(panel)
    G, D = m.shape[0], m.shape[1]
    c, k = clean.mean(), corrupt.mean()
    e_ind = effect(task, m[..., :64].mean(-1).mean(0), c, k)
    f_ind = np.abs(e_ind) >= TAU
    mean_idx = 88 if task == "sva" else 72
    e_mean = effect(task, m[..., mean_idx].mean(0), c, k)
    ma = saved["mean_ablation"]
    check(f"{panel}/mean_ablation", ma["misclassified"] == int(((np.abs(e_mean) >= TAU) != f_ind).sum())
          and close(ma["mae"], np.abs(e_mean - e_ind)[live].mean()), str(ma))
    rng = np.random.default_rng(4343)
    for kk, rec in saved["sampled_donors"].items():
        kk = int(kk)
        mis, mae = [], []
        for _ in range(300):
            sel = np.stack([[rng.permutation(64)[:kk] for _ in range(D)] for _ in range(G)])      # [G, D, k]
            gm = np.stack([[m[g, d][..., sel[g, d]].mean(-1) for d in range(D)] for g in range(G)]).mean(0)
            e = effect(task, gm, c, k)
            mis.append(int(((np.abs(e) >= TAU) != f_ind).sum())); mae.append(float(np.abs(e - e_ind)[live].mean()))
        tol = max(0.25, 0.15 * rec["misclassified_mean"])
        check(f"{panel}/sampled_donors/{kk}", abs(np.mean(mis) - rec["misclassified_mean"]) <= tol
              and abs(np.mean(mae) - rec["mae_mean"]) <= 0.05 * rec["mae_mean"],
              f"saved {rec['misclassified_mean']:.3f}/{rec['mae_mean']:.5f}, redrawn {np.mean(mis):.3f}/{np.mean(mae):.5f}")
    halves = []
    for _ in range(500):
        perm = np.stack([rng.permutation(64) for _ in range(G)])
        a = np.stack([m[g][..., perm[g, :32]].mean(-1) for g in range(G)]).mean(0)
        b = np.stack([m[g][..., perm[g, 32:]].mean(-1) for g in range(G)]).mean(0)
        halves.append(int(((np.abs(effect(task, a, c, k)) >= TAU) != (np.abs(effect(task, b, c, k)) >= TAU)).sum()))
    sh = saved["split_half"]
    check(f"{panel}/split_half", abs(np.mean(halves) - sh["halves_disagree_mean"]) <= 0.1 and max(halves) <= sh["halves_disagree_max"] + 1,
          f"saved {sh['halves_disagree_mean']:.3f} (max {sh['halves_disagree_max']}), redrawn {np.mean(halves):.3f} (max {max(halves)})")
    if task == "sva":
        rule = np.empty(m.shape[:-1])
        for p in range(m.shape[3]):
            if p == 0:
                cols = list(range(80, 88))          # ADJ8 bank centroids at the adjective
            elif p == m.shape[3] - 1:
                cols = list(range(72, 80))          # NOUN8 bank centroids at the final token
            else:
                cols = [88]                         # one group: mean ablation
            rule[..., p] = m[..., p, :][..., cols].mean(-1)
        e_rule = effect(task, rule.mean(0), c, k)
        tr = saved["token_rule"]["rule"]
        check(f"{panel}/token_rule", tr["misclassified"] == int(((np.abs(e_rule) >= TAU) != f_ind).sum())
              and close(tr["mae"], np.abs(e_rule - e_ind)[live].mean()), str(tr))
    if task == "ioi":
        verify_ioi(saved["ioi_count"], rows, m, clean, corrupt, strata)


def verify_ioi(saved, rows, m, clean, corrupt, strata):
    assert [r["id"] for r in rows[:4]] == [f"sweep-ioi-{i:05d}" for i in range(4)]
    rng = np.random.default_rng(777)
    for name, start, fp_only in [("as_published", 0, False), ("original_definition", 0, True),
                                 ("without_development_prompts", 2, False),
                                 ("without_development_prompts_original_definition", 2, True),
                                 ("without_calibration_prompts", 4, False)]:
        idx = np.arange(start, len(rows))
        mm, cc, kk = m[idx], clean[idx], corrupt[idx]

        def count(w):
            gi = np.tensordot(w, mm[..., :64].mean(-1), axes=(0, 0))
            gm = np.tensordot(w, mm[..., 72], axes=(0, 0))
            fi = np.abs(effect("ioi", gi, w @ cc, w @ kk)) >= TAU
            fm = np.abs(effect("ioi", gm, w @ cc, w @ kk)) >= TAU
            return int((fm & ~fi).sum()) if fp_only else int((fm != fi).sum())
        est = count(np.full(len(idx), 1 / len(idx)))
        st = [strata[i] for i in idx]
        boot = []
        for _ in range(2000):
            w = np.zeros(len(idx))
            for s in sorted(set(st)):
                ix = [i for i, x in enumerate(st) if x == s]
                for j in rng.choice(ix, size=len(ix), replace=True):
                    w[j] += 1
            boot.append(count(w / w.sum()))
        lo, hi = np.percentile(boot, [100 * 0.05 / 6, 100 * (1 - 0.05 / 6)])
        s = saved[name]
        check(f"ioi_count/{name}", s["estimate"] == est and s["ci"][0] >= 1 and lo >= 1 and abs(hi - s["ci"][1]) <= 1,
              f"saved {s['estimate']} {s['ci']}, recomputed {est} [{lo:.0f}, {hi:.0f}]")


# ---------------------------------------------------------------- Study 1 slopes
def verify_slopes(saved):
    rng = np.random.default_rng(99)
    for model in ["gpt2-small", "qwen3-0.6b"]:
        z = np.load(S1 / f"token/results/analysis-test/{model}-components.npz")
        gap = np.abs(z["gap"][:, [1, 3, 5]][..., 0])            # basic indexing on one axis keeps the order
        msr = z["dispersion"][:, [1, 3, 5]][..., 0]
        lg, lr = np.log(gap + 1e-12), np.log(msr)
        for name, ks, within in [("all_banks", [0, 1, 2, 3], False), ("within_policy", [0, 1, 2, 3], True),
                                 ("k_2_4_8_only", [1, 2, 3], False)]:
            num = np.zeros(lg.shape[0]); den = np.zeros(lg.shape[0])
            for r in range(lg.shape[0]):
                for s in range(3):
                    for d in range(2):
                        cells = [[ki] for ki in ks] if within else [ks]
                        for cell in cells:
                            y, x = lg[r, s, d, cell].ravel(), lr[r, s, d, cell].ravel()
                            num[r] += ((y - y.mean()) * (x - x.mean())).sum(); den[r] += ((x - x.mean()) ** 2).sum()
            b = num.sum() / den.sum()
            draws = rng.integers(0, len(num), size=(2000, len(num)))
            ci = np.percentile(num[draws].sum(1) / den[draws].sum(1), [2.5, 97.5])
            s = saved[model][name]
            check(f"slopes/{model}/{name}", close(s["slope"], b) and np.allclose(s["ci95"], ci, atol=0.03),
                  f"saved {s['slope']:.4f} {s['ci95']}, recomputed {b:.4f} {ci.tolist()}")


# ---------------------------------------------------------------- separation
def recipients():
    rows = read(ROOT / "data/ioi-selected-test.json")
    pops = read(ROOT / "results/reference/screen-selection.json")["strata"]
    strata = [f"{r['family']}/{r['stratum_label']}" for r in rows]
    n = {s: strata.count(s) for s in set(strata)}
    w = np.array([pops[s]["population"] / n[s] for s in strata])
    error = np.array([r["stratum_label"] == "pair_error" for r in rows])
    scores = {}
    for i, r in enumerate(rows):
        with np.load(ROOT / "results/reference/test" / f"{r['id']}.npz") as a, \
                np.load(ROOT / "results/posthoc/random-circuits/test" / f"{r['id']}.npz") as b:
            for cell in ["N8", "N128", "NPO8", "NPO128"]:
                units = {"circuit": (a, f"{cell}__reference"), "empty": (a, f"{cell}__empty")}
                units.update({f"random{k}": (b, f"{cell}__c{k}") for k in range(5)})
                for u, (z, pre) in units.items():
                    scores.setdefault((u, "mean", cell), []).append(float(z[f"{pre}__centroid_margin"]))
                    scores.setdefault((u, "resample", cell), []).append(float(np.mean(z[f"{pre}__donor_margins"])))
    return rows, error, w, {k: np.array(v) for k, v in scores.items()}


def auroc_pairs(score, error, w):
    """Weighted P(error score < correct score), ties counted half, by explicit pairs."""
    se, sc = score[error], score[~error]
    we, wc = w[error], w[~error]
    less = (se[:, None] < sc[None, :]).astype(float) + 0.5 * (se[:, None] == sc[None, :])
    return float((we[:, None] * wc[None, :] * less).sum() / (we.sum() * wc.sum()))


def wquant(x, w, q):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o]) / w[o].sum()
    return float(x[o][np.argmax(cw >= q)])


def wrank(a, b, w):
    ra, rb = rankdata(a, method="ordinal") - 1.0, rankdata(b, method="ordinal") - 1.0
    ma, mb = (w * ra).sum() / w.sum(), (w * rb).sum() / w.sum()
    return float((w * (ra - ma) * (rb - mb)).sum() / np.sqrt((w * (ra - ma) ** 2).sum() * (w * (rb - mb) ** 2).sum()))


def verify_separation(saved):
    rows, error, w, scores = recipients()
    full = np.array([r["screen_margin"] for r in rows])
    cut = {q: wquant(full[~error], w[~error], float(q)) for q in saved["intact_margin_cutoffs"]}
    check("separation/cutoffs", all(close(cut[q], saved["intact_margin_cutoffs"][q]) for q in cut), str(cut))
    for key, rec in saved["near_boundary"].items():
        unit, pool, cell = key.split("/")
        s = scores[(unit, pool, cell)]
        ok = close(rec["auroc_all"], auroc_pairs(s, error, w))
        for q in cut:
            keep = error | (full <= cut[q])
            ok &= close(rec[f"auroc_below_q{int(float(q) * 100)}"], auroc_pairs(s[keep], error[keep], w[keep]))
        ok &= close(rec["rank_corr_with_intact_errors"], wrank(s[error], full[error], w[error]))
        ok &= close(rec["rank_corr_with_intact_correct"], wrank(s[~error], full[~error], w[~error]))
        check(f"separation/{key}", ok, str(rec))
    screen = read(ROOT / "data/ioi-screen.json")
    with np.load(ROOT / "results/reference/screen.npz") as z:
        margin = dict(zip(z["ids"].tolist(), z["margin"].tolist()))
    chosen = {r["id"] for r in rows}
    train = [r for r in screen if r["id"] not in chosen]
    field = {"template": "template", "order": "order", "io": "target", "s": "subject", "place": "place", "object": "object"}
    for name, rec in saved["surface_baseline"].items():
        cols_tr, cols_te = [np.ones(len(train))], [np.ones(len(rows))]
        for f in rec["features"]:
            levels = sorted({r[field[f]] for r in screen})
            for lev in levels[1:]:
                cols_tr.append(np.array([r[field[f]] == lev for r in train], float))
                cols_te.append(np.array([r[field[f]] == lev for r in rows], float))
        X, y = np.stack(cols_tr, 1), np.array([margin[r["id"]] for r in train])
        beta = scipy.linalg.lstsq(X, y, lapack_driver="gelsy")[0]
        pred = np.stack(cols_te, 1) @ beta
        check(f"surface/{name}", abs(rec["auroc"] - auroc_pairs(pred, error, w)) <= 1e-6 and rec["parameters"] == X.shape[1],
              f"saved {rec['auroc']:.6f}, recomputed {auroc_pairs(pred, error, w):.6f}")


def main():
    saved = read(SAVED)
    for panel in PANELS:
        verify_panel(panel, saved["panels"][panel])
    verify_slopes(saved["spread_within_policy"])
    verify_separation(saved["separation"])
    report = {"checked": str(SAVED.relative_to(ROOT)),
              "review_controls_sha256": hashlib.sha256(SAVED.read_bytes()).hexdigest(),
              "checks_performed": len(performed), "checks": performed,
              "failures": failures, "passed": not failures}
    (ROOT / "results/posthoc/verification-review-controls.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(performed)} checks;", "passed" if not failures else "FAILED\n" + "\n".join(failures))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()

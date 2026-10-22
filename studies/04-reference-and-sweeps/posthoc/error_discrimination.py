"""Post hoc: does an ablated circuit pick out the prompts the model gets wrong?

Added after the Experiment A and random-circuit results were known; frozen files are only read.
For the IOI circuit, each of the five random circuits and the empty control, under both ablations and
all four references, on the 1,299 Experiment A recipients:
  (a) P(ablated model picks the subject | model-error prompt) and P(... | model-correct prompt);
      under resampling, the expected share of donors for which it picks the subject;
  (b) their odds ratio;
  (c) the AUROC of the ablated IO-minus-S margin for separating model-error from model-correct prompts
      (the centroid margin under mean ablation, the mean donor margin under resampling); and the same
      AUROC computed only within templates, which removes template differences between the two sets.
Population weights: within each template family and correctness stratum, prompts carry inverse
inclusion weights from the screen record. Intervals: 2,000 stratified bootstrap draws.
Output: results/posthoc/error-discrimination.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, dump, read, utc  # noqa: E402

CELLS = ["N8", "N128", "NPO8", "NPO128"]
DRAWS, SEED, N_RANDOM = 2000, 26100671, 5


def load():
    rows = read(ROOT / "data/ioi-selected-test.json")
    pops = read(ROOT / "results/reference/screen-selection.json")["strata"]
    units = ["circuit"] + [f"random{k}" for k in range(N_RANDOM)] + ["empty"]
    score = {(u, p, c): [] for u in units for p in ["mean", "resample"] for c in CELLS}
    picks = {k: [] for k in score}
    for r in rows:
        with np.load(ROOT / "results/reference/test" / f"{r['id']}.npz") as a, \
                np.load(ROOT / "results/posthoc/random-circuits/test" / f"{r['id']}.npz") as b:
            for c in CELLS:
                for u in units:
                    if u == "circuit":
                        cm, dm = float(a[f"{c}__reference__centroid_margin"]), a[f"{c}__reference__donor_margins"]
                    elif u == "empty":
                        cm, dm = float(a[f"{c}__empty__centroid_margin"]), a[f"{c}__empty__donor_margins"]
                    else:
                        k = int(u[6:])
                        cm, dm = float(b[f"{c}__c{k}__centroid_margin"]), b[f"{c}__c{k}__donor_margins"]
                    score[(u, "mean", c)].append(cm)
                    picks[(u, "mean", c)].append(float(cm < 0))
                    score[(u, "resample", c)].append(float(np.mean(dm)))
                    picks[(u, "resample", c)].append(float(np.mean(dm < 0)))
    strata = [f"{r['family']}/{r['stratum_label']}" for r in rows]
    error = np.array([r["stratum_label"] == "pair_error" for r in rows])
    templates = np.array([r["template"] for r in rows])
    n = {s: strata.count(s) for s in set(strata)}
    base_w = np.array([pops[s]["population"] / n[s] for s in strata])
    return rows, strata, error, templates, base_w, {k: np.array(v) for k, v in score.items()}, \
        {k: np.array(v) for k, v in picks.items()}


def weighted_auroc(score, error, w):
    """P(score of an error prompt < score of a correct prompt), pairs weighted by w_i * w_j; w: [B, n] or [n]."""
    w = np.atleast_2d(w)
    e, c = np.where(error)[0], np.where(~error)[0]
    order = c[np.argsort(score[c], kind="stable")]
    sc = score[order]
    cw = np.cumsum(w[:, order], axis=1)                                  # [B, n_c]
    total_c = cw[:, -1]
    lo = np.searchsorted(sc, score[e], side="left")                       # correct prompts strictly below
    hi = np.searchsorted(sc, score[e], side="right")
    below = np.where(lo > 0, cw[:, np.maximum(lo - 1, 0)], 0.0)
    upto = np.where(hi > 0, cw[:, np.maximum(hi - 1, 0)], 0.0)
    above = total_c[:, None] - upto
    ties = upto - below
    num = (w[:, e] * (above + 0.5 * ties)).sum(1)
    return num / (w[:, e].sum(1) * total_c)


def within_template_auroc(score, error, templates, w):
    num, den = 0.0, 0.0
    for t in np.unique(templates):
        m = templates == t
        if error[m].any() and (~error[m]).any():
            pair_w = w[m & error].sum() * w[m & ~error].sum()
            num += pair_w * weighted_auroc(score[m], error[m], w[m])[0]
            den += pair_w
    return num / den


def main():
    rows, strata, error, templates, base_w, score, picks = load()
    rng = np.random.default_rng(SEED)
    # Stratified multinomial bootstrap: counts within each family x correctness stratum.
    counts = np.zeros((DRAWS, len(rows)))
    for s in sorted(set(strata)):
        ix = np.array([i for i, x in enumerate(strata) if x == s])
        counts[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=DRAWS)
    W = counts * base_w[None, :]

    def p_pick(v, w, cls):
        w = np.atleast_2d(w)
        return (w[:, cls] * v[cls]).sum(1) / w[:, cls].sum(1)

    out = {"created_utc": utc(), "posthoc": True, "draws": DRAWS, "seed": SEED, "cells": CELLS, "results": {}}
    for (u, pool, c), s in score.items():
        v = picks[(u, pool, c)]
        pe, pc = p_pick(v, base_w, error)[0], p_pick(v, base_w, ~error)[0]
        be, bc = p_pick(v, W, error), p_pick(v, W, ~error)
        odds = lambda p: p / (1 - p)
        with np.errstate(divide="ignore", invalid="ignore"):
            orr, bor = odds(pe) / odds(pc), odds(be) / odds(bc)
        auc, bauc = weighted_auroc(s, error, base_w)[0], weighted_auroc(s, error, W)
        out["results"][f"{u}/{pool}/{c}"] = {
            "p_subject_given_error": float(pe), "p_subject_given_error_ci95": np.percentile(be, [2.5, 97.5]).tolist(),
            "p_subject_given_correct": float(pc), "p_subject_given_correct_ci95": np.percentile(bc, [2.5, 97.5]).tolist(),
            "odds_ratio": float(orr), "odds_ratio_ci95": np.nanpercentile(bor, [2.5, 97.5]).tolist(),
            "auroc": float(auc), "auroc_ci95": np.percentile(bauc, [2.5, 97.5]).tolist(),
            "auroc_within_template": float(within_template_auroc(s, error, templates, base_w))}
    # Random circuits summarized by the mean over the five circuits, with a bootstrap interval for that mean.
    for pool in ["mean", "resample"]:
        for c in CELLS:
            keys = [f"random{k}/{pool}/{c}" for k in range(N_RANDOM)]
            res = [out["results"][k] for k in keys]
            summary = {}
            for metric in ["p_subject_given_error", "p_subject_given_correct", "odds_ratio", "auroc", "auroc_within_template"]:
                vals = [x[metric] for x in res]
                summary[metric] = {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals))}
            bauc = np.mean([weighted_auroc(score[(f"random{k}", pool, c)], error, W) for k in range(N_RANDOM)], axis=0)
            summary["auroc"]["ci95_mean"] = np.percentile(bauc, [2.5, 97.5]).tolist()
            out["results"][f"random/{pool}/{c}"] = summary
    out["error_prompts"], out["correct_prompts"] = int(error.sum()), int((~error).sum())
    out["templates_with_both_classes"] = int(sum(1 for t in np.unique(templates)
                                                if error[templates == t].any() and (~error[templates == t]).any()))
    dump(ROOT / "results/posthoc/error-discrimination.json", out)
    for pool in ["mean", "resample"]:
        for c in CELLS:
            ci, em, ra = (out["results"][f"{u}/{pool}/{c}"] for u in ["circuit", "empty", "random"])
            print(f"{pool:8s} {c:6s} | circuit pS|e {ci['p_subject_given_error']:.3f} pS|c {ci['p_subject_given_correct']:.3f} "
                  f"OR {ci['odds_ratio']:7.1f} AUC {ci['auroc']:.3f} (within {ci['auroc_within_template']:.3f}) | "
                  f"random OR {ra['odds_ratio']['mean']:5.2f} AUC {ra['auroc']['mean']:.3f} | "
                  f"empty OR {em['odds_ratio']:5.2f} AUC {em['auroc']:.3f} (within {em['auroc_within_template']:.3f})")


if __name__ == "__main__":
    main()

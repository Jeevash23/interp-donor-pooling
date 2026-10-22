"""Post hoc controls added after the readiness review of 12 October 2026 (fifth revision of the manuscript).

Every analysis here was specified after all results were known; frozen files are only read.
  1. Sampled donors at increasing cost (Study 2 sweeps): misclassified sites and mean absolute error of
     k = 1, 2, 4, 8, 16 and 32 donors drawn per recipient group and direction (the same k donors at every
     site), 1,000 draws each, against the 64-donor individual average. Mean ablation costs one evaluation.
  2. Stability of the reference: each group's 64 donors are split at random into two halves of 32, and the
     sites whose flag differs between the two halves' individual averages are counted (2,000 splits).
  3. The token rule as defined ("group donors by the token at the patched position") in the agreement
     sweeps: ADJ8 banks at the adjective, whose donors share every token up to it; one group, which is mean
     ablation, at the head noun and the two following words; NOUN8 banks at the final token.
  4. The IOI sweep count under the original definition (sites flagged by mean ablation only) and without
     the prompts used for development (the first two) and calibration (the first four), each with the
     prespecified 98.33% interval (20,000 stratified group draws with the frozen seed).
  5. Study 1 spread slopes within a grouping policy (the eight rotations of one K) and across K = 2, 4, 8
     only, with the 4,000-draw recipient bootstrap of retrospective_study3.py.
  6. Error separation on the Study 2 Experiment A recipients: the AUROC of the ablated margin with correct
     prompts restricted to those nearest the model's decision boundary; the weighted rank correlation of
     ablated and intact margins within each stratum; and a surface-feature baseline, an additive
     least-squares model of the intact margin on template, name order, the two names, place and object,
     fitted on the screened prompts that were not selected as recipients.
Output: results/posthoc/review-controls.json. If the file already exists, the script recomputes everything
and stops with an error unless the result is identical apart from its creation time; it never overwrites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze_sweep import DRAWS as FROZEN_DRAWS, SEED as FROZEN_SEED, boot_counts, interval, load_panel, normalized  # noqa: E402
from common import OCTOBER, ROOT, dump, read, utc  # noqa: E402
from error_discrimination import load as load_recipients, weighted_auroc, within_template_auroc  # noqa: E402

OUT = ROOT / "results/posthoc/review-controls.json"
TAU = 0.05
PANELS = ["sva-gpt2-small", "sva-qwen3-0.6b", "ioi-gpt2-small"]
COST_K, COST_DRAWS, COST_SEED = [1, 2, 4, 8, 16, 32], 1000, 26101301
SPLITS, SPLIT_SEED = 2000, 26101302
SLOPE_DRAWS, SLOPE_SEED = 4000, 26100533              # as retrospective_study3.py
SEP_QUANTILES, SEP_DRAWS, SEP_SEED = [0.10, 0.25, 0.50], 2000, 26101303
CELLS, POOLS, N_RANDOM = ["N8", "N128", "NPO8", "NPO128"], ["mean", "resample"], 5


def sweep_panel(name):
    P = load_panel(name, "test")
    task, raw = P["task"], P["raw"]
    G, D = raw.shape[0], raw.shape[1]
    c, k = P["clean"].mean(), P["corrupt"].mean()
    live = ~P["structural"]
    e_ind = normalized(task, P["methods"]["individual"].mean(0), c, k)
    f_ind = np.abs(e_ind) >= TAU
    e_mean = normalized(task, P["methods"]["mean64"].mean(0), c, k)
    out = {"groups": G, "sites": int(e_ind.size), "live_sites": int(live.sum()),
           "mean_ablation": {"evaluations": 1, "misclassified": int(((np.abs(e_mean) >= TAU) != f_ind).sum()),
                             "mae": float(np.abs(e_mean - e_ind)[live].mean())}}
    donors = raw[..., :64]

    # 1. k sampled donors per group and direction.
    rng = np.random.default_rng(COST_SEED)
    cost = {}
    for kk in COST_K:
        mis, mae = [], []
        for _ in range(COST_DRAWS):
            pick = np.argsort(rng.random((G, D, 64)), axis=-1)[..., :kk]
            gm = np.take_along_axis(donors, pick[:, :, None, None, :], axis=-1).mean(-1)
            e = normalized(task, gm.mean(0), c, k)
            mis.append(int(((np.abs(e) >= TAU) != f_ind).sum()))
            mae.append(float(np.abs(e - e_ind)[live].mean()))
        cost[str(kk)] = {"evaluations": kk, "misclassified_mean": float(np.mean(mis)),
                         "misclassified_range95": np.percentile(mis, [2.5, 97.5]).tolist(),
                         "misclassified_max": int(np.max(mis)), "mae_mean": float(np.mean(mae))}
    out["sampled_donors"] = cost

    # 2. Two disjoint halves of 32 donors.
    rng = np.random.default_rng(SPLIT_SEED)
    halves, half_full = [], []
    for _ in range(SPLITS):
        perm = np.argsort(rng.random((G, 64)), axis=1)[:, None, None, None, :]
        a = np.take_along_axis(donors, perm[..., :32], axis=-1).mean(-1).mean(0)
        b = np.take_along_axis(donors, perm[..., 32:], axis=-1).mean(-1).mean(0)
        fa = np.abs(normalized(task, a, c, k)) >= TAU
        fb = np.abs(normalized(task, b, c, k)) >= TAU
        halves.append(int((fa != fb).sum()))
        half_full.append(int((fa != f_ind).sum()))
    out["split_half"] = {"splits": SPLITS,
                         "halves_disagree_mean": float(np.mean(halves)), "halves_disagree_max": int(np.max(halves)),
                         "half_vs_all_mean": float(np.mean(half_full)), "half_vs_all_max": int(np.max(half_full))}

    # 3. The token rule as defined (agreement grid only).
    if task == "sva":
        assert P["positions"][0] == "adj" and P["positions"][-1] == "END", P["positions"]
        rule = P["methods"]["mean64"].copy()
        rule[..., 0] = P["methods"]["adj8"][..., 0]
        rule[..., -1] = P["methods"]["noun8"][..., -1]
        e_rule = normalized(task, rule.mean(0), c, k)
        e_noun = normalized(task, P["methods"]["noun8"].mean(0), c, k)
        out["token_rule"] = {
            "rule": {"misclassified": int(((np.abs(e_rule) >= TAU) != f_ind).sum()),
                     "mae": float(np.abs(e_rule - e_ind)[live].mean())},
            "noun8_everywhere": {"misclassified": int(((np.abs(e_noun) >= TAU) != f_ind).sum()),
                                 "mae": float(np.abs(e_noun - e_ind)[live].mean())},
            "max_abs_noun8_minus_mean64_before_final": float(np.abs(e_noun[..., :-1] - e_mean[..., :-1]).max()),
            "max_abs_adj8_minus_individual_at_adjective": float(np.abs(e_rule[..., 0] - e_ind[..., 0]).max())}
    return out, P


def ioi_count(P, keep, fp_only):
    """Primary count and 98.33% interval, as in analyze_sweep.py, for the groups in keep."""
    task = P["task"]
    G = len(keep)
    strata = [P["strata"][i] for i in keep]
    W = boot_counts(strata, np.random.default_rng(FROZEN_SEED))
    ind, mean = P["methods"]["individual"][keep], P["methods"]["mean64"][keep]
    clean, corrupt = P["clean"][keep], P["corrupt"][keep]

    def flags(weights, arr):
        gm = (weights @ arr.reshape(G, -1)).reshape(weights.shape[:-1] + arr.shape[1:])
        return np.abs(normalized(task, gm, weights @ clean, weights @ corrupt)) >= TAU
    pw = np.full(G, 1 / G)
    fi, fm = flags(pw, ind), flags(pw, mean)
    bi, bm = flags(W, ind), flags(W, mean)
    if fp_only:
        est, boot = int((fm & ~fi).sum()), (bm & ~bi).reshape(len(W), -1).sum(1)
    else:
        est, boot = int((fm != fi).sum()), (bm != bi).reshape(len(W), -1).sum(1)
    return {"groups": G, "estimate": est, "ci": interval(boot, 0.05 / 3), "confidence": 1 - 0.05 / 3,
            "draws": int(len(W)), "seed": FROZEN_SEED}


def ioi_robustness(P):
    ids = [r["id"] for r in P["rows"]]
    assert ids[:4] == [f"sweep-ioi-{i:05d}" for i in range(4)], ids[:4]
    allg = np.arange(len(ids))
    return {"as_published": ioi_count(P, allg, False),
            "original_definition": ioi_count(P, allg, True),
            "without_development_prompts": ioi_count(P, allg[2:], False),
            "without_development_prompts_original_definition": ioi_count(P, allg[2:], True),
            "without_calibration_prompts": ioi_count(P, allg[4:], False)}


def fe_slope(lg, lr, cell_axes, seed):
    yc = lg - lg.mean(axis=cell_axes, keepdims=True)
    xc = lr - lr.mean(axis=cell_axes, keepdims=True)
    num = (yc * xc).reshape(len(yc), -1).sum(1)
    den = (xc ** 2).reshape(len(xc), -1).sum(1)
    b = float(num.sum() / den.sum())
    draws = np.random.default_rng(seed).integers(0, len(num), size=(SLOPE_DRAWS, len(num)))
    return {"slope": b, "ci95": np.percentile(num[draws].sum(1) / den[draws].sum(1), [2.5, 97.5]).tolist(),
            "r2": float(1 - ((yc - b * xc) ** 2).sum() / (yc ** 2).sum()),
            "sd_log_spread_within_cells": float(np.sqrt((xc ** 2).mean()))}


def spread_within_policy():
    out = {}
    for model in ["gpt2-small", "qwen3-0.6b"]:
        z = np.load(OCTOBER / f"token/results/analysis-test/{model}-components.npz")
        gap = np.take(z["gap"], [1, 3, 5], axis=1)[..., 0]           # [recipients, 3 sites, 2 directions, 4 K, 8 rotations]
        msr = np.take(z["dispersion"], [1, 3, 5], axis=1)[..., 0]
        lg, lr = np.log(np.abs(gap) + 1e-12), np.log(msr)
        r = {"all_banks": fe_slope(lg, lr, (3, 4), SLOPE_SEED),
             "within_policy": fe_slope(lg, lr, (4,), SLOPE_SEED),
             "k_2_4_8_only": fe_slope(lg[:, :, :, 1:], lr[:, :, :, 1:], (3, 4), SLOPE_SEED)}
        for ki, kk in enumerate([1, 2, 4, 8]):
            r[f"within_policy_k{kk}"] = fe_slope(lg[:, :, :, ki:ki + 1], lr[:, :, :, ki:ki + 1], (3, 4), SLOPE_SEED)
        out[model] = r
    return out


def wquantile(x, w, q):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o]) / w[o].sum()
    return float(x[o][np.searchsorted(cw, q)])


def wrankcorr(a, b, w):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    ma, mb = np.average(ra, weights=w), np.average(rb, weights=w)
    cov = np.average((ra - ma) * (rb - mb), weights=w)
    return float(cov / np.sqrt(np.average((ra - ma) ** 2, weights=w) * np.average((rb - mb) ** 2, weights=w)))


FEATURES = {"template": "template", "order": "order", "io": "target", "s": "subject", "place": "place", "object": "object"}


def surface_design(rows, levels, feats):
    cols = [np.ones(len(rows))]
    for f in feats:
        for lev in levels[f][1:]:
            cols.append(np.array([r[FEATURES[f]] == lev for r in rows], float))
    return np.stack(cols, 1)


def separation():
    rows, strata, error, templates, base_w, score, picks = load_recipients()
    full = np.array([r["screen_margin"] for r in rows])
    assert ((full <= 0) == error).all()
    G = len(rows)
    counts = np.zeros((SEP_DRAWS, G))
    rng = np.random.default_rng(SEP_SEED)
    for s in sorted(set(strata)):
        ix = np.array([i for i, x in enumerate(strata) if x == s])
        counts[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=SEP_DRAWS)
    W = counts * base_w[None, :]
    cut = {str(q): wquantile(full[~error], base_w[~error], q) for q in SEP_QUANTILES}
    near = {}
    for unit in ["circuit", "empty"] + [f"random{k}" for k in range(N_RANDOM)]:
        for pool in POOLS:
            for cell in CELLS:
                s = score[(unit, pool, cell)]
                rec = {"auroc_all": float(weighted_auroc(s, error, base_w)[0])}
                for q in SEP_QUANTILES:
                    keep = error | (full <= cut[str(q)])
                    rec[f"auroc_below_q{int(q * 100)}"] = float(weighted_auroc(s[keep], error[keep], base_w[keep])[0])
                    if q == 0.10 and unit in ("circuit", "empty"):
                        rec["auroc_below_q10_ci95"] = np.percentile(weighted_auroc(s[keep], error[keep], W[:, keep]), [2.5, 97.5]).tolist()
                rec["rank_corr_with_intact_errors"] = wrankcorr(s[error], full[error], base_w[error])
                rec["rank_corr_with_intact_correct"] = wrankcorr(s[~error], full[~error], base_w[~error])
                near[f"{unit}/{pool}/{cell}"] = rec
    # Surface-feature baseline.
    screen = read(ROOT / "data/ioi-screen.json")
    z = np.load(ROOT / "results/reference/screen.npz")
    margin = dict(zip(z["ids"].tolist(), z["margin"].tolist()))
    selected = {r["id"] for r in rows}
    train = [r for r in screen if r["id"] not in selected]
    base = {}
    for name, feats in [("template_order", ["template", "order"]), ("names", ["io", "s"]),
                        ("template_order_names", ["template", "order", "io", "s"]),
                        ("all_surface_features", list(FEATURES))]:
        levels = {f: sorted({r[FEATURES[f]] for r in screen}) for f in feats}
        X, y = surface_design(train, levels, feats), np.array([margin[r["id"]] for r in train])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = surface_design(rows, levels, feats) @ beta
        base[name] = {"features": feats, "parameters": int(X.shape[1]), "train_prompts": len(train),
                      "train_r2": float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                      "auroc": float(weighted_auroc(pred, error, base_w)[0]),
                      "auroc_ci95": np.percentile(weighted_auroc(pred, error, W), [2.5, 97.5]).tolist(),
                      "auroc_within_template": float(within_template_auroc(pred, error, templates, base_w))}
    return {"recipients": G, "errors": int(error.sum()), "intact_margin_cutoffs": cut, "draws": SEP_DRAWS,
            "seed": SEP_SEED, "near_boundary": near, "surface_baseline": base}


def compute():
    out = {"posthoc": True, "tau": TAU, "cost_draws": COST_DRAWS, "cost_seed": COST_SEED,
           "split_seed": SPLIT_SEED, "panels": {}}
    for name in PANELS:
        res, P = sweep_panel(name)
        if name == "ioi-gpt2-small":
            res["ioi_count"] = ioi_robustness(P)
        out["panels"][name] = res
    out["frozen_bootstrap"] = {"draws": FROZEN_DRAWS, "seed": FROZEN_SEED}
    out["spread_within_policy"] = spread_within_policy()
    out["separation"] = separation()
    return out


def main():
    out = compute()
    if OUT.exists():
        saved = read(OUT)
        saved.pop("created_utc", None)
        if json.loads(json.dumps(out, sort_keys=True)) != json.loads(json.dumps(saved, sort_keys=True)):
            sys.exit(f"{OUT.relative_to(ROOT)} exists and differs from a fresh computation; not overwritten")
        print(f"{OUT.relative_to(ROOT)} reproduced exactly (apart from its creation time); not rewritten")
        return
    dump(OUT, {"created_utc": utc(), **out})
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

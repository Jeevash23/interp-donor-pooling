"""Post hoc re-analysis of the saved Study 3 (October) outputs.

These are the audit analyses that motivated Study 4. They were written after the
Study 3 results were known, read only saved arrays, and write one summary file:
results/retrospective-study3.json.
"""
from __future__ import annotations

import collections

import numpy as np

from common import OCTOBER, ROOT, dump, read, utc

END = [1, 3, 5]              # nonterminal END sites in the Study 3 component arrays
K = [1, 2, 4, 8]


def fe_slope(y, x, cell_axes):
    yc = y - y.mean(axis=cell_axes, keepdims=True)
    xc = x - x.mean(axis=cell_axes, keepdims=True)
    b = float((yc * xc).sum() / (xc ** 2).sum())
    r2 = float(1 - ((yc - b * xc) ** 2).sum() / (yc ** 2).sum())
    return b, r2, yc, xc


def grouping(model):
    z = np.load(OCTOBER / f"token/results/analysis-test/{model}-components.npz")
    s = read(OCTOBER / f"token/results/analysis-test/{model}.json")
    comp = s["competence"]
    gap = np.take(z["gap"], END, axis=1)[..., 0]        # [64 groups, 3 sites, 2 directions, 4 K, 8 rotations]
    msr = np.take(z["dispersion"], END, axis=1)[..., 0]
    lg, lr = np.log(np.abs(gap) + 1e-12), np.log(msr)
    slope, r2, yc, xc = fe_slope(lg, lr, (3, 4))
    # Compare spread alone, K labels alone, and both, all with the same cell fixed effects.
    y = yc.reshape(-1)
    dummies = []
    for ki in range(1, 4):
        d = np.zeros_like(lg)
        d[:, :, :, ki] = 1
        dummies.append((d - d.mean(axis=(3, 4), keepdims=True)).reshape(-1))
    def fit(cols):
        X = np.stack(cols, 1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(1 - ((y - X @ beta) ** 2).sum() / (y ** 2).sum())
    r2_k = fit(dummies)
    r2_both = fit([xc.reshape(-1)] + dummies)
    # Recipient bootstrap for the slope (cells stay within recipients).
    rng = np.random.default_rng(26100533)
    num = (yc * xc).reshape(len(yc), -1).sum(1)
    den = (xc ** 2).reshape(len(xc), -1).sum(1)
    draws = rng.integers(0, len(num), size=(4000, len(num)))
    slope_ci = np.percentile(num[draws].sum(1) / den[draws].sum(1), [2.5, 97.5]).tolist()
    # Centroid bias relative to the individual-patch effect, by site and direction.
    clean, corrupt = comp["recipient_clean_margin"]["mean"], comp["recipient_corrupt_clean_target_margin"]["mean"]
    full = clean - corrupt
    rows = []
    for di, (direction, base) in enumerate([("restore", corrupt), ("destroy", clean)]):
        for si in range(3):
            indiv = float(z["individual"][:, END[si], di, :, :, 0].mean())
            effect = indiv - base
            for ki, k in enumerate(K):
                bias = float(z["mean"][:, END[si], di, ki, :, 0].mean()) - indiv
                rows.append({"direction": direction, "depth_index": si + 1, "k": k, "individual_effect": effect,
                             "centroid_bias": bias, "bias_over_effect": bias / abs(effect),
                             "bias_over_clean_minus_corrupt": bias / full})
    curve = {str(k): {"mean_abs_gap": s["secondary_dose_curve"][str(k)]["margin_gap_absolute"]["mean"],
                      "mean_squared_radius": s["secondary_dose_curve"][str(k)]["mean_squared_radius"]["mean"]} for k in K}
    return {"loglog_slope_gap_on_spread": slope, "loglog_slope_ci95": slope_ci, "r2_spread": r2, "r2_k_labels": r2_k, "r2_spread_and_k": r2_both,
            "clean_margin": clean, "corrupt_margin": corrupt, "clean_minus_corrupt": full,
            "bias_by_site": rows, "dose_curve": curve,
            "max_bias_over_effect_k8": max(abs(r["bias_over_effect"]) for r in rows if r["k"] == 8),
            "max_bias_over_clean_minus_corrupt_k8": max(abs(r["bias_over_clean_minus_corrupt"]) for r in rows if r["k"] == 8)}


def circuit():
    a = read(OCTOBER / "circuit/results/analysis.json")
    out = {}
    for c in ["reference", "empty_attention"]:
        st = a["circuits"][c]["strata"]
        for s in ["full_pair_error", "full_pair_success"]:
            out[f"{c}/{s}"] = {"mean_agreement": st[s]["mean_pair_agreement"]["mean"],
                               "resample_agreement": st[s]["expected_resample_pair_agreement"]["mean"], "n": st[s]["n"]}
    # Error drivers in the 8,192-prompt screen: template and a held-out name-pair score.
    rows = read(OCTOBER / "circuit/data/screen.json")
    margin = np.load(OCTOBER / "circuit/results/screen.npz")["metrics"][:, 0]
    err = margin <= 0
    tpl = np.array([r["template"] for r in rows])
    by_template = {str(t): {"error_rate": float(err[tpl == t].mean()), "errors": int(err[tpl == t].sum())} for t in range(4)}
    score = np.zeros(len(rows))
    for t in range(4):
        io, sb = collections.defaultdict(list), collections.defaultdict(list)
        for r, m, keep in zip(rows, margin, tpl != t):
            if keep:
                io[r["target"]].append(m)
                sb[r["subject"]].append(m)
        val = {n: (np.mean(io[n]) if io[n] else 0) - (np.mean(sb[n]) if sb[n] else 0) for n in set(io) | set(sb)}
        for i, r in enumerate(rows):
            if tpl[i] == t:
                score[i] = val.get(r["target"], 0) - val.get(r["subject"], 0)
    q = np.quantile(score, [0, 0.1, 0.9, 1])
    out["screen"] = {"prompts": len(rows), "errors": int(err.sum()), "by_template": by_template,
                     "error_rate_lowest_decile_name_score": float(err[score <= q[1]].mean()),
                     "error_rate_highest_decile_name_score": float(err[score >= q[2]].mean())}
    return out


def main():
    result = {"created_utc": utc(), "source": str(OCTOBER.relative_to(ROOT.parents[1])),
              "note": "post hoc audit analysis of saved Study 3 outputs",
              "grouping": {m: grouping(m) for m in ["gpt2-small", "qwen3-0.6b"]}, "circuit": circuit()}
    dump(ROOT / "results/retrospective-study3.json", result)
    for m, g in result["grouping"].items():
        print(m, {k: round(v, 3) for k, v in g.items() if isinstance(v, float)})
    print(result["circuit"])


if __name__ == "__main__":
    main()

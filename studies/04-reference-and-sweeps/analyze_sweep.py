"""Analysis for Experiment B (frozen with the protocol).

Normalized effects use ratios of group-mean margins, so a group with a small
clean-minus-corrupt difference cannot dominate. A site is "flagged" when the
absolute normalized effect is at least tau. The target estimand for every site is
the average of individual donor patches ("individual"); a method's false positives
are sites it flags that the individual average does not.
"""
from __future__ import annotations

import argparse

import numpy as np

from common import ROOT, dump, read, utc
from run_sweep import ADJ, MIX, NOUN

DRAWS, SEED = 20000, 26100532
TAUS = [0.05, 0.10]
PANELS = {"sva-gpt2-small": ("sva", "gpt2-small", 64), "sva-qwen3-0.6b": ("sva", "qwen3-0.6b", 32),
          "ioi-gpt2-small": ("ioi", "gpt2-small", 64)}
IOI_LABELS = ["prefix", "N1", "and", "N2", "went", "to", "the", "PLACE", "punct", "S2", "gave", "a", "OBJ", "END"]


def bank_indices(partition):
    return [[s * 8 + n for s, n in bank] for bank in partition]


def load_panel(panel, stage):
    task, model, n = PANELS[panel]
    folder = ROOT / "results/sweep" / f"{task}-{model}" / stage
    if task == "sva":
        rows = read(ROOT / f"data/sva-{'development' if stage == 'development' else 'test'}.json")
    else:
        rows = read(ROOT / "data/ioi-sweep.json")
    rows = [r for r in rows if (folder / f"{r['id']}.npz").exists()]
    if stage == "test":
        rows = rows[:n]
        if len(rows) != n:   # Amendment 3
            raise RuntimeError(f"{panel}: expected {n} groups, found {len(rows)}")
    data = [np.load(folder / f"{r['id']}.npz") for r in rows]
    if task == "sva":
        m = np.stack([d["margins"] for d in data])                     # [G, 2, L, P, 89]
        methods = {"individual": m[..., :64].mean(-1), "mix8": m[..., 64:72].mean(-1), "noun8": m[..., 72:80].mean(-1),
                   "adj8": m[..., 80:88].mean(-1), "mean64": m[..., 88]}
        clean = np.array([d["baseline"][0] for d in data])
        corrupt = np.array([d["baseline"][1] for d in data])
        strata = [f"{r['template']}-{r['class']}" for r in rows]
        positions = [str(x) for x in data[0]["position_names"]]
        positions[-1] = "END"
        spread = np.stack([d["spread"] for d in data])                  # [G, 2, L, P, 25]
        extra = {"raw": m, "spread": spread}
    else:
        t = 14
        ms, sp = [], []
        for r, d in zip(rows, data):
            off = d["margins"].shape[1] - t                              # right-align the two templates
            ms.append(d["margins"][:, off:])
            sp.append(d["spread"][:, off:])
        m = np.stack(ms)[:, None]                                       # [G, 1, L, T, 73]
        methods = {"individual": m[..., :64].mean(-1), "mix8": m[..., 64:72].mean(-1), "mean64": m[..., 72]}
        clean = np.array([d["baseline"][0] for d in data])
        corrupt = np.array([d["donor_margins"].mean() for d in data])
        strata = [f"{r['template']}-{r['order']}" for r in rows]
        positions = IOI_LABELS
        extra = {"raw": m, "spread": np.stack(sp)[:, None]}
    # Amendment 4: sites where no patch changes any margin in any group (structural zeros).
    if task == "sva":
        base = np.stack([corrupt, clean], 1)[:, :, None, None, None]
    else:
        base = clean[:, None, None, None, None]
    structural = (np.abs(m - base).max(axis=(0, -1)) <= 1e-4)
    return {"task": task, "model": model, "rows": rows, "methods": methods, "clean": clean, "corrupt": corrupt,
            "structural": structural,
            "strata": strata, "positions": positions, **extra}


def normalized(task, means, clean, corrupt):
    """means: [..., D, L, P] group-averaged margins; clean/corrupt: matching group averages."""
    c = np.asarray(clean)[..., None, None]
    k = np.asarray(corrupt)[..., None, None]
    denom = c - k
    e = np.empty_like(means)
    if task == "sva":
        e[..., 0, :, :] = (means[..., 0, :, :] - k) / denom      # restore: share of the gap recovered
        e[..., 1, :, :] = (c - means[..., 1, :, :]) / denom      # destroy: share of the gap removed
    else:
        e[..., 0, :, :] = (c - means[..., 0, :, :]) / denom      # noising
    return e


def boot_counts(strata, rng):
    keys = sorted(set(strata))
    G = len(strata)
    W = np.zeros((DRAWS, G))
    for k in keys:
        ix = np.array([i for i, s in enumerate(strata) if s == k])
        c = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=DRAWS)
        W[:, ix] = c / G
    return W                                                           # rows sum to 1


def interval(x, alpha=0.05):
    lo, hi = np.nanpercentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def analyze_panel(panel, stage, alpha_primary):
    P = load_panel(panel, stage)
    task, G = P["task"], len(P["rows"])
    rng = np.random.default_rng(SEED)
    W = boot_counts(P["strata"], rng)
    point_w = np.full(G, 1 / G)

    def maps(weights):
        clean = weights @ P["clean"]
        corrupt = weights @ P["corrupt"]
        out = {}
        for name, arr in P["methods"].items():
            flat = arr.reshape(G, -1)
            means = (weights @ flat).reshape(weights.shape[:-1] + arr.shape[1:])
            out[name] = normalized(task, means, clean, corrupt)
        return out

    point = maps(point_w)
    boot = maps(W)
    res = {"groups": G, "positions": P["positions"], "denominator": float(P["clean"].mean() - P["corrupt"].mean()),
           "clean_margin": float(P["clean"].mean()), "corrupt_margin": float(P["corrupt"].mean()),
           "maps": {k: v.tolist() for k, v in point.items()}, "flags": {}, "bias": {}}
    others = [m for m in P["methods"] if m != "individual"]
    for tau in TAUS:
        fr_p, fr_b = np.abs(point["individual"]) >= tau, np.abs(boot["individual"]) >= tau
        for m in others:
            fm_p, fm_b = np.abs(point[m]) >= tau, np.abs(boot[m]) >= tau
            fp_p, fp_b = int((fm_p & ~fr_p).sum()), (fm_b & ~fr_b).reshape(DRAWS, -1).sum(1)
            fn_p, fn_b = int((~fm_p & fr_p).sum()), (~fm_b & fr_b).reshape(DRAWS, -1).sum(1)
            nm_p, nm_b = int(fm_p.sum()), fm_b.reshape(DRAWS, -1).sum(1)
            a = alpha_primary if (m == "mean64" and tau == 0.05) else 0.05
            res["flags"][f"{m}/tau{tau}"] = {
                "misclassified_sites": {"estimate": fp_p + fn_p, "ci": interval(fp_b + fn_b, a), "confidence": 1 - a},
                "false_positive_sites": {"estimate": fp_p, "ci": interval(fp_b)},
                "false_negative_sites": {"estimate": fn_p, "ci": interval(fn_b)},
                "flagged_by_method": {"estimate": nm_p, "ci": interval(nm_b)},
                "flagged_by_individual": {"estimate": int(fr_p.sum()), "ci": interval(fr_b.reshape(DRAWS, -1).sum(1))},
                "false_discovery_proportion": {"estimate": fp_p / nm_p if nm_p else None,
                                               "ci": interval(np.where(nm_b > 0, fp_b / np.maximum(nm_b, 1), np.nan))},
                "sites": int(fr_p.size),
                "false_positive_list": [[int(x) for x in ix] for ix in np.argwhere(fm_p & ~fr_p)],
                "false_negative_list": [[int(x) for x in ix] for ix in np.argwhere(~fm_p & fr_p)]}
    # Bias relative to the individual average, by position, and its size relative to the effect.
    for m in others:
        b_p, b_b = point[m] - point["individual"], boot[m] - boot["individual"]
        byp = {}
        for pi, name in enumerate(P["positions"]):
            byp[name] = {"mean_abs_bias": {"estimate": float(np.abs(b_p[..., pi]).mean()),
                                           "ci": interval(np.abs(b_b[..., pi]).reshape(DRAWS, -1).mean(1))},
                         "max_abs_bias": float(np.abs(b_p[..., pi]).max())}
        live = ~P["structural"]
        weak = (np.abs(point["individual"]) < 0.05) & live
        res["bias"][m] = {"by_position": byp, "max_abs_bias": float(np.abs(b_p).max()),
                          "mean_abs_bias_all_sites": {"estimate": float(np.abs(b_p).mean()),
                                                      "ci": interval(np.abs(b_b).reshape(DRAWS, -1).mean(1))},
                          "mean_abs_bias_weak_sites": float(np.abs(b_p[weak]).mean()) if weak.any() else None,
                          "mean_abs_bias_strong_sites": float(np.abs(b_p[~weak & live]).mean()) if (~weak & live).any() else None,
                          "mean_abs_bias_live_sites": float(np.abs(b_p[live]).mean())}
    # Rank agreement of site importance.
    from scipy.stats import spearmanr
    live = ~P["structural"]
    res["structural_zero_sites"] = int(P["structural"].sum())
    res["spearman_abs_effect"] = {m: float(spearmanr(np.abs(point[m])[live], np.abs(point["individual"])[live]).statistic)
                                  for m in others}
    # Bank-level spread law (secondary): log |centroid gap| on log within-bank mean squared radius,
    # with a fixed effect for every group x direction x layer x position cell.
    raw, spread = P["raw"], P["spread"]
    if task == "sva":
        banks = bank_indices(MIX) + bank_indices(NOUN) + bank_indices(ADJ)
        cent = raw[..., 64:88]
    else:
        banks = [list(range(8 * b, 8 * b + 8)) for b in range(8)]
        cent = raw[..., 64:72]
    indiv = np.stack([raw[..., ix].mean(-1) for ix in banks], -1)
    gap = np.abs(cent - indiv)
    sp = spread[..., :len(banks)]
    ok = (sp > 1e-6) & (gap > 1e-9)
    lg, ls = np.where(ok, np.log(np.maximum(gap, 1e-30)), np.nan), np.where(ok, np.log(np.maximum(sp, 1e-30)), np.nan)
    cnt = ok.sum(-1, keepdims=True)
    use = (cnt >= 3)
    lgc = lg - np.nanmean(np.where(use, lg, np.nan), -1, keepdims=True)
    lsc = ls - np.nanmean(np.where(use, ls, np.nan), -1, keepdims=True)
    mask = ok & use
    slope = float(np.nansum((lgc * lsc)[mask]) / np.nansum((lsc ** 2)[mask]))
    r2 = float(1 - np.nansum(((lgc - slope * lsc) ** 2)[mask]) / np.nansum((lgc ** 2)[mask]))
    res["spread_law"] = {"loglog_slope": slope, "within_cell_r2": r2, "banks_used": int(mask.sum())}
    if task == "sva":
        endi = len(P["positions"]) - 1
        g_end = gap[..., endi, :]
        res["end_gap_by_partition"] = {name: float(g_end[..., sl].mean()) for name, sl in
                                       [("mix8", slice(0, 8)), ("noun8", slice(8, 16)), ("adj8", slice(16, 24))]}
        res["end_spread_by_partition"] = {name: float(sp[..., endi, sl].mean()) for name, sl in
                                          [("mix8", slice(0, 8)), ("noun8", slice(8, 16)), ("adj8", slice(16, 24))]}
    return res


def analyze(stage):
    out = {"created_utc": utc(), "stage": stage, "draws": DRAWS, "seed": SEED, "taus": TAUS, "panels": {}}
    for panel in PANELS:
        task, model, _ = PANELS[panel]
        if not (ROOT / "results/sweep" / f"{task}-{model}" / stage).exists():
            continue
        out["panels"][panel] = analyze_panel(panel, stage, alpha_primary=0.05 / 3)
    out["primary"] = {f"B_{panel}_mean64_misclassified_sites_tau0.05": p["flags"]["mean64/tau0.05"]["misclassified_sites"]
                      for panel, p in out["panels"].items()}
    dump(ROOT / f"results/sweep/analysis-{stage}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["development", "test"])
    a = analyze(ap.parse_args().stage)
    for k, v in a["primary"].items():
        print(k, v)
    for panel, p in a["panels"].items():
        print(panel, "groups", p["groups"], "spread law", p["spread_law"], "spearman", p["spearman_abs_effect"])

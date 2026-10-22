"""Post hoc analyses of the saved Experiment B sweep outputs (Study 2), requested after review.

Every analysis here was specified after the sweep results were known; frozen files are only read.
  1. Qwen3-0.6B task competence on the sweep groups and misclassified sites restricted to groups
     where it prefers the grammatical verb.
  2. Equal-budget comparison with the 64-donor individual average: full centroid, MIX8, NOUN8, ADJ8
     and eight sampled donors (1,000 draws).
  3. Group sizes under "group donors by the token at the patched position" in the IOI sweep.
  4. Threshold robustness: counts for tau = 0.02..0.20, top-k overlap, distance of each
     misclassified site from the threshold.
  5. Cancellation at disputed sites: mean absolute per-donor effect and the share of donors whose
     effect has the opposite sign to the average, against non-disputed weak sites.
  6. Donor spread at misclassified sites (within-layer percentile of the 64-donor spread).
Output: results/posthoc/sweep-posthoc.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze_sweep import ADJ, MIX, NOUN, load_panel, normalized  # noqa: E402
from common import ROOT, dump, read, utc  # noqa: E402

TAU = 0.05
TAUS = [round(0.02 + 0.01 * i, 2) for i in range(19)]
DRAWS, SEED = 1000, 26100651


def bank_cols(partition):
    return [[s * 8 + n for s, n in bank] for bank in partition]


def maps(P, idx=None):
    """Normalized maps for every method over the groups in idx (default all)."""
    idx = np.arange(len(P["rows"])) if idx is None else np.asarray(idx)
    c, k = P["clean"][idx].mean(), P["corrupt"][idx].mean()
    return {m: normalized(P["task"], a[idx].mean(0), c, k) for m, a in P["methods"].items()}


def counts(e, e_ref, tau=TAU, live=None):
    f, fr = np.abs(e) >= tau, np.abs(e_ref) >= tau
    if live is not None:
        f, fr = f & live, fr & live
    return int((f != fr).sum()), int((f & ~fr).sum()), int((~f & fr).sum())


def per_donor_effects(P):
    """Normalized per-donor effects [G, D, L, P, 64] using the aggregate denominator."""
    raw = P["raw"][..., :64]
    c, k = P["clean"].mean(), P["corrupt"].mean()
    denom = c - k
    if P["task"] == "sva":
        base = np.stack([P["corrupt"], P["clean"]], 1)[:, :, None, None, None]
        delta = raw - base
        delta[:, 1] *= -1                       # destruction: clean minus patched
    else:
        delta = P["clean"][:, None, None, None, None] - raw
    return delta / denom


def panel(name):
    P = load_panel(name, "test")
    rng = np.random.default_rng(SEED)
    raw = P["raw"]
    G = raw.shape[0]
    m = maps(P)
    e_ind, e_mean = m["individual"], m["mean64"]
    live = ~(np.abs(raw - (np.stack([P["corrupt"], P["clean"]], 1)[:, :, None, None, None] if P["task"] == "sva"
                           else P["clean"][:, None, None, None, None])).max(axis=(0, -1)) <= 1e-4)
    out = {"groups": G, "sites": int(e_ind.size), "live_sites": int(live.sum())}

    # 1. Competence and restricted recount.
    acc = float(np.mean(P["clean"] > 0)) if P["task"] == "sva" else float(np.mean(P["clean"] > 0))
    keep = np.where(P["clean"] > 0)[0]
    mr = maps(P, keep)
    out["clean_accuracy"] = acc
    out["groups_correct"] = int(len(keep))
    out["misclassified_all_groups"] = counts(e_mean, e_ind)[0]
    out["misclassified_correct_groups_only"] = counts(mr["mean64"], mr["individual"])[0]

    # 2. Equal-budget comparison (MAE over live sites).
    budget = {"mean64": 1, "mix8": 8, "noun8": 8, "adj8": 8}
    eq = {}
    for meth, ev in budget.items():
        if meth in m:
            mis, fp, fn = counts(m[meth], e_ind)
            eq[meth] = {"evaluations": ev, "misclassified": mis, "false_positive": fp, "false_negative": fn,
                        "mae": float(np.abs(m[meth] - e_ind)[live].mean())}
    ind = raw[..., :64]
    D = ind.shape[1]
    c, k = P["clean"].mean(), P["corrupt"].mean()
    s_mis, s_mae, s_fp, s_fn = [], [], [], []
    for _ in range(DRAWS):
        sel = np.argsort(rng.random((G, D, 64)), axis=-1)[..., :8]
        gm = np.take_along_axis(ind, sel[:, :, None, None, :], axis=-1).mean(-1)
        e = normalized(P["task"], gm.mean(0), c, k)
        mis, fp, fn = counts(e, e_ind)
        s_mis.append(mis), s_fp.append(fp), s_fn.append(fn)
        s_mae.append(float(np.abs(e - e_ind)[live].mean()))
    eq["sample8"] = {"evaluations": 8, "misclassified_mean": float(np.mean(s_mis)),
                     "misclassified_range95": np.percentile(s_mis, [2.5, 97.5]).tolist(),
                     "false_positive_mean": float(np.mean(s_fp)), "false_negative_mean": float(np.mean(s_fn)),
                     "mae_mean": float(np.mean(s_mae)), "mae_range95": np.percentile(s_mae, [2.5, 97.5]).tolist()}
    eq["individual"] = {"evaluations": 64}
    out["equal_budget"] = eq

    # 4. Threshold robustness.
    out["misclassified_by_tau"] = {str(t): counts(e_mean, e_ind, t)[0] for t in TAUS}
    kflag = int((np.abs(e_ind) >= TAU).sum())
    top_ind = set(np.argsort(-np.abs(e_ind).ravel())[:kflag])
    top_mean = set(np.argsort(-np.abs(e_mean).ravel())[:kflag])
    out["topk"] = {"k": kflag, "overlap": len(top_ind & top_mean), "share": len(top_ind & top_mean) / kflag}
    disputed = (np.abs(e_mean) >= TAU) != (np.abs(e_ind) >= TAU)
    sites = []
    for d, l, p in np.argwhere(disputed):
        sites.append({"direction": int(d), "layer": int(l), "position": P["positions"][p],
                      "individual": float(e_ind[d, l, p]), "mean": float(e_mean[d, l, p]),
                      "distance_individual": float(abs(e_ind[d, l, p]) - TAU),
                      "distance_mean": float(abs(e_mean[d, l, p]) - TAU)})
    out["disputed_sites"] = sites
    out["distance_summary"] = {
        "individual_median": float(np.median([abs(s["distance_individual"]) for s in sites])) if sites else None,
        "mean_median": float(np.median([abs(s["distance_mean"]) for s in sites])) if sites else None,
        "individual_max": float(max(abs(s["distance_individual"]) for s in sites)) if sites else None,
        "mean_max": float(max(abs(s["distance_mean"]) for s in sites)) if sites else None}

    # 5. Cancellation.
    pd = per_donor_effects(P)                                   # [G, D, L, P, 64]
    mabs = np.abs(pd).mean(axis=(0, -1))
    avg = pd.mean(axis=(0, -1))
    opposite = (np.sign(pd) != np.sign(avg)[None, :, :, :, None]).mean(axis=(0, -1))
    weak = (np.abs(e_ind) < TAU) & live & ~disputed
    def summ(mask):
        return {"sites": int(mask.sum()), "mean_abs_per_donor_effect": float(mabs[mask].mean()) if mask.any() else None,
                "abs_signed_average": float(np.abs(avg[mask]).mean()) if mask.any() else None,
                "share_opposite_sign": float(opposite[mask].mean()) if mask.any() else None}
    out["cancellation"] = {"disputed": summ(disputed), "other_weak": summ(weak),
                           "disputed_sites": [{"mean_abs_per_donor_effect": float(mabs[d, l, p]),
                                               "signed_average": float(avg[d, l, p]),
                                               "share_opposite_sign": float(opposite[d, l, p])}
                                              for d, l, p in np.argwhere(disputed)]}

    # 6. Spread percentile within layer (64-donor spread, averaged over groups and directions).
    spread = P["spread"][..., -1].mean(axis=0)                  # [D, L, P]
    pct_rows = []
    for d, l, p in np.argwhere(disputed):
        layer_vals = spread[:, l, :][live[:, l, :]]
        pct_rows.append(float((layer_vals < spread[d, l, p]).mean()))
    others = []
    for d, l, p in np.argwhere(live & ~disputed):
        layer_vals = spread[:, l, :][live[:, l, :]]
        others.append(float((layer_vals < spread[d, l, p]).mean()))
    eff_pct = [float((np.abs(e_ind[live]) < abs(e_ind[d, l, p])).mean()) for d, l, p in np.argwhere(disputed)]
    out["spread_percentile_within_layer"] = {"disputed_median": float(np.median(pct_rows)) if pct_rows else None,
                                             "disputed_values": pct_rows, "other_live_median": float(np.median(others)),
                                             "disputed_effect_percentile_median": float(np.median(eff_pct)) if eff_pct else None}
    return out


def ioi_group_sizes():
    """Group sizes under the token rule at each aligned IOI sweep position."""
    rows = read(ROOT / "data/ioi-sweep.json")
    labels = ["prefix", "N1", "and", "N2", "went", "to", "the", "PLACE", "punct", "S2", "gave", "a", "OBJ", "END"]
    per = {lab: [] for lab in labels}
    for r in rows:
        off = len(r["recipient"]["ids"]) - 14
        for q, lab in enumerate(labels):
            toks = [d["ids"][off + q] for d in r["donors"]]
            _, sizes = np.unique(toks, return_counts=True)
            per[lab].append((len(sizes), float(sizes.mean()), int(sizes.max())))
    return {lab: {"groups_mean": float(np.mean([x[0] for x in v])), "groups_min": int(min(x[0] for x in v)),
                  "groups_max": int(max(x[0] for x in v)), "mean_group_size": float(np.mean([x[1] for x in v])),
                  "largest_group": int(max(x[2] for x in v))} for lab, v in per.items()}


def main():
    out = {"created_utc": utc(), "posthoc": True, "tau": TAU, "draws": DRAWS, "seed": SEED, "panels": {}}
    for name in ["sva-gpt2-small", "sva-qwen3-0.6b", "ioi-gpt2-small"]:
        out["panels"][name] = panel(name)
    out["ioi_token_groups"] = ioi_group_sizes()
    dump(ROOT / "results/posthoc/sweep-posthoc.json", out)
    for name, p in out["panels"].items():
        print(name, {k: p[k] for k in ["clean_accuracy", "groups_correct", "misclassified_all_groups", "misclassified_correct_groups_only"]})
        print("  equal budget", {k: (v.get("misclassified", v.get("misclassified_mean")), round(v.get("mae", v.get("mae_mean", 0)) or 0, 4)) for k, v in p["equal_budget"].items() if k != "individual"})
        print("  topk", p["topk"], "cancellation", p["cancellation"]["disputed"], p["cancellation"]["other_weak"])
        print("  spread pct", {k: v for k, v in p["spread_percentile_within_layer"].items() if k != "disputed_values"})


if __name__ == "__main__":
    main()

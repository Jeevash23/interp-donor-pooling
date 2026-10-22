"""Frozen analysis for the confirmation study.

Methods at every head-site, all compared with IND (average of the 64 individual patches):
  MEAN  centroid of all 64 donors (1 evaluation)
  TOK   token-grouped averaging: donors grouped by the token at the patched position, group-centroid
        scores averaged with weights proportional to group size (sva-end: 8 groups of 8 sharing the final
        noun; ioi-end: one group, identical to MEAN; ioi-s2: one group per distinct name)
  MIX8  eight mixed banks of eight (8 evaluations)
  S8    eight donors sampled without replacement from the 64 saved individual patches, the same eight for
        every head of a recipient and direction; 1,000 draws (8 evaluations)
Normalized effects are ratios of group-mean margins. A head-site is flagged when |effect| >= tau.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from common import ROOT, dump, read, utc

SETTINGS = {"sva-end-gpt2-small": ("sva-end", 64), "ioi-end-gpt2-small": ("ioi-end", 64),
            "ioi-s2-gpt2-small": ("ioi-s2", 64), "sva-end-qwen3-0.6b": ("sva-end", 32)}
TAUS, PRIMARY_TAU = [0.02, 0.05, 0.10], 0.05
BOOT, S8_DRAWS, BOOT_S8 = 5000, 1000, 100
SEED = 26100641


def load(setting, stage):
    kind, n = SETTINGS[setting]
    folder = ROOT / "results" / setting / stage
    task = "sva" if kind == "sva-end" else "ioi"
    rows = read(ROOT / f"data/{task}-{'development' if stage == 'development' else 'test'}.json")
    rows = [r for r in rows if (folder / f"{r['id']}.npz").exists()]
    if stage == "test":
        rows = rows[:n]
        if len(rows) != n:
            raise RuntimeError(f"{setting}: expected {n} rows, found {len(rows)}")
    ind, meth, clean, corrupt, strata, evals_tok = [], {"MEAN": [], "TOK": [], "MIX8": []}, [], [], [], []
    for r in rows:
        with np.load(folder / f"{r['id']}.npz") as z:
            m = z["margins"]
            if kind == "sva-end":
                ind.append(m[..., :64])
                meth["MEAN"].append(m[..., 64])
                meth["TOK"].append(m[..., 65:73].mean(-1))
                meth["MIX8"].append(m[..., 73:81].mean(-1))
                clean.append(z["baseline"][0])
                corrupt.append(z["baseline"][1])
                strata.append(f"{r['template']}-{r['class']}")
                evals_tok.append(8)
            else:
                m = m[None]                                        # direction axis of length 1
                ind.append(m[..., :64])
                meth["MEAN"].append(m[..., 64])
                meth["MIX8"].append(m[..., 65:73].mean(-1))
                n_tok = int(z["token_groups_evaluated"])
                if n_tok:
                    sizes = z["token_group_sizes"].astype(float)
                    meth["TOK"].append((m[..., 73:73 + n_tok] * sizes).sum(-1) / sizes.sum())
                else:
                    meth["TOK"].append(m[..., 64])                 # every donor shares the token: TOK is MEAN
                evals_tok.append(max(n_tok, 1))
                clean.append(z["baseline"][0])
                corrupt.append(z["donor_margins"].mean())
                strata.append(f"{r['template']}-{r['order']}")
    return {"kind": kind, "rows": rows, "ind": np.stack(ind), "methods": {k: np.stack(v) for k, v in meth.items()},
            "clean": np.array(clean), "corrupt": np.array(corrupt), "strata": strata,
            "evals_tok": float(np.mean(evals_tok))}


def normalized(kind, means, clean, corrupt):
    """means [..., D, L, H] of group-averaged margins -> normalized effects."""
    c = np.asarray(clean)[..., None, None]
    k = np.asarray(corrupt)[..., None, None]
    e = np.empty_like(means)
    if kind == "sva-end":
        e[..., 0, :, :] = (means[..., 0, :, :] - k) / (c - k)
        e[..., 1, :, :] = (c - means[..., 1, :, :]) / (c - k)
    else:
        e[..., 0, :, :] = (c - means[..., 0, :, :]) / (c - k)
    return e


def weights(strata, rng, draws):
    G = len(strata)
    W = np.zeros((draws, G))
    for s in sorted(set(strata)):
        ix = np.array([i for i, x in enumerate(strata) if x == s])
        W[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=draws) / G
    return W


def summary_vs_ind(e, e_ind):
    out = {"mae": float(np.abs(e - e_ind).mean())}
    for tau in TAUS:
        f, fi = np.abs(e) >= tau, np.abs(e_ind) >= tau
        out[f"tau{tau}"] = {"misclassified": int((f != fi).sum()), "false_positive": int((f & ~fi).sum()),
                            "false_negative": int((~f & fi).sum()), "flagged": int(f.sum()), "flagged_ind": int(fi.sum())}
    return out


def analyze_setting(setting, stage):
    P = load(setting, stage)
    kind, G = P["kind"], len(P["rows"])
    rng = np.random.default_rng(SEED)
    ind_mean = P["ind"].mean(-1)                                       # [G, D, L, H]
    cbar, kbar = P["clean"].mean(), P["corrupt"].mean()
    e_ind = normalized(kind, ind_mean.mean(0), cbar, kbar)
    res = {"groups": G, "head_sites": int(e_ind.size), "denominator": float(cbar - kbar),
           "maps": {"IND": e_ind.tolist()}, "methods": {},
           "evaluations_per_recipient_site": {"IND": 64, "MEAN": 1, "TOK": P["evals_tok"], "MIX8": 8, "S8": 8}}
    for name, arr in P["methods"].items():
        e = normalized(kind, arr.mean(0), cbar, kbar)
        res["maps"][name] = e.tolist()
        res["methods"][name] = summary_vs_ind(e, e_ind)
    # S8: eight sampled donors per recipient and direction, the same eight at every head.
    ind = P["ind"]
    Gn, D = ind.shape[0], ind.shape[1]
    s8_mae, s8_mis, s8_group_means = [], {t: [] for t in TAUS}, []
    for draw in range(S8_DRAWS):
        sel = np.argsort(rng.random((Gn, D, 64)), axis=-1)[..., :8]
        gathered = np.take_along_axis(ind, sel[:, :, None, None, :], axis=-1)   # [G, D, L, H, 8]
        gm = gathered.mean(-1)
        if draw < BOOT_S8:
            s8_group_means.append(gm)
        e = normalized(kind, gm.mean(0), cbar, kbar)
        s = summary_vs_ind(e, e_ind)
        s8_mae.append(s["mae"])
        for t in TAUS:
            s8_mis[t].append(s[f"tau{t}"]["misclassified"])
    res["methods"]["S8"] = {"mae_mean": float(np.mean(s8_mae)), "mae_range95": np.percentile(s8_mae, [2.5, 97.5]).tolist(),
                            **{f"tau{t}": {"misclassified_mean": float(np.mean(s8_mis[t])),
                                           "misclassified_range95": np.percentile(s8_mis[t], [2.5, 97.5]).tolist()}
                               for t in TAUS}}
    # Primary 1: misclassified head-sites under MEAN at tau = 0.05, bootstrap over groups within strata.
    W = weights(P["strata"], rng, BOOT)
    def maps_for(arr):
        flat = arr.reshape(Gn, -1)
        return normalized(kind, (W @ flat).reshape((BOOT,) + arr.shape[1:]), W @ P["clean"], W @ P["corrupt"])
    b_ind = maps_for(ind_mean)
    b_mean = maps_for(P["methods"]["MEAN"])
    mis = ((np.abs(b_mean) >= PRIMARY_TAU) != (np.abs(b_ind) >= PRIMARY_TAU)).reshape(BOOT, -1).sum(1)
    res["primary_mean_misclassified"] = {"estimate": res["methods"]["MEAN"][f"tau{PRIMARY_TAU}"]["misclassified"],
                                         "ci": np.percentile(mis, [100 * 0.05 / 6, 100 * (1 - 0.05 / 6)]).tolist(),
                                         "confidence": 1 - 0.05 / 3}
    # Primary 2 (where TOK differs from MEAN): TOK against S8 at equal or stated budget.
    tok = res["methods"]["TOK"]
    res["tok_vs_s8"] = {
        "tok_mae": tok["mae"], "s8_mae_mean": res["methods"]["S8"]["mae_mean"],
        "s8_mae_2.5th_percentile": res["methods"]["S8"]["mae_range95"][0],
        "tok_misclassified": tok[f"tau{PRIMARY_TAU}"]["misclassified"],
        "s8_misclassified_mean": res["methods"]["S8"][f"tau{PRIMARY_TAU}"]["misclassified_mean"],
        "tok_better": bool(tok["mae"] < res["methods"]["S8"]["mae_range95"][0]
                           and tok[f"tau{PRIMARY_TAU}"]["misclassified"] <= res["methods"]["S8"][f"tau{PRIMARY_TAU}"]["misclassified_mean"]),
        "tok_identical_to_mean": bool(np.allclose(P["methods"]["TOK"], P["methods"]["MEAN"]))}
    # Secondary: bootstrap of MAE(TOK) - E[MAE(S8)] with 100 S8 draws per replicate.
    sgm = np.stack(s8_group_means)                                       # [100, G, D, L, H]
    Wb = W[:1000]
    b_tok = normalized(kind, (Wb @ P["methods"]["TOK"].reshape(Gn, -1)).reshape((1000,) + ind_mean.shape[1:]),
                       Wb @ P["clean"], Wb @ P["corrupt"])
    b_ind1 = b_ind[:1000]
    diffs = []
    for b in range(1000):
        w = Wb[b]
        e_s8 = normalized(kind, np.tensordot(sgm, w, axes=([1], [0])), w @ P["clean"], w @ P["corrupt"])
        diffs.append(np.abs(b_tok[b] - b_ind1[b]).mean() - np.abs(e_s8 - b_ind1[b]).mean(axis=(1, 2, 3)).mean())
    res["tok_vs_s8"]["mae_difference_ci95"] = np.percentile(diffs, [2.5, 97.5]).tolist()
    # Every candidate head-site, with effects under every method.
    names = ["IND", "MEAN", "TOK", "MIX8"]
    with (ROOT / f"results/heads-{setting}-{stage}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["direction", "layer", "head"] + [f"effect_{n}" for n in names] + [f"flag_{n}_tau{PRIMARY_TAU}" for n in names])
        maps = {n: np.array(res["maps"][n]) for n in names}
        layers = list(range(e_ind.shape[1])) if not setting.endswith("qwen3-0.6b") else [0, 4, 8, 12, 16, 20, 24]
        for d in range(e_ind.shape[0]):
            for li in range(e_ind.shape[1]):
                for h in range(e_ind.shape[2]):
                    w.writerow([["restore", "destroy"][d] if kind == "sva-end" else "noising", layers[li], h]
                               + [f"{maps[n][d, li, h]:.6f}" for n in names]
                               + [int(abs(maps[n][d, li, h]) >= PRIMARY_TAU) for n in names])
    return res


def analyze(stage):
    out = {"created_utc": utc(), "stage": stage, "taus": TAUS, "primary_tau": PRIMARY_TAU, "seed": SEED, "settings": {}}
    for setting in SETTINGS:
        if (ROOT / "results" / setting / stage).exists():
            out["settings"][setting] = analyze_setting(setting, stage)
    dump(ROOT / f"results/analysis-{stage}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["development", "test"])
    a = analyze(ap.parse_args().stage)
    for s, r in a["settings"].items():
        print(s, "MEAN misclassified", r["primary_mean_misclassified"], "TOK vs S8", {k: v for k, v in r["tok_vs_s8"].items()})

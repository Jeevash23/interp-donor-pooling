"""Post hoc: centroid bias against the sampling noise of eight sampled donors (Study 2 panels, Study 3 settings).

Added after the results were known; frozen outputs are only read. At every site:
  bias  = |centroid effect - individual-average effect| (normalized units), for mean ablation and, where it
          exists, for token grouping (Study 2 agreement: NOUN8; Study 3: agreement and repeated subject);
  sd8   = the standard deviation of the eight-donor estimate of the same site effect, exact for sampling eight
          of the 64 donors without replacement in every group: sqrt(sum_g s2_g/8 * 56/63)/G, where s2_g is the
          variance of the 64 per-donor effects in group g.
The expected squared error is bias^2 for a centroid and sd8^2 for eight sampled donors, so a centroid is the
more accurate choice at a site exactly when its bias is below sd8. Sites near the threshold are those with
0.025 <= |individual-average effect| <= 0.10. Study 2 sites that no patch can change are left out.
Output: results/posthoc/bias-noise.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze_sweep import load_panel  # noqa: E402
from common import ROOT, dump, read, utc  # noqa: E402

S3 = ROOT.parent / "05-head-confirmation"
TAU = 0.05


def per_donor(task, raw, clean, unpatched):
    """Per-donor normalized effects [G, D, ..., 64] with the aggregate denominator."""
    c, k = clean.mean(), unpatched.mean()
    if task == "sva":
        base = np.stack([unpatched, clean], 1).reshape((len(clean), 2) + (1,) * (raw.ndim - 2))
        delta = raw - base
        delta[:, 1] *= -1
    else:
        delta = clean.reshape((len(clean),) + (1,) * (raw.ndim - 1)) - raw
    return delta / (c - k)


def summarize(ind, cents, sd8, mask):
    """Near the threshold: the individual average or the mean-ablation centroid within a factor of two of tau.
    Disputed: sites whose flag differs between mean ablation and the individual average."""
    band = lambda x: (np.abs(x) >= TAU / 2) & (np.abs(x) <= 2 * TAU)
    near = mask & (band(ind) | band(cents["mean"]))
    disputed = mask & ((np.abs(cents["mean"]) >= TAU) != (np.abs(ind) >= TAU))
    out = {"sites": int(mask.sum()), "near_threshold_sites": int(near.sum()), "disputed_sites": int(disputed.sum())}
    for label, m in [("near", near), ("disputed", disputed), ("all", mask)]:
        out[f"median_sd8_{label}"] = float(np.median(sd8[m])) if m.any() else None
    for name, e in cents.items():
        b = np.abs(e - ind)
        out[name] = {}
        for label, m in [("near", near), ("disputed", disputed), ("all", mask)]:
            out[name][f"median_bias_{label}"] = float(np.median(b[m])) if m.any() else None
            out[name][f"share_bias_above_sd8_{label}"] = float(np.mean(b[m] > sd8[m])) if m.any() else None
            out[name][f"median_ratio_bias_to_sd8_{label}"] = float(np.median(b[m] / sd8[m])) if m.any() else None
    return out


def sd_eight(per):
    """SD of the group-averaged eight-donor estimate at each site; per: [G, ..., 64]."""
    var = per.var(axis=-1)                                                  # population variance over donors
    return np.sqrt((var / 8 * 56 / 63).sum(0)) / per.shape[0]


def study2():
    res = {}
    for panel in ["sva-gpt2-small", "sva-qwen3-0.6b", "ioi-gpt2-small"]:
        P = load_panel(panel, "test")
        task, raw = P["task"], P["raw"][..., :64]
        per = per_donor(task, raw, P["clean"], P["corrupt"])
        ind = per.mean(axis=(0, -1))
        c, k = P["clean"].mean(), P["corrupt"].mean()
        cents = {}
        for meth in (["mean64", "noun8"] if task == "sva" else ["mean64"]):
            g = P["methods"][meth].mean(0)
            e = np.empty_like(g)
            if task == "sva":
                e[0], e[1] = (g[0] - k) / (c - k), (c - g[1]) / (c - k)
            else:
                e[0] = (c - g[0]) / (c - k)
            cents["mean" if meth == "mean64" else "token"] = e
        res[panel] = summarize(ind, cents, sd_eight(per), ~P["structural"])
    return res


def end_only():
    """Equal-budget comparison restricted to the final token of the agreement sweeps (mean absolute error)."""
    out = {}
    rng = np.random.default_rng(26100691)
    for panel in ["sva-gpt2-small", "sva-qwen3-0.6b"]:
        P = load_panel(panel, "test")
        raw = P["raw"][..., :64]
        G, end = raw.shape[0], P["positions"].index("END")
        c, k = P["clean"].mean(), P["corrupt"].mean()

        def eff(g):
            e = np.empty_like(g)
            e[0], e[1] = (g[0] - k) / (c - k), (c - g[1]) / (c - k)
            return e
        ind = eff(P["methods"]["individual"].mean(0))[:, :, end]
        res = {m: float(np.abs(eff(P["methods"][m].mean(0))[:, :, end] - ind).mean()) for m in ["mean64", "mix8", "noun8", "adj8"]}
        s8 = []
        for _ in range(1000):
            pick = np.argsort(rng.random((G, 2, 64)), axis=-1)[..., :8]
            gm = np.take_along_axis(raw, pick[:, :, None, None, :], axis=-1).mean(-1)
            s8.append(float(np.abs(eff(gm.mean(0))[:, :, end] - ind).mean()))
        res["sample8_mean"], res["sample8_range95"] = float(np.mean(s8)), np.percentile(s8, [2.5, 97.5]).tolist()
        out[panel] = res
    return out


def study3():
    res = {}
    for setting, task in [("sva-end-gpt2-small", "sva"), ("ioi-end-gpt2-small", "ioi"), ("ioi-s2-gpt2-small", "ioi")]:
        rows = read(S3 / f"data/{task}-test.json")[:64]
        M, clean, unpatched = [], [], []
        for r in rows:
            with np.load(S3 / "results" / setting / "test" / f"{r['id']}.npz") as z:
                if task == "sva":
                    M.append(z["margins"])
                    clean.append(z["baseline"][0]), unpatched.append(z["baseline"][1])
                else:
                    m = z["margins"][None]
                    n_tok = int(z["token_groups_evaluated"])
                    if n_tok:
                        w = z["token_group_sizes"].astype(float)
                        tok = (m[..., 73:73 + n_tok] * w).sum(-1) / w.sum()
                    else:
                        tok = m[..., 64]
                    M.append(np.concatenate([m[..., :65], tok[..., None]], -1))
                    clean.append(z["baseline"][0]), unpatched.append(z["donor_margins"].mean())
        M, clean, unpatched = np.stack(M), np.array(clean), np.array(unpatched)
        per = per_donor(task, M[..., :64], clean, unpatched)
        ind = per.mean(axis=(0, -1))
        c, k = clean.mean(), unpatched.mean()

        def eff(g):
            g = g.mean(0)
            e = np.empty_like(g)
            if task == "sva":
                e[0], e[1] = (g[0] - k) / (c - k), (c - g[1]) / (c - k)
            else:
                e[0] = (c - g[0]) / (c - k)
            return e
        tok = M[..., 65:73].mean(-1) if task == "sva" else M[..., 65]
        cents = {"mean": eff(M[..., 64])}
        if not setting.startswith("ioi-end"):
            cents["token"] = eff(tok)
        res[setting] = summarize(ind, cents, sd_eight(per), np.ones_like(ind, dtype=bool))
    return res


def main():
    out = {"created_utc": utc(), "posthoc": True, "tau": TAU, "near_threshold": [TAU / 2, 2 * TAU],
           "study2": study2(), "study3": study3(), "final_token_equal_budget": end_only()}
    dump(ROOT / "results/posthoc/bias-noise.json", out)
    for study in ["study2", "study3"]:
        for k, v in out[study].items():
            for label in ["near", "disputed", "all"]:
                line = f"{k:20s} {label:8s} n={v[{'near': 'near_threshold_sites', 'disputed': 'disputed_sites', 'all': 'sites'}[label]]:4d} sd8 {v['median_sd8_' + label] or 0:.4f}"
                for name in ["mean", "token"]:
                    if name in v and v[name][f"median_bias_{label}"] is not None:
                        line += f" | {name} bias {v[name]['median_bias_' + label]:.4f} ratio {v[name]['median_ratio_bias_to_sd8_' + label]:.2f} share>sd {v[name]['share_bias_above_sd8_' + label]:.2f}"
                print(line)


if __name__ == "__main__":
    main()

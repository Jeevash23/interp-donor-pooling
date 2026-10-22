"""Frozen analysis for Study 4 (GPT-2 Small, agreement, final token, four kinds of site).

Kinds: resid (residual stream after the block), attn (attention-layer output), mlp (MLP output), head
(one attention head). Each kind has 12 layers (heads: 12 x 12) x 2 directions of sites.
Methods at every site, all compared with IND (the average of the 64 individual patches):
  MEAN  centroid of all 64 donors (1 evaluation)
  TOK   token grouping: centroids of the eight banks that share the final noun, scores averaged (8)
  MIX8  centroids of eight mixed banks, scores averaged (8)
  S8    eight donors sampled from the 64 individual patches, the same eight for every site of a group and
        direction, 1,000 draws (8)
Normalized effects are ratios of group-mean margins: restore (patched - corrupt)/(clean - corrupt),
destroy (clean - patched)/(clean - corrupt). A site is flagged when |effect| >= 0.05.

Prespecified outcomes (PROTOCOL.md):
  P1  misclassified sites under MEAN, per kind, with a 98.75% stratified bootstrap interval (four kinds);
  P2  token grouping against eight sampled donors, per kind, by mean absolute error;
  P3  the ordering of the mean absolute centroid bias across kinds;
  P4  donor spread per kind, and whether bias follows spread.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from common import ROOT, dump, read, utc

KINDS = ["resid", "mlp", "attn", "head"]
TAU, TAUS = 0.05, [0.02, 0.05, 0.10]
BOOT, S8_DRAWS, SEED = 5000, 1000, 26100681
CONF = 1 - 0.05 / 4                                                      # four kinds


def load(stage):
    rows = read(ROOT / f"data/sva-{'development' if stage == 'development' else 'test'}.json")
    folder = ROOT / "results" / stage
    rows = [r for r in rows if (folder / f"{r['id']}.npz").exists()]
    if stage == "test" and len(rows) != 64:
        raise RuntimeError(f"expected 64 test groups, found {len(rows)}")
    data = {k: [] for k in KINDS}
    spread = {k: [] for k in KINDS}
    clean, corrupt, strata = [], [], []
    for r in rows:
        with np.load(folder / f"{r['id']}.npz") as z:
            for k in KINDS:
                m = z[f"margins_{k}"]
                data[k].append(m.reshape(2, -1, 81))                      # [2, sites, 81]
                spread[k].append(z[f"spread_{k}"].reshape(2, -1, 17))
            clean.append(z["baseline"][0])
            corrupt.append(z["baseline"][1])
        strata.append(f"{r['template']}-{r['class']}")
    return rows, {k: np.stack(v) for k, v in data.items()}, {k: np.stack(v) for k, v in spread.items()}, \
        np.array(clean), np.array(corrupt), strata


def normalized(group_means, c, k):
    """group_means [..., 2, S] -> normalized effects, using population means c and k."""
    c, k = np.asarray(c)[..., None], np.asarray(k)[..., None]
    e = np.empty_like(group_means)
    e[..., 0, :] = (group_means[..., 0, :] - k) / (c - k)
    e[..., 1, :] = (c - group_means[..., 1, :]) / (c - k)
    return e


def flags(e, ref, tau):
    a, b = np.abs(e) >= tau, np.abs(ref) >= tau
    return {"misclassified": int((a != b).sum()), "false_positive": int((a & ~b).sum()),
            "false_negative": int((~a & b).sum()), "flagged": int(a.sum()), "flagged_ind": int(b.sum())}


def analyze(stage):
    rows, M, SP, clean, corrupt, strata = load(stage)
    G = len(rows)
    c, k = clean.mean(), corrupt.mean()
    rng = np.random.default_rng(SEED)
    keys = sorted(set(strata))
    W = np.zeros((BOOT, G))
    for s in keys:
        ix = np.array([i for i, x in enumerate(strata) if x == s])
        W[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=BOOT) / G
    out = {"created_utc": utc(), "stage": stage, "groups": G, "tau": TAU, "taus": TAUS, "seed": SEED,
           "confidence": CONF, "kinds": {}}
    boot_bias = {}
    for kind in KINDS:
        m = M[kind]                                                       # [G, 2, S, 81]
        ind_g, mean_g = m[..., :64].mean(-1), m[..., 64]
        tok_g, mix_g = m[..., 65:73].mean(-1), m[..., 73:81].mean(-1)
        e = {name: normalized(x.mean(0), c, k) for name, x in
             [("IND", ind_g), ("MEAN", mean_g), ("TOK", tok_g), ("MIX8", mix_g)]}
        res = {"sites": int(e["IND"].size), "methods": {}}
        for name in ["MEAN", "TOK", "MIX8"]:
            res["methods"][name] = {"mae": float(np.abs(e[name] - e["IND"]).mean()),
                                    "max_abs_error": float(np.abs(e[name] - e["IND"]).max()),
                                    **{f"tau{t}": flags(e[name], e["IND"], t) for t in TAUS}}
        # S8: eight sampled donors per group and direction, the same eight at every site of the kind.
        maes, mis = [], {t: [] for t in TAUS}
        raw = m[..., :64]
        for _ in range(S8_DRAWS):
            pick = np.argsort(rng.random((G, 2, 64)), axis=-1)[..., :8]
            gm = np.take_along_axis(raw, pick[:, :, None, :], axis=-1).mean(-1)
            es = normalized(gm.mean(0), c, k)
            maes.append(float(np.abs(es - e["IND"]).mean()))
            for t in TAUS:
                mis[t].append(flags(es, e["IND"], t)["misclassified"])
        res["methods"]["S8"] = {"mae_mean": float(np.mean(maes)), "mae_range95": np.percentile(maes, [2.5, 97.5]).tolist(),
                                **{f"tau{t}": {"misclassified_mean": float(np.mean(mis[t])),
                                               "misclassified_range95": np.percentile(mis[t], [2.5, 97.5]).tolist()}
                                   for t in TAUS}}
        # P1: misclassified sites under MEAN with a stratified bootstrap interval.
        def boot_maps(x):
            flat = x.reshape(G, -1)
            return normalized((W @ flat).reshape((BOOT,) + x.shape[1:]), W @ clean, W @ corrupt)
        bi, bm = boot_maps(ind_g), boot_maps(mean_g)
        counts = ((np.abs(bm) >= TAU) != (np.abs(bi) >= TAU)).reshape(BOOT, -1).sum(1)
        lo, hi = np.percentile(counts, [100 * (1 - CONF) / 2, 100 * (1 + CONF) / 2])
        predicted = "present" if kind in ("resid", "mlp") else "absent"
        res["p1_mean_misclassified"] = {"estimate": res["methods"]["MEAN"][f"tau{TAU}"]["misclassified"],
                                        "interval": [float(lo), float(hi)], "confidence": CONF,
                                        "predicted": predicted,
                                        "supported": bool(lo >= 1) if predicted == "present" else bool(lo == 0)}
        # P2: token grouping against eight sampled donors, by mean absolute error.
        q_lo, q_hi = res["methods"]["S8"]["mae_range95"]
        tok_mae = res["methods"]["TOK"]["mae"]
        decision = "TOK better" if tok_mae < q_lo else "S8 better" if tok_mae > q_hi else "no difference"
        res["p2_tok_vs_s8"] = {"tok_mae": tok_mae, "s8_mae_mean": res["methods"]["S8"]["mae_mean"],
                               "s8_mae_2.5": q_lo, "s8_mae_97.5": q_hi, "decision": decision,
                               "predicted": "TOK better", "supported": decision == "TOK better"}
        # P3 inputs: mean absolute centroid bias per kind (normalized), with bootstrap draws.
        bias = np.abs(e["MEAN"] - e["IND"])
        res["mean_abs_bias"] = float(bias.mean())
        res["median_abs_bias"] = float(np.median(bias))
        boot_bias[kind] = np.abs(bm - bi).reshape(BOOT, -1).mean(1)
        # Bias and sampling noise per site: SD of an eight-donor average (without replacement from 64).
        per = raw.copy()
        delta = np.where(np.arange(2)[None, :, None, None] == 0, per - k, c - per) / (c - k)  # per-donor effects
        var_g = delta.var(axis=-1, ddof=0)                                # population variance over the 64 donors
        sd8 = np.sqrt((var_g / 8 * (64 - 8) / (64 - 1)).sum(0)) / G      # SD of the group-averaged 8-donor effect
        near = (np.abs(e["IND"]) >= TAU / 2) & (np.abs(e["IND"]) <= 2 * TAU)
        res["bias_vs_noise"] = {"near_threshold_sites": int(near.sum()),
                                "median_bias_near": float(np.median(bias[near])) if near.any() else None,
                                "median_sd8_near": float(np.median(sd8[near])) if near.any() else None,
                                "share_bias_above_sd8_near": float(np.mean(bias[near] > sd8[near])) if near.any() else None,
                                "median_bias_all": float(np.median(bias)), "median_sd8_all": float(np.median(sd8)),
                                "share_bias_above_sd8_all": float(np.mean(bias > sd8))}
        # P4: donor spread (mean squared radius of the 64 donors in residual space), per site.
        sp = SP[kind][..., 0].mean(0)                                     # [2, S]
        res["spread"] = {"median": float(np.median(sp)), "mean": float(sp.mean()),
                         "noun_bank_median": float(np.median(SP[kind][..., 1:9].mean(axis=(0, -1))))}
        ok = (bias > 0) & (sp > 0)
        res["spread"]["loglog_slope_bias_on_spread"] = float(np.polyfit(np.log(sp[ok]), np.log(bias[ok]), 1)[0]) if ok.sum() > 2 else None
        res["maps"] = {name: v.tolist() for name, v in e.items()}
        res["spread_map"] = sp.tolist()
        res["sd8_map"] = sd8.tolist()
        out["kinds"][kind] = res
    # P3: predicted ordering resid > mlp > attn > head of mean absolute centroid bias.
    pairs = [("resid", "mlp"), ("mlp", "attn"), ("attn", "head")]
    out["p3_ordering"] = {f"{a}-{b}": {"difference": out["kinds"][a]["mean_abs_bias"] - out["kinds"][b]["mean_abs_bias"],
                                       "ci95": np.percentile(boot_bias[a] - boot_bias[b], [2.5, 97.5]).tolist()}
                          for a, b in pairs}
    out["p3_ordering"]["all_pairs_above_zero"] = bool(all(out["p3_ordering"][f"{a}-{b}"]["ci95"][0] > 0 for a, b in pairs))
    out["p3_ordering"]["bootstrap_share_full_order"] = float(np.mean(
        (boot_bias["resid"] > boot_bias["mlp"]) & (boot_bias["mlp"] > boot_bias["attn"]) & (boot_bias["attn"] > boot_bias["head"])))
    # P4: across kinds, the order of median spread against the order of mean bias.
    med_sp = {kd: out["kinds"][kd]["spread"]["median"] for kd in KINDS}
    out["p4_spread_order"] = {"kinds_by_spread": sorted(KINDS, key=lambda kd: -med_sp[kd]),
                              "kinds_by_bias": sorted(KINDS, key=lambda kd: -out["kinds"][kd]["mean_abs_bias"])}
    out["p4_spread_order"]["orders_match"] = out["p4_spread_order"]["kinds_by_spread"] == out["p4_spread_order"]["kinds_by_bias"]
    out["p4_spread_order"]["spread_order_as_predicted"] = out["p4_spread_order"]["kinds_by_spread"] == KINDS
    out["p4_spread_order"]["supported"] = bool(out["p4_spread_order"]["orders_match"]
                                              and out["p4_spread_order"]["spread_order_as_predicted"])
    out["p3_ordering"]["supported"] = out["p3_ordering"]["all_pairs_above_zero"]
    dump(ROOT / f"results/analysis-{stage}.json", out)
    # Every site, with its effect under every method, flags, spread and eight-donor SD.
    with (ROOT / f"results/sites-{stage}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "direction", "layer", "head", "effect_IND", "effect_MEAN", "effect_TOK", "effect_MIX8",
                    "flag_IND", "flag_MEAN", "flag_TOK", "flag_MIX8", "spread64", "sd8"])
        for kind in KINDS:
            r = out["kinds"][kind]
            maps = {n: np.array(r["maps"][n]) for n in ["IND", "MEAN", "TOK", "MIX8"]}
            sp, sd = np.array(r["spread_map"]), np.array(r["sd8_map"])
            for d in range(2):
                for s in range(maps["IND"].shape[1]):
                    layer, head = (s // 12, s % 12) if kind == "head" else (s, "")
                    w.writerow([kind, ["restore", "destroy"][d], layer, head]
                               + [f"{maps[n][d, s]:.6f}" for n in maps] + [int(abs(maps[n][d, s]) >= TAU) for n in maps]
                               + [f"{sp[d, s]:.4f}", f"{sd[d, s]:.6f}"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["development", "test"])
    a = analyze(ap.parse_args().stage)
    for kind in KINDS:
        r = a["kinds"][kind]
        print(kind, "P1", r["p1_mean_misclassified"], "| P2", r["p2_tok_vs_s8"]["decision"],
              f"TOK {r['p2_tok_vs_s8']['tok_mae']:.2e} S8 {r['p2_tok_vs_s8']['s8_mae_mean']:.2e}",
              f"| bias {r['mean_abs_bias']:.4f} | spread {r['spread']['median']:.1f} | noise", r["bias_vs_noise"])
    print("P3", a["p3_ordering"])
    print("P4", a["p4_spread_order"])

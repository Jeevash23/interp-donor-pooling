"""Frozen analysis for Study 5 (Qwen3-1.7B), ported from the frozen analyses of Studies 4, 3 and 2.

    python analyze.py <stage> <arrays folder> <output folder> <rows.json>

Changes against the frozen files (the full diff is in data/analysis-port-diff.json):
  - input: the panel folders A, B1, B2, C1, C2 and compact row metadata instead of each study's own paths;
  - 28 layers and 16 heads (the CSV writers decode head-sites with 16 heads); 64 or, under the cost rule, 32 rows;
  - new seeds (26101781, 26101782, 26101783);
  - Study 4: P1 and P2 report estimates and decisions without Study 4's predictions (secondary in Study 5);
  - Study 3: settings sva-end (panel A's heads), ioi-end (B1), ioi-s2 (B2); its bootstrap interval for TOK - S8 is
    not repeated (PROTOCOL.md: Appendix F shows it is biased towards TOK);
  - Study 2: panels C1 and C2, primary intervals at 97.5% (Bonferroni over two panels); an interval over draws that
    are all undefined (a false discovery proportion when no site is flagged) is reported as null;
  - decisions(): the primary rules R1, R2 and R3 of PROTOCOL.md (new code).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import ADJ, MIX, NOUN, dump, read, utc

# ============================================================================ Study 4 (panel A)
KINDS = ["resid", "mlp", "attn", "head"]
TAU, TAUS = 0.05, [0.02, 0.05, 0.10]
BOOT, S8_DRAWS, SEED = 5000, 1000, 26101781
CONF = 1 - 0.05 / 4                                                      # four kinds
N_HEADS = 16


def load(rows, folder):
    folder = Path(folder) / "A"
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


def analyze_components(rows, folder, out_dir, stage):
    rows, M, SP, clean, corrupt, strata = load(rows, folder)
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
        res["p1_mean_misclassified"] = {"estimate": res["methods"]["MEAN"][f"tau{TAU}"]["misclassified"],
                                        "interval": [float(lo), float(hi)], "confidence": CONF}
        # P2: token grouping against eight sampled donors, by mean absolute error.
        q_lo, q_hi = res["methods"]["S8"]["mae_range95"]
        tok_mae = res["methods"]["TOK"]["mae"]
        decision = "TOK better" if tok_mae < q_lo else "S8 better" if tok_mae > q_hi else "no difference"
        res["p2_tok_vs_s8"] = {"tok_mae": tok_mae, "s8_mae_mean": res["methods"]["S8"]["mae_mean"],
                               "s8_mae_2.5": q_lo, "s8_mae_97.5": q_hi, "decision": decision}
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
    dump(Path(out_dir) / f"analysis-A-{stage}.json", out)
    # Every site, with its effect under every method, flags, spread and eight-donor SD.
    with (Path(out_dir) / f"sites-A-{stage}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "direction", "layer", "head", "effect_IND", "effect_MEAN", "effect_TOK", "effect_MIX8",
                    "flag_IND", "flag_MEAN", "flag_TOK", "flag_MIX8", "spread64", "sd8"])
        for kind in KINDS:
            r = out["kinds"][kind]
            maps = {n: np.array(r["maps"][n]) for n in ["IND", "MEAN", "TOK", "MIX8"]}
            sp, sd = np.array(r["spread_map"]), np.array(r["sd8_map"])
            for d in range(2):
                for s in range(maps["IND"].shape[1]):
                    layer, head = (s // N_HEADS, s % N_HEADS) if kind == "head" else (s, "")
                    w.writerow([kind, ["restore", "destroy"][d], layer, head]
                               + [f"{maps[n][d, s]:.6f}" for n in maps] + [int(abs(maps[n][d, s]) >= TAU) for n in maps]
                               + [f"{sp[d, s]:.4f}", f"{sd[d, s]:.6f}"])
    return out


# ============================================================================ Study 3 (head settings)
H_TAUS, PRIMARY_TAU = [0.02, 0.05, 0.10], 0.05
H_BOOT, H_S8_DRAWS = 5000, 1000
H_SEED = 26101782
SETTINGS = {"sva-end": ("sva-end", "A"), "ioi-end": ("ioi-end", "B1"), "ioi-s2": ("ioi-s2", "B2")}


def load_heads(setting, rows, folder):
    kind, panel = SETTINGS[setting]
    folder = Path(folder) / panel
    ind, meth, clean, corrupt, strata, evals_tok = [], {"MEAN": [], "TOK": [], "MIX8": []}, [], [], [], []
    for r in rows:
        with np.load(folder / f"{r['id']}.npz") as z:
            if kind == "sva-end":
                m = z["margins_head"]
                ind.append(m[..., :64])
                meth["MEAN"].append(m[..., 64])
                meth["TOK"].append(m[..., 65:73].mean(-1))
                meth["MIX8"].append(m[..., 73:81].mean(-1))
                clean.append(z["baseline"][0])
                corrupt.append(z["baseline"][1])
                strata.append(f"{r['template']}-{r['class']}")
                evals_tok.append(8)
            else:
                m = z["margins"][None]                                 # direction axis of length 1
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


def normalized_heads(kind, means, clean, corrupt):
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
    for tau in H_TAUS:
        f, fi = np.abs(e) >= tau, np.abs(e_ind) >= tau
        out[f"tau{tau}"] = {"misclassified": int((f != fi).sum()), "false_positive": int((f & ~fi).sum()),
                            "false_negative": int((~f & fi).sum()), "flagged": int(f.sum()), "flagged_ind": int(fi.sum())}
    return out


def analyze_setting(setting, rows, folder, out_dir, stage):
    P = load_heads(setting, rows, folder)
    kind, G = P["kind"], len(P["rows"])
    rng = np.random.default_rng(H_SEED)
    ind_mean = P["ind"].mean(-1)                                       # [G, D, L, H]
    cbar, kbar = P["clean"].mean(), P["corrupt"].mean()
    e_ind = normalized_heads(kind, ind_mean.mean(0), cbar, kbar)
    res = {"groups": G, "head_sites": int(e_ind.size), "denominator": float(cbar - kbar),
           "maps": {"IND": e_ind.tolist()}, "methods": {},
           "evaluations_per_recipient_site": {"IND": 64, "MEAN": 1, "TOK": P["evals_tok"], "MIX8": 8, "S8": 8}}
    for name, arr in P["methods"].items():
        e = normalized_heads(kind, arr.mean(0), cbar, kbar)
        res["maps"][name] = e.tolist()
        res["methods"][name] = summary_vs_ind(e, e_ind)
    # S8: eight sampled donors per recipient and direction, the same eight at every head.
    ind = P["ind"]
    Gn, D = ind.shape[0], ind.shape[1]
    s8_mae, s8_mis = [], {t: [] for t in H_TAUS}
    for draw in range(H_S8_DRAWS):
        sel = np.argsort(rng.random((Gn, D, 64)), axis=-1)[..., :8]
        gathered = np.take_along_axis(ind, sel[:, :, None, None, :], axis=-1)   # [G, D, L, H, 8]
        gm = gathered.mean(-1)
        e = normalized_heads(kind, gm.mean(0), cbar, kbar)
        s = summary_vs_ind(e, e_ind)
        s8_mae.append(s["mae"])
        for t in H_TAUS:
            s8_mis[t].append(s[f"tau{t}"]["misclassified"])
    res["methods"]["S8"] = {"mae_mean": float(np.mean(s8_mae)), "mae_range95": np.percentile(s8_mae, [2.5, 97.5]).tolist(),
                            **{f"tau{t}": {"misclassified_mean": float(np.mean(s8_mis[t])),
                                           "misclassified_range95": np.percentile(s8_mis[t], [2.5, 97.5]).tolist()}
                               for t in H_TAUS}}
    # Primary (R2): misclassified head-sites under MEAN at tau = 0.05, bootstrap over groups within strata.
    W = weights(P["strata"], rng, H_BOOT)
    def maps_for(arr):
        flat = arr.reshape(Gn, -1)
        return normalized_heads(kind, (W @ flat).reshape((H_BOOT,) + arr.shape[1:]), W @ P["clean"], W @ P["corrupt"])
    b_ind = maps_for(ind_mean)
    b_mean = maps_for(P["methods"]["MEAN"])
    mis = ((np.abs(b_mean) >= PRIMARY_TAU) != (np.abs(b_ind) >= PRIMARY_TAU)).reshape(H_BOOT, -1).sum(1)
    res["primary_mean_misclassified"] = {"estimate": res["methods"]["MEAN"][f"tau{PRIMARY_TAU}"]["misclassified"],
                                         "ci": np.percentile(mis, [100 * 0.05 / 6, 100 * (1 - 0.05 / 6)]).tolist(),
                                         "confidence": 1 - 0.05 / 3}
    # H2 (where TOK differs from MEAN): TOK against S8 at equal or stated budget.
    tok = res["methods"]["TOK"]
    res["tok_vs_s8"] = {
        "tok_mae": tok["mae"], "s8_mae_mean": res["methods"]["S8"]["mae_mean"],
        "s8_mae_2.5th_percentile": res["methods"]["S8"]["mae_range95"][0],
        "tok_misclassified": tok[f"tau{PRIMARY_TAU}"]["misclassified"],
        "s8_misclassified_mean": res["methods"]["S8"][f"tau{PRIMARY_TAU}"]["misclassified_mean"],
        "tok_better": bool(tok["mae"] < res["methods"]["S8"]["mae_range95"][0]
                           and tok[f"tau{PRIMARY_TAU}"]["misclassified"] <= res["methods"]["S8"][f"tau{PRIMARY_TAU}"]["misclassified_mean"]),
        "tok_identical_to_mean": bool(np.allclose(P["methods"]["TOK"], P["methods"]["MEAN"]))}
    # Every candidate head-site, with effects under every method.
    names = ["IND", "MEAN", "TOK", "MIX8"]
    with (Path(out_dir) / f"heads-{setting}-{stage}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["direction", "layer", "head"] + [f"effect_{n}" for n in names] + [f"flag_{n}_tau{PRIMARY_TAU}" for n in names])
        maps = {n: np.array(res["maps"][n]) for n in names}
        for d in range(e_ind.shape[0]):
            for li in range(e_ind.shape[1]):
                for h in range(e_ind.shape[2]):
                    w.writerow([["restore", "destroy"][d] if kind == "sva-end" else "noising", li, h]
                               + [f"{maps[n][d, li, h]:.6f}" for n in names]
                               + [int(abs(maps[n][d, li, h]) >= PRIMARY_TAU) for n in names])
    return res


def analyze_heads(sva_rows, ioi_rows, folder, out_dir, stage):
    out = {"created_utc": utc(), "stage": stage, "taus": H_TAUS, "primary_tau": PRIMARY_TAU, "seed": H_SEED, "settings": {}}
    for setting, (kind, panel) in SETTINGS.items():
        rows = sva_rows if kind == "sva-end" else ioi_rows
        if rows and (Path(folder) / panel).exists():
            out["settings"][setting] = analyze_setting(setting, rows, folder, out_dir, stage)
    dump(Path(out_dir) / f"analysis-heads-{stage}.json", out)
    return out


# ============================================================================ Study 2 (sweeps)
DRAWS, S_SEED = 20000, 26101783
S_TAUS = [0.05, 0.10]
IOI_LABELS = ["prefix", "N1", "and", "N2", "went", "to", "the", "PLACE", "punct", "S2", "gave", "a", "OBJ", "END"]


def bank_indices(partition):
    return [[s * 8 + n for s, n in bank] for bank in partition]


def load_panel(task, rows, folder):
    folder = Path(folder) / ("C1" if task == "sva" else "C2")
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
    # Study 2's Amendment 4: sites where no patch changes any margin in any group (structural zeros).
    if task == "sva":
        base = np.stack([corrupt, clean], 1)[:, :, None, None, None]
    else:
        base = clean[:, None, None, None, None]
    structural = (np.abs(m - base).max(axis=(0, -1)) <= 1e-4)
    return {"task": task, "rows": rows, "methods": methods, "clean": clean, "corrupt": corrupt,
            "structural": structural, "strata": strata, "positions": positions, **extra}


def normalized_sweep(task, means, clean, corrupt):
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
    if np.isnan(np.asarray(x, dtype=float)).all():      # e.g. no site flagged in any draw (Study 2's code would stop here)
        return None
    lo, hi = np.nanpercentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def analyze_panel(task, rows, folder, alpha_primary):
    P = load_panel(task, rows, folder)
    G = len(P["rows"])
    rng = np.random.default_rng(S_SEED)
    W = boot_counts(P["strata"], rng)
    point_w = np.full(G, 1 / G)

    def maps(weights):
        clean = weights @ P["clean"]
        corrupt = weights @ P["corrupt"]
        out = {}
        for name, arr in P["methods"].items():
            flat = arr.reshape(G, -1)
            means = (weights @ flat).reshape(weights.shape[:-1] + arr.shape[1:])
            out[name] = normalized_sweep(task, means, clean, corrupt)
        return out

    point = maps(point_w)
    boot = maps(W)
    res = {"groups": G, "positions": P["positions"], "denominator": float(P["clean"].mean() - P["corrupt"].mean()),
           "clean_margin": float(P["clean"].mean()), "corrupt_margin": float(P["corrupt"].mean()),
           "maps": {k: v.tolist() for k, v in point.items()}, "flags": {}, "bias": {}}
    others = [m for m in P["methods"] if m != "individual"]
    for tau in S_TAUS:
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


def analyze_sweeps(sva_rows, ioi_rows, folder, out_dir, stage):
    out = {"created_utc": utc(), "stage": stage, "draws": DRAWS, "seed": S_SEED, "taus": S_TAUS, "panels": {}}
    for panel, task, rows in (("sva-qwen3-1.7b", "sva", sva_rows), ("ioi-qwen3-1.7b", "ioi", ioi_rows)):
        if rows and (Path(folder) / ("C1" if task == "sva" else "C2")).exists():
            out["panels"][panel] = analyze_panel(task, rows, folder, alpha_primary=0.05 / 2)
    out["primary"] = {f"R3_{panel}_mean64_misclassified_sites_tau0.05": p["flags"]["mean64/tau0.05"]["misclassified_sites"]
                      for panel, p in out["panels"].items()}
    dump(Path(out_dir) / f"analysis-sweeps-{stage}.json", out)
    return out


# ============================================================================ decisions (PROTOCOL.md, primary outcomes)
EARLIER_RATES = {"GPT-2 Small agreement": 6 / 120, "Qwen3-0.6B agreement": 27 / 280, "GPT-2 Small IOI": 6 / 168}


def decisions(a4, a3, a2):
    out = {}
    if a4 is not None:
        r1 = {"p3_supported": bool(a4["p3_ordering"]["supported"]), "p4_supported": bool(a4["p4_spread_order"]["supported"]),
              "p3_ordering": {k: v for k, v in a4["p3_ordering"].items()},
              "kinds_by_bias": a4["p4_spread_order"]["kinds_by_bias"], "kinds_by_spread": a4["p4_spread_order"]["kinds_by_spread"],
              "mean_abs_bias": {k: a4["kinds"][k]["mean_abs_bias"] for k in KINDS},
              "median_spread": {k: a4["kinds"][k]["spread"]["median"] for k in KINDS}}
        r1["replicates"] = r1["p3_supported"] and r1["p4_supported"]
        out["R1"] = r1
    if a3 is not None:
        r2 = {}
        for setting, res in a3["settings"].items():
            lo, hi = res["primary_mean_misclassified"]["ci"]
            n = res["head_sites"]
            bound = int(np.floor(0.02 * n))
            verdict = "contradicts" if lo >= 1 else ("replicates" if hi <= bound else "inconclusive")
            r2[setting] = {"estimate": res["primary_mean_misclassified"]["estimate"], "ci": [lo, hi],
                           "confidence": res["primary_mean_misclassified"]["confidence"], "head_sites": n,
                           "upper_bound_2pct": bound, "study3_rule_changes_flags": bool(lo >= 1), "verdict": verdict}
        out["R2"] = r2
    if a2 is not None:
        r3 = {}
        for panel, p in a2["panels"].items():
            f = p["flags"]["mean64/tau0.05"]["misclassified_sites"]
            sites = p["flags"]["mean64/tau0.05"]["sites"]
            r3[panel] = {"estimate": f["estimate"], "ci": f["ci"], "confidence": f["confidence"], "sites": sites,
                         "rate": f["estimate"] / sites, "replicates": bool(f["ci"][0] >= 1)}
        r3["earlier_rates"] = EARLIER_RATES
        out["R3"] = r3
    return out


def analyze(stage, folder, out_dir, rows_path):
    meta = read(rows_path)
    sva_rows, ioi_rows = meta["sva"], meta["ioi"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    a4 = analyze_components(sva_rows, folder, out_dir, stage) if sva_rows and (Path(folder) / "A").exists() else None
    a3 = analyze_heads(sva_rows, ioi_rows, folder, out_dir, stage)
    a2 = analyze_sweeps(sva_rows, ioi_rows, folder, out_dir, stage)
    out = {"created_utc": utc(), "stage": stage, "sva_rows": [r["id"] for r in sva_rows], "ioi_rows": [r["id"] for r in ioi_rows],
           **decisions(a4, a3 if a3["settings"] else None, a2 if a2["panels"] else None)}
    dump(Path(out_dir) / f"decisions-{stage}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["development", "test"])
    ap.add_argument("folder")
    ap.add_argument("out_dir")
    ap.add_argument("rows")
    a = ap.parse_args()
    print(analyze(a.stage, a.folder, a.out_dir, a.rows))

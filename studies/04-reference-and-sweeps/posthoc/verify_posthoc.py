"""Independent check of the post hoc sweep analyses (results/posthoc/sweep-posthoc.json).

Imports nothing from the study's code. It reads the raw sweep arrays directly and recomputes, with
separate code, every quantity the manuscript quotes from sweep-posthoc.json: clean accuracy and the
restricted counts, the bank-centroid counts and errors, the threshold curve, the top-k overlap, the
cancellation summaries, the spread percentiles and the IOI token-group sizes. Sampled-donor results
are redrawn with a separate seed and compared within tolerances.
Output: results/posthoc/verification-posthoc.json (exit code 1 if anything differs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = {"sva-gpt2-small": ("sva", "gpt2-small", 64), "sva-qwen3-0.6b": ("sva", "qwen3-0.6b", 32),
          "ioi-gpt2-small": ("ioi", "gpt2-small", 64)}
TAU, TOL, DRAWS, SEED = 0.05, 1e-9, 300, 4242


def read(p):
    return json.loads(Path(p).read_text())


def load(panel):
    task, model, n = PANELS[panel]
    folder = ROOT / "results/sweep" / f"{task}-{model}" / "test"
    rows = read(ROOT / ("data/sva-test.json" if task == "sva" else "data/ioi-sweep.json"))
    rows = [r for r in rows if (folder / f"{r['id']}.npz").exists()][:n]
    margins, spread, clean, unpatched_k = [], [], [], []
    for r in rows:
        with np.load(folder / f"{r['id']}.npz") as z:
            m, s = z["margins"], z["spread"]
            if task == "ioi":
                m, s = m[None, :, -14:], s[None, :, -14:]          # right-aligned 14 positions, one direction
            margins.append(m)
            spread.append(s)
            clean.append(float(z["baseline"][0]))
            unpatched_k.append(float(z["baseline"][1]) if task == "sva" else float(z["donor_margins"].mean()))
    return task, np.array(margins), np.array(spread), np.array(clean), np.array(unpatched_k)


def effects(task, group_margins, c, k):
    """group_margins [G, D, L, P] -> normalized effect map [D, L, P] from group means."""
    g = group_margins.mean(0)
    out = np.empty_like(g)
    if task == "sva":
        out[0] = (g[0] - k) / (c - k)
        out[1] = (c - g[1]) / (c - k)
    else:
        out[0] = (c - g[0]) / (c - k)
    return out


def flips(a, b, tau=TAU):
    return int(((np.abs(a) >= tau) != (np.abs(b) >= tau)).sum())


def check_panel(panel, saved):
    task, m, spread, clean, kk = load(panel)
    G = m.shape[0]
    cols = {"individual": slice(0, 64)}
    if task == "sva":
        cols.update({"mix8": slice(64, 72), "noun8": slice(72, 80), "adj8": slice(80, 88), "mean64": 88})
    else:
        cols.update({"mix8": slice(64, 72), "mean64": 72})
    grp = {k: (m[..., v].mean(-1) if isinstance(v, slice) else m[..., v]) for k, v in cols.items()}
    c, k = clean.mean(), kk.mean()
    e = {name: effects(task, a, c, k) for name, a in grp.items()}
    if task == "sva":
        base = np.stack([kk, clean], 1)[:, :, None, None, None]
    else:
        base = clean[:, None, None, None, None]
    live = ~(np.abs(m - base).max(axis=(0, -1)) <= 1e-4)
    out, problems = {}, []

    def expect(name, got, want, tol=TOL):
        out[name] = got
        if want is None or (isinstance(got, (int, float)) and abs(got - want) > tol) or (not isinstance(got, (int, float)) and got != want):
            problems.append(f"{name}: recomputed {got}, saved {want}")

    expect("clean_accuracy", float(np.mean(clean > 0)), saved["clean_accuracy"])
    keep = np.where(clean > 0)[0]
    ck, kk2 = clean[keep].mean(), kk[keep].mean()
    expect("restricted_count", flips(effects(task, grp["mean64"][keep], ck, kk2), effects(task, grp["individual"][keep], ck, kk2)),
           saved["misclassified_correct_groups_only"])
    for name in grp:
        if name == "individual":
            continue
        eq = saved["equal_budget"][name]
        expect(f"{name}_misclassified", flips(e[name], e["individual"]), eq["misclassified"])
        expect(f"{name}_mae", float(np.abs(e[name] - e["individual"])[live].mean()), eq["mae"])
    for t, want in saved["misclassified_by_tau"].items():
        expect(f"tau_{t}", flips(e["mean64"], e["individual"], float(t)), want)
    kflag = int((np.abs(e["individual"]) >= TAU).sum())
    top = lambda x: set(np.argsort(-np.abs(x).ravel(), kind="stable")[:kflag])
    expect("topk_overlap", len(top(e["individual"]) & top(e["mean64"])), saved["topk"]["overlap"])
    # Cancellation: per-donor normalized effects with the aggregate denominator.
    raw = m[..., :64]
    delta = (raw - base) if task == "sva" else (clean[:, None, None, None, None] - raw)
    if task == "sva":
        delta[:, 1] *= -1
    per = delta / (c - k)
    avg = per.mean(axis=(0, -1))
    opp = (np.sign(per) != np.sign(avg)[None, ..., None]).mean(axis=(0, -1))
    absm = np.abs(per).mean(axis=(0, -1))
    disputed = (np.abs(e["mean64"]) >= TAU) != (np.abs(e["individual"]) >= TAU)
    weak = (np.abs(e["individual"]) < TAU) & live & ~disputed
    for label, mask in [("disputed", disputed), ("other_weak", weak)]:
        s = saved["cancellation"][label]
        expect(f"{label}_abs_effect", float(absm[mask].mean()), s["mean_abs_per_donor_effect"], 1e-12)
        expect(f"{label}_abs_average", float(np.abs(avg[mask]).mean()), s["abs_signed_average"], 1e-12)
        expect(f"{label}_opposite", float(opp[mask].mean()), s["share_opposite_sign"], 1e-12)
    # Spread percentile within layer (64-donor spread, averaged over groups).
    sp = spread[..., -1].mean(0)
    def pct(d, l, p):
        vals = sp[:, l, :][live[:, l, :]]
        return float((vals < sp[d, l, p]).mean())
    dis = [pct(*ix) for ix in np.argwhere(disputed)]
    oth = [pct(*ix) for ix in np.argwhere(live & ~disputed)]
    sv = saved["spread_percentile_within_layer"]
    expect("spread_pct_disputed", float(np.median(dis)), sv["disputed_median"])
    expect("spread_pct_other", float(np.median(oth)), sv["other_live_median"])
    # Eight sampled donors, redrawn with a separate seed.
    rng = np.random.default_rng(SEED)
    mis, mae = [], []
    for _ in range(DRAWS):
        pick = np.argsort(rng.random((G, raw.shape[1], 64)), axis=-1)[..., :8]
        gm = np.take_along_axis(raw, pick[:, :, None, None, :], axis=-1).mean(-1)
        es = effects(task, gm, c, k)
        mis.append(flips(es, e["individual"]))
        mae.append(float(np.abs(es - e["individual"])[live].mean()))
    s8 = saved["equal_budget"]["sample8"]
    out["sample8_misclassified_mean"], out["sample8_mae_mean"] = float(np.mean(mis)), float(np.mean(mae))
    if abs(np.mean(mis) - s8["misclassified_mean"]) > 0.15 or abs(np.mean(mae) - s8["mae_mean"]) > 0.05 * s8["mae_mean"]:
        problems.append(f"sample8: recomputed {np.mean(mis):.3f}, {np.mean(mae):.5f}; saved {s8['misclassified_mean']}, {s8['mae_mean']}")
    return out, problems


def token_groups(saved):
    rows = read(ROOT / "data/ioi-sweep.json")
    labels = ["prefix", "N1", "and", "N2", "went", "to", "the", "PLACE", "punct", "S2", "gave", "a", "OBJ", "END"]
    if set(labels) != set(saved):
        return [f"position labels differ: {sorted(saved)}"]
    problems = []
    for q, lab in enumerate(labels):
        counts = []
        for r in rows:
            off = len(r["recipient"]["ids"]) - 14
            counts.append(len({d["ids"][off + q] for d in r["donors"]}))
        if abs(np.mean(counts) - saved[lab]["groups_mean"]) > TOL:
            problems.append(f"token groups at {lab}: recomputed {np.mean(counts)}, saved {saved[lab]['groups_mean']}")
    return problems


def random_circuits():
    """Population-weighted agreement of the random circuits, recomputed from the raw margins."""
    path = ROOT / "results/posthoc/random-circuits.json"
    if not path.exists():
        return ["random-circuits.json missing"]
    saved = read(path)
    design = read(ROOT / "results/posthoc/random-circuits/design.json")
    pops = read(ROOT / "results/reference/screen-selection.json")["strata"]
    rows = read(ROOT / "data/ioi-selected-test.json")
    problems, n_circ = [], len(design["circuits"])
    for c in design["circuits"]:
        if len(c) != 26:
            problems.append(f"random circuit with {len(c)} nodes")
    for label in ["pair_error", "pair_correct"]:
        sums = {}
        for r in rows:
            if r["stratum_label"] != label:
                continue
            with np.load(ROOT / "results/posthoc/random-circuits/test" / f"{r['id']}.npz") as z:
                full = float(z["full_margin"]) > 0
                for cell in ["N8", "NPO8", "N128", "NPO128"]:
                    for k in range(n_circ):
                        cm = float(z[f"{cell}__c{k}__centroid_margin"]) > 0
                        dm = (z[f"{cell}__c{k}__donor_margins"] > 0) == full
                        for pool, v in [("mean", float(cm == full)), ("resample", float(dm.mean()))]:
                            sums.setdefault((cell, pool, k, r["family"]), []).append(v)
        for cell in ["N8", "NPO8", "N128", "NPO128"]:
            for pool in ["mean", "resample"]:
                per = []
                for k in range(n_circ):
                    fams = sorted({f for (c, q, kk, f) in sums if (c, q, kk) == (cell, pool, k)})
                    total = sum(pops[f"{f}/{label}"]["population"] for f in fams)
                    per.append(sum(pops[f"{f}/{label}"]["population"] * np.mean(sums[(cell, pool, k, f)]) for f in fams) / total)
                want = saved["tables"][label][f"{cell}/{pool}"]
                if max(abs(a - b) for a, b in zip(per, want["per_circuit"])) > 1e-9:
                    problems.append(f"random circuits {label} {cell}/{pool}: recomputed {per}, saved {want['per_circuit']}")
    return problems


def error_discrimination():
    """P(subject | error), P(subject | correct), odds ratio and AUROC for every unit and cell, by brute-force pairs."""
    path = ROOT / "results/posthoc/error-discrimination.json"
    if not path.exists():
        return ["error-discrimination.json missing"]
    saved = read(path)["results"]
    rows = read(ROOT / "data/ioi-selected-test.json")
    pops = read(ROOT / "results/reference/screen-selection.json")["strata"]
    keys = [f"{r['family']}/{r['stratum_label']}" for r in rows]
    w = np.array([pops[k]["population"] / keys.count(k) for k in keys])
    err = np.array([r["stratum_label"] == "pair_error" for r in rows])
    problems = []
    for cell in ["N8", "N128", "NPO8", "NPO128"]:
        units = {"circuit": [], "empty": [], **{f"random{k}": [] for k in range(5)}}
        for r in rows:
            with np.load(ROOT / "results/reference/test" / f"{r['id']}.npz") as a, \
                    np.load(ROOT / "results/posthoc/random-circuits/test" / f"{r['id']}.npz") as b:
                units["circuit"].append((float(a[f"{cell}__reference__centroid_margin"]), a[f"{cell}__reference__donor_margins"]))
                units["empty"].append((float(a[f"{cell}__empty__centroid_margin"]), a[f"{cell}__empty__donor_margins"]))
                for k in range(5):
                    units[f"random{k}"].append((float(b[f"{cell}__c{k}__centroid_margin"]), b[f"{cell}__c{k}__donor_margins"]))
        for u, vals in units.items():
            for pool in ["mean", "resample"]:
                score = np.array([cm if pool == "mean" else float(np.mean(dm)) for cm, dm in vals])
                pick = np.array([float(cm < 0) if pool == "mean" else float(np.mean(dm < 0)) for cm, dm in vals])
                pe = (w[err] * pick[err]).sum() / w[err].sum()
                pc = (w[~err] * pick[~err]).sum() / w[~err].sum()
                orr = (pe / (1 - pe)) / (pc / (1 - pc))
                se, sc, we, wc = score[err], score[~err], w[err], w[~err]
                less = (se[:, None] < sc[None, :]).astype(float) + 0.5 * (se[:, None] == sc[None, :])
                auc = float((we[:, None] * wc[None, :] * less).sum() / (we.sum() * wc.sum()))
                x = saved[f"{u}/{pool}/{cell}"]
                for name, got, want in [("p_error", pe, x["p_subject_given_error"]), ("p_correct", pc, x["p_subject_given_correct"]),
                                        ("odds_ratio", orr, x["odds_ratio"]), ("auroc", auc, x["auroc"])]:
                    if abs(got - want) > 1e-9 * max(1.0, abs(want)):
                        problems.append(f"error discrimination {u}/{pool}/{cell} {name}: {got} vs {want}")
    return problems


def bias_noise():
    """Eight-donor SD and centroid bias at near-threshold sites for the GPT-2 agreement panel, recomputed."""
    path = ROOT / "results/posthoc/bias-noise.json"
    if not path.exists():
        return ["bias-noise.json missing"]
    saved = read(path)["study2"]["sva-gpt2-small"]
    task, m, spread, clean, kk = load("sva-gpt2-small")
    c, k = clean.mean(), kk.mean()
    raw = m[..., :64]
    base = np.stack([kk, clean], 1)[:, :, None, None, None]
    per = (raw - base) / (c - k)
    per[:, 1] *= -1
    ind = per.mean(axis=(0, -1))
    mean = effects(task, m[..., 88], c, k)
    sd8 = np.sqrt((per.var(axis=-1) / 8 * 56 / 63).sum(0)) / per.shape[0]
    live = ~(np.abs(m - base).max(axis=(0, -1)) <= 1e-4)
    band = lambda x: (np.abs(x) >= 0.025) & (np.abs(x) <= 0.10)
    near = live & (band(ind) | band(mean))
    got = {"n": int(near.sum()), "sd": float(np.median(sd8[near])), "bias": float(np.median(np.abs(mean - ind)[near]))}
    want = {"n": saved["near_threshold_sites"], "sd": saved["median_sd8_near"], "bias": saved["mean"]["median_bias_near"]}
    return [f"bias-noise {key}: {got[key]} vs {want[key]}" for key in got if abs(got[key] - want[key]) > 1e-9]


def head_gaps():
    """Study 3 head-level gaps in head-spread.npz against Study 3's saved margins."""
    s3 = ROOT.parent / "05-head-confirmation"
    z = np.load(s3 / "results/posthoc/head-spread.npz")
    problems = []
    for setting, key, task in [("sva-end-gpt2-small", "sva-end", "sva"), ("ioi-end-gpt2-small", "ioi-end", "ioi"),
                               ("ioi-s2-gpt2-small", "ioi-s2", "ioi")]:
        rows = read(s3 / f"data/{task}-test.json")[:64]
        gaps = []
        for r in rows:
            with np.load(s3 / "results" / setting / "test" / f"{r['id']}.npz") as a:
                mm = a["margins"] if task == "sva" else a["margins"][None]
                gaps.append(mm[..., 64] - mm[..., :64].mean(-1))
        if np.max(np.abs(np.array(gaps) - z[f"{key}__gap"])) > 1e-12:
            problems.append(f"head gaps differ for {key}")
        if not (np.all(z[f"{key}__head_spread"] >= 0) and np.all(z[f"{key}__resid_spread"] > 0)):
            problems.append(f"invalid spreads for {key}")
    return problems


def main():
    saved = read(ROOT / "results/posthoc/sweep-posthoc.json")
    report, ok = {"panels": {}}, True
    for panel in PANELS:
        values, problems = check_panel(panel, saved["panels"][panel])
        report["panels"][panel] = {"values": values, "problems": problems}
        ok &= not problems
    report["token_group_problems"] = token_groups(saved["ioi_token_groups"])
    ok &= not report["token_group_problems"]
    report["random_circuit_problems"] = random_circuits()
    ok &= not report["random_circuit_problems"]
    for name, fn in [("error_discrimination_problems", error_discrimination), ("bias_noise_problems", bias_noise),
                     ("head_gap_problems", head_gaps)]:
        report[name] = fn()
        ok &= not report[name]
        print(name.replace("_problems", "").replace("_", " "), "ok" if not report[name] else report[name][:3])
    report["passed"] = bool(ok)
    (ROOT / "results/posthoc/verification-posthoc.json").write_text(json.dumps(report, indent=2) + "\n")
    for panel, r in report["panels"].items():
        print(panel, "ok" if not r["problems"] else r["problems"])
    print("token groups", "ok" if not report["token_group_problems"] else report["token_group_problems"])
    print("random circuits", "ok" if not report["random_circuit_problems"] else report["random_circuit_problems"])
    print("passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

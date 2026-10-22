"""Independent check of Study 4's primary numbers from the raw response arrays.

This file imports nothing from the study's analysis code. It recomputes, with separate
plain code, the point estimates and bootstrap intervals of A1 and A2 and the primary
counts B1-B3, and compares them with the saved summaries. Strata and population sizes
come from the screen record, not from the test run. It also confirms that every frozen
file either matches its frozen digest or matches a recorded amendment.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import sys as _release_sys  # release: frozen-file checks accept documented changes
_release_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import release_paths as rp  # noqa: E402

import numpy as np

ROOT = Path(__file__).resolve().parent
POINT_TOL = 1e-9          # point estimates must agree exactly (up to float rounding)
INTERVAL_TOL = 0.01       # independent bootstrap limits within 1 percentage point


def read(p):
    return json.loads(Path(p).read_text())


def frozen_files():
    rec = read(ROOT / "data/freeze.json")["files"]
    amended = {}
    if (ROOT / "AMENDMENTS.json").exists():
        amended = {a["file"]: a["amended_sha256"] for a in read(ROOT / "AMENDMENTS.json")["files"]}
    bad = []
    for rel, digest in rec.items():
        h = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if h != digest and amended.get(rel) != h and not rp.matches(ROOT / rel, digest) \
                and not (rel in amended and rp.matches(ROOT / rel, amended[rel])):
            bad.append(rel)
    return bad


def experiment_a(draws=4000, seed=7):
    rows = read(ROOT / "data/ioi-selected-test.json")
    strata = read(ROOT / "results/reference/screen-selection.json")["strata"]
    by = {}
    flips = 0
    for r in rows:
        if r["stratum_label"] != "pair_error":
            continue
        with np.load(ROOT / "results/reference/test" / f"{r['id']}.npz") as z:
            full = float(z["full_margin"])
            flips += int((full > 0) != (r["screen_margin"] > 0))
            agree = lambda key: float((float(z[key]) > 0) == (full > 0))
            by.setdefault(r["family"], []).append(
                (agree("NPO128__reference__centroid_margin"), agree("N8__reference__centroid_margin"),
                 agree("N8__empty__centroid_margin")))
    pops = {fam: strata[f"{fam}/pair_error"]["population"] for fam in by}
    arrays = {fam: np.array(v) for fam, v in by.items()}

    def estimate(samples):
        total = sum(pops.values())
        m = sum(pops[f] * samples[f].mean(0) for f in samples) / total
        return np.array([m[0] - m[1], m[1] - m[2]])

    point = estimate(arrays)
    rng = np.random.default_rng(seed)
    boot = np.array([estimate({f: a[rng.integers(0, len(a), len(a))] for f, a in arrays.items()}) for _ in range(draws)])
    lo, hi = np.percentile(boot, [1.25, 98.75], axis=0)
    return {"A1": (point[0], [lo[0], hi[0]]), "A2": (point[1], [lo[1], hi[1]])}, flips


def flags_differ(task, m, clean, corrupt, tau=0.05):
    """m: [G, D, L, P, methods with individual first 64 and mean64 last]."""
    ind = m[..., :64].mean(-1).mean(0)
    mean64 = m[..., -1].mean(0)
    c, k = clean.mean(), corrupt.mean()

    def norm(x):
        out = np.empty_like(x)
        if task == "sva":
            out[0] = (x[0] - k) / (c - k)
            out[1] = (c - x[1]) / (c - k)
        else:
            out[0] = (c - x[0]) / (c - k)
        return out
    return int(((np.abs(norm(mean64)) >= tau) != (np.abs(norm(ind)) >= tau)).sum())


def load_panel(task, model, n):
    folder = ROOT / "results/sweep" / f"{task}-{model}" / "test"
    rows = read(ROOT / ("data/sva-test.json" if task == "sva" else "data/ioi-sweep.json"))[:n]
    ms, clean, corrupt = [], [], []
    for r in rows:
        with np.load(folder / f"{r['id']}.npz") as z:
            if task == "sva":
                ms.append(z["margins"])
                clean.append(z["baseline"][0])
                corrupt.append(z["baseline"][1])
            else:
                off = z["margins"].shape[1] - 14
                ms.append(np.concatenate([z["margins"][None, :, off:, :64], z["margins"][None, :, off:, 72:73]], -1))
                clean.append(z["baseline"][0])
                corrupt.append(z["donor_margins"].mean())
    if len(ms) != n:
        raise RuntimeError(f"{task}-{model}: expected {n} rows, found {len(ms)}")
    return np.stack(ms), np.array(clean), np.array(corrupt)


def experiment_b():
    out = {}
    for panel, task, model, n in [("sva-gpt2-small", "sva", "gpt2-small", 64), ("sva-qwen3-0.6b", "sva", "qwen3-0.6b", 32),
                                  ("ioi-gpt2-small", "ioi", "gpt2-small", 64)]:
        m, clean, corrupt = load_panel(task, model, n)
        out[panel] = flags_differ(task, m, clean, corrupt)
        if task == "ioi":
            # The IOI calibration and development runs used the first prompts of this panel.
            out["ioi-gpt2-small/without_first_two_prompts"] = flags_differ(task, m[2:], clean[2:], corrupt[2:])
    return out


def main():
    problems = []
    bad = frozen_files()
    if bad:
        problems.append(f"frozen files changed without a recorded amendment: {bad}")
    a, flips = experiment_a()
    saved = read(ROOT / "results/reference/analysis-test.json")["primary"]
    for key, name in [("A1", "A1_npo128_minus_n8_error_agreement_mean_ablation"),
                      ("A2", "A2_circuit_minus_empty_error_agreement_n8_mean_ablation")]:
        point, (lo, hi) = a[key]
        s = saved[name]
        if abs(point - s["estimate"]) > POINT_TOL:
            problems.append(f"{key}: independent point {point:.6f} vs saved {s['estimate']:.6f}")
        if abs(lo - s["ci"][0]) > INTERVAL_TOL or abs(hi - s["ci"][1]) > INTERVAL_TOL:
            problems.append(f"{key}: independent interval [{lo:.4f}, {hi:.4f}] vs saved {s['ci']}")
    b = experiment_b()
    saved_b = read(ROOT / "results/sweep/analysis-test.json")["primary"]
    for panel in ["sva-gpt2-small", "sva-qwen3-0.6b", "ioi-gpt2-small"]:
        s = saved_b[f"B_{panel}_mean64_misclassified_sites_tau0.05"]["estimate"]
        if b[panel] != s:
            problems.append(f"{panel}: independent {b[panel]} vs saved {s}")
    report = {"frozen_files_unchanged_or_amended": not bad, "screen_vs_test_sign_flips_among_selected_errors": flips,
              "A": {k: {"estimate": v[0], "ci_independent_bootstrap": v[1]} for k, v in a.items()},
              "B_misclassified_sites": b, "problems": problems, "passed": not problems}
    (ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()

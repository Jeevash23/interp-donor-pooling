"""Independent check of Study 4, written after the test runs; imports nothing from the study's code.

From the raw arrays it recomputes, for every kind of site, the IND, MEAN, TOK and MIX8 effects, the
misclassified, false-positive and false-negative counts at tau = 0.02, 0.05 and 0.10, the mean absolute
errors and the mean absolute centroid bias, and compares them with results/analysis-test.json. It redraws
the eight-donor comparison and the P1 bootstrap with its own seeds and compares them within tolerances.
It also checks the freeze, calibration, completion audits, fresh-process replays and prompt separation.
Output: results/verification.json (exit code 1 if anything differs).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
import sys as _release_sys  # release: frozen-file checks accept documented changes
_release_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import release_paths as rp  # noqa: E402

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
KINDS = ["resid", "mlp", "attn", "head"]
TOL, DRAWS, BOOT, SEED = 1e-9, 300, 2000, 77


def read(p):
    return json.loads(Path(p).read_text())


def effects(g, c, k):
    """g [2, S] group-averaged margins -> normalized effects."""
    e = np.empty_like(g)
    e[0] = (g[0] - k) / (c - k)
    e[1] = (c - g[1]) / (c - k)
    return e


def flags(e, ref, tau):
    a, b = np.abs(e) >= tau, np.abs(ref) >= tau
    return int((a != b).sum()), int((a & ~b).sum()), int((~a & b).sum())


def prompt_texts(obj, out):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("text", "prefix") and isinstance(value, str):
                out.add(value)
            elif key != "encoded":
                prompt_texts(value, out)
    elif isinstance(obj, list):
        for value in obj:
            prompt_texts(value, out)
    return out


def separation():
    test = prompt_texts(read(ROOT / "data/sva-test.json"), set())
    own = set()
    for name in ["sva-calibration.json", "sva-development.json"]:
        prompt_texts(read(ROOT / "data" / name), own)
    prior, seen = set(), set()
    studies = PROJECT / "studies"
    for folder in [studies / "01-pilot/data", PROJECT / "submission-study/data", studies / "03-grouping-and-circuit/token/data",
                   studies / "03-grouping-and-circuit/circuit/data", studies / "04-reference-and-sweeps/data",
                   studies / "05-head-confirmation/data"]:
        for path in sorted(folder.glob("*.json")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                prompt_texts(read(path), prior)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    banks = studies / "04-reference-and-sweeps/results/reference/donor-banks-selected.json.gz"
    for entry in json.load(gzip.open(banks, "rt"))["banks"].values():
        for donors in entry.values():
            prior.update(donors)
    return {"test_strings": len(test), "overlap_with_calibration_or_development": len(test & own),
            "prior_strings": len(prior), "overlap_with_earlier_studies": len(test & prior)}


def main():
    report = {"frozen_files_changed": [rel for rel, d in read(ROOT / "data/freeze.json")["files"].items()
                                       if not rp.matches(ROOT / rel, d)]}
    saved = read(ROOT / "results/analysis-test.json")
    rows = read(ROOT / "data/sva-test.json")[:64]
    data = {k: [] for k in KINDS}
    clean, corrupt, strata = [], [], []
    for r in rows:
        with np.load(ROOT / "results/test" / f"{r['id']}.npz") as z:
            for kind in KINDS:
                data[kind].append(z[f"margins_{kind}"].reshape(2, -1, 81))
            clean.append(float(z["baseline"][0]))
            corrupt.append(float(z["baseline"][1]))
        strata.append((r["template"], r["class"]))
    clean, corrupt = np.array(clean), np.array(corrupt)
    c, k = clean.mean(), corrupt.mean()
    rng = np.random.default_rng(SEED)
    idx = {s: np.array([i for i, x in enumerate(strata) if x == s]) for s in sorted(set(strata))}
    picks = [np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()]) for _ in range(BOOT)]
    problems, values = [], {}
    for kind in KINDS:
        m = np.stack(data[kind])
        s = saved["kinds"][kind]
        e = {"IND": effects(m[..., :64].mean(-1).mean(0), c, k), "MEAN": effects(m[..., 64].mean(0), c, k),
             "TOK": effects(m[..., 65:73].mean(-1).mean(0), c, k), "MIX8": effects(m[..., 73:81].mean(-1).mean(0), c, k)}
        for name in ["MEAN", "TOK", "MIX8"]:
            mae = float(np.abs(e[name] - e["IND"]).mean())
            if abs(mae - s["methods"][name]["mae"]) > TOL:
                problems.append(f"{kind} {name} MAE {mae} vs {s['methods'][name]['mae']}")
            for tau in [0.02, 0.05, 0.10]:
                got = flags(e[name], e["IND"], tau)
                t = s["methods"][name][f"tau{tau}"]
                if got != (t["misclassified"], t["false_positive"], t["false_negative"]):
                    problems.append(f"{kind} {name} tau {tau}: {got} vs saved")
        bias = float(np.abs(e["MEAN"] - e["IND"]).mean())
        if abs(bias - s["mean_abs_bias"]) > TOL:
            problems.append(f"{kind} mean bias {bias} vs {s['mean_abs_bias']}")
        maes = []
        raw = m[..., :64]
        for _ in range(DRAWS):
            pk = np.argsort(rng.random((len(rows), 2, 64)), axis=-1)[..., :8]
            gm = np.take_along_axis(raw, pk[:, :, None, :], axis=-1).mean(-1)
            maes.append(float(np.abs(effects(gm.mean(0), c, k) - e["IND"]).mean()))
        if abs(np.mean(maes) - s["methods"]["S8"]["mae_mean"]) > 0.05 * s["methods"]["S8"]["mae_mean"]:
            problems.append(f"{kind} S8 MAE {np.mean(maes)} vs {s['methods']['S8']['mae_mean']}")
        counts = []
        for pick in picks:
            cb, kb = clean[pick].mean(), corrupt[pick].mean()
            counts.append(flags(effects(m[pick, ..., 64].mean(0), cb, kb), effects(m[pick, ..., :64].mean(-1).mean(0), cb, kb), 0.05)[0])
        conf = s["p1_mean_misclassified"]["confidence"]
        lo, hi = np.percentile(counts, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
        if abs(lo - s["p1_mean_misclassified"]["interval"][0]) > 1 or abs(hi - s["p1_mean_misclassified"]["interval"][1]) > 1:
            problems.append(f"{kind} P1 interval [{lo}, {hi}] vs {s['p1_mean_misclassified']['interval']}")
        values[kind] = {"mean_abs_bias": bias, "s8_mae_redrawn": float(np.mean(maes)), "p1_interval_redrawn": [float(lo), float(hi)]}
    report["values"], report["problems"] = values, problems
    comp = read(ROOT / "results/test-completion.json")
    report["completion_rows"] = comp["rows"]
    report["write_audits_nonzero"] = sum(1 for a in comp["audits"].values() if a["outside"] != 0 or a["write"] != 0)
    rep = read(ROOT / "results/replay-verification.json")
    errs = []
    for path in sorted((ROOT / "results/replay").glob("*.npz")):
        with np.load(path) as a, np.load(ROOT / "results/test" / path.name) as b:
            errs.append(max(float(np.max(np.abs(a[x] - b[x]))) for x in a.files if x.startswith("margins")))
    report["replay_max_error_recomputed"], report["replay_saved"] = max(errs), rep["maximum_error"]
    report["calibration_passed"] = read(ROOT / "results/calibration.json")["passed"]
    report["separation"] = separation()
    report["passed"] = bool(not report["frozen_files_changed"] and not problems and comp["rows"] == 64
                            and report["write_audits_nonzero"] == 0 and len(errs) == 2 and max(errs) <= 1e-3
                            and report["calibration_passed"]
                            and report["separation"]["overlap_with_calibration_or_development"] == 0
                            and report["separation"]["overlap_with_earlier_studies"] == 0)
    (ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "values"}, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

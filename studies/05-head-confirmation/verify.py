"""Independent check of the confirmation study (Study 3), written after the test runs.

This file imports nothing from the study's code. From the raw arrays it recomputes, for every
setting, the IND, MEAN, TOK and MIX8 maps, the misclassified, false-positive and false-negative
counts at tau = 0.02, 0.05 and 0.10, and the mean absolute errors, and compares them with
results/analysis-test.json. It redraws S8 and the primary bootstrap with its own seeds and
compares them within tolerances. It also checks the freeze digests, the completion records and
write audits, the fresh-process replays and the separation of test prompts from calibration,
development and every earlier study.
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
SETTINGS = {"sva-end-gpt2-small": ("sva", 64), "ioi-end-gpt2-small": ("ioi", 64), "ioi-s2-gpt2-small": ("ioi", 64)}
TAUS = [0.02, 0.05, 0.10]
POINT_TOL = 1e-9
S8_DRAWS, BOOT, SEED = 300, 2000, 99


def read(p):
    return json.loads(Path(p).read_text())


def frozen_files():
    rec = read(ROOT / "data/freeze.json")["files"]
    return [rel for rel, digest in rec.items() if not rp.matches(ROOT / rel, digest)]


def load(setting):
    task, n = SETTINGS[setting]
    rows = read(ROOT / f"data/{task}-test.json")[:n]
    folder = ROOT / "results" / setting / "test"
    ind, mean, tok, mix, clean, corrupt, strata = [], [], [], [], [], [], []
    for r in rows:
        with np.load(folder / f"{r['id']}.npz") as z:
            m = z["margins"] if task == "sva" else z["margins"][None]
            ind.append(m[..., :64].mean(-1))
            mean.append(m[..., 64])
            if task == "sva":
                tok.append(m[..., 65:73].mean(-1))
                mix.append(m[..., 73:81].mean(-1))
                clean.append(z["baseline"][0])
                corrupt.append(z["baseline"][1])
                strata.append((r["template"], r["class"]))
            else:
                mix.append(m[..., 65:73].mean(-1))
                k = int(z["token_groups_evaluated"])
                if k:
                    w = z["token_group_sizes"].astype(float)
                    tok.append(m[..., 73:73 + k] @ w / w.sum())
                else:
                    tok.append(m[..., 64])
                clean.append(z["baseline"][0])
                corrupt.append(z["donor_margins"].mean())
                strata.append((r["template"], r["order"]))
    return task, rows, {"IND": np.array(ind), "MEAN": np.array(mean), "TOK": np.array(tok), "MIX8": np.array(mix)}, \
        np.array(clean), np.array(corrupt), strata


def effect(task, group_means, c, k):
    """group_means [D, L, H] averaged over recipients -> normalized effects."""
    e = np.empty_like(group_means)
    if task == "sva":
        e[0] = (group_means[0] - k) / (c - k)
        e[1] = (c - group_means[1]) / (c - k)
    else:
        e[0] = (c - group_means[0]) / (c - k)
    return e


def flags(e, e_ref, tau):
    a, b = np.abs(e) >= tau, np.abs(e_ref) >= tau
    return int((a != b).sum()), int((a & ~b).sum()), int((~a & b).sum())


def check_setting(setting, saved):
    task, rows, arrays, clean, corrupt, strata = load(setting)
    c, k = clean.mean(), corrupt.mean()
    e = {name: effect(task, a.mean(0), c, k) for name, a in arrays.items()}
    problems = []
    for name in ["MEAN", "TOK", "MIX8"]:
        s = saved["methods"][name]
        mae = float(np.abs(e[name] - e["IND"]).mean())
        if abs(mae - s["mae"]) > POINT_TOL:
            problems.append(f"{name} MAE {mae} vs saved {s['mae']}")
        for tau in TAUS:
            mis, fp, fn = flags(e[name], e["IND"], tau)
            t = s[f"tau{tau}"]
            if (mis, fp, fn) != (t["misclassified"], t["false_positive"], t["false_negative"]):
                problems.append(f"{name} tau {tau}: {(mis, fp, fn)} vs saved {(t['misclassified'], t['false_positive'], t['false_negative'])}")
    # S8 redrawn with a separate seed: the mean error and mean count must agree closely.
    rng = np.random.default_rng(SEED)
    folder = ROOT / "results" / setting / "test"
    raw = []
    for r in rows:
        with np.load(folder / f"{r['id']}.npz") as z:
            raw.append((z["margins"] if task == "sva" else z["margins"][None])[..., :64])
    raw = np.array(raw)
    G, D = raw.shape[:2]
    maes, mis5 = [], []
    for _ in range(S8_DRAWS):
        pick = np.argsort(rng.random((G, D, 64)), axis=-1)[..., :8]
        gm = np.take_along_axis(raw, pick[:, :, None, None, :], axis=-1).mean(-1)
        es = effect(task, gm.mean(0), c, k)
        maes.append(float(np.abs(es - e["IND"]).mean()))
        mis5.append(flags(es, e["IND"], 0.05)[0])
    s8 = saved["methods"]["S8"]
    if abs(np.mean(maes) - s8["mae_mean"]) > 0.05 * s8["mae_mean"]:
        problems.append(f"S8 MAE {np.mean(maes)} vs saved {s8['mae_mean']}")
    if abs(np.mean(mis5) - s8["tau0.05"]["misclassified_mean"]) > 0.5:
        problems.append(f"S8 misclassified {np.mean(mis5)} vs saved {s8['tau0.05']['misclassified_mean']}")
    # Primary interval: stratified bootstrap over recipients with its own seed; limits within one count.
    keys = sorted(set(strata))
    idx = {s: np.array([i for i, x in enumerate(strata) if x == s]) for s in keys}
    counts = []
    for _ in range(BOOT):
        pick = np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()])
        cb, kb = clean[pick].mean(), corrupt[pick].mean()
        eb = effect(task, arrays["IND"][pick].mean(0), cb, kb)
        mb = effect(task, arrays["MEAN"][pick].mean(0), cb, kb)
        counts.append(flags(mb, eb, 0.05)[0])
    lo, hi = np.percentile(counts, [100 * 0.05 / 6, 100 * (1 - 0.05 / 6)])
    pm = saved["primary_mean_misclassified"]
    if flags(e["MEAN"], e["IND"], 0.05)[0] != pm["estimate"]:
        problems.append("primary estimate differs")
    if abs(lo - pm["ci"][0]) > 1 or abs(hi - pm["ci"][1]) > 1:
        problems.append(f"primary interval [{lo}, {hi}] vs saved {pm['ci']}")
    # Completion, write audits and replays.
    comp = read(ROOT / "results" / setting / "test-completion.json")
    if comp["rows"] != SETTINGS[setting][1]:
        problems.append(f"completion rows {comp['rows']}")
    if any(a["outside"] != 0 or a["write"] != 0 for a in comp["audits"].values()):
        problems.append("write audit nonzero")
    rep = ROOT / "results" / setting / "replay"
    errs = []
    for path in sorted(rep.glob("*.npz")):
        with np.load(path) as a, np.load(folder / path.name) as b:
            errs.append(float(np.max(np.abs(a["margins"] - b["margins"]))))
    if len(errs) != 2 or max(errs) > 1e-3:
        problems.append(f"replay errors {errs}")
    return {"problems": problems, "s8_mae_redrawn": float(np.mean(maes)), "primary_interval_redrawn": [float(lo), float(hi)],
            "replay_max_error": max(errs) if errs else None}


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
    test = set()
    for name in ["sva-test.json", "ioi-test.json"]:
        prompt_texts(read(ROOT / "data" / name), test)
    rows = read(ROOT / "data/sva-test.json")
    for r in rows:                                   # agreement prompt texts live under "encoded"
        for p in r["encoded"]["gpt2-small"]:
            test.add(p["text"])
    own = set()
    for name in ["sva-calibration.json", "sva-development.json", "ioi-calibration.json", "ioi-development.json"]:
        data = read(ROOT / "data" / name)
        prompt_texts(data, own)
        if name.startswith("sva"):
            for r in data:
                for p in r["encoded"]["gpt2-small"]:
                    own.add(p["text"])
    prior, seen = set(), set()
    studies = PROJECT / "studies"
    folders = [studies / "01-pilot/data", PROJECT / "submission-study/data", studies / "03-grouping-and-circuit/token/data",
               studies / "03-grouping-and-circuit/circuit/data", studies / "04-reference-and-sweeps/data"]
    for folder in folders:
        for path in sorted(folder.glob("*.json")):
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
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
    report = {"frozen_files_changed": frozen_files()}
    saved = read(ROOT / "results/analysis-test.json")["settings"]
    report["settings"] = {s: check_setting(s, saved[s]) for s in SETTINGS}
    report["separation"] = separation()
    gate = read(ROOT / "results/qwen-gate/gate.json")
    report["qwen_gate"] = {"accuracy": gate["clean_two_verb_accuracy"], "required": gate["required"], "passed": gate["passed"]}
    ok = (not report["frozen_files_changed"] and all(not v["problems"] for v in report["settings"].values())
          and report["separation"]["overlap_with_calibration_or_development"] == 0
          and report["separation"]["overlap_with_earlier_studies"] == 0)
    report["passed"] = ok
    (ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Independent check of Study 5; imports nothing from the study's code (only numpy and the standard library).

Two parts:
  arrays()  runs in a Modal CPU container next to the raw arrays (called by modal_app.py after the analysis):
            every array's SHA-256 against its run record; from the raw margins it recomputes, with its own code,
            the effects and the primary counts of R1, R2 and R3, redraws the bootstrap intervals with its own
            seeds, and compares them with the saved analysis (counts and biases exactly; intervals within one site
            or a tenth of their width; the P3 intervals by sign). Output: verification-arrays.json in the analysis
            folder.
  local     python verify.py   (local machine, after the records are pulled back): the three freezes, prompt separation
            from every earlier AI-03 study, calibration, gates, completion records, write audits and replays, and the
            array part. Output: results/verification.json; exit code 1 if anything fails.
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

HERE = Path(__file__).resolve().parent
TOL = 1e-9
SEED = 5170
KIND_ORDER = ["resid", "mlp", "attn", "head"]


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def strata_draws(labels, draws, rng):
    """Resample indices within strata; returns a list of index arrays."""
    groups = {}
    for i, s in enumerate(labels):
        groups.setdefault(s, []).append(i)
    groups = [np.array(v) for _, v in sorted(groups.items())]
    return [np.concatenate([rng.choice(ix, len(ix)) for ix in groups]) for _ in range(draws)]


def two_way(mean_margins, clean, corrupt):
    """[D=2, ...] restore/destroy effects; mean_margins already averaged over groups."""
    e = np.empty_like(mean_margins)
    e[0] = (mean_margins[0] - corrupt) / (clean - corrupt)
    e[1] = (clean - mean_margins[1]) / (clean - corrupt)
    return e


def noising(mean_margins, clean, donor_mean):
    return (clean - mean_margins) / (clean - donor_mean)


def close(lo, hi, saved):
    """Redrawn interval against the saved one: within one site or a tenth of the saved width (Monte Carlo error)."""
    tol = max(1.0, 0.1 * (saved[1] - saved[0]))
    return abs(lo - saved[0]) <= tol and abs(hi - saved[1]) <= tol


def mismatch(a, b, tau=0.05):
    return int(((np.abs(a) >= tau) != (np.abs(b) >= tau)).sum())


def arrays(folder, analysis_dir, meta_path, out_path):
    folder, analysis_dir = Path(folder), Path(analysis_dir)
    meta = read(meta_path)
    problems, values = [], {}
    # 1. Every array against its run record.
    hashes = 0
    for rec_path in sorted(p for panel in ("A", "B1", "B2", "C1", "C2") for p in (folder / panel).glob("*.json")):
        if rec_path.name.endswith(".failed.json"):
            problems.append(f"failed record present: {rec_path}")
            continue
        rec = read(rec_path)
        if digest(rec_path.with_suffix(".npz")) != rec["sha256"]:
            problems.append(f"hash mismatch {rec_path}")
        hashes += 1
    values["arrays_hashed"] = hashes
    rng = np.random.default_rng(SEED)
    sva, ioi = meta["sva"], meta["ioi"]
    # 2. R1 (panel A) and the agreement head setting.
    if sva and (folder / "A").exists():
        saved = read(analysis_dir / "analysis-A-test.json")
        files = [np.load(folder / "A" / f"{r['id']}.npz") for r in sva]
        clean = np.array([f["baseline"][0] for f in files])
        corrupt = np.array([f["baseline"][1] for f in files])
        labels = [(r["template"], r["class"]) for r in sva]
        draws = strata_draws(labels, 2000, rng)
        bias, spread, boot_bias = {}, {}, {}
        for kind in KIND_ORDER:
            m = np.stack([f[f"margins_{kind}"].reshape(2, -1, 81) for f in files])        # [G, 2, S, 81]
            sp = np.stack([f[f"spread_{kind}"].reshape(2, -1, 17)[..., 0] for f in files]).mean(0)
            ind = two_way(m[..., :64].mean(-1).mean(0), clean.mean(), corrupt.mean())
            mean = two_way(m[..., 64].mean(0), clean.mean(), corrupt.mean())
            bias[kind] = float(np.abs(mean - ind).mean())
            spread[kind] = float(np.median(sp))
            if abs(bias[kind] - saved["kinds"][kind]["mean_abs_bias"]) > TOL:
                problems.append(f"A {kind} bias {bias[kind]} vs {saved['kinds'][kind]['mean_abs_bias']}")
            if abs(spread[kind] - saved["kinds"][kind]["spread"]["median"]) > 1e-6 * max(1, spread[kind]):
                problems.append(f"A {kind} spread {spread[kind]} vs {saved['kinds'][kind]['spread']['median']}")
            est = mismatch(mean, ind)
            if est != saved["kinds"][kind]["p1_mean_misclassified"]["estimate"]:
                problems.append(f"A {kind} misclassified {est} vs saved")
            bb = []
            for pick in draws:
                c, k = clean[pick].mean(), corrupt[pick].mean()
                bb.append(np.abs(two_way(m[pick, ..., 64].mean(0), c, k) - two_way(m[pick, ..., :64].mean(-1).mean(0), c, k)).mean())
            boot_bias[kind] = np.array(bb)
        p3 = {}
        for a, b in [("resid", "mlp"), ("mlp", "attn"), ("attn", "head")]:
            lo, hi = np.percentile(boot_bias[a] - boot_bias[b], [2.5, 97.5])
            p3[f"{a}-{b}"] = [float(lo), float(hi)]
            saved_lo = saved["p3_ordering"][f"{a}-{b}"]["ci95"][0]
            if (lo > 0) != (saved_lo > 0):
                problems.append(f"P3 {a}-{b}: redrawn lower {lo} vs saved {saved_lo} differ in sign")
        by_bias = sorted(KIND_ORDER, key=lambda x: -bias[x])
        by_spread = sorted(KIND_ORDER, key=lambda x: -spread[x])
        if by_bias != saved["p4_spread_order"]["kinds_by_bias"] or by_spread != saved["p4_spread_order"]["kinds_by_spread"]:
            problems.append("P4 orders differ")
        values["R1"] = dict(mean_abs_bias=bias, median_spread=spread, p3_ci95_redrawn=p3, kinds_by_bias=by_bias,
                            kinds_by_spread=by_spread,
                            replicates=bool(all(v[0] > 0 for v in p3.values()) and by_bias == by_spread == KIND_ORDER))
    # 3. R2: head settings.
    heads_saved = read(analysis_dir / "analysis-heads-test.json")["settings"] if (analysis_dir / "analysis-heads-test.json").exists() else {}
    r2 = {}
    for setting, panel, rows in (("sva-end", "A", sva), ("ioi-end", "B1", ioi), ("ioi-s2", "B2", ioi)):
        if not rows or setting not in heads_saved:
            continue
        files = [np.load(folder / panel / f"{r['id']}.npz") for r in rows]
        if setting == "sva-end":
            m = np.stack([f["margins_head"] for f in files])                              # [G, 2, L, H, 81]
            clean = np.array([f["baseline"][0] for f in files])
            other = np.array([f["baseline"][1] for f in files])
            labels = [(r["template"], r["class"]) for r in rows]
            eff = lambda mm, c, k: two_way(mm, c, k)                                       # noqa: E731
        else:
            m = np.stack([f["margins"][None, ..., :73] for f in files])                    # [G, 1, L, H, 73]
            clean = np.array([f["baseline"][0] for f in files])
            other = np.array([f["donor_margins"].mean() for f in files])
            labels = [(r["template"], r["order"]) for r in rows]
            eff = lambda mm, c, k: noising(mm, c, k)                                       # noqa: E731
        ind_g, mean_g = m[..., :64].mean(-1), m[..., 64]
        est = mismatch(eff(mean_g.mean(0), clean.mean(), other.mean()), eff(ind_g.mean(0), clean.mean(), other.mean()))
        counts = []
        for pick in strata_draws(labels, 5000, rng):
            c, k = clean[pick].mean(), other[pick].mean()
            counts.append(mismatch(eff(mean_g[pick].mean(0), c, k), eff(ind_g[pick].mean(0), c, k)))
        lo, hi = np.percentile(counts, [100 * 0.05 / 6, 100 * (1 - 0.05 / 6)])
        s = heads_saved[setting]["primary_mean_misclassified"]
        if est != s["estimate"]:
            problems.append(f"R2 {setting} estimate {est} vs {s['estimate']}")
        if not close(lo, hi, s["ci"]):
            problems.append(f"R2 {setting} interval [{lo}, {hi}] vs {s['ci']}")
        n = int(ind_g[0].size)
        r2[setting] = dict(estimate=est, ci_redrawn=[float(lo), float(hi)], head_sites=n,
                           verdict_redrawn="contradicts" if lo >= 1 else ("replicates" if hi <= int(0.02 * n) else "inconclusive"))
    values["R2"] = r2
    # 4. R3: residual sweeps.
    sweeps_saved = read(analysis_dir / "analysis-sweeps-test.json")["panels"] if (analysis_dir / "analysis-sweeps-test.json").exists() else {}
    r3 = {}
    for name, panel, rows in (("sva-qwen3-1.7b", "C1", sva), ("ioi-qwen3-1.7b", "C2", ioi)):
        if not rows or name not in sweeps_saved:
            continue
        files = [np.load(folder / panel / f"{r['id']}.npz") for r in rows]
        if panel == "C1":
            m = np.stack([f["margins"] for f in files])                                    # [G, 2, L, P, 89]
            clean = np.array([f["baseline"][0] for f in files])
            other = np.array([f["baseline"][1] for f in files])
            labels = [(r["template"], r["class"]) for r in rows]
            ind_g, mean_g, eff = m[..., :64].mean(-1), m[..., 88], two_way
        else:
            m = np.stack([f["margins"][:, -14:] for f in files])                           # [G, L, 14, 73]
            clean = np.array([f["baseline"][0] for f in files])
            other = np.array([f["donor_margins"].mean() for f in files])
            labels = [(r["template"], r["order"]) for r in rows]
            ind_g, mean_g, eff = m[..., :64].mean(-1), m[..., 72], noising
        est = mismatch(eff(mean_g.mean(0), clean.mean(), other.mean()), eff(ind_g.mean(0), clean.mean(), other.mean()))
        counts = []
        for pick in strata_draws(labels, 10000, rng):
            c, k = clean[pick].mean(), other[pick].mean()
            counts.append(mismatch(eff(mean_g[pick].mean(0), c, k), eff(ind_g[pick].mean(0), c, k)))
        lo, hi = np.percentile(counts, [1.25, 98.75])
        s = sweeps_saved[name]["flags"]["mean64/tau0.05"]["misclassified_sites"]
        if est != s["estimate"]:
            problems.append(f"R3 {name} estimate {est} vs {s['estimate']}")
        if not close(lo, hi, s["ci"]):
            problems.append(f"R3 {name} interval [{lo}, {hi}] vs {s['ci']}")
        r3[name] = dict(estimate=est, ci_redrawn=[float(lo), float(hi)], sites=int(ind_g[0].size), replicates_redrawn=bool(lo >= 1))
    values["R3"] = r3
    report = dict(values=values, problems=problems, passed=not problems)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


# ---------------------------------------------------------------------------------------- local part
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
    project = HERE.parents[1]
    studies = project / "studies"
    test = set()
    for name in ("sva-test.json", "ioi-test.json"):
        prompt_texts(read(HERE / "data" / name), test)
    own = set()
    for name in ("sva-calibration.json", "sva-development.json", "ioi-calibration.json", "ioi-development.json"):
        prompt_texts(read(HERE / "data" / name), own)
    prior, seen = set(), set()
    for folder in [studies / "01-pilot/data", project / "submission-study/data", studies / "03-grouping-and-circuit/token/data",
                   studies / "03-grouping-and-circuit/circuit/data", studies / "04-reference-and-sweeps/data",
                   studies / "05-head-confirmation/data", studies / "06-component-confirmation/data"]:
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
    portfolio = HERE.parents[2]
    report, problems = {}, []
    changed = {}
    later = {}                                                            # hashes recorded by amendments (AMENDMENTS.md)
    amendments = sorted((HERE / "data").glob("input-freeze-amendment-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    for path in amendments:
        later.update(read(path)["files"])
    for name in ["protocol-freeze.json", "input-freeze.json", "stage4-freeze.json"] + [p.name for p in amendments]:
        files = read(HERE / "data" / name)["files"]
        copy = HERE / "provenance/README-at-protocol-freeze.md"           # the README is updated as the study runs
        allowed = later if name.startswith("input-freeze") else {}       # later amendments supersede earlier records
        changed[name] = [r for r, h in files.items() if not rp.matches(rp.resolve_key(r), h, r)
                         and not (r.endswith("07-larger-model/README.md") and copy.exists() and rp.matches(copy, h))
                         and not (r in allowed and rp.matches(rp.resolve_key(r), allowed[r], r))]
        problems += [f"{name}: {r} changed" for r in changed[name]]
    report["freezes_changed"] = changed
    report["separation"] = sep = separation()
    if sep["overlap_with_calibration_or_development"] or sep["overlap_with_earlier_studies"]:
        problems.append("test prompts overlap other prompts")
    cal = read(HERE / "results/calibration/calibration.json")["calibration"]
    report["calibration"] = {k: cal[k] for k in ("maximum_logit_error", "maximum_probability_error",
                                                 "maximum_engine_difference", "exact_write_locality", "passed")}
    if not cal["passed"]:
        problems.append("calibration did not pass")
    gates = read(HERE / "results/gates/gates.json")["gates"]
    report["gates"] = {"agreement": gates["agreement"]["passed"], "ioi": gates["ioi"]["passed"]}
    decision = read(HERE / "results/cost-decision.json")
    expected = {"A": decision["agreement_groups"], "C1": decision["agreement_groups"]} if gates["agreement"]["passed"] else {}
    if gates["ioi"]["passed"]:
        expected.update({"B1": decision["ioi_prompts"], "B2": decision["ioi_prompts"], "C2": decision["ioi_prompts"]})
    comp = read(HERE / "results/test-completion.json")
    counts, bad_audit, worst = {}, 0, 0.0
    for p in sorted((HERE / "results/test").glob("*/*.json")):
        rec = read(p)
        counts[rec["panel"]] = counts.get(rec["panel"], 0) + 1
        bad_audit += int(rec["audit"]["outside"] != 0 or rec["audit"]["write"] != 0 or not rec["passed"])
        worst = max(worst, rec["max_check_error"])
        if comp["records"].get(f"{p.parent.name}/{p.stem}") != rec["sha256"]:
            problems.append(f"completion record differs for {p}")
    report["test_rows"], report["records_failing"], report["largest_built_in_check_error"] = counts, bad_audit, worst
    if counts != expected or bad_audit or comp["failures"]:
        problems.append(f"test rows {counts} against {expected}, failing records {bad_audit}, failures {comp['failures']}")
    rep = read(HERE / "results/replay-verification.json")
    report["replays"] = {k: rep[k] for k in ("maximum_error_primary", "maximum_error_l4", "passed")}
    if not rep["passed"]:
        problems.append("replays outside tolerance")
    arr = read(HERE / "results/analysis/verification-arrays.json")
    report["arrays"] = arr
    problems += [f"arrays: {p}" for p in arr["problems"]]
    report["problems"], report["passed"] = problems, not problems
    (HERE / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "arrays"}, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

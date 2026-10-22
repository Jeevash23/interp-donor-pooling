"""Post-test descriptive QA; does not change frozen inference or measurements."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import read, dump, sha, utc, freeze_check
from analyze import bootstrap_draws, estimate

def check(model, filename):
    completed = read(ROOT / "results" / model / "test-completion.json")
    assert completed["groups"] == 64
    frozen = freeze_check()
    summary_file = ROOT / "results/analysis-test" / f"{model}.json"
    summary = read(summary_file)
    rows = read(ROOT / "data/test.json")
    draws = bootstrap_draws(rows)
    with np.load(ROOT / "results/analysis-test" / f"{model}-components.npz") as archive:
        comp = {key: archive[key] for key in ["gap", "mean", "individual", "propagation", "normalization", "scoring", "primary_recipient_values"]}
    end_gap = np.take(comp["gap"], [1, 3, 5], axis=1)
    contrasts = abs(end_gap[:, :, :, 3, :, 0]) - abs(end_gap[:, :, :, 0, :, 0])
    direct_primary = contrasts.mean(axis=(1, 2, 3))
    assert np.array_equal(direct_primary, comp["primary_recipient_values"])
    reconstructed = estimate(contrasts, draws, alpha=.025)
    assert reconstructed == summary["primary"]
    curves = {metric: {str(k): summary["secondary_dose_curve"][str(k)][field] for k in [1, 2, 4, 8]} for metric, field in [("absolute_margin_gap", "margin_gap_absolute"), ("absolute_probability_gap", "probability_gap_absolute"), ("mean_squared_radius", "mean_squared_radius")]}
    direction = {label: estimate(contrasts[:, :, di], draws) for di, label in enumerate(["restore", "destroy"])}
    depth = {str(summary["environment"]["layers"][i]): estimate(contrasts[:, i], draws) for i in range(3)}
    adjacent_probability = {f"{b}-{a}": estimate(abs(end_gap[:, :, :, bi, :, 1]) - abs(end_gap[:, :, :, ai, :, 1]), draws) for ai, a, bi, b in [(0, 1, 1, 2), (1, 2, 2, 4), (2, 4, 3, 8)]}
    adjacent_margin = {label: summary["secondary_pairwise_dose_contrasts"][label] for label in ["2-1", "4-2", "8-4"]}
    cell_exceptions = []
    for site in [1, 3, 5]:
        for di, label in enumerate(["restore", "destroy"]):
            curve = [float(abs(comp["gap"][:, site, di, k, :, 0]).mean()) for k in range(4)]
            if not all(b >= a for a, b in zip(curve, curve[1:])):
                cell_exceptions.append({"layer": summary["environment"]["layers"][site // 2], "direction": label, "absolute_margin_gap_curve": curve})
    audits = [a for row in rows for a in read(ROOT / "results" / model / "test" / (row["id"] + "-audit.json"))]
    maxima = {key: max(a.get(key, 0.) for a in audits) for key in sorted({key for a in audits for key in a if key not in ["layer", "position", "write_norm"]})}
    assert maxima["outside_max"] == maxima["target_error"] == 0
    assert max(value for key, value in maxima.items() if "probability" not in key) <= .001
    assert max(value for key, value in maxima.items() if "probability" in key) <= .00002
    replay_path = ROOT / "results" / model / "replay-verification.json"
    record = {"created_utc": utc(), "scope": "Post-test descriptive QA; no changed endpoints, altered primary analysis or new GPU inference. Direction/depth/probability contrasts are descriptive secondary checks.", "model": model, "groups": len(rows), "frozen_file_and_checkpoint_hashes_passed": True, "freeze_sha256": sha(ROOT / "data/freeze.json"), "freeze_created_utc": frozen["created_utc"], "analyzed_summary_sha256": sha(summary_file), "primary_exact_reconstruction_passed": True, "primary": summary["primary"], "primary_positive_recipients": int((direct_primary > 0).sum()), "primary_min": float(direct_primary.min()), "primary_max": float(direct_primary.max()), "curves": curves, "secondary_adjacent_margin_contrasts": adjacent_margin, "secondary_adjacent_probability_contrasts": adjacent_probability, "secondary_direction_contrasts": direction, "secondary_depth_contrasts": depth, "nonmonotonic_margin_cells": cell_exceptions, "competence": summary["competence"], "numerical_summary": summary["numerical"], "raw_audit_maxima": maxima, "replay": read(replay_path) if replay_path.exists() else {"status": "not yet present at review time"}, "passed": True}
    dump(ROOT / "docs" / filename, record)
    print(json.dumps({"model": model, "primary": record["primary"], "positive_recipients": record["primary_positive_recipients"], "margin_curve": [curves["absolute_margin_gap"][str(k)]["mean"] for k in [1, 2, 4, 8]], "probability_curve": [curves["absolute_probability_gap"][str(k)]["mean"] for k in [1, 2, 4, 8]], "nonmonotonic_margin_cells": cell_exceptions, "passed": True}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt2-small", "qwen3-0.6b"], required=True)
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()
    check(args.model, args.filename)

"""Prospective paired recipient-group analysis; no bank-level pseudoreplication."""
from __future__ import annotations
import argparse
import csv
import numpy as np
from common import *

def bootstrap_draws(rows, count=20000):
    rng = np.random.default_rng(26100299)
    strata = {}
    for i, row in enumerate(rows):
        strata.setdefault((row["template"], row["class"]), []).append(i)
    return np.concatenate([rng.choice(indices, size=(count, len(indices)), replace=True) for _, indices in sorted(strata.items())], axis=1)

def estimate(values, draws, alpha=.05):
    array = np.asarray(values, dtype=np.float64)
    recipient = array.reshape(len(array), -1).mean(axis=1)
    distribution = recipient[draws].mean(axis=1)
    return {"mean": float(recipient.mean()), "ci": np.quantile(distribution, [alpha / 2, 1 - alpha / 2]).tolist(), "confidence": 1 - alpha, "n_recipient_groups": len(array), "bootstrap_draws": len(draws)}

def load(model, split):
    rows = read(ROOT / f"data/{split}.json")
    folder = ROOT / "results" / model
    design = read(folder / "design.json")
    arrays = {key: [] for key in ["metrics", "endpoints", "dispersion", "baseline"]}
    all_audits = []
    for row in rows:
        with np.load(folder / split / (row["id"] + ".npz")) as data:
            assert str(data["id"]) == row["id"]
            assert data["condition_names"].tolist() == design["names"]
            assert data["bank_names"].tolist() == [b["name"] for b in design["banks"]]
            for key in arrays:
                assert np.isfinite(data[key]).all(), (model, row["id"], key)
                arrays[key].append(data[key])
        all_audits.extend(read(folder / split / (row["id"] + "-audit.json")))
    return rows, {key: np.stack(value) for key, value in arrays.items()}, design, all_audits

def components(data, design):
    # Bank ordering is direction, dose, rotation; metrics are margin, probability.
    expected = [(direction, k, rotation) for direction in ["restore", "destroy"] for k in DOSES for rotation in range(8)]
    actual = [(b["direction"], b["k"], b["rotation"]) for b in design["banks"]]
    assert actual == expected
    mean = data["metrics"][:, :, [b["mean"] for b in design["banks"]], :2]
    individual = np.stack([data["metrics"][:, :, b["indices"], :2].mean(axis=2) for b in design["banks"]], axis=2)
    b = data["endpoints"][:, :, :, 0, :2]
    c = data["endpoints"][:, :, :, 1, :2]
    arrays = {"mean": mean, "individual": individual, "gap": mean - individual, "propagation": mean - b, "normalization": b - c, "scoring": c - individual}
    shape = (*mean.shape[:2], 2, 4, 8, 2)
    return {key: value.reshape(shape) for key, value in arrays.items()}

def analyze(model, split="test"):
    rows, data, design, audits = load(model, split)
    draws = bootstrap_draws(rows)
    comp = components(data, design)
    nonterminal_end = [1, 3, 5]
    gap = comp["gap"]
    end_comp = {key: np.take(value, nonterminal_end, axis=1) for key, value in comp.items()}
    end_gap = end_comp["gap"]
    primary = abs(end_gap[:, :, :, -1, :, 0]) - abs(end_gap[:, :, :, 0, :, 0])
    assert primary.shape == (len(rows), 3, 2, 8)
    secondary = {}
    for ki, k in enumerate(DOSES):
        values = end_gap[:, :, :, ki]
        secondary[str(k)] = {
            "margin_gap_absolute": estimate(abs(values[..., 0]), draws),
            "margin_gap_signed": estimate(values[..., 0], draws),
            "probability_gap_absolute": estimate(abs(values[..., 1]), draws),
            "probability_gap_signed": estimate(values[..., 1], draws),
            "propagation_margin_absolute": estimate(abs(end_comp["propagation"][:, :, :, ki, :, 0]), draws),
            "normalization_margin_absolute": estimate(abs(end_comp["normalization"][:, :, :, ki, :, 0]), draws),
            "propagation_probability_absolute": estimate(abs(end_comp["propagation"][:, :, :, ki, :, 1]), draws),
            "normalization_probability_absolute": estimate(abs(end_comp["normalization"][:, :, :, ki, :, 1]), draws),
            "scoring_probability_absolute": estimate(abs(end_comp["scoring"][:, :, :, ki, :, 1]), draws),
            "gap_exceeds_point_one_logit": estimate(abs(values[..., 0]) > .1, draws),
        }
    dispersion = data["dispersion"].reshape(len(rows), 8, 2, 4, 8, 3)
    end_dispersion = np.take(dispersion, nonterminal_end, axis=1)
    for ki, k in enumerate(DOSES):
        secondary[str(k)]["mean_squared_radius"] = estimate(end_dispersion[:, :, :, ki, :, 0], draws)
    dose_contrasts = {f"{b}-{a}": estimate(abs(end_gap[:, :, :, bi, :, 0]) - abs(end_gap[:, :, :, ai, :, 0]), draws) for ai, a in enumerate(DOSES) for bi, b in enumerate(DOSES) if bi > ai}
    cells = []
    for si in range(8):
        for di, direction in enumerate(["restore", "destroy"]):
            for ki, k in enumerate(DOSES):
                cell = {"layer": design["layers"][si // 2], "position": POSITIONS[si % 2], "site": si, "direction": direction, "k": k}
                for key in ["gap", "propagation", "normalization"]:
                    for mi, metric in enumerate(["margin", "probability"]):
                        value = comp[key][:, si, di, ki, :, mi]
                        cell[f"{key}_{metric}_absolute"] = estimate(abs(value), draws)
                        cell[f"{key}_{metric}_signed"] = estimate(value, draws)
                cell["scoring_probability_signed"] = estimate(comp["scoring"][:, si, di, ki, :, 1], draws)
                cell["mean_squared_radius"] = estimate(dispersion[:, si, di, ki, :, 0], draws)
                cells.append(cell)
    baseline = data["baseline"]
    clean_donor_indices = [rows[0]["grid"][i][j][0] for i in range(8) for j in range(8)]
    corrupt_donor_indices = [rows[0]["grid"][i][j][1] for i in range(8) for j in range(8)]
    competence = {"recipient_clean_pair_accuracy": estimate(baseline[:, 0, 3], draws), "recipient_clean_top1_accuracy": estimate(baseline[:, 0, 2], draws), "recipient_corrupt_clean_target_preference": estimate(baseline[:, 1, 3], draws), "recipient_clean_margin": estimate(baseline[:, 0, 0], draws), "recipient_corrupt_clean_target_margin": estimate(baseline[:, 1, 0], draws), "clean_donor_pair_accuracy": estimate(baseline[:, clean_donor_indices, 3], draws), "corrupt_donor_clean_target_preference": estimate(baseline[:, corrupt_donor_indices, 3], draws)}
    pooled_individual = comp["individual"].mean(axis=4)
    pooled_reference = pooled_individual[:, :, :, :1]
    balanced_resample_error = float(abs(pooled_individual - pooled_reference).max())
    anchor_mean = np.take(comp["mean"], [0, 2, 4, 6], axis=1)
    anchor_delta = anchor_mean - anchor_mean[:, :, :, :1, :1, :]
    closure = comp["gap"] - comp["propagation"] - comp["normalization"] - comp["scoring"]
    numerical = {
        "algebraic_closure_max": float(abs(closure).max()),
        "margin_linear_readout_roundoff_max": float(abs(comp["scoring"][..., 0]).max()),
        "global_resample_across_doses_max": balanced_resample_error,
        "anchor_centroid_patch_across_doses_max": float(abs(anchor_delta).max()),
        "terminal_anchor_gap_max": float(abs(gap[:, 6]).max()),
        "terminal_end_propagation_max": float(abs(comp["propagation"][:, 7]).max()),
        "write_outside_max": max(a["outside_max"] for a in audits),
        "write_target_max": max(a["target_error"] for a in audits),
        "identity_logit_max": max(a["identity_logit_max"] for a in audits),
        "identity_probability_max": max(a["identity_probability_max"] for a in audits),
        "grand_centroid_roundoff_max": max(a["grand_centroid_roundoff_max"] for a in audits),
        "causal_prefix_state_max": max(a.get("causal_prefix_state_max", 0) for a in audits),
    }
    assert numerical["algebraic_closure_max"] < 1e-10
    assert balanced_resample_error < 1e-10
    assert numerical["write_outside_max"] == numerical["write_target_max"] == 0
    for key in ["margin_linear_readout_roundoff_max", "anchor_centroid_patch_across_doses_max", "terminal_anchor_gap_max", "terminal_end_propagation_max", "identity_logit_max", "grand_centroid_roundoff_max", "causal_prefix_state_max"]:
        assert numerical[key] <= .001, (key, numerical[key])
    assert numerical["identity_probability_max"] <= .00002
    summary = {"model": model, "split": split, "groups": len(rows), "primary": estimate(primary, draws, alpha=.025), "primary_description": "K8 minus K1 absolute margin gap, paired recipient-group average across 3 nonterminal END depths, both directions, 8 partition rotations", "secondary_confidence": .95, "secondary_dose_curve": secondary, "secondary_pairwise_dose_contrasts": dose_contrasts, "competence": competence, "cells": cells, "numerical": numerical, "interventions": int(np.prod(data["metrics"].shape[:3])), "environment": read(ROOT / "results" / model / f"environment-{split}.json"), "interpretation": "Fixed donor support, different balanced partitions; not isolated token identity or an assertion of monotonicity"}
    folder = ROOT / "results" / f"analysis-{split}"
    dump(folder / f"{model}.json", summary)
    np.savez_compressed(folder / f"{model}-components.npz", **comp, dispersion=dispersion, ids=np.array([r["id"] for r in rows]), primary_recipient_values=primary.reshape(len(rows), -1).mean(1))
    with (folder / f"{model}-recipient-primary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "template", "class", "primary_k8_minus_k1_absolute_gap"])
        for row, value in zip(rows, primary.reshape(len(rows), -1).mean(1)):
            writer.writerow([row["id"], row["template"], row["class"], value])
    print(json.dumps({"model": model, "split": split, "primary": summary["primary"], "numerical": numerical}), flush=True)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS))
    parser.add_argument("--split", choices=["development", "test"], default="test")
    args = parser.parse_args()
    for model in [args.model] if args.model else MODEL_PATHS:
        analyze(model, args.split)

if __name__ == "__main__":
    main()

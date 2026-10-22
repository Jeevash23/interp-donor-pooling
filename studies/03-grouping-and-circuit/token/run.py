"""Auditable residual interventions on matched 8-by-8 donor grids."""
from __future__ import annotations
import argparse
import time
import numpy as np
import torch
from common import *
from native_engine import Engine

LOGIT_TOLERANCE = .001
PROBABILITY_TOLERANCE = .00002

def conditions(src, row):
    names, vectors, recipients, banks = [], [], [], []
    def add(name, vector, recipient):
        names.append(name)
        vectors.append(vector)
        recipients.append(recipient)
        return len(names) - 1
    for direction, recipient, donor_side in [("restore", 1, 0), ("destroy", 0, 1)]:
        indices = {}
        for slot in range(8):
            for column in range(8):
                source_index = row["grid"][slot][column][donor_side]
                indices[source_index] = add(f"{direction}/grid/{slot}/{column}", src[source_index], recipient)
        for bank in row["banks"]:
            source_indices = [pair[donor_side] for pair in bank["pairs"]]
            patch_indices = [indices[i] for i in source_indices]
            mean_index = add(f"{direction}/k{bank['k']}/rotation{bank['rotation']}/mean8", src[source_indices].mean(0), recipient)
            banks.append({"name": f"{direction}/k{bank['k']}/rotation{bank['rotation']}", "direction": direction, "k": bank["k"], "rotation": bank["rotation"], "indices": patch_indices, "mean": mean_index, "source_indices": source_indices})
        add(f"{direction}/identity", src[recipient], recipient)
        add(f"{direction}/paired", src[1 - recipient], recipient)
        add(f"{direction}/zero", torch.zeros_like(src[recipient]), recipient)
    assert len(names) == 198 and len(banks) == 64
    return names, torch.stack(vectors), recipients, banks

def validate(engine, rows):
    errors = {}
    positive = 0.
    for ri, row in enumerate(rows):
        prompts = row["encoded"][engine.name]
        logits, state, _ = engine.run(prompts, capture=True)
        reconstruction = engine.readout(state["final"])
        errors[f"readout_{ri}"] = float((logits - reconstruction).abs().max())
        errors[f"prob_readout_{ri}"] = float(abs(engine.metrics(logits, row)[:, 1] - engine.metrics(reconstruction, row)[:, 1]).max())
        for recipient in [0, 1]:
            p = prompts[recipient]
            with torch.inference_mode():
                native = engine.model(torch.tensor([p["ids"]], device=engine.device), use_cache=False, logits_to_keep=1).logits[:, 0]
            errors[f"native_{ri}_{recipient}"] = float((native[0] - logits[recipient]).abs().max())
            errors[f"prob_native_{ri}_{recipient}"] = float(abs(engine.metrics(native, row)[0, 1] - engine.metrics(logits[recipient:recipient + 1], row)[0, 1]))
            donor_side = 1 - recipient
            donor_ix = row["grid"][0][0][donor_side]
            for layer in engine.layers:
                for pi, pos in enumerate(POSITIONS):
                    source = state["source"][layer][:, pi]
                    vectors = torch.stack([source[recipient], source[donor_ix], source[donor_ix].repeat(8, 1).mean(0)])
                    patched, _, audit = engine.run([p] * 3, (layer, pos, vectors, None))
                    suffix = f"{ri}_{recipient}_{layer}_{pos}"
                    errors[f"identity_{suffix}"] = float((patched[0] - logits[recipient]).abs().max())
                    errors[f"duplicate_{suffix}"] = float((patched[1] - patched[2]).abs().max())
                    errors[f"prob_identity_{suffix}"] = float(abs(engine.metrics(patched[:1], row)[0, 1] - engine.metrics(logits[recipient:recipient + 1], row)[0, 1]))
                    errors[f"prob_duplicate_{suffix}"] = float(abs(engine.metrics(patched[1:2], row)[0, 1] - engine.metrics(patched[2:3], row)[0, 1]))
                    errors[f"outside_{suffix}"] = audit["outside_max"]
                    errors[f"write_{suffix}"] = audit["target_error"]
                    positive = max(positive, float((patched[1] - logits[recipient]).abs().max()))
                    if layer == engine.layers[-1]:
                        expected = logits[recipient] if pos == "anchor" else logits[donor_ix]
                        errors[f"terminal_{suffix}"] = float((patched[1] - expected).abs().max())
            layer = engine.layers[1]
            src = state["source"][layer][:, 1]
            for k in [1, 8]:
                bank = next(b for b in row["banks"] if b["k"] == k and b["rotation"] == 0)
                indices = [p[donor_side] for p in bank["pairs"]]
                x = src[indices]
                log, st, audit = engine.run([p] * 9, (layer, "end", torch.cat([x, x.mean(0, keepdim=True)]), None))
                metrics = engine.metrics(log, row)
                a, individual = metrics[-1], metrics[:8].mean(0)
                b = engine.metrics(engine.readout(st["final"][:8].mean(0, keepdim=True)), row)[0]
                c = engine.metrics(log[:8].mean(0, keepdim=True), row)[0]
                errors[f"margin_closure_{ri}_{recipient}_{k}"] = float(abs((a[0] - individual[0]) - (a[0] - b[0]) - (b[0] - c[0])))
                errors[f"prob_closure_{ri}_{recipient}_{k}"] = float(abs((a[1] - individual[1]) - (a[1] - b[1]) - (b[1] - c[1]) - (c[1] - individual[1])))
        for layer in engine.layers:
            src = state["source"][layer][:, 0]
            differences = [float((src[row["grid"][i][j][d]] - src[row["grid"][i][0][d]]).abs().max()) for i in range(8) for j in range(8) for d in range(2)]
            errors[f"causal_prefix_{ri}_{layer}"] = max(differences)
    probability_max = max(v for k, v in errors.items() if k.startswith("prob_"))
    maximum = max(v for k, v in errors.items() if not k.startswith("prob_"))
    exact_write = all(value == 0 for key, value in errors.items() if key.startswith(("outside_", "write_")))
    return {"created_utc": utc(), "calibration_groups": len(rows), "errors": errors, "maximum_logit_or_state_error": maximum, "maximum_probability_error": probability_max, "exact_write_locality_passed": exact_write, "positive_logit_change": positive, "logit_tolerance": LOGIT_TOLERANCE, "probability_tolerance": PROBABILITY_TOLERANCE, "passed": maximum <= LOGIT_TOLERANCE and probability_max <= PROBABILITY_TOLERANCE and positive > .001 and exact_write}

def evaluate(engine, rows, destination):
    destination.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for ri, row in enumerate(rows):
        path = destination / (row["id"] + ".npz")
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        prompts = row["encoded"][engine.name]
        base, captured, _ = engine.run(prompts, capture=True)
        base_metrics = engine.metrics(base, row)
        all_metrics, endpoints, dispersions, audits, final_states, source_states = [], [], [], [], [], []
        for layer in engine.layers:
            for pi, position in enumerate(POSITIONS):
                src = captured["source"][layer][:, pi]
                names, vectors, recipients, banks = conditions(src, row)
                logits, state, audit = engine.run([prompts[r] for r in recipients], (layer, position, vectors, None))
                metrics = engine.metrics(logits, row)
                all_metrics.append(metrics)
                # Use exactly the original single-row readout branch separately
                # for each centroid endpoint, avoiding cross-bank kernel shapes.
                endpoint, dispersion = [], []
                max_closure = max_probability_closure = 0.
                grand_mean_errors = []
                for bank in banks:
                    ix = bank["indices"]
                    final_centroid = state["final"][ix].mean(0, keepdim=True)
                    b = engine.metrics(engine.readout(final_centroid), row)[0]
                    c = engine.metrics(logits[ix].mean(0, keepdim=True), row)[0]
                    endpoint.append(np.stack([b, c]))
                    a, resample = metrics[bank["mean"]], metrics[ix].mean(0)
                    max_closure = max(max_closure, float(abs((a[0] - resample[0]) - (a[0] - b[0]) - (b[0] - c[0]))))
                    max_probability_closure = max(max_probability_closure, float(abs((a[1] - resample[1]) - (a[1] - b[1]) - (b[1] - c[1]) - (c[1] - resample[1]))))
                    x = src[bank["source_indices"]].float()
                    mu = x.mean(0)
                    centered = x - mu
                    dispersion.append([float(centered.square().sum(-1).mean()), float(mu.square().sum()), float(centered.norm(dim=-1).max())])
                for direction in ["restore", "destroy"]:
                    grids = [i for i, name in enumerate(names) if name.startswith(direction + "/grid/")]
                    grand = vectors[grids].double().mean(0)
                    for k in DOSES:
                        selected = [b for b in banks if b["direction"] == direction and b["k"] == k]
                        grand_mean_errors.append(float((vectors[[b["mean"] for b in selected]].double().mean(0) - grand).abs().max()))
                audit["margin_closure_max"] = max_closure
                audit["probability_closure_max"] = max_probability_closure
                audit["grand_centroid_roundoff_max"] = max(grand_mean_errors)
                audit["identity_logit_max"] = max(float((logits[names.index(d + "/identity")] - base[r]).abs().max()) for d, r in [("restore", 1), ("destroy", 0)])
                audit["identity_probability_max"] = max(abs(metrics[names.index(d + "/identity"), 1] - base_metrics[r, 1]) for d, r in [("restore", 1), ("destroy", 0)])
                if layer == engine.layers[-1]:
                    if position == "anchor":
                        audit["terminal_null_logit_max"] = max(float((logits[i] - base[r]).abs().max()) for i, r in enumerate(recipients))
                    else:
                        audit["terminal_donor_logit_max"] = max(float((logits[bank["indices"][j]] - base[source_index]).abs().max()) for bank in banks for j, source_index in enumerate(bank["source_indices"]))
                if position == "anchor":
                    audit["causal_prefix_state_max"] = max(float((src[row["grid"][i][j][d]] - src[row["grid"][i][0][d]]).abs().max()) for i in range(8) for j in range(8) for d in range(2))
                    audit["anchor_mean_patch_dose_max"] = max(float((logits[b["mean"]] - logits[next(bb["mean"] for bb in banks if bb["direction"] == b["direction"] and bb["k"] == 1 and bb["rotation"] == 0)]).abs().max()) for b in banks)
                if audit["outside_max"] != 0 or audit["target_error"] != 0:
                    raise RuntimeError("Write locality failed")
                gated = [v for key, v in audit.items() if key != "write_norm" and "probability" not in key]
                if max(gated) > LOGIT_TOLERANCE or max(audit["identity_probability_max"], audit["probability_closure_max"]) > PROBABILITY_TOLERANCE:
                    dump(destination / (row["id"] + "-failed-audit.json"), {"layer": layer, "position": position, "audit": audit})
                    raise RuntimeError(f"Numerical control failed: {row['id']}/{layer}/{position}")
                endpoints.append(np.stack(endpoint))
                dispersions.append(dispersion)
                audits.append({"layer": layer, "position": position, **audit})
                if row["group"] in [0, 16, 32, 48]:
                    final_states.append(state["final"].float().cpu().numpy())
                    source_states.append(src.float().cpu().numpy())
        data = {"metrics": np.stack(all_metrics), "endpoints": np.stack(endpoints), "dispersion": np.array(dispersions), "baseline": base_metrics, "condition_names": np.array(names), "bank_names": np.array([b["name"] for b in banks]), "id": np.array(row["id"])}
        if final_states:
            data.update(final=np.stack(final_states), source=np.stack(source_states))
        np.savez_compressed(path, **data)
        dump(destination / (row["id"] + "-audit.json"), audits)
        if ri == 0:
            dump(destination.parent / "design.json", {"names": names, "banks": banks, "metrics": METRICS, "layers": engine.layers, "positions": POSITIONS, "dispersion": ["mean_squared_radius", "centroid_squared_norm", "maximum_radius"]})
        print(json.dumps({"model": engine.name, "row": ri + 1, "total": len(rows), "id": row["id"], "seconds": round(time.time() - start, 2)}), flush=True)
    dump(destination.parent / (destination.name + "-completion.json"), {"finished_utc": utc(), "groups": len(rows), "source_prompt_evaluations": len(rows) * 130, "interventions": len(rows) * 8 * 198, "seconds": time.time() - start, "peak_cuda_bytes": torch.cuda.max_memory_allocated() if engine.device == "cuda" else 0})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["calibrate", "development", "test", "replay"])
    parser.add_argument("--model", choices=list(MODEL_PATHS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.stage in ["test", "replay"]:
        freeze_check()
    dest = args.out / args.model
    dest.mkdir(parents=True, exist_ok=True)
    if args.stage == "calibrate" and (dest / "calibration.json").exists():
        raise RuntimeError("Use a separate output directory to preserve prior calibration attempts")
    engine = Engine(args.model, "float32", args.device, args.batch)
    dump(dest / (f"environment-{args.stage}.json"), engine.environment())
    if args.stage == "calibrate":
        result = validate(engine, read(ROOT / "data/calibration.json"))
        dump(dest / "calibration.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "errors"}), flush=True)
        if not result["passed"]:
            raise RuntimeError("Calibration failed; test is prohibited")
        return
    if not read(dest / "calibration.json")["passed"]:
        raise RuntimeError("Passed calibration required")
    calibration_env = read(dest / "environment-calibrate.json")
    if calibration_env["batch"] != args.batch or calibration_env["dtype"] != "float32":
        raise RuntimeError("Execution configuration differs from admitted calibration")
    split = "development" if args.stage == "development" else "test"
    rows = read(ROOT / f"data/{split}.json")
    if args.stage == "replay":
        rows = [r for r in rows if r["group"] in [0, 16, 32, 48]]
    evaluate(engine, rows, dest / args.stage)
    if args.stage == "replay":
        errors = {}
        for row in rows:
            with np.load(dest / "test" / (row["id"] + ".npz")) as a, np.load(dest / "replay" / (row["id"] + ".npz")) as b:
                for key in ["metrics", "endpoints", "dispersion", "baseline", "final", "source"]:
                    errors[row["id"] + "/" + key] = float(abs(a[key] - b[key]).max())
        maximum = max(errors.values())
        dump(dest / "replay-verification.json", {"errors": errors, "maximum_error": maximum, "passed": maximum <= LOGIT_TOLERANCE})
        if maximum > LOGIT_TOLERANCE:
            raise RuntimeError("Replay failed")

if __name__ == "__main__":
    main()

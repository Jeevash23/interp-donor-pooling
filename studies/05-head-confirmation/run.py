"""Head-level patching runs for the confirmation study.

Settings:
  sva-end  agreement, every head at the final token (the final noun), restore and destroy
  ioi-end  IOI noising, every head at the final token ("to" for every donor)
  ioi-s2   IOI noising, every head at the repeated subject (a different name for most donors)

At every head-site the run patches each of the 64 donors individually, the centroid of all 64
(mean ablation), the centroids of the token groups (donors sharing the token at the patched
position), and the centroids of eight mixed banks of eight (MIX8).

Stages: calibrate -> development -> (freeze) -> gate -> test -> replay.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import ROOT, check_checkpoint, dump, freeze_check, read, utc
from engine import HeadPatchEngine

LOGIT_TOL, PROB_TOL = 1e-3, 2e-5
GATE = 0.90                                  # Qwen3-0.6B clean two-verb accuracy required on the test groups
QWEN_LAYERS = [0, 4, 8, 12, 16, 20, 24]      # heads examined in Qwen if it passes the gate
QWEN_GROUPS = 32
MIX = [[(s, (r + s) % 8) for s in range(8)] for r in range(8)]
NOUN = [[(s, n) for s in range(8)] for n in range(8)]


def margin(logits, good, bad):
    return (logits[:, good].double() - logits[:, bad].double()).cpu().numpy()


def sva_group(engine, row, layers):
    prompts = row["encoded"][engine.name]
    ids = [p["ids"] for p in prompts]
    good, bad, end = prompts[0]["good_id"], prompts[0]["bad_id"], prompts[0]["end"]
    logits, z = engine.capture(ids, [end])                     # z: [L, 130, 1, H, d]
    base = margin(logits, good, bad)
    out = np.zeros((2, len(layers), engine.n_heads, 64 + 1 + 8 + 8))
    worst = {"outside": 0.0, "write": 0.0}
    for d, (recipient, side) in enumerate([(1, 0), (0, 1)]):
        idx = [row["grid"][s][n][side] for s in range(8) for n in range(8)]
        for li, layer in enumerate(layers):
            for head in range(engine.n_heads):
                v = z[layer, idx, 0, head]                      # [64, d], index s*8+n
                tok = [v[[s * 8 + n for s, n in bank]].mean(0) for bank in NOUN]
                mix = [v[[s * 8 + n for s, n in bank]].mean(0) for bank in MIX]
                vectors = torch.cat([v, v.mean(0, keepdim=True), torch.stack(tok), torch.stack(mix)])
                log, audit = engine.patch(ids[recipient], layer, end, head, vectors)
                out[d, li, head] = margin(log, good, bad)
                worst["outside"] = max(worst["outside"], audit["outside_max"])
                worst["write"] = max(worst["write"], audit["target_error"])
    return {"margins": out, "baseline": base[:2], "layers": np.array(layers)}, worst


def ioi_row(engine, row, setting):
    rec = row["recipient"]
    ids = [rec["ids"]] + [d["ids"] for d in row["donors"]]
    pos = rec["end"] if setting == "ioi-end" else rec["s2"]
    logits, z = engine.capture(ids, [pos])
    base = margin(logits, rec["target_id"], rec["contrast_id"])
    keys = row["s2_tokens"] if setting == "ioi-s2" else [rec["ids"][rec["end"]]] * 64
    groups = sorted(set(keys))
    members = [[j for j, k in enumerate(keys) if k == g] for g in groups]
    n_tok = len(groups) if len(groups) > 1 else 0          # one group means the token rule equals the mean
    out = np.zeros((engine.n_layers, engine.n_heads, 64 + 1 + 8 + n_tok))
    worst = {"outside": 0.0, "write": 0.0}
    for layer in range(engine.n_layers):
        for head in range(engine.n_heads):
            v = z[layer, 1:, 0, head]
            parts = [v, v.mean(0, keepdim=True), torch.stack([v[8 * b:8 * b + 8].mean(0) for b in range(8)])]
            if n_tok:
                parts.append(torch.stack([v[m].mean(0) for m in members]))
            log, audit = engine.patch(rec["ids"], layer, pos, head, torch.cat(parts))
            out[layer, head] = margin(log, rec["target_id"], rec["contrast_id"])
            worst["outside"] = max(worst["outside"], audit["outside_max"])
            worst["write"] = max(worst["write"], audit["target_error"])
    return {"margins": out, "baseline": base[:1], "donor_margins": base[1:],
            "token_group_sizes": np.array([len(m) for m in members]), "token_groups_evaluated": np.array(n_tok)}, worst


def calibrate(engine, setting):
    errors, probs = {}, {}

    def prob_err(a, b):
        return float((a.double().softmax(-1) - b.double().softmax(-1)).abs().max())

    if setting == "sva-end":
        rows = read(ROOT / "data/sva-calibration.json")
        cases = []
        for row in rows:
            p = row["encoded"][engine.name]
            cases.append(([x["ids"] for x in p], 1, row["grid"][0][0][0], p[0]["end"]))
    else:
        rows = read(ROOT / "data/ioi-calibration.json")
        cases = [([r["recipient"]["ids"]] + [d["ids"] for d in r["donors"][:2]], 0, 1,
                  r["recipient"]["end"] if setting == "ioi-end" else r["recipient"]["s2"]) for r in rows]
    for ci, (ids, recipient, donor, pos) in enumerate(cases):
        logits, z = engine.capture(ids, [pos])
        with torch.inference_mode():
            native = engine.model(torch.tensor([ids[recipient]], device=engine.device), use_cache=False).logits[:, -1]
        errors[f"{ci}/passive"] = float((native[0] - logits[recipient]).abs().max())
        probs[f"{ci}/passive"] = prob_err(native, logits[recipient:recipient + 1])
        for layer in [0, engine.n_layers // 2, engine.n_layers - 1]:
            for head in [0, engine.n_heads - 1]:
                vec = torch.stack([z[layer, recipient, 0, head], z[layer, donor, 0, head],
                                   z[layer, donor, 0, head].repeat(8, 1).mean(0)])
                log, audit = engine.patch(ids[recipient], layer, pos, head, vec)
                k = f"{ci}/{layer}/{head}"
                errors[k + "/identity"] = float((log[0] - logits[recipient]).abs().max())
                probs[k + "/identity"] = prob_err(log[:1], logits[recipient:recipient + 1])
                errors[k + "/duplicate"] = float((log[1] - log[2]).abs().max())
                probs[k + "/duplicate"] = prob_err(log[1:2], log[2:3])
                errors[k + "/outside"], errors[k + "/write"] = audit["outside_max"], audit["target_error"]
    exact = all(v == 0 for k, v in errors.items() if k.endswith(("/outside", "/write")))
    gated = max(v for k, v in errors.items() if not k.endswith(("/outside", "/write")))
    return {"created_utc": utc(), "setting": setting, "model": engine.name, "errors": errors, "probability_errors": probs,
            "maximum_logit_error": gated, "maximum_probability_error": max(probs.values()), "exact_write_locality": exact,
            "passed": gated <= LOGIT_TOL and max(probs.values()) <= PROB_TOL and exact}


def gate(engine):
    """Qwen3-0.6B clean two-verb accuracy on the fresh test groups, from intact-model outputs only."""
    rows = read(ROOT / "data/sva-test.json")
    correct = []
    for row in rows:
        p = row["encoded"][engine.name][0]
        logits, _ = engine.capture([p["ids"]], [p["end"]])
        correct.append(bool(margin(logits, p["good_id"], p["bad_id"])[0] > 0))
    acc = float(np.mean(correct))
    return {"created_utc": utc(), "model": engine.name, "groups": len(rows), "clean_two_verb_accuracy": acc,
            "required": GATE, "passed": acc >= GATE}


def run(engine, setting, stage):
    task = "sva" if setting == "sva-end" else "ioi"
    rows = read(ROOT / f"data/{task}-{'development' if stage == 'development' else 'test'}.json")
    if engine.name == "qwen3-0.6b":
        rows = rows[:QWEN_GROUPS]
    if stage == "replay":
        rows = rows[:2]
    layers = QWEN_LAYERS if engine.name == "qwen3-0.6b" else list(range(engine.n_layers))
    dest = ROOT / "results" / f"{setting}-{engine.name}" / stage
    dest.mkdir(parents=True, exist_ok=True)
    start, audits = time.time(), {}
    for i, row in enumerate(rows):
        path = dest / f"{row['id']}.npz"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        data, worst = sva_group(engine, row, layers) if task == "sva" else ioi_row(engine, row, setting)
        if worst["outside"] != 0 or worst["write"] != 0:
            raise RuntimeError(f"write locality failed: {row['id']}")
        np.savez_compressed(path, **data)
        audits[row["id"]] = worst
        print(f'{{"setting": "{setting}", "model": "{engine.name}", "stage": "{stage}", "row": {i + 1}, "of": {len(rows)}, "seconds": {time.time() - start:.1f}}}', flush=True)
    dump(dest.parent / f"{stage}-completion.json", {"finished_utc": utc(), "rows": len(rows), "seconds": time.time() - start,
                                                     "audits": audits, "environment": engine.environment()})
    if stage == "replay":
        errors = {}
        for row in rows:
            with np.load(dest / f"{row['id']}.npz") as a, np.load(dest.parent / "test" / f"{row['id']}.npz") as b:
                errors[row["id"]] = float(np.max(np.abs(a["margins"] - b["margins"])))
        dump(dest.parent / "replay-verification.json", {"errors": errors, "maximum_error": max(errors.values()),
                                                         "passed": max(errors.values()) <= LOGIT_TOL})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "development", "gate", "test", "replay"])
    ap.add_argument("--setting", choices=["sva-end", "ioi-end", "ioi-s2"], default="sva-end")
    ap.add_argument("--model", choices=["gpt2-small", "qwen3-0.6b"], default="gpt2-small")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    if args.model == "qwen3-0.6b" and args.setting != "sva-end":
        raise ValueError("Qwen3-0.6B is only considered for the agreement setting")
    if args.stage in ("gate", "test", "replay"):
        freeze_check()
    hashes = check_checkpoint(args.model)
    engine = HeadPatchEngine(args.model, "cuda", args.batch)
    folder = ROOT / "results" / (f"{args.setting}-{args.model}" if args.stage != "gate" else "qwen-gate")
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / f"environment-{args.stage}.json", {**engine.environment(), "checkpoint_sha256": hashes})
    if args.stage == "calibrate":
        dest = folder / "calibration.json"
        if dest.exists():
            raise RuntimeError("keep earlier calibration records")
        result = calibrate(engine, args.setting)
        dump(dest, result)
        print({k: v for k, v in result.items() if k not in ("errors", "probability_errors")}, flush=True)
        if not result["passed"]:
            raise RuntimeError("calibration failed")
        return
    if args.stage == "gate":
        if args.model != "qwen3-0.6b":
            raise ValueError("the gate applies to Qwen3-0.6B")
        dest = folder / "gate.json"
        if dest.exists():
            raise RuntimeError("the gate is evaluated once")
        result = gate(engine)
        dump(dest, result)
        print(result, flush=True)
        return
    if not read(folder / "calibration.json")["passed"]:
        raise RuntimeError("passed calibration required")
    if args.model == "qwen3-0.6b" and args.stage != "development":
        if not read(ROOT / "results/qwen-gate/gate.json")["passed"]:
            raise RuntimeError("Qwen3-0.6B did not pass the prespecified accuracy gate")
    run(engine, args.setting, args.stage)


if __name__ == "__main__":
    main()

"""Experiment B: full residual patching sweeps with averaged versus individual donors.

Panels:
  sva  (gpt2-small, qwen3-0.6b)  64-donor adjective-by-noun grids, restore and destroy
  ioi  (gpt2-small)              64 canonical ABC donors per prompt, noising only

Per site (layer x position) every donor is patched individually, and centroids of
several partitions of the same donors are patched: eight mixed banks (MIX8), the
mean of all 64 donors (MEAN64) and, for the agreement grid, banks that share the
final noun (NOUN8) or the adjective (ADJ8).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import ROOT, check_checkpoint, dump, freeze_check, read, utc
from engines import ResidualEngine

LOGIT_TOL = 1e-3
MIX = [[(s, (r + s) % 8) for s in range(8)] for r in range(8)]
NOUN = [[(s, n) for s in range(8)] for n in range(8)]
ADJ = [[(s, n) for n in range(8)] for s in range(8)]
SVA_METHODS = ["individual"] * 64 + ["mix8"] * 8 + ["noun8"] * 8 + ["adj8"] * 8 + ["mean64"]
IOI_METHODS = ["individual"] * 64 + ["mix8"] * 8 + ["mean64"]


def margin(logits, good, bad):
    return (logits[:, good].double() - logits[:, bad].double()).cpu().numpy()


def msr(x):
    x = x.double()
    return float((x - x.mean(0)).square().sum(-1).mean())


def sva_vectors(states, grid, side, layer, pos):
    """64 individual donor states (index s*8+n), then MIX8, NOUN8, ADJ8 centroids and MEAN64."""
    idx = [grid[s][n][side] for s in range(8) for n in range(8)]
    v = states[idx, layer, pos]                      # [64, d]
    cell = lambda s, n: s * 8 + n
    cents, spread = [], []
    for partition in (MIX, NOUN, ADJ):
        for bank in partition:
            rows = v[[cell(s, n) for s, n in bank]]
            cents.append(rows.mean(0))
            spread.append(msr(rows))
    cents.append(v.mean(0))
    spread.append(msr(v))
    return torch.cat([v, torch.stack(cents)]), spread


def sva_group(engine, row, check=False):
    prompts = row["encoded"][engine.name]
    ids = [p["ids"] for p in prompts]
    good, bad = prompts[0]["good_id"], prompts[0]["bad_id"]
    logits, states = engine.capture(ids)
    base = margin(logits, good, bad)
    positions = [prompts[0]["adj"]] + list(range(prompts[0]["anchor"], prompts[0]["end"] + 1))
    out = np.zeros((2, engine.n_layers, len(positions), len(SVA_METHODS)))
    spread = np.zeros((2, engine.n_layers, len(positions), 25))
    worst = {"outside": 0.0, "write": 0.0, "identity": 0.0}
    for d, (recipient, side) in enumerate([(1, 0), (0, 1)]):        # restore: clean donors into corrupt; destroy: reverse
        for layer in range(engine.n_layers):
            for pi, pos in enumerate(positions):
                vectors, s = sva_vectors(states, row["grid"], side, layer, pos)
                if check and layer == engine.n_layers // 2 and pi == len(positions) - 1:
                    vectors = torch.cat([vectors, states[recipient, layer, pos][None]])
                log, audit = engine.patch(ids[recipient], layer, pos, vectors)
                m = margin(log, good, bad)
                if check and layer == engine.n_layers // 2 and pi == len(positions) - 1:
                    worst["identity"] = max(worst["identity"], float(abs(m[-1] - base[recipient])))
                    m = m[:-1]
                out[d, layer, pi] = m
                spread[d, layer, pi] = s
                worst["outside"] = max(worst["outside"], audit["outside_max"])
                worst["write"] = max(worst["write"], audit["target_error"])
    if worst["outside"] != 0 or worst["write"] != 0:
        raise RuntimeError(f"write locality failed: {row['id']}")
    return {"margins": out, "spread": spread, "baseline": base[:2], "positions": np.array(positions),
            "position_names": np.array(["adj"] + [f"anchor+{k}" for k in range(len(positions) - 1)])}, worst


def ioi_row(engine, row, check=False):
    rec = row["recipient"]
    ids = [rec["ids"]] + [d["ids"] for d in row["donors"]]
    logits, states = engine.capture(ids)
    base = margin(logits, rec["target_id"], rec["contrast_id"])
    t = len(rec["ids"])
    out = np.full((engine.n_layers, t, len(IOI_METHODS)), np.nan)
    spread = np.zeros((engine.n_layers, t, 9))
    null_positions = []
    worst = {"outside": 0.0, "write": 0.0, "identity": 0.0}
    for pos in range(t):
        if float((states[1:, :, pos] - states[:1, :, pos]).abs().max()) == 0.0:
            null_positions.append(pos)              # every donor state equals the recipient state here
            out[:, pos] = base[0]
            continue
        for layer in range(engine.n_layers):
            v = states[1:, layer, pos]
            cents = [v[8 * b:8 * b + 8].mean(0) for b in range(8)] + [v.mean(0)]
            spread[layer, pos] = [msr(v[8 * b:8 * b + 8]) for b in range(8)] + [msr(v)]
            vectors = torch.cat([v, torch.stack(cents)])
            extra = check and layer == 6 and pos == t - 1
            if extra:
                vectors = torch.cat([vectors, states[0, layer, pos][None]])
            log, audit = engine.patch(rec["ids"], layer, pos, vectors)
            m = margin(log, rec["target_id"], rec["contrast_id"])
            if extra:
                worst["identity"] = max(worst["identity"], float(abs(m[-1] - base[0])))
                m = m[:-1]
            out[layer, pos] = m
            worst["outside"] = max(worst["outside"], audit["outside_max"])
            worst["write"] = max(worst["write"], audit["target_error"])
    if worst["outside"] != 0 or worst["write"] != 0:
        raise RuntimeError(f"write locality failed: {row['id']}")
    return {"margins": out, "spread": spread, "baseline": base[:1], "donor_margins": base[1:],
            "null_positions": np.array(null_positions, dtype=np.int64)}, worst


def calibrate(engine, panel):
    """Identity, duplicate-donor, structural-null and final-block checks on calibration prompts."""
    errors = {}
    if panel == "sva":
        rows = read(ROOT / "data/sva-calibration.json")
        for row in rows:
            prompts = row["encoded"][engine.name]
            ids = [p["ids"] for p in prompts]
            logits, states = engine.capture(ids)
            for r in (0, 1):
                with torch.inference_mode():
                    native = engine.model(torch.tensor([ids[r]], device=engine.device), use_cache=False).logits[:, -1]
                errors[f"{row['id']}/native_{r}"] = float((native[0] - logits[r]).abs().max())
            donor = row["grid"][0][0][0]
            pre = prompts[0]["adj"] - 1
            for layer in range(engine.n_layers):
                for pos in [pre, prompts[0]["adj"], prompts[0]["anchor"], prompts[0]["end"]]:
                    vec = torch.stack([states[1, layer, pos], states[donor, layer, pos], states[donor, layer, pos].repeat(8, 1).mean(0)])
                    log, audit = engine.patch(ids[1], layer, pos, vec)
                    k = f"{row['id']}/{layer}/{pos}"
                    errors[k + "/identity"] = float((log[0] - logits[1]).abs().max())
                    errors[k + "/duplicate"] = float((log[1] - log[2]).abs().max())
                    errors[k + "/outside"], errors[k + "/write"] = audit["outside_max"], audit["target_error"]
                    if pos == pre:          # identical prefix: a donor state here equals the recipient state
                        errors[k + "/structural_null"] = float((log[1] - logits[1]).abs().max())
                    if layer == engine.n_layers - 1 and pos == prompts[0]["end"]:
                        errors[k + "/final_block_end"] = float((log[1] - logits[donor]).abs().max())
    else:
        rows = read(ROOT / "data/ioi-sweep.json")[:4]
        for row in rows:
            rec = row["recipient"]
            ids = [rec["ids"]] + [d["ids"] for d in row["donors"][:2]]
            logits, states = engine.capture(ids)
            with torch.inference_mode():
                native = engine.model(torch.tensor([rec["ids"]], device=engine.device), use_cache=False).logits[:, -1]
            errors[f"{row['id']}/native"] = float((native[0] - logits[0]).abs().max())
            for layer in range(engine.n_layers):
                for pos in [0, rec["s2"], rec["end"]]:
                    vec = torch.stack([states[0, layer, pos], states[1, layer, pos], states[1, layer, pos].repeat(8, 1).mean(0)])
                    log, audit = engine.patch(rec["ids"], layer, pos, vec)
                    k = f"{row['id']}/{layer}/{pos}"
                    errors[k + "/identity"] = float((log[0] - logits[0]).abs().max())
                    errors[k + "/duplicate"] = float((log[1] - log[2]).abs().max())
                    errors[k + "/outside"], errors[k + "/write"] = audit["outside_max"], audit["target_error"]
                    if pos == 0:
                        errors[k + "/structural_null"] = float((log[1] - logits[0]).abs().max())
                    if layer == engine.n_layers - 1 and pos == rec["end"]:
                        errors[k + "/final_block_end"] = float((log[1] - logits[1]).abs().max())
    exact = all(v == 0 for k, v in errors.items() if k.endswith(("/outside", "/write")))
    gated = max(v for k, v in errors.items() if not k.endswith(("/outside", "/write")))
    return {"created_utc": utc(), "panel": panel, "model": engine.name, "errors": errors, "maximum_logit_error": gated,
            "exact_write_locality": exact, "tolerance": LOGIT_TOL, "passed": gated <= LOGIT_TOL and exact}


def run(engine, panel, stage, limit=None):
    if panel == "sva":
        rows = read(ROOT / f"data/sva-{'development' if stage == 'development' else 'test'}.json")
    else:
        rows = read(ROOT / "data/ioi-sweep.json")
        if stage == "development":
            rows = rows[:2]
    if limit:
        rows = rows[:limit]
    if stage == "replay":
        rows = rows[:2]
    dest = ROOT / "results/sweep" / f"{panel}-{engine.name}" / stage
    dest.mkdir(parents=True, exist_ok=True)
    start, worst_all = time.time(), {}
    for i, row in enumerate(rows):
        path = dest / f"{row['id']}.npz"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        data, worst = (sva_group if panel == "sva" else ioi_row)(engine, row, check=True)
        if worst["identity"] > LOGIT_TOL:
            raise RuntimeError(f"identity patch failed: {row['id']}")
        np.savez_compressed(path, **data)
        worst_all[row["id"]] = worst
        print(f'{{"panel": "{panel}", "model": "{engine.name}", "stage": "{stage}", "row": {i + 1}, "of": {len(rows)}, "seconds": {time.time() - start:.1f}}}', flush=True)
    dump(dest.parent / f"{stage}-completion.json", {"finished_utc": utc(), "rows": len(rows), "seconds": time.time() - start,
                                                     "audits": worst_all, "environment": engine.environment()})
    if stage == "replay":
        errors = {}
        for row in rows:
            with np.load(dest / f"{row['id']}.npz") as a, np.load(dest.parent / "test" / f"{row['id']}.npz") as b:
                errors[row["id"]] = float(np.nanmax(np.abs(a["margins"] - b["margins"])))
        dump(dest.parent / "replay-verification.json", {"errors": errors, "maximum_error": max(errors.values()),
                                                         "passed": max(errors.values()) <= LOGIT_TOL})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", choices=["sva", "ioi"])
    ap.add_argument("stage", choices=["calibrate", "development", "test", "replay"])
    ap.add_argument("--model", default="gpt2-small", choices=["gpt2-small", "qwen3-0.6b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.panel == "ioi" and args.model != "gpt2-small":
        raise ValueError("the IOI panel uses GPT-2 Small")
    if args.stage in ("test", "replay"):
        freeze_check()
    hashes = check_checkpoint(args.model)
    engine = ResidualEngine(args.model, args.device, args.batch)
    folder = ROOT / "results/sweep" / f"{args.panel}-{args.model}"
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / f"environment-{args.stage}.json", {**engine.environment(), "checkpoint_sha256": hashes})
    if args.stage == "calibrate":
        dest = folder / "calibration.json"
        if dest.exists():
            raise RuntimeError("keep earlier calibration records")
        result = calibrate(engine, args.panel)
        dump(dest, result)
        print({k: v for k, v in result.items() if k != "errors"}, flush=True)
        if not result["passed"]:
            raise RuntimeError("calibration failed")
        return
    if not read(folder / "calibration.json")["passed"]:
        raise RuntimeError("passed calibration required")
    run(engine, args.panel, args.stage, args.limit)


if __name__ == "__main__":
    main()

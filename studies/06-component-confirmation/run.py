"""Component-level patching runs for Study 4 (GPT-2 Small, agreement, final token).

For every group and direction (restore: clean donors into the corrupted recipient; destroy: corrupted
donors into the clean recipient), at the final token, every layer, and four kinds of site (residual
stream, attention-layer output, MLP output, single head), the run patches:
  the 64 donors one at a time; their centroid (mean ablation); the centroids of the eight banks that share
  the final noun (token grouping); and the centroids of eight mixed banks (MIX8).
It also records the donors' spread in residual space: the mean squared distance from the centroid of the
64 donors and within each bank (heads are mapped into residual space through W_O).

Stages: calibrate -> development -> (freeze) -> test -> replay.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import ROOT, check_checkpoint, dump, freeze_check, read, utc
from engine import KINDS, ComponentEngine

LOGIT_TOL, PROB_TOL = 1e-3, 2e-5
MIX = [[(s, (r + s) % 8) for s in range(8)] for r in range(8)]
NOUN = [[(s, n) for s in range(8)] for n in range(8)]


def margin(logits, good, bad):
    return (logits[:, good].double() - logits[:, bad].double()).cpu().numpy()


def spreads(r):
    """r: [64, 768] donor vectors in residual space, index s*8+n -> [17]: all 64, eight NOUN banks, eight MIX banks."""
    def msr(x):
        return float(((x - x.mean(0, keepdim=True)) ** 2).sum(-1).mean())
    out = [msr(r)]
    for banks in (NOUN, MIX):
        out += [msr(r[[s * 8 + n for s, n in bank]]) for bank in banks]
    return np.array(out)


def run_group(engine, row):
    prompts = row["encoded"]["gpt2-small"]
    ids = [p["ids"] for p in prompts]
    good, bad, end = prompts[0]["good_id"], prompts[0]["bad_id"], prompts[0]["end"]
    logits, comp = engine.capture(ids, end)
    base = margin(logits, good, bad)
    L, H = engine.n_layers, engine.n_heads
    out = {f"margins_{k}": np.zeros((2, L, H, 81) if k == "head" else (2, L, 81)) for k in KINDS}
    out.update({f"spread_{k}": np.zeros((2, L, H, 17) if k == "head" else (2, L, 17)) for k in KINDS})
    worst = {"outside": 0.0, "write": 0.0}
    for d, (recipient, side) in enumerate([(1, 0), (0, 1)]):
        idx = [row["grid"][s][n][side] for s in range(8) for n in range(8)]
        for kind in KINDS:
            for layer in range(L):
                for head in (range(H) if kind == "head" else [None]):
                    v = comp[kind][layer, idx] if head is None else comp[kind][layer, idx, head]      # [64, dim]
                    tok = torch.stack([v[[s * 8 + n for s, n in bank]].mean(0) for bank in NOUN])
                    mix = torch.stack([v[[s * 8 + n for s, n in bank]].mean(0) for bank in MIX])
                    vectors = torch.cat([v, v.mean(0, keepdim=True), tok, mix])
                    log, audit = engine.patch(ids[recipient], kind, layer, end, vectors, head=head)
                    r = engine.head_in_residual(v, layer, head) if kind == "head" else v
                    where = (d, layer) if head is None else (d, layer, head)
                    out[f"margins_{kind}"][where] = margin(log, good, bad)
                    out[f"spread_{kind}"][where] = spreads(r.double())
                    worst["outside"] = max(worst["outside"], audit["outside_max"])
                    worst["write"] = max(worst["write"], audit["target_error"])
    out["baseline"] = base[:2]
    return out, worst


def calibrate(engine):
    rows = read(ROOT / "data/sva-calibration.json")
    errors, probs = {}, {}

    def prob_err(a, b):
        return float((a.double().softmax(-1) - b.double().softmax(-1)).abs().max())

    for ci, row in enumerate(rows):
        prompts = row["encoded"]["gpt2-small"]
        ids = [p["ids"] for p in prompts]
        end = prompts[0]["end"]
        recipient, donor = 1, row["grid"][0][0][0]
        logits, comp = engine.capture(ids, end)
        with torch.inference_mode():
            native = engine.model(torch.tensor([ids[recipient]], device=engine.device), use_cache=False).logits[:, -1]
        errors[f"{ci}/passive"] = float((native[0] - logits[recipient]).abs().max())
        probs[f"{ci}/passive"] = prob_err(native, logits[recipient:recipient + 1])
        for kind in KINDS:
            for layer in [0, engine.n_layers // 2, engine.n_layers - 1]:
                for head in ([0, engine.n_heads - 1] if kind == "head" else [None]):
                    pick = (lambda i: comp[kind][layer, i]) if head is None else (lambda i: comp[kind][layer, i, head])
                    vec = torch.stack([pick(recipient), pick(donor), pick(donor).repeat(8, 1).mean(0)])
                    log, audit = engine.patch(ids[recipient], kind, layer, end, vec, head=head)
                    k = f"{ci}/{kind}/{layer}/{head}"
                    errors[k + "/identity"] = float((log[0] - logits[recipient]).abs().max())
                    probs[k + "/identity"] = prob_err(log[:1], logits[recipient:recipient + 1])
                    errors[k + "/duplicate"] = float((log[1] - log[2]).abs().max())
                    probs[k + "/duplicate"] = prob_err(log[1:2], log[2:3])
                    errors[k + "/outside"], errors[k + "/write"] = audit["outside_max"], audit["target_error"]
                    if kind == "resid" and layer == engine.n_layers - 1:
                        errors[k + "/final_block_reproduces_donor"] = float((log[1] - logits[donor]).abs().max())
    exact = all(v == 0 for k, v in errors.items() if k.endswith(("/outside", "/write")))
    gated = max(v for k, v in errors.items() if not k.endswith(("/outside", "/write")))
    return {"created_utc": utc(), "errors": errors, "probability_errors": probs, "maximum_logit_error": gated,
            "maximum_probability_error": max(probs.values()), "exact_write_locality": exact,
            "passed": gated <= LOGIT_TOL and max(probs.values()) <= PROB_TOL and exact}


def run(engine, stage):
    rows = read(ROOT / f"data/sva-{'development' if stage == 'development' else 'test'}.json")
    if stage == "replay":
        rows = rows[:2]
    dest = ROOT / "results" / stage
    dest.mkdir(parents=True, exist_ok=True)
    start, audits = time.time(), {}
    for i, row in enumerate(rows):
        path = dest / f"{row['id']}.npz"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        data, worst = run_group(engine, row)
        if worst["outside"] != 0 or worst["write"] != 0:
            raise RuntimeError(f"write locality failed: {row['id']}")
        np.savez_compressed(path, **data)
        audits[row["id"]] = worst
        print(f'{{"stage": "{stage}", "row": {i + 1}, "of": {len(rows)}, "seconds": {time.time() - start:.1f}}}', flush=True)
    dump(ROOT / "results" / f"{stage}-completion.json", {"finished_utc": utc(), "rows": len(rows),
                                                        "seconds": time.time() - start, "audits": audits,
                                                        "environment": engine.environment()})
    if stage == "replay":
        errors = {}
        for row in rows:
            with np.load(dest / f"{row['id']}.npz") as a, np.load(ROOT / "results/test" / f"{row['id']}.npz") as b:
                errors[row["id"]] = max(float(np.max(np.abs(a[k] - b[k]))) for k in a.files if k.startswith("margins"))
        dump(ROOT / "results/replay-verification.json", {"errors": errors, "maximum_error": max(errors.values()),
                                                         "passed": max(errors.values()) <= LOGIT_TOL})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "development", "test", "replay"])
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    if args.stage in ("test", "replay"):
        freeze_check()
    hashes = check_checkpoint("gpt2-small")
    engine = ComponentEngine("cuda", args.batch)
    dump(ROOT / "results" / f"environment-{args.stage}.json", {**engine.environment(), "checkpoint_sha256": hashes})
    if args.stage == "calibrate":
        dest = ROOT / "results/calibration.json"
        if dest.exists():
            raise RuntimeError("keep earlier calibration records")
        result = calibrate(engine)
        dump(dest, result)
        print({k: result[k] for k in ["maximum_logit_error", "maximum_probability_error", "exact_write_locality", "passed"]})
        if not result["passed"]:
            raise SystemExit("calibration failed")
    else:
        run(engine, args.stage)


if __name__ == "__main__":
    main()

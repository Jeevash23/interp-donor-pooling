"""Exploratory (written after the freeze): does the second-order term predict the pooling gap?

For agreement banks at the final token, compute -1/2 * mean_j d_j^T H d_j, where H is the
Hessian of the clean-answer margin with respect to the patched residual vector at the
bank centroid and d_j are the donor deviations. Compare with the measured gap M - R from
the saved sweep outputs. GPT-2 Small only; blocks 2, 5 and 8 (Study 3's sites).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import ROOT, dump, read, utc
from engines import ResidualEngine, _tensor
from run_sweep import ADJ, MIX, NOUN

LAYERS = [2, 5, 8]
PARTITIONS = {"mix8": (MIX, 64), "noun8": (NOUN, 72), "adj8": (ADJ, 80)}


def margin_and_hvps(engine, ids, layer, pos, mu, deltas, good, bad):
    """Margin at the centroid and d^T H d for each deviation, by double backward."""
    v = mu.clone().requires_grad_(True)

    def write(module, args, out):
        w = _tensor(out).clone()
        w[:, pos] = v
        return (w, *out[1:]) if isinstance(out, tuple) else w

    handle = engine.blocks[layer].register_forward_hook(write)
    try:
        with torch.enable_grad():
            logits = engine.model(torch.tensor([ids], device=engine.device), use_cache=False, logits_to_keep=1).logits[0, -1]
            f = logits[good] - logits[bad]
            (g,) = torch.autograd.grad(f, v, create_graph=True)
            q = []
            for d in deltas:
                (hd,) = torch.autograd.grad(g @ d, v, retain_graph=True)
                q.append(float(hd @ d))
    finally:
        handle.remove()
    return float(f), np.array(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    engine = ResidualEngine("gpt2-small", args.device, 96)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(False)      # double backward; exploratory analysis only
    rows = read(ROOT / "data/sva-test.json")[:args.groups]
    folder = ROOT / "results/sweep/sva-gpt2-small/test"
    records, start = [], time.time()
    for gi, row in enumerate(rows):
        prompts = row["encoded"]["gpt2-small"]
        ids = [p["ids"] for p in prompts]
        good, bad = prompts[0]["good_id"], prompts[0]["bad_id"]
        _, states = engine.capture(ids)
        saved = np.load(folder / f"{row['id']}.npz")
        end = len(saved["positions"]) - 1
        assert int(saved["positions"][end]) == prompts[0]["end"]
        for d, (recipient, side) in enumerate([(1, 0), (0, 1)]):
            idx = [row["grid"][s][n][side] for s in range(8) for n in range(8)]
            for layer in LAYERS:
                v = states[idx, layer, prompts[0]["end"]].float()
                m = saved["margins"][d, layer, end]
                for name, (partition, offset) in PARTITIONS.items():
                    for b, bank in enumerate(partition):
                        cells = [s * 8 + n for s, n in bank]
                        x = v[cells]
                        mu = x.mean(0)
                        fmu, q = margin_and_hvps(engine, ids[recipient], layer, prompts[0]["end"], mu, x - mu, good, bad)
                        measured = float(m[offset + b] - m[cells].mean())
                        records.append({"group": gi, "direction": d, "layer": layer, "partition": name, "bank": b,
                                        "measured_gap": measured, "predicted_gap": float(-0.5 * q.mean()),
                                        "centroid_margin_recomputed": fmu, "centroid_margin_saved": float(m[offset + b]),
                                        "spread": float((x - mu).double().square().sum(-1).mean())})
        print(f'{{"group": {gi + 1}, "of": {len(rows)}, "seconds": {time.time() - start:.1f}}}', flush=True)
    meas = np.array([r["measured_gap"] for r in records])
    pred = np.array([r["predicted_gap"] for r in records])
    def fit(mask):
        a, b = meas[mask], pred[mask]
        slope = float((a * b).sum() / (b * b).sum())
        r2 = float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
        return {"banks": int(mask.sum()), "corr": float(np.corrcoef(a, b)[0, 1]), "slope_measured_on_predicted": slope,
                "r2_identity_prediction": r2, "mean_abs_measured": float(np.abs(a).mean()), "mean_abs_predicted": float(np.abs(b).mean()),
                "sign_agreement": float(np.mean(np.sign(a) == np.sign(b)))}
    parts = np.array([r["partition"] for r in records])
    summary = {"created_utc": utc(), "exploratory": True, "groups": len(rows), "layers": LAYERS,
               "all": fit(np.ones(len(records), bool)), **{p: fit(parts == p) for p in PARTITIONS},
               "max_centroid_margin_mismatch": float(max(abs(r["centroid_margin_recomputed"] - r["centroid_margin_saved"]) for r in records))}
    dump(ROOT / "results/curvature.json", {"summary": summary, "banks": records})
    print(summary)


if __name__ == "__main__":
    main()

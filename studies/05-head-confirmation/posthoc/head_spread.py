"""Post hoc: donor spread of single heads in the residual space, against the residual stream (Study 3).

Added after the Study 3 results were known; frozen files are only read. Study 3 saved patched margins,
not activations, so this script reruns the unpatched forward passes of the Study 3 test prompts (the same
prompts, checkpoint and settings; Study 3's replays reproduced its outputs exactly) and captures, at the
patched position:
  - every head's output written into the residual stream, z_h @ W_O^h (the projection bias is common to
    all donors and does not affect spread);
  - the residual stream after every block.
For each recipient, direction and site it records the donors' mean squared distance from their centroid.
The pooling gap of each head-site (centroid margin minus the average individual margin, in logits) comes
from Study 3's saved margins.
Output: results/posthoc/head-spread.npz and results/posthoc/head-spread.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, check_checkpoint, dump, read, utc  # noqa: E402
from engine import HeadPatchEngine  # noqa: E402

SETTINGS = {"sva-end-gpt2-small": "sva", "ioi-end-gpt2-small": "ioi", "ioi-s2-gpt2-small": "ioi"}


@torch.inference_mode()
def capture(engine, ids, position):
    """Head outputs in residual space [L, n, H, 768] and the residual after every block [L, n, 768] at `position`."""
    blocks = engine.model.transformer.h
    w_o = torch.stack([b.attn.c_proj.weight.reshape(engine.n_heads, engine.head_dim, -1) for b in blocks])
    zs, rs = [], []
    for start in range(0, len(ids), engine.batch):
        batch = torch.tensor(ids[start:start + engine.batch], device=engine.device)
        cz, cr, handles = {}, {}, []
        for layer, b in enumerate(blocks):
            handles.append(b.attn.c_proj.register_forward_pre_hook(
                lambda m, a, layer=layer: cz.__setitem__(layer, a[0][:, position].reshape(-1, engine.n_heads, engine.head_dim).clone())))
            handles.append(b.register_forward_hook(lambda m, a, o, layer=layer: cr.__setitem__(layer, o[:, position].clone())))
        try:
            engine.model(batch, use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        z = torch.stack([cz[l] for l in range(engine.n_layers)])            # [L, n, H, 64]
        zs.append(torch.einsum("lnhd,lhde->lnhe", z.double(), w_o.double()))
        rs.append(torch.stack([cr[l] for l in range(engine.n_layers)]).double())
    return torch.cat(zs, 1), torch.cat(rs, 1)


def msr(x, axis):
    return ((x - x.mean(axis, keepdims=True)) ** 2).sum(-1).mean(axis)


def main():
    hashes = check_checkpoint("gpt2-small")
    engine = HeadPatchEngine("gpt2-small", "cuda", 64)
    out_npz, summary = {}, {"created_utc": utc(), "posthoc": True, "checkpoint_sha256": hashes, "settings": {}}
    for setting, task in SETTINGS.items():
        rows = read(ROOT / f"data/{task}-test.json")[:64]
        head_sp, resid_sp, gaps = [], [], []
        for r in rows:
            with np.load(ROOT / "results" / setting / "test" / f"{r['id']}.npz") as z:
                m = z["margins"] if task == "sva" else z["margins"][None]
            gaps.append(m[..., 64] - m[..., :64].mean(-1))                     # [D, L, H] logits
            if task == "sva":
                prompts = r["encoded"]["gpt2-small"]
                ids, pos = [p["ids"] for p in prompts], prompts[0]["end"]
                hz, rz = capture(engine, ids, pos)
                hs, rsd = [], []
                for side in (0, 1):                                            # restore donors, destroy donors
                    idx = [r["grid"][s][n][side] for s in range(8) for n in range(8)]
                    hs.append(msr(hz[:, idx].cpu().numpy(), 1))               # [L, H]
                    rsd.append(msr(rz[:, idx].cpu().numpy(), 1))              # [L]
                head_sp.append(np.stack(hs))
                resid_sp.append(np.stack(rsd))
            else:
                ids = [r["recipient"]["ids"]] + [d["ids"] for d in r["donors"]]
                pos = r["recipient"]["end"] if setting.startswith("ioi-end") else r["recipient"]["s2"]
                hz, rz = capture(engine, ids, pos)
                head_sp.append(msr(hz[:, 1:].cpu().numpy(), 1)[None])
                resid_sp.append(msr(rz[:, 1:].cpu().numpy(), 1)[None])
        H, R, Gp = np.stack(head_sp), np.stack(resid_sp), np.stack(gaps)       # [G, D, L, H], [G, D, L], [G, D, L, H]
        key = setting.replace("-gpt2-small", "")
        out_npz[f"{key}__head_spread"], out_npz[f"{key}__resid_spread"], out_npz[f"{key}__gap"] = H, R, Gp
        ratio = H / R[..., None]
        site_h, site_r = np.median(H, 0), np.median(R, 0)
        site_gap = np.median(np.abs(Gp), 0)
        ok = (site_gap > 0) & (site_h > 0)
        slope = float(np.polyfit(np.log10(site_h[ok]), np.log10(site_gap[ok]), 1)[0])
        rho = float(np.corrcoef(np.log10(site_h[ok]), np.log10(site_gap[ok]))[0, 1])
        summary["settings"][key] = {
            "median_head_spread": float(np.median(H)), "median_resid_spread": float(np.median(R)),
            "median_ratio_head_to_resid": float(np.median(ratio)),
            "ratio_percentiles_5_50_95": np.percentile(ratio, [5, 50, 95]).tolist(),
            "max_ratio": float(ratio.max()),
            "median_head_spread_by_layer": np.median(H, axis=(0, 1, 3)).tolist(),
            "median_resid_spread_by_layer": np.median(R, axis=(0, 1)).tolist(),
            "median_abs_gap": float(np.median(np.abs(Gp))), "max_site_median_abs_gap": float(site_gap.max()),
            "site_loglog_slope_gap_on_head_spread": slope, "site_log_correlation": rho}
    np.savez_compressed(ROOT / "results/posthoc/head-spread.npz", **out_npz)
    dump(ROOT / "results/posthoc/head-spread.json", summary)
    for k, v in summary["settings"].items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if "by_layer" not in kk})


if __name__ == "__main__":
    main()

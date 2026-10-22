"""CPU code test of analyze.py and verify.arrays() on synthetic arrays with the study's shapes (no model).

    python -B test_analysis.py

Writes random but well-formed arrays for the first 8 agreement groups and 8 IOI prompts to a temporary folder,
runs the frozen analysis and the independent array check, and requires the check to agree with the analysis.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

import analyze
import verify
from common import ROOT, read


def main():
    rng = np.random.default_rng(1)
    L, H = 28, 16
    sva = [{k: r[k] for k in ("id", "template", "class")} for r in read(ROOT / "data/sva-test.json")[:8]]
    ioi = [{k: r[k] for k in ("id", "template", "order")} for r in read(ROOT / "data/ioi-test.json")[:8]]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for r in sva:
            clean, corrupt = 3 + rng.normal(), -2 + rng.normal()
            base = np.array([clean, corrupt])
            a = {"baseline": base}
            for kind in ("resid", "attn", "mlp", "head"):
                shape = (2, L, H) if kind == "head" else (2, L)
                scale = {"resid": 2.0, "mlp": 0.6, "attn": 0.3, "head": 0.05}[kind]
                centre = np.stack([np.full(shape[1:], corrupt), np.full(shape[1:], clean)]) + rng.normal(0, scale, shape)
                m = centre[..., None] + rng.normal(0, scale, shape + (81,))
                m[..., 64] += rng.normal(0, scale / 4, shape)
                a[f"margins_{kind}"] = m
                a[f"spread_{kind}"] = np.abs(rng.normal(scale * 1000, scale * 100, shape + (17,)))
            (tmp / "A").mkdir(exist_ok=True)
            np.savez(tmp / "A" / f"{r['id']}.npz", **a)
            m = np.stack([np.full((L, 5), corrupt), np.full((L, 5), clean)])[..., None] + rng.normal(0, 0.5, (2, L, 5, 89))
            (tmp / "C1").mkdir(exist_ok=True)
            np.savez(tmp / "C1" / f"{r['id']}.npz", margins=m, spread=np.abs(rng.normal(100, 10, (2, L, 5, 25))), baseline=base,
                     positions=np.arange(6, 11), position_names=np.array(["adj", "anchor+0", "anchor+1", "anchor+2", "anchor+3"]))
        for r in ioi:
            t = 14 if r["template"] == "O0" else 15
            clean, donors = 4 + rng.normal(), rng.normal(0, 1, 64)
            for panel, extra in (("B1", 0), ("B2", 38)):
                (tmp / panel).mkdir(exist_ok=True)
                m = clean + rng.normal(0, 0.1, (L, H, 73 + extra))
                sizes = np.full(extra, 64 // max(extra, 1)) if extra else np.array([64])
                np.savez(tmp / panel / f"{r['id']}.npz", margins=m, baseline=np.array([clean]), donor_margins=donors,
                         token_group_sizes=sizes, token_groups_evaluated=np.array(extra))
            (tmp / "C2").mkdir(exist_ok=True)
            np.savez(tmp / "C2" / f"{r['id']}.npz", margins=clean + rng.normal(0, 0.5, (L, t, 73)),
                     spread=np.abs(rng.normal(100, 10, (L, t, 9))), baseline=np.array([clean]), donor_margins=donors,
                     null_positions=np.array([0]))
        for panel in ("A", "B1", "B2", "C1", "C2"):
            for p in (tmp / panel).glob("*.npz"):
                p.with_suffix(".json").write_text(json.dumps({"sha256": verify.digest(p)}))
        meta = tmp / "meta.json"
        meta.write_text(json.dumps({"sva": sva, "ioi": ioi}))
        out = tmp / "analysis"
        dec = analyze.analyze("test", str(tmp), str(out), str(meta))
        rep = verify.arrays(str(tmp), str(out), str(meta), str(out / "verification-arrays.json"))
        print(json.dumps({k: dec[k] for k in ("R1", "R2", "R3") if k in dec}, default=str)[:1500])
        print("verification problems:", rep["problems"])
        assert rep["values"]["arrays_hashed"] == 40
        assert not [p for p in rep["problems"] if "interval" not in p], rep["problems"]
    print("analysis code test passed")


if __name__ == "__main__":
    main()

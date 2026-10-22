"""Post hoc control for Experiment A: size-matched random 26-head circuits.

Added after the Experiment A results were known. Each random circuit keeps the same
number of heads as the Wang et al. circuit in every (layer, position role) cell
(final token, second subject mention, token after the first subject mention),
drawn from the heads of that layer that the real circuit does not use at that
position. It is evaluated on the same 1,299 recipients, the same saved donor banks
(results/reference/donor-banks-selected.json.gz), all four references and both
ablations. Frozen Experiment A files are only read.

Usage: python posthoc/random_circuits.py run | analyze
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import MODEL_PATHS, ROOT, check_checkpoint, dump, read, utc  # noqa: E402
from run_reference import CELLS, GROUPS, NODE_POSITION, LOGIT_TOL  # noqa: E402

N_CIRCUITS, SEED = 5, 26100611
OUT = ROOT / "results/posthoc/random-circuits"


def random_circuits():
    """Composition-matched random circuits: same head count per (layer, role), disjoint from the real circuit."""
    rng = random.Random(SEED)
    cells = {}
    for group, heads in GROUPS.items():
        for layer, head in heads:
            cells.setdefault((layer, NODE_POSITION[group]), []).append(head)
    circuits = []
    for _ in range(N_CIRCUITS):
        nodes = []
        for (layer, role), real in sorted(cells.items()):
            pool = [h for h in range(12) if h not in real]
            nodes += [(layer, role, h) for h in rng.sample(pool, len(real))]
        circuits.append(nodes)
    return circuits, {f"{l}/{r}": len(h) for (l, r), h in sorted(cells.items())}


def keep_mask(rec, nodes):
    import torch
    t = len(rec["ids"])
    mask = torch.zeros((12, t, 12, 1), dtype=torch.bool)
    for layer, role, head in nodes:
        mask[layer, rec[role], head] = True
    assert int(mask.sum()) == 26
    return mask


def margins(logits, rec):
    return (logits[:, rec["target_id"]].double() - logits[:, rec["contrast_id"]].double()).cpu().numpy()


def run():
    import torch
    from transformers import AutoTokenizer
    from engines import HeadEngine
    hashes = check_checkpoint("gpt2-small")
    circuits, composition = random_circuits()
    engine = HeadEngine("cuda", 64)
    tok = AutoTokenizer.from_pretrained(MODEL_PATHS["gpt2-small"], local_files_only=True)
    rows = read(ROOT / "data/ioi-selected-test.json")
    banks = json.load(gzip.open(ROOT / "results/reference/donor-banks-selected.json.gz", "rt"))["banks"]
    dest = OUT / "test"
    dest.mkdir(parents=True, exist_ok=True)
    dump(OUT / "design.json", {"created_utc": utc(), "posthoc": True, "seed": SEED, "circuits": circuits,
                               "composition_per_layer_and_role": composition, "checkpoint_sha256": hashes,
                               "environment": engine.environment()})
    start = time.time()
    for i, row in enumerate(rows):
        path = dest / f"{row['id']}.npz"
        if path.exists():
            continue
        rec = row["recipient"]
        p = rec["ids"]
        full, _, _ = engine.run([p])
        if abs(margins(full, rec)[0] - row["screen_margin"]) > LOGIT_TOL:
            raise RuntimeError(f"recipient margin not reproduced: {row['id']}")
        out = {"full_margin": margins(full, rec)[0]}
        for content in ["N", "NPO"]:
            texts = banks[row["id"]][content + "128"]
            ids = [tok.encode(t, add_special_tokens=False) for t in texts]
            _, src, _ = engine.run(ids, capture=True)
            for size in (8, 128):
                cell = f"{content}{size}"
                rep = torch.cat([src[:, :size].mean(1, keepdim=True), src[:, :size]], 1)
                for ci, nodes in enumerate(circuits):
                    log, _, audit = engine.run([p] * rep.shape[1], replacement=rep, keep=keep_mask(rec, nodes))
                    if audit["outside_changed_max"] != 0 or audit["replacement_error_max"] != 0:
                        raise RuntimeError("write locality failed")
                    m = margins(log, rec)
                    out[f"{cell}__c{ci}__centroid_margin"] = m[0]
                    out[f"{cell}__c{ci}__donor_margins"] = m[1:]
            del src
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in out.items()})
        if i % 50 == 0 or i == len(rows) - 1:
            print(json.dumps({"row": i + 1, "of": len(rows), "seconds": round(time.time() - start, 1)}), flush=True)
    dump(OUT / "test-completion.json", {"finished_utc": utc(), "rows": len(rows), "seconds": time.time() - start})


def analyze(draws=20000, seed=26100612):
    rows = read(ROOT / "data/ioi-selected-test.json")
    strata = read(ROOT / "results/reference/screen-selection.json")["strata"]
    keys = sorted({(r["family"], r["stratum_label"]) for r in rows})
    fields = [f"{c}/c{k}/{pool}" for c in CELLS for k in range(N_CIRCUITS) for pool in ["mean", "resample"]]
    data = {k: [] for k in keys}
    for r in rows:
        with np.load(OUT / "test" / f"{r['id']}.npz") as z:
            full = float(z["full_margin"])
            vals = []
            for c in CELLS:
                for k in range(N_CIRCUITS):
                    cm, dm = float(z[f"{c}__c{k}__centroid_margin"]), z[f"{c}__c{k}__donor_margins"]
                    vals += [float((cm > 0) == (full > 0)), float(np.mean((dm > 0) == (full > 0)))]
        data[(r["family"], r["stratum_label"])].append(vals)
    rng = np.random.default_rng(seed)
    point, boot, pop = {}, {}, {}
    for k in keys:
        F = np.array(data[k])
        n = len(F)
        counts = rng.multinomial(n, np.full(n, 1 / n), size=draws)
        point[k], boot[k] = F.mean(0), counts @ F / n
        pop[k] = strata[f"{k[0]}/{k[1]}"]["population"]
    out = {"created_utc": utc(), "posthoc": True, "circuits": N_CIRCUITS, "draws": draws, "tables": {}}
    for label in ["pair_error", "pair_correct"]:
        sel = [k for k in keys if k[1] == label]
        total = sum(pop[k] for k in sel)
        p = sum(pop[k] * point[k] for k in sel) / total
        b = sum(pop[k] * boot[k] for k in sel) / total
        table = {}
        for c in CELLS:
            for pool in ["mean", "resample"]:
                idx = [fields.index(f"{c}/c{k}/{pool}") for k in range(N_CIRCUITS)]
                per = [float(p[j]) for j in idx]
                avg_b = b[:, idx].mean(1)
                table[f"{c}/{pool}"] = {"per_circuit": per, "mean_over_circuits": float(np.mean(per)),
                                        "min": float(min(per)), "max": float(max(per)),
                                        "ci95_mean_over_circuits": np.percentile(avg_b, [2.5, 97.5]).tolist()}
        out["tables"][label] = table
    dump(ROOT / "results/posthoc/random-circuits.json", out)
    return out


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()

"""Post hoc (written after the replay check failed, 17 October 2026): could replay-sized numerical noise change a flag?

    python -B posthoc/replay_noise.py      (run from the study folder)

For each panel, the largest margin difference seen in its primary replays (study engine in fresh containers and
reference engine) is divided by the panel's denominator (clean minus corrupted, or clean minus mean donor margin).
That bounds how far any group-averaged normalised effect could move if every row carried that much noise (a
conservative bound: averaging over 64 rows can only shrink it). The script counts sites whose IND or MEAN effect
lies within that bound of tau = 0.05, and gives the closest distance. Reads only saved records; writes
results/posthoc/replay-noise.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
TAU = 0.05


def main():
    rep = read(RES / "replay-verification.json")["errors"]
    primary = {k: v for k, v in rep.items() if not k.startswith("replay-l4")}
    worst = {p: max(v for k, v in primary.items() if f"/{p}/" in k) for p in ("A", "B1", "B2", "C1", "C2")}
    a4 = read(RES / "analysis/analysis-A-test.json")
    heads = read(RES / "analysis/analysis-heads-test.json")["settings"]
    sweeps = read(RES / "analysis/analysis-sweeps-test.json")["panels"]
    denom_sva = heads["sva-end"]["denominator"]
    cases = [("A/" + k, worst["A"], denom_sva, a4["kinds"][k]["maps"]["IND"], a4["kinds"][k]["maps"]["MEAN"])
             for k in ("resid", "mlp", "attn", "head")]
    cases += [("B1", worst["B1"], heads["ioi-end"]["denominator"], heads["ioi-end"]["maps"]["IND"], heads["ioi-end"]["maps"]["MEAN"]),
              ("B2", worst["B2"], heads["ioi-s2"]["denominator"], heads["ioi-s2"]["maps"]["IND"], heads["ioi-s2"]["maps"]["MEAN"]),
              ("C1", worst["C1"], sweeps["sva-qwen3-1.7b"]["denominator"], sweeps["sva-qwen3-1.7b"]["maps"]["individual"],
               sweeps["sva-qwen3-1.7b"]["maps"]["mean64"]),
              ("C2", worst["C2"], sweeps["ioi-qwen3-1.7b"]["denominator"], sweeps["ioi-qwen3-1.7b"]["maps"]["individual"],
               sweeps["ioi-qwen3-1.7b"]["maps"]["mean64"])]
    out = {"post_hoc": True, "tau": TAU, "panels": {}}
    for name, err, denom, ind, mean in cases:
        ind, mean = np.abs(np.array(ind)), np.abs(np.array(mean))
        bound = err / denom
        dist = np.minimum(np.abs(ind - TAU), np.abs(mean - TAU))
        out["panels"][name] = {"largest_replay_margin_difference": err, "denominator": denom, "effect_bound": bound,
                               "sites_within_bound_of_tau": int((dist < bound).sum()), "closest_distance_to_tau": float(dist.min()),
                               "sites": int(dist.size)}
    out["any_flag_could_change"] = any(v["sites_within_bound_of_tau"] for v in out["panels"].values())
    target = RES / "posthoc/replay-noise.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


def read(p):
    return json.loads(Path(p).read_text())


if __name__ == "__main__":
    main()

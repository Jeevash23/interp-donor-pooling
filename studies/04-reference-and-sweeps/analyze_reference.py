"""Analysis for Experiment A (frozen with the protocol).

Unit: one recipient prompt with its complete donor banks. Estimates for model errors
and model successes are conditional means over the screened population, using
inverse inclusion weights within each template family. Intervals come from a
bootstrap over recipients within family-by-correctness strata (multinomial
resampling counts, 20,000 draws, fixed seed).
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from common import ROOT, dump, read, utc
from run_reference import CELLS, MODES

DRAWS, SEED = 20000, 26100531
METRICS = ["mean_agree", "resample_agree", "meanlogit_agree", "mean_top1_agree", "resample_top1_agree",
           "mean_margin", "resample_margin"]


def load(stage):
    if stage == "development":
        rows = read(ROOT / "data/ioi-development.json")
        for r in rows:
            r["weight"] = 1.0
    else:
        rows = read(ROOT / "data/ioi-selected-test.json")
        # Amendment 1: strata and population sizes come from the screen record.
        strata = read(ROOT / "results/reference/screen-selection.json")["strata"]
        for r in rows:
            r["population"] = strata[f"{r['family']}/{r['stratum_label']}"]["population"]
    recs = []
    for r in rows:
        with np.load(ROOT / "results/reference" / stage / f"{r['id']}.npz") as z:
            d = {k.replace("__", "/"): z[k] for k in z.files}
        full = float(d["full_margin"])
        rec = {"id": r["id"], "family": r["family"], "template": r["template"], "order": r["order"],
               "weight": float(r["weight"]),
               "error": int(r["stratum_label"] == "pair_error") if "stratum_label" in r else int(full <= 0),
               "population": r.get("population"), "full_margin": full,
               "full_argmax": int(d["full_argmax"])}
        for c in CELLS:
            rec[f"{c}/corrupt_margin"] = float(np.mean(d[f"{c}/donor_prompt_margin"]))
            for m in MODES:
                cm, dm = float(d[f"{c}/{m}/centroid_margin"]), d[f"{c}/{m}/donor_margins"]
                rec[f"{c}/{m}/mean_agree"] = float((cm > 0) == (full > 0))
                rec[f"{c}/{m}/resample_agree"] = float(np.mean((dm > 0) == (full > 0)))
                rec[f"{c}/{m}/meanlogit_agree"] = float((float(d[f"{c}/{m}/meanlogit_margin"]) > 0) == (full > 0))
                rec[f"{c}/{m}/mean_top1_agree"] = float(int(d[f"{c}/{m}/centroid_argmax"]) == rec["full_argmax"])
                rec[f"{c}/{m}/resample_top1_agree"] = float(np.mean(d[f"{c}/{m}/donor_argmax"] == rec["full_argmax"]))
                rec[f"{c}/{m}/mean_margin"] = cm
                rec[f"{c}/{m}/resample_margin"] = float(np.mean(dm))
        recs.append(rec)
    return recs


class Bootstrap:
    """Stratified bootstrap of population-weighted means for many fields at once."""

    def __init__(self, recs, fields):
        rng = np.random.default_rng(SEED)
        self.fields = {f: i for i, f in enumerate(fields)}
        keys = sorted({(r["family"], r["error"]) for r in recs})
        self.keys = keys
        self.point, self.draws, self.population = {}, {}, {}
        for k in keys:
            ix = [i for i, r in enumerate(recs) if (r["family"], r["error"]) == k]
            F = np.array([[recs[i][f] for f in fields] for i in ix], dtype=float)
            n = len(ix)
            counts = rng.multinomial(n, np.full(n, 1.0 / n), size=DRAWS)
            self.point[k] = F.mean(0)
            self.draws[k] = counts @ F / n
            pop = recs[ix[0]].get("population")
            self.population[k] = float(pop) if pop is not None else recs[ix[0]]["weight"] * n

    def mean(self, field, select):
        j = self.fields[field]
        ks = [k for k in self.keys if select(k)]
        p = np.array([self.population[k] for k in ks])
        point = sum(self.population[k] * self.point[k][j] for k in ks) / p.sum()
        draws = sum(self.population[k] * self.draws[k][:, j] for k in ks) / p.sum()
        return point, draws


def summarize(point, draws, alpha=0.05):
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(point), "ci": [float(lo), float(hi)], "confidence": 1 - alpha}


def analyze(stage):
    recs = load(stage)
    fields = ["full_margin"] + [f"{c}/{m}/{x}" for c in CELLS for m in MODES for x in METRICS]
    bs = Bootstrap(recs, fields)
    fams = {"pooled": lambda k: True, "O": lambda k: k[0] == "O", "W": lambda k: k[0] == "W"}
    out = {"created_utc": utc(), "stage": stage, "recipients": len(recs), "draws": DRAWS, "seed": SEED,
           "counts": {f"{f}/{'error' if e else 'correct'}": sum(1 for r in recs if (r["family"], r["error"]) == (f, e))
                      for f, e in bs.keys},
           "population": {f"{f}/{'error' if e else 'correct'}": bs.population[(f, e)] for f, e in bs.keys},
           "tables": {}}
    for fam, fsel in fams.items():
        if not any(fsel(k) for k in bs.keys):
            continue
        for label, want in [("error", 1), ("correct", 0)]:
            sel = lambda k, fsel=fsel, want=want: fsel(k) and k[1] == want
            if not any(sel(k) for k in bs.keys):
                continue
            table = {}
            for c in CELLS:
                for m in MODES:
                    for x in METRICS:
                        table[f"{c}/{m}/{x}"] = summarize(*bs.mean(f"{c}/{m}/{x}", sel))
                for x in ["mean_agree", "resample_agree"]:
                    p1, d1 = bs.mean(f"{c}/reference/{x}", sel)
                    p2, d2 = bs.mean(f"{c}/empty/{x}", sel)
                    table[f"{c}/excess_over_empty/{x}"] = summarize(p1 - p2, d1 - d2)
                p1, d1 = bs.mean(f"{c}/reference/mean_agree", sel)
                p2, d2 = bs.mean(f"{c}/reference/resample_agree", sel)
                table[f"{c}/reference/mean_minus_resample_agree"] = summarize(p1 - p2, d1 - d2)
            out["tables"][f"{fam}/{label}"] = table
        faith = {}
        for c in CELLS:
            for pool in ["mean", "resample"]:
                lf, dlf = bs.mean("full_margin", fsel)
                lc, dlc = bs.mean(f"{c}/reference/{pool}_margin", fsel)
                le, dle = bs.mean(f"{c}/empty/{pool}_margin", fsel)
                faith[f"{c}/{pool}"] = {**summarize((lc - le) / (lf - le), (dlc - dle) / (dlf - dle)),
                                        "denominator": float(lf - le), "reference_margin": float(lc),
                                        "empty_margin": float(le), "full_margin": float(lf)}
        out["tables"][f"{fam}/faithfulness"] = faith
    err = lambda k: k[1] == 1
    if not any(err(k) for k in bs.keys):
        out["primary"] = out["secondary_contrasts"] = None
        dump(ROOT / f"results/reference/analysis-{stage}.json", out)
        return out
    a1p, a1d = bs.mean("NPO128/reference/mean_agree", err)
    b1p, b1d = bs.mean("N8/reference/mean_agree", err)
    a2p, a2d = bs.mean("N8/empty/mean_agree", err)
    out["primary"] = {
        "A1_npo128_minus_n8_error_agreement_mean_ablation": summarize(a1p - b1p, a1d - b1d, alpha=0.025),
        "A2_circuit_minus_empty_error_agreement_n8_mean_ablation": summarize(b1p - a2p, b1d - a2d, alpha=0.025),
    }
    sec = {}
    for x in ["mean_agree", "resample_agree"]:
        def g(c, x=x):
            return bs.mean(f"{c}/reference/{x}", err)
        for name, (hi, lo) in {"content_effect_size8": ("NPO8", "N8"), "content_effect_size128": ("NPO128", "N128"),
                               "size_effect_names_only": ("N128", "N8"),
                               "size_effect_names_place_object": ("NPO128", "NPO8")}.items():
            (ph, dh), (pl, dl) = g(hi), g(lo)
            sec[f"{name}/{x}"] = summarize(ph - pl, dh - dl)
    for x in ["mean_agree", "resample_agree"]:
        # Amendment 2: main effects as averages of the two simple effects.
        g = {c: bs.mean(f"{c}/reference/{x}", err) for c in CELLS}
        content = (0.5 * (g["NPO8"][0] - g["N8"][0] + g["NPO128"][0] - g["N128"][0]),
                   0.5 * (g["NPO8"][1] - g["N8"][1] + g["NPO128"][1] - g["N128"][1]))
        size = (0.5 * (g["N128"][0] - g["N8"][0] + g["NPO128"][0] - g["NPO8"][0]),
                0.5 * (g["N128"][1] - g["N8"][1] + g["NPO128"][1] - g["NPO8"][1]))
        sec[f"content_main_effect/{x}"] = summarize(*content)
        sec[f"size_main_effect/{x}"] = summarize(*size)
    out["secondary_contrasts"] = sec
    dump(ROOT / f"results/reference/analysis-{stage}.json", out)
    with (ROOT / f"results/reference/recipients-{stage}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["development", "test"])
    a = analyze(ap.parse_args().stage)
    print(a["counts"])
    for k, v in (a["primary"] or {}).items():
        print(k, round(v["estimate"], 4), [round(x, 4) for x in v["ci"]])

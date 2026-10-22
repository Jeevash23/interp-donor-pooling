"""Independent input and weighted-summary checks; no model inference."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("circuit_experiment", ROOT / "experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def main():
    all_strings = set()
    counts = {}
    groups = [tuple(x) for vals in e.GROUPS.values() for x in vals]
    assert len(groups) == len(set(groups)) == 26
    assert {(9,9),(9,6),(10,0)} <= set(groups)
    for split,n in [("calibration",16),("development",32),("screen",8192)]:
        rows = e.read(ROOT / f"data/{split}.json")
        assert len(rows) == n
        strata = {}
        for r in rows:
            strata[r["stratum"]] = strata.get(r["stratum"],0)+1
            p = r["recipient"]
            assert len(r["donors"]) == len(r["donor_names"]) == 8
            assert 0 <= p["s1"] < p["s1_plus1"] < p["s2"] < p["end"]
            assert p["target_id"] != p["contrast_id"]
            for d,names in zip(r["donors"],r["donor_names"]):
                assert len(set(names)) == 3
                assert not (set(names) & {r["target"],r["subject"]})
                assert len(d["ids"]) == len(p["ids"])
                assert [j for j,(a,b) in enumerate(zip(d["ids"],p["ids"])) if a != b] == r["name_positions"]
            texts = [p["text"]]+[d["text"] for d in r["donors"]]
            assert len(set(texts)) == 9
            assert not (set(texts) & all_strings)
            all_strings.update(texts)
        assert len(strata) == 8 and len(set(strata.values())) == 1
        counts[split] = {"groups":n,"strata":strata}
    prior,_ = e.old_strings()
    assert not (prior & all_strings)

    # The deliberately enriched sample has two errors and six successes,
    # whereas the screen has 10% errors. Known paired margin effects are +2
    # for errors and -1 for successes, so correct population weighting is -0.7
    # (the incorrect unweighted enriched-sample mean would be -0.25).
    with tempfile.TemporaryDirectory(prefix="circuit-analysis-check-",dir=ROOT) as tmp:
        out = Path(tmp)
        rows = []
        e.dump(out / "data/freeze.json",{"files":{}})
        e.dump(out / "results/screen-selection.json",{"synthetic_unit_test":True})
        (out/"results/test").mkdir()
        for i in range(8):
            error = i < 2
            base_margin = -1 if error else 1
            gap = 2 if error else -1
            baseline = np.zeros(9)
            baseline[0] = base_margin
            baseline[3:5] = not error
            baseline[5:7] = 1
            m = np.zeros((2,9,9))
            m[0,:,0] = base_margin
            m[0,0,0] += gap
            m[0,:,6] = 0.5
            m[0,0,6] = 0.75 if error else 0.25
            r = {"id":f"synthetic-{i}","selection_stratum":"pair_error" if error else "pair_correct",
                 "population_weight":5 if error else 15,"template":i%4}
            rows.append(r)
            np.savez_compressed(out / "results/test" / (r["id"]+".npz"),
                metrics=m,baseline=baseline,mixture_metrics=np.zeros((2,9)),mean_logit_metrics=np.zeros((2,9)),
                prediction=np.zeros((2,9),dtype=int),baseline_prediction=np.array([0]),
                identity_logit_error=np.array(0.),all_kept_logit_error=np.array(0.))
        e.dump(out / "data/selected-test.json",rows)
        e.analyze(SimpleNamespace(out=out))
        got=e.read(out/"results/analysis.json")["circuits"]["reference"]
        assert np.isclose(got["mean_minus_expected_resample"]["margin"]["mean"],-.7)
        assert np.isclose(got["strata"]["full_pair_error"]["paired_pair_agreement_difference"]["mean"],.25)
        assert np.isclose(got["strata"]["full_pair_success"]["paired_pair_agreement_difference"]["mean"],-.25)
        assert got["strata"]["full_pair_error"]["n"] == 2
    report={"passed":True,"created_utc":e.utc(),"data":counts,"unique_strings":len(all_strings),
            "prior_overlap":0,"projection_mask_reference_nodes":26,
            "analysis_weighting_fixture_expected":-.7,"analysis_weighting_fixture_passed":True,
            "synthetic_fixtures_are_temporary_and_not_scientific_results":True,
            "source_sha256":e.sha(ROOT/"experiment.py")}
    e.dump(ROOT/"results/input-analysis-checks.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()

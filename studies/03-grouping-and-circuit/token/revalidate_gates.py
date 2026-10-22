"""Admit recorded calibration/development audits under stricter pretest gates.

No model execution and no scientific measurement is changed. The initial
calibration had already recorded exact write errors; the initial development
runner had already recorded probability closure. Two missing explicit gate
checks were added before any test results, then applied to these saved records.
"""
from common import *

def main():
    models = {}
    for model in MODEL_PATHS:
        folder = ROOT / "results" / model
        calibration = read(folder / "calibration.json")
        exact = {k: v for k, v in calibration["errors"].items() if k.startswith(("outside_", "write_"))}
        audits = [a for row in read(ROOT / "data/development.json") for a in read(folder / "development" / (row["id"] + "-audit.json"))]
        models[model] = {"calibration_original_sha256": sha(folder / "calibration.json"), "calibration_original_passed": calibration["passed"], "calibration_exact_write_checks": len(exact), "calibration_exact_write_max": max(exact.values()), "development_audits": len(audits), "development_probability_closure_max": max(a["probability_closure_max"] for a in audits), "development_outside_max": max(a["outside_max"] for a in audits), "development_write_max": max(a["target_error"] for a in audits)}
        m = models[model]
        m["passed"] = m["calibration_original_passed"] and m["calibration_exact_write_max"] == m["development_outside_max"] == m["development_write_max"] == 0 and m["development_probability_closure_max"] <= .00002
    report = {"created_utc": utc(), "scope": "Audit-only gate correction before any test measurements; original calibration and development records retained unchanged", "gate_changes": ["Calibration requires recorded outside/write errors exactly zero, rather than accepting them under generic .001 tolerance", "Development/test explicitly require recorded probability closure <=.00002, in addition to identity-probability gate"], "scientific_forward_or_metric_changes": False, "models": models, "passed": all(m["passed"] for m in models.values())}
    dump(ROOT / "results/pretest-gate-revalidation.json", report)
    if not report["passed"]:
        raise RuntimeError("Recorded audits fail tightened pretest gates")
    print("Recorded calibration/development audits pass both explicit gate corrections.")

if __name__ == "__main__":
    main()

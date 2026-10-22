"""Compare the Modal reproduction with the saved AI-03 arrays (project environment, CPU, no model).

    python -B compare_smoke.py

Tolerance: 0.001 logits, the frozen replay and calibration tolerance of Studies 2-4, on every margin.
Spreads (squared distances, up to about 2,000) are compared by relative difference where the saved value is at
least 0.001, and reported only; entries below that are zero up to rounding (identical donors) and get their
absolute difference. (The first version of this script, whose output is records/comparison-v1.json, used a
1e-12 floor and so reported a relative error of 8.45 from an entry saved as 1.8e-10.)
The Study 3 gate is compared as the saved accuracy (the gate saved no margins).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parents[1]
REC = HERE / "records"
TOL = 1e-3
FLOOR = 1e-3


def compare(a, b):
    d = np.abs(a.astype(float) - b.astype(float))
    big = np.abs(b) >= FLOOR
    return dict(shape=list(b.shape), max_abs=float(d.max()),
                max_rel=float((d[big] / np.abs(b[big])).max()) if big.any() else None,
                entries_below_floor=int((~big).sum()), max_abs_below_floor=float(d[~big].max()) if (~big).any() else None)


checks = {}
with np.load(REC / "modal-s4.npz") as got, \
        np.load(STUDIES / "06-component-confirmation/results/test/sva-test-sva-000.npz") as saved:
    assert sorted(got.files) == sorted(saved.files), (got.files, saved.files)
    checks["s4"] = {k: compare(got[k], saved[k]) for k in sorted(saved.files)}
with np.load(REC / "modal-s2.npz") as got, \
        np.load(STUDIES / "04-reference-and-sweeps/results/sweep/sva-qwen3-0.6b/test/sva-test-sva-000.npz") as saved:
    assert sorted(got.files) == sorted(saved.files), (got.files, saved.files)
    checks["s2"] = {}
    for k in sorted(saved.files):
        if saved[k].dtype.kind in "fi":
            checks["s2"][k] = compare(got[k], saved[k])
        else:
            checks["s2"][k] = dict(shape=list(saved[k].shape), equal=bool(np.array_equal(got[k], saved[k])))
gate_saved = json.loads((STUDIES / "05-head-confirmation/results/qwen-gate/gate.json").read_text())
gate_got = json.loads((REC / "modal-gate.json").read_text())
margins = np.array(gate_got["margins"])
checks["gate"] = dict(saved_accuracy=gate_saved["clean_two_verb_accuracy"], modal_accuracy=gate_got["clean_two_verb_accuracy"],
                      groups=len(margins), correct=int((margins > 0).sum()),
                      smallest_abs_margin=float(np.abs(margins).min()),
                      equal=gate_got["clean_two_verb_accuracy"] == gate_saved["clean_two_verb_accuracy"])

margin_keys = [("s4", k) for k in checks["s4"] if k.startswith("margins") or k == "baseline"] + \
              [("s2", k) for k in ("margins", "baseline")]
worst = max(checks[s][k]["max_abs"] for s, k in margin_keys)
spread_rel = max(checks[s][k]["max_rel"] for s in ("s4", "s2") for k in checks[s] if k.startswith("spread"))
spread_tiny = max(checks[s][k]["max_abs_below_floor"] or 0.0 for s in ("s4", "s2") for k in checks[s] if k.startswith("spread"))
other_equal = all(v.get("equal", True) for v in checks["s2"].values())
passed = worst <= TOL and checks["gate"]["equal"] and other_equal
target = REC / "comparison.json"
assert not target.exists(), target
target.write_text(json.dumps(dict(
    tolerance_logits=TOL, max_abs_margin_error=worst, max_rel_spread_error=spread_rel, spread_floor=FLOOR,
    max_abs_spread_error_below_floor=spread_tiny, passed=passed,
    per_array=checks,
    compared_with=dict(s4="studies/06-component-confirmation/results/test/sva-test-sva-000.npz",
                       s2="studies/04-reference-and-sweeps/results/sweep/sva-qwen3-0.6b/test/sva-test-sva-000.npz",
                       gate="studies/05-head-confirmation/results/qwen-gate/gate.json")), indent=2) + "\n")
print(json.dumps(dict(passed=passed, max_abs_margin_error=worst, max_rel_spread_error=spread_rel,
                      max_abs_spread_error_below_floor=spread_tiny,
                      gate=checks["gate"])))

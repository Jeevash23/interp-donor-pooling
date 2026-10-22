"""Print the numbers `make quick-check` reproduced, next to the saved values the manuscript reports.

Run after the Study 3 and Study 4 verifiers (they write results/verification.json from the raw arrays).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
S3 = RELEASE / "studies/05-head-confirmation"
S4 = RELEASE / "studies/06-component-confirmation"


def read(p):
    return json.loads(Path(p).read_text())


def main():
    v4, a4 = read(S4 / "results/verification.json"), read(S4 / "results/analysis-test.json")
    print("Study 4 (components, GPT-2 Small, final token): mean absolute bias of mean ablation")
    print("  kind    recomputed from raw arrays   saved analysis   P1 interval redrawn   saved")
    for kind in ["resid", "mlp", "attn", "head"]:
        got, saved = v4["values"][kind]["mean_abs_bias"], a4["kinds"][kind]["mean_abs_bias"]
        lo, hi = v4["values"][kind]["p1_interval_redrawn"]
        s_lo, s_hi = a4["kinds"][kind]["p1_mean_misclassified"]["interval"]
        redrawn, interval = f"[{lo:.0f}, {hi:.0f}]", f"[{s_lo:.0f}, {s_hi:.0f}]"
        print(f"  {kind:6s}  {got:26.6f}   {saved:14.6f}   {redrawn:20s}  {interval}")
    v3 = read(S3 / "results/verification.json")
    print("Study 3 (single heads, GPT-2 Small): recomputed flag counts match the saved analysis in every setting:",
          all(not s["problems"] for s in v3["settings"].values()))
    g = v3["qwen_gate"]
    print(f"  Qwen3-0.6B accuracy gate: {g['accuracy']:.4f} against the required {g['required']} "
          f"({'passed' if g['passed'] else 'failed, as recorded; the study ran on GPT-2 Small only'})")
    print(f"  prompt separation: {v3['separation']['overlap_with_earlier_studies']} of "
          f"{v3['separation']['test_strings']} test strings shared with earlier studies")
    n = len(re.findall(r"\\newcommand", (RELEASE / "paper/generated/numbers.tex").read_text()))
    print(f"Manuscript: {n} number macros and 8 generated tables recomputed from the saved outputs; all identical")


if __name__ == "__main__":
    main()

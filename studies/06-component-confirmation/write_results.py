"""Write RESULTS.md from the saved analysis, verification and run records (no hand-typed numbers)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {"resid": "Residual stream", "mlp": "MLP output", "attn": "Attention-layer output", "head": "Single head"}


def read(p):
    return json.loads(Path(p).read_text())


def m3(x):
    return f"{1e3 * x:.2f}"


def main():
    a = read(ROOT / "results/analysis-test.json")
    v = read(ROOT / "results/verification.json")
    cal = read(ROOT / "results/calibration.json")
    rep = read(ROOT / "results/replay-verification.json")
    done = read(ROOT / "results/test-completion.json")["finished_utc"]
    frozen = read(ROOT / "data/freeze.json")["frozen_utc"]
    lines = ["# Study 4 results (folder `06-component-confirmation`)", "",
             f"Test runs finished at {done}, after the freeze at {frozen}. Every outcome below follows `PROTOCOL.md`; nothing was "
             "changed after test outputs. This file is written by `write_results.py` from `results/analysis-test.json`, "
             "`results/verification.json` and the run records.", "",
             "## Checks", "",
             f"- Calibration: {'passed' if cal['passed'] else 'FAILED'}; largest logit error {cal['maximum_logit_error']:.1e}, largest "
             f"probability error {cal['maximum_probability_error']:.1e}, exact write locality: {cal['exact_write_locality']}.",
             f"- Fresh-process replays of the first two test groups: maximum difference {rep['maximum_error']:.1e} logits "
             f"({'passed' if rep['passed'] else 'FAILED'}).",
             f"- Independent verification (`verify.py`): {'passed' if v['passed'] else 'FAILED'}. Frozen files changed: "
             f"{len(v['frozen_files_changed'])}. Test prompts overlapping calibration/development: "
             f"{v['separation']['overlap_with_calibration_or_development']}; overlapping earlier studies: "
             f"{v['separation']['overlap_with_earlier_studies']} (of {v['separation']['prior_strings']:,} earlier strings).", "",
             "## Outcomes", "",
             "Sites are site-directions at the final token. Misclassified sites at tau = 0.05 against the average of 64 individual "
             "patches (IND); mean absolute error in normalized effect, in units of 1e-3, in brackets. S8: mean over 1,000 draws (95% range).", "",
             "| Kind | Sites | Flagged by IND | MEAN (98.75% interval) | TOK | MIX8 | S8 | Mean abs. bias | Median spread |",
             "|---|---:|---:|---|---|---|---|---:|---:|"]
    for kind, name in NAMES.items():
        r = a["kinds"][kind]
        m = r["methods"]
        p1 = r["p1_mean_misclassified"]
        s8 = m["S8"]
        lines.append(f"| {name} | {r['sites']} | {m['MEAN']['tau0.05']['flagged_ind']} | {p1['estimate']} [{p1['interval'][0]:g}, "
                     f"{p1['interval'][1]:g}] ({m3(m['MEAN']['mae'])}) | {m['TOK']['tau0.05']['misclassified']} ({m3(m['TOK']['mae'])}) | "
                     f"{m['MIX8']['tau0.05']['misclassified']} ({m3(m['MIX8']['mae'])}) | {s8['tau0.05']['misclassified_mean']:.2f}, "
                     f"{s8['tau0.05']['misclassified_range95'][0]:g}-{s8['tau0.05']['misclassified_range95'][1]:g} ({m3(s8['mae_mean'])}) | "
                     f"{r['mean_abs_bias']:.4f} | {r['spread']['median']:,.1f} |")
    lines += ["", "**Decisions.**", ""]
    for kind, name in NAMES.items():
        p1 = a["kinds"][kind]["p1_mean_misclassified"]
        lines.append(f"- P1, {name}: predicted {p1['predicted']}; estimate {p1['estimate']}, interval [{p1['interval'][0]:g}, "
                     f"{p1['interval'][1]:g}]: **{'supported' if p1['supported'] else 'not supported'}**.")
    for kind, name in NAMES.items():
        p2 = a["kinds"][kind]["p2_tok_vs_s8"]
        lines.append(f"- P2, {name}: TOK error {m3(p2['tok_mae'])} against S8 2.5th-97.5th percentiles {m3(p2['s8_mae_2.5'])}-"
                     f"{m3(p2['s8_mae_97.5'])} (x1e-3): {p2['decision']}; predicted TOK better: **{'supported' if p2['supported'] else 'not supported'}**.")
    o = a["p3_ordering"]
    lines.append("- P3, ordering resid > mlp > attn > head of mean absolute bias: " + "; ".join(
        f"{pair} {o[pair]['difference']:.4f} [{o[pair]['ci95'][0]:.4f}, {o[pair]['ci95'][1]:.4f}]" for pair in ["resid-mlp", "mlp-attn", "attn-head"])
        + f": **{'supported' if o['supported'] else 'not supported'}** (full order in {100 * o['bootstrap_share_full_order']:.0f}% of bootstrap draws).")
    p4 = a["p4_spread_order"]
    lines.append(f"- P4, spread: kinds by spread {p4['kinds_by_spread']}, by bias {p4['kinds_by_bias']}: "
                 f"**{'supported' if p4['supported'] else 'not supported'}**.")
    lines += ["", "**Bias against sampling noise at sites near the threshold** (IND between 0.025 and 0.10):", ""]
    for kind, name in NAMES.items():
        bn = a["kinds"][kind]["bias_vs_noise"]
        if bn["median_bias_near"] is not None:
            lines.append(f"- {name}: {bn['near_threshold_sites']} sites; median centroid bias {bn['median_bias_near']:.4f}, median "
                         f"eight-donor SD {bn['median_sd8_near']:.4f}; bias above SD at {100 * bn['share_bias_above_sd8_near']:.0f}%.")
        else:
            lines.append(f"- {name}: no sites near the threshold.")
    lines += ["", "Every site, with its effect under IND, MEAN, TOK and MIX8, its flags, spread and eight-donor SD, is in "
              "`results/sites-test.csv`. The manuscript's Section 8 reports these results.", "",
              "Note: the protocol's development section quotes 0.0032 for the final-token error of eight sampled donors in Study 2, "
              "from a 300-draw check run while writing it; the saved 1,000-draw value (Study 2 `results/posthoc/bias-noise.json`) is 0.0033.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

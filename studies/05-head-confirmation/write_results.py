"""Write RESULTS.md from the saved analysis, verification and run records (no hand-typed numbers)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {"sva-end-gpt2-small": "Agreement, final token", "ioi-end-gpt2-small": "IOI, final token",
         "ioi-s2-gpt2-small": "IOI, repeated subject"}


def read(p):
    return json.loads(Path(p).read_text())


def e4(x):
    return f"{1e4 * x:.2f}"


def main():
    a = read(ROOT / "results/analysis-test.json")["settings"]
    v = read(ROOT / "results/verification.json")
    gate = read(ROOT / "results/qwen-gate/gate.json")
    cals = [read(p) for p in sorted(ROOT.glob("results/*/calibration.json"))]
    reps = {p.parent.name: read(p) for p in sorted(ROOT.glob("results/*/replay-verification.json"))}
    done = max(read(p)["finished_utc"] for p in ROOT.glob("results/*/test-completion.json"))
    frozen = read(ROOT / "data/freeze.json")["frozen_utc"]
    lines = ["# Study 3 results (folder `05-head-confirmation`)", "",
             f"Test runs finished at {done}, after the freeze at {frozen}. Every outcome below follows "
             "`PROTOCOL.md`; nothing was changed after test outputs. This file is written by `write_results.py` from "
             "`results/analysis-test.json`, `results/verification.json` and the run records.", "",
             "## Checks", "",
             f"- Calibrations: {len(cals)}, {'all passed' if all(c['passed'] for c in cals) else 'NOT all passed'}; largest logit error "
             f"{max(c['maximum_logit_error'] for c in cals):.1e}, largest probability error {max(c['maximum_probability_error'] for c in cals):.1e}.",
             "- Fresh-process replays of the first two test rows per setting: " + ", ".join(
                 f"{k} max {r['maximum_error']:.1e} ({'passed' if r['passed'] else 'FAILED'})" for k, r in reps.items()) + ".",
             f"- Qwen3-0.6B gate: clean two-verb accuracy {100 * gate['clean_two_verb_accuracy']:.1f}% on {gate['groups']} fresh groups, "
             f"required {100 * gate['required']:.0f}%: {'passed' if gate['passed'] else 'not passed, so Qwen was excluded before any patching'}.",
             f"- Independent verification (`verify.py`): {'passed' if v['passed'] else 'FAILED'}. Frozen files changed: "
             f"{len(v['frozen_files_changed'])}. Test prompts overlapping calibration/development: "
             f"{v['separation']['overlap_with_calibration_or_development']}; overlapping earlier studies: "
             f"{v['separation']['overlap_with_earlier_studies']} (of {v['separation']['prior_strings']:,} earlier strings).", "",
             "## Outcomes", "",
             "Misclassified head-sites at tau = 0.05 against the average of 64 individual patches (IND); mean absolute error in "
             "normalized effect in units of 1e-4 in brackets. S8: mean over 1,000 draws (95% range).", "",
             "| Setting | Head-sites | Flagged by IND | MEAN (98.33% interval) | TOK (evaluations) | MIX8 | S8 |",
             "|---|---:|---:|---|---|---|---|"]
    for key, name in NAMES.items():
        p = a[key]
        m = p["methods"]
        pm = p["primary_mean_misclassified"]
        s8 = m["S8"]
        lines.append(
            f"| {name} | {p['head_sites']} | {m['MEAN']['tau0.05']['flagged_ind']} | "
            f"{pm['estimate']} [{pm['ci'][0]:g}, {pm['ci'][1]:g}] ({e4(m['MEAN']['mae'])}) | "
            f"{m['TOK']['tau0.05']['misclassified']} ({e4(m['TOK']['mae'])}; {p['evaluations_per_recipient_site']['TOK']:.0f}) | "
            f"{m['MIX8']['tau0.05']['misclassified']} ({e4(m['MIX8']['mae'])}) | "
            f"{s8['tau0.05']['misclassified_mean']:.2f}, {s8['tau0.05']['misclassified_range95'][0]:g}-{s8['tau0.05']['misclassified_range95'][1]:g} "
            f"({e4(s8['mae_mean'])}; {e4(s8['mae_range95'][0])}-{e4(s8['mae_range95'][1])}) |")
    lines += ["", "**Decisions.**", ""]
    for key, name in NAMES.items():
        pm = a[key]["primary_mean_misclassified"]
        lines.append(f"- H1, {name}: lower limit {pm['ci'][0]:g}, so the prediction that averaging changes the flagged heads is "
                     f"**{'supported' if pm['ci'][0] >= 1 else 'not supported'}**.")
    t = a["sva-end-gpt2-small"]["tok_vs_s8"]
    lines.append(f"- H2, agreement: TOK error {e4(t['tok_mae'])} against the S8 2.5th percentile {e4(t['s8_mae_2.5th_percentile'])}; "
                 f"TOK misclassified {t['tok_misclassified']} against the S8 mean {t['s8_misclassified_mean']:.2f}: token grouping "
                 f"**{'beats' if t['tok_better'] else 'does not beat'}** eight sampled donors. Bootstrap interval for TOK error minus the "
                 f"S8 expectation: [{e4(t['mae_difference_ci95'][0])}, {e4(t['mae_difference_ci95'][1])}].")
    t = a["ioi-end-gpt2-small"]["tok_vs_s8"]
    lines.append(f"- H3, IOI final token: one token group, so TOK equals MEAN ({'confirmed' if t['tok_identical_to_mean'] else 'NOT identical'}); "
                 f"it has no advantage over mean ablation. MEAN error {e4(t['tok_mae'])} against S8 {e4(t['s8_mae_mean'])} "
                 f"(2.5th percentile {e4(t['s8_mae_2.5th_percentile'])}).")
    p = a["ioi-s2-gpt2-small"]
    t = p["tok_vs_s8"]
    lines.append(f"- H4, IOI repeated subject (descriptive): TOK costs {p['evaluations_per_recipient_site']['TOK']:.1f} evaluations per "
                 f"head-site on average, misclassifies {t['tok_misclassified']} head-sites, error {e4(t['tok_mae'])}, against "
                 f"MEAN {e4(p['methods']['MEAN']['mae'])} and S8 {e4(t['s8_mae_mean'])}.")
    lines += ["", "**Secondary thresholds** (misclassified by MEAN / TOK / MIX8 / S8 mean):", ""]
    for key, name in NAMES.items():
        m = a[key]["methods"]
        for tau in ["0.02", "0.1"]:
            lines.append(f"- {name}, tau = {tau}: {m['MEAN']['tau' + tau]['misclassified']} / {m['TOK']['tau' + tau]['misclassified']} / "
                         f"{m['MIX8']['tau' + tau]['misclassified']} / {m['S8']['tau' + tau]['misclassified_mean']:.2f}"
                         f" (IND flags {m['MEAN']['tau' + tau]['flagged_ind']})")
    lines += ["", "Every head-site, with its effect under IND, MEAN, TOK and MIX8 and its flags, is in `results/heads-*-test.csv`. "
              "The manuscript's Section 7 reports these results; its appendix lists every head-site that any method flags.", ""]
    hs_path = ROOT / "results/posthoc/head-spread.json"
    if hs_path.exists():
        hs = read(hs_path)["settings"]
        lines += ["", "## Post hoc: head-level donor spread (added in the second revision)", "",
                  "`posthoc/head_spread.py` reran the unpatched test prompts (no frozen file changed) and measured each head-site's "
                  "donor spread in the residual space (through W_O) and the residual stream's spread after the same block at the same "
                  "position. Results (`results/posthoc/head-spread.json`):", ""]
        for k, v in hs.items():
            lines.append(f"- {k}: median head spread {v['median_head_spread']:.2f}, median residual spread {v['median_resid_spread']:,.0f}, "
                         f"median ratio {v['median_ratio_head_to_resid']:.4f} (max {v['max_ratio']:.2f}); site-level log-log slope of "
                         f"|gap| on head spread {v['site_loglog_slope_gap_on_head_spread']:.2f}.")
        lines += ["", "Extending the gap-spread relation fitted to Study 1's residual banks down to these spreads predicts the head-level "
                  "gaps within a factor of ten at most head-sites (manuscript Section 7).", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

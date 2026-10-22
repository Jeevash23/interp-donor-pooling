"""Write RESULTS.md from the saved records only (results/), after verify.py.

    python -B write_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
KIND_NAMES = {"resid": "Residual stream", "mlp": "MLP output", "attn": "Attention-layer output", "head": "Single head"}
SETTING_NAMES = {"sva-end": "Agreement, final token (panel A)", "ioi-end": "IOI, final token (B1)", "ioi-s2": "IOI, repeated subject (B2)"}
PANEL_NAMES = {"sva-qwen3-1.7b": "Agreement (C1)", "ioi-qwen3-1.7b": "IOI (C2)"}


def read(p):
    return json.loads(Path(p).read_text())


def f(x, d=4):
    return "–" if x is None else f"{x:.{d}f}"


def e(x):
    return "–" if x is None else f"{x:.1e}"


def iv(ci, d=0):
    return f"[{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def token_rule(maps, positions):
    """Secondary: the token rule as defined (ADJ8 at the adjective, NOUN8 at the final token, MEAN64 elsewhere)."""
    ind = np.array(maps["individual"])
    rule = np.array(maps["mean64"]).copy()
    rule[..., 0] = np.array(maps["adj8"])[..., 0]
    rule[..., -1] = np.array(maps["noun8"])[..., -1]
    mis = int(((np.abs(rule) >= 0.05) != (np.abs(ind) >= 0.05)).sum())
    return mis, float(np.abs(rule - ind).mean())


def main():
    out = []
    w = out.append
    stage_log = [json.loads(x) for x in (RES / "stage-log.jsonl").read_text().splitlines()]
    billing = [json.loads(x) for x in (RES / "billing-log.jsonl").read_text().splitlines()]
    cal = read(RES / "calibration/calibration.json")["calibration"]
    cost = read(RES / "cost-decision.json")
    gates = read(RES / "gates/gates.json")["gates"]
    comp = read(RES / "test-completion.json")
    rep = read(RES / "replay-verification.json")
    ver = read(RES / "verification.json")
    dec = read(RES / "analysis/decisions-test.json")
    a4 = read(RES / "analysis/analysis-A-test.json") if (RES / "analysis/analysis-A-test.json").exists() else None
    a3 = read(RES / "analysis/analysis-heads-test.json")["settings"]
    a2 = read(RES / "analysis/analysis-sweeps-test.json")["panels"]
    proto = read(HERE / "data/protocol-freeze.json")["utc"]
    inp = read(HERE / "data/input-freeze.json")["utc"]
    st4 = read(HERE / "data/stage4-freeze.json")["utc"]
    w("# Study 5 results (folder `07-larger-model`)\n")
    w(f"Qwen3-1.7B (revision `70d244cc`), float32, on a Modal A10G. Protocol and predictions hashed at {proto[:19]} UTC; "
      f"code and prompts frozen at {inp[:19]} UTC (three amendments followed: a GPU-memory fix before any development output, "
      f"stages moved inside Modal after the local connection dropped, and a wider stage-4 freeze; `AMENDMENTS.md`); "
      f"calibration, development and the cost decision frozen at {st4[:19]} UTC, before the gates and any test output. "
      f"Test runs finished at {comp['finished_utc'][:19]} UTC. Every outcome below follows `PROTOCOL.md`. "
      "This file is written by `write_results.py` from the saved records in `results/`.\n")
    w("## Checks\n")
    w(f"- Calibration (reference engine, calibration prompts): passed; largest logit error {e(cal['maximum_logit_error'])}, "
      f"largest probability error {e(cal['maximum_probability_error'])}, exact write locality: {cal['exact_write_locality']}. "
      f"Study engine against reference engine: largest difference {e(cal['maximum_engine_difference'])} logits over "
      f"{len(cal['engine_agreement'])} comparisons (tolerance 1e-3).")
    w(f"- Built-in checks in every test row (identity patches, cached passive run, write locality): largest error "
      f"{e(ver['largest_built_in_check_error'])}; failing records: {ver['records_failing']}.")
    w(f"- Replays: fresh-container study engine and reference engine, largest margin difference {e(rep['maximum_error_primary'])} "
      f"(passed: {rep['passed']}); on an L4 (secondary) {e(rep['maximum_error_l4'])}.")
    w(f"- Independent verification (`verify.py`): passed: {ver['passed']}. Frozen files changed: "
      f"{sum(len(v) for v in ver['freezes_changed'].values())}. Test prompts overlapping calibration or development: "
      f"{ver['separation']['overlap_with_calibration_or_development']}; overlapping earlier studies: "
      f"{ver['separation']['overlap_with_earlier_studies']} (of {ver['separation']['prior_strings']:,} earlier strings).")
    w(f"- Cost rule: development projected ${cost['projections'][0]['remaining_usd']:.2f} for the rest of the study "
      f"(rule ${cost['rule_usd']:.0f}); {cost['agreement_groups']} agreement groups and {cost['ioi_prompts']} IOI prompts were used.\n")
    if not rep["passed"]:
        noise = read(RES / "posthoc/replay-noise.json")
        over = {k: v for k, v in rep["errors"].items() if v > rep["tolerance"]}
        w("### The replay check failed (protocol stage 7)\n")
        w("The protocol requires every primary replay to match the test outputs within 0.001 logits, and a failed check stops "
          "the study for the author's review. Four replays exceeded it, all in the IOI panels: "
          + ", ".join(f"{k} {v:.2e}" for k, v in sorted(over.items(), key=lambda x: -x[1])) + " (the L4 replay is secondary). "
          "Every agreement replay was within 1e-4; the study engine's fresh-container agreement replays were exact. "
          "The IOI computations therefore carry float32 differences of about 1e-3 logits between containers, consistent with the "
          f"largest study-against-reference difference in calibration ({e(cal['maximum_engine_difference'])}, also in the IOI sweep).\n")
        w(f"Post hoc (`posthoc/replay_noise.py`, written after the failure): dividing each panel's largest replay difference by its "
          f"denominator bounds how far any normalised effect could move if every row carried that much noise. The bound is at most "
          f"{max(v['effect_bound'] for v in noise['panels'].values()):.1e}; no IND or MEAN effect lies that close to τ (closest "
          f"{min(v['closest_distance_to_tau'] for v in noise['panels'].values()):.1e}, in C2). So noise of the replayed size cannot "
          f"change any flag or primary count (any flag could change: {noise['any_flag_could_change']}). The author accepted the results "
          "with the replay failure and this bound disclosed (`NOTES-AFTER-TEST.md`), and the manuscript's sixth revision reports them.\n")
        w("Order of stages: the analysis (stage 8) was started while the replays (stage 7) were still running, so the outcomes "
          "below were seen before the replay result. The analysis is frozen code and no choice depended on it.\n")
    w("## Competence gates\n")
    ga, gi = gates["agreement"], gates["ioi"]
    w(f"- Agreement: clean recipients {100 * ga['clean_accuracy']:.1f}%, corrupted recipients {100 * ga['corrupted_accuracy']:.1f}% "
      f"(both required at least 90%): **{'passed' if ga['passed'] else 'not passed'}**.")
    w(f"- IOI: IO above S on {100 * gi['io_over_s_accuracy']:.1f}% of prompts (required 90%): **{'passed' if gi['passed'] else 'not passed'}**.\n")
    if a4 is not None:
        w("## R1: bias follows component size (panel A)\n")
        w("Sites are site-directions at the agreement final token. Misclassified sites at τ = 0.05 against the average of 64 individual "
          "patches (IND); mean absolute error in normalised effect, in units of 1e-3, in brackets. S8: mean over 1,000 draws.\n")
        w("| Kind | Sites | Flagged by IND | MEAN (98.75% interval) | TOK | MIX8 | S8 | Mean abs. bias | Median spread |")
        w("|---|---:|---:|---|---|---|---|---:|---:|")
        for k in ["resid", "mlp", "attn", "head"]:
            r = a4["kinds"][k]
            m = r["methods"]
            p1 = r["p1_mean_misclassified"]
            w(f"| {KIND_NAMES[k]} | {r['sites']} | {m['MEAN']['tau0.05']['flagged_ind']} | {p1['estimate']} {iv(p1['interval'])} "
              f"({1e3 * m['MEAN']['mae']:.2f}) | {m['TOK']['tau0.05']['misclassified']} ({1e3 * m['TOK']['mae']:.2f}) | "
              f"{m['MIX8']['tau0.05']['misclassified']} ({1e3 * m['MIX8']['mae']:.2f}) | {m['S8']['tau0.05']['misclassified_mean']:.2f} "
              f"({1e3 * m['S8']['mae_mean']:.2f}) | {r['mean_abs_bias']:.4f} | {r['spread']['median']:,.1f} |")
        r1 = dec["R1"]
        w("")
        for pair in ["resid-mlp", "mlp-attn", "attn-head"]:
            p = r1["p3_ordering"][pair]
            w(f"- P3, {pair}: difference {p['difference']:.4f}, 95% interval {iv(p['ci95'], 4)}.")
        w(f"- P3 (all three intervals above zero): **{'supported' if r1['p3_supported'] else 'not supported'}**; "
          f"full order in {100 * r1['p3_ordering']['bootstrap_share_full_order']:.0f}% of bootstrap draws.")
        w(f"- P4: kinds by spread {r1['kinds_by_spread']}, by bias {r1['kinds_by_bias']}: **{'supported' if r1['p4_supported'] else 'not supported'}**.")
        w(f"- **R1 {'replicates' if r1['replicates'] else 'does not replicate'}.**\n")
        w("Secondary (Study 4's P2): " + "; ".join(f"{KIND_NAMES[k]} {a4['kinds'][k]['p2_tok_vs_s8']['decision']}" for k in ["resid", "mlp", "attn", "head"]) + ".\n")
    if a3:
        w("## R2: single-head flags\n")
        w("Misclassified head-sites under MEAN at τ = 0.05 against IND, with 98.33% intervals. Replicates if the lower limit is 0 and the "
          "upper limit is at most 2% of head-sites; contradicts Study 3 if the lower limit is at least 1.\n")
        w("| Setting | Head-sites | Flagged by IND | MEAN (98.33% interval) | 2% bound | TOK | S8 | Verdict |")
        w("|---|---:|---:|---|---:|---|---|---|")
        a3 = {s: a3[s] for s in SETTING_NAMES if s in a3}
        for s, r in a3.items():
            d = dec["R2"][s]
            m = r["methods"]
            w(f"| {SETTING_NAMES[s]} | {r['head_sites']} | {m['MEAN']['tau0.05']['flagged_ind']} | {d['estimate']} {iv(d['ci'])} | "
              f"{d['upper_bound_2pct']} | {m['TOK']['tau0.05']['misclassified']} | {m['S8']['tau0.05']['misclassified_mean']:.2f} | "
              f"**{d['verdict']}** |")
        w("")
        for s, r in a3.items():
            ind, mean, tok = (np.array(r["maps"][n]) for n in ("IND", "MEAN", "TOK"))
            for d_, l_, h_ in np.argwhere((np.abs(ind) >= 0.05) != (np.abs(mean) >= 0.05)):
                w(f"- Changed flag, {SETTING_NAMES[s]}: layer {l_}, head {h_}: IND {ind[d_, l_, h_]:.4f}, MEAN {mean[d_, l_, h_]:.4f}, "
                  f"TOK {tok[d_, l_, h_]:.4f}.")
        for s, r in a3.items():
            t = r["tok_vs_s8"]
            w(f"- {SETTING_NAMES[s]}: TOK error {1e4 * t['tok_mae']:.2f} against S8 {1e4 * t['s8_mae_mean']:.2f} (2.5th percentile "
              f"{1e4 * t['s8_mae_2.5th_percentile']:.2f}; units of 1e-4); TOK better by Study 3's H2 rule: {t['tok_better']}; "
              f"TOK identical to MEAN: {t['tok_identical_to_mean']}; TOK evaluations per head-site {r['evaluations_per_recipient_site']['TOK']:.1f}.")
        w("")
    if a2:
        w("## R3: residual sweeps\n")
        w("Sites whose flag under MEAN64 differs from IND at τ = 0.05, with 97.5% intervals. Replicates if the lower limit is at least 1.\n")
        w("| Panel | Sites | Misclassified (97.5% interval) | False positives | False negatives | Rate | MIX8 | NOUN8 | Verdict |")
        w("|---|---:|---|---:|---:|---:|---:|---:|---|")
        a2 = {p: a2[p] for p in PANEL_NAMES if p in a2}
        for p, r in a2.items():
            d = dec["R3"][p]
            fl = r["flags"]
            noun = fl.get("noun8/tau0.05", {}).get("misclassified_sites", {}).get("estimate", "–")
            w(f"| {PANEL_NAMES[p]} | {d['sites']} | {d['estimate']} {iv(d['ci'])} | {fl['mean64/tau0.05']['false_positive_sites']['estimate']} | "
              f"{fl['mean64/tau0.05']['false_negative_sites']['estimate']} | {100 * d['rate']:.1f}% | "
              f"{fl['mix8/tau0.05']['misclassified_sites']['estimate']} | {noun} | **{'replicates' if d['replicates'] else 'does not replicate'}** |")
        w("")
        w("Earlier rates (Study 2): GPT-2 Small agreement 5.0%, Qwen3-0.6B agreement 9.6%, GPT-2 Small IOI 3.6%.\n")
        if "sva-qwen3-1.7b" in a2:
            r = a2["sva-qwen3-1.7b"]
            mis, mae = token_rule(r["maps"], r["positions"])
            w(f"- Token rule as defined (agreement sweep; secondary): {mis} misclassified site{'s' if mis != 1 else ''}, "
              f"mean absolute error {mae:.4f}.")
            w(f"- Final-token centroid gap by bank partition (logits): " + ", ".join(f"{k} {v:.3f}" for k, v in r["end_gap_by_partition"].items()) + ".")
        for p, r in a2.items():
            w(f"- {PANEL_NAMES[p]}: bank-level spread slope {r['spread_law']['loglog_slope']:.2f} (within-cell R² {r['spread_law']['within_cell_r2']:.2f}).")
        w("")
    w("## Predictions against outcomes\n")
    w("The designer's predictions (`PREDICTIONS.md`, hashed with the protocol) are not decision rules.\n")
    w("| Prediction | Confidence | Outcome |")
    w("|---|---|---|")
    yn = lambda b: "held" if b else "did not hold"                                          # noqa: E731
    w(f"| Agreement gate passes | about 55% | {yn(ga['passed'])} ({100 * ga['clean_accuracy']:.1f}% / {100 * ga['corrupted_accuracy']:.1f}%) |")
    w(f"| IOI gate passes | about 80% | {yn(gi['passed'])} ({100 * gi['io_over_s_accuracy']:.1f}%) |")
    if a4 is not None:
        w(f"| P3: resid > mlp > attn > head, all intervals above zero | about 75% | {yn(dec['R1']['p3_supported'])} |")
        w(f"| P4: spread in the same order | about 75% | {yn(dec['R1']['p4_supported'])} |")
        w(f"| Mean absolute residual bias about 0.04–0.08 | – | {a4['kinds']['resid']['mean_abs_bias']:.4f} |")
        tok_all = all(a4["kinds"][k]["p2_tok_vs_s8"]["decision"] == "TOK better" for k in ["resid", "mlp", "attn", "head"])
        w(f"| Token grouping beats eight sampled donors for all four kinds | about 75% | {yn(tok_all)} |")
    conf = {"sva-end": "about 85%", "ioi-end": "about 80%", "ioi-s2": "about 85%"}
    for s in a3:
        w(f"| R2 replicates: {SETTING_NAMES[s]} | {conf[s]} | {dec['R2'][s]['verdict']} |")
    if len(a3) == 3:
        w(f"| R2 replicates in all three settings | about 65% | {yn(all(dec['R2'][s]['verdict'] == 'replicates' for s in a3))} |")
    if "sva-qwen3-1.7b" in dec.get("R3", {}):
        d = dec["R3"]["sva-qwen3-1.7b"]
        w(f"| R3 replicates in the agreement sweep, with 10–35 misclassified sites | about 85% | {yn(d['replicates'])}; {d['estimate']} sites |")
    if "ioi-qwen3-1.7b" in dec.get("R3", {}):
        w(f"| R3 replicates in the IOI sweep | about 60% | {yn(dec['R3']['ioi-qwen3-1.7b']['replicates'])} |")
    if "sva-qwen3-1.7b" in a2:
        mis, _ = token_rule(a2["sva-qwen3-1.7b"]["maps"], a2["sva-qwen3-1.7b"]["positions"])
        w(f"| Token rule as defined misclassifies no agreement-sweep site | about 70% | {yn(mis == 0)} ({mis}) |")
    w(f"| Total spend $3.50–6.50 | – | ${billing[-1]['study5_usd']:.2f} (all ai03-study5 apps, including the smoke test) |\n")
    w("## Cost and data\n")
    w(f"- Modal billing report, all `ai03-study5` apps (including the smoke test): ${billing[-1]['study5_usd']:.2f} at "
      f"{billing[-1]['utc'][:16]} UTC; stop at $10.")
    starts = [s for s in stage_log if "started_utc" in s]
    w("- Stage starts (spend before each, by the billing report; some development starts were interrupted, see AMENDMENTS.md): "
      + "; ".join(f"{s['stage']} {s['started_utc'][11:16]} UTC (${s['spend_usd']:.2f})" for s in starts) + ".")
    final = read(RES / "billing-final.json")
    w(f"- Final billing report: ${final['total_usd']:.2f} for Study 5, of which ${final['total_by_app']['ai03-study5']:.2f} for the "
      f"study run itself (A10G ${final['by_app']['ai03-study5']['A10G']:.2f}, L4 ${final['by_app']['ai03-study5']['L4']:.2f}, "
      f"CPU ${final['by_app']['ai03-study5']['CPU']:.2f}, memory ${final['by_app']['ai03-study5']['Memory']:.2f}); $0.00 billed after credits.")
    size = sum(p.stat().st_size for p in RES.rglob("*") if p.is_file())
    w(f"- Data: the records pulled back to `results/` total {size / 1e6:.1f} MB; uploads were code and compact prompt rows. "
      "Raw arrays stay on the Modal Volume `ai03-study5`; every array's SHA-256 is in its run record under `results/`.\n")
    (HERE / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()

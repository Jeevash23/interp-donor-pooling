"""Write paper/generated/numbers.tex: every study number quoted in the manuscript.

Each macro is computed from a saved, verified result file. `--check` regenerates the
macros in memory and fails if the committed file differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
S2 = PROJECT / "submission-study"   # release: the original reached P2 through a link, studies/02-six-model-audit
S3 = PROJECT / "studies/03-grouping-and-circuit"
S4 = PROJECT / "studies/04-reference-and-sweeps"


def read(p):
    return json.loads(Path(p).read_text())


def f(x, d=3):
    """Fixed decimals; a typeset minus sign for negatives and no negative zero."""
    text = f"{x:.{d}f}"
    if float(text) == 0:
        return f"{0:.{d}f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def pct(x, d=1):
    return f(100 * x, d)


def ci(c, d=3, scale=1.0):
    return f"[{f(c[0] * scale, d)}, {f(c[1] * scale, d)}]"


def numbers():
    n = {}
    # ---------------- Study 3: balanced grouping (frozen primary analysis)
    for key, model in [("Gpt", "gpt2-small"), ("Qwen", "qwen3-0.6b")]:
        a = read(S3 / f"token/results/analysis-test/{model}.json")
        dc = a["secondary_dose_curve"]
        n[f"GroupK{key}"] = f(dc["1"]["margin_gap_absolute"]["mean"])
        n[f"GroupKeight{key}"] = f(dc["8"]["margin_gap_absolute"]["mean"])
        n[f"GroupDiff{key}"] = f(a["primary"]["mean"])
        n[f"GroupDiffCI{key}"] = ci(a["primary"]["ci"])
        n[f"GroupFold{key}"] = str(round(dc["8"]["margin_gap_absolute"]["mean"] / dc["1"]["margin_gap_absolute"]["mean"]))
        n[f"RadiusKone{key}"] = f"{dc['1']['mean_squared_radius']['mean']:,.0f}"
        n[f"RadiusKeight{key}"] = f"{dc['8']['mean_squared_radius']['mean']:,.0f}"
        for kk, kname in [("2", "two"), ("4", "four")]:
            n[f"GroupK{kname}{key}"] = f(dc[kk]["margin_gap_absolute"]["mean"])
            n[f"RadiusK{kname}{key}"] = f"{dc[kk]['mean_squared_radius']['mean']:,.0f}"
        pg = [dc[kk]["probability_gap_absolute"]["mean"] for kk in ["1", "2", "4", "8"]]
        n[f"ProbGapKone{key}"] = f(pg[0], 5)
        n[f"ProbGapMax{key}"] = f(max(pg), 4)
        n[f"ProbGapMaxK{key}"] = ["1", "2", "4", "8"][pg.index(max(pg))]
        comp = a["competence"]
        n[f"PairAcc{key}"] = pct(comp["recipient_clean_pair_accuracy"]["mean"])
        n[f"TopOneCount{key}"] = str(round(comp["recipient_clean_top1_accuracy"]["mean"] * comp["recipient_clean_top1_accuracy"]["n_recipient_groups"]))
        n[f"TopOneGroups{key}"] = str(comp["recipient_clean_top1_accuracy"]["n_recipient_groups"])
        n[f"CleanMargin{key}"] = f(comp["recipient_clean_margin"]["mean"])
        mant, exp = f"{a['numerical']['anchor_centroid_patch_across_doses_max']:.1e}".split("e")
        n[f"AnchorMax{key}"] = f"${mant}\\times 10^{{{int(exp)}}}$"
    if n["TopOneCountGpt"] == n["TopOneCountQwen"]:
        n["TopOneSentence"] = f"each rank the correct verb first in only {n['TopOneCountGpt']} of {n['TopOneGroupsGpt']} clean prompts"
    else:
        n["TopOneSentence"] = (f"rank the correct verb first in only {n['TopOneCountGpt']} and {n['TopOneCountQwen']} "
                               f"of {n['TopOneGroupsGpt']} clean prompts")
    # ---------------- Retrospective analyses of Study 3 (post hoc audit)
    r = read(S4 / "results/retrospective-study3.json")
    for key, model in [("Gpt", "gpt2-small"), ("Qwen", "qwen3-0.6b")]:
        g = r["grouping"][model]
        n[f"Slope{key}"] = f(g["loglog_slope_gap_on_spread"], 2)
        n[f"SlopeCI{key}"] = ci(g["loglog_slope_ci95"], 2)
        n[f"RsqSpread{key}"] = f(g["r2_spread"], 2)
        n[f"RsqK{key}"] = f(g["r2_k_labels"], 2)
        n[f"RsqBoth{key}"] = f(g["r2_spread_and_k"], 3)
        n[f"RsqSpreadThree{key}"] = f(g["r2_spread"], 3)
        n[f"MaxBiasRatio{key}"] = f(g["max_bias_over_effect_k8"], 1)
        n[f"MaxBiasGap{key}"] = pct(g["max_bias_over_clean_minus_corrupt_k8"])
        n[f"CleanMinusCorrupt{key}"] = f(g["clean_minus_corrupt"], 2)
        rows = {(x["direction"], x["depth_index"], x["k"]): x for x in g["bias_by_site"]}
        for d in ["restore", "destroy"]:
            for depth in [1, 2, 3]:
                tag = f"{key}{d.capitalize()}{'One Two Three'.split()[depth - 1]}"
                n[f"Effect{tag}"] = f(rows[(d, depth, 8)]["individual_effect"])
                n[f"Effect{tag}Abs"] = f(abs(rows[(d, depth, 8)]["individual_effect"]), 2)
                n[f"BiasKeight{tag}"] = f(rows[(d, depth, 8)]["centroid_bias"])
                n[f"BiasKone{tag}"] = f(rows[(d, depth, 1)]["centroid_bias"])
                n[f"Ratio{tag}"] = f(rows[(d, depth, 8)]["bias_over_effect"], 2)
    eff, bias = [], []
    for model in ["gpt2-small", "qwen3-0.6b"]:
        for x in r["grouping"][model]["bias_by_site"]:
            if x["k"] == 8 and x["depth_index"] in (1, 2):
                eff.append(abs(x["individual_effect"]))
                bias.append(abs(x["centroid_bias"]))
    n["WeakEffectLow"], n["WeakEffectHigh"] = f(min(eff), 2), f(max(eff), 2)
    n["WeakBiasLow"], n["WeakBiasHigh"] = f(min(bias), 2), f(max(bias), 2)
    c = r["circuit"]
    n["SThreeScreened"] = f"{read(S3 / 'circuit/results/screen-selection.json')['screened_n']:,}"
    n["SThreeCircuitErrMean"] = pct(c["reference/full_pair_error"]["mean_agreement"])
    n["SThreeCircuitErrRes"] = pct(c["reference/full_pair_error"]["resample_agreement"])
    n["SThreeCircuitOkMean"] = pct(c["reference/full_pair_success"]["mean_agreement"])
    n["SThreeCircuitOkRes"] = pct(c["reference/full_pair_success"]["resample_agreement"])
    n["SThreeEmptyErrMean"] = pct(c["empty_attention/full_pair_error"]["mean_agreement"])
    n["SThreeEmptyErrRes"] = pct(c["empty_attention/full_pair_error"]["resample_agreement"])
    n["SThreeEmptyOkMean"] = pct(c["empty_attention/full_pair_success"]["mean_agreement"])
    n["SThreeEmptyOkRes"] = pct(c["empty_attention/full_pair_success"]["resample_agreement"])
    sc = c["screen"]
    n["SThreeScreenErrors"] = f"{sc['errors']}"
    n["SThreeTemplateThreeErrors"] = f"{sc['by_template']['3']['errors']}"
    n["SThreeTemplateThreeRate"] = pct(sc["by_template"]["3"]["error_rate"], 2)
    n["SThreeTemplateOneRate"] = pct(sc["by_template"]["1"]["error_rate"], 2)
    n["SThreeNameLowRate"] = pct(sc["error_rate_lowest_decile_name_score"])
    n["SThreeNameHighRate"] = pct(sc["error_rate_highest_decile_name_score"])
    a = read(S3 / "circuit/results/analysis.json")
    ref = a["circuits"]["reference"]
    n["SThreeFaithMean"] = f(ref["mean_empty_adjusted_faithfulness"]["estimate"])
    n["SThreeFaithRes"] = f(ref["resample_empty_adjusted_faithfulness"]["estimate"])
    st = ref["strata"]
    ok, err = st["full_pair_success"]["paired_pair_agreement_difference"], st["full_pair_error"]["paired_pair_agreement_difference"]
    n["SThreeOkDiff"] = pct(ok["mean"])
    n["SThreeOkDiffCI"] = ci(ok["ci95"], 1, 100)
    n["SThreeErrDiff"] = pct(err["mean"])
    n["SThreeTopOneOkMean"] = pct(st["full_pair_success"]["mean_exact_answer_agreement"]["mean"])
    n["SThreeTopOneOkRes"] = pct(st["full_pair_success"]["expected_resample_exact_answer_agreement"]["mean"])
    n["SThreeErrDiffCI"] = ci(err["ci95"], 1, 100)
    # ---------------- Study 2 (September six-model audit), supporting values only
    import math
    models = ["gpt2-small", "gpt2-medium", "gpt2-large", "pythia-410m", "tinyllama-1.1b", "qwen3-0.6b"]
    half = [read(S2 / f"results/analysis/{m}.json")["radius"]["0.5"]["ratio_to_full"]["ratio"] for m in models]
    n["STwoRadiusLow"], n["STwoRadiusHigh"] = pct(min(half)), pct(max(half))
    slopes = [math.log(h) / math.log(0.25) for h in half]
    n["STwoSlopeLow"], n["STwoSlopeHigh"] = f(min(slopes), 2), f(max(slopes), 2)
    n["STwoInterventions"] = f"{read(S2 / 'results/analysis/paper-summary.json')['core_interventions']:,}"
    prev = read(S3 / "results/previous-study-diagnostics.json")["models"]
    shares = [v["sva_end_share_of_old_primary"] for v in prev.values()]
    n["STwoShareLow"], n["STwoShareHigh"] = pct(min(shares), 0), pct(max(shares), 0)
    # ---------------- Study 1 (12 September pilot)
    S1 = PROJECT / "studies/01-pilot"
    for key, model in [("Small", "gpt2-small"), ("Medium", "gpt2-medium")]:
        prim = read(S1 / f"results/analysis/{model}.json")["primary"]
        n[f"SOneGap{key}"] = f(prim["gap_ld_abs"]["mean"])
        n[f"SOneHalfReduction{key}"] = pct(1 - prim["half_full_ratio_of_mean_absolute_gaps"])
    audit = read(S4 / "data/generation-audit.json")
    n["NumNames"], n["NumPlaces"], n["NumObjects"] = (str(len(audit[k])) for k in ["ioi_names", "ioi_places", "ioi_objects"])
    n["SvaNounPool"], n["SvaAdjPool"] = str(len(audit["sva_nouns"])), str(len(audit["sva_adjectives"]))
    n["NumTemplatesW"] = str(sum(k.startswith("W") for k in audit["ioi_templates"]))
    n["NumTemplatesO"] = str(sum(k.startswith("O") for k in audit["ioi_templates"]))
    n["ScreenSize"] = f"{audit['splits']['ioi-screen/O']['rows'] + audit['splits']['ioi-screen/W']['rows']:,}"
    n.update(study4())
    n.update(posthoc())
    n.update(study5())
    n.update(revision2())
    n.update(study6())
    n.update(revision5(n))
    n.update(study7())
    return n


def posthoc():
    """Post hoc analyses of saved Study 2 outputs and the post hoc random-circuit control."""
    n = {}
    ph = S4 / "results/posthoc/sweep-posthoc.json"
    if ph.exists():
        d = read(ph)
        for panel, key in PANEL.items():
            p = d["panels"][panel]
            n[f"Acc{key}"] = pct(p["clean_accuracy"])
            n[f"CorrectGroups{key}"] = str(p["groups_correct"])
            n[f"MisRestricted{key}"] = str(p["misclassified_correct_groups_only"])
            eq = p["equal_budget"]
            for meth, mk in [("mean64", "Mean"), ("mix8", "Mix"), ("noun8", "Noun"), ("adj8", "Adj")]:
                if meth in eq:
                    n[f"Eq{mk}Mis{key}"] = str(eq[meth]["misclassified"])
                    n[f"Eq{mk}Mae{key}"] = f(eq[meth]["mae"], 4)
            s8 = eq["sample8"]
            n[f"EqSampleMis{key}"] = f(s8["misclassified_mean"], 2)
            n[f"EqSampleMisRange{key}"] = "{:.0f}--{:.0f}".format(*s8["misclassified_range95"])
            n[f"EqSampleMae{key}"] = f(s8["mae_mean"], 4)
            bt = p["misclassified_by_tau"]
            for t, tk in [("0.02", "Two"), ("0.1", "Ten"), ("0.2", "Twenty")]:
                n[f"Tau{tk}{key}"] = str(bt[t])
            taus = sorted(bt, key=float)
            upto = taus[0]
            for t in taus:
                if bt[t] == 0:
                    break
                upto = t
            n[f"TauNonzeroUpTo{key}"] = f"{float(upto):.2f}"
            n[f"EffPctMis{key}"] = pct(p["spread_percentile_within_layer"]["disputed_effect_percentile_median"], 0)
            n[f"TopK{key}"] = str(p["topk"]["k"])
            n[f"TopKOverlap{key}"] = str(p["topk"]["overlap"])
            n[f"TopKShare{key}"] = pct(p["topk"]["share"], 0)
            ds = p["distance_summary"]
            n[f"DistInd{key}"] = f(ds["individual_median"], 3)
            n[f"DistMean{key}"] = f(ds["mean_median"], 3)
            c = p["cancellation"]
            n[f"CancAbs{key}"] = f(c["disputed"]["mean_abs_per_donor_effect"], 3)
            n[f"CancAvg{key}"] = f(c["disputed"]["abs_signed_average"], 3)
            n[f"CancOpp{key}"] = pct(c["disputed"]["share_opposite_sign"], 0)
            n[f"WeakAbs{key}"] = f(c["other_weak"]["mean_abs_per_donor_effect"], 3)
            n[f"WeakAvg{key}"] = f(c["other_weak"]["abs_signed_average"], 3)
            n[f"WeakOpp{key}"] = pct(c["other_weak"]["share_opposite_sign"], 0)
            n[f"CancRatio{key}"] = f(c["disputed"]["mean_abs_per_donor_effect"] / c["disputed"]["abs_signed_average"], 0)
            sp = p["spread_percentile_within_layer"]
            n[f"SpreadPctMis{key}"] = pct(sp["disputed_median"], 0)
            n[f"SpreadPctOther{key}"] = pct(sp["other_live_median"], 0)
        n["IoiMisAtEnd"] = str(sum(1 for x in d["panels"]["ioi-gpt2-small"]["disputed_sites"] if x["position"] == "END"))
        g = d["ioi_token_groups"]
        for lab, lk in [("N1", "NOne"), ("N2", "NTwo"), ("S2", "STwo"), ("PLACE", "Place"), ("OBJ", "Obj"), ("END", "End")]:
            n[f"TokGroups{lk}"] = f(g[lab]["groups_mean"], 0)
            n[f"TokSize{lk}"] = f(g[lab]["mean_group_size"], 1)
        n["TokSharedPositions"] = str(sum(1 for v in g.values() if v["groups_max"] == 1))
    rc = S4 / "results/posthoc/random-circuits.json"
    if rc.exists():
        r = read(rc)
        n["RandCircuits"] = str(r["circuits"])
        for label, lk in [("pair_error", "Err"), ("pair_correct", "Ok")]:
            for c, ck in CELL.items():
                for pool, pk in [("mean", "Mean"), ("resample", "Res")]:
                    t = r["tables"][label][f"{c}/{pool}"]
                    n[f"Rand{lk}{ck}{pk}"] = pct(t["mean_over_circuits"])
                    n[f"Rand{lk}{ck}{pk}Range"] = f"{pct(t['min'])}--{pct(t['max'])}"
            for pool, pk in [("mean", "Mean"), ("resample", "Res")]:
                vals = [r["tables"][label][f"{c}/{pool}"]["mean_over_circuits"] for c in CELL]
                n[f"Rand{lk}{pk}AllLow"], n[f"Rand{lk}{pk}AllHigh"] = pct(min(vals)), pct(max(vals))
                lows = [r["tables"][label][f"{c}/{pool}"]["min"] for c in CELL]
                highs = [r["tables"][label][f"{c}/{pool}"]["max"] for c in CELL]
                n[f"Rand{lk}{pk}AnyLow"], n[f"Rand{lk}{pk}AnyHigh"] = pct(min(lows)), pct(max(highs))
    return n


S5 = PROJECT / "studies/05-head-confirmation"
S6 = PROJECT / "studies/06-component-confirmation"


COMP = {"resid": "Res", "mlp": "Mlp", "attn": "Attn", "head": "Head"}


def study6():
    """Study 4 (folder 06-component-confirmation): component-level test at the final token."""
    n = {}
    audit = S6 / "data/generation-audit.json"
    if audit.exists():
        n["CompPriorStrings"] = f"{read(audit)['prior_unique_strings']:,}"
    cal = S6 / "results/calibration.json"
    if cal.exists():
        c = read(cal)
        mant, exp = f"{c['maximum_logit_error']:.1e}".split("e")
        n["CompCalMax"] = f"${mant}\\times 10^{{{int(exp)}}}$"
    rep = S6 / "results/replay-verification.json"
    if rep.exists():
        r = read(rep)
        mx = r["maximum_error"]
        mant, exp = f"{mx:.1e}".split("e")
        n["CompReplayText"] = "reproduced exactly" if mx == 0 else f"reproduced to within ${mant}\\times 10^{{{int(exp)}}}$ logits"
    a6 = S6 / "results/analysis-test.json"
    if a6.exists():
        a = read(a6)
        n["CompGroups"] = str(a["groups"])
        n["CompSites"] = str(sum(v["sites"] for v in a["kinds"].values()))
        for kind, kk in COMP.items():
            v = a["kinds"][kind]
            m = v["methods"]
            p1 = v["p1_mean_misclassified"]
            show = lambda x: f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"
            n[f"CompSites{kk}"] = str(v["sites"])
            n[f"CompFlagInd{kk}"] = str(m["MEAN"]["tau0.05"]["flagged_ind"])
            n[f"CompMis{kk}"] = str(p1["estimate"])
            n[f"CompMisCI{kk}"] = "[{}, {}]".format(*(show(x) for x in p1["interval"]))
            n[f"CompMisVerdict{kk}"] = "supported" if p1["supported"] else "not supported"
            n[f"CompFp{kk}"], n[f"CompFn{kk}"] = str(m["MEAN"]["tau0.05"]["false_positive"]), str(m["MEAN"]["tau0.05"]["false_negative"])
            for meth, mk in [("TOK", "Tok"), ("MIX8", "Mix")]:
                n[f"Comp{mk}Mis{kk}"] = str(m[meth]["tau0.05"]["misclassified"])
                n[f"Comp{mk}Err{kk}"] = f(1e3 * m[meth]["mae"], 2)
            n[f"CompMeanErr{kk}"] = f(1e3 * m["MEAN"]["mae"], 2)
            n[f"CompSampleMis{kk}"] = f(m["S8"]["tau0.05"]["misclassified_mean"], 2)
            n[f"CompSampleMisRange{kk}"] = "{:.0f}--{:.0f}".format(*m["S8"]["tau0.05"]["misclassified_range95"])
            n[f"CompSampleErr{kk}"] = f(1e3 * m["S8"]["mae_mean"], 2)
            n[f"CompSampleErrLow{kk}"] = f(1e3 * m["S8"]["mae_range95"][0], 2)
            n[f"CompSampleErrHigh{kk}"] = f(1e3 * m["S8"]["mae_range95"][1], 2)
            n[f"CompTokDecision{kk}"] = v["p2_tok_vs_s8"]["decision"].replace("TOK", "token grouping").replace("S8", "sampled donors")
            n[f"CompTokVerdict{kk}"] = "supported" if v["p2_tok_vs_s8"]["supported"] else "not supported"
            n[f"CompBias{kk}"] = f(v["mean_abs_bias"], 4)
            n[f"CompBiasMilli{kk}"] = f(1e3 * v["mean_abs_bias"], 2)
            n[f"CompMaxErr{kk}"] = f(m["MEAN"]["max_abs_error"], 3)
            n[f"CompSpread{kk}"] = f"{v['spread']['median']:,.1f}" if v["spread"]["median"] < 100 else f"{v['spread']['median']:,.0f}"
            bn = v["bias_vs_noise"]
            n[f"CompBnNear{kk}"] = str(bn["near_threshold_sites"])
            if bn["median_bias_near"] is not None:
                n[f"CompBnBias{kk}"] = f(bn["median_bias_near"], 4)
                n[f"CompBnSd{kk}"] = f(bn["median_sd8_near"], 4)
                n[f"CompBnShare{kk}"] = pct(bn["share_bias_above_sd8_near"], 0)
            if v["spread"]["loglog_slope_bias_on_spread"] is not None:
                n[f"CompSlope{kk}"] = f(v["spread"]["loglog_slope_bias_on_spread"], 2)
        import numpy as np

        def blocks(ls):
            ls = sorted(ls)
            if len(ls) > 2 and ls == list(range(ls[0], ls[-1] + 1)):
                return f"{ls[0]}--{ls[-1]}"
            return " and ".join(str(x) for x in ls) if len(ls) <= 2 else ", ".join(str(x) for x in ls[:-1]) + f" and {ls[-1]}"
        for kind, kk in COMP.items():
            if kind == "head":
                continue
            v = a["kinds"][kind]
            ind, mean = np.array(v["maps"]["IND"]), np.array(v["maps"]["MEAN"])
            dis = (np.abs(mean) >= 0.05) != (np.abs(ind) >= 0.05)
            for d, dk in [(0, "Restore"), (1, "Destroy")]:
                ls = [int(x) for x in np.where(dis[d])[0]]
                n[f"CompMis{dk}{kk}"] = str(len(ls))
                if ls:
                    n[f"CompMis{dk}Blocks{kk}"] = blocks(ls)
                    n[f"CompMis{dk}IndMax{kk}"] = pct(float(np.abs(ind[d, ls]).max()), 1)
                    n[f"CompMis{dk}MeanLow{kk}"] = pct(float(np.abs(mean[d, ls]).min()), 1)
                    n[f"CompMis{dk}MeanHigh{kk}"] = pct(float(np.abs(mean[d, ls]).max()), 1)
                    if len(ls) == 1:
                        n[f"CompMis{dk}Ind{kk}"] = f(float(ind[d, ls[0]]), 3)
                        n[f"CompMis{dk}Mean{kk}"] = f(float(mean[d, ls[0]]), 3)
        o = a["p3_ordering"]
        for pair, pk in [("resid-mlp", "ResMlp"), ("mlp-attn", "MlpAttn"), ("attn-head", "AttnHead")]:
            n[f"CompDiff{pk}"] = f(o[pair]["difference"], 4)
            n[f"CompDiffCI{pk}"] = ci(o[pair]["ci95"], 4)
        n["CompOrderVerdict"] = "supported" if o["supported"] else "not supported"
        n["CompOrderShare"] = pct(o["bootstrap_share_full_order"], 0)
        p4 = a["p4_spread_order"]
        n["CompSpreadVerdict"] = "supported" if p4["supported"] else "not supported"
    return n


def revision2():
    """Second-revision post hoc analyses: error discrimination, head-level spread, bias against sampling noise."""
    n = {}
    ed = S4 / "results/posthoc/error-discrimination.json"
    if ed.exists():
        r = read(ed)["results"]
        cells_n, cells_npo = ["N8", "N128"], ["NPO8", "NPO128"]
        pools = ["mean", "resample"]

        def rng(unit, metric, cells, d, scale=1.0, sub=None):
            vals = []
            for pool in pools:
                for c in cells:
                    x = r[f"{unit}/{pool}/{c}"][metric]
                    vals.append((x[sub] if sub else x) * scale)
            return f(min(vals), d), f(max(vals), d)

        for unit, uk in [("circuit", "Circ"), ("empty", "Empty")]:
            for cells, ck in [(cells_n, "N"), (cells_npo, "NPO")]:
                n[f"DiscOr{uk}{ck}Low"], n[f"DiscOr{uk}{ck}High"] = rng(unit, "odds_ratio", cells, 1)
                n[f"DiscAuc{uk}{ck}Low"], n[f"DiscAuc{uk}{ck}High"] = rng(unit, "auroc", cells, 2)
                n[f"DiscAucWithin{uk}{ck}Low"], n[f"DiscAucWithin{uk}{ck}High"] = rng(unit, "auroc_within_template", cells, 2)
            n[f"DiscOr{uk}Low"], n[f"DiscOr{uk}High"] = rng(unit, "odds_ratio", cells_n + cells_npo, 1)
            n[f"DiscAuc{uk}Low"], n[f"DiscAuc{uk}High"] = rng(unit, "auroc", cells_n + cells_npo, 2)
        n["DiscOrRandLow"], n["DiscOrRandHigh"] = rng("random", "odds_ratio", cells_n + cells_npo, 1, sub="mean")
        n["DiscAucRandLow"], n["DiscAucRandHigh"] = rng("random", "auroc", cells_n + cells_npo, 2, sub="mean")
        n["DiscAucRandAnyHigh"] = f(max(r[f"random{k}/{p}/{c}"]["auroc"] for k in range(5) for p in pools
                                        for c in cells_n + cells_npo), 2)
        sep = sum(1 for p in pools for c in cells_n + cells_npo
                  if r[f"circuit/{p}/{c}"]["auroc_ci95"][0] > max(r[f"empty/{p}/{c}"]["auroc_ci95"][1],
                                                                   r[f"random/{p}/{c}"]["auroc"]["max"]))
        n["DiscAucSeparated"] = str(sep)
        for unit, uk in [("circuit", "Circ"), ("empty", "Empty")]:
            for pool, pk in [("mean", "Mean"), ("resample", "Res")]:
                for c, ck in CELL.items():
                    x = r[f"{unit}/{pool}/{c}"]
                    n[f"DiscPErr{uk}{ck}{pk}"] = pct(x["p_subject_given_error"], 0)
                    n[f"DiscPCorr{uk}{ck}{pk}"] = pct(x["p_subject_given_correct"], 0)
                    n[f"DiscOr{uk}{ck}{pk}"] = f(x["odds_ratio"], 1)
                    n[f"DiscAuc{uk}{ck}{pk}"] = f(x["auroc"], 2)
        n["DiscCells"] = str(len(pools) * 4)
    hs = S5 / "results/posthoc/head-spread.json"
    if hs.exists():
        h = read(hs)["settings"]
        for key, hk in [("sva-end", "Agr"), ("ioi-end", "IoiEnd"), ("ioi-s2", "IoiSTwo")]:
            v = h[key]
            n[f"HSpreadRatio{hk}"] = f(v["median_ratio_head_to_resid"], 4)
            n[f"HSpreadRatioPct{hk}"] = f(100 * v["median_ratio_head_to_resid"], 2)
            n[f"HSpreadRatioMax{hk}"] = f(v["max_ratio"], 2)
            n[f"HSpreadHead{hk}"] = f(v["median_head_spread"], 2)
            n[f"HSpreadResid{hk}"] = f"{v['median_resid_spread']:,.0f}"
            n[f"HSpreadSlope{hk}"] = f(v["site_loglog_slope_gap_on_head_spread"], 2)
            n[f"HSpreadCorr{hk}"] = f(v["site_log_correlation"], 2)
            n[f"HSpreadInvRatio{hk}"] = f"{1 / v['median_ratio_head_to_resid']:,.0f}"
        # Extrapolate the Study 1 residual-bank relation (site medians, GPT-2 Small) to the head-level spreads.
        # Arrays are [recipient, site, direction, K, rotation, part]; the medians run over recipients and
        # rotations, one point per site, direction and K (24 points). np.take keeps the axis order; the
        # earlier a[:, [1, 3, 5], :, :, :, 0] moved the site axis first, so its medians ran over sites
        # instead of recipients (corrected 9 October 2026).
        import numpy as np
        s1 = np.load(S3 / "token/results/analysis-test/gpt2-small-components.npz")
        gm = np.median(np.abs(np.take(s1["gap"], [1, 3, 5], axis=1)[..., 0]), axis=(0, 4)).ravel()
        sm = np.median(np.take(s1["dispersion"], [1, 3, 5], axis=1)[..., 0], axis=(0, 4)).ravel()
        b1, a1 = np.polyfit(np.log10(sm), np.log10(gm), 1)
        n["SOneSiteSlope"] = f(b1, 2)
        n["SOneSpreadLow"], n["SOneSpreadHigh"] = f"{sm.min():,.0f}", f"{sm.max():,.0f}"
        z = np.load(S5 / "results/posthoc/head-spread.npz")
        within = []
        for key, hk in [("sva-end", "Agr"), ("ioi-end", "IoiEnd"), ("ioi-s2", "IoiSTwo")]:
            g = np.median(np.abs(z[f"{key}__gap"]), 0).ravel()
            hsp = np.median(z[f"{key}__head_spread"], 0).ravel()
            ok = (g > 0) & (hsp > 0)
            err = np.log10(g[ok]) - (a1 + b1 * np.log10(hsp[ok]))
            n[f"HExtrapErr{hk}"] = f(float(np.median(err)), 2)
            n[f"HExtrapWithin{hk}"] = pct(float(np.mean(np.abs(err) < 1)), 0)
            within.append(float(np.mean(np.abs(err) < 1)))
        n["HExtrapWithinLow"], n["HExtrapWithinHigh"] = pct(min(within), 0), pct(max(within), 0)
    bn = S4 / "results/posthoc/bias-noise.json"
    if bn.exists():
        b = read(bn)
        keys = {("study2", "sva-gpt2-small"): "SvaGpt", ("study2", "sva-qwen3-0.6b"): "SvaQwen",
                ("study2", "ioi-gpt2-small"): "IoiGpt", ("study3", "sva-end-gpt2-small"): "HAgr",
                ("study3", "ioi-end-gpt2-small"): "HIoiEnd", ("study3", "ioi-s2-gpt2-small"): "HIoiSTwo"}
        for panel, k in [("sva-gpt2-small", "SvaGpt"), ("sva-qwen3-0.6b", "SvaQwen")]:
            ft = b["final_token_equal_budget"][panel]
            n[f"FinalTokNoun{k}"] = f(ft["noun8"], 4)
            n[f"FinalTokMean{k}"] = f(ft["mean64"], 4)
            n[f"FinalTokSample{k}"] = f(ft["sample8_mean"], 4)
        for (study, panel), k in keys.items():
            v = b[study][panel]
            n[f"BnNear{k}"] = str(v["near_threshold_sites"])
            n[f"BnSdNear{k}"] = f(v["median_sd8_near"], 4)
            n[f"BnBiasNear{k}"] = f(v["mean"]["median_bias_near"], 4)
            n[f"BnRatioNear{k}"] = f(v["mean"]["median_ratio_bias_to_sd8_near"], 1)
            n[f"BnShareNear{k}"] = pct(v["mean"]["share_bias_above_sd8_near"], 0)
            if v["disputed_sites"]:
                n[f"BnRatioMis{k}"] = f(v["mean"]["median_ratio_bias_to_sd8_disputed"], 0)
            if "token" in v:
                n[f"BnTokBiasNear{k}"] = f(v["token"]["median_bias_near"], 4)
                n[f"BnTokRatioNear{k}"] = f(v["token"]["median_ratio_bias_to_sd8_near"], 1)
                n[f"BnTokShareNear{k}"] = pct(v["token"]["share_bias_above_sd8_near"], 0)
    return n
SETTING = {"sva-end-gpt2-small": "Agr", "ioi-end-gpt2-small": "IoiEnd", "ioi-s2-gpt2-small": "IoiSTwo"}


def study5():
    n = {}
    audit = S5 / "data/generation-audit.json"
    if audit.exists():
        n["ConfPriorStrings"] = f"{read(audit)['prior_unique_strings']:,}"
        groups = [len(set(r["s2_tokens"])) for r in read(S5 / "data/ioi-test.json")[:64]]
        n["ConfSTwoGroupsLow"], n["ConfSTwoGroupsHigh"] = str(min(groups)), str(max(groups))
        n["ConfSTwoGroupsMean"] = f(sum(groups) / len(groups), 1)
    gate = S5 / "results/qwen-gate/gate.json"
    if gate.exists():
        g = read(gate)
        n["QwenGateAcc"] = pct(g["clean_two_verb_accuracy"])
        n["QwenGateRequired"] = pct(g["required"], 0)
        n["QwenGatePassed"] = "passed" if g["passed"] else "did not pass"
    cals = [read(c) for c in sorted(S5.glob("results/*/calibration.json"))]
    if cals:
        n["ConfCalibrations"] = str(len(cals))
        n["ConfCalPassed"] = "all passed" if all(c["passed"] for c in cals) else "not all passed"
        mx = max(c["maximum_logit_error"] for c in cals)
        mant, exp = f"{mx:.1e}".split("e")
        n["ConfCalMax"] = f"${mant}\\times 10^{{{int(exp)}}}$"
    reps = [read(r) for r in sorted(S5.glob("results/*/replay-verification.json"))]
    if reps:
        mx = max(r["maximum_error"] for r in reps)
        mant, exp = f"{mx:.1e}".split("e")
        n["ConfReplayMax"] = f"${mant}\\times 10^{{{int(exp)}}}$" if mx > 0 else "0"
        n["ConfReplayText"] = "reproduced exactly" if mx == 0 else f"reproduced to within {n['ConfReplayMax']} logits"
        n["ConfReplayPassed"] = "all passed" if all(r["passed"] for r in reps) else "not all passed"
        n["ConfReplays"] = str(len(reps))
    a5 = S5 / "results/analysis-test.json"
    if a5.exists():
        a = read(a5)
        n["HSitesTotal"] = str(sum(a["settings"][k]["head_sites"] for k in SETTING if k in a["settings"]))
        import numpy as np
        n["HMaxBiasAll"] = f(max(float(np.abs(np.array(a["settings"][k]["maps"]["MEAN"]) - np.array(a["settings"][k]["maps"]["IND"])).max())
                                 for k in SETTING if k in a["settings"]), 3)
        for setting, key in SETTING.items():
            if setting not in a["settings"]:
                continue
            p = a["settings"][setting]
            n[f"HSites{key}"] = str(p["head_sites"])
            pm = p["primary_mean_misclassified"]
            n[f"HMisMean{key}"] = str(pm["estimate"])
            show = lambda x: f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"
            n[f"HMisMeanCI{key}"] = "[{}, {}]".format(*(show(x) for x in pm["ci"]))
            n[f"HMisMeanSupported{key}"] = "supported" if pm["ci"][0] >= 1 else "not supported"
            for meth, mk in [("MEAN", "Mean"), ("TOK", "Tok"), ("MIX8", "Mix")]:
                mm = p["methods"][meth]
                n[f"H{mk}Mis{key}"] = str(mm["tau0.05"]["misclassified"])
                n[f"H{mk}Fp{key}"] = str(mm["tau0.05"]["false_positive"])
                n[f"H{mk}Fn{key}"] = str(mm["tau0.05"]["false_negative"])
                n[f"H{mk}Mae{key}"] = f(mm["mae"], 4)
                n[f"H{mk}MisTen{key}"] = str(mm["tau0.1"]["misclassified"])
                n[f"H{mk}MisTwo{key}"] = str(mm["tau0.02"]["misclassified"])
            n[f"HFlagInd{key}"] = str(p["methods"]["MEAN"]["tau0.05"]["flagged_ind"])
            s8 = p["methods"]["S8"]
            n[f"HSampleMis{key}"] = f(s8["tau0.05"]["misclassified_mean"], 2)
            n[f"HSampleMisRange{key}"] = "{:.0f}--{:.0f}".format(*s8["tau0.05"]["misclassified_range95"])
            n[f"HSampleMae{key}"] = f(s8["mae_mean"], 4)
            n[f"HSampleMaeLow{key}"] = f(s8["mae_range95"][0], 4)
            tv = p["tok_vs_s8"]
            n[f"HTokBetter{key}"] = "yes" if tv["tok_better"] else "no"
            n[f"HTokVerdict{key}"] = "beats" if tv["tok_better"] else "does not beat"
            n[f"HTokDiffCI{key}"] = ci(tv["mae_difference_ci95"], 4)
            n[f"HTokEvals{key}"] = f(p["evaluations_per_recipient_site"]["TOK"], 0)
            n[f"HGroups{key}"] = str(p["groups"])
            import numpy as np
            ind_map = np.array(p["maps"]["IND"])
            n[f"HMaxBias{key}"] = f(float(np.abs(np.array(p["maps"]["MEAN"]) - ind_map).max()), 4)
            if setting.startswith("ioi"):
                # Overlap of the flagged heads with the IOI circuit's heads at the same position (run_reference.GROUPS).
                import ast
                src = (S4 / "run_reference.py").read_text()
                groups = ast.literal_eval(src[src.index("GROUPS = {") + len("GROUPS = "):src.index("}", src.index("GROUPS = {")) + 1])
                role = "end" if setting.startswith("ioi-end") else "s2"
                at_role = {g for g in groups if ("s2" if g in ("induction", "duplicate_token") else
                                                  "s1_plus1" if g == "previous_token" else "end") == role}
                circuit = {tuple(h) for g in at_role for h in groups[g]}
                flagged = {(int(l), int(h)) for l, h in np.argwhere(np.abs(ind_map[0]) >= 0.05)}
                n[f"HCircFlag{key}"] = str(len(flagged & circuit))
                n[f"HCircHeads{key}"] = str(len(circuit))
            n[f"HMaxBiasTok{key}"] = f(float(np.abs(np.array(p["maps"]["TOK"]) - ind_map).max()), 4)
            n[f"HMaxEffect{key}"] = f(float(np.abs(ind_map).max()), 2)
            n[f"HFlagMean{key}"] = str(p["methods"]["MEAN"]["tau0.05"]["flagged"])
            n[f"HFlagIndTwo{key}"] = str(p["methods"]["MEAN"]["tau0.02"]["flagged_ind"])
            n[f"HFlagIndTen{key}"] = str(p["methods"]["MEAN"]["tau0.1"]["flagged_ind"])
            n[f"HSampleMisTwo{key}"] = f(s8["tau0.02"]["misclassified_mean"], 2)
            n[f"HSampleMisTen{key}"] = f(s8["tau0.1"]["misclassified_mean"], 2)
            n[f"HSampleMaeHigh{key}"] = f(s8["mae_range95"][1], 4)
            # Mean absolute errors in units of 1e-4 (normalized effect), for the Study 3 table.
            for meth, mk in [("MEAN", "Mean"), ("TOK", "Tok"), ("MIX8", "Mix")]:
                n[f"H{mk}Err{key}"] = f(1e4 * p["methods"][meth]["mae"], 2)
            n[f"HSampleErr{key}"] = f(1e4 * s8["mae_mean"], 2)
            n[f"HSampleErrRange{key}"] = f"{f(1e4 * s8['mae_range95'][0], 2)}--{f(1e4 * s8['mae_range95'][1], 2)}"
            n[f"HSampleErrLow{key}"] = f(1e4 * s8["mae_range95"][0], 2)
            if p["methods"]["TOK"]["mae"] > 0:
                n[f"HTokRatio{key}"] = f(p["methods"]["MEAN"]["mae"] / p["methods"]["TOK"]["mae"], 0)
            n[f"HTokDiffErrCI{key}"] = "[{}, {}]".format(*(f(1e4 * x, 2) for x in tv["mae_difference_ci95"]))
            n[f"HTokDiffErr{key}"] = f(1e4 * (tv["tok_mae"] - tv["s8_mae_mean"]), 2)
    return n


SVA_POS = {"adj": "adjective", "anchor+0": "head noun", "anchor+1": "word after the head noun",
           "anchor+2": "second word after the head noun", "anchor+3": "third word after the head noun", "END": "final token"}
IOI_POS = {"prefix": "opening word", "N1": "first name", "and": "``and''", "N2": "second name", "went": "``went''",
           "to": "``to''", "the": "``the''", "PLACE": "place word", "punct": "punctuation", "S2": "repeated subject",
           "gave": "verb", "a": "``a''", "OBJ": "object word", "END": "final token"}


def np_maps(p, method):
    return p["maps"][method]


def describe_sites(lst, positions, sva):
    """'object word, blocks 0--2; final token, blocks 2 and 5' from [direction, layer, position] triples."""
    if not lst:
        return "none"
    names = SVA_POS if sva else IOI_POS
    groups = {}
    for d, l, q in lst:
        label = names.get(positions[q], positions[q])
        if sva:
            label += " (" + ("restoration" if d == 0 else "destruction") + ")"
        groups.setdefault(label, []).append(l)
    parts = []
    for label, layers in groups.items():
        layers = sorted(layers)
        if len(layers) > 2 and layers == list(range(layers[0], layers[-1] + 1)):
            span = f"in blocks {layers[0]}--{layers[-1]}"
        elif len(layers) > 4:
            span = f"in {len(layers)} blocks between {layers[0]} and {layers[-1]}"
        elif len(layers) == 1:
            span = f"in block {layers[0]}"
        else:
            span = "in blocks " + ", ".join(map(str, layers[:-1])) + " and " + str(layers[-1])
        parts.append(f"{label} {span}")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and the " + parts[-1]


CELL = {"N8": "NEight", "NPO8": "NPOEight", "N128": "NBig", "NPO128": "NPOBig"}
PANEL = {"sva-gpt2-small": "SvaGpt", "sva-qwen3-0.6b": "SvaQwen", "ioi-gpt2-small": "IoiGpt"}


def study4():
    n = {}
    curv = S4 / "results/curvature.json"
    if curv.exists():
        c = read(curv)["summary"]
        n["CurvGroups"] = str(c["groups"])
        n["CurvBanks"] = f"{c['all']['banks']:,}"
        n["CurvCorrAll"] = f(c["all"]["corr"], 2)
        n["CurvSignAll"] = pct(c["all"]["sign_agreement"], 0)
        n["CurvSlopeAll"] = f(c["all"]["slope_measured_on_predicted"], 2)
        n["CurvCorrNoun"] = f(c["noun8"]["corr"], 3)
        n["CurvRsqNoun"] = f(c["noun8"]["r2_identity_prediction"], 3)
        n["CurvSlopeNoun"] = f(c["noun8"]["slope_measured_on_predicted"], 2)
        mixed = [c["mix8"], c["adj8"]]
        n["CurvCorrMixed"] = f(min(x["corr"] for x in mixed), 2)
        n["CurvSlopeMixed"] = f(sum(x["slope_measured_on_predicted"] for x in mixed) / 2, 2)
        n["CurvOverMixed"] = pct(1 / (sum(x["slope_measured_on_predicted"] for x in mixed) / 2) - 1, 0)
        n["CurvRsqMixed"] = f(sum(x["r2_identity_prediction"] for x in mixed) / 2, 2)
        n["CurvRatioMixed"] = pct(sum(x["mean_abs_predicted"] for x in mixed) / sum(x["mean_abs_measured"] for x in mixed) - 1, 0)
    sweep = S4 / "results/sweep/analysis-test.json"
    if sweep.exists():
        a = read(sweep)
        for panel, key in PANEL.items():
            if panel not in a["panels"]:
                continue
            p = a["panels"][panel]
            n[f"Groups{key}"] = str(p["groups"])
            fl = p["flags"]
            m = fl["mean64/tau0.05"]
            n[f"Sites{key}"] = str(m["sites"])
            n[f"MisMean{key}"] = str(m["misclassified_sites"]["estimate"])
            n[f"MisMeanCI{key}"] = "[{:.0f}, {:.0f}]".format(*m["misclassified_sites"]["ci"])
            n[f"FPMean{key}"] = str(m["false_positive_sites"]["estimate"])
            n[f"FNMean{key}"] = str(m["false_negative_sites"]["estimate"])
            n[f"FlagInd{key}"] = str(m["flagged_by_individual"]["estimate"])
            n[f"FlagMean{key}"] = str(m["flagged_by_method"]["estimate"])
            fdp = m["false_discovery_proportion"]["estimate"]
            n[f"FDPMean{key}"] = pct(fdp, 0) if fdp is not None else "--"
            n[f"MisMeanTen{key}"] = str(fl["mean64/tau0.1"]["misclassified_sites"]["estimate"])
            for meth, mk in [("mix8", "Mix"), ("noun8", "Noun"), ("adj8", "Adj")]:
                if f"{meth}/tau0.05" in fl:
                    n[f"Mis{mk}{key}"] = str(fl[f"{meth}/tau0.05"]["misclassified_sites"]["estimate"])
            b = p["bias"]["mean64"]
            n[f"MaxBias{key}"] = f(b["max_abs_bias"])
            n[f"BiasWeak{key}"] = f(b["mean_abs_bias_weak_sites"])
            n[f"BiasStrong{key}"] = f(b["mean_abs_bias_strong_sites"])
            n[f"Spearman{key}"] = f(p["spearman_abs_effect"]["mean64"])
            if panel.startswith("ioi"):
                for pos, pk in [("N1", "NOne"), ("N2", "NTwo")]:
                    n[f"Bias{pk}{key}"] = f(b["by_position"][pos]["mean_abs_bias"]["estimate"])
            n[f"SweepSlope{key}"] = f(p["spread_law"]["loglog_slope"], 2)
            n[f"SweepRsq{key}"] = f(p["spread_law"]["within_cell_r2"], 2)
            # Plain-language description of where the flags differ, generated from the site lists.
            ind = np_maps(p, "individual")
            mean = np_maps(p, "mean64")
            for kind, lst in [("FP", m["false_positive_list"]), ("FN", m["false_negative_list"])]:
                n[f"{kind}Where{key}"] = describe_sites(lst, p["positions"], panel.startswith("sva"))
                if lst:
                    e_ind = [abs(ind[d][l][q]) for d, l, q in lst]
                    e_mean = [abs(mean[d][l][q]) for d, l, q in lst]
                    n[f"{kind}IndLow{key}"], n[f"{kind}IndHigh{key}"] = pct(min(e_ind)), pct(max(e_ind))
                    n[f"{kind}MeanLow{key}"], n[f"{kind}MeanHigh{key}"] = pct(min(e_mean)), pct(max(e_mean))
                    flips = sum(1 for d, l, q in lst if ind[d][l][q] * mean[d][l][q] < 0)
                    n[f"{kind}SignFlips{key}"] = str(flips)
                    n[f"{kind}SignFlipsText{key}"] = ("at all of these sites" if flips == len(lst) else
                                                      f"at {flips} of these {len(lst)} sites")
            # Bias relative to effect at weak sites, structural zeros excluded (post hoc description).
            import sys
            import numpy as np
            from scipy.stats import spearmanr
            sys.path.insert(0, str(S4))
            from analyze_sweep import load_panel
            live = ~load_panel(panel, "test")["structural"]
            e = np.abs(np.array(ind))[live]
            bias_abs = np.abs(np.array(mean) - np.array(ind))[live]
            weak = e < 0.05
            n[f"WeakSites{key}"] = str(int(weak.sum()))
            n[f"WeakBiasOver{key}"] = str(int((bias_abs[weak] > e[weak]).sum()))
            n[f"WeakBiasOverPct{key}"] = pct(float((bias_abs[weak] > e[weak]).mean()), 0)
            n[f"WeakBiasAboveTau{key}"] = str(int((bias_abs[weak] >= 0.05).sum()))
            n[f"MisRate{key}"] = pct(m["misclassified_sites"]["estimate"] / m["sites"], 1)
            n[f"EffectMax{key}"] = f(float(e.max()), 2)
            dis = m["false_positive_list"] + m["false_negative_list"]
            n[f"MisSiteMax{key}"] = f(max(max(abs(ind[d][l][q]), abs(mean[d][l][q])) for d, l, q in dis), 3)
            n[f"BiasEffectRho{key}"] = f(float(spearmanr(bias_abs, e).statistic), 2)
            ok = (e > 1e-6) & (bias_abs > 1e-8)
            slope_be = float(np.polyfit(np.log(e[ok]), np.log(bias_abs[ok]), 1)[0])
            n[f"BiasEffectSlope{key}"] = f(slope_be, 2)
            if "end_gap_by_partition" in p:
                for meth, mk in [("mix8", "Mix"), ("noun8", "Noun"), ("adj8", "Adj")]:
                    n[f"EndGap{mk}{key}"] = f(p["end_gap_by_partition"][meth])
                    n[f"EndSpread{mk}{key}"] = f"{p['end_spread_by_partition'][meth]:,.0f}"
    n.update(numerics())
    if all(f"MisNoun{k}" in n for k in ["SvaGpt", "SvaQwen"]):
        total = int(n["MisMeanSvaGpt"]) + int(n["MisMeanSvaQwen"])
        left = int(n["MisNounSvaGpt"]) + int(n["MisNounSvaQwen"])
        n["NounRemovesText"] = "all" if left == 0 else f"{total - left} of the {total}"
    ref = S4 / "results/reference/analysis-test.json"
    if ref.exists():
        a = read(ref)
        sel = read(S4 / "results/reference/screen-selection.json")
        n["RefRecipients"] = f"{a['recipients']:,}"
        n["RefScreened"] = f"{sel['screened']:,}"
        for fam in ["O", "W"]:
            st = sel["strata"]
            n[f"RefPopErr{fam}"] = f"{st[f'{fam}/pair_error']['population']:,}"
            n[f"RefSelErr{fam}"] = f"{st[f'{fam}/pair_error']['selected']:,}"
            n[f"RefAcc{fam}"] = pct(sel["population_pair_accuracy"][fam], 2)
        pr = a["primary"]
        for k, name in [("AOne", "A1_npo128_minus_n8_error_agreement_mean_ablation"),
                        ("ATwo", "A2_circuit_minus_empty_error_agreement_n8_mean_ablation")]:
            n[k] = pct(pr[name]["estimate"])
            n[k + "CI"] = ci(pr[name]["ci"], 1, 100)
        for fam_key, fam in [("pooled", ""), ("O", "FamO"), ("W", "FamW")]:
            for stratum, sk in [("error", "Err"), ("correct", "Ok")]:
                t = a["tables"].get(f"{fam_key}/{stratum}")
                if not t:
                    continue
                for c, ck in CELL.items():
                    for pool, pk in [("mean_agree", "Mean"), ("resample_agree", "Res")]:
                        for mode, mk in [("reference", "Circ"), ("empty", "Empty")]:
                            n[f"{sk}{mk}{ck}{pk}{fam}"] = pct(t[f"{c}/{mode}/{pool}"]["estimate"])
                            if not fam:
                                n[f"{sk}{mk}{ck}{pk}CI"] = ci(t[f"{c}/{mode}/{pool}"]["ci"], 1, 100)
                        ex = t[f"{c}/excess_over_empty/{pool}"]
                        n[f"{sk}Excess{ck}{pk}{fam}"] = pct(ex["estimate"])
                        if not fam:
                            n[f"{sk}Excess{ck}{pk}CI"] = ci(ex["ci"], 1, 100)
            fa = a["tables"].get(f"{fam_key}/faithfulness")
            if fa:
                for c, ck in CELL.items():
                    for pool, pk in [("mean", "Mean"), ("resample", "Res")]:
                        n[f"Faith{ck}{pk}{fam}"] = f(fa[f"{c}/{pool}"]["estimate"])
        tok = a["tables"]["pooled/correct"]
        circ = [tok[f"{c}/reference/{pl}"]["estimate"] for c in CELL for pl in ["mean_agree", "resample_agree"]]
        emp = [tok[f"{c}/empty/{pl}"]["estimate"] for c in CELL for pl in ["mean_agree", "resample_agree"]]
        n["OkCircLow"], n["OkCircHigh"] = pct(min(circ), 0), pct(max(circ), 0)
        n["OkEmptyLow"], n["OkEmptyHigh"] = pct(min(emp), 0), pct(max(emp), 0)
        n["AOneAbsRounded"] = str(round(abs(pr["A1_npo128_minus_n8_error_agreement_mean_ablation"]["estimate"]) * 100))
        te = a["tables"]["pooled/error"]
        n["ErrMarginNPOBigMean"] = f(te["NPO128/reference/mean_margin"]["estimate"], 2)
        n["ErrExcessNPOBigResAbs"] = pct(abs(te["NPO128/excess_over_empty/resample_agree"]["estimate"]))
        tw = a["tables"]["W/error"]
        n["ErrCircNPOBigMeanFamWCI"] = ci(tw["NPO128/reference/mean_agree"]["ci"], 1, 100)
        n["ExplainedGap"] = str(round(100 * (te["N8/reference/mean_agree"]["estimate"] - te["NPO128/reference/mean_agree"]["estimate"])))
        zhang_mean_error_agreement = 15.2   # Zhang et al. (2026), Table 1 (literal-numbers.json)
        s3_err = read(S4 / "results/retrospective-study3.json")["circuit"]["reference/full_pair_error"]["mean_agreement"]
        n["StudyThreeGap"] = str(round(100 * s3_err - zhang_mean_error_agreement))
        ver = S4 / "results/verification.json"
        if ver.exists():
            n["IoiWithoutTwo"] = str(read(ver)["B_misclassified_sites"]["ioi-gpt2-small/without_first_two_prompts"])
        sec = a["secondary_contrasts"]
        for name, key in [("content_effect_size8", "ContentEight"), ("content_effect_size128", "ContentBig"),
                          ("size_effect_names_only", "SizeN"), ("size_effect_names_place_object", "SizeNPO"),
                          ("content_main_effect", "ContentMain"), ("size_main_effect", "SizeMain")]:
            if f"{name}/mean_agree" not in sec:
                continue
            for pool, pk in [("mean_agree", "Mean"), ("resample_agree", "Res")]:
                v = sec[f"{name}/{pool}"]
                n[f"{key}{pk}"] = pct(v["estimate"])
                n[f"{key}{pk}CI"] = ci(v["ci"], 1, 100)
    return n


def numerics():
    """Largest identity-patch error in Studies 1 and 2, and the largest fresh-process replay difference in Study 2.

    Identity checks only: the identity entries of every Study 2 calibration, the identity audits of the Study 2
    sweep test runs, the identity errors of the Experiment A replays, the identity audits of Study 1's token test
    runs and Study 1's circuit audit. (Until the fifth revision this macro took each calibration's overall
    maximum, which includes a comparison with the native forward pass, and read Study 2 only.)"""
    import numpy as np
    n, ident, replay = {}, [], []
    cals = [S4 / "results/reference/calibration.json"] + sorted((S4 / "results/sweep").glob("*/calibration.json"))
    for c in cals:
        if c.exists():
            ident += [v for k, v in read(c)["errors"].items() if k.endswith("/identity")]
    for comp in sorted((S4 / "results/sweep").glob("*/test-completion.json")):
        ident += [a["identity"] for a in read(comp)["audits"].values()]
    rdir = S4 / "results/reference/replay"
    if rdir.exists():
        for path in sorted(rdir.glob("*.npz")):
            with np.load(path) as a, np.load(S4 / "results/reference/test" / path.name) as b:
                for k in a.files:
                    if "argmax" not in k:
                        replay.append(float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))))
                ident.append(float(a["identity_error"]))
    for audit in sorted(S3.glob("token/results/*/test/*-audit.json")):
        ident += [row["identity_logit_max"] for row in read(audit)]
    circuit_audit = S3 / "circuit/results/final-audit.json"
    if circuit_audit.exists():
        ident.append(read(circuit_audit)["maximum_identity_error"])
    for rv in sorted((S4 / "results/sweep").glob("*/replay-verification.json")):
        replay.append(read(rv)["maximum_error"])
    def sci(x):
        mant, exp = f"{x:.1e}".split("e")
        return f"${mant}\\times 10^{{{int(exp)}}}$" if x > 0 else "0"
    if ident:
        n["NumIdentityMax"] = sci(max(ident))
    if replay:
        n["NumReplayMax"] = sci(max(replay))
        n["NumReplayText"] = ("reproduced exactly" if max(replay) == 0
                              else f"reproduced to within {sci(max(replay))} logits")
    return n


def revision5(prev):
    """Fifth revision (13 October 2026): prespecified results added and post hoc controls from the readiness review.

    Post hoc controls come from studies/04-reference-and-sweeps/results/posthoc/review-controls.json (checked by
    posthoc/verify_review_controls.py); the other values come from the frozen analyses' saved outputs."""
    import numpy as np
    n = {}
    # Plain-language figures for the abstract and introduction.
    sweep = read(S4 / "results/sweep/analysis-test.json")["panels"]
    rates = [100 * p["flags"]["mean64/tau0.05"]["misclassified_sites"]["estimate"] / p["flags"]["mean64/tau0.05"]["sites"]
             for p in sweep.values()]
    n["MisRatePctLow"], n["MisRatePctHigh"] = str(round(min(rates))), str(round(max(rates)))
    ref = read(S4 / "results/reference/analysis-test.json")
    te, tc = ref["tables"]["pooled/error"], ref["tables"]["pooled/correct"]
    n["ErrCircNEightMeanInt"] = str(round(100 * te["N8/reference/mean_agree"]["estimate"]))
    n["ErrCircNPOBigMeanInt"] = str(round(100 * te["NPO128/reference/mean_agree"]["estimate"]))
    # Study 1: the K = 8 gap as a share of the clean-minus-corrupted margin difference.
    r = read(S4 / "results/retrospective-study3.json")["grouping"]
    for key, model in [("Gpt", "gpt2-small"), ("Qwen", "qwen3-0.6b")]:
        dc = read(S3 / f"token/results/analysis-test/{model}.json")["secondary_dose_curve"]
        n[f"GroupKeightShare{key}"] = pct(dc["8"]["margin_gap_absolute"]["mean"] / r[model]["clean_minus_corrupt"], 0)
    # Experiment A: how much each choice moves each summary (from the cells of Table 4).
    cells = ["N8", "N128", "NPO8", "NPO128"]
    pairs = {"N8": "NPO8", "N128": "NPO128"}
    fa = ref["tables"]["pooled/faithfulness"]
    comb_err = [abs(te[f"{c}/reference/mean_agree"]["estimate"] - te[f"{c}/reference/resample_agree"]["estimate"]) for c in cells]
    comb_ok = [abs(tc[f"{c}/reference/mean_agree"]["estimate"] - tc[f"{c}/reference/resample_agree"]["estimate"]) for c in cells]
    comb_fa = [abs(fa[f"{c}/mean"]["estimate"] - fa[f"{c}/resample"]["estimate"]) for c in cells]
    cont_err = [abs(te[f"{a}/reference/{pl}"]["estimate"] - te[f"{b}/reference/{pl}"]["estimate"])
                for a, b in pairs.items() for pl in ["mean_agree", "resample_agree"]]
    cont_ok = [abs(tc[f"{a}/reference/{pl}"]["estimate"] - tc[f"{b}/reference/{pl}"]["estimate"])
               for a, b in pairs.items() for pl in ["mean_agree", "resample_agree"]]
    cont_fa = [abs(fa[f"{a}/{pl}"]["estimate"] - fa[f"{b}/{pl}"]["estimate"]) for a, b in pairs.items() for pl in ["mean", "resample"]]
    n["CombErrLow"], n["CombErrHigh"] = pct(min(comb_err), 0), pct(max(comb_err), 0)
    n["ContentErrLow"], n["ContentErrHigh"] = pct(min(cont_err), 0), pct(max(cont_err), 0)
    n["CombOkLow"], n["CombOkHigh"] = pct(min(comb_ok), 0), pct(max(comb_ok), 0)
    n["ContentOkHigh"] = pct(max(cont_ok), 0)
    n["CombFaithLow"], n["CombFaithHigh"] = f(min(comb_fa), 2), f(max(comb_fa), 2)
    n["ContentFaithHigh"] = f(max(cont_fa), 2)
    # Experiment A, prespecified secondary: unrestricted top-1 agreement and the choice after averaging donor logits.
    def rng(tab, metric, cs, d=0):
        vals = [tab[f"{c}/reference/{metric}"]["estimate"] for c in cs]
        return pct(min(vals), d), pct(max(vals), d)
    n["TopOneOkCircMeanLow"], n["TopOneOkCircMeanHigh"] = rng(tc, "mean_top1_agree", cells)
    n["TopOneOkCircResLow"], n["TopOneOkCircResHigh"] = rng(tc, "resample_top1_agree", cells)
    n["LogitErrCircNLow"], n["LogitErrCircNHigh"] = rng(te, "meanlogit_agree", ["N8", "N128"])
    n["LogitErrCircNPOLow"], n["LogitErrCircNPOHigh"] = rng(te, "meanlogit_agree", ["NPO8", "NPO128"])
    ed = read(S4 / "results/posthoc/error-discrimination.json")["results"]
    n["DiscPCorrCircNPOBigMeanOne"] = pct(ed["circuit/mean/NPO128"]["p_subject_given_correct"], 1)
    # Study 4, prespecified secondary: misclassified site-directions at the secondary thresholds.
    a6 = read(S6 / "results/analysis-test.json")["kinds"]
    for kind, kk in COMP.items():
        for t_, tk in [("0.02", "Two"), ("0.1", "Ten")]:
            n[f"CompMis{tk}{kk}"] = str(a6[kind]["methods"]["MEAN"][f"tau{t_}"]["misclassified"])
    # Head-level extrapolation: head-sites left out because their gap is exactly zero (they cannot change the output).
    z = np.load(S5 / "results/posthoc/head-spread.npz")
    for key, hk in [("sva-end", "Agr"), ("ioi-end", "IoiEnd"), ("ioi-s2", "IoiSTwo")]:
        g = np.median(np.abs(z[f"{key}__gap"]), 0).ravel()
        hsp = np.median(z[f"{key}__head_spread"], 0).ravel()
        ok = (g > 0) & (hsp > 0)
        n[f"HExtrapSites{hk}"] = str(int(ok.sum()))
        n[f"HExtrapExcluded{hk}"] = str(int((~ok).sum()))
    # Post hoc review controls.
    rc = read(S4 / "results/posthoc/review-controls.json")
    for panel, key in PANEL.items():
        p = rc["panels"][panel]
        one = p["sampled_donors"]["1"]
        n[f"SampleOneMis{key}"] = f(one["misclassified_mean"], 1)
        n[f"SampleOneMisRange{key}"] = "{:.0f}--{:.0f}".format(*one["misclassified_range95"])
        n[f"SampleOneMae{key}"] = f(one["mae_mean"], 4)
        n[f"MeanAblMae{key}"] = f(p["mean_ablation"]["mae"], 4)
        n[f"SplitHalf{key}"] = f(p["split_half"]["halves_disagree_mean"], 2)
        if "token_rule" in p:
            n[f"TokRuleMis{key}"] = str(p["token_rule"]["rule"]["misclassified"])
            n[f"TokRuleMae{key}"] = f(p["token_rule"]["rule"]["mae"], 4)
    n["SplitHalfMax"] = str(max(rc["panels"][p]["split_half"]["halves_disagree_max"] for p in PANEL))
    n["SampleDraws"] = f"{rc['cost_draws']:,}"
    n["SplitDraws"] = f"{rc['panels']['ioi-gpt2-small']['split_half']['splits']:,}"
    ic = rc["panels"]["ioi-gpt2-small"]["ioi_count"]
    show = lambda c: "[{:.0f}, {:.0f}]".format(*c)
    for name, k in [("original_definition", "IoiOrigDef"), ("without_development_prompts", "IoiNoDev"),
                    ("without_development_prompts_original_definition", "IoiNoDevOrig"), ("without_calibration_prompts", "IoiNoCal")]:
        n[k], n[k + "CI"] = str(ic[name]["estimate"]), show(ic[name]["ci"])
    sw = rc["spread_within_policy"]
    for key, model in [("Gpt", "gpt2-small"), ("Qwen", "qwen3-0.6b")]:
        n[f"SlopeWithin{key}"] = f(sw[model]["within_policy"]["slope"], 2)
        n[f"SlopeWithinCI{key}"] = ci(sw[model]["within_policy"]["ci95"], 2)
        n[f"SlopeHighK{key}"] = f(sw[model]["k_2_4_8_only"]["slope"], 2)
        n[f"SlopeHighKCI{key}"] = ci(sw[model]["k_2_4_8_only"]["ci95"], 2)
    n["RsqWithinHigh"] = f(max(sw[m][k]["r2"] for m in sw for k in ["within_policy", "k_2_4_8_only"]), 2)
    sep = rc["separation"]
    n["NearCutTen"] = f(sep["intact_margin_cutoffs"]["0.1"], 1)
    nb = sep["near_boundary"]
    def nrng(unit, cs, field, d=2):
        vals = [nb[f"{unit}/{pl}/{c}"][field] for pl in ["mean", "resample"] for c in cs]
        return f(min(vals), d), f(max(vals), d)
    n["NearAucCircNLow"], n["NearAucCircNHigh"] = nrng("circuit", ["N8", "N128"], "auroc_below_q10")
    n["NearAucCircNPOLow"], n["NearAucCircNPOHigh"] = nrng("circuit", ["NPO8", "NPO128"], "auroc_below_q10")
    n["NearAucEmptyNPOLow"], n["NearAucEmptyNPOHigh"] = nrng("empty", ["NPO8", "NPO128"], "auroc_below_q10")
    n["NearAucEmptyNLow"], n["NearAucEmptyNHigh"] = nrng("empty", ["N8", "N128"], "auroc_below_q10")
    n["NearAucCircNPOBigMean"] = f(nb["circuit/mean/NPO128"]["auroc_below_q10"], 2)
    n["NearAucCircNPOBigMeanCI"] = ci(nb["circuit/mean/NPO128"]["auroc_below_q10_ci95"], 2)
    n["NearAucEmptyNPOBigMean"] = f(nb["empty/mean/NPO128"]["auroc_below_q10"], 2)
    n["NearAucEmptyNPOBigMeanCI"] = ci(nb["empty/mean/NPO128"]["auroc_below_q10_ci95"], 2)
    n["RankErrCircNPOLow"], n["RankErrCircNPOHigh"] = nrng("circuit", ["NPO8", "NPO128"], "rank_corr_with_intact_errors")
    n["RankErrCircNLow"], n["RankErrCircNHigh"] = nrng("circuit", ["N8", "N128"], "rank_corr_with_intact_errors")
    sb = sep["surface_baseline"]
    n["SurfAuc"], n["SurfAucCI"] = f(sb["all_surface_features"]["auroc"], 2), ci(sb["all_surface_features"]["auroc_ci95"], 2)
    n["SurfAucWithin"] = f(sb["all_surface_features"]["auroc_within_template"], 2)
    n["SurfAucNames"] = f(sb["names"]["auroc"], 2)
    n["SurfAucTemplate"] = f(sb["template_order"]["auroc"], 2)
    n["SurfTrain"] = f"{sb['all_surface_features']['train_prompts']:,}"
    # Study 1's IOI word pools (studies/03-grouping-and-circuit/circuit/experiment.py).
    import re
    src = (S3 / "circuit/experiment.py").read_text()
    for var, key in [("PLACES", "SOnePlaces"), ("OBJECTS", "SOneObjects")]:
        n[key] = str(len(re.search(rf'^{var} = "([^"]*)"\.split\(\)', src, flags=re.M).group(1).split()))
    return n


def misclassified_table():
    """Appendix table: every site whose flag differs between mean ablation and the individual average
    (misclassified sites; the post hoc result file calls them "disputed")."""
    ph = S4 / "results/posthoc/sweep-posthoc.json"
    if not ph.exists():
        return None
    d = read(ph)
    names = {"sva-gpt2-small": "Agreement, GPT-2 Small", "sva-qwen3-0.6b": "Agreement, Qwen3-0.6B",
             "ioi-gpt2-small": "IOI, GPT-2 Small"}
    lines = ["% Generated by paper/make_numbers.py from studies/04-reference-and-sweeps/results/posthoc/sweep-posthoc.json.",
             "\\begin{tabular}{@{}lrlrrrrrr@{}}", "\\toprule",
             " & & & \\multicolumn{2}{c}{Effect} & \\multicolumn{2}{c}{Distance from $\\tau$} & \\multicolumn{2}{c}{Single donors} \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
             "Direction & Block & Position & Individual & Mean & Individual & Mean & Mean $|$effect$|$ & Opposite (\\%) \\\\"]
    for panel, label in names.items():
        p = d["panels"][panel]
        sva = panel.startswith("sva")
        lines.append(f"\\midrule\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
        for s, c in zip(p["disputed_sites"], p["cancellation"]["disputed_sites"]):
            direction = ("restore" if s["direction"] == 0 else "destroy") if sva else "noise"
            pos = (SVA_POS if sva else IOI_POS).get(s["position"], s["position"])
            lines.append(" & ".join([direction, str(s["layer"]), pos, f(s["individual"]), f(s["mean"]),
                                     f(s["distance_individual"]), f(s["distance_mean"]),
                                     f(c["mean_abs_per_donor_effect"]), pct(c["share_opposite_sign"], 0)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def token_group_table():
    """Appendix table: groups per prompt under the token rule at each aligned IOI sweep position."""
    ph = S4 / "results/posthoc/sweep-posthoc.json"
    if not ph.exists():
        return None
    g = read(ph)["ioi_token_groups"]
    order = ["prefix", "N1", "and", "N2", "went", "to", "the", "PLACE", "punct", "S2", "gave", "a", "OBJ", "END"]
    label = {"prefix": "opening", "and": "and", "went": "went", "to": "to", "the": "the", "punct": "punct.",
             "gave": "gave", "a": "a", "END": "final"}
    lines = ["% Generated by paper/make_numbers.py from studies/04-reference-and-sweeps/results/posthoc/sweep-posthoc.json.",
             "\\begin{tabular}{@{}l" + "r" * len(order) + "@{}}", "\\toprule",
             "Position & " + " & ".join(label.get(k, k) for k in order) + " \\\\", "\\midrule",
             "Groups per prompt & " + " & ".join(f(g[k]["groups_mean"], 1) if g[k]["groups_max"] > 1 else "1" for k in order) + " \\\\",
             "Range & " + " & ".join(f"{g[k]['groups_min']}--{g[k]['groups_max']}" if g[k]["groups_max"] > 1 else "1" for k in order) + " \\\\",
             "Mean group size & " + " & ".join(f(g[k]["mean_group_size"], 1) for k in order) + " \\\\",
             "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def discrimination_table():
    """Appendix table: error discrimination for the circuit, each random circuit and the empty control."""
    path = S4 / "results/posthoc/error-discrimination.json"
    if not path.exists():
        return None
    r = read(path)["results"]
    units = [("circuit", "IOI circuit")] + [(f"random{k}", f"random {k + 1}") for k in range(5)] + [("empty", "no heads")]
    lines = ["% Generated by paper/make_numbers.py from studies/04-reference-and-sweeps/results/posthoc/error-discrimination.json.",
             "\\begin{tabular}{@{}lllrrrrr@{}}", "\\toprule",
             " & & & \\multicolumn{2}{c}{P(subject)} & & & \\\\", "\\cmidrule(lr){4-5}",
             "Ablation & Reference & Circuit & Errors & Correct & Odds ratio & AUROC & Within template \\\\"]
    for pool, plabel in [("mean", "Mean"), ("resample", "Resample")]:
        lines.append("\\midrule")
        for ci_, cell in enumerate(["N8", "N128", "NPO8", "NPO128"]):
            for ui, (u, ulabel) in enumerate(units):
                x = r[f"{u}/{pool}/{cell}"]
                lead = [plabel if (ci_ == 0 and ui == 0) else "", cell if ui == 0 else ""]
                orr = f"{x['odds_ratio']:.1f} [{x['odds_ratio_ci95'][0]:.1f}, {x['odds_ratio_ci95'][1]:.1f}]"
                auc = f"{f(x['auroc'], 3)} [{f(x['auroc_ci95'][0], 3)}, {f(x['auroc_ci95'][1], 3)}]"
                lines.append(" & ".join(lead + [ulabel, f(x["p_subject_given_error"], 3), f(x["p_subject_given_correct"], 3),
                                                orr, auc, f(x["auroc_within_template"], 3)]) + " \\\\")
            if ci_ < 3:
                lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def components_table(stage="test"):
    """Appendix table: every Study 4 site-direction that any method flags at tau = 0.05, with all four effects."""
    import csv
    path = S6 / f"results/sites-{stage}.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    names = {"resid": "Residual stream", "mlp": "MLP output", "attn": "Attention-layer output", "head": "Single head"}
    lines = ["% Generated by paper/make_numbers.py from studies/06-component-confirmation/results/sites-" + stage + ".csv.",
             "\\begin{tabular}{@{}lrrrrrrl@{}}", "\\toprule",
             "Direction & Layer & Head & IND & Mean & TOK & MIX8 & Flagged by \\\\"]
    for kind, label in names.items():
        sub = [r for r in rows if r["kind"] == kind]
        hit = [r for r in sub if any(r[f"flag_{m}"] == "1" for m in ["IND", "MEAN", "TOK", "MIX8"])]
        lines.append(f"\\midrule\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{label}: {len(hit)} of {len(sub)} site-directions}}}} \\\\")
        for r in hit:
            flagged = [m if m != "MEAN" else "Mean" for m in ["IND", "MEAN", "TOK", "MIX8"] if r[f"flag_{m}"] == "1"]
            text = "all" if len(flagged) == 4 else ", ".join(flagged)
            lines.append(" & ".join([r["direction"], r["layer"], r["head"] or "--"]
                                    + [f(float(r[f"effect_{m}"])) for m in ["IND", "MEAN", "TOK", "MIX8"]] + [text]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def heads_table(stage="test"):
    """Appendix table: every Study 3 head-site that any method flags at tau = 0.05, with all four effects."""
    import csv
    names = {"sva-end-gpt2-small": "Agreement, final token", "ioi-end-gpt2-small": "IOI, final token",
             "ioi-s2-gpt2-small": "IOI, repeated subject"}
    if not all((S5 / f"results/heads-{k}-{stage}.csv").exists() for k in names):
        return None
    lines = ["% Generated by paper/make_numbers.py from studies/05-head-confirmation/results/heads-*-" + stage + ".csv.",
             "\\begin{tabular}{@{}lrrrrrrl@{}}", "\\toprule",
             "Direction & Layer & Head & IND & Mean & TOK & MIX8 & Flagged by \\\\"]
    for setting, label in names.items():
        rows = list(csv.DictReader((S5 / f"results/heads-{setting}-{stage}.csv").open()))
        hit = [r for r in rows if any(r[f"flag_{m}_tau0.05"] == "1" for m in ["IND", "MEAN", "TOK", "MIX8"])]
        lines.append(f"\\midrule\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{label}: {len(hit)} of {len(rows)} head-sites}}}} \\\\")
        for r in hit:
            flagged = [m if m != "MEAN" else "Mean" for m in ["IND", "MEAN", "TOK", "MIX8"] if r[f"flag_{m}_tau0.05"] == "1"]
            text = "all" if len(flagged) == 4 else ", ".join(flagged)
            lines.append(" & ".join([r["direction"].replace("noising", "noise"), r["layer"], r["head"]]
                                    + [f(float(r[f"effect_{m}"])) for m in ["IND", "MEAN", "TOK", "MIX8"]] + [text]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def cost_table():
    """Appendix table (post hoc): mean ablation, the token rule and k sampled donors against the individual average."""
    path = S4 / "results/posthoc/review-controls.json"
    if not path.exists():
        return None
    rc = read(path)
    names = {"sva-gpt2-small": "Agreement, GPT-2 Small", "sva-qwen3-0.6b": "Agreement, Qwen3-0.6B", "ioi-gpt2-small": "IOI, GPT-2 Small"}
    ks = ["1", "2", "4", "8", "16", "32"]
    lines = ["% Generated by paper/make_numbers.py from studies/04-reference-and-sweeps/results/posthoc/review-controls.json.",
             "\\begin{tabular}{@{}l" + "c" * (len(ks) + 2) + "@{}}", "\\toprule",
             " & Mean ablation & Token rule & \\multicolumn{" + str(len(ks)) + "}{c}{Sampled donors per input} \\\\",
             "\\cmidrule(l){4-" + str(len(ks) + 3) + "}",
             "Evaluations & 1 & 1--8 & " + " & ".join(ks) + " \\\\", "\\midrule"]
    for panel, label in names.items():
        p = rc["panels"][panel]
        cells = [f"{p['mean_ablation']['misclassified']} ({f(1e3 * p['mean_ablation']['mae'], 1)})"]
        tr = p.get("token_rule")
        cells.append(f"{tr['rule']['misclassified']} ({f(1e3 * tr['rule']['mae'], 1)})" if tr else "--")
        for k in ks:
            s = p["sampled_donors"][k]
            cells.append(f"{f(s['misclassified_mean'], 2)} ({f(1e3 * s['mae_mean'], 1)})")
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def reference_secondary_table():
    """Appendix table (prespecified secondary, Study 2 Experiment A): unrestricted top-1 agreement and the choice
    after averaging donor logits, for the circuit and the empty control."""
    path = S4 / "results/reference/analysis-test.json"
    if not path.exists():
        return None
    tabs = read(path)["tables"]
    lines = ["% Generated by paper/make_numbers.py from studies/04-reference-and-sweeps/results/reference/analysis-test.json.",
             "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
             " & & \\multicolumn{3}{c}{Circuit} & \\multicolumn{3}{c}{No heads} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}",
             "Prompts & Reference & Top-1, mean & Top-1, resample & Logit average & Top-1, mean & Top-1, resample & Logit average \\\\"]
    for stratum, label in [("error", "Model errors"), ("correct", "Correct cases")]:
        t = tabs[f"pooled/{stratum}"]
        lines.append("\\midrule")
        for i, c in enumerate(["N8", "N128", "NPO8", "NPO128"]):
            vals = [pct(t[f"{c}/{unit}/{m}"]["estimate"]) for unit in ["reference", "empty"]
                    for m in ["mean_top1_agree", "resample_top1_agree", "meanlogit_agree"]]
            lines.append((label if i == 0 else "") + f" & {c} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


S7 = PROJECT / "studies/07-larger-model"
LG_SETTING = {"sva-end": "Agr", "ioi-end": "IoiEnd", "ioi-s2": "IoiSTwo"}
LG_PANEL = {"sva-qwen3-1.7b": "Sva", "ioi-qwen3-1.7b": "Ioi"}


def sci(x, d=1):
    """x as a typeset power of ten, e.g. $2.2\\times 10^{-4}$."""
    mant, exp = f"{x:.{d}e}".split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def study7():
    """Study 5 (folder 07-larger-model): three prespecified tests in Qwen3-1.7B on Modal (sixth revision)."""
    import numpy as np
    n = {}
    R = S7 / "results"
    if not (R / "analysis/decisions-test.json").exists():
        return n
    dec = read(R / "analysis/decisions-test.json")
    a4 = read(R / "analysis/analysis-A-test.json")
    a3 = read(R / "analysis/analysis-heads-test.json")["settings"]
    a2 = read(R / "analysis/analysis-sweeps-test.json")["panels"]
    gates = read(R / "gates/gates.json")["gates"]
    cal = read(R / "calibration/calibration.json")["calibration"]
    ver = read(R / "verification.json")
    rep = read(R / "replay-verification.json")
    noise = read(R / "posthoc/replay-noise.json")
    bill = read(R / "billing-final.json")
    cost = read(R / "cost-decision.json")
    n["LgPriorStrings"] = f"{read(S7 / 'data/generation-audit.json')['prior_unique_strings']:,}"
    n["LgGroups"], n["LgIoiPrompts"] = str(cost["agreement_groups"]), str(cost["ioi_prompts"])
    # gates and checks
    n["LgGateClean"] = pct(gates["agreement"]["clean_accuracy"])
    n["LgGateCorrupt"] = pct(gates["agreement"]["corrupted_accuracy"])
    n["LgGateIoi"] = pct(gates["ioi"]["io_over_s_accuracy"], 0)
    n["LgCalMax"] = sci(cal["maximum_logit_error"])
    n["LgEngineMax"] = sci(cal["maximum_engine_difference"])
    n["LgEngineComparisons"] = str(len(cal["engine_agreement"]))
    n["LgCheckMax"] = sci(ver["largest_built_in_check_error"])
    n["LgTestRows"] = str(sum(ver["test_rows"].values()))
    primary = {k: v for k, v in rep["errors"].items() if not k.startswith("replay-l4")}
    over = [k for k, v in rep["errors"].items() if v > rep["tolerance"]]
    n["LgReplayMax"] = sci(rep["maximum_error_primary"])
    n["LgReplayLow"] = sci(min(rep["errors"][k] for k in over if not k.startswith("replay-l4")))
    n["LgReplayOver"] = str(len(over))
    n["LgReplayOverPrimary"] = str(len([k for k in over if not k.startswith("replay-l4")]))
    n["LgReplayAgrMax"] = sci(max(v for k, v in rep["errors"].items() if "/A/" in k or "/C1/" in k))
    n["LgReplayBound"] = sci(max(v["effect_bound"] for v in noise["panels"].values()))
    n["LgReplayClosest"] = sci(min(v["closest_distance_to_tau"] for v in noise["panels"].values()))
    n["LgCalProb"] = sci(cal["maximum_probability_error"])
    n["LgReplayLFour"] = sci(rep["maximum_error_l4"])
    n["LgSmokeMax"] = sci(read(S7 / "modal-smoke-test/records/comparison.json")["max_abs_margin_error"])
    n["LgCostTotal"] = f"{bill['total_usd']:.2f}"
    n["LgCostRun"] = f"{bill['total_by_app']['ai03-study5']:.2f}"
    n["LgGpuHours"] = f"{bill['by_app']['ai03-study5']['A10G'] / 1.10:.1f}"
    n["LgProjected"] = f"{cost['projections'][0]['remaining_usd']:.2f}"
    # R1, panel A
    for kind, kk in COMP.items():
        v = a4["kinds"][kind]
        m = v["methods"]
        p1 = v["p1_mean_misclassified"]
        n[f"LgSites{kk}"] = str(v["sites"])
        n[f"LgFlagInd{kk}"] = str(m["MEAN"]["tau0.05"]["flagged_ind"])
        n[f"LgMis{kk}"] = str(p1["estimate"])
        n[f"LgMisCI{kk}"] = "[{:.0f}, {:.0f}]".format(*p1["interval"])
        n[f"LgMeanErr{kk}"] = f(1e3 * m["MEAN"]["mae"], 2)
        for meth, mk in [("TOK", "Tok"), ("MIX8", "Mix")]:
            n[f"Lg{mk}Mis{kk}"] = str(m[meth]["tau0.05"]["misclassified"])
            n[f"Lg{mk}Err{kk}"] = f(1e3 * m[meth]["mae"], 2)
        n[f"LgSampleMis{kk}"] = f(m["S8"]["tau0.05"]["misclassified_mean"], 2)
        n[f"LgSampleErr{kk}"] = f(1e3 * m["S8"]["mae_mean"], 2)
        n[f"LgBias{kk}"] = f(v["mean_abs_bias"], 4)
        n[f"LgSpread{kk}"] = f"{v['spread']['median']:,.1f}" if v["spread"]["median"] < 100 else f"{v['spread']['median']:,.0f}"
    o = dec["R1"]["p3_ordering"]
    for pair, pk in [("resid-mlp", "ResMlp"), ("mlp-attn", "MlpAttn"), ("attn-head", "AttnHead")]:
        n[f"LgDiff{pk}"] = f(o[pair]["difference"], 4)
        n[f"LgDiffCI{pk}"] = ci(o[pair]["ci95"], 4)
    n["LgOrderShare"] = pct(o["bootstrap_share_full_order"], 0)
    n["LgROneVerdict"] = "replicates" if dec["R1"]["replicates"] else "does not replicate"
    n["LgBiasRatioRes"] = f(read(S6 / "results/analysis-test.json")["kinds"]["resid"]["mean_abs_bias"] / a4["kinds"]["resid"]["mean_abs_bias"], 1)
    # R2, head settings
    for s, sk in LG_SETTING.items():
        d, r = dec["R2"][s], a3[s]
        n[f"LgHSites{sk}"] = str(d["head_sites"])
        n[f"LgHFlag{sk}"] = str(r["methods"]["MEAN"]["tau0.05"]["flagged_ind"])
        n[f"LgHMis{sk}"] = str(d["estimate"])
        n[f"LgHMisCI{sk}"] = "[{:.0f}, {:.0f}]".format(*d["ci"])
        n[f"LgHBound{sk}"] = str(d["upper_bound_2pct"])
        n[f"LgHVerdict{sk}"] = d["verdict"]
        n[f"LgHTokMis{sk}"] = str(r["methods"]["TOK"]["tau0.05"]["misclassified"])
        n[f"LgHSampleMis{sk}"] = f(r["methods"]["S8"]["tau0.05"]["misclassified_mean"], 2)
        n[f"LgHTokEvals{sk}"] = f(r["evaluations_per_recipient_site"]["TOK"], 1)
        n[f"LgHTokErr{sk}"] = f(1e4 * r["tok_vs_s8"]["tok_mae"], 2)
        n[f"LgHSampleErr{sk}"] = f(1e4 * r["tok_vs_s8"]["s8_mae_mean"], 2)
        ind, mean, tok = (np.array(r["maps"][k]) for k in ("IND", "MEAN", "TOK"))
        changed = np.argwhere((np.abs(ind) >= 0.05) != (np.abs(mean) >= 0.05))
        for i, (dd, l, h) in enumerate(changed):
            tag = "ABCD"[i]
            n[f"LgHChanged{sk}{tag}"] = f"L{l}H{h}"
            n[f"LgHChangedInd{sk}{tag}"] = f(float(ind[dd, l, h]), 3)
            n[f"LgHChangedMean{sk}{tag}"] = f(float(mean[dd, l, h]), 3)
            n[f"LgHChangedTok{sk}{tag}"] = f(float(tok[dd, l, h]), 3)
    # R3, residual sweeps
    for panel, pk in LG_PANEL.items():
        d, r = dec["R3"][panel], a2[panel]
        fl = r["flags"]["mean64/tau0.05"]
        n[f"LgSw{pk}Sites"] = str(d["sites"])
        n[f"LgSw{pk}Mis"] = str(d["estimate"])
        n[f"LgSw{pk}MisCI"] = "[{:.0f}, {:.0f}]".format(*d["ci"])
        n[f"LgSw{pk}Rate"] = pct(d["rate"], 1)
        n[f"LgSw{pk}Fp"], n[f"LgSw{pk}Fn"] = str(fl["false_positive_sites"]["estimate"]), str(fl["false_negative_sites"]["estimate"])
        n[f"LgSw{pk}MisTen"] = str(r["flags"]["mean64/tau0.1"]["misclassified_sites"]["estimate"])
        n[f"LgSw{pk}Where"] = describe_sites(fl["false_positive_list"] + fl["false_negative_list"], r["positions"], panel.startswith("sva"))
        n[f"LgSw{pk}Slope"] = f(r["spread_law"]["loglog_slope"], 2)
    sva = a2["sva-qwen3-1.7b"]
    n["LgSwSvaNounMis"] = str(sva["flags"]["noun8/tau0.05"]["misclassified_sites"]["estimate"])
    n["LgSwSvaAdjMis"] = str(sva["flags"]["adj8/tau0.05"]["misclassified_sites"]["estimate"])
    g = sva["end_gap_by_partition"]
    n["LgEndGapMix"], n["LgEndGapAdj"], n["LgEndGapNoun"] = f(g["mix8"]), f(g["adj8"]), f(g["noun8"])
    ind = np.array(sva["maps"]["individual"])
    rule = np.array(sva["maps"]["mean64"]).copy()
    rule[..., 0] = np.array(sva["maps"]["adj8"])[..., 0]
    rule[..., -1] = np.array(sva["maps"]["noun8"])[..., -1]
    n["LgTokRuleMis"] = str(int(((np.abs(rule) >= 0.05) != (np.abs(ind) >= 0.05)).sum()))
    noun = np.array(sva["maps"]["noun8"])
    n["LgNounMisAdj"] = str(int(((np.abs(noun[..., 0]) >= 0.05) != (np.abs(ind[..., 0]) >= 0.05)).sum()))
    miss = np.argwhere((np.abs(rule) >= 0.05) != (np.abs(ind) >= 0.05))
    if len(miss) == 1:
        dd, l, q = miss[0]
        n["LgTokRuleWhere"] = SVA_POS.get(sva["positions"][q], sva["positions"][q])
        n["LgTokRuleBlock"] = str(l)
        n["LgTokRuleInd"], n["LgTokRuleMean"] = f(float(ind[dd, l, q]), 4), f(float(rule[dd, l, q]), 4)
    n["LgTokRuleErr"] = f(float(np.abs(rule - ind).mean()), 4)
    return n


def large_heads_table(stage="test"):
    """Appendix table (sixth revision): every Study 5 head-site that any method flags at tau = 0.05 (Qwen3-1.7B)."""
    import csv
    names = {"sva-end": "Agreement, final token", "ioi-end": "IOI, final token", "ioi-s2": "IOI, repeated subject"}
    folder = S7 / "results/analysis"
    if not all((folder / f"heads-{k}-{stage}.csv").exists() for k in names):
        return None
    lines = ["% Generated by paper/make_numbers.py from studies/07-larger-model/results/analysis/heads-*-" + stage + ".csv.",
             "\\begin{tabular}{@{}lrrrrrrl@{}}", "\\toprule",
             "Direction & Layer & Head & IND & Mean & TOK & MIX8 & Flagged by \\\\"]
    for setting, label in names.items():
        rows = list(csv.DictReader((folder / f"heads-{setting}-{stage}.csv").open()))
        hit = [r for r in rows if any(r[f"flag_{m}_tau0.05"] == "1" for m in ["IND", "MEAN", "TOK", "MIX8"])]
        lines.append(f"\\midrule\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{label}: {len(hit)} of {len(rows)} head-sites}}}} \\\\")
        for r in hit:
            flagged = [m if m != "MEAN" else "Mean" for m in ["IND", "MEAN", "TOK", "MIX8"] if r[f"flag_{m}_tau0.05"] == "1"]
            text = "all" if len(flagged) == 4 else ", ".join(flagged)
            lines.append(" & ".join([r["direction"].replace("noising", "noise"), r["layer"], r["head"]]
                                    + [f(float(r[f"effect_{m}"])) for m in ["IND", "MEAN", "TOK", "MIX8"]] + [text]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def render(n):
    lines = ["% Generated by paper/make_numbers.py from saved study outputs. Do not edit by hand."]
    for k in sorted(n):
        if not k.isalpha():
            raise ValueError(f"macro names must be letters only: {k}")
        lines.append(f"\\newcommand{{\\{k}}}{{{n[k]}}}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render(numbers())
    out = PAPER / "generated/numbers.tex"
    tables = {PAPER / "generated/misclassified-sites.tex": misclassified_table(), PAPER / "generated/heads-flagged.tex": heads_table(),
              PAPER / "generated/ioi-token-groups.tex": token_group_table(),
              PAPER / "generated/error-discrimination.tex": discrimination_table(),
              PAPER / "generated/components-flagged.tex": components_table(),
              PAPER / "generated/sampled-donors.tex": cost_table(),
              PAPER / "generated/reference-secondary.tex": reference_secondary_table(),
              PAPER / "generated/large-heads-flagged.tex": large_heads_table()}
    if args.check:
        if not out.exists() or out.read_text() != text:
            sys.exit("generated/numbers.tex is out of date; run make numbers")
        for tout, table in tables.items():
            if table is not None and (not tout.exists() or tout.read_text() != table):
                sys.exit(f"{tout.relative_to(PAPER)} is out of date; run make numbers")
        print("numbers.tex and generated tables match the saved outputs")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        for tout, table in tables.items():
            if table is not None:
                tout.write_text(table)
        print(f"wrote {len(text.splitlines()) - 1} macros")

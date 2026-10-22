# Study 2 results (folder `04-reference-and-sweeps`)

In the manuscript this is **Study 2**. Earlier drafts called it Study 4; the folder keeps its name because frozen records refer to it.

Test runs finished on 5 October 2026. All primary decisions follow the rules in `PROTOCOL.md`. The analysis changes made after the freeze are in `AMENDMENTS.md`; none changes a primary number. `verify.py` recomputes every primary estimate from the raw arrays with separate code (`results/verification.json`: passed). Fresh-process replays of the prespecified cases reproduced exactly (maximum difference 0).

## Experiment A: what the reference keeps

The screen covered 65,528 IOI prompts. Family O had 429 errors (pair accuracy 98.69%) and family W had 147 (99.55%). The selected set was 384 + 147 errors and 384 + 384 correct cases. Estimates are weighted to the screened population; brackets are 95% intervals unless marked.

| Error agreement (%) | Mean: circuit | Mean: no heads | Resample: circuit | Resample: no heads |
|---|---:|---:|---:|---:|
| N8 | 72.0 | 77.1 | 81.6 | 71.1 |
| N128 | 71.7 | 75.4 | 82.0 | 70.7 |
| NPO8 | 22.5 | 65.2 | 47.4 | 60.0 |
| NPO128 | 21.7 | 63.9 | 47.1 | 60.3 |

- **A1 (primary).** NPO128 minus N8 under mean ablation is −50.2 points [−55.3, −45.3] (97.5% interval). **Supported.**
- **A2 (primary).** Circuit minus no heads, N8, mean ablation, is −5.1 points [−10.4, 0.3] (97.5% interval). The upper limit is below 10, so by the protocol's rule the circuit matches errors **little better than no circuit**.
- **Main effects.** Content −49.7 [−53.8, −45.6]; size −0.5 [−2.1, 1.0] (mean ablation).
- **Correct cases.** The circuit agrees 89–98% of the time in every cell; the no-heads control 47–49%.
- **Faithfulness (mean / resample).** N8 0.774 / 0.572; NPO128 0.913 / 0.566.
- **Comparison with Zhang et al. (2026).** They report 15.2% error agreement, 99.5% correct-case agreement and faithfulness 0.956 under mean ablation, and 40.6%, 91.2% and 0.644 under resampling. NPO128 gives 21.7%, 98.2%, 0.913 and 47.1%, 88.6%, 0.566.

## Experiment B: patching sweeps

| Panel | Sites | Misclassified, mean (98.33% interval) | False positives | False negatives | NOUN8 misclassified |
|---|---:|---:|---:|---:|---:|
| Agreement, GPT-2 Small (64 groups) | 120 | 6 [2, 9] | 6 | 0 | 0 |
| Agreement, Qwen3-0.6B (32 groups) | 280 | 27 [15, 38] | 26 | 1 | 0 |
| IOI, GPT-2 Small (64 prompts) | 168 | 6 [1, 10] | 5 | 1 | — |

- **B1–B3 (primary).** All three lower limits are at least 1. **Supported.** Without the two IOI prompts that the development run also used, the IOI count is 5.
- **Agreement panels.** All false positives are at the final token, where the donors' final nouns differ:
  - GPT-2: destruction direction, blocks 0–5. Individual effects are at most 1.1% of the clean-minus-corrupted margin; mean ablation reports 5.1–10.3%, with the opposite sign.
  - Qwen: individual effects at most 3.8%; mean ablation reports 5.5–22.6%.
- **IOI panel.**
  - False positives at the object word (blocks 0–2) and the final token (blocks 2, 5).
  - One false negative at the final token in block 6, where mean ablation reverses a 5.3% effect.
- **Bank partitions at the final token.** Same-adjective banks have almost the same gap as fully mixed banks, while same-noun banks have 2–6% of it:
  - GPT-2: 0.334, 0.340 and 0.008 logits;
  - Qwen: 0.451, 0.459 and 0.027 logits.
- **Spread law.** Bank-level log–log slopes are 0.98 (GPT-2) and 1.00 (Qwen) at the final token.

## Exploratory: second-order prediction (post hoc)

Run after the freeze on 16 GPT-2 groups (`curvature.py`, `results/curvature.json`), covering 2,304 banks. For same-noun banks, −½ tr(HΣ) predicts the measured gap with R² 0.997 and slope 1.00. For mixed-noun banks the correlation is at least 0.93, but R² against the identity line is 0.43 and the prediction is 42% too large in mean absolute value. Signs agree in 97% of banks.

## Post hoc analyses added in revision (5 October 2026)

All of these were specified after the results above were known and only read the saved outputs (`posthoc/sweep_posthoc.py` -> `results/posthoc/sweep-posthoc.json`; `posthoc/random_circuits.py` -> `results/posthoc/random-circuits.json`).

**Task competence.** Qwen3-0.6B prefers the grammatical verb on 27 of 32 clean sweep groups (84.4%). Restricted to those 27 groups, mean ablation misclassifies 28 sites (27 with all groups). GPT-2 Small is correct on all 64 agreement groups (count unchanged at 6) and on 63 of 64 IOI prompts (count 5 without the remaining prompt).

**Equal budget** (misclassified sites at tau = 0.05; mean absolute error over sites a patch can change; evaluations per recipient-site in brackets; S8 = eight donors sampled from the 64 individual patches, 1,000 draws, mean and 95% range):

| Panel | Centroid (1) | MIX8 (8) | NOUN8 (8) | ADJ8 (8) | S8 (8) |
|---|---|---|---|---|---|
| Agreement, GPT-2 | 6 (0.0111) | 6 (0.0111) | 0 (0.0026) | 6 (0.0085) | 0.28, 0-1 (0.0011) |
| Agreement, Qwen | 27 (0.0166) | 27 (0.0165) | 0 (0.0041) | 28 (0.0127) | 0.21, 0-2 (0.0029) |
| IOI, GPT-2 | 6 (0.0366) | 6 (0.0299) | - | - | 0.47, 0-2 (0.0029) |

Eight sampled donors do about as well as NOUN8 (0.28 and 0.21 misclassified sites on average, against none) and have a smaller error than NOUN8 banks used at every position. The token rule as defined uses NOUN8 banks only at the final token, ADJ8 banks at the adjective and one group (mean ablation) at the head noun and the two following words; it also misclassifies no site, with errors of 0.0012 and 0.0026 (see the fifth-revision controls below).

**Token groups in the IOI sweep.** Grouping donors by the token at the patched position gives one group of 64 (identical to mean ablation) at 9 of 14 positions, including the final "to"; about 39, 38 and 39 groups (mean size 1.6-1.7) at the two names and the repeated subject; about 15 and 19 groups at the place and object words.

**Threshold robustness.** Misclassified counts at tau = 0.02 / 0.05 / 0.10 / 0.20: GPT-2 agreement 11 / 6 / 2 / 0 (nonzero up to 0.12); Qwen 33 / 27 / 15 / 2; IOI 6 / 6 / 7 / 1. Top-k overlap with k = sites flagged by the individual average: 42 of 44, 77 of 94, 43 of 47. Median distance of misclassified sites from tau: 0.019, 0.043, 0.056 under mean ablation; 0.046, 0.035, 0.023 under the individual average. Every site is listed in the manuscript's appendix table.

**Cancellation.** At misclassified sites the mean absolute single-donor effect is 0.111, 0.199 and 0.171 (normalized), against an absolute average of 0.005, 0.020 and 0.027; 49%, 46% and 45% of donors oppose the sign of the average. At other weak sites: 0.029, 0.049, 0.025 and 33%, 41%, 43%. The individual average hides large two-sided effects at these sites, so the manuscript calls them misclassified sites (a disagreement with the individual average) rather than false positives.

**Donor spread.** Median within-layer percentile of the 64-donor spread: 60, 70 and 69 at misclassified sites, against 40, 40 and 46 at other sites a patch can change.

**Random-circuit control** (`posthoc/random_circuits.py`, run after the Experiment A results were known). Five random circuits, each with the same number of heads as the Wang et al. circuit in every (layer, position role) cell and none of its heads, evaluated on the same 1,299 recipients, saved donor banks, references and ablations. Agreement in percent, weighted to the screened population, as circuit / random mean (lowest-highest circuit) / no heads:

| Stratum, ablation | N8 | N128 | NPO8 | NPO128 |
|---|---|---|---|---|
| Errors, mean | 72.0 / 85.5 (77.4-93.0) / 77.1 | 71.7 / 85.7 (76.9-94.8) / 75.4 | 22.5 / 79.7 (65.9-91.9) / 65.2 | 21.7 / 81.3 (66.2-94.0) / 63.9 |
| Errors, resample | 81.6 / 79.7 (72.1-85.9) / 71.1 | 82.0 / 79.8 (72.0-85.9) / 70.7 | 47.4 / 71.9 (61.6-79.6) / 60.0 | 47.1 / 72.6 (62.5-79.9) / 60.3 |
| Correct cases, mean | 95.7 / 31.2 (23.3-43.9) / 46.7 | 95.8 / 29.3 (20.8-42.6) / 47.9 | 97.8 / 29.5 (21.1-43.5) / 47.5 | 98.2 / 28.0 (18.5-41.7) / 47.1 |
| Correct cases, resample | 89.6 / 36.3 (30.5-45.3) / 48.5 | 89.5 / 36.6 (30.6-46.2) / 49.2 | 88.6 / 36.1 (30.1-45.5) / 48.6 | 88.6 / 36.3 (30.3-45.9) / 48.8 |

Random circuits match more errors than both the IOI circuit and the empty control under mean ablation with every reference, and agree with the model on fewer correct cases than the empty control: they mostly choose the subject. Error agreement on its own does not separate the IOI circuit from random heads; correct-case agreement does. `posthoc/verify_posthoc.py` recomputes every value above from the raw margins.

## Second-revision post hoc analyses (6 October 2026)

**Error discrimination** (`posthoc/error_discrimination.py`). P(subject | error) / P(subject | correct), odds ratio and AUROC of the ablated IO-S margin for separating model-error from model-correct prompts, population-weighted, 2,000 stratified bootstrap draws; within-template AUROC in parentheses. Random circuits: mean over five.

| Ablation | Reference | IOI circuit P(S) err / corr | IOI circuit OR [95%] | IOI circuit AUROC [95%] (within) | Random OR / AUROC | No heads OR / AUROC (within) |
|---|---|---|---|---|---|---|
| mean | N8 | 0.720 / 0.043 | 57.3 [39.9, 88.3] | 0.964 [0.955, 0.972] (0.942) | 2.86 / 0.658 | 2.95 / 0.688 (0.688) |
| mean | N128 | 0.717 / 0.042 | 58.3 [40.2, 91.5] | 0.965 [0.957, 0.974] (0.942) | 2.85 / 0.668 | 2.82 / 0.700 (0.694) |
| mean | NPO8 | 0.225 / 0.022 | 12.9 [8.0, 24.2] | 0.823 [0.797, 0.846] (0.757) | 1.85 / 0.593 | 1.69 / 0.593 (0.587) |
| mean | NPO128 | 0.217 / 0.018 | 15.0 [9.1, 31.1] | 0.824 [0.798, 0.847] (0.754) | 1.99 / 0.608 | 1.58 / 0.610 (0.602) |
| resample | N8 | 0.816 / 0.104 | 38.3 [31.0, 47.8] | 0.970 [0.962, 0.978] (0.948) | 2.28 / 0.658 | 2.32 / 0.680 (0.677) |
| resample | N128 | 0.820 / 0.105 | 39.0 [32.2, 47.8] | 0.974 [0.967, 0.981] (0.951) | 2.32 / 0.666 | 2.34 / 0.688 (0.681) |
| resample | NPO8 | 0.474 / 0.114 | 7.0 [5.9, 8.4] | 0.845 [0.822, 0.866] (0.774) | 1.47 / 0.587 | 1.42 / 0.587 (0.581) |
| resample | NPO128 | 0.471 / 0.114 | 6.9 [5.9, 8.2] | 0.853 [0.830, 0.874] (0.778) | 1.53 / 0.600 | 1.45 / 0.600 (0.591) |

The circuit separates errors from correct prompts far better than the controls in every cell, although its raw error agreement is below theirs in every cell under mean ablation (under resampling it is above both only with names-only donors). Caveat: error prompts differ from correct prompts in template and names; the within-template AUROC removes the template difference only. The fifth-revision controls below add a surface-feature baseline and the AUROC against correct prompts near the decision boundary.

**Bias against sampling noise** (`posthoc/bias_noise.py`). Median centroid bias and eight-donor SD at sites near the threshold (individual average or centroid between 0.025 and 0.10):

- study2 sva-gpt2-small: 21 sites; mean ablation bias 0.0153 vs SD 0.0036 (bias above SD at 76%); token grouping bias 0.0003.
- study2 sva-qwen3-0.6b: 62 sites; mean ablation bias 0.0099 vs SD 0.0026 (bias above SD at 82%); token grouping bias 0.0023.
- study2 ioi-gpt2-small: 17 sites; mean ablation bias 0.0259 vs SD 0.0026 (bias above SD at 100%).
- study3 sva-end-gpt2-small: 6 sites; mean ablation bias 0.0002 vs SD 0.0006 (bias above SD at 17%); token grouping bias 0.0001.
- study3 ioi-end-gpt2-small: 10 sites; mean ablation bias 0.0016 vs SD 0.0015 (bias above SD at 40%).
- study3 ioi-s2-gpt2-small: 5 sites; mean ablation bias 0.0030 vs SD 0.0013 (bias above SD at 100%); token grouping bias 0.0002.

At the final token alone, NOUN8's mean absolute error is 0.0009 (GPT-2) and 0.0030 (Qwen), against 0.0033 and 0.0078 for eight sampled donors.

## Fifth-revision post hoc controls (13 October 2026)

Specified after every result above was known, for the fifth revision of the manuscript; they only read saved outputs (`posthoc/review_controls.py` -> `results/posthoc/review-controls.json`). `posthoc/verify_review_controls.py` recomputes every value with separate code that imports nothing from this study (98 checks, `results/posthoc/verification-review-controls.json`); `make check-paper` confirms that the verified file is the one the manuscript quotes.

- **Sampled donors at increasing cost.** k donors drawn per recipient and direction, the same k at every site, 1,000 draws, against the 64-donor individual average (misclassified sites at tau = 0.05, mean over draws; mean absolute error in brackets). Mean ablation (one evaluation): 6 (0.0111), 27 (0.0166), 6 (0.0366). One sampled donor (one evaluation): 0.53 (0.0034), 6.24 (0.0088), 1.96 (0.0085). Every doubling of k up to 32 lowers the error in all three panels.
- **Stability of the reference.** Two random halves of 32 donors per group disagree with each other at 0.13, 0.00 and 0.17 sites on average, at most 1 in 2,000 splits.
- **The token rule as defined** (agreement panels): 0 misclassified sites in both models, errors 0.0012 and 0.0026.
- **IOI sweep count.** Under the original definition (sites flagged by mean ablation only) 5 [1, 9]; without the two test prompts used in development 5 [1, 10] (original definition 5 [1, 9]); without the four calibration prompts 6 [1, 10] (98.33% intervals, 20,000 stratified draws, frozen seed).
- **Study 1 spread slopes with K fixed.** Within the eight rotations of one grouping policy 0.87 [0.69, 1.05] (GPT-2) and 1.14 [0.98, 1.29] (Qwen); across K = 2, 4 and 8 only 0.90 [0.77, 1.02] and 0.80 [0.67, 0.92] (4,000-draw recipient bootstrap of `retrospective_study3.py`).
- **Error separation near the decision boundary** (Experiment A recipients). With correct prompts restricted to the tenth with the smallest intact margins (below 1.5 logits), the circuit's AUROC is 0.83-0.87 (names-only references) and 0.62-0.66 (names-place-object), against 0.54-0.63 for the empty control; under NPO128 and mean ablation 0.63 [0.56, 0.69] against 0.56 [0.48, 0.63]. On error prompts the circuit's ablated margin has weighted rank correlations of 0.35-0.37 (names-only) and -0.06 to -0.01 (names-place-object) with the intact margin.
- **Surface-feature baseline.** An additive least-squares model of the intact margin on template, name order, the two names, place and object, fitted on the 64,229 screened prompts that were not selected as recipients, separates the recipients' error prompts from their correct ones with AUROC 0.93 [0.91, 0.94] (0.87 within templates; the two names alone 0.76, template and name order alone 0.76).

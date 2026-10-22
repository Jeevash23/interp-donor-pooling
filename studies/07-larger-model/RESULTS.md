# Study 5 results (folder `07-larger-model`)

Qwen3-1.7B (revision `70d244cc`), float32, on a Modal A10G. Protocol and predictions hashed at 2026-10-17T05:43:12 UTC; code and prompts frozen at 2026-10-17T08:18:49 UTC (three amendments followed: a GPU-memory fix before any development output, stages moved inside Modal after the local connection dropped, and a wider stage-4 freeze; `AMENDMENTS.md`); calibration, development and the cost decision frozen at 2026-10-17T09:59:53 UTC, before the gates and any test output. Test runs finished at 2026-10-17T11:04:22 UTC. Every outcome below follows `PROTOCOL.md`. This file is written by `write_results.py` from the saved records in `results/`.

## Checks

- Calibration (reference engine, calibration prompts): passed; largest logit error 2.2e-04, largest probability error 5.0e-06, exact write locality: True. Study engine against reference engine: largest difference 6.5e-04 logits over 106 comparisons (tolerance 1e-3).
- Built-in checks in every test row (identity patches, cached passive run, write locality): largest error 3.8e-04; failing records: 0.
- Replays: fresh-container study engine and reference engine, largest margin difference 1.6e-03 (passed: False); on an L4 (secondary) 1.2e-03.
- Independent verification (`verify.py`): passed: False. Frozen files changed: 0. Test prompts overlapping calibration or development: 0; overlapping earlier studies: 0 (of 532,560 earlier strings).
- Cost rule: development projected $6.32 for the rest of the study (rule $7); 64 agreement groups and 64 IOI prompts were used.

### The replay check failed (protocol stage 7)

The protocol requires every primary replay to match the test outputs within 0.001 logits, and a failed check stops the study for the author's review. Four replays exceeded it, all in the IOI panels: replay-study/B1/ioi-test-ioi-00001 1.57e-03, replay-l4/C2/ioi-test-ioi-00000 1.19e-03, replay-reference/C2/ioi-test-ioi-00000 1.15e-03, replay-study/C2/ioi-test-ioi-00001 1.11e-03 (the L4 replay is secondary). Every agreement replay was within 1e-4; the study engine's fresh-container agreement replays were exact. The IOI computations therefore carry float32 differences of about 1e-3 logits between containers, consistent with the largest study-against-reference difference in calibration (6.5e-04, also in the IOI sweep).

Post hoc (`posthoc/replay_noise.py`, written after the failure): dividing each panel's largest replay difference by its denominator bounds how far any normalised effect could move if every row carried that much noise. The bound is at most 1.8e-04; no IND or MEAN effect lies that close to τ (closest 3.2e-04, in C2). So noise of the replayed size cannot change any flag or primary count (any flag could change: False). The author accepted the results with the replay failure and this bound disclosed (`NOTES-AFTER-TEST.md`), and the manuscript's sixth revision reports them.

Order of stages: the analysis (stage 8) was started while the replays (stage 7) were still running, so the outcomes below were seen before the replay result. The analysis is frozen code and no choice depended on it.

## Competence gates

- Agreement: clean recipients 90.6%, corrupted recipients 98.4% (both required at least 90%): **passed**.
- IOI: IO above S on 100.0% of prompts (required 90%): **passed**.

## R1: bias follows component size (panel A)

Sites are site-directions at the agreement final token. Misclassified sites at τ = 0.05 against the average of 64 individual patches (IND); mean absolute error in normalised effect, in units of 1e-3, in brackets. S8: mean over 1,000 draws.

| Kind | Sites | Flagged by IND | MEAN (98.75% interval) | TOK | MIX8 | S8 | Mean abs. bias | Median spread |
|---|---:|---:|---|---|---|---|---:|---:|
| Residual stream | 56 | 19 | 3 [2, 24] (29.17) | 0 (2.06) | 3 (29.17) | 0.33 (5.13) | 0.0292 | 8,576.9 |
| MLP output | 56 | 10 | 2 [0, 6] (6.99) | 0 (0.81) | 2 (6.92) | 0.35 (2.13) | 0.0070 | 1,372.0 |
| Attention-layer output | 56 | 8 | 0 [0, 2] (1.78) | 0 (0.44) | 0 (1.76) | 0.00 (1.17) | 0.0018 | 267.9 |
| Single head | 896 | 8 | 0 [0, 0] (0.16) | 0 (0.04) | 0 (0.16) | 0.00 (0.20) | 0.0002 | 8.5 |

- P3, resid-mlp: difference 0.0222, 95% interval [0.0186, 0.0269].
- P3, mlp-attn: difference 0.0052, 95% interval [0.0042, 0.0065].
- P3, attn-head: difference 0.0016, 95% interval [0.0014, 0.0024].
- P3 (all three intervals above zero): **supported**; full order in 100% of bootstrap draws.
- P4: kinds by spread ['resid', 'mlp', 'attn', 'head'], by bias ['resid', 'mlp', 'attn', 'head']: **supported**.
- **R1 replicates.**

Secondary (Study 4's P2): Residual stream TOK better; MLP output TOK better; Attention-layer output TOK better; Single head TOK better.

## R2: single-head flags

Misclassified head-sites under MEAN at τ = 0.05 against IND, with 98.33% intervals. Replicates if the lower limit is 0 and the upper limit is at most 2% of head-sites; contradicts Study 3 if the lower limit is at least 1.

| Setting | Head-sites | Flagged by IND | MEAN (98.33% interval) | 2% bound | TOK | S8 | Verdict |
|---|---:|---:|---|---:|---|---|---|
| Agreement, final token (panel A) | 896 | 8 | 0 [0, 0] | 17 | 0 | 0.00 | **replicates** |
| IOI, final token (B1) | 448 | 13 | 1 [0, 5] | 8 | 1 | 0.27 | **replicates** |
| IOI, repeated subject (B2) | 448 | 4 | 2 [1, 4] | 8 | 0 | 0.01 | **contradicts** |

- Changed flag, IOI, final token (B1): layer 21, head 1: IND -0.0383, MEAN -0.0575, TOK -0.0575.
- Changed flag, IOI, repeated subject (B2): layer 0, head 3: IND 0.0740, MEAN 0.0166, TOK 0.0740.
- Changed flag, IOI, repeated subject (B2): layer 0, head 12: IND 0.2211, MEAN 0.0482, TOK 0.2211.
- Agreement, final token (panel A): TOK error 0.37 against S8 2.03 (2.5th percentile 1.80; units of 1e-4); TOK better by Study 3's H2 rule: True; TOK identical to MEAN: False; TOK evaluations per head-site 8.0.
- IOI, final token (B1): TOK error 9.50 against S8 8.24 (2.5th percentile 7.33; units of 1e-4); TOK better by Study 3's H2 rule: False; TOK identical to MEAN: True; TOK evaluations per head-site 1.0.
- IOI, repeated subject (B2): TOK error 0.49 against S8 2.96 (2.5th percentile 2.56; units of 1e-4); TOK better by Study 3's H2 rule: True; TOK identical to MEAN: False; TOK evaluations per head-site 38.6.

## R3: residual sweeps

Sites whose flag under MEAN64 differs from IND at τ = 0.05, with 97.5% intervals. Replicates if the lower limit is at least 1.

| Panel | Sites | Misclassified (97.5% interval) | False positives | False negatives | Rate | MIX8 | NOUN8 | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Agreement (C1) | 280 | 7 [5, 29] | 7 | 0 | 2.5% | 7 | 4 | **replicates** |
| IOI (C2) | 392 | 30 [19, 35] | 26 | 4 | 7.7% | 30 | – | **replicates** |

Earlier rates (Study 2): GPT-2 Small agreement 5.0%, Qwen3-0.6B agreement 9.6%, GPT-2 Small IOI 3.6%.

- Token rule as defined (agreement sweep; secondary): 1 misclassified site, mean absolute error 0.0026.
- Final-token centroid gap by bank partition (logits): adj8 0.415, mix8 0.421, noun8 0.031.
- Agreement (C1): bank-level spread slope 0.47 (within-cell R² 0.45).
- IOI (C2): bank-level spread slope 1.02 (within-cell R² 0.03).

## Predictions against outcomes

The designer's predictions (`PREDICTIONS.md`, hashed with the protocol) are not decision rules.

| Prediction | Confidence | Outcome |
|---|---|---|
| Agreement gate passes | about 55% | held (90.6% / 98.4%) |
| IOI gate passes | about 80% | held (100.0%) |
| P3: resid > mlp > attn > head, all intervals above zero | about 75% | held |
| P4: spread in the same order | about 75% | held |
| Mean absolute residual bias about 0.04–0.08 | – | 0.0292 |
| Token grouping beats eight sampled donors for all four kinds | about 75% | held |
| R2 replicates: Agreement, final token (panel A) | about 85% | replicates |
| R2 replicates: IOI, final token (B1) | about 80% | replicates |
| R2 replicates: IOI, repeated subject (B2) | about 85% | contradicts |
| R2 replicates in all three settings | about 65% | did not hold |
| R3 replicates in the agreement sweep, with 10–35 misclassified sites | about 85% | held; 7 sites |
| R3 replicates in the IOI sweep | about 60% | held |
| Token rule as defined misclassifies no agreement-sweep site | about 70% | did not hold (1) |
| Total spend $3.50–6.50 | – | $6.67 (all ai03-study5 apps, including the smoke test) |

## Cost and data

- Modal billing report, all `ai03-study5` apps (including the smoke test): $6.67 at 2026-10-17T11:45 UTC; stop at $10.
- Stage starts (spend before each, by the billing report; some development starts were interrupted, see AMENDMENTS.md): calibrate 08:20 UTC ($0.28); development 08:33 UTC ($0.47); development 09:25 UTC ($0.48); development 09:46 UTC ($0.63); test 10:00 UTC ($0.86); replays 11:25 UTC ($5.92); analyse 11:26 UTC ($5.92).
- Final billing report: $6.67 for Study 5, of which $6.39 for the study run itself (A10G $5.31, L4 $0.08, CPU $0.50, memory $0.50); $0.00 billed after credits.
- Data: the records pulled back to `results/` total 2.7 MB; uploads were code and compact prompt rows. Raw arrays stay on the Modal Volume `ai03-study5`; every array's SHA-256 is in its run record under `results/`.


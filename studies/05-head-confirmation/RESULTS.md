# Study 3 results (folder `05-head-confirmation`)

Test runs finished at 2026-10-05T19:17:54Z, after the freeze at 2026-10-05T17:20:49Z. Every outcome below follows `PROTOCOL.md`; nothing was changed after test outputs. This file is written by `write_results.py` from `results/analysis-test.json`, `results/verification.json` and the run records.

## Checks

- Calibrations: 4, all passed; largest logit error 1.6e-04, largest probability error 7.6e-06.
- Fresh-process replays of the first two test rows per setting: ioi-end-gpt2-small max 0.0e+00 (passed), ioi-s2-gpt2-small max 0.0e+00 (passed), sva-end-gpt2-small max 0.0e+00 (passed).
- Qwen3-0.6B gate: clean two-verb accuracy 79.7% on 64 fresh groups, required 90%: not passed, so Qwen was excluded before any patching.
- Independent verification (`verify.py`): passed. Frozen files changed: 0. Test prompts overlapping calibration/development: 0; overlapping earlier studies: 0 (of 509,688 earlier strings).

## Outcomes

Misclassified head-sites at tau = 0.05 against the average of 64 individual patches (IND); mean absolute error in normalized effect in units of 1e-4 in brackets. S8: mean over 1,000 draws (95% range).

| Setting | Head-sites | Flagged by IND | MEAN (98.33% interval) | TOK (evaluations) | MIX8 | S8 |
|---|---:|---:|---|---|---|---|
| Agreement, final token | 288 | 9 | 0 [0, 1] (0.83) | 0 (0.20; 8) | 0 (0.82) | 0.46, 0-1 (1.26; 1.00-1.58) |
| IOI, final token | 144 | 11 | 0 [0, 2] (7.37) | 0 (7.37; 1) | 0 (6.55) | 0.00, 0-0 (5.06; 3.87-6.79) |
| IOI, repeated subject | 144 | 6 | 0 [0, 1] (3.64) | 0 (0.52; 38) | 0 (3.24) | 0.00, 0-0 (1.66; 1.33-2.07) |

**Decisions.**

- H1, Agreement, final token: lower limit 0, so the prediction that averaging changes the flagged heads is **not supported**.
- H1, IOI, final token: lower limit 0, so the prediction that averaging changes the flagged heads is **not supported**.
- H1, IOI, repeated subject: lower limit 0, so the prediction that averaging changes the flagged heads is **not supported**.
- H2, agreement: TOK error 0.20 against the S8 2.5th percentile 1.00; TOK misclassified 0 against the S8 mean 0.46: token grouping **beats** eight sampled donors. Bootstrap interval for TOK error minus the S8 expectation: [-1.79, -1.33].
- H3, IOI final token: one token group, so TOK equals MEAN (confirmed); it has no advantage over mean ablation. MEAN error 7.37 against S8 5.06 (2.5th percentile 3.87).
- H4, IOI repeated subject (descriptive): TOK costs 38.3 evaluations per head-site on average, misclassifies 0 head-sites, error 0.52, against MEAN 3.64 and S8 1.66.

**Secondary thresholds** (misclassified by MEAN / TOK / MIX8 / S8 mean):

- Agreement, final token, tau = 0.02: 0 / 0 / 0 / 0.00 (IND flags 15)
- Agreement, final token, tau = 0.1: 0 / 0 / 0 / 0.00 (IND flags 6)
- IOI, final token, tau = 0.02: 2 / 2 / 2 / 1.47 (IND flags 21)
- IOI, final token, tau = 0.1: 0 / 0 / 0 / 0.10 (IND flags 7)
- IOI, repeated subject, tau = 0.02: 0 / 0 / 0 / 0.27 (IND flags 9)
- IOI, repeated subject, tau = 0.1: 0 / 0 / 0 / 0.03 (IND flags 3)

Every head-site, with its effect under IND, MEAN, TOK and MIX8 and its flags, is in `results/heads-*-test.csv`. The manuscript's Section 7 reports these results; its appendix lists every head-site that any method flags.


## Post hoc: head-level donor spread (added in the second revision)

`posthoc/head_spread.py` reran the unpatched test prompts (no frozen file changed) and measured each head-site's donor spread in the residual space (through W_O) and the residual stream's spread after the same block at the same position. Results (`results/posthoc/head-spread.json`):

- ioi-end: median head spread 2.42, median residual spread 398, median ratio 0.0117 (max 0.49); site-level log-log slope of |gap| on head spread 0.62.
- ioi-s2: median head spread 1.28, median residual spread 2,294, median ratio 0.0006 (max 0.11); site-level log-log slope of |gap| on head spread 0.61.
- sva-end: median head spread 0.93, median residual spread 1,817, median ratio 0.0006 (max 0.19); site-level log-log slope of |gap| on head spread 0.66.

Extending the gap-spread relation fitted to Study 1's residual banks down to these spreads predicts the head-level gaps within a factor of ten at most head-sites (manuscript Section 7).

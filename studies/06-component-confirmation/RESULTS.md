# Study 4 results (folder `06-component-confirmation`)

Test runs finished at 2026-10-06T03:38:33Z, after the freeze at 2026-10-06T02:58:16Z. Every outcome below follows `PROTOCOL.md`; nothing was changed after test outputs. This file is written by `write_results.py` from `results/analysis-test.json`, `results/verification.json` and the run records.

## Checks

- Calibration: passed; largest logit error 9.9e-05, largest probability error 1.8e-06, exact write locality: True.
- Fresh-process replays of the first two test groups: maximum difference 0.0e+00 logits (passed).
- Independent verification (`verify.py`): passed. Frozen files changed: 0. Test prompts overlapping calibration/development: 0; overlapping earlier studies: 0 (of 523,452 earlier strings).

## Outcomes

Sites are site-directions at the final token. Misclassified sites at tau = 0.05 against the average of 64 individual patches (IND); mean absolute error in normalized effect, in units of 1e-3, in brackets. S8: mean over 1,000 draws (95% range).

| Kind | Sites | Flagged by IND | MEAN (98.75% interval) | TOK | MIX8 | S8 | Mean abs. bias | Median spread |
|---|---:|---:|---|---|---|---|---:|---:|
| Residual stream | 24 | 12 | 9 [5, 10] (42.32) | 0 (0.88) | 9 (42.29) | 0.01, 0-0 (3.47) | 0.0423 | 1,869.3 |
| MLP output | 24 | 7 | 1 [0, 3] (6.75) | 0 (0.18) | 1 (6.74) | 0.00, 0-0 (1.62) | 0.0068 | 166.0 |
| Attention-layer output | 24 | 8 | 0 [0, 1] (1.11) | 0 (0.23) | 0 (1.10) | 0.00, 0-0 (0.80) | 0.0011 | 15.9 |
| Single head | 288 | 10 | 0 [0, 1] (0.09) | 0 (0.02) | 0 (0.09) | 0.04, 0-1 (0.14) | 0.0001 | 1.0 |

**Decisions.**

- P1, Residual stream: predicted present; estimate 9, interval [5, 10]: **supported**.
- P1, MLP output: predicted present; estimate 1, interval [0, 3]: **not supported**.
- P1, Attention-layer output: predicted absent; estimate 0, interval [0, 1]: **supported**.
- P1, Single head: predicted absent; estimate 0, interval [0, 1]: **supported**.
- P2, Residual stream: TOK error 0.88 against S8 2.5th-97.5th percentiles 1.44-6.34 (x1e-3): TOK better; predicted TOK better: **supported**.
- P2, MLP output: TOK error 0.18 against S8 2.5th-97.5th percentiles 1.08-2.35 (x1e-3): TOK better; predicted TOK better: **supported**.
- P2, Attention-layer output: TOK error 0.23 against S8 2.5th-97.5th percentiles 0.49-1.25 (x1e-3): TOK better; predicted TOK better: **supported**.
- P2, Single head: TOK error 0.02 against S8 2.5th-97.5th percentiles 0.11-0.18 (x1e-3): TOK better; predicted TOK better: **supported**.
- P3, ordering resid > mlp > attn > head of mean absolute bias: resid-mlp 0.0356 [0.0328, 0.0387]; mlp-attn 0.0056 [0.0050, 0.0063]; attn-head 0.0010 [0.0009, 0.0011]: **supported** (full order in 100% of bootstrap draws).
- P4, spread: kinds by spread ['resid', 'mlp', 'attn', 'head'], by bias ['resid', 'mlp', 'attn', 'head']: **supported**.

**Bias against sampling noise at sites near the threshold** (IND between 0.025 and 0.10):

- Residual stream: 3 sites; median centroid bias 0.0176, median eight-donor SD 0.0046; bias above SD at 100%.
- MLP output: 6 sites; median centroid bias 0.0068, median eight-donor SD 0.0023; bias above SD at 100%.
- Attention-layer output: 5 sites; median centroid bias 0.0012, median eight-donor SD 0.0010; bias above SD at 80%.
- Single head: 6 sites; median centroid bias 0.0002, median eight-donor SD 0.0006; bias above SD at 17%.

Every site, with its effect under IND, MEAN, TOK and MIX8, its flags, spread and eight-donor SD, is in `results/sites-test.csv`. The manuscript's Section 8 reports these results.

Note: the protocol's development section quotes 0.0032 for the final-token error of eight sampled donors in Study 2, from a 300-draw check run while writing it; the saved 1,000-draw value (Study 2 `results/posthoc/bias-noise.json`) is 0.0033.

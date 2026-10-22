# Study 3 (folder `05-head-confirmation`): does donor averaging change which heads an analysis flags?

In the manuscript this is **Study 3**, the prespecified confirmation study (Section 7). The earlier studies are P1 (`../01-pilot`), P2 (`../../submission-study`), Study 1 (`../03-grouping-and-circuit`) and Study 2 (`../04-reference-and-sweeps`).

Run on 5 October 2026 on GPT-2 Small. The protocol ([PROTOCOL.md](PROTOCOL.md)), code, generated prompts, calibrations and development records were frozen at 2026-10-05 17:20:49 UTC (`data/freeze.json`, 21 files), after calibration and development and before any test output. This is a local freeze with file digests, not a public preregistration. The study was designed after Study 2's prespecified results were known. The post hoc sweep analyses of Study 2 were saved 87 seconds after this freeze (`../04-reference-and-sweeps/results/posthoc/sweep-posthoc.json`, created 2026-10-05 17:22:16 UTC); the paper states this order.

**Questions.**
1. Does mean ablation change which attention heads a head-level patching analysis flags, compared with the average of individual donor patches?
2. Does grouping donors by the token at the patched position preserve the individual-average conclusions better than eight sampled donors at the same evaluation budget?

**Settings** (GPT-2 Small, every head patched at one position, 64 donors per recipient):
- `sva-end`: agreement, final token (the final noun), restore and destroy, 64 groups, 288 head-sites; token groups = final noun (8 of 8);
- `ioi-end`: IOI noising at the final token ("to" for every donor), 64 prompts, 144 head-sites; one token group, so the rule equals mean ablation;
- `ioi-s2`: IOI noising at the repeated subject, same prompts, 144 head-sites; 33-43 token groups per prompt.

Qwen3-0.6B was to be added only if its clean two-verb accuracy on the 64 fresh agreement groups reached 90%. It reached 79.7% (`results/qwen-gate/gate.json`), so it was excluded before any patching.

Results: [RESULTS.md](RESULTS.md); corrections and notes on them: [AMENDMENTS.md](AMENDMENTS.md).

## Files

| Path | Contents |
|---|---|
| `PROTOCOL.md` | Questions, settings, methods, prespecified outcomes H1-H4, the Qwen gate and rules; frozen |
| `common.py`, `engine.py` | Paths, checkpoint and freeze checks; head-patching engine (input of the attention output projection) |
| `prepare.py` | Prompt generation with new seeds; excludes every prompt string of the earlier studies |
| `run.py` | Calibration, development, Qwen gate, test and replay stages |
| `analyze.py` | Frozen analysis; writes `results/analysis-test.json` and `results/heads-*-test.csv` (every head-site) |
| `freeze.py` | Writes `data/freeze.json` |
| `verify.py` | Independent check written after the test runs; writes `results/verification.json` |
| `write_results.py` | Writes `RESULTS.md` from the saved analysis, verification and run records |
| `posthoc/head_spread.py` | Post hoc: head-level donor spread in the residual space against the residual stream's spread |
| `data/` | Generated prompts (calibration, development and test splits), generation audit, freeze record |
| `results/` | Raw per-recipient arrays (`*/test/`, `*/development/`, `*/replay/`), calibrations, completion records, analyses |
| `logs/` | Run logs |

Reproduce the analysis from the saved arrays with `python analyze.py test` and `python verify.py` (Python environment as in the project's REPRODUCIBILITY.md). Model reruns need the GPU and the checkpoints listed there; the run stages refuse to overwrite saved outputs.

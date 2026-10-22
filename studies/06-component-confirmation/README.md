# Study 4 (folder `06-component-confirmation`): which components does donor averaging bias?

In the manuscript this is **Study 4**, the prespecified component-level test (Section 8). The earlier studies are P1 (`../01-pilot`), P2 (`../../submission-study`), Study 1 (`../03-grouping-and-circuit`), Study 2 (`../04-reference-and-sweeps`) and Study 3 (`../05-head-confirmation`).

Run on 6 October 2026 (UTC) on GPT-2 Small. The protocol ([PROTOCOL.md](PROTOCOL.md)), code, generated prompts, calibration and development records were frozen at 2026-10-06 02:58:16 UTC (`data/freeze.json`, 14 files), after calibration and a four-group development run and before any test output. This is a local freeze with file digests, not a public preregistration. The study was designed after the results of Studies 1-3 and the post hoc sweep, random-circuit and error-discrimination analyses of Study 2. Two post hoc analyses were saved after this freeze: the head-level spread of Study 3 (`../05-head-confirmation/results/posthoc/head-spread.json`, 02:59:47 UTC) and the bias-against-noise comparison, which also holds the final-token comparison of Study 2's sweeps quoted in the manuscript's Section 5.4 (`../04-reference-and-sweeps/results/posthoc/bias-noise.json`, 03:10:45 UTC). The protocol's statement that the design followed all post hoc analyses is corrected in [AMENDMENTS.md](AMENDMENTS.md).

**Question.** Does the bias of mean ablation, and the comparison between token grouping and eight sampled donors, depend on the size of the patched component?

**Design.** 64 fresh agreement groups (8 x 8 donor grids), with separate calibration and development groups and no prompt string from any earlier study. At the final token, in both directions and at every layer, four kinds of site are patched: the residual stream after the block, the MLP output, the attention-layer output and every single head (360 site-directions). At each site: every donor individually, their centroid (mean ablation), token-group centroids (banks sharing the final noun), mixed-bank centroids, and eight donors sampled from the individual patches. The donors' spread is measured in the residual space (heads through W_O).

**Prespecified outcomes.** P1 misclassified sites under mean ablation per kind (98.75% interval); P2 token grouping against eight sampled donors by mean absolute error; P3 ordering of the mean absolute centroid bias across kinds; P4 spread order and its match with the bias order. The predictions and decision rules are in the protocol; the four-group development results that informed them are listed there.

Results: [RESULTS.md](RESULTS.md).

## Files

| Path | Contents |
|---|---|
| `PROTOCOL.md` | Question, design, prespecified outcomes, predictions and decision rules; frozen |
| `common.py`, `engine.py` | Paths, checkpoint and freeze checks; component-patching engine (residual stream, MLP, attention layer, single head) |
| `prepare.py` | Prompt generation with new seeds; excludes every prompt string of the earlier studies |
| `run.py` | Calibration, development, test and replay stages |
| `analyze.py` | Frozen analysis; writes `results/analysis-test.json` and `results/sites-test.csv` (every site-direction) |
| `freeze.py` | Writes `data/freeze.json` |
| `verify.py` | Independent check written after the test runs; writes `results/verification.json` |
| `write_results.py` | Writes `RESULTS.md` from the saved analysis, verification and run records |
| `data/` | Generated prompts (calibration, development and test splits), generation audit, freeze record |
| `results/` | Raw per-group arrays (`test/`, `development/`, `replay/`), calibration, completion records, analyses |
| `logs/` | Run logs |

Reproduce the analysis from the saved arrays with `python analyze.py test`, `python verify.py` and `python write_results.py` (Python environment as in the project's REPRODUCIBILITY.md). Model reruns need the GPU and the GPT-2 Small checkpoint; the run stages refuse to overwrite saved outputs.

# Study 2 (folder `04-reference-and-sweeps`): what the reference keeps, and where averaging misleads a sweep

In the manuscript this is **Study 2**; Study 1 is `../03-grouping-and-circuit` and Study 3 is `../05-head-confirmation`. The folder keeps its original name because frozen records refer to it.

Run on 5 October 2026 on GPT-2 Small and Qwen3-0.6B. The protocol ([PROTOCOL.md](PROTOCOL.md)) was frozen with the code and generated prompts at 2026-10-05 12:08:08 UTC (`data/freeze.json`), after calibration and development and before any test output.

**Experiment A** asks how often the 26-head IOI circuit reproduces GPT-2 Small's errors under four ablation references:
- names-only or names-place-object ABC donors;
- 8 or 128 donors.

Every reference is compared with a control that keeps no attention heads. The recipients are all model errors (up to 384 per template family) and 384 correct cases per family, drawn from a 65,528-prompt screen.

**Experiment B** runs full residual patching sweeps on fresh agreement prompts (GPT-2 Small, 64 groups; Qwen3-0.6B, 32 groups) and IOI prompts (GPT-2 Small, 64 prompts). At each site it compares every donor patched individually with the patch of their mean.

Results: [RESULTS.md](RESULTS.md).

## Files

| Path | Contents |
|---|---|
| `PROTOCOL.md` | Hypotheses, design, endpoints and decision rules, frozen before test runs |
| `common.py`, `engines.py` | Paths and checkpoint checks; residual-patching and head-ablation engines |
| `prepare.py` | Prompt generation (no model outputs); the fixed rule that builds each recipient's donor banks |
| `run_reference.py`, `run_sweep.py` | Calibration, development, screen, test and replay stages |
| `analyze_reference.py`, `analyze_sweep.py` | Frozen analyses; write `results/reference/analysis-test.json` and `results/sweep/analysis-test.json` |
| `retrospective_study3.py` | Post hoc re-analysis of Study 1's saved outputs that motivated this study (named after Study 1's earlier label) |
| `curvature.py` | Exploratory second-order (Hessian) prediction of the gap, run after the freeze |
| `posthoc/sweep_posthoc.py` | Post hoc sweep analyses added in revision: task competence, equal budget, token groups, thresholds, cancellation, spread |
| `posthoc/random_circuits.py` | Post hoc random-circuit control for Experiment A |
| `posthoc/error_discrimination.py` | Post hoc: odds ratio and AUROC for separating model errors from correct prompts (circuit, random circuits, empty control) |
| `posthoc/bias_noise.py` | Post hoc: centroid bias against the sampling SD of eight donors (Study 2 panels and Study 3 settings); final-token equal budget |
| `posthoc/verify_posthoc.py` | Independent recomputation from the raw arrays of the numbers of the four scripts above and of the head-level gaps saved by Study 3's post hoc spread analysis |
| `posthoc/review_controls.py` | Post hoc controls added for the fifth revision of the manuscript (13 October 2026): sampled donors at increasing cost, split-half stability of the reference, the token rule as defined, the IOI count under its original definition and without development or calibration prompts, Study 1 slopes within a grouping policy, error separation near the decision boundary and a surface-feature baseline. Refuses to overwrite its output |
| `posthoc/verify_review_controls.py` | Independent recomputation of `results/posthoc/review-controls.json` (separate code, other seeds) |
| `verify.py` | Independent recomputation of the primary numbers from the raw arrays |
| `test_design.py` | Design invariants (pytest) |
| `data/` | Generated prompts, the selected test recipients, generation audit, freeze record |
| `results/` | Raw per-recipient arrays (`test/`, `development/`, `replay/`), calibrations, analyses, replay checks |
| `logs/` | Run logs |

Run everything from the project root with `make verify`, `make test` and `make analysis`. Model reruns are described in the project's [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).

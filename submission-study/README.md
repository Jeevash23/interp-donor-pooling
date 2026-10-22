# Donor Pooling in Activation Patching

A completed local experimental study and an initial full manuscript in TMLR style. The paper is an unsubmitted research draft. The earlier pilot remains in the parent directory; this extension uses separate measurements.

## Read the work

- [Complete paper](output/pdf/donor-pooling-boundary-audit.pdf) — figures, proofs, results and appendices.
- [Editable manuscript](paper/main.tex) in official TMLR preprint mode.
- [Anonymous-format entry point](paper/submission.tex) — compiling does not submit anything.
- [Claim–evidence register](docs/CLAIM-EVIDENCE.md) — exact estimates, scope and evidence locations.
- [Literature audit](docs/LITERATURE-AUDIT.md) — closest prior work and the novelty boundary.
- [Data dictionary](docs/DATA-DICTIONARY.md) — experimental units, arrays and metric definitions.
- `REPRODUCIBILITY.md` and `MODEL-SOURCES.md` — execution, environment and checkpoint provenance.
- `docs/EXECUTION-NOTES.md` — calibration failures, remedies and reporting corrections.
- `docs/REVIEW-QUESTIONS.md` — internal critical appraisal for human review.

## What was executed

Six primary float32 checkpoints from four model families were evaluated on 288 generated groups each: 96 IOI, 96 subject–verb agreement and 96 key–value retrieval groups. Each group includes narrow/broad eight-donor banks, restoration/destruction, eight residual sites and 74 conditions per site. The primary panel contains **1,022,976 intervention executions**.

Completed supplementary evidence adds **455,280 intervention executions**: a 31-cluster/279-recipient BLiMP assay on all six models, attention-head scans in three families, nine predetermined lower-precision cases per primary model, and a separate full 288-group Qwen3-1.7B bfloat16 extension. There are also 47,424 verification repeats and 170,496 earlier development interventions. These are execution counts, not independent sample sizes; later archive-integration reruns are recorded separately.

## Primary result

The endpoint is the mean absolute propagation contribution minus the mean absolute final-normalization contribution, averaging the three tasks, two directions and six nonterminal sites under narrow donor banks. Intervals below are 99.1667%, with a six-comparison Bonferroni adjustment.

| Checkpoint | Estimate [adjusted interval] | Decision |
|---|---:|---|
| GPT-2 Small | 0.0692 [0.0607, 0.0786] | positive |
| GPT-2 Medium | 0.0572 [0.0506, 0.0648] | positive |
| GPT-2 Large | 0.0511 [0.0436, 0.0591] | positive |
| Pythia-410M | 0.0243 [0.0198, 0.0292] | positive |
| TinyLlama-1.1B | 0.0371 [0.0325, 0.0417] | positive |
| Qwen3-0.6B | 0.0833 [0.0775, 0.0893] | positive |

This is a fixed-checkpoint aggregate result. It does not assert that every task, direction, bank or layer has the same behavior. The full cell tables, model competence and secondary intervals are retained.

## Practical estimator comparison

Mean absolute error against the measured eight-donor average, restoration/narrow/nonterminal sites. The correction and five-sample resampling both use five recipient forwards. Resampling error averages all 56 five-of-eight subsets; shared donor collection is accounted for separately.

| Checkpoint | Mean patch (1) | Correction (5) | Resampling (5) | Correction advantage over resampling [95% interval] |
|---|---:|---:|---:|---:|
| GPT-2 Small | 0.0804 | 0.0533 | 0.0460 | -0.0073 [-0.0134, -0.0015] |
| GPT-2 Medium | 0.0707 | 0.0514 | 0.0505 | -0.0008 [-0.0055, 0.0035] |
| GPT-2 Large | 0.0946 | 0.0643 | 0.0545 | -0.0099 [-0.0177, -0.0027] |
| Pythia-410M | 0.0518 | 0.0513 | 0.0546 | 0.0033 [-0.0015, 0.0078] |
| TinyLlama-1.1B | 0.0584 | 0.0556 | 0.0501 | -0.0055 [-0.0100, -0.0010] |
| Qwen3-0.6B | 0.0949 | 0.0817 | 0.0617 | -0.0200 [-0.0269, -0.0132] |

A positive advantage means the correction has lower error. These comparisons are descriptive; an interval containing zero does not establish superiority or equivalence.

## Verification

All primary numerical gates passed at their original tolerance. An attempted Pythia CPU configuration failed its duplicate-donor gate before external measurements; that model remains in the external panel using its validated GPU configuration. Every test patch passed exact write-locality checks; terminal controls and decomposition closure passed over the complete primary data. Every primary model passed fresh-model replays and a direct float64 checkpoint-tensor readout on 5,328 saved states. Data regeneration and complete primary statistical recomputation matched exactly. Details, including near-tie top-token changes, are in `results/verification.json` and per-model audit records.

Reduced-precision calibration required a documented one-row readout adapter and, for some models, singleton native execution. Original failures and the calibration-only diagnosis are preserved. The reduced-precision configurations can produce nonzero pooling gaps at a structural null; these numerical artifacts are reported explicitly and do not enter the primary float32 claims. In the 1.7B run, the identity-margin discrepancy reaches 1.875 logits and the structural-null pooling gap reaches 0.3828125 logits. Its calibration passed, but its full-test numerical diagnostics prevent treating it as a successful robustness replication.

## Publication status

This is a complete experimental manuscript for author review, not an accepted paper or a promise of acceptance. The candidate contribution is an empirical boundary audit and comparison-stability evidence. The mean/resample distinction, telescoping identity and Taylor expansion are established mathematics. Human authors still need to evaluate the contribution against the closest literature, review the calculations and claims, agree authorship/disclosures, and approve any public release or submission.

The delivered manuscript uses TMLR formatting. An ARR version would need a separate, shorter main paper and a language-research framing; it is not presented as simultaneously compliant with both formats.

## Build and reuse

From `paper/`, run `latexmk -pdf main.tex` for the marked draft, or `latexmk -pdf submission.tex` for anonymous submission formatting. Install the pinned Python dependencies and use `scripts/reproduce.py` with a new output directory for experiments. On another machine, provide checkpoint directories through `--models-root` as described in the guide. No weights are included and no model training is performed.

The compact review archive includes all response metrics, data, source and summary records. The separate dense-state archive restores the saved activation arrays at the same relative paths; extract it over the review archive for exact state replay. Statistical recomputation works from the compact archive alone. The local project retains machine-specific execution logs; the review archives omit those logs and temporary reproduction copies. Archive manifests list exactly what each contains.

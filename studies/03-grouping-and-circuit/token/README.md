# Balanced donor grouping experiment

This new panel tests partitioning of a fixed donor set. It does not change the September study. The controlling plan is [the prospective protocol](../PROTOCOL.md).

Each recipient has 64 distinct clean donors in an eight-context by eight-final-noun grid and a paired corrupted grid. The recipient, head noun, answer, template, and prompt length are fixed. For each K in 1, 2, 4, 8, eight banks contain eight donors each and exactly K distinct final tokens. Every one of the same 64 donors appears exactly once at every K. Consequently, the global resample outcome and grand source centroid are fixed; only the grouping and bank centroids change. This controls marginal donor composition, while within-bank lexical diversity, semantics, covariance, and centroids change together.

The primary endpoint is the recipient-level mean difference in absolute logit-margin gap between K=8 and K=1, across three nonterminal END depths, both patch directions and eight partition rotations. Two checkpoints are tested with 97.5% intervals, using 20,000 group-level bootstrap draws stratified by template and answer. Other intervals are descriptive 95% intervals. A positive extreme contrast alone is not a monotonicity test.

The exact sample is 4 calibration, 8 development and 64 test recipients with two templates and two answer classes. There are 198 distinct interventions at each of eight sites, or 101,376 test interventions per model. Rotations, donors, sites, and directions are repeated measurements, not independent samples. The test set has 64 unique recipient pairs; chance donor reuse includes 154 repeated donor-prompt occurrences and two recipient/donor shared strings. All old/new-split and new cross-split exact overlaps are zero. The finite vocabulary and shared language constructions limit the generalization supported by group-level intervals.

Calibration includes exact write locality, native and reconstructed logits, identity patches, duplicate-donor means, causal-prefix states, and final-depth structural controls. Full test measurements repeat the locality, identity and structural checks. ANCHOR donors have the same context-state multiset at every K because the differing noun is causally later, so ANCHOR is a negative control for the grouping effect. Final-depth ANCHOR cannot affect the END output; final-depth END has no remaining propagation contribution.

The native Engine class is copied from the previous audited implementation without scientific changes. `common.py` supplies the relocated cached model paths. All runs are native float32, eager attention, deterministic, evaluation mode, no gradients, no generation, no cache and no TF32. Two gate-only omissions found during development review are documented in `results/pretest-gate-revalidation.json`; saved calibration/development values satisfy the corrected gates, and no model outcome was altered.

## Execution

From the portfolio workspace, use `<environment>/.venv/bin/python` as the Python executable. The experiment directory below is `<project>/followups/2026-10-pooling-consequences/token`.

1. `python token/prepare.py` generates tokenizer-qualified data, refusing existing files.
2. `python token/test_design.py` checks analytic pooling cases, exact partition support, index shape, causal-prefix invariance, bootstrap sampling unit, and all saved prompt invariants.
3. `python token/run.py calibrate --model gpt2-small`, followed by the same command for `qwen3-0.6b`.
4. `python token/run.py development --model MODEL` runs the eight reserved development cases; `python token/analyze.py --split development` analyzes them.
5. After implementation review, `python token/freeze.py` verifies checks, independently regenerates prompt files, binds the source/data/protocol/checkpoint record and creates the local freeze.
6. `python token/run.py test --model MODEL`, then `python token/run.py replay --model MODEL`, then `python token/analyze.py --model MODEL --split test`.

The test/replay stages require the frozen inputs and checkpoint files to match. Existing measurement outputs are never overwritten. GPU access is required for the admitted execution configuration. The four replay recipients are fixed before test: indices 0, 16, 32 and 48, using a fresh model instance.

## Saved measurements

Per-recipient compressed arrays retain `metrics[8 sites,198 interventions,5 metrics]`, `endpoints[8 sites,64 banks,2 endpoints,5 metrics]`, `dispersion[8 sites,64 banks,3 summaries]`, and `baseline[130 prompts,5 metrics]`. Metric order is margin, full-vocabulary target probability, target top-one indicator, positive target-minus-competitor margin indicator, and target log probability. In the corrupted baseline, that pair indicator measures preference for the original clean target; it is not relabeled corrupted accuracy.

Endpoint B applies the unchanged final norm/unembedding to the mean final residual from the eight donor interventions; C is the mean donor logit vector scored afterward. Thus margin D=P+N up to measured linear-readout roundoff, and probability D=P+N+S. These are ordered accounting terms, not unique causal mediation shares. Dispersion fields are mean squared radius, squared centroid norm and maximum radius. Fixed replay recipients additionally retain dense final/source states.

`results/analysis-test/*.json` includes the primary estimate, dose curves, all direction/site cells, competence, numerical checks and execution environment. The companion component archive has axes recipient, site, direction, dose, rotation, metric; the primary CSV contains one value per recipient.

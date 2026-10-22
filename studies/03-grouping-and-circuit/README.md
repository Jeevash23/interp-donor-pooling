# Donor grouping in activation patching

This is the completed October revision of the donor-pooling study, with two new experiments and a shorter paper. The controlled grouping effect is positive on both fixed checkpoints. The circuit experiment demonstrates a practical evaluation difference, while its error-agreement comparison remains inconclusive. The September measurements and original paper remain in `../../submission-study/`.

## Read the revision

- [Revised paper](output/pdf/donor-grouping-revised-paper.pdf), in TMLR preprint style and marked not submitted.
- [Editable LaTeX](paper/main.tex).
- [Claim and evidence map](CLAIM-EVIDENCE.md).
- [Prospective local protocol](PROTOCOL.md), frozen before new test interventions.
- [Reproduction guide](REPRODUCIBILITY.md) and [checkpoint sources](MODEL-SOURCES.md).
- [New study review archive](output/donor-pooling-revision-review.zip).
- [Independent response verification](results/revision-verification.json) and [archive extraction check](output/archive-integration.json).

## New evidence

Each recipient uses the same 64 context-by-noun donors in four different partitions. Each partition forms eight banks of eight; every donor appears once at each setting. The number of distinct final tokens per bank is K=1,2,4,8. The primary comparison averages three nonterminal END depths, both intervention directions and all eight rotations within each of 64 recipient groups. Intervals adjust for the two checkpoint comparisons. Values are logits.

| Checkpoint | Mean absolute gap K1 | Mean absolute gap K8 | Paired K8 minus K1 [97.5% interval] |
|---|---:|---:|---:|
| GPT-2 Small | 0.00777 | 0.36456 | 0.35679 [0.33809, 0.37629] |
| Qwen3-0.6B | 0.03164 | 0.46987 | 0.43823 [0.40779, 0.47016] |

The donor population and its global average individual-patch outcome remain fixed. The grouping changes bank centroids, semantic mixtures and activation dispersion; this does not isolate a pure token-identity mechanism. Both models use the same two-template agreement panel. Clean two-verb accuracy is 64/64 for GPT-2 but only 46/64 (71.9%) for Qwen; both choose the target verb as unrestricted top-one in only 1/64 cases. These are pooling measurements, not evidence that both models consistently solve grammatical agreement. Rotations and intervention executions are not independent samples. Probability gaps require separate interpretation.

The reference-circuit study screens 8,192 fresh GPT-2 Small IOI prompts, finds 119 two-name errors, then evaluates the predeclared 64 errors and 128 correct cases. Mean ablation versus expected coherent resampling changes correct-case agreement by **8.50 [5.66, 11.62] percentage points** and error-case agreement by **-3.91 [-10.74, 2.34] points** (descriptive 95% intervals). Empty-adjusted faithfulness is 0.710 versus 0.490.

Our high error agreement does not reproduce the severe loss reported in the September paper. The donor distribution, bank size, templates and resampling endpoint differ. [The verified comparison](circuit/COMPARISON-WITH-SEPTEMBER-PAPER.md) explains those differences without claiming which causes the discrepancy.

## What changed from the first draft

The new paper centers the controlled grouping test. The old propagation/final-normalization comparison is supporting evidence, with its donor-token confound made explicit. The old head-scan stability result is limited by small effect sizes and omission of the primary GPT-2 name movers. The five-forward correction is not presented as a superior estimator. The earlier reduced-precision failures remain disclosed.

The first circuit calibration's separate float32 projection diagnostic failed; a documented float64 algebra check resolved the structural question without changing native model computation or relaxing native-output gates. An analysis indexing bug was caught and corrected on development data before test inference. Failed and corrected records are retained.

## Research decision

Keep this as a focused empirical methods study and stop the planned experimental expansion here. The fresh controls address the strongest donor-composition criticism; the circuit assay supplies a bounded practical consequence. More arbitrary model sweeps are not the next requirement.

The remaining decision is scientific positioning, especially whether the controlled grouping lesson is sufficiently useful beyond established mean/resample sensitivity. The paper is ready for human scientific review, not represented as submission-ready or guaranteed publishable. TMLR remains a candidate for an empirical/reproducibility contribution. ARR would require a separate shorter manuscript and language-research framing; this artifact does not claim ARR formatting compliance. AI contributed substantially to design, implementation, execution, analysis and drafting, and the manuscript discloses that role. No public release or submission was performed.

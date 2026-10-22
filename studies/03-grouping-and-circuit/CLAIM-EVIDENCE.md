# Claim and evidence map

All October primary hypotheses and panel sizes were fixed before test interventions. The questions were motivated by observed September results; this is a local prospective plan, not public preregistration.

| Claim | Estimate and status | Direct evidence | Limit |
|---|---|---|---|
| Fixed-support K8 versus K1 grouping increases absolute margin pooling gap in GPT-2 Small | 0.35679 [0.33809, 0.37629] logits, adjusted 97.5% interval | `token/results/analysis-test/gpt2-small.json`, `token/results/gpt2-small/test/*.npz` | 64 recipient groups, two SVA templates, one fixed checkpoint; no pure token identity claim |
| Fixed-support K8 versus K1 grouping increases absolute margin pooling gap in Qwen3-0.6B | 0.43823 [0.40779, 0.47016] logits, adjusted 97.5% interval | `token/results/analysis-test/qwen3-0.6b.json`, `token/results/qwen3-0.6b/test/*.npz` | 64 recipient groups, two SVA templates, one fixed checkpoint; no pure token identity claim |
| Mean/resample circuit agreement differs on correct cases | 8.50 [5.66, 11.62] percentage points, descriptive 95% interval | `circuit/results/analysis.json`, raw `circuit/results/test/*.npz` | Same-bank eight-donor conditional assay; not circuit discovery |
| Error-agreement difference is inconclusive | -3.91 [-10.74, 2.34] percentage points | Same files, full pair-error stratum | 64 sampled errors; neither superiority nor equivalence established |
| Whole-donor support and average resample outcome are fixed across K | Exact combinatorial partition plus numerical invariants | `token/data/test.json`, `review/token-design-check.json`, both numerical summaries | Per-bank centroids and per-bank resample scores can change |
| Instrumentation is numerically stable in admitted float32 configurations | Native/identity/structural checks pass; fresh replay records retained | `results/revision-verification.json`, panel calibration/audit/replay JSON | Not a guarantee for other hardware, precision or larger models |
| Earlier six-checkpoint aggregate survives recomputation but is confounded | SVA-END supplies about 29–77% of the old primary aggregate | `results/previous-study-diagnostics.json`; original study raw arrays | Retrospective supporting evidence; task/site contrast does not isolate donor diversity |

## Claims intentionally not made

This study does not establish universal token-diversity causality, robust circuit membership, completeness or minimality, general monotonicity across output metrics, superiority of the earlier correction estimator, a successful larger-model reduced-precision replication, independent human review, or venue acceptance.

## Evidence separation

- New grouping test: 64 recipient groups per checkpoint, same texts across two checkpoints; 202,752 total intervention executions including residual controls.
- New circuit test: 192 selected recipients from an 8,192-example screen; 3,456 mean/resample reference/empty comparisons plus 576 identity/all-kept controls. Source and baseline forwards are recorded separately.
- Calibration, development, replay and archival recomputation are verification/development, not added scientific sample sizes.
- The original six-model results are separately archived prior evidence. The new review archive includes a retrospective summary and its input hashes, not the old raw-response archive or any model weights.

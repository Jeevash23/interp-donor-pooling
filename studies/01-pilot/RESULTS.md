# What the experiment found

The project measures why patching an average internal state differs from averaging the outcomes of individual donor patches. The broad distinction is already established in prior work. This study contributes a bounded diagnosis of where the discrepancy arises.

**The measured gaps are small, but most enter during downstream computation before the final LayerNorm.** Halving donor deviations around their unchanged centroid reduces the gap by roughly three quarters in both models. The overall ordering of the six nonterminal patch sites is unchanged.

## Main measurements

The same 64 groups and 256 recipient configurations are evaluated in both checkpoints. Each recipient has eight donors agreeing on its intended answer. Every model receives 40,960 intervention executions across eight sites and 20 conditions. These repeated measurements are clustered by group.

| Quantity | GPT-2 Small | GPT-2 Medium |
|---|---|---|
| Mean absolute total logit gap | 0.0268 [0.0234, 0.0306] | 0.0164 [0.0148, 0.0182] |
| Mean absolute propagation term | 0.0257 [0.0224, 0.0292] | 0.0166 [0.0150, 0.0183] |
| Mean absolute final normalization term | 0.0017 [0.0015, 0.0020] | 0.0011 [0.0010, 0.0013] |
| Propagation minus normalization | 0.0240 [0.0205, 0.0278] | 0.0154 [0.0137, 0.0173] |

Intervals for the last row are 97.5%, adjusted for the two-model primary comparison. Other displayed intervals are pointwise 95%. Values average the six nonterminal sites. Absolute terms do not necessarily add because signed contributions can cancel.

Halving donor deviations reduces the mean absolute logit gap by 75.7% in Small and 74.1% in Medium. That agrees locally with a quadratic approximation; a single radius comparison is not a general scaling law.

![All site logit components](results/analysis/logit-decomposition.png)

## Accuracy and probability

Clean-recipient top-one accuracy is 251/256 for Small and 254/256 for Medium. Corruption lowers it to 83/256 and 115/256. Donor accuracy is 1,963/2,048 and 2,014/2,048. No failures are filtered out of the primary analysis.

At the fixed three-quarter-depth END sites, mean-patch accuracy is 254/256 and 255/256, while individual-donor accuracy is 1,935/2,048 and 1,976/2,048. The paired differences are 4.74 and 3.13 percentage points. These secondary comparisons use oracle-matched donors and do not establish a practical accuracy-improvement method.

Mean absolute probability gaps across nonterminal sites are 0.449 and 0.557 percentage points. At the final block END site, no transformer computation remains, but the final normalization and softmax still produce a pooling gap. The explicit decomposition separates that output-scoring contribution.

## What did not happen

The aggregate site ranking did not change in either model. The results do not show that mean patches generally invalidate circuit conclusions, or that resampling is always preferable. Model size alone is not tested as a causal variable. The experiment identifies neither a new circuit nor a particular downstream nonlinearity.

## Verification

- All pre-evaluation model calibration checks passed their fixed logit tolerance.
- Every saved write was local to the intended position. Terminal null and positive controls passed.
- A freshly loaded model repeated all conditions and sites for 12 recipient configurations per model, with exact agreement on saved outputs and states.
- Independent readout from checkpoint tensors exactly reproduced the same 1,920 intervention states per model after matching native tensor shape. Earlier shape-dependent rounding records are retained.
- All-group statistics and uncertainty intervals reproduced exactly from saved measurements.
- Deterministic prompt and model-manifest regeneration reproduced all four checked files exactly.

The audit found four donor strings shared by generated calibration and evaluation banks; one was actually used in calibration. Recipient strings do not overlap. All data remain in the analysis. Algebraic closure was checked after inference, a disclosed timing deviation from the protocol. See [execution notes](docs/EXECUTION-NOTES.md).

## Read the work

The [paper PDF](output/pdf/donor-averaging-in-activation-patching.pdf), [editable Word draft](paper/manuscript.docx), [text manuscript](paper/manuscript.md), [full numerical table](results/analysis/site-results.csv), [literature comparison](docs/RELATED-WORK.md) and [reproduction instructions](REPRODUCIBILITY.md) are included.

This is an initial research draft with measured results. The incremental novelty claim needs broader review, and generalization beyond these two GPT-2 checkpoints and two grammar families remains untested.

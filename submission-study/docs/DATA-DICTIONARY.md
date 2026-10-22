# Reading the data and results

## Experimental unit and execution counts

A controlled group is one recipient configuration with its paired donor banks. There are 96 test groups per task and 288 total; these same groups are reused across the fixed models. Donors, sites, directions and conditions are repeated measurements within a group. A BLiMP cluster contains nine distinct prefixes, each of which is a recipient once with the other eight as donors.

An intervention execution means one intervened input example. Batched examples can share a framework forward call; the reported million-plus count is not the number of Python calls or an independent sample size. Baseline prompt executions and repeated verification runs are recorded separately.

## Controlled prompt records

`data/calibration.json`, `development.json` and `test.json` contain complete text, target/contrast words, character anchor spans, group identifiers, template and class assignments, and tokenized inputs for every included checkpoint. `encoded[model]` orders its 34 prompts as clean recipient, corrupt recipient, eight alternating clean/corrupt narrow pairs, then eight alternating clean/corrupt broad pairs. An explicit beginning token precedes the text. Every continuation is qualified as one appended token; its token ID is stored with the full prefix.

The legacy field `heldout_template` means the recipient template was unused **as a development recipient**. Broad development donor banks can contain those templates. It does not mark globally unseen templates. The frozen protocol's original wording and its reporting correction are both retained.

## Raw controlled arrays

Each `results/<model>-<dtype>/<split>/<group>.npz` contains:

| Array | Shape | Meaning |
|---|---|---|
| `metrics` | 8 × 74 × 5 | Responses at four depths × ANCHOR/END, for all named conditions |
| `endpoints` | 8 × 7 × 2 × 5 | Seven banks, each with B = readout of mean final state, and C = mean full-vocabulary logits |
| `baseline` | 34 × 5 | Unintervened responses in the prompt order above |
| `dispersion` | 8 × 4 × 3 | Narrow clean/corrupt then broad clean/corrupt; mean squared donor deviation norm, centroid squared norm, and largest donor deviation norm |
| `final` | 8 × 74 × hidden size | Saved pre-final-normalization states on the nine fixed audit groups only |
| `source` | 8 × 34 × hidden size | Source vectors on the same audit groups |

`design.json` specifies the exact condition/bank order, coordinates and metric order. The 74 conditions include 53 restoration and 21 destruction conditions. The first six sites are nonterminal; site 6 is the terminal ANCHOR control and site 7 the terminal END control. Arrays saved from bfloat16 executions are represented as float32/float64 where needed for storage and analysis; this casting does not restore lost precision.

The five metrics are: target-minus-contrast logit margin, full-vocabulary target probability, target top-one indicator, positive-margin indicator, and full-vocabulary target log probability. Probability is not renormalized to just the two selected tokens. A zero margin counts as not pair-correct. On corrupted inputs, the target remains the original clean answer; those indicators are clean-target preferences, not relabeled corrupted-answer accuracy.

Each companion audit JSON records exact write locality, target-vector equality and identity comparison. The compact archive retains all response metrics and omits only dense `final`/`source` arrays; the separate overlay restores the original files losslessly.

## Supplementary arrays

BLiMP files contain 9 × 4 × 10 × 5 response arrays per cluster: recipient, depth, condition, metric. Conditions are eight individual donors, their centroid, and identity. The head scan contains `head_sites × 11 × 5` responses per IOI group: eight donors, centroid, identity, paired recipient. The saved `sites` array specifies every layer/head coordinate.

Five BLiMP panels use CPU; Pythia BLiMP and all head panels use GPU. Device records, separate CPU calibrations, and same-device fresh-model replay records accompany them. Pythia's failed CPU gate is retained but supplies no external measurements.

## Derived summaries

`results/analysis/<model>.json` is the frozen primary analysis output. Its `primary` endpoint uses a 99.1667% interval; other intervals are descriptive 95% intervals. `cells` contains every task/direction/bank combination. `*-components.npz` stores signed gaps and ordered contributions. `all-cells.csv` gives a compact cross-model export.

The relative-effect ratio divides aggregate absolute pooling gap by aggregate absolute individual-average patch effect relative to the appropriate recipient baseline. It is not an average of per-case ratios. The `competent_subset_n` field always requires the clean recipient and all eight clean **narrow** donors to be pair-correct, including when the same filter is displayed beside a broad-bank result.

The resampling-5 error is the exact average absolute error over all 56 five-of-eight donor subsets. `curvature_improvement_over_resample5` is resampling error minus correction error, so a positive value favors the correction. A confidence interval containing zero establishes neither superiority nor equivalence.

State-array replays compare newly loaded model outputs with stored values. Direct readout uses explicit float64 normalization/unembedding without constructing a Transformers model. These are computational checks on the recorded local devices, not independent human or laboratory replication.

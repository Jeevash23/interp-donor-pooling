# Claim–evidence register

All values below come from completed test analyses. The local protocol and primary analysis were hashed before test inference; this timing record is not public preregistration. The earlier pilot informed the design.

| Claim | Evidence | Scope / decision |
|---|---|---|
| Ordered margin and probability identities close | Equations 3–4, Appendix proofs; `results/verification.json` | Algebraic accounting at specified boundaries; no uniqueness or causal-mediation claim. |
| Propagation exceeds final normalization in the aggregate | `results/analysis/<model>.json`, field `primary`; 6/6 intervals have positive lower limits | Fixed checkpoints, narrow banks, equal task/direction/site mixture; 99.1667% intervals. |
| Radius contraction changes pooling gaps | Per-model `radius` and `quadratic_vs_zero` fields | Controlled artificial rays around an unchanged centroid; standard Taylor context, not a universal power law. |
| The five-forward correction estimates finite-bank response | Per-model `estimators` fields; full five-sample budget comparison | Exploratory comparator; decisions below retain unfavorable results. |
| Pooling can affect score comparisons | Per-cell `rank`; `<model>-heads.json`; `rank-confidence.json` | Fixed six-site or partial-depth head universe; thresholds 0/0.05/0.1 plus opposite pointwise interval signs, descriptive comparisons. No complete circuit claim. |
| External and numerical robustness | `<model>-blimp.json`, `<model>-precision.json`, `qwen3-1.7b-scale.json` | External cluster assay and lower-precision diagnostics; not pooled into the primary six-test family. |

## Per-model decisions

### GPT-2 Small

- Primary propagation-minus-normalization: 0.0692 [0.0607, 0.0786]; positive.
- Half/full radius absolute-gap ratio: 0.3281 [0.3187, 0.3368] (95%, descriptive).
- Quadratic prediction improvement over predicting no gap: 0.0485 [0.0428, 0.0545].
- Correction improvement over the centroid: 0.0271 [0.0208, 0.0336].
- Correction improvement over five-sample resampling: -0.0073 [-0.0134, -0.0015]; negative.
- External BLiMP propagation-minus-normalization: 0.5728 [0.4214, 0.7389]; positive (95%, descriptive).
- Head top-5/10/20 overlaps: 5/5, 10/10, 20/20. Aggregate reversals at 0.05: 0/1128; 0 also have opposite pointwise 95% interval signs. These intervals are not simultaneous.

### GPT-2 Medium

- Primary propagation-minus-normalization: 0.0572 [0.0506, 0.0648]; positive.
- Half/full radius absolute-gap ratio: 0.3457 [0.3275, 0.3639] (95%, descriptive).
- Quadratic prediction improvement over predicting no gap: 0.0337 [0.0292, 0.0381].
- Correction improvement over the centroid: 0.0193 [0.0139, 0.0247].
- Correction improvement over five-sample resampling: -0.0008 [-0.0055, 0.0035]; interval includes zero.
- External BLiMP propagation-minus-normalization: 0.8861 [0.6584, 1.1245]; positive (95%, descriptive).

### GPT-2 Large

- Primary propagation-minus-normalization: 0.0511 [0.0436, 0.0591]; positive.
- Half/full radius absolute-gap ratio: 0.3877 [0.3729, 0.4029] (95%, descriptive).
- Quadratic prediction improvement over predicting no gap: 0.0380 [0.0310, 0.0450].
- Correction improvement over the centroid: 0.0303 [0.0207, 0.0399].
- Correction improvement over five-sample resampling: -0.0099 [-0.0177, -0.0027]; negative.
- External BLiMP propagation-minus-normalization: 0.7118 [0.4880, 0.9488]; positive (95%, descriptive).

### Pythia-410M

- Primary propagation-minus-normalization: 0.0243 [0.0198, 0.0292]; positive.
- Half/full radius absolute-gap ratio: 0.3043 [0.2926, 0.3163] (95%, descriptive).
- Quadratic prediction improvement over predicting no gap: 0.0322 [0.0287, 0.0357].
- Correction improvement over the centroid: 0.0004 [-0.0050, 0.0055].
- Correction improvement over five-sample resampling: 0.0033 [-0.0015, 0.0078]; interval includes zero.
- External BLiMP propagation-minus-normalization: 0.2238 [0.1447, 0.3127]; positive (95%, descriptive).
- Head top-5/10/20 overlaps: 5/5, 10/10, 19/20. Aggregate reversals at 0.05: 0/2016; 0 also have opposite pointwise 95% interval signs. These intervals are not simultaneous.

### TinyLlama-1.1B

- Primary propagation-minus-normalization: 0.0371 [0.0325, 0.0417]; positive.
- Half/full radius absolute-gap ratio: 0.3997 [0.3852, 0.4137] (95%, descriptive).
- Quadratic prediction improvement over predicting no gap: 0.0152 [0.0122, 0.0182].
- Correction improvement over the centroid: 0.0028 [-0.0021, 0.0076].
- Correction improvement over five-sample resampling: -0.0055 [-0.0100, -0.0010]; negative.
- External BLiMP propagation-minus-normalization: 0.1725 [0.1041, 0.2435]; positive (95%, descriptive).

### Qwen3-0.6B

- Primary propagation-minus-normalization: 0.0833 [0.0775, 0.0893]; positive.
- Half/full radius absolute-gap ratio: 0.3673 [0.3523, 0.3828] (95%, descriptive).
- Quadratic prediction improvement over predicting no gap: 0.0378 [0.0318, 0.0438].
- Correction improvement over the centroid: 0.0132 [0.0049, 0.0215].
- Correction improvement over five-sample resampling: -0.0200 [-0.0269, -0.0132]; negative.
- External BLiMP propagation-minus-normalization: 0.4018 [0.3216, 0.4883]; positive (95%, descriptive).
- Head top-5/10/20 overlaps: 5/5, 10/10, 20/20. Aggregate reversals at 0.05: 0/2016; 0 also have opposite pointwise 95% interval signs. These intervals are not simultaneous.

## Evidence limits that change interpretation

- All task failures remain in the main analysis; clean-target competence is shown explicitly. Corrupted-input preference is scored against the original clean target.
- Recipient templates unused during development may appear in broad development donor banks. This is recipient-template transfer, not globally unseen-template testing.
- The primary panel ends at about 1.1B parameters. Qwen3-1.7B is a separate bfloat16 diagnostic with material identity and structural-null errors; it does not establish a successful scaling or numerical robustness replication.
- Reduced-precision comparisons report their selected native batches and the one-row reconstruction adapter. Their observed structural-null artifacts limit interpretation; they do not establish hardware-independent precision robustness.
- The direct readout and fresh-model checks are independently implemented/re-executed computations on the same machine, not independent human or laboratory replication.
- A nonzero pooling gap does not establish that either intervention reference is the right causal ground truth.

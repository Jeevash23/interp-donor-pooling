# Fixed IOI circuit follow-up

This directory contains a self-contained native Transformers implementation of a paired mean8 versus coherent-resample8 circuit-evaluation assay. Read [PROTOCOL.md](PROTOCOL.md) for the exact scope, estimands, limitations, and numerical gates. No test circuit is selected from outcomes.

From the AI interpretability portfolio root, use the cached runtime and checkpoint:

```sh
<environment>/.venv/bin/python <project>/followups/2026-10-pooling-consequences/circuit/experiment.py calibrate
<environment>/.venv/bin/python <project>/followups/2026-10-pooling-consequences/circuit/experiment.py development
```

After reviewing development and freezing the final protocol/source, continue sequentially:

```sh
<environment>/.venv/bin/python <project>/followups/2026-10-pooling-consequences/circuit/experiment.py freeze
<environment>/.venv/bin/python <project>/followups/2026-10-pooling-consequences/circuit/experiment.py screen
<environment>/.venv/bin/python <project>/followups/2026-10-pooling-consequences/circuit/experiment.py test
<environment>/.venv/bin/python <project>/followups/2026-10-pooling-consequences/circuit/experiment.py replay
<environment>/.venv/bin/python <project>/followups/2026-10-pooling-consequences/circuit/experiment.py analyze
```

Preparation already generated 16 calibration groups, 32 development groups, and 8,192 screening groups. Each has one recipient and eight outcome-independent ABC donor strings, making 74,160 unique prompt strings overall. All strings are excluded against prior AI-03 prompt files. `prepare` refuses to overwrite an existing screen. Model files are checked against their preparation-time SHA-256 hashes before inference. The existing checkpoint is GPT-2 Small revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

The default batch is nine, float32/eager, no KV cache. On the local 6 GB NVIDIA GPU the model fits comfortably in 6 GiB; actual runtime is to be measured rather than guaranteed. The fixed screen uses 8,192 unpatched recipient rows. Development uses 960 inference rows. With 64 errors and 128 correct recipients, test uses 5,760 inference rows and the fresh-process replay uses 240. Calibration adds its explicitly recorded gate runs. Each intervention retains all MLPs and the residual/embedding scaffold. Test data selection uses pair correctness and frozen priorities only; all eight donors remain together in a bootstrap recipient cluster.

Key outputs:

- `data/generation-audit.json`: token qualification, exclusions, data-generation counts, cached checkpoint hashes.
- `results/calibration.json`: numerical gates and execution environment.
- `results/development/`: separate complete-procedure development measurements.
- `data/freeze.json`: final source/protocol/input hashes and local freeze timestamp.
- `results/screen.npz` and `results/screen-selection.json`: full-screen metrics, selection counts, probabilities, and hashes.
- `data/selected-test.json`: selected recipients, donor banks, screening outcomes, and inverse inclusion weights.
- `results/test/*.npz`: two circuit modes × nine conditions × nine metrics, predictor summaries and identity controls.
- `results/replay-verification.json`: fresh-process comparison.
- `results/analysis.json`: weighted paired effects, raw and normalized faithfulness, uncertainty, error/success strata, and template descriptions.

Array order in the per-recipient files is `metrics[circuit_mode, condition, metric]`: circuit modes reference/empty-attention; conditions mean8 followed by coherent donor0 through donor7; metrics are the explicit `METRICS` constant in the source. `mixture_metrics` refers to the average of normalized donor distributions; `mean_logit_metrics` refers to softmax/argmax of average donor logits. Neither is the expectation of per-donor accuracy or agreement. `prediction` retains unrestricted argmax IDs for every direct condition. `baseline` is the intact-model row. Output probabilities use float64 softmax of float32 logits.

This follow-up is supporting replication of an already studied question. It does not test circuit discovery, membership changes, isolated MLP-free circuits, broad natural-language reliability, or a population beyond the finite donor banks and fixed template generator.

# Traceability and reproduction

These notes are explanatory packaging documentation written after the token-panel freeze. Frozen files remain unchanged. The records establish local computational traceability, not public preregistration, human peer review, or scientific validity by themselves.

## What was frozen

`data/freeze.json` was created at `2026-10-02T09:32:08Z`, after four calibration and eight development recipients per model and before any test intervention. Its SHA-256 is `30b09b0e2d15872f12690780b4268c4e8ff8b754e2a7b04e9dc04a21945c789b`. It binds 20 files: the token source, prompt data, original README, protocol, checkpoint/environment record, independent design review, independent analysis verification, analytic tests, data regeneration and gate-revalidation records. Paths in the manifest are relative to `token/`, including three references into the parent revision directory.

The final frozen analysis SHA-256 is `5d7ca5bfc8eb00ee4bca5377be7a5c7f58e92e749e4d5e3a24bcbf4bac3ebf05`. A second [redacted] independently calculated development primary values and all four dose curves with explicit loops and obtained zero discrepancy from the vectorized code. The same review identified a pretest NumPy advanced-indexing error; it was corrected before test and a regression test now compares each recipient's result against an explicit nested loop. This is implementation verification, not independent replication by another research group.

The `native_engine.py` Engine class is copied from the September implementation. Only its module documentation and the imported local `common.py` definitions differ; model loading, hooks, numerical execution and scoring use the proven class implementation. The model files are the existing cached GPT-2 Small checkpoint revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` and Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`. Their precise config, tokenizer and weight hashes are recorded in `../results/environment-and-inputs.json`. Both test and replay check those actual files against the record before model loading.

The calibration/development runner initially recorded two checks without enforcing the exact intended gates: calibration write errors were included under the generic 0.001 bound rather than explicitly requiring zero, and development probability closure was recorded but omitted from its explicit probability gate. The implementation was tightened before test. Audit-only revalidation of the preserved original records showed zero write errors and probability closure below 0.00002; no numerical measurement or outcome was changed. See `results/pretest-gate-revalidation.json`.

## Software and execution

The admitted environment is Python 3.12.14, PyTorch 2.13.0 with CUDA 13.0, Transformers 5.16.1, NumPy 2.5.2, SciPy 1.18.1, Matplotlib 3.11.1 and Safetensors 0.8.0. The experiment ran on a [6 GB NVIDIA GPU] with batch size 16 for both models. Environment JSONs are emitted separately for calibration, development, test and replay; their timestamps and package metadata describe the actual stage.

Both models use native float32, evaluation mode, eager attention, deterministic algorithms, no gradients, no generation, `use_cache=False`, one-position final logits, and disabled TF32. Engine initialization sets four PyTorch threads and manual seed `260912`. Prompt batches group equal sequence lengths and are not sequence-padded. The final partial execution batch can therefore have a different row count. Numerical differences caused by arithmetic or kernel shapes remain subject to the same fixed bounds and are saved, not silently removed.

The identity/native/readout tolerance is 0.001 in logit or state coordinates and 0.00002 in target probability. Write locality and target-write error must be exactly zero. Test-time audits additionally check causal-prefix states, ANCHOR dose invariance, final-ANCHOR structural nulls, final-END donor equality, grand-centroid roundoff, and decomposition closure. A failing numerical check stops the run and writes a failed audit. No post-test tolerance adjustment is part of the plan.

## Directory dependencies

The frozen code locates paths relative to its own file location. Preserve this structure for a direct rerun:

```text
WORKSPACE/
  <models>/gpt2-small/
  <models>/Qwen3-0.6B/
  <project>/
    data/                         # historical pilot inputs for overlap recheck
    submission-study/data/        # historical study inputs for overlap recheck
    followups/2026-10-pooling-consequences/
      PROTOCOL.md
      results/environment-and-inputs.json
      review/token-design-check.json
      token/                      # all frozen code/data and measurement results
```

The model directories may be symlinks to the exact verified local checkpoints. Weights need not be redistributed with the review archive. `common.py` resolves this layout and does not expose a `--models-root` option. A portable archive launcher should construct this tree or checkpoint links outside the frozen scientific files. It should not edit the frozen paths and then claim an unchanged-source reproduction.

The old prompt files are needed to repeat the historical overlap audit. They are not needed to analyze existing measurements. If they are omitted from a compact package, disclose that limitation and retain the original overlap evidence and its listed sources. New prompt-data byte equality can still be checked against the frozen files, but that alone is not a fresh historical-overlap verification.

## Commands

From `WORKSPACE`, the original runtime is `<environment>/.venv/bin/python`. Set `TOKEN` to the experiment directory above and use that Python as `PYTHON` in the following examples. These names are descriptive placeholders; no shell configuration is assumed.

```text
PYTHON TOKEN/test_design.py
PYTHON TOKEN/analyze.py --model gpt2-small --split test
PYTHON TOKEN/analyze.py --model qwen3-0.6b --split test
```

Those commands reconstruct tests/statistics from saved inputs and arrays. They need no GPU inference; analysis writes only derived analysis outputs. Prompt regeneration should go into a new directory:

```text
PYTHON TOKEN/prepare.py --out NEW_EMPTY_DATA_DIRECTORY
```

Compare the regenerated calibration, development and test JSON bytes with the original files. The generation-audit timestamp is expected to differ. The original pretest regeneration record already confirms byte equality for all three splits.

GPU stages are:

```text
PYTHON TOKEN/run.py calibrate --model gpt2-small
PYTHON TOKEN/run.py development --model gpt2-small
PYTHON TOKEN/run.py test --model gpt2-small
PYTHON TOKEN/run.py replay --model gpt2-small
```

Repeat for `qwen3-0.6b`. The saved evidence directories intentionally reject overwriting measurements, so a fresh rerun should use `--out NEW_RESULTS_DIRECTORY` on every stage or run in a fresh project copy. `run.py` requires the matching passed calibration record and admitted batch size in that output directory. A replay additionally requires its corresponding full test results to compare against; it evaluates the four predetermined recipients with a fresh model instance and compares metrics, boundary endpoints, baselines, dispersion and dense residuals at tolerance 0.001.

`analyze.py` reads the fixed default `token/results` directory, not an arbitrary `--out` path. To analyze a fresh isolated rerun while preserving originals, use a separate project copy whose `results` directory points to the new outputs, or independently compare the new arrays against the released arrays. The frozen script has no resume or output-overwrite option. This makes its scope explicit rather than silently replacing evidence.

## What a compact source/data package must retain

Retain every relative file in `data/freeze.json`, including the parent protocol, environment/checkpoint record and design-check record. Keep all per-recipient NPZ metric/endpoint/baseline/dispersion arrays and their JSON audits, `design.json`, stage environments, calibration, completion, replay verification and analysis files. Keep the four predetermined dense replay cases if claiming full dense-state replay compatibility. Logs and explanatory documents help audit chronology but are not substitutes for numerical arrays.

A package may omit the other model weights and the surrounding unrelated research projects. The archived source should include all local Python modules imported by the execution path: `common.py`, `native_engine.py`, `prepare.py`, `run.py`, `analyze.py`, `test_design.py`, `revalidate_gates.py` and `freeze.py`. Their frozen imports use standard Python, NumPy, PyTorch and Transformers; SciPy and Matplotlib versions appear in environment metadata although this token analysis does not require plotting or SciPy routines.

Rechecking an archive should verify member hashes, regenerate the prompt bytes, run analytic/design tests, recompute both model summaries from archived arrays, and compare a fresh-model replay if GPU time and exact checkpoints are available. Do not describe a lightweight statistics-only check as a fresh model replication.

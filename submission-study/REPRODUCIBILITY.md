# Reproducing the study

The original results are preserved. Every reproduction uses a new output directory and refuses to overwrite an existing run. The paper's primary evidence uses float32 computation on six fixed checkpoints, with Pythia executed one example at a time following calibration. bfloat16 checks and the Qwen3-1.7B extension are separate evidence.

## Environment

Exact executed versions, parameter counts, GPU, precision and batching are in `results/<model>-<dtype>/environment.json`. The shared local Python used for this study is `../../<sibling-project>/.venv/bin/python` relative to this directory. A fresh environment can install `requirements.txt`; use a PyTorch build compatible with the host GPU. Matching software does not guarantee bitwise agreement on a different device.

The primary model keys are `gpt2-small`, `gpt2-medium`, `gpt2-large`, `pythia-410m`, `tinyllama-1.1b`, and `qwen3-0.6b`. Their upstream identifiers, exact cached revisions and file hashes are in `data/checkpoints.json` and `MODEL-SOURCES.md`. Weights are not redistributed. In the original workspace the scripts resolve existing cached directories. On another machine, put each model in `<models-root>/<model-key>` and pass `--models-root` to the isolated reproduction launcher. Full data regeneration also needs the Qwen3-1.7B tokenizer/config directory, since the initial generator qualified that optional scale extension; its weights are unnecessary for data generation.

## Isolated verification

The compact review ZIP contains all response metrics, endpoints, baselines and dispersion arrays, plus input data, analysis and execution source. To keep the review package below the venue's supplementary limit, it omits only dense `final`/`source` activation arrays from selected NPZ files. The separate dense-state ZIP restores those original files losslessly when extracted over the same project directory. Its manifest gives the original hashes. Statistical recomputation works from the compact package alone; exact saved-state replay requires the overlay. A full new inference run can regenerate states without the overlay.

Run from this study directory using the intended Python interpreter. Replace output directories with new, unused locations.

```sh
python scripts/test_mathematics.py
python scripts/reproduce.py regenerate --out /tmp/pooling-data-check
python scripts/reproduce.py statistics --model gpt2-small --out /tmp/pooling-statistics-check
python scripts/reproduce.py replay --model gpt2-small --out /tmp/pooling-model-check
```

`regenerate` rebuilds calibration, development, test and BLiMP data and checks exact file hashes. It includes the pilot's pre-existing prompt exclusions in a private staging directory. `statistics` recomputes the complete selected model summary and compares it to the saved result. `replay` reloads the model and repeats every condition/site for the first group in each task/seed block, comparing raw responses, boundary endpoints and stored states. The launcher copies inputs before execution and writes a verification record outside the original experiment.

To repeat a full selected model with a separate cache:

```sh
python scripts/reproduce.py full --model gpt2-small --models-root /path/to/models --out /path/to/new-run
```

This validates instrumentation first, then evaluates all 288 test groups and performs the fixed-subset replay. The model cache is read only. A failed calibration stops evaluation. For a complete regeneration on a new machine, supply every tokenizer/config listed above; for a selected-model replay only that model's weights/tokenizer are needed because exact tokenized data are already included.

## Main execution and supplementary assays

`run.py` implements the primary residual experiment and `analyze_core.py` the frozen primary/secondary estimators. In the original directory, `batch_run.py main` runs models sequentially and skips already completed stages; individual data files cannot be overwritten or silently resumed. To avoid altering delivered results, use the isolated launcher for new runs.

`external.py` prepares/evaluates/analyzes the BLiMP assay. `heads.py` evaluates and analyzes the predefined attention-head universe. `audit_readout.py` computes final normalization and unembedding directly from checkpoint tensors in float64, without constructing a Transformers model. The independent readout is an approximate higher-precision numerical reference, so its measured errors and fixed tolerances are reported; exact equality is not presumed.

Five delivered BLiMP panels use `cpu_assays.py`, a device-only wrapper around the unchanged frozen assay functions. It requires separate CPU calibration at the same float32 tolerances, then runs the assay and a fresh-model CPU replay. `cpu_blimp_batch_v3.py` executes the five eligible CPU panels while primary GPU inference runs separately. Pythia failed CPU calibration before any external measurement and uses the original GPU runner for BLiMP; its failed CPU gate is retained. The earlier all-CPU scheduling proposal is preserved; measured CPU throughput led to keeping all head scans on GPU before any head outcomes. Per-assay device records and `cpu-calibration.json` identify the executed environment. This scheduling addition was frozen before any BLiMP/head outcomes; its timing is recorded in the execution notes. The original `external.py`/`heads.py` default to GPU. Use the CPU wrapper for the five delivered CPU BLiMP panels; delivered head scans use `heads.py` on GPU.

The portable isolated launcher also supports the delivered BLiMP (CPU except Pythia) and head (GPU) assays:

```sh
python scripts/reproduce.py blimp --model gpt2-small --models-root /path/to/models --out /path/to/new-blimp-run
python scripts/reproduce.py heads --model gpt2-small --models-root /path/to/models --out /path/to/new-head-run
```

They perform fresh calibration on the relevant device, execute every planned group, reload the model for the fixed replay subset, and compare the resulting summary with the delivered result. The launcher dispatches BLiMP to its recorded device. Head and Pythia-BLiMP reproduction default to GPU; an explicit `--device cpu` requests a different-device diagnostic and may differ numerically. Exact summary equality is expected only in the recorded environment; a different machine can require examining measured numerical differences. The launcher verifies relevant checkpoint files against the original hashes before data regeneration or model inference. Statistical recomputation needs no model cache.

## Interpreting the evidence

The independent unit is a generated group (96 per task), or a BLiMP cluster (31). Donors, sites, directions and conditions are paired measurements. The six fixed checkpoints are not repeated training seeds. Primary intervals use 20,000 task/seed-stratified group bootstrap draws and a six-comparison Bonferroni adjustment. Other intervals are descriptive and pointwise. See `docs/PROTOCOL.md`, `docs/EXECUTION-NOTES.md` and the manuscript for precise boundaries.

`data/freeze.json` records primary scientific inputs before test inference. `data/supplement-freeze.json` records additional execution/audit scripts. These are local timing records, not public preregistration or third-party attestations. File hashes establish identity of inspected inputs; the failed-calibration records explain engineering changes. Raw outputs, replay records and regenerated summaries provide the empirical chain to the claims.

## Reduced-precision execution

The separate bfloat16 panel uses `reduced_precision.py`, a documented adapter around the unchanged frozen `run.py` and `engine.py`. It evaluates multi-row final-readout reconstruction one row at a time. Every scientific mean-state endpoint has one row and follows the exact original readout branch. Calibration first tests the native default batch, then singleton execution if needed, using the original 0.5-logit and 0.05-probability gates. `batching-selection.json` and `environment.json` record the chosen batch. The first GPT-2 Small default and singleton attempts before this adapter failed and remain preserved; its calibration-only readout diagnosis is also included.

The primary float32 results use their original batches. Cross-precision comparisons therefore report both batch sizes and describe a numerical-configuration comparison, without isolating dtype from all kernel-shape effects. `data/bfloat16-batching-amendment.json` and `data/bfloat16-readout-amendment.json` record timing and source hashes before any reduced-precision test measurement. `finish_reduced_precision.py` is the recorded local coordinator; it must not be used to overwrite delivered results.

The isolated launcher supports the reduced-precision diagnostics and the larger checkpoint:

```sh
python scripts/reproduce.py statistics --model gpt2-small --dtype bfloat16 --out /path/to/new-bfloat16-statistics
python scripts/reproduce.py precision --model gpt2-small --dtype bfloat16 --models-root /path/to/models --out /path/to/new-bfloat16-check
python scripts/reproduce.py full --model qwen3-1.7b --dtype bfloat16 --models-root /path/to/models --out /path/to/new-scale-run
```

The precision command performs fresh calibration and repeats all nine predetermined cases at the recorded batch size, then compares response metrics, endpoints, baselines and dispersion with the delivered arrays. It works from the compact archive without dense states. The full larger-model command recalibrates, evaluates all 288 groups and performs its fixed-case fresh-model replay using the recorded adapter and batch. All outputs remain separate from the delivered study.

The Qwen3-1.7B run passed reserved calibration but exhibits substantial identity and structural-null errors over the complete test set. Replaying those responses establishes repeatability, not numerical validity. It is retained as a failed robustness diagnostic and is excluded from the primary float32 claims. The execution ledger excludes subsequent archive-integration reruns, which have their own verification record.

The shared checkpoint paths were updated on 18 September 2026. Exact source relocations are recorded separately in `data/consolidation-source-relocations.json`; original freeze records and numerical results are unchanged.

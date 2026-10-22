# Reproducing the experiment

Run commands from this project directory. The local interpreter is `../<sibling-project>/.venv/bin/python`; no packages were changed. Package versions used for the scientific run are in `requirements-experiment.txt`, with actual environment records under `results/gpt2-small/` and `results/gpt2-medium/`. The exact CUDA build and hardware can affect floating-point results.

## Fast saved-output reproduction

```
../<sibling-project>/.venv/bin/python scripts/reproduce.py analysis
```

This recomputes every summary and interval from the complete saved outcome arrays, compares the resulting JSON objects exactly, and checks the frozen experiment/protocol/data hashes. It requires NumPy and Matplotlib but no model loading or GPU. It does not independently re-execute the model.

To regenerate figures and CSV tables from the saved measurements, run `scripts/analyze.py` using the same interpreter. The primary two-model intervals are 97.5%; secondary intervals are pointwise 95%. The resampling unit is the group, preserving all four within-group variants and all intervention conditions.

## Deterministic data regeneration

```
../<sibling-project>/.venv/bin/python scripts/reproduce.py data
```

This creates a new timestamped directory under `tmp/reproductions`, regenerates prompt files and checkpoint hashes, and checks them against the original four data files. It reads the original model/tokenizer directory. Do not run `experiment.py prepare` over the published data: that command is for an empty project, whereas this wrapper isolates regeneration.

## New full model replay

```
../<sibling-project>/.venv/bin/python scripts/reproduce.py full
```

This creates an isolated copy under `tmp/reproductions`, links the existing checkpoint directory, preserves the original frozen inputs, and runs calibration, evaluation, fixed-subset replay, independent readout and statistical analysis. Original result files are never overwritten. The experiment refuses to overwrite an existing group journal. Interrupted runs are preserved; start another isolated replay rather than changing settings and appending to a partial scientific run.

GPU execution may need host access because the app sandbox cannot initialize CUDA. The actual run used a working PyTorch CUDA path despite the system utility's NVML mismatch. No driver changes or paid cloud resources are needed. On a different machine, create a compatible environment and obtain the public checkpoints at the revisions recorded in `data/checkpoint-provenance.json`; compare actual files with `data/model-manifest.json`. CPU execution is supported by `experiment.py --device cpu`, but exact GPU reproducibility is not implied.

## Checkpoints and dependencies

- GPT-2 Small: `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- GPT-2 Medium: `openai-community/gpt2-medium`, revision `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`.

Revision provenance comes from the earlier download manifest. The actual currently loaded files have independently recorded SHA-256 hashes. Weights are stored in the sibling `<sibling-project>/models` directory, not duplicated in this project. Scripts import no sibling scientific implementation.

The frozen pre-inference record covers `scripts/experiment.py`, `docs/PROTOCOL.md`, and the original generated data/weight manifest. Analysis and document code were authored subsequently under the stated estimands. The freeze is a local timing record, not public preregistration. `docs/EXECUTION-NOTES.md` reports the post-inference closure-check timing and the calibration donor overlap.

## Raw array schema

Each model has 64 `results/<model>/raw/group-NNN.npz` files. In each group:

- `baseline`: shape 4 x 10 x 3, in clean recipient, corrupt recipient, donor 0 through donor 7 order.
- `metrics`: shape 4 x 8 x 20 x 3, for four variants, eight sites and 20 conditions.
- `final`: shape 4 x 8 x 20 x width, the END residual before final LayerNorm.
- `source`: shape 4 x 8 x 10 x width, patch-site source states in baseline order.
- `endpoints`: shape 4 x 8 x 4 x 2 x 3, for first-two, first-four, all-eight and half-radius banks; B then C readout comparator.
- `ids`: four identifiers matching the fixed evaluation data.

The metric axis is correct-minus-subject logit difference, full-vocabulary correct-answer probability, then full-vocabulary top-one correctness. Sites iterate over the four block indices, each S2 then END. Conditions are donor0 through donor7, mean2, mean4, mean8, half0 through half7, identity. Float32 forward states are retained; probability calculation uses float64 softmax of the measured float32 logits. Full-vocabulary logits can be reconstructed from final states and checkpoint weights; they are not all duplicated on disk.

## What verification establishes

Pre-evaluation calibration checks native execution, capture, identity writes, write locality, duplicate donors, singleton/batch consistency and terminal controls. Complete saved-output checks cover every primary measurement and algebraic closure. Newly loaded model replay covers all conditions/sites for groups 0, 31 and 63: 1,920 interventions per model. Independent final-state readout uses direct checkpoint tensors rather than the Transformers model. Statistical replay covers all groups. These checks validate implementation and transcription, not novelty or population generalization.

## Rebuilding the draft

Use the bundled document Python runtime to execute `scripts/write_paper.py`, which reads the measured summaries. Render its DOCX with the document skill's `render_docx.py` using bundled LibreOffice, inspect every page, then copy the resulting manuscript PDF to `output/pdf/donor-averaging-in-activation-patching.pdf`. The document toolchain is separate from the scientific environment. Plain text manuscript and figures remain readable without that toolchain.

The shared checkpoint paths were updated on 18 September 2026. Exact source relocations are recorded separately in `data/consolidation-source-relocations.json`; original freeze records and numerical results are unchanged.

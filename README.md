# How donors are chosen and combined changes activation-patching conclusions

Code and results for the preprint *How donors are chosen and combined changes activation-patching conclusions*, by Richard Jeevash (link to be added when it is posted).

This repository holds the code, frozen inputs, study records and saved outputs behind it; the saved outputs and the
frozen protocols are attached to its v1.0 release (see [Release files](#release-files)). Everything the manuscript
reports can be rechecked from the saved outputs on a CPU without downloading a model; the model runs can be
repeated with the pinned checkpoints.

## Release files

The saved outputs and the frozen protocols are not in this repository's files; they are attached to its v1.0 release:

| File | Size | Contents |
|---|---|---|
| `per-run-outputs.tar.gz` | 47.4 MB | The saved outputs: the 7 compressed archives in `data/saved-outputs/`, which `make unpack` places in the study folders and checks file by file |
| `frozen-protocols.tar.gz` | 53 kB | The studies' protocols, predictions, amendments, approvals and freeze records (36 files), at their places in the folder, where the checks read them |
| `SHA256SUMS` | | Checksums of the two archives |

Download the three files into an empty folder, check them there, and unpack them into this one:

```sh
sha256sum -c SHA256SUMS
tar -xzf per-run-outputs.tar.gz -C /path/to/interp-donor-pooling
tar -xzf frozen-protocols.tar.gz -C /path/to/interp-donor-pooling
```

After unpacking, `sha256sum -c SHA256SUMS` in this folder checks every file that its `SHA256SUMS` lists, the
unpacked attachments included, and the checks below run as described.

## What the paper asks

An activation-patching analysis with many donor inputs makes two choices: how the donors are combined, and
what they contain. Mean ablation patches the average of the donor activations; resample ablation patches each
donor in turn and averages the outcomes. The paper measures when either choice changes a conclusion, in GPT-2
Small, Qwen3-0.6B and Qwen3-1.7B: how the gap between the two grows with the spread of the donors, which sites
a patching sweep flags, how often an ablated circuit agrees with the model's errors, and whether the effect
reaches single attention heads and other components. Three of the studies were prespecified and frozen before
their test runs; their protocols, predictions, amendments and freeze records are included.

## What is in this folder

| Path | Contents |
|---|---|
| `studies/` | One folder per study: code, protocol and other records, frozen inputs (`data/`), and, after `make unpack`, saved outputs (`results/`) |
| `submission-study/` | The September six-model audit (P2), kept at its own path because its scripts find their inputs from there |
| `paper/` | `make_numbers.py` (every number and generated table in the manuscript) and `make_figures.py` (its figures); `generated/` holds the values the manuscript was built with |
| `data/saved-outputs/` | The saved outputs as compressed archives (47.7 MB); `make unpack` places them in the study folders (attached to the release: `per-run-outputs.tar.gz`) |
| `data/PACKED-FILES.json` | SHA-256 of every packed file, checked when it is unpacked |
| `tools/` | Unpacking, freeze checking, model download and rerun helpers |
| `models/` | Empty; `models/README.md` and `models/models.json` say which checkpoints to download and how |
| `pyproject.toml`, `uv.lock` | The exact Python environment the studies ran in (see `ENVIRONMENT.md`) |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original project, with the original's SHA-256 |
| `SHA256SUMS` | Checksums of every file in the release |
| `LICENSE` | The MIT licence |

## Studies

Folder names predate the manuscript's study names and are kept because the frozen records refer to them.

| Manuscript | Folder | Models | What it did | From this release |
|---|---|---|---|---|
| P1 | `studies/01-pilot` | GPT-2 Small, Medium | Mean against individual patches at eight residual sites (appendix) | Saved analyses; reruns with the models |
| P2 | `submission-study` | six float32 checkpoints, Qwen3-1.7B (bfloat16) | Gap split into propagation and final-normalisation parts; confounded by donor spread (appendix) | Saved analyses; reruns with the models |
| Study 1 | `studies/03-grouping-and-circuit` | GPT-2 Small, Qwen3-0.6B | Balanced regrouping of fixed donors; first IOI circuit comparison | Saved analyses; reruns with the models |
| Study 2 | `studies/04-reference-and-sweeps` | GPT-2 Small, Qwen3-0.6B | 2 × 2 reference experiment for the IOI circuit; full patching sweeps; post hoc controls in `posthoc/` | Every verifier runs from the saved arrays |
| Study 3 | `studies/05-head-confirmation` | GPT-2 Small (Qwen3-0.6B failed its gate) | Prespecified head-level test on fresh prompts | Verifier runs from the saved arrays |
| Study 4 | `studies/06-component-confirmation` | GPT-2 Small | Prespecified test across components at the final token | Verifier runs from the saved arrays |
| Study 5 | `studies/07-larger-model` | Qwen3-1.7B (float32, Modal) | Studies 2–4's prespecified tests in a larger model | Freezes and records verified locally; reruns on Modal |

## Install

The studies ran with Python 3.12.14, PyTorch 2.13.0, Transformers 5.16.1, NumPy 2.5.2, SciPy 1.18.1 and
Matplotlib 3.11.1. With [uv](https://docs.astral.sh/uv/):

```sh
make setup        # uv sync --frozen --extra experiment --group dev: creates .venv from uv.lock
```

`make setup` downloads about 3.0 GB of packages (most of it PyTorch's CUDA libraries); `ENVIRONMENT.md` lists
them and gives a smaller CPU-only route. To use another environment with the same packages, pass
`PYTHON=/path/to/python` to any `make` target.

## Quick check (CPU, a few minutes, no model, no network)

First unpack the two release attachments into this folder (see [Release files](#release-files)): the checks read
the saved outputs and the freeze records they contain.

```sh
make quick-check
```

This unpacks the saved outputs and the compressed inputs (checking every SHA-256), then:

1. checks every freeze record against the files it names (`tools/check_freezes.py`);
2. runs the independent verifiers of Study 4 and Study 3, which recompute the prespecified primary numbers
   from the raw arrays and compare them with the saved analyses;
3. verifies Study 5's protocol, input, amendment and stage-4 freezes;
4. recomputes every number and generated table in the manuscript from the saved outputs and compares them with
   `paper/generated/` (`paper/make_numbers.py --check`);
5. prints a few of the recomputed numbers next to the saved ones.

Expected end of the output:

```
Study 4 (components, GPT-2 Small, final token): mean absolute bias of mean ablation
  kind    recomputed from raw arrays   saved analysis   P1 interval redrawn   saved
  resid                     0.042324         0.042324   [5, 10]               [5, 10]
  mlp                       0.006751         0.006751   [0, 3]                [0, 3]
  attn                      0.001108         0.001108   [0, 1]                [0, 1]
  head                      0.000091         0.000091   [0, 1]                [0, 1]
Study 3 (single heads, GPT-2 Small): recomputed flag counts match the saved analysis in every setting: True
  Qwen3-0.6B accuracy gate: 0.7969 against the required 0.9 (failed, as recorded; the study ran on GPT-2 Small only)
  prompt separation: 0 of 11944 test strings shared with earlier studies
Manuscript: 1502 number macros and 8 generated tables recomputed from the saved outputs; all identical
```

The whole check, unpacking included, took between one and four minutes on a CPU that was busy with other work.

## Recheck everything from the saved outputs

```sh
make verify          # Studies 2-5: every verifier, including the post hoc ones (about 5 minutes on a CPU)
make numbers-check   # every manuscript number and generated table
make figures         # Figures 2-12 into paper/figures/
make test            # Study 2 design invariants; needs the GPT-2 Small tokeniser (make models)
```

Study 5's verifier reports one problem, as recorded: four IOI replays exceeded their 1e-3 tolerance (by 1.1–1.6
× 10⁻³ logits). The author accepted the results with this disclosed and with a post hoc bound showing that no
flag can change (`studies/07-larger-model/NOTES-AFTER-TEST.md`, `posthoc/replay_noise.py`). `make verify-study5`
passes when that is the only problem. Study 5's raw arrays (about 130 MB) stayed on the cloud volume; the part of
its verifier that reads them ran there, and its report is `studies/07-larger-model/results/analysis/verification-arrays.json`.

Study 1's array-level verifier (`studies/03-grouping-and-circuit/scripts/verify_revision.py`) needs the raw
token arrays (44 MB, not compressible), which are not included; the manuscript's Study 1 numbers are rechecked
from its saved analyses by `make numbers-check`, and the arrays can be regenerated as described below.

## Models

`make models` downloads GPT-2 Small and Qwen3-0.6B (about 2.1 GB), which Studies 1–4 use; `make test` needs only
GPT-2 Small's tokeniser files. `python tools/get_models.py --list` shows every checkpoint with its Hugging Face
repository, revision and size, and `models/README.md` explains the rest. Each file is checked against the
SHA-256 recorded before the studies ran, and the study code checks them again before every model stage.

## Full reruns

The model stages write into `results/` and refuse to overwrite existing outputs, so a rerun starts from a fresh
copy of the study folder:

```sh
make fresh-copy STUDY=studies/06-component-confirmation   # copies to reruns/06-component-confirmation and prints the stages
```

The original runs used one 6 GB NVIDIA GPU in float32. Measured times, each job alone unless
noted:

| Study | Stages (documented in each study's README and `tools/fresh_copy.py`) | Time per unit | Units in the test |
|---|---|---|---|
| Study 2 | calibrate, development, freeze, screen, test, replay; sweeps for agreement and IOI | 1.4 s per Experiment A recipient; 14 s per GPT-2 agreement group; 16 s per IOI sweep prompt; 146 s per Qwen agreement group | 64, 64 and 32 sweep groups |
| Study 3 | prepare, calibrate, development, freeze, gate, test, replay | 47 s per agreement group; 22 s per IOI prompt at the final token; 40 s at the repeated subject (sharing the GPU) | 64 per setting |
| Study 4 | prepare, calibrate, development, freeze, test, replay | 37 s per agreement group (360 site-directions, 81 patches each) | 64 groups |
| Study 1 | see `studies/03-grouping-and-circuit/REPRODUCIBILITY.md` (`scripts/reproduce.py` takes explicit checkpoint paths) | | 64 groups per model |
| P1, P2 | see their `REPRODUCIBILITY.md`; P2's GPT-2 Large, Pythia-410M, TinyLlama and Qwen3-1.7B runs need those checkpoints | | |

The post hoc random-circuit control reruns with `python posthoc/random_circuits.py run`, then `analyze`, in Study
2's folder; it skips recipients whose outputs exist. The head-level spread (`studies/05-head-confirmation/posthoc/head_spread.py`)
reruns the unpatched Study 3 test prompts on a GPU in about a minute.

Matching versions does not guarantee bitwise agreement on other hardware; each protocol's numerical tolerances
say what counts as a reproduction.

### Study 5 on Modal (optional, your own account)

Study 5 ran Qwen3-1.7B in float32 on Modal A10G GPUs, because the model does not fit a 6 GB GPU in float32. Its
stages run inside Modal from `studies/07-larger-model/modal_app.py`; the weights download inside the image at the
pinned revision with every digest asserted, and the image is built from this release's `pyproject.toml` and
`uv.lock`. With the Modal CLI installed and authenticated to your own account (`modal setup`):

```sh
cd studies/07-larger-model
export STUDY5_APP=donor-pooling-larger-model     # app and Volume name in your workspace (this is the default)
modal run modal_app.py::calibrate
modal run modal_app.py::development
modal run modal_app.py::test
modal run modal_app.py::replays
modal run modal_app.py::analyse
```

After each stage, `modal run modal_app.py::pull --stage <stage>` copies its small records back into `results/`;
the raw arrays stay on the Volume.

`run_stage.sh <stage>` starts one stage detached and retries when the connection fails before the stage starts.
Before each stage the app reads your billing report and stops once the study's apps have spent $10
(PROTOCOL.md, "Stopping"); `watch_billing.sh` repeats that check every 15 minutes during long stages. The
original run cost $6.39, plus $0.28 for the smoke test in `modal-smoke-test/` (A10G about $1.10 an hour). A rerun
from scratch also needs fresh freezes: the frozen records shipped here must not be overwritten.

## Tables and figures

All numbers in the manuscript are macros in `paper/generated/numbers.tex`, written by `paper/make_numbers.py` from
the saved outputs; tables that list many sites are written as whole files in `paper/generated/`.

| Manuscript | Source |
|---|---|
| Tables 1–10 and 12 | macros from `make_numbers.py`: Table 1 and Table 9 from Study 1 (`studies/03-grouping-and-circuit/token/results/analysis-test/`, `studies/04-reference-and-sweeps/results/retrospective-study3.json`); Tables 2–4, 10 and 12 from Study 2 (`results/sweep/`, `results/reference/`, `results/posthoc/`); Table 5 from Study 3; Table 6 from Study 4; Tables 7 and 8 from Study 5 (`results/analysis/`) |
| Table 11 | `paper/generated/reference-secondary.tex` |
| Table 13 | `paper/generated/error-discrimination.tex` |
| Table 14 | `paper/generated/sampled-donors.tex` |
| Table 15 | `paper/generated/misclassified-sites.tex` |
| Table 16 | `paper/generated/ioi-token-groups.tex` |
| Table 17 | `paper/generated/heads-flagged.tex` |
| Table 18 | `paper/generated/components-flagged.tex` |
| Table 19 | `paper/generated/large-heads-flagged.tex` |
| Figure 1 | a diagram drawn in the manuscript's source; no data |
| Figures 2–12 | `paper/make_figures.py`: `fig-partitions` (2), `fig-spread` (3), `fig-sweeps-sva` (4), `fig-sweeps-ioi` (5), `fig-agreement` (6), `fig-discrimination` (7), `fig-head-spread` (8), `fig-components` (9), `fig-weak-sites` (10), `fig-threshold` (11), `fig-heads` (12) |

The analyses behind those files are the studies' own scripts: `analyze*.py` in each study folder (frozen before
the test runs), Study 2's `retrospective_study3.py` (post hoc re-analysis of Study 1) and `curvature.py`, and the
post hoc scripts in `studies/04-reference-and-sweeps/posthoc/` and `studies/05-head-confirmation/posthoc/`.

## What is not included

- Model weights and tokenisers (download them; see Models).
- Study 1's raw token arrays, replays and development arrays (95 MB), and the raw per-group arrays and other
  outputs of P1 and P2 (about 1 GB);
  their saved analyses are included and the arrays can be regenerated.
- Study 5's raw arrays, which stayed on the cloud volume (records and digests of every array are included).
- Logs, the manuscript and its LaTeX source, rendered figures (regenerate them with `make figures`), working
  notes and review notes, and the original project's provenance history. Frozen files that were not shipped are
  listed with their SHA-256 in `RELEASE-CHANGES.md`.

## Changes from the original project

The release copies the project's files byte for byte, with three kinds of exception, all listed in
`RELEASE-CHANGES.md` with the original's SHA-256:

- paths that pointed into other folders on the original machine now use `MODELS_DIR` (default `models/`) or
  paths inside the release, and the Modal code no longer names the original workspace's apps;
- records had personal paths, machine details and the names of software tools in working records replaced by placeholders such as `<project>`, `<models>` and `[redacted]` (the assistants used are named under "Use of AI assistants" below);
- the verifiers' frozen-file checks accept exactly those documented changes (`tools/release_paths.py`).

`make check-freezes` shows, for every frozen file, whether it is unchanged, changed in a documented way, or
documented as not shipped.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs. The BLiMP subset in `submission-study/vendor/` is from BLiMP
(Warstadt et al., 2020; CC BY 4.0; see `submission-study/vendor/blimp-README.md`), and the checkpoints carry
their own licences.

## Study codes

The records and code name the studies by the codes they had while the work was done. Each code that occurs here,
and where that study is released:

| Code | Released in |
|---|---|
| AI-01 | `interp-answer-repair` |
| AI-03 | this repository |

## Use of AI assistants

As the preprint states:

> Language-model assistants were used throughout this work, under the author's
> direction, to help design the studies, to write and run the code for the experiments and analyses, to draft and
> revise the text and to check the references against their online records. All reported results come from the
> saved study outputs, and the numbers in the text were generated from those outputs or checked against them.

In the working records shipped here, the names of software tools were replaced by placeholders such as
`[redacted]`.

## Citation

Please cite the preprint (the link will be added here when it is posted) and, for the code and results, this
repository; [CITATION.cff](CITATION.cff) has the details.

```bibtex
@software{interp_donor_pooling_2026,
  author  = {Jeevash, Richard},
  title   = {{How donors are chosen and combined changes activation-patching conclusions: code and results}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Jeevash23/interp-donor-pooling}
}
```

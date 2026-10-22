# Reproducing the October donor grouping study

The review archive contains the complete new experiment: generated prompts, fixed inputs, raw response arrays, selected activation states, analyses, controls, failed calibration history, fresh-model replays and the revised manuscript. Model weights are external. The September six-model raw-response archive remains a separate supporting artifact; this archive contains the newly computed retrospective summary and its original input hashes.

## Verify and recompute without model weights

Use Python 3.12 and the recorded dependencies in `requirements.txt`. NumPy is enough for the independent verifier; statistical recomputation also uses SciPy, and figures use Matplotlib. Model inference additionally needs PyTorch, Transformers and Safetensors. The recorded inference build used CUDA 13.0 on a 6 GB NVIDIA GPU; matching package versions does not guarantee bitwise agreement on another device.

From this directory:

```sh
python scripts/reproduce.py verify
python scripts/reproduce.py analyze
python scripts/render_results.py
```

Verification checks both frozen input manifests and independently reconstructs the two primary recipient-level estimates and their adjusted bootstrap intervals from the raw response arrays. It also verifies circuit sampling weights and agreement indicators directly from saved predictions. Analysis runs the actual frozen analyzers. Derived JSON files may be rewritten; the circuit summary's creation timestamp will change. Raw arrays and frozen scientific inputs remain unchanged. For preservation-sensitive work, run these commands in a fresh extraction of the review archive.

`output/archive-integration.json` is an adjacent delivery record, rather than a member that hashes its own archive. It records a clean extraction, every manifest hash, exact recomputation of both token summaries, and circuit-summary equality excluding only its top-level creation timestamp. The independent verifier must pass in that extraction without model weights or access to the original project.

## Fresh model replays in a separate copy

Provide an existing local checkpoint directory containing the exact recorded files. The launcher checks every required file digest, copies the artifact to a new directory, removes only the copied prior replay outputs, and runs the frozen implementation. It does not download weights or modify the delivered measurements.

```sh
python scripts/reproduce.py token-replay --model gpt2-small --checkpoint /path/to/gpt2-small --output /tmp/new-gpt2-replay
python scripts/reproduce.py token-replay --model qwen3-0.6b --checkpoint /path/to/Qwen3-0.6B --output /tmp/new-qwen-replay
python scripts/reproduce.py circuit-replay --checkpoint /path/to/gpt2-small --output /tmp/new-circuit-replay
```

Every output path must be new and outside the source artifact tree. The four residual replay groups are 0, 16, 32 and 48; they include the stored dense states. The circuit replay uses the first four selected cases in each correctness stratum. Each replay reloads model weights in a fresh process, reruns per-case controls and compares saved metrics, states where available, and discrete predictions. It uses the admitted float32 CUDA batches (16 residual, nine circuit). The original calibration records stay bound to the frozen inputs. A numerical failure on another configuration must be reported, not resolved by loosening the recorded gate.

## Full execution and data regeneration

The frozen panel commands are documented in `token/README.md`, `token/docs/TRACEABILITY-AND-REPRODUCTION.md` and `circuit/README.md`. Their sequence is prepare, calibration, development, local freeze, test and replay; circuit screening and selection occur after its freeze and before circuit test. The original local study tree supplies portfolio-relative checkpoint paths. The portable replay launcher changes the runtime checkpoint mapping without editing the frozen files.

Exact new prompt regeneration additionally needs the historical prompt files used to exclude overlap. In the original project these remain under `<project>/data/` and `submission-study/data/`. They are intentionally not replaced by an empty exclusion set in this archive. The saved generation audits identify those files, and the pretest regeneration record confirms byte-for-byte equality for all three residual splits. Statistical recomputation and replay need only the generated inputs already included here.

The protocol is a local prospective plan, not a public preregistration. The questions were motivated by the September results. Corrections made during calibration/development are preserved. No test-driven model, template, sample-size or threshold expansion was performed.

## Build the paper

This is an existing multi-file TMLR LaTeX project using the official style files. From `paper/`, run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The released PDF is copied to `output/pdf/donor-grouping-revised-paper.pdf`. Its complete page-render review is recorded in `results/visual-qa.json`. Rendering or compiling a paper does not submit it. This remains an author-review draft; a public release needs human scientific review and an explicit decision about authorship, disclosures, licensing and venue formatting.

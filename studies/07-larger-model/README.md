# Study 5 (folder `07-larger-model`): AI-03's prespecified tests in Qwen3-1.7B, on Modal

**Status, 17 October 2026: complete, and reported as Study 5 (Section 9 and Appendix H) in the manuscript's sixth revision.** The author approved the protocol as written (`AUTHOR-APPROVAL.txt`). The protocol and predictions were hashed at 05:43:12 UTC, before any prompt was built (`data/protocol-freeze.json`). Code and prompts were frozen at 08:18:49 UTC, and three amendments followed (`AMENDMENTS.md`). Calibration, development and the cost decision were frozen at 09:59:53 UTC. Both competence gates passed, and all five panels ran on 64 agreement groups and 64 IOI prompts.
- **R1 replicates:** bias residual stream > MLP > attention layer > head, following spread.
- **R2:** replicates for agreement heads and at the IOI final token. At the IOI repeated subject it contradicts Study 3: two layer-0 heads lose their flags under mean ablation, and token grouping restores them.
- **R3 replicates** in both sweeps.

Four IOI replays exceeded the 10⁻³-logit tolerance (up to 1.6 × 10⁻³), and the protocol stopped the study for review. A post hoc bound shows this noise cannot change any flag, and the author accepted the results with the failure disclosed (`NOTES-AFTER-TEST.md`). The optional Qwen3-4B arm was not chosen. Details: [RESULTS.md](RESULTS.md).

In the manuscript this would be **Study 5**; the earlier studies are P1 (`../01-pilot`), P2 (`../../submission-study`), Study 1 (`../03-grouping-and-circuit`), Study 2 (`../04-reference-and-sweeps`), Study 3 (`../05-head-confirmation`) and Study 4 (`../06-component-confirmation`).

**Question.** The manuscript's first limitation is that the models are small. Do three of AI-03's prespecified tests hold in Qwen3-1.7B in float32?
1. Mean ablation's bias is ordered residual stream > MLP > attention layer > single head, following donor spread (Study 4).
2. Mean ablation leaves single-head flags unchanged at τ = 0.05 (Study 3).
3. Mean ablation misclassifies sites in residual-stream sweeps (Study 2).

**Where it runs.** Modal (cloud GPUs), because float32 Qwen3-1.7B does not fit the local machine's 6 GB GPU and downloads are limited. The weights download inside Modal. The raw arrays stay on the Modal Volume `ai03-study5`, and only code, prompts and small records cross the local connection.

## Files

| Path | Contents |
|---|---|
| `PROTOCOL.md`, `PREDICTIONS.md` | Frozen protocol and predictions |
| `AUTHOR-APPROVAL.txt`, `AMENDMENTS.md`, `NOTES-AFTER-TEST.md` | The approval; changes after a freeze; events after the test stage (replay failure, stage order) |
| `common.py`, `engine.py`, `stages.py` | Paths and constants; Qwen3 patching engines (study and reference); calibration, panel runners and gates |
| `prepare.py` | Prompt generation (tokenisation only), copied from Studies 3 and 4 with new seeds |
| `modal_app.py` | Modal app: image, Volume and one entry point per stage; reads the billing report before each stage |
| `analyze.py` | Frozen analyses of Studies 4, 3 and 2, ported (diff in `data/analysis-port-diff.json`), and the decision rules R1–R3 |
| `verify.py` | Independent check: arrays (on Modal) and freezes, separation, records and replays (on the local machine) |
| `freeze.py` | Protocol, input, amendment and stage-4 freezes |
| `test_cpu.py`, `test_analysis.py` | CPU code tests on a tiny random Qwen3 and on synthetic arrays; logs in `logs/` |
| `data/` | Prompts, generation audit, freezes |
| `RESULTS.md`, `write_results.py` | Results, written from the saved records |
| `results/` | Records pulled back from the Volume: calibration, development, cost decision, gates, per-row run records with array hashes, replays, analyses (`analysis/`), verification, billing logs, the post hoc replay-noise bound (`posthoc/`) |
| `posthoc/replay_noise.py` | Post hoc: could replay-sized noise move any effect across τ? |
| `run_stage.sh`, `watch_billing.sh` | Start a stage as a standalone process (retrying a failed connection); stop the app if spend reaches $10 |
| `logs/` | Code-test and stage logs |
| `provenance/README-at-protocol-freeze.md` | This README as it was hashed with the protocol |
| `modal-smoke-test/` | The smoke test of 17 October 2026 and its profiling addendum |

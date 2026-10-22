# Modal smoke test for Study 5 (17 October 2026)

**Passed.** On a Modal T4, AI-03's frozen code, imported unchanged, recomputed three saved results. Every margin matches the saved arrays within 5.3 × 10⁻⁵ logits (tolerance 10⁻³, the frozen replay tolerance of Studies 2–4). This is the first AI-03 computation outside the local machine. No Qwen3-1.7B or Qwen3-4B computation used a task prompt.

| Reproduced | Saved result | Largest margin deviation | Compute time on the T4 (local) |
|---|---|---:|---:|
| Study 4, first test group (`sva-test-sva-000`), GPT-2 Small: 360 site-directions × 81 patches, every component kind | `../../06-component-confirmation/results/test/sva-test-sva-000.npz` | 5.3 × 10⁻⁵ | 38 s (37 s) |
| Study 2, first Qwen3-0.6B agreement sweep test group: 280 sites × 89 patches | `../../04-reference-and-sweeps/results/sweep/sva-qwen3-0.6b/test/sva-test-sva-000.npz` | 3.0 × 10⁻⁵ | 142 s (146 s) |
| Study 3, Qwen3-0.6B accuracy gate on its 64 test groups | `../../05-head-confirmation/results/qwen-gate/gate.json` | 51 of 64 correct (0.796875), as saved; smallest margin 0.079 logits | 6 s |

The donor spreads match to a relative 6.0 × 10⁻⁵ wherever the saved value is at least 0.001; the entries below that are zero up to rounding and differ by at most 3.3 × 10⁻⁸. Write locality was exact in both patching runs, and Study 2's built-in identity patch was within 8.6 × 10⁻⁶. Before running, the frozen `check_checkpoint` confirmed every model file against the digests recorded before Study 1.

| | |
|---|---|
| GPUs | Tesla T4 (reproduction); NVIDIA L4, A10G and L40S (Qwen3 timing); A10G (profiling addendum) |
| Image | `modal.Image.debian_slim(python_version="3.12")` + `uv_sync(extras=["experiment"], frozen=True)` from AI-01's `pyproject.toml` and `uv.lock`, the environment AI-03's Makefile uses (AI-03 has no lock file of its own). It resolved to Python 3.12.10, torch 2.13.0+cu130 (CUDA 13.0), transformers 5.16.1, numpy 2.5.2, scipy 1.18.1, huggingface-hub 1.29.0, safetensors 0.8.0 and tokenizers 0.23.1. The local machine's recorded build has Python 3.12.14 and the same libraries. |
| Weights | Downloaded inside Modal at the image build, at the pinned revisions, with SHA-256 asserted: GPT-2 Small and Qwen3-0.6B (all seven files against `../../03-grouping-and-circuit/results/environment-and-inputs.json`), Qwen3-1.7B (`70d244cc…`; both shards against `../../../provenance/environment-2026-10-09.md`). Qwen3-1.7B's config, index and tokeniser files equal the local machine's copies, and its tokeniser files equal Qwen3-0.6B's. Qwen3-4B: config only (revision `1cfa9a72…`); its shard digests were read from the Hugging Face record. |
| Runtime | T4 function 255 s in total, including three model loads; the three timing functions ran alongside; whole run 282 s (05:11–05:17 UTC). Profiling addendum 191 s (05:21–05:25 UTC). |
| Cost | **$0.28** by Modal's billing report: smoke test $0.215 (T4 $0.042, L4 $0.062, A10G $0.023, L40S $0.058, CPU $0.020, memory $0.002, egress $0.008), profiling $0.064. $0.00 billed after credits. |
| Data | About 0.52 MB uploaded (code, the three input rows, AI-01's `pyproject.toml` and `uv.lock`); 0.23 MB received. The weights (about 6.1 GB) and Python packages were downloaded inside Modal, not through the local machine. |

## Qwen3 timing on random tokens

Float32, eager attention, deterministic algorithms, no TF32, no KV cache, 2 CPU cores; random token inputs only. Seconds per patched site with the frozen engines' one-site-per-call design (81 vectors at 11 tokens, as in Study 4; median of three):

| Model | L4 | A10G | L40S | Peak GPU memory |
|---|---:|---:|---:|---:|
| Qwen3-1.7B (pinned weights) | 0.735 | 0.377 | 0.233 | 7.4 GB |
| Qwen3-4B (random weights from its config) | 1.44 | — | 0.42 | 17.8 GB |

The time hardly changes with kind of site or layer. 73 vectors at 14 tokens (Study 3's IOI settings) take about the same time. Per site, the A10G is the cheapest of the three: about $1.2 × 10⁻⁴ against $1.6 × 10⁻⁴ (L4) and $1.3 × 10⁻⁴ (L40S).

The calibration checks of Studies 2–4 (passive, identity, duplicate donor, final block reproduces the donor) on the same random inputs: largest logit error 3.9 × 10⁻⁵ (L4), 4.3 × 10⁻⁵ (A10G) and 3.7 × 10⁻⁵ (L40S) for Qwen3-1.7B, all within 10⁻³; write locality exact. The bfloat16 run of P2 had identity errors up to 1.9 logits.

## Profiling addendum (`modal_profile.py`, A10G, random tokens)

At that speed a replication with 32 agreement groups and 32 IOI prompts would need about 10 A10G hours (about $12), so a profile was run before the protocol was written (`profile-records/profile.json`):

- Half the time of an 81-row forward pass (197 of 379 ms) is one matrix-vector kernel in the output layer. The model takes the last position as a strided slice, so the 151,936 × 2,048 output matrix is read once per row. A plain forward pass of 81 rows takes 0.040 s at 1 token and 0.38 s at 11 tokens.
- **Prefix caching** for a patch at the last position: the first 10 positions are computed once and only the last position runs per vector. The logits agree with full recomputation within 9.3 × 10⁻⁵ (residual stream) and 5.5 × 10⁻⁵ (head). Time: 0.047 s for 81 rows, or 0.37 s for 1,296 rows (16 sites in one batch, 0.023 s per site): 16 times faster than one site per call.
- Several sites in one batch without caching gain nothing (16 heads: 5.9 s against 6.1 s); their logits agree within 6.2 × 10⁻⁵.
- Deterministic algorithms cost nothing measurable.

## Files

`modal_smoke.py` (Modal app), `remote_driver.py` (runs one frozen computation per process in the container), `qwen_timing.py` (timing and random-input checks), `compare_smoke.py` (local comparison, project environment, CPU), `modal_profile.py` and `qwen_profile.py` (profiling addendum), `records/` (`remote-record.json`, `modal-*.npz` and `modal-*.json`, `comparison.json`, `billing-report.json`), `profile-records/profile.json`, `run.log`, `profile-run.log` and the start and end times.

`records/comparison-v1.json` is the first output of `compare_smoke.py`. It divided spread differences by the saved value with a floor of 10⁻¹², so it reported a relative spread error of 8.45 from an entry saved as 1.8 × 10⁻¹⁰ (Modal 1.7 × 10⁻⁹). The script now uses a floor of 0.001 and reports the entries below it by absolute difference. The margins and the pass were the same in both versions.

Both apps are stopped; nothing is deployed or kept warm.

Reproduce from this folder:

```bash
~/.local/bin/modal run modal_smoke.py
python -B compare_smoke.py
~/.local/bin/modal run modal_profile.py
```

`modal_smoke.py` refuses to run while `records/` exists, `compare_smoke.py` refuses to overwrite `records/comparison.json`, and `modal_profile.py` refuses to run while `profile-records/` exists.

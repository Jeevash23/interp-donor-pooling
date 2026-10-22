# Checkpoint sources

| Key | Public model identifier | Revision |
|---|---|---|
| gpt2-small | openai-community/gpt2 | 607a30d783dfa663caf39e06633721c8d4cfcd7e |
| gpt2-medium | openai-community/gpt2-medium | 6dcaa7a952f72f9298047fd5137cd6e4f05f41da |
| gpt2-large | openai-community/gpt2-large | 32b71b12589c2f8d625668d2335a01cac3249519 |
| pythia-410m | EleutherAI/pythia-410m-deduped | c4fc8d586d62df497f1f9b69d66d3ca419992d3e |
| tinyllama-1.1b | TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | 59f6f375b26bde864a6ca194a9a3044570490064 |
| qwen3-0.6b | Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca |
| qwen3-1.7b (supplementary) | Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |

Revisions come from the existing local cache metadata and are checked against actual checkpoint file hashes in `data/checkpoints.json`. The primary panel runs in float32, including checkpoints distributed at a lower storage precision; converting a checkpoint to float32 does not undo its original weight rounding. The supplementary Qwen3-1.7B run uses bfloat16 for memory feasibility. No weights are trained or downloaded during this study.

The Qwen checkpoints are post-trained releases, evaluated as raw text completion. TinyLlama is the indicated base intermediate checkpoint. Architecture and training differences limit comparisons; nominal size alone is not an experimental treatment.

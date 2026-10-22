# Fixed checkpoint sources

The new experiments reuse two unchanged checkpoints from the September study. Every required local file was rehashed before measurement and matched its recorded SHA-256. The full per-file digests and revision identifiers are in `results/environment-and-inputs.json` and the circuit generation audit.

| Checkpoint | Upstream source | Fixed revision | Evaluation |
|---|---|---|---|
| GPT-2 Small | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | Float32 native causal completion; residual and circuit assays |
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | Float32 raw-text completion without chat/reasoning generation; residual assay |

The archive does not redistribute model weights. The residual generator qualifies words jointly with both tokenizers and prepends the configured beginning-token/BOS/EOS fallback recorded in the encoded inputs. The circuit generator uses GPT-2 without a BOS token, matching the original authors' reference convention. These input conventions are part of the experiment and must not be silently unified in a reproduction.

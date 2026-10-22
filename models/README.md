# Model checkpoints

No weights are included. The studies read checkpoints from this folder, or from the folder named by the
`MODELS_DIR` environment variable, one subfolder per model:

| Subfolder | Hugging Face repository | Revision | Size | Used by |
|---|---|---|---|---|
| `gpt2-small` | `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 0.55 GB | P1, P2, Studies 1–4; `make test` reads only its tokeniser |
| `Qwen3-0.6B` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 1.52 GB | P2, Studies 1–3 |
| `gpt2-medium` | `openai-community/gpt2-medium` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | 1.52 GB | P1, P2 |
| `gpt2-large` | `openai-community/gpt2-large` | `32b71b12589c2f8d625668d2335a01cac3249519` | 3.25 GB | P2 |
| `pythia-410m-deduped` | `EleutherAI/pythia-410m-deduped` | `c4fc8d586d62df497f1f9b69d66d3ca419992d3e` | 0.91 GB | P2 |
| `tinyllama-1.1b-base` | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `59f6f375b26bde864a6ca194a9a3044570490064` | 4.40 GB | P2 |
| `Qwen3-1.7B` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 4.08 GB | P2's bfloat16 extension; Study 5 downloads it inside its Modal image instead |

`models.json` lists every file of every checkpoint with the SHA-256 recorded before the studies ran.

```sh
python tools/get_models.py --list                    # the table above
python tools/get_models.py gpt2-small qwen3-0.6b     # what Studies 1-4 need (make models does this)
python tools/get_models.py --check                   # check what is already here, without network access
```

`get_models.py` uses `huggingface_hub` (part of the environment) and stops if any file's digest differs from the
record. The same files can be fetched by hand, for example
`hf download openai-community/gpt2 --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --local-dir models/gpt2-small`,
followed by `python tools/get_models.py --check`. Before each model stage the study code checks the digests
again against `studies/03-grouping-and-circuit/results/environment-and-inputs.json` (recorded before Study 1)
and refuses to run on a different file.

The checkpoints are distributed under their own licences (MIT for GPT-2, Apache 2.0 for Qwen3, Pythia and
TinyLlama); see each repository.

"""Shared paths, constants and file helpers for Study 5 (Qwen3-1.7B on Modal). No torch import here."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1] if len(ROOT.parents) > 1 else ROOT
PORTFOLIO = PROJECT.parent
MODELS = Path(os.environ.get("MODELS_DIR", PROJECT / "models"))   # release: see models/README.md

MODEL = dict(name="qwen3-1.7b", repo="Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
             sha256={"model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
                     "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
                     # equal to the local machine's copies (smoke test, 17 October 2026)
                     "config.json": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
                     "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
                     "model.safetensors.index.json": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
                     "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
                     "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                     "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                     "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"})
REMOTE_MODEL = "/root/models/qwen3-1.7b"
# Tokenisers on the local machine (tokenisation only): Qwen3-1.7B's config and tokeniser files, kept without weights
# since the 8 October cleanup; they equal the files Modal downloads (smoke test) and Qwen3-0.6B's tokeniser.
LOCAL_TOKENIZERS = {
    "gpt2-small": MODELS / "gpt2-small",
    "qwen3": MODELS / "Qwen3-1.7B",
}

KINDS = ("resid", "attn", "mlp", "head")
LOGIT_TOL, PROB_TOL = 1e-3, 2e-5
MAX_ROWS = 1296
MIX = [[(s, (r + s) % 8) for s in range(8)] for r in range(8)]
NOUN = [[(s, n) for s in range(8)] for n in range(8)]
ADJ = [[(s, n) for n in range(8)] for s in range(8)]


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# Compact rows for the remote stages: only what the runners need (Qwen3-1.7B encodings), so little is uploaded.
def compact_sva(row):
    p = row["encoded"][MODEL["name"]]
    return {"id": row["id"], "template": row["template"], "class": row["class"], "grid": row["grid"],
            "ids": [x["ids"] for x in p], "adj": p[0]["adj"], "anchor": p[0]["anchor"], "end": p[0]["end"],
            "good_id": p[0]["good_id"], "bad_id": p[0]["bad_id"]}


def compact_ioi(row):
    r = row["recipient"]
    return {"id": row["id"], "template": row["template"], "order": row["order"],
            "recipient": {k: r[k] for k in ("ids", "end", "s2", "target_id", "contrast_id")},
            "donors": [d["ids"] for d in row["donors"]], "s2_tokens": row["s2_tokens"]}

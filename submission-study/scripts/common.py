from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
MODELS = Path(os.environ.get("MODELS_DIR", ROOT.parent / "models"))   # release: see models/README.md
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
MODEL_PATHS = {
    "gpt2-small": MODELS/"gpt2-small",
    "gpt2-medium": MODELS/"gpt2-medium",
    "gpt2-large": MODELS/"gpt2-large",
    "pythia-410m": MODELS/"pythia-410m-deduped",
    "tinyllama-1.1b": MODELS/"tinyllama-1.1b-base",
    "qwen3-0.6b": MODELS/"Qwen3-0.6B",
    "qwen3-1.7b": MODELS/"Qwen3-1.7B",
}
PRIMARY_MODELS = list(MODEL_PATHS)[:6]
TASKS = ["ioi", "sva", "kv"]
METRICS = ["margin", "probability", "top1", "pair_correct", "log_probability"]
def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()
def utc():return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
def freeze_check():
    f=read(ROOT/"data/freeze.json")
    from relocation import frozen_matches
    bad=[p for p,h in f["files"].items() if not frozen_matches(ROOT,p,h)]
    if bad:raise RuntimeError(f"Frozen inputs changed: {bad}")

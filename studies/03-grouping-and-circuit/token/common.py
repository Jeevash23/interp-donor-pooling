"""Shared definitions for the new, isolated donor-label policy experiment."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[3]
MODELS = Path(os.environ.get("MODELS_DIR", ROOT.parents[2] / "models"))   # release: see models/README.md
STUDY = ROOT.parents[2] / "submission-study"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
MODEL_PATHS = {
    "gpt2-small": MODELS / "gpt2-small",
    "qwen3-0.6b": MODELS / "Qwen3-0.6B",
}
METRICS = ["margin", "probability", "top1", "pair_correct", "log_probability"]
DOSES = [1, 2, 4, 8]
ROTATIONS = 8
POSITIONS = ["anchor", "end"]

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()

def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def freeze_check():
    record = read(ROOT / "data/freeze.json")
    changed = [p for p, h in record["files"].items() if sha(ROOT / p) != h]
    if changed:
        raise RuntimeError(f"Frozen inputs changed: {changed}")
    checkpoint_record = read(ROOT.parent / "results/environment-and-inputs.json")
    changed_models = [f"{name}/{filename}" for name, files in checkpoint_record["models"].items() if name in MODEL_PATHS for filename, metadata in files.items() if sha(MODEL_PATHS[name] / filename) != metadata["sha256"]]
    if changed_models:
        raise RuntimeError(f"Frozen checkpoint files changed: {changed_models}")
    return record

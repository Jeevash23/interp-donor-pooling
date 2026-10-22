"""Shared paths, checkpoint checks and file helpers for Study 4."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PORTFOLIO = PROJECT.parent
MODELS = Path(os.environ.get("MODELS_DIR", PROJECT / "models"))   # release: see models/README.md
OCTOBER = PROJECT / "studies/03-grouping-and-circuit"
if not OCTOBER.exists():
    OCTOBER = PROJECT / "followups/2026-10-pooling-consequences"

MODEL_PATHS = {
    "gpt2-small": MODELS / "gpt2-small",
    "qwen3-0.6b": MODELS / "Qwen3-0.6B",
}
# SHA-256 digests recorded before the October experiments (results/environment-and-inputs.json).
CHECKPOINT_RECORD = OCTOBER / "results/environment-and-inputs.json"


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


def check_checkpoint(name):
    """Every model file must match the digest recorded for the October study."""
    record = read(CHECKPOINT_RECORD)["models"][name]
    bad = [f for f, meta in record.items() if sha(MODEL_PATHS[name] / f) != meta["sha256"]]
    if bad:
        raise RuntimeError(f"{name}: checkpoint files differ from the recorded digests: {bad}")
    return {f: meta["sha256"] for f, meta in record.items()}


def freeze_check():
    path = ROOT / "data/freeze.json"
    if not path.exists():
        raise RuntimeError("Test runs require data/freeze.json")
    record = read(path)
    changed = [p for p, h in record["files"].items() if sha(ROOT / p) != h]
    if changed:
        raise RuntimeError(f"Frozen files changed after the freeze: {changed}")
    return record


def prior_prompt_strings():
    """Every prompt text used by Studies 1-3, for exclusion from the new data."""
    files = [PROJECT / "data/calibration.json", PROJECT / "data/evaluation.json"]
    for base in [PROJECT / "studies/01-pilot", PROJECT / "studies/02-six-model-audit", PROJECT / "submission-study"]:
        files += [base / "data" / f"{s}.json" for s in ["calibration", "evaluation", "development", "test", "blimp"]]
    files += [OCTOBER / "token/data" / f"{s}.json" for s in ["calibration", "development", "test"]]
    files += [OCTOBER / "circuit/data" / f"{s}.json" for s in ["calibration", "development", "screen"]]
    seen, used = set(), []
    texts = set()

    def descend(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("text", "prefix") and isinstance(value, str):
                    texts.add(value)
                elif key != "encoded":
                    descend(value)
        elif isinstance(obj, list):
            for value in obj:
                descend(value)

    for path in files:
        real = path.resolve()
        if real in seen or not path.exists():
            continue
        seen.add(real)
        used.append(str(path.relative_to(PROJECT)))
        descend(read(path))
    if not any("token/data/test.json" in p for p in used) or not any("circuit/data/screen.json" in p for p in used):
        raise FileNotFoundError("October prompt files are required for exclusion")
    return texts, used

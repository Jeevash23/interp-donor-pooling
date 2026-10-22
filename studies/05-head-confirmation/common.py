"""Shared paths, checkpoint checks and file helpers for the head-level confirmation study."""
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
MODEL_PATHS = {
    "gpt2-small": MODELS / "gpt2-small",
    "qwen3-0.6b": MODELS / "Qwen3-0.6B",
}
CHECKPOINT_RECORD = PROJECT / "studies/03-grouping-and-circuit/results/environment-and-inputs.json"


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
    record = read(CHECKPOINT_RECORD)["models"][name]
    bad = [f for f, meta in record.items() if sha(MODEL_PATHS[name] / f) != meta["sha256"]]
    if bad:
        raise RuntimeError(f"{name}: checkpoint files differ from the recorded digests: {bad}")
    return {f: meta["sha256"] for f, meta in record.items()}


def freeze_check():
    path = ROOT / "data/freeze.json"
    if not path.exists():
        raise RuntimeError("test runs require data/freeze.json")
    changed = [p for p, h in read(path)["files"].items() if sha(ROOT / p) != h]
    if changed:
        raise RuntimeError(f"frozen files changed after the freeze: {changed}")


def prior_prompt_strings():
    """Every prompt text used by any earlier AI-03 study (pilots, Study 1, Study 2)."""
    studies = PROJECT / "studies"
    files = sorted((studies / "01-pilot/data").glob("*.json"))
    files += sorted((PROJECT / "submission-study/data").glob("*.json"))
    files += sorted((studies / "03-grouping-and-circuit/token/data").glob("*.json"))
    files += sorted((studies / "03-grouping-and-circuit/circuit/data").glob("*.json"))
    files += sorted((studies / "04-reference-and-sweeps/data").glob("*.json"))
    texts, used, seen = set(), [], set()

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
        if real in seen:
            continue
        seen.add(real)
        try:
            descend(read(path))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        used.append(str(path.relative_to(PROJECT)))
    banks = studies / "04-reference-and-sweeps/results/reference/donor-banks-selected.json.gz"
    if banks.exists():
        import gzip
        for entry in json.load(gzip.open(banks, "rt"))["banks"].values():
            for donors in entry.values():
                texts.update(donors)
        used.append(str(banks.relative_to(PROJECT)))
    return texts, used

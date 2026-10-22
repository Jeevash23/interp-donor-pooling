"""Profiling addendum to the smoke test: where the time goes in float32 Qwen3-1.7B patching on an A10.

Random token inputs only, no task prompts. Uses the smoke test's image (same layers, so nothing is rebuilt
or downloaded again) plus qwen_profile.py. Run from this folder:

    modal run modal_profile.py

Writes profile-records/profile.json and refuses to run if that folder exists.
"""
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from modal_smoke import HERE, RESOURCES, WEIGHTS, _environment, image

image = image.add_local_file(HERE / "qwen_profile.py", "/root/smoke/qwen_profile.py").add_local_python_source("modal_smoke")
app = modal.App("donor-pooling-larger-model-profile", image=image)


@app.function(gpu="A10", timeout=900, **RESOURCES)
def profile_a10() -> bytes:
    import sys
    sys.path.insert(0, "/root/smoke")
    import torch
    from transformers import AutoModelForCausalLM
    import qwen_profile
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(WEIGHTS["qwen3-1.7b"]["dest"], local_files_only=True, dtype=torch.float32,
                                                 attn_implementation="eager").to("cuda").eval()
    load = time.perf_counter() - t0
    out = qwen_profile.run_profile(model)
    return pickle.dumps(dict(environment=_environment(), load_seconds=load, seconds=time.perf_counter() - t0, **out))


@app.local_entrypoint()
def main():
    out_dir = HERE / "profile-records"
    assert not out_dir.exists(), out_dir
    t0 = time.perf_counter()
    raw = profile_a10.remote()
    wall = time.perf_counter() - t0
    result = pickle.loads(raw)
    result["environment"].pop("build_record")
    out_dir.mkdir()
    (out_dir / "profile.json").write_text(json.dumps(dict(
        purpose="Profiling addendum to the smoke test: float32 Qwen3-1.7B on an A10, random token inputs only",
        client_wall_seconds=wall, received_bytes=len(raw),
        uploaded_bytes=sum((HERE / n).stat().st_size for n in ("qwen_profile.py", "modal_smoke.py", "modal_profile.py")),
        utc=datetime.now(timezone.utc).isoformat(), **result), indent=2, default=str) + "\n")
    print(json.dumps(dict(wall=wall, multi_site=result["multi_site"], prefix_cache=result["prefix_cache"])))

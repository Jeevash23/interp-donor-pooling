"""Runs one frozen AI-03 computation inside the Modal container (called by modal_smoke.py).

    python remote_driver.py <task> <study folder> <input.json> <output>

The study folder goes first on sys.path, so `common`, `engine(s)` and `run*` are that study's frozen
modules, imported unchanged. Each task runs in its own process because the studies reuse module names.

  s4    Study 4 (folder 06): run.run_group on one test group (GPT-2 Small, every component kind)
  s2    Study 2 (folder 04): run_sweep.sva_group on one Qwen3-0.6B agreement sweep test group
  gate  Study 3 (folder 05): the Qwen3-0.6B accuracy gate, the body of run.gate on the given prompts
"""
import json
import sys
import time

task, folder, source, target = sys.argv[1:5]
sys.path.insert(0, folder)

import numpy as np  # noqa: E402

out = {}
started = time.perf_counter()
if task == "s4":
    from common import check_checkpoint
    from engine import ComponentEngine
    from run import run_group
    out["checkpoint_sha256"] = check_checkpoint("gpt2-small")
    engine = ComponentEngine("cuda", 128)
    out["load_seconds"] = time.perf_counter() - started
    row = json.load(open(source))
    t0 = time.perf_counter()
    data, worst = run_group(engine, row)
    out["group_seconds"] = time.perf_counter() - t0
    out["audit"] = worst
    np.savez_compressed(target, **data)
elif task == "s2":
    from common import check_checkpoint
    from engines import ResidualEngine
    from run_sweep import sva_group
    out["checkpoint_sha256"] = check_checkpoint("qwen3-0.6b")
    engine = ResidualEngine("qwen3-0.6b", "cuda", 96)
    out["load_seconds"] = time.perf_counter() - started
    row = json.load(open(source))
    t0 = time.perf_counter()
    data, worst = sva_group(engine, row, check=True)
    out["group_seconds"] = time.perf_counter() - t0
    out["audit"] = worst
    np.savez_compressed(target, **data)
elif task == "gate":
    from common import check_checkpoint
    from engine import HeadPatchEngine
    from run import margin
    out["checkpoint_sha256"] = check_checkpoint("qwen3-0.6b")
    engine = HeadPatchEngine("qwen3-0.6b", "cuda", 128)
    out["load_seconds"] = time.perf_counter() - started
    rows = json.load(open(source))
    t0 = time.perf_counter()
    margins = []
    for row in rows:                      # as in run.gate: the first clean prompt of each group
        p = row["prompt"]
        logits, _ = engine.capture([p["ids"]], [p["end"]])
        margins.append(float(margin(logits, p["good_id"], p["bad_id"])[0]))
    out["gate_seconds"] = time.perf_counter() - t0
    out["ids"] = [row["id"] for row in rows]
    out["margins"] = margins
    out["clean_two_verb_accuracy"] = float(np.mean([m > 0 for m in margins]))
else:
    raise SystemExit(f"unknown task {task}")
out["environment"] = engine.environment()
json.dump(out, open(target + ".json", "w"))
print(json.dumps({"task": task, "seconds": time.perf_counter() - started}), flush=True)

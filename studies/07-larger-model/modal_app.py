"""Modal app for Study 5 (Qwen3-1.7B, float32, A10G). Run from this folder with the Modal CLI:

    modal run modal_app.py::calibrate      # stage 2 (after the input freeze)
    modal run modal_app.py::development    # stage 3, then the cost decision (results/cost-decision.json)
    modal run modal_app.py::test           # stages 5-6 (after the stage-4 freeze): gates, then the panels
    modal run modal_app.py::replays        # stage 7
    modal run modal_app.py::analyse        # stage 8: frozen analysis and the array part of verify.py

Raw arrays go to the Modal Volume named by STUDY5_APP (default `donor-pooling-larger-model`); small records come back to results/. Weights download inside
the image build at the pinned revision with every digest asserted. Before each stage the billing report is read,
and no stage starts once Study 5's apps have spent $10 (PROTOCOL.md, "Stopping").
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

# Duplicated from common.py because the image build imports this file before the study code is mounted;
# the local entrypoints assert that the two agree.
REPO, REVISION, REMOTE_MODEL = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "/root/models/qwen3-1.7b"
DIGESTS = {"model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
           "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
           "config.json": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
           "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
           "model.safetensors.index.json": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
           "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
           "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
           "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
           "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"}
CODE = ["common.py", "engine.py", "stages.py", "analyze.py", "verify.py"]
STOP_USD, RULE_USD = 10.0, 7.0
RATES = {"A10G": 1.10, "L4": 0.80, "cpu_core": 0.0473, "memory_gib": 0.008}       # dollars per hour (modal billing rates)
CPU_CORES, MEMORY_MIB = 2.0, 12288


def download_weights():
    from huggingface_hub import hf_hub_download
    Path(REMOTE_MODEL).mkdir(parents=True, exist_ok=True)
    record = {}
    for f, want in DIGESTS.items():
        hf_hub_download(REPO, f, revision=REVISION, local_dir=REMOTE_MODEL)
        with open(Path(REMOTE_MODEL) / f, "rb") as stream:
            got = hashlib.file_digest(stream, "sha256").hexdigest()
        assert got == want, (f, got)
        record[f] = dict(sha256=got, bytes=(Path(REMOTE_MODEL) / f).stat().st_size)
    (Path(REMOTE_MODEL) / "build-record.json").write_text(json.dumps(dict(repo=REPO, revision=REVISION, files=record), indent=2))


if modal.is_local():
    HERE = Path(__file__).resolve().parent
    AI01 = HERE.parents[1]   # release: the release root holds pyproject.toml and uv.lock
    sys.path.insert(0, str(HERE))
    from common import MODEL, compact_ioi, compact_sva, dump, read          # noqa: E402
    assert MODEL["repo"] == REPO and MODEL["revision"] == REVISION and MODEL["sha256"] == DIGESTS
else:
    HERE = AI01 = Path("/nonexistent")

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(uv_project_dir=str(AI01), extras=["experiment"], frozen=True)
         .run_function(download_weights)
         .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))
for name in CODE:
    image = image.add_local_file(HERE / name, f"/root/s5/{name}")
APP_NAME = os.environ.get("STUDY5_APP", "donor-pooling-larger-model")   # release: app and Volume name
app = modal.App(APP_NAME, image=image, tags={"study": "07-larger-model"})
volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)
VOL = "/vol"
GPU_OPTS = dict(timeout=3600, cpu=CPU_CORES, memory=MEMORY_MIB, scaledown_window=10, volumes={VOL: volume})


# ======================================================================================== remote side
def _engine():
    sys.path.insert(0, "/root/s5")
    from engine import Qwen3Engine
    t0 = time.perf_counter()
    engine = Qwen3Engine(REMOTE_MODEL, "cuda")
    env = engine.environment()
    env.update(load_seconds=time.perf_counter() - t0,
               build_record=json.loads((Path(REMOTE_MODEL) / "build-record.json").read_text()))
    return engine, env


def _npz_bytes(data):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **data)
    return buffer.getvalue()


def _panels(task):
    """Run panels on rows; arrays and per-row records go to the Volume. Development and test resume over rows already
    saved whose arrays match their records (Amendment 2); replays never overwrite."""
    import stages
    engine, env = _engine()
    stage, mode = task["stage"], task.get("mode", "study")
    base = Path(VOL) / stage
    stop = Path(VOL) / "test" / "STOP"
    records = []
    for row in task["rows"]:
        for panel in task["panels"]:
            if stage == "test":
                volume.reload()
                if stop.exists():
                    raise RuntimeError(f"stopped: {stop.read_text()}")
            path = base / panel / f"{row['id']}.npz"
            meta = base / panel / f"{row['id']}.json"
            if path.exists():
                if stage not in ("test", "development"):
                    raise RuntimeError(f"refusing to overwrite {path}")
                rec = json.loads(meta.read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
                    raise RuntimeError(f"saved array does not match its record: {path}")
                records.append(dict(rec, resumed=True))
                continue
            data, rec = stages.RUNNERS[panel](engine, row, mode=mode)
            rec.update(panel=panel, id=row["id"], stage=stage, mode=mode, gpu=env["gpu"], finished_utc=stages.utc())
            path.parent.mkdir(parents=True, exist_ok=True)
            if not rec["passed"]:
                meta.with_suffix(".failed.json").write_text(json.dumps(rec, indent=2))
                if stage == "test":
                    stop.write_text(f"{panel} {row['id']} failed a built-in check")
                volume.commit()
                raise RuntimeError(f"{panel} {row['id']} failed a built-in check: {rec['max_check_error']}, {rec['audit']}")
            blob = _npz_bytes(data)
            path.write_bytes(blob)
            rec["sha256"] = hashlib.sha256(blob).hexdigest()
            rec["bytes"] = len(blob)
            meta.write_text(json.dumps(rec, indent=2))
            volume.commit()
            records.append(rec)
            print(json.dumps({k: rec[k] for k in ("stage", "panel", "id", "seconds", "max_check_error")}), flush=True)
    return dict(environment=env, records=records)


def _job(task):
    sys.path.insert(0, "/root/s5")
    if task["kind"] == "panels":
        out = _panels(task)
    elif task["kind"] == "calibrate":
        import stages
        engine, env = _engine()
        t0 = time.perf_counter()
        out = dict(environment=env, calibration=stages.calibrate(engine, task["sva"], task["ioi"]),
                   seconds=time.perf_counter() - t0)
    elif task["kind"] == "gates":
        saved = Path(VOL) / task.get("record_dir", "gates") / "gates.json"
        volume.reload()
        if saved.exists():                                     # measured once; a resumed stage reads the saved gates
            return saved.read_bytes()
        import stages
        engine, env = _engine()
        out = dict(environment=env, gates=stages.gates(engine, task["sva"], task["ioi"]))
    else:
        raise ValueError(task["kind"])
    for name in ("calibration", "gates"):
        if name in out:
            target = Path(VOL) / task.get("record_dir", name) / f"{name}.json"
            if target.exists():
                raise RuntimeError(f"refusing to overwrite {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(out, indent=2))
            volume.commit()
    return json.dumps(out).encode()


@app.function(gpu="A10", **GPU_OPTS, max_containers=4)
def gpu_job(task: dict) -> bytes:
    return _job(task)


@app.function(gpu="A10", **GPU_OPTS, max_containers=2, single_use_containers=True)
def gpu_job_fresh(task: dict) -> bytes:
    return _job(task)


@app.function(gpu="L4", **GPU_OPTS, max_containers=1, single_use_containers=True)
def gpu_job_l4(task: dict) -> bytes:
    return _job(task)


@app.function(cpu=4.0, memory=16384, timeout=3600, scaledown_window=10, volumes={VOL: volume}, max_containers=1)
def cpu_job(task: dict) -> bytes:
    sys.path.insert(0, "/root/s5")
    volume.reload()
    out = {}
    if task["kind"] == "analyse":
        import analyze
        stage = task["stage"]
        meta = Path(VOL) / f"meta-{stage}.json"
        meta.write_text(json.dumps(task["meta"]))
        out_dir = Path(VOL) / f"analysis-{stage}"
        if (out_dir / f"decisions-{stage}.json").exists():
            raise RuntimeError(f"refusing to overwrite the {stage} analysis")
        analyze.analyze(stage, str(Path(VOL) / stage), str(out_dir), str(meta))
        if task.get("verify"):
            import verify
            verify.arrays(str(Path(VOL) / stage), str(out_dir), str(meta), str(out_dir / "verification-arrays.json"))
        volume.commit()
        out["files"] = {p.name: p.read_text() for p in sorted(out_dir.iterdir()) if p.suffix in (".json", ".csv")}
    elif task["kind"] == "replays":
        import numpy as np
        rep = {}
        for kind in task["replay_stages"]:
            for path in sorted((Path(VOL) / kind).glob("*/*.npz")):
                saved = Path(VOL) / "test" / path.parent.name / path.name
                with np.load(path) as a, np.load(saved) as b:
                    rep[f"{kind}/{path.parent.name}/{path.stem}"] = max(
                        float(np.nanmax(np.abs(a[k] - b[k]))) for k in a.files if k.startswith("margins"))
        out["replays"] = rep
    elif task["kind"] == "records":
        out["files"] = {str(p.relative_to(VOL)): p.read_text() for p in sorted(Path(VOL).glob(task["pattern"]))}
    return json.dumps(out).encode()


def _gpu_records(stage, panels, rows, mode="study"):
    return dict(kind="panels", stage=stage, panels=panels, rows=rows, mode=mode)


@app.function(cpu=0.25, memory=1024, timeout=6 * 3600, volumes={VOL: volume}, max_containers=1)
def orchestrate(task: dict) -> str:
    """Runs one stage inside Modal (Amendment 2), so that a dropped local connection cannot stop it. Writes
    orchestration/<stage>.json on the Volume when the stage ends, with its status and the per-row records."""
    stage = task["stage"]
    summary = dict(stage=stage, started_utc=datetime.now(timezone.utc).isoformat(), status="running")
    target = Path(VOL) / "orchestration" / f"{stage}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2))
    volume.commit()
    try:
        if stage == "development":
            a = gpu_job.spawn(_gpu_records("development", ["A", "C1"], task["sva"]))
            b = gpu_job.spawn(_gpu_records("development", ["B1", "B2", "C2"], task["ioi"]))
            ra, rb = json.loads(a.get()), json.loads(b.get())
            cpu_job.remote(dict(kind="analyse", stage="development", meta=task["meta"]))
            gpu_job.remote(dict(kind="gates", sva=task["sva"], ioi=task["ioi"], record_dir="development-accuracy"))
            summary.update(records=ra["records"] + rb["records"], environment=ra["environment"])
        elif stage == "test":
            g = json.loads(gpu_job.remote(dict(kind="gates", sva=task["sva"], ioi=task["ioi"])))["gates"]
            tasks = []
            if g["agreement"]["passed"]:
                tasks += [_gpu_records("test", ["A", "C1"], task["sva"][i:i + 16]) for i in range(0, len(task["sva"]), 16)]
            if g["ioi"]["passed"]:
                tasks += [_gpu_records("test", ["B1", "B2", "C2"], task["ioi"][i:i + 8]) for i in range(0, len(task["ioi"]), 8)]
            failures, records, envs = [], [], []
            for result in gpu_job.map(tasks, return_exceptions=True):
                if isinstance(result, Exception):
                    failures.append(repr(result))
                    continue
                out = json.loads(result)
                records += out["records"]
                envs.append(out["environment"])
            volume.reload()
            summary.update(gates=dict(agreement=g["agreement"]["passed"], ioi=g["ioi"]["passed"]), failures=failures,
                           records=records, environment=envs[0] if envs else None,
                           agreement_groups=len(task["sva"]), ioi_prompts=len(task["ioi"]))
            if failures:
                raise RuntimeError(f"test stage stopped: {failures}")
        elif stage == "replays":
            calls = [(fn, t, fn.spawn(t)) for fn, t in
                     ([(gpu_job_fresh, t) for t in task["fresh"]] + [(gpu_job_l4, t) for t in task["l4"]])]
            records = []
            for fn, t, call in calls:
                records += json.loads(call.get())["records"]
            rep = json.loads(cpu_job.remote(dict(kind="replays", replay_stages=["replay-study", "replay-reference", "replay-l4"])))["replays"]
            primary = {k: v for k, v in rep.items() if not k.startswith("replay-l4")}
            l4 = [v for k, v in rep.items() if k.startswith("replay-l4")]
            summary.update(records=records, replay_verification=dict(
                created_utc=datetime.now(timezone.utc).isoformat(), tolerance=1e-3, errors=rep,
                maximum_error_primary=max(primary.values()), maximum_error_l4=max(l4) if l4 else None,
                passed=max(primary.values()) <= 1e-3))
        elif stage == "analyse":
            cpu_job.remote(dict(kind="analyse", stage="test", meta=task["meta"], verify=True))
        summary["status"] = "finished"
    except Exception as error:                                                   # noqa: BLE001
        summary.update(status="failed", error=repr(error))
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary["status"]


# ======================================================================================== local side
def _billing():
    """Study 5's spend so far by Modal's billing report (every app whose name starts with APP_NAME this month)."""
    for attempt in range(6):                                  # the local connection sometimes drops for a minute
        proc = subprocess.run([os.environ.get("MODAL_CLI", "modal"), "billing", "report", "--for", "this month",
                               "--show-resources", "--json"], capture_output=True, text=True)
        if proc.returncode == 0:
            break
        time.sleep(20)
    else:
        raise SystemExit(f"could not read the billing report: {proc.stderr[-300:]}")
    rows = [r for r in json.loads(proc.stdout) if r["description"].startswith(APP_NAME)]
    total = sum(float(r["cost"]) for r in rows)
    by_app = {}
    for r in rows:
        by_app[r["description"]] = by_app.get(r["description"], 0.0) + float(r["cost"])
    entry = dict(utc=datetime.now(timezone.utc).isoformat(), study5_usd=round(total, 4), by_app=by_app)
    with (HERE / "results/billing-log.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return total


def _guard(stage):
    (HERE / "results").mkdir(exist_ok=True)
    spent = _billing()
    print(f"[{stage}] Study 5 spend so far ${spent:.2f}", flush=True)
    if spent >= STOP_USD:
        raise SystemExit(f"stop: Study 5 has spent ${spent:.2f} (limit ${STOP_USD})")
    with (HERE / "results/stage-log.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(stage=stage, started_utc=datetime.now(timezone.utc).isoformat(), spend_usd=round(spent, 4))) + "\n")
    return spent


def _rows(split, n=None):
    task = "sva" if split.startswith("sva") else "ioi"
    rows = read(HERE / f"data/{split}.json")[:n]
    return [compact_sva(r) if task == "sva" else compact_ioi(r) for r in rows]


def _meta(sva, ioi):
    return {"sva": [{k: r[k] for k in ("id", "template", "class")} for r in sva],
            "ioi": [{k: r[k] for k in ("id", "template", "order")} for r in ioi]}


def _require(freeze):
    if not (HERE / "data" / freeze).exists():
        raise SystemExit(f"data/{freeze} is required first")


def _start(stage, task):
    """Guard, then hand the stage to orchestrate() and return at once (run with --detach)."""
    _guard(stage)
    call = orchestrate.spawn(dict(task, stage=stage))
    with (HERE / "results/stage-log.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(stage=stage, spawned_utc=datetime.now(timezone.utc).isoformat(), call=call.object_id)) + "\n")
    print(f"[{stage}] running in Modal as {call.object_id}; pull the records with: modal run modal_app.py::pull --stage {stage}",
          flush=True)


@app.local_entrypoint()
def calibrate():
    _require("input-freeze.json")
    _guard("calibrate")
    target = HERE / "results/calibration/calibration.json"
    assert not target.exists(), target
    out = json.loads(gpu_job.remote(dict(kind="calibrate", sva=_rows("sva-calibration"), ioi=_rows("ioi-calibration"))))
    dump(target, out)
    c = out["calibration"]
    print(json.dumps({k: c[k] for k in ("maximum_logit_error", "maximum_probability_error", "maximum_engine_difference",
                                         "exact_write_locality", "passed")}), flush=True)
    _billing()


@app.local_entrypoint()
def development():
    _require("input-freeze.json")
    if not read(HERE / "results/calibration/calibration.json")["calibration"]["passed"]:
        raise SystemExit("calibration did not pass")
    sva, ioi = _rows("sva-development"), _rows("ioi-development")
    _start("development", dict(sva=sva, ioi=ioi, meta=_meta(sva, ioi)))


@app.local_entrypoint()
def test():
    _require("stage4-freeze.json")
    decision = read(HERE / "results/cost-decision.json")
    _start("test", dict(sva=_rows("sva-test", decision["agreement_groups"]), ioi=_rows("ioi-test", decision["ioi_prompts"])))


@app.local_entrypoint()
def replays():
    _require("stage4-freeze.json")
    decision = read(HERE / "results/cost-decision.json")
    g = read(HERE / "results/gates/gates.json")["gates"]
    sva, ioi = _rows("sva-test", decision["agreement_groups"]), _rows("ioi-test", decision["ioi_prompts"])
    fresh, l4 = [], []
    if g["agreement"]["passed"]:
        fresh += [_gpu_records("replay-study", ["A", "C1"], sva[:2]), _gpu_records("replay-reference", ["A", "C1"], sva[:1], "reference")]
        l4 += [_gpu_records("replay-l4", ["A", "C1"], sva[:1])]
    if g["ioi"]["passed"]:
        fresh += [_gpu_records("replay-study", ["B1", "B2", "C2"], ioi[:2]),
                  _gpu_records("replay-reference", ["B1", "B2", "C2"], ioi[:1], "reference")]
        l4 += [_gpu_records("replay-l4", ["B1", "B2", "C2"], ioi[:1])]
    _start("replays", dict(fresh=fresh, l4=l4))


@app.local_entrypoint()
def analyse():
    decision = read(HERE / "results/cost-decision.json")
    g = read(HERE / "results/gates/gates.json")["gates"]
    sva = _rows("sva-test", decision["agreement_groups"]) if g["agreement"]["passed"] else []
    ioi = _rows("ioi-test", decision["ioi_prompts"]) if g["ioi"]["passed"] else []
    _start("analyse", dict(meta=_meta(sva, ioi)))


def _read_volume(path):
    return b"".join(volume.read_file(path)).decode()


def _listing(path):
    try:
        return [e.path for e in volume.listdir(path, recursive=True)]
    except Exception:                                                          # noqa: BLE001
        return []


@app.local_entrypoint()
def pull(stage: str):
    """Copy a finished stage's small records from the Volume to results/ (arrays stay on the Volume)."""
    status = json.loads(_read_volume(f"orchestration/{stage}.json"))
    print(json.dumps({k: status.get(k) for k in ("stage", "status", "started_utc", "finished_utc", "error")}), flush=True)
    if status["status"] == "running":
        return
    dump(HERE / "results" / "orchestration" / f"{stage}.json", status)
    folders = {"development": ["development", "analysis-development", "development-accuracy"], "test": ["test", "gates"],
               "replays": ["replay-study", "replay-reference", "replay-l4"], "analyse": ["analysis-test"]}[stage]
    for folder in folders:
        for path in _listing(folder):
            if path.endswith((".json", ".csv")):
                target = HERE / "results" / ("analysis/" + Path(path).name if stage == "analyse" else path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(_read_volume(path))
    if status.get("environment"):
        dump(HERE / "results" / ("development" if stage == "development" else stage) / "environment.json", status["environment"])
    if stage == "development" and status["status"] == "finished":
        if (HERE / "results/development-accuracy/gates.json").exists():
            (HERE / "results/development/development-accuracy.json").write_text(
                (HERE / "results/development-accuracy/gates.json").read_text())
        _cost_decision(status["records"], status["environment"], _billing())
    if stage == "test":
        dump(HERE / "results/test-completion.json", dict(
            finished_utc=status["finished_utc"], status=status["status"], gates=status.get("gates"),
            agreement_groups=status.get("agreement_groups"), ioi_prompts=status.get("ioi_prompts"), failures=status.get("failures", []),
            records={p.parent.name + "/" + p.stem: read(p)["sha256"] for p in sorted((HERE / "results/test").glob("*/*.json"))}))
    if stage == "replays" and status["status"] == "finished":
        dump(HERE / "results/replay-verification.json", status["replay_verification"])
    _billing()


def _cost_decision(records, env, spent):
    """PROTOCOL.md cost rule: project the remaining spend from the development timings."""
    per_row = {}
    for rec in records:
        per_row.setdefault(rec["panel"], []).append(rec["seconds"])
    per_row = {p: sum(v) / len(v) for p, v in per_row.items()}
    hourly = RATES["A10G"] + CPU_CORES * RATES["cpu_core"] + MEMORY_MIB / 1024 * RATES["memory_gib"]
    hourly_l4 = RATES["L4"] + CPU_CORES * RATES["cpu_core"] + MEMORY_MIB / 1024 * RATES["memory_gib"]
    reference_sites = {"A": 1064, "B1": 448, "B2": 448 * 1.5, "C1": 280 * 89 / 81, "C2": 392 * 73 / 81}

    def project(n_sva, n_ioi):
        sva_s = n_sva * (per_row["A"] + per_row["C1"])
        ioi_s = n_ioi * (per_row["B1"] + per_row["B2"] + per_row["C2"])
        loads = 14 * 60.0                                                  # container starts and model loads
        gates = 120.0
        replay_study = 2 * (per_row["A"] + per_row["C1"] + per_row["B1"] + per_row["B2"] + per_row["C2"])
        replay_reference = 0.38 * sum(reference_sites.values())           # smoke test: 0.38 s per one-site call
        replay_l4 = replay_study                                           # one row per panel at about twice the time
        a10_hours = (sva_s + ioi_s + loads + gates + replay_study + replay_reference) / 3600
        usd = a10_hours * hourly + replay_l4 / 3600 * hourly_l4 + 0.15     # 0.15: CPU analysis and verification
        return dict(n_sva=n_sva, n_ioi=n_ioi, a10_hours=a10_hours, remaining_usd=usd)

    steps = [project(64, 64)]
    if steps[-1]["remaining_usd"] > RULE_USD:
        steps.append(project(64, 32))
    if steps[-1]["remaining_usd"] > RULE_USD:
        steps.append(project(32, 32))
    decision = dict(created_utc=datetime.now(timezone.utc).isoformat(), rule_usd=RULE_USD, spent_before_test_usd=spent,
                    development_seconds_per_row=per_row, hourly_usd=hourly, projections=steps,
                    agreement_groups=steps[-1]["n_sva"], ioi_prompts=steps[-1]["n_ioi"],
                    rule_triggered=len(steps) > 1, gpu=env["gpu"])
    target = HERE / "results/cost-decision.json"
    assert not target.exists(), target
    dump(target, decision)
    print(json.dumps(decision, indent=1), flush=True)

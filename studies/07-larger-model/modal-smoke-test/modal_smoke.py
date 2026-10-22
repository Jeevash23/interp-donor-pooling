"""Modal smoke test for AI-03 Study 5 (folder 07-larger-model): reproduce saved results remotely, time Qwen3.

Run from this folder with the Modal CLI (its own Python needs no numpy or torch):

    modal run modal_smoke.py

1. Reproduction on a Modal T4, with the frozen study code imported unchanged:
   - Study 4 (folder 06): the first test group, every component kind, GPT-2 Small;
   - Study 2 (folder 04): the first Qwen3-0.6B agreement sweep test group;
   - Study 3 (folder 05): the Qwen3-0.6B accuracy gate on its 64 test groups (saved: 0.796875, failed).
   compare_smoke.py then compares the arrays with the saved ones at the frozen tolerance (0.001 logits).
2. Timing of Qwen3-1.7B (pinned weights) and Qwen3-4B (random weights from its config) in float32 on
   random token inputs only, no task prompts, on an L4, an A10 and an L40S, with calibration-style
   numerical checks on those random inputs.

Weights download inside Modal at the image build, at pinned revisions, with SHA-256 asserted. The local machine
uploads code and small inputs and receives small result files. Nothing is written outside records/.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

PORT = "/root/release"                        # release: the release layout is mirrored in the container
P03 = PORT
STUDY = {"s2": f"{P03}/studies/04-reference-and-sweeps", "gate": f"{P03}/studies/05-head-confirmation",
         "s4": f"{P03}/studies/06-component-confirmation"}
QWEN06 = "models/Qwen3-0.6B"
FILES7 = ["config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json",
          "tokenizer_config.json", "vocab.json"]
# Digests recorded before Study 1 (studies/03-grouping-and-circuit/results/environment-and-inputs.json) and,
# for Qwen3-1.7B, in provenance/environment-2026-10-09.md.
WEIGHTS = {
    "gpt2-small": dict(repo="openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
                       dest=f"{PORT}/models/gpt2-small", sha256={
                           "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
                           "generation_config.json": "ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b",
                           "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
                           "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
                           "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
                           "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
                           "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"}),
    "qwen3-0.6b": dict(repo="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
                       dest=f"{PORT}/{QWEN06}", sha256={
                           "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
                           "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
                           "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
                           "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
                           "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                           "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                           "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"}),
    "qwen3-1.7b": dict(repo="Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                       dest="/root/models/qwen3-1.7b", sha256={
                           "config.json": None, "generation_config.json": None, "merges.txt": None,
                           "tokenizer.json": None, "tokenizer_config.json": None, "vocab.json": None,
                           "model.safetensors.index.json": None,
                           "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
                           "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"}),
}
QWEN4B = dict(repo="Qwen/Qwen3-4B", dest="/root/models/qwen3-4b")       # config only; weights are not downloaded


def _sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_weights():
    """Image build step: pinned downloads inside Modal, every recorded digest asserted."""
    from huggingface_hub import HfApi, hf_hub_download
    record = {}
    for name, spec in WEIGHTS.items():
        Path(spec["dest"]).mkdir(parents=True, exist_ok=True)
        record[name] = dict(repo=spec["repo"], revision=spec["revision"], files={})
        for f, want in spec["sha256"].items():
            hf_hub_download(spec["repo"], f, revision=spec["revision"], local_dir=spec["dest"])
            got = _sha(Path(spec["dest"]) / f)
            assert want is None or got == want, (name, f, got)
            record[name]["files"][f] = dict(sha256=got, bytes=(Path(spec["dest"]) / f).stat().st_size)
    info = HfApi().model_info(QWEN4B["repo"], files_metadata=True)
    Path(QWEN4B["dest"]).mkdir(parents=True, exist_ok=True)
    hf_hub_download(QWEN4B["repo"], "config.json", revision=info.sha, local_dir=QWEN4B["dest"])
    record["qwen3-4b"] = dict(repo=QWEN4B["repo"], revision=info.sha, config_sha256=_sha(Path(QWEN4B["dest"]) / "config.json"),
                              files={s.rfilename: dict(bytes=s.size, sha256=(s.lfs.sha256 if s.lfs else None))
                                     for s in info.siblings})
    Path("/root/models/build-record.json").write_text(json.dumps(record, indent=2))


if modal.is_local():
    HERE = Path(__file__).resolve().parent
    PROJECT = HERE.parents[2]
    PORTFOLIO = PROJECT.parent
    AI01 = PROJECT   # release: pyproject.toml and uv.lock
else:
    HERE = PROJECT = PORTFOLIO = AI01 = Path("/nonexistent")
CODE = {"studies/04-reference-and-sweeps": ["common.py", "engines.py", "run_sweep.py"],
        "studies/05-head-confirmation": ["common.py", "engine.py", "run.py"],
        "studies/06-component-confirmation": ["common.py", "engine.py", "run.py"]}
INPUTS = ["studies/03-grouping-and-circuit/results/environment-and-inputs.json"]

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(uv_project_dir=str(AI01), extras=["experiment"], frozen=True)
         .run_function(download_weights)
         .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))
for folder, names in CODE.items():
    for n in names:
        image = image.add_local_file(PROJECT / folder / n, f"{P03}/{folder}/{n}")
for f in INPUTS:
    image = image.add_local_file(PROJECT / f, f"{P03}/{f}")
for n in ("remote_driver.py", "qwen_timing.py"):
    image = image.add_local_file(HERE / n, f"/root/smoke/{n}")

app = modal.App("donor-pooling-larger-model-smoke", image=image)
RESOURCES = dict(cpu=2.0, max_containers=1, scaledown_window=10)


def _environment():
    import importlib.metadata as md
    import platform
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=str(torch.version.cuda),
                device=str(torch.cuda.get_device_name(0)), capability=[int(x) for x in torch.cuda.get_device_capability(0)],
                packages={p: md.version(p) for p in ("numpy", "scipy", "transformers", "huggingface-hub", "safetensors",
                                                     "tokenizers")},
                build_record=json.loads(Path("/root/models/build-record.json").read_text()))


@app.function(gpu="T4", timeout=1500, **RESOURCES)
def reproduce(payload: bytes) -> bytes:
    task = pickle.loads(payload)
    result = dict(started=datetime.now(timezone.utc).isoformat(), environment=_environment(), runs={}, files={})
    for name in ("gate", "s2", "s4"):
        src, dst = f"/tmp/{name}-input.json", f"/tmp/{name}-output.npz"
        Path(src).write_text(json.dumps(task[name]))
        t0 = time.perf_counter()
        proc = subprocess.run(["python", "/root/smoke/remote_driver.py", name, STUDY[name], src, dst],
                              cwd=STUDY[name], capture_output=True, text=True)
        result["runs"][name] = dict(returncode=proc.returncode, wall_seconds=time.perf_counter() - t0,
                                    stdout=proc.stdout[-4000:], stderr=proc.stderr[-4000:])
        if proc.returncode == 0:
            result["files"][f"{name}.json"] = Path(dst + ".json").read_bytes()
            if Path(dst).exists():
                result["files"][f"{name}.npz"] = Path(dst).read_bytes()
    result["finished"] = datetime.now(timezone.utc).isoformat()
    return pickle.dumps(result)


def _timing(names):
    import sys
    sys.path.insert(0, "/root/smoke")
    import qwen_timing
    models = {n: (WEIGHTS[n]["dest"], None) if n == "qwen3-1.7b" else (QWEN4B["dest"], QWEN4B["dest"]) for n in names}
    return dict(environment=_environment(), models=qwen_timing.timing(models))


@app.function(gpu="L4", timeout=900, **RESOURCES)
def time_l4() -> bytes:
    return pickle.dumps(_timing(["qwen3-1.7b", "qwen3-4b"]))


@app.function(gpu="A10", timeout=900, **RESOURCES)
def time_a10() -> bytes:
    return pickle.dumps(_timing(["qwen3-1.7b"]))


@app.function(gpu="L40S", timeout=900, **RESOURCES)
def time_l40s() -> bytes:
    return pickle.dumps(_timing(["qwen3-1.7b", "qwen3-4b"]))


@app.local_entrypoint()
def main():
    out_dir = HERE / "records"
    assert not out_dir.exists(), out_dir
    studies = PROJECT / "studies"
    s4 = json.loads((studies / "06-component-confirmation/data/sva-test.json").read_text())[0]
    s2 = json.loads((studies / "04-reference-and-sweeps/data/sva-test.json").read_text())[0]
    gate = [dict(id=r["id"], prompt=r["encoded"]["qwen3-0.6b"][0])
            for r in json.loads((studies / "05-head-confirmation/data/sva-test.json").read_text())]
    payload = pickle.dumps(dict(s4=s4, s2=s2, gate=gate))
    uploaded = {f"{folder}/{n}": (PROJECT / folder / n).stat().st_size for folder, names in CODE.items() for n in names}
    uploaded.update({f: (PROJECT / f).stat().st_size for f in INPUTS})
    uploaded.update({n: (HERE / n).stat().st_size for n in ("remote_driver.py", "qwen_timing.py")})
    uploaded.update({f"environment/{f}": (AI01 / f).stat().st_size for f in ("pyproject.toml", "uv.lock")})
    calls, received = {}, {}
    t0 = time.perf_counter()
    handles = dict(l4=time_l4.spawn(), a10=time_a10.spawn(), l40s=time_l40s.spawn())
    t4_bytes = reproduce.remote(payload)
    received["t4"] = len(t4_bytes)
    calls["t4_wall_seconds"] = time.perf_counter() - t0
    timing = {}
    for gpu, handle in handles.items():
        try:
            b = handle.get()
            received[gpu] = len(b)
            timing[gpu] = pickle.loads(b)
        except Exception as error:                 # a timing failure must not lose the reproduction
            timing[gpu] = dict(error=repr(error))
    calls["all_wall_seconds"] = time.perf_counter() - t0
    t4 = pickle.loads(t4_bytes)
    out_dir.mkdir()
    for name, data in t4.pop("files").items():
        (out_dir / f"modal-{name}").write_bytes(data)
    record = dict(
        purpose="Smoke test of Modal for AI-03 Study 5: reproduce saved Study 2, 3 and 4 results remotely; "
                "time Qwen3-1.7B and Qwen3-4B in float32 on random tokens",
        reproduced=dict(s4="studies/06-component-confirmation results/test/sva-test-sva-000.npz",
                        s2="studies/04-reference-and-sweeps results/sweep/sva-qwen3-0.6b/test/sva-test-sva-000.npz",
                        gate="studies/05-head-confirmation results/qwen-gate/gate.json"),
        t4=t4, timing=timing, calls=calls, weights={k: {kk: vv for kk, vv in v.items() if kk != "dest"} for k, v in WEIGHTS.items()},
        image=dict(base='modal.Image.debian_slim(python_version="3.12")',
                   uv_sync=dict(project="the release root (pyproject.toml, uv.lock)",
                                extras=["experiment"], frozen=True,
                                lock_sha256=hashlib.sha256((AI01 / "uv.lock").read_bytes()).hexdigest())),
        transfer=dict(uploaded_bytes=sum(uploaded.values()) + len(payload), uploaded_detail=uploaded,
                      payload_bytes=len(payload), received_bytes=sum(received.values()), received_detail=received,
                      note="Weights and Python packages were downloaded inside Modal, not through the local machine"),
        utc=datetime.now(timezone.utc).isoformat())
    (out_dir / "remote-record.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps(dict(calls=calls, uploaded_mb=record["transfer"]["uploaded_bytes"] / 1e6,
                          received_mb=record["transfer"]["received_bytes"] / 1e6,
                          runs={k: (v["returncode"], round(v["wall_seconds"], 1)) for k, v in t4["runs"].items()})))

"""Study 5 local freezes (not public preregistration).

    python freeze.py protocol   # PROTOCOL.md, PREDICTIONS.md, the approval and the inherited inputs; after approval,
                                # before any prompt is built or Qwen3-1.7B runs on a task prompt
    python freeze.py input      # code, prompts, image definition, lock hash and analysis diff; before calibration
    python freeze.py amend N    # amendment N to the input freeze (AMENDMENTS.md): new hashes of the changed files
    python freeze.py stage4     # calibration and development records and the cost decision; before the gates
    python freeze.py verify     # every freeze written so far

Each freeze refuses to overwrite itself and checks the freezes before it.
"""
from __future__ import annotations

import difflib
import os
import sys
from datetime import datetime, timezone

from common import PORTFOLIO, PROJECT, ROOT, dump, read, sha

sys.path.insert(0, str(PROJECT / "tools"))   # release: freeze keys are "<project>/..." (RELEASE-CHANGES.md)
import release_paths as rp  # noqa: E402

STUDIES = PROJECT / "studies"
INHERITED = [STUDIES / f"{folder}/{name}" for folder in ("04-reference-and-sweeps", "05-head-confirmation",
                                                           "06-component-confirmation")
             for name in ("PROTOCOL.md", "common.py", "prepare.py")]
INHERITED += [STUDIES / "04-reference-and-sweeps" / n for n in ("engines.py", "run_sweep.py", "analyze_sweep.py", "AMENDMENTS.md")]
INHERITED += [STUDIES / "05-head-confirmation" / n for n in ("engine.py", "run.py", "analyze.py")]
INHERITED += [STUDIES / "06-component-confirmation" / n for n in ("engine.py", "run.py", "analyze.py")]
INHERITED += [PROJECT / "provenance/environment-2026-10-09.md",
              STUDIES / "03-grouping-and-circuit/results/environment-and-inputs.json",
              PROJECT / "pyproject.toml", PROJECT / "uv.lock"]   # release: this release's environment files
CODE = ["common.py", "prepare.py", "engine.py", "stages.py", "modal_app.py", "analyze.py", "verify.py", "freeze.py",
        "test_cpu.py", "test_analysis.py"]
# The study README is updated as the study runs; its frozen version is kept here and checked instead.
README_COPY = ROOT / "provenance/README-at-protocol-freeze.md"
PORTED = {"analyze.py": ["06-component-confirmation/analyze.py", "05-head-confirmation/analyze.py",
                         "04-reference-and-sweeps/analyze_sweep.py"]}


def rel(p):
    return rp.PROJECT_KEY + os.path.relpath(p, PROJECT)


def record(path, files, kind, **extra):
    if path.exists():
        raise RuntimeError(f"{path.name} exists; freezes are never overwritten")
    dump(path, {"kind": kind, "public_preregistration": False,
                "utc": datetime.now(timezone.utc).isoformat(), "files": {rel(p): sha(p) for p in files}, **extra})
    print(f"wrote {path.relative_to(ROOT)} ({len(files)} files)")


def amended():
    """Current hashes recorded by the amendments to the input freeze (later amendments win)."""
    out = {}
    for path in sorted((ROOT / "data").glob("input-freeze-amendment-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        out.update(read(path)["files"])
    return out


def check(name):
    path = ROOT / "data" / name
    readme = rel(ROOT / "README.md")
    later = amended() if name.startswith("input-freeze") else {}       # later amendments supersede earlier records
    bad = [r for r, h in read(path)["files"].items() if not rp.matches(rp.resolve_key(r), h, r)
           and not (r == readme and README_COPY.exists() and rp.matches(README_COPY, h))
           and not (r in later and rp.matches(rp.resolve_key(r), later[r], r))]
    if bad:
        raise RuntimeError(f"{name}: files changed after the freeze: {bad}")
    return len(read(path)["files"])


def protocol():
    files = [ROOT / n for n in ("PROTOCOL.md", "PREDICTIONS.md", "AUTHOR-APPROVAL.txt", "README.md")]
    files += sorted(p for p in (ROOT / "modal-smoke-test").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    record(ROOT / "data/protocol-freeze.json", files + INHERITED,
           "Local protocol freeze after the author's approval and before any prompt is built or Qwen3-1.7B runs on a "
           "task prompt")


def input_freeze():
    check("protocol-freeze.json")
    if (ROOT / "results").exists():
        raise RuntimeError("results predate the input freeze")
    diffs = {}
    for target, sources in PORTED.items():
        new = (ROOT / target).read_text().splitlines(keepends=True)
        diffs[target] = {src: "".join(difflib.unified_diff((STUDIES / src).read_text().splitlines(keepends=True), new,
                                                            fromfile=src, tofile=f"07-larger-model/{target}"))
                         for src in sources}
    dump(ROOT / "data/analysis-port-diff.json", diffs)
    files = [ROOT / n for n in CODE] + sorted((ROOT / "data").glob("*.json")) + sorted((ROOT / "logs").glob("test_*.log"))
    files = [p for p in files if p.name not in ("input-freeze.json", "stage4-freeze.json")]
    record(ROOT / "data/input-freeze.json", files + INHERITED,
           "Local input freeze of code, prompts, image definition, lock file and analysis port, before calibration")


def amend(n):
    check("protocol-freeze.json")
    frozen = read(ROOT / "data/input-freeze.json")["files"]
    changed = [PORTFOLIO / r for r, h in frozen.items() if sha(PORTFOLIO / r) != h]
    files = changed + [ROOT / "AMENDMENTS.md"] + sorted((ROOT / "logs").glob(f"test_*amendment{n}*.log"))
    record(ROOT / f"data/input-freeze-amendment-{n}.json", files,
           f"Amendment {n} to the input freeze (AMENDMENTS.md): new hashes of the changed files")


def stage4():
    check("protocol-freeze.json")
    check("input-freeze.json")
    files = sorted((ROOT / "results/calibration").rglob("*.json")) + sorted((ROOT / "results/development").rglob("*.json"))
    files += sorted((ROOT / "results/analysis-development").rglob("*")) + sorted((ROOT / "results/development-accuracy").rglob("*.json"))
    files += [ROOT / "results/orchestration/development.json", ROOT / "results/cost-decision.json"]
    record(ROOT / "data/stage4-freeze.json", files,
           "Local freeze of calibration, development and the cost decision, before the gates and any test output")


def verify():
    names = ["protocol-freeze.json", "input-freeze.json", "stage4-freeze.json"]
    names += sorted(p.name for p in (ROOT / "data").glob("input-freeze-amendment-*.json"))
    n = sum(check(name) for name in names if (ROOT / "data" / name).exists())
    print(f"freezes verified ({n} file hashes)")


if __name__ == "__main__":
    if sys.argv[1] == "amend":
        amend(int(sys.argv[2]))
    else:
        {"protocol": protocol, "input": input_freeze, "stage4": stage4, "verify": verify}[sys.argv[1]]()

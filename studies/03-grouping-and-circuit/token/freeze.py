"""Freeze reviewed inputs only after calibration/development and independent checks."""
from __future__ import annotations
import subprocess
import sys
import tempfile
from common import *

def main():
    target = ROOT / "data/freeze.json"
    if target.exists():
        raise RuntimeError("Freeze already exists; never silently replace it")
    for model in MODEL_PATHS:
        folder = ROOT / "results" / model
        if (folder / "test").exists() or (folder / "replay").exists():
            raise RuntimeError("Freeze must precede every test intervention")
        assert read(folder / "calibration.json")["passed"]
        assert read(folder / "development-completion.json")["groups"] == 8
        assert read(ROOT / "results/analysis-development" / f"{model}.json")["groups"] == 8
    assert read(ROOT / "results/pretest-gate-revalidation.json")["passed"]
    test = subprocess.run([sys.executable, str(ROOT / "test_design.py")], capture_output=True, text=True)
    dump(ROOT / "results/pretest-analytic-tests.json", {"created_utc": utc(), "command": "python token/test_design.py", "returncode": test.returncode, "stdout": test.stdout, "stderr": test.stderr, "passed": test.returncode == 0})
    if test.returncode:
        raise RuntimeError(test.stderr)
    with tempfile.TemporaryDirectory(prefix="donor-grouping-regeneration-") as temporary:
        output = Path(temporary)
        regenerated = subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--out", str(output)], capture_output=True, text=True)
        if regenerated.returncode:
            raise RuntimeError(regenerated.stderr)
        equality = {split: sha(output / f"{split}.json") == sha(ROOT / f"data/{split}.json") for split in ["calibration", "development", "test"]}
        assert all(equality.values())
        dump(ROOT / "results/pretest-data-regeneration.json", {"created_utc": utc(), "split_byte_equality": equality, "passed": all(equality.values()), "audit_timestamp_excluded_from_equality": True})
    paths = sorted([*ROOT.glob("*.py"), ROOT / "README.md", *[ROOT / f"data/{name}.json" for name in ["calibration", "development", "test", "generation-audit"]], ROOT.parent / "PROTOCOL.md", ROOT.parent / "results/environment-and-inputs.json", ROOT.parent / "review/token-design-check.json", ROOT / "results/pretest-gate-revalidation.json", ROOT / "results/pretest-analytic-tests.json", ROOT / "results/pretest-data-regeneration.json", ROOT / "results/pretest-independent-analysis-review.json"])
    files = {os.path.relpath(path, ROOT): sha(path) for path in paths}
    record = {"created_utc": utc(), "local_prospective_freeze": True, "public_preregistration": False, "test_results_seen": False, "calibration_groups_per_model": 4, "development_groups_per_model": 8, "test_groups_per_model": 64, "source_of_design": "Critique of September study; balanced fixed-support partition control", "files": files}
    dump(target, record)
    freeze_check()
    print(json.dumps({"created_utc": record["created_utc"], "frozen_files": len(files), "passed": True, "test_groups_per_model": 64}))

if __name__ == "__main__":
    main()

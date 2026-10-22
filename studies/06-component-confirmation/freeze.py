"""Bind the protocol, code, generated prompts, calibration and development records before test runs."""
from __future__ import annotations

import os

from common import ROOT, dump, sha, utc


def main():
    path = ROOT / "data/freeze.json"
    if path.exists():
        raise RuntimeError("freeze already exists; do not overwrite")
    files = [ROOT / "PROTOCOL.md"] + sorted(ROOT.glob("*.py")) + sorted((ROOT / "data").glob("*.json"))
    files += [ROOT / "results/calibration.json", ROOT / "results/development-completion.json",
              ROOT / "results/analysis-development.json"]
    dump(path, {"frozen_utc": utc(), "public_preregistration": False,
                "files": {os.path.relpath(p, ROOT): sha(p) for p in files}})
    print(f"froze {len(files)} files")


if __name__ == "__main__":
    main()

"""Make a fresh copy of one study for a full rerun, then print the study's documented stage commands.

    python tools/fresh_copy.py studies/06-component-confirmation

The copy goes to reruns/<folder>, at the same depth as studies/, so the code still finds the release
root, the other studies' prompts (for the prompt-separation checks) and models/. Saved results are left
out, and so are the frozen prompts of studies whose rerun starts by generating them again (prepare.py
refuses to overwrite prompts); compare the regenerated prompts with the shipped ones afterwards
(`sha256sum` against the study's data/freeze.json). The shipped study folder is never touched.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]

# Stage commands as documented for each study (run inside reruns/<folder>; GPU recommended).
COMMANDS = {
    "04-reference-and-sweeps": [
        "python run_reference.py calibrate && python run_reference.py development",
        "python run_sweep.py sva calibrate --model gpt2-small   # likewise ioi, and sva with --model qwen3-0.6b",
        "python freeze.py",
        "python run_reference.py screen && python run_reference.py test && python run_reference.py replay",
        "python run_sweep.py sva test --model gpt2-small && python run_sweep.py ioi test --model gpt2-small",
        "python run_sweep.py sva test --model qwen3-0.6b --limit 32",
        "python analyze_reference.py test && python analyze_sweep.py test && python verify.py",
        "python posthoc/random_circuits.py run && python posthoc/random_circuits.py analyze   # post hoc control",
    ],
    "05-head-confirmation": [
        "python prepare.py",
        "python run.py calibrate --setting sva-end && python run.py development --setting sva-end   # likewise ioi-end, ioi-s2",
        "python run.py calibrate --setting sva-end --model qwen3-0.6b",
        "python freeze.py",
        "python run.py gate --model qwen3-0.6b",
        "python run.py test --setting sva-end && python run.py test --setting ioi-end && python run.py test --setting ioi-s2",
        "python run.py replay --setting sva-end && python run.py replay --setting ioi-end && python run.py replay --setting ioi-s2",
        "python analyze.py test && python verify.py && python write_results.py",
    ],
    "06-component-confirmation": [
        "python prepare.py",
        "python run.py calibrate && python run.py development && python analyze.py development",
        "python freeze.py",
        "python run.py test && python run.py replay",
        "python analyze.py test && python verify.py && python write_results.py",
    ],
}
# Studies whose rerun regenerates the prompts first: the copy leaves data/ out.
REGENERATE = {"05-head-confirmation", "06-component-confirmation"}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = (RELEASE / sys.argv[1]).resolve()
    if src.parent != (RELEASE / "studies").resolve() or not src.is_dir():
        sys.exit("give a folder under studies/, for example studies/06-component-confirmation")
    if src.name not in COMMANDS:
        sys.exit(f"{src.name}: see its README.md and REPRODUCIBILITY.md (and the release README) for its rerun")
    dst = RELEASE / "reruns" / src.name
    if dst.exists():
        sys.exit(f"{dst} exists; remove it first")
    skip = {"results"} | ({"data"} if src.name in REGENERATE else set())

    def ignore(folder, names):
        top = Path(folder).resolve() == src
        out = {n for n in names if n == "__pycache__"}
        if top:
            out |= {n for n in names if n in skip}
        return out

    shutil.copytree(src, dst, ignore=ignore)
    if src.name not in REGENERATE:
        (dst / "data/freeze.json").unlink(missing_ok=True)      # freeze.py writes a new one
    print(f"fresh copy: {dst.relative_to(RELEASE)} (left out: {', '.join(sorted(skip))}"
          f"{'' if src.name in REGENERATE else ', data/freeze.json'})")
    print("then, from that folder (GPU recommended; see README.md for times):")
    for c in COMMANDS[src.name]:
        print("   ", c)


if __name__ == "__main__":
    main()

# How donors are chosen and combined changes activation-patching conclusions: code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` (uv sync) creates.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export MODELS_DIR ?= $(CURDIR)/models
export PYTHONDONTWRITEBYTECODE := 1
.NOTPARALLEL:

# Folder names predate the manuscript's study names (see README.md).
P1 := studies/01-pilot
P2 := submission-study
S1 := studies/03-grouping-and-circuit
S2 := studies/04-reference-and-sweeps
S3 := studies/05-head-confirmation
S4 := studies/06-component-confirmation
S5 := studies/07-larger-model

.PHONY: help setup models unpack check-freezes quick-check verify verify-study2 verify-study2-posthoc \
        verify-study3 verify-study4 verify-study5 test numbers numbers-check figures fresh-copy python-check

help:
	@echo "make setup          create .venv with the pinned environment (uv sync; downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check    CPU only, a few minutes, no model and no network: freezes, Studies 3-5 checks, all paper numbers"
	@echo "make verify         every verifier the saved outputs support (about 5 minutes on a CPU)"
	@echo "make numbers-check  recompute every number and generated table in the manuscript and compare"
	@echo "make figures        redraw the manuscript's figures into paper/figures/"
	@echo "make models         download GPT-2 Small and Qwen3-0.6B (about 2.1 GB), the checkpoints Studies 1-4 use"
	@echo "make test           Study 2 design invariants (needs the GPT-2 Small tokeniser in models/)"
	@echo "make fresh-copy STUDY=studies/06-component-confirmation   start a full rerun in reruns/ (README.md)"

setup:
	uv sync --frozen --extra experiment --group dev

python-check:
	@$(PY) -c "import numpy, torch, transformers" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

models: python-check
	$(PY) tools/get_models.py gpt2-small qwen3-0.6b

unpack: python-check
	$(PY) tools/unpack.py

check-freezes: unpack
	$(PY) tools/check_freezes.py

# A few minutes on a CPU. The verifiers recompute the prespecified primary numbers of Studies 3 and 4 from the
# raw arrays and compare them with the saved analyses; make_numbers.py --check recomputes every number in the
# manuscript from the saved outputs and compares it with paper/generated/.
quick-check: unpack
	CUDA_VISIBLE_DEVICES= $(PY) tools/check_freezes.py
	cd $(S4) && CUDA_VISIBLE_DEVICES= $(PY) verify.py > /dev/null && echo "Study 4: verify.py passed"
	cd $(S3) && CUDA_VISIBLE_DEVICES= $(PY) verify.py > /dev/null && echo "Study 3: verify.py passed"
	cd $(S5) && CUDA_VISIBLE_DEVICES= $(PY) freeze.py verify
	CUDA_VISIBLE_DEVICES= $(PY) paper/make_numbers.py --check
	CUDA_VISIBLE_DEVICES= $(PY) tools/quick_summary.py

verify: unpack verify-study2 verify-study2-posthoc verify-study3 verify-study4 verify-study5
	@echo "all verifiers passed (Study 5 with its recorded replay failure; see README.md)"

verify-study2:
	cd $(S2) && $(PY) verify.py

verify-study2-posthoc:
	cd $(S2) && $(PY) posthoc/verify_posthoc.py
	cd $(S2) && $(PY) posthoc/verify_review_controls.py

verify-study3:
	cd $(S3) && $(PY) verify.py > /dev/null && echo "Study 3: verify.py passed"

verify-study4:
	cd $(S4) && $(PY) verify.py > /dev/null && echo "Study 4: verify.py passed"

# Study 5's verifier exits with status 1 because four IOI replays exceeded their tolerance; the author accepted
# the results with this disclosed (studies/07-larger-model/NOTES-AFTER-TEST.md). This target passes when that is
# the only problem it reports.
verify-study5:
	cd $(S5) && $(PY) freeze.py verify
	cd $(S5) && { $(PY) verify.py > /dev/null; true; }
	$(PY) -c "import json,sys; p=json.load(open('$(S5)/results/verification.json'))['problems']; \
	print('Study 5 problems:', p); sys.exit(0 if p == ['replays outside tolerance'] else 1)"

test: unpack
	cd $(S2) && $(PY) -m pytest -q -p no:cacheprovider test_design.py

numbers: unpack
	$(PY) paper/make_numbers.py

numbers-check: unpack
	$(PY) paper/make_numbers.py --check

figures: unpack
	$(PY) paper/make_figures.py

fresh-copy: python-check
	@test -n "$(STUDY)" || { echo "usage: make fresh-copy STUDY=studies/<folder>"; exit 1; }
	$(PY) tools/fresh_copy.py $(STUDY)

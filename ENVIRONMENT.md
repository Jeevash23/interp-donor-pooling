# Environment

The studies ran in one Python environment, recorded exactly in `uv.lock` (180 locked packages, including the
release itself as a virtual project). `pyproject.toml` declares it the same way the original project did: the
`experiment` extra is the full research environment, and the `dev` group adds pytest.

| Package | Version |
|---|---|
| Python | 3.12.14 (`uv` installs it if it is missing) |
| torch | 2.13.0 (CUDA 13 build from PyPI) |
| transformers | 5.16.1 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.29.0 |

The code itself imports torch, transformers, numpy, scipy, matplotlib, safetensors and huggingface-hub (and
pytest for `make test`); the other packages in the lock (JupyterLab, datasets, TransformerLens and their
dependencies) were part of the environment and are kept so that the lock is the one the runs used. The Modal
image for Study 5 is built from the same two files (`uv_sync` with the `experiment` extra).

## A fresh install

```sh
uv sync --frozen --extra experiment --group dev        # what make setup runs
```

On Linux x86-64 this downloads 176 packages, about 3.0 GB of wheels. The largest are torch (527 MB) and the
NVIDIA CUDA libraries it depends on (about 2.0 GB together: cuBLAS 423 MB, cuDNN 366 MB, cuFFT 214 MB, NCCL 206
MB, cuSOLVER 201 MB, cuSPARSELt 170 MB, cuSPARSE 146 MB and smaller ones), then triton (198 MB). The sizes come
from `uv.lock`; installing creates `.venv/` in this folder.

## A smaller CPU-only environment

The quick check, the verifiers and the number and figure scripts run on a CPU and need no CUDA. To avoid the
CUDA libraries, install the same versions of the packages the code imports, with PyTorch's CPU build:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv transformers==5.16.1 numpy==2.5.2 scipy==1.18.1 matplotlib==3.11.1 \
    safetensors==0.8.0 huggingface-hub==1.29.0 pytest
```

This is not the locked environment, so use it for checking saved outputs; for model reruns that are meant to
reproduce the recorded numbers to the protocols' tolerances, use `make setup`.

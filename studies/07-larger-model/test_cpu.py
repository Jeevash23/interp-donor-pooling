"""CPU code test of the Study 5 engine and runners on a tiny random Qwen3 (no checkpoint, no GPU).

    python -B test_cpu.py

On one calibration group and one calibration IOI prompt, every panel runs with the study engine and the
reference engine, and the arrays must agree; the built-in checks, calibration and gates must run and pass.
The tiny model (16 layers, width 64, 4 heads of 16) is written to a temporary folder outside the project.
"""
import tempfile

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import stages
from common import LOCAL_TOKENIZERS, ROOT, compact_ioi, compact_sva, read
from engine import Qwen3Engine


def main():
    cfg = AutoConfig.from_pretrained(LOCAL_TOKENIZERS["qwen3"])
    cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads = 16, 64, 4
    cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size, cfg.max_window_layers = 2, 16, 128, 16
    cfg.layer_types = cfg.layer_types[:16]
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(tmp)
        engine = Qwen3Engine(tmp, device="cpu", strict=False, max_rows=200)
        sva = [compact_sva(r) for r in read(ROOT / "data/sva-calibration.json")[:1]]
        ioi = [compact_ioi(r) for r in read(ROOT / "data/ioi-calibration.json")[:1]]
        results = {}
        for panel, run in stages.RUNNERS.items():
            row = sva[0] if stages.TASK[panel] == "sva" else ioi[0]
            a, rec_a = run(engine, row, mode="study")
            b, rec_b = run(engine, row, mode="reference")
            assert rec_a["passed"] and rec_b["passed"], (panel, rec_a["max_check_error"], rec_b["max_check_error"])
            diff = max(float(np.nanmax(np.abs(a[k] - b[k]))) for k in a if k.startswith("margins"))
            same = all(np.array_equal(a[k], b[k]) for k in a if not k.startswith("margins") and not k.startswith("spread"))
            assert diff <= 1e-4 and same, (panel, diff, same)
            results[panel] = dict(max_difference=diff, seconds_study=rec_a["seconds"], seconds_reference=rec_b["seconds"])
            print(panel, results[panel], flush=True)
        cal = stages.calibrate(engine, sva, ioi)
        assert cal["passed"], {k: cal[k] for k in ("maximum_logit_error", "maximum_probability_error",
                                                    "maximum_engine_difference", "exact_write_locality")}
        print("calibration", {k: cal[k] for k in ("maximum_logit_error", "maximum_probability_error",
                                                    "maximum_engine_difference", "exact_write_locality")})
        g = stages.gates(engine, sva, ioi)
        print("gates ran", g["agreement"]["clean_accuracy"], g["ioi"]["io_over_s_accuracy"])
    print("all CPU code tests passed")


if __name__ == "__main__":
    main()

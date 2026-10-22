"""Experiment A: does the ablation reference decide how often the IOI circuit matches model errors?

Stages: calibrate -> development -> (freeze) -> screen -> test -> replay.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from common import MODEL_PATHS, ROOT, check_checkpoint, dump, freeze_check, prior_prompt_strings, read, sha, utc
from engines import HeadEngine
from prepare import BANK_SIZES, ioi_banks

GROUPS = {
    "name_mover": [(9, 9), (10, 0), (9, 6)],
    "backup_name_mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative": [(10, 7), (11, 10)],
    "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate_token": [(0, 1), (0, 10), (3, 0)],
    "previous_token": [(2, 2), (4, 11)],
}
NODE_POSITION = {g: ("s2" if g in ("induction", "duplicate_token") else "s1_plus1" if g == "previous_token" else "end")
                 for g in GROUPS}
CELLS = list(BANK_SIZES)          # N8, NPO8, N128, NPO128
MODES = ["reference", "empty"]
LOGIT_TOL, PROB_TOL = 1e-3, 2e-5
SELECT = {"pair_error": 384, "pair_correct": 384}   # per template family


def keep_mask(rec, mode):
    t = len(rec["ids"])
    mask = torch.zeros((12, t, 12, 1), dtype=torch.bool)
    if mode == "all":
        mask[:] = True
    elif mode == "reference":
        for group, heads in GROUPS.items():
            for layer, head in heads:
                mask[layer, rec[NODE_POSITION[group]], head] = True
        assert int(mask.sum()) == 26
    elif mode != "empty":
        raise ValueError(mode)
    return mask


def margins(logits, rec):
    return (logits[:, rec["target_id"]].double() - logits[:, rec["contrast_id"]].double()).cpu().numpy()


def prob_error(a, b):
    return float((a.double().softmax(-1) - b.double().softmax(-1)).abs().max())


def calibrate(engine, tok, prior):
    rows = read(ROOT / "data/ioi-calibration.json")
    errors, probs, positive = {}, {}, 0.0
    for row in rows:
        rec = row["recipient"]
        p = rec["ids"]
        full, own, audit = engine.run([p], capture=True)
        with torch.inference_mode():
            native = engine.model(torch.tensor([p], device=engine.device), use_cache=False).logits[:, -1]
        errors[row["id"] + "/passive_hook"] = float((full - native).abs().max())
        probs[row["id"] + "/passive_hook"] = prob_error(full, native)
        banks = ioi_banks(tok, row, prior)
        for cell in ["N8", "NPO8"]:
            dl, src, _ = engine.run([d["ids"] for d in banks[cell]], capture=True)
            mean = src.mean(1, keepdim=True)
            checks = [("all_kept", mean, "all", full), ("self_empty", own, "empty", full), ("self_reference", own, "reference", full)]
            for name, rep, mode, expected in checks:
                got, _, a = engine.run([p], replacement=rep, keep=keep_mask(rec, mode))
                key = f"{row['id']}/{cell}/{name}"
                errors[key] = float((got - expected).abs().max())
                probs[key] = prob_error(got, expected)
                errors[key + "/outside"], errors[key + "/write"] = a["outside_changed_max"], a["replacement_error_max"]
            duplicate = src[:, :1].repeat(1, 8, 1, 1, 1).mean(1, keepdim=True)
            log, _, a = engine.run([p] * 2, replacement=torch.cat([src[:, :1], duplicate], 1), keep=keep_mask(rec, "reference"))
            key = f"{row['id']}/{cell}/duplicate_donor"
            errors[key], probs[key] = float((log[0] - log[1]).abs().max()), prob_error(log[:1], log[1:])
            errors[key + "/outside"], errors[key + "/write"] = a["outside_changed_max"], a["replacement_error_max"]
            positive = max(positive, float((log[0] - full[0]).abs().max()))
            replay, _, _ = engine.run([banks[cell][0]["ids"]], replacement=src[:, :1], keep=keep_mask(rec, "empty"))
            key = f"{row['id']}/{cell}/donor_reconstruction"
            errors[key], probs[key] = float((replay[0] - dl[0]).abs().max()), prob_error(replay[:1], dl[:1])
    exact = all(v == 0 for k, v in errors.items() if k.endswith(("/outside", "/write")))
    gated = max(v for k, v in errors.items() if not k.endswith(("/outside", "/write")))
    passed = gated <= LOGIT_TOL and max(probs.values()) <= PROB_TOL and positive > 0.01 and exact
    return {"created_utc": utc(), "rows": len(rows), "errors": errors, "probability_errors": probs,
            "maximum_logit_error": gated, "maximum_probability_error": max(probs.values()),
            "exact_write_locality": exact, "positive_logit_change": positive,
            "logit_tolerance": LOGIT_TOL, "probability_tolerance": PROB_TOL, "passed": passed}


def screen(engine):
    rows = read(ROOT / "data/ioi-screen.json")
    values = np.empty((len(rows), 2))           # margin, argmax token
    bylength = {}
    for i, r in enumerate(rows):
        bylength.setdefault(len(r["recipient"]["ids"]), []).append(i)
    start = time.time()
    for _, idx in sorted(bylength.items()):
        for s in range(0, len(idx), 256):
            ix = idx[s:s + 256]
            log, _, _ = engine.run([rows[i]["recipient"]["ids"] for i in ix])
            for j, i in enumerate(ix):
                values[i, 0] = margins(log[j:j + 1], rows[i]["recipient"])[0]
                values[i, 1] = int(log[j].argmax())
    np.savez_compressed(ROOT / "results/reference/screen.npz", margin=values[:, 0], argmax=values[:, 1].astype(np.int64),
                        ids=np.array([r["id"] for r in rows]))
    selected, strata = [], {}
    for fam in ["O", "W"]:
        for label, correct in [("pair_error", False), ("pair_correct", True)]:
            ix = [i for i, r in enumerate(rows) if r["family"] == fam and (values[i, 0] > 0) == correct]
            chosen = sorted(ix, key=lambda i: rows[i]["selection_priority"])[:SELECT[label]]
            strata[f"{fam}/{label}"] = {"population": len(ix), "selected": len(chosen)}
            for i in chosen:
                row = dict(rows[i])
                row.update(screen_index=i, stratum_label=label, weight=len(ix) / len(chosen),
                           screen_margin=float(values[i, 0]), screen_argmax=int(values[i, 1]))
                selected.append(row)
    selected.sort(key=lambda r: r["screen_index"])
    dump(ROOT / "data/ioi-selected-test.json", selected)
    audit = {"created_utc": utc(), "screened": len(rows), "strata": strata, "selection_caps": SELECT,
             "error_definition": "IO-minus-S logit margin <= 0", "seconds": time.time() - start,
             "selected_sha256": sha(ROOT / "data/ioi-selected-test.json"),
             "screen_sha256": sha(ROOT / "results/reference/screen.npz"),
             "population_pair_accuracy": {fam: float(np.mean([values[i, 0] > 0 for i, r in enumerate(rows) if r["family"] == fam])) for fam in ["O", "W"]},
             "selection_uses_intervention_outputs": False, "environment": engine.environment()}
    dump(ROOT / "results/reference/screen-selection.json", audit)
    return audit


def evaluate_row(engine, tok, row, prior):
    rec = row["recipient"]
    p = rec["ids"]
    full, own, _ = engine.run([p], capture=True)
    out = {"full_margin": margins(full, rec)[0], "full_argmax": int(full.argmax())}
    banks = ioi_banks(tok, row, prior)
    audits = {}
    for cell in CELLS:
        donors = banks[cell]
        dlog, src, _ = engine.run([d["ids"] for d in donors], capture=True)
        out[f"{cell}/donor_prompt_margin"] = margins(dlog, rec)
        replacement = torch.cat([src.mean(1, keepdim=True), src], 1)
        for mode in MODES:
            log, _, a = engine.run([p] * replacement.shape[1], replacement=replacement, keep=keep_mask(rec, mode))
            if a["outside_changed_max"] != 0 or a["replacement_error_max"] != 0:
                raise RuntimeError(f"write locality failed: {row['id']}/{cell}/{mode}")
            m = margins(log, rec)
            out[f"{cell}/{mode}/centroid_margin"] = m[0]
            out[f"{cell}/{mode}/donor_margins"] = m[1:]
            out[f"{cell}/{mode}/centroid_argmax"] = int(log[0].argmax())
            out[f"{cell}/{mode}/donor_argmax"] = log[1:].argmax(-1).cpu().numpy()
            out[f"{cell}/{mode}/meanlogit_margin"] = margins(log[1:].mean(0, keepdim=True), rec)[0]
            audits[f"{cell}/{mode}"] = a
        del src, replacement
    identity, _, _ = engine.run([p], replacement=own, keep=keep_mask(rec, "empty"))
    keepall, _, _ = engine.run([p], replacement=own * 0, keep=keep_mask(rec, "all"))
    out["identity_error"] = float((identity - full).abs().max())
    out["all_kept_error"] = float((keepall - full).abs().max())
    if out["identity_error"] > LOGIT_TOL or out["all_kept_error"] > LOGIT_TOL:
        raise RuntimeError(f"identity controls failed for {row['id']}")
    if "screen_margin" in row and abs(out["full_margin"] - row["screen_margin"]) > LOGIT_TOL:
        raise RuntimeError(f"screen margin not reproduced for {row['id']}")
    return out, audits


def evaluate(engine, tok, prior, stage):
    if stage == "development":
        rows = read(ROOT / "data/ioi-development.json")
    else:
        rows = read(ROOT / "data/ioi-selected-test.json")
        assert sha(ROOT / "data/ioi-selected-test.json") == read(ROOT / "results/reference/screen-selection.json")["selected_sha256"]
        if stage == "replay":
            rows = [r for key in sorted({(r["family"], r["stratum_label"]) for r in rows})
                    for r in [x for x in rows if (x["family"], x["stratum_label"]) == key][:3]]
    dest = ROOT / "results/reference" / stage
    dest.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for i, row in enumerate(rows):
        path = dest / f"{row['id']}.npz"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        out, audits = evaluate_row(engine, tok, row, prior)
        np.savez_compressed(path, **{k.replace("/", "__"): np.asarray(v) for k, v in out.items()})
        if i % 32 == 0 or i == len(rows) - 1:
            print(f'{{"stage": "{stage}", "row": {i + 1}, "of": {len(rows)}, "seconds": {time.time() - start:.1f}}}', flush=True)
    dump(ROOT / f"results/reference/{stage}-completion.json",
         {"finished_utc": utc(), "rows": len(rows), "seconds": time.time() - start, "environment": engine.environment(),
          "patched_forwards_per_row": 2 * sum(1 + n for n in BANK_SIZES.values()),
          "donor_forwards_per_row": sum(BANK_SIZES.values())})
    if stage == "replay":
        worst = {}
        for row in rows:
            with np.load(dest / f"{row['id']}.npz") as a, np.load(ROOT / "results/reference/test" / f"{row['id']}.npz") as b:
                worst[row["id"]] = max(float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))) for k in a.files)
        dump(ROOT / "results/reference/replay-verification.json",
             {"errors": worst, "maximum_error": max(worst.values()), "passed": max(worst.values()) <= LOGIT_TOL})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "development", "screen", "test", "replay"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    (ROOT / "results/reference").mkdir(parents=True, exist_ok=True)
    if args.stage in ("screen", "test", "replay"):
        freeze_check()
    hashes = check_checkpoint("gpt2-small")
    engine = HeadEngine(args.device, args.batch)
    tok = AutoTokenizer.from_pretrained(MODEL_PATHS["gpt2-small"], local_files_only=True)
    prior, _ = prior_prompt_strings()
    dump(ROOT / f"results/reference/environment-{args.stage}.json", {**engine.environment(), "checkpoint_sha256": hashes})
    if args.stage == "calibrate":
        dest = ROOT / "results/reference/calibration.json"
        if dest.exists():
            raise RuntimeError("keep earlier calibration records; use a new file name")
        result = calibrate(engine, tok, prior)
        dump(dest, result)
        print({k: v for k, v in result.items() if k not in ("errors", "probability_errors")}, flush=True)
        if not result["passed"]:
            raise RuntimeError("calibration failed")
        return
    if not read(ROOT / "results/reference/calibration.json")["passed"]:
        raise RuntimeError("passed calibration required")
    if args.stage == "screen":
        print(screen(engine)["strata"], flush=True)
    else:
        evaluate(engine, tok, prior, args.stage)


if __name__ == "__main__":
    main()

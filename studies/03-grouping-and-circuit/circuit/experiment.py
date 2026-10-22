"""Finite-bank pooling comparison for a fixed, position-specific IOI circuit.

Scientific scope: 26 attention-head/position nodes plus the full MLP and
embedding/residual scaffold; this is node ablation, not edge circuit discovery.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PORTFOLIO = PROJECT.parent
DEFAULT_MODEL = Path(os.environ.get("MODELS_DIR", PROJECT / "models")) / "gpt2-small"   # release: see models/README.md

# Membership and relevant positions verified against the original authors'
# ioi_circuit_extraction.py and independently ACDC's acdc/ioi/utils.py.
GROUPS = {
    "name_mover": [(9, 9), (10, 0), (9, 6)],
    "backup_name_mover": [(10, 10), (10, 6), (10, 2), (10, 1),
                          (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative": [(10, 7), (11, 10)],
    "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate_token": [(0, 1), (0, 10), (3, 0)],
    "previous_token": [(2, 2), (4, 11)],
}
POSITIONS = {k: ("s2" if k in ["induction", "duplicate_token"] else
                 "s1_plus1" if k == "previous_token" else "end") for k in GROUPS}
assert sum(map(len, GROUPS.values())) == 26
TEMPLATES = [
    "When {n1} and {n2} went to the {place}, {s} gave a {obj} to",
    "After {n1} and {n2} visited the {place}, {s} handed a {obj} to",
    "While {n1} and {n2} were at the {place}, {s} passed a {obj} to",
    "Before {n1} and {n2} left the {place}, {s} offered a {obj} to",
]
NAMES = "Alice Bob Carol David Emma Frank Grace Henry Jack James Jane John Julia Kate Laura Lucy Luke Mark Mary Michael Mike Nick Oliver Paul Peter Rachel Richard Robert Rose Sam Sarah Simon Sophie Steve Susan Thomas Tom Victor William Anna Ben Daniel Edward George Helen Joe Kevin Linda Lisa Maria Martin Nancy Patrick Rebecca Roger Steven Tony Walter Wendy".split()
PLACES = "store park garden school office station market hotel museum library beach church hospital theater restaurant".split()
OBJECTS = "book ring ball cup bag box hat map toy key letter flower ticket bottle apple".split()
METRICS = ["margin", "target_probability", "target_log_probability", "top1_correct",
           "pair_correct", "top1_agreement", "pair_agreement", "forward_kl", "total_variation"]


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def text_strings(value):
    """Prompt-text exclusion, excluding non-prompt natural-language metadata."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ["text", "prefix"] and isinstance(v, str):
                yield v
            else:
                yield from text_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from text_strings(v)


def old_strings():
    files = [PROJECT / "data/calibration.json", PROJECT / "data/evaluation.json"]
    files += [PROJECT / "submission-study/data" / (s + ".json")
              for s in ["calibration", "development", "test", "blimp"]]
    absent = [str(p) for p in files if not p.exists()]
    if absent:
        raise FileNotFoundError(f"Required prior-prompt exclusion inputs absent: {absent}")
    return set(t for p in files for t in text_strings(read(p))), files


def encode(tok, text, target, contrast, subject):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    a = text.find(subject)
    b = text.rfind(subject)
    def token_at(start, word):
        indices = [i for i, (x, y) in enumerate(offsets) if y > start and x < start + len(word)]
        if len(indices) != 1:
            raise ValueError("name is not one token in context")
        return indices[0]
    s1, s2 = token_at(a, subject), token_at(b, subject)
    if s1 >= s2:
        raise ValueError("recipient subject is not repeated")
    appended = []
    for name in [target, contrast]:
        ext = tok.encode(text + " " + name, add_special_tokens=False)
        if len(ext) != len(ids) + 1 or ext[:-1] != ids:
            raise ValueError("answer is not one appended token")
        appended.append(ext[-1])
    # No BOS matches the original authors' experiment (prepend_bos=False).
    return {"text": text, "ids": ids, "s1": s1, "s2": s2, "s1_plus1": s1 + 1,
            "end": len(ids) - 1, "target_id": appended[0], "contrast_id": appended[1]}


def prepare(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    old, prior_files = old_strings()
    used = set(old)
    names = [n for n in NAMES if len(tok.encode(" " + n, add_special_tokens=False)) == 1]
    if len(names) < 30:
        raise ValueError("Insufficient tokenizer-qualified names")
    output = args.out
    if (output / "data/screen.json").exists():
        raise RuntimeError("Refusing to overwrite prepared data")
    counts = {"calibration": args.calibration_n, "development": args.development_n, "screen": args.screen_n}
    if any(n % 8 for n in counts.values()):
        raise ValueError("Counts must be multiples of 8 (four templates by two orders)")
    stats = {}
    pools = {}
    for si, (split, n) in enumerate(counts.items()):
        rows, local = [], set()
        rejects = {"overlap": 0, "encoding": 0}
        for g in range(n):
            template, order = g % 4, (g // 4) % 2
            rng = random.Random(args.seed + si * 100000 + g * 101)
            for attempt in range(10000):
                target, subject = rng.sample(names, 2)
                place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
                n1, n2 = (target, subject) if order == 0 else (subject, target)
                text = TEMPLATES[template].format(n1=n1, n2=n2, s=subject, place=place, obj=obj)
                try:
                    recipient = encode(tok, text, target, subject, subject)
                    donors, donor_names = [], []
                    available = [x for x in names if x not in [target, subject]]
                    donor_used = set()
                    for j in range(8):
                        for _ in range(1000):
                            triplet = tuple(rng.sample(available, 3))
                            if triplet not in donor_used:
                                donor_used.add(triplet)
                                break
                        donor_text = TEMPLATES[template].format(
                            n1=triplet[0], n2=triplet[1], s=triplet[2], place=place, obj=obj)
                        donor_ids = tok.encode(donor_text, add_special_tokens=False)
                        if len(donor_ids) != len(recipient["ids"]):
                            raise ValueError("donor/recipient lengths differ")
                        # Apart from the three name slots, token sequences must agree.
                        changed = [k for k, (x, y) in enumerate(zip(donor_ids, recipient["ids"])) if x != y]
                        # Explicit positions of both first names, independent of order.
                        offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                        slots = sorted(set([next(k for k, (x, y) in enumerate(offsets) if y > text.find(nm) and x < text.find(nm) + len(nm)) for nm in [target, subject]] + [recipient["s2"]]))
                        if changed != slots or len(slots) != 3:
                            raise ValueError("ABC corruption changes non-name slots")
                        donors.append({"text": donor_text, "ids": donor_ids})
                        donor_names.append(triplet)
                except ValueError:
                    rejects["encoding"] += 1
                    continue
                texts = [text] + [d["text"] for d in donors]
                if len(set(texts)) != 9 or set(texts) & used or set(texts) & local:
                    rejects["overlap"] += 1
                    continue
                row = {"id": f"{split}-circuit-{g:04d}", "split": split, "group": g,
                       "template": template, "order": "ABBA" if order == 0 else "BABA",
                       "stratum": f"t{template}-o{order}", "target": target, "subject": subject,
                       "recipient": recipient, "donors": donors, "donor_names": donor_names,
                       "name_positions": slots, "place": place, "object": obj}
                row["selection_priority"] = hashlib.sha256(f"261002-priority-{g}".encode()).hexdigest()
                rows.append(row)
                local.update(texts)
                break
            else:
                raise RuntimeError(f"Could not prepare {split} row {g}")
        dump(output / f"data/{split}.json", rows)
        stats[split] = {"groups": n, "unique_strings": len(local), "rejections": rejects,
                        "prior_overlap": len(local & old)}
        pools[split] = local
        used.update(local)
    checkpoint = {p.name: sha(p) for p in args.model_path.iterdir()
                  if p.is_file() and p.suffix in [".json", ".txt", ".safetensors"]}
    dump(output / "data/generation-audit.json", {
        "created_utc": utc(), "seed": args.seed, "counts": stats,
        "cross_split_overlap": {a+"/"+b:len(pools[a]&pools[b]) for a in pools for b in pools if a < b},
        "prior_unique_strings": len(old), "prior_files": {str(p.relative_to(PROJECT)): sha(p) for p in prior_files},
        "all_donor_changes_at_exactly_three_name_slots": True,
        "all_target_contrasts_one_appended_token": True, "prepend_bos": False,
        "selection_uses_no_model_outputs": True, "checkpoint_files": checkpoint,
        "model_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "selection_targets": {"pair_error": args.error_n, "pair_correct": args.correct_n},
        "screen_selection_endpoint": "target logit minus subject logit > 0; ties counted as pair error"})
    print(json.dumps(stats), flush=True)


def keep_mask(row, mode):
    import torch
    length = len(row["recipient"]["ids"])
    mask = torch.zeros((12, length, 12, 1), dtype=torch.bool)
    if mode == "all":
        mask[:] = True
    elif mode == "reference":
        for group, heads in GROUPS.items():
            p = row["recipient"][POSITIONS[group]]
            for layer, head in heads:
                mask[layer, p, head] = True
        assert int(mask.sum()) == 26
    elif mode != "empty":
        raise ValueError(mode)
    return mask


class Engine:
    def __init__(self, model_path, device="cuda", batch=9):
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(4)
        torch.manual_seed(261002)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device, self.batch = device, batch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32,
            attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        assert self.model.config.model_type == "gpt2"
        assert (self.model.config.n_layer, self.model.config.n_head, self.model.config.n_embd) == (12, 12, 768)

    def run(self, ids, capture=False, replacement=None, keep=None):
        import torch
        if not ids or len(set(map(len, ids))) != 1:
            raise ValueError("Unpadded equal-length rows required")
        logs, caches = [], []
        audit = {"outside_changed_max": 0., "replacement_error_max": 0., "write_change_max": 0.}
        with torch.inference_mode():
            for start in range(0, len(ids), self.batch):
                stop = min(start + self.batch, len(ids))
                tensors = torch.tensor(ids[start:stop], device=self.device)
                handles, cache = [], {}
                for layer, block in enumerate(self.model.transformer.h):
                    def hook(module, inputs, layer=layer):
                        original = inputs[0]
                        z = original.view(*original.shape[:2], 12, 64)
                        if capture:
                            cache[layer] = z.clone()
                        if replacement is None:
                            return None
                        rep = replacement[layer, start:stop].to(device=self.device, dtype=z.dtype)
                        retained = keep[layer].to(self.device)[None].expand_as(z)
                        w = torch.where(retained, z, rep)
                        outside = (w-z).masked_select(retained)
                        target = (w-rep).masked_select(~retained)
                        audit["outside_changed_max"] = max(audit["outside_changed_max"], float(outside.abs().max()) if outside.numel() else 0.)
                        audit["replacement_error_max"] = max(audit["replacement_error_max"], float(target.abs().max()) if target.numel() else 0.)
                        audit["write_change_max"] = max(audit["write_change_max"], float((w-z).abs().max()))
                        return (w.reshape_as(original), *inputs[1:])
                    handles.append(block.attn.c_proj.register_forward_pre_hook(hook))
                try:
                    logits = self.model(tensors, use_cache=False, logits_to_keep=1).logits[:, -1]
                finally:
                    for h in handles:
                        h.remove()
                logs.append(logits)
                if capture:
                    caches.append(torch.stack([cache[l] for l in range(12)]))
        return torch.cat(logs), torch.cat(caches, 1) if capture else None, audit

    def environment(self):
        import torch
        return {"created_utc": utc(), "model": "gpt2-small", "dtype": "float32",
                "native_batch": self.batch, "device": self.device, "python": sys.version,
                "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "scipy"]},
                "hook": "attention c_proj input reshaped [batch,position,head,64]",
                "MLPs": "all kept", "attention_projection_biases": "kept unchanged"}


def metrics(logits, reference, row):
    import torch
    logits, reference = logits.double(), reference.double()
    lp = logits.log_softmax(-1)
    ref_lp = reference.log_softmax(-1)
    p, ref_p = lp.exp(), ref_lp.exp()
    good, bad = row["recipient"]["target_id"], row["recipient"]["contrast_id"]
    margin, ref_margin = logits[:, good] - logits[:, bad], reference[:, good] - reference[:, bad]
    argmax, ref_argmax = logits.argmax(-1), reference.argmax(-1)
    return torch.stack([margin, p[:, good], lp[:, good], (argmax == good).double(),
                        (margin > 0).double(), (argmax == ref_argmax).double(),
                        ((margin > 0) == (ref_margin > 0)).double(),
                        (ref_p * (ref_lp - lp)).sum(-1), (ref_p-p).abs().sum(-1)/2], -1).cpu().numpy()


def freeze(args):
    path = args.out / "data/freeze.json"
    if path.exists():
        raise RuntimeError("Freeze already exists; do not overwrite")
    required = [ROOT / "experiment.py", ROOT / "PROTOCOL.md", ROOT.parent / "PROTOCOL.md"]
    required += list((args.out / "data").glob("*.json"))
    required += [args.out / "results/calibration.json", args.out / "results/development-completion.json"]
    required += [args.out / "results" / n for n in ["calibration-initial-f32-projection-failed.json", "projection-calibration-amendment.json", "source-before-projection-remedy.py"] if (args.out / "results" / n).exists()]
    dump(path, {"frozen_utc": utc(), "public_preregistration": False,
                "files": {os.path.relpath(p,args.out): sha(p) for p in required}})


def freeze_check(out):
    f = read(out / "data/freeze.json")
    bad = [p for p, h in f["files"].items() if sha(out/p) != h]
    if bad:
        raise RuntimeError(f"Frozen files changed: {bad}")


def check_model(args):
    expected = read(args.out / "data/generation-audit.json")["checkpoint_files"]
    bad = [p for p, h in expected.items() if sha(args.model_path/p) != h]
    if bad:
        raise RuntimeError(f"Checkpoint mismatch: {bad}")


def calibrate(args):
    import numpy as np
    import torch
    if (args.out / "data/freeze.json").exists():
        freeze_check(args.out)
    check_model(args)
    measured_source_sha256 = sha(Path(__file__))
    engine = Engine(args.model_path, args.device, args.batch)
    rows = read(args.out / "data/calibration.json")
    dest = args.out / "results/calibration.json"
    if dest.exists():
        raise RuntimeError("Refusing to overwrite calibration")
    errors, probability_errors, projection_diagnostics, projection_float64_errors, positive = {}, {}, {}, {}, 0.
    for row in rows:
        p = row["recipient"]["ids"]
        full, own, _ = engine.run([p], capture=True)
        with torch.inference_mode():
            native = engine.model(torch.tensor([p],device=engine.device),use_cache=False,logits_to_keep=1).logits[:,-1]
        errors[row["id"]+"/passive_hook_native"] = float((full-native).abs().max())
        probability_errors[row["id"]+"/passive_hook_native"] = float((full.double().softmax(-1)-native.double().softmax(-1)).abs().max())
        donors, source, _ = engine.run([d["ids"] for d in row["donors"]], capture=True)
        source_mean = source.mean(1, keepdim=True)
        conditions = [("all_keep", source_mean, "all", full),
                      ("self_identity_empty", own, "empty", full),
                      ("self_identity_reference", own, "reference", full)]
        for name, replacement, mode, expected in conditions:
            got, _, audit = engine.run([p], replacement=replacement, keep=keep_mask(row, mode))
            errors[row["id"] + "/" + name] = float((got-expected).abs().max())
            probability_errors[row["id"] + "/" + name] = float((got.double().softmax(-1)-expected.double().softmax(-1)).abs().max())
            errors[row["id"] + "/" + name + "/outside"] = audit["outside_changed_max"]
            errors[row["id"] + "/" + name + "/assigned"] = audit["replacement_error_max"]
            assert audit["outside_changed_max"] == audit["replacement_error_max"] == 0.
        duplicate = source[:, :1].repeat(1, 8, 1, 1, 1).mean(1, keepdim=True)
        replacement = torch.cat([source[:, :1], duplicate], 1)
        log, _, audit = engine.run([p]*2, replacement=replacement, keep=keep_mask(row, "reference"))
        errors[row["id"] + "/duplicate_donor"] = float((log[0]-log[1]).abs().max())
        probability_errors[row["id"] + "/duplicate_donor"] = float((log[0].double().softmax(-1)-log[1].double().softmax(-1)).abs().max())
        assert audit["outside_changed_max"] == audit["replacement_error_max"] == 0.
        positive = max(positive, float((log[0]-full[0]).abs().max()))
        # A full donor's cached pre-projection states, replayed into that same
        # donor, must reconstruct its logits despite capturing eight at once.
        replay, _, _ = engine.run([row["donors"][0]["ids"]],
                                 replacement=source[:, :1], keep=keep_mask(row, "empty"))
        errors[row["id"] + "/donor_reconstruction"] = float((replay[0]-donors[0]).abs().max())
        probability_errors[row["id"] + "/donor_reconstruction"] = float((replay[0].double().softmax(-1)-donors[0].double().softmax(-1)).abs().max())
        # Algebraic equivalence to replacing each projected head result is
        # checked directly (one shared c_proj bias stays outside the head sum).
        for layer, block in enumerate(engine.model.transformer.h):
            x = source[layer, :1]
            with torch.inference_mode():
                linear = torch.einsum("bthd,hdo->bto", x, block.attn.c_proj.weight.view(12,64,768)) + block.attn.c_proj.bias
                native = block.attn.c_proj(x.flatten(-2))
            projection_diagnostics[row["id"] + f"/projection_equivalence_{layer}"] = float((linear-native).abs().max())
            # This is an algebra/layout check, distinct from a native model
            # execution gate. Promote the SAME actual cached states/weights;
            # changing float32 contraction order can differ by >1e-3 in large
            # late-layer activations even when the algebra and head axes match.
            xd = x.double().cpu()
            wd = block.attn.c_proj.weight.double().cpu()
            bd = block.attn.c_proj.bias.double().cpu()
            explicit = torch.einsum("bthd,hdo->bto",xd,wd.view(12,64,768))+bd
            flat = xd.flatten(-2) @ wd + bd
            projection_float64_errors[row["id"]+f"/projection_equivalence_{layer}"] = float((explicit-flat).abs().max())
    passed = max(errors.values()) <= .001 and max(probability_errors.values()) <= .00002 and positive > .01 and max(projection_float64_errors.values()) <= 1e-9
    dump(dest, {"created_utc": utc(), "groups": len(rows), "errors": errors,
                "maximum_error": max(errors.values()), "tolerance": .001,
                "positive_logit_change": positive, "passed": passed,
                "probability_errors":probability_errors,"maximum_probability_error":max(probability_errors.values()),
                "probability_tolerance":.00002,"source_sha256":measured_source_sha256,
                "float32_projection_diagnostics":projection_diagnostics,
                "float32_projection_diagnostic_max":max(projection_diagnostics.values()),
                "float64_projection_errors":projection_float64_errors,
                "float64_projection_error_max":max(projection_float64_errors.values()),
                "float64_projection_tolerance":1e-9,
                "environment": engine.environment()})
    print(json.dumps({"passed": passed, "maximum_error": max(errors.values()), "positive": positive}), flush=True)
    if not passed:
        raise RuntimeError("Calibration failed; no test inference authorized by this program")


def screen(args):
    """Frozen finite screening population; selects only on intact model outputs."""
    import numpy as np
    freeze_check(args.out)
    check_model(args)
    cal = read(args.out / "results/calibration.json")
    if not cal["passed"] or cal["environment"]["native_batch"] != args.batch or cal["environment"]["device"] != args.device:
        raise RuntimeError("Passed calibration required")
    if (args.out / "data/selected-test.json").exists():
        raise RuntimeError("Refusing to rescreen or overwrite selected test data")
    engine = Engine(args.model_path, args.device, args.batch)
    rows = read(args.out / "data/screen.json")
    values, predictions = np.empty((len(rows), len(METRICS))), np.empty(len(rows), dtype=np.int64)
    bylength = {}
    for i, row in enumerate(rows):
        bylength.setdefault(len(row["recipient"]["ids"]), []).append(i)
    done, started = 0, time.time()
    for length, indices in sorted(bylength.items()):
        for start in range(0,len(indices),args.batch):
            ix = indices[start:start+args.batch]
            log, _, _ = engine.run([rows[i]["recipient"]["ids"] for i in ix])
            for j,i in enumerate(ix):
                values[i] = metrics(log[j:j+1],log[j:j+1],rows[i])[0]
                predictions[i] = int(log[j].argmax())
            done += len(ix)
            if done % 256 < args.batch or done == len(rows):
                print(json.dumps({"screened":done,"of":len(rows),"seconds":time.time()-started}),flush=True)
    np.savez_compressed(args.out / "results/screen.npz", metrics=values, prediction=predictions)
    targets = read(args.out / "data/generation-audit.json")["selection_targets"]
    selected, strata = [], {}
    for correct,label in [(False,"pair_error"),(True,"pair_correct")]:
        ix = [i for i in range(len(rows)) if bool(values[i,4])==correct]
        ranked = sorted(ix,key=lambda i:rows[i]["selection_priority"])
        chosen = ranked[:targets[label]]
        strata[label] = {"screen_population_n":len(ix),"selected_n":len(chosen),
                        "inclusion_probability":len(chosen)/len(ix) if ix else None}
        for i in chosen:
            row = dict(rows[i])
            row.update(screen_index=i, selection_stratum=label,
                       population_weight=len(ix)/len(chosen),
                       screen_baseline=values[i].tolist(), screen_prediction=int(predictions[i]))
            selected.append(row)
    selected.sort(key=lambda r:r["screen_index"])
    dump(args.out / "data/selected-test.json",selected)
    audit = {"screened_n":len(rows),"selected_n":len(selected),"strata":strata,
             "created_utc":utc(),"targets":targets,"error_definition":"two-name margin <= 0",
             "pair_errors_available":int((values[:,4]==0).sum()),
             "unrestricted_top1_errors_available":int((values[:,3]==0).sum()),
             "population_baseline_means":dict(zip(METRICS,values.mean(0).tolist())),
             "selected_template_order_counts": {s:sum(r["stratum"]==s for r in selected) for s in sorted({r["stratum"] for r in rows})},
             "selected_test_sha256":sha(args.out / "data/selected-test.json"),
             "screen_sha256":sha(args.out / "results/screen.npz"),
             "selection_does_not_use_any_intervention_output":True,
             "environment":engine.environment(),"seconds":time.time()-started}
    dump(args.out / "results/screen-selection.json",audit)
    print(json.dumps({k:audit[k] for k in ["screened_n","selected_n","pair_errors_available","unrestricted_top1_errors_available"]}),flush=True)


def evaluate(args):
    import numpy as np
    import torch
    development = args.stage == "development"
    if not development:
        freeze_check(args.out)
    check_model(args)
    cal = read(args.out / "results/calibration.json")
    if not cal["passed"] or cal["environment"]["native_batch"] != args.batch or cal["environment"]["device"] != args.device:
        raise RuntimeError("Matching passed calibration required")
    engine = Engine(args.model_path, args.device, args.batch)
    rows = read(args.out / ("data/development.json" if development else "data/selected-test.json"))
    if not development:
        assert sha(args.out / "data/selected-test.json") == read(args.out / "results/screen-selection.json")["selected_test_sha256"]
    if args.stage == "replay":
        rows = [r for s in sorted({r["selection_stratum"] for r in rows}) for r in [r for r in rows if r["selection_stratum"]==s][:4]]
    stage = "development" if development else "replay" if args.stage == "replay" else "test"
    dest = args.out / "results" / stage
    dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for i, row in enumerate(rows):
        path = dest / (row["id"] + ".npz")
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        p = row["recipient"]["ids"]
        full, own, _ = engine.run([p], capture=True)
        _, source, _ = engine.run([d["ids"] for d in row["donors"]], capture=True)
        replacement = torch.cat([source.mean(1, keepdim=True), source], 1)
        outcomes, predictions, mixtures, meanlogits, audits = [], [], [], [], {}
        for mode in ["reference", "empty"]:
            log, _, audit = engine.run([p]*9, replacement=replacement, keep=keep_mask(row, mode))
            assert audit["outside_changed_max"] == audit["replacement_error_max"] == 0.
            outcomes.append(metrics(log, full, row))
            predictions.append(log.argmax(-1).cpu().numpy())
            # Mixture metrics are separate from average per-donor metrics.
            # The latter measures expectation under a sampled joint donor;
            # the former measures a deterministic mixture predictor.
            mixture_logp = torch.logsumexp(log[1:].double().log_softmax(-1), 0, keepdim=True) - np.log(8)
            mixtures.append(metrics(mixture_logp, full, row)[0])
            meanlogits.append(metrics(log[1:].mean(0,keepdim=True),full,row)[0])
            audits[mode] = audit
        identity, _, audit = engine.run([p]*2,
                                       replacement=own.repeat(1,2,1,1,1), keep=keep_mask(row, "empty"))
        errors = float((identity-full).abs().max())
        assert errors <= .001
        keepall, _, _ = engine.run([p], replacement=source[:, :1], keep=keep_mask(row, "all"))
        all_error = float((keepall-full).abs().max())
        assert all_error <= .001
        if not development:
            screen_margin_error = abs(metrics(full,full,row)[0,0]-row["screen_baseline"][0])
            assert screen_margin_error <= .001
            assert int(full.argmax(-1)) == row["screen_prediction"]
            assert int(metrics(full,full,row)[0,4]) == int(row["screen_baseline"][4])
        np.savez_compressed(path, metrics=np.stack(outcomes), prediction=np.stack(predictions),
                            mixture_metrics=np.stack(mixtures), baseline=metrics(full, full, row)[0],
                            mean_logit_metrics=np.stack(meanlogits),
                            baseline_prediction=full.argmax(-1).cpu().numpy(),
                            target_id=np.array(row["recipient"]["target_id"]),
                            contrast_id=np.array(row["recipient"]["contrast_id"]),
                            identity_logit_error=np.array(errors), all_kept_logit_error=np.array(all_error))
        dump(dest / (row["id"] + "-audit.json"), audits)
        if i % 8 == 0 or i == len(rows)-1:
            print(json.dumps({"stage": stage, "group": i+1, "of": len(rows), "seconds": time.time()-started}), flush=True)
    dump(args.out / f"results/{stage}-completion.json", {
        "finished_utc": utc(), "groups": len(rows), "seconds": time.time()-started,
        "per_group_inference_rows": 30, "total_inference_rows": len(rows)*30,
        "conditions": ["full", "8 donor capture", "reference mean and 8 joint samples",
                       "empty attention mean and 8 joint samples", "2 self identity", "all kept"],
        "environment": engine.environment()})
    if stage == "replay":
        errors = {}
        for row in rows:
            with np.load(dest / (row["id"]+".npz")) as a, np.load(args.out / "results/test" / (row["id"]+".npz")) as b:
                errors[row["id"]] = max(float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a.files)
        dump(args.out / "results/replay-verification.json", {"errors": errors,
             "maximum_error": max(errors.values()), "passed": max(errors.values()) <= .001})
        if max(errors.values()) > .001:
            raise RuntimeError("Fresh-process replay failed")


def analyze(args):
    import numpy as np
    from scipy.stats import binomtest
    freeze_check(args.out)
    rows = read(args.out / "data/selected-test.json")
    packed = [np.load(args.out / "results/test" / (r["id"]+".npz")) for r in rows]
    m = np.stack([x["metrics"] for x in packed])
    b = np.stack([x["baseline"] for x in packed])
    mix = np.stack([x["mixture_metrics"] for x in packed])
    meanlogit = np.stack([x["mean_logit_metrics"] for x in packed])
    pred = np.stack([x["prediction"] for x in packed])
    basepred = np.stack([x["baseline_prediction"] for x in packed]).ravel()
    assert np.isfinite(m).all() and np.isfinite(b).all() and np.isfinite(mix).all()
    rng = np.random.default_rng(26100231)
    stratum = np.array([r["selection_stratum"] for r in rows])
    weights = np.array([r["population_weight"] for r in rows])
    draws = np.concatenate([rng.choice(np.where(stratum == s)[0], size=(20000, sum(stratum == s)), replace=True)
                            for s in sorted(set(stratum))], axis=1)
    def est(x, selection=None):
        x = np.asarray(x, dtype=float)
        if selection is None:
            boot = (x[draws]*weights[draws]).sum(1)/weights[draws].sum(1)
            n = len(x)
            point = float(np.average(x,weights=weights))
        else:
            selection = np.asarray(selection, dtype=bool)
            n = int(selection.sum())
            if n == 0:
                return {"mean": None, "ci95": None, "n": 0, "identified": False}
            count = (selection*weights)[draws].sum(1)
            values = (x*selection*weights)[draws].sum(1)
            boot = values[count > 0]/count[count > 0]
            point = float(np.average(x[selection],weights=weights[selection]))
        return {"mean": point, "ci95": np.quantile(boot,[.025,.975]).tolist(), "n": n, "identified": True}
    def ratio(num, den):
        nb = (num[draws]*weights[draws]).sum(1)/weights[draws].sum(1)
        db = (den[draws]*weights[draws]).sum(1)/weights[draws].sum(1)
        if abs(float(np.average(den,weights=weights))) < .1:
            return {"estimate": None, "ci95": None, "denominator_guard_failed": True}
        if np.any(db * np.average(den,weights=weights) <= 0):
            return {"estimate":float(np.average(num,weights=weights)/np.average(den,weights=weights)), "ci95":None, "denominator_guard_failed":False,"interval_cross_zero_guard_failed":True}
        return {"estimate": float(np.average(num,weights=weights)/np.average(den,weights=weights)), "ci95": np.quantile(nb/db,[.025,.975]).tolist(), "denominator_guard_failed": False}
    result = {"created_utc": utc(), "groups": len(rows), "bootstrap": {"draws": 20000, "seed": 26100231,
               "unit": "recipient plus its complete donor bank", "stratification": "intact-model pair success/error only", "weighting":"inverse inclusion probability from finite screening population; template and order are descriptive only", "intervals": "pointwise percentile 95%; conditional recipient variation at fixed screening correctness proportions; no finite-population correction, no screening-prevalence uncertainty, no across-metric multiplicity claim"},
              "model": "gpt2-small", "donor_bank_size": 8, "metric_order": METRICS,
              "screening":read(args.out / "results/screen-selection.json"),
              "baseline": {name: est(b[:, j]) for j, name in enumerate(METRICS)}, "circuits": {}}
    for ci, mode in enumerate(["reference", "empty_attention"]):
        mean, resample = m[:, ci, 0], m[:, ci, 1:].mean(1)
        out = {"mean": {name: est(mean[:,j]) for j,name in enumerate(METRICS)},
               "expected_joint_resample": {name: est(resample[:,j]) for j,name in enumerate(METRICS)},
               "mean_minus_expected_resample": {name: est(mean[:,j]-resample[:,j]) for j,name in enumerate(METRICS)},
               "mixture_predictor": {name: est(mix[:,ci,j]) for j,name in enumerate(METRICS)},
               "mean_logit_predictor": {name: est(meanlogit[:,ci,j]) for j,name in enumerate(METRICS)},
               "mean_absolute_margin_gap": est(abs(mean[:,0]-resample[:,0])),
               "mean_margin_retention": ratio(mean[:,0],b[:,0]),
               "resample_margin_retention": ratio(resample[:,0],b[:,0]),
               "mean_empty_adjusted_faithfulness": ratio(mean[:,0]-m[:,1,0,0],b[:,0]-m[:,1,0,0]),
               "resample_empty_adjusted_faithfulness": ratio(resample[:,0]-m[:,1,1:,0].mean(1),b[:,0]-m[:,1,1:,0].mean(1)),
               "mean_empty_adjusted_denominator": est(b[:,0]-m[:,1,0,0]),
               "resample_empty_adjusted_denominator": est(b[:,0]-m[:,1,1:,0].mean(1)),
               "mean_abs_margin_error_from_full": est(abs(mean[:,0]-b[:,0])),
               "expected_abs_margin_error_from_full": est(abs(m[:,ci,1:,0]-b[:,None,0]).mean(1)),
               "per_template": {}, "strata": {}}
        for label, sel in [("full_top1_success", b[:,3] == 1), ("full_top1_error", b[:,3] == 0),
                           ("full_pair_success", b[:,4] == 1), ("full_pair_error", b[:,4] == 0)]:
            out["strata"][label] = {"n": int(sel.sum()),
                "mean_exact_answer_agreement": est(mean[:,5],sel),
                "expected_resample_exact_answer_agreement": est(resample[:,5],sel),
                "paired_agreement_difference": est(mean[:,5]-resample[:,5],sel),
                "paired_pair_agreement_difference": est(mean[:,6]-resample[:,6],sel),
                "mean_pair_agreement": est(mean[:,6],sel),
                "expected_resample_pair_agreement": est(resample[:,6],sel)}
        for t in range(4):
            sel = np.array([r["template"] == t for r in rows])
            out["per_template"][str(t)] = {"gap":est(mean[:,0]-resample[:,0],sel),
                                            "top1_agreement_difference":est(mean[:,5]-resample[:,5],sel)}
        # Fraction of individual (recipient, donor) assignments with a decision
        # differing from the deterministic mean intervention; inference remains
        # recipient-clustered, never treats 8 donors as 8 independent samples.
        out["mean_vs_sampled_donor_prediction_disagreement"] = est((pred[:,ci,0,None] != pred[:,ci,1:]).mean(1))
        out["mean_vs_sampled_donor_pair_disagreement"] = est(((m[:,ci,0,0,None] > 0) != (m[:,ci,1:,0] > 0)).mean(1))
        result["circuits"][mode] = out
    result["verification"] = {"all_finite": True,
        "maximum_identity_logit_error": max(float(x["identity_logit_error"]) for x in packed),
        "maximum_all_kept_logit_error": max(float(x["all_kept_logit_error"]) for x in packed),
        "source_assignment_is_joint_across_all_ablated_nodes": True,
        "donor_index_is_shared_across_layers_heads_and_positions":True,
        "frozen_files_checked": True}
    dump(args.out / "results/analysis.json", result)
    print(json.dumps({"groups":len(rows), "reference_mean_minus_resample_margin":result["circuits"]["reference"]["mean_minus_expected_resample"]["margin"],
                      "reference_agreement_difference":result["circuits"]["reference"]["mean_minus_expected_resample"]["top1_agreement"]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "freeze", "calibrate", "development", "screen", "test", "replay", "analyze"])
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=9)
    ap.add_argument("--screen-n", type=int, default=8192)
    ap.add_argument("--error-n", type=int, default=64)
    ap.add_argument("--correct-n", type=int, default=128)
    ap.add_argument("--development-n", type=int, default=32)
    ap.add_argument("--calibration-n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=26100217)
    args = ap.parse_args()
    if args.stage == "prepare": prepare(args)
    elif args.stage == "freeze": freeze(args)
    elif args.stage == "calibrate": calibrate(args)
    elif args.stage == "screen": screen(args)
    elif args.stage in ["development", "test", "replay"]: evaluate(args)
    else: analyze(args)


if __name__ == "__main__":
    main()

"""Study 5 stages: calibration, panel runners, gates and replays (run inside Modal; see modal_app.py).

Panels (PROTOCOL.md):
  A   components at the agreement final token, both directions (Study 4's run_group)
  B1  every head at the IOI final token, noising (Study 3's ioi_row, ioi-end)
  B2  every head at the IOI repeated subject, noising (Study 3's ioi_row, ioi-s2)
  C1  residual sweep of the agreement prompts, both directions (Study 2's sva_group)
  C2  residual sweep of the IOI prompts, noising (Study 2's ioi_row)
Vectors per site keep the inherited order: A [64 IND, MEAN, 8 TOK, 8 MIX]; B [64 IND, MEAN, 8 MIX, TOK groups];
C1 [64 IND, 8 MIX, 8 NOUN, 8 ADJ, MEAN64]; C2 [64 IND, 8 MIX, MEAN64].
Each runner takes mode "study" (prefix cache, batched sites) or "reference" (one site per call, full recomputation)
and returns the arrays, the write audit and the built-in checks of PROTOCOL.md stage 6.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from common import ADJ, KINDS, LOGIT_TOL, MIX, NOUN, PROB_TOL, utc

CHECK_LAYER = 14


def margin(logits, good, bad):
    return (logits[:, good].double() - logits[:, bad].double()).cpu().numpy()


def msr(x):
    x = x.double()
    return float(((x - x.mean(0, keepdim=True)) ** 2).sum(-1).mean())


def spreads17(r):
    """Study 4: r [64, d] in residual space, index s*8+n -> all 64, eight NOUN banks, eight MIX banks."""
    out = [msr(r)]
    for banks in (NOUN, MIX):
        out += [msr(r[[s * 8 + n for s, n in bank]]) for bank in banks]
    return np.array(out)


class Runner:
    """Runs a list of sites at one position with either engine and keeps the audit."""

    def __init__(self, engine, mode):
        self.engine, self.mode = engine, mode
        self.audit = {"outside": 0.0, "write": 0.0}

    def run(self, ids, position, sites, reduce=None):
        """Logits per site, or reduce(logits) per site (the panels keep only margins, so GPU memory stays bounded)."""
        if self.mode == "study":
            logits, audit = self.engine.patch_sites(ids, position, sites, reduce)
            self._note(audit)
            return logits
        out = []
        for kind, layer, head, vectors in sites:
            log, audit = self.engine.reference_patch(ids, kind, layer, position, vectors, head)
            self._note(audit)
            out.append(log if reduce is None else reduce(log))
        return out

    def _note(self, audit):
        self.audit["outside"] = max(self.audit["outside"], audit["outside_max"])
        self.audit["write"] = max(self.audit["write"], audit["target_error"])


def identity_checks(runner, ids, position, own, reference_logits, checks, tag):
    """own: {kind: (layer, head, vector)}; each patched with the recipient's own value must reproduce its logits."""
    sites = [(kind, layer, head, vec[None]) for kind, (layer, head, vec) in own.items()]
    logs = runner.run(ids, position, sites)
    for (kind, *_), log in zip(sites, logs):
        checks[f"{tag}/identity/{kind}"] = float((log[0] - reference_logits).abs().max())
    if runner.mode == "study":
        checks[f"{tag}/cached_passive"] = float((runner.engine.cached_logits(ids, position) - reference_logits).abs().max())


def finish(out, runner, checks, started):
    gated = {k: v for k, v in checks.items()}
    worst = max(gated.values()) if gated else 0.0
    record = {"seconds": time.time() - started, "audit": runner.audit, "checks": checks, "max_check_error": worst,
              "passed": bool(worst <= LOGIT_TOL and runner.audit["outside"] == 0 and runner.audit["write"] == 0)}
    return out, record


# ---------------------------------------------------------------------------------------------- panel A
def run_A(engine, row, mode="study"):
    started, runner, checks = time.time(), Runner(engine, mode), {}
    ids, good, bad, end = row["ids"], row["good_id"], row["bad_id"], row["end"]
    logits, comp = engine.capture(ids, [end])
    base = margin(logits, good, bad)
    L, H = engine.n_layers, engine.n_heads
    out = {f"margins_{k}": np.zeros((2, L, H, 81) if k == "head" else (2, L, 81)) for k in KINDS}
    out.update({f"spread_{k}": np.zeros((2, L, H, 17) if k == "head" else (2, L, 17)) for k in KINDS})
    for d, (recipient, side) in enumerate([(1, 0), (0, 1)]):
        idx = [row["grid"][s][n][side] for s in range(8) for n in range(8)]
        sites, where = [], []
        for kind in KINDS:
            for layer in range(L):
                for head in (range(H) if kind == "head" else [None]):
                    v = comp[kind][layer, idx, 0] if head is None else comp[kind][layer, idx, 0, head]     # [64, dim]
                    tok = torch.stack([v[[s * 8 + n for s, n in bank]].mean(0) for bank in NOUN])
                    mix = torch.stack([v[[s * 8 + n for s, n in bank]].mean(0) for bank in MIX])
                    sites.append((kind, layer, head, torch.cat([v, v.mean(0, keepdim=True), tok, mix])))
                    r = engine.head_in_residual(v, layer, head) if kind == "head" else v
                    w = (d, layer) if head is None else (d, layer, head)
                    out[f"spread_{kind}"][w] = spreads17(r.double())
                    where.append((kind, w))
        margins = runner.run(ids[recipient], end, sites, reduce=lambda l: margin(l, good, bad))
        for (kind, w), m in zip(where, margins):
            out[f"margins_{kind}"][w] = m
        own = {kind: (CHECK_LAYER, 0 if kind == "head" else None,
                      comp[kind][CHECK_LAYER, recipient, 0] if kind != "head" else comp[kind][CHECK_LAYER, recipient, 0, 0])
               for kind in KINDS}
        identity_checks(runner, ids[recipient], end, own, logits[recipient], checks, ["restore", "destroy"][d])
    out["baseline"] = base[:2]
    return finish(out, runner, checks, started)


# ---------------------------------------------------------------------------------------------- panels B1, B2
def run_B(engine, row, setting, mode="study"):
    started, runner, checks = time.time(), Runner(engine, mode), {}
    rec = row["recipient"]
    ids = [rec["ids"]] + row["donors"]
    pos = rec["end"] if setting == "ioi-end" else rec["s2"]
    logits, comp = engine.capture(ids, [pos], kinds=("head",))
    z = comp["head"]                                                                       # [L, 65, 1, H, 128]
    base = margin(logits, rec["target_id"], rec["contrast_id"])
    keys = row["s2_tokens"] if setting == "ioi-s2" else [rec["ids"][rec["end"]]] * 64
    groups = sorted(set(keys))
    members = [[j for j, k in enumerate(keys) if k == g] for g in groups]
    n_tok = len(groups) if len(groups) > 1 else 0                       # one group means the token rule equals the mean
    out = np.zeros((engine.n_layers, engine.n_heads, 64 + 1 + 8 + n_tok))
    sites = []
    for layer in range(engine.n_layers):
        for head in range(engine.n_heads):
            v = z[layer, 1:, 0, head]
            parts = [v, v.mean(0, keepdim=True), torch.stack([v[8 * b:8 * b + 8].mean(0) for b in range(8)])]
            if n_tok:
                parts.append(torch.stack([v[m].mean(0) for m in members]))
            sites.append(("head", layer, head, torch.cat(parts)))
    margins = runner.run(rec["ids"], pos, sites, reduce=lambda l: margin(l, rec["target_id"], rec["contrast_id"]))
    for (_, layer, head, _), m in zip(sites, margins):
        out[layer, head] = m
    identity_checks(runner, rec["ids"], pos, {"head": (CHECK_LAYER, 0, z[CHECK_LAYER, 0, 0, 0])}, logits[0], checks, setting)
    data = {"margins": out, "baseline": base[:1], "donor_margins": base[1:],
            "token_group_sizes": np.array([len(m) for m in members]), "token_groups_evaluated": np.array(n_tok)}
    return finish(data, runner, checks, started)


# ---------------------------------------------------------------------------------------------- panel C1
def sva_vectors(states, grid, side, layer, pos):
    """Study 2: 64 individual donor states (index s*8+n), then MIX8, NOUN8, ADJ8 centroids and MEAN64."""
    idx = [grid[s][n][side] for s in range(8) for n in range(8)]
    v = states[layer, idx, pos]
    cents, spread = [], []
    for partition in (MIX, NOUN, ADJ):
        for bank in partition:
            rows = v[[s * 8 + n for s, n in bank]]
            cents.append(rows.mean(0))
            spread.append(msr(rows))
    cents.append(v.mean(0))
    spread.append(msr(v))
    return torch.cat([v, torch.stack(cents)]), spread


def run_C1(engine, row, mode="study"):
    started, runner, checks = time.time(), Runner(engine, mode), {}
    ids, good, bad = row["ids"], row["good_id"], row["bad_id"]
    positions = [row["adj"]] + list(range(row["anchor"], row["end"] + 1))
    logits, comp = engine.capture(ids, positions, kinds=("resid",))
    states = comp["resid"]                                                                 # [L, 130, P, 2048]
    base = margin(logits, good, bad)
    out = np.zeros((2, engine.n_layers, len(positions), 89))
    spread = np.zeros((2, engine.n_layers, len(positions), 25))
    for d, (recipient, side) in enumerate([(1, 0), (0, 1)]):
        for pi, pos in enumerate(positions):
            sites = []
            for layer in range(engine.n_layers):
                vectors, s = sva_vectors(states, row["grid"], side, layer, pi)
                sites.append(("resid", layer, None, vectors))
                spread[d, layer, pi] = s
            margins = runner.run(ids[recipient], pos, sites, reduce=lambda l: margin(l, good, bad))
            for layer, m in enumerate(margins):
                out[d, layer, pi] = m
            identity_checks(runner, ids[recipient], pos, {"resid": (CHECK_LAYER, None, states[CHECK_LAYER, recipient, pi])},
                            logits[recipient], checks, f"{['restore', 'destroy'][d]}/p{pos}")
    data = {"margins": out, "spread": spread, "baseline": base[:2], "positions": np.array(positions),
            "position_names": np.array(["adj"] + [f"anchor+{k}" for k in range(len(positions) - 1)])}
    return finish(data, runner, checks, started)


# ---------------------------------------------------------------------------------------------- panel C2
def run_C2(engine, row, mode="study"):
    started, runner, checks = time.time(), Runner(engine, mode), {}
    rec = row["recipient"]
    ids = [rec["ids"]] + row["donors"]
    t = len(rec["ids"])
    logits, comp = engine.capture(ids, list(range(t)), kinds=("resid",))
    states = comp["resid"]                                                                 # [L, 65, T, 2048]
    base = margin(logits, rec["target_id"], rec["contrast_id"])
    out = np.full((engine.n_layers, t, 73), np.nan)
    spread = np.zeros((engine.n_layers, t, 9))
    null_positions = []
    for pos in range(t):
        if float((states[:, 1:, pos] - states[:, :1, pos]).abs().max()) == 0.0:
            null_positions.append(pos)                       # every donor state equals the recipient state here
            out[:, pos] = base[0]
            continue
        sites = []
        for layer in range(engine.n_layers):
            v = states[layer, 1:, pos]
            cents = [v[8 * b:8 * b + 8].mean(0) for b in range(8)] + [v.mean(0)]
            spread[layer, pos] = [msr(v[8 * b:8 * b + 8]) for b in range(8)] + [msr(v)]
            sites.append(("resid", layer, None, torch.cat([v, torch.stack(cents)])))
        margins = runner.run(rec["ids"], pos, sites, reduce=lambda l: margin(l, rec["target_id"], rec["contrast_id"]))
        for layer, m in enumerate(margins):
            out[layer, pos] = m
        identity_checks(runner, rec["ids"], pos, {"resid": (CHECK_LAYER, None, states[CHECK_LAYER, 0, pos])}, logits[0],
                        checks, f"p{pos}")
    data = {"margins": out, "spread": spread, "baseline": base[:1], "donor_margins": base[1:],
            "null_positions": np.array(null_positions, dtype=np.int64)}
    return finish(data, runner, checks, started)


RUNNERS = {"A": run_A, "B1": lambda e, r, mode="study": run_B(e, r, "ioi-end", mode),
           "B2": lambda e, r, mode="study": run_B(e, r, "ioi-s2", mode), "C1": run_C1, "C2": run_C2}
TASK = {"A": "sva", "C1": "sva", "B1": "ioi", "B2": "ioi", "C2": "ioi"}


# ---------------------------------------------------------------------------------------------- calibration
def prob_err(a, b):
    return float((a.double().softmax(-1) - b.double().softmax(-1)).abs().max())


def calibrate(engine, sva_rows, ioi_rows):
    """Reference-engine checks of Studies 2-4, then the study engine against the reference engine."""
    errors, probs, agreement = {}, {}, {}
    L, H = engine.n_layers, engine.n_heads
    layers = [0, L // 2, L - 1]
    ref, study = Runner(engine, "reference"), Runner(engine, "study")

    def compare(tag, ids, position, sites):
        a = ref.run(ids, position, sites)
        b = study.run(ids, position, sites)
        agreement[tag] = max(float((x - y).abs().max()) for x, y in zip(a, b))

    for row in sva_rows:
        ids, end = row["ids"], row["end"]
        positions = [row["adj"] - 1, row["adj"]] + list(range(row["anchor"], end + 1))
        logits, comp = engine.capture(ids, [end])
        _, sweep = engine.capture(ids, positions, kinds=("resid",))
        recipient, donor = 1, row["grid"][0][0][0]
        for r in (0, 1):
            native = engine.native(ids[r])
            errors[f"{row['id']}/passive_{r}"] = float((native - logits[r]).abs().max())
            probs[f"{row['id']}/passive_{r}"] = prob_err(native[None], logits[r:r + 1])
        for kind in KINDS:
            for layer in layers:
                for head in ([0, H - 1] if kind == "head" else [None]):
                    pick = (lambda i: comp[kind][layer, i, 0]) if head is None else (lambda i: comp[kind][layer, i, 0, head])
                    vec = torch.stack([pick(recipient), pick(donor), pick(donor).repeat(8, 1).mean(0)])
                    log, audit = engine.reference_patch(ids[recipient], kind, layer, end, vec, head)
                    k = f"{row['id']}/{kind}/{layer}/{head}"
                    errors[k + "/identity"] = float((log[0] - logits[recipient]).abs().max())
                    probs[k + "/identity"] = prob_err(log[:1], logits[recipient:recipient + 1])
                    errors[k + "/duplicate"] = float((log[1] - log[2]).abs().max())
                    probs[k + "/duplicate"] = prob_err(log[1:2], log[2:3])
                    errors[k + "/outside"], errors[k + "/write"] = audit["outside_max"], audit["target_error"]
                    if kind == "resid" and layer == L - 1:
                        errors[k + "/final_block_reproduces_donor"] = float((log[1] - logits[donor]).abs().max())
        pre = 0                                                        # identical prefix: the donor state equals the recipient's
        for layer in layers:
            vec = sweep["resid"][layer, donor, pre][None]
            log, audit = engine.reference_patch(ids[recipient], "resid", layer, positions[pre], vec)
            errors[f"{row['id']}/resid/{layer}/p{positions[pre]}/structural_null"] = float((log[0] - logits[recipient]).abs().max())
        # Study engine against reference engine: panel A sites at three layers, C1 sites at every sweep position.
        for d, (rcp, side) in enumerate([(1, 0), (0, 1)]):
            idx = [row["grid"][s][n][side] for s in range(8) for n in range(8)]
            sites = []
            for kind in KINDS:
                for layer in layers:
                    for head in (range(H) if kind == "head" else [None]):
                        v = comp[kind][layer, idx, 0] if head is None else comp[kind][layer, idx, 0, head]
                        tok = torch.stack([v[[s * 8 + n for s, n in bank]].mean(0) for bank in NOUN])
                        mix = torch.stack([v[[s * 8 + n for s, n in bank]].mean(0) for bank in MIX])
                        sites.append((kind, layer, head, torch.cat([v, v.mean(0, keepdim=True), tok, mix])))
            compare(f"{row['id']}/A/d{d}", ids[rcp], end, sites)
            for pi, pos in enumerate(positions[1:], start=1):
                sites = [("resid", layer, None, sva_vectors(sweep["resid"], row["grid"], side, layer, pi)[0]) for layer in layers]
                compare(f"{row['id']}/C1/d{d}/p{pos}", ids[rcp], pos, sites)
    for row in ioi_rows:
        rec = row["recipient"]
        ids = [rec["ids"]] + row["donors"]
        t = len(rec["ids"])
        logits, comp = engine.capture(ids, list(range(t)), kinds=("resid", "head"))
        native = engine.native(rec["ids"])
        errors[f"{row['id']}/passive"] = float((native - logits[0]).abs().max())
        probs[f"{row['id']}/passive"] = prob_err(native[None], logits[:1])
        for pos in [0, rec["s2"], rec["end"]]:
            for layer in layers:
                for kind, head in (("resid", None), ("head", 0), ("head", H - 1)):
                    pick = (lambda i: comp["resid"][layer, i, pos]) if head is None else (lambda i: comp["head"][layer, i, pos, head])
                    vec = torch.stack([pick(0), pick(1), pick(1).repeat(8, 1).mean(0)])
                    log, audit = engine.reference_patch(rec["ids"], kind, layer, pos, vec, head)
                    k = f"{row['id']}/{kind}/{layer}/{head}/p{pos}"
                    errors[k + "/identity"] = float((log[0] - logits[0]).abs().max())
                    probs[k + "/identity"] = prob_err(log[:1], logits[:1])
                    errors[k + "/duplicate"] = float((log[1] - log[2]).abs().max())
                    probs[k + "/duplicate"] = prob_err(log[1:2], log[2:3])
                    errors[k + "/outside"], errors[k + "/write"] = audit["outside_max"], audit["target_error"]
                    if pos == 0 and kind == "resid":
                        errors[k + "/structural_null"] = float((log[1] - logits[0]).abs().max())
                    if kind == "resid" and layer == L - 1 and pos == rec["end"]:
                        errors[k + "/final_block_reproduces_donor"] = float((log[1] - logits[1]).abs().max())
        keys = row["s2_tokens"]
        members = [[j for j, k in enumerate(keys) if k == g] for g in sorted(set(keys))]
        sites = []
        for head in range(H):
            v = comp["head"][L // 2, 1:, rec["s2"], head]
            sites.append(("head", L // 2, head, torch.cat([v, v.mean(0, keepdim=True),
                                                           torch.stack([v[8 * b:8 * b + 8].mean(0) for b in range(8)]),
                                                           torch.stack([v[m].mean(0) for m in members])])))
        compare(f"{row['id']}/B2/heads-L{L // 2}", rec["ids"], rec["s2"], sites)
        for pos in range(1, t):
            sites = []
            for layer in layers:
                v = comp["resid"][layer, 1:, pos]
                cents = [v[8 * b:8 * b + 8].mean(0) for b in range(8)] + [v.mean(0)]
                sites.append(("resid", layer, None, torch.cat([v, torch.stack(cents)])))
            compare(f"{row['id']}/C2/p{pos}", rec["ids"], pos, sites)
    exact = all(v == 0 for k, v in errors.items() if k.endswith(("/outside", "/write")))
    exact = exact and ref.audit["outside"] == 0 and ref.audit["write"] == 0 and study.audit["outside"] == 0 and study.audit["write"] == 0
    gated = max(v for k, v in errors.items() if not k.endswith(("/outside", "/write")))
    worst_agreement = max(agreement.values())
    return {"created_utc": utc(), "errors": errors, "probability_errors": probs, "engine_agreement": agreement,
            "maximum_logit_error": gated, "maximum_probability_error": max(probs.values()),
            "maximum_engine_difference": worst_agreement, "exact_write_locality": exact,
            "tolerance": {"logits": LOGIT_TOL, "probability": PROB_TOL},
            "passed": bool(gated <= LOGIT_TOL and max(probs.values()) <= PROB_TOL and exact and worst_agreement <= LOGIT_TOL)}


# ---------------------------------------------------------------------------------------------- gates
def share(flags):
    return float(np.mean(flags)) if flags else None


@torch.inference_mode()
def gates(engine, sva_rows, ioi_rows):
    """Intact-model competence on the test prompts (PROTOCOL.md, "Competence gates")."""
    clean, corrupt = [], []
    for row in sva_rows:
        logits, _ = engine.capture(row["ids"][:2], [row["end"]], kinds=())
        m = margin(logits, row["good_id"], row["bad_id"])
        clean.append(float(m[0]))
        corrupt.append(float(m[1]))
    ioi = []
    for row in ioi_rows:
        rec = row["recipient"]
        logits, _ = engine.capture([rec["ids"]], [rec["end"]], kinds=())
        ioi.append(float(margin(logits, rec["target_id"], rec["contrast_id"])[0]))
    acc_clean = float(np.mean([m > 0 for m in clean]))
    acc_corrupt = float(np.mean([m < 0 for m in corrupt]))
    acc_ioi = float(np.mean([m > 0 for m in ioi]))
    return {"created_utc": utc(), "required": 0.90,
            "agreement": {"groups": len(sva_rows), "clean_accuracy": acc_clean, "corrupted_accuracy": acc_corrupt,
                          "passed": bool(acc_clean >= 0.90 and acc_corrupt >= 0.90),
                          "ids": [r["id"] for r in sva_rows], "clean_margins": clean, "corrupted_margins": corrupt,
                          "by_stratum": {f"{t}-{c}": {"clean": share([m > 0 for m, r in zip(clean, sva_rows) if (r['template'], r['class']) == (t, c)]),
                                                      "corrupted": share([m < 0 for m, r in zip(corrupt, sva_rows) if (r['template'], r['class']) == (t, c)])}
                                         for t in (0, 1) for c in (0, 1)}},
            "ioi": {"prompts": len(ioi_rows), "io_over_s_accuracy": acc_ioi, "passed": bool(acc_ioi >= 0.90),
                    "ids": [r["id"] for r in ioi_rows], "margins": ioi,
                    "by_stratum": {f"{t}-{o}": share([m > 0 for m, r in zip(ioi, ioi_rows) if (r['template'], r['order']) == (t, o)])
                                   for t in ("O0", "W0") for o in ("ABBA", "BABA")}}}

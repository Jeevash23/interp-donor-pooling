"""Float32 Qwen3 timing and numerical checks on random token inputs (no task prompts); used by modal_smoke.py.

For each model: one capture pass over 128 random prompts, then component patches at the last position as in
Study 4 (residual stream after a block, attention-layer output, MLP output, one head's slice of the input to
o_proj) at the first, middle and last layer, with 81 vectors (Study 4) or 73 (Study 3's IOI settings), and a
Study 2-style residual patch at a middle position with 89 vectors. The calibration checks of Studies 2-4
(passive, identity, duplicate donor, final block reproduces the donor, exact write locality) are run on the
same random inputs. Prompt lengths 11 (agreement) and 14 (IOI).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load(name, path, device, config=None):
    """The pinned checkpoint, or random weights from a config (timing only)."""
    if config is None:
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32,
                                                     attn_implementation="eager").to(device).eval()
        return model, "pinned checkpoint"
    cfg = AutoConfig.from_pretrained(path) if isinstance(config, str) else config
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="eager").eval()
    return model, "random initialisation from the published config (timing only)"


def measure(model, device="cuda", repeats=3, sizes=((11, 130), (14, 65))):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.requires_grad_(False)
    cfg = model.config
    blocks = model.model.layers
    L, H, hd = len(blocks), cfg.num_attention_heads, cfg.head_dim
    rng = np.random.default_rng(0)
    rec = dict(layers=L, heads=H, head_dim=hd, hidden=cfg.hidden_size,
               parameters=int(sum(p.numel() for p in model.parameters())), dtype=str(next(model.parameters()).dtype))

    def module(kind, layer):
        b = blocks[layer]
        return {"resid": b, "attn": b.self_attn, "mlp": b.mlp, "head": b.self_attn.o_proj}[kind]

    @torch.inference_mode()
    def run(ids, kind=None, layer=None, position=None, vectors=None, head=None, capture=None):
        audit = {"outside": 0.0, "write": 0.0}
        store, handles = {}, []
        if capture is not None:
            for l, b in enumerate(blocks):
                handles.append(b.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(
                    ("resid", l), (o[0] if isinstance(o, tuple) else o)[:, capture].clone())))
                handles.append(b.self_attn.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(
                    ("attn", l), o[0][:, capture].clone())))
                handles.append(b.mlp.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(
                    ("mlp", l), o[:, capture].clone())))
                handles.append(b.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: store.__setitem__(
                    ("head", l), a[0][:, capture].reshape(-1, H, hd).clone())))
        if kind is not None:
            sl = slice(head * hd, (head + 1) * hd) if kind == "head" else slice(None)
            vv = vectors.to(device, torch.float32)

            def write(x):
                w = x.clone()
                w[:, position, sl] = vv
                diff = w - x
                diff[:, position, sl] = 0
                audit["outside"] = max(audit["outside"], float(diff.abs().max()))
                audit["write"] = max(audit["write"], float((w[:, position, sl] - vv).abs().max()))
                return w
            m = module(kind, layer)
            if kind == "head":
                handles.append(m.register_forward_pre_hook(lambda mm, a: (write(a[0]), *a[1:])))
            elif kind == "attn":
                handles.append(m.register_forward_hook(lambda mm, a, o: (write(o[0]), *o[1:])))
            else:
                handles.append(m.register_forward_hook(
                    lambda mm, a, o: (write(o[0]), *o[1:]) if isinstance(o, tuple) else write(o)))
        try:
            logits = model(torch.tensor(ids, device=device), use_cache=False, logits_to_keep=1).logits[:, -1]
        finally:
            for h in handles:
                h.remove()
        return logits, store, audit

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def timed(fn):
        sync()
        t = time.perf_counter()
        r = fn()
        sync()
        return r, time.perf_counter() - t

    checks, timings = {}, {}
    for T, n_prompts in sizes:
        ids = rng.integers(1000, min(150000, cfg.vocab_size), size=(n_prompts, T)).tolist()
        end = T - 1
        (logits, store, _), cap = timed(lambda: run(ids[:128], capture=end))
        timings[f"capture_T{T}_{min(128, n_prompts)}_prompts"] = cap
        recipient, donor = 1, 2
        alone, _, _ = run([ids[recipient]])
        checks[f"T{T}/passive"] = float((alone[0] - logits[recipient]).abs().max())
        n_vec = 81 if T == 11 else 73                          # Study 4 / Study 3 IOI vectors per site
        for kind in ("resid", "attn", "mlp", "head"):
            for layer in (0, L // 2, L - 1):
                head = H - 1 if kind == "head" else None
                pick = (lambda i: store[(kind, layer)][i]) if head is None else (lambda i: store[(kind, layer)][i, head])
                vec = torch.stack([pick(recipient), pick(donor), pick(donor).repeat(8, 1).mean(0)])
                log, _, audit = run([ids[recipient]] * 3, kind, layer, end, vec, head)
                k = f"T{T}/{kind}/{layer}"
                checks[k + "/identity"] = float((log[0] - logits[recipient]).abs().max())
                checks[k + "/duplicate"] = float((log[1] - log[2]).abs().max())
                checks[k + "/outside"], checks[k + "/write"] = audit["outside"], audit["write"]
                if kind == "resid" and layer == L - 1:
                    checks[k + "/final_block_reproduces_donor"] = float((log[1] - logits[donor]).abs().max())
                v = torch.stack([pick(i) for i in range(64)])
                vectors = torch.cat([v, v[:n_vec - 64]])
                run([ids[recipient]] * n_vec, kind, layer, end, vectors, head)               # warm-up
                secs = [timed(lambda: run([ids[recipient]] * n_vec, kind, layer, end, vectors, head))[1]
                        for _ in range(repeats)]
                timings[f"patch_T{T}_{kind}_L{layer}_{n_vec}"] = float(np.median(secs))
        mid = T // 2                                           # Study 2-style sweep site, 89 vectors
        _, store_mid, _ = run(ids[:89], capture=mid)
        v = store_mid[("resid", L // 2)]
        vectors = v[torch.arange(89) % len(v)]
        run([ids[recipient]] * 89, "resid", L // 2, mid, vectors)
        secs = [timed(lambda: run([ids[recipient]] * 89, "resid", L // 2, mid, vectors))[1] for _ in range(repeats)]
        timings[f"patch_T{T}_resid_mid_89"] = float(np.median(secs))
    gated = {k: v for k, v in checks.items() if not k.endswith(("/outside", "/write"))}
    rec.update(checks=checks, timings=timings, max_logit_error=max(gated.values()),
               exact_write_locality=all(v == 0 for k, v in checks.items() if k.endswith(("/outside", "/write"))))
    if device == "cuda":
        rec["peak_gpu_gb"] = float(torch.cuda.max_memory_allocated() / 1e9)
    return rec


def timing(models, device="cuda"):
    """models: {name: (path, config or None)}. A failure in one model is recorded and the next one still runs."""
    out = {}
    for name, (path, config) in models.items():
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            model, weights = load(name, path, device, config)
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            out[name] = dict(weights=weights, load_seconds=seconds, **measure(model, device))
            del model
        except Exception as error:                              # noqa: BLE001
            out[name] = dict(error=repr(error))
    return out

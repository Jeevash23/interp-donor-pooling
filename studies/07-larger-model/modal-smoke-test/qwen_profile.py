"""Where the time goes in float32 Qwen3-1.7B patching, on random token inputs only (used by modal_profile.py).

1. Plain forward passes: batch size 1-1,296 at 11 tokens, with deterministic algorithms on and off; 81 rows at
   1-22 tokens.
2. Several sites in one batch: the 16 heads of one layer (16 x 81 rows), each row patching its own head, and
   four layers of the residual stream (4 x 81 rows), against one site per call; the logits must agree with
   the one-site runs within 0.001.
3. Prefix caching for patches at the last position: the first T-1 positions are computed once, the last
   position is run for every vector; the logits must agree with full recomputation within 0.001.
4. The CUDA kernels that take the most time in one 81-row forward pass.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def run_profile(model, device="cuda", repeats=3):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.requires_grad_(False)
    cfg = model.config
    blocks = model.model.layers
    L, H, hd = len(blocks), cfg.num_attention_heads, cfg.head_dim
    rng = np.random.default_rng(1)
    out = {}

    def sync():
        torch.cuda.synchronize()

    def timed(fn):
        fn()
        secs = []
        for _ in range(repeats):
            sync()
            t = time.perf_counter()
            fn()
            sync()
            secs.append(time.perf_counter() - t)
        return float(np.median(secs))

    @torch.inference_mode()
    def forward(ids, writes=(), cache=None):
        """writes: [(kind, layer, head, rows, position, vectors)], each applied only to its rows."""
        handles = []
        for kind, layer, head, rows, position, vv in writes:
            sl = slice(head * hd, (head + 1) * hd) if kind == "head" else slice(None)
            rows_t = torch.as_tensor(rows, device=device)

            def write(x, rows_t=rows_t, sl=sl, position=position, vv=vv):
                w = x.clone()
                w[rows_t, position, sl] = vv
                return w
            b = blocks[layer]
            if kind == "head":
                handles.append(b.self_attn.o_proj.register_forward_pre_hook(lambda m, a, f=write: (f(a[0]), *a[1:])))
            elif kind == "attn":
                handles.append(b.self_attn.register_forward_hook(lambda m, a, o, f=write: (f(o[0]), *o[1:])))
            elif kind == "mlp":
                handles.append(b.mlp.register_forward_hook(lambda m, a, o, f=write: f(o)))
            else:
                handles.append(b.register_forward_hook(
                    lambda m, a, o, f=write: (f(o[0]), *o[1:]) if isinstance(o, tuple) else f(o)))
        try:
            ids_t = torch.tensor(ids, device=device)
            if cache is None:
                return model(ids_t, use_cache=False, logits_to_keep=1).logits[:, -1]
            return model(ids_t, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
        finally:
            for h in handles:
                h.remove()

    @torch.inference_mode()
    def capture(ids, position):
        store, handles = {}, []
        for l, b in enumerate(blocks):
            handles.append(b.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(
                ("resid", l), (o[0] if isinstance(o, tuple) else o)[:, position].clone())))
            handles.append(b.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: store.__setitem__(
                ("head", l), a[0][:, position].reshape(-1, H, hd).clone())))
        try:
            model(torch.tensor(ids, device=device), use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        return store

    # 1. Plain forward passes.
    T = 11
    ids = rng.integers(1000, 150000, size=(1296, 22)).tolist()
    scaling = {}
    for det in (True, False):
        torch.use_deterministic_algorithms(det)
        for B in (1, 8, 32, 81, 162, 324, 648, 1296):
            scaling[f"det{int(det)}_T{T}_B{B}"] = timed(lambda: forward([x[:T] for x in ids[:B]]))
    torch.use_deterministic_algorithms(True)
    for t in (1, 6, 11, 14, 22):
        scaling[f"det1_T{t}_B81"] = timed(lambda: forward([x[:t] for x in ids[:81]]))
    out["plain_forward_seconds"] = scaling

    # 2. Several sites in one batch.
    donors = [x[:T] for x in ids[:64]]
    recipient = ids[100][:T]
    end = T - 1
    store = capture(donors, end)
    layer = L // 2
    v_head = [store[("head", layer)][:, h] for h in range(H)]                 # 16 x [64, hd]
    vec81 = lambda v: torch.cat([v, v[:17]])                                  # 81 rows, as a Study 4 site
    one_site = []
    for h in range(H):
        one_site.append(forward([recipient] * 81, [("head", layer, h, list(range(81)), end, vec81(v_head[h]))]))
    t_one = timed(lambda: [forward([recipient] * 81, [("head", layer, h, list(range(81)), end, vec81(v_head[h]))])
                           for h in range(H)])
    writes = [("head", layer, h, list(range(81 * h, 81 * (h + 1))), end, vec81(v_head[h])) for h in range(H)]
    multi = forward([recipient] * (81 * H), writes)
    t_multi = timed(lambda: forward([recipient] * (81 * H), writes))
    err_heads = float((multi - torch.cat(one_site)).abs().max())
    layers4 = [0, L // 3, 2 * L // 3, L - 1]
    one_r = [forward([recipient] * 81, [("resid", l, None, list(range(81)), end, vec81(store[("resid", l)]))]) for l in layers4]
    writes_r = [("resid", l, None, list(range(81 * i, 81 * (i + 1))), end, vec81(store[("resid", l)]))
                for i, l in enumerate(layers4)]
    multi_r = forward([recipient] * (81 * 4), writes_r)
    err_resid = float((multi_r - torch.cat(one_r)).abs().max())
    out["multi_site"] = dict(heads_16_one_per_call_seconds=t_one, heads_16_one_batch_seconds=t_multi,
                             max_logit_difference_heads=err_heads, max_logit_difference_resid_4_layers=err_resid)

    # 3. Prefix caching at the last position.
    @torch.inference_mode()
    def prefix_cache():
        return model(torch.tensor([recipient[:-1]], device=device), use_cache=True).past_key_values

    base_cache = prefix_cache()
    cached_err, cached_time = {}, {}
    for kind, head in (("resid", None), ("head", H - 1)):
        v = vec81(store[(kind, layer)] if head is None else store[("head", layer)][:, head])
        full = forward([recipient] * 81, [(kind, layer, head, list(range(81)), end, v)])

        def cached_call(B=81, v=v, kind=kind, head=head):
            c = copy.deepcopy(base_cache)
            c.batch_repeat_interleave(B)
            vv = v if B == 81 else v.repeat(B // 81, 1)
            return forward([[recipient[-1]]] * B, [(kind, layer, head, list(range(B)), 0, vv)], cache=c)
        cached_err[kind] = float((cached_call() - full).abs().max())
        cached_time[f"{kind}_B81"] = timed(lambda: cached_call(81))
        cached_time[f"{kind}_B1296"] = timed(lambda: cached_call(1296))
    out["prefix_cache"] = dict(max_logit_difference_vs_full=cached_err, seconds=cached_time,
                               full_recompute_B81_seconds=scaling["det1_T11_B81"])

    # 4. Kernels.
    from torch.profiler import ProfilerActivity, profile
    batch = [x[:T] for x in ids[:81]]
    forward(batch)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        forward(batch)
        sync()
    out["kernels"] = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    out["peak_gpu_gb"] = float(torch.cuda.max_memory_allocated() / 1e9)
    return out

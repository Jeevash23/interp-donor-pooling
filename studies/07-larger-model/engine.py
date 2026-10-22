"""Patching engines for Qwen3-1.7B (Study 5).

Four kinds of site, replaced at one layer and one token position while everything downstream is recomputed
(the ports to Qwen3 of Studies 2-4's engines):
  resid  the output of decoder layer l (model.model.layers[l])
  attn   the attention output, the output of o_proj (no bias in Qwen3)
  mlp    the MLP output
  head   one head's 128-wide slice of the input to o_proj
Float32, eager attention, deterministic algorithms, no TF32.

Two ways to run the patches (PROTOCOL.md, "Engine"):
  reference_patch  the frozen design of Studies 2-4: one site per call, the whole prompt recomputed, no cache;
  patch_sites      the study engine: positions before the patched one are computed once and cached, only the
                   patched position onwards runs for each vector; several sites that share a position go in one
                   batch of at most MAX_ROWS rows, each row writing only its own site; the final hidden state is
                   made contiguous before the output layer.
Every write is audited: values outside the written rows, position and slice must be unchanged, and the written
values must equal the vectors.
"""
from __future__ import annotations

import copy
import importlib.metadata
import platform
import sys

import torch
from transformers import AutoModelForCausalLM

from common import MAX_ROWS, MODEL, utc

KINDS = ("resid", "attn", "mlp", "head")


def _tensor(out):
    return out[0] if isinstance(out, tuple) else out


class Qwen3Engine:
    def __init__(self, path, device="cuda", batch=128, seed=261017, max_rows=MAX_ROWS, strict=True):
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device, self.batch, self.max_rows = device, batch, max_rows
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32,
                                                          attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        dims = (cfg.model_type, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size)
        if strict:                                      # strict=False only for the CPU code test on a tiny random model
            assert dims == ("qwen3", 28, 16, 128, 2048), f"unexpected architecture {dims}"
        self.blocks = self.model.model.layers
        self.n_layers, self.n_heads, self.head_dim, self.width = dims[1:]
        # o_proj is a Linear with weight [out=2048, in=2048]; input columns h*128:(h+1)*128 belong to head h.
        self.w_o = torch.stack([b.self_attn.o_proj.weight.detach().T.reshape(self.n_heads, self.head_dim, -1)
                                for b in self.blocks])                                   # [L, H, 128, 2048]

    def module(self, kind, layer):
        b = self.blocks[layer]
        return {"resid": b, "attn": b.self_attn, "mlp": b.mlp, "head": b.self_attn.o_proj}[kind]

    # ------------------------------------------------------------------ capture
    @torch.inference_mode()
    def capture(self, ids, positions, kinds=KINDS):
        """Last-position logits and components at `positions`: resid/attn/mlp [L, n, P, 2048], head [L, n, P, H, 128]."""
        if len({len(x) for x in ids}) != 1:
            raise ValueError("equal-length prompts required")
        positions = list(positions)
        logits, store = [], {k: [] for k in kinds}
        for start in range(0, len(ids), self.batch):
            batch = torch.tensor(ids[start:start + self.batch], device=self.device)
            cache, handles = {k: {} for k in kinds}, []
            for layer, b in enumerate(self.blocks):
                if "resid" in kinds:
                    handles.append(b.register_forward_hook(
                        lambda m, a, o, layer=layer: cache["resid"].__setitem__(layer, _tensor(o)[:, positions].clone())))
                if "attn" in kinds:
                    handles.append(b.self_attn.register_forward_hook(
                        lambda m, a, o, layer=layer: cache["attn"].__setitem__(layer, o[0][:, positions].clone())))
                if "mlp" in kinds:
                    handles.append(b.mlp.register_forward_hook(
                        lambda m, a, o, layer=layer: cache["mlp"].__setitem__(layer, o[:, positions].clone())))
                if "head" in kinds:
                    handles.append(b.self_attn.o_proj.register_forward_pre_hook(
                        lambda m, a, layer=layer: cache["head"].__setitem__(
                            layer, a[0][:, positions].reshape(a[0].shape[0], len(positions), self.n_heads,
                                                              self.head_dim).clone())))
            try:
                logits.append(self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1])
            finally:
                for h in handles:
                    h.remove()
            for k in kinds:
                store[k].append(torch.stack([cache[k][l] for l in range(self.n_layers)]))
        return torch.cat(logits), {k: torch.cat(v, 1) for k, v in store.items()}

    @torch.inference_mode()
    def native(self, ids):
        """Last-position logits of one prompt run alone."""
        return self.model(torch.tensor([ids], device=self.device), use_cache=False).logits[0, -1]

    # ------------------------------------------------------------------ reference engine (frozen design)
    @torch.inference_mode()
    def reference_patch(self, ids, kind, layer, position, vectors, head=None):
        """Run one prompt once per vector, replacing the component at (layer, position) with that vector."""
        audit = {"outside_max": 0.0, "target_error": 0.0}
        sl = slice(head * self.head_dim, (head + 1) * self.head_dim) if kind == "head" else slice(None)
        logits = []
        for start in range(0, len(vectors), self.batch):
            vv = vectors[start:start + self.batch].to(self.device, torch.float32)
            batch = torch.tensor([ids] * len(vv), device=self.device)

            def write(x):
                w = x.clone()
                if w[:, position, sl].shape != vv.shape:
                    raise RuntimeError("replacement shape does not match the patched slice")
                w[:, position, sl] = vv
                diff = w - x
                diff[:, position, sl] = 0
                audit["outside_max"] = max(audit["outside_max"], float(diff.abs().max()))
                audit["target_error"] = max(audit["target_error"], float((w[:, position, sl] - vv).abs().max()))
                return w

            module = self.module(kind, layer)
            if kind == "head":
                handle = module.register_forward_pre_hook(lambda m, a: (write(a[0]), *a[1:]))
            elif kind == "attn":
                handle = module.register_forward_hook(lambda m, a, o: (write(o[0]), *o[1:]))
            else:
                handle = module.register_forward_hook(lambda m, a, o: (write(o[0]), *o[1:]) if isinstance(o, tuple) else write(o))
            try:
                logits.append(self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1])
            finally:
                handle.remove()
        return torch.cat(logits), audit

    # ------------------------------------------------------------------ study engine
    @torch.inference_mode()
    def _prefix(self, ids, position):
        if position == 0:
            return None
        return self.model(torch.tensor([ids[:position]], device=self.device), use_cache=True).past_key_values

    @torch.inference_mode()
    def _suffix(self, ids, position, cache, n_rows, hooks=()):
        handles = [self.model.lm_head.register_forward_pre_hook(lambda m, a: (a[0].contiguous(), *a[1:]))]
        for module, kind, fn in hooks:
            if kind == "head":
                handles.append(module.register_forward_pre_hook(lambda m, a, fn=fn: (fn(a[0]), *a[1:])))
            elif kind == "attn":
                handles.append(module.register_forward_hook(lambda m, a, o, fn=fn: (fn(o[0]), *o[1:])))
            else:
                handles.append(module.register_forward_hook(
                    lambda m, a, o, fn=fn: (fn(o[0]), *o[1:]) if isinstance(o, tuple) else fn(o)))
        try:
            batch = torch.tensor([ids[position:]] * n_rows, device=self.device)
            if cache is None:
                return self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1]
            c = copy.deepcopy(cache)
            c.batch_repeat_interleave(n_rows)
            return self.model(batch, past_key_values=c, use_cache=True, logits_to_keep=1).logits[:, -1]
        finally:
            for h in handles:
                h.remove()

    @torch.inference_mode()
    def cached_logits(self, ids, position):
        """The unpatched prompt's last-position logits through the cached path (the passive check of the study engine)."""
        return self._suffix(ids, position, self._prefix(ids, position), 1)[0]

    @torch.inference_mode()
    def patch_sites(self, ids, position, sites, reduce=None):
        """sites: [(kind, layer, head, vectors [n, d])], all at `position`. Returns, per site, its logits (or
        reduce(logits), e.g. the margins, so that a batch's full logits are freed at once) and the audit.

        Sites are packed in order into batches of at most max_rows rows; a site is never split across batches.
        """
        audit = {"outside_max": 0.0, "target_error": 0.0}
        cache = self._prefix(ids, position)
        out = [None] * len(sites)
        order, batches, cur, rows = list(range(len(sites))), [], [], 0
        for i in order:
            n = len(sites[i][3])
            if cur and rows + n > self.max_rows:
                batches.append(cur)
                cur, rows = [], 0
            cur.append(i)
            rows += n
        if cur:
            batches.append(cur)
        for members in batches:
            spans, start = {}, 0
            for i in members:
                spans[i] = (start, start + len(sites[i][3]))
                start += len(sites[i][3])
            n_rows = start
            by_module = {}
            for i in members:
                kind, layer, head, vectors = sites[i]
                sl = slice(head * self.head_dim, (head + 1) * self.head_dim) if kind == "head" else slice(None)
                by_module.setdefault((kind, layer), []).append((spans[i], sl, vectors.to(self.device, torch.float32)))
            hooks = []
            for (kind, layer), writes in by_module.items():
                def fn(x, writes=writes):
                    w = x.clone()
                    mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
                    for (a, b), sl, vv in writes:
                        if w[a:b, 0, sl].shape != vv.shape:
                            raise RuntimeError("replacement shape does not match the patched slice")
                        w[a:b, 0, sl] = vv
                        mask[a:b, 0, sl] = True
                    diff = (w - x).masked_fill(mask, 0)
                    audit["outside_max"] = max(audit["outside_max"], float(diff.abs().max()))
                    for (a, b), sl, vv in writes:
                        audit["target_error"] = max(audit["target_error"], float((w[a:b, 0, sl] - vv).abs().max()))
                    return w
                hooks.append((self.module(kind, layer), kind, fn))
            logits = self._suffix(ids, position, cache, n_rows, hooks)
            for i in members:
                a, b = spans[i]
                out[i] = logits[a:b] if reduce is None else reduce(logits[a:b])
            del logits
        return out, audit

    def head_in_residual(self, z, layer, head):
        """A head's output written into the residual stream: z [n, 128] @ W_O^h [128, 2048]."""
        return z.to(self.w_o.device) @ self.w_o[layer, head]

    def environment(self):
        return {"created_utc": utc(), "model": MODEL["name"], "revision": MODEL["revision"], "dtype": "float32",
                "batch": self.batch, "max_rows": self.max_rows, "device": self.device, "layers": self.n_layers,
                "heads": self.n_heads, "head_dim": self.head_dim, "python": sys.version, "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None, "cuda": torch.version.cuda,
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "scipy"]}}

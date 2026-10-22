"""Component-level patching in GPT-2 Small at one position.

Four kinds of site, each replaced at one layer and one token position while everything downstream
is recomputed:
  resid  the residual stream after the block (output of transformer.h[l])
  attn   the attention sublayer's output, including the output-projection bias (transformer.h[l].attn)
  mlp    the MLP sublayer's output (transformer.h[l].mlp)
  head   one head's slice of the input to the attention output projection (transformer.h[l].attn.c_proj)
Float32, eager attention, deterministic kernels, no KV cache, no TF32. Every write is audited.
"""
from __future__ import annotations

import importlib.metadata
import platform
import sys

import torch
from transformers import AutoModelForCausalLM

from common import MODEL_PATHS, utc

KINDS = ("resid", "attn", "mlp", "head")


class ComponentEngine:
    def __init__(self, device="cuda", batch=128, seed=261007):
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device, self.batch = device, batch
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["gpt2-small"], local_files_only=True, dtype=torch.float32,
            attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        assert (cfg.n_layer, cfg.n_head, cfg.n_embd) == (12, 12, 768)
        self.blocks = self.model.transformer.h
        self.n_layers, self.n_heads, self.head_dim = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
        # W_O per head: c_proj is a Conv1D with weight [in=768, out=768]; rows h*64:(h+1)*64 belong to head h.
        self.w_o = torch.stack([b.attn.c_proj.weight.detach().reshape(self.n_heads, self.head_dim, -1)
                                for b in self.blocks])                       # [L, H, 64, 768]

    def _module(self, kind, layer):
        b = self.blocks[layer]
        return {"resid": b, "attn": b.attn, "mlp": b.mlp, "head": b.attn.c_proj}[kind]

    @torch.inference_mode()
    def capture(self, ids, position):
        """Last-position logits and every component at `position`: resid/attn/mlp [L, n, 768], head [L, n, H, 64]."""
        if len({len(x) for x in ids}) != 1:
            raise ValueError("equal-length prompts required")
        logits, store = [], {k: [] for k in KINDS}
        for start in range(0, len(ids), self.batch):
            batch = torch.tensor(ids[start:start + self.batch], device=self.device)
            cache, handles = {k: {} for k in KINDS}, []
            for layer, b in enumerate(self.blocks):
                handles.append(b.register_forward_hook(
                    lambda m, a, o, layer=layer: cache["resid"].__setitem__(layer, o[:, position].clone())))
                handles.append(b.attn.register_forward_hook(
                    lambda m, a, o, layer=layer: cache["attn"].__setitem__(layer, o[0][:, position].clone())))
                handles.append(b.mlp.register_forward_hook(
                    lambda m, a, o, layer=layer: cache["mlp"].__setitem__(layer, o[:, position].clone())))
                handles.append(b.attn.c_proj.register_forward_pre_hook(
                    lambda m, a, layer=layer: cache["head"].__setitem__(
                        layer, a[0][:, position].reshape(-1, self.n_heads, self.head_dim).clone())))
            try:
                logits.append(self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1])
            finally:
                for h in handles:
                    h.remove()
            for k in KINDS:
                store[k].append(torch.stack([cache[k][l] for l in range(self.n_layers)]))
        return torch.cat(logits), {k: torch.cat(v, 1) for k, v in store.items()}

    @torch.inference_mode()
    def patch(self, ids, kind, layer, position, vectors, head=None):
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

            module = self._module(kind, layer)
            if kind == "head":
                handle = module.register_forward_pre_hook(lambda m, a: (write(a[0]), *a[1:]))
            elif kind == "attn":
                handle = module.register_forward_hook(lambda m, a, o: (write(o[0]), *o[1:]))
            else:
                handle = module.register_forward_hook(lambda m, a, o: write(o))
            try:
                logits.append(self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1])
            finally:
                handle.remove()
        return torch.cat(logits), audit

    def head_in_residual(self, z, layer, head):
        """A head's output written into the residual stream: z [n, 64] @ W_O^h [64, 768] (projection bias excluded)."""
        return z.to(self.w_o.device) @ self.w_o[layer, head]

    def environment(self):
        return {"created_utc": utc(), "model": "gpt2-small", "dtype": "float32", "batch": self.batch,
                "device": self.device, "layers": self.n_layers, "heads": self.n_heads, "head_dim": self.head_dim,
                "python": sys.version, "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None, "cuda": torch.version.cuda,
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "scipy"]}}

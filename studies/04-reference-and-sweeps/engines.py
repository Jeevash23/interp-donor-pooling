"""Native Hugging Face engines for residual patching and attention-head ablation.

Both engines follow the October implementations (studies/03-grouping-and-circuit):
float32 weights, eager attention, evaluation mode, no KV cache, no TF32 and
deterministic kernels. Patches are written with forward hooks and audited for
exact write locality.
"""
from __future__ import annotations

import importlib.metadata
import platform
import sys

import torch
from transformers import AutoModelForCausalLM

from common import MODEL_PATHS, utc


def _configure(seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _tensor(out):
    return out[0] if isinstance(out, tuple) else out


class ResidualEngine:
    """Replace the residual stream after one block at one position."""

    def __init__(self, name, device="cuda", batch=96, seed=261005):
        _configure(seed)
        self.name, self.device, self.batch = name, device, batch
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS[name], local_files_only=True, dtype=torch.float32,
            attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        kind = self.model.config.model_type
        if kind == "gpt2":
            self.blocks = self.model.transformer.h
        elif kind == "qwen3":
            self.blocks = self.model.model.layers
        else:
            raise ValueError(kind)
        self.n_layers = len(self.blocks)

    @torch.inference_mode()
    def capture(self, ids):
        """Final-position logits and the residual after every block at every position."""
        if len({len(x) for x in ids}) != 1:
            raise ValueError("equal-length prompts required")
        logits, states = [], []
        for start in range(0, len(ids), self.batch):
            batch = torch.tensor(ids[start:start + self.batch], device=self.device)
            cache = {}
            handles = []
            for layer, block in enumerate(self.blocks):
                def save(module, args, out, layer=layer):
                    cache[layer] = _tensor(out).clone()
                handles.append(block.register_forward_hook(save))
            try:
                out = self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1]
            finally:
                for h in handles:
                    h.remove()
            logits.append(out)
            states.append(torch.stack([cache[l] for l in range(self.n_layers)], 1))
        return torch.cat(logits), torch.cat(states)

    @torch.inference_mode()
    def patch(self, ids, layer, position, vectors):
        """Run one prompt once per vector, replacing (layer, position) with that vector."""
        audit = {"outside_max": 0.0, "target_error": 0.0}
        logits = []
        for start in range(0, len(vectors), self.batch):
            vv = vectors[start:start + self.batch].to(self.device, torch.float32)
            batch = torch.tensor([ids] * len(vv), device=self.device)

            def write(module, args, out):
                v = _tensor(out)
                w = v.clone()
                w[:, position] = vv
                diff = (w - v)
                diff[:, position] = 0
                audit["outside_max"] = max(audit["outside_max"], float(diff.abs().max()))
                audit["target_error"] = max(audit["target_error"], float((w[:, position] - vv).abs().max()))
                return (w, *out[1:]) if isinstance(out, tuple) else w

            handle = self.blocks[layer].register_forward_hook(write)
            try:
                logits.append(self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1])
            finally:
                handle.remove()
        return torch.cat(logits), audit

    def environment(self):
        return {"created_utc": utc(), "model": self.name, "dtype": "float32", "batch": self.batch,
                "device": self.device, "layers": self.n_layers, "python": sys.version,
                "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
                "cuda": torch.version.cuda,
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "scipy"]}}


class HeadEngine:
    """GPT-2 Small attention-head ablation at the input of each attention output projection.

    Matches the October circuit engine: the concatenated head results are viewed as
    [batch, position, head, 64]; excluded (layer, position, head) slices are replaced,
    retained slices are recomputed through the intervened network, and MLPs,
    embeddings, normalizations and projection biases stay active.
    """

    def __init__(self, device="cuda", batch=64, seed=261005):
        _configure(seed)
        self.device, self.batch = device, batch
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["gpt2-small"], local_files_only=True, dtype=torch.float32,
            attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        assert (self.model.config.n_layer, self.model.config.n_head, self.model.config.n_embd) == (12, 12, 768)

    @torch.inference_mode()
    def run(self, ids, capture=False, replacement=None, keep=None):
        """ids: equal-length prompts. replacement: [12, len(ids), T, 12, 64]; keep: [12, T, 12, 1] bool."""
        if not ids or len({len(x) for x in ids}) != 1:
            raise ValueError("equal-length prompts required")
        audit = {"outside_changed_max": 0.0, "replacement_error_max": 0.0}
        logits, caches = [], []
        for start in range(0, len(ids), self.batch):
            stop = min(start + self.batch, len(ids))
            batch = torch.tensor(ids[start:stop], device=self.device)
            cache, handles = {}, []
            for layer, block in enumerate(self.model.transformer.h):
                def hook(module, inputs, layer=layer, start=start, stop=stop):
                    original = inputs[0]
                    z = original.view(*original.shape[:2], 12, 64)
                    if capture:
                        cache[layer] = z.clone()
                    if replacement is None:
                        return None
                    rep = replacement[layer, start:stop].to(self.device, z.dtype)
                    retained = keep[layer].to(self.device)[None].expand_as(z)
                    w = torch.where(retained, z, rep)
                    outside = (w - z).masked_select(retained)
                    target = (w - rep).masked_select(~retained)
                    if outside.numel():
                        audit["outside_changed_max"] = max(audit["outside_changed_max"], float(outside.abs().max()))
                    if target.numel():
                        audit["replacement_error_max"] = max(audit["replacement_error_max"], float(target.abs().max()))
                    return (w.reshape_as(original), *inputs[1:])
                handles.append(block.attn.c_proj.register_forward_pre_hook(hook))
            try:
                logits.append(self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1])
            finally:
                for h in handles:
                    h.remove()
            if capture:
                caches.append(torch.stack([cache[l] for l in range(12)]))
        return torch.cat(logits), (torch.cat(caches, 1) if capture else None), audit

    def environment(self):
        return {"created_utc": utc(), "model": "gpt2-small", "dtype": "float32", "batch": self.batch,
                "device": self.device, "python": sys.version, "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
                "cuda": torch.version.cuda,
                "hook": "attention c_proj input viewed as [batch, position, head, 64]",
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "scipy"]}}

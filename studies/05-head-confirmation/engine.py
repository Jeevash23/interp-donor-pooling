"""Head-level patching: replace one attention head's output at one position.

The head's slice of the input to the attention output projection (GPT-2: attn.c_proj;
Qwen3: self_attn.o_proj) is replaced; every other head and position is recomputed
normally. Float32, eager attention, deterministic kernels, no KV cache, no TF32.
"""
from __future__ import annotations

import importlib.metadata
import platform
import sys

import torch
from transformers import AutoModelForCausalLM

from common import MODEL_PATHS, utc


class HeadPatchEngine:
    def __init__(self, name, device="cuda", batch=128, seed=261006):
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.name, self.device, self.batch = name, device, batch
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS[name], local_files_only=True, dtype=torch.float32,
            attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        if cfg.model_type == "gpt2":
            self.proj = [b.attn.c_proj for b in self.model.transformer.h]
        elif cfg.model_type == "qwen3":
            self.proj = [b.self_attn.o_proj for b in self.model.model.layers]
        else:
            raise ValueError(cfg.model_type)
        self.n_layers = len(self.proj)
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    @torch.inference_mode()
    def capture(self, ids, positions):
        """Logits at the last position and head outputs z[layer, prompt, position, head, dim] at the given positions."""
        if len({len(x) for x in ids}) != 1:
            raise ValueError("equal-length prompts required")
        logits, zs = [], []
        for start in range(0, len(ids), self.batch):
            batch = torch.tensor(ids[start:start + self.batch], device=self.device)
            cache, handles = {}, []
            for layer, proj in enumerate(self.proj):
                def save(module, args, layer=layer):
                    x = args[0][:, positions]
                    cache[layer] = x.reshape(*x.shape[:2], self.n_heads, self.head_dim).clone()
                handles.append(proj.register_forward_pre_hook(save))
            try:
                logits.append(self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1])
            finally:
                for h in handles:
                    h.remove()
            zs.append(torch.stack([cache[l] for l in range(self.n_layers)]))
        return torch.cat(logits), torch.cat(zs, 1)

    @torch.inference_mode()
    def patch(self, ids, layer, position, head, vectors):
        """Run one prompt once per vector, replacing head `head` at (layer, position) with that vector."""
        audit = {"outside_max": 0.0, "target_error": 0.0}
        sl = slice(head * self.head_dim, (head + 1) * self.head_dim)
        logits = []
        for start in range(0, len(vectors), self.batch):
            vv = vectors[start:start + self.batch].to(self.device, torch.float32)
            batch = torch.tensor([ids] * len(vv), device=self.device)

            def hook(module, args):
                x = args[0]
                w = x.clone()
                if w[:, position, sl].shape != vv.shape:
                    raise RuntimeError("replacement shape does not match the patched slice")
                w[:, position, sl] = vv
                diff = w - x
                diff[:, position, sl] = 0
                audit["outside_max"] = max(audit["outside_max"], float(diff.abs().max()))
                audit["target_error"] = max(audit["target_error"], float((w[:, position, sl] - vv).abs().max()))
                return (w, *args[1:])

            handle = self.proj[layer].register_forward_pre_hook(hook)
            try:
                logits.append(self.model(batch, use_cache=False, logits_to_keep=1).logits[:, -1])
            finally:
                handle.remove()
        return torch.cat(logits), audit

    def environment(self):
        return {"created_utc": utc(), "model": self.name, "dtype": "float32", "batch": self.batch,
                "device": self.device, "layers": self.n_layers, "heads": self.n_heads, "head_dim": self.head_dim,
                "python": sys.version, "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None, "cuda": torch.version.cuda,
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "scipy"]}}

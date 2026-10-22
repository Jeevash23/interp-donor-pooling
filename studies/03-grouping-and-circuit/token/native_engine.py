"""Copied Engine class from submission-study/scripts/engine.py; only its imported common definitions are local. Scientific implementation unchanged."""
from __future__ import annotations
import math, importlib.metadata, platform, sys
from common import *
import numpy as np
import torch
from transformers import AutoModelForCausalLM

class Engine:
    def __init__(self,name,dtype="float32",device="cuda",batch=16):
        torch.set_num_threads(4);torch.manual_seed(260912)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.name=name;self.dtype=dtype;self.device=device;self.batch=1 if name=='pythia-410m' else batch
        self.model=AutoModelForCausalLM.from_pretrained(MODEL_PATHS[name],local_files_only=True,dtype=getattr(torch,dtype),attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        typ=self.model.config.model_type
        if typ=="gpt2":
            self.blocks=self.model.transformer.h;self.norm=self.model.transformer.ln_f;self.head=self.model.lm_head
            self.projections=[b.attn.c_proj for b in self.blocks]
        elif typ=="gpt_neox":
            self.blocks=self.model.gpt_neox.layers;self.norm=self.model.gpt_neox.final_layer_norm;self.head=self.model.get_output_embeddings()
            self.projections=[b.attention.dense for b in self.blocks]
        elif typ in ["llama","qwen3"]:
            self.blocks=self.model.model.layers;self.norm=self.model.model.norm;self.head=self.model.lm_head
            self.projections=[b.self_attn.o_proj for b in self.blocks]
        else:raise ValueError(typ)
        self.layers=[math.ceil(len(self.blocks)*f)-1 for f in [.25,.5,.75,1]]
        self.nheads=self.model.config.num_attention_heads
        self.head_dim=getattr(self.model.config,"head_dim",None) or self.model.config.hidden_size//self.nheads
        self.hidden=self.model.config.hidden_size

    @torch.inference_mode()
    def readout(self,z):
        # Match a native one-position sequence; no model weights or norm parameters are folded.
        return self.head(self.norm(z[:,None,:]))[:,0,:]

    @torch.inference_mode()
    def run(self,prompts,patch=None,capture=False,head_capture=False):
        n=len(prompts); result=[None]*n;final=[None]*n
        sources={l:[None]*n for l in self.layers} if capture else {}
        hsources={l:[None]*n for l in self.layers} if head_capture else {}
        audit={"outside_max":0.,"target_error":0.,"write_norm":0.}
        lengths={}
        for i,p in enumerate(prompts):lengths.setdefault(len(p["ids"]),[]).append(i)
        for length,ixall in sorted(lengths.items()):
            for start in range(0,len(ixall),self.batch):
                ix=ixall[start:start+self.batch];ps=[prompts[i] for i in ix]
                ids=torch.tensor([p["ids"] for p in ps],device=self.device)
                handles=[];st={};ar=torch.arange(len(ix),device=self.device)
                def tensor(out):return out[0] if isinstance(out,tuple) else out
                if patch:
                    layer,pos,vectors,head_index=patch
                    pp=torch.tensor([p[pos] for p in ps],device=self.device)
                    vv=vectors[ix].to(device=self.device,dtype=self.model.dtype)
                    def replace(v):
                        w=v.clone()
                        sl=slice(None) if head_index is None else slice(head_index*self.head_dim,(head_index+1)*self.head_dim)
                        old=w[ar,pp,sl].clone();w[ar,pp,sl]=vv
                        diff=w-v;diff[ar,pp,sl]=0
                        audit["outside_max"]=max(audit["outside_max"],float(diff.abs().max()))
                        audit["target_error"]=max(audit["target_error"],float((w[ar,pp,sl]-vv).abs().max()))
                        audit["write_norm"]=max(audit["write_norm"],float((w[ar,pp,sl]-old).float().norm(dim=-1).max()))
                        return w
                    if head_index is None:
                        def write(module,args,out):
                            w=replace(tensor(out));return (w,*out[1:]) if isinstance(out,tuple) else w
                        handles.append(self.blocks[layer].register_forward_hook(write))
                    else:
                        def writehead(module,args):return (replace(args[0]),*args[1:])
                        handles.append(self.projections[layer].register_forward_pre_hook(writehead))
                if capture:
                    for l in self.layers:
                        def save(module,args,out,l=l):
                            v=tensor(out)
                            st[l]=torch.stack([v[ar,torch.tensor([p[k] for p in ps],device=self.device)] for k in ["anchor","end"]],1).clone()
                        handles.append(self.blocks[l].register_forward_hook(save))
                if head_capture:
                    for l in self.layers:
                        def savehead(module,args,l=l):st[f"head-{l}"]=args[0][:,-1].clone()
                        handles.append(self.projections[l].register_forward_pre_hook(savehead))
                def savefinal(module,args):st["final"]=args[0][:,-1].clone()
                handles.append(self.norm.register_forward_pre_hook(savefinal))
                try:log=self.model(ids,use_cache=False,logits_to_keep=1).logits[:,0]
                finally:
                    for h in handles:h.remove()
                for j,i in enumerate(ix):
                    result[i]=log[j];final[i]=st["final"][j]
                    for l in sources:sources[l][i]=st[l][j]
                    for l in hsources:hsources[l][i]=st[f"head-{l}"][j]
        return torch.stack(result),{"final":torch.stack(final),"source":{l:torch.stack(x) for l,x in sources.items()},"head_source":{l:torch.stack(x) for l,x in hsources.items()}},audit

    def metrics(self,log,row):
        p=row["encoded"][self.name][0];good,bad=p["good_id"],p["bad_id"]
        logp=log.double().log_softmax(-1)[:,good]
        margin=log[:,good].double()-log[:,bad].double()
        return torch.stack([margin,logp.exp(),(log.argmax(-1)==good).double(),(margin>0).double(),logp],-1).cpu().numpy()

    def environment(self):
        return {"model":self.name,"model_type":self.model.config.model_type,"parameters":sum(p.numel() for p in self.model.parameters()),"layers":self.layers,"n_heads":self.nheads,"head_dim":self.head_dim,"dtype":self.dtype,"batch":self.batch,"python":sys.version,"platform":platform.platform(),"packages":{p:importlib.metadata.version(p) for p in ["torch","transformers","numpy","scipy","matplotlib","safetensors"]},"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if self.device=="cuda" else None,"created_utc":utc()}

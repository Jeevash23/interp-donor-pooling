"""Native unpadded model execution and auditable vector/single-head writes."""
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

def conditions(src,row,layer,pi):
    """Source order: recipient clean/corrupt, 8 narrow pairs, 8 broad pairs."""
    names=[];vecs=[];recipient=[];banks=[]
    def add(name,v,r):names.append(name);vecs.append(v);recipient.append(r)
    for direction,target,donor_offset in [("restore",1,0),("destroy",0,1)]:
        r=target
        for bank,offset in [("narrow",2),("broad",18)]:
            x=src[offset+donor_offset:offset+16:2];mean=x.mean(0)
            indices=[]
            for j in range(8):indices.append(len(vecs));add(f"{direction}/{bank}/donor{j}",x[j],r)
            mi=len(vecs);add(f"{direction}/{bank}/mean8",mean,r)
            banks.append({"name":f"{direction}/{bank}/full","indices":indices,"mean":mi})
            if direction=="restore" and bank=="narrow":
                for radius in [.25,.5,.75]:
                    indices=[]
                    for j in range(8):indices.append(len(vecs));add(f"restore/narrow/r{radius}/donor{j}",mean+radius*(x[j]-mean),r)
                    banks.append({"name":f"restore/narrow/r{radius}","indices":indices,"mean":mi})
                for order_seed in range(3):
                    rng=np.random.default_rng(260920+row["group"]*31+order_seed)
                    perm=rng.permutation(8)
                    for k in [2,4]:add(f"restore/narrow/mean{k}/order{order_seed}",x[perm[:k]].mean(0),r)
                for j in range(2):add(f"restore/narrow/antithetic{j}",mean-.5*(x[j]-mean),r)
        add(f"{direction}/paired",src[1-target],r)
        add(f"{direction}/zero",torch.zeros_like(src[0]),r)
        add(f"{direction}/identity",src[target],r)
    return names,torch.stack(vecs),recipient,banks

def validate(engine,rows):
    errors={};positive=0.
    selected=[next(r for r in rows if r["task"]==t) for t in TASKS]
    for ri,row in enumerate(selected):
        ps=row["encoded"][engine.name];log,st,_=engine.run(ps,capture=True,head_capture=True)
        p=ps[1]
        with torch.inference_mode():native=engine.model(torch.tensor([p["ids"]],device=engine.device),use_cache=False,logits_to_keep=1).logits[:,0]
        errors[f"native_{ri}"]=float((native[0]-log[1]).abs().max())
        errors[f"readout_{ri}"]=float((log-engine.readout(st["final"])).abs().max())
        errors[f"prob_native_{ri}"]=float(abs(engine.metrics(native,row)[0,1]-engine.metrics(log[1:2],row)[0,1]))
        errors[f"prob_readout_{ri}"]=float(abs(engine.metrics(log,row)[:,1]-engine.metrics(engine.readout(st["final"]),row)[:,1]).max())
        for l in engine.layers:
            for pi,pos in enumerate(["anchor","end"]):
                x=st["source"][l][:,pi]
                vectors=torch.stack([x[1],x[2],x[2].repeat(8,1).mean(0)])
                patched,_,a=engine.run([p]*3,(l,pos,vectors,None))
                errors[f"identity_{ri}_{l}_{pos}"]=float((patched[0]-log[1]).abs().max())
                errors[f"prob_identity_{ri}_{l}_{pos}"]=float(abs(engine.metrics(patched[:1],row)[0,1]-engine.metrics(log[1:2],row)[0,1]))
                errors[f"duplicate_{ri}_{l}_{pos}"]=float((patched[1]-patched[2]).abs().max())
                errors[f"outside_{ri}_{l}_{pos}"]=a["outside_max"]
                errors[f"target_{ri}_{l}_{pos}"]=a["target_error"]
                positive=max(positive,float((patched[1]-log[1]).abs().max()))
                if l==engine.layers[-1]:errors[f"terminal_{ri}_{pos}"]=float((patched[1]-(log[1] if pos=="anchor" else log[2])).abs().max())
            # Single head is explicitly checked independently of full-residual writes.
            h=engine.nheads//2;v=st["head_source"][l][:,h*engine.head_dim:(h+1)*engine.head_dim]
            hlog,_,a=engine.run([p]*3,(l,"end",torch.stack([v[1],v[2],v[2]]),h))
            errors[f"head_identity_{ri}_{l}"]=float((hlog[0]-log[1]).abs().max())
            errors[f"head_duplicate_{ri}_{l}"]=float((hlog[1]-hlog[2]).abs().max())
            errors[f"head_outside_{ri}_{l}"]=a["outside_max"]
            errors[f"head_target_{ri}_{l}"]=a["target_error"]
        # Endpoint reconstruction and finite-bank decomposition are also checked before test.
        l=engine.layers[1];x=st["source"][l][2:18:2,1];mean=x.mean(0)
        y,s,_=engine.run([p]*9,(l,"end",torch.cat([x,mean[None]]),None))
        a=engine.metrics(y[-1:],row)[0];avg=engine.metrics(y[:8],row).mean(0)
        b=engine.metrics(engine.readout(s["final"][:8].mean(0,keepdim=True)),row)[0]
        c=engine.metrics(y[:8].mean(0,keepdim=True),row)[0]
        errors[f"closure_margin_{ri}"]=float(abs((a[0]-avg[0])-(a[0]-b[0])-(b[0]-c[0])))
        errors[f"closure_prob_{ri}"]=float(abs((a[1]-avg[1])-(a[1]-b[1])-(b[1]-c[1])-(c[1]-avg[1])))
    tol=.001 if engine.dtype=="float32" else .5
    ptol=.00002 if engine.dtype=="float32" else .05
    pmax=max(v for k,v in errors.items() if k.startswith('prob_'))
    result={"validation_version":2,"errors":errors,"maximum_error":max(errors.values()),"probability_max_error":pmax,"probability_tolerance":ptol,"positive_logit_change":positive,"logit_tolerance":tol,"passed":max(errors.values())<=tol and pmax<=ptol and positive>.001}
    return result

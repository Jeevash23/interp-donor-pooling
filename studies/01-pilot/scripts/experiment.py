"""Finite-bank activation-pooling experiment using native GPT-2 forward hooks."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {"gpt2-small": [2,5,8,11], "gpt2-medium": [5,11,17,23]}
CONDITIONS = [f"donor{j}" for j in range(8)] + ["mean2","mean4","mean8"] + [f"half{j}" for j in range(8)] + ["identity"]

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""): h.update(b)
    return h.hexdigest()

MODELS = Path(os.environ.get("MODELS_DIR", ROOT.parents[1] / "models"))   # release: see models/README.md
def model_path(name): return MODELS / name

def prepare():
    tok = AutoTokenizer.from_pretrained(model_path("gpt2-small"), local_files_only=True)
    names = "Alice Bob Carol David Emma Frank Grace Henry Jack James Jane John Julia Kate Laura Lucy Luke Mark Mary Michael Mike Nick Oliver Paul Peter Rachel Richard Robert Rose Sam Sarah Simon Sophie Steve Susan Thomas Tom Victor William Anna Ben Daniel Edward George Helen Joe Kevin Linda Lisa Maria Martin Nancy Patrick Rebecca Roger Steven Tony Walter Wendy".split()
    places = "store park garden school office station market hotel museum library beach church hospital theater restaurant".split()
    objects = "book ring ball cup bag box hat map toy key letter flower ticket bottle apple".split()
    pools = {k:[s for s in v if len(tok.encode(" "+s)) == 1] for k,v in {"names":names,"places":places,"objects":objects}.items()}
    tok2=AutoTokenizer.from_pretrained(model_path("gpt2-medium"),local_files_only=True)
    assert all(tok.encode(" "+s)==tok2.encode(" "+s) for v in pools.values() for s in v)
    dump(ROOT/"data/pools.json",pools)
    all_rows=[]
    for split,seed,n in [("calibration",120926,4),("evaluation",120927,64)]:
        rng=random.Random(seed); rows=[]
        for g in range(n):
            io,subject,alt=rng.sample(pools["names"],3)
            contexts=rng.sample([(p,o) for p in pools["places"] for o in pools["objects"]],9)
            for family in range(2):
                for order in range(2):
                    first,second=(io,subject) if order==0 else (subject,io)
                    def make(context, sub):
                        place,obj=context
                        prefix=(f"When {first} and {second} went to the {place}," if family==0 else f"After {first} and {second} arrived at the {place},")
                        text=prefix+f" {sub} "+("gave" if family==0 else "handed")+f" the {obj} to"
                        ids=[tok.eos_token_id]+tok.encode(text)
                        s2=1+len(tok.encode(prefix))
                        assert ids[s2]==tok.encode(" "+sub)[0]
                        return {"text":text,"ids":ids,"s2":s2,"end":len(ids)-1}
                    clean=make(contexts[0],subject); corrupt=make(contexts[0],alt)
                    donors=[make(c,subject) for c in contexts[1:]]
                    assert len({len(x["ids"]) for x in [clean,corrupt,*donors]})==1
                    assert sum(a!=b for a,b in zip(clean["ids"],corrupt["ids"]))==1
                    rows.append(dict(id=f"{split}-{g:03d}-f{family}-o{order}",split=split,group=g,family=family,order=order,io=io,subject=subject,alt=alt,io_id=tok.encode(" "+io)[0],subject_id=tok.encode(" "+subject)[0],clean=clean,corrupt=corrupt,donors=donors))
        dump(ROOT/f"data/{split}.json",rows); all_rows.extend(rows)
    assert len({r["clean"]["text"] for r in all_rows})==len(all_rows)
    manifest={name:{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in model_path(name).iterdir() if p.is_file()} for name in MODELS}
    dump(ROOT/"data/model-manifest.json",manifest)
    freeze={str(p.relative_to(ROOT)):sha(p) for folder in ["scripts","docs","data"] for p in sorted((ROOT/folder).rglob("*")) if p.is_file()}
    dump(ROOT/"data/freeze.json",{"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"files":freeze})
    print("Prepared 16 calibration and 256 evaluation configurations; 8 donors each.",flush=True)

class Engine:
    def __init__(self,name,device):
        torch.set_num_threads(4); torch.manual_seed(120926)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        self.name=name; self.layers=MODELS[name]; self.device=device
        self.model=AutoModelForCausalLM.from_pretrained(model_path(name),local_files_only=True,dtype=torch.float32,attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        self.d=self.model.config.n_embd
    @torch.inference_mode()
    def run(self,prompts,patch=None,capture=False):
        ids=torch.tensor([p["ids"] for p in prompts],device=self.device)
        assert all(len(p["ids"])==ids.shape[1] for p in prompts)
        handles=[]; states={}; audit={}
        if patch is not None:
            layer,position,vectors=patch
            def write(module,args,out):
                assert isinstance(out,torch.Tensor)
                v=out.clone(); idx=torch.arange(len(v),device=v.device)
                pos=torch.tensor([p[position] for p in prompts],device=v.device)
                before=v[idx,pos].clone(); v[idx,pos]=vectors.to(v)
                diff=v-out; diff[idx,pos]=0
                audit["outside_max"]=float(diff.abs().max())
                audit["target_error"]=float((v[idx,pos]-vectors.to(v)).abs().max())
                audit["write_norm"]=float((v[idx,pos]-before).norm(dim=-1).max())
                return v
            handles.append(self.model.transformer.h[layer].register_forward_hook(write))
        if capture:
            for layer in self.layers:
                def save(module,args,out,layer=layer):
                    ix=torch.arange(len(out),device=out.device)
                    states[layer]=torch.stack([out[ix,torch.tensor([p[pos] for p in prompts],device=out.device)] for pos in ["s2","end"]],1).clone()
                handles.append(self.model.transformer.h[layer].register_forward_hook(save))
        def last(module,args): states["final"]=args[0][:,-1].clone()
        handles.append(self.model.transformer.ln_f.register_forward_pre_hook(last))
        try:
            logits=self.model(ids,use_cache=False,logits_to_keep=1).logits[:,0]
        finally:
            for h in handles: h.remove()
        return logits,states,audit
    @torch.inference_mode()
    def readout(self,z): return self.model.lm_head(self.model.transformer.ln_f(z))
    def metrics(self,logits,row):
        # Compute probabilities in float64 so the statistical replay uses a stable readout.
        ld=logits[:,row["io_id"]]-logits[:,row["subject_id"]]
        prob=logits.double().softmax(-1)[:,row["io_id"]]
        top=(logits.argmax(-1)==row["io_id"]).double()
        return torch.stack([ld.double(),prob,top],1).cpu().numpy()

def validate(engine,rows):
    errors={}; maxwrite=0.
    for ri,row in enumerate([rows[i] for i in [0,5,10,15]]):
        ps=[row["corrupt"],*row["donors"]]
        log,st,_=engine.run(ps,capture=True)
        ids=torch.tensor([p["ids"] for p in ps],device=engine.device)
        with torch.inference_mode(): native=engine.model(ids,use_cache=False,logits_to_keep=1).logits[:,0]
        errors[f"native_{ri}"]=float((log-native).abs().max())
        errors[f"readout_{ri}"]=float((log-engine.readout(st["final"])).abs().max())
        serial,_,_=engine.run([ps[0]])
        errors[f"batch_{ri}"]=float((log[0]-serial[0]).abs().max())
        for layer in engine.layers:
            for pi,position in enumerate(["s2","end"]):
                own=st[layer][0,pi].repeat(2,1)
                identity,_,a=engine.run([ps[0]]*2,patch=(layer,position,own))
                errors[f"identity_{ri}_{layer}_{position}"]=float((identity-log[0]).abs().max())
                donor=st[layer][1,pi]
                dup=donor.repeat(8,1).mean(0)
                test,_,a=engine.run([ps[0]]*2,patch=(layer,position,torch.stack([donor,dup])))
                errors[f"duplicate_{ri}_{layer}_{position}"]=float((test[0]-test[1]).abs().max())
                errors[f"outside_{ri}_{layer}_{position}"]=a["outside_max"]
                errors[f"target_{ri}_{layer}_{position}"]=a["target_error"]
                maxwrite=max(maxwrite,float((test[0]-log[0]).abs().max()))
                if layer==engine.layers[-1]:
                    target=log[1] if position=="end" else log[0]
                    errors[f"terminal_{ri}_{position}"]=float((test[0]-target).abs().max())
    result={"errors":errors,"maximum_error":max(errors.values()),"positive_control_logit_change":maxwrite,"tolerance":0.0005}
    result["passed"]=result["maximum_error"]<=0.0005 and maxwrite>0.001
    return result

def freeze_check():
    f=json.loads((ROOT/"data/freeze.json").read_text())
    from relocation import frozen_matches
    changed=[p for p,h in f["files"].items() if not frozen_matches(ROOT,p,h)]
    if changed: raise RuntimeError(f"Frozen inputs changed: {changed}")

def evaluate(engine,rows,out):
    out.mkdir(parents=True,exist_ok=True)
    for g in sorted({r["group"] for r in rows}):
        path=out/f"group-{g:03d}.npz"
        if path.exists(): raise RuntimeError(f"Refusing to overwrite/resume {path}")
        selected=[r for r in rows if r["group"]==g]
        finals=[]; source=[]; baselines=[]; endpoints=[]; metrics=[]; audit=[]
        for row in selected:
            ps=[row["clean"],row["corrupt"],*row["donors"]]
            base,st,_=engine.run(ps,capture=True)
            baselines.append(engine.metrics(base,row))
            rowfinal=[]; rowsource=[]; rowmetrics=[]; rowend=[]
            for layer in engine.layers:
                for pi,position in enumerate(["s2","end"]):
                    donors=st[layer][2:,pi]; mean=donors.mean(0)
                    vectors=torch.cat([donors,torch.stack([donors[:2].mean(0),donors[:4].mean(0),mean]),mean+0.5*(donors-mean),st[layer][1:2,pi]])
                    log,patched,a=engine.run([row["corrupt"]]*len(vectors),patch=(layer,position,vectors))
                    z=patched["final"]
                    rowfinal.append(z.cpu().numpy()); rowsource.append(st[layer][:,pi].cpu().numpy())
                    rowmetrics.append(engine.metrics(log,row))
                    # B=readout(mean final state); C=mean complete logits, for all nested banks and half radius.
                    ep=[]
                    for idx in [list(range(2)),list(range(4)),list(range(8)),list(range(11,19))]:
                        b=engine.readout(z[idx].mean(0,keepdim=True))
                        c=log[idx].mean(0,keepdim=True)
                        ep.append(np.stack([engine.metrics(b,row)[0],engine.metrics(c,row)[0]]))
                    rowend.append(np.stack(ep)); audit.append(a)
                    assert a["outside_max"]==0 and a["target_error"]==0
                    assert float((log[-1]-base[1]).abs().max())<=0.0005
            finals.append(np.stack(rowfinal)); source.append(np.stack(rowsource)); metrics.append(np.stack(rowmetrics)); endpoints.append(np.stack(rowend))
        np.savez_compressed(path,final=np.stack(finals),source=np.stack(source),baseline=np.stack(baselines),metrics=np.stack(metrics),endpoints=np.stack(endpoints),ids=np.array([r["id"] for r in selected]))
        dump(out/f"group-{g:03d}-audit.json",audit)
        print(json.dumps({"model":engine.name,"group":g,"groups":len({r["group"] for r in rows}),"elapsed_seconds":round(time.time()-START,1)}),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["prepare","calibrate","evaluate","replay"]); ap.add_argument("--model",choices=list(MODELS),default="gpt2-small"); ap.add_argument("--device",default="cuda"); args=ap.parse_args()
    if args.stage=="prepare": prepare(); return
    freeze_check(); engine=Engine(args.model,args.device)
    dest=ROOT/"results"/args.model; dest.mkdir(parents=True,exist_ok=True)
    if args.stage=="calibrate":
        env={"python":sys.version,"platform":platform.platform(),"packages":{p:importlib.metadata.version(p) for p in ["torch","transformers","numpy","scipy","matplotlib","safetensors"]},"device":args.device,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"free_total_cuda_bytes":torch.cuda.mem_get_info() if torch.cuda.is_available() else None,"parameters":sum(p.numel() for p in engine.model.parameters()),"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
        dump(dest/"environment.json",env)
        rows=json.loads((ROOT/"data/calibration.json").read_text())
        result=validate(engine,rows); dump(dest/"validation.json",result); print(json.dumps(result),flush=True)
        if not result["passed"]: raise RuntimeError("Calibration gate failed")
    elif args.stage=="evaluate":
        if not json.loads((dest/"validation.json").read_text())["passed"]: raise RuntimeError("Calibration must pass first")
        rows=json.loads((ROOT/"data/evaluation.json").read_text())
        evaluate(engine,rows,dest/"raw")
        dump(dest/"completion.json",{"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"elapsed_seconds":time.time()-START,"rows":len(rows),"conditions":CONDITIONS,"layers":engine.layers,"peak_cuda_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None})
    else:
        rows=json.loads((ROOT/"data/evaluation.json").read_text())
        evaluate(engine,[r for r in rows if r["group"] in [0,31,63]],dest/"replay")
        checks={}
        for g in [0,31,63]:
            a=np.load(dest/"raw"/f"group-{g:03d}.npz"); b=np.load(dest/"replay"/f"group-{g:03d}.npz")
            for k in ["final","source","baseline","metrics","endpoints"]: checks[f"{g}_{k}"]=float(np.max(np.abs(a[k]-b[k])))
        dump(dest/"replay.json",{"max_errors":checks,"passed":max(checks.values())<=0.0005})

START=time.time()
if __name__=="__main__": main()

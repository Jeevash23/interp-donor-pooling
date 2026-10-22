from __future__ import annotations
import argparse,time
from common import *
from engine import Engine, conditions, validate
import numpy as np
import torch

def evaluate(engine,rows,out,replay=False):
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    for ri,row in enumerate(rows):
        path=out/(row["id"]+".npz")
        if path.exists():raise RuntimeError(f"Refusing to overwrite {path}")
        ps=row["encoded"][engine.name]
        base,source,_=engine.run(ps,capture=True)
        metrics=[];endpoints=[];finals=[];spatial=[];audits=[];dispersion=[]
        for layer in engine.layers:
            for pi,pos in enumerate(["anchor","end"]):
                src=source["source"][layer][:,pi]
                names,vectors,targets,banks=conditions(src,row,layer,pi)
                log,st,audit=engine.run([ps[t] for t in targets],(layer,pos,vectors,None))
                m=engine.metrics(log,row);metrics.append(m)
                ep=[]
                for bank in banks:
                    idx=bank["indices"]
                    b=engine.readout(st["final"][idx].mean(0,keepdim=True))
                    c=log[idx].mean(0,keepdim=True)
                    ep.append(np.stack([engine.metrics(b,row)[0],engine.metrics(c,row)[0]]))
                endpoints.append(np.stack(ep))
                audit["identity_margin_error"]=max(abs(m[names.index(d+"/identity"),0]-engine.metrics(base,row)[t,0]) for d,t in [("restore",1),("destroy",0)])
                assert audit["outside_max"]==0 and audit["target_error"]==0
                if engine.dtype=="float32":assert audit["identity_margin_error"]<=.001
                audits.append(audit)
                ds=[]
                for off in [2,18]:
                    for od in [0,1]:
                        x=src[off+od:off+16:2].float();mu=x.mean(0);delta=x-mu
                        ds.append([float(delta.square().sum(-1).mean()),float(mu.square().sum()),float(delta.norm(dim=-1).max())])
                dispersion.append(ds)
                # Save all responses; dense states are reserved for the fixed audit subset.
                if row["group"] in [0,32,64]:
                    finals.append(st["final"].float().cpu().numpy());spatial.append(src.float().cpu().numpy())
        data={"metrics":np.stack(metrics),"endpoints":np.stack(endpoints),"baseline":engine.metrics(base,row),"dispersion":np.array(dispersion),"condition_names":np.array(names),"bank_names":np.array([b["name"] for b in banks]),"id":np.array(row["id"])}
        if finals:data.update(final=np.stack(finals),source=np.stack(spatial))
        np.savez_compressed(path,**data)
        dump(out/(row["id"]+"-audit.json"),audits)
        if ri==0:
            dump(out.parent/"design.json",{"names":names,"banks":banks,"metrics":METRICS,"layers":engine.layers,"positions":["anchor","end"]})
        if ri%8==0 or ri==len(rows)-1:print(json.dumps({"model":engine.name,"dtype":engine.dtype,"row":ri+1,"total":len(rows),"id":row["id"],"seconds":round(time.time()-start,1)}),flush=True)
    dump(out.parent/(out.name+"-completion.json"),{"groups":len(rows),"interventions":len(rows)*8*len(names),"baseline_executions":len(rows)*34,"seconds":time.time()-start,"finished_utc":utc(),"peak_cuda_bytes":torch.cuda.max_memory_allocated() if engine.device=="cuda" else 0})

def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["calibrate","development","test","replay","precision"]);ap.add_argument("--model",required=True,choices=list(MODEL_PATHS));ap.add_argument("--dtype",default="float32",choices=["float32","bfloat16"]);ap.add_argument("--device",default="cuda");ap.add_argument("--batch",type=int,default=16);a=ap.parse_args()
    if a.stage in ["test","replay","precision"]:freeze_check()
    engine=Engine(a.model,a.dtype,a.device,a.batch)
    dest=ROOT/"results"/f"{a.model}-{a.dtype}";dest.mkdir(parents=True,exist_ok=True)
    dump(dest/"environment.json",engine.environment())
    if a.stage=="calibrate":
        result=validate(engine,read(ROOT/"data/calibration.json"));dump(dest/"calibration.json",result)
        print(json.dumps({k:v for k,v in result.items() if k!="errors"}),flush=True)
        if not result["passed"]:raise RuntimeError("Calibration failed; test evaluation prohibited")
        return
    if not read(dest/"calibration.json")["passed"]:raise RuntimeError("Calibration required")
    split="development" if a.stage=="development" else "test";rows=read(ROOT/f"data/{split}.json")
    if a.stage in ["replay","precision"]:rows=[r for r in rows if r["group"] in [0,32,64]]
    evaluate(engine,rows,dest/a.stage)
    if a.stage=="replay":
        errs={}
        for row in rows:
            x=np.load(dest/"test"/(row["id"]+".npz"));y=np.load(dest/"replay"/(row["id"]+".npz"))
            for k in ["metrics","endpoints","baseline","dispersion","final","source"]:errs[row["id"]+"/"+k]=float(np.abs(x[k]-y[k]).max())
        dump(dest/"replay-verification.json",{"errors":errs,"maximum_error":max(errs.values()),"passed":max(errs.values())<=.001})

if __name__=="__main__":main()

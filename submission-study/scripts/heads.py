"""Fixed-depth exhaustive head scan for three architecture families."""
from __future__ import annotations
import argparse,time
from common import *
from engine import Engine
import numpy as np
import torch
HEAD_MODELS=["gpt2-small","pythia-410m","qwen3-0.6b"]

def evaluate(name,replay=False):
    freeze_check();engine=Engine(name);root=ROOT/"results"/f"{name}-float32"
    if not read(root/"calibration.json")["passed"]:raise RuntimeError("Calibration required")
    dest=root/("heads-replay" if replay else "heads");dest.mkdir(parents=True,exist_ok=True)
    rows=[r for r in read(ROOT/"data/test.json") if r['task']=='ioi']
    if replay:rows=[r for r in rows if r['group'] in [0,32,64]]
    start=time.time()
    for ri,row in enumerate(rows):
        path=dest/(row["id"]+".npz")
        if path.exists():raise RuntimeError(f"Refusing to overwrite {path}")
        ps=row['encoded'][name][:18]
        base,st,_=engine.run(ps,head_capture=True)
        metrics=[];audits=[];sites=[]
        for layer in engine.layers:
            for h in range(engine.nheads):
                x=st['head_source'][layer][:,h*engine.head_dim:(h+1)*engine.head_dim]
                donors=x[2:18:2];mu=donors.mean(0)
                vec=torch.cat([donors,mu[None],x[1:2],x[0:1]])
                log,_,a=engine.run([ps[1]]*11,(layer,'end',vec,h))
                assert a['outside_max']==0 and a['target_error']==0
                identity=float(abs(log[9]-base[1]).max());assert identity<=.001
                a['identity_logit_error']=identity;audits.append(a);metrics.append(engine.metrics(log,row));sites.append([layer,h])
        np.savez_compressed(path,metrics=np.stack(metrics),baseline=engine.metrics(base,row),sites=np.array(sites))
        dump(dest/(row['id']+'-audit.json'),audits)
        if ri%8==0 or ri==len(rows)-1:print(json.dumps({"model":name,"head_group":ri+1,"of":len(rows),"seconds":time.time()-start}),flush=True)
    dump(root/(dest.name+'-completion.json'),{"groups":len(rows),"head_sites":len(sites),"interventions":len(rows)*len(sites)*11,"baseline_executions":len(rows)*18,"seconds":time.time()-start,"finished_utc":utc()})
    if replay:
        errors={}
        for r in rows:
            x=np.load(root/'heads'/(r['id']+'.npz'));y=np.load(dest/(r['id']+'.npz'))
            errors[r['id']]=max(float(abs(x[k]-y[k]).max()) for k in ['metrics','baseline'])
        dump(root/'heads-replay-verification.json',{"errors":errors,"maximum_error":max(errors.values()),"passed":max(errors.values())<=.001})

def analyze(name):
    from analyze_core import estimate,draws_for,reversal
    from scipy.stats import spearmanr
    rows=[r for r in read(ROOT/'data/test.json') if r['task']=='ioi'];ms=[];bs=[]
    for row in rows:
        with np.load(ROOT/'results'/f'{name}-float32/heads'/(row['id']+'.npz')) as z:ms.append(z['metrics']);bs.append(z['baseline']);sites=z['sites']
    m=np.stack(ms);base=np.stack(bs);draw=draws_for(rows)
    a=m[:,:,8,0];b=m[:,:,:8,0].mean(2);ma=a.mean(0);mb=b.mean(0);gap=a-b
    top={}
    for k in [5,10,20]:
        aa=np.argsort(-ma)[:k];bb=np.argsort(-mb)[:k]
        top[str(k)]={"intersection":len(set(aa)&set(bb)),"k":k,"mean_heads":sites[aa].tolist(),"individual_heads":sites[bb].tolist()}
    result={"model":name,"groups":len(rows),"head_sites":len(sites),"gap_abs":estimate(abs(gap),draw),"gap_over_0.1":estimate(abs(gap)>.1,draw),"spearman":float(spearmanr(ma,mb).statistic),"topk":top,"pair_count":int(len(sites)*(len(sites)-1)/2),"aggregate_reversals":{str(t):int(reversal(ma,mb,t).sum()) for t in [0,.05,.1]},"per_group_reversal_fraction":{str(t):estimate(reversal(a,b,t),draw) for t in [0,.05,.1]},"sites":[{"layer":int(l),"head":int(h),"mean_score":float(ma[i]),"individual_score":float(mb[i]),"gap_abs":estimate(abs(gap[:,i]),draw)} for i,(l,h) in enumerate(sites)]}
    dump(ROOT/'results/analysis'/f'{name}-heads.json',result);print(json.dumps({k:v for k,v in result.items() if k!='sites'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['evaluate','replay','analyze']);ap.add_argument('--model',choices=HEAD_MODELS);a=ap.parse_args()
    if a.stage=='analyze':
        for m in [a.model] if a.model else HEAD_MODELS:analyze(m)
    else:evaluate(a.model,a.stage=='replay')

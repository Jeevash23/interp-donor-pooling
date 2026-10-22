"""Disjoint-cluster BLiMP pooling assay using published grammatical prefixes."""
from __future__ import annotations
import argparse,collections,random,time
from common import *
from prepare import encode_prompt
import numpy as np
import torch
from transformers import AutoTokenizer
from engine import Engine

def prepare():
    toks={m:AutoTokenizer.from_pretrained(MODEL_PATHS[m],local_files_only=True) for m in PRIMARY_MODELS}
    configs={m:read(MODEL_PATHS[m]/"config.json") for m in PRIMARY_MODELS}
    buckets=collections.defaultdict(list);seen=set();rejected=collections.Counter()
    allowed={('is','are'),('are','is'),('was','were'),('were','was'),('has','have'),('have','has'),('do','does'),('does','do')}
    for line in (ROOT/"vendor/blimp-relational-noun.jsonl").read_text().splitlines():
        r=json.loads(line);good,bad=r['one_prefix_word_good'],r['one_prefix_word_bad'];text=r['one_prefix_prefix']
        if (good,bad) not in allowed:rejected["contrast"]+=1;continue
        if text in seen:rejected["duplicate_prefix"]+=1;continue
        p={"text":text,"anchor_span":[text.rfind(' ')+1,len(text)]}
        try:encoded={m:encode_prompt(tok,configs[m],p,good,bad) for m,tok in toks.items()}
        except ValueError:rejected["tokenization"]+=1;continue
        seen.add(text);buckets[good,bad].append({"pairID":r['pairID'],"good":good,"bad":bad,"encoded":encoded})
    groups=[];discarded=0
    for key,rr in sorted(buckets.items()):
        random.Random(26091288).shuffle(rr)
        for i in range(0,len(rr)-8,9):groups.append({"id":f"blimp-{len(groups):03d}","task":"blimp","seed_block":0,"good":key[0],"bad":key[1],"members":rr[i:i+9]})
        discarded+=len(rr)%9
    dump(ROOT/"data/blimp.json",groups)
    texts=[r['encoded'][PRIMARY_MODELS[0]]['text'] for g in groups for r in g['members']]
    assert len(texts)==len(set(texts))
    dump(ROOT/"data/blimp-audit.json",{"source":read(ROOT/"vendor/source-manifest.json")["blimp-relational-noun.jsonl"],"source_records":1000,"eligible_unique":sum(map(len,buckets.values())),"rejected":dict(rejected),"unused_bucket_remainders":discarded,"clusters":len(groups),"prefixes":len(texts),"within_cluster_recipients":9,"donors_per_recipient":8,"disjoint_cluster_prefixes":True,"license":"CC-BY, as stated in upstream README; attribution retained in manuscript and vendor/README.md"})
    print(f"BLiMP: {len(groups)} disjoint clusters, {len(texts)} recipient prefixes",flush=True)

def evaluate(name,replay=False):
    freeze_check();engine=Engine(name);root=ROOT/"results"/f"{name}-float32"
    if not read(root/"calibration.json")["passed"]:raise RuntimeError("Calibration required")
    dest=root/("blimp-replay" if replay else "blimp");dest.mkdir(parents=True,exist_ok=True)
    groups=read(ROOT/"data/blimp.json");start=time.time()
    if replay:groups=groups[:1]+groups[-1:]
    for gi,g in enumerate(groups):
        path=dest/(g["id"]+".npz")
        if path.exists():raise RuntimeError(f"Refusing to overwrite {path}")
        ps=[r["encoded"][name] for r in g["members"]];row={"encoded":{name:ps}}
        base,st,_=engine.run(ps,capture=True)
        metrics=[];endpoints=[];audits=[]
        for recipient in range(9):
            rm=[];re=[];indices=[j for j in range(9) if j!=recipient]
            for layer in engine.layers:
                x=st["source"][layer][indices,1];mu=x.mean(0)
                vec=torch.cat([x,mu[None],st["source"][layer][recipient:recipient+1,1]])
                log,z,a=engine.run([ps[recipient]]*10,(layer,"end",vec,None))
                rm.append(engine.metrics(log,row));audits.append(a)
                b=engine.readout(z["final"][:8].mean(0,keepdim=True));c=log[:8].mean(0,keepdim=True)
                re.append(np.stack([engine.metrics(b,row)[0],engine.metrics(c,row)[0]]))
                assert a["outside_max"]==0 and a["target_error"]==0
                assert float((log[9]-base[recipient]).abs().max())<=.001
            metrics.append(rm);endpoints.append(re)
        np.savez_compressed(path,metrics=np.array(metrics),endpoints=np.array(endpoints),baseline=engine.metrics(base,row),pair_ids=np.array([r['pairID'] for r in g['members']]))
        dump(dest/(g["id"]+"-audit.json"),audits)
        if gi%8==0 or gi==len(groups)-1:print(json.dumps({"model":name,"blimp_cluster":gi+1,"of":len(groups),"seconds":time.time()-start}),flush=True)
    dump(root/(dest.name+"-completion.json"),{"clusters":len(groups),"recipients":9*len(groups),"interventions":len(groups)*9*4*10,"baseline_executions":len(groups)*9,"seconds":time.time()-start,"finished_utc":utc()})
    if replay:
        err={}
        for g in groups:
            x=np.load(root/"blimp"/(g['id']+'.npz'));y=np.load(dest/(g['id']+'.npz'))
            for k in ['metrics','endpoints','baseline']:err[g['id']+'/'+k]=float(abs(x[k]-y[k]).max())
        dump(root/"blimp-replay-verification.json",{"maximum_error":max(err.values()),"errors":err,"passed":max(err.values())<=.001})

def analyze(name):
    from analyze_core import estimate,draws_for
    groups=read(ROOT/"data/blimp.json");ms=[];es=[];bs=[]
    for g in groups:
        with np.load(ROOT/"results"/f"{name}-float32/blimp"/(g['id']+'.npz')) as z:ms.append(z['metrics']);es.append(z['endpoints']);bs.append(z['baseline'])
    m=np.stack(ms);e=np.stack(es);base=np.stack(bs);a=m[:,:,:,8,:2];avg=m[:,:,:,:8,:2].mean(3);b=e[:,:,:,0,:2];c=e[:,:,:,1,:2]
    gap=a-avg;prop=a-b;norm=b-c;soft=c[:,:,:,1]-avg[:,:,:,1]
    draw=draws_for(groups)
    result={"model":name,"clusters":len(groups),"recipients":9*len(groups),"pair_accuracy":estimate(base[:,:,3],draw),"top1_accuracy":estimate(base[:,:,2],draw),"gap_margin_abs":estimate(abs(gap[:,:,:3,0]),draw),"prop_margin_abs":estimate(abs(prop[:,:,:3,0]),draw),"norm_margin_abs":estimate(abs(norm[:,:,:3,0]),draw),"gap_probability_abs":estimate(abs(gap[:,:,:3,1]),draw),"soft_probability_abs":estimate(abs(soft[:,:,:3]),draw),"prop_minus_norm":estimate(abs(prop[:,:,:3,0])-abs(norm[:,:,:3,0]),draw),"terminal_prop_max":float(abs(prop[:,:,-1]).max()),"closure_margin_max":float(abs(gap[:,:,:,0]-prop[:,:,:,0]-norm[:,:,:,0]).max())}
    dump(ROOT/"results/analysis"/f"{name}-blimp.json",result);print(json.dumps(result),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","evaluate","replay","analyze"]);ap.add_argument("--model",choices=PRIMARY_MODELS);a=ap.parse_args()
    if a.stage=="prepare":prepare()
    elif a.stage=="analyze":
        for m in [a.model] if a.model else PRIMARY_MODELS:analyze(m)
    else:evaluate(a.model,a.stage=="replay")

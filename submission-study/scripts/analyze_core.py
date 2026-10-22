"""Pre-specified group-level estimation from saved interventions."""
from __future__ import annotations
import argparse,itertools,math
from common import *
import numpy as np
from scipy.stats import spearmanr

def draws_for(rows,n=20000):
    rng=np.random.default_rng(26091299);strata={}
    for i,r in enumerate(rows):strata.setdefault((r["task"],r["seed_block"]),[]).append(i)
    return np.concatenate([rng.choice(ix,size=(n,len(ix)),replace=True) for _,ix in sorted(strata.items())],1)

def estimate(x,draws,alpha=.05):
    x=np.asarray(x,dtype=float);g=x.reshape(len(x),-1).mean(1)
    sampled=g[draws].mean(1)
    return {"mean":float(g.mean()),"ci":[float(v) for v in np.quantile(sampled,[alpha/2,1-alpha/2])],"n_groups":len(g),"confidence":1-alpha}

def ratio(a,b,draws):
    aa=np.asarray(a,dtype=float).reshape(len(a),-1).mean(1);bb=np.asarray(b,dtype=float).reshape(len(b),-1).mean(1)
    den=bb[draws].mean(1);val=aa[draws].mean(1)/np.maximum(den,1e-15)
    return {"ratio":float(aa.mean()/max(bb.mean(),1e-15)),"ci":[float(v) for v in np.quantile(val,[.025,.975])]}

def reversal(a,b,margin=.05):
    ii,jj=np.triu_indices(a.shape[-1],1)
    da=a[...,ii]-a[...,jj];db=b[...,ii]-b[...,jj]
    return (da*db<0)&(np.abs(da)>margin)&(np.abs(db)>margin)

def rank_diagnostic(a,b,rows,draws):
    # Both are responses at the same sites; changing direction preserves reversal status.
    ma=a.mean(0);mb=b.mean(0);ii,jj=np.triu_indices(a.shape[-1],1)
    da=a[:,ii]-a[:,jj];db=b[:,ii]-b[:,jj]
    pair_records=[]
    for k,(i,j) in enumerate(zip(ii,jj)):
        ea=estimate(da[:,k],draws);eb=estimate(db[:,k],draws)
        support=(ea["ci"][0]>0 and eb["ci"][1]<0) or (eb["ci"][0]>0 and ea["ci"][1]<0)
        pair_records.append({"i":int(i),"j":int(j),"mean_patch_difference":ea,"individual_difference":eb,"opposite_pointwise95_signs":support})
    return {"same_order":bool(np.array_equal(np.argsort(ma),np.argsort(mb))),"spearman":float(spearmanr(ma,mb).statistic),"aggregate_reversals":{str(t):int(reversal(ma,mb,t).sum()) for t in [0,.05,.1]},"per_group_reversal_fraction":{str(t):estimate(reversal(a,b,t),draws) for t in [0,.05,.1]},"pair_count":len(ii),"pairs":pair_records}

def load_model(name,split="test",dtype="float32"):
    rows=read(ROOT/f"data/{'test' if split in ['replay','precision'] else split}.json")
    if split in ["replay","precision"]:rows=[r for r in rows if r["group"] in [0,32,64]]
    path=ROOT/"results"/f"{name}-{dtype}"
    design=read(path/"design.json");data={k:[] for k in ["metrics","endpoints","baseline","dispersion"]}
    for r in rows:
        with np.load(path/split/(r["id"]+".npz")) as a:
            assert str(a["id"])==r["id"]
            assert a["condition_names"].tolist()==design["names"]
            for k in data:
                assert np.isfinite(a[k]).all();data[k].append(a[k])
    return rows,{k:np.stack(v) for k,v in data.items()},design

def decompose(data,design):
    result={};maxclose=0.
    for bi,bank in enumerate(design["banks"]):
        m=data["metrics"];ep=data["endpoints"]
        a=m[:,:,bank["mean"],:2];b=ep[:,:,bi,0,:2];c=ep[:,:,bi,1,:2];avg=m[:,:,bank["indices"],:2].mean(2)
        comp={"gap":a-avg,"prop":a-b,"norm":b-c,"soft":c[:,:,1]-avg[:,:,1],"mean":a,"individual":avg}
        close=(a[:,:,0]-avg[:,:,0])-(a[:,:,0]-b[:,:,0])-(b[:,:,0]-c[:,:,0])
        maxclose=max(maxclose,float(abs(close).max()))
        result[bank["name"]]=comp
    assert maxclose<.001
    return result,maxclose

def analyze(name,split="test",outfolder=None):
    rows,data,design=load_model(name,split);draws=draws_for(rows);comp,closure=decompose(data,design)
    nc={n:i for i,n in enumerate(design["names"])};m=data["metrics"];base=data["baseline"]
    primary=np.stack([np.abs(comp[d+"/narrow/full"]["prop"][:,:6,0])-np.abs(comp[d+"/narrow/full"]["norm"][:,:6,0]) for d in ["restore","destroy"]],-1)
    summary={"model":name,"split":split,"groups":len(rows),"primary":estimate(primary,draws,alpha=.05/6),"cells":[],"closure_max":closure,"interventions":int(m.shape[0]*m.shape[1]*m.shape[2]),"environment":read(ROOT/"results"/f"{name}-float32/environment.json")}
    for task in TASKS:
        ix=np.array([i for i,r in enumerate(rows) if r["task"]==task]);rr=[rows[i] for i in ix];dd=draws_for(rr)
        competence={"clean_pair":estimate(base[ix,0,3],dd),"corrupt_pair":estimate(base[ix,1,3],dd),"clean_top1":estimate(base[ix,0,2],dd),"corrupt_top1":estimate(base[ix,1,2],dd),"clean_narrow_donor_pair":estimate(base[ix,2:18:2,3],dd),"clean_broad_donor_pair":estimate(base[ix,18:34:2,3],dd)}
        for direction,target in [("restore",1),("destroy",0)]:
            for bank in ["narrow","broad"]:
                c=comp[f"{direction}/{bank}/full"]
                cell={"task":task,"direction":direction,"bank":bank,"competence":competence,"stats":{}}
                for k in ["gap","prop","norm"]:
                    for j,metric in enumerate(["margin","probability"]):
                        cell["stats"][f"{k}_{metric}_absolute"]=estimate(np.abs(c[k][ix,:6,j]),dd)
                        cell["stats"][f"{k}_{metric}_signed"]=estimate(c[k][ix,:6,j],dd)
                cell["stats"]["soft_probability_absolute"]=estimate(np.abs(c["soft"][ix,:6]),dd)
                cell["stats"]["prop_minus_norm"]=estimate(np.abs(c["prop"][ix,:6,0])-np.abs(c["norm"][ix,:6,0]),dd)
                cell["stats"]["gap_over_0.1_margin"]=estimate(np.abs(c["gap"][ix,:6,0])>.1,dd)
                cell["stats"]["gap_over_1pp"]=estimate(np.abs(c["gap"][ix,:6,1])>.01,dd)
                cell["stats"]["gap_relative_to_individual_patch_effect"]=ratio(np.abs(c["gap"][ix,:6,0]),np.abs(c["individual"][ix,:6,0]-base[ix,target,0,None]),dd)
                cell["rank"]=rank_diagnostic(c["mean"][ix,:6,0],c["individual"][ix,:6,0],rr,dd)
                cell["sites"]=[{"layer":design["layers"][s//2],"position":design["positions"][s%2],"gap_abs":estimate(abs(c["gap"][ix,s,0]),dd),"prop_abs":estimate(abs(c["prop"][ix,s,0]),dd),"norm_abs":estimate(abs(c["norm"][ix,s,0]),dd),"prob_gap_abs":estimate(abs(c["gap"][ix,s,1]),dd)} for s in range(8)]
                competent=(base[ix,0,3]>0)&(base[ix,2:18:2,3].min(1)>0)
                cell["competent_subset"]={"n":int(competent.sum()),"of":len(ix),"gap_abs_mean":float(np.abs(c["gap"][ix[competent],:6,0]).mean()) if competent.any() else None}
                cell["heldout_templates"]={}
                for heldout in [False,True]:
                    jj=np.array([j for j,r in enumerate(rr) if r["heldout_template"]==heldout])
                    if not len(jj):continue
                    dx=draws_for([rr[j] for j in jj])
                    cell["heldout_templates"][str(heldout)]={"gap_abs":estimate(abs(c["gap"][ix[jj],:6,0]),dx),"prop_minus_norm":estimate(abs(c["prop"][ix[jj],:6,0])-abs(c["norm"][ix[jj],:6,0]),dx)}
                cell["seed_blocks"]={str(s):float(abs(c["gap"][[i for i in ix if rows[i]["seed_block"]==s],:6,0]).mean()) for s in sorted({r["seed_block"] for r in rr})}
                summary["cells"].append(cell)
    # Fixed local radius approximation and budget-matched finite-bank comparison.
    c=comp["restore/narrow/full"];full=c["gap"][:,:6,0];half=comp["restore/narrow/r0.5"]["gap"][:,:6,0]
    summary["radius"]={}
    for radius in [.25,.5,.75,1.]:
        g=full if radius==1 else comp[f"restore/narrow/r{radius}"]["gap"][:,:6,0]
        summary["radius"][str(radius)]={"gap_abs":estimate(abs(g),draws),"ratio_to_full":ratio(abs(g),abs(full),draws),"quadratic_prediction_error":estimate(abs(full-g/(radius**2)),draws)}
    summary["quadratic_vs_zero"]=estimate(abs(full)-abs(full-4*half),draws)
    mu=m[:,:6,nc["restore/narrow/mean8"],0]
    pos=np.stack([m[:,:6,nc[f"restore/narrow/r0.5/donor{j}"],0] for j in range(2)],-1)
    neg=np.stack([m[:,:6,nc[f"restore/narrow/antithetic{j}"],0] for j in range(2)],-1)
    correction=((pos+neg-2*mu[...,None])/(2*.5**2)).mean(-1)
    expected=c["individual"][:,:6,0]
    individual=np.stack([m[:,:6,nc[f"restore/narrow/donor{j}"],0] for j in range(8)],-1)
    mc4_error=np.mean([abs(individual[:,:,list(ss)].mean(-1)-expected) for ss in itertools.combinations(range(8),4)],axis=0)
    mc5_error=np.mean([abs(individual[:,:,list(ss)].mean(-1)-expected) for ss in itertools.combinations(range(8),5)],axis=0)
    summary["estimators"]={"mean_1_forward":estimate(abs(mu-expected),draws),"curvature_5_forward":estimate(abs(mu+correction-expected),draws),"resample4_4_forward_expected_absolute_error":estimate(mc4_error,draws),"paired_1_forward":estimate(abs(m[:,:6,nc["restore/paired"],0]-expected),draws),"zero_1_forward":estimate(abs(m[:,:6,nc["restore/zero"],0]-expected),draws),"curvature_improvement_over_mean":estimate(abs(mu-expected)-abs(mu+correction-expected),draws)}
    summary['estimators']['resample5_5_forward_expected_absolute_error']=estimate(mc5_error,draws)
    summary['estimators']['curvature_improvement_over_resample5']=estimate(mc5_error-abs(mu+correction-expected),draws)
    for k in [2,4]:
        err=np.mean([abs(m[:,:6,nc[f"restore/narrow/mean{k}/order{s}"],0]-expected) for s in range(3)],axis=0)
        summary["estimators"][f"mean{k}_three_orders_1_forward_each"]=estimate(err,draws)
    dest=outfolder or ROOT/"results"/("analysis" if split=="test" else f"analysis-{split}")
    dump(dest/f"{name}.json",summary)
    np.savez_compressed(dest/f"{name}-components.npz",**{key.replace('/','__')+'__'+k:v for key,x in comp.items() for k,v in x.items()},ids=np.array([r["id"] for r in rows]))
    print(json.dumps({"model":name,"split":split,"primary":summary["primary"],"radius_half_ratio":summary["radius"]["0.5"]["ratio_to_full"],"curvature_improvement":summary["estimators"]["curvature_improvement_over_mean"]}),flush=True)
    return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=PRIMARY_MODELS);ap.add_argument("--split",default="test",choices=["test","development"]);a=ap.parse_args()
    for model in [a.model] if a.model else PRIMARY_MODELS:analyze(model,a.split)

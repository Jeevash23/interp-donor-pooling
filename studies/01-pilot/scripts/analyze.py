"""Paired group analysis of saved measurements; no model execution or outcome filtering."""
from __future__ import annotations
import argparse, csv, hashlib, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"tmp/matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS={"gpt2-small":[2,5,8,11],"gpt2-medium":[5,11,17,23]}
DRAW=np.random.default_rng(120928).integers(0,64,size=(20000,64))

def estimate(x,alpha=.05):
    x=np.asarray(x,dtype=float)
    assert x.shape[0]==64
    groups=x.reshape(64,-1).mean(1)
    draws=groups[DRAW].mean(1)
    return {"mean":float(groups.mean()),"ci":[float(v) for v in np.quantile(draws,[alpha/2,1-alpha/2])],"groups":64,"confidence":1-alpha}

def load(name):
    path=ROOT/"results"/name/"raw"
    files=sorted(path.glob("group-*.npz")); assert len(files)==64
    data={k:[] for k in ["baseline","metrics","endpoints","final","source"]}
    # Final/source arrays are inspected per file for finite values and integrity, not held in memory.
    ids=[]; records=[]
    for f in files:
        a=np.load(f)
        assert a["metrics"].shape==(4,8,20,3)
        for k in data:
            assert np.isfinite(a[k]).all()
            if k not in ["final","source"]: data[k].append(a[k])
        ids.extend(a["ids"].tolist())
        records.append({"file":str(f.relative_to(ROOT)),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()})
    expected=json.loads((ROOT/"data/evaluation.json").read_text())
    assert ids==[r["id"] for r in expected]
    return {k:np.stack(v) for k,v in data.items() if v},records

def analyze(name):
    d,hashes=load(name); m=d["metrics"]; e=d["endpoints"]; base=d["baseline"]
    all_banks={}; comp={}; max_ld_close=0.; max_prob_close=0.
    for bank,indices,mi,ei in [("k2",list(range(2)),8,0),("k4",list(range(4)),9,1),("k8",list(range(8)),10,2),("half",list(range(11,19)),10,3)]:
        a=m[:,:,:,mi,:2]; b=e[:,:,:,ei,0,:2]; c=e[:,:,:,ei,1,:2]; avg=m[:,:,:,indices,:2].mean(3)
        gap=a-avg; prop=a-b; norm=b-c; soft=c[:,:,:,1]-avg[:,:,:,1]
        max_ld_close=max(max_ld_close,float(np.abs(gap[:,:,:,0]-prop[:,:,:,0]-norm[:,:,:,0]).max()))
        max_prob_close=max(max_prob_close,float(np.abs(gap[:,:,:,1]-prop[:,:,:,1]-norm[:,:,:,1]-soft).max()))
        comp[bank]={"gap":gap,"prop":prop,"norm":norm,"soft":soft}
        all_banks[bank]={"ld_abs":estimate(np.abs(gap[:,:,:6,0])),"prob_abs":estimate(np.abs(gap[:,:,:6,1])),"ld_signed":estimate(gap[:,:,:6,0]),"prob_signed":estimate(gap[:,:,:6,1])}
    q=comp["k8"]
    summary={"model":name,"evaluation_groups":64,"recipient_configurations":256,"donor_configurations":2048,"intervention_evaluations":40960,"primary":{},"banks":all_banks,"sites":[],"baseline":{},"audit":{},"raw_manifest":hashes}
    primary=summary["primary"]
    for k in ["gap","prop","norm"]:
        primary[k+"_ld_abs"]=estimate(np.abs(q[k][:,:,:6,0]))
        primary[k+"_prob_abs"]=estimate(np.abs(q[k][:,:,:6,1]))
    primary["soft_prob_abs"]=estimate(np.abs(q["soft"][:,:,:6]))
    primary["prop_minus_norm_ld_abs"]=estimate(np.abs(q["prop"][:,:,:6,0])-np.abs(q["norm"][:,:,:6,0]),alpha=.025)
    primary["half_minus_full_abs_ld"]=estimate(np.abs(comp["half"]["gap"][:,:,:6,0])-np.abs(q["gap"][:,:,:6,0]))
    primary["half_full_ratio_of_mean_absolute_gaps"]=all_banks["half"]["ld_abs"]["mean"]/all_banks["k8"]["ld_abs"]["mean"]
    for label,i in [("clean",0),("corrupt",1)]:
        summary["baseline"][label]={"ld":estimate(base[:,:,i,0]),"prob":estimate(base[:,:,i,1]),"top1":estimate(base[:,:,i,2]),"correct_count":int(base[:,:,i,2].sum()),"n":256,"pairwise_correct_count":int((base[:,:,i,0]>0).sum())}
    summary["baseline"]["donors"]={"top1":estimate(base[:,:,2:,2]),"correct_count":int(base[:,:,2:,2].sum()),"n":2048,"pairwise_correct_count":int((base[:,:,2:,0]>0).sum())}
    for site in range(8):
        a=m[:,:,site,10,:]; indiv=m[:,:,site,:8,:].mean(2)
        item={"block":MODELS[name][site//2],"position":["S2","END"][site%2],"terminal":site>=6}
        for label,idx in [("ld",0),("prob",1)]:
            for k in ["gap","prop","norm"]:
                item[k+"_"+label+"_signed"]=estimate(q[k][:,:,site,idx])
                item[k+"_"+label+"_abs"]=estimate(np.abs(q[k][:,:,site,idx]))
        item["soft_prob_signed"]=estimate(q["soft"][:,:,site]); item["soft_prob_abs"]=estimate(np.abs(q["soft"][:,:,site]))
        for label,array in [("mean",a),("individual",indiv)]:
            item[label+"_ld"]=float(array[:,:,0].mean()); item[label+"_prob"]=float(array[:,:,1].mean()); item[label+"_top1"]=float(array[:,:,2].mean())
        item["top1_difference"]=estimate(a[:,:,2]-indiv[:,:,2])
        item["opposite_recovery_sign_count"]=int((((a[:,:,0]-base[:,:,1,0])*(indiv[:,:,0]-base[:,:,1,0]))<0).sum())
        item["large_opposite_recovery_sign_count"]=int((((a[:,:,0]-base[:,:,1,0])*(indiv[:,:,0]-base[:,:,1,0]))<0)[(np.abs(a[:,:,0]-base[:,:,1,0])>.01)&(np.abs(indiv[:,:,0]-base[:,:,1,0])>.01)].sum())
        summary["sites"].append(item)
    identity=float(np.abs(m[:,:,:,-1,:2]-base[:,:,None,1,:2]).max())
    terminal_null=float(np.abs(m[:,:,6,:,:2]-base[:,:,None,1,:2]).max())
    final_end_prop=float(np.abs(q["prop"][:,:,7,0]).max())
    final_end_states=[]; all_unchanged=True
    for f in sorted((ROOT/"results"/name/"raw").glob("group-*.npz")):
        a=np.load(f)
        final_end_states.append(float(np.abs(a["final"][:,7,:8]-a["source"][:,7,2:]).max()))
        log=json.loads(f.with_name(f.stem+"-audit.json").read_text())
        all_unchanged=all_unchanged and all(v["outside_max"]==0 and v["target_error"]==0 for v in log)
    summary["audit"]={"ld_decomposition_error":max_ld_close,"prob_decomposition_error":max_prob_close,"identity_metric_error":identity,"terminal_S2_metric_error":terminal_null,"terminal_END_prop_ld_max":final_end_prop,"terminal_END_state_error":max(final_end_states),"all_writes_exact_and_local":all_unchanged,"raw_file_count":len(hashes)}
    summary["audit"]["passed"]=max_ld_close<=.0005 and max_prob_close<=.000005 and identity<=.0005 and terminal_null<=.0005 and final_end_prop<=.0005 and max(final_end_states)==0 and all_unchanged
    assert summary["audit"]["passed"],summary["audit"]
    # Conditional summaries are diagnostics and keep the unconditioned primary result intact.
    competent=(base[:,:,0,2]==1)&(base[:,:,2:,2].min(2)==1)
    summary["competent_subset"]={"rows":int(competent.sum()),"total_rows":256,"criterion":"Clean recipient and all eight clean donors are full-vocabulary top-1 correct","ld_abs_gap_nonterminal":float(np.abs(q["gap"][:,:,:6,0])[competent].mean()) if competent.any() else None}
    # Fixed site rankings are descriptive, not searched or independently tested claims.
    mean_scores=np.array([s["mean_ld"] for s in summary["sites"][:6]])
    donor_scores=np.array([s["individual_ld"] for s in summary["sites"][:6]])
    summary["rank_diagnostic"]={"mean_order":np.argsort(-mean_scores).tolist(),"individual_order":np.argsort(-donor_scores).tolist(),"same_order":bool(np.array_equal(np.argsort(-mean_scores),np.argsort(-donor_scores)))}
    return summary,comp

def figures(summaries,out):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    colors={"gap":"#202b38","prop":"#126c9e","norm":"#cf7c1d","soft":"#8261a9"}
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharey="row",layout="constrained")
    for mi,s in enumerate(summaries):
        for pi,pos in enumerate(["S2","END"]):
            ax=axes[mi,pi]; sites=s["sites"][pi::2]; x=[v["block"] for v in sites]
            for k,label in [("gap","Total gap"),("prop","Propagation"),("norm","Final normalization")]:
                vals=[v[k+"_ld_abs"]["mean"] for v in sites]
                lows=[v[k+"_ld_abs"]["ci"][0] for v in sites]; highs=[v[k+"_ld_abs"]["ci"][1] for v in sites]
                ax.errorbar(x,vals,yerr=[np.array(vals)-lows,np.array(highs)-vals],label=label,color=colors[k],marker="o",capsize=3,lw=1.8)
            ax.set_title("GPT-2 "+s["model"].split("-")[1].capitalize()+" | "+pos)
            ax.set_xticks(x); ax.set_xlabel("Block index (zero based)"); ax.grid(alpha=.18)
            if pi==0: ax.set_ylabel("Mean absolute logit gap")
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    fig.savefig(out/"logit-decomposition.png"); fig.savefig(out/"logit-decomposition.pdf"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for ax,s in zip(axes,summaries):
        for k,offset,label in [("prop",-.22,"Propagation"),("norm",0,"Final normalization"),("soft",.22,"Softmax")]:
            vals=[v[k+"_prob_signed"]["mean"]*100 for v in s["sites"]]
            ax.bar(np.arange(8)+offset,vals,width=.22,color=colors[k],label=label)
        ax.plot(np.arange(8),[v["gap_prob_signed"]["mean"]*100 for v in s["sites"]],"ko",ms=3,label="Total gap")
        ax.axhline(0,color="#555",lw=.8); ax.set_xticks(range(8),[f'{v["block"]}\n{v["position"]}' for v in s["sites"]],fontsize=8)
        ax.set_title("GPT-2 "+s["model"].split("-")[1].capitalize()); ax.set_xlabel("Block and position"); ax.set_ylabel("Signed probability gap (percentage points)"); ax.grid(axis="y",alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False,fontsize=9)
    fig.savefig(out/"probability-decomposition.png"); fig.savefig(out/"probability-decomposition.pdf"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
    for ax,s in zip(axes,summaries):
        labels=["2 donors","4 donors","8 donors","8 donors\nhalf radius"]
        est=[s["banks"][k]["ld_abs"] for k in ["k2","k4","k8","half"]]
        vals=[v["mean"] for v in est]
        ax.bar(range(4),vals,color=["#126c9e"]*3+["#79aca6"],width=.6)
        ax.errorbar(range(4),vals,yerr=[[v["mean"]-v["ci"][0] for v in est],[v["ci"][1]-v["mean"] for v in est]],fmt="none",color="black",capsize=4)
        ax.set_xticks(range(4),labels); ax.set_ylabel("Mean absolute logit gap"); ax.set_title("GPT-2 "+s["model"].split("-")[1].capitalize()); ax.grid(axis="y",alpha=.15)
    fig.savefig(out/"donor-dispersion.png"); fig.savefig(out/"donor-dispersion.pdf"); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=ROOT/"results/analysis"); ap.add_argument("--model",choices=list(MODELS)); ap.add_argument("--no-figures",action="store_true"); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for name in ([args.model] if args.model else list(MODELS)):
        s,c=analyze(name); summaries.append(s)
        (args.output/f"{name}.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
        np.savez_compressed(args.output/f"{name}-components.npz",**{bank+"_"+k:v for bank,vals in c.items() for k,v in vals.items()})
        print(json.dumps({"model":name,"primary":s["primary"],"baseline":s["baseline"],"audit":s["audit"]},indent=2))
    if len(summaries)==2 and not args.no_figures: figures(summaries,args.output)
    keys=["model","block","position","mean_ld","individual_ld","gap_ld_abs","prop_ld_abs","norm_ld_abs","gap_prob_signed","soft_prob_signed","mean_top1","individual_top1"]
    with (args.output/"site-results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for s in summaries:
            for site in s["sites"]:
                w.writerow({k:s["model"] if k=="model" else site[k]["mean"] if isinstance(site[k],dict) else site[k] for k in keys})

if __name__=="__main__": main()

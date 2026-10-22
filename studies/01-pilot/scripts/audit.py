"""Independent readout, data, provenance and saved-analysis checks."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")

def data_audit():
    rows=json.loads((ROOT/"data/evaluation.json").read_text())
    cal=json.loads((ROOT/"data/calibration.json").read_text())
    def texts(rs):return {p["text"] for r in rs for p in [r["clean"],r["corrupt"],*r["donors"]]}
    new=texts(rows); calibration=texts(cal)
    # Release: the earlier projects whose prompt files this audit compared against are not included;
    # give their data folders in PRIOR_DATA_DIRS (separated by ':') to repeat the comparison.
    candidates=[Path(p) for p in os.environ.get("PRIOR_DATA_DIRS","").split(":") if p]
    checked=[]; matches=[]
    for d in candidates:
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix not in [".json",".jsonl",".csv",".txt"]:continue
            if p.stat().st_size>100_000_000:continue
            content=p.read_text(errors="replace");checked.append(str(p))
            for s in new:
                if s in content:matches.append({"file":str(p),"text":s})
    actual={"evaluated_rows":len(rows),"calibration_rows_generated":len(cal),"unique_evaluation_prompt_strings":len(new),"calibration_evaluation_overlap":sorted(new&calibration),"prior_files_checked":checked,"prior_exact_string_matches":matches,"scope":"Literal string scan of predecessor top-level data directories and bridge study data directories, for text/JSON/JSONL/CSV files smaller than 100 MB; not a full historical-output or training-data deduplication."}
    # Independent draws from a finite word pool can overlap. Record overlaps; never filter observed evaluation data.
    actual["calibration_evaluation_disjoint"]=not bool(actual["calibration_evaluation_overlap"])
    assert len(rows)==256 and len({r["group"] for r in rows})==64
    dump(ROOT/"results/data-audit.json",actual)
    # Release: the predecessor model manifest is not included; set PREDECESSOR_MODEL_MANIFEST to rewrite this record.
    prior=json.loads(Path(os.environ["PREDECESSOR_MODEL_MANIFEST"]).read_text()) if os.environ.get("PREDECESSOR_MODEL_MANIFEST") else None
    if prior is not None: dump(ROOT/"data/checkpoint-provenance.json",{"source":"Predecessor model-manifest.json; current actual files are separately hashed in model-manifest.json","models":[m for m in prior["models"] if m["name"] in ["gpt2-small","gpt2-medium"]]})
    print(json.dumps({"previous_files":len(checked),"matches":len(matches),"distinct_prompts":len(new)}))

def readout_audit(name,device):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    mp=Path(os.environ.get("MODELS_DIR", ROOT.parents[1]/"models"))/name   # release: see models/README.md
    with safe_open(mp/"model.safetensors",framework="pt",device="cpu") as f:
        w=f.get_tensor("wte.weight").to(device)
        gamma=f.get_tensor("ln_f.weight").to(device); beta=f.get_tensor("ln_f.bias").to(device)
    config=json.loads((mp/"config.json").read_text()); eps=config["layer_norm_epsilon"]
    def read(z,length=None):
        if length is None:
            return F.linear(F.layer_norm(z,(len(gamma),),gamma,beta,eps),w)
        # LayerNorm acts independently at each position. Match the native 3D kernel
        # shape using repeated copies; only the final position is read out.
        expanded=z[:,None,:].expand(-1,length,-1).contiguous()
        norm=F.layer_norm(expanded,(len(gamma),),gamma,beta,eps)
        return F.linear(norm[:,-1:,:],w)[:,0,:]
    def metrics(log,row):
        return torch.stack([log[:,row["io_id"]].double()-log[:,row["subject_id"]].double(),log.double().softmax(-1)[:,row["io_id"]],(log.argmax(-1)==row["io_id"]).double()],1).cpu().numpy()
    rows=json.loads((ROOT/"data/evaluation.json").read_text()); maxerr=np.zeros(3); eperr=np.zeros(3); count=0
    with torch.inference_mode():
        for g in [0,31,63]:
            a=np.load(ROOT/"results"/name/"raw"/f"group-{g:03d}.npz")
            for ri,row in enumerate(rows[4*g:4*g+4]):
                for si in range(8):
                    z=torch.tensor(a["final"][ri,si],device=device)
                    log=read(z,len(row['corrupt']['ids'])); mt=metrics(log,row)
                    maxerr=np.maximum(maxerr,np.abs(mt-a["metrics"][ri,si]).max(0)); count+=len(z)
                    for ei,idx in enumerate([list(range(2)),list(range(4)),list(range(8)),list(range(11,19))]):
                        b=metrics(read(z[idx].mean(0,keepdim=True)),row)[0]
                        c=metrics(log[idx].mean(0,keepdim=True),row)[0]
                        eperr=np.maximum(eperr,np.abs(np.stack([b,c])-a["endpoints"][ri,si,ei]).max(0))
    result={"model":name,"independent_formula":"Safetensors final LayerNorm weights and tied token embedding matrix; torch functional layer_norm then linear with native 3D shape for intervention metrics; no Transformers model or experiment Engine","groups":[0,31,63],"intervention_states_checked":count,"metric_errors":{"ld":float(maxerr[0]),"probability":float(maxerr[1]),"top1":float(maxerr[2])},"endpoint_errors":{"ld":float(eperr[0]),"probability":float(eperr[1]),"top1":float(eperr[2])}}
    result["passed"]=bool(maxerr[0]<=.0005 and maxerr[1]<=.000005 and maxerr[2]==0 and eperr[0]<=.0005 and eperr[1]<=.000005 and eperr[2]==0)
    dump(ROOT/"results"/name/"readout-audit.json",result);print(json.dumps(result,indent=2));assert result["passed"]

def statistics_audit():
    import importlib.util
    spec=importlib.util.spec_from_file_location("analysis",ROOT/"scripts/analyze.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    checks={}
    for name in ["gpt2-small","gpt2-medium"]:
        current,_=mod.analyze(name)
        saved=json.loads((ROOT/"results/analysis"/f"{name}.json").read_text())
        checks[name]=current==saved
    freeze=json.loads((ROOT/"data/freeze.json").read_text())
    from relocation import frozen_matches
    checks["frozen_inputs_or_documented_relocations_valid"]=all(frozen_matches(ROOT,p,h) for p,h in freeze["files"].items())
    dump(ROOT/"results/statistics-replay.json",{"checks":checks,"passed":all(checks.values())});assert all(checks.values());print(checks)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["data","readout","statistics"]);ap.add_argument("--model",default="gpt2-small");ap.add_argument("--device",default="cuda");args=ap.parse_args()
    if args.mode=="data":data_audit()
    elif args.mode=="readout":readout_audit(args.model,args.device)
    else:statistics_audit()

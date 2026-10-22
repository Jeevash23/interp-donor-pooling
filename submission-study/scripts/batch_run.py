"""Sequential GPU execution; stops on a failed gate, never overwrites a run."""
import argparse,subprocess,sys
from common import *
ap=argparse.ArgumentParser();ap.add_argument("phase",choices=["develop","main","supplement"]);a=ap.parse_args()
for model in PRIMARY_MODELS:
    stages=[("calibrate","float32"),("development","float32")] if a.phase=="develop" else [("calibrate","float32"),("test","float32"),("replay","float32")] if a.phase=="main" else [("calibrate","bfloat16"),("precision","bfloat16")]
    for stage,dtype in stages:
        dest=ROOT/"results"/f"{model}-{dtype}"
        done=dest/("calibration.json" if stage=="calibrate" else stage+"-completion.json")
        if done.exists() and stage=="calibrate" and read(done).get('validation_version',1)<2:
            done.rename(dest/'calibration-v1.json')
        if done.exists():
            if stage=="calibrate" and not read(done)["passed"]:raise RuntimeError(f"Failed gate exists: {model} {dtype}")
            print(f"Already complete: {model} {stage} {dtype}",flush=True);continue
        log=ROOT/"logs"/f"{stage}-{model}-{dtype}.log"
        print(f"Starting {model} {stage} {dtype} {utc()}",flush=True)
        with log.open("w") as stream:subprocess.run([sys.executable,str(ROOT/"scripts/run.py"),stage,"--model",model,"--dtype",dtype],stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f"Finished {model} {stage} {dtype} {utc()}",flush=True)

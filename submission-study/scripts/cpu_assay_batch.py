"""Finite CPU supplement worker; every model gets the original numerical gate."""
from common import *
import subprocess,sys,traceback

status=ROOT/'results/cpu-assay-status.json'
dump(status,{'state':'running','started_utc':utc(),'plan':'Six BLiMP panels followed by all three planned head scans; original frozen scientific scripts; CPU device, same float32 gates.'})
try:
    for assay,models in [('blimp',PRIMARY_MODELS),('heads',['gpt2-small','qwen3-0.6b','pythia-410m'])]:
        for model in models:
            print('Starting CPU',assay,model,utc(),flush=True)
            with (ROOT/'logs'/f'cpu-{assay}-{model}.log').open('w') as out:
                subprocess.run([sys.executable,str(ROOT/'scripts/cpu_assays.py'),assay,'--model',model],stdout=out,stderr=subprocess.STDOUT,check=True)
            print('Finished CPU',assay,model,utc(),flush=True)
    dump(status,{'state':'complete','finished_utc':utc()})
except Exception as e:
    dump(status,{'state':'failed','failed_utc':utc(),'error':str(e)});raise

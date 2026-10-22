"""CPU scheduling revision: BLiMP only; the unchanged head scans stay on GPU."""
from common import *
import subprocess,sys

status=ROOT/'results/cpu-assay-status.json'
dump(status,{'state':'running','started_utc':utc(),'plan':'Six BLiMP CPU panels only; all head scans are assigned to the existing GPU supplement runner.'})
try:
    for model in PRIMARY_MODELS:
        root=ROOT/'results'/f'{model}-float32'
        if (root/'blimp-replay-verification.json').exists() and (ROOT/'results/analysis'/f'{model}-blimp.json').exists():
            assert read(root/'blimp-replay-verification.json')['passed'];print('Already complete CPU BLiMP',model,flush=True);continue
        print('Starting CPU blimp',model,utc(),flush=True)
        with (ROOT/'logs'/f'cpu-blimp-{model}.log').open('w') as out:
            subprocess.run([sys.executable,str(ROOT/'scripts/cpu_assays.py'),'blimp','--model',model],stdout=out,stderr=subprocess.STDOUT,check=True)
        print('Finished CPU blimp',model,utc(),flush=True)
    dump(status,{'state':'complete','finished_utc':utc(),'scope':'Six BLiMP CPU panels; GPU head scans remain with the main supplementary runner.'})
except Exception as e:
    dump(status,{'state':'failed','failed_utc':utc(),'error':str(e)});raise

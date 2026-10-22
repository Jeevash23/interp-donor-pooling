"""Finite orchestration of this study's already running core and planned supplements."""
import subprocess,sys,time
from common import *

def run(script,*args,tag=None):
    tag=tag or '-'.join([script.replace('.py',''),*[x for x in args if not x.startswith('--')]])
    print('Starting',tag,utc(),flush=True)
    with (ROOT/'logs'/(tag+'.log')).open('w') as out:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],stdout=out,stderr=subprocess.STDOUT,check=True)
    print('Finished',tag,utc(),flush=True)

for model in PRIMARY_MODELS:
    root=ROOT/'results'/f'{model}-float32';deadline=time.time()+4*3600
    while not (root/'replay-verification.json').exists():
        if time.time()>deadline:raise RuntimeError('Timed out waiting for core '+model)
        if 'Traceback' in (ROOT/'logs/main-batch.log').read_text():raise RuntimeError('Core batch stopped; inspect main-batch.log')
        time.sleep(5)
    assert read(root/'replay-verification.json')['passed']
    if not (ROOT/'results/analysis'/f'{model}.json').exists():run('analyze_core.py','--model',model,tag='analysis-'+model)
    if not (root/'independent-readout.json').exists():run('audit_readout.py','--model',model,'--device','cpu',tag='audit-readout-'+model+'-cpu')
    assert read(root/'independent-readout.json')['passed']
    if not (ROOT/'logs'/('development-analysis-'+model+'-final.log')).exists():
        run('analyze_core.py','--model',model,'--split','development',tag='development-analysis-'+model+'-final')
    job=ROOT/'tmp'/('statistics-reproduction-'+model)
    if not job.exists():run('reproduce.py','statistics','--model',model,'--out',job,tag='reproduce-statistics-'+model)
    assert read(job/'verification.json')['passed']

cpu=ROOT/'results/cpu-assay-status.json'
while cpu.exists() and read(cpu)['state']=='running':time.sleep(5)
if cpu.exists() and read(cpu)['state']!='complete':raise RuntimeError('CPU supplementary worker failed; inspect its preserved logs')
time.sleep(3)
run('supplement_run.py',tag='supplement-batch')
run('external.py','analyze',tag='analysis-blimp-all')
run('heads.py','analyze',tag='analysis-heads-all')
run('analyze_supplements.py','precision',tag='analysis-precision-all')
run('analyze_supplements.py','scale',tag='analysis-scale')
run('report.py',tag='report-primary')
print('ALL PLANNED MEASUREMENTS AND ANALYSES COMPLETE',utc(),flush=True)

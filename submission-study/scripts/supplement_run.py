"""Sequential planned supplementary runs, after all primary evaluations."""
import subprocess,sys
from common import *
from heads import HEAD_MODELS
def run(script,*args):
    name='-'.join([script.replace('.py',''),*[x for x in args if not x.startswith('--')]])
    print('Starting',name,utc(),flush=True)
    with (ROOT/'logs'/f'{name}.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Finished',name,utc(),flush=True)
for model in PRIMARY_MODELS:
    root=ROOT/'results'/f'{model}-float32'
    if not (root/'test-completion.json').exists():raise RuntimeError('All primary runs must complete first')
    if not (root/'independent-readout.json').exists():run('audit_readout.py','--model',model)
    for stage,done in [('evaluate','blimp-completion.json'),('replay','blimp-replay-completion.json')]:
        if not (root/done).exists():run('external.py',stage,'--model',model)
    if model in HEAD_MODELS:
        for stage,done in [('evaluate','heads-completion.json'),('replay','heads-replay-completion.json')]:
            if not (root/done).exists():run('heads.py',stage,'--model',model)
run('batch_run.py','supplement')
for stage in ['calibrate','test','replay']:
    root=ROOT/'results/qwen3-1.7b-bfloat16';done=root/('calibration.json' if stage=='calibrate' else stage+'-completion.json')
    if not done.exists():run('run.py',stage,'--model','qwen3-1.7b','--dtype','bfloat16')

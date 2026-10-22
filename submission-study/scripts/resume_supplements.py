"""Calibration-only batch fallback for the remaining reduced-precision panel.

Preserves failed default configurations and all frozen scientific code. Only
the original default batch (16, or Pythia's built-in singleton) and batch 1
are considered. A failed singleton gate still stops the evaluation.
"""
from common import *
import shutil,subprocess,sys

def run(label,script,*args,check=True):
    print('Starting',label,utc(),flush=True)
    with (ROOT/'logs'/('resume-'+label+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
    print('Finished',label,'return code',result.returncode,utc(),flush=True)
    if check:result.check_returncode()
    return result.returncode

def choose_batch(model):
    root=ROOT/'results'/f'{model}-bfloat16';root.mkdir(parents=True,exist_ok=True)
    if (root/'batching-selection.json').exists():
        selection=read(root/'batching-selection.json');assert read(root/'calibration.json')['passed']
        return selection['selected_batch']
    attempts=[]
    if not (root/'calibration.json').exists():run(model+'-calibrate-default','run.py','calibrate','--model',model,'--dtype','bfloat16',check=False)
    cal=read(root/'calibration.json');env=read(root/'environment.json');batch=env['batch']
    if not cal['passed']:
        assert batch!=1,'The default singleton configuration failed; evaluation remains prohibited'
        for filename in ['calibration','environment']:
            dest=root/(filename+'-default-failed.json');assert not dest.exists()
            shutil.copy2(root/(filename+'.json'),dest)
        attempts.append({'batch':batch,'passed':False,'logit_max':cal['maximum_error'],'probability_max':cal['probability_max_error'],'record':'calibration-default-failed.json'})
        run(model+'-calibrate-singleton','run.py','calibrate','--model',model,'--dtype','bfloat16','--batch','1',check=False)
        cal=read(root/'calibration.json');env=read(root/'environment.json');batch=env['batch']
        assert batch==1
    attempts.append({'batch':batch,'passed':cal['passed'],'logit_max':cal['maximum_error'],'probability_max':cal['probability_max_error'],'record':'calibration.json'})
    dump(root/'batching-selection.json',{'model':model,'dtype':'bfloat16','selected_batch':batch if cal['passed'] else None,'attempts':attempts,'basis':'Reserved calibration only; original numerical gates. No reduced-precision test outcomes used.','created_utc':utc()})
    if not cal['passed']:raise RuntimeError('Singleton reduced-precision calibration failed: '+model)
    return batch

def main():
    freeze_check()
    for model in PRIMARY_MODELS:
        root=ROOT/'results'/f'{model}-bfloat16'
        batch=choose_batch(model)
        if not (root/'precision-completion.json').exists():
            run(model+'-precision','run.py','precision','--model',model,'--dtype','bfloat16','--batch',batch)
    model='qwen3-1.7b';batch=choose_batch(model);root=ROOT/'results'/f'{model}-bfloat16'
    for stage in ['test','replay']:
        if not (root/(stage+'-completion.json')).exists():run(model+'-'+stage,'run.py',stage,'--model',model,'--dtype','bfloat16','--batch',batch)
    for label,script,args in [('blimp-analysis','external.py',['analyze']),('head-analysis','heads.py',['analyze']),('precision-analysis','analyze_supplements.py',['precision']),('scale-analysis','analyze_supplements.py',['scale']),('primary-report','report.py',[])]:
        run(label,script,*args)
    print('ALL PLANNED MEASUREMENTS AND ANALYSES COMPLETE',utc(),flush=True)
    run('build-deliverables','build_deliverables.py')

if __name__=='__main__':main()

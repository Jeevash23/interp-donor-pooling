"""Complete calibrated reduced-precision panels using the documented adapter."""
from common import *
import shutil,subprocess,sys

def run(label,script,*args,check=True):
    print('Starting',label,utc(),flush=True)
    with (ROOT/'logs'/('consistent-'+label+'.log')).open('w') as out:
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],stdout=out,stderr=subprocess.STDOUT)
    print('Finished',label,'return code',result.returncode,utc(),flush=True)
    if check:result.check_returncode()
    return result.returncode

def calibrate(model):
    root=ROOT/'results'/f'{model}-bfloat16';root.mkdir(parents=True,exist_ok=True)
    if (root/'batching-selection.json').exists():
        selection=read(root/'batching-selection.json')
        if selection.get('readout_batch')==1:
            assert selection['passed'];return selection['selected_batch']
        # Retain the earlier failed, unadapted singleton attempt in full.
        for filename in ['calibration','environment','batching-selection']:
            dest=root/(filename+'-before-readout-fix.json');assert not dest.exists()
            shutil.copy2(root/(filename+'.json'),dest)
    attempts=[]
    for requested_batch in ([1] if model=='pythia-410m' else [16,1]):
        run(model+'-calibrate-batch'+str(requested_batch),'reduced_precision.py','calibrate','--model',model,'--dtype','bfloat16','--batch',requested_batch,check=False)
        cal=read(root/'calibration.json');env=read(root/'environment.json')
        filename='calibration.json'
        if not cal['passed']:
            filename=f'calibration-readout1-batch{requested_batch}-failed.json'
            assert not (root/filename).exists();shutil.copy2(root/'calibration.json',root/filename)
        attempts.append({'batch':env['batch'],'readout_batch':1,'passed':cal['passed'],'maximum_error':cal['maximum_error'],'probability_max_error':cal['probability_max_error'],'record':filename})
        if cal['passed']:
            dump(root/'batching-selection.json',{'model':model,'dtype':'bfloat16','selected_batch':env['batch'],'readout_batch':1,'passed':True,'attempts':attempts,'basis':'Reserved calibration only; unchanged gates; native default then singleton if required; one-row readout adapter.','created_utc':utc()})
            return env['batch']
    dump(root/'batching-selection.json',{'model':model,'dtype':'bfloat16','selected_batch':None,'readout_batch':1,'passed':False,'attempts':attempts,'created_utc':utc()})
    raise RuntimeError('Reduced precision remains inadmissible under original gates: '+model)

def main():
    freeze_check()
    for model in PRIMARY_MODELS:
        batch=calibrate(model);root=ROOT/'results'/f'{model}-bfloat16'
        if not (root/'precision-completion.json').exists():run(model+'-precision','reduced_precision.py','precision','--model',model,'--dtype','bfloat16','--batch',batch)
    model='qwen3-1.7b';batch=calibrate(model);root=ROOT/'results'/f'{model}-bfloat16'
    for stage in ['test','replay']:
        if not (root/(stage+'-completion.json')).exists():run(model+'-'+stage,'reduced_precision.py',stage,'--model',model,'--dtype','bfloat16','--batch',batch)
    for label,script,args in [('blimp-analysis','external.py',['analyze']),('head-analysis','heads.py',['analyze']),('precision-analysis','analyze_supplements.py',['precision']),('scale-analysis','analyze_supplements.py',['scale']),('primary-report','report.py',[])]:run(label,script,*args)
    print('ALL PLANNED MEASUREMENTS AND ANALYSES COMPLETE',utc(),flush=True)
    run('build-deliverables','build_deliverables.py')

if __name__=='__main__':main()

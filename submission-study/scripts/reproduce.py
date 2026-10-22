"""Reproduce in a new directory, leaving the original experiment untouched."""
from __future__ import annotations
import argparse,importlib,shutil,sys
from pathlib import Path
import common

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['regenerate','statistics','replay','full','blimp','heads','precision'])
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--model',choices=list(common.MODEL_PATHS),default='gpt2-small')
    ap.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    ap.add_argument('--models-root',type=Path,help='Optional cache with one subdirectory per model key')
    ap.add_argument('--device',default='cuda');a=ap.parse_args()
    src=common.ROOT;job=a.out.resolve()
    if a.model=='qwen3-1.7b' and a.stage!='regenerate' and a.dtype!='bfloat16':
        raise RuntimeError('The recorded Qwen3-1.7B extension uses --dtype bfloat16 only')
    if a.stage in ['blimp','heads','regenerate'] and a.dtype!='float32':raise RuntimeError('This stage uses the original float32 configuration')
    if a.stage=='precision' and (a.dtype!='bfloat16' or a.model not in common.PRIMARY_MODELS):raise RuntimeError('Precision reproduction requires a primary model and --dtype bfloat16')
    if a.stage=='replay' and a.dtype=='bfloat16' and a.model in common.PRIMARY_MODELS:raise RuntimeError('Use the precision stage to repeat the nine bfloat16 cases for a primary model')
    if a.stage=='replay':
        import numpy as np
        for row in common.read(src/'data/test.json'):
            if row['group'] in [0,32,64]:
                with np.load(src/'results'/f'{a.model}-{a.dtype}/test'/(row['id']+'.npz')) as z:
                    if not all(k in z.files for k in ['final','source']):
                        raise RuntimeError('Exact saved-state replay needs the dense-state archive extracted over the compact review archive. Statistical reproduction works without it; full inference can regenerate new states.')
    if job.exists():raise RuntimeError(f'Refusing to overwrite an existing output directory: {job}')
    job.mkdir(parents=True);dest=job/'study';dest.mkdir()
    for folder in ['scripts','data','docs','vendor']:
        shutil.copytree(src/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__'))
    (job/'data').mkdir()
    for split in ['calibration','evaluation']:shutil.copy2(src/'data'/f'pilot-{split}.json',job/'data'/f'{split}.json')
    if a.models_root:
        mr=a.models_root.resolve();common.MODEL_PATHS.update({m:mr/m for m in common.MODEL_PATHS})
    if a.stage!='statistics':
        manifest=common.read(src/'data/checkpoints.json')
        models=list(common.MODEL_PATHS) if a.stage=='regenerate' else [a.model]
        for model in models:
            for filename,meta in manifest[model]['files'].items():
                if a.stage=='regenerate' and filename.endswith(('.safetensors','.bin')):continue
                file=common.MODEL_PATHS[model]/filename
                if not file.exists() or common.sha(file)!=meta['sha256']:
                    raise RuntimeError(f'Checkpoint dependency differs from the recorded experiment: {model}/{filename}')
    common.ROOT=dest
    sys.path.insert(0,str(dest/'scripts'))
    common.freeze_check()
    if a.dtype=='bfloat16':
        record=common.read(src/'data/bfloat16-readout-amendment.json')
        assert all(common.sha(dest/p)==h for p,h in record['files'].items()),'Reduced-precision adapter source differs from its recorded amendment'
    if a.stage=='regenerate':
        importlib.import_module('prepare').main()
        importlib.import_module('external').prepare()
        names=['calibration.json','development.json','test.json','blimp.json']
        checks={n:common.sha(src/'data'/n)==common.sha(dest/'data'/n) for n in names}
        common.dump(job/'verification.json',{'stage':a.stage,'files_match':checks,'passed':all(checks.values())})
        if not all(checks.values()):raise RuntimeError('Generated data differ')
    elif a.stage=='statistics':
        shutil.copytree(src/'results'/f'{a.model}-{a.dtype}',dest/'results'/f'{a.model}-{a.dtype}')
        if a.dtype=='float32':
            actual=importlib.import_module('analyze_core').analyze(a.model);filename=f'{a.model}.json'
        else:
            analyzer=importlib.import_module('analyze_supplements')
            if a.model=='qwen3-1.7b':analyzer.scale();filename='qwen3-1.7b-scale.json'
            else:
                shutil.copytree(src/'results'/f'{a.model}-float32',dest/'results'/f'{a.model}-float32')
                analyzer.precision(a.model);filename=f'{a.model}-precision.json'
            actual=common.read(dest/'results/analysis'/filename)
        expected=common.read(src/'results/analysis'/filename)
        common.dump(job/'verification.json',{'stage':a.stage,'model':a.model,'dtype':a.dtype,'passed':actual==expected})
        if actual!=expected:raise RuntimeError('Statistical recomputation differs')
    elif a.stage in ['blimp','heads']:
        if a.stage=='heads' and a.model not in ['gpt2-small','pythia-410m','qwen3-0.6b']:
            raise RuntimeError('This model was not in the prespecified attention-head panel')
        target=dest/'results'/f'{a.model}-float32';target.mkdir(parents=True)
        recorded_device=common.read(src/'results'/f'{a.model}-float32'/(a.stage+'-device.json'))['device']
        if a.stage=='blimp' and recorded_device=='cpu':
            # The wrapper runs a fresh CPU gate before any new measurements.
            shutil.copy2(src/'results'/f'{a.model}-float32/cpu-calibration.json',target/'calibration.json')
            sys.argv=['cpu_assays.py',a.stage,'--model',a.model]
            importlib.import_module('cpu_assays').main()
        else:
            from engine import Engine,validate
            engine=Engine(a.model,device=a.device)
            gate=validate(engine,common.read(dest/'data/calibration.json'))
            common.dump(target/'calibration.json',gate)
            common.dump(target/(a.stage+'-device.json'),{'device':a.device,'environment':engine.environment()})
            del engine
            if not gate['passed']:raise RuntimeError('Fresh supplementary calibration failed')
            assay=importlib.import_module('heads' if a.stage=='heads' else 'external');assay.Engine=lambda name:Engine(name,device=a.device)
            assay.evaluate(a.model,False);assay.evaluate(a.model,True);assay.analyze(a.model)
        actual=common.read(dest/'results/analysis'/f'{a.model}-{a.stage}.json')
        expected=common.read(src/'results/analysis'/f'{a.model}-{a.stage}.json')
        checks=common.read(target/(a.stage+'-replay-verification.json'))
        common.dump(job/'verification.json',{'stage':a.stage,'model':a.model,'replay_passed':checks['passed'],'maximum_error':checks['maximum_error'],'original_summary_exact':actual==expected,'passed':checks['passed'] and actual==expected})
        if actual!=expected:raise RuntimeError('Supplementary recomputation differs; inspect device/environment and raw values')
    else:
        source_model=src/'results'/f'{a.model}-{a.dtype}'
        batch=common.read(source_model/'environment.json')['batch']
        if a.stage=='replay':
            target=dest/'results'/f'{a.model}-{a.dtype}';target.mkdir(parents=True)
            for n in ['design.json','calibration.json','environment.json']:shutil.copy2(source_model/n,target/n)
            (target/'test').mkdir()
            for row in common.read(src/'data/test.json'):
                if row['group'] in [0,32,64]:shutil.copy2(source_model/'test'/(row['id']+'.npz'),target/'test'/(row['id']+'.npz'))
        runner=importlib.import_module('reduced_precision' if a.dtype=='bfloat16' else 'run')
        stages=['replay'] if a.stage=='replay' else ['calibrate','precision'] if a.stage=='precision' else ['calibrate','test','replay']
        for stage in stages:
            sys.argv=['run.py',stage,'--model',a.model,'--device',a.device,'--dtype',a.dtype,'--batch',str(batch)];runner.main()
        if a.stage=='precision':
            import numpy as np
            errors={}
            for row in common.read(src/'data/test.json'):
                if row['group'] not in [0,32,64]:continue
                with np.load(source_model/'precision'/(row['id']+'.npz')) as old,np.load(dest/'results'/f'{a.model}-bfloat16/precision'/(row['id']+'.npz')) as new:
                    assert np.array_equal(old['condition_names'],new['condition_names'])
                    errors[row['id']]=max(float(abs(old[k]-new[k]).max()) for k in ['metrics','endpoints','baseline','dispersion'])
            checks={'passed':max(errors.values())<=.001,'maximum_error':max(errors.values()),'errors':errors}
            if not checks['passed']:raise RuntimeError('Reduced-precision response reproduction differs')
        else:checks=common.read(dest/'results'/f'{a.model}-{a.dtype}/replay-verification.json')
        common.dump(job/'verification.json',{'stage':a.stage,'model':a.model,'dtype':a.dtype,'passed':checks['passed'],'maximum_error':checks['maximum_error']})
    print('Reproduction complete:',job/'verification.json',flush=True)

if __name__=='__main__':main()

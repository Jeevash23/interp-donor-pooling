"""Run the unchanged supplementary assays on a separately calibrated CPU engine."""
from common import *
from engine import Engine,validate
import argparse,time

def cpu_engine(name):return Engine(name,device='cpu')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('assay',choices=['blimp','heads']);ap.add_argument('--model',required=True,choices=PRIMARY_MODELS);a=ap.parse_args()
    freeze_check();root=ROOT/'results'/f'{a.model}-float32';record=root/'cpu-calibration.json'
    if not record.exists():
        start=time.time();engine=cpu_engine(a.model);cal=validate(engine,read(ROOT/'data/calibration.json'));cal['environment']=engine.environment();cal['seconds']=time.time()-start
        dump(record,cal);del engine
    if not read(record)['passed']:raise RuntimeError('CPU gate failed; CPU assay prohibited')
    # Device is the only adapter override; frozen assay and engine code are unchanged.
    if a.assay=='blimp':
        import external as assay
    else:
        import heads as assay
    assay.Engine=cpu_engine
    dump(root/(a.assay+'-device.json'),{'device':'cpu','dtype':'float32','calibration':'cpu-calibration.json','environment':read(record)['environment'],'adapter':'scripts/cpu_assays.py','started_utc':utc()})
    for replay in [False,True]:
        stage=a.assay+('-replay' if replay else '')
        if not (root/(stage+'-completion.json')).exists():assay.evaluate(a.model,replay)
    assert read(root/(a.assay+'-replay-verification.json'))['passed']
    assay.analyze(a.model)

if __name__=='__main__':main()

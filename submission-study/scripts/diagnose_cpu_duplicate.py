"""Calibration-only diagnosis; never changes the measured scientific mean operator."""
from common import *
from engine import Engine
import torch

engine=Engine('pythia-410m',device='cpu')
row=next(r for r in read(ROOT/'data/calibration.json') if r['task']=='kv')
ps=row['encoded']['pythia-410m'];_,states,_=engine.run(ps,capture=True)
source=states['source'][17][2,1]
repeat32=source.repeat(8,1).mean(0)
repeat64=source.double().repeat(8,1).mean(0).float()
log,_,audit=engine.run([ps[1]]*3,(17,'end',torch.stack([source,repeat32,repeat64]),None))
record={'model':'pythia-410m','device':'cpu','split':'calibration','id':row['id'],'layer':17,'position':'end',
    'float32_repeated_mean_input_error':float(abs(source-repeat32).max()),
    'float64_repeated_mean_input_error':float(abs(source-repeat64).max()),
    'float32_repeated_mean_output_error':float(abs(log[0]-log[1]).max()),
    'float64_repeated_mean_output_error':float(abs(log[0]-log[2]).max()),
    'write_audit':audit,'note':'The float64 mean is a numerical diagnosis only. Main and supplementary scientific means are unchanged; the CPU Pythia gate remains failed and its assay uses GPU.',
    'finished_utc':utc()}
dump(ROOT/'results/pythia-410m-float32/cpu-duplicate-diagnosis.json',record)
print(json.dumps(record),flush=True)

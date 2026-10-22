"""Calibration-only diagnosis of readout batch shape in reduced precision."""
from common import *
from engine import Engine,validate
import torch

def main():
    model='gpt2-small';e=Engine(model,'bfloat16','cuda',1)
    original=e.readout
    def one_row(z):return torch.cat([original(z[i:i+1]) for i in range(len(z))])
    e.readout=one_row
    result=validate(e,read(ROOT/'data/calibration.json'))
    result['environment']=e.environment();result['readout_batch_size']=1
    result['scope']='Reserved calibration only. The readout applies the same frozen function one row at a time, matching native singleton scoring. Scientific one-row endpoint readouts are unchanged.'
    dump(ROOT/'results'/f'{model}-bfloat16/calibration-consistent-readout-diagnostic.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['errors','environment']}),flush=True)

if __name__=='__main__':main()

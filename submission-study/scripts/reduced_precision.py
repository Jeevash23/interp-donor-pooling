"""Reduced-precision adapter with one-row final readout evaluation.

The frozen experiment already calls readout on one-row mean states. Those
scientific calls take the exact original branch below. Multi-row readout is
used by calibration reconstruction; splitting it aligns its kernel shape
with native singleton scoring. No mean, score, threshold or weights change.
"""
import sys
import torch
from engine import Engine as OriginalEngine
import run as frozen_run

class ConsistentReadoutEngine(OriginalEngine):
    def readout(self,z):
        if len(z)==1:return super().readout(z)
        return torch.cat([super(ConsistentReadoutEngine,self).readout(z[i:i+1]) for i in range(len(z))])
    def environment(self):
        value=super().environment();value.update(readout_batch=1,execution_adapter='scripts/reduced_precision.py')
        return value

def main():
    if '--dtype' not in sys.argv or sys.argv[sys.argv.index('--dtype')+1]!='bfloat16':
        raise RuntimeError('This adapter is restricted to the separate bfloat16 panel')
    frozen_run.Engine=ConsistentReadoutEngine
    frozen_run.main()

if __name__=='__main__':main()

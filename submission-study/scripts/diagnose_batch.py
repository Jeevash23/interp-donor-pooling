from common import *
from engine import Engine
import torch
e=Engine('pythia-410m');rows=read(ROOT/'data/calibration.json');checks={}
for task in TASKS:
 r=next(x for x in rows if x['task']==task);ps=r['encoded'][e.name];p=ps[1]
 with torch.inference_mode():native=e.model(torch.tensor([p['ids']],device=e.device),use_cache=False,logits_to_keep=1).logits[:,0]
 for size in [1,2,3,4,8,16]:
  e.batch=size;log,st,_=e.run(ps,capture=True)
  errs=[]
  for layer in e.layers:
   for pi,pos in enumerate(['anchor','end']):
    x=st['source'][layer][1,pi];v=x.repeat(3,1);y,_,_=e.run([p]*3,(layer,pos,v,None));errs.append(float(abs(y-log[1]).max()))
  checks[f'{task}/{size}']={'native_full_vocab':float(abs(log[1]-native[0]).max()),'native_target_probability':float(abs(e.metrics(log[1:2],r)[0,1]-e.metrics(native,r)[0,1])),'max_identity':max(errs)}
  print(task,size,checks[f'{task}/{size}'],flush=True)
dump(ROOT/'results/pythia-batch-diagnostic.json',checks)

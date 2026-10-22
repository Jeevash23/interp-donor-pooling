"""Independent float64 normalization/unembedding from safetensors, no Transformers model."""
from __future__ import annotations
import argparse,time
from common import *
import numpy as np
import torch
from safetensors import safe_open

def checkpoint_tensors(name,device):
    cfg=read(MODEL_PATHS[name]/'config.json')
    if name.startswith('gpt2'):
        names=['ln_f.weight','ln_f.bias','wte.weight'];kind='layernorm';eps=cfg['layer_norm_epsilon']
    elif name.startswith('pythia'):
        names=['gpt_neox.final_layer_norm.weight','gpt_neox.final_layer_norm.bias','embed_out.weight'];kind='layernorm';eps=cfg['layer_norm_eps']
    else:
        names=['model.norm.weight',None,'lm_head.weight'];kind='rmsnorm';eps=cfg['rms_norm_eps']
    tensors={}
    for file in MODEL_PATHS[name].glob('*.safetensors'):
        with safe_open(file,framework='pt',device='cpu') as f:
            for key in names:
                if key is not None and key in f.keys():tensors[key]=f.get_tensor(key).to(device=device,dtype=torch.float64)
    assert all(k is None or k in tensors for k in names)
    return tensors[names[0]],tensors.get(names[1]),tensors[names[2]],kind,eps

def readout(z,w,b,head,kind,eps):
    if kind=='layernorm':
        centered=z-z.mean(-1,keepdim=True)
        normed=centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+eps)
    else:normed=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps)
    normed=normed*w
    if b is not None:normed=normed+b
    return normed@head.T

def scores(log,good,bad):
    return torch.stack([log[:,good]-log[:,bad],log.log_softmax(-1)[:,good].exp()],-1).cpu().numpy()

def audit(name,device='cuda'):
    start=time.time();torch.set_num_threads(4)
    w,b,head,kind,eps=checkpoint_tensors(name,device)
    rows=[r for r in read(ROOT/'data/test.json') if r['group'] in [0,32,64]]
    base=ROOT/'results'/f'{name}-float32';design=read(base/'design.json');errors={};n=0
    with torch.inference_mode():
        for row in rows:
            x=np.load(base/'test'/(row['id']+'.npz'));good=row['encoded'][name][0]['good_id'];bad=row['encoded'][name][0]['bad_id']
            mm=0.;mp=0.;em=0.;ep=0.;top_disagree=0
            for site in range(8):
                zz=torch.tensor(x['final'][site],device=device,dtype=torch.float64)
                logits=torch.cat([readout(zz[j:j+16],w,b,head,kind,eps) for j in range(0,len(zz),16)])
                q=scores(logits,good,bad);ref=x['metrics'][site,:,:2]
                mm=max(mm,float(abs(q[:,0]-ref[:,0]).max()));mp=max(mp,float(abs(q[:,1]-ref[:,1]).max()))
                tops=(logits.argmax(-1)==good).cpu().numpy().astype(float)
                top_disagree+=int((tops!=x['metrics'][site,:,2]).sum())
                for bi,bank in enumerate(design['banks']):
                    # Oracle uses higher-precision means as well as higher-precision readout.
                    lb=readout(zz[bank['indices']].mean(0,keepdim=True),w,b,head,kind,eps)
                    lc=logits[bank['indices']].mean(0,keepdim=True)
                    endpoints=np.stack([scores(lb,good,bad)[0],scores(lc,good,bad)[0]])
                    er=abs(endpoints-x['endpoints'][site,bi,:,:2]);em=max(em,float(er[:,0].max()));ep=max(ep,float(er[:,1].max()))
                n+=len(zz)
            errors[row['id']]={'margin_max':mm,'probability_max':mp,'endpoint_margin_max':em,'endpoint_probability_max':ep,'target_top1_disagreements':top_disagree}
            print(name,row['id'],errors[row['id']],flush=True)
    maxima={k:max(v[k] for v in errors.values()) for k in next(iter(errors.values()))}
    passed=max(maxima['margin_max'],maxima['endpoint_margin_max'])<=.001 and max(maxima['probability_max'],maxima['endpoint_probability_max'])<=.00002
    result={'model':name,'method':'Explicit float64 final normalization and unembedding from checkpoint tensors; no Transformers model','states_checked':n,'groups':len(rows),'maximum_errors':maxima,'per_group':errors,'margin_tolerance':.001,'probability_tolerance':.00002,'passed':passed,'seconds':time.time()-start,'finished_utc':utc()}
    dump(base/'independent-readout.json',result)
    if not passed:raise RuntimeError('Independent readout check failed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=PRIMARY_MODELS);ap.add_argument('--device',default='cuda');a=ap.parse_args();audit(a.model,a.device)

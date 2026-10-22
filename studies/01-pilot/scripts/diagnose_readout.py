from experiment import Engine,ROOT,MODELS
import torch, numpy as np
from safetensors import safe_open
import torch.nn.functional as F
m=Engine('gpt2-small','cuda')
a=np.load(ROOT/'results/gpt2-small/raw/group-000.npz')
z=torch.tensor(a['final'][0,5],device='cuda')
with safe_open(MODELS/'gpt2-small/model.safetensors',framework='pt',device='cpu') as f:
 w=f.get_tensor('wte.weight').cuda();g=f.get_tensor('ln_f.weight').cuda();b=f.get_tensor('ln_f.bias').cuda()
with torch.inference_mode():
 print('eps',m.model.transformer.ln_f.eps,'dtype',w.dtype)
 print('weights',float((w-m.model.lm_head.weight).abs().max()),float((g-m.model.transformer.ln_f.weight).abs().max()),float((b-m.model.transformer.ln_f.bias).abs().max()))
 manual=F.linear(F.layer_norm(z,(768,),g,b,1e-5),w)
 lib=m.readout(z)
 print('manual_library',float((manual-lib).abs().max()))
 print('norm',float((F.layer_norm(z,(768,),g,b,1e-5)-m.model.transformer.ln_f(z)).abs().max()))
 print('readoutmetric',m.metrics(lib, __import__('json').loads((ROOT/'data/evaluation.json').read_text())[0])[:2],a['metrics'][0,5,:2])

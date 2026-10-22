"""Descriptive lower-precision checks; never merge these into the float32 primary family."""
from __future__ import annotations
import argparse
from common import *
from analyze_core import load_model,draws_for,estimate,rank_diagnostic
import numpy as np

def bank_scores(data,design):
    return {b['name']:(data['metrics'][:,:,b['mean'],:2],data['metrics'][:,:,b['indices'],:2].mean(2)) for b in design['banks']}

def closure_error(data,design):
    # Quantized averaging of logits may fail to commute with the linear margin.
    errors=[]
    for i,b in enumerate(design['banks']):
        expected=data['metrics'][:,:,b['indices'],0].mean(2)
        errors.append(abs(data['endpoints'][:,:,i,1,0]-expected).max())
    return float(max(errors))

def null_diagnostics(scores):
    full={k:a-r for k,(a,r) in scores.items() if k.endswith('/full')}
    return {'terminal_anchor_gap_max':max(float(abs(g[:,6,0]).max()) for g in full.values()),
        'terminal_anchor_probability_gap_max':max(float(abs(g[:,6,1]).max()) for g in full.values()),
        'terminal_anchor_narrow_gap_max':max(float(abs(g[:,6,0]).max()) for k,g in full.items() if '/narrow/' in k)}

def precision(name):
    rows,fp,design=load_model(name,'replay','float32')
    br,bf,bd=load_model(name,'precision','bfloat16')
    assert [r['id'] for r in rows]==[r['id'] for r in br] and design==bd
    sf=bank_scores(fp,design);sb=bank_scores(bf,bd);cells=[]
    for task in TASKS:
        ix=[i for i,r in enumerate(rows) if r['task']==task]
        for direction in ['restore','destroy']:
            for bank in ['narrow','broad']:
                key=f'{direction}/{bank}/full';af,rf=sf[key];ab,rb=sb[key]
                gf=(af-rf)[ix,:6,0];gb=(ab-rb)[ix,:6,0]
                cells.append({'task':task,'direction':direction,'bank':bank,'groups':len(ix),
                    'fp32_gap_abs':float(abs(gf).mean()),'bf16_gap_abs':float(abs(gb).mean()),
                    'gap_absolute_difference':float(abs(gb-gf).mean()),'gap_difference_max':float(abs(gb-gf).max()),
                    'margin_absolute_difference':float(abs(bf['metrics'][ix,:,:,0]-fp['metrics'][ix,:,:,0]).mean()),
                    'probability_absolute_difference':float(abs(bf['metrics'][ix,:,:,1]-fp['metrics'][ix,:,:,1]).mean())})
    result={'model':name,'groups':len(rows),'ids':[r['id'] for r in rows],
        'interpretation':'Nine predetermined cases; numerical-configuration diagnostic with both dtype and selected batch size reported, no isolated causal dtype claim or population confidence interval',
        'float32_batch':read(ROOT/'results'/f'{name}-float32/environment.json')['batch'],
        'bfloat16_batch':read(ROOT/'results'/f'{name}-bfloat16/environment.json')['batch'],
        'readout_batch':1,
        'cells':cells,'bf16_linear_endpoint_closure_error':closure_error(bf,bd),
        'fp32_linear_endpoint_closure_error':closure_error(fp,design),
        'baseline_pair_disagreements':int((bf['baseline'][:,:,3]!=fp['baseline'][:,:,3]).sum()),
        'baseline_comparisons':int(fp['baseline'][:,:,3].size),
        'all_intervention_margin_mae':float(abs(bf['metrics'][:,:,:,0]-fp['metrics'][:,:,:,0]).mean()),
        'all_intervention_probability_mae':float(abs(bf['metrics'][:,:,:,1]-fp['metrics'][:,:,:,1]).mean()),
        'identity_margin_max':max(a['identity_margin_error'] for r in rows for a in read(ROOT/'results'/f'{name}-bfloat16/precision'/(r['id']+'-audit.json'))),
        'calibration':read(ROOT/'results'/f'{name}-bfloat16/calibration.json')}
    result.update(null_diagnostics(sb))
    dump(ROOT/'results/analysis'/f'{name}-precision.json',result)
    print(name,'precision',result['all_intervention_margin_mae'],result['bf16_linear_endpoint_closure_error'],flush=True)

def scale():
    name='qwen3-1.7b';rows,data,design=load_model(name,'test','bfloat16');scores=bank_scores(data,design)
    result={'model':name,'dtype':'bfloat16','groups':len(rows),'interventions':int(data['metrics'].shape[0]*data['metrics'].shape[1]*data['metrics'].shape[2]),
        'interpretation':'Numerically limited size diagnostic on a post-trained model; identity and structural-null artifacts prevent treating this run as a successful robustness replication; excluded from float32 primary comparisons',
        'linear_endpoint_closure_error':closure_error(data,design),'cells':[],
        'environment':read(ROOT/'results'/f'{name}-bfloat16/environment.json')}
    for task in TASKS:
        ix=np.array([i for i,r in enumerate(rows) if r['task']==task]);rr=[rows[i] for i in ix];draw=draws_for(rr)
        for direction in ['restore','destroy']:
            for bank in ['narrow','broad']:
                a,r=scores[f'{direction}/{bank}/full'];g=a-r
                result['cells'].append({'task':task,'direction':direction,'bank':bank,
                    'gap_margin_abs':estimate(abs(g[ix,:6,0]),draw),
                    'gap_probability_abs':estimate(abs(g[ix,:6,1]),draw),
                    'clean_pair':estimate(data['baseline'][ix,0,3],draw),
                    'clean_top1':estimate(data['baseline'][ix,0,2],draw),
                    'rank':rank_diagnostic(a[ix,:6,0],r[ix,:6,0],rr,draw)})
    draw=draws_for(rows)
    result['gap_margin_abs']=estimate(np.stack([abs((scores[f'{d}/narrow/full'][0]-scores[f'{d}/narrow/full'][1])[:,:6,0]) for d in ['restore','destroy']],-1),draw)
    audits=[a for r in rows for a in read(ROOT/'results'/f'{name}-bfloat16/test'/(r['id']+'-audit.json'))]
    result['identity_margin_max']=max(a['identity_margin_error'] for a in audits)
    result['identity_audit_count']=len(audits)
    result['identity_margin_above_half_logit']=sum(a['identity_margin_error']>.5 for a in audits)
    result['identity_reference_scope']='Post-run descriptive half-logit margin reference; no bfloat16 per-test identity exclusion rule was enforced by the frozen runner'
    result.update(null_diagnostics(scores))
    dump(ROOT/'results/analysis'/f'{name}-scale.json',result)
    print(name,'scale',result['gap_margin_abs'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['precision','scale']);ap.add_argument('--model',choices=PRIMARY_MODELS);a=ap.parse_args()
    if a.stage=='scale':scale()
    else:
        for m in [a.model] if a.model else PRIMARY_MODELS:precision(m)

"""Descriptive pointwise sign support for prespecified aggregate rank margins.

This extends the frozen head summary, without new model inference. Residual
pair intervals already exist in the frozen primary analysis. Head intervals
use the same group/seed-stratified bootstrap and preserve paired head scores.
They are not simultaneous confidence intervals or circuit discoveries.
"""
from common import *
from analyze_core import draws_for
from heads import HEAD_MODELS
from report import LABELS,GEN,table
import numpy as np

def main():
    result={'interval_confidence':.95,'simultaneous':False,'models':{},'head_models':{}}
    for model in PRIMARY_MODELS:
        summary=read(ROOT/'results/analysis'/f'{model}.json');counts={str(t):0 for t in [0,.05,.1]};support={str(t):0 for t in [0,.05,.1]}
        for cell in summary['cells']:
            for pair in cell['rank']['pairs']:
                a=pair['mean_patch_difference']['mean'];b=pair['individual_difference']['mean']
                for t in [0,.05,.1]:
                    rev=a*b<0 and abs(a)>t and abs(b)>t
                    counts[str(t)]+=int(rev);support[str(t)]+=int(rev and pair['opposite_pointwise95_signs'])
        assert counts=={str(t):sum(c['rank']['aggregate_reversals'][str(t)] for c in summary['cells']) for t in [0,.05,.1]}
        result['models'][model]={'pairs':180,'point_reversals':counts,'reversals_with_opposite_pointwise95_signs':support}
    rows=[r for r in read(ROOT/'data/test.json') if r['task']=='ioi'];draw=draws_for(rows)
    weights=np.stack([np.bincount(d,minlength=len(rows)) for d in draw]).astype(np.float64)/len(rows)
    for model in HEAD_MODELS:
        responses=[]
        for row in rows:
            with np.load(ROOT/'results'/f'{model}-float32/heads'/(row['id']+'.npz')) as z:
                responses.append(z['metrics']);sites=z['sites']
        responses=np.stack(responses);a=responses[:,:,8,0];b=responses[:,:,:8,0].mean(2)
        ba=weights@a;bb=weights@b;ma=a.mean(0);mb=b.mean(0)
        ii,jj=np.triu_indices(a.shape[1],1);pairs=[]
        counts={str(t):0 for t in [0,.05,.1]};support={str(t):0 for t in [0,.05,.1]}
        for start in range(0,len(ii),64):
            iii=ii[start:start+64];jjj=jj[start:start+64]
            ca=np.quantile(ba[:,iii]-ba[:,jjj],[.025,.975],axis=0)
            cb=np.quantile(bb[:,iii]-bb[:,jjj],[.025,.975],axis=0)
            for k,(i,j) in enumerate(zip(iii,jjj)):
                da=float(ma[i]-ma[j]);db=float(mb[i]-mb[j])
                opposite=bool((ca[0,k]>0 and cb[1,k]<0) or (cb[0,k]>0 and ca[1,k]<0))
                for t in [0,.05,.1]:
                    rev=da*db<0 and abs(da)>t and abs(db)>t
                    counts[str(t)]+=int(rev);support[str(t)]+=int(rev and opposite)
                pairs.append({'i':sites[i].tolist(),'j':sites[j].tolist(),'mean_patch_difference':{'mean':da,'ci':ca[:,k].tolist()},'individual_difference':{'mean':db,'ci':cb[:,k].tolist()},'opposite_pointwise95_signs':opposite})
        expected=read(ROOT/'results/analysis'/f'{model}-heads.json')
        assert counts==expected['aggregate_reversals']
        detail={'model':model,'groups':len(rows),'bootstrap_draws':len(draw),'interval_confidence':.95,'simultaneous':False,'pairs':pairs}
        dump(ROOT/'results/analysis'/f'{model}-head-pair-intervals.json',detail)
        result['head_models'][model]={'pairs':len(ii),'point_reversals':counts,'reversals_with_opposite_pointwise95_signs':support}
        print(model,'head rank intervals complete',flush=True)
    dump(ROOT/'results/analysis/rank-confidence.json',result)
    rr=[]
    for kind,key in [('Residual','models'),('Heads','head_models')]:
        for model,x in result[key].items():
            rr.append([LABELS[model],kind,x['pairs'],x['point_reversals']['0.05'],x['reversals_with_opposite_pointwise95_signs']['0.05']])
    (GEN/'rank-confidence.tex').write_text(table(['Checkpoint','Panel','Pairs','Rev. at 0.05','Opposite 95\\% signs'],rr,'Aggregate rank margins with descriptive uncertainty. The last column additionally requires the two paired score-difference intervals to lie on opposite sides of zero. These are pointwise 95\\% intervals, not simultaneous tests; pair comparisons share components and groups. They support a narrower reading of point-order changes without establishing different circuits.','tab:rank-confidence',small=True))

if __name__=='__main__':main()

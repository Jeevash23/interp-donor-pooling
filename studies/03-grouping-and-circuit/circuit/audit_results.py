"""Independent saved-data audit after inference; does not run a model."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    frozen=read(ROOT/'data/freeze.json')
    for p,h in frozen['files'].items():assert sha(ROOT/p)==h,p
    screen=read(ROOT/'data/screen.json')
    selected=read(ROOT/'data/selected-test.json')
    sel_audit=read(ROOT/'results/screen-selection.json')
    assert sha(ROOT/'data/selected-test.json')==sel_audit['selected_test_sha256']
    assert sha(ROOT/'results/screen.npz')==sel_audit['screen_sha256']
    with np.load(ROOT/'results/screen.npz') as z:
        population=z['metrics'];screen_pred=z['prediction']
    assert population.shape==(8192,9) and np.isfinite(population).all()
    assert np.array_equal(population[:,4],population[:,0]>0)
    assert np.array_equal(population[:,3],screen_pred==np.array([r['recipient']['target_id'] for r in screen]))
    expected=[]
    for correct,key in [(False,'pair_error'),(True,'pair_correct')]:
        candidates=[i for i in range(len(screen)) if bool(population[i,4])==correct]
        chosen=sorted(candidates,key=lambda i:screen[i]['selection_priority'])[:sel_audit['targets'][key]]
        expected+=chosen
        assert sel_audit['strata'][key]['selected_n']==len(chosen)
        assert sel_audit['strata'][key]['screen_population_n']==len(candidates)
    assert sorted(expected)==[r['screen_index'] for r in selected]
    testfiles=list((ROOT/'results/test').glob('*.npz'))
    assert len(testfiles)==len(selected)
    arrays=[];bases=[];weights=[];top1errors=0
    identity_max=allkept_max=0.
    for r in selected:
        si=r['screen_index'];key=r['selection_stratum']
        assert r['population_weight']==sel_audit['strata'][key]['screen_population_n']/sel_audit['strata'][key]['selected_n']
        assert r['screen_baseline']==population[si].tolist()
        with np.load(ROOT/'results/test'/f"{r['id']}.npz") as z:
            m=z['metrics'];b=z['baseline'];pred=z['prediction'];bp=int(z['baseline_prediction'][0])
            assert m.shape==(2,9,9) and np.isfinite(m).all()
            assert np.isfinite(z['mixture_metrics']).all() and np.isfinite(z['mean_logit_metrics']).all()
            assert np.array_equal(m[:,:,4],m[:,:,0]>0)
            assert np.array_equal(m[:,:,3],pred==int(z['target_id']))
            assert np.array_equal(m[:,:,5],pred==bp)
            assert np.array_equal(m[:,:,6],(m[:,:,0]>0)==bool(b[4]))
            assert np.allclose(np.exp(m[:,:,2]),m[:,:,1],atol=1e-15,rtol=1e-12)
            assert np.array_equal(b[4:7],np.array([population[si,4],1,1]))
            assert bp==screen_pred[si]
            assert abs(b[0]-population[si,0])<=.001
            identity_max=max(identity_max,float(z['identity_logit_error']))
            allkept_max=max(allkept_max,float(z['all_kept_logit_error']))
            arrays.append(m);bases.append(b);weights.append(r['population_weight'])
        hooks=read(ROOT/'results/test'/f"{r['id']}-audit.json")
        for v in hooks.values():assert v['outside_changed_max']==v['replacement_error_max']==0.
    assert identity_max<=.001 and allkept_max<=.001
    arrays=np.stack(arrays);weights=np.array(weights)
    analysis=read(ROOT/'results/analysis.json')
    assert analysis['groups']==len(selected)
    for ci,key in enumerate(['reference','empty_attention']):
        for j,metric in enumerate(analysis['metric_order']):
            expected_value=float(np.average(arrays[:,ci,0,j]-arrays[:,ci,1:,j].mean(1),weights=weights))
            assert np.isclose(expected_value,analysis['circuits'][key]['mean_minus_expected_resample'][metric]['mean'],atol=1e-14)
    replay=read(ROOT/'results/replay-verification.json')
    assert replay['passed'] and replay['maximum_error']<=.001
    replay_rows=[r for key in sorted({r['selection_stratum'] for r in selected}) for r in [r for r in selected if r['selection_stratum']==key][:4]]
    assert set(replay['errors'])=={r['id'] for r in replay_rows}
    for r in replay_rows:
        with np.load(ROOT/'results/test'/f"{r['id']}.npz") as a,np.load(ROOT/'results/replay'/f"{r['id']}.npz") as b:
            for k in ['prediction','baseline_prediction','target_id','contrast_id']:
                assert np.array_equal(a[k],b[k])
    report={'passed':True,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'frozen_files_verified':len(frozen['files']),'screened':len(screen),'selected':len(selected),
            'selected_pair_errors':sum(r['selection_stratum']=='pair_error' for r in selected),
            'selected_pair_correct':sum(r['selection_stratum']=='pair_correct' for r in selected),
            'screen_population_errors':int((population[:,4]==0).sum()),
            'full_screen_pair_accuracy':float(population[:,4].mean()),
            'full_screen_top1_accuracy':float(population[:,3].mean()),
            'selection_priority_and_weights_independently_reconstructed':True,
            'metric_indices_verified_from_predictions_and_margins':True,
            'all_paired_point_estimates_independently_recomputed':True,
            'maximum_identity_error':identity_max,'maximum_allkept_error':allkept_max,
            'all_hook_locality_errors_zero':True,'replayed':len(replay_rows),
            'replay_discrete_predictions_exact':True,'replay_maximum_error':replay['maximum_error'],
            'analysis_sha256':sha(ROOT/'results/analysis.json')}
    (ROOT/'results/final-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

"""Audit completeness, numerical controls, frozen inputs, and real replay records."""
from common import *
from analyze_core import load_model,decompose
from heads import HEAD_MODELS
import numpy as np

def main():
    freeze_check();sf=read(ROOT/'data/supplement-freeze.json')
    assert all(sha(ROOT/p)==h for p,h in sf['files'].items())
    cf=read(ROOT/'data/cpu-assay-freeze.json')
    assert all(sha(ROOT/p)==h for p,h in cf['files'].items())
    cf2=read(ROOT/'data/cpu-scheduling-v2.json')
    assert all(sha(ROOT/p)==h for p,h in cf2['files'].items())
    cf3=read(ROOT/'data/cpu-scheduling-v3.json')
    assert all(sha(ROOT/p)==h for p,h in cf3['files'].items())
    rf=read(ROOT/'data/rank-audit-timing.json')
    assert all(sha(ROOT/p)==h for p,h in rf['files'].items())
    rank=read(ROOT/'results/analysis/rank-confidence.json')
    assert set(rank['models'])==set(PRIMARY_MODELS) and set(rank['head_models'])==set(HEAD_MODELS)
    for filename in ['bfloat16-batching-amendment.json','bfloat16-readout-amendment.json']:
        record=read(ROOT/'data'/filename)
        assert all(sha(ROOT/p)==h for p,h in record['files'].items())
    assert read(ROOT/'results/cpu-assay-status.json')['state']=='complete'
    results={'frozen_primary_inputs_unchanged':True,'frozen_supplement_scripts_unchanged':True,'models':{},'checkpoint_files_checked':0,'created_utc':utc()}
    for name,record in read(ROOT/'data/checkpoints.json').items():
        for fn,v in record['files'].items():
            assert sha(MODEL_PATHS[name]/fn)==v['sha256'],(name,fn)
            results['checkpoint_files_checked']+=1
    tasks=read(ROOT/'data/test.json');assert len(tasks)==288
    for split in ['calibration','development','test']:
        for row in read(ROOT/'data'/f'{split}.json'):
            for ps in row['encoded'].values():
                assert len(ps)==34
                assert len({p['good_id'] for p in ps})==len({p['bad_id'] for p in ps})==1
                assert ps[0]['good_id']!=ps[0]['bad_id']
                assert all(0<=p['anchor']<p['end']==len(p['ids'])-1 for p in ps)
    for cluster in read(ROOT/'data/blimp.json'):
        for model in cluster['members'][0]['encoded']:
            ps=[r['encoded'][model] for r in cluster['members']]
            assert len({p['good_id'] for p in ps})==len({p['bad_id'] for p in ps})==1
            assert ps[0]['good_id']!=ps[0]['bad_id']
    results['scoring_token_ids_and_controlled_anchor_order_checked']=True
    for model in PRIMARY_MODELS:
        root=ROOT/'results'/f'{model}-float32';rows,data,design=load_model(model)
        assert data['metrics'].shape==(288,8,74,5)
        assert data['baseline'].shape==(288,34,5)
        assert len(list((root/'test').glob('*.npz')))==288
        comp,closure=decompose(data,design);max_prob_close=0.;terminal_anchor=[0.,0.];terminal_end=[0.,0.]
        for v in comp.values():
            terminal_anchor=[max(terminal_anchor[j],float(abs(v['gap'][:,6,j]).max())) for j in [0,1]]
            terminal_end=[max(terminal_end[j],float(abs(v['prop'][:,7,j]).max())) for j in [0,1]]
            max_prob_close=max(max_prob_close,float(abs(v['gap'][:,:,1]-v['prop'][:,:,1]-v['norm'][:,:,1]-v['soft']).max()))
        assert terminal_anchor[0]<=.001 and terminal_anchor[1]<=.00002
        assert terminal_end[0]<=.001 and terminal_end[1]<=.00002
        assert max_prob_close<1e-12
        for key in ['metrics','baseline','endpoints']:
            v=data[key];assert np.array_equal(v[...,3],(v[...,0]>0).astype(float))
            assert np.max(abs(v[...,1]-np.exp(v[...,4])))<1e-12
            assert set(np.unique(v[...,2]))<={0.,1.}
        write_max=0.;identity=0.
        for row in rows:
            audits=read(root/'test'/(row['id']+'-audit.json'));assert len(audits)==8
            for a in audits:
                write_max=max(write_max,a['outside_max'],a['target_error']);identity=max(identity,a['identity_margin_error'])
        assert write_max==0 and identity<=.001
        replay=read(root/'replay-verification.json');ind=read(root/'independent-readout.json');cal=read(root/'calibration.json')
        assert replay['passed'] and ind['passed'] and cal['passed'] and ind['states_checked']==5328
        cpu_passed=read(root/'cpu-calibration.json')['passed']
        expected_device='cuda' if model=='pythia-410m' else 'cpu'
        assert read(root/'blimp-device.json')['device']==expected_device
        assert cpu_passed==(model!='pythia-410m')
        for stage,expected in [('test',170496),('replay',5328),('blimp',11160),('blimp-replay',720)]:
            assert read(root/(stage+'-completion.json'))['interventions']==expected
        assert read(root/'blimp-replay-verification.json')['passed']
        for stage in ['blimp','blimp-replay']:
            for file in (root/stage).glob('*.npz'):
                with np.load(file) as z:
                    for k in ['metrics','endpoints','baseline']:assert np.isfinite(z[k]).all()
            for file in (root/stage).glob('*-audit.json'):
                for a in read(file):assert a['outside_max']==a['target_error']==0
        if model in HEAD_MODELS:
            assert read(root/'heads-device.json')['device']=='cuda'
            assert read(root/'heads-replay-verification.json')['passed']
            for stage in ['heads','heads-replay']:
                for file in (root/stage).glob('*.npz'):
                    with np.load(file) as z:assert np.isfinite(z['metrics']).all() and np.isfinite(z['baseline']).all()
                for file in (root/stage).glob('*-audit.json'):
                    for a in read(file):assert a['outside_max']==a['target_error']==0 and a['identity_logit_error']<=.001
        assert read(ROOT/'tmp'/('statistics-reproduction-'+model)/'verification.json')['passed']
        assert read(ROOT/'results'/f'{model}-bfloat16/calibration.json')['passed']
        selection=read(ROOT/'results'/f'{model}-bfloat16/batching-selection.json')
        assert selection['passed'] and selection['readout_batch']==1
        assert selection['selected_batch']==read(ROOT/'results'/f'{model}-bfloat16/environment.json')['batch']
        assert read(ROOT/'results'/f'{model}-bfloat16/precision-completion.json')['groups']==9
        results['models'][model]={'groups':len(rows),'interventions':170496,'margin_closure_max':closure,
            'probability_closure_max':max_prob_close,'terminal_anchor_gap_max':terminal_anchor,'terminal_end_propagation_max':terminal_end,
            'write_locality_max':write_max,'identity_margin_max':identity,'replay_max':replay['maximum_error'],
            'independent_readout_passed':ind['passed'],'independent_readout_top1_changes':sum(v['target_top1_disagreements'] for v in ind['per_group'].values()),
            'statistical_recomputation_exact':True}
        print(model,'all checks passed',flush=True)
    assert read(ROOT/'tmp/reproduce-data/verification.json')['passed']
    scale=ROOT/'results/qwen3-1.7b-bfloat16'
    assert read(scale/'calibration.json')['passed'] and read(scale/'replay-verification.json')['passed']
    assert read(scale/'batching-selection.json')['passed']
    assert not read(ROOT/'results/gpt2-small-bfloat16/calibration-default-failed.json')['passed']
    assert not read(ROOT/'results/gpt2-small-bfloat16/calibration-before-readout-fix.json')['passed']
    assert read(scale/'test-completion.json')['interventions']==170496
    sr,sd,ss=load_model('qwen3-1.7b','test','bfloat16');assert len(sr)==288
    assert sd['metrics'].shape==(288,8,74,5)
    scale_identity=0.
    for row in sr:
        audits=read(scale/'test'/(row['id']+'-audit.json'));assert len(audits)==8
        for audit in audits:
            assert audit['outside_max']==audit['target_error']==0
            scale_identity=max(scale_identity,audit['identity_margin_error'])
    results['scale_write_locality_max']=0.
    results['scale_identity_margin_max_descriptive']=scale_identity
    results['scale_numerical_robustness_established']=False
    results['verification_scope']='Artifact integrity and prespecified primary checks pass; this does not mean every supplemental robustness diagnostic succeeded. The larger bfloat16 run retains material identity and structural-null errors.'
    assert read(ROOT/'tmp/reproduce-data-delivery-launcher/verification.json')['passed']
    results.update(passed=True,data_regeneration_exact=True,scale_replay_passed=True)
    dump(ROOT/'results/verification.json',results)
    print('Delivery audit passed.',flush=True)

if __name__=='__main__':main()

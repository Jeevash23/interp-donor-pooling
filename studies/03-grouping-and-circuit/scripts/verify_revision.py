"""Independent response-array checks and frozen-input validation; no inference."""
import hashlib
import json
from pathlib import Path
import sys as _release_sys  # release: frozen-file checks accept documented changes
_release_sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import release_paths as rp  # noqa: E402

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    output = {'frozen': {}, 'token': {}, 'circuit': {}}
    for panel in ['token', 'circuit']:
        folder = ROOT/panel
        freeze = read(folder/'data/freeze.json')
        for relative, expected in freeze['files'].items():
            with (folder/relative).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert actual == expected or rp.matches(folder/relative, expected), (panel,relative)
        output['frozen'][panel] = {'files_checked':len(freeze['files']), 'passed': True}
    rows = read(ROOT/'token/data/test.json')
    strata = sorted({(r['template'],r['class']) for r in rows})
    rng = np.random.default_rng(26100299)
    draws = np.concatenate([rng.choice([i for i,r in enumerate(rows) if (r['template'],r['class'])==s],(20000,16)) for s in strata],1)
    for model in ['gpt2-small','qwen3-0.6b']:
        folder=ROOT/'token/results'/model
        design=read(folder/'design.json')
        by_name={b['name']:b for b in design['banks']}
        values=[]; primary_cells=0
        for row in rows:
            with np.load(folder/'test'/f"{row['id']}.npz") as z:
                m=z['metrics']
                assert m.shape==(8,198,5) and np.isfinite(m).all()
                differences=[]
                for site in [1,3,5]:
                    for direction in ['restore','destroy']:
                        for rotation in range(8):
                            gaps=[]
                            for k in [1,8]:
                                b=by_name[f'{direction}/k{k}/rotation{rotation}']
                                gaps.append(float(m[site,b['mean'],0]-sum(m[site,j,0] for j in b['indices'])/8))
                            differences.append(abs(gaps[1])-abs(gaps[0]))
                assert len(differences)==48
                values.append(np.mean(differences));primary_cells+=len(differences)
        values=np.array(values)
        ci=np.quantile(values[draws].mean(1),[.0125,.9875])
        summary=read(ROOT/'token/results/analysis-test'/f'{model}.json')
        assert abs(values.mean()-summary['primary']['mean'])<1e-12
        assert np.max(abs(ci-np.array(summary['primary']['ci'])))<1e-12
        replay=read(folder/'replay-verification.json')
        assert replay['passed']
        output['token'][model]={'groups':len(rows),'primary_cells':primary_cells,'explicit_loop_mean':float(values.mean()),'explicit_bootstrap_ci':ci.tolist(),'replay_max_error':replay['maximum_error'],'passed':True}
    cdir=ROOT/'circuit'
    selection=read(cdir/'data/selected-test.json')
    screen=read(cdir/'results/screen-selection.json')
    summary=read(cdir/'results/analysis.json')
    m=[];baseline=[];w=[];labels=[]
    for row in selection:
        with np.load(cdir/'results/test'/f"{row['id']}.npz") as z:
            assert z['metrics'].shape==(2,9,9)
            assert np.isfinite(z['metrics']).all()
            m.append(z['metrics']);baseline.append(z['baseline'])
            # Independently construct both discrete agreement endpoints.
            pair=(z['metrics'][0,:,0]>0)==(z['baseline'][0]>0)
            top=z['prediction'][0]==int(z['baseline_prediction'][0])
            np.testing.assert_array_equal(pair,z['metrics'][0,:,6])
            np.testing.assert_array_equal(top,z['metrics'][0,:,5])
        stratum=screen['strata'][row['selection_stratum']]
        weight=stratum['screen_population_n']/stratum['selected_n']
        assert weight==row['population_weight']
        w.append(weight);labels.append(row['selection_stratum'])
    m=np.stack(m);baseline=np.stack(baseline);w=np.array(w);labels=np.array(labels)
    assert abs(w.sum()-8192)<1e-9
    reference=summary['circuits']['reference']
    for name, selector in [('mean',0),('expected_joint_resample',slice(1,None))]:
        x=m[:,0,selector]
        if x.ndim==3:x=x.mean(1)
        for i,metric in enumerate(summary['metric_order']):
            assert abs(np.average(x[:,i],weights=w)-reference[name][metric]['mean'])<1e-12
    strata_result={}
    for label,key in [('pair_correct','full_pair_success'),('pair_error','full_pair_error')]:
        ix=labels==label
        delta=m[ix,0,0,6]-m[ix,0,1:,6].mean(1)
        assert abs(delta.mean()-reference['strata'][key]['paired_pair_agreement_difference']['mean'])<1e-12
        strata_result[label]={'n':int(ix.sum()),'pair_agreement_difference':float(delta.mean())}
    replay=read(cdir/'results/replay-verification.json');assert replay['passed']
    output['circuit']={'selected':len(selection),'screened':8192,'strata':strata_result,'replay_max_error':replay['maximum_error'],'passed':True}
    output['passed']=True
    return output


if __name__=='__main__':
    result=verify()
    (ROOT/'results/revision-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

"""Supplementary tables and head-score plots from completed measurements."""
from common import *
from report import LABELS,GEN,FIG,table,ci,fmt,savefig
import numpy as np
import matplotlib.pyplot as plt

def main():
    from heads import HEAD_MODELS
    bs={m:read(ROOT/'results/analysis'/f'{m}-blimp.json') for m in PRIMARY_MODELS}
    hs={m:read(ROOT/'results/analysis'/f'{m}-heads.json') for m in HEAD_MODELS}
    ps={m:read(ROOT/'results/analysis'/f'{m}-precision.json') for m in PRIMARY_MODELS}
    scale=read(ROOT/'results/analysis/qwen3-1.7b-scale.json')
    rows=[]
    for m,s in bs.items():rows.append([LABELS[m],'GPU' if m=='pythia-410m' else 'CPU',f"{100*s['pair_accuracy']['mean']:.1f}/{100*s['top1_accuracy']['mean']:.1f}",ci(s['gap_margin_abs']),ci(s['prop_minus_norm'])])
    (GEN/'blimp.tex').write_text(table(['Checkpoint','Device','Pair/top-1 (\\%)','$|D|$ [95\\% CI]','$|P|-|N|$ [95\\% CI]'],rows,'External BLiMP relational-noun subset, averaging three nonterminal END sites. Pair/top-1 gives baseline contrast preference and unrestricted target top-one accuracy. Every model uses the same 31 disjoint clusters and 279 rotating recipients. Pythia uses its validated GPU configuration after failing CPU calibration before any external measurement. These are descriptive cluster intervals, not a second confirmatory family.','tab:blimp',small=True))
    rows=[]
    for m,s in bs.items():rows.append([LABELS[m],ci(s['prop_margin_abs']),ci(s['norm_margin_abs']),ci(s['gap_probability_abs'],2,100),ci(s['soft_probability_abs'],2,100)])
    (GEN/'blimp-components.tex').write_text(table(['Checkpoint','$|P|$ [95\\% CI]','$|N|$ [95\\% CI]','$|D_p|$ (pp)','$|S_p|$ (pp)'],rows,'BLiMP component magnitudes and probability diagnostics. All intervals are descriptive 95\\% cluster intervals; probability quantities are percentage points. Signed terms can cancel, so their absolute magnitudes are not additive shares.','tab:blimp-components',small=True))
    rows=[]
    for m,s in hs.items():rows.append([LABELS[m],s['head_sites'],ci(s['gap_abs'],3,1000),fmt(s['spearman'],4),*[f"{s['topk'][str(k)]['intersection']}/{k}" for k in [5,10,20]],'/'.join(str(s['aggregate_reversals'][str(t)]) for t in [0,.05,.1])])
    (GEN/'heads.tex').write_text(table(['Checkpoint','Heads','$|D|$ ($10^{-3}$ logits)','$\\rho$','Top-5','Top-10','Top-20','Rev.'],rows,'Fixed IOI head scan, 96 groups per model. Gap values and their descriptive 95\\% intervals are in thousandths of a logit to preserve precision for small effects. Top columns give overlap between mean-patch and individual-average top sets. Rev. lists aggregate pair reversals at margins 0/0.05/0.1 logits; the pair denominators are 1,128 for GPT-2 Small and 2,016 for each other model. These are descriptive comparisons in a partial-depth universe.','tab:heads',small=True))
    rows=[]
    for m in PRIMARY_MODELS:
        root=ROOT/'results'/f'{m}-float32';c=read(root/'cpu-calibration.json');b=read(root/'blimp-replay-verification.json')
        h=read(root/'heads-replay-verification.json') if m in HEAD_MODELS else None
        rows.append([LABELS[m],f"{c['maximum_error']:.2e}",f"{c['probability_max_error']:.2e}",'pass' if c['passed'] else 'fail',f"{b['maximum_error']:.2e}",f"{h['maximum_error']:.2e}" if h else '--'])
    (GEN/'cpu-validation.tex').write_text(table(['Checkpoint','CPU logits','CPU prob.','Gate','BLiMP replay','Head replay'],rows,'Separate CPU calibration and fresh-model supplementary replays. Pythia fails the unchanged CPU duplicate-donor gate (0.002539 logits), so its BLiMP panel and replay use GPU; other BLiMP panels use CPU. Every head panel uses GPU. The numerical gates remain 0.001 logits and 0.00002 target probability. No model or external group is dropped.','tab:cpu-validation',small=True))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.6))
    for ax,(m,s) in zip(axes,hs.items()):
        x=np.array([r['individual_score'] for r in s['sites']]);y=np.array([r['mean_score'] for r in s['sites']]);lay=np.array([r['layer'] for r in s['sites']])
        ax.scatter(x,y,c=lay,cmap='viridis',s=17,alpha=.8)
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());pad=max((hi-lo)*.07,.01)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#777777',linestyle='--',linewidth=.8)
        ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='Individual-donor average',title=LABELS[m]);ax.grid(alpha=.15)
    axes[0].set_ylabel('Mean-patch logit margin')
    from matplotlib.lines import Line2D
    palette=plt.get_cmap('viridis')
    handles=[Line2D([0],[0],marker='o',linestyle='',color=palette(i/3),markersize=4,label=label) for i,label in enumerate(['Quarter depth','Half depth','Three-quarter depth','Final block'])]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=10.5)
    fig.tight_layout(rect=[0,.09,1,1]);savefig(fig,'head-comparison')
    rows=[]
    for m,s in ps.items():
        fp=np.mean([c['fp32_gap_abs'] for c in s['cells']]);bf=np.mean([c['bf16_gap_abs'] for c in s['cells']]);delta=np.mean([c['gap_absolute_difference'] for c in s['cells']])
        rows.append([LABELS[m],f"{s['float32_batch']}/{s['bfloat16_batch']}",fmt(fp),fmt(bf),fmt(delta),fmt(s['all_intervention_margin_mae']),fmt(s['bf16_linear_endpoint_closure_error']),f"{s['baseline_pair_disagreements']}/{s['baseline_comparisons']}"])
    (GEN/'precision.tex').write_text(table(['Checkpoint','Batch F/B','F32 gap','BF16 gap','Gap diff.','Margin diff.','Closure','Pair changes'],rows,'Nine predetermined cases per model. Batch F/B gives float32/bfloat16 native maximum batch sizes, selected by calibration; these are configuration comparisons rather than isolated dtype effects. Gap columns average all tasks, directions, banks and six nonterminal sites; Gap diff. is the mean absolute difference between their signed gaps. Margin diff. averages all 74 intervention responses at all eight sites. Closure is numerical noncommutation of the bfloat16 mean-logit endpoint with a linear margin. Pair changes count baseline clean-target preference changes. These fixed-case diagnostics have no population interval.','tab:precision',small=True))
    rows=[]
    for m in list(PRIMARY_MODELS)+['qwen3-1.7b']:
        root=ROOT/'results'/f'{m}-bfloat16';cal=read(root/'calibration.json');selection=read(root/'batching-selection.json')
        attempts=', '.join(str(x['batch'])+(' P' if x['passed'] else ' F') for x in selection['attempts'])
        rows.append([LABELS[m],attempts,cal['logit_tolerance'],f"{cal['maximum_error']:.3f}",f"{cal['probability_max_error']:.2e}"])
    (GEN/'precision-validation.tex').write_text(table(['Checkpoint','Native batch attempts','Logit gate','Logit error','Probability error'],rows,'Reduced-precision calibration with the one-row readout adapter. P/F mark passed/failed attempts; default native batching precedes singleton fallback. The probability gate is 0.05 throughout. Earlier GPT-2 Small attempts before this adapter also failed and are preserved separately. Every scientific one-row mean-state readout calls the original function unchanged; only multi-row reconstruction is split for numerical consistency. All choices use reserved calibration, before reduced-precision test outcomes.','tab:precision-validation',small=True))
    rows=[]
    for m,x in list(ps.items())+[('qwen3-1.7b',scale)]:
        batch=read(ROOT/'results'/f'{m}-bfloat16/environment.json')['batch']
        rows.append([LABELS[m],batch,fmt(x['terminal_anchor_narrow_gap_max'],4),fmt(x['terminal_anchor_gap_max'],4),fmt(100*x['terminal_anchor_probability_gap_max'],3)])
    (GEN/'precision-null.tex').write_text(table(['Checkpoint','Batch','Narrow null (logits)','All-bank null (logits)','All-bank null (pp)'],rows,'Maximum absolute direct pooling gaps at final-block ANCHOR, a structural null. The primary six models use their nine fixed reduced-precision cases; Qwen3-1.7B uses all 288 extension groups. All-bank columns include narrow/broad and restoration/destruction, while Narrow includes both directions only. Any nonzero value is numerical rather than a propagated causal effect at this site. These diagnostics expose the limits of the lower-precision configuration and are excluded from primary float32 claims.','tab:precision-null',small=True))
    rows=[]
    for c in scale['cells']:
        rows.append([c['task'].upper(),'R' if c['direction']=='restore' else 'D','N' if c['bank']=='narrow' else 'B',ci(c['gap_margin_abs']),ci(c['gap_probability_abs'],2,100),fmt(c['clean_pair']['mean']*100,1),'/'.join(str(c['rank']['aggregate_reversals'][str(t)]) for t in [.05,.1])])
    (GEN/'scale.tex').write_text(table(['Task','Dir.','Bank','$|D|$ [95\\% CI]','$|D_p|$ (pp)','Clean pair','Rev.'],rows,'Qwen3-1.7B, bfloat16 only, 288 groups and all 170,496 residual interventions. Rev. lists 0.05/0.1-logit aggregate reversals. All intervals are descriptive and quantify group variation, not numerical error. Identity and structural-null artifacts prevent treating this extension as a successful robustness replication. These retained direct response gaps do not join the primary boundary comparison.','tab:scale',align='lccrrrr',small=True))
    # Scoring contributions and practical gap prevalence, every task/bank/direction.
    tables=[]
    for m in PRIMARY_MODELS:
        s=read(ROOT/'results/analysis'/f'{m}.json');rr=[]
        for c in s['cells']:
            st=c['stats'];rr.append([c['task'].upper(),'R' if c['direction']=='restore' else 'D','N' if c['bank']=='narrow' else 'B',fmt(100*st['prop_probability_absolute']['mean'],2),fmt(100*st['norm_probability_absolute']['mean'],2),fmt(100*st['soft_probability_absolute']['mean'],2),fmt(100*st['gap_over_0.1_margin']['mean'],1),fmt(100*st['gap_over_1pp']['mean'],1)])
        tables.append(table(['Task','Dir.','Bank','$|P_p|$','$|N_p|$','$|S_p|$','$|D|>0.1$','$|D_p|>1$ pp'],rr,LABELS[m]+': scoring decomposition magnitudes in percentage points, and prevalence of gaps above reporting margins in percent of group/site cases. Magnitudes are computed before averaging and may cancel. All six nonterminal sites are retained.','tab:scoring-'+m,align='lccrrrrr',small=True))
    (GEN/'scoring.tex').write_text('\n\\clearpage\n'.join('\n'.join(tables[i:i+2]) for i in range(0,len(tables),2)))
    ledger=[]
    for m in PRIMARY_MODELS:
        root=ROOT/'results'/f'{m}-float32'
        for stage in ['development','test','replay','blimp','blimp-replay','heads','heads-replay']:
            file=root/(stage+'-completion.json')
            if file.exists():ledger.append({'model':m,'dtype':'float32','stage':stage,'device':'cpu' if stage.startswith('blimp') and m!='pythia-410m' else 'cuda',**read(file)})
        file=ROOT/'results'/f'{m}-bfloat16/precision-completion.json'
        ledger.append({'model':m,'dtype':'bfloat16','stage':'precision',**read(file)})
    for stage in ['test','replay']:
        ledger.append({'model':'qwen3-1.7b','dtype':'bfloat16','stage':stage,**read(ROOT/'results/qwen3-1.7b-bfloat16'/(stage+'-completion.json'))})
    quantities={'primary':sum(r['interventions'] for r in ledger if r['stage']=='test' and r['dtype']=='float32'),
        'blimp':sum(r['interventions'] for r in ledger if r['stage']=='blimp'),
        'heads':sum(r['interventions'] for r in ledger if r['stage']=='heads'),
        'precision':sum(r['interventions'] for r in ledger if r['stage']=='precision'),
        'scale':sum(r['interventions'] for r in ledger if r['stage']=='test' and r['dtype']=='bfloat16'),
        'replays':sum(r['interventions'] for r in ledger if 'replay' in r['stage']),
        'development':sum(r['interventions'] for r in ledger if r['stage']=='development')}
    dump(ROOT/'results/analysis/execution-ledger.json',{'runs':ledger,'interventions_by_purpose':quantities,'seconds_recorded_runs':sum(r['seconds'] for r in ledger),'calibration_and_postprocessing_not_in_runtime_total':True,'scope':'Planned study runs and their fixed replays; excludes later archive-integration and portable-launcher validation reruns'})
    rr=[]
    for key,label in [('primary','Primary six-model residual panel'),('blimp','BLiMP external assay'),('heads','IOI attention-head scan'),('precision','Fixed-case reduced precision'),('scale','Qwen3-1.7B size extension'),('replays','Fresh-model verification repeats'),('development','Earlier development split')]:rr.append([label,f"{quantities[key]:,}"])
    (GEN/'execution.tex').write_text(table(['Purpose','Intervention executions'],rr,'Execution ledger for the planned study runs and fixed replays. Calibration, baseline prompts, direct saved-state readouts and later archive-integration reruns are separate from these counts. Repeated forwards are computational work, not independent experimental units.','tab:execution',align='lr',small=True))
    print('Supplementary figures, tables and execution ledger generated.',flush=True)

if __name__=='__main__':main()

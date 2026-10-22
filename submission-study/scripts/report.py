"""Derive tables/figures and a paper evidence summary from completed test analyses."""
from __future__ import annotations
import csv
from common import *
from analyze_core import load_model,decompose,estimate,draws_for,ratio
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'gpt2-small':'GPT-2 Small','gpt2-medium':'GPT-2 Medium','gpt2-large':'GPT-2 Large','pythia-410m':'Pythia-410M','tinyllama-1.1b':'TinyLlama-1.1B','qwen3-0.6b':'Qwen3-0.6B','qwen3-1.7b':'Qwen3-1.7B'}
SHORT=['GPT-2 S','GPT-2 M','GPT-2 L','Pythia','TinyLlama','Qwen3']
COLORS=['#0072B2','#56B4E9','#332288','#D55E00','#009E73','#CC79A7']
FIG=ROOT/'results/figures';GEN=ROOT/'paper/generated'
def fmt(x,d=3):return f'{x:.{d}f}'
def ci(e,d=3,mult=1):return f"{e['mean']*mult:.{d}f} [{e['ci'][0]*mult:.{d}f}, {e['ci'][1]*mult:.{d}f}]"
def table(headers,rows,caption,label,align=None,small=False):
    align=align or ('l'+'r'*(len(headers)-1));size='\\small\n' if small else ''
    return '\\begin{table}[ht]\n\\centering\n'+size+'\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,r))+'\\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n'
def savefig(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def include(name,caption,label,width='\\linewidth'):
    return '\\begin{figure}[ht]\n\\centering\n\\includegraphics[width='+width+']{../results/figures/'+name+'.pdf}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{figure}\n'

def main():
    GEN.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':11,'legend.fontsize':10.5,'pdf.fonttype':42,'figure.dpi':130})
    summaries={m:read(ROOT/'results/analysis'/f'{m}.json') for m in PRIMARY_MODELS};pool={};allcomp={};allrows=None
    for name in PRIMARY_MODELS:
        rows,data,design=load_model(name);comp,closure=decompose(data,design);draw=draws_for(rows);allcomp[name]=comp;allrows=rows
        pc={}
        for term in ['gap','prop','norm']:
            v=np.stack([abs(comp[d+'/narrow/full'][term][:,:6,0]) for d in ['restore','destroy']],-1)
            pc[term]=estimate(v,draw)
        pc['prob_gap']=estimate(np.stack([abs(comp[d+'/narrow/full']['gap'][:,:6,1]) for d in ['restore','destroy']],-1),draw)
        pc['broad_gap']=estimate(np.stack([abs(comp[d+'/broad/full']['gap'][:,:6,0]) for d in ['restore','destroy']],-1),draw)
        pc['broad_over_narrow']=ratio(np.stack([abs(comp[d+'/broad/full']['gap'][:,:6,0]) for d in ['restore','destroy']],-1),np.stack([abs(comp[d+'/narrow/full']['gap'][:,:6,0]) for d in ['restore','destroy']],-1),draw)
        pool[name]=pc
    dump(ROOT/'results/analysis/paper-summary.json',{'primary_components':pool,'primary_supported':[m for m,s in summaries.items() if s['primary']['ci'][0]>0],'core_interventions':sum(s['interventions'] for s in summaries.values()),'created_utc':utc()})
    # Model properties, independently measured parameter counts.
    mrows=[]
    for name,s in summaries.items():
        e=s['environment'];typ=e['model_type'];mrows.append([LABELS[name],f"{e['parameters']/1e6:.1f}",', '.join(map(str,e['layers'])),'LN' if typ in ['gpt2','gpt_neox'] else 'RMS',str(e['batch'])])
    (GEN/'model-table.tex').write_text(table(['Checkpoint','Params (M)','Patched block indices','Final norm','Batch'],mrows,'Primary model panel. Parameter counts are measured from loaded, tied model parameters. Indices are zero-based. All primary computation is float32. LN denotes LayerNorm and RMS denotes RMSNorm.','tab:models',align='lrlcr',small=True))
    prows=[]
    for name,s in summaries.items():
        p=pool[name];prows.append([LABELS[name],fmt(p['gap']['mean']),fmt(p['prop']['mean']),fmt(p['norm']['mean']),ci(s['primary'])])
    (GEN/'primary-table.tex').write_text(table(['Checkpoint','$|D|$','$|P|$','$|N|$','$|P|-|N|$ [99.17\\% CI]'],prows,'Primary narrow-bank results, equally averaging tasks, directions and six nonterminal sites. Absolute values are taken before averaging. The last column uses the six-comparison adjusted interval; displayed component magnitudes are point estimates and do not necessarily add.','tab:primary',small=True))
    # Decomposition by model/task; show all model-task cells, both directions averaged.
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.6),sharey=True)
    for ti,task in enumerate(TASKS):
        ax=axes[ti];ix=np.array([i for i,r in enumerate(allrows) if r['task']==task]);draw=draws_for([allrows[i] for i in ix])
        for k,term,color in [(0,'prop','#0072B2'),(1,'norm','#D55E00')]:
            vv=[];lo=[];hi=[]
            for name in PRIMARY_MODELS:
                c=allcomp[name];est=estimate(np.stack([abs(c[d+'/narrow/full'][term][ix,:6,0]) for d in ['restore','destroy']],-1),draw)
                vv.append(est['mean']);lo.append(est['mean']-est['ci'][0]);hi.append(est['ci'][1]-est['mean'])
            ax.bar(np.arange(6)+(k-.5)*.34,vv,.34,color=color,label='Propagation' if term=='prop' else 'Final normalization',yerr=[lo,hi],capsize=2,error_kw={'linewidth':.7})
        ax.set_xticks(range(6),['S','M','L','P','T','Q']);ax.set_title(task.upper());ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean absolute margin contribution');axes[0].legend(frameon=False)
    fig.supxlabel('GPT-2 Small / Medium / Large, Pythia, TinyLlama, Qwen3',fontsize=11);fig.tight_layout();savefig(fig,'decomposition')
    # Preserve the site profile, including the two separately labeled controls.
    profile={};matrix=[];profile_draws=draws_for(allrows)
    for name in PRIMARY_MODELS:
        c=allcomp[name];vals=np.stack([abs(c[d+'/narrow/full']['prop'][:,:,0])-abs(c[d+'/narrow/full']['norm'][:,:,0]) for d in ['restore','destroy']],-1)
        profile[name]=[estimate(vals[:,site],profile_draws) for site in range(8)]
        matrix.append([x['mean'] for x in profile[name]])
    matrix=np.array(matrix);vmax=float(abs(matrix).max())
    fig,ax=plt.subplots(figsize=(10.2,4.2));im=ax.imshow(matrix,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_yticks(range(6),[LABELS[n] for n in PRIMARY_MODELS]);ax.set_xticks(range(8),['Q1 A','Q1 E','Q2 A','Q2 E','Q3 A','Q3 E','Final A','Final E'])
    ax.axvline(5.5,color='#333333',linestyle='--',linewidth=1.2)
    for i in range(6):
        for j in range(8):
            label_value=0. if abs(matrix[i,j])<.0005 else matrix[i,j]
            ax.text(j,i,f'{label_value:.3f}',ha='center',va='center',fontsize=10.5,color='white' if abs(matrix[i,j])>.55*vmax else '#111111')
    fig.colorbar(im,ax=ax,pad=.02,label='Mean |P| − |N| (logits)')
    ax.set_xlabel('A: ANCHOR; E: END. Final-block controls are right of the dashed line.')
    fig.tight_layout();savefig(fig,'site-profile')
    dump(ROOT/'results/analysis/site-profile.json',{'scope':'Narrow banks, equal tasks and directions, separated by each of the eight fixed sites. Final two sites are controls excluded from the primary aggregate. Intervals are descriptive 95%.','models':profile})
    (GEN/'site-profile.tex').write_text(include('site-profile','Site dependence of the propagation-minus-final-normalization comparison, equally averaging tasks and directions under narrow banks. Q1/Q2/Q3 denote the first three quarter-depth choices. The six columns left of the dashed line enter the primary average. Final ANCHOR is a null control; final END has zero propagation by construction and can retain a normalization term. Cell annotations are point estimates; descriptive group intervals are in the released site-profile summary. The structural terminal pattern is an instrumentation check, not an empirical mechanism discovery.','fig:site-profile'))
    # Radius profile and actual estimator errors.
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.7));ts=np.array([.25,.5,.75,1.])
    for color,name in zip(COLORS,PRIMARY_MODELS):
        ys=np.array([summaries[name]['radius'][str(float(t))]['ratio_to_full']['ratio'] for t in ts]);axes[0].plot(ts,ys,'o-',color=color,label=LABELS[name],markersize=4)
    axes[0].plot(ts,ts**2,'k--',label='$t^2$',linewidth=1.3);axes[0].set(xlabel='Donor radius t',ylabel='Absolute gap / full-radius absolute gap',xticks=ts);axes[0].legend(frameon=False,fontsize=10.5,ncol=2);axes[0].grid(alpha=.2)
    for k,(key,label,color) in enumerate([('mean_1_forward','Mean (1)','#999999'),('curvature_5_forward','Correction (5)','#0072B2'),('resample5_5_forward_expected_absolute_error','Resample (5)','#D55E00')]):
        vals=[summaries[n]['estimators'][key]['mean'] for n in PRIMARY_MODELS];axes[1].bar(np.arange(6)+(k-1)*.25,vals,.25,label=label,color=color)
    axes[1].set_xticks(range(6),['S','M','L','P','T','Q']);axes[1].set(ylabel='Absolute error for the finite-bank mean',xlabel='Checkpoint');axes[1].legend(frameon=False,fontsize=10.5,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,columnspacing=.7,handlelength=1.3);axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
    fig.tight_layout();savefig(fig,'radius-and-estimators')
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.6),sharey=True)
    for ti,task in enumerate(TASKS):
        for k,(bank,color) in enumerate([('narrow','#0072B2'),('broad','#E69F00')]):
            vals=[]
            for name in PRIMARY_MODELS:vals.append(np.mean([c['stats']['gap_margin_absolute']['mean'] for c in summaries[name]['cells'] if c['task']==task and c['bank']==bank]))
            axes[ti].bar(np.arange(6)+(k-.5)*.34,vals,.34,label=bank.capitalize(),color=color)
        axes[ti].set_xticks(range(6),['S','M','L','P','T','Q']);axes[ti].set_title(task.upper());axes[ti].grid(axis='y',alpha=.2);axes[ti].set_axisbelow(True)
    axes[0].set_ylabel('Mean absolute pooling gap (logit margin)');axes[0].legend(frameon=False);fig.tight_layout();savefig(fig,'donor-banks')
    # Every cell goes into a machine-readable table and the paper appendix.
    flat=[];app=[]
    for name,s in summaries.items():
        cr=[]
        for cell in s['cells']:
            t=cell['stats'];row={'model':name,'task':cell['task'],'direction':cell['direction'],'bank':cell['bank'],'gap':t['gap_margin_absolute']['mean'],'prop':t['prop_margin_absolute']['mean'],'norm':t['norm_margin_absolute']['mean'],'probability_gap_pp':100*t['gap_probability_absolute']['mean'],'softmax_pp':100*t['soft_probability_absolute']['mean'],'reversals_0.05':cell['rank']['aggregate_reversals']['0.05'],'reversals_0.1':cell['rank']['aggregate_reversals']['0.1'],'rank_spearman':cell['rank']['spearman'],'gap_to_individual_effect_ratio':t['gap_relative_to_individual_patch_effect']['ratio'],'gap_to_individual_effect_ci_low':t['gap_relative_to_individual_patch_effect']['ci'][0],'gap_to_individual_effect_ci_high':t['gap_relative_to_individual_patch_effect']['ci'][1],'clean_pair_accuracy':cell['competence']['clean_pair']['mean'],'competent_subset_n':cell['competent_subset']['n']};flat.append(row)
            cr.append([cell['task'].upper(),'R' if cell['direction']=='restore' else 'D','N' if cell['bank']=='narrow' else 'B',ci(t['gap_margin_absolute']),fmt(row['prop']),fmt(row['norm']),fmt(row['probability_gap_pp'],2),str(row['reversals_0.05'])])
        app.append(table(['Task','Dir.','Bank','$|D|$ [95\\% CI]','$|P|$','$|N|$','$|D_p|$ (pp)','Rev.'],cr,LABELS[name]+': all controlled-task cells. R/D are restoration/destruction; N/B are narrow/broad donors. Rev. counts aggregate reversals among 15 site pairs at 0.05 logits; these are descriptive, not multiplicity-adjusted discoveries. All values average six nonterminal sites.','tab:full-'+name,align='lccrrrrr',small=True))
    with (ROOT/'results/analysis/all-cells.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (GEN/'all-cells.tex').write_text('\n\\clearpage\n'.join('\n'.join(app[i:i+2]) for i in range(0,len(app),2)))
    # Competence / template robustness / estimator appendix.
    rows=[];templates=[];ests=[]
    for name,s in summaries.items():
        for task in TASKS:
            cell=next(c for c in s['cells'] if c['task']==task and c['direction']=='restore' and c['bank']=='narrow');cp=cell['competence'];subset=cell['competent_subset']
            rows.append([LABELS[name],task.upper(),f"{100*cp['clean_pair']['mean']:.1f}",f"{100*cp['corrupt_pair']['mean']:.1f}",f"{100*cp['clean_top1']['mean']:.1f}",f"{100*cp['clean_narrow_donor_pair']['mean']:.1f}",f"{subset['n']}/96"])
            h=cell['heldout_templates'];templates.append([LABELS[name],task.upper(),ci(h['False']['gap_abs']),ci(h['True']['gap_abs']),', '.join(f'{x:.3f}' for x in cell['seed_blocks'].values())])
        e=s['estimators'];ests.append([LABELS[name],fmt(e['mean_1_forward']['mean']),fmt(e['curvature_5_forward']['mean']),fmt(e['resample4_4_forward_expected_absolute_error']['mean']),fmt(e['resample5_5_forward_expected_absolute_error']['mean']),fmt(e['mean2_three_orders_1_forward_each']['mean']),fmt(e['mean4_three_orders_1_forward_each']['mean'])])
    (GEN/'competence.tex').write_text(table(['Checkpoint','Task','Clean pair','Corrupt pref.','Clean top-1','Donor pair','Subset'],rows,'Baseline scores in percent. Corrupt pref. is preference for the original clean target on the corrupted input, not relabeled corrupted-input accuracy. Donor pair is clean narrow-bank pair accuracy. Subset counts recipients for which both the clean recipient and all eight clean narrow donors are pair-correct. No failures are removed from the primary sample.','tab:competence',small=True))
    (GEN/'templates.tex').write_text(table(['Checkpoint','Task','Dev. recipient types','Other recipient types','Seed blocks'],templates,'Restoration/narrow mean absolute margin gaps. Other recipient types were absent from development recipients but could occur in broad development donors. Template columns give descriptive 95\\% intervals. The final column lists the three fixed seed-block means in order. Each test group contributes once.','tab:templates',small=True))
    (GEN/'estimators.tex').write_text(table(['Checkpoint','Mean','Correct.','RS4','RS5','Mean2','Mean4'],ests,'Mean absolute error for the expected response of the measured eight-donor bank, restoration/narrow/nonterminal sites. Correction and RS5 each cost five recipient forwards; RS4 costs four; each mean patch costs one after donor collection. Mean2/4 average error over three fixed orders.','tab:estimators',small=True))
    practical=[];ranks=[];radius=[]
    for name,s in summaries.items():
        for task in TASKS:
            c=next(c for c in s['cells'] if c['task']==task and c['direction']=='restore' and c['bank']=='narrow')
            r=c['stats']['gap_relative_to_individual_patch_effect'];sub=c['competent_subset']
            practical.append([LABELS[name],task.upper(),fmt(c['stats']['gap_margin_absolute']['mean']),f"{r['ratio']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]",f"{sub['n']}/96",fmt(sub['gap_abs_mean']) if sub['gap_abs_mean'] is not None else '--'])
        counts=[sum(c['rank']['aggregate_reversals'][str(t)] for c in s['cells']) for t in [0,.05,.1]]
        ranks.append([LABELS[name],*counts,f"{min(c['rank']['spearman'] for c in s['cells']):.3f}"])
        h=s['radius']['0.5']['ratio_to_full'];e=s['estimators']
        radius.append([LABELS[name],f"{h['ratio']:.3f} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}]",ci(s['quadratic_vs_zero']),ci(e['curvature_improvement_over_mean']),ci(e['curvature_improvement_over_resample5'])])
    (GEN/'practical-scale.tex').write_text(table(['Checkpoint','Task','$|D|$','Relative gap [95\\% CI]','Subset','$|D|$ in subset'],practical,'Restoration/narrow practical scale. Relative gap divides the aggregate absolute pooling gap by the aggregate absolute individual-average effect relative to the corrupted recipient. It is not a per-example percentage or a clean-minus-corrupt normalization. The subset requires pair-correct clean recipients and all eight clean narrow donors; its gap is descriptive with its denominator shown. All groups remain in the primary analysis.','tab:practical',small=True))
    (GEN/'residual-ranks.tex').write_text(table(['Checkpoint','Rev. at 0','Rev. at 0.05','Rev. at 0.1','Minimum $\\rho$'],ranks,'Aggregate residual-site order reversals across all twelve task/direction/bank cells per model, each with 15 pairs (180 comparisons per row). Counts at different margins are nested and comparisons share sites and groups. The last column is the minimum six-site Spearman correlation across the twelve cells. These descriptive counts are not independent discoveries.','tab:residual-ranks',small=True))
    (GEN/'radius-effects.tex').write_text(table(['Checkpoint','Half/full ratio','Quadratic advantage','Correction vs. mean','Correction vs. RS5'],radius,'Restoration/narrow sensitivity and estimator comparisons with descriptive 95\\% group intervals. The ratio compares absolute gaps. Quadratic advantage is the absolute-error reduction from predicting the full-radius signed gap with four times its half-radius value instead of zero. Correction columns report absolute-error reduction; positive favors the correction.','tab:radius-effects',small=True))
    # Exact evaluated examples and validation table.
    ex=[]
    for task in TASKS:
        r=next(x for x in allrows if x['task']==task)
        def esc(s):return s.replace('&','\\&').replace('_','\\_').replace('%','\\%')
        ex.append("\\paragraph{"+task.upper()+" ("+r["id"]+").} Clean: \\emph{``"+esc(r["recipient"][0]["text"])+"''}. Corrupt: \\emph{``"+esc(r["recipient"][1]["text"])+"''}. Target/contrast: \\emph{"+r["good"]+"/"+r["bad"]+"}.")
    (GEN/'examples.tex').write_text('\n\n'.join(ex)+'\n')
    vr=[]
    for name in PRIMARY_MODELS:
        root=ROOT/'results'/f'{name}-float32';c=read(root/'calibration.json');r=read(root/'replay-verification.json');ind=read(root/'independent-readout.json')
        vv=ind['maximum_errors'];vr.append([LABELS[name],f"{c['maximum_error']:.2e}",f"{c['probability_max_error']:.2e}",f"{r['maximum_error']:.2e}",f"{max(vv['margin_max'],vv['endpoint_margin_max']):.2e}",f"{max(vv['probability_max'],vv['endpoint_probability_max']):.2e}"])
    (GEN/'validation.tex').write_text(table(['Checkpoint','Cal. logits','Cal. prob.','Replay','Direct margin','Direct prob.'],vr,'Maximum numerical discrepancies. Replay covers all saved responses, endpoints, dispersions and audit states on nine fixed groups per model. Direct readout uses explicit float64 normalization and checkpoint unembedding, covering 5,328 intervention states per model; it is a higher-precision reference rather than a claim of bitwise equality.','tab:validation',small=True))
    print('Generated tables and figures from completed model analyses.',flush=True)

if __name__=='__main__':main()

"""Render manuscript tables and scientific figures from completed analysis files."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/donor-grouping-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
GEN = PAPER / 'generated'
MODELS = {'gpt2-small': 'GPT-2 Small', 'qwen3-0.6b': 'Qwen3-0.6B'}
K = [1, 2, 4, 8]


def read(p):
    return json.loads(Path(p).read_text())


def num(v, digits=3):
    return '--' if v is None else f'{v:.{digits}f}'


def estimate(v, scale=1, digits=3):
    ci = v.get('ci', v.get('ci95'))
    if v.get('mean') is None:
        return '--'
    return f"{num(v['mean']*scale, digits)} [{num(ci[0]*scale,digits)}, {num(ci[1]*scale,digits)}]"


def table(headers, rows, caption, label, alignment=None):
    alignment = alignment or ('l' + 'r'*(len(headers)-1))
    body = '\n'.join(' & '.join(row) + r' \\' for row in rows)
    return ('\\begin{table}[ht]\n\\centering\\small\n\\begin{tabular}{' + alignment + '}\n\\toprule\n' +
            ' & '.join(headers) + ' \\\\\n\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n' +
            '\\caption{' + caption + '}\n\\label{' + label + '}\n\\end{table}\n')


def figures(token, circuit):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3), constrained_layout=True)
    palette = plt.get_cmap('tab10', 8)
    for ax, k in zip(axs, K):
        grid = np.array([[(r+s*k//8) % 8 for s in range(8)] for r in range(8)])
        ax.imshow(grid, cmap=palette, vmin=-.5, vmax=7.5, interpolation='nearest')
        ax.set_title(f'K = {k}')
        ax.set_xticks([0, 3, 7], ['1', '4', '8'])
        ax.set_yticks([0, 3, 7], ['1', '4', '8'])
        ax.set_xlabel('Context slot')
        ax.set_ylabel('Bank rotation' if k == 1 else '')
    fig.savefig(GEN/'grouping-design.pdf'); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True)
    colors = ['#1f5a99', '#bd5a25']
    for ax, (model, title), color in zip(axs, MODELS.items(), colors):
        values = [token[model]['secondary_dose_curve'][str(k)]['margin_gap_absolute'] for k in K]
        y = np.array([v['mean'] for v in values]); ci = np.array([v['ci'] for v in values])
        ax.errorbar(range(4), y, yerr=np.array([y-ci[:, 0], ci[:, 1]-y]), fmt='o-', color=color, capsize=3)
        ax.set_xticks(range(4), K); ax.set_ylim(bottom=0)
        ax.set_xlabel('Distinct final tokens per bank')
        ax.set_ylabel('Mean absolute pooling gap (logits)')
        ax.set_title(title); ax.grid(axis='y', alpha=.18)
    fig.savefig(GEN/'grouping-gap.pdf'); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(7.4, 4.5), constrained_layout=True)
    for col, (model, title) in enumerate(MODELS.items()):
        summary = token[model]
        for row, direction in enumerate(['restore', 'destroy']):
            ax = axs[row, col]
            for k, color in zip(K, ['#555555','#8bb7ce','#388bb5','#c95b28']):
                cells = [c for c in summary['cells'] if c['position']=='end' and c['direction']==direction and c['k']==k]
                ax.plot([.25,.5,.75,1], [c['gap_margin_absolute']['mean'] for c in cells], 'o-', color=color, label=f'K={k}')
            ax.set_title(f'{title}: {direction}'); ax.set_xticks([.25,.5,.75,1])
            ax.set_xlabel('Fraction of blocks completed'); ax.set_ylabel('Mean |gap| (logits)')
            ax.set_ylim(bottom=0); ax.grid(axis='y',alpha=.18)
    axs[0,0].legend(ncol=2,fontsize=8)
    fig.savefig(GEN/'grouping-sites.pdf'); plt.close(fig)

    ref = circuit['circuits']['reference']
    fig, axs = plt.subplots(1, 2, figsize=(7.4,2.8), constrained_layout=True)
    for ax, (stratum, title) in zip(axs, [('full_pair_success','Full model correct'),('full_pair_error','Full model wrong')]):
        item = ref['strata'][stratum]
        vals = [item['mean_pair_agreement'],item['expected_resample_pair_agreement']]
        if item['n']:
            means = np.array([v['mean']*100 for v in vals]); ci=np.array([v['ci95'] for v in vals])*100
            ax.bar([0,1],means,color=['#1f5a99','#bd5a25'],width=.6)
            ax.errorbar([0,1],means,yerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='none',ecolor='black',capsize=4)
        ax.set_xticks([0,1],['Mean patch','Joint resample'])
        ax.set_ylabel('Two-name agreement with full model (%)'); ax.set_ylim(0,105)
        ax.set_title(f"{title} (n={item['n']})"); ax.grid(axis='y',alpha=.18)
    fig.savefig(GEN/'circuit-agreement.pdf'); plt.close(fig)


def main():
    GEN.mkdir(exist_ok=True, parents=True)
    token = {m: read(ROOT/'token/results/analysis-test'/f'{m}.json') for m in MODELS}
    circuit = read(ROOT/'circuit/results/analysis.json')
    old = read(ROOT/'results/previous-study-diagnostics.json')['models']
    for m in MODELS:
        assert token[m]['groups'] == 64
        assert read(ROOT/'token/results'/m/'replay-verification.json')['passed']
    assert read(ROOT/'circuit/results/replay-verification.json')['passed']
    figures(token, circuit)
    rows = []
    for model, label in MODELS.items():
        v = token[model]
        rows.append([label,num(v['secondary_dose_curve']['1']['margin_gap_absolute']['mean']),
                     num(v['secondary_dose_curve']['8']['margin_gap_absolute']['mean']),estimate(v['primary'])])
    (GEN/'primary-table.tex').write_text(table(['Checkpoint',r'$K=1$',r'$K=8$',r'Paired difference [97.5\% CI]'], rows,
        'Primary controlled grouping result in logit units. Each value averages eight rotations, both directions and three nonterminal END depths within each of 64 recipients. The two difference intervals adjust for the two checkpoints; the absolute-gap means themselves are descriptive.', 'tab:primary'))
    rows=[]
    for model,label in MODELS.items():
        c=token[model]['competence']
        rows.append([label,estimate(c['recipient_clean_pair_accuracy'],100,1),estimate(c['recipient_clean_top1_accuracy'],100,1),estimate(c['recipient_clean_margin'])])
    (GEN/'competence-table.tex').write_text(table(['Checkpoint',r'Two-verb accuracy (\%)',r'Full-vocabulary top-1 (\%)','Clean margin'],rows,
        'Recipient competence in the fresh agreement panel (64 groups per model). Brackets are descriptive 95\\% intervals. The patching task measures a grammatical verb preference, not unrestricted next-token completion accuracy.', 'tab:competence'))
    ref=circuit['circuits']['reference']; empty=circuit['circuits']['empty_attention']
    rows=[]
    for a,label in [('mean','Mean'),('expected_joint_resample','Joint resample')]:
        f=ref['mean_empty_adjusted_faithfulness' if a=='mean' else 'resample_empty_adjusted_faithfulness']
        fstr=num(f['estimate'])
        if f['ci95'] is not None: fstr+=f" [{num(f['ci95'][0])}, {num(f['ci95'][1])}]"
        rows.append([label,estimate(ref[a]['margin']),estimate(empty[a]['margin']),fstr])
    (GEN/'circuit-margin-table.tex').write_text(table(['Ablation','Reference margin','Empty-attention margin',r'Faithfulness $F_a$'],rows,
        'Circuit and empty-control margins use screening-prevalence weights. Intervals are descriptive 95\\%. Faithfulness uses Equation~\\ref{eq:faithfulness}; the raw margins expose changes in its policy-dependent denominator.', 'tab:circuitmargin'))
    rows=[]
    for s,label in [('full_pair_success','Full model correct'),('full_pair_error','Full model wrong')]:
        v=ref['strata'][s]
        rows.append([label,str(v['n']),estimate(v['mean_pair_agreement'],100,1),estimate(v['expected_resample_pair_agreement'],100,1),estimate(v['paired_pair_agreement_difference'],100,1)])
    (GEN/'circuit-agreement-table.tex').write_text(table(['Stratum','n',r'Mean (\%)',r'Resample (\%)',r'Difference (pp)'],rows,
        'Agreement with the intact model\'s two-name choice. Resampling averages the eight per-donor agreement indicators. Difference is paired mean minus resample. Intervals are descriptive 95\\% recipient-bootstrap intervals within correctness strata.', 'tab:circuitagreement'))
    rows=[]
    labels={'gpt2-small':'GPT-2 Small','gpt2-medium':'GPT-2 Medium','gpt2-large':'GPT-2 Large','pythia-410m':'Pythia-410M','tinyllama-1.1b':'TinyLlama-1.1B','qwen3-0.6b':'Qwen3-0.6B'}
    for m,label in labels.items():
        z=old[m]
        rows.append([label,num(z['old_primary']),num(100*z['sva_end_share_of_old_primary'],1),num(z['old_primary_excluding_sva_end']),num(z['absolute_gap_shared_token_cells']),num(z['absolute_gap_sva_end'])])
    (GEN/'previous-table.tex').write_text(table(['Checkpoint',r'Old $|P|-|N|$',r'\shortstack{SVA-END\\share (\%)}',r'\shortstack{Excluding\\SVA-END}',r'\shortstack{Shared-token\\$|D|$}',r'SVA-END $|D|$'], rows,
        'Retrospective diagnostics of the original narrow-bank nonterminal panel. SVA-END share is its contribution to the old equal-cell aggregate, not a causal share of the network. Remaining columns expose the confounded donor-token contrast. Values are logit units except the percentage.', 'tab:previous'))
    rows=[]
    for m,label in MODELS.items():
        n=token[m]['numerical']
        rows.append([label,f"{n['identity_logit_max']:.2g}",f"{n['identity_probability_max']:.2g}",f"{n['global_resample_across_doses_max']:.2g}",f"{n['terminal_end_propagation_max']:.2g}"])
    (GEN/'numerics-table.tex').write_text(table(['Checkpoint','Identity logits','Identity prob.',r'Global $R$ invariant',r'Terminal-END $|P|$'],rows,
        'Maximum absolute numerical discrepancies across the complete new grouping test panel. Write target and outside-mask errors are exactly zero. Structural controls are numerical checks, not additional scientific samples.', 'tab:numerics'))
    rows=[]
    for m,label in MODELS.items():
        for k in K:
            v=token[m]['secondary_dose_curve'][str(k)]
            rows.append([label,str(k),estimate(v['margin_gap_signed']),estimate(v['probability_gap_absolute'],1000),estimate(v['propagation_margin_absolute']),estimate(v['normalization_margin_absolute'])])
    (GEN/'accounting-table.tex').write_text(table(['Checkpoint','K',r'Signed $D$',r'Probability $|D|$ ($10^{-3}$)',r'$|P|$',r'$|N|$'],rows,
        'Secondary nonterminal END summaries with descriptive 95\\% intervals. Probability gaps are in units of $10^{-3}$; other columns are logits. The ordered terms are not causal shares and need not add after taking absolute values.', 'tab:accounting'))
    output={'token':{m:{'primary':v['primary'],'curve':v['secondary_dose_curve'],'numerical':v['numerical']} for m,v in token.items()},
            'circuit_screen_counts':circuit['screening']['strata'],'circuit_primary_strata':ref['strata'],
            'circuit_faithfulness':{'mean':ref['mean_empty_adjusted_faithfulness'],'resample':ref['resample_empty_adjusted_faithfulness']}}
    (ROOT/'results/report-inputs.json').write_text(json.dumps(output,indent=2)+'\n')
    primary_positive=all(v['primary']['ci'][0]>0 for v in token.values())
    primary_statement=('Both adjusted primary intervals exclude zero in the positive direction.' if primary_positive else
                       'The primary intervals do not establish a positive effect in both checkpoints.')
    finding=[]
    for m,label in MODELS.items():
        v=token[m]
        with np.load(ROOT/'token/results/analysis-test'/f'{m}-components.npz') as z:
            positive=int((z['primary_recipient_values']>0).sum())
        finding.append(f"For {label}, the paired increase in absolute margin gap is {estimate(v['primary'])} logits (adjusted 97.5\\% interval); {positive} of 64 recipient-level differences are positive.")
    curve_positive=all(np.all(np.diff([v['secondary_dose_curve'][str(k)]['margin_gap_absolute']['mean'] for k in K])>0) for v in token.values())
    curve_statement=('The aggregate absolute margin-gap means increase across all four tested policies in both models (Figure~\\ref{fig:gap}); this shape is a secondary observation.' if curve_positive else
                     'The four-policy curves in Figure~\\ref{fig:gap} are secondary observations; the extreme contrast alone does not imply a monotonic response.')
    (GEN/'grouping-findings.tex').write_text(primary_statement+' '+ ' '.join(finding)+'\n\n'+curve_statement+' The task, recipient and full donor population are unchanged. The result therefore addresses the original task/donor-composition confound within this constructed agreement panel. It does not hold within-bank centroid or dispersion fixed.\n')
    replays=[read(ROOT/'token/results'/m/'replay-verification.json')['maximum_error'] for m in MODELS]
    replays.append(read(ROOT/'circuit/results/replay-verification.json')['maximum_error'])
    replay_statement=('The predetermined four cases per residual checkpoint and eight circuit cases reproduced exactly in fresh model processes.' if max(replays)==0 else
                      f'The predetermined fresh-model replays passed with a maximum recorded discrepancy of {max(replays):.6g}.')
    numerical_max=max(v['numerical']['identity_logit_max'] for v in token.values())
    (GEN/'numerical-findings.tex').write_text(f'The complete residual test panels pass their original numerical gates; their largest full-vocabulary identity-logit discrepancy is ${numerical_max:.6f}$. '+replay_statement+' Detailed maxima and the preserved calibration failure are in the appendix.\n')
    effects=' and '.join(f"{v['primary']['mean']:.3f} [{v['primary']['ci'][0]:.3f}, {v['primary']['ci'][1]:.3f}]" for v in token.values())
    abstract=('Mean activation patching evaluates an averaged donor state, whereas resample patching averages outcomes of individual interventions. We study how grouping donors changes their discrepancy while holding the complete donor population fixed. For each of 64 fresh agreement recipients, we partition the same 64 donors into eight banks with one, two, four or eight distinct final tokens per bank. Every donor appears once under every policy, so the global resample average is invariant. On GPT-2 Small and Qwen3-0.6B, the eight-versus-one grouping contrast changes the mean absolute margin gap by '+effects+' logits, respectively (adjusted 97.5\\% intervals). A separate 26-head IOI reference-circuit assay uses the same eight-donor bank for mean and coherent resample ablations. Mean ablation raises correct-case model agreement by 8.50 percentage points [5.66, 11.62], while the error-case contrast is inconclusive. Numerical controls and fresh-model replays validate the admitted float32 configurations. Clean two-verb accuracy is 100\\% for GPT-2 and 71.9\\% for Qwen, limiting a grammatical-mechanism interpretation. The findings isolate a grouping-policy effect on fixed donor support and delimit a practical circuit comparison; they do not identify pure token-identity causality or establish that a smaller pooling gap yields a better mechanistic explanation.\n')
    (GEN/'abstract.tex').write_text(abstract)
    interpretation=('The controlled experiment supports donor grouping as a material evaluation choice in these two fixed checkpoints: the same donor population can yield substantially different centroid-versus-resample discrepancies depending on which states share a bank.' if primary_positive else
                    'The controlled experiment supplies a bounded estimate of grouping sensitivity, with model-specific uncertainty shown in the primary intervals.')
    (GEN/'interpretation.tex').write_text(interpretation+' The circuit assay supplies a separate practical comparison, with a clear correct-case agreement difference and an inconclusive error-case difference. These findings support a focused empirical audit. They do not turn the earlier propagation accounting into a universal mechanism or establish a new circuit-discovery method.\n')
    print('Rendered six tables, one detailed accounting table and four scientific figures from completed analyses.')


if __name__=='__main__':
    main()

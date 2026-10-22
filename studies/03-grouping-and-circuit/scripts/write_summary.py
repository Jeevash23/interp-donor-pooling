"""Write the author-facing revision summary and claim/evidence map from results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS={'gpt2-small':'GPT-2 Small','qwen3-0.6b':'Qwen3-0.6B'}


def read(p):return json.loads(Path(p).read_text())


def interval(x,scale=1,digits=3):
    ci=x.get('ci',x.get('ci95'))
    return f"{x['mean']*scale:.{digits}f} [{ci[0]*scale:.{digits}f}, {ci[1]*scale:.{digits}f}]"


def main():
    token={m:read(ROOT/'token/results/analysis-test'/f'{m}.json') for m in MODELS}
    circuit=read(ROOT/'circuit/results/analysis.json')
    ref=circuit['circuits']['reference']
    supported=all(v['primary']['ci'][0]>0 for v in token.values())
    finding=('The controlled grouping effect is positive on both fixed checkpoints.' if supported else
             'The controlled grouping results require a model-specific interpretation; inspect the intervals below.')
    table='\n'.join(f"| {MODELS[m]} | {v['secondary_dose_curve']['1']['margin_gap_absolute']['mean']:.5f} | {v['secondary_dose_curve']['8']['margin_gap_absolute']['mean']:.5f} | {interval(v['primary'],digits=5)} |" for m,v in token.items())
    correct=ref['strata']['full_pair_success']['paired_pair_agreement_difference']
    error=ref['strata']['full_pair_error']['paired_pair_agreement_difference']
    text=f'''# Donor grouping in activation patching

This is the completed October revision of the donor-pooling study, with two new experiments and a shorter paper. {finding} The circuit experiment demonstrates a practical evaluation difference, while its error-agreement comparison remains inconclusive. The September measurements and original paper remain in `../../submission-study/`.

## Read the revision

- [Revised paper](output/pdf/donor-grouping-revised-paper.pdf), in TMLR preprint style and marked not submitted.
- [Editable LaTeX](paper/main.tex).
- [Claim and evidence map](CLAIM-EVIDENCE.md).
- [Prospective local protocol](PROTOCOL.md), frozen before new test interventions.
- [Reproduction guide](REPRODUCIBILITY.md) and [checkpoint sources](MODEL-SOURCES.md).
- [New study review archive](output/donor-pooling-revision-review.zip).
- [Independent response verification](results/revision-verification.json) and [archive extraction check](output/archive-integration.json).

## New evidence

Each recipient uses the same 64 context-by-noun donors in four different partitions. Each partition forms eight banks of eight; every donor appears once at each setting. The number of distinct final tokens per bank is K=1,2,4,8. The primary comparison averages three nonterminal END depths, both intervention directions and all eight rotations within each of 64 recipient groups. Intervals adjust for the two checkpoint comparisons. Values are logits.

| Checkpoint | Mean absolute gap K1 | Mean absolute gap K8 | Paired K8 minus K1 [97.5% interval] |
|---|---:|---:|---:|
{table}

The donor population and its global average individual-patch outcome remain fixed. The grouping changes bank centroids, semantic mixtures and activation dispersion; this does not isolate a pure token-identity mechanism. Both models use the same two-template agreement panel. Clean two-verb accuracy is 64/64 for GPT-2 but only 46/64 (71.9%) for Qwen; both choose the target verb as unrestricted top-one in only 1/64 cases. These are pooling measurements, not evidence that both models consistently solve grammatical agreement. Rotations and intervention executions are not independent samples. Probability gaps require separate interpretation.

The reference-circuit study screens 8,192 fresh GPT-2 Small IOI prompts, finds 119 two-name errors, then evaluates the predeclared 64 errors and 128 correct cases. Mean ablation versus expected coherent resampling changes correct-case agreement by **{interval(correct,100,2)} percentage points** and error-case agreement by **{interval(error,100,2)} points** (descriptive 95% intervals). Empty-adjusted faithfulness is {ref['mean_empty_adjusted_faithfulness']['estimate']:.3f} versus {ref['resample_empty_adjusted_faithfulness']['estimate']:.3f}.

Our high error agreement does not reproduce the severe loss reported in the September paper. The donor distribution, bank size, templates and resampling endpoint differ. [The verified comparison](circuit/COMPARISON-WITH-SEPTEMBER-PAPER.md) explains those differences without claiming which causes the discrepancy.

## What changed from the first draft

The new paper centers the controlled grouping test. The old propagation/final-normalization comparison is supporting evidence, with its donor-token confound made explicit. The old head-scan stability result is limited by small effect sizes and omission of the primary GPT-2 name movers. The five-forward correction is not presented as a superior estimator. The earlier reduced-precision failures remain disclosed.

The first circuit calibration's separate float32 projection diagnostic failed; a documented float64 algebra check resolved the structural question without changing native model computation or relaxing native-output gates. An analysis indexing bug was caught and corrected on development data before test inference. Failed and corrected records are retained.

## Research decision

Keep this as a focused empirical methods study and stop the planned experimental expansion here. The fresh controls address the strongest donor-composition criticism; the circuit assay supplies a bounded practical consequence. More arbitrary model sweeps are not the next requirement.

The remaining decision is scientific positioning, especially whether the controlled grouping lesson is sufficiently useful beyond established mean/resample sensitivity. The paper is ready for human scientific review, not represented as submission-ready or guaranteed publishable. TMLR remains a candidate for an empirical/reproducibility contribution. ARR would require a separate shorter manuscript and language-research framing; this artifact does not claim ARR formatting compliance. AI contributed substantially to design, implementation, execution, analysis and drafting, and the manuscript discloses that role. No public release or submission was performed.
'''
    (ROOT/'README.md').write_text(text)
    claims=f'''# Claim and evidence map

All October primary hypotheses and panel sizes were fixed before test interventions. The questions were motivated by observed September results; this is a local prospective plan, not public preregistration.

| Claim | Estimate and status | Direct evidence | Limit |
|---|---|---|---|
'''
    for m,v in token.items():
        claims+=f"| Fixed-support K8 versus K1 grouping increases absolute margin pooling gap in {MODELS[m]} | {interval(v['primary'],digits=5)} logits, adjusted 97.5% interval | `token/results/analysis-test/{m}.json`, `token/results/{m}/test/*.npz` | 64 recipient groups, two SVA templates, one fixed checkpoint; no pure token identity claim |\n"
    claims+=f'''| Mean/resample circuit agreement differs on correct cases | {interval(correct,100,2)} percentage points, descriptive 95% interval | `circuit/results/analysis.json`, raw `circuit/results/test/*.npz` | Same-bank eight-donor conditional assay; not circuit discovery |
| Error-agreement difference is inconclusive | {interval(error,100,2)} percentage points | Same files, full pair-error stratum | 64 sampled errors; neither superiority nor equivalence established |
| Whole-donor support and average resample outcome are fixed across K | Exact combinatorial partition plus numerical invariants | `token/data/test.json`, `review/token-design-check.json`, both numerical summaries | Per-bank centroids and per-bank resample scores can change |
| Instrumentation is numerically stable in admitted float32 configurations | Native/identity/structural checks pass; fresh replay records retained | `results/revision-verification.json`, panel calibration/audit/replay JSON | Not a guarantee for other hardware, precision or larger models |
| Earlier six-checkpoint aggregate survives recomputation but is confounded | SVA-END supplies about 29–77% of the old primary aggregate | `results/previous-study-diagnostics.json`; original study raw arrays | Retrospective supporting evidence; task/site contrast does not isolate donor diversity |

## Claims intentionally not made

This study does not establish universal token-diversity causality, robust circuit membership, completeness or minimality, general monotonicity across output metrics, superiority of the earlier correction estimator, a successful larger-model reduced-precision replication, independent human review, or venue acceptance.

## Evidence separation

- New grouping test: 64 recipient groups per checkpoint, same texts across two checkpoints; 202,752 total intervention executions including residual controls.
- New circuit test: 192 selected recipients from an 8,192-example screen; 3,456 mean/resample reference/empty comparisons plus 576 identity/all-kept controls. Source and baseline forwards are recorded separately.
- Calibration, development, replay and archival recomputation are verification/development, not added scientific sample sizes.
- The original six-model results are separately archived prior evidence. The new review archive includes a retrospective summary and its input hashes, not the old raw-response archive or any model weights.
'''
    (ROOT/'CLAIM-EVIDENCE.md').write_text(claims)
    print(finding)


if __name__=='__main__':main()

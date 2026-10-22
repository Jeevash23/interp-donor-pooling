# Donor averaging in activation patching

**Current revision (2 October 2026):** [Donor grouping: controlled experiment and circuit replication](../03-grouping-and-circuit/README.md). It contains two completed new experiments, a 13-page revised paper, raw measurements and a portable review archive. Use that revision for the current claims and limitations. The material below describes the original pilot; the later September six-model study remains in `submission-study/`.

An autonomous local research study begun and completed on 12 September 2026. The question is where the difference between patching an average activation and averaging individual patch outcomes enters the model's computation.

**Measured finding:** across six nonterminal sites, the mean absolute logit gaps are 0.0268 in GPT-2 Small and 0.0164 in Medium. Most of the gap enters during downstream propagation, before the final LayerNorm. Halving donor variation reduces the gap by 75.7% and 74.1%. The aggregate site rankings remain unchanged. These are modest effects in a controlled two-checkpoint experiment, not proof that mean patching generally gives incorrect circuit conclusions.

- [Initial manuscript in PDF](output/pdf/donor-averaging-in-activation-patching.pdf)
- [Editable Word draft](paper/manuscript.docx)
- [Text manuscript](paper/manuscript.md)
- [Results in plain language](RESULTS.md)
- [Reproduction instructions](REPRODUCIBILITY.md)
- [Prospective protocol](docs/PROTOCOL.md)
- [Closest related work and novelty limits](docs/RELATED-WORK.md)
- [Execution notes and disclosed deviations](docs/EXECUTION-NOTES.md)
- [Claim and evidence map](docs/CLAIM-EVIDENCE-MAP.md)

The study contains 64 evaluation groups, four recipient configurations per group, eight matched donors per recipient, two fixed models, eight sites per model, and 20 conditions per site: **81,920 intervention executions** in the main evaluation. There are 2,560 baseline prompt executions per model. These are repeated measurements, not 81,920 independent scientific observations.

`scripts/` contains the implementation and analysis; `data/` contains deterministic prompts and hashes; `results/` contains all measurements, saved states and audits; `paper/` contains the draft. The code reads the existing checkpoints and environment but imports no scientific code from another project. Earlier projects were not edited. No external publication, message or submission was made.

The broad mean-versus-individual distinction is already established in the literature. The proposed incremental contribution is the matched-bank decomposition and controlled dispersion test. The draft is ready for initial scientific review; novelty and wider generalization remain open.

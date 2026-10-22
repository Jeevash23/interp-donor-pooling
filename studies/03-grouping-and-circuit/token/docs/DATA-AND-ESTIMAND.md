# Data and estimand reconstruction

These explanatory notes were written after the local pretest freeze. They do not modify the frozen protocol, source, data or analysis. They describe the code and saved arrays so another researcher can reconstruct the estimand directly. The internal implementation checks were performed by [redacted]; they are not independent human peer review.

## Prompt construction and fixed support

There are 4 calibration, 8 development and 64 test recipient groups. Within each split, group index `g` selects template `g % 2` and clean answer class `(g // 2) % 2`. Class 0 has a singular head and clean target `is`; class 1 has a plural head and clean target `are`. The competitor is the other verb. A paired corruption changes only head number, preserving the original clean target used by every metric. Distractors have the number opposite to the clean head.

The two templates are `At the station, the {adj} {head} near the {distr}` and `At the office, the {adj} {head} beside the {distr}`. Each group fixes the grammatical head, template, answer class and recipient. It samples eight adjective contexts and eight distractor noun labels. The recipient's adjective and final noun are excluded from their respective donor pools; the head noun is excluded from donor final nouns. Only nouns whose singular and regular plural forms are each one appended token in both tokenizers are eligible. Adjectives must also be a single token in both. Full-prompt tokenization and offsets subsequently verify the qualification rather than relying on isolated-word tokenization alone.

The 8×8 grid crosses every context with every noun label. Every grid point has a clean and corrupted prompt, giving 128 donor prompts plus two recipient prompts. Each of the 130 prompt strings in a group is distinct. Within a model/group, token counts, anchor index and answer-token IDs are equal across all 130 prompts. Within one context row and answer side, the final token is the only token that changes across noun columns. Every donor final-token ID differs from the recipient's final-token ID.

For K in 1, 2, 4, 8 and rotation r in 0,…,7, bank slot s selects grid cell `(s, (r + floor(s*K/8)) % 8)`. Each bank has eight unique donors, exactly K distinct final-token IDs, and equal multiplicity 8/K of those IDs. Across the eight banks at a fixed K, all 64 cells occur exactly once. No donor is discarded or reweighted as K changes. Noun and context order are randomized before constructing the partitions.

The design therefore varies a **balanced partition policy of fixed donor support**. The overall average individual-patch outcome, the overall source-token counts and the grand source centroid are invariant across K in exact arithmetic. Individual bank centroids and within-bank covariance are not fixed. The interpretation is not a mechanism-specific effect of token identity separated from lexical meaning or residual geometry.

## Deterministic generation and overlap

The split seed constants are calibration `26100211`, development `26100212` and test `26100213`. Group RNG seed is `split_seed + 1009*g + 10000019*attempt`. The `attempt` starts at zero and increases only after outcome-independent overlap/tokenization rejection. Every saved row includes its seed and accepted attempt. Python's `random.Random` supplies sampling and ordering. Eight-donor bank rotations are deterministic given the sampled grid.

Generation rejects exact prompt overlap with the old pilot/calibration/development/test inputs and with any earlier new split, and rejects duplicate recipient pairs within the current split. Chance donor reuse within a split is allowed and counted. Test contains 8,320 prompt occurrences and 8,164 distinct strings: 154 extra occurrences from repeated donor prompts and two recipient/donor shared strings across different groups. No prompt occurs more than twice, and all 64 recipient pairs are unique. Calibration and development have no within-split text reuse. These counts describe strings, not semantic or vocabulary independence.

`data/generation-audit.json` records the qualified lexicons, templates, rejection counts and overlap checks. Rejected attempts are determined without model scores. `results/pretest-data-regeneration.json` confirms all three prompt JSON files were reproduced byte for byte before the freeze. Historical overlap evidence is conditional on the listed old data files being available; a recipient of a reduced archive should not imply that missing historical inputs were rechecked.

## Tokenization and BOS

The tokenizer receives the prompt with `add_special_tokens=False`. One checkpoint-configured BOS token is explicitly prepended afterward: GPT-2 Small uses ID 50256 and Qwen3-0.6B uses ID 151643. Qwen is used as a raw text completion model without a chat template or generated reasoning. Appending a space and the target or competitor must add exactly one token without changing any prefix ID.

ANCHOR is the one token containing the grammatical head noun. END is the final distractor token. Both are zero-based indices in the BOS-prefixed input. Since the final noun is causally later than ANCHOR and every bank contains every context once, each ANCHOR bank has the same context-state multiset. Contexts themselves differ at the preceding adjective; their individual anchor states need not coincide.

## Intervention layout

The four block-output sites are GPT-2 layers `[2,5,8,11]` or Qwen layers `[6,13,20,27]`, in zero-based numbering. Each is evaluated at ANCHOR then END, so the site order is quarter-ANCHOR, quarter-END, half-ANCHOR, half-END, three-quarter-ANCHOR, three-quarter-END, final-ANCHOR, final-END.

Restoration patches clean donor vectors into the corrupted recipient. Destruction patches corrupted donor vectors into the clean recipient. All scoring retains the original clean target. Each direction evaluates 64 cached individual donor patches, 32 mean-of-eight patches and three controls: identity, paired other recipient, and zero. Thus there are 99 interventions/direction and 198/site. The final intervention count per model's 64-recipient test is 64×8×198 = 101,376. This is a computational count; the inferential sample size is 64 recipients.

The first 99 condition names belong to restoration. Indices 0–63 are grid patches in context-major/noun-minor order; 64–95 are centroids in K-major/rotation-minor order; 96,97,98 are identity, paired and zero. Destruction repeats this layout offset by 99. `results/MODEL/design.json` records the exact names and all bank-to-condition mappings; downstream consumers should read it instead of relying only on this description.

## Array schema

Each per-recipient NPZ is a compressed collection of ordinary NumPy arrays. It has no pickled objects.

| Array | Shape | Meaning |
|---|---:|---|
| `metrics` | `[8,198,5]` | Native patched output metrics. |
| `endpoints` | `[8,64,2,5]` | Boundary B and C metrics for each bank. |
| `baseline` | `[130,5]` | Recipient clean/corrupt followed by context-major, noun-minor, clean/corrupt donor pairs. |
| `dispersion` | `[8,64,3]` | Mean squared source radius, squared centroid norm, maximum source radius. |
| `condition_names` | `[198]` | Strings indexing the intervention axis. |
| `bank_names` | `[64]` | Direction-major, K-major, rotation-minor bank labels. |
| `id` | scalar string | Recipient ID matching the generated input. |
| `final` | `[8,198,H]` | Final pre-normalization residuals, only for fixed replay recipients. |
| `source` | `[8,130,H]` | Captured block-output source residuals, only for fixed replay recipients. |

H is 768 for GPT-2 Small and 1024 for Qwen3-0.6B. The metric axis is `[margin, probability, top1, pair_correct, log_probability]`. Margin is target logit minus competitor logit. Probability is the target's full-vocabulary softmax probability. Top-one is the indicator that the unrestricted argmax equals the target; `pair_correct` is the indicator that margin is strictly positive. In corrupted contexts, the latter is preference for the clean target, not relabeled corrupted-task accuracy. Log probability is the full-vocabulary target log-softmax.

Native logits, source means, mean logits, norms and unembedding are float32. Margin subtraction and softmax/log-softmax scoring are performed in float64. Saved metric arrays are float64; dense residual arrays are float32. Source mean squared radius uses the unnormalized residual Euclidean norm and is not directly comparable in magnitude across architectures with different residual geometry.

## Reconstructing the gap

For each bank, A is `metrics[site, bank.mean]`; R is the average over its eight `bank.indices`; B is `endpoints[site,bank,0]`; C is `endpoints[site,bank,1]`. B scores the unchanged final normalization and output head applied to the mean final residual of the eight donor-patched executions. C scores the mean of their native logit vectors.

Define D=A−R, P=A−B, N=B−C and S=C−R. For margin, S is a linear-readout roundoff diagnostic. For probability, S is the scoring nonlinearity term. The exact ordered accounting identity is D=P+N+S; D=P+N holds for margin up to measured rounding. Absolute components do not add to an absolute gap, and these are not unique causal mediation effects.

`analyze.py` reshapes bank axes into `[recipient,site,direction,dose,rotation,metric]` after asserting the full bank order. The primary recipient value averages `abs(D[K=8])-abs(D[K=1])` across END sites `[1,3,5]`, both directions and eight rotations. Each recipient supplies one value. Twenty thousand bootstrap draws with NumPy RNG seed `26100299` resample recipient indices within template×answer strata. Primary intervals are 97.5%, controlling the two-checkpoint family with Bonferroni; secondary intervals are pointwise 95%.

`seed_block` in the generated rows is `g//16`; it labels the fixed replay spacing and is not a bootstrap stratum. Replay recipients are indices 0,16,32,48. Their repeated execution uses a fresh model instance. The analysis output includes one primary value per recipient in CSV and a component NPZ with the axes described above.

# Norm-relative steering variance check and pythia-1.0b dense k-sweep follow-up

## Research Question

This follow-up run addressed two open questions left by the earlier accepted norm-relative steering scaling result.

1. **Variance / robustness of the scaling pattern:** the previous margins were single point estimates from one held-out split per model, so it was unclear whether the very low `pythia-2.8b` result and the unexpectedly low `pythia-1.0b` result were real effects or just noise from a lucky/unlucky split or steering draw.
2. **Trustworthiness of `pythia-1.0b`'s previously selected `best_k=0.4`:** the earlier pilot sweep used a sparse grid, and `0.4` was much smaller than the other models’ best-k values. This raised the possibility that the 1.0b result was an artifact of coarse tuning rather than a true low-k optimum.

The working hypothesis was:
- a denser `k` sweep for `pythia-1.0b` would either confirm the low-k optimum or reveal that a better higher-`k` value had been missed;
- repeated seeded evaluations would show whether `pythia-2.8b`’s near-collapse and `pythia-1.0b`’s anomaly persist after estimating variance.

## Protocol

The run **reused the exact accepted norm-relative steering pipeline** from the prior run, keeping the steering method, prompt source, metric definition, normalization rule, layer choice, and generation/evaluation settings unchanged. The only intentional changes were:

- adding explicit seed control for repeated evaluation;
- running a **denser pilot `k` sweep for `pythia-1.0b`**;
- repeating the final test evaluation across multiple seeds for all four models.

GPU usage was restricted to **GPU 0 only**, and artifacts were kept compact: CSV/JSON summaries plus the report.

### Seeded repeated evaluation

The evaluation pipeline was made seedable so that each run could vary:
- held-out test-prompt subsampling;
- per-prompt random control-direction draws.

Five seeds were used for each model:
- `101, 202, 303, 404, 505`

Each seed used **60 held-out test prompts**.

### Dense `k` sweep for `pythia-1.0b`

`pythia-1.0b` was re-tuned using the same pilot/tuning procedure as before, but on a denser and wider grid:

`[0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6, 2.4, 3.2, 4.8, 6.4, 8.0]`

The goal was to test both:
- whether the earlier `k=0.4` was really optimal in the low-k region, and
- whether a better value existed at the larger scales that were preferred by the other Pythia models.

### Best-k choices used in the repeated evaluations

- `pythia-1.0b`: used the **new dense-sweep best k = 0.2**
- `pythia-410m`, `pythia-1.4b`, `pythia-2.8b`: reused their previously selected best-k values from the accepted prior run

### Aggregate statistics

For each model, the repeated runs were summarized with:
- mean margin
- sample standard deviation (`ddof=1`)
- standard error of the mean
- 95% t-interval confidence interval
- min and max across seeds

Pairwise comparisons were also computed for the comparisons most relevant to the anomaly/collapse claims:
- `1.0b` vs `410m`
- `1.0b` vs `1.4b`
- `2.8b` vs `410m`
- `2.8b` vs `1.4b`

## Results

## Dense `pythia-1.0b` k-sweep

The dense sweep changed the selected best k:

- **old sparse-grid best k:** `0.4`
- **new dense-grid best k:** `0.2`
- **margin at new best k=0.2:** `0.009620891534723341`
- **margin at old k=0.4:** `0.00402`
- **old k=0.4 rank in dense grid:** `3rd`

The important qualitative finding is that **the optimum stayed in the low-k region**, but it was **not exactly at 0.4**. The previous `0.4` choice was therefore not the true pilot optimum.

Just as important, the denser sweep found **no better high-k alternative**: all tested values at or above `1.0` had margin `<= 0`, and nothing in the `3.2–6.4` range used by the other models beat the low-k region.

So the dense sweep says two things at once:
- the previous `k=0.4` was somewhat off; and
- the broader story that `pythia-1.0b` prefers unusually low k still holds.

## Multi-seed steering margins

All four models were evaluated for **5 seeds each**.

### Per-model summary

| Model | Seeds | Mean margin | Std | 95% CI |
|---|---:|---:|---:|---:|
| pythia-410m | 5 | 0.05647209545131773 | 0.010959663025129317 | [0.0428638764405697, 0.07008031446206575] |
| pythia-1.0b | 5 | -0.000020035472698509692 | 0.002458589432429222 | [-0.0030727774573101004, 0.0030327065119130814] |
| pythia-1.4b | 5 | 0.0403575612232089 | 0.005537659529914902 | [0.03348164875060359, 0.04723347369581422] |
| pythia-2.8b | 5 | 0.00867582024075091 | 0.0029841016982758628 | [0.004970568595023059, 0.012381071886478759] |

### Comparison to the previous single-run point estimates

The earlier accepted run had reported:
- `410m`: `0.0534`
- `1.0b`: `0.0043`
- `1.4b`: `0.0353`
- `2.8b`: `0.0010`

The new multi-seed means are broadly consistent with that overall pattern:
- `410m` remains clearly strong: now mean `0.0565`
- `1.4b` remains clearly positive: now mean `0.0404`
- `2.8b` remains much smaller than `410m` and `1.4b`: now mean `0.00868`
- `1.0b` becomes effectively **zero on average** after retuning and reseeding: mean `-0.00002`

In other words, adding variance estimates did **not** wash out the unusual low performance of `1.0b` and `2.8b`; if anything, it made the `1.0b` weakness look even clearer.

## Interpretation of the Two Open Questions

## Open question A: Are the `2.8b` collapse and `1.0b` anomaly real, or could they be noise?

### `pythia-2.8b`

`pythia-2.8b` has:
- mean margin `0.00868`
- 95% CI `[0.00497, 0.01238]`

This interval is far below both:
- `pythia-410m` CI `[0.04286, 0.07008]`
- `pythia-1.4b` CI `[0.03348, 0.04723]`

The run summary states that **none of the relevant pairwise confidence intervals overlap**. So although `2.8b` is not literally zero, it is still **dramatically below** the better-performing smaller models, and this gap is not plausibly explained by seed noise alone.

### `pythia-1.0b`

`pythia-1.0b` has:
- mean margin `-0.00002`
- 95% CI `[-0.00307, 0.00303]`

This interval sits near zero and is also far below the intervals for `410m` and `1.4b`, again with **no CI overlap** in the reported pairwise comparisons.

So the `1.0b` anomaly also appears **real**, not an artifact of a single split.

### Verdict on A

**Yes: both effects still hold up after variance estimation.**

- The `2.8b` collapse remains real in the practical sense that its performance stays far below `410m` and `1.4b`, with non-overlapping 95% CIs.
- The `1.0b` anomaly also remains real: even after retuning and repeating across seeds, it stays near zero and far below the neighboring model sizes.

This means the paper can state the low-margin behavior of `1.0b` and `2.8b` with substantially more confidence than before.

## Open question B: Was `pythia-1.0b`'s old `best_k=0.4` trustworthy?

The dense sweep shows that:
- **No, `k=0.4` was not the true pilot optimum.**
- The better value on the denser grid was **`k=0.2`**.
- `k=0.4` ranked only **3rd** on the dense grid.

So in a narrow tuning sense, the earlier `best_k=0.4` was a **sparse-grid artifact**.

However, this does **not** rescue the 1.0b model’s overall steering result. After switching to the corrected `best_k=0.2`, the repeated test-set margin for `1.0b` is still essentially zero:
- mean `-0.00002`
- 95% CI `[-0.00307, 0.00303]`

Also, no larger-k value was competitive: all `k >= 1.0` gave non-positive pilot margins. So the anomaly is **not explained by having missed a better high-k regime**.

### Verdict on B

**No: the previous `best_k=0.4` was not fully trustworthy as the exact optimum.** The denser sweep overturns it in favor of **`k=0.2`**.

But the more important substantive conclusion is:
- the low-k preference itself is still real, and
- correcting the exact k does **not materially improve** `pythia-1.0b`’s final steering margin.

So the paper should say that the original exact `k=0.4` claim was too specific, but the broader conclusion that `pythia-1.0b` behaves anomalously under norm-relative steering remains supported.

## Success Criteria Check

The run met the requested goals:

1. **Dense 1.0b k-sweep completed** on the required grid, including values from `0.05` up to `8.0`.
2. **A new best k was selected** for `1.0b`, with ranking information saved.
3. **Repeated seeded evaluations were completed** for all four models, with **5 seeds each**.
4. **Per-model mean, standard deviation, and 95% confidence intervals** were computed.
5. The run produced **explicit verdicts** on both the variance question and the 1.0b best-k question.

## Limitations

A few limitations remain.

- **Only 5 seeds per model** were used. This is enough to estimate variance and show the large gaps here, but still a modest sample for precise confidence intervals.
- The repeated evaluation varied seeded test-prompt subsampling and random control-direction draws, but it is still tied to the same overall pipeline and task setup as the earlier accepted run.
- The conclusions are about the **specific norm-relative steering method and metric already established in the prior run**. They do not by themselves tell us why `1.0b` and `2.8b` underperform.
- The report does not include deeper mechanistic analysis of why `1.0b` prefers such small k or why `2.8b` collapses relative to the smaller models.

## Follow-up Questions

The main next steps suggested by this run are:

1. **Mechanistic diagnosis of the 1.0b anomaly.** Since retuning k did not fix it, the low margin likely reflects something about the model’s internal geometry or steering sensitivity rather than a simple hyperparameter miss.
2. **Mechanistic diagnosis of the 2.8b collapse.** Its margin is consistently positive but much smaller than `410m` and `1.4b`; understanding whether this comes from weaker steerability, stronger resistance to control directions, or evaluation-specific effects would be valuable.
3. **More seeds or alternate held-out splits** if the eventual paper wants tighter uncertainty estimates, though the current intervals already appear sufficient to support the main qualitative claim.
4. **Check whether the anomaly persists under nearby protocol variants** only if the paper scope allows it, since that would test whether the effect is specific to this exact norm-relative setup or reflects a broader scaling behavior.

## Final Verdict

- **Open question A:** the `pythia-2.8b` collapse and `pythia-1.0b` anomaly both **survive variance estimation** and do **not** look like single-run noise.
- **Open question B:** the old `pythia-1.0b best_k=0.4` was **not** the exact true optimum; a denser sweep selects **`k=0.2`** instead. But this correction does **not** materially change the substantive result: `pythia-1.0b` still has essentially zero steering margin.

Overall, this follow-up strengthens the paper’s ability to claim that the non-monotonic scaling pattern is real under the established norm-relative steering protocol, while also correcting the exact 1.0b tuning detail.

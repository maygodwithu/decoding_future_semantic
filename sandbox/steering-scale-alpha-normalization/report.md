# Norm-relative alpha scaling follow-up for causal semantic steering across Pythia sizes

## Research Question

This follow-up tested whether the near-zero steering margin previously observed at Pythia-2.8b was a real scale-dependent effect or a tuning artifact from reusing one fixed raw steering magnitude across model sizes.

The specific hypothesis was:

- If the old 2.8b result was mostly a methodology artifact, then redefining steering strength as a model-relative quantity
  `alpha = k × mean hidden-state norm at the intervention site`
  and selecting `k` on held-out pilot prompts should substantially increase the 2.8b test-set steering margin, ideally into the same rough range as the smaller models.
- If the collapse is genuine, then even after this normalization the 2.8b margin should remain much smaller than the others.

## Prior Result Being Re-tested

The accepted earlier scaling run reported the following test-set steering margins (semantic direction minus random-direction control):

- 410m: `0.00343199516646564`
- 1.0b: `0.011274605058133602`
- 1.4b: `0.008364170556887984`
- 2.8b: `0.0000422820448875`

The unexplained issue was the collapse at 2.8b.

## Experimental Setup

The run reused the accepted scaling pipeline as closely as possible, changing only alpha handling.

Reused from the accepted run:

- same probe-derived sentiment steering direction
- same random-direction control procedure
- same intervention logic: last prompt token, prefill only
- same greedy decoding with `MAX_NEW_TOKENS = 24`
- same sentiment-axis scoring metric
- same model-specific accepted best layer:
  - 410m: layer 20
  - 1.0b: layer 13
  - 1.4b: layer 20
  - 2.8b: layer 27
- same seed-42 prompt split family, with disjoint pilot and test subsets verified by prompt ID overlap = 0

Prompt counts reported by the run:

- pilot prompts: `40`
- test prompts: `60`

## Protocol

### 1. Measure hidden-state norms per model

For each model, the run measured the mean L2 norm of the hidden states at the exact intervention site using unsteered activations from the pilot split only.

Measured mean hidden-state norms:

- 410m: `48.07675552368164`
- 1.0b: `91.55453491210938`
- 1.4b: `173.2047882080078`
- 2.8b: `272.3387145996094`

These norms increased monotonically with model size.

### 2. Compare old alpha to model-relative scale

The run computed each model’s implied old relative coefficient:

`old_implied_k = old_raw_alpha / mean_hidden_norm`

Old raw alphas and implied relative coefficients were:

- 410m: old raw alpha `99.11671447753906`, old implied `k = 2.061634846151716`
- 1.0b: old raw alpha `200.23184204101562`, old implied `k = 2.187022655221114`
- 1.4b: old raw alpha `271.2640075683594`, old implied `k = 1.56614612318101`
- 2.8b: old raw alpha `576.7410278320312`, old implied `k = 2.117734265875317`

This is important: the old 2.8b run was **not** obviously underpowered in relative terms. Its implied `k` was not much smaller than for the other models.

### 3. Pilot sweep for norm-relative alpha

The run swept a shared `k` grid on pilot prompts only:

- `0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.4, 0.8, 1.6, 3.2, 6.4`

The grid was extended beyond the originally suggested compact range because the small grid alone did not reach magnitudes comparable to the old effective alpha. This was documented as a protocol adaptation.

For each model, `best_k` was selected as the pilot value with the largest semantic-minus-random margin.

Selected pilot-optimal `k` values:

- 410m: `6.4`
- 1.0b: `0.4`
- 1.4b: `6.4`
- 2.8b: `3.2`

These gave the following new raw alphas:

- 410m: `307.6912353515625`
- 1.0b: `36.62181396484375`
- 1.4b: `1108.5106445312501`
- 2.8b: `871.48388671875`

### 4. Final held-out test evaluation

Using only the pilot-selected `best_k`, the run evaluated the final test split with semantic steering and random-direction control.

It also reran the old fixed-alpha condition through the same code path for apples-to-apples comparison.

## Results

### Main comparison table

| Model | Mean hidden norm | Old raw alpha | Old implied k | Best k (pilot) | New raw alpha | Old margin | New margin | Delta margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 410m | 48.07675552368164 | 99.11671447753906 | 2.061634846151716 | 6.4 | 307.6912353515625 | 0.00343199516646564 | 0.053391492168884724 | 0.049959497002419084 |
| 1.0b | 91.55453491210938 | 200.23184204101562 | 2.187022655221114 | 0.4 | 36.62181396484375 | 0.011274605058133602 | 0.0043181838700547814 | -0.006956421188078821 |
| 1.4b | 173.2047882080078 | 271.2640075683594 | 1.56614612318101 | 6.4 | 1108.5106445312501 | 0.008364170556887984 | 0.03530193818733096 | 0.026937767630442977 |
| 2.8b | 272.3387145996094 | 576.7410278320312 | 2.117734265875317 | 3.2 | 871.48388671875 | 0.0000422820448875 | 0.001017189584672451 | 0.0009749075397849083 |

### Per-model outcomes

- **410m** improved strongly:
  - old margin `0.00343199516646564`
  - new margin `0.053391492168884724`
  - increase `0.049959497002419084`

- **1.0b** got worse under the pilot-selected norm-relative alpha:
  - old margin `0.011274605058133602`
  - new margin `0.0043181838700547814`
  - decrease `0.006956421188078821`

- **1.4b** improved strongly:
  - old margin `0.008364170556887984`
  - new margin `0.03530193818733096`
  - increase `0.026937767630442977`

- **2.8b** improved relative to its old near-zero baseline, but remained small in absolute terms:
  - old margin `0.0000422820448875`
  - new margin `0.001017189584672451`
  - increase `0.0009749075397849083`

The run also reported:

- ratio of new to old margin at 2.8b: `24.057`
- mean new margin across the three smaller models: `0.031004`
- ratio of 2.8b new margin to that smaller-model mean: `0.0328`

So even after normalization, 2.8b reached only about **3.3%** of the average new margin of the smaller models.

## Interpretation

### Did hidden-state norms grow with model size?

Yes. Mean intervention-site hidden-state norm increased steadily:

- `48.08 → 91.55 → 173.20 → 272.34`

So the motivation for checking norm-relative alpha was valid.

### Was the old 2.8b alpha relatively much weaker than at smaller scales?

No. The old implied `k` at 2.8b was `2.117734265875317`, which was not unusually small relative to the other models:

- 410m: `2.0616`
- 1.0b: `2.1870`
- 1.4b: `1.5661`
- 2.8b: `2.1177`

This argues against the original collapse being explained simply by raw-alpha under-scaling at 2.8b.

### Did norm-relative alpha rescue the 2.8b steering effect?

**No.**

Although 2.8b improved from `0.0000422820448875` to `0.001017189584672451`, that test-set margin is still near zero in absolute terms and still far below the smaller models’ new margins (`0.004318...` to `0.053391...`).

## Success Criteria Check

The run met the experiment’s stated goals:

- produced `norm_relative_scaling_summary.csv` with 4 model rows and required columns
- produced `pilot_k_sweep.csv` with at least 4 tested `k` values per model
- saved disjoint pilot and test prompt artifacts
- made a direct conclusion between the two paper-facing interpretations
- reported the numeric old vs new 2.8b margin clearly

## Final Conclusion

The evidence supports **collapse persists after normalization**, not “pure methodology artifact.”

Paper-facing conclusion:

- Hidden-state norms do increase with model size.
- But the old fixed-alpha setup did **not** give 2.8b an unusually tiny relative perturbation.
- Norm-relative pilot tuning increases the 2.8b margin somewhat, but only from `0.0000423` to `0.001017`, which remains far below the smaller models.
- Therefore, the 2.8b collapse is **not rescued** by correcting alpha for hidden-state norm scaling.

## Limitations and Follow-up Questions

- The conclusion is about this specific steering setup: same sentiment direction construction, same intervention layer choice inherited from the accepted run, same decoding, and same metric. It does not rule out that larger models might respond better under a different intervention layer, token position, or direction-construction method.
- The pilot-selected best `k` varied substantially across models (`0.4` to `6.4`), suggesting the response surface is not simple and may differ qualitatively by scale.
- One smaller model, 1.0b, actually worsened under norm-relative tuning on test, which suggests pilot-selected `k` may be somewhat noisy or that the margin-vs-alpha relationship is unstable.
- The 2.8b gain was large in relative terms (~24x) but tiny in absolute terms; understanding why the absolute margin stays suppressed remains an open mechanistic question.
- A natural next step would be to test whether larger models need changes beyond alpha scaling alone, such as retuning intervention layer, applying multi-token or multi-layer steering, or rebuilding the semantic direction specifically for each scale.

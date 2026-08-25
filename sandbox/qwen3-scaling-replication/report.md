# Qwen3 family replication of Claims A and B using the reused Qwen pipeline

## Research Question

This run asked whether the earlier replication pipeline that already worked on **Qwen2.5-1.5B base** would also replicate the paper’s two main claims on two additional **Qwen3 base checkpoints**:

- **Qwen/Qwen3-1.7B-Base**
- **Qwen/Qwen3-4B-Base**

The goal was to extend the existing cross-architecture result into a **within-Qwen-family scaling comparison**.

The hypotheses were:

1. **Claim A** would replicate on both models: the learned probe should outperform the weak, token-identity, and rollout baselines on the same reused dataset and split.
2. **Claim B** would replicate on both models: at least one steering layer should show a positive mean steering margin with a 95% CI whose lower bound is above 0.
3. The two Qwen3 models might differ by scale, potentially revealing whether a larger same-family model gives stronger probing and/or stronger steering.

## Experimental Setup

### Reused pipeline

The run **reused the existing Qwen-specific replication pipeline directly**, rather than rewriting it. The coder reports that the following components were carried over unchanged from the earlier successful Qwen2.5-1.5B replication:

- prompt handling
- 549-usable prompt filtering from the reused 600-prompt source
- fixed split with **seed 42** into **383 train / 83 val / 83 test**
- ridge probe training/evaluation
- weak baselines
- token-identity baseline
- rollout baseline
- **all-MiniLM-L6-v2** semantic target
- norm-relative causal steering procedure
- CI computation
- plotting and summary artifact generation

This matters because the main purpose here was comparability with the earlier references, not a new pipeline.

### Explicit checkpoint confirmation

The run explicitly confirmed that the models used were the **BASE, non-instruct, non-chat** checkpoints:

- **Qwen/Qwen3-1.7B-Base**
- **Qwen/Qwen3-4B-Base**

The coder also notes these were confirmed as distinct from the chat repos.

### Model-specific adaptations

Only thin Qwen3 adapter changes were made. Reported differences from the Qwen2.5 adapter were:

- model name/tag parameterization via environment variables
- reading **layer count** and **hidden size** generically from `model.config`
- confirming that the residual-stream injection path remained `model.model.layers[L-1]`
- confirming that the logit-lens norm module remained `model.model.norm`
- reading per-model steering candidate layers and probe/logit-lens layer from each model’s own results rather than hardcoding

The coder explicitly reports that the hook location and norm path were **identical to Qwen2.5** after inspecting the Qwen3 architecture code.

## Per-Cycle Execution Summary

### Cycle 1: Full run with one-model-at-a-time workflow

The approved run completed in a single successful cycle, though it resumed from a partially completed earlier failed attempt.

What happened:

1. The coder reused the validated prior Qwen pipeline.
2. They recovered a partially completed prior attempt for **Qwen3-1.7B**, which had already finished Claim A and most of the Claim B steering sweep before interruption.
3. They killed an orphaned leftover process that was still occupying GPU 0.
4. They resumed and completed the interrupted **Qwen3-1.7B** steering run.
5. After saving metrics and plots, they performed cleanup.
6. They then ran the full **Qwen3-4B** Claim A and Claim B pipeline from scratch.

This preserved the requested **GPU 0 only, one model fully at a time, cleanup before the next model** workflow.

## Protocol Details

### Shared data and split

Both models used the same:

- reused 600-prompt source set
- **549 usable prompts** after filtering
- **383 / 83 / 83** train/val/test split
- **seed 42**
- **all-MiniLM-L6-v2** semantic target

### Claim A protocol

For each model, the reused pipeline evaluated:

- learned probe
- weak baseline
- token-identity baseline
- rollout baseline

The objective replication criterion was that the **probe test cosine** exceed each of the baselines for that same model.

### Claim B protocol

For each model, the reused norm-relative causal steering protocol performed a steering-specific multi-layer scan with:

- at least 5 candidate layers spanning early/mid/late depth
- **5 seeds per layer**
- **95% confidence intervals**

The objective replication criterion was that at least one scanned layer have:

- positive mean steering margin, and
- **95% CI lower bound > 0**

## Results

## Claim A: Probe vs baselines

### Qwen3-1.7B-Base

Claim A metrics:

- **Probe test cosine:** 0.42738377733735833
- **Weak baseline best:** 0.35796793064029014
- **Token-identity baseline best:** 0.27330813288643524
- **Rollout-m1 baseline:** 0.14916101805246976

Replication judgment:

- The probe beat **all three** baselines.
- **Claim A replicated** on Qwen3-1.7B-Base.

### Qwen3-4B-Base

Claim A metrics:

- **Probe test cosine:** 0.4093715060154242
- **Weak baseline best:** 0.3211177613735958
- **Token-identity baseline best:** 0.2746024634654634
- **Rollout-m1 baseline:** 0.1411399840801563

Replication judgment:

- The probe beat **all three** baselines.
- **Claim A replicated** on Qwen3-4B-Base.

### Comparison to references

Provided references:

- **Pythia-1.4B probe cosine:** 0.4282612058059962
- **Qwen2.5-1.5B probe cosine:** 0.4283822861654986

Comparison:

- **Qwen3-1.7B** probe cosine (**0.4274**) is essentially aligned with both references (**0.4283**, **0.4284**).
- **Qwen3-4B** probe cosine (**0.4094**) is lower than both references, but still clearly above all baselines and therefore still a successful Claim A replication.

## Claim B: Norm-relative causal steering

### Qwen3-1.7B-Base

Scanned layers:

- **4, 10, 14, 20, 24, 28**

Best steering result:

- **Best layer:** 14
- **Best mean steering margin:** 0.03058764087036252
- **95% CI:** [0.021088826503538625, 0.040086455237186416]

Passive-best result:

- **Passive-best layer:** 24
- **Passive-best margin:** 0.004983733012340963

Replication judgment:

- The best layer had a positive mean margin.
- The CI lower bound was above 0.
- **Claim B replicated** on Qwen3-1.7B-Base.

### Qwen3-4B-Base

Scanned layers:

- **5, 13, 18, 26, 31, 32, 36**

Best steering result:

- **Best layer:** 31
- **Best mean steering margin:** 0.0740326458006166
- **95% CI:** [0.07038233811523019, 0.077682953486003]

Passive-best result:

- **Passive-best layer:** 32
- **Passive-best margin:** 0.06138396704918705

Replication judgment:

- The best layer had a positive mean margin.
- The CI lower bound was above 0.
- **Claim B replicated** on Qwen3-4B-Base.

### Comparison to references

Provided references:

- **Pythia-1.4B best steering margin:** about 0.0404
- **Qwen2.5-1.5B best steering layer:** 28
- **Qwen2.5-1.5B best steering margin:** 0.1283140833955258

Comparison:

- **Qwen3-1.7B** best margin (**0.0306**) is slightly below the Pythia reference (**~0.0404**) and well below Qwen2.5-1.5B (**0.1283**), but still positive and statistically reliable.
- **Qwen3-4B** best margin (**0.0740**) is above the Pythia reference and below Qwen2.5-1.5B, placing it between those two references.

## Side-by-Side Summary

| Model | Base checkpoint confirmed | Claim A probe cosine | Weak baseline | Token-identity | Rollout | Claim A replicated? | Scanned layers | Best steering layer | Best steering margin | 95% CI | Passive-best layer | Passive-best margin | Claim B replicated? |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|---|
| Pythia-1.4B reference | not part of this run | 0.4282612058059962 | — | — | — | reference | — | — | ~0.0404 | — | — | — | reference |
| Qwen2.5-1.5B reference | prior base run | 0.4283822861654986 | — | — | — | reference | — | 28 | 0.1283140833955258 | — | 24 | 0.0461 | reference |
| Qwen3-1.7B-Base | yes | 0.42738377733735833 | 0.35796793064029014 | 0.27330813288643524 | 0.14916101805246976 | yes | 4, 10, 14, 20, 24, 28 | 14 | 0.03058764087036252 | [0.021088826503538625, 0.040086455237186416] | 24 | 0.004983733012340963 | yes |
| Qwen3-4B-Base | yes | 0.4093715060154242 | 0.3211177613735958 | 0.2746024634654634 | 0.1411399840801563 | yes | 5, 13, 18, 26, 31, 32, 36 | 31 | 0.0740326458006166 | [0.07038233811523019, 0.077682953486003] | 32 | 0.06138396704918705 | yes |

## Did the Success Criteria Pass?

Yes. The run met the requested criteria:

1. Both checkpoint IDs were explicitly confirmed as **base/non-instruct**.
2. Both models produced Claim A outputs on the fixed reused dataset/split.
3. Both models produced Claim B scans over at least 5 layers, with 5 seeds and 95% CIs.
4. Final comparison artifacts included the two new models plus the two references.
5. The run made binary replication judgments for both claims on both models.
6. **Claim A objective threshold passed** for both models: probe cosine exceeded weak, token-identity, and rollout baselines.
7. **Claim B objective threshold passed** for both models: at least one scanned layer had positive mean margin with CI lower bound above 0.
8. Cleanup logs confirmed the requested one-model-at-a-time workflow.

## Same-Family Scaling Comparison: Qwen3-1.7B vs Qwen3-4B

The most interesting outcome is that scaling within the Qwen3 family did **not** move both claims in the same direction.

### For Claim A

The larger model was **not** better on probe cosine:

- **Qwen3-1.7B:** 0.4274
- **Qwen3-4B:** 0.4094

So for the passive probing metric, the 1.7B model was actually closer to the earlier Pythia and Qwen2.5 reference values.

### For Claim B

The larger model was clearly stronger on steering:

- **Qwen3-1.7B best margin:** 0.0306
- **Qwen3-4B best margin:** 0.0740

The coder’s summary also notes that the 4B model showed a broader pattern of positive layers, whereas the 1.7B model’s steering effect was weaker and more localized.

### Layer-location pattern

The best steering layer locations also differed:

- **Qwen3-1.7B best steering layer:** 14 out of 28 layers, about mid-depth
- **Qwen3-4B best steering layer:** 31 out of 36 layers, late depth

The passive-best layer nearly matched the best steering layer for Qwen3-4B (**31 vs 32**), but not for Qwen3-1.7B (**14 vs 24**). That suggests stronger alignment between passive probe structure and causal steering location in the larger model.

## Deviations, Issues, and Fixes During the Run

Two real implementation issues were discovered and fixed.

### Cleanup bug

The cleanup script originally used a faulty cache-directory name transformation:

- `tr '/' '--'`

This silently produced the wrong Hugging Face cache name and skipped actual deletion of the weights cache. The coder fixed it to:

- `sed 's#/#--#g'`

They then manually corrected the already-run 1.7B cleanup and report that this removed a missed **3.3 GB** cache. The fixed script then correctly deleted the 4B cache, reported as **7.6 GB**.

### Report-generation substring bug

A second bug affected report building:

- matching `"4B" in model_name` caused **Pythia-1.4B** to be confused with **Qwen3-4B-Base** in one section of the generated report

This was fixed and the report regenerated.

These bugs do not appear to invalidate the final metrics, but they are worth documenting because they affected workflow verification and reporting integrity.

## Limitations

Several limitations remain.

1. **Only two new Qwen3 base models were tested.** That is enough for the requested within-family comparison, but not enough to establish a smooth scaling law.
2. **Qwen3-4B underperformed Qwen3-1.7B on probe cosine.** This shows that same-family scaling is not monotonic for Claim A in this setup, but the run does not explain why.
3. **Claim B effect sizes varied substantially across models.** Qwen3-1.7B replicated only modestly, while Qwen3-4B was stronger and cleaner, and Qwen2.5-1.5B remained stronger still.
4. The report history gives only the final summary numbers, not the full per-layer tables in text here, so interpretation of the full layerwise profile depends on the saved CSV/plot artifacts.
5. The Qwen3-1.7B results were partially resumed from an interrupted earlier attempt. The final accepted run states that completion and verification were done correctly, but this is still operationally messier than a fresh uninterrupted run.

## Overall Conclusion

Using the **reused Qwen pipeline** with only minimal model-adapter changes, both requested **Qwen3 BASE** checkpoints were successfully evaluated:

- **Qwen/Qwen3-1.7B-Base**
- **Qwen/Qwen3-4B-Base**

### Final replication judgments

- **Claim A replicated on Qwen3-1.7B-Base:** yes
- **Claim A replicated on Qwen3-4B-Base:** yes
- **Claim B replicated on Qwen3-1.7B-Base:** yes
- **Claim B replicated on Qwen3-4B-Base:** yes

### Interpretation

The cross-architecture finding extends cleanly to these additional Qwen-family models: both show probe superiority over baselines and at least one statistically reliable positive steering layer.

But the same-family scaling story is mixed:

- **Qwen3-1.7B** had the stronger passive probe cosine, nearly identical to the earlier Pythia and Qwen2.5 references.
- **Qwen3-4B** had the stronger causal steering effect, with a best margin between the Pythia and Qwen2.5 reference strengths.

So the main qualitative claims replicate on both new models, but scaling within Qwen3 does **not** produce a simple monotonic improvement across both Claim A and Claim B metrics.

## Follow-Up Questions

1. Why does **Qwen3-4B** have lower probe cosine than **Qwen3-1.7B** while showing stronger steering?
2. Would additional Qwen3 sizes clarify whether this is a noisy two-point result or a real family-level pattern?
3. Is the stronger alignment of passive-best and steering-best layers in **Qwen3-4B** a reliable sign of more causally usable representations?
4. Would repeating the same protocol on more seeds or more layer candidates change the relative ordering of Qwen3-1.7B and Qwen3-4B?
5. Since Qwen2.5-1.5B still showed the largest steering margin among the Qwen-family results referenced here, what architectural or training differences between Qwen2.5 and Qwen3 might explain that gap?

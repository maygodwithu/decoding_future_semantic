# Probe vs. Greedy-Rollout Controls Across Longer Continuation Horizons in Pythia-1.4B

## Research Question

This run tested whether the earlier probe-vs-rollout result was mainly an artifact of short continuation targets.

In the prior accepted run on `EleutherAI/pythia-1.4b` with greedy decoding, a linear probe from the last prompt hidden state predicted the full continuation sentence embedding with test cosine **0.4283**, beating token/logit-lens controls (best **0.181**) but losing to short greedy-rollout controls: embedding the model’s own next **3** tokens gave **0.476**, and next **5** tokens gave **0.637**. The main hypothesis for this follow-up was that those rollout baselines were advantaged because the earlier continuation targets were short (mean about 16 tokens after sentence-boundary cutting), so even a 3–5 token rollout already exposed a large fraction of the target semantics.

The question here was:

- as the target continuation horizon increases, does the probe degrade more slowly than rollout controls?
- is there a horizon where the probe beats **all tested rollout controls**?
- if not, does the gap at least narrow enough to support the interpretation that rollouts win mainly by directly copying a short prefix of the target?

## Reused Pipeline and Experimental Setup

This run reused the earlier accepted pipeline rather than rebuilding from scratch. The reused components included the same data-processing, hidden-state extraction, embedding, and probe-training scripts.

Key choices preserved for comparability:

- **Model:** `EleutherAI/pythia-1.4b`
- **Decoding:** greedy
- **Hidden state source:** last-prompt-token hidden state from a prompt-only forward pass
- **Candidate layers checked in the reused pipeline:** 4, 8, 12, 20
- **Probe type:** ridge regression linear probe
- **Sentence embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dataset base:** the reused original **600-prompt** set
- **GPU:** GPU 0 only

To keep compute and disk use conservative, the run did **one** greedy generation pass to `max_new_tokens=96` and then derived shorter horizons by truncation. This ensured the horizon conditions were nested prefixes of the same continuation.

## Protocol

### Horizon conditions

The run evaluated three fixed continuation horizons:

- **H=16**
- **H=48**
- **H=96**

These were chosen to cover:

- a short condition comparable to the earlier approximately 16-token setting,
- a medium condition around 50 tokens,
- a longer condition around 100 tokens.

### Prompt set and filtering

- Total prompts: **600**
- Usable prompts: **600**
- Test split size: **90**

The logs note a **>=16 realized-token filter**, under which all 600 prompts were usable. Only **1.5%** of continuations hit EOS before 96 tokens.

### Targets and controls

For each horizon `H`, the target was the embedding of the full generated continuation truncated to the first `H` tokens.

For each same target, rollout controls were computed by embedding only the first `m` generated tokens, for:

- `m=3`
- `m=5`
- `m=10`
- `m=20`

The metric throughout was cosine similarity on the **test** set between:

- probe prediction and true full-continuation embedding, or
- rollout-prefix embedding and true full-continuation embedding.

### Probe training details

The probe was trained separately for each horizon, using the same training/validation/test methodology and alpha grid as in the reused setup.

Observed selection results:

- **best layer:** 20 at all three horizons
- **validation-selected alpha:** `1e4` at all three horizons

## Results

### Main metric table

| Horizon H | Probe cosine | Rollout m=3 | Rollout m=5 | Rollout m=10 | Rollout m=20 | Best rollout | Best m | Probe - Best rollout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.438978 | 0.424675 | 0.609991 | 0.831929 | 0.938148 | 0.938148 | 20 | -0.499169 |
| 48 | 0.419148 | 0.337223 | 0.491810 | 0.678734 | 0.859946 | 0.859946 | 20 | -0.440798 |
| 96 | 0.406147 | 0.299540 | 0.436403 | 0.606032 | 0.779775 | 0.779775 | 20 | -0.373628 |

### Probe behavior across horizon

The probe weakened only modestly as the target horizon increased:

- **H=16:** 0.438978
- **H=48:** 0.419148
- **H=96:** 0.406147

So from 16 to 96 tokens, the absolute drop was about **0.0328**.

### Rollout behavior across horizon

Every rollout control degraded with longer horizons, and the degradation was larger than for the probe.

For example:

- `m=3`: **0.424675 → 0.337223 → 0.299540**
- `m=5`: **0.609991 → 0.491810 → 0.436403**
- `m=10`: **0.831929 → 0.678734 → 0.606032**
- `m=20`: **0.938148 → 0.859946 → 0.779775**

This is the core positive result of the run: **rollout controls do lose ground as the horizon grows**, which matches the motivating hypothesis.

## Did a Crossover Occur?

### Strict criterion: probe beats all tested rollout controls

No.

Under the pre-registered strict comparison against the **best tested rollout** at each horizon, there was **no crossover up to H=96**.

The key quantity was:

- **H=16:** probe - best rollout = **-0.499169**
- **H=48:** **-0.440798**
- **H=96:** **-0.373628**

So the deficit remained negative at every tested horizon.

The summary flag in the run artifacts was:

- `crossover_exists_strict_all_m = false`
- `crossover_horizon_strict_all_m = none`

### Partial crossover against weaker rollout baselines

Yes, but only against the shortest rollout.

The probe already beat the `m=3` rollout at every tested horizon:

- **H=16:** 0.438978 vs 0.424675, delta **+0.014303**
- **H=48:** 0.419148 vs 0.337223, delta **+0.081924**
- **H=96:** 0.406147 vs 0.299540, delta **+0.106607**

Against `m=5`, the probe still lost, but the gap nearly vanished by the longest horizon tested:

- **H=16:** delta **-0.171013**
- **H=48:** delta **-0.072662**
- **H=96:** delta **-0.030256**

So although there was no strict crossover against all tested rollouts, the probe was increasingly competitive as horizon increased.

## Gap-Narrowing Result

The run explicitly recorded that the gap to the best rollout **did narrow** from H=16 to H=96.

- Gap at H=16: **0.499169**
- Gap at H=96: **0.373628**
- Narrowing fraction: **0.251533**

That is, the probe erased about **25.15%** of its deficit to the best rollout over the tested horizon range.

This matters because it supports the main interpretation even though the strict crossover was not reached: **the probe’s semantic signal is more horizon-stable than rollout-prefix signal**, but not yet strong enough by 96 tokens to beat larger rollout controls like `m=10` or `m=20`.

## Interpretation

### What this says about the original hypothesis

The original idea was partly confirmed and partly not.

Confirmed:

- rollout controls are indeed helped by short targets;
- as targets get longer, rollout controls degrade faster than the hidden-state probe;
- the probe already outperforms the smallest rollout baseline (`m=3`) and nearly catches `m=5` by 96 tokens.

Not confirmed:

- there was **no full crossover** up to the longest tested horizon of **96 tokens**;
- larger rollout prefixes, especially `m=10` and `m=20`, remained substantially stronger in raw cosine.

### Why larger rollout controls still win

The clearest explanation is that `m=10` and especially `m=20` still embed a large literal chunk of the target continuation. Since the target is itself the first `H` generated tokens, a 20-token rollout is not just a weak cue about the future; it is a direct embedding of a sizable prefix of the exact target text. That gives it a strong advantage in this semantic-similarity metric.

This is most obvious at short horizon:

- at **H=16**, `m=20` effectively sees the entire target continuation and more, producing **0.938148**.

But even at **H=96**, 20 observed generated tokens are still a meaningful portion of the continuation, and `m=20` remains very strong at **0.779775**.

### Relation to Future Lens, ESP, and simple rollout baselines

This run helps sharpen the distinction.

- **Versus simple rollout baselines:** the probe is not just reproducing what a tiny rollout can already reveal. It beats `m=3` at all tested horizons and degrades more slowly as the target extends.
- **Versus stronger rollout baselines:** by raw semantic cosine, observed generated prefixes of length 10 or 20 still dominate up to 96 tokens, so the probe does not yet substitute for an actually generated partial future.
- **Relative to Future Lens / ESP-style claims:** the results support a modest but real latent-future signal in the prompt-side hidden state, not a signal strong enough in this setup to outperform all explicit rollout evidence. The probe seems to encode some future semantics beyond what very short prefix exposure provides, but this should not be overstated as beating direct rollout-based baselines in general.

A reasonable summary is: **there is future-semantic information in the hidden state that is more robust to horizon length than small rollout controls, but not enough by 96 tokens to surpass larger observed-prefix controls.**

## Compute-Normalized Secondary Finding

The run also noted a secondary, back-of-the-envelope compute interpretation.

- The **probe** requires the prompt forward pass plus a negligible linear map.
- A rollout control of length `m` requires roughly **m additional sequential decoding steps** beyond the prompt pass.

The logs report that rollout cosine-per-extra-decode-step falls with horizon, while the probe adds essentially **zero extra decode steps**. So even when rollout still wins on raw cosine, the probe can be argued to be **more compute-efficient per unit of recovered semantic signal**.

This was not developed into a full benchmark with wall-clock numbers, so it should be presented as a qualitative efficiency point rather than a definitive throughput result.

## Success Criteria and Outcome

The run met the planned success criteria.

It produced:

- three horizon conditions: **16, 48, 96**,
- probe and rollout metrics for **m=3,5,10,20** at each horizon,
- explicit `delta_best_rollout` values,
- a direct answer to the crossover question.

The final answer is:

- **No strict crossover exists up to 96 tokens** when comparing the probe to the best tested rollout baseline.
- The first strict crossover horizon was therefore **not observed**.
- However, the probe **does** beat `m=3` already, nearly catches `m=5` by H=96, and its deficit to the best rollout shrinks by about **25%** from H=16 to H=96.

## Limitations

Several limitations matter for interpreting the result.

1. **The longest tested horizon was 96 tokens.**  
   Since the probe-vs-`m=5` gap shrank to only **-0.030256** at H=96, a crossover against `m=5` may occur beyond the tested range. The logs mention a rough extrapolative suggestion around **125–135 tokens**, but that was not directly tested and should not be treated as an empirical result.

2. **Large rollout prefixes are intrinsically advantaged on this metric.**  
   A rollout embedding from the first 10 or 20 generated tokens is a semantic summary of literal target content. That makes raw cosine to the full continuation embedding a favorable metric for rollout baselines.

3. **Only one model and one decoding regime were tested.**  
   All results here are for `pythia-1.4b` under **greedy decoding**.

4. **The sentence embedding model may compress prefix and full-continuation semantics in a way that benefits rollouts.**  
   Since `all-MiniLM-L6-v2` maps text into a global embedding space, even a relatively short exact prefix may capture much of the full continuation’s embedding.

5. **The evaluation set was the fixed reused split with test n=90.**  
   This is enough for a controlled follow-up, but not large enough to settle subtle differences between close conditions with high precision.

## Follow-Up Questions

The most useful next steps are:

1. **Push to longer horizons beyond 96 tokens.**  
   This is the cleanest test of whether the near-crossover versus `m=5` becomes an actual crossover.

2. **Use horizon-relative rollout baselines.**  
   For example, fix rollouts to a smaller fraction of the target length, or compare against controls that cannot simply copy a large portion of the target.

3. **Add stronger compute-normalized analysis.**  
   A more explicit cost-vs-signal comparison could better support the claim that the probe is attractive even where raw rollout cosine remains higher.

4. **Replicate on other models or decoding schemes.**  
   The balance between latent future information and rollout-prefix advantage may differ under sampling or in larger models.

## Bottom Line

This run did **not** find a full probe-over-rollout crossover by **96 tokens**. The best rollout baseline remained stronger at every tested horizon.

But the central pattern was still informative: the **probe degrades slowly** with horizon, whereas all rollout controls degrade faster. The probe already beats the smallest rollout (`m=3`) and nearly catches `m=5` by the longest tested horizon. So the earlier short-target result was not just a complete failure of latent future prediction; rather, it reflected that short explicit rollouts are extremely strong when the semantic target is itself short and heavily overlapped with those observed tokens.

The conservative conclusion is that, in this setup, **hidden states contain nontrivial future-semantic information that becomes relatively more competitive as the horizon grows, but not yet enough to beat larger direct rollout baselines by 96 tokens.**

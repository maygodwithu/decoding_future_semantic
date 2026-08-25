# Follow-up on continuation-semantic recoverability: token-identity controls, Pythia scaling, and greedy vs sampled decoding

## Research Question

This follow-up reused the exact prompt set from the accepted prior run to test three user-prioritized questions:

1. **Token-identity control baseline (most important):** does the semantic probe recover continuation meaning beyond what is already available from simple next-token predictability?
2. **Model-size scaling across Pythia:** does semantic recoverability, and the causal steering effect derived from the probe, change with model size?
3. **Greedy vs sampled decoding:** does the result depend on greedy decoding, or does it remain under stochastic continuations?

The original accepted result on **EleutherAI/pythia-1.4b** under **greedy decoding** was reproduced exactly as the anchor point for comparison: at **layer 20**, a single hidden state at the last prompt token predicted the sentence embedding of the full continuation with **test cosine 0.4283** versus **mean baseline 0.2310** and **random 0.0616**, with **Recall@5 0.759** versus **chance 0.060**.

The central hypothesis for this follow-up was that last-prompt hidden states contain continuation-level semantic information that is not reducible to immediate token prediction alone, and that this signal should be visible across model scales and not disappear under sampled decoding.

## Reused Data and Reproducibility

The run reused the same prior prompt artifact:

- **600 prompts total**
- **549 usable prompts** after the same filtering logic as the accepted run
- the same split was reconstructed and matched the prior setup:
  - **383 train**
  - **83 validation**
  - **83 test**
- seed: **42**

Before running the new conditions, the pipeline verified that it could reproduce the previous 1.4b greedy result at layer 20 **bit-for-bit**. This matters because the whole point of the follow-up was direct comparability rather than using a fresh prompt sample.

## Experimental Protocol

A fresh pipeline was implemented around the reused prompt set. For each condition, it:

1. loaded a Pythia model on **GPU 0**,
2. extracted the hidden state at the **last prompt token**,
3. generated continuations,
4. embedded those continuations with the **same sentence embedding model** used previously,
5. trained/evaluated a **linear probe** from hidden state to continuation embedding,
6. evaluated cosine similarity and retrieval metrics,
7. where relevant, tested causal steering using the learned probe direction versus a random-direction control.

Disk usage was kept conservative: models were processed one at a time and caches for non-1.4b models were cleaned after use. The final run directory was reported as about **104 MB**, and all four planned scaling models fit on GPU 0.

## Priority 1: Token-Identity Control Baseline

### What was tested

For **pythia-1.4b, greedy decoding**, the follow-up compared the semantic probe against two families of controls derived from prompt-only information:

1. **Top-k token-text controls** from the same hidden state via logit-lens/unembedding-style next-token prediction.
   - Variants included concatenated or weighted text built from top-k predicted tokens.
2. **Short rollout controls** using the model’s own prompt-only continuation for only a few tokens.
   - In particular, short greedy continuations of length **m = 1, 3, 5** tokens were embedded and compared against the true full continuation embedding.

This directly addressed the objection that continuation-semantic prediction might simply be a repackaging of known next-token predictability.

### Main probe result

At the reproduced comparison point, **layer 20** on **pythia-1.4b greedy**:

- **Probe test cosine:** **0.4282612058059962**
- **Probe Recall@5:** **0.7349397590361446**

The run notes that the earlier accepted run had Recall@5 **0.759** at this same setting; the new pipeline verified reproduction before the expanded control experiments.

### Comparison to top-k token-text controls

The strongest top-k token-text control reached:

- **Best top-k control test cosine:** **0.1812326643494623**

So the semantic probe beat the best such control by:

- **Margin:** **0.24702854145653388**

The run summary states that the probe beat **all 6 top-k logit-lens controls** by roughly **0.25–0.32 cosine**.

### Comparison to short rollout controls

The strongest short rollout control was much stronger:

- **Short greedy rollout, m = 5, test cosine:** **0.637128441743362**
- **Probe minus short-rollout margin:** **-0.20886723593736584**

The run summary also reports that short greedy rollouts at **m = 3** and **m = 5** matched or beat the probe, with **m = 3** around **0.476** and **m = 5** around **0.637**.

### Interpretation

This is the most important finding of the run, and it is mixed:

- **Strengthens differentiation from very weak token-identity baselines:** yes. The semantic probe clearly beats top-k token-list/text controls derived from the same hidden state.
- **Weakens the stronger claim that the probe beats prompt-only token predictability in general:** yes. Once the control is allowed to use the model’s own first few greedily generated tokens, the control can outperform the probe.

So the follow-up does **not** support the strongest version of the claim that continuation semantics are recoverable in a way clearly beyond token-level prediction. Instead, it supports a narrower claim:

- the hidden state contains information that is much richer than a bare top-k next-token list,
- but for this dataset and setup, a few actual greedily decoded continuation tokens already carry enough information to beat the probe on continuation embedding prediction.

The run explicitly notes a likely reason: the continuations are fairly short, with mean length about **16 tokens** after the sentence-boundary cut, so a **3–5 token rollout** may already reveal a large fraction of continuation semantics.

### Verdict for Priority 1

**Overall effect on the original hypothesis: weakens it.**

More specifically:

- **Against simple Future Lens-style top-k token prediction:** the original claim looks stronger.
- **Against stronger prompt-only short-rollout controls:** the claim is weakened, because the semantic probe did **not** beat these controls.

## Priority 2: Model-Size Scaling Across Pythia

### Models tested

The scaling study successfully ran on all four planned models:

- **pythia-410m**
- **pythia-1.0b**
- **pythia-1.4b**
- **pythia-2.8b**

No fallback was needed.

### Probe performance across scale

Best probe test cosine by model:

- **410m:** **0.4260184860374043**
- **1.0b:** **0.4019609994779997**
- **1.4b:** **0.4282612058059962**
- **2.8b:** **0.4312234267626699**

The run summary also reports cosine gain over the mean baseline as approximately:

- **410m:** **0.195**
- **1.0b:** **0.178**
- **1.4b:** **0.197**
- **2.8b:** **0.207**

### Scaling interpretation for probe recoverability

Semantic recoverability was **present at all tested scales** and did **not collapse** at smaller or larger models. However, the scaling trend was modest rather than dramatic:

- performance is fairly flat from **410m** to **2.8b**,
- with a small dip at **1.0b**,
- and the best value at **2.8b** is only slightly above **1.4b**.

So the evidence suggests **stable semantic recoverability across this scale range**, with at most a **slight positive trend**, not a sharp scaling law.

### Steering across scale

A probe-derived semantic direction was compared to a random-direction control, using a steering test matched to the prior run’s methodology.

Reported steering margins (semantic direction minus random direction):

- **410m:** **0.00343199516646564**
- **1.0b:** **0.011274605058133602**
- **1.4b:** **0.008364170556887984**
- **2.8b:** **0.0000423**

All four scales had **positive** steering margins, so the semantic direction beat the random direction at every tested scale. But the margins were small, and the pattern with scale was not monotonic.

### Scaling interpretation for steering

The steering result says that the probe-derived direction remains somewhat causally meaningful across scales, but the effect size is **small** and **not clearly increasing with model size**.

In particular:

- the largest reported margin was at **1.0b** (**0.0113**),
- the **2.8b** margin was nearly zero (**0.0000423**), though still positive.

So the steering results support the existence of a signal, but not a strong claim that steering gets more powerful as models get larger over this range.

### Verdict for Priority 2

**Overall effect on the original hypothesis: mostly leaves it unchanged to slightly strengthens it.**

Why:

- semantic recoverability above baseline appears **robust across 410m–2.8b**,
- probe quality is **roughly stable** with perhaps a small upward drift,
- causal steering remains **positive relative to random** at all tested sizes,
- but neither probe quality nor steering showed a strong monotonic scale-up.

## Priority 3: Greedy vs Sampled Decoding

### What changed in protocol

For **pythia-1.4b**, a second extraction was run with sampled decoding:

- **temperature = 0.8**
- **top_p = 0.95**
- fixed **seed = 42**

The hidden state source stayed the same: the last prompt token before generation. The same train/test split and probe setup were used so the greedy and sampled results were directly comparable.

### Results

For **greedy decoding**:

- **Test cosine:** **0.4282612058059962**

For **sampled decoding**:

- **Test cosine:** **0.3507694629728986**
- **Sampled mean baseline cosine:** **0.20420030874432446**
- **Sampled Recall@5:** **0.6987951807228916**

The run summary states that under sampling, the probe still beat the top-k logit-lens controls by a similar margin to greedy, while **short-rollout controls still beat the probe**.

### Interpretation

Sampling clearly makes the task harder:

- cosine drops from **0.4283** to **0.3508**, a decrease of about **0.0775**.

But the effect does **not** disappear:

- sampled probe cosine remains well above the sampled mean baseline (**0.3508 vs 0.2042**),
- Recall@5 remains high at **0.6988**,
- and the probe still beats the weaker top-k token-text controls.

So the result is **not merely a greedy-decoding artifact**. However, the same caveat from Priority 1 remains: if one allows short prompt-only rollout controls, those still outperform the probe even in the sampled setting.

### Verdict for Priority 3

**Overall effect on the original hypothesis: strengthens it, but only partially.**

It strengthens the claim that continuation-level semantic recoverability is not exclusive to greedy decoding. But it does **not** remove the concern that short rollout token information may explain much of the recoverable signal.

## Overall Assessment

The follow-up gives a more precise and more qualified picture than the original run.

### What held up

1. **Direct reproducibility:** the original 1.4b greedy result was reproduced on the reused prompt set.
2. **Semantic probe beats weak token-identity controls:** against top-k token-text baselines from the same hidden state, the probe wins by a large margin.
3. **Cross-scale robustness:** semantic recoverability is present from **410m** through **2.8b**.
4. **Not just greedy:** the effect persists under **temperature 0.8 / top-p 0.95** sampled continuations.
5. **Causal signal remains detectable:** probe-derived steering beats random-direction steering at all tested scales, though weakly.

### What was weakened

1. **The strongest anti-token-identity claim failed.** A short prompt-only greedy rollout baseline, especially **5 tokens**, outperformed the semantic probe (**0.6371 vs 0.4283** cosine).
2. **Scaling is mild, not dramatic.** There is no strong monotonic increase in probe quality or steering strength with model size.
3. **Steering effect sizes are small.** Positive margins exist, but they are limited and nearly vanish at **2.8b**.

## Did the Success Criteria Get Met?

Yes.

The run produced the required outputs and answered all three requested priorities:

- **Priority 1:** token-identity control comparisons were completed, including honest reporting that short rollout controls beat the probe.
- **Priority 2:** scaling results were produced for **4** Pythia sizes with probe and steering summaries.
- **Priority 3:** greedy vs sampled comparisons were produced for **pythia-1.4b**.

As an experiment, this was a success. As a substantive defense of the strongest original claim, the outcome is mixed.

## Limitations

1. **Short continuations make rollout controls unusually powerful.** If average continuations are only about **16 tokens**, then seeing the first **3–5** tokens may reveal a large share of the semantic content.
2. **Only one prompt set was used.** This was necessary for direct comparability, but it limits how confidently the findings generalize.
3. **Sampling was tested only on 1.4b.** That answers the user’s minimum request, but not whether decoding robustness itself scales with model size.
4. **Steering metrics were small.** Positive margins over random are encouraging, but they are not large enough to support strong causal claims without additional replication.
5. **The report summary provides only some control numbers explicitly.** The full CSV artifacts contain the detailed per-control comparisons, but not all of those values are surfaced in the cycle summary.

## Follow-up Questions

1. **Longer continuations:** does the semantic probe regain an advantage over short-rollout controls if continuations are substantially longer?
2. **Fixed-information controls:** how does the probe compare to controls constrained by equal text budget, such as only 1–2 rollout tokens, or a matched-information compressed token baseline?
3. **Alternative target representations:** does the same pattern hold if continuation meaning is measured with different embedders or with more local semantic labels rather than a sentence embedding?
4. **Scale and decoding jointly:** does sampled-decoding recoverability stay stable across 410m–2.8b, or does it degrade differently by scale?
5. **Stronger causal tests:** can one design steering evaluations with larger and more interpretable effect sizes than the small margins seen here?

## Bottom Line

This follow-up **partly strengthens** and **partly weakens** the original story.

- It **strengthens** the claim that last-prompt hidden states contain continuation-level information that survives across model sizes and under sampled decoding, and that this signal is richer than a simple top-k next-token list.
- It **weakens** the stronger claim that the semantic probe captures something clearly beyond prompt-only token predictability in a practically important sense, because **short rollout controls beat the probe**.

A fair summary is:

> continuation semantics are recoverable from a single last-prompt hidden state, but in this dataset that recoverability is not cleanly separable from the information already exposed by the model’s own first few continuation tokens.

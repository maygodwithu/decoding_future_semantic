# Qwen Base Cross-Architecture Replication of Claims A and B

## Research Question

This run tested whether two core findings from the paper, originally established within the Pythia family, also hold on a different architecture: a Qwen base model in the 1–2B parameter range.

The two claims checked were:

- **Claim A:** A linear probe from a single early/intermediate hidden state can predict the sentence embedding of the model’s eventual continuation better than weak baselines and token-identity/logit-lens baselines, but it should still lose to a short greedy-rollout baseline at short target horizons.
- **Claim B:** Causal semantic steering using the probe-derived direction should produce a positive effect relative to a random-direction control, with a **95% confidence interval excluding zero**.

A further methodological lesson from the earlier Pythia work was incorporated into the steering test: **the best layer for passive probing may not be the best layer for steering**, so steering was evaluated on a small layer scan rather than only at the passive-best layer.

## Model and Setup

The model used was **`Qwen/Qwen2.5-1.5B`**.

This was explicitly confirmed to be a **base, non-instruction-tuned causal LM**, which is important because the paper is about raw next-token-prediction dynamics rather than instruction-following behavior.

Key setup details:

- Model family: **Qwen2.5 base**
- Parameters: **1,543,714,304**
- Layers: **28**
- Hidden size: **1536**
- Prompt set: same paper prompt set, **600 raw prompts**, of which **549 were usable** in this run
- Split: **383 train / 83 validation / 83 test**
- Target sentence embeddings: **`all-MiniLM-L6-v2`**
- Execution: **GPU 0 only**
- Methodology: reused the existing paper pipeline for passive probing, baselines, norm-relative steering, multi-seed evaluation, and layer scan

The only meaningful deviation from the idealized protocol was that **549/600 prompts** were usable after preprocessing/filtering, which was documented in the run outputs.

## Experimental Protocol

### Cycle 1: Passive probe replication and baseline comparison

The first part of the run reproduced the paper’s passive semantic prediction setup on Qwen.

What was done:

- A **single-layer linear probe** was fit from one hidden state to the continuation sentence embedding target.
- A **layer sweep** was run to find the best passive-probing layer.
- The best probe was compared against the same classes of baselines used in the prior pipeline:
  - mean-embedding baseline
  - random-match baseline
  - lexical TF-IDF baseline
  - token-identity/logit-lens baselines
  - greedy rollout embedding baselines for **m = 1, 3, 5, 10** generated tokens

Why this matched the paper’s logic:

- Claim A is specifically about whether hidden-state semantics beat weak baselines and token-identity controls, while still being worse than a short direct rollout baseline at short horizons.
- Using the same prompt set and sentence-embedding target was necessary for comparability to the Pythia results.

### Cycle 1: Steering replication with layer scan

The second part of the same run tested causal steering.

What was done:

- Instead of only steering at the passive-best layer, the run scanned **5 candidate layers** spanning the network:
  - **4, 10, 14, 24, 28**
- This included the passive-best layer (**24**) plus earlier and later layers.
- At each layer, steering was evaluated with:
  - **norm-relative alpha tuning**
  - **5 seeds per layer**
  - comparison between **probe-derived direction** and **random-direction control**
  - a **95% CI** over seeds

Why the protocol changed from a naive one-layer steering test:

- Prior Pythia results had shown that the passive-best layer can be a poor steering layer.
- So, to fairly test Claim B on Qwen, the run intentionally looked for the **best steering layer**, not just the best passive layer.

## Results

## Claim A: Passive semantic probing vs baselines

### Best probe result

- **Best probe layer:** **24**
- **Held-out test cosine:** **0.4284**

This is essentially identical to the Pythia-1.4B reference:

- **Pythia-1.4B probe cosine:** **0.428**

### Weak baselines and token-identity baselines

Held-out cosine values:

- **Mean-embedding baseline:** **0.2335**
- **Random-match baseline:** **0.0479**
- **TF-IDF baseline:** **0.3512**
- **Logit-lens top-k=1 concat:** **0.1710**
- **Logit-lens top-k=5 concat:** **0.2271**
- **Logit-lens top-k=10 concat:** **0.2652**

Comparison to the best probe (**0.4284**):

- Probe beat mean-embedding by **0.1949**
- Probe beat random-match by **0.3805**
- Probe beat TF-IDF by **0.0772**
- Probe beat logit-lens top-k=10 by **0.1632**

So the Qwen probe clearly beat all reported weak baselines and all reported token-identity/logit-lens baselines.

### Rollout baselines

Held-out rollout cosine values:

- **m=1:** **0.1508**
- **m=3:** **0.4434**
- **m=5:** **0.6154**
- **m=10:** **0.8536**

The best reported rollout was:

- **Best rollout:** **m=10**, cosine **0.8536**

For direct comparison to the paper’s headline reference:

- **Qwen m=5 rollout cosine:** **0.6154**
- **Pythia-1.4B best m=5 rollout cosine:** **0.637**

Interpretation:

- The rollout baseline was **below** the probe at **m=1**.
- It **overtook** the probe already at **m=3** (**0.4434 > 0.4284**).
- At **m=5**, rollout was substantially stronger (**0.6154 vs 0.4284**).

This matches the intended qualitative pattern of Claim A: the probe contains substantial semantic information and beats weak controls, but a short greedy rollout remains stronger once a few generated tokens are available.

## Claim B: Causal semantic steering

### Layers scanned

The run evaluated steering at **5 candidate layers**:

- **Layer 4**
- **Layer 10**
- **Layer 14**
- **Layer 24** (passive-best layer)
- **Layer 28**

Each layer used **5 seeds** and norm-relative alpha tuning.

### Steering margins by layer

Reported mean steering margin and 95% CI:

- **Layer 4:** **-0.0029**, CI **[-0.0072, 0.0014]**
- **Layer 10:** **0.0001**, CI **[-0.0020, 0.0023]**
- **Layer 14:** **0.0012**, CI **[-0.0059, 0.0083]**
- **Layer 24:** **0.0461**, CI **[0.0393, 0.0530]**
- **Layer 28:** **0.1283**, CI **[0.1230, 0.1337]**

### Best steering layer

- **Best steering layer:** **28**
- **Best mean steering margin:** **0.1283**
- **95% CI:** **[0.1230, 0.1337]**

This CI clearly excludes zero.

At the best layer, the component effects were:

- **Probe-direction semantic effect:** **0.1301**
- **Random-direction control effect:** **0.0017**

So the observed improvement was not just a generic activation perturbation effect; it was much larger for the semantically derived direction than for the matched-norm random control.

### Comparison to Pythia reference

The paper’s Pythia-1.4B steering reference was approximately:

- **Pythia-1.4B best steering margin:** **~0.04**

Qwen comparisons:

- **Layer 24 steering margin:** **0.0461**, very close to and slightly above the Pythia reference
- **Layer 28 steering margin:** **0.1283**, about **3.2×** the Pythia reference

### Passive-best layer vs steering-best layer

A key outcome is that the best steering layer was **not** the passive-best layer:

- **Passive-best probing layer:** **24**
- **Best steering layer:** **28**

This reproduces the earlier methodological lesson that passive semantic decodability and causal steerability need not peak at the same depth.

## Side-by-Side Comparison with Pythia-1.4B

| Metric | Pythia-1.4B reference | Qwen2.5-1.5B result | Comparison |
|---|---:|---:|---|
| Best probe cosine | 0.428 | 0.4284 | Essentially identical |
| Best rollout cosine at m=5 | 0.637 | 0.6154 | Slightly lower on Qwen |
| Best steering margin | ~0.04 | 0.1283 | Much larger on Qwen |
| Passive-best layer | not the focus here | 24 | — |
| Steering-best layer | paper lesson: may differ | 28 | Differs from passive-best |

## Did the Success Criteria for the Claims Replicate?

### Claim A

**Yes, Claim A replicated on Qwen.**

Reason:

- The best probe cosine was **0.4284**.
- It beat all weak baselines:
  - mean embedding **0.2335**
  - random match **0.0479**
  - TF-IDF **0.3512**
- It beat all reported token-identity/logit-lens baselines:
  - top-k1 **0.1710**
  - top-k5 **0.2271**
  - top-k10 **0.2652**
- A short rollout baseline overtook it once a few tokens were generated:
  - **m=3: 0.4434 > 0.4284**
  - **m=5: 0.6154 > 0.4284**

This is the same qualitative pattern as the Pythia result and nearly the same absolute probe performance.

### Claim B

**Yes, Claim B replicated on Qwen.**

Reason:

- At least one scanned layer had a **positive steering margin with a 95% CI excluding zero**.
- In fact, **two layers** did:
  - Layer 24: **0.0461**, CI **[0.0393, 0.0530]**
  - Layer 28: **0.1283**, CI **[0.1230, 0.1337]**
- At the best layer, the semantic steering effect (**0.1301**) was far larger than the random-direction control (**0.0017**).

This satisfies the causal-steering replication criterion cleanly.

## Interpretation

The main conclusion is that **both Claim A and Claim B generalized from Pythia to a different model family**.

What is strongest here:

- The **probe result is almost numerically identical** to the Pythia-1.4B reference (**0.4284 vs 0.428**), which is a notably strong replication for Claim A.
- The same **rank ordering** held: probe beats weak and token-identity baselines, but loses to short rollout once enough generated tokens are included.
- Steering was not only present on Qwen, but **substantially stronger than the Pythia headline reference** at the best layer.
- The run also reproduced the practical lesson that **steering-specific layer selection matters**.

Overall, this **does strengthen confidence that the findings are general rather than Pythia-specific**.

## Limitations

Several limitations should be kept in mind.

- This was a replication on **one** non-Pythia architecture, not a broad architecture sweep.
- Only **549 of 600 prompts** were usable in this run, though this was documented and the run was still accepted as satisfying the protocol.
- The rollout comparison is somewhat sensitive to horizon choice. For example, Qwen’s **m=5** rollout (**0.6154**) was slightly below the Pythia reference (**0.637**), while **m=10** became extremely strong (**0.8536**).
- Claim C was not itself fully re-run as a standalone study; rather, its lesson was operationalized by scanning steering layers. The result does support that lesson, but it is not a full reproduction of the original six-experiment arc.
- The report here relies on the pipeline’s existing effect metric and CI machinery rather than introducing new diagnostics.

## Follow-Up Questions

Useful next steps would be:

- Repeat the same Claim A/B check on **additional base architectures** to determine whether Qwen is representative or unusually favorable for steering.
- Test whether the very strong Qwen steering at **layer 28** is robust across prompt subsets or target-horizon choices.
- Compare whether the **gap between passive-best and steering-best layers** is systematic across families.
- Investigate why Qwen’s passive probe score matches Pythia almost exactly, while its best steering effect appears considerably larger.

## Final Verdict

**Claims A and B both replicated on `Qwen/Qwen2.5-1.5B`, a confirmed base non-instruct model.**

- **Claim A:** replicated
- **Claim B:** replicated
- **Cross-architecture confidence:** **strengthened**

The replication is especially convincing because it preserves the main qualitative structure of the Pythia findings while showing that the causal steering effect is not confined to the original model family.

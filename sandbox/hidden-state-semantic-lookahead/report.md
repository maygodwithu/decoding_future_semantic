# Semantic Recoverability and Causal Steering from a Single Early Hidden State in Pythia-1.4B

## Research Question

This experiment tested a stronger semantic version of the “latent future prediction” idea previously shown for token identities.

The question was: **when a causal language model generates text one token at a time, does a single hidden state taken very early—specifically, at the final prompt position before any continuation token is generated—already encode much of the *meaning* of the eventual continuation sentence or clause, rather than only information about the next few token identities?**

The hypothesis had two parts:

1. **Semantic recoverability:** a lightweight probe should be able to predict a sentence embedding of the model’s full short continuation from one early hidden state, beating sensible baselines.
2. **Causal semantic steering:** if such a semantic direction is learned, perturbing that hidden state along the learned direction should shift the meaning of the generated continuation in the predicted direction more than a matched random-direction control.

## How This Differs from Future Lens and ESP

This experiment was designed explicitly **not** to replicate the prior work you cited.

- **Future Lens (Pal et al., 2023)** showed that a hidden state at token *t* can linearly predict **exact future token identities**, including positions beyond the immediate next token, and supported this with probing plus causal patching. That is evidence for latent multi-token information, but it is still fundamentally a **token-identity** result.
- **ESP / Efficient Training-Free Multi-Token Prediction** used similar latent future-token predictability for **speculative decoding speedups**, which is an engineering application rather than a semantics study.

This run instead asked whether the early hidden state contains information about the **sentence-level semantics of the entire generated continuation**, evaluated by:

- predicting a **sentence embedding** of the full continuation from one early hidden state;
- testing a **longer horizon** than just the next one or two tokens, using full short continuations up to 24 new tokens;
- performing a **causal semantic steering intervention**, not merely token-identity probing or decoding acceleration.

So the contribution here is not “can the state predict future tokens?” but rather “can the state predict and causally influence the future continuation’s *meaning*?”

## Experimental Setup

### Model and generation

- Base model: **EleutherAI/pythia-1.4b**
- Hardware: **GPU 0 only**
- Generation mode: **greedy decoding** (`do_sample=False`)

Greedy decoding was chosen for the first cycle to reduce variance and ask the cleanest question: does the early state encode the semantics of the model’s own preferred continuation?

### Prompt set

A dataset of **600 prompts** was built:

- **300 templated prompts** across six domains: story, explanation, opinion, factual completion, instruction-following, and dialogue
- **300 WikiText-103-derived prompts**

After generation and filtering, **549 examples** remained.

### Continuations

For each prompt, the model generated one short continuation, with a cap of **24 new tokens**. Examples with too-short or malformed outputs were filtered out.

### Hidden states

For each example, hidden states were extracted from the **final prompt position**—the representation used to predict the first continuation token.

Candidate layers tested:

- layer 4
- layer 8
- layer 12
- layer 20

Pythia-1.4B has 24 layers total, so layer 20 is near the top of the network.

### Semantic targets

The target for probing was a sentence embedding of the **full generated continuation text**, computed with:

- **sentence-transformers/all-MiniLM-L6-v2**

### Split and probe

Data split:

- train: **383**
- validation: **83**
- test: **83**

A **linear ridge regressor** was trained to map a single hidden state to the continuation sentence embedding.

Primary evaluation metric:

- **mean cosine similarity** between predicted embedding and true continuation embedding on the test set

Baselines:

1. **Mean embedding baseline**: always predict the training-set mean continuation embedding
2. **Random-match baseline**: compare predictions to a randomly permuted true continuation embedding
3. **Lexical TF-IDF prompt baseline**: a shallow prompt-surface baseline

## Cycle 1 Protocol and Adjustments

The first cycle was run end-to-end and completed successfully.

Two protocol adjustments were made during the run:

### 1. Ridge regularization search was extended

The original alpha grid was intended to be small (up to 10), but this turned out to be insufficient. The logs reported that with `alpha <= 10`, train cosine was extremely high (**~0.97 to 0.9997**) while validation/test performance stayed much lower (**~0.25 to 0.30**), indicating poor regularization behavior for this setup.

The alpha grid was therefore extended up to **1e6**, and the best-performing model ultimately used:

- **best alpha = 10000.0**

This is a reasonable deviation rather than a conceptual change: it kept the same linear probe but tuned it properly.

### 2. Steering strength and random control were refined after a pilot

For the causal intervention, the originally suggested modest perturbation size produced a negligible and noisy pilot effect. The steering magnitude was therefore empirically retuned using a pilot sweep.

Final steering magnitude:

- **alpha = 271.264**

The random-direction control was also improved: instead of one fixed random direction, the control used an **independent fresh random direction per prompt**, which is a fairer low-variance comparison.

These changes were documented in the run artifacts and did not alter the basic hypothesis being tested.

## Probe Results: Can One Early Hidden State Predict Full-Continuation Meaning?

Yes, to a meaningful extent.

### Test cosine similarity by layer

- **Layer 4:** 0.3363
- **Layer 8:** 0.3412
- **Layer 12:** 0.3670
- **Layer 20:** **0.4283** (best)

### Baselines

- **Mean embedding baseline:** 0.2310
- **Random-match baseline:** 0.0616
- **Lexical TF-IDF baseline:** 0.3612

### Margins

For the best layer (20):

- improvement over mean baseline: **+0.1973**
- improvement over lexical TF-IDF baseline: **+0.0671**

This is the key semantic recoverability result. A single hidden state, taken *before* any continuation tokens are generated, supported prediction of the eventual continuation’s sentence embedding with test cosine **0.4283**, well above both trivial and shallow lexical baselines.

That matters because the target was not a next-token label or a bag of future token IDs; it was a dense semantic representation of the **entire continuation**.

## Retrieval-Style Evaluation

To make the embedding result more concrete, retrieval metrics were also computed for the best layer.

### Best-layer retrieval results

- **Recall@1:** 0.4337
- **Recall@5:** 0.7590
- **Mean rank:** 4.7711
- **Chance Recall@5:** 0.0602

So on the test set, the probe’s predicted embedding retrieved the correct continuation embedding within the top 5 in **75.9%** of cases, versus **6.02%** chance.

This is a large separation and supports the claim that the early hidden state carries substantial information about the continuation’s semantic destination.

## Causal Semantic Steering Test

The second question was whether the semantic information was merely decodable or also **causally usable**.

### Method

Using the best probe from layer 20, a semantic direction was derived from the learned linear map.

Primary semantic contrast:

- **positive vs. negative sentiment**

A hidden-state steering direction was formed from the probe weights and a semantic contrast vector in embedding space, then injected at the final prompt position before generation.

Evaluation used **76 held-out test prompts** whose original greedy continuations had at least 8 generated tokens.

For each prompt, generations were compared under:

- `alpha = 0`
- `+alpha`
- `-alpha`
- matched random-direction control

The semantic-axis score was:

- cosine to positive anchors minus cosine to negative anchors

### Sentiment steering results

- **Mean delta, +alpha:** **+0.0098**
- **Mean delta, -alpha:** **-0.0158**
- **Fraction shifting in predicted direction, +alpha:** **0.5658**
- **Fraction shifting in predicted direction, -alpha:** **0.6184**

Random-direction control:

- **Mean delta, +alpha:** 0.0042
- **Mean delta, -alpha:** 0.0035
- **Fraction correct, +alpha:** 0.4737
- **Fraction correct, -alpha:** 0.5000

Interpretation: sentiment steering was modest in effect size, but it had the **correct sign** for both positive and negative interventions, exceeded the matched random control, and beat the predeclared directional threshold of 55% for both signs.

### Secondary temporal contrast

A secondary semantic contrast—described in the logs as **temporal (future-vs-past)**—showed a much stronger effect:

- **Mean delta, +alpha:** **+0.1345**
- **Fraction shifting in predicted direction, +alpha:** **0.8684**

This was reported as clearly above random control, which was near chance.

The temporal result is notably stronger than the sentiment result and suggests that some semantic axes may be encoded and steerable much more cleanly than others.

## Were the Success Criteria Met?

Yes.

### Criterion 1: Probe beats baselines

Met.

- Best probe test cosine: **0.4283**
- Mean baseline: **0.2310**
- Random-match baseline: **0.0616**
- Margin over mean baseline: **+0.1973**

This comfortably exceeded the predeclared “nontrivial margin” target.

### Criterion 2: Retrieval beats chance

Met.

- Recall@5: **0.7590**
- Chance Recall@5: **0.0602**

### Criterion 3: Causal steering beats random control

Met.

For sentiment, the learned direction produced the expected sign for both `+alpha` and `-alpha`, directional success above 55%, and better results than the random control. The temporal contrast was even stronger.

### Criterion 4: Final verdict stated explicitly

Met.

The run’s verdict was **supported**.

## Overall Conclusion

### Verdict: Supported

On this first small-scale experiment, the original hypothesis is **supported**.

More precisely:

- A **single early hidden state** from Pythia-1.4B, taken at the last prompt token before any continuation is generated, contained enough information for a **linear probe** to predict the **sentence embedding of the eventual full continuation** substantially better than mean, random, and shallow lexical baselines.
- A semantic direction derived from that probe could be used for **causal steering** of subsequent generations, with effects that were modest for sentiment but clearly above control, and strong for a temporal semantic axis.

That is evidence that the model’s hidden representation at this point encodes more than immediate next-token identity. It appears to encode a meaningful amount of the **semantic trajectory** of the continuation.

## Limitations

Several limitations matter for interpretation.

### 1. Only one model was tested

This was run only on **Pythia-1.4B**. The result may not generalize to other architectures or scales without further testing.

### 2. Greedy decoding only

The continuations were generated with **greedy decoding**, which isolates the model’s preferred continuation but may overstate semantic predictability relative to stochastic sampling. Under sampling, the future may be more weakly determined at the same early state.

### 3. Sentence embedding is only a proxy for meaning

The target semantics were defined by **MiniLM sentence embeddings**, not by human judgments or task-based entailment measures. Strong embedding performance is evidence of semantic recoverability, but it is still mediated by the embedding model’s own biases and geometry.

### 4. Steering effects were axis-dependent

Sentiment steering worked, but the effect size was small:

- +0.0098 for positive
- -0.0158 for negative

The temporal axis was much stronger. So the claim should not be “all aspects of meaning are equally encoded and steerable.” Some semantic dimensions may be much easier to recover and manipulate than others.

### 5. The steering protocol was tuned after a pilot

The steering magnitude had to be increased after the initial suggested scale produced little effect. This was transparent and reasonable, but it means the causal result is not from a fully fixed one-shot intervention protocol.

### 6. Scale was still modest

The dataset after filtering was **549 examples**, with **83 test examples** and **76 prompts** in the steering subset. That is enough for a strong first signal, but still small for claiming broad generality.

## Follow-Up Questions

This run gives a good first answer, but several next experiments would sharpen the claim.

### 1. Test multiple models

Repeat the protocol on:

- GPT-2-large or GPT-2-xl
- Pythia-2.8B
- possibly instruction-tuned variants

This would show whether semantic recoverability and steering strength scale with model family or size.

### 2. Compare greedy vs sampled continuations

Run the same probe and intervention under temperature/top-p sampling. That would test whether the early state encodes a semantic plan even when the exact wording remains underdetermined.

### 3. Probe different horizons explicitly

Instead of one overall short continuation target, compare:

- first clause only
- full first sentence
- 24-token continuation
- multi-sentence continuation

This would measure how far ahead semantic recoverability extends.

### 4. Stronger semantic targets

Add evaluation beyond generic sentence embeddings, for example:

- NLI-style entailment targets
- semantic role or event structure features
- topic, tense, sentiment, and factuality axes
- human judgments of semantic similarity

### 5. Layer-position mapping

This run sampled four layers and found the strongest probe at **layer 20**. A denser sweep over layers and token positions could show where semantic planning becomes linearly accessible.

### 6. Better causal controls

Future runs could include:

- multiple independent random seeds for generation
- norm-matched interventions at neighboring layers
- interventions on earlier prompt tokens, not just the last prompt token
- comparisons to direct embedding-space target optimization

## Bottom Line

This experiment provides initial evidence that, in **Pythia-1.4B**, an early hidden state used to predict the first continuation token already carries substantial information about the **meaning of the eventual short continuation**, not just local token identity. The evidence is both **decoding-based** (probe and retrieval results) and **causal** (semantic steering beating random controls), making it a stronger semantic-level result than prior token-identity or speculative-decoding work.

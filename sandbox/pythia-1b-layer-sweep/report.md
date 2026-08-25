# Pythia-1.0B causal steering layer sweep: follow-up on the null result

## Research Question

The prior accepted multi-seed steering variance run found positive causal semantic steering for three Pythia model sizes, but **pythia-1.0b** was an outlier. At its previously chosen intervention layer, the mean steering margin was essentially zero (**-0.00002**, 95% CI **[-0.0031, 0.0030]**), even after a dense k-sweep.

The key question in this follow-up was:

- Was that null result a **layer-selection artifact** because the chosen layer was optimized for **passive semantic probing** rather than **causal steering**?
- Or does pythia-1.0b genuinely fail to support this kind of steering at **any** layer?

The working hypothesis was that sweeping multiple layers across the network, with per-layer alpha tuning, would reveal at least one layer with a positive steering margin whose 95% CI excludes zero.

## Experimental Setup

The run reused the same causal steering methodology as the accepted prior experiment, changing only the intervention layer:

- model: **EleutherAI/pythia-1b**
- hardware: **GPU 0 only**
- model config checked programmatically rather than assumed
- actual config found:
  - **16 transformer layers**
  - **hidden size 2048**
- chosen candidate layers spanning early, middle, and late network positions:
  - **[0, 3, 5, 8, 10, 12, 15]**

The steering pipeline was reused as-is from the accepted run:

- probe-derived semantic steering direction
- norm-relative alpha scaling
- prefill-token intervention
- greedy decoding
- sentiment-axis margin metric
- same train/validation/test split procedure
- same 95% CI computation via t-interval

## Protocol by Cycle

### Cycle 1

This project completed in one successful experimental cycle.

#### What was tried

For each candidate layer, the run followed the same two-stage procedure used previously:

1. **Pilot k-sweep** on the validation/pilot split to choose the best scaling constant for that layer.
2. **Multi-seed evaluation** using the selected k and resulting alpha.

The dense k-grid reused from the prior accepted run was:

- **[0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6, 2.4, 3.2, 4.8, 6.4, 8.0]**

Per-layer alpha was then computed using the same norm-relative rule as before:

- **alpha = k × mean hidden-state norm at that layer**

Multi-seed evaluation used:

- **5 seeds**: [101, 202, 303, 404, 505]
- pilot/validation size: **n = 40**
- held-out test size: **n = 60 per seed**

#### Why the protocol did not change further

No mid-course protocol revision was needed, because the first complete run met the success criteria and produced clear layer-dependent differences, including layers with confidence intervals excluding zero.

## Layer-wise k Selection

The pilot sweep selected different k values for different layers, indicating that optimal steering strength was layer-dependent rather than uniform across the network.

| Layer | Selected k |
|---|---:|
| 0 | 1.0 |
| 3 | 2.4 |
| 5 | 8.0 |
| 8 | 8.0 |
| 10 | 1.6 |
| 12 | 0.2 |
| 15 | 3.2 |

A notable detail is that the formerly used layer for pythia-1.0b was **layer 12**, and its selected k here was very small (**0.2**), consistent with it being a weak steering site under this methodology.

## Results

### Per-layer steering margins

All results below are from the 5-seed evaluation, aggregated as mean margin, sample standard deviation, and 95% CI.

| Layer | Mean margin | 95% CI | Interpretation |
|---|---:|---|---|
| 0 | 0.00416 | [-0.00178, 0.01010] | indistinguishable from zero |
| 3 | -0.01378 | [-0.02211, -0.00545] | significantly negative |
| 5 | 0.03167 | [0.02298, 0.04037] | significantly positive |
| 8 | 0.01268 | [0.00408, 0.02128] | significantly positive |
| 10 | 0.00724 | [0.00177, 0.01271] | significantly positive |
| 12 | 0.00019 | [-0.00307, 0.00345] | indistinguishable from zero |
| 15 | 0.00169 | [-0.00604, 0.00943] | indistinguishable from zero |

### Best layer

The best-performing layer was:

- **Layer 5**
- mean margin: **0.03167**
- 95% CI: **[0.02298, 0.04037]**

This is a clear positive effect, and its CI excludes zero by a substantial margin.

### Comparison to prior cross-model results

Previously reported positive margins for the other model sizes were:

- **410m**: **0.0565**, CI **[0.043, 0.070]**
- **1.4b**: **0.0404**, CI **[0.033, 0.047]**
- **2.8b**: **0.0087**, CI **[0.0050, 0.0124]**

Compared with those:

- pythia-1.0b at **layer 5** (**0.03167**, CI **[0.02298, 0.04037]**) is
  - smaller than **410m**,
  - somewhat smaller but close to **1.4b**,
  - clearly larger than **2.8b**.

So the best pythia-1.0b layer is not just barely above zero; it is in the same general effect-size range as the other successful models, especially relative to 1.4b and 2.8b.

## Interpretation

The main outcome is that **pythia-1.0b does support positive causal steering at some layers**.

Three tested layers had positive 95% CIs excluding zero:

- **layer 5**: 0.03167, CI [0.02298, 0.04037]
- **layer 8**: 0.01268, CI [0.00408, 0.02128]
- **layer 10**: 0.00724, CI [0.00177, 0.01271]

At the same time, the effect was strongly **layer-dependent**:

- **layer 12**, the previously used passive-probing-selected layer, remained null: **0.00019**, CI **[-0.00307, 0.00345]**
- **layer 3** was significantly negative: **-0.01378**, CI **[-0.02211, -0.00545]**

This means the earlier pythia-1.0b null result was not evidence that the model lacks steerability overall. Instead, it appears that the specific layer chosen from passive probing was a poor choice for causal steering.

## Success Criteria

The run met the stated success criteria:

- covered **7 distinct layers** spanning early/mid/late positions
- used **5 seeds per layer**
- reported per-layer **mean**, **std**, and **95% CI**
- answered the key decision question unambiguously

Most importantly, at least one layer had **ci95_low > 0**; in fact, **three** layers did.

## Final Verdict

**Layer-selection artifact: at least one pythia-1.0b layer has CI excluding zero.**

More strongly, the best layer (**layer 5**) shows a steering margin (**0.03167**, CI **[0.02298, 0.04037]**) that is meaningfully positive and comparable in magnitude to previously successful model sizes.

## Limitations and Follow-up Questions

- This sweep tested a **representative subset** of layers, not all 16 layers. The conclusion is sufficient to reject the claim that “no layer works,” but it does not identify the globally optimal layer with certainty.
- The steering effect is clearly **not monotonic with depth**. Some middle layers work well, one early-middle layer is significantly negative, and the previously chosen late-middle layer is null. That deserves mechanistic follow-up.
- Because the methodology intentionally matched the prior accepted run, this experiment isolates **layer choice** as the main changed variable. That is a strength for causal interpretation, but it also means other potentially useful variations were not explored here.
- A natural next step would be a **denser full-layer sweep** across all 16 layers, or a targeted sweep around **layers 5–10**, to map the steering profile more precisely.
- Another follow-up would be to compare **passive probe quality** and **causal steering quality** layer by layer, since this run suggests those two criteria can diverge sharply in pythia-1.0b.

## Bottom Line

The earlier pythia-1.0b anomaly was **not** a model-wide absence of causal semantic steerability. It was a **layer-selection mistake**: the previously chosen passive-probing-optimal layer was near-null for causal steering, while several other layers—especially **layer 5**—showed clear positive steering with confidence intervals excluding zero.

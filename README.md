# Decoding Future Semantics

Code and experiment pipelines for a study of whether a language model's hidden state — at the moment it is about to generate a continuation — already linearly encodes the *meaning* of that continuation, not just the identity of the next few tokens.

**Working title:** *Beyond the Next Few Tokens: Semantic Recoverability, Strong Baselines, and Layer-Dependent Causal Steering in Language Models*

Target: interpretability-focused venue (workshop-track framing originally; extended with additional robustness/comparison experiments toward a journal submission).

## What's here

This repo holds two things:

1. **`paper_agent/`** — a small autonomous Planner → Coder → Critic pipeline (`paper-agent` CLI) that proposed and ran the first eight experiments end-to-end against a GPU server, iterating per experiment until a Critic pass accepted the result.
2. **`sandbox/`** — one directory per experiment, each self-contained (its own scripts, `report.md`, and result summaries). Some were produced by the `paper_agent` pipeline above; later ones (`p1`–`p4`, `paper-figures`) were follow-up analyses run directly against the same reused data/checkpoints, mostly for IEEE Access-level robustness checks the original workshop-scope experiments didn't need.

## Core method (common to nearly every experiment)

- **Models**: EleutherAI Pythia (410M / 1.0B / 1.4B / 2.8B, base) as the primary scaling suite; Qwen2.5-1.5B and Qwen3-1.7B/4B-Base for cross-architecture generalization.
- **Probe**: linear ridge regression from a single hidden state (at the model's own last prompt token, before any continuation token exists) to a sentence embedding of the eventual continuation.
- **Baselines**: mean/random/lexical embeddings, a top-*k* logit-lens token-identity control, and — the paper's central methodological point — an explicit behavioral rollout baseline (the model's own next *m* generated tokens, embedded directly).
- **Causal steering**: the probe's learned direction, projected back into hidden-state space and added to the last prompt-token's hidden state during prefill only, evaluated against a magnitude-matched random-direction control across 5 seeds.

Full method details, exact hyperparameters, and honest negative/mixed results (including one retracted-then-corrected finding) are in each experiment's own `report.md`.

## Repository layout

```
paper_agent/            Autonomous experiment-running CLI (Planner/Coder/Critic loop)
run_paper.sh             Launches paper_agent inside a detached tmux session
status.sh                One-line status for every project under sandbox/
pyproject.toml

sandbox/
  hidden-state-semantic-lookahead/     4.1  baseline semantic recoverability (Pythia-1.4B anchor)
  pythia-scaling-and-controls/         4.2  token-identity + rollout controls; 4.4 scaling + sampling
  horizon-crossover-analysis/          4.3  horizon crossover (probe vs. rollout as target length grows)
  steering-scale-alpha-normalization/  4.5  norm-relative steering alpha (step 1 of the correction arc)
  steering-variance-and-1b-recheck/    4.5  multi-seed steering variance; retracts a single-seed artifact
  pythia-1b-layer-sweep/               4.5  resolves the Pythia-1.0B steering anomaly (layer-selection artifact)
  qwen-cross-architecture-replication/ 4.6  cross-architecture replication (Qwen2.5-1.5B)
  qwen3-scaling-replication/           4.7  within-family scaling (Qwen3-1.7B/4B-Base)

  semantic-target-robustness/  (P1) does recoverability hold across MiniLM / BGE-base / E5-base target spaces?
  p2-horizon-rollout-encoder/  (P2) joint horizon x rollout-budget x semantic-encoder analysis
  p3-pythia1.4b-steering-layer-sweep/ (P3) steering-specific layer sweep on the anchor model (Pythia-1.4B)
  p4-future-decoding-comparison/ (P4) comparison with Tuned-Lens-style and Future-Lens-style decoders
  paper-figures/               Cross-experiment figures assembled for the manuscript
```

Each `sandbox/<experiment>/` directory follows roughly the same pattern: numbered/staged Python scripts, an `artifacts/` folder with result tables and (small) figures, and a `report.md` with the full written findings, caveats, and reproducibility notes for that experiment.

## Data / artifact availability

Raw hidden-state tensors, sentence embeddings, and probe weights (`*.pt` / `*.npy` / `*.npz`) are **not** committed to this repo — they total several hundred MB and individual files exceed GitHub's per-file size limit. Every script that produces them is included, so they can be regenerated locally given:

- the model checkpoints listed in each experiment's `report.md` (all public on the Hugging Face Hub),
- a CUDA GPU,
- the environment below.

Result *summaries* (CSV/JSON tables, markdown reports, PNG/PDF figures) are committed in full.

## Environment

```
Python 3.11
torch >= 2.11 (CUDA build)
transformers >= 5.15
sentence-transformers >= 6.0
scikit-learn >= 1.9
numpy >= 2.4
```

`paper_agent`'s own dependencies are listed in `pyproject.toml`. Each `sandbox/` experiment additionally expects `sentence-transformers`, `scikit-learn`, and `scipy` in whatever environment runs it.

## Running an experiment

Most `sandbox/<experiment>/` directories are run as a sequence of plain Python scripts from within that directory, e.g.:

```bash
cd sandbox/p3-pythia1.4b-steering-layer-sweep
python run_p3.py --stage prep
python run_p3.py --stage k_sweep
python run_p3.py --stage multiseed
python run_p3.py --stage aggregate
```

See the top-of-file docstring in each experiment's main script (and its `report.md`) for the exact invocation and any artifacts it expects to reuse from an earlier experiment.

To launch the autonomous pipeline on a new topic:

```bash
./run_paper.sh --topic "<research question for the Planner>" --project <folder-name> [--reuse-from <path-to-prior-project>] [--gpu 0] [--hours 8]
```

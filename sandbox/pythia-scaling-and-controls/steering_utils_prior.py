"""
Utilities for the causal semantic-steering intervention (step 8).

We hook the output of a specific GPT-NeoX decoder layer (the module whose
output equals `output_hidden_states[layer_idx]`) and add a fixed vector to
the hidden state at the LAST position of the prompt during the prefill
forward pass only (not during subsequent single-token decode steps). Because
attention is causal, this perturbation propagates into the KV cache for that
position at layers >= layer_idx, and therefore influences every subsequently
generated token.
"""
import torch


def layer_module_for_hidden_states_index(model, layer_idx):
    """`output_hidden_states` tuple index 0 is the embedding output, and index
    L (for L>=1) is the output of transformer block L-1. Returns the module
    whose forward-hook output corresponds to hidden_states[layer_idx]."""
    assert layer_idx >= 1, "layer_idx 0 is the embedding layer; no module produces it directly"
    return model.gpt_neox.layers[layer_idx - 1]


class InjectionHook:
    """Adds `delta` to the last-token hidden state on the first forward call
    after `reset()`, then no-ops on subsequent calls (the autoregressive
    decode steps), until `reset()` is called again."""

    def __init__(self, model, layer_idx, delta=None):
        self.module = layer_module_for_hidden_states_index(model, layer_idx)
        self.delta = delta
        self.applied = True  # no-op until a delta is set and reset() called
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.delta is None or self.applied:
            return output
        if output.shape[1] < 2:
            # a decode step (seq_len==1) reached before any prefill was seen;
            # nothing sensible to inject into, skip.
            return output
        out = output.clone()
        out[:, -1, :] = out[:, -1, :] + self.delta.to(dtype=out.dtype, device=out.device)
        self.applied = True
        return out

    def set_delta(self, delta):
        self.delta = delta

    def reset(self):
        self.applied = False

    def remove(self):
        self.handle.remove()


def compute_pilot_alpha(hidden_states_subset, fraction=0.5):
    """hidden_states_subset: [N, H] numpy/torch array of hidden states at the
    target layer/position (e.g. validation examples). We define the "mean
    token-state std" as the L2 norm of the per-dimension standard deviation
    vector (an aggregate, norm-comparable scale for how much a typical
    hidden-state vector fluctuates across examples). alpha = fraction * that
    norm, so that ||alpha * unit_direction|| ~= fraction * this scale.
    """
    import numpy as np
    if hasattr(hidden_states_subset, "numpy"):
        hidden_states_subset = hidden_states_subset.numpy()
    std_per_dim = hidden_states_subset.std(axis=0)
    agg_std = float(np.linalg.norm(std_per_dim))
    return fraction * agg_std, agg_std

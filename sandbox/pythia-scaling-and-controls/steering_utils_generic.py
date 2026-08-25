"""
Batched version of the prior run's InjectionHook (steering_utils.py),
generalized to add a PER-ROW delta vector to the last token's hidden state
during the prefill forward pass of a left-padded batch (so index -1 is the
true last token for every row). Same injection point/semantics as prior:
hook the output of `model.gpt_neox.layers[layer_idx-1]` (the module whose
output equals `output_hidden_states[layer_idx]`).
"""
import numpy as np
import torch


def layer_module_for_hidden_states_index(model, layer_idx):
    assert layer_idx >= 1
    return model.gpt_neox.layers[layer_idx - 1]


class BatchedInjectionHook:
    """delta: None, or a [B, H] tensor added to out[:, -1, :] on the first
    forward call after reset() (the prefill step), matching batch size B of
    that call. No-ops on subsequent (decode) calls until reset() again."""

    def __init__(self, model, layer_idx):
        self.module = layer_module_for_hidden_states_index(model, layer_idx)
        self.delta = None
        self.applied = True
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.delta is None or self.applied:
            return output
        if output.shape[1] < 2:
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


def compute_pilot_alpha(hidden_states_subset, fraction=4.0):
    if hasattr(hidden_states_subset, "numpy"):
        hidden_states_subset = hidden_states_subset.numpy()
    std_per_dim = hidden_states_subset.std(axis=0)
    agg_std = float(np.linalg.norm(std_per_dim))
    return fraction * agg_std, agg_std

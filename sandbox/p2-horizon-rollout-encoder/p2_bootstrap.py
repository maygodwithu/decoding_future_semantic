"""Paired bootstrap CI helper (shared with P1)."""
import numpy as np


def paired_bootstrap_ci(values_a, values_b=None, n_boot=10000, seed=42, alpha=0.05):
    values_a = np.asarray(values_a, dtype=np.float64)
    n = len(values_a)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))

    if values_b is None:
        stat = values_a[idx].mean(axis=1)
        point = float(values_a.mean())
    else:
        values_b = np.asarray(values_b, dtype=np.float64)
        assert len(values_b) == n
        diff = values_a - values_b
        stat = diff[idx].mean(axis=1)
        point = float(diff.mean())

    lo = float(np.percentile(stat, 100 * (alpha / 2)))
    hi = float(np.percentile(stat, 100 * (1 - alpha / 2)))
    return {"point": point, "ci_low": lo, "ci_high": hi, "excludes_zero": (lo > 0) or (hi < 0)}

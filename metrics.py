"""
Evaluation metrics: per-task error, total parameter change, locality of change.
"""
import numpy as np


def mse(network, X, y):
    y_pred, _ = network.forward(X)
    return float(np.mean((y_pred - y) ** 2))


def _flat(params):
    return np.concatenate([W.ravel() for W in params])


def parameter_change_norm(params_before, params_after):
    return float(np.linalg.norm(_flat(params_after) - _flat(params_before)))


def locality_of_change(params_before, params_after, frac=0.1):
    """
    Fraction of total |Δw| concentrated in the top-`frac` most-changed parameters.
    1.0 = fully localized in a few edges; equals `frac` if change is uniform.
    """
    delta = np.abs(_flat(params_after) - _flat(params_before))
    total = float(delta.sum())
    if total <= 0.0:
        return float(frac)
    n = delta.size
    k = max(1, int(np.ceil(frac * n)))
    return float(np.sort(delta)[-k:].sum() / total)


def forgetting(mse_old_after_old, mse_old_after_new):
    """How much the old-task error grew after training on the new task."""
    return float(mse_old_after_new - mse_old_after_old)


def concentration(values_per_edge, frac=0.1):
    """
    Top-`frac` concentration of |values| across all edges.
    1.0 = fully localized in a few edges; equals `frac` if uniform.
    Used for both Δw locality and heat locality.
    """
    flat = np.abs(np.concatenate([v.ravel() for v in values_per_edge]))
    total = float(flat.sum())
    if total <= 0.0:
        return float(frac)
    n = flat.size
    k = max(1, int(np.ceil(frac * n)))
    return float(np.sort(flat)[-k:].sum() / total)

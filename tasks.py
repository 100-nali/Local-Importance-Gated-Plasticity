"""
Sequential linear regression tasks with a controllable overlap parameter.

A task is defined by a unit-norm target direction d in R^input_dim. Samples are
drawn x ~ N(0, I); targets are y = d.x + eps. Two consecutive tasks A, B are
constructed so that d_A . d_B == overlap. overlap = 1 means identical tasks
(no interference), overlap = 0 means orthogonal tasks (maximum interference).
"""
import numpy as np


def _linreg_samples(input_dim, direction, n_train, n_test, noise, rng):
    X_train = rng.standard_normal((n_train, input_dim))
    y_train = X_train @ direction + noise * rng.standard_normal((n_train,))
    X_test = rng.standard_normal((n_test, input_dim))
    y_test = X_test @ direction
    return {
        "X_train": X_train,
        "y_train": y_train.reshape(-1, 1),
        "X_test": X_test,
        "y_test": y_test.reshape(-1, 1),
        "target_dir": direction.copy(),
    }


def make_two_task_sequence(input_dim, overlap,
                           n_train=500, n_test=200,
                           noise=0.05, seed=0):
    rng = np.random.default_rng(seed)

    d_a = rng.standard_normal((input_dim,))
    d_a /= np.linalg.norm(d_a)

    perp = rng.standard_normal((input_dim,))
    perp -= (perp @ d_a) * d_a
    perp /= max(np.linalg.norm(perp), 1e-12)

    overlap = float(np.clip(overlap, -1.0, 1.0))
    d_b = overlap * d_a + np.sqrt(max(1.0 - overlap ** 2, 0.0)) * perp
    d_b /= max(np.linalg.norm(d_b), 1e-12)

    task_a = _linreg_samples(input_dim, d_a, n_train, n_test, noise, rng)
    task_b = _linreg_samples(input_dim, d_b, n_train, n_test, noise, rng)
    task_a["name"] = "A"
    task_b["name"] = "B"
    return [task_a, task_b]


def make_orthogonal_task_sequence(input_dim, n_tasks,
                                  n_train=500, n_test=200,
                                  noise=0.05, seed=0):
    """N mutually-orthogonal linear-regression tasks (n_tasks <= input_dim).

    Directions are the first n_tasks columns of a Haar-random orthonormal
    matrix obtained via QR of a Gaussian. Each pair has zero overlap, so
    interference between consecutive tasks is maximal — the regime where any
    retention benefit of imp_gated over vanilla / thresh has to show up.
    """
    if n_tasks > input_dim:
        raise ValueError(
            f"orthogonal n_tasks ({n_tasks}) must be <= input_dim ({input_dim})"
        )
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((input_dim, input_dim)))
    tasks = []
    for k in range(n_tasks):
        d = Q[:, k]
        t = _linreg_samples(input_dim, d, n_train, n_test, noise, rng)
        t["name"] = chr(ord("A") + k)
        tasks.append(t)
    return tasks

"""Gold-standard check: ``SpringNetworkSubstrate.backward`` vs loss FD.

Compares ``net.backward`` (with ``backward_mode='fd'`` — parameter central
differences through the full free-phase solve) to an independent FD estimate
of ``∇_w L`` for ``L = (1/2) * mean((y_pred - y)^2)``.

With ``backward_mode='adjoint'`` (the default for fast training), small
mismatches vs this FD are expected when the elastic Hessian is regularized;
use this script to validate ``fd`` mode and to probe hyperparameters.
"""
import numpy as np

from spring_network import SpringNetworkSubstrate


def loss_at(net, X, y_target):
    y_pred, _ = net.forward(X)
    return float(0.5 * np.mean((y_pred - y_target) ** 2))


def fd_gradient(net, X, y_target, h=1e-4):
    """Finite-difference gradient of MSE loss w.r.t. each parameter."""
    fd_grads = []
    for i, W in enumerate(net.weights):
        g = np.zeros_like(W)
        flat = W.ravel()
        for k in range(flat.size):
            saved = flat[k]
            flat[k] = saved + h
            L_plus = loss_at(net, X, y_target)
            flat[k] = saved - h
            L_minus = loss_at(net, X, y_target)
            flat[k] = saved
            g.ravel()[k] = (L_plus - L_minus) / (2.0 * h)
        fd_grads.append(g)
    return fd_grads


def main():
    # Input scale: keep source y-separations moderate (see module docstring).
    s = 0.2
    inputs = np.array([[-s, -s]])
    targets = (inputs[:, 0:1] - inputs[:, 1:2])

    net = SpringNetworkSubstrate(
        rows=5,
        cols=4,
        n_input=2,
        out_pos_row=2,
        out_neg_row=3,
        eta=0.05,
        seed=0,
        backward_mode="fd",
        fd_eps=1e-4,
    )
    y_pred, cache = net.forward(inputs)
    g_back = net.backward(cache, y_pred, targets)
    g_fd = fd_gradient(net, inputs, targets, h=net.fd_eps)
    names = ["k_h", "k_v", "ell_h", "ell_v"]
    print("Spring backward (fd) vs independent FD of L = 0.5 * mean((y_pred-y)^2)")
    for name, gc, gf in zip(names, g_back, g_fd):
        ratio = gc / (gf + 1e-20)
        mask = np.abs(gf) > 1e-8
        mean_ratio = float(np.mean(ratio[mask])) if mask.any() else float("nan")
        print(
            f"  {name}: backward=[{gc.min():+.4f},{gc.max():+.4f}]  "
            f"FD=[{gf.min():+.4f},{gf.max():+.4f}]  "
            f"mean ratio={mean_ratio:.3f}"
        )


if __name__ == "__main__":
    main()

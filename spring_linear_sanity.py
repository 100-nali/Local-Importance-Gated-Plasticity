"""Linear-task sanity for SpringNetworkSubstrate.

Fits y = x_1 - x_2 — the simplest linearly-separable task on the same
four-point input set as XOR. The substrate is *more than capable* of
representing this (a single linear chain of springs can do it). If this
fails, training is broken. If this succeeds and XOR still fails, the
issue is substrate expressivity rather than training.
"""
import numpy as np

from learning_rules import SGDRule
from spring_network import SpringNetworkSubstrate


def main():
    s = 0.2
    X = np.array([[-s, -s], [-s, +s], [+s, -s], [+s, +s]])
    y = (X[:, 0:1] - X[:, 1:2])

    for label, kwargs, lr in [
        ("k=1.0 ell=1.0 (default)",       dict(k_init=1.0, ell_init=1.0, init_std=0.05), 0.5),
        ("k=1.0 ell=0.5 (pre-strained)",  dict(k_init=1.0, ell_init=0.5, init_std=0.05), 0.5),
        ("k=0.3 ell=1.0 (soft)",          dict(k_init=0.3, ell_init=1.0, init_std=0.05), 1.0),
        ("k=0.3 ell=0.5 (soft+stretched)",dict(k_init=0.3, ell_init=0.5, init_std=0.05), 1.0),
        ("init_std=0.3 (broken symmetry)",dict(k_init=1.0, ell_init=1.0, init_std=0.3),  0.5),
    ]:
        net = SpringNetworkSubstrate(
            rows=5, cols=5, n_input=2,
            out_pos_row=2, out_neg_row=3,
            eta=0.01, seed=0, **kwargs,
        )
        rule = SGDRule(lr=lr)
        state = rule.init_state(net)
        losses = []
        for epoch in range(500):
            yp, cache = net.forward(X)
            grads = net.backward(cache, yp, y)
            state = rule.step(net, grads, state, cache=cache)
            net.project_weights()
            losses.append(float(np.mean((yp - y) ** 2)))
        yp_final, _ = net.forward(X)
        sign_correct = ((yp_final * y) > 0)
        n_correct = int(sign_correct.sum())
        # Also report monotonicity: predictions should grow in the direction of x1 - x2
        x1mx2 = (X[:, 0] - X[:, 1])
        order_match = np.argsort(yp_final.ravel()) == np.argsort(x1mx2)
        print(f"\n[{label}]  lr={lr}")
        print(f"  loss[0]={losses[0]:.4f}  loss[100]={losses[100]:.4f}  "
              f"loss[final]={losses[-1]:.4f}")
        print(f"  preds={yp_final.ravel()}")
        print(f"  targets={y.ravel()}")
        print(f"  sign_correct={n_correct}/4   order_match={bool(order_match.all())}")


if __name__ == "__main__":
    main()

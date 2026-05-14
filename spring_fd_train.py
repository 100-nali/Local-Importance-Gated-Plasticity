"""Train spring substrate on y = x_1 - x_2 using TRUE finite-difference gradients.

If FD training fits the linear task, the substrate is fine and the contrastive
gradient has a bug. If FD training also fails, the substrate setup itself
can't represent this task.
"""
import numpy as np

from spring_network import SpringNetworkSubstrate


def loss_at(net, X, y_target):
    y_pred, _ = net.forward(X)
    return float(np.mean((y_pred - y_target) ** 2))


def fd_gradient(net, X, y_target, h=1e-3):
    grads = []
    for W in net.weights:
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
        grads.append(g)
    return grads


def main():
    s = 0.2
    X = np.array([[-s, -s], [-s, +s], [+s, -s], [+s, +s]])
    y = (X[:, 0:1] - X[:, 1:2])

    for lr in [0.05, 0.01]:
        net = SpringNetworkSubstrate(
            rows=5, cols=5, n_input=2,
            out_pos_row=2, out_neg_row=3,
            eta=0.01, seed=0,
        )
        print(f"\nFD-gradient training (lr={lr})")
        print(f"  initial loss: {loss_at(net, X, y):.5f}")
        for epoch in range(200):
            grads = fd_gradient(net, X, y, h=1e-3)
            for i in range(len(net.weights)):
                net.weights[i] -= lr * grads[i]
            net.project_weights()
            if epoch % 20 == 0 or epoch == 199:
                yp, _ = net.forward(X)
                loss = float(np.mean((yp - y) ** 2))
                print(f"  epoch {epoch:3d}  loss={loss:.5f}  "
                      f"preds={yp.ravel()}")

    print()
    yp_final, _ = net.forward(X)
    sign_correct = ((yp_final * y) > 0)
    print(f"Final preds={yp_final.ravel()}  targets={y.ravel()}")
    print(f"Sign-correct: {int(sign_correct.sum())}/4 "
          f"(note: two targets are 0, only two can be sign-decided)")


if __name__ == "__main__":
    main()

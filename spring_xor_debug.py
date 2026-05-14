"""Debug script: inspect spring network forward pass and gradients at init."""
import numpy as np

from spring_network import SpringNetworkSubstrate


def main():
    net = SpringNetworkSubstrate(
        rows=5, cols=4, n_input=2,
        out_pos_row=2, out_neg_row=3,
        eta=0.05, seed=0,
    )
    print(f"num_params: {net.num_params}")
    print(f"source nodes: {net.source_nodes}")
    print(f"output nodes: pos={net.out_pos_node}, neg={net.out_neg_node}")
    print(f"rest positions:")
    for r in range(net.rows):
        for c in range(net.cols):
            i = net._idx(r, c)
            print(f"  node {i} ({r},{c}): rest={net.rest_pos[i]}")

    print()
    print("Forward pass at init for each XOR input:")
    inputs = np.array([
        [-0.5, -0.5],
        [-0.5, +0.5],
        [+0.5, -0.5],
        [+0.5, +0.5],
    ])
    for x in inputs:
        y_pred, cache = net.forward(x[None, :])
        all_pos = cache["all_positions_F"][0]
        out_pos_y = all_pos[net.out_pos_node, 1]
        out_neg_y = all_pos[net.out_neg_node, 1]
        rp = net.rest_pos[net.out_pos_node, 1]
        rn = net.rest_pos[net.out_neg_node, 1]
        print(f"  input={x}  y_pred={float(y_pred[0,0]):+.4f}  "
              f"out_pos.y={out_pos_y:+.4f} (rest {rp:+.1f})  "
              f"out_neg.y={out_neg_y:+.4f} (rest {rn:+.1f})")

    print()
    print("Gradients on one mini-batch (all four XOR pairs):")
    X = inputs
    y_targets = np.array([[-0.5], [+0.5], [+0.5], [-0.5]])
    y_pred, cache = net.forward(X)
    grads = net.backward(cache, y_pred, y_targets)
    for i, (g, name) in enumerate(zip(grads, ["k_h", "k_v", "ell_h", "ell_v"])):
        print(f"  {name}: shape={g.shape}  range=[{g.min():+.4f}, {g.max():+.4f}]  "
              f"|mean|={np.abs(g).mean():.4f}")

    # Check first gradient step direction: does it reduce loss?
    print()
    print("First gradient step (small step, check loss):")
    lr_test = 0.001
    w_save = net.copy_params()
    loss_before = float(np.mean((y_pred - y_targets) ** 2))
    for i in range(len(net.weights)):
        net.weights[i] -= lr_test * grads[i]
    net.project_weights()
    y_pred_after, _ = net.forward(X)
    loss_after = float(np.mean((y_pred_after - y_targets) ** 2))
    print(f"  loss before: {loss_before:.5f}")
    print(f"  loss after step (lr={lr_test}): {loss_after:.5f}")
    print(f"  delta: {loss_after - loss_before:+.5f}")
    net.set_params(w_save)

    # Linear-task sanity: try fitting y = x_1 - x_2 (which IS linearly
    # separable, so even a linear substrate should fit it). If this fails,
    # the substrate has a fundamental bug; if it works, the XOR issue is
    # nonlinearity, not the training pipeline.
    print()
    print("Linear-task sanity: fitting y = x_1 - x_2 with vanilla coupled learning")
    print("(linear-substrate-compatible target; this should ALWAYS work)")
    from learning_rules import SGDRule
    y_linear = (inputs[:, 0:1] - inputs[:, 1:2])
    for k_init, ell_init, lr, label in [
        (1.0, 1.0, 0.05, "default k=1.0 ell=1.0"),
        (0.3, 1.0, 0.1,  "softer  k=0.3 ell=1.0"),
        (1.0, 0.5, 0.05, "pre-strained ell=0.5"),
        (0.3, 0.5, 0.1,  "soft + pre-strained"),
    ]:
        net_t = SpringNetworkSubstrate(
            rows=5, cols=4, n_input=2,
            out_pos_row=2, out_neg_row=3,
            eta=0.05, seed=0,
            k_init=k_init, ell_init=ell_init, init_std=0.05,
        )
        rule_t = SGDRule(lr=lr)
        state = rule_t.init_state(net_t)
        losses = []
        for epoch in range(500):
            yp, cache = net_t.forward(inputs)
            grads = net_t.backward(cache, yp, y_linear)
            state = rule_t.step(net_t, grads, state, cache=cache)
            net_t.project_weights()
            losses.append(float(np.mean((yp - y_linear) ** 2)))
        yp_final, _ = net_t.forward(inputs)
        print(f"  [{label}]  lr={lr}  loss[0]={losses[0]:.4f}  "
              f"loss[100]={losses[100]:.4f}  loss[final]={losses[-1]:.4f}")
        print(f"      preds={yp_final.ravel()}  targets={y_linear.ravel()}")


if __name__ == "__main__":
    main()

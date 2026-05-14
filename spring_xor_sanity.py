"""
XOR sanity test for SpringNetworkSubstrate.

Act 1's resistor mesh is linear in input clamps and cannot represent XOR
regardless of substrate size. Act 2's spring network has geometric
nonlinearity from the Euclidean distance function r = ||x_i - x_j||, so the
substrate should — in principle — fit XOR. This script tests that claim
with a small network trained by vanilla coupled learning (no continual
learning, no importance gating, no context — just the substrate's raw
expressivity).

Pass criterion: all four XOR inputs produce outputs with the correct sign
(i.e., final outputs are sign-consistent with the centered targets ±scale).
If this fails, the geometric nonlinearity is not doing real work and we
need to debug before going further with continual experiments.

Run:
    python spring_xor_sanity.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from learning_rules import SGDRule
from spring_network import SpringNetworkSubstrate


CFG = {
    # Larger grid (5 rows): default interior sources on column 0 (rows 1,3)
    # and output readout on the right at rows 2 and 3, leaving hidden slack.
    "rows": 5,
    "cols": 5,
    "n_input": 2,
    "out_pos_row": 2,
    "out_neg_row": 3,
    "n_epochs": 2000,
    # Input scale 0.2 keeps source-node separations safely > 0.5 (no two
    # source nodes can collide), while still pushing the substrate
    # measurably into its nonlinear regime relative to rest spacing 1.0.
    "input_scale": 0.2,
    "lr": 0.5,
    # Smaller eta -> better EP-approximation accuracy; the project's
    # resistor experiments use eta=0.005 for the same reason.
    "eta": 0.01,
    "seed": 0,
    "log_every": 100,
}


XOR_TABLE = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])


def make_xor_dataset(scale):
    """Centered XOR: 0 -> -scale, 1 -> +scale for both inputs and output."""
    X = (XOR_TABLE[:, :2] * 2 - 1) * scale
    y = (XOR_TABLE[:, 2:3] * 2 - 1) * scale
    return X, y


def train(net, rule, X, y, n_epochs, log_every):
    state = rule.init_state(net)
    history = []
    for epoch in range(n_epochs):
        y_pred, cache = net.forward(X)
        grads = net.backward(cache, y_pred, y)
        state = rule.step(net, grads, state, cache=cache)
        net.project_weights()
        loss = float(np.mean((y_pred - y) ** 2))
        history.append(loss)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            preds_str = ", ".join(f"{v:+.3f}" for v in y_pred.ravel())
            print(f"  epoch {epoch:4d}  loss={loss:.5f}  preds=[{preds_str}]")
    return history


def plot_results(net, X, y, y_pred, history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].semilogy(history)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss (log scale)")
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    # Scatter of predicted vs target across the 4 XOR cases.
    ax = axes[1]
    labels = ["(0,0)->0", "(0,1)->1", "(1,0)->1", "(1,1)->0"]
    colors = ["C0", "C3", "C3", "C0"]
    for k, label in enumerate(labels):
        ax.scatter([y[k, 0]], [y_pred[k, 0]], s=160, c=colors[k],
                   edgecolors="black", linewidths=0.5, label=label, zorder=3)
    lim = max(np.abs(y).max(), np.abs(y_pred).max()) * 1.3
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("target")
    ax.set_ylabel("predicted")
    ax.set_title("XOR predictions (sign quadrant = correct)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Network state for input (1, 1): rest config + free-phase positions.
    ax = axes[2]
    rest = net.rest_pos
    # Draw rest config as gray.
    for (i, j) in net.edges_h + net.edges_v:
        ax.plot([rest[i, 0], rest[j, 0]], [rest[i, 1], rest[j, 1]],
                color="lightgray", linewidth=0.8, zorder=1)
    # Draw trained equilibrium for input (1,1) as colored by strain.
    all_pos, _, _ = net._equilibrate(X[3])  # (1, 1) input
    strains = []
    for (i, j) in net.edges_h + net.edges_v:
        d = all_pos[i] - all_pos[j]
        strains.append(np.linalg.norm(d))
    strains = np.array(strains)
    cmap = plt.get_cmap("coolwarm")
    s_max = max(abs(strains - 1.0).max(), 0.05)
    for idx, (i, j) in enumerate(net.edges_h + net.edges_v):
        d = strains[idx] - 1.0
        color = cmap(0.5 + 0.5 * d / s_max)
        ax.plot([all_pos[i, 0], all_pos[j, 0]],
                [all_pos[i, 1], all_pos[j, 1]],
                color=color, linewidth=2.2, zorder=2)
    # Node markers
    for node in range(net.n_nodes):
        x_, y_ = all_pos[node]
        if node in net.source_nodes:
            ax.plot(x_, y_, "s", color="red", markersize=10,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        elif node == net.out_pos_node:
            ax.plot(x_, y_, "^", color="dodgerblue", markersize=12,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        elif node == net.out_neg_node:
            ax.plot(x_, y_, "v", color="darkorange", markersize=12,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        else:
            ax.plot(x_, y_, "o", color="white", markersize=5,
                    markeredgecolor="dimgray", markeredgewidth=0.4, zorder=3)
    ax.set_title("Trained network @ input (1,1) — strain colored")
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        f"Spring-network XOR sanity test "
        f"(grid {CFG['rows']}x{CFG['cols']}, {CFG['n_epochs']} epochs, "
        f"input scale {CFG['input_scale']})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    print("Spring-network XOR sanity")
    print(f"  grid={CFG['rows']}x{CFG['cols']}  n_input={CFG['n_input']}  "
          f"output rows={CFG['out_pos_row']}/{CFG['out_neg_row']}")
    print(f"  lr={CFG['lr']}  eta={CFG['eta']}  "
          f"input_scale={CFG['input_scale']}  epochs={CFG['n_epochs']}")

    net = SpringNetworkSubstrate(
        rows=CFG["rows"], cols=CFG["cols"], n_input=CFG["n_input"],
        out_pos_row=CFG["out_pos_row"], out_neg_row=CFG["out_neg_row"],
        eta=CFG["eta"], seed=CFG["seed"],
    )
    print(f"  num_params = {net.num_params} "
          f"(n_edges = {len(net.edges_h) + len(net.edges_v)})")

    rule = SGDRule(lr=CFG["lr"])
    X, y = make_xor_dataset(CFG["input_scale"])

    history = train(net, rule, X, y, CFG["n_epochs"], CFG["log_every"])

    y_pred, _ = net.forward(X)
    sign_correct = ((y_pred * y) > 0).ravel()
    n_correct = int(sign_correct.sum())
    print()
    print(f"Final MSE: {history[-1]:.5f}")
    print(f"Final preds: {y_pred.ravel()}")
    print(f"Targets:     {y.ravel()}")
    print(f"Sign-correct XOR cases: {n_correct}/4  -> "
          f"{'PASS' if n_correct == 4 else 'FAIL'}")

    plot_results(net, X, y, y_pred, history,
                 save_path=out_dir / "spring_xor_sanity.png")
    with open(out_dir / "spring_xor_sanity.json", "w") as f:
        json.dump({
            "config": CFG,
            "final_loss": history[-1],
            "history": history,
            "final_preds": y_pred.ravel().tolist(),
            "targets": y.ravel().tolist(),
            "n_correct": n_correct,
        }, f, indent=2)
    print(f"Saved {out_dir / 'spring_xor_sanity.png'}")


if __name__ == "__main__":
    main()

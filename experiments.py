"""
Sweep three local learning rules over an overlap axis on the edge-coupled
substrate, tracking per-edge heat dissipation alongside the existing metrics.

Heat per edge per step is the work done by the contrastive force on that edge:
    Q_e = - g_e * Δw_e
For SGD this reduces to the canonical viscous dissipation lr * g_e^2; for
imp_gated and thresh it tracks the actual force-displacement product the
edge experiences. Cumulative Q_total over a task is the total work the
loss-landscape force did on the network during that task — the closest
substrate-faithful proxy for thermodynamic dissipation.

Run:
    python experiments.py
"""
import json
from pathlib import Path

import numpy as np

from network import MeshCoupledSubstrate
from tasks import make_two_task_sequence
from learning_rules import SGDRule, ThresholdedSGDRule, ImportanceGatedRule
from metrics import mse, parameter_change_norm, locality_of_change, concentration


CONFIG = {
    "rows": 8,
    "cols": 10,
    "n_input": 8,
    "out_pos_row": 3,
    "out_neg_row": 4,
    "n_train": 500,
    "n_test": 200,
    "noise": 0.05,
    "n_epochs": 80,
    "batch_size": 32,
    "overlaps": [0.0, 0.25, 0.5, 0.75, 1.0],
    "n_seeds": 12,
}


def train_on_task(network, rule, state, X, y, n_epochs, batch_size, rng,
                  step_callback=None):
    """Train one task. Returns (final state, cumulative heat per edge, steps).

    heat_per_edge[i][idx] = sum over training steps of (-g_e * Δw_e). Positive
    when motion is descent along the contrastive force; can be locally negative
    for imp_gated when the anchor pull dominates. After each rule step we call
    project_weights() if the substrate provides it (positive-conductance
    constraint on the mesh substrate); without that, raw weights drift below
    w_min and the rule's view diverges from the physics.
    """
    n = len(X)
    heat = [np.zeros_like(W) for W in network.weights]
    has_project = hasattr(network, "project_weights")
    step_count = 0
    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            xb, yb = X[b], y[b]
            y_pred, cache = network.forward(xb)
            grads = network.backward(cache, y_pred, yb)
            w_before = [W.copy() for W in network.weights]
            state = rule.step(network, grads, state)
            if has_project:
                network.project_weights()
            for i in range(len(network.weights)):
                delta_w = network.weights[i] - w_before[i]
                heat[i] += -grads[i] * delta_w
            step_count += 1
            if step_callback is not None:
                step_callback(step_count)
    return state, heat, step_count


def run_sequence(rule, tasks, network, n_epochs, batch_size, seed,
                 trace_mse=False, eval_every=None):
    rng = np.random.default_rng(seed)
    state = rule.init_state(network)
    log = {"rule": rule.name, "seed": seed, "tasks": []}
    pre = [mse(network, t["X_test"], t["y_test"]) for t in tasks]
    if trace_mse:
        log["learning_curve"] = []

    global_iteration = 0

    def record_mse(task_idx, task_iteration):
        log["learning_curve"].append({
            "global_iteration": int(global_iteration + task_iteration),
            "task_iteration": int(task_iteration),
            "train_task_idx": int(task_idx),
            "train_task_name": tasks[task_idx]["name"],
            "mse_test_each_task": [
                mse(network, t["X_test"], t["y_test"]) for t in tasks
            ],
        })

    for k, task in enumerate(tasks):
        if trace_mse:
            record_mse(k, 0)
        params_before = network.copy_params()

        def maybe_record_mse(task_iteration):
            if eval_every is None:
                return
            if task_iteration % eval_every == 0:
                record_mse(k, task_iteration)

        state, heat, steps_taken = train_on_task(
            network, rule, state,
            task["X_train"], task["y_train"],
            n_epochs, batch_size, rng,
            step_callback=maybe_record_mse if trace_mse else None,
        )
        if trace_mse and (eval_every is None or steps_taken % eval_every != 0):
            record_mse(k, steps_taken)
        global_iteration += steps_taken
        params_after = network.copy_params()
        per_edge_dw = [pa - pb for pb, pa in zip(params_before, params_after)]
        log["tasks"].append({
            "task_idx": k,
            "task_name": task["name"],
            "mse_test_each_task": [
                mse(network, t["X_test"], t["y_test"]) for t in tasks
            ],
            "param_change": parameter_change_norm(params_before, params_after),
            "locality": locality_of_change(params_before, params_after, frac=0.1),
            "heat_total": float(sum(h.sum() for h in heat)),
            "heat_locality": concentration(heat, frac=0.1),
            "per_edge_dw_layers": [w.tolist() for w in per_edge_dw],
            "per_edge_heat_layers": [h.tolist() for h in heat],
        })
        state = rule.on_task_boundary(network, state)
    log["mse_pre"] = pre
    return log


# Hyperparameters tuned on the 2D-mesh coupled substrate. Contrastive-force
# scale and equilibrium relaxation differ from the layered substrate; lr,
# threshold, and lambda are retuned accordingly. eta is the nudge strength
# used by the substrate's clamped phase.
HPARAMS = {
    "lr": 5.0,
    "threshold": 0.005,
    "lam": 100.0,
    "beta": 0.95,
    "eta": 0.005,
}


def make_rules():
    return [
        SGDRule(lr=HPARAMS["lr"]),
        ThresholdedSGDRule(lr=HPARAMS["lr"], threshold=HPARAMS["threshold"]),
        ImportanceGatedRule(lr=HPARAMS["lr"], lam=HPARAMS["lam"], beta=HPARAMS["beta"]),
    ]


def make_substrate(cfg, seed):
    return MeshCoupledSubstrate(
        rows=cfg["rows"], cols=cfg["cols"],
        n_input=cfg["n_input"],
        out_pos_row=cfg["out_pos_row"],
        out_neg_row=cfg["out_neg_row"],
        eta=HPARAMS["eta"], seed=seed,
    )


def overlap_sweep(cfg):
    results = {}
    for ov in cfg["overlaps"]:
        for seed in range(cfg["n_seeds"]):
            tasks = make_two_task_sequence(
                input_dim=cfg["n_input"],
                overlap=ov,
                n_train=cfg["n_train"],
                n_test=cfg["n_test"],
                noise=cfg["noise"],
                seed=seed,
            )
            for rule in make_rules():
                network = make_substrate(cfg, seed)
                log = run_sequence(
                    rule, tasks, network,
                    n_epochs=cfg["n_epochs"],
                    batch_size=cfg["batch_size"],
                    seed=seed,
                )
                log["overlap"] = ov
                results.setdefault(rule.name, {}).setdefault(ov, []).append(log)
    return results


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Sweep: {len(CONFIG['overlaps'])} overlaps x "
          f"{CONFIG['n_seeds']} seeds x 3 rules on coupled substrate...")
    results = overlap_sweep(CONFIG)

    payload = {"config": CONFIG, "hparams": HPARAMS,
               "results": _to_jsonable(results)}
    with open(out_dir / "sweep.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved logs -> {out_dir / 'sweep.json'}")

    from plots import plot_overlap_sweep, plot_heat_pareto, plot_mesh_substrate_activity
    plot_overlap_sweep(results, save_path=out_dir / "overlap_sweep.png")
    plot_heat_pareto(results, save_path=out_dir / "heat_pareto.png")
    plot_mesh_substrate_activity(
        results,
        rows=CONFIG["rows"], cols=CONFIG["cols"],
        n_input=CONFIG["n_input"],
        out_pos_row=CONFIG["out_pos_row"], out_neg_row=CONFIG["out_neg_row"],
        save_path=out_dir / "substrate_activity.png",
    )
    print(f"Saved plots -> {out_dir / 'overlap_sweep.png'}, "
          f"{out_dir / 'heat_pareto.png'}, "
          f"{out_dir / 'substrate_activity.png'}")


if __name__ == "__main__":
    main()

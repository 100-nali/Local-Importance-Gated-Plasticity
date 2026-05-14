"""
Thermodynamic analysis on the trainable context-modulated mesh.

Each substrate update dissipates heat into the edge it touches:

    Q_e(t) = - g_e(t) * Delta w_e(t)

For vanilla contrastive learning this reduces to lr * g_e^2 — the canonical
viscous dissipation. For the importance-gated rules the proximal pull
modifies Delta w_e on high-importance edges, so the per-edge dissipation
profile is expected to differ. Two competing readings of cumulative
importance gating are worth distinguishing on this substrate:

    H1 (genuine retention): protecting load-bearing edges of past tasks
        prevents the learn--forget--relearn cycle; cumulative Q_total
        across the sequence is *lower* than vanilla.

    H2 (restoring force only): the anchor adds work without changing the
        trajectory's footprint; retention improves while cumulative
        Q_total *grows* relative to vanilla.

The infrastructure to compute Q_e per training step is already in
experiments.train_on_task (which records per-task heat_total and
per_edge_heat_layers in run_sequence). This script consumes that output
on the eight-task context-modulated sequence and produces the deliverable
from progress.tex:

    - retention vs cumulative dissipation Pareto across rules
    - per-edge spatial heat maps on the mesh (base + sum-over-context-axes
      collapsed onto each edge's spatial location)
    - per-task heat (does heat per task shrink as edges get protected?)
    - cumulative heat trajectory (does the gap to vanilla open or close?)

Rules and operating points are taken from progress.tex:
    vanilla (no parameters)
    threshold (tau = 1e-4)
    cumulative importance (lambda = 50)         -- current winner
    slow consolidation (lambda = 0.3)

Run:
    python thermodynamic_analysis.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import run_sequence
from learning_rules import (
    CumulativeImportanceGatedRule,
    SGDRule,
    SlowConsolidatedImportanceRule,
    ThresholdedSGDRule,
)
from network import TrainableContextModulatedMeshSubstrate
from plots import _draw_mesh
from tasks import make_modulated_context_task_sequence


CFG = {
    # Mirrors context_modulated_prototype.py — the headline 8-task experiment.
    "rows": 8,
    "cols": 10,
    "n_sensory": 8,
    "context_dim": 8,
    "n_tasks": 8,
    "out_pos_row": 3,
    "out_neg_row": 4,
    "n_train": 500,
    "n_test": 200,
    "noise": 0.05,
    "n_epochs": 60,
    "batch_size": 32,
    "n_seeds": 12,
    "lr": 20.0,
    "eta": 0.005,
    "context_init_std": 0.0,
    "max_log_extra": 4.0,
    # Operating points reported in progress.tex.
    "lam_cum": 50.0,
    "lam_slow": 0.3,
    "thresh_tau": 1e-4,
}


RULE_COLOR = {
    "vanilla": "C0",
    "thresh": "C1",
    "cum_imp_gated": "C3",
    "slow_consolidated": "C6",
}

RULE_LABEL = {
    "vanilla": "vanilla",
    "thresh": "threshold",
    "cum_imp_gated": "cum. importance (λ=50)",
    "slow_consolidated": "slow consolidation",
}


def make_substrate(seed):
    return TrainableContextModulatedMeshSubstrate(
        rows=CFG["rows"],
        cols=CFG["cols"],
        n_sensory=CFG["n_sensory"],
        context_dim=CFG["context_dim"],
        out_pos_row=CFG["out_pos_row"],
        out_neg_row=CFG["out_neg_row"],
        eta=CFG["eta"],
        context_init_std=CFG["context_init_std"],
        max_log_extra=CFG["max_log_extra"],
        seed=seed,
    )


def make_rules():
    return [
        SGDRule(lr=CFG["lr"]),
        ThresholdedSGDRule(lr=CFG["lr"], threshold=CFG["thresh_tau"]),
        CumulativeImportanceGatedRule(lr=CFG["lr"], lam=CFG["lam_cum"]),
        SlowConsolidatedImportanceRule(lr=CFG["lr"], lam=CFG["lam_slow"]),
    ]


def _collapse_to_edges(per_edge_heat_layers):
    """Sum base + context-gain heat onto each edge's spatial location.

    Layers from train_on_task on TrainableContextModulatedMeshSubstrate:
        [0] base_h    shape (rows, cols-1)
        [1] base_v    shape (rows-1, cols)
        [2] gain_h    shape (context_dim, rows, cols-1)
        [3] gain_v    shape (context_dim, rows-1, cols)

    Both base and per-context gains are edge-local parameters; for the
    spatial map we want total heat dissipated through any parameter
    attached to that edge.
    """
    base_h = np.asarray(per_edge_heat_layers[0])
    base_v = np.asarray(per_edge_heat_layers[1])
    gain_h = np.asarray(per_edge_heat_layers[2]).sum(axis=0)
    gain_v = np.asarray(per_edge_heat_layers[3]).sum(axis=0)
    return base_h + gain_h, base_v + gain_v


def run_all():
    """Per-rule list of per-seed records with heat + MSE trajectories."""
    records = {}
    for seed in range(CFG["n_seeds"]):
        tasks = make_modulated_context_task_sequence(
            input_dim=CFG["n_sensory"],
            n_tasks=CFG["n_tasks"],
            context_dim=CFG["context_dim"],
            n_train=CFG["n_train"],
            n_test=CFG["n_test"],
            noise=CFG["noise"],
            seed=seed,
        )
        for rule in make_rules():
            net = make_substrate(seed)
            log = run_sequence(
                rule, tasks, net,
                n_epochs=CFG["n_epochs"],
                batch_size=CFG["batch_size"],
                seed=seed,
            )
            heat_per_task = np.array(
                [float(t["heat_total"]) for t in log["tasks"]]
            )
            mse_each_task = np.array(
                [t["mse_test_each_task"] for t in log["tasks"]]
            )
            edge_heat_h = np.zeros((CFG["rows"], CFG["cols"] - 1))
            edge_heat_v = np.zeros((CFG["rows"] - 1, CFG["cols"]))
            for t in log["tasks"]:
                eh, ev = _collapse_to_edges(t["per_edge_heat_layers"])
                edge_heat_h += eh
                edge_heat_v += ev
            records.setdefault(rule.name, []).append({
                "seed": seed,
                "heat_per_task": heat_per_task,
                "mse_each_task": mse_each_task,
                "edge_heat_h": edge_heat_h,
                "edge_heat_v": edge_heat_v,
            })
        print(f"  seed={seed} done")
    return records


def compute_stats(records):
    """Per-rule scalars used for the Pareto and the summary table."""
    stats = {}
    for name, recs in records.items():
        cum_heat = np.array([r["heat_per_task"].sum() for r in recs])
        final = np.stack([r["mse_each_task"][-1] for r in recs])
        past = final[:, :-1].mean(axis=1)
        current = final[:, -1]
        overall = final.mean(axis=1)
        stats[name] = {
            "cum_heat_mu": float(cum_heat.mean()),
            "cum_heat_se": float(cum_heat.std() / np.sqrt(len(cum_heat))),
            "past_mu": float(past.mean()),
            "past_se": float(past.std() / np.sqrt(len(past))),
            "current_mu": float(current.mean()),
            "current_se": float(current.std() / np.sqrt(len(current))),
            "overall_mu": float(overall.mean()),
            "overall_se": float(overall.std() / np.sqrt(len(overall))),
        }
    return stats


def summarize(records, stats):
    print()
    print("Final MSE and cumulative dissipation:")
    print(f"  {'rule':<22}{'cum heat':>14}{'past':>10}{'current':>10}{'overall':>10}")
    for name, s in stats.items():
        print(
            f"  {name:<22}"
            f"{s['cum_heat_mu']:>9.2f}±{s['cum_heat_se']:<3.2f}"
            f"{s['past_mu']:>10.3f}"
            f"{s['current_mu']:>10.3f}"
            f"{s['overall_mu']:>10.3f}"
        )

    print()
    print("Paired vs vanilla (negative = lower dissipation / lower error):")
    print(f"  {'rule':<22}{'Δ cum heat':>14}{'Δ overall MSE':>18}")
    v_cum = np.array([r["heat_per_task"].sum() for r in records["vanilla"]])
    v_overall = np.array(
        [r["mse_each_task"][-1].mean() for r in records["vanilla"]]
    )
    for name, recs in records.items():
        if name == "vanilla":
            continue
        cum_heat = np.array([r["heat_per_task"].sum() for r in recs])
        overall = np.array([r["mse_each_task"][-1].mean() for r in recs])
        dh = cum_heat - v_cum
        do = overall - v_overall
        print(
            f"  {name:<22}"
            f"{dh.mean():>9.2f}±{dh.std() / np.sqrt(len(dh)):<3.2f}"
            f"{do.mean():>13.3f}±{do.std() / np.sqrt(len(do)):<3.3f}"
        )


def to_jsonable(records, stats):
    return {
        "config": CFG,
        "rule_stats": stats,
        "records": {
            name: [
                {
                    "seed": int(r["seed"]),
                    "heat_per_task": r["heat_per_task"].tolist(),
                    "mse_each_task": r["mse_each_task"].tolist(),
                    "edge_heat_h": r["edge_heat_h"].tolist(),
                    "edge_heat_v": r["edge_heat_v"].tolist(),
                }
                for r in recs
            ]
            for name, recs in records.items()
        },
    }


def plot_thermo(records, stats, save_path):
    rule_names = list(records.keys())
    n_rules = len(rule_names)

    fig = plt.figure(figsize=(4.0 * n_rules, 12.5))
    gs = fig.add_gridspec(
        3, n_rules,
        height_ratios=[1.05, 1.0, 0.9],
        hspace=0.55, wspace=0.35,
    )

    # ---- Panel A: retention–dissipation Pareto (top row, spans all columns)
    ax_pareto = fig.add_subplot(gs[0, :])
    for name in rule_names:
        s = stats[name]
        ax_pareto.errorbar(
            s["cum_heat_mu"], s["overall_mu"],
            xerr=s["cum_heat_se"], yerr=s["overall_se"],
            fmt="o", color=RULE_COLOR.get(name, "k"),
            markersize=11, capsize=4, linewidth=1.4,
            markeredgecolor="black", markeredgewidth=0.6,
            label=RULE_LABEL.get(name, name),
        )
        ax_pareto.annotate(
            RULE_LABEL.get(name, name),
            (s["cum_heat_mu"], s["overall_mu"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=RULE_COLOR.get(name, "k"),
        )
    ax_pareto.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax_pareto.set_xlabel("cumulative dissipation Σ Q_total over 8-task sequence")
    ax_pareto.set_ylabel("overall final MSE")
    ax_pareto.set_title(
        "Retention–dissipation Pareto "
        "(bottom-left = high retention at low heat; dotted = no-info MSE)"
    )
    ax_pareto.legend(fontsize=8.5, loc="best", framealpha=0.9)
    ax_pareto.grid(True, alpha=0.25)

    # ---- Panel B: per-edge spatial heat maps (middle row, one per rule)
    # Shared color scale across rules so concentration patterns are
    # visually comparable.
    edge_heat_mean = {}
    for name in rule_names:
        eh_h = np.mean(np.stack([r["edge_heat_h"] for r in records[name]]), axis=0)
        eh_v = np.mean(np.stack([r["edge_heat_v"] for r in records[name]]), axis=0)
        edge_heat_mean[name] = (eh_h, eh_v)
    vmax = max(
        max(np.abs(eh_h).max(), np.abs(eh_v).max())
        for (eh_h, eh_v) in edge_heat_mean.values()
    )

    for col, name in enumerate(rule_names):
        ax = fig.add_subplot(gs[1, col])
        eh_h, eh_v = edge_heat_mean[name]
        _draw_mesh(
            ax,
            rows=CFG["rows"], cols=CFG["cols"],
            n_input=CFG["n_sensory"],
            out_pos_row=CFG["out_pos_row"],
            out_neg_row=CFG["out_neg_row"],
            W_h=eh_h, W_v=eh_v,
            cmap_name="inferno",
            vmax=vmax,
            max_lw=4.5,
        )
        ax.set_title(
            f"{RULE_LABEL.get(name, name)}\n|per-edge heat|, mean across seeds",
            fontsize=9,
        )

    # ---- Panel C: per-task heat (bottom-left half) and cumulative heat
    # trajectory (bottom-right half).
    split = max(1, n_rules // 2)
    ax_task = fig.add_subplot(gs[2, :split])
    ax_cum = fig.add_subplot(gs[2, split:])
    task_axis = np.arange(1, CFG["n_tasks"] + 1)

    for name in rule_names:
        heat_traj = np.stack([r["heat_per_task"] for r in records[name]])
        mu = heat_traj.mean(axis=0)
        se = heat_traj.std(axis=0) / np.sqrt(heat_traj.shape[0])
        ax_task.errorbar(
            task_axis, mu, yerr=se,
            color=RULE_COLOR.get(name, "k"),
            label=RULE_LABEL.get(name, name),
            linewidth=1.6, marker="o", markersize=4, capsize=2,
        )
        cum = heat_traj.cumsum(axis=1)
        cmu = cum.mean(axis=0)
        cse = cum.std(axis=0) / np.sqrt(cum.shape[0])
        ax_cum.fill_between(
            task_axis, cmu - cse, cmu + cse,
            color=RULE_COLOR.get(name, "k"), alpha=0.18,
        )
        ax_cum.plot(
            task_axis, cmu,
            color=RULE_COLOR.get(name, "k"),
            label=RULE_LABEL.get(name, name),
            linewidth=1.8, marker="o", markersize=4,
        )

    ax_task.set_xlabel("task index (A=1 .. H=8)")
    ax_task.set_ylabel("heat Q_total per task")
    ax_task.set_title("Per-task dissipation")
    ax_task.grid(True, alpha=0.25)
    ax_task.legend(fontsize=8)

    ax_cum.set_xlabel("task index (A=1 .. H=8)")
    ax_cum.set_ylabel("cumulative Σ Q_total through task k")
    ax_cum.set_title("Cumulative dissipation trajectory")
    ax_cum.grid(True, alpha=0.25)
    ax_cum.legend(fontsize=8)

    fig.suptitle(
        "Thermodynamic analysis on the eight-task trainable context-modulated mesh "
        f"({CFG['n_seeds']} seeds)",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", action="store_true",
        help="Small smoke run (2 seeds, 10 epochs/task) to verify the pipeline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        CFG["n_seeds"] = 2
        CFG["n_epochs"] = 10

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    print("Thermodynamic analysis (8 tasks, trainable context-modulated mesh)")
    print(
        f"  seeds={CFG['n_seeds']}  epochs/task={CFG['n_epochs']}  "
        f"lr={CFG['lr']}  rules=vanilla / threshold(tau={CFG['thresh_tau']}) "
        f"/ cum_imp(λ={CFG['lam_cum']}) / slow_consolidated(λ={CFG['lam_slow']})"
    )

    records = run_all()
    stats = compute_stats(records)
    summarize(records, stats)

    tag = "_smoke" if args.smoke else ""
    json_path = out_dir / f"thermodynamic_analysis{tag}.json"
    png_path = out_dir / f"thermodynamic_analysis{tag}.png"
    with open(json_path, "w") as f:
        json.dump(to_jsonable(records, stats), f, indent=2)
    plot_thermo(records, stats, png_path)
    print(f"\nSaved {json_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()

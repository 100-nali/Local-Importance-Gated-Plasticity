"""
Baseline sweep for the trainable context-modulated mesh.

Purpose:
  1. Verify the corrected contextual task setup over enough seeds.
  2. Measure single-task capacity task-by-task.
  3. Compare vanilla, threshold, slow consolidation, and a lambda sweep of
     cumulative importance gating.
  4. Report paired effect sizes relative to vanilla.

Run:
    python context_modulated_baseline_sweep.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import run_sequence
from learning_rules import (
    ActivityCumulativeImportanceRule,
    CumulativeImportanceGatedRule,
    SGDRule,
    SlowConsolidatedImportanceRule,
    ThresholdedSGDRule,
)
from network import TrainableContextModulatedMeshSubstrate
from tasks import make_modulated_context_task_sequence


CFG = {
    "rows": 8,
    "cols": 10,
    "n_sensory": 8,
    "context_dim": 5,
    "n_tasks": 5,
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
    # Gradients in the trainable log/context parameterization are much smaller
    # than raw conductance gradients in the original mesh experiments.
    "thresh_tau": 1e-4,
    "lambdas_cum": [5, 10, 20, 50, 100, 200],
    "lam_slow": 0.3,
    # Activity importance is on the same scale as squared voltage drops
    # rather than squared gradients, ~10^5x larger than per-step g^2 at
    # init, so its lambda decade is correspondingly smaller. Single-seed
    # smoke puts the working point near 3e-3.
    "lambdas_act": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "act_beta": 0.95,
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


def make_tasks(seed):
    return make_modulated_context_task_sequence(
        input_dim=CFG["n_sensory"],
        n_tasks=CFG["n_tasks"],
        context_dim=CFG["context_dim"],
        n_train=CFG["n_train"],
        n_test=CFG["n_test"],
        noise=CFG["noise"],
        seed=seed,
    )


def run_rule(rule_factory):
    finals = []
    stage_mse = []
    for seed in range(CFG["n_seeds"]):
        tasks = make_tasks(seed)
        net = make_substrate(seed)
        log = run_sequence(
            rule_factory(),
            tasks,
            net,
            n_epochs=CFG["n_epochs"],
            batch_size=CFG["batch_size"],
            seed=seed,
        )
        mat = np.array([t["mse_test_each_task"] for t in log["tasks"]])
        stage_mse.append(mat)
        finals.append(mat[-1])
    return {
        "stage_mse": np.array(stage_mse),
        "final_mse": np.array(finals),
    }


def run_single_task_capacity():
    """Train one task at a time from scratch with vanilla."""
    by_seed = []
    for seed in range(CFG["n_seeds"]):
        tasks = make_tasks(seed)
        row = []
        for task in tasks:
            net = make_substrate(seed)
            log = run_sequence(
                SGDRule(lr=CFG["lr"]),
                [task],
                net,
                n_epochs=CFG["n_epochs"],
                batch_size=CFG["batch_size"],
                seed=seed,
            )
            row.append(log["tasks"][-1]["mse_test_each_task"][0])
        by_seed.append(row)
    return np.array(by_seed)


def mean_se(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    return arr.mean(axis=axis), arr.std(axis=axis) / np.sqrt(arr.shape[axis])


def summarize(records, capacity):
    print()
    cap_mu, cap_se = mean_se(capacity, axis=0)
    print("Single-task vanilla capacity:")
    print("  " + " ".join(
        f"{chr(ord('A') + i)}={cap_mu[i]:.3f}+/-{cap_se[i]:.3f}"
        for i in range(CFG["n_tasks"])
    ))
    print(f"  overall={capacity.mean():.3f}")

    print()
    print("Sequential final MSE:")
    print(f"  {'rule':<18}{'past':>9}{'current':>10}{'overall':>10}")
    for name, rec in records.items():
        final = rec["final_mse"]
        past = final[:, :-1].mean(axis=1)
        current = final[:, -1]
        overall = final.mean(axis=1)
        print(
            f"  {name:<18}"
            f"{past.mean():>9.3f}"
            f"{current.mean():>10.3f}"
            f"{overall.mean():>10.3f}"
        )

    vanilla = records["vanilla"]["final_mse"]
    vanilla_past = vanilla[:, :-1].mean(axis=1)
    vanilla_current = vanilla[:, -1]

    print()
    print("Paired effect vs vanilla (negative is better):")
    print(f"  {'rule':<18}{'past Δ':>9}{'current Δ':>12}{'overall Δ':>12}")
    for name, rec in records.items():
        if name == "vanilla":
            continue
        final = rec["final_mse"]
        past_diff = final[:, :-1].mean(axis=1) - vanilla_past
        current_diff = final[:, -1] - vanilla_current
        overall_diff = final.mean(axis=1) - vanilla.mean(axis=1)
        print(
            f"  {name:<18}"
            f"{past_diff.mean():>9.3f}"
            f"{current_diff.mean():>12.3f}"
            f"{overall_diff.mean():>12.3f}"
        )


def to_jsonable(records, capacity):
    out = {
        "config": CFG,
        "single_task_capacity": capacity.tolist(),
        "records": {},
    }
    for name, rec in records.items():
        out["records"][name] = {
            "stage_mse": rec["stage_mse"].tolist(),
            "final_mse": rec["final_mse"].tolist(),
        }
    return out


def plot(records, capacity, save_path):
    names = list(records.keys())
    stats = {}
    for name in names:
        final = records[name]["final_mse"]
        past = final[:, :-1].mean(axis=1)
        current = final[:, -1]
        overall = final.mean(axis=1)
        stats[name] = {
            "past_mu": past.mean(),
            "past_se": past.std() / np.sqrt(len(past)),
            "current_mu": current.mean(),
            "current_se": current.std() / np.sqrt(len(current)),
            "overall_mu": overall.mean(),
            "overall_se": overall.std() / np.sqrt(len(overall)),
        }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(names))

    ax = axes[0]
    def plot_lambda_sweep(prefix, color, marker, label, endpoint_offsets):
        sweep_names = sorted(
            [name for name in names if name.startswith(prefix)],
            key=lambda s: float(s.split("=")[1]),
        )
        if not sweep_names:
            return None
        p = np.array([stats[name]["past_mu"] for name in sweep_names])
        c = np.array([stats[name]["current_mu"] for name in sweep_names])
        p_se = np.array([stats[name]["past_se"] for name in sweep_names])
        c_se = np.array([stats[name]["current_se"] for name in sweep_names])
        ax.plot(p, c, color=color, linewidth=1.8, alpha=0.8, label=label)
        ax.errorbar(p, c, xerr=p_se, yerr=c_se, fmt=marker, color=color,
                    markersize=5, capsize=2, linewidth=0.8)
        for name, offset in zip((sweep_names[0], sweep_names[-1]), endpoint_offsets):
            lam = name.split("=")[1]
            ax.annotate(
                f"λ={lam}",
                (stats[name]["past_mu"], stats[name]["current_mu"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color=color,
                ha="right" if offset[0] < 0 else "left",
            )
        return sweep_names

    cum_names = plot_lambda_sweep(
        "cum_lam=", "C3", "o", "cum_imp_gated sweep",
        [(-6, -12), (-6, 8)],
    )

    baseline_styles = {
        "vanilla": ("s", "C0", "vanilla"),
        "thresh": ("D", "C1", "threshold"),
        "slow_consolidated": ("X", "C6", "slow consolidation"),
    }
    for name, (marker, color, label) in baseline_styles.items():
        if name not in stats:
            continue
        ax.errorbar(
            stats[name]["past_mu"],
            stats[name]["current_mu"],
            xerr=stats[name]["past_se"],
            yerr=stats[name]["current_se"],
            fmt=marker,
            color=color,
            markersize=8,
            capsize=3,
            linewidth=1.0,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=label,
        )

    if cum_names:
        sweep_names = cum_names
        color = "C3"
        yoff = -18
        best_name = min(sweep_names, key=lambda name: stats[name]["overall_mu"])
        ax.scatter(
            [stats[best_name]["past_mu"]],
            [stats[best_name]["current_mu"]],
            s=160,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            zorder=5,
        )
        ax.annotate(
            f"best λ={best_name.split('=')[1]}",
            (stats[best_name]["past_mu"], stats[best_name]["current_mu"]),
            xytext=(8, yoff),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    axes[0].axvline(1.0, color="gray", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("past-mean MSE")
    axes[0].set_ylabel("current-task MSE")
    axes[0].set_title("Stability-plasticity")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    axes[0].grid(True, alpha=0.25)

    overall_mu = [stats[name]["overall_mu"] for name in names]
    overall_se = [stats[name]["overall_se"] for name in names]
    axes[1].bar(x, overall_mu, yerr=overall_se, capsize=3)
    axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=35, ha="right")
    axes[1].set_ylabel("mean final MSE over tasks")
    axes[1].set_title("Overall final error")
    axes[1].grid(True, alpha=0.25, axis="y")

    cap_mu, cap_se = mean_se(capacity, axis=0)
    task_x = np.arange(CFG["n_tasks"])
    axes[2].bar(task_x, cap_mu, yerr=cap_se, capsize=3, color="C7")
    axes[2].axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    axes[2].set_xticks(task_x)
    axes[2].set_xticklabels([chr(ord("A") + i) for i in range(CFG["n_tasks"])])
    axes[2].set_ylabel("single-task MSE")
    axes[2].set_title("Vanilla single-task capacity")
    axes[2].grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "Trainable context-modulated mesh baseline sweep "
        f"({CFG['n_seeds']} seeds, lr={CFG['lr']})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    print("Trainable context-modulated baseline sweep")
    print(f"  seeds={CFG['n_seeds']}  epochs/task={CFG['n_epochs']}  "
          f"lr={CFG['lr']}  context_dim={CFG['context_dim']}")

    print("single-task capacity...")
    capacity = run_single_task_capacity()

    records = {}
    rule_factories = {
        "vanilla": lambda: SGDRule(lr=CFG["lr"]),
        "thresh": lambda: ThresholdedSGDRule(
            lr=CFG["lr"], threshold=CFG["thresh_tau"],
        ),
        "slow_consolidated": lambda: SlowConsolidatedImportanceRule(
            lr=CFG["lr"], lam=CFG["lam_slow"],
        ),
    }
    for lam in CFG["lambdas_cum"]:
        rule_factories[f"cum_lam={lam}"] = (
            lambda lam=lam: CumulativeImportanceGatedRule(
                lr=CFG["lr"], lam=lam,
            )
        )
    for lam in CFG["lambdas_act"]:
        rule_factories[f"act_lam={lam}"] = (
            lambda lam=lam: ActivityCumulativeImportanceRule(
                lr=CFG["lr"], lam=lam, beta=CFG["act_beta"],
            )
        )

    for name, factory in rule_factories.items():
        print(f"{name}...")
        records[name] = run_rule(factory)

    payload = to_jsonable(records, capacity)
    with open(out_dir / "context_modulated_baseline_sweep.json", "w") as f:
        json.dump(payload, f, indent=2)
    plot(records, capacity, out_dir / "context_modulated_baseline_sweep.png")
    summarize(records, capacity)
    print(f"\nSaved {out_dir / 'context_modulated_baseline_sweep.json'}")
    print(f"Saved {out_dir / 'context_modulated_baseline_sweep.png'}")


if __name__ == "__main__":
    main()

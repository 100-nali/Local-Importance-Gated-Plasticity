"""
Compare importance-gated variants on the trainable context-modulated mesh.

This is separate from context_modulated_prototype.py so the quick prototype
plot stays readable. Here we sweep the major importance variants and compare
their stability-plasticity tradeoffs against vanilla.

Run:
    python context_modulated_importance_variants.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import run_sequence
from learning_rules import (
    CumulativeImportanceGatedRule,
    HeatCumImportanceGatedRule,
    MultiAnchorImportanceGatedRule,
    SGDRule,
    SlowConsolidatedImportanceRule,
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
    "n_seeds": 6,
    "lr": 20.0,
    "eta": 0.005,
    "context_init_std": 0.0,
    "max_log_extra": 4.0,
    "lambdas_cum": [10, 20, 50, 100, 200],
    "lambdas_multi": [10, 20, 50, 100, 200],
    "lambdas_heat": [0.0001, 0.0003, 0.001, 0.003, 0.01],
    "heat_xi": 1e-3,
    "lambdas_slow": [0.03, 0.1, 0.3, 1.0],
}


STYLE = {
    "cum_imp_gated": ("C3", "o", "cum. importance"),
    "multi_anchor": ("C4", "v", "multi-anchor"),
    "heat_cum_imp_gated": ("C5", "P", "SI / heat importance"),
    "slow_consolidated": ("C6", "X", "slow consolidation"),
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
        finals.append(log["tasks"][-1]["mse_test_each_task"])
    return np.array(finals)


def summarize(records):
    print()
    print(f"{'rule':<28}{'past':>9}{'current':>10}{'overall':>10}")
    best = None
    for name, final in records.items():
        past = final[:, :-1].mean(axis=1)
        current = final[:, -1]
        overall = final.mean(axis=1)
        print(f"{name:<28}{past.mean():>9.3f}{current.mean():>10.3f}{overall.mean():>10.3f}")
        if name != "vanilla" and (best is None or overall.mean() < best[0]):
            best = (overall.mean(), name, past.mean(), current.mean())
    if best is not None:
        print()
        print(f"Best variant by overall MSE: {best[1]} "
              f"(past={best[2]:.3f}, current={best[3]:.3f}, overall={best[0]:.3f})")


def to_jsonable(records):
    return {
        "config": CFG,
        "records": {name: final.tolist() for name, final in records.items()},
    }


def _stats(final):
    past = final[:, :-1].mean(axis=1)
    current = final[:, -1]
    overall = final.mean(axis=1)
    return {
        "past_mu": past.mean(),
        "past_se": past.std() / np.sqrt(len(past)),
        "current_mu": current.mean(),
        "current_se": current.std() / np.sqrt(len(current)),
        "overall_mu": overall.mean(),
        "overall_se": overall.std() / np.sqrt(len(overall)),
    }


def plot(records, save_path):
    stats = {name: _stats(final) for name, final in records.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    v = stats["vanilla"]
    ax.errorbar(v["past_mu"], v["current_mu"],
                xerr=v["past_se"], yerr=v["current_se"],
                fmt="s", color="C0", markersize=9, capsize=3,
                markeredgecolor="black", markeredgewidth=0.4,
                label="vanilla")

    for prefix, (color, marker, label) in STYLE.items():
        names = sorted(
            [name for name in records if name.startswith(prefix + ":")],
            key=lambda s: float(s.split("=")[1]),
        )
        if not names:
            continue
        x = np.array([stats[name]["past_mu"] for name in names])
        y = np.array([stats[name]["current_mu"] for name in names])
        xerr = np.array([stats[name]["past_se"] for name in names])
        yerr = np.array([stats[name]["current_se"] for name in names])
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8, label=label)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, color=color,
                    markersize=6, capsize=2, linewidth=0.7,
                    markeredgecolor="black", markeredgewidth=0.3)

        best = min(names, key=lambda name: stats[name]["overall_mu"])
        ax.scatter([stats[best]["past_mu"]], [stats[best]["current_mu"]],
                   s=140, facecolors="none", edgecolors=color,
                   linewidths=1.6, zorder=5)
        ax.annotate(
            f"best {best.split('=')[1]}",
            (stats[best]["past_mu"], stats[best]["current_mu"]),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("past-mean MSE")
    ax.set_ylabel("current-task MSE")
    ax.set_title("Importance-variant stability-plasticity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)

    names = list(records.keys())
    overall = [stats[name]["overall_mu"] for name in names]
    overall_se = [stats[name]["overall_se"] for name in names]
    x = np.arange(len(names))
    axes[1].bar(x, overall, yerr=overall_se, capsize=2)
    axes[1].axhline(stats["vanilla"]["overall_mu"], color="C0",
                    linestyle="--", linewidth=1.0, label="vanilla")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("mean final MSE over tasks")
    axes[1].set_title("Overall final error")
    axes[1].grid(True, alpha=0.25, axis="y")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Importance-gated variants on trainable context-modulated mesh "
        f"({CFG['n_seeds']} seeds)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    print("Importance variants on trainable context-modulated mesh")
    print(f"  seeds={CFG['n_seeds']} epochs/task={CFG['n_epochs']} lr={CFG['lr']}")

    records = {}
    print("vanilla...")
    records["vanilla"] = run_rule(lambda: SGDRule(lr=CFG["lr"]))

    for lam in CFG["lambdas_cum"]:
        name = f"cum_imp_gated:lam={lam}"
        print(name + "...")
        records[name] = run_rule(
            lambda lam=lam: CumulativeImportanceGatedRule(lr=CFG["lr"], lam=lam)
        )

    for lam in CFG["lambdas_multi"]:
        name = f"multi_anchor:lam={lam}"
        print(name + "...")
        records[name] = run_rule(
            lambda lam=lam: MultiAnchorImportanceGatedRule(lr=CFG["lr"], lam=lam)
        )

    for lam in CFG["lambdas_heat"]:
        name = f"heat_cum_imp_gated:lam={lam}"
        print(name + "...")
        records[name] = run_rule(
            lambda lam=lam: HeatCumImportanceGatedRule(
                lr=CFG["lr"], lam=lam, xi=CFG["heat_xi"],
            )
        )

    for lam in CFG["lambdas_slow"]:
        name = f"slow_consolidated:lam={lam}"
        print(name + "...")
        records[name] = run_rule(
            lambda lam=lam: SlowConsolidatedImportanceRule(lr=CFG["lr"], lam=lam)
        )

    with open(out_dir / "context_modulated_importance_variants.json", "w") as f:
        json.dump(to_jsonable(records), f, indent=2)
    plot(records, out_dir / "context_modulated_importance_variants.png")
    summarize(records)
    print(f"\nSaved {out_dir / 'context_modulated_importance_variants.json'}")
    print(f"Saved {out_dir / 'context_modulated_importance_variants.png'}")


if __name__ == "__main__":
    main()

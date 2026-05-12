"""
Standalone quick test for the fixed-size trainable context-modulated mesh.

This does not run the full multi-task experiment. It only asks whether the
trainable context sensitivities can learn a sequence of identifiable
orthogonal regression tasks with one shared output pair.

Run:
    python context_modulated_prototype.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import run_sequence
from learning_rules import (
    CumulativeImportanceGatedRule,
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
    "curve_eval_every": 16,
    "n_seeds": 3,
    "lr": 20.0,
    "eta": 0.005,
    "context_init_std": 0.0,
    "max_log_extra": 4.0,
    "lam_cum": 50.0,
    "lam_slow": 0.3,
}


RULE_COLOR = {
    "vanilla": "C0",
    "cum_imp_gated": "C3",
    "slow_consolidated": "C6",
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
        CumulativeImportanceGatedRule(lr=CFG["lr"], lam=CFG["lam_cum"]),
        SlowConsolidatedImportanceRule(lr=CFG["lr"], lam=CFG["lam_slow"]),
    ]


def run_all():
    results = {}
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
                rule,
                tasks,
                net,
                n_epochs=CFG["n_epochs"],
                batch_size=CFG["batch_size"],
                seed=seed,
                trace_mse=True,
                eval_every=CFG["curve_eval_every"],
            )
            results.setdefault(rule.name, {})[seed] = {
                "stage_mse": [t["mse_test_each_task"] for t in log["tasks"]],
                "learning_curve": log["learning_curve"],
            }
        print(f"  seed={seed} done")
    return results


def _stack(results, rule, key):
    seeds = sorted(results[rule].keys())
    return [results[rule][s][key] for s in seeds]


def summarize(results):
    print()
    print("Final MSE after contextual sequence:")
    print(f"  {'rule':<18}" + "".join(f"{chr(ord('A') + j):>9}" for j in range(CFG["n_tasks"]))
          + f"  {'past-mean':>11}  {'current':>9}")
    for rule in results:
        mats = np.array(_stack(results, rule, "stage_mse"))
        final = mats[:, -1, :]
        mu = final.mean(axis=0)
        past = mu[:-1].mean()
        current = mu[-1]
        row = "".join(f"{v:>9.3f}" for v in mu)
        print(f"  {rule:<18}{row}  {past:>11.3f}  {current:>9.3f}")


def plot_results(results, save_path):
    names = [chr(ord("A") + i) for i in range(CFG["n_tasks"])]
    rules = list(results.keys())
    fig, axes = plt.subplots(len(rules), 2, figsize=(12, 3.2 * len(rules)),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")

    for row, rule in enumerate(rules):
        mats = np.array(_stack(results, rule, "stage_mse"))
        final = mats[:, -1, :]
        mu = final.mean(axis=0)
        se = final.std(axis=0) / np.sqrt(final.shape[0])
        axes[row, 0].bar(np.arange(CFG["n_tasks"]), mu, yerr=se,
                         capsize=2, color=RULE_COLOR.get(rule, "C0"))
        axes[row, 0].axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
        axes[row, 0].set_xticks(np.arange(CFG["n_tasks"]))
        axes[row, 0].set_xticklabels(names)
        axes[row, 0].set_ylabel("final test MSE")
        axes[row, 0].set_title(rule)
        axes[row, 0].grid(True, alpha=0.25, axis="y")

        curves = _stack(results, rule, "learning_curve")
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        x = np.array([p["global_iteration"] for p in curves[0]])
        mse = np.array([[p["mse_test_each_task"] for p in c] for c in curves])
        curve_mu = mse.mean(axis=0)
        for j in range(CFG["n_tasks"]):
            color = cmap(j / max(CFG["n_tasks"] - 1, 1))
            axes[row, 1].plot(x, curve_mu[:, j], color=color, label=f"task {names[j]}")
        axes[row, 1].axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
        axes[row, 1].set_ylabel("test MSE")
        axes[row, 1].set_title(f"{rule} learning curves")
        axes[row, 1].grid(True, alpha=0.25)

    axes[-1, 0].set_xlabel("task")
    axes[-1, 1].set_xlabel("minibatch update iteration")
    axes[0, 1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(
        "Context-modulated mesh prototype "
        f"({CFG['n_seeds']} seeds, {CFG['n_epochs']} epochs/task)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    print("Context-modulated mesh prototype")
    print(f"  tasks={CFG['n_tasks']}  seeds={CFG['n_seeds']}  "
          f"epochs/task={CFG['n_epochs']}  lr={CFG['lr']}")
    print(f"  mesh={CFG['rows']}x{CFG['cols']}  sensory={CFG['n_sensory']}  "
          f"context_dim={CFG['context_dim']}  trainable_context=True")

    results = run_all()
    with open(out_dir / "context_modulated_prototype.json", "w") as f:
        json.dump({"config": CFG, "results": results}, f, indent=2)
    plot_results(results, out_dir / "context_modulated_prototype.png")
    summarize(results)
    print(f"\nSaved {out_dir / 'context_modulated_prototype.json'}")
    print(f"Saved {out_dir / 'context_modulated_prototype.png'}")


if __name__ == "__main__":
    main()

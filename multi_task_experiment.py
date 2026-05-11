"""
Five-task sequential learning on the mesh substrate. Orthogonal tasks (max
interference) — after training on each task k we evaluate MSE on all five.
The question this experiment asks: does the importance-gated rule retain past
tasks better than vanilla and thresh over a longer sequence than the 2-task
baseline?

Run:
    python multi_task_experiment.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from network import MeshCoupledSubstrate
from tasks import make_orthogonal_task_sequence
from learning_rules import (SGDRule, ThresholdedSGDRule,
                            ImportanceGatedRule, CumulativeImportanceGatedRule)
from experiments import run_sequence


CFG = {
    "rows": 8, "cols": 10, "n_input": 8,
    "out_pos_row": 3, "out_neg_row": 4,
    "n_tasks": 5,
    "n_train": 500, "n_test": 200, "noise": 0.05,
    "n_epochs": 80, "batch_size": 32,
    "n_seeds": 12,
    "lr": 5.0, "eta": 0.005,
    "lam": 100.0,        # snapshot imp_gated operating point (2-task pareto)
    "lam_cum": 500.0,    # cumulative imp_gated operating point — crosses no-info floor
    "beta": 0.95,
    "thresh_tau": 0.005,
}

RULE_ORDER = ["vanilla", "thresh", "imp_gated", "cum_imp_gated"]
RULE_COLOR = {"vanilla": "C0", "thresh": "C1", "imp_gated": "C2", "cum_imp_gated": "C3"}


def make_substrate(seed):
    return MeshCoupledSubstrate(
        rows=CFG["rows"], cols=CFG["cols"],
        n_input=CFG["n_input"],
        out_pos_row=CFG["out_pos_row"], out_neg_row=CFG["out_neg_row"],
        eta=CFG["eta"], seed=seed,
    )


def make_rules():
    return [
        SGDRule(lr=CFG["lr"]),
        ThresholdedSGDRule(lr=CFG["lr"], threshold=CFG["thresh_tau"]),
        ImportanceGatedRule(lr=CFG["lr"], lam=CFG["lam"], beta=CFG["beta"]),
        CumulativeImportanceGatedRule(lr=CFG["lr"], lam=CFG["lam_cum"], beta=CFG["beta"]),
    ]


def run_all():
    """
    results[rule][seed] = mse_matrix of shape [n_tasks, n_tasks] where
    mse_matrix[k, j] = MSE on task j evaluated AFTER training on task k.
    """
    out = {}
    for seed in range(CFG["n_seeds"]):
        tasks = make_orthogonal_task_sequence(
            input_dim=CFG["n_input"], n_tasks=CFG["n_tasks"],
            n_train=CFG["n_train"], n_test=CFG["n_test"],
            noise=CFG["noise"], seed=seed,
        )
        for rule in make_rules():
            net = make_substrate(seed)
            log = run_sequence(rule, tasks, net,
                                n_epochs=CFG["n_epochs"],
                                batch_size=CFG["batch_size"], seed=seed)
            mat = np.array([t["mse_test_each_task"] for t in log["tasks"]])
            out.setdefault(rule.name, {})[seed] = mat.tolist()
        print(f"  seed={seed} done")
    return out


def _stack(results, rule):
    seeds = sorted(results[rule].keys())
    return np.array([results[rule][s] for s in seeds])  # [n_seeds, n_tasks, n_tasks]


def plot_multi_task(results, save_path):
    n_tasks = CFG["n_tasks"]
    names = [chr(ord("A") + i) for i in range(n_tasks)]
    stages = np.arange(n_tasks)
    rules_present = [r for r in RULE_ORDER if r in results]
    n_rules = len(rules_present)

    fig = plt.figure(figsize=(16.5, 9.5))
    gs = fig.add_gridspec(2, n_rules, height_ratios=[1.1, 1.0],
                          hspace=0.42, wspace=0.30)

    # Top row: per-rule forgetting curves (one line per task)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(n_rules)]
    cmap = plt.get_cmap("viridis")
    y_top_max = 0.0
    for ax, rule in zip(axes_top, rules_present):
        mats = _stack(results, rule)              # [seeds, stage, task]
        mu = mats.mean(axis=0)
        se = mats.std(axis=0) / np.sqrt(mats.shape[0])
        for j in range(n_tasks):
            x = stages[j:]; y = mu[j:, j]; e = se[j:, j]
            ax.errorbar(x, y, yerr=e, marker="o", capsize=2,
                        color=cmap(j / max(n_tasks - 1, 1)),
                        label=f"task {names[j]}")
            y_top_max = max(y_top_max, (y + e).max())
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(rule, fontsize=11)
        ax.set_xticks(stages)
        ax.set_xticklabels([f"after {n}" for n in names], rotation=0, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    for ax in axes_top:
        ax.set_ylim(0, y_top_max * 1.05)
    axes_top[0].set_ylabel("MSE on task j")
    axes_top[-1].legend(fontsize=8, loc="upper left",
                        bbox_to_anchor=(1.02, 1.0), title="evaluated on")

    # Bottom row: 3 summary panels split across the available width
    gs_bot = gs[1, :].subgridspec(1, 3, wspace=0.30)

    # Bottom-left: final MSE per task per rule (end of sequence)
    ax_bar = fig.add_subplot(gs_bot[0, 0])
    width = 0.8 / n_rules
    offsets = (np.arange(n_rules) - (n_rules - 1) / 2.0) * width
    x = np.arange(n_tasks)
    for i, rule in enumerate(rules_present):
        mats = _stack(results, rule)
        final = mats[:, -1, :]
        mu = final.mean(axis=0)
        se = final.std(axis=0) / np.sqrt(final.shape[0])
        ax_bar.bar(x + offsets[i], mu, width, yerr=se, capsize=1.5,
                   color=RULE_COLOR[rule], label=rule)
    ax_bar.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names)
    ax_bar.set_xlabel("task")
    ax_bar.set_ylabel("MSE after final task")
    ax_bar.set_title("Final retention per task (lower = better)", fontsize=10)
    ax_bar.legend(fontsize=8, loc="upper right", ncol=2)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Bottom-middle: avg past-task MSE vs stage k (mean over j < k)
    ax_past = fig.add_subplot(gs_bot[0, 1])
    for rule in rules_present:
        mats = _stack(results, rule)
        past_curve_per_seed = []
        for s in range(mats.shape[0]):
            row = [np.nan]
            for k in range(1, n_tasks):
                row.append(mats[s, k, :k].mean())
            past_curve_per_seed.append(row)
        past = np.array(past_curve_per_seed)
        mu = np.nanmean(past, axis=0)
        se = np.nanstd(past, axis=0) / np.sqrt(past.shape[0])
        ax_past.errorbar(stages, mu, yerr=se, marker="o", capsize=2,
                         color=RULE_COLOR[rule], label=rule)
    ax_past.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
                    label="no-info floor")
    ax_past.set_xticks(stages)
    ax_past.set_xticklabels([f"after {n}" for n in names], fontsize=8)
    ax_past.set_xlabel("training stage")
    ax_past.set_ylabel("mean MSE on past tasks")
    ax_past.set_title("Past-task retention (avg over j<k)", fontsize=10)
    ax_past.legend(fontsize=8, loc="best")
    ax_past.grid(True, alpha=0.3)

    # Bottom-right: current-task fit vs stage k (diagonal)
    ax_cur = fig.add_subplot(gs_bot[0, 2])
    for rule in rules_present:
        mats = _stack(results, rule)
        diag = np.array([np.diag(mats[s]) for s in range(mats.shape[0])])
        mu = diag.mean(axis=0)
        se = diag.std(axis=0) / np.sqrt(diag.shape[0])
        ax_cur.errorbar(stages, mu, yerr=se, marker="o", capsize=2,
                        color=RULE_COLOR[rule], label=rule)
    ax_cur.set_xticks(stages)
    ax_cur.set_xticklabels([f"after {n}" for n in names], fontsize=8)
    ax_cur.set_xlabel("training stage k")
    ax_cur.set_ylabel("MSE on task k")
    ax_cur.set_title("Current-task fit (diagonal)", fontsize=10)
    ax_cur.legend(fontsize=8)
    ax_cur.grid(True, alpha=0.3)

    fig.suptitle(
        f"Five orthogonal tasks on the mesh substrate "
        f"({CFG['n_seeds']} seeds, lr={CFG['lr']}, "
        f"imp_gated λ={CFG['lam']:g}, cum_imp_gated λ={CFG['lam_cum']:g}, "
        f"τ={CFG['thresh_tau']})",
        fontsize=12, y=0.995,
    )
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def _summarize(results):
    n_tasks = CFG["n_tasks"]
    print()
    print("Final retention (MSE per task after training all 5):")
    print(f"  {'rule':<16}" + "".join(f"{chr(ord('A')+j):>9}" for j in range(n_tasks))
          + f"  {'past-mean':>11}  {'current':>9}")
    for rule in [r for r in RULE_ORDER if r in results]:
        mats = _stack(results, rule)
        final = mats[:, -1, :].mean(axis=0)
        past_mean = final[:-1].mean()
        current = final[-1]
        row = "".join(f"{v:>9.3f}" for v in final)
        print(f"  {rule:<16}{row}  {past_mean:>11.3f}  {current:>9.3f}")


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    print(f"Five-task sequential experiment on coupled mesh substrate")
    print(f"  n_tasks={CFG['n_tasks']}  n_seeds={CFG['n_seeds']}  "
          f"lr={CFG['lr']}  lam={CFG['lam']}  thresh_tau={CFG['thresh_tau']}")
    results = run_all()

    with open(out_dir / "multi_task.json", "w") as f:
        json.dump({"config": CFG, "results": results}, f, indent=2)
    print(f"Saved {out_dir / 'multi_task.json'}")

    plot_multi_task(results, out_dir / "multi_task.png")
    print(f"Saved {out_dir / 'multi_task.png'}")

    _summarize(results)


if __name__ == "__main__":
    main()

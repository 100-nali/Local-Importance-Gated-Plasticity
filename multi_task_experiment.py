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

from network import ContextModulatedMeshSubstrate
from tasks import make_modulated_context_task_sequence
from learning_rules import (SGDRule, ThresholdedSGDRule,
                            CumulativeImportanceGatedRule,
                            SlowConsolidatedImportanceRule)
from experiments import run_sequence


CFG = {
    # Fixed-size context-modulated mesh: 8 sensory inputs plus a compact
    # context code that multiplicatively modulates edge conductances.
    "n_sensory": 8,
    "context_dim": 5,
    "n_tasks": 5,
    "rows": 8, "cols": 10,
    "out_pos_row": 3, "out_neg_row": 4,
    "context_gain_std": 0.35,
    "max_log_gain": 1.0,
    "n_train": 500, "n_test": 200, "noise": 0.05,
    "n_epochs": 80, "batch_size": 32,
    # MSE learning curves are evaluated every this many minibatch updates.
    # 16 = once per epoch for n_train=500, batch_size=32.
    "curve_eval_every": 16,
    "n_seeds": 12,
    "lr": 5.0, "eta": 0.005,
    # Balanced operating points: lower than the no-info-floor crossing lambdas,
    # because lambdas like 500 mostly suppress plasticity (current MSE ~ 1).
    "lam_cum": 100.0,
    "lam_slow": 1.0,
    "slow_pull": 0.02,
    "slow_consolidation_rate": 0.25,
    "slow_stability_decay": 0.95,
    "slow_stability_cap": 3.0,
    "slow_importance_scale": 1.0,
    "slow_xi": 1e-3,
    "beta": 0.95,
    "thresh_tau": 0.005,
}

RULE_ORDER = [
    "vanilla",
    "thresh",
    "cum_imp_gated",
    "slow_consolidated",
]
RULE_COLOR = {
    "vanilla": "C0",
    "thresh": "C1",
    "cum_imp_gated": "C3",
    "slow_consolidated": "C6",
}

NO_INFO_MSE = 1.0


def make_substrate(seed):
    return ContextModulatedMeshSubstrate(
        rows=CFG["rows"], cols=CFG["cols"],
        n_sensory=CFG["n_sensory"],
        context_dim=CFG["context_dim"],
        out_pos_row=CFG["out_pos_row"], out_neg_row=CFG["out_neg_row"],
        eta=CFG["eta"],
        context_gain_std=CFG["context_gain_std"],
        max_log_gain=CFG["max_log_gain"],
        seed=seed,
    )


def make_rules():
    return [
        SGDRule(lr=CFG["lr"]),
        ThresholdedSGDRule(lr=CFG["lr"], threshold=CFG["thresh_tau"]),
        CumulativeImportanceGatedRule(lr=CFG["lr"], lam=CFG["lam_cum"], beta=CFG["beta"]),
        SlowConsolidatedImportanceRule(
            lr=CFG["lr"],
            lam=CFG["lam_slow"],
            pull=CFG["slow_pull"],
            consolidation_rate=CFG["slow_consolidation_rate"],
            stability_decay=CFG["slow_stability_decay"],
            stability_cap=CFG["slow_stability_cap"],
            importance_scale=CFG["slow_importance_scale"],
            xi=CFG["slow_xi"],
        ),
    ]


def run_all():
    """
    results[rule][seed]["stage_mse"] = matrix of shape [n_tasks, n_tasks]
    where stage_mse[k, j] = MSE on task j evaluated AFTER training on task k.
    results[rule][seed]["learning_curve"] stores test MSE during training,
    sampled every CFG["curve_eval_every"] updates.
    """
    out = {}
    for seed in range(CFG["n_seeds"]):
        tasks = make_modulated_context_task_sequence(
            input_dim=CFG["n_sensory"], n_tasks=CFG["n_tasks"],
            context_dim=CFG["context_dim"],
            n_train=CFG["n_train"], n_test=CFG["n_test"],
            noise=CFG["noise"], seed=seed,
        )
        for rule in make_rules():
            net = make_substrate(seed)
            log = run_sequence(rule, tasks, net,
                                n_epochs=CFG["n_epochs"],
                                batch_size=CFG["batch_size"], seed=seed,
                                trace_mse=True,
                                eval_every=CFG["curve_eval_every"])
            mat = np.array([t["mse_test_each_task"] for t in log["tasks"]])
            out.setdefault(rule.name, {})[seed] = {
                "stage_mse": mat.tolist(),
                "learning_curve": log["learning_curve"],
            }
        print(f"  seed={seed} done")
    return out


def _seed_keys(results, rule):
    return sorted(results[rule].keys(), key=lambda s: int(s))


def _stage_mse(entry):
    # Backward-compatible with results/multi_task.json generated before
    # learning curves were added, where each seed stored the matrix directly.
    if isinstance(entry, dict) and "stage_mse" in entry:
        return entry["stage_mse"]
    return entry


def _stack(results, rule):
    seeds = _seed_keys(results, rule)
    return np.array([_stage_mse(results[rule][s]) for s in seeds])  # [n_seeds, n_tasks, n_tasks]


def _curve_stack(results, rule):
    seeds = _seed_keys(results, rule)
    curves = []
    for s in seeds:
        entry = results[rule][s]
        if not isinstance(entry, dict) or "learning_curve" not in entry:
            return None
        curves.append(entry["learning_curve"])

    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    global_iterations = np.array(
        [p["global_iteration"] for p in curves[0]], dtype=int,
    )
    task_iterations = np.array(
        [p["task_iteration"] for p in curves[0]], dtype=int,
    )
    train_task_idxs = np.array(
        [p["train_task_idx"] for p in curves[0]], dtype=int,
    )
    mse_values = np.array(
        [[p["mse_test_each_task"] for p in curve] for curve in curves],
        dtype=float,
    )
    return global_iterations, task_iterations, train_task_idxs, mse_values


def plot_multi_task(results, save_path):
    n_tasks = CFG["n_tasks"]
    names = [chr(ord("A") + i) for i in range(n_tasks)]
    stages = np.arange(n_tasks)
    rules_present = [r for r in RULE_ORDER if r in results]
    n_rules = len(rules_present)

    fig = plt.figure(figsize=(max(16.5, 4.0 * n_rules), 9.5))
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
        ax.axhline(NO_INFO_MSE, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
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
    ax_bar.axhline(NO_INFO_MSE, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
                   label="zero predictor")
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
        mu = np.full(n_tasks, np.nan)
        se = np.full(n_tasks, np.nan)
        mu[1:] = np.nanmean(past[:, 1:], axis=0)
        se[1:] = np.nanstd(past[:, 1:], axis=0) / np.sqrt(past.shape[0])
        ax_past.errorbar(stages, mu, yerr=se, marker="o", capsize=2,
                         color=RULE_COLOR[rule], label=rule)
    ax_past.axhline(NO_INFO_MSE, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
                    label="zero predictor")
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
        f"cum λ={CFG['lam_cum']:g}, slow λ={CFG['lam_slow']:g}, "
        f"τ={CFG['thresh_tau']})",
        fontsize=12, y=0.995,
    )
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_learning_curves(results, save_path):
    n_tasks = CFG["n_tasks"]
    names = [chr(ord("A") + i) for i in range(n_tasks)]
    rules_present = [r for r in RULE_ORDER if r in results]
    n_rules = len(rules_present)

    fig, axes = plt.subplots(n_rules, 1, figsize=(13.5, 2.8 * n_rules),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    cmap = plt.get_cmap("viridis")

    plotted_any = False
    for ax, rule in zip(axes, rules_present):
        curve = _curve_stack(results, rule)
        if curve is None:
            ax.text(0.5, 0.5, "learning curve not recorded",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(rule, fontsize=11)
            continue

        plotted_any = True
        global_iterations, task_iterations, train_task_idxs, mse_values = curve
        mu = mse_values.mean(axis=0)  # [n_points, n_tasks]
        se = mse_values.std(axis=0) / np.sqrt(mse_values.shape[0])

        for j in range(n_tasks):
            color = cmap(j / max(n_tasks - 1, 1))
            ax.plot(global_iterations, mu[:, j], color=color, linewidth=1.6,
                    label=f"task {names[j]}")
            ax.fill_between(global_iterations, mu[:, j] - se[:, j],
                            mu[:, j] + se[:, j], color=color, alpha=0.12,
                            linewidth=0)

        task_starts = []
        for k in range(n_tasks):
            mask = (train_task_idxs == k) & (task_iterations == 0)
            if np.any(mask):
                task_starts.append(int(global_iterations[np.flatnonzero(mask)[0]]))
        task_starts = sorted(set(task_starts))
        for x in task_starts[1:]:
            ax.axvline(x, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.axhline(NO_INFO_MSE, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

        y0, y1 = ax.get_ylim()
        for k, start in enumerate(task_starts):
            end = task_starts[k + 1] if k + 1 < len(task_starts) else int(global_iterations.max())
            ax.text((start + end) / 2.0, y1, f"train {names[k]}",
                    ha="center", va="top", fontsize=8, color="black", alpha=0.65)

        ax.set_title(rule, fontsize=11)
        ax.set_ylabel("test MSE")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("minibatch update iteration")
    if plotted_any:
        axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       title="evaluated on")
    fig.suptitle(
        "Learning curves across the five-task sequence "
        f"(mean ± SE over {CFG['n_seeds']} seeds; "
        f"sampled every {CFG['curve_eval_every']} updates)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
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
    steps_per_task = int(np.ceil(CFG["n_train"] / CFG["batch_size"]) * CFG["n_epochs"])
    print(f"Five-task sequential experiment on coupled mesh substrate")
    print(f"  n_tasks={CFG['n_tasks']}  n_seeds={CFG['n_seeds']}  "
          f"lr={CFG['lr']}  thresh_tau={CFG['thresh_tau']}")
    print(f"  lam_cum={CFG['lam_cum']}  lam_slow={CFG['lam_slow']}")
    print(f"  budget={steps_per_task} minibatch updates/task  "
          f"curve_eval_every={CFG['curve_eval_every']} updates")
    results = run_all()

    with open(out_dir / "multi_task.json", "w") as f:
        json.dump({"config": CFG, "results": results}, f, indent=2)
    print(f"Saved {out_dir / 'multi_task.json'}")

    plot_multi_task(results, out_dir / "multi_task.png")
    print(f"Saved {out_dir / 'multi_task.png'}")

    plot_learning_curves(results, out_dir / "multi_task_learning_curves.png")
    print(f"Saved {out_dir / 'multi_task_learning_curves.png'}")

    _summarize(results)


if __name__ == "__main__":
    main()

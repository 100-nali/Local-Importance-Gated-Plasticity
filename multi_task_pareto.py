"""
Five-task lambda sweep on the mesh substrate. Pareto axes are:
    x = past-mean MSE  (mean over tasks 0..N-2 evaluated after final task)
    y = current MSE    (task N-1 evaluated after final task)
Reference lines drawn at MSE = 1.0 — the zero-output baseline — so the
"genuine retention" region is the bottom-left quadrant (past_mean < 1.0 and
current < 1.0). vanilla and thresh are reference points.

The question: is there a lambda where a cumulative local-importance rule
lands in the retention quadrant while strictly beating vanilla on past_mean
AND beating thresh on current?

Run:
    python multi_task_pareto.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from network import ContextModulatedMeshSubstrate
from tasks import make_modulated_context_task_sequence
from learning_rules import (SGDRule, ThresholdedSGDRule,
                            CumulativeImportanceGatedRule,
                            MultiAnchorImportanceGatedRule,
                            HeatCumImportanceGatedRule,
                            SlowConsolidatedImportanceRule)
from experiments import run_sequence


CFG = {
    "n_sensory": 8,
    "context_dim": 5,
    "n_tasks": 5,
    "rows": 8, "cols": 10,
    "out_pos_row": 3, "out_neg_row": 4,
    "context_gain_std": 0.35,
    "max_log_gain": 1.0,
    "n_train": 500, "n_test": 200, "noise": 0.05,
    "n_epochs": 80, "batch_size": 32,
    "n_seeds": 12,
    "lr": 5.0, "eta": 0.005,
    "beta": 0.95,
    # Cumulative I* grows ~N x larger over N tasks, so the protection-equivalent
    # lambda is ~N x smaller; widen the low end to catch the right operating point.
    "lambdas_cum": [5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    # Multi-anchor has the same S accumulation as cum_imp_gated (so denominator
    # scales the same), so we reuse the same lambda grid.
    "lambdas_multi": [5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    # SI-normalized heat importance can be much larger than raw heat on edges
    # with small net task displacement, so sweep a wider low-lambda range.
    "lambdas_heat": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    "heat_xi": 1e-3,
    "lambdas_slow": [0.1, 0.3, 1, 3, 10],
    "slow_pull": 0.02,
    "slow_consolidation_rate": 0.25,
    "slow_stability_decay": 0.95,
    "slow_stability_cap": 3.0,
    "slow_importance_scale": 1.0,
    "slow_xi": 1e-3,
    "thresh_tau": 0.005,
}


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


def run_rule(rule_factory):
    """Return arrays of (past_mean_after_last, current_after_last) over seeds."""
    past, curr = [], []
    for seed in range(CFG["n_seeds"]):
        tasks = make_modulated_context_task_sequence(
            input_dim=CFG["n_sensory"], n_tasks=CFG["n_tasks"],
            context_dim=CFG["context_dim"],
            n_train=CFG["n_train"], n_test=CFG["n_test"],
            noise=CFG["noise"], seed=seed,
        )
        net = make_substrate(seed)
        rule = rule_factory()
        log = run_sequence(rule, tasks, net,
                            n_epochs=CFG["n_epochs"],
                            batch_size=CFG["batch_size"], seed=seed)
        final_row = log["tasks"][-1]["mse_test_each_task"]  # length n_tasks
        past.append(float(np.mean(final_row[:-1])))
        curr.append(float(final_row[-1]))
    return np.array(past), np.array(curr)


def mu_se(a):
    a = np.array(a)
    return float(a.mean()), float(a.std() / np.sqrt(len(a)))


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Multi-task lambda sweep: {CFG['n_tasks']} orthogonal tasks, "
          f"{CFG['n_seeds']} seeds, lr={CFG['lr']}, beta={CFG['beta']}")
    print(f"  lambdas_cum = {CFG['lambdas_cum']}")
    print(f"  lambdas_multi = {CFG['lambdas_multi']}")
    print(f"  lambdas_heat = {CFG['lambdas_heat']}  xi={CFG['heat_xi']}")
    print(f"  lambdas_slow = {CFG['lambdas_slow']}")
    print()

    records = {}

    print("vanilla...")
    p, c = run_rule(lambda: SGDRule(lr=CFG["lr"]))
    records["vanilla"] = {"past": p.tolist(), "current": c.tolist()}
    pM, pS = mu_se(p); cM, cS = mu_se(c)
    print(f"  vanilla   past_mean = {pM:.3f} +/- {pS:.3f}   current = {cM:.3f} +/- {cS:.3f}")

    print("thresh...")
    p, c = run_rule(lambda: ThresholdedSGDRule(lr=CFG["lr"], threshold=CFG["thresh_tau"]))
    records["thresh"] = {"past": p.tolist(), "current": c.tolist()}
    pM, pS = mu_se(p); cM, cS = mu_se(c)
    print(f"  thresh    past_mean = {pM:.3f} +/- {pS:.3f}   current = {cM:.3f} +/- {cS:.3f}")

    records["cum_imp_gated"] = {}
    print("cum_imp_gated...")
    for lam in CFG["lambdas_cum"]:
        p, c = run_rule(lambda lam=lam: CumulativeImportanceGatedRule(
            lr=CFG["lr"], lam=lam, beta=CFG["beta"]))
        records["cum_imp_gated"][f"lam={lam}"] = {"past": p.tolist(), "current": c.tolist()}
        pM, pS = mu_se(p); cM, cS = mu_se(c)
        print(f"  lam={lam:>5}  past_mean = {pM:.3f} +/- {pS:.3f}   "
              f"current = {cM:.3f} +/- {cS:.3f}")

    records["multi_anchor"] = {}
    print("multi_anchor...")
    for lam in CFG["lambdas_multi"]:
        p, c = run_rule(lambda lam=lam: MultiAnchorImportanceGatedRule(
            lr=CFG["lr"], lam=lam, beta=CFG["beta"]))
        records["multi_anchor"][f"lam={lam}"] = {"past": p.tolist(), "current": c.tolist()}
        pM, pS = mu_se(p); cM, cS = mu_se(c)
        print(f"  lam={lam:>5}  past_mean = {pM:.3f} +/- {pS:.3f}   "
              f"current = {cM:.3f} +/- {cS:.3f}")

    records["heat_cum_imp_gated"] = {}
    print("heat_cum_imp_gated...")
    for lam in CFG["lambdas_heat"]:
        p, c = run_rule(lambda lam=lam: HeatCumImportanceGatedRule(
            lr=CFG["lr"], lam=lam, xi=CFG["heat_xi"]))
        records["heat_cum_imp_gated"][f"lam={lam}"] = {"past": p.tolist(), "current": c.tolist()}
        pM, pS = mu_se(p); cM, cS = mu_se(c)
        print(f"  lam={lam:>7.3f}  past_mean = {pM:.3f} +/- {pS:.3f}   "
              f"current = {cM:.3f} +/- {cS:.3f}")

    records["slow_consolidated"] = {}
    print("slow_consolidated...")
    for lam in CFG["lambdas_slow"]:
        p, c = run_rule(lambda lam=lam: SlowConsolidatedImportanceRule(
            lr=CFG["lr"],
            lam=lam,
            pull=CFG["slow_pull"],
            consolidation_rate=CFG["slow_consolidation_rate"],
            stability_decay=CFG["slow_stability_decay"],
            stability_cap=CFG["slow_stability_cap"],
            importance_scale=CFG["slow_importance_scale"],
            xi=CFG["slow_xi"],
        ))
        records["slow_consolidated"][f"lam={lam}"] = {"past": p.tolist(), "current": c.tolist()}
        pM, pS = mu_se(p); cM, cS = mu_se(c)
        print(f"  lam={lam:>5.1f}  past_mean = {pM:.3f} +/- {pS:.3f}   "
              f"current = {cM:.3f} +/- {cS:.3f}")

    with open(out_dir / "multi_task_pareto.json", "w") as f:
        json.dump({"config": CFG, "records": records}, f, indent=2)
    print(f"\nSaved {out_dir / 'multi_task_pareto.json'}")

    plot_pareto(records, out_dir / "multi_task_pareto.png")


def _curve(records, key):
    keys = list(records[key].keys())
    lams_str = sorted(keys, key=lambda s: float(s.split("=")[1]))
    p, pse, c, cse = [], [], [], []
    for k in lams_str:
        pM, pS = mu_se(records[key][k]["past"])
        cM, cS = mu_se(records[key][k]["current"])
        p.append(pM); pse.append(pS); c.append(cM); cse.append(cS)
    # Return numeric lambdas (int if integer-valued, else float) for labelling
    lams_num = [float(s.split("=")[1]) for s in lams_str]
    lams_num = [int(v) if v.is_integer() else v for v in lams_num]
    return (lams_num, np.array(p), np.array(pse), np.array(c), np.array(cse))


def plot_pareto(records, save_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    van_p, van_p_se = mu_se(records["vanilla"]["past"])
    van_c, van_c_se = mu_se(records["vanilla"]["current"])
    thr_p, thr_p_se = mu_se(records["thresh"]["past"])
    thr_c, thr_c_se = mu_se(records["thresh"]["current"])

    has_cum = "cum_imp_gated" in records
    if has_cum:
        cum_lams, cum_p, cum_p_se, cum_c, cum_c_se = _curve(records, "cum_imp_gated")
    else:
        cum_lams = []; cum_p = cum_c = np.array([])
    has_multi = "multi_anchor" in records
    if has_multi:
        ma_lams, ma_p, ma_p_se, ma_c, ma_c_se = _curve(records, "multi_anchor")
    else:
        ma_lams = []; ma_p = ma_c = np.array([])
    has_heat = "heat_cum_imp_gated" in records
    if has_heat:
        ht_lams, ht_p, ht_p_se, ht_c, ht_c_se = _curve(records, "heat_cum_imp_gated")
    else:
        ht_lams = []; ht_p = ht_c = np.array([])
    has_slow = "slow_consolidated" in records
    if has_slow:
        sl_lams, sl_p, sl_p_se, sl_c, sl_c_se = _curve(records, "slow_consolidated")
    else:
        sl_lams = []; sl_p = sl_c = np.array([])

    # Axis limits
    x_all = np.concatenate([cum_p, ma_p, ht_p, sl_p, [van_p, thr_p]])
    y_all = np.concatenate([cum_c, ma_c, ht_c, sl_c, [van_c, thr_c]])
    xpad = 0.1 * (x_all.max() - x_all.min() + 0.1)
    ypad = 0.1 * (y_all.max() - y_all.min() + 0.1)
    xmin, xmax = x_all.min() - xpad, x_all.max() + xpad
    ymin, ymax = y_all.min() - ypad, y_all.max() + ypad
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    # Retention region: rectangle (x < 1, y < 1)
    rect = plt.Rectangle((xmin, ymin), max(0.0, 1.0 - xmin), max(0.0, 1.0 - ymin),
                         color="green", alpha=0.07, zorder=0)
    ax.add_patch(rect)

    # No-info floor lines
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
    ax.text(1.0, ymax, "  no-info floor (zero output)",
            fontsize=8, color="gray", va="top", ha="left")
    if 1.0 > xmin:
        ax.text(xmin + 0.01 * (xmax - xmin), 1.0 - 0.01 * (ymax - ymin),
                "true retention region\n(both axes below no-info floor)",
                fontsize=8, color="green", alpha=0.8, va="top")

    # cum_imp_gated curve
    if has_cum:
        order = np.argsort(cum_p)
        ax.plot(cum_p[order], cum_c[order], "-", color="C3", linewidth=1.6, alpha=0.85, zorder=3)
        ax.errorbar(cum_p, cum_c, xerr=cum_p_se, yerr=cum_c_se, fmt="^", color="C3",
                    markersize=8, capsize=2, linewidth=0.6,
                    markeredgecolor="black", markeredgewidth=0.4,
                    zorder=4, label="cum_imp_gated (online EWC I*)")
        for i, lam in enumerate(cum_lams):
            ax.annotate(f"λ={lam}", (cum_p[i], cum_c[i]), xytext=(-7, -10),
                        textcoords="offset points", fontsize=7, color="C3",
                        ha="right", zorder=5)

    # multi_anchor curve
    if has_multi:
        order = np.argsort(ma_p)
        ax.plot(ma_p[order], ma_c[order], "-", color="C4", linewidth=1.6, alpha=0.85, zorder=3)
        ax.errorbar(ma_p, ma_c, xerr=ma_p_se, yerr=ma_c_se, fmt="v", color="C4",
                    markersize=8, capsize=2, linewidth=0.6,
                    markeredgecolor="black", markeredgewidth=0.4,
                    zorder=4, label="multi_anchor (per-task anchors via S, R)")
        for i, lam in enumerate(ma_lams):
            ax.annotate(f"λ={lam}", (ma_p[i], ma_c[i]), xytext=(7, -10),
                        textcoords="offset points", fontsize=7, color="C4",
                        ha="left", zorder=5)

    # heat_cum_imp_gated curve
    if has_heat:
        order = np.argsort(ht_p)
        ax.plot(ht_p[order], ht_c[order], "-", color="C5", linewidth=1.6, alpha=0.85, zorder=3)
        ax.errorbar(ht_p, ht_c, xerr=ht_p_se, yerr=ht_c_se, fmt="P", color="C5",
                    markersize=8, capsize=2, linewidth=0.6,
                    markeredgecolor="black", markeredgewidth=0.4,
                    zorder=4, label="heat_cum_imp_gated (SI importance)")
        for i, lam in enumerate(ht_lams):
            lam_str = f"{lam:g}"
            ax.annotate(f"λ={lam_str}", (ht_p[i], ht_c[i]), xytext=(-7, 8),
                        textcoords="offset points", fontsize=7, color="C5",
                        ha="right", zorder=5)

    # slow_consolidated curve
    if has_slow:
        order = np.argsort(sl_p)
        ax.plot(sl_p[order], sl_c[order], "-", color="C6", linewidth=1.6, alpha=0.85, zorder=3)
        ax.errorbar(sl_p, sl_c, xerr=sl_p_se, yerr=sl_c_se, fmt="X", color="C6",
                    markersize=8, capsize=2, linewidth=0.6,
                    markeredgecolor="black", markeredgewidth=0.4,
                    zorder=4, label="slow_consolidated (metaplastic z, S)")
        for i, lam in enumerate(sl_lams):
            lam_str = f"{lam:g}"
            ax.annotate(f"λ={lam_str}", (sl_p[i], sl_c[i]), xytext=(8, 8),
                        textcoords="offset points", fontsize=7, color="C6",
                        ha="left", zorder=5)

    # Reference points
    ax.errorbar(van_p, van_c, xerr=van_p_se, yerr=van_c_se, fmt="s", color="C0",
                markersize=12, capsize=3, linewidth=1.2, zorder=5, label="vanilla")
    ax.errorbar(thr_p, thr_c, xerr=thr_p_se, yerr=thr_c_se, fmt="D", color="C1",
                markersize=11, capsize=3, linewidth=1.2, zorder=5, label="thresh")

    ax.set_xlabel("Past-mean MSE  (mean over tasks A–D after final task; lower = better retention)")
    ax.set_ylabel("Current-task MSE  (task E after final training; lower = better new-task fit)")
    ax.set_title(f"Five-task stability–plasticity Pareto on the mesh substrate\n"
                  f"(orthogonal tasks, {CFG['n_seeds']} seeds, lr={CFG['lr']}, β={CFG['beta']})",
                  fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()

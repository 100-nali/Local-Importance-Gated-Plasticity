"""
Focused experiment: (A_after_B, B_after_B) Pareto frontier at overlap=0 on
the mesh substrate. Sweep lambda for both imp_gated variants and plot
against vanilla and thresh single points. The question this experiment is
designed to answer: can any local rule reach B_after_B ≈ vanilla's B while
strictly improving A_after_B?

Run:
    python pareto_experiment.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from network import MeshCoupledSubstrate
from tasks import make_two_task_sequence
from learning_rules import (SGDRule, ThresholdedSGDRule, ImportanceGatedRule)
from experiments import run_sequence


CFG = {
    "rows": 8, "cols": 10, "n_input": 8, "out_pos_row": 3, "out_neg_row": 4,
    "n_train": 500, "n_test": 200, "noise": 0.05,
    "n_epochs": 80, "batch_size": 32,
    "n_seeds": 12,
    "lr": 5.0,
    "eta": 0.005,
    "lambdas": [20, 30, 50, 100, 200, 500, 1000],
    "thresh_tau": 0.005,
}


def make_substrate(seed):
    return MeshCoupledSubstrate(
        rows=CFG["rows"], cols=CFG["cols"],
        n_input=CFG["n_input"],
        out_pos_row=CFG["out_pos_row"], out_neg_row=CFG["out_neg_row"],
        eta=CFG["eta"], seed=seed,
    )


def run_rule(rule_factory):
    """Return arrays of (A_after_B, B_after_B) across seeds at overlap=0."""
    A_vals, B_vals = [], []
    for seed in range(CFG["n_seeds"]):
        tasks = make_two_task_sequence(
            input_dim=CFG["n_input"], overlap=0.0,
            n_train=CFG["n_train"], n_test=CFG["n_test"],
            noise=CFG["noise"], seed=seed,
        )
        net = make_substrate(seed)
        rule = rule_factory()
        log = run_sequence(rule, tasks, net,
                            n_epochs=CFG["n_epochs"],
                            batch_size=CFG["batch_size"], seed=seed)
        A_vals.append(log["tasks"][1]["mse_test_each_task"][0])
        B_vals.append(log["tasks"][1]["mse_test_each_task"][1])
    return np.array(A_vals), np.array(B_vals)


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Pareto experiment at overlap=0, {CFG['n_seeds']} seeds")
    print(f"lr={CFG['lr']}, eta={CFG['eta']}")
    print()

    records = {}

    print("vanilla...")
    A, B = run_rule(lambda: SGDRule(lr=CFG["lr"]))
    records["vanilla"] = {"A": A.tolist(), "B": B.tolist()}
    van_A_mean, van_A_se = float(A.mean()), float(A.std() / np.sqrt(len(A)))
    van_B_mean, van_B_se = float(B.mean()), float(B.std() / np.sqrt(len(B)))
    print(f"  vanilla   A = {van_A_mean:.3f} +/- {van_A_se:.3f}   B = {van_B_mean:.3f} +/- {van_B_se:.3f}")

    print("thresh...")
    A, B = run_rule(lambda: ThresholdedSGDRule(lr=CFG["lr"], threshold=CFG["thresh_tau"]))
    records["thresh"] = {"A": A.tolist(), "B": B.tolist()}
    print(f"  thresh    A = {A.mean():.3f} +/- {A.std()/np.sqrt(len(A)):.3f}   B = {B.mean():.3f} +/- {B.std()/np.sqrt(len(B)):.3f}")

    records["imp_gated"] = {}
    print("imp_gated...")
    for lam in CFG["lambdas"]:
        A, B = run_rule(lambda lam=lam: ImportanceGatedRule(lr=CFG["lr"], lam=lam, beta=0.95))
        records["imp_gated"][f"lam={lam}"] = {"A": A.tolist(), "B": B.tolist()}
        print(f"  lam={lam:>5}  A = {A.mean():.3f} +/- {A.std()/np.sqrt(len(A)):.3f}   "
              f"B = {B.mean():.3f} +/- {B.std()/np.sqrt(len(B)):.3f}")

    # Save records
    with open(out_dir / "pareto.json", "w") as f:
        json.dump({"config": CFG, "records": records}, f, indent=2)
    print(f"Saved pareto.json")

    # Pareto plot — single imp_gated curve, vanilla/thresh as reference points,
    # vanilla B drawn as a dashed reference line so the "match vanilla B" claim is visible.
    fig, ax = plt.subplots(figsize=(8, 5.5))

    def mu_se(arr):
        a = np.array(arr)
        return float(a.mean()), float(a.std() / np.sqrt(len(a)))

    van_A, van_A_se = mu_se(records["vanilla"]["A"])
    van_B, van_B_se = mu_se(records["vanilla"]["B"])
    thr_A, thr_A_se = mu_se(records["thresh"]["A"])
    thr_B, thr_B_se = mu_se(records["thresh"]["B"])

    # Vanilla B reference band
    ax.axhline(van_B, color="C0", linestyle="--", alpha=0.5, linewidth=1.2, zorder=1)
    ax.axhspan(van_B - van_B_se, van_B + van_B_se, color="C0", alpha=0.08, zorder=0)

    # imp_gated Pareto curve
    lams = sorted(int(k.split("=")[1]) for k in records["imp_gated"])
    ig_A, ig_A_se, ig_B, ig_B_se = [], [], [], []
    for lam in lams:
        a_mu, a_se = mu_se(records["imp_gated"][f"lam={lam}"]["A"])
        b_mu, b_se = mu_se(records["imp_gated"][f"lam={lam}"]["B"])
        ig_A.append(a_mu); ig_A_se.append(a_se); ig_B.append(b_mu); ig_B_se.append(b_se)
    ig_A = np.array(ig_A); ig_B = np.array(ig_B)
    ig_A_se = np.array(ig_A_se); ig_B_se = np.array(ig_B_se)
    order = np.argsort(ig_A)
    ax.plot(ig_A[order], ig_B[order], "-", color="C2", linewidth=1.8, alpha=0.85, zorder=3)
    ax.fill_between(ig_A[order], (ig_B - ig_B_se)[order], (ig_B + ig_B_se)[order],
                    color="C2", alpha=0.10, zorder=2)
    ax.scatter(ig_A, ig_B, s=60, color="C2", edgecolor="black", linewidth=0.6, zorder=4,
               label="imp_gated (snapshot)")

    # Vanilla and thresh reference points with full error bars
    ax.errorbar(van_A, van_B, xerr=van_A_se, yerr=van_B_se, fmt="o", color="C0",
                markersize=12, capsize=3, linewidth=1.2, zorder=5, label="vanilla")
    ax.errorbar(thr_A, thr_B, xerr=thr_A_se, yerr=thr_B_se, fmt="s", color="C1",
                markersize=11, capsize=3, linewidth=1.2, zorder=5, label="thresh")

    # λ annotations: endpoints + operating point
    def annot(lam, dx, dy, ha="left"):
        i = lams.index(lam)
        ax.annotate(f"λ={lam}", (ig_A[i], ig_B[i]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9, color="C2", ha=ha,
                    zorder=6, arrowprops=dict(arrowstyle="-", color="C2",
                                              alpha=0.4, lw=0.6))
    annot(lams[0], 8, -2)
    if 100 in lams: annot(100, 8, -10)
    annot(lams[-1], -8, 4, ha="right")

    # Highlight operating point
    if 100 in lams:
        i100 = lams.index(100)
        ax.scatter([ig_A[i100]], [ig_B[i100]], s=200, facecolors="none",
                   edgecolors="C2", linewidth=2.0, zorder=7)

    ax.set_xlabel("A_after_B  (lower = better past-task retention)")
    ax.set_ylabel("B_after_B  (lower = better new-task fit)")
    ax.set_title(f"Stability–plasticity Pareto on the 2D mesh\n"
                  f"(overlap=0, {CFG['n_seeds']} seeds, lr={CFG['lr']})",
                  fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    if 100 in lams:
        i100 = lams.index(100)
        ax.text(0.02, 0.97,
                f"vanilla   A={van_A:.3f}, B={van_B:.3f}\n"
                f"imp_gated λ=100   A={ig_A[i100]:.3f}, B={ig_B[i100]:.3f}\n"
                f"Pareto improvement over vanilla",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=140, bbox_inches="tight")
    print(f"Saved pareto.png")


if __name__ == "__main__":
    main()

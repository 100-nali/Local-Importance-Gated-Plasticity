"""
Re-plot the per-edge heat maps from thermodynamic_analysis.json with
per-rule normalization, and report the top-10% heat-concentration metric
for each rule.

The original 4-panel plot used a shared color scale across all rules so
the absolute heat magnitudes were comparable. That choice washes out
within-rule concentration patterns because the highest-vmax rule
(slow_consolidated) dominates the colormap. Progress.tex specifically
asks for "where heat is concentrated under each rule", which is a
per-rule question -- this script answers it.

Concentration metric: metrics.concentration(values, frac=0.1) is the
fraction of total |heat| carried by the top 10% of edges. 0.10 means
heat is uniform; 1.0 means it is fully localized in 10% of edges.

Run (after thermodynamic_analysis.py has been run):
    python thermodynamic_replot.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics import concentration
from plots import _draw_mesh


RULE_LABEL = {
    "vanilla": "vanilla",
    "thresh": "threshold",
    "cum_imp_gated": "cum. importance (λ=50)",
    "slow_consolidated": "slow consolidation",
}


def mean_edge_heat(records):
    """Mean per-edge heat (eh_h, eh_v) across seeds for one rule."""
    eh_h = np.mean(
        np.stack([np.array(r["edge_heat_h"]) for r in records]),
        axis=0,
    )
    eh_v = np.mean(
        np.stack([np.array(r["edge_heat_v"]) for r in records]),
        axis=0,
    )
    return eh_h, eh_v


def per_seed_concentration(records, frac=0.1):
    """Top-`frac` concentration per seed, summarized as mean ± SE."""
    vals = np.array([
        concentration(
            [np.array(r["edge_heat_h"]), np.array(r["edge_heat_v"])],
            frac=frac,
        )
        for r in records
    ])
    return float(vals.mean()), float(vals.std() / np.sqrt(len(vals)))


def per_column_heat(eh_h, eh_v):
    """Mean horizontal-edge heat per column-index (col c = edges c↔c+1).
    Returned alongside vertical-edge heat per node-column.
    """
    col_h = eh_h.mean(axis=0)
    col_v = eh_v.mean(axis=0)
    return col_h, col_v


def main():
    out_dir = Path(__file__).parent / "results"
    with open(out_dir / "thermodynamic_analysis.json") as f:
        data = json.load(f)

    cfg = data["config"]
    rule_names = list(data["records"].keys())
    n_rules = len(rule_names)

    # --- Per-rule numerics: total heat, top-10% concentration, column profile
    print(f"{'rule':<22}{'top-10% conc.':>18}{'col 0→1 mean':>16}{'col 4→5 mean':>16}{'ratio':>10}")
    summary_rows = {}
    for name in rule_names:
        recs = data["records"][name]
        eh_h, eh_v = mean_edge_heat(recs)
        col_h, col_v = per_column_heat(eh_h, eh_v)
        conc_mu, conc_se = per_seed_concentration(recs, frac=0.1)
        col0_mean = col_h[0]
        col4_mean = col_h[4]
        ratio = col0_mean / max(col4_mean, 1e-12)
        summary_rows[name] = {
            "eh_h": eh_h, "eh_v": eh_v,
            "col_h": col_h, "col_v": col_v,
            "conc_mu": conc_mu, "conc_se": conc_se,
            "col0_mean": col0_mean, "col4_mean": col4_mean,
            "ratio_col0_col4": ratio,
        }
        print(
            f"{name:<22}"
            f"{conc_mu:>13.3f}±{conc_se:<3.3f}"
            f"{col0_mean:>16.3f}"
            f"{col4_mean:>16.3f}"
            f"{ratio:>10.1f}"
        )
    print()
    print("(top-10% conc.: 0.10 = uniform across edges; 1.0 = all heat in 10% of edges)")
    print()

    # --- Plot: per-rule normalized per-edge maps, with concentration annotation
    fig = plt.figure(figsize=(4.0 * n_rules, 5.5))
    gs = fig.add_gridspec(
        2, n_rules,
        height_ratios=[1.0, 0.45],
        hspace=0.35, wspace=0.25,
    )

    for col, name in enumerate(rule_names):
        ax = fig.add_subplot(gs[0, col])
        row = summary_rows[name]
        eh_h, eh_v = row["eh_h"], row["eh_v"]
        per_rule_vmax = float(max(np.abs(eh_h).max(), np.abs(eh_v).max()))
        _draw_mesh(
            ax,
            rows=cfg["rows"], cols=cfg["cols"],
            n_input=cfg["n_sensory"],
            out_pos_row=cfg["out_pos_row"],
            out_neg_row=cfg["out_neg_row"],
            W_h=eh_h, W_v=eh_v,
            cmap_name="inferno",
            vmax=per_rule_vmax,
            max_lw=5.5,
        )
        ax.set_title(
            f"{RULE_LABEL.get(name, name)}\n"
            f"vmax={per_rule_vmax:.2f}   "
            f"top-10% conc.={row['conc_mu']:.2f}±{row['conc_se']:.2f}",
            fontsize=9,
        )

    # --- Per-column heat profile (horizontal edges by column index)
    ax_prof = fig.add_subplot(gs[1, :])
    cols_h = np.arange(cfg["cols"] - 1)
    for name in rule_names:
        col_h = summary_rows[name]["col_h"]
        ax_prof.plot(
            cols_h, col_h, marker="o", linewidth=1.4,
            label=RULE_LABEL.get(name, name),
        )
    ax_prof.set_yscale("log")
    ax_prof.set_xlabel("horizontal edge: column c → column c+1   (c=0 is input-adjacent)")
    ax_prof.set_ylabel("mean per-edge heat (log scale)")
    ax_prof.set_title(
        "Per-column profile -- heat decays ~exponentially from the input boundary "
        "for every rule"
    )
    ax_prof.grid(True, which="both", alpha=0.25)
    ax_prof.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Per-edge heat maps: per-rule normalization "
        f"({cfg['n_seeds']} seeds, eight-task sequence)",
        fontsize=12,
    )
    save_path = out_dir / "thermodynamic_per_edge_maps.png"
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()

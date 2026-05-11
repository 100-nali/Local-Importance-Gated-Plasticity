"""
Three figures, coupled substrate only:

  overlap_sweep.png : six panels along the overlap axis — old/new MSE,
    parameter change, locality of change, total heat dissipated, locality
    of heat. One curve per rule.

  heat_pareto.png : retention vs total heat dissipated, per (rule, overlap)
    cell.

  substrate_activity.png : the layered graph drawn explicitly, with edges
    colored/sized by per-task |Δw| (averaged across seeds). One row per rule;
    columns show task A activity, task B activity, and the per-edge overlap
    |Δw_A| · |Δw_B|. The cosine similarity of the two |Δw| vectors is shown
    in each row's title — this is the topology-agnostic "edge-set overlap"
    metric. Note the topology is fully-connected layered, so high cosine
    similarity is expected; this plot makes the actual structure visible.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PANEL_METRICS = [
    ("Old-task (A) MSE after B", lambda log: log["tasks"][1]["mse_test_each_task"][0]),
    ("New-task (B) MSE after B", lambda log: log["tasks"][1]["mse_test_each_task"][1]),
    ("||Δw|| during B",          lambda log: log["tasks"][1]["param_change"]),
    ("Locality of Δw (top 10%)", lambda log: log["tasks"][1]["locality"]),
    ("Total heat during B",      lambda log: log["tasks"][1]["heat_total"]),
    ("Locality of heat (top 10%)", lambda log: log["tasks"][1]["heat_locality"]),
]


def _agg(rows):
    arr = np.array(rows, dtype=float)
    return arr.mean(axis=1), arr.std(axis=1) / np.sqrt(arr.shape[1])


def _agg_robust(rows):
    """Median and (q25, q75) per row. Robust to bimodal / divergent seeds."""
    arr = np.array(rows, dtype=float)
    med = np.median(arr, axis=1)
    q25 = np.percentile(arr, 25, axis=1)
    q75 = np.percentile(arr, 75, axis=1)
    err_lo = med - q25
    err_hi = q75 - med
    return med, np.stack([err_lo, err_hi], axis=0)


def plot_overlap_sweep(results, save_path=None):
    rule_names = list(results.keys())
    overlaps = sorted(next(iter(results.values())).keys())

    n_cols = 3
    n_rows = (len(PANEL_METRICS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4.0 * n_rows), squeeze=False,
    )

    for panel_i, (title, extractor) in enumerate(PANEL_METRICS):
        ax = axes[panel_i // n_cols, panel_i % n_cols]
        is_heat = "heat" in title.lower() and "locality" not in title.lower()
        for r in rule_names:
            rows = []
            for ov in overlaps:
                logs = results[r][ov]
                rows.append([extractor(log) for log in logs])
            if is_heat:
                med, err = _agg_robust(rows)
                ax.errorbar(overlaps, med, yerr=err, marker="o", capsize=3, label=r)
            else:
                mu, se = _agg(rows)
                ax.errorbar(overlaps, mu, yerr=se, marker="o", capsize=3, label=r)
        ax.set_title(title + (" — median, IQR" if is_heat else ""), fontsize=10)
        ax.set_xlabel("task A · task B overlap")
        ax.grid(True, alpha=0.3)
        if is_heat:
            ax.set_yscale("symlog", linthresh=1.0)
        if panel_i == 0:
            ax.legend(loc="best", fontsize=9)

    # Hide unused axes if any
    for k in range(len(PANEL_METRICS), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.suptitle(
        "Coupled substrate — local rules under varying task overlap",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=130, bbox_inches="tight")
    return fig


def plot_heat_pareto(results, save_path=None):
    """
    Per-cell scatter: total heat during task B vs retained memory of A.
    Memory retention = 1 - A_after_B / pre-train A MSE, clipped to [0, 1].
    """
    rule_names = list(results.keys())
    overlaps = sorted(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    markers = {"sgd": "o", "thresh": "s", "imp_gated": "^"}

    for r in rule_names:
        for ov_i, ov in enumerate(overlaps):
            logs = results[r][ov]
            heat = np.array([log["tasks"][1]["heat_total"] for log in logs])
            a_after_b = np.array([log["tasks"][1]["mse_test_each_task"][0] for log in logs])
            a_pre = np.array([log["mse_pre"][0] for log in logs])
            retention = np.clip(1.0 - a_after_b / np.maximum(a_pre, 1e-9), 0.0, 1.0)
            heat_med = float(np.median(heat))
            heat_lo = float(np.percentile(heat, 25))
            heat_hi = float(np.percentile(heat, 75))
            ret_med = float(np.median(retention))
            ret_lo = float(np.percentile(retention, 25))
            ret_hi = float(np.percentile(retention, 75))
            ax.errorbar(
                max(heat_med, 0.1), ret_med,
                xerr=[[max(heat_med - heat_lo, 0)], [max(heat_hi - heat_med, 0)]],
                yerr=[[max(ret_med - ret_lo, 0)], [max(ret_hi - ret_med, 0)]],
                marker=markers.get(r, "o"),
                color=cmap(ov_i / max(len(overlaps) - 1, 1)),
                capsize=3,
                markersize=10,
            )

    ax.set_xscale("log")

    # Custom legend: rules by marker, overlap by color
    rule_handles = [
        plt.Line2D([0], [0], marker=markers.get(r, "o"), linestyle="",
                   color="black", markersize=10, label=r)
        for r in rule_names
    ]
    ov_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=cmap(i / max(len(overlaps) - 1, 1)),
                   markersize=10, label=f"overlap={ov}")
        for i, ov in enumerate(overlaps)
    ]
    leg1 = ax.legend(handles=rule_handles, loc="lower right",
                     title="rule", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=ov_handles, loc="lower left",
              title="task overlap", fontsize=9)

    ax.set_xlabel("Total heat dissipated during task B (log scale, median ± IQR)")
    ax.set_ylabel("Retained memory of A: 1 − MSE_A_after_B / MSE_A_pre")
    ax.set_title("Retention vs dissipation, per (rule, overlap) cell — coupled substrate")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=130, bbox_inches="tight")
    return fig


def _draw_substrate(ax, layer_sizes, edge_value_layers,
                    cmap_name="viridis", vmax=None,
                    edge_alpha=0.7, max_lw=3.0):
    """
    Draw a layered weighted graph: nodes in vertical columns per layer,
    edges between consecutive layers colored / thickened by |edge_value|.
    edge_value_layers: list of arrays, one per slab, shape matching weights.
    """
    n_layers = len(layer_sizes)
    max_size = max(layer_sizes)
    node_pos = {}
    for li, size in enumerate(layer_sizes):
        ys = np.linspace(0, max_size - 1, size) - (max_size - 1) / 2
        for j, y in enumerate(ys):
            node_pos[(li, j)] = (li, y)

    if vmax is None:
        all_vals = np.concatenate(
            [np.abs(np.asarray(v)).ravel() for v in edge_value_layers]
        )
        vmax = float(all_vals.max()) if all_vals.size > 0 else 1.0
    vmax = max(vmax, 1e-12)
    cmap = plt.get_cmap(cmap_name)

    # Edges first (so nodes draw on top).
    for li in range(n_layers - 1):
        E = np.abs(np.asarray(edge_value_layers[li]))
        # Sort edges by value so high-value edges draw on top.
        ij = sorted(
            ((i, j) for i in range(layer_sizes[li]) for j in range(layer_sizes[li + 1])),
            key=lambda p: E[p[0], p[1]],
        )
        for i, j in ij:
            v = float(E[i, j])
            frac = v / vmax
            color = cmap(min(max(frac, 0.0), 1.0))
            lw = 0.15 + max_lw * frac
            x0, y0 = node_pos[(li, i)]
            x1, y1 = node_pos[(li + 1, j)]
            ax.plot([x0, x1], [y0, y1], color=color,
                    linewidth=lw, alpha=edge_alpha, zorder=1)

    # Nodes.
    for li, size in enumerate(layer_sizes):
        xs = [node_pos[(li, j)][0] for j in range(size)]
        ys = [node_pos[(li, j)][1] for j in range(size)]
        ax.scatter(xs, ys, s=70, c="white", edgecolors="black",
                   linewidths=1.2, zorder=2)

    ax.set_xlim(-0.4, n_layers - 1 + 0.4)
    ax.set_xticks(range(n_layers))
    layer_labels = ["input"] + [f"hidden{li}" for li in range(1, n_layers - 1)] + ["output"]
    ax.set_xticklabels(layer_labels[:n_layers], fontsize=9)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def _stack_per_edge(logs, task_idx, key="per_edge_dw_layers"):
    """Aggregate per-edge value across seeds: median across seed dimension."""
    layers_per_seed = [
        [np.abs(np.asarray(layer)) for layer in log["tasks"][task_idx][key]]
        for log in logs
    ]
    n_layers = len(layers_per_seed[0])
    return [
        np.median(np.stack([s[li] for s in layers_per_seed], axis=0), axis=0)
        for li in range(n_layers)
    ]


def _cos_sim(layers_a, layers_b):
    a = np.concatenate([v.ravel() for v in layers_a])
    b = np.concatenate([v.ravel() for v in layers_b])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, 1e-12))


def _rank_corr(layers_a, layers_b):
    """Spearman rank correlation — scale-invariant edge-set overlap."""
    a = np.concatenate([v.ravel() for v in layers_a])
    b = np.concatenate([v.ravel() for v in layers_b])
    if a.size < 2:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / max(np.linalg.norm(ra) * np.linalg.norm(rb), 1e-12))


def plot_substrate_activity(results, layer_sizes, save_path=None,
                             overlap=0.0, cmap_name="viridis"):
    """
    Per-rule, three-panel layout:
      (1) median |Δw| during task A across seeds, drawn on the substrate
      (2) same for task B
      (3) per-edge product min(|Δw_A|, |Δw_B|) — edges modified by both tasks
    Each row's title reports the cosine similarity of |Δw|_A and |Δw|_B,
    averaged across seeds (the "edge-set overlap" between tasks).
    """
    rule_names = list(results.keys())
    overlap = float(overlap)
    if overlap not in results[rule_names[0]]:
        # JSON deserialization may have left float keys as strings
        overlap = list(results[rule_names[0]].keys())[0]

    n_rows = len(rule_names)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.0 * n_rows), squeeze=False)

    for row_i, r in enumerate(rule_names):
        logs = results[r][overlap]
        dw_A = _stack_per_edge(logs, task_idx=0)
        dw_B = _stack_per_edge(logs, task_idx=1)

        # Per-seed overlap metrics, averaged
        cos_sims, rank_corrs = [], []
        for log in logs:
            la = [np.abs(np.asarray(v)) for v in log["tasks"][0]["per_edge_dw_layers"]]
            lb = [np.abs(np.asarray(v)) for v in log["tasks"][1]["per_edge_dw_layers"]]
            cos_sims.append(_cos_sim(la, lb))
            rank_corrs.append(_rank_corr(la, lb))
        cos_mean = float(np.mean(cos_sims))
        rank_mean = float(np.mean(rank_corrs))

        # Cols 0–1: shared scale (so magnitude comparison is honest)
        all_vals = np.concatenate([v.ravel() for v in dw_A + dw_B])
        vmax_shared = float(all_vals.max()) if all_vals.size > 0 else 1.0
        _draw_substrate(axes[row_i, 0], layer_sizes, dw_A,
                        cmap_name=cmap_name, vmax=vmax_shared)
        axes[row_i, 0].set_title(f"{r} — task A |Δw| (shared scale)", fontsize=10)
        _draw_substrate(axes[row_i, 1], layer_sizes, dw_B,
                        cmap_name=cmap_name, vmax=vmax_shared)
        axes[row_i, 1].set_title(f"{r} — task B |Δw| (shared scale)", fontsize=10)

        # Col 2: task B with its own scale, so the structure of B is visible
        vmax_B = float(max(np.concatenate([v.ravel() for v in dw_B]).max(), 1e-12))
        _draw_substrate(axes[row_i, 2], layer_sizes, dw_B,
                        cmap_name=cmap_name, vmax=vmax_B)
        axes[row_i, 2].set_title(f"{r} — task B |Δw| (own scale)", fontsize=10)

        # Col 3: per-edge overlap min(|Δw_A|,|Δw_B|), shared scale
        overlap_layers = [np.minimum(a, b) for a, b in zip(dw_A, dw_B)]
        _draw_substrate(axes[row_i, 3], layer_sizes, overlap_layers,
                        cmap_name=cmap_name, vmax=vmax_shared)
        axes[row_i, 3].set_title(
            f"{r} — min(|Δw_A|, |Δw_B|)\n"
            f"cos = {cos_mean:.3f},  rank corr = {rank_mean:.3f}",
            fontsize=10,
        )

    fig.suptitle(
        f"Edge activity per task on the coupled substrate "
        f"(overlap = {overlap}; edge color = |Δw|, brightness = magnitude)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=140, bbox_inches="tight")
    return fig


def _draw_mesh(ax, rows, cols, n_input, out_pos_row, out_neg_row,
               W_h, W_v, cmap_name="viridis", vmax=None,
               edge_alpha=0.85, max_lw=4.0):
    """
    Draw a 2D mesh substrate with edge color/width = |W_h| or |W_v|.
    Inputs are red squares on the left; outputs are blue (+) and orange (-)
    diamonds on the right; other free nodes are small white circles.
    """
    Wh = np.abs(np.asarray(W_h))
    Wv = np.abs(np.asarray(W_v))
    if vmax is None:
        vmax = float(max(Wh.max(), Wv.max(), 1e-12))
    cmap = plt.get_cmap(cmap_name)

    def node_xy(r, c):
        return c, -r

    # Edges (sorted by magnitude so big ones draw on top)
    edge_records = []
    for r in range(rows):
        for c in range(cols - 1):
            edge_records.append((Wh[r, c], (r, c), (r, c + 1)))
    for r in range(rows - 1):
        for c in range(cols):
            edge_records.append((Wv[r, c], (r, c), (r + 1, c)))
    edge_records.sort(key=lambda e: e[0])
    for v, (r0, c0), (r1, c1) in edge_records:
        frac = v / vmax
        color = cmap(min(max(frac, 0.0), 1.0))
        lw = 0.2 + max_lw * frac
        x0, y0 = node_xy(r0, c0); x1, y1 = node_xy(r1, c1)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                alpha=edge_alpha, zorder=1)

    # Nodes
    for r in range(rows):
        for c in range(cols):
            x, y = node_xy(r, c)
            if c == 0 and r < n_input:
                ax.scatter([x], [y], s=80, c="red", marker="s",
                           edgecolors="black", linewidths=0.7, zorder=3,
                           label="input" if (r == 0) else None)
            elif c == cols - 1 and r == out_pos_row:
                ax.scatter([x], [y], s=120, c="dodgerblue", marker="D",
                           edgecolors="black", linewidths=0.8, zorder=3,
                           label="output (+)" if r == out_pos_row else None)
            elif c == cols - 1 and r == out_neg_row:
                ax.scatter([x], [y], s=120, c="darkorange", marker="D",
                           edgecolors="black", linewidths=0.8, zorder=3,
                           label="output (-)")
            else:
                ax.scatter([x], [y], s=24, c="white",
                           edgecolors="black", linewidths=0.6, zorder=2)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-(rows - 1) - 0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _stack_mesh_dw(logs, task_idx):
    """Median |Δw| per edge across seeds, returned as (Wh, Wv) arrays."""
    per_seed = [
        [np.abs(np.asarray(layer)) for layer in log["tasks"][task_idx]["per_edge_dw_layers"]]
        for log in logs
    ]
    Wh = np.median(np.stack([s[0] for s in per_seed], axis=0), axis=0)
    Wv = np.median(np.stack([s[1] for s in per_seed], axis=0), axis=0)
    return Wh, Wv


def plot_mesh_substrate_activity(results, rows, cols, n_input,
                                   out_pos_row, out_neg_row,
                                   save_path=None, overlap=0.0,
                                   cmap_name="viridis"):
    """
    For each rule: draw the 2D mesh with edge brightness = |Δw| during task A,
    task B (own scale), and the per-edge overlap min(|Δw_A|, |Δw_B|). Title
    reports cosine similarity and rank correlation across seeds.
    """
    rule_names = list(results.keys())
    overlap = float(overlap)
    if overlap not in results[rule_names[0]]:
        overlap = list(results[rule_names[0]].keys())[0]

    n_rows_grid = len(rule_names)
    fig, axes = plt.subplots(n_rows_grid, 4,
                              figsize=(4.5 * 4, 3.5 * n_rows_grid),
                              squeeze=False)

    for row_i, r in enumerate(rule_names):
        logs = results[r][overlap]
        Wh_A, Wv_A = _stack_mesh_dw(logs, task_idx=0)
        Wh_B, Wv_B = _stack_mesh_dw(logs, task_idx=1)

        cos_sims, rank_corrs = [], []
        for log in logs:
            la = [np.abs(np.asarray(v)) for v in log["tasks"][0]["per_edge_dw_layers"]]
            lb = [np.abs(np.asarray(v)) for v in log["tasks"][1]["per_edge_dw_layers"]]
            cos_sims.append(_cos_sim(la, lb))
            rank_corrs.append(_rank_corr(la, lb))
        cos_mean = float(np.mean(cos_sims))
        rank_mean = float(np.mean(rank_corrs))

        vmax_shared = float(max(Wh_A.max(), Wv_A.max(),
                                Wh_B.max(), Wv_B.max(), 1e-12))

        _draw_mesh(axes[row_i, 0], rows, cols, n_input,
                   out_pos_row, out_neg_row, Wh_A, Wv_A,
                   cmap_name=cmap_name, vmax=vmax_shared)
        axes[row_i, 0].set_title(f"{r} — task A |Δw|", fontsize=10)

        _draw_mesh(axes[row_i, 1], rows, cols, n_input,
                   out_pos_row, out_neg_row, Wh_B, Wv_B,
                   cmap_name=cmap_name, vmax=vmax_shared)
        axes[row_i, 1].set_title(f"{r} — task B |Δw| (shared scale)", fontsize=10)

        vmax_B = float(max(Wh_B.max(), Wv_B.max(), 1e-12))
        _draw_mesh(axes[row_i, 2], rows, cols, n_input,
                   out_pos_row, out_neg_row, Wh_B, Wv_B,
                   cmap_name=cmap_name, vmax=vmax_B)
        axes[row_i, 2].set_title(f"{r} — task B |Δw| (own scale)", fontsize=10)

        Wh_min = np.minimum(Wh_A, Wh_B)
        Wv_min = np.minimum(Wv_A, Wv_B)
        _draw_mesh(axes[row_i, 3], rows, cols, n_input,
                   out_pos_row, out_neg_row, Wh_min, Wv_min,
                   cmap_name=cmap_name, vmax=vmax_shared)
        axes[row_i, 3].set_title(
            f"{r} — min(|Δw_A|, |Δw_B|)\n"
            f"cos = {cos_mean:.3f},  rank corr = {rank_mean:.3f}",
            fontsize=10,
        )

    fig.suptitle(
        f"2D mesh substrate — per-task edge activity (overlap = {overlap}; "
        f"red=input, blue/orange=output ±, color=|Δw|)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=140, bbox_inches="tight")
    return fig

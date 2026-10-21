"""Small descriptive audit; no model fitting, scoring or threshold selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import (
    AGGREGATE_COUNT,
    ILLICIT,
    LICIT,
    LOCAL_NUMERIC_COUNT,
    TOTAL_TIME_STEPS,
    UNKNOWN,
    EllipticData,
    TemporalMasks,
    training_graph,
    training_labels,
    write_json,
)


def per_step_label_summary(data: EllipticData) -> pd.DataFrame:
    """Report observed illicit prevalence among known labels, never impute unknowns."""
    rows: list[dict[str, Any]] = []
    for time in range(1, TOTAL_TIME_STEPS + 1):
        labels = data.y[data.time_step == time]
        n, known = len(labels), int((labels != UNKNOWN).sum())
        illicit, licit = int((labels == ILLICIT).sum()), int((labels == LICIT).sum())
        rows.append(
            {
                "time_step": time,
                "nodes": n,
                "labeled": known,
                "licit": licit,
                "illicit": illicit,
                "unknown": n - known,
                "label_coverage": known / n if n else np.nan,
                "observed_illicit_rate_labeled": illicit / known if known else np.nan,
                "known_illicit_share_all_nodes": illicit / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_stage1_report(data: EllipticData, masks: TemporalMasks, output: Path) -> dict[str, Any]:
    """Write compact EDA, split/graph audits and machine-readable evidence.

    Full-period EDA is descriptive. It must not guide fitting, hyperparameters,
    model choice or thresholds. Model-performance test results are not inspected.
    """
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = per_step_label_summary(data)
    summary.to_csv(output / "label_coverage_by_step.csv", index=False)
    splits: list[dict[str, Any]] = []
    for name in ("train", "validation", "test"):
        nodes, labeled = masks.node_mask(name), masks.labeled_mask(name)
        splits.append(
            {
                "split": name,
                "first_step": int(data.time_step[nodes].min()),
                "last_step": int(data.time_step[nodes].max()),
                "nodes": int(nodes.sum()),
                "labeled": int(labeled.sum()),
                "illicit": int((labeled & (data.y == ILLICIT)).sum()),
                "licit": int((labeled & (data.y == LICIT)).sum()),
                "unknown": int((nodes & (data.y == UNKNOWN)).sum()),
            }
        )
    pd.DataFrame(splits).to_csv(output / "split_summary.csv", index=False)
    edges = data.edge_index
    in_degree = np.bincount(edges[1], minlength=data.num_nodes)
    out_degree = np.bincount(edges[0], minlength=data.num_nodes)
    degree = in_degree + out_degree
    visible = training_labels(data, masks)
    labeled_neighbor_edge = (visible[edges[0]] != UNKNOWN) | (visible[edges[1]] != UNKNOWN)
    eval_edge = (~masks.train_nodes[edges[0]]) | (~masks.train_nodes[edges[1]])
    graph = training_graph(data, masks)
    audit: dict[str, Any] = {
        "nodes": data.num_nodes,
        "edges": edges.shape[1],
        "time_steps": len(np.unique(data.time_step)),
        "local_numeric_features": LOCAL_NUMERIC_COUNT,
        "aggregate_numeric_features": AGGREGATE_COUNT,
        "time_stored_separately": True,
        "class_counts": {
            "illicit": int((data.y == ILLICIT).sum()),
            "licit": int((data.y == LICIT).sum()),
            "unknown": int((data.y == UNKNOWN).sum()),
        },
        "splits": splits,
        "cross_time_edges": int((data.time_step[edges[0]] != data.time_step[edges[1]]).sum()),
        "duplicate_directed_edges": int(edges.shape[1] - np.unique(edges, axis=1).shape[1]),
        "self_loops": int((edges[0] == edges[1]).sum()),
        "train_graph_nodes_including_unknown": len(graph.node_indices),
        "train_graph_edges": graph.edge_index.shape[1],
        "evaluation_edges_touching_train_labeled_nodes": int(
            (labeled_neighbor_edge & eval_edge).sum()
        ),
        "degree_definition": (
            "in_degree + out_degree; raw directed payment edges, no added self-loops"
        ),
        "degree_mean": float(degree.mean()),
        "degree_max": int(degree.max()),
        "no_preprocessing_fitted": True,
        "no_model_trained": True,
        "unresolved_protocol": [
            "include time as a feature",
            "completed-snapshot versus transaction-time information availability",
            "AP versus trapezoidal PR-AUC",
            "validation threshold selection objective",
            "alert budget and labeled-only versus operational top-k denominators",
        ],
    }
    write_json(output / "data_audit.json", audit)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    counts = audit["class_counts"]
    axes[0, 0].bar(counts.keys(), counts.values(), color=["#dc6b56", "#267b88", "#adb8bc"])
    axes[0, 0].set(title="Class counts: unknown is not a negative", ylabel="Transactions")
    axes[0, 1].plot(summary.time_step, summary.label_coverage, color="#267b88")
    axes[0, 1].set(title="Label coverage", ylabel="Known labels / all nodes", ylim=(0, 1))
    axes[1, 0].plot(summary.time_step, summary.observed_illicit_rate_labeled, color="#dc6b56")
    axes[1, 0].set(
        title="Observed illicit rate among labeled nodes", ylabel="Illicit / known labels"
    )
    for ax in (axes[0, 1], axes[1, 0]):
        ax.axvspan(24.5, 34.5, color="#f4ce72", alpha=0.15, label="Validation")
        ax.axvspan(34.5, 49.5, color="#dc6b56", alpha=0.08, label="Test")
        ax.axvline(24.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(34.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Time step", xlim=(1, 49))
    axes[0, 1].legend(loc="upper left", fontsize=8)
    for label, values, color in (
        ("Train 1–24", degree[masks.train_nodes], "#267b88"),
        ("Validation 25–34", degree[masks.validation_nodes], "#d39b28"),
        ("Test 35–49", degree[masks.test_nodes], "#dc6b56"),
    ):
        positive, freq = np.unique(values[values > 0], return_counts=True)
        axes[1, 1].loglog(positive, freq, ".", label=label, color=color, alpha=0.65)
    axes[1, 1].set(
        title=f"Directed total degree (zero-degree nodes: {int((degree == 0).sum())})",
        xlabel="In-degree + out-degree (log scale)",
        ylabel="Number of nodes (log scale)",
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Elliptic data audit · descriptive statistics only", fontsize=15)
    fig.tight_layout()
    fig.savefig(figures / "data_overview.png", dpi=160)
    plt.close(fig)
    return audit

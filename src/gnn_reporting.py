"""Validation-only GNN comparisons, learning curves and measured ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import write_json
from src.reporting import aggregate_runs, run_rows

LABELS = {"logreg": "LogReg", "xgboost": "XGBoost", "gcn": "GCN", "graphsage": "GraphSAGE"}


def gnn_markdown_table(frame: pd.DataFrame) -> str:
    """Render measured means/stds; never represent one seed as zero uncertainty."""
    text = "| Model | Inputs | Features | Seeds | AP (PR-AUC) | Precision | Recall | F1 |\n"
    text += "|---|---|---:|---:|---:|---:|---:|---:|\n"
    for row in frame.to_dict("records"):
        values = []
        for metric in ("average_precision", "precision", "recall", "f1"):
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            values.append(f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f} (std N/A)")
        text += (
            f"| {LABELS[row['model']]} | {row['feature_group']} | {row['n_features']} | "
            f"{row['n_seeds']} | " + " | ".join(values) + " |\n"
        )
    return text


def build_gnn_report(
    selected: list[dict[str, Any]],
    tuning: list[dict[str, Any]],
    choices: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Compare frozen candidates across seeds and retain explicit validation caveats."""
    summary = aggregate_runs(selected)
    rows = run_rows(selected)
    rows["best_epoch"] = [run["best_epoch"] for run in selected]
    rows["epochs_run"] = [run["epochs_run"] for run in selected]
    rows.to_csv(output / "validation_per_seed.csv", index=False)
    run_rows(tuning).to_csv(output / "tuning_candidates.csv", index=False)
    summary.to_csv(output / "validation_summary.csv", index=False)
    primary = summary[summary.feature_group == "all"]
    primary.to_csv(output / "validation_primary.csv", index=False)
    baseline = pd.read_csv(config["experiment"]["baseline_summary"])
    graph = pd.read_csv(config["experiment"]["graph_summary"])
    comparison = pd.concat(
        [
            baseline[(baseline.model != "logreg") | (baseline.feature_group == "all")],
            graph[graph.feature_group.isin(["all_structure", "all_graph"])],
            primary,
        ],
        ignore_index=True,
    )
    comparison.to_csv(output / "comparison_validation.csv", index=False)
    delta_rows = []
    for model_name, group in rows.groupby("model"):
        message = group[group.feature_group == "all"].set_index("seed")
        control = group[group.feature_group == "all_no_messages"].set_index("seed")
        if control.empty:
            continue
        if set(message.index) != set(control.index):
            raise ValueError("Message/no-message comparisons require matching seeds.")
        for seed in sorted(message.index):
            delta_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "ap_delta_message_minus_no_message": float(
                        message.loc[seed, "average_precision"]
                        - control.loc[seed, "average_precision"]
                    ),
                    "f1_delta_message_minus_no_message": float(
                        message.loc[seed, "f1"] - control.loc[seed, "f1"]
                    ),
                }
            )
    pd.DataFrame(
        delta_rows,
        columns=[
            "model",
            "seed",
            "ap_delta_message_minus_no_message",
            "f1_delta_message_minus_no_message",
        ],
    ).to_csv(output / "message_passing_lift.csv", index=False)
    report: dict[str, Any] = {
        "evaluation_split": "validation",
        "time_steps": [25, 34],
        "test_evaluated": False,
        "primary_metric": "average_precision",
        "choices": choices,
        "std_definition": "sample standard deviation across seeds, ddof=1",
        "selected_run_count": len(selected),
        "tuning_run_count": len(tuning),
        "unique_fit_count": len({run["run_id"] for run in selected + tuning}),
        "primary_results": primary.to_dict("records"),
        "interpretation": (
            "Validation selects candidates, epochs and thresholds; scores are optimistic "
            "development diagnostics. Five seeds measure fitting variation, not temporal "
            "uncertainty. Final test and monitoring remain Stages 5 and 6."
        ),
    }
    # Strict JSON cannot serialize NaN from a deliberately single-seed CLI run.
    report["primary_results"] = [
        {key: (None if pd.isna(value) else value) for key, value in row.items()}
        for row in report["primary_results"]
    ]
    write_json(output / "selection.json", report)
    seed = config["experiment"]["representative_seed"]
    representatives = [run for run in selected if run["seed"] == seed]
    frames = []
    for run in representatives:
        with np.load(run["artifacts"]["validation_predictions"]["path"]) as scores:
            known = scores["label"] != -1
            frames.append(
                pd.DataFrame(
                    {
                        "model": run["model"],
                        "feature_group": run["feature_group"],
                        "seed": seed,
                        "tx_id": scores["tx_id"][known],
                        "time_step": scores["time_step"][known],
                        "label": scores["label"][known],
                        "score": scores["score"][known],
                        "threshold": run["threshold"]["value"],
                    }
                )
            )
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            output / "representative_validation_scores.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        plot_diagnostics(selected, summary, seed, output)
    text = (
        "# Stage 4 — GCN and GraphSAGE\n\n"
        "**Validation only, known labels in steps 25–34. Test remains unevaluated.**\n\n"
        "## Comparison with earlier stages\n\n"
        + gnn_markdown_table(comparison)
        + "\n## GNN feature and message ablations\n\n"
        + gnn_markdown_table(summary)
        + "\n## Training and selection\n\n"
        "Two message-passing layers, ReLU and dropout; GCN uses normalized self-loops, "
        "GraphSAGE uses mean neighbor aggregation with a separate root transformation. "
        "Raw edges are made bidirectional and deduplicated within each snapshot. "
        "Payment direction is not preserved by messages. GCN adjacency caching is disabled "
        "because training and validation use different graphs. No batch normalization or "
        "globally learned graph normalization is used.\n\n"
        "The scaler fits known training rows only and transforms all available nodes. "
        "Unknown nodes participate in messages but never in loss, early stopping or metrics. "
        "Weighted BCE derives its positive weight from known training labels only. "
        "Full-batch updates use the disjoint training snapshots 1–24; validation graphs "
        "25–34 are separate, evaluated without gradients or dropout.\n\n"
        f"Each architecture/feature group compares {len(config['training']['candidates'])} "
        f"predeclared candidates on seed {config['experiment']['tuning_seed']}. "
        "The selected candidate is frozen for seed replications, reusing its tuning-seed fit. "
        "Early stopping maximizes validation AP, exact ties keeping the earliest epoch. "
        "After restoring that checkpoint, each run selects its maximum-F1 validation "
        "threshold, ties favoring the higher threshold. No best-seed choice, test tuning, "
        "or train+validation refit occurs. Exact settings are in config and every run JSON.\n\n"
        "Structure-only inputs are the six snapshot-local NetworkX features from Stage 3; "
        "there are no transaction attributes or time input in this ablation. "
        "The all_no_messages control removes every inter-node edge and uses the selected "
        "all-feature hyperparameters without a separate search. GCN still has self-loops; "
        "GraphSAGE retains its root transformation. Controls select epochs/thresholds "
        "independently on validation. They test the implemented message contribution, "
        "not every possible neural architecture.\n\n"
        f"Actual unique fits: {report['unique_fit_count']}; selected seed results: "
        f"{len(selected)}. Mean ± sample std, ddof=1; one seed has unavailable std. "
        "The representative diagnostic seed is predeclared, not the best seed.\n\n"
        "![GNN feature ablations](figures/gnn_comparison.png)\n\n"
        "![Representative learning curves](figures/learning_curves.png)\n\n"
        "## Limits\n\n"
        "Validation selects settings, epochs and thresholds, so these are optimistic "
        "development estimates. The search is deliberately small. Seed variability is "
        "not a confidence interval or evidence of temporal stability. Differences between "
        "architectures also reflect optimization and feature choices; graph learning is "
        "not isolated by comparing a GNN to a tree alone. The no-message control helps "
        "address that narrower question.\n\n"
        "All-feature inputs already contain supplied neighbor aggregates. Completed "
        "snapshots and supplied forward-neighbor aggregates do not demonstrate "
        "transaction-arrival availability. Label timestamps are unavailable. Unknown "
        "alerts cannot be judged correct/incorrect. Scores are uncalibrated. "
        "Test comparisons and monitoring remain later review stages.\n"
    )
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    return report


def plot_diagnostics(
    selected: list[dict[str, Any]],
    summary: pd.DataFrame,
    seed: int,
    output: Path,
) -> None:
    """Plot measured AP/std and each representative primary run's full epoch history."""
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (model_name, group) in zip(axes, summary.groupby("model")):
        axis.bar(
            group.feature_group,
            group.average_precision_mean,
            yerr=group.average_precision_std.fillna(0),
            capsize=4,
            color="#297f9e",
        )
        axis.set(title=LABELS[model_name], ylabel="Validation AP", ylim=(0, 1.04))
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Feature and message ablations — mean ± sample std")
    fig.tight_layout()
    fig.savefig(directory / "gnn_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in selected:
        if run["seed"] != seed or run["feature_group"] != "all":
            continue
        history = pd.read_csv(run["artifacts"]["history"]["path"])
        label = LABELS[run["model"]]
        axes[0].plot(history.epoch, history.train_weighted_bce, label=label)
        axes[1].plot(history.epoch, history.validation_ap, label=label)
        axes[1].scatter(run["best_epoch"], run["best_validation_ap"], s=35)
    axes[0].set(xlabel="Epoch", ylabel="Training weighted BCE")
    axes[1].set(xlabel="Epoch", ylabel="Validation AP", ylim=(0, 1.02))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle(f"All features, predeclared seed {seed}; dots mark restored checkpoints")
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)

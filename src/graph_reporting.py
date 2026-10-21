"""Measured validation lift from paired Stage 3 runs; no test or GNN claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import write_json
from src.reporting import aggregate_runs, markdown_table, run_rows

LABELS = {
    "all_control": "All-feature control",
    "all_structure": "All + structure",
    "all_graph": "All + structure + neighbor scores",
    "structure_only": "Structure only",
}


def paired_lift(runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair each graph variant with the same seed's freshly fitted control."""
    rows = run_rows(runs)
    control = rows[rows.feature_group == "all_control"].set_index("seed")
    differences = []
    for row in rows.to_dict("records"):
        if row["feature_group"] == "all_control":
            continue
        if row["seed"] not in control.index:
            raise ValueError("Every graph result requires a control with the same seed.")
        base = control.loc[row["seed"]]
        differences.append(
            {
                "variant": row["feature_group"],
                "seed": row["seed"],
                "ap_delta": row["average_precision"] - float(base.average_precision),
                "f1_delta": row["f1"] - float(base.f1),
            }
        )
    detail = pd.DataFrame(differences, columns=["variant", "seed", "ap_delta", "f1_delta"])
    summaries = []
    for variant, group in detail.groupby("variant"):
        summaries.append(
            {
                "variant": variant,
                "n_seeds": len(group),
                "ap_delta_mean": float(group.ap_delta.mean()),
                "ap_delta_std": float(group.ap_delta.std(ddof=1)) if len(group) > 1 else None,
                "f1_delta_mean": float(group.f1_delta.mean()),
                "f1_delta_std": float(group.f1_delta.std(ddof=1)) if len(group) > 1 else None,
                "seeds_with_positive_ap_delta": int((group.ap_delta > 0).sum()),
            }
        )
    return detail, pd.DataFrame(summaries)


def build_graph_report(
    selected: list[dict[str, Any]],
    baseline: dict[int, dict[str, Any]],
    feature_evidence: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Build actual mean/std and paired deltas, plus explicit availability diagnostics."""
    frame = run_rows(selected)
    summary = aggregate_runs(selected)
    detail, lift = paired_lift(selected)
    frame.to_csv(output / "validation_per_seed.csv", index=False)
    summary.to_csv(output / "validation_summary.csv", index=False)
    detail.to_csv(output / "paired_lift.csv", index=False)
    lift.to_csv(output / "paired_lift_summary.csv", index=False)
    snapshots = pd.DataFrame(feature_evidence["snapshot_audit"])
    snapshots.to_csv(output / "snapshot_audit.csv", index=False)
    baseline_checks = []
    for run in selected:
        if run["feature_group"] == "all_control":
            previous = baseline[run["seed"]]
            baseline_checks.append(
                {
                    "seed": run["seed"],
                    "stage2_run_id": previous["run_id"],
                    "stage3_run_id": run["run_id"],
                    "train_inputs_identical": run["input_fingerprints"]["train"]
                    == previous["input_fingerprints"]["train"],
                    "validation_inputs_identical": run["input_fingerprints"]["validation"]
                    == previous["input_fingerprints"]["validation"],
                    "frozen_threshold_identical": run["threshold"]["value"]
                    == previous["threshold"]["value"],
                    "validation_metrics_identical": run["validation_metrics"]
                    == previous["validation_metrics"],
                    "validation_score_digest_identical": run["threshold"][
                        "validation_scores_sha256"
                    ]
                    == previous["threshold"]["validation_scores_sha256"],
                }
            )
    write_json(output / "baseline_reproduction.json", baseline_checks)
    validation_snapshots = snapshots[snapshots.time_step >= 25]
    report = {
        "evaluation_split": "validation",
        "test_evaluated": False,
        "hyperparameters": "frozen Stage 2 XGBoost candidate for every variant; no retuning",
        "std_definition": "sample standard deviation over seeds, ddof=1",
        "baseline_reproduction": baseline_checks,
        "raw_train_label_neighbor_coverage_validation_nodes": int(
            validation_snapshots.nodes_with_visible_train_label_neighbor.sum()
        ),
        "paired_lift": json.loads(lift.to_json(orient="records", double_precision=15)),
        "limits": [
            "Validation has been used for development; final test performance is deferred.",
            "No GNN was trained in Stage 3.",
            "Completed-snapshot features are retrospective, not transaction-arrival features.",
            "Past train labels are assumed available by the next step; label delays are unknown.",
            "Forward-chained score helpers change training sizes; cold start uses a fixed prior.",
            "Seed std measures fitting variation, not temporal uncertainty or significance.",
            "Adding columns changes stochastic feature subsampling; paired lift is descriptive.",
        ],
    }
    write_json(output / "selection.json", report)
    order = {name: i for i, name in enumerate(config["experiment"]["variants"])}
    summary = summary.assign(order=summary.feature_group.map(order)).sort_values("order")
    text = (
        "# Stage 3 — additional graph features\n\n"
        "**Validation only: steps 25–34, known labels only. Test remains unevaluated.**\n\n"
        + markdown_table(summary)
        + "\nEvery variant uses the Stage 2 selected XGBoost parameters without retuning. "
        "Thresholds maximize validation illicit F1 and remain frozen for later test use. "
        "Precision/recall/F1 describe validation selection, not a final test estimate.\n\n"
        "## Feature definitions\n\n"
        "Six structural inputs: directed total degree, in-degree, out-degree, directed PageRank, "
        "PageRank multiplied by current snapshot node count, and clustering on the undirected "
        "projection. PageRank normalization uses one snapshot, excluding future steps. "
        "Unknown nodes stay in these snapshots. No test graph statistics are computed.\n\n"
        "Two neighbor-score inputs: mean/max over the unique union of predecessors and successors, "
        "excluding the center node. A fixed balanced LogReg on 93 local numeric features generates "
        "scores from earlier training steps only; validation uses a frozen steps-1–24 helper. "
        "Its scaler fits earlier known labels only. First-step/insufficient-class cases use a "
        "configured 0.5 cold-start score, and isolates use a configured zero sentinel. Helpers are "
        "shared deterministic feature generators across downstream seeds, not five independent "
        "upstream ensembles. Scores have not been calibrated as illicit probabilities.\n\n"
        "Raw train-label neighbor coverage is diagnostic only. Held-out snapshots have no edges "
        "to train-labeled nodes, so naive neighbor-label features are unavailable there. They are "
        "excluded from model inputs. Every helper fit/cutoff is logged separately.\n\n"
        "## Paired validation lift\n\n"
        "Each row subtracts the same seed's freshly fitted all-feature control. "
        "Positive AP means higher validation average precision.\n\n"
        "| Variant | Seeds | AP delta mean ± std | F1 delta mean ± std | Positive AP seeds |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    for row in lift.to_dict("records"):

        def formatted(metric: str) -> str:
            mean, std = row[f"{metric}_delta_mean"], row[f"{metric}_delta_std"]
            return f"{mean:+.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:+.4f} (std N/A)"

        text += (
            f"| {LABELS[row['variant']]} | {row['n_seeds']} | {formatted('ap')} | "
            f"{formatted('f1')} | {row['seeds_with_positive_ap_delta']} |\n"
        )
    text += (
        "\n![Validation comparison and paired AP lift](figures/graph_comparison.png)\n\n"
        "## Limits\n\n" + "\n".join(f"- {limit}" for limit in report["limits"]) + "\n\n"
        "The control already includes supplied neighbor aggregates. This comparison measures "
        "incremental evidence from the new inputs. Structure-only is a deliberately restricted "
        "ablation, not an optimized graph-only model. A GNN comparison remains Stage 4.\n"
    )
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(np.arange(len(summary)), summary.average_precision_mean, color="#267b88")
    axes[0].errorbar(
        np.arange(len(summary)),
        summary.average_precision_mean,
        yerr=summary.average_precision_std.fillna(0),
        fmt="none",
        color="black",
        capsize=4,
    )
    axes[0].set_xticks(
        np.arange(len(summary)),
        [LABELS[x] for x in summary.feature_group],
        rotation=15,
        ha="right",
        fontsize=8,
    )
    axes[0].set(ylabel="Average precision", ylim=(0, 1.03), title="Validation mean ± seed std")
    for variant in ("all_structure", "all_graph"):
        values = detail[detail.variant == variant]
        if len(values):
            axes[1].plot(values.seed, values.ap_delta, "o-", label=LABELS[variant])
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set(
        xlabel="Seed",
        ylabel="AP difference from paired control",
        title="Incremental validation lift",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle("Stage 3 · validation only · fixed XGBoost hyperparameters")
    fig.tight_layout()
    fig.savefig(figure_dir / "graph_comparison.png", dpi=150)
    plt.close(fig)
    representative = []
    for run in selected:
        if run["seed"] == config["experiment"]["representative_seed"]:
            with np.load(run["artifacts"]["validation_predictions"]["path"]) as scores:
                representative.append(
                    pd.DataFrame(
                        {
                            "run_id": run["run_id"],
                            "variant": run["feature_group"],
                            "tx_id": scores["tx_id"],
                            "time_step": scores["time_step"],
                            "label": scores["label"],
                            "score": scores["score"],
                            "frozen_threshold": run["threshold"]["value"],
                        }
                    )
                )
            if run["feature_group"] == "all_graph":
                model = XGBClassifier()
                model.load_model(run["artifacts"]["model"]["path"])
                pd.DataFrame(
                    {
                        "feature": run["feature_names"],
                        "normalized_gain_importance": model.feature_importances_,
                    }
                ).to_csv(output / "training_feature_importance_seed0.csv", index=False)
    if representative:
        pd.concat(representative, ignore_index=True).to_csv(
            output / "representative_validation_scores.csv.gz", index=False, compression="gzip"
        )
    return report

"""Readable final-test evidence with frozen-policy and sparse-label caveats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import write_json
from src.evaluation_metrics import budget_metrics

METRICS = ("average_precision", "pr_auc_trapezoid", "precision", "recall", "f1", "roc_auc")
NAMES = {
    "s2_logreg_all_time": "LogReg (all)",
    "s2_xgboost_local_time": "XGBoost (local)",
    "s2_xgboost_all_time": "XGBoost (all)",
    "s3_xgboost_all_structure": "XGBoost + structure",
    "s3_xgboost_all_graph": "XGBoost + structure + scores",
    "s4_gcn_all": "GCN (all)",
    "s4_graphsage_all": "GraphSAGE (all)",
}


def summarize(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Aggregate distinct seeds; std is sample variability, not temporal uncertainty."""
    rows = []
    for values, group in frame.groupby(keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        if group.seed.duplicated().any():
            raise ValueError("Each summary requires distinct seeds.")
        row = dict(zip(keys, values))
        row["n_seeds"] = len(group)
        for metric in metrics:
            values_array = group[metric].to_numpy(dtype=float)
            finite = values_array[np.isfinite(values_array)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else None
            row[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def pretty(row: dict[str, Any], metric: str) -> str:
    """Format measured values, retaining undefined std as unavailable."""
    mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
    if pd.isna(mean):
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f} (std N/A)"


def comparison_table(frame: pd.DataFrame) -> str:
    """Keep the headline focused on imbalanced-classification metrics."""
    text = "| Model | Seeds | AP (PR-AUC) | Illicit precision | Illicit recall | Illicit F1 |\n"
    text += "|---|---:|---:|---:|---:|---:|\n"
    for row in frame.to_dict("records"):
        text += (
            f"| {NAMES.get(row['variant'], row['variant'])} | {row['n_seeds']} | "
            + " | ".join(
                pretty(row, metric) for metric in ("average_precision", "precision", "recall", "f1")
            )
            + " |\n"
        )
    return text


def build_final_report(
    runs: list[dict[str, Any]],
    plan: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Report every declared fit and preserve the pre-test model selection."""
    primary_order = plan["config"]["experiment"]["primary_variants"]
    rows, budget_rows, policy_rows, confusion_rows, workload_rows = [], [], [], [], []
    for run in runs:
        rows.append(
            {
                "variant": run["variant"],
                "seed": run["seed"],
                "primary": run["primary"],
                "model": run["model"],
                "feature_group": run["feature_group"],
                "include_time": run["include_time"],
                "n_features": run["n_features"],
                "source_run_id": run["source_run_id"],
                "threshold": run["frozen_threshold"],
                **{metric: run["test_metrics"][metric] for metric in METRICS},
            }
        )
        tn, fp = run["test_metrics"]["confusion_matrix"][0]
        fn, tp = run["test_metrics"]["confusion_matrix"][1]
        confusion_rows.append(
            {
                "variant": run["variant"],
                "seed": run["seed"],
                "threshold": run["frozen_threshold"],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        workload_rows.append(
            {
                "variant": run["variant"],
                "seed": run["seed"],
                "threshold": run["frozen_threshold"],
                **run["frozen_threshold_workload"],
            }
        )
        for budget in run["per_step_budget_metrics"]:
            budget_rows.append({"variant": run["variant"], "seed": run["seed"], **budget})
        for policy in run["recall_policies"]:
            row = {
                "variant": run["variant"],
                "seed": run["seed"],
                "target_validation_recall": policy["target_recall"],
                "frozen_threshold": policy["threshold"],
            }
            for split in ("validation", "test"):
                row.update(
                    {
                        f"{split}_{key}": policy[f"{split}_metrics"][key]
                        for key in ("precision", "recall", "f1")
                    }
                )
                row.update(
                    {f"{split}_{key}": value for key, value in policy[f"{split}_workload"].items()}
                )
            policy_rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "test_per_seed.csv", index=False)
    summary = summarize(frame, ["variant"], list(METRICS))
    summary.to_csv(output / "test_summary_all_variants.csv", index=False)
    primary = summary.set_index("variant").loc[primary_order].reset_index()
    primary.to_csv(output / "test_comparison.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrices.csv", index=False)
    pd.DataFrame(workload_rows).to_csv(output / "frozen_threshold_workload.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(output / "recall_policy_transfer.csv", index=False)
    budget_frame = pd.DataFrame(budget_rows)
    budget_frame.to_csv(output / "budget_per_step.csv", index=False)
    budget_micro = []
    for (variant, seed, view), group in budget_frame.groupby(["variant", "seed", "view"]):
        alerts = int(group.alerts_selected.sum())
        illicit = int(group.known_illicit_alerts.sum())
        known = int(group.known_alerts.sum())
        positives = int(group.support_known_illicit.sum())
        budget_micro.append(
            {
                "variant": variant,
                "seed": seed,
                "view": view,
                "alerts_selected": alerts,
                "known_illicit_alerts": illicit,
                "known_licit_alerts": int(group.known_licit_alerts.sum()),
                "unknown_alerts": int(group.unknown_alerts.sum()),
                "known_alerts": known,
                "precision_at_k_labeled": illicit / alerts
                if view == "labeled_only" and alerts
                else None,
                "precision_among_known_alerts": illicit / known if known else None,
                "known_label_coverage": known / alerts if alerts else None,
                "recall_known_illicit": illicit / positives if positives else None,
            }
        )
    micro = pd.DataFrame(budget_micro)
    micro.to_csv(output / "budget_per_seed.csv", index=False)
    budget_metrics_names = [key for key in micro.columns if key not in {"variant", "seed", "view"}]
    budget_summary = summarize(micro, ["variant", "view"], budget_metrics_names)
    budget_summary.to_csv(output / "budget_summary.csv", index=False)
    validation_test = []
    for row in primary.to_dict("records"):
        validation_ap = plan["primary_validation_ap_means"][row["variant"]]
        validation_test.append(
            {
                "variant": row["variant"],
                "validation_ap_mean": validation_ap,
                "test_ap_mean": row["average_precision_mean"],
                "test_minus_validation_ap": row["average_precision_mean"] - validation_ap,
            }
        )
    pd.DataFrame(validation_test).to_csv(output / "validation_test_comparison.csv", index=False)
    baseline = frame[frame.variant == "s2_xgboost_all_time"].set_index("seed")
    paired = []
    if not baseline.empty:
        for variant in primary_order:
            if variant == "s2_xgboost_all_time":
                continue
            for row in frame[frame.variant == variant].to_dict("records"):
                paired.append(
                    {
                        "variant": variant,
                        "seed": row["seed"],
                        "ap_delta_vs_xgboost_all": row["average_precision"]
                        - float(baseline.loc[row["seed"], "average_precision"]),
                        "f1_delta_vs_xgboost_all": row["f1"]
                        - float(baseline.loc[row["seed"], "f1"]),
                    }
                )
    pd.DataFrame(paired).to_csv(output / "paired_test_lift.csv", index=False)
    representative_seed = plan["config"]["experiment"]["representative_seed"]
    representatives = [run for run in runs if run["primary"] and run["seed"] == representative_seed]
    evidence, queue_evidence = [], []
    for run in representatives:
        with np.load(run["test_predictions"]["path"]) as prediction:
            known = prediction["label"] != -1
            evidence.append(
                pd.DataFrame(
                    {
                        "variant": run["variant"],
                        "seed": representative_seed,
                        "tx_id": prediction["tx_id"][known],
                        "time_step": prediction["time_step"][known],
                        "label": prediction["label"][known],
                        "score": prediction["score"][known],
                        "threshold": run["frozen_threshold"],
                    }
                )
            )
            for step in range(35, 50):
                current = prediction["time_step"] == step
                y, score, ids = (prediction[key][current] for key in ("label", "score", "tx_id"))
                if "all_transactions" in plan["config"]["alert_budget"]["views"]:
                    _, chosen = budget_metrics(
                        y,
                        score,
                        ids,
                        k=plan["config"]["alert_budget"]["per_time_step"],
                        view="all_transactions",
                    )
                    queue_evidence.append(
                        pd.DataFrame(
                            {
                                "variant": run["variant"],
                                "seed": representative_seed,
                                "time_step": step,
                                "rank": np.arange(1, len(chosen) + 1),
                                "tx_id": ids[chosen],
                                "score": score[chosen],
                                "label": y[chosen],
                            }
                        )
                    )
    if evidence:
        pd.concat(evidence, ignore_index=True).to_csv(
            output / "representative_test_scores_labeled.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    if queue_evidence:
        pd.concat(queue_evidence, ignore_index=True).to_csv(
            output / "representative_operational_queues.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    plot_final(primary, representatives, plan, output)
    selected = plan["validation_selected_variant"]
    selected_row = primary[primary.variant == selected].iloc[0]
    report = {
        "evaluation_split": "test",
        "time_steps": [35, 49],
        "evaluated_runs": len(runs),
        "primary_variants": primary_order,
        "validation_selected_variant": selected,
        "validation_selected_variant_test_ap": float(selected_row.average_precision_mean),
        "test_used_for_model_or_threshold_selection": False,
        "std_definition": "sample std over training seeds, ddof=1; not temporal uncertainty",
        "alert_budget_per_step": plan["config"]["alert_budget"]["per_time_step"],
        "unknown_labels_assigned_negative": False,
        "monitoring_completed": False,
    }
    write_json(output / "evaluation_summary.json", report)
    budget_text = (
        "| Model | Labeled-only precision@k | Known illicit in all-node queue | "
        "Unknown share in queue |\n"
    )
    budget_text += "|---|---:|---:|---:|\n"
    for variant in primary_order:
        subset = budget_summary[budget_summary.variant == variant].set_index("view")
        labeled = (
            pretty(subset.loc["labeled_only"].to_dict(), "precision_at_k_labeled")
            if "labeled_only" in subset.index
            else "N/A"
        )
        if "all_transactions" in subset.index:
            operational = subset.loc["all_transactions"]
            count = pretty(operational.to_dict(), "known_illicit_alerts")
            unknown_share = 1 - operational.known_label_coverage_mean
            share = f"{unknown_share:.1%}"
        else:
            count, share = "N/A", "N/A"
        budget_text += f"| {NAMES.get(variant, variant)} | {labeled} | {count} | {share} |\n"
    text = (
        "# Stage 5 — frozen-model temporal test evaluation\n\n"
        "**Test steps 35–49. Unknown labels are excluded from classification metrics.**\n\n"
        "## Main comparison\n\n"
        + comparison_table(primary)
        + "\nAP is sklearn average precision; trapezoidal PR-AUC and secondary ROC-AUC are "
        "separate columns in the CSV. Each metric uses the same known test transactions. "
        "Mean ± sample std over the declared training seeds; no best-seed selection.\n\n"
        f"The pre-test validation-selected variant was **{NAMES.get(selected, selected)}**. "
        "It remains the selected candidate regardless of test ranking. Every model, scaler, "
        "hyperparameter, primary cutoff, tie rule and alert budget was frozen before test "
        "scoring. No model/helper/preprocessor was fitted in this stage.\n\n"
        "![Test AP and frozen-threshold F1](figures/test_comparison.png)\n\n"
        "## Alert budget: two different questions\n\n"
        f"The declared budget is up to **{report['alert_budget_per_step']} alerts per time step**. "
        "The choice form returned no selections; this explicit benchmark default was recorded "
        "before test scoring. It is not a measured staffing requirement. These queues rank "
        "scores and are separate from the fixed-F1-threshold alerts. Ties use lexical "
        "transaction ID, never labels.\n\n"
        + budget_text
        + "\nLabeled-only precision@k filters to known labels before ranking. It is a "
        "retrospective benchmark, not an operational queue. All-node queues rank every "
        "transaction and report known illicit/licit counts, unknown counts and label coverage. "
        "Their overall precision cannot be measured from this release. Unknown alerts are "
        "never false positives. Counts are totals over test steps, averaged over seeds; "
        "they are not sums of distinct cases across repeated seeds. Labeled precision@k is "
        "micro-aggregated from selected labeled alerts, with k capped by cohort size.\n\n"
        "## Frozen thresholds and recall targets\n\n"
        "Primary precision/recall/F1 use each historical run's saved validation-max-F1 cutoff. "
        "No cutoff is optimized on test. Additional recall targets are diagnostics: the "
        "highest validation cutoff meeting each declared target is frozen and transferred "
        "to test. `recall_policy_transfer.csv` shows achieved recall and all-node alert "
        "workload in both periods. A validation recall target is not a test recall guarantee.\n\n"
        "![Test ranking and selected-candidate confusion matrix](figures/test_diagnostics.png)\n\n"
        "![Validation-only threshold tradeoffs](figures/validation_tradeoffs.png)\n\n"
        "## Ablations and evidence\n\n"
        "`test_summary_all_variants.csv` retains local/all, time/no-time, structure-only and "
        "no-message controls from earlier stages. `paired_test_lift.csv` subtracts the "
        "same seed's all-feature XGBoost result. Paired differences are descriptive; five "
        "seeds do not create five independent datasets or a significance test.\n\n"
        "All per-seed confusion matrices, frozen-threshold workloads, budget queues and "
        "recall policies have CSV evidence. The representative seed is predeclared. "
        "Exact full-node scores live under ignored `artifacts/`; committed compressed CSVs "
        "retain known-label test scores and all-node queue evidence for primary seed 0. "
        "Use pandas `float_precision='round_trip'` to preserve score/threshold ties.\n\n"
        "## Limits and next review\n\n"
        "Validation was reused for settings, checkpoint selection and cutoffs. Test is "
        "one later historical period, not proof of deployment performance. Label coverage "
        "and prevalence differ over time; PR metrics reflect prevalence as well as ranking. "
        "Unknown labels are not representative negatives. Scores are uncalibrated. "
        "All-feature baselines already contain supplied graph aggregates. Completed-snapshot "
        "topology and forward-neighbor features do not establish "
        "transaction-arrival availability.\n\n"
        "Per-step performance monitoring, score/feature drift and the reported step-43 "
        "regime change remain Stage 6. No shutdown causality or retraining conclusion is "
        "claimed here. README/About redesign is deferred until after the Stage 6 review.\n"
    )
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    return report


def plot_final(
    primary: pd.DataFrame, representatives: list[dict], plan: dict, output: Path
) -> None:
    """Plot aggregate metrics and predeclared representative policy diagnostics."""
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    labels = [NAMES.get(variant, variant) for variant in primary.variant]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, metric, title in zip(
        axes, ("average_precision", "f1"), ("Test AP", "Test F1 at frozen thresholds")
    ):
        axis.barh(
            labels,
            primary[f"{metric}_mean"],
            xerr=primary[f"{metric}_std"].fillna(0),
            color="#247d94",
            capsize=3,
        )
        axis.set(xlim=(0, 1.03), title=title)
        axis.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle("Future test period: mean ± sample std over training seeds")
    fig.tight_layout()
    fig.savefig(directory / "test_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    selected_run = None
    for run in representatives:
        with np.load(run["test_predictions"]["path"]) as prediction:
            known = prediction["label"] != -1
            precision, recall, _ = precision_recall_curve(
                prediction["label"][known], prediction["score"][known]
            )
            axes[0].plot(recall, precision, label=NAMES.get(run["variant"], run["variant"]))
        if run["variant"] == plan["validation_selected_variant"]:
            selected_run = run
    axes[0].set(xlabel="Illicit recall", ylabel="Illicit precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    if selected_run:
        matrix = np.asarray(selected_run["test_metrics"]["confusion_matrix"])
        axes[1].imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                axes[1].text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                )
        axes[1].set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Licit", "Illicit"],
            yticklabels=["Licit", "Illicit"],
            xlabel="Predicted",
            ylabel="Known label",
            title="Validation-selected candidate; frozen cutoff",
        )
    fig.suptitle(
        f"Test known labels; predeclared seed {plan['config']['experiment']['representative_seed']}"
    )
    fig.tight_layout()
    fig.savefig(directory / "test_diagnostics.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for run in representatives:
        entry = next(
            item for item in plan["references"] if item["source_run_id"] == run["source_run_id"]
        )
        with np.load(entry["validation_predictions"]["path"]) as prediction:
            known = prediction["label"] != -1
            precision, recall, _ = precision_recall_curve(
                prediction["label"][known], prediction["score"][known]
            )
            axes[0].plot(recall, precision, label=NAMES.get(run["variant"], run["variant"]))
            point = entry["validation_replay"]["metrics"]
            axes[0].scatter(point["recall"], point["precision"], s=20)
        if run["variant"] == plan["validation_selected_variant"]:
            policies = entry["recall_policies"]
            axes[1].plot(
                [p["achieved_validation_recall"] for p in policies],
                [p["validation_workload"]["alert_count_all"] for p in policies],
                "o-",
            )
            for policy in policies:
                axes[1].annotate(
                    f"target {policy['target_recall']:.0%}",
                    (
                        policy["achieved_validation_recall"],
                        policy["validation_workload"]["alert_count_all"],
                    ),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=8,
                )
    axes[0].set(
        xlabel="Validation illicit recall",
        ylabel="Validation illicit precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="Dots: saved max-F1 thresholds",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="Validation recall among known illicit",
        ylabel="All-node validation alerts",
        title="Validation-selected candidate: review workload",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Validation-only policy selection; test uses these frozen cutoffs")
    fig.tight_layout()
    fig.savefig(directory / "validation_tradeoffs.png", dpi=160)
    plt.close(fig)

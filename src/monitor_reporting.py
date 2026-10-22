"""Monitoring evidence, readable figures and explicitly retrospective warning rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.atomic_artifacts import write_json
from src.final_reporting import NAMES, summarize

QUALITY = [
    "average_precision",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "alert_rate_all",
    "mean_score_all",
    "label_coverage",
    "observed_illicit_rate_labeled",
]
COLORS = {
    "s3_xgboost_all_structure": "#007f8b",
    "s2_xgboost_all_time": "#d28c19",
    "s4_gcn_all": "#c85745",
    "s4_graphsage_all": "#6c5aa5",
}


def warning_actions(score: pd.DataFrame, rolling: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Persist drift warnings and label-dependent review recommendations without retraining."""
    candidate, config = reference["validation_selected_variant"], reference["config"]
    result = (
        rolling[rolling.variant == candidate]
        .merge(
            score[score.split == "test"][["variant", "seed", "time_step", "psi", "ks_statistic"]],
            on=["variant", "seed", "time_step"],
            validate="one_to_one",
        )
        .sort_values(["seed", "time_step"])
    )
    result["score_drift_warning"] = result.psi > config["drift"]["psi_warning"]
    result["persistent_score_drift"] = False
    for _, group in result.groupby("seed"):
        consecutive = 0
        previous = None
        for index, row in group.iterrows():
            consecutive = (
                consecutive + 1
                if row.score_drift_warning and previous == row.time_step - 1
                else int(row.score_drift_warning)
            )
            result.loc[index, "persistent_score_drift"] = (
                consecutive >= config["drift"]["consecutive_steps"]
            )
            previous = row.time_step
    result["investigate"] = result.score_drift_warning | result.performance_warning
    result["retraining_review"] = result.performance_warning
    result["recommended_action"] = np.select(
        [result.performance_warning, result.persistent_score_drift, result.score_drift_warning],
        [
            "review_retraining_with_recent_labels",
            "investigate_persistent_shift_and_sample_labels",
            "investigate_score_shift",
        ],
        default="observe",
    )
    result["supervised_metrics_retrospective"] = True
    result["automatic_retraining_or_promotion"] = False
    return result


def _timeline(ax: Any, *, full: bool = False) -> None:
    """Use the literature marker as context, with fixed split boundaries."""
    ax.axvline(43, color="#c85745", linestyle=":", linewidth=1.4)
    if full:
        ax.axvline(24.5, color="#65747b", linestyle="--", linewidth=0.8)
        ax.axvline(34.5, color="#65747b", linestyle="--", linewidth=0.8)
        ax.axvspan(24.5, 34.5, color="#e9bb65", alpha=0.10)
        ax.set_xlim(1, 49)
    else:
        ax.set_xlim(35, 49)
    ax.set_xlabel("Time step")


def monitor_figures(
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    features: pd.DataFrame,
    rolling: pd.DataFrame,
    reference: dict,
    output: Path,
) -> None:
    """Write three concise scientific figures, using fixed representative/model choices."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    candidate = reference["validation_selected_variant"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, metric, title in zip(
        axes.flat,
        [
            "f1",
            "average_precision",
            "alert_rate_all",
            "mean_score_all",
            "label_coverage",
            "observed_illicit_rate_labeled",
        ],
        [
            "Illicit F1 · frozen thresholds",
            "AP (PR-AUC) · known labels",
            "Alerts / all transactions",
            "Mean score · all transactions",
            "Known labels / all transactions",
            "Illicit / known labels",
        ],
    ):
        for variant, color in COLORS.items():
            group = summary[summary.variant == variant]
            if (
                metric in {"label_coverage", "observed_illicit_rate_labeled"}
                and variant != candidate
            ):
                continue
            x, mean, std = (
                group[k].to_numpy(float) for k in ("time_step", f"{metric}_mean", f"{metric}_std")
            )
            ax.plot(x, mean, color=color, label=NAMES[variant], linewidth=1.8)
            ax.fill_between(
                x, np.maximum(0, mean - std), np.minimum(1, mean + std), color=color, alpha=0.12
            )
        _timeline(ax)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Future-period reliability · mean ± sample std over five training seeds\n"
        "Dotted line: reported event at step 43; labels are retrospective",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures / "performance_over_time.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, metric, title in zip(
        axes[:2],
        ["psi", "ks_statistic"],
        [
            "Score PSI vs training · fixed train quantile bins",
            "Score KS distance vs training · descriptive effect size",
        ],
    ):
        aggregate = scores.groupby("time_step")[metric].agg(["mean", "std"])
        ax.plot(aggregate.index, aggregate["mean"], color="#007f8b")
        ax.fill_between(
            aggregate.index,
            np.maximum(0, aggregate["mean"] - aggregate["std"]),
            aggregate["mean"] + aggregate["std"],
            color="#007f8b",
            alpha=0.15,
        )
        _timeline(ax, full=True)
        ax.set_title(title)
    axes[0].axhline(
        reference["config"]["drift"]["psi_warning"],
        color="#d28c19",
        linestyle="--",
        label="Heuristic PSI warning",
    )
    axes[0].legend(fontsize=8)
    selected = rolling[rolling.variant == candidate]
    agg = selected.groupby("time_step")[["f1", "f1_floor"]].mean()
    axes[2].plot(agg.index, agg.f1, color="#007f8b", label="Pooled F1, mean over seeds")
    axes[2].plot(
        agg.index,
        agg.f1_floor,
        color="#d28c19",
        linestyle="--",
        label="Validation-derived F1 floor",
    )
    eligible = selected.groupby("time_step").support_sufficient.all()
    for step in eligible[~eligible].index:
        axes[2].axvspan(step - 0.4, step + 0.4, color="gray", alpha=0.12)
    axes[2].set(
        title="Performance warning · gray means incomplete window or insufficient support",
        ylim=(0, 1),
    )
    axes[2].legend(fontsize=8)
    _timeline(axes[2])
    fig.suptitle(
        "Selected XGBoost + structure · training reference is in-sample\n"
        "Warnings investigate risk; no automatic retraining or promotion",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures / "drift_and_warnings.png", dpi=160)
    plt.close(fig)

    table = features.pivot(index="feature", columns="time_step", values="psi").reindex(
        reference["top_feature_names"]
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    maximum = min(3.0, float(table.max().max()))
    heat = ax.imshow(table.to_numpy(float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=maximum)
    ax.set_yticks(range(len(table)), table.index)
    ticks = [1, 12, 24, 25, 34, 35, 43, 49]
    ax.set_xticks([x - 1 for x in ticks], ticks)
    for boundary in (23.5, 33.5):
        ax.axvline(boundary, color="#263b4a", linestyle="--", linewidth=1)
    ax.axvline(42, color="#263b4a", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Time step · train 1–24 | validation 25–34 | test 35–49",
        title=(
            "Top 10 features by frozen seed-0 training gain · "
            "PSI against all-node training reference"
        ),
    )
    fig.colorbar(heat, ax=ax, label=f"PSI (color capped at {maximum:.2f}; full values in CSV)")
    fig.tight_layout()
    fig.savefig(figures / "feature_drift.png", dpi=160)
    plt.close(fig)


def build_monitor_report(
    reference: dict,
    step_rows: list[dict],
    rolling_rows: list[dict],
    period_rows: list[dict],
    score_rows: list[dict],
    feature_rows: list[dict],
    output: Path,
) -> dict[str, Any]:
    """Save exact per-seed CSVs, JSON, summaries, recommendations and measured findings."""
    step, rolling, periods = map(pd.DataFrame, (step_rows, rolling_rows, period_rows))
    scores, features = map(pd.DataFrame, (score_rows, feature_rows))
    summary = summarize(step, ["variant", "time_step"], QUALITY)
    period_summary = summarize(periods, ["variant", "period"], QUALITY)
    warnings = warning_actions(scores, rolling, reference)
    frames = {
        "per_step_metrics": step,
        "per_step_summary": summary,
        "rolling_metrics": rolling,
        "regime_per_seed": periods,
        "regime_summary": period_summary,
        "score_drift": scores,
        "feature_drift": features,
        "warnings": warnings,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    write_json(
        output / "per_step_metrics.json",
        json.loads(step.to_json(orient="records", double_precision=15)),
    )
    candidate = reference["validation_selected_variant"]
    selected = step[step.variant == candidate]
    selected_periods = period_summary[period_summary.variant == candidate]
    representative = reference["config"]["experiment"]["representative_seed"]
    actions = warnings[warnings.seed == representative]

    def first(col: str) -> int | None:
        return int(actions.loc[actions[col], "time_step"].min()) if actions[col].any() else None

    train_val = scores[scores.split == "validation"]
    result = {
        "status": "completed",
        "validation_selected_variant": candidate,
        "primary_variants": int(step.variant.nunique()),
        "model_seed_combinations": int(step.groupby(["variant", "seed"]).ngroups),
        "per_step_metric_rows": len(step),
        "score_drift_rows": len(scores),
        "feature_drift_rows": len(features),
        "reference_training_nodes": reference["reference_nodes"],
        "training_score_reference_is_in_sample": True,
        "top_feature_names": reference["top_feature_names"],
        "representative_first_score_psi_warning": first("score_drift_warning"),
        "representative_first_persistent_score_drift": first("persistent_score_drift"),
        "representative_first_performance_warning": first("performance_warning"),
        "validation_score_psi_warning_fraction": float(
            (train_val.psi > reference["config"]["drift"]["psi_warning"]).mean()
        ),
        "selected_candidate_periods": json.loads(
            selected_periods.to_json(orient="records", double_precision=15)
        ),
        "minimum_test_step_illicit_support": int(selected.support_illicit.min()),
        "models_fitted": False,
        "decision_thresholds_changed": False,
        "shutdown_causality_established": False,
        "prospective_warning_detection_validated": False,
        "automatic_retraining_performed": False,
    }
    write_json(output / "monitoring_summary.json", result)
    monitor_figures(summary, scores, features, rolling, reference, output)
    table = "| Model | Period | AP (PR-AUC) | Illicit F1 |\n|---|---|---:|---:|\n"
    for row in period_summary.to_dict("records"):
        if row["variant"] in COLORS:
            table += (
                f"| {NAMES[row['variant']]} | {row['period']} | "
                f"{row['average_precision_mean']:.4f} ± {row['average_precision_std']:.4f} | "
                f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f} |\n"
            )
    counts = periods[(periods.variant == candidate) & (periods.seed == representative)]
    support_text = "; ".join(
        f"{r['period']}: {r['support_illicit']} illicit / {r['support_labeled']} known labels"
        for r in counts.to_dict("records")
    )
    text = f"""# Temporal reliability and drift monitoring

The model and each seed's decision threshold remain frozen. The selected candidate is **XGBoost
+ structure**, chosen from validation before Stage 5. This is retrospective monitoring of the
already-open test period, not a live alert evaluation.

## Measured performance around the reported event

{table}
Periods are pooled transaction cohorts, not averages of time-step metrics. Mean ± sample std
covers seeds 0–4, not independent datasets. Support: {support_text}. Changing prevalence and
sparse labels affect AP and uncertainty.

![Performance and label coverage](figures/performance_over_time.png)

## Drift and warning policy

All-node score and feature references use steps 1–24 only, including unknown nodes. Training
scores are in-sample predictions from the frozen models, so train-to-holdout differences mix
fitting optimism with temporal shift. Validation is shown as an out-of-time sanity check. PSI
uses {reference["config"]["drift"]["quantile_bins"]} requested training quantile bins, collapsed
ties, open outer bins, and proportion flooring at
{reference["config"]["drift"]["probability_floor"]:g} followed by renormalization. Constant
references receive three bins. Exact cuts/counts are frozen in `frozen_reference.json`.

The PSI > {reference["config"]["drift"]["psi_warning"]} warning is a heuristic, not a universal
threshold. {result["validation_score_psi_warning_fraction"]:.1%} of candidate validation
step/seed comparisons already exceed it. Seed {representative}'s first test score warning is
step **{first("score_drift_warning")}**, and its first persistent warning is
**{first("persistent_score_drift")}**. These are retrospective rule outputs, not
detection-latency estimates.

The performance warning pools the latest
{reference["config"]["performance"]["rolling_steps"]} test steps, requires at least
{reference["config"]["performance"]["minimum_known_labels"]} known and
{reference["config"]["performance"]["minimum_illicit_labels"]} illicit labels, and compares F1
with {reference["config"]["performance"]["f1_floor_fraction_of_validation"]:.0%} of that
seed's saved validation F1. Incomplete windows
do not warn. Seed {representative}'s first eligible performance warning is
**{first("performance_warning")}**. The CSV records each seed's exact floor and support.

![Score drift and pooled performance warnings](figures/drift_and_warnings.png)

KS distance is a descriptive effect size. P-values are retained as diagnostics only:
transactions share graph structure, features/scores have ties, the reference model was fitted on
training labels, and repeated comparisons are not independent. No p-value triggers action. Drift
alone does not establish lower detection quality.

## Top 10 feature drift

Features are ranked once by the frozen representative model's training gain, with lexical name
ties. No held-out importance, labels or drift magnitude chooses them. Drift uses all nodes and
original released feature values, with no newly fitted scaler. Anonymous feature names limit
business interpretation. If time appears, its deterministic advance is expected and must not be
treated as a surprise failure.

![Training-selected feature drift](figures/feature_drift.png)

## Investigation and retraining recommendation

1. A single PSI breach prompts input/pipeline and label-coverage checks. Two consecutive
breaches prompt investigation of persistent shift and targeted labeling.
2. A supported pooled-F1 breach prompts a retraining review using recent, adjudicated labels and
their actual availability times. Supervised warnings here use retrospectively available labels;
their online timing is unknown.
3. Train a challenger only after that review. Use a new chronological development window, choose
its threshold on validation under the real review budget, and test on a later untouched cohort.
Keep the existing candidate until the challenger passes quality/workload checks; no automatic
promotion is implemented.

This stage does not retrain a model or claim that retraining repairs the shift. The original
study associates step 43 with a dark-market shutdown and reports degradation even with its
repeated random-forest retraining. Our fixed periods investigate that reported boundary; this
dataset does not supply a causal counterfactual or identifiable event timestamps.

## Evidence and limits

`per_step_metrics.csv` / `.json` record known-label AP/F1, precision/recall, confusion counts,
all-node alert rate and mean score, label coverage and observed illicit prevalence. Unknown
labels never become false positives. `rolling_metrics.csv`, `score_drift.csv`,
`feature_drift.csv`, `warnings.csv` and the pooled `regime_*.csv` tables preserve denominators
and five-seed evidence. Original Stage 5 test scores are reused unchanged for all seven primary
variants; only the selected candidate receives train-referenced score/feature drift.

Completed-snapshot graph and forward-neighbor input availability remain retrospective. The
release lacks real label delays, entities, feature semantics and real-time deployment evidence.
Five training seeds quantify fitting variation, not temporal confidence intervals. These
warnings need prospective calibration on future adjudicated cohorts. README/About presentation
remains deferred for the next review.

Sources: [Weber et al., 2019, section 4, The Dark Market
Shutdown](https://arxiv.org/pdf/1908.02591); [SciPy two-sample KS
documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
"""
    (output / "REPORT.md").write_text(text)
    return result

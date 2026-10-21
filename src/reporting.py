"""Build validation-only tables from logged runs, with no invented metrics."""

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

METRICS = (
    "average_precision",
    "pr_auc_trapezoid",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "alert_rate_labeled",
    "mean_score_labeled",
    "threshold",
)


def run_rows(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten evidence logs without adding unmeasured model outputs."""
    rows = []
    for run in runs:
        rows.append(
            {
                **{
                    key: run[key]
                    for key in (
                        "run_id",
                        "model",
                        "feature_group",
                        "include_time",
                        "n_features",
                        "seed",
                        "candidate_index",
                        "fit_seconds",
                    )
                },
                **{name: run["validation_metrics"][name] for name in METRICS},
            }
        )
    return pd.DataFrame(rows)


def aggregate_runs(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Mean and sample std over distinct seeds with the same selected candidate."""
    rows = []
    frame = run_rows(runs)
    for keys, group in frame.groupby(["model", "feature_group", "include_time", "n_features"]):
        if group.seed.duplicated().any() or group.candidate_index.nunique() != 1:
            raise ValueError("Aggregation requires unique seeds and a single frozen candidate.")
        row = dict(zip(("model", "feature_group", "include_time", "n_features"), keys))
        row.update(n_seeds=len(group), candidate_index=int(group.candidate_index.iloc[0]))
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "feature_group", "include_time"])


def markdown_table(frame: pd.DataFrame) -> str:
    """Format only measured validation statistics; single-run std remains unavailable."""
    text = "| Model | Inputs | Features | Seeds | AP (PR-AUC) | Precision | Recall | F1 |\n"
    text += "|---|---|---:|---:|---:|---:|---:|---:|\n"
    for row in frame.to_dict("records"):
        metrics = []
        for key in ("average_precision", "precision", "recall", "f1"):
            mean, std = row[f"{key}_mean"], row[f"{key}_std"]
            metrics.append(f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f} (std N/A)")
        model = "LogReg" if row["model"] == "logreg" else "XGBoost"
        text += (
            f"| {model} | {row['feature_group']} | {row['n_features']} | {row['n_seeds']} | "
            + " | ".join(metrics)
            + " |\n"
        )
    return text


def build_stage2_report(
    selected: list[dict[str, Any]],
    tuning: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Report held-out validation only, and a predeclared seed-0 diagnostic plot."""
    summary = aggregate_runs(selected)
    primary_time = config["experiment"]["primary_include_time"]
    primary = summary[summary.include_time == primary_time]
    ablation = summary[summary.include_time != primary_time]
    run_rows(selected).to_csv(output / "validation_per_seed.csv", index=False)
    run_rows(tuning).to_csv(output / "tuning_candidates.csv", index=False)
    summary.to_csv(output / "validation_summary.csv", index=False)
    primary.to_csv(output / "validation_primary.csv", index=False)
    ablation.to_csv(output / "validation_time_ablation.csv", index=False)
    primary_metric = config["experiment"]["primary_metric"]
    best = primary.sort_values(f"{primary_metric}_mean", ascending=False, kind="stable").iloc[0]
    report = {
        "evaluation_split": "validation",
        "time_steps": [25, 34],
        "test_evaluated": False,
        "primary_metric": primary_metric,
        "std_definition": "sample standard deviation across seeds, ddof=1",
        "primary_include_time": primary_time,
        "selected_run_count": len(selected),
        "tuning_run_count": len(tuning),
        "best_primary_tabular_variant": {
            "model": str(best.model),
            "feature_group": str(best.feature_group),
            "include_time": bool(best.include_time),
            "n_features": int(best.n_features),
            "validation_mean_primary_metric": float(best[f"{primary_metric}_mean"]),
            "candidate_index": int(best.candidate_index),
        },
        "interpretation": (
            "Validation was used for candidate and threshold selection. These validation metrics "
            "are optimistic selection diagnostics, not final test estimates. Seed std measures "
            "training variability on the same split, not uncertainty across future periods."
        ),
    }
    write_json(output / "selection.json", report)
    tuning_seed = config["experiment"]["tuning_seed"]
    representative_seed = config["experiment"]["representative_seed"]
    actual_seeds = sorted({run["seed"] for run in selected})
    candidate_counts = {
        name: len(specification["candidates"]) for name, specification in config["models"].items()
    }
    representatives = [
        run
        for run in selected
        if run["include_time"] == primary_time and run["seed"] == representative_seed
    ]
    diagnostic_note = (
        "![Validation PR and threshold diagnostics]"
        "(figures/validation_diagnostics.png)\n\n"
        f"The diagnostic uses the configured representative seed {representative_seed}, "
        "never the best-performing seed. "
        if representatives
        else "The representative seed was not requested; no diagnostic is generated. "
    )
    lines = (
        "# Stage 2 — tabular baseline results\n\n"
        "**Validation only: steps 25–34. Test performance is not evaluated.**\n\n"
        "Mean ± sample std across the listed seeds; no accuracy headline. Each seed's "
        "precision/recall/F1 uses its own validation-selected maximum-F1 threshold. "
        "Those thresholds are recorded and frozen for later test use.\n\n"
        "## Primary feature convention\n\n"
        + markdown_table(primary)
        + "\n## Time-feature ablation\n\n"
        + markdown_table(ablation)
        + "\n## Protocol and interpretation\n\n"
        f"Predeclared candidate counts per feature variant: {candidate_counts}. "
        f"Candidates are compared on fixed tuning seed {tuning_seed} using validation PR "
        f"ranking performance. Winners are frozen and replicated on seeds {actual_seeds}; "
        "the tuning-seed fit is reused when requested. No best-seed "
        "selection, test-based tuning, random split, SMOTE, or train+validation refit occurs. "
        "Exact candidate counts/seeds come from the saved configuration; smaller CLI runs "
        "may intentionally use fewer candidates or seeds.\n\n"
        "Average precision and trapezoidal PR-AUC are separate columns in the CSV files. "
        "Class weights and XGBoost scale_pos_weight use known training labels only. "
        "LogReg's StandardScaler fits known training rows only; XGBoost's histogram bins "
        "are constructed without an evaluation set. Weighted outputs are uncalibrated scores.\n\n"
        "LogReg uses deterministic LBFGS, so repeating its seed need not produce variation. "
        "XGBoost's row/column subsampling provides seed-dependent variation. A small std "
        "does not establish robustness to temporal shift. Validation has been used both "
        "for model selection and threshold optimization, so these are selection-biased "
        "development estimates. Final test comparison remains Stage 5.\n\n"
        "The all-feature baseline includes the supplied neighbor aggregates; it already "
        "has graph-derived information. This stage cannot establish whether additional "
        "graph features or GNNs improve it.\n\n"
        + diagnostic_note
        + "Labeled-only alert rates are not all-node analyst workloads. Alert-budget "
        "evaluation and operational denominators remain Stage 5 decisions.\n"
    )
    (output / "REPORT.md").write_text(lines, encoding="utf-8")
    if representatives:
        figure_dir = output / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        evidence = []
        for run in representatives:
            prediction = np.load(run["artifacts"]["validation_predictions"]["path"])
            y, score = prediction["label"], prediction["score"]
            p, r, thresholds = precision_recall_curve(y, score)
            label = f"{run['model']} {run['feature_group']} ({run['n_features']})"
            axes[0].plot(r, p, label=label)
            f1 = np.divide(
                2 * p[:-1] * r[:-1],
                p[:-1] + r[:-1],
                out=np.zeros_like(thresholds),
                where=(p[:-1] + r[:-1]) != 0,
            )
            axes[1].plot(thresholds, f1, label=label)
            axes[1].scatter(run["threshold"]["value"], run["validation_metrics"]["f1"], s=24)
            evidence.append(
                pd.DataFrame(
                    {
                        "run_id": run["run_id"],
                        "tx_id": prediction["tx_id"],
                        "time_step": prediction["time_step"],
                        "label": y,
                        "score": score,
                        "frozen_threshold": run["threshold"]["value"],
                    }
                )
            )
        axes[0].axhline(float(y.mean()), color="gray", linestyle="--", label="Labeled prevalence")
        axes[0].set(
            xlabel="Illicit recall", ylabel="Illicit precision", title="Validation PR curves"
        )
        axes[1].set(
            xlabel="Decision threshold",
            ylabel="Illicit F1",
            title="Validation threshold sensitivity",
        )
        for ax in axes:
            ax.legend(fontsize=8)
            ax.set(xlim=(0, 1), ylim=(0, 1.03))
        fig.suptitle(f"Stage 2 · validation only · representative seed {representative_seed}")
        fig.tight_layout()
        fig.savefig(figure_dir / "validation_diagnostics.png", dpi=150)
        plt.close(fig)
        pd.concat(evidence, ignore_index=True).to_csv(
            output / "representative_validation_scores.csv.gz", index=False, compression="gzip"
        )
    else:
        # A single-seed CLI run must not display an older run's diagnostic.
        (output / "figures/validation_diagnostics.png").unlink(missing_ok=True)
        (output / "representative_validation_scores.csv.gz").unlink(missing_ok=True)
    return report

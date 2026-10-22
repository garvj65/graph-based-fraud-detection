"""Generate measured monitoring interpretation and final interview findings from evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.atomic_artifacts import write_json  # noqa: E402
from src.data import sha256_file  # noqa: E402
from src.inference import verify_artifact  # noqa: E402


def summarize(output: Path, docs: Path) -> None:
    """Explain the measured outcome without fitting or selecting on test evidence."""
    reference = json.loads((output / "frozen_reference.json").read_text())
    audit = json.loads((output / "independent_validation.json").read_text())
    assert audit["status"] == "passed" and audit["batch_id"] == reference["batch_id"]
    candidate = reference["validation_selected_variant"]
    periods = pd.read_csv(output / "regime_summary.csv", float_precision="round_trip")
    selected = periods[periods.variant == candidate].set_index("period")
    before, after = selected.loc["before_event"], selected.loc["event_and_after"]
    feature_rows = []
    with np.load(verify_artifact(reference["training_reference"])) as archive:
        for column, name in enumerate(archive["feature_names"]):
            values = archive["feature_values"][:, column]
            rho = (
                float(spearmanr(archive["time_step"], values).statistic) if np.ptp(values) else None
            )
            index = reference["top_feature_indices"][column]
            feature_rows.append(
                {
                    "feature": str(name),
                    "training_gain_rank": column + 1,
                    "training_gain_importance": reference["training_gain_importance"][index],
                    "training_spearman_with_time": rho,
                    "training_reference_nodes": reference["reference_nodes"],
                    "psi_actual_bins": reference["feature_histograms"][str(name)]["actual_bins"],
                }
            )
    feature_diagnostics = pd.DataFrame(feature_rows)
    feature_diagnostics.to_csv(output / "training_feature_diagnostics.csv", index=False)
    temporal = feature_diagnostics[feature_diagnostics.training_spearman_with_time.abs() >= 0.95]
    drift = pd.read_csv(output / "score_drift.csv")
    threshold = reference["config"]["drift"]["psi_warning"]
    drift_summary = []
    for split, group in drift.groupby("split", sort=False):
        drift_summary.append(
            {
                "split": split,
                "step_seed_comparisons": len(group),
                "psi_warning_fraction": float((group.psi > threshold).mean()),
                "psi_minimum": float(group.psi.min()),
                "psi_median": float(group.psi.median()),
                "psi_maximum": float(group.psi.max()),
            }
        )
    pd.DataFrame(drift_summary).to_csv(output / "drift_warning_calibration.csv", index=False)
    warnings = pd.read_csv(output / "warnings.csv")
    warning_rows = []
    for seed, group in warnings.groupby("seed"):
        row = {"seed": int(seed)}
        for field in ("score_drift_warning", "persistent_score_drift", "performance_warning"):
            positive = group.loc[group[field], "time_step"]
            row[f"first_{field}_step"] = int(positive.min()) if len(positive) else None
        warning_rows.append(row)
    pd.DataFrame(warning_rows).to_csv(output / "warning_timing_by_seed.csv", index=False)
    first_performance_steps = sorted(
        {
            row["first_performance_warning_step"]
            for row in warning_rows
            if row["first_performance_warning_step"] is not None
        }
    )
    timing = (
        f"step {first_performance_steps[0]} for all {len(warning_rows)} seeds"
        if len(first_performance_steps) == 1
        and all(row["first_performance_warning_step"] is not None for row in warning_rows)
        else f"steps {first_performance_steps}, depending on seed"
    )
    validation_fraction = next(
        r["psi_warning_fraction"] for r in drift_summary if r["split"] == "validation"
    )
    proxy_text = (
        ", ".join(
            f"{r['feature']} (rho {r['training_spearman_with_time']:.3f})"
            for r in temporal.to_dict("records")
        )
        or "none at |rho| ≥ 0.95"
    )
    section = f"""
## What these findings mean

The selected candidate's pooled AP falls from **{before.average_precision_mean:.4f} ±
{before.average_precision_std:.4f}** in steps 35–42 to **{after.average_precision_mean:.4f} ±
{after.average_precision_std:.4f}** in steps 43–49. F1 falls from **{before.f1_mean:.4f} ±
{before.f1_std:.4f}** to **{after.f1_mean:.4f} ± {after.f1_std:.4f}**. Post-event recall is
**{after.recall_mean:.1%}**, so the frozen policy misses approximately
**{1 - after.recall_mean:.1%}** of known illicit transactions in that period.
Its all-node alert rate remains **{after.alert_rate_all_mean:.1%}**: investigation workload
continues while detection quality is poor. This is why a strong pooled test score does not
establish reliable operation over time.

Observed illicit prevalence among known labels changes from
{before.observed_illicit_rate_labeled_mean:.2%} to {after.observed_illicit_rate_labeled_mean:.2%}.
That changes AP's baseline. ROC-AUC also drops from {before.roc_auc_mean:.3f} to
{after.roc_auc_mean:.3f}, and recall falls sharply, providing evidence beyond AP alone.
These are historical associations, not proof that the shutdown caused the change.

The PSI rule is already active in **{validation_fraction:.0%}** of validation step/seed
comparisons. It indicates distribution differences but does not isolate the step-43 failure.
The pooled-F1 rule first warns at {timing} in this recorded experiment.
Its three-step window initially dilutes the sharp step-43 decline; actual live warning timing
cannot be established without label-availability timestamps. The rule is an investigation and
retraining-review proposal, not an automatic retraining policy.

Training-only diagnostics find strongly time-associated features among the top ten:
**{proxy_text}**. Their drift is already substantial within training and during validation.
This explains why high feature PSI alone is not a shutdown-specific signal. Their anonymous
names do not establish their semantics. The earlier no-time ablation removes the explicit time
column only; it does not establish removal of implicit temporal information. A future follow-up
could test temporal-proxy ablations using a new chronological development/evaluation protocol.
No columns or models are changed after observing this test result.

`training_feature_diagnostics.csv`, `drift_warning_calibration.csv` and
`warning_timing_by_seed.csv` preserve these checks. No feature is selected by this diagnostic;
the monitored top ten remain those chosen by training-tree gain.
"""
    report = output / "REPORT.md"
    marker = "<!-- measured-monitoring-interpretation -->"
    base_report = report.read_text().split(marker)[0].rstrip()
    base_report = base_report.replace("| before_event |", "| Steps 35–42 |")
    base_report = base_report.replace("| event_and_after |", "| Steps 43–49 |")
    report.write_text(base_report + "\n\n" + marker + section)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "stage6_findings.md").write_text(
        "# Monitoring interview findings\n\n"
        + "\n".join(
            [
                f"- The frozen candidate's F1 changes from {before.f1_mean:.4f} before step 43 "
                f"to {after.f1_mean:.4f} afterward; an aggregate test score concealed "
                "a severe reliability failure.",
                f"- Post-event recall is {after.recall_mean:.1%}, while the all-node alert "
                f"rate remains {after.alert_rate_all_mean:.1%}; the review queue keeps "
                "consuming effort while missing known illicit activity.",
                f"- PSI > {threshold} warns during {validation_fraction:.0%} of validation "
                "step/seed comparisons, so generic drift thresholds need calibration and "
                "cannot substitute for label-based quality monitoring.",
                f"- The pooled F1 policy first warns at {timing}; this is retrospective "
                "because true label delays are unavailable.",
                "- Retraining should be reviewed with recent adjudicated labels, followed "
                "by a new temporal challenger evaluation. Neither drift nor this historical "
                "event guarantees a successful retrain.",
            ]
        )
        + "\n"
    )
    stage5_path = Path(reference["stage5_manifest"]["path"]).parent
    comparison = pd.read_csv(stage5_path / "test_comparison.csv").set_index("variant")
    tree = comparison.loc["s2_xgboost_all_time"]
    graph = comparison.loc[candidate]
    sage = comparison.loc["s4_graphsage_all"]
    budget = pd.read_csv(stage5_path / "budget_summary.csv")
    queue = budget[(budget.variant == candidate) & (budget.view == "all_transactions")].iloc[0]
    (docs / "project_findings.md").write_text(
        "# Five findings to explain in an interview\n\n"
        + "\n".join(
            [
                "- I built a chronological fraud benchmark with train 1–24, validation "
                "25–34 and test 35–49. Unknown transactions participate in the graph but "
                "never become negative labels or enter supervised loss/quality metrics.",
                "- A strong all-feature XGBoost baseline achieved test AP "
                f"{tree.average_precision_mean:.4f} ± {tree.average_precision_std:.4f}; "
                f"GraphSAGE reached {sage.average_precision_mean:.4f} ± "
                f"{sage.average_precision_std:.4f}. The tested GNNs did not beat the "
                "tabular baseline over the whole future period.",
                f"- Added structural features reached AP {graph.average_precision_mean:.4f} "
                f"± {graph.average_precision_std:.4f}, without improving mean F1. That "
                "small ranking gain does not establish that extra graph complexity "
                "pays for itself.",
                "- At 100 reviews per step, the selected model's all-node queue found "
                f"{queue.known_illicit_alerts_mean:.0f} ± {queue.known_illicit_alerts_std:.0f} "
                f"known illicit transactions over {queue.alerts_selected_mean:.0f} reviews. "
                f"{(queue.unknown_alerts_mean / queue.alerts_selected_mean):.1%} of alerts "
                "were unlabeled, so overall operational precision remains unknown.",
                f"- Monitoring exposed F1 dropping from {before.f1_mean:.3f} in steps "
                f"35–42 to {after.f1_mean:.3f} in steps 43–49. PSI alone was not selective; "
                "I proposed supported performance warnings, label investigation and a "
                "new temporal challenger evaluation before any retraining promotion.",
            ]
        )
        + "\n"
    )
    write_json(
        output / "interpretation_audit.json",
        {
            "batch_id": reference["batch_id"],
            "source": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
            "training_correlations_use_steps": [1, 24],
            "heldout_feature_ranking_used": False,
            "diagnostics_changed_models_or_policies": False,
            "features_with_abs_training_time_correlation_at_least_0_95": temporal.feature.tolist(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage6"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    summarize(args.output_dir, args.docs_dir)

"""Write measured conclusions and interview findings from verified Stage 5 evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    """Summarize only verified outputs; never alter models, thresholds or the README."""
    root = Path("results/stage5")
    batch = json.loads((root / "latest_run.json").read_text())
    audit = json.loads((root / "independent_validation.json").read_text())
    assert audit["status"] == "passed" and audit["batch_id"] == batch["batch_id"]
    primary = pd.read_csv(root / "test_comparison.csv", float_precision="round_trip").set_index(
        "variant"
    )
    budget = pd.read_csv(root / "budget_summary.csv", float_precision="round_trip")
    paired = pd.read_csv(root / "paired_test_lift.csv", float_precision="round_trip")
    plan = json.loads((root / "frozen_protocol.json").read_text())
    selected = plan["validation_selected_variant"]
    row = primary.loc[selected]
    control = primary.loc["s2_xgboost_all_time"]
    graph = primary.loc["s3_xgboost_all_structure"]
    sage, gcn = primary.loc["s4_graphsage_all"], primary.loc["s4_gcn_all"]
    delta = paired.loc[paired.variant == "s3_xgboost_all_structure", "ap_delta_vs_xgboost_all"]
    queue = budget[(budget.variant == selected) & (budget.view == "all_transactions")].iloc[0]
    benchmark = budget[(budget.variant == selected) & (budget.view == "labeled_only")].iloc[0]
    unknown_share = 1 - queue.known_label_coverage_mean
    validation_ap = plan["primary_validation_ap_means"][selected]
    recap = (
        "## Measured findings\n\n"
        f"The validation-selected structural XGBoost candidate reached test AP "
        f"**{row.average_precision_mean:.4f} ± {row.average_precision_std:.4f}** and F1 "
        f"**{row.f1_mean:.4f} ± {row.f1_std:.4f}**. The all-feature XGBoost control reached "
        f"AP {control.average_precision_mean:.4f} ± {control.average_precision_std:.4f}. "
        f"The paired structural AP difference was {delta.mean():+.6f} ± {delta.std(ddof=1):.6f}, "
        f"positive in {int((delta > 0).sum())}/{len(delta)} seeds. Its mean F1 difference was "
        f"{graph.f1_mean - control.f1_mean:+.6f}; this is a small AP improvement, not a clear "
        "operating-point or business win.\n\n"
        f"The implemented GraphSAGE and GCN all-feature models reached test AP "
        f"{sage.average_precision_mean:.4f} ± {sage.average_precision_std:.4f} and "
        f"{gcn.average_precision_mean:.4f} ± {gcn.average_precision_std:.4f}, respectively. "
        "Neither beats the strong tabular baseline on this future-period comparison. "
        "This conclusion is limited to the declared architectures, direction policy "
        "and tuning budget.\n\n"
        f"At the declared per-step budget, the selected model's all-node queues contained "
        f"{queue.alerts_selected_mean:.0f} reviews across the test period and identified "
        f"{queue.known_illicit_alerts_mean:.1f} ± {queue.known_illicit_alerts_std:.1f} "
        "known illicit "
        f"transactions per seed. Approximately {unknown_share:.1%} of those alerts have unknown "
        "labels; their correctness is not measurable. The retrospective labeled-only "
        f"precision@k was {benchmark.precision_at_k_labeled_mean:.4f} ± "
        f"{benchmark.precision_at_k_labeled_std:.4f}. "
        "These two views answer different questions.\n\n"
        f"The selected model's validation AP was {validation_ap:.4f}; "
        f"its test AP was {row.average_precision_mean:.4f}. This supports investigating "
        "temporal reliability. Changed label prevalence also affects PR metrics; these "
        "aggregate results do not locate the decline or establish shutdown causality. "
        "That analysis remains Stage 6.\n\n"
        "All 100 frozen models were independently reloaded. Artifact recovery preserved "
        "the original metric table and protocol; its only source amendment changes artifact "
        "serialization, with unchanged evaluation logic verified by AST comparison.\n"
    )
    start, end = "<!-- stage5-findings-start -->", "<!-- stage5-findings-end -->"
    report = (root / "REPORT.md").read_text()
    block = start + "\n" + recap + end
    if start in report:
        before, remainder = report.split(start, 1)
        _, after = remainder.split(end, 1)
        report = before + block + after
    else:
        report += "\n" + block + "\n"
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    findings = [
        "I froze every model, preprocessing step, alert policy and threshold before evaluating "
        "steps 35–49. Unknown nodes stayed in graph inference and stayed out of "
        "classification metrics.",
        f"The all-feature XGBoost baseline reached test AP {control.average_precision_mean:.4f} ± "
        f"{control.average_precision_std:.4f}, compared with GraphSAGE "
        f"{sage.average_precision_mean:.4f} ± "
        f"{sage.average_precision_std:.4f} and GCN {gcn.average_precision_mean:.4f} ± "
        f"{gcn.average_precision_std:.4f}. These GNNs did not outperform the strong tabular model.",
        f"Structural features added paired AP {delta.mean():+.6f} ± {delta.std(ddof=1):.6f}, "
        "but did not improve mean F1. I would describe this as a small ranking gain, rather "
        "than claim additional graph complexity delivered a clear business benefit.",
        "A 100-alert-per-step all-transaction queue identified "
        f"{queue.known_illicit_alerts_mean:.0f} "
        f"known illicit transactions on average over {queue.alerts_selected_mean:.0f} reviews. "
        f"About {unknown_share:.1%} of alerts were unlabeled, "
        "so I cannot claim overall operational precision.",
        f"The selected model moved from validation AP {validation_ap:.4f} "
        f"to test AP {row.average_precision_mean:.4f}. "
        "The next task is to locate that deterioration "
        "with per-step metrics and drift checks, rather than assume a particular event caused it.",
    ]
    history = Path("docs/history")
    history.mkdir(parents=True, exist_ok=True)
    (history / "stage5_findings.md").write_text(
        "# Stage 5 interview findings\n\n"
        + "\n".join(f"- “{finding}”" for finding in findings)
        + "\n",
        encoding="utf-8",
    )
    print("Measured report and five interview findings updated; README/About unchanged.")


if __name__ == "__main__":
    main()

"""Render measured Stage 4 conclusions and historical findings from run evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

START, END = "<!-- stage4-measured-start -->", "<!-- stage4-measured-end -->"


def replace_block(text: str, block: str) -> str:
    """Replace an existing measured block or append it once."""
    if START in text:
        before, after = text.split(START, 1)
        _, after = after.split(END, 1)
        return before + block + after
    return text.rstrip() + "\n\n" + block + "\n"


def main() -> None:
    """Use completed validation evidence; preserve the current project README."""
    output = Path("results/stage4")
    batch = json.loads((output / "latest_run.json").read_text())
    assert batch["status"] == "completed" and batch["test_evaluated"] is False
    verification = json.loads((output / "independent_validation.json").read_text())
    assert verification["status"] == "passed" and verification["batch_id"] == batch["batch_id"]
    summary = pd.read_csv(output / "validation_summary.csv", float_precision="round_trip")
    comparison = pd.read_csv(output / "comparison_validation.csv", float_precision="round_trip")
    paired = pd.read_csv(output / "message_passing_lift.csv", float_precision="round_trip")
    primary = summary[summary.feature_group == "all"].set_index("model")
    trees = comparison[comparison.model == "xgboost"]
    best_tree = trees.loc[trees.average_precision_mean.idxmax()]
    best_gnn = primary.loc[primary.average_precision_mean.idxmax()]
    advantage = float(best_gnn.average_precision_mean - best_tree.average_precision_mean)
    conclusion = (
        "The best primary GNN mean validation AP exceeds the strongest tabular mean "
        "in this experiment. This is a development-set difference, not a final test win."
        if advantage > 0
        else "Neither primary GNN demonstrates a mean validation AP advantage over the "
        "strongest tabular result under this compute and tuning budget."
    )
    records = [
        json.loads((output / "runs" / f"{run_id}.json").read_text())
        for run_id in sorted(set(batch["selected_runs"] + batch["trial_runs"]))
    ]
    interrupted_count = len(list((output / "interrupted_runs").glob("*.json")))
    selected_ids = set(batch["selected_runs"])
    budget_rows = [
        {
            "run_id": run["run_id"],
            "model": run["model"],
            "feature_group": run["feature_group"],
            "seed": run["seed"],
            "candidate_index": run["candidate_index"],
            "selected_result": run["run_id"] in selected_ids,
            "epochs_run": run["epochs_run"],
            "best_epoch": run["best_epoch"],
            "stop_reason": run["stop_reason"],
            "reached_epoch_cap": run["epochs_run"] == batch["config"]["training"]["max_epochs"],
            "best_in_last_patience_window": run["best_epoch"]
            > (batch["config"]["training"]["max_epochs"] - batch["config"]["training"]["patience"]),
        }
        for run in records
    ]
    budget = pd.DataFrame(budget_rows)
    budget.to_csv(output / "budget_audit.csv", index=False)
    selected_budget = budget[budget.selected_result]
    cap_count = int(selected_budget.reached_epoch_cap.sum())
    recent_count = int(
        (selected_budget.reached_epoch_cap & selected_budget.best_in_last_patience_window).sum()
    )
    delta_text = "| Architecture | AP change, messages − no messages | Positive seeds |\n"
    delta_text += "|---|---:|---:|\n"
    for name, group in paired.groupby("model"):
        values = group.ap_delta_message_minus_no_message.to_numpy()
        std = f"{np.std(values, ddof=1):.5f}" if len(values) > 1 else "N/A"
        delta_text += (
            f"| {name} | {np.mean(values):+.5f} ± {std} | "
            f"{int((values > 0).sum())}/{len(values)} |\n"
        )
    measured = (
        f"{START}\n## Measured interpretation\n\n{conclusion}\n\n"
        f"The best primary GNN mean AP differs from the strongest tabular mean by "
        f"{advantage:+.5f}. Candidate and epoch selection used validation; these comparisons "
        "are descriptive, not significance tests.\n\n"
        + delta_text
        + "\nThe controls share all-feature hyperparameters but select their own validation "
        "checkpoint and threshold. Differences concern these implementations and do not "
        "isolate every architecture or optimization effect.\n\n"
        f"There were {len(records)} completed fits. Interrupted attempts excluded from metrics: "
        f"{interrupted_count}. Completed artifacts were preserved during resumption. "
        f"{cap_count}/{len(selected_budget)} selected fits reached the configured epoch cap; "
        f"{recent_count} of those had their best epoch in the final patience window. "
        "Convergence is therefore not established for every fit. The declared budget was "
        "not expanded after seeing the results. See `budget_audit.csv` for all candidates. "
        "Test performance and over-time reliability remain unevaluated.\n" + END
    )
    report_path = output / "REPORT.md"
    report_text = report_path.read_text().replace(
        "one seed has unavailable std.", "single-seed invocations have unavailable std."
    )
    report_path.write_text(replace_block(report_text, measured), encoding="utf-8")
    sage_delta = paired.loc[paired.model == "graphsage", "ap_delta_message_minus_no_message"]
    findings = [
        "I compared two-layer GCN and GraphSAGE against the same temporal tabular benchmark, "
        "keeping unknown nodes in messages and excluding them from loss and metrics.",
        f"On validation, GCN with all features reached AP "
        f"{primary.loc['gcn', 'average_precision_mean']:.5f} ± "
        f"{primary.loc['gcn', 'average_precision_std']:.5f}; GraphSAGE reached "
        f"{primary.loc['graphsage', 'average_precision_mean']:.5f} ± "
        f"{primary.loc['graphsage', 'average_precision_std']:.5f}. {conclusion}",
        f"GraphSAGE's message-passing AP lift over its no-message control was "
        f"{sage_delta.mean():+.5f} ± {sage_delta.std(ddof=1):.5f}, positive in "
        f"{int((sage_delta > 0).sum())}/{len(sage_delta)} seeds. Messages helped this neural "
        "model, while the stronger tabular baseline still performed better. Its supplied "
        "neighbor aggregates already contain graph-derived information.",
        f"I completed {len(records)} unique fits and reported {len(selected_ids)} selected seed "
        f"results. {cap_count} selected fits reached the epoch cap, so I report the compute "
        "limit rather than claiming the architectures were fully optimized.",
        "I reloaded every checkpoint and independently checked AP, alert thresholds and "
        "confusion counts. Validation was reused for development, so the future-period "
        "comparison and any claim about deterioration after step 43 still need their own runs.",
    ]
    history = Path("docs/history")
    history.mkdir(parents=True, exist_ok=True)
    (history / "stage4_findings.md").write_text(
        "# Stage 4 findings — interview explanation\n\n"
        "Validation only; final test and monitoring remain unevaluated.\n\n"
        + "\n".join(f"- “{line}”" for line in findings)
        + "\n",
        encoding="utf-8",
    )
    print(conclusion)


if __name__ == "__main__":
    main()

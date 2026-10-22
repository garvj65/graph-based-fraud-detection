"""Describe cross-runtime score replay without relaxing the original exact-score audit."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.atomic_artifacts import write_json  # noqa: E402
from src.data import environment_versions, load_csv, sha256_file  # noqa: E402
from src.evaluate import binary_metrics  # noqa: E402
from src.final_evaluate import make_cache  # noqa: E402
from src.inference import predict_frozen, verify_artifact  # noqa: E402


def audit(stage5: Path, output: Path) -> dict:
    """Reload every frozen artifact and report all differences, never replacing evidence."""
    batch = json.loads((stage5 / "latest_run.json").read_text())
    plan = json.loads(verify_artifact(batch["protocol"]).read_text())
    config = plan["config"]
    for name, digest in plan["input_files_sha256"].items():
        assert sha256_file(Path(config["data"]["raw_dir"]) / name) == digest
    data = load_csv(Path(config["data"]["raw_dir"]), require_full_dataset=True)
    cache = make_cache(data, config)
    rows = []
    for run_id in batch["evaluation_runs"]:
        record = json.loads((stage5 / "runs" / f"{run_id}.json").read_text())
        source = json.loads(verify_artifact(record["source_record"]).read_text())
        nodes, scores, _ = predict_frozen(
            cache.observed, (35, 49), source, cache, thread_limit=config["runtime"]["thread_limit"]
        )
        with np.load(verify_artifact(record["test_predictions"])) as saved:
            original, labels, ids, times = (
                saved[key] for key in ("score", "label", "tx_id", "time_step")
            )
        np.testing.assert_array_equal(ids, data.tx_ids[nodes])
        np.testing.assert_array_equal(labels, data.y[nodes])
        metrics = binary_metrics(labels, scores, threshold=record["frozen_threshold"])
        old = record["test_metrics"]
        ordered_changes = membership_changes = known_count_changes = 0
        for step in range(35, 50):
            positions = np.flatnonzero(times == step)
            k = config["alert_budget"]["per_time_step"]
            before = positions[np.lexsort((ids[positions], -original[positions]))[:k]]
            after = positions[np.lexsort((ids[positions], -scores[positions]))[:k]]
            ordered_changes += int(not np.array_equal(before, after))
            membership_changes += int(set(before) != set(after))
            known_count_changes += int((labels[before] == 1).sum() != (labels[after] == 1).sum())
        rows.append(
            {
                "variant": record["variant"],
                "seed": record["seed"],
                "model": record["model"],
                "bitwise_identical": bool(np.array_equal(original, scores)),
                "maximum_absolute_score_difference": float(np.abs(original - scores).max()),
                "changed_threshold_predictions_all_nodes": int(
                    (
                        (original >= record["frozen_threshold"])
                        != (scores >= record["frozen_threshold"])
                    ).sum()
                ),
                "confusion_identical": metrics["confusion_matrix"] == old["confusion_matrix"],
                "absolute_ap_difference": abs(
                    metrics["average_precision"] - old["average_precision"]
                ),
                "absolute_f1_difference": abs(metrics["f1"] - old["f1"]),
                "changed_ordered_top100_step_queues": ordered_changes,
                "changed_top100_membership_steps": membership_changes,
                "changed_known_illicit_alert_count_steps": known_count_changes,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "numerical_replay.csv", index=False)
    result = {
        "status": "bitwise_identical"
        if frame.bitwise_identical.all()
        else "numerical_portability_difference",
        "models_reloaded": len(frame),
        "bitwise_identical_models": int(frame.bitwise_identical.sum()),
        "maximum_absolute_score_difference": float(frame.maximum_absolute_score_difference.max()),
        "maximum_absolute_ap_difference": float(frame.absolute_ap_difference.max()),
        "changed_threshold_predictions_all_nodes": int(
            frame.changed_threshold_predictions_all_nodes.sum()
        ),
        "changed_ordered_top100_step_queues": int(frame.changed_ordered_top100_step_queues.sum()),
        "changed_top100_membership_steps": int(frame.changed_top100_membership_steps.sum()),
        "changed_known_illicit_alert_count_steps": int(
            frame.changed_known_illicit_alert_count_steps.sum()
        ),
        "versions": environment_versions(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_build": torch.__config__.show(),
        "script_sha256": sha256_file(Path(__file__)),
        "original_stage5_exact_audit_relaxed": False,
        "saved_predictions_or_metrics_replaced": False,
        "hardware_cause_proven": False,
        "interpretation": (
            "Descriptive portability check. Original saved scores remain canonical; "
            "differences are disclosed without overriding the strict audit."
        ),
    }
    write_json(output / "numerical_replay.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage5-dir", type=Path, default=Path("results/stage5"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_review"))
    args = parser.parse_args()
    print(json.dumps(audit(args.stage5_dir, args.output_dir), indent=2))

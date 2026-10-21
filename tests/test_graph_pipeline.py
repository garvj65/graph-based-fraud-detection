"""Tiny end-to-end graph experiment uses genuine fits on synthetic fixtures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.graph_train import run_graph_experiment, validate_graph_config
from src.train import run_experiment


def test_graph_experiment_logs_five_seeds_and_reproduces_control(
    raw_csvs: Path,
    tmp_path: Path,
) -> None:
    baseline = yaml.safe_load(Path("configs/stage2.yaml").read_text())
    baseline["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    baseline["output_dir"] = str(tmp_path / "stage2")
    baseline["artifact_dir"] = str(tmp_path / "baseline_artifacts")
    baseline["experiment"].update(feature_groups=["all"], include_time_options=[True])
    baseline["models"]["xgboost"]["candidates"] = [{"max_depth": 2}]
    baseline["models"]["xgboost"]["common"].update(n_estimators=5, n_jobs=1)
    run_experiment(baseline, models=["xgboost"])
    config = yaml.safe_load(Path("configs/stage3.yaml").read_text())
    config["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    config["output_dir"] = str(tmp_path / "stage3")
    config["artifact_dir"] = str(tmp_path / "graph_artifacts")
    config["experiment"].update(
        selection_record=str(tmp_path / "stage2/selection.json"),
        baseline_run_manifest=str(tmp_path / "stage2/latest_run.json"),
        fixed_candidate_index=0,
    )
    result = run_graph_experiment(config)
    assert result["status"] == "completed"
    assert len(result["selected_runs"]) == 20
    output = Path(config["output_dir"])
    summary = pd.read_csv(output / "validation_summary.csv")
    assert (summary.n_seeds == 5).all()
    assert set(summary.n_features) == {166, 172, 174, 6}
    checks = json.loads((output / "baseline_reproduction.json").read_text())
    for check in checks:
        for key, value in check.items():
            if key.endswith(("identical",)):
                assert value is True
    assert result["report"]["raw_train_label_neighbor_coverage_validation_nodes"] == 0
    assert not result["test_evaluated"]
    helpers = list((output / "helper_fit_logs").glob("*.json"))
    assert len(helpers) == 25  # 24 fits plus one explicitly logged cold-start fallback.
    for path in helpers:
        record = json.loads(path.read_text())
        assert record["fit_cutoff"] <= 24
        assert record["status"] == "completed"
        journal = (
            Path(config["artifact_dir"])
            / result["batch_id"]
            / "feature_build/score_models"
            / f"prefix_{record['fit_cutoff']:02d}.json"
        )
        assert json.loads(journal.read_text())["status"] == "completed"
    for run_id in result["selected_runs"]:
        record = json.loads((output / "runs" / f"{run_id}.json").read_text())
        assert record["stage"] == 3
        assert record["threshold"]["selection_split"] == "validation"
        with np.load(record["artifacts"]["validation_predictions"]["path"]) as scores:
            assert scores["time_step"].min() == 25
            assert scores["time_step"].max() == 34
    assert (output / "figures/graph_comparison.png").is_file()


@pytest.mark.parametrize("field", ["test", "snapshot_scope", "metric"])
def test_graph_config_rejects_protocol_changes(field: str) -> None:
    config = yaml.safe_load(Path("configs/stage3.yaml").read_text())
    if field == "test":
        config["experiment"]["evaluate_test"] = True
    elif field == "snapshot_scope":
        config["experiment"]["graph_availability"] = "full_future_graph"
    else:
        config["experiment"]["primary_metric"] = "accuracy"
    with pytest.raises(ValueError):
        validate_graph_config(config)

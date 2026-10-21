"""Small real fits on artificial fixtures validate logs and the CLI workflow."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluate import binary_metrics
from src.reporting import aggregate_runs
from src.train import run_experiment, validate_config


def test_stage2_runs_five_seeds_without_test_metrics(raw_csvs: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/stage2.yaml").read_text())
    config["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    config["output_dir"] = str(tmp_path / "results")
    config["artifact_dir"] = str(tmp_path / "artifacts")
    config["experiment"].update(
        feature_groups=["local"], include_time_options=[False], primary_include_time=False
    )
    config["models"]["logreg"]["candidates"] = [{"C": 1.0}]
    config["models"]["xgboost"]["candidates"] = [{"max_depth": 2}]
    config["models"]["xgboost"]["common"].update(n_estimators=5, n_jobs=1)
    result = run_experiment(config)
    assert result["status"] == "completed"
    assert result["test_evaluated"] is False
    assert len(result["selected_runs"]) == 10
    assert len(result["trial_runs"]) == 10  # Seed-0 tuning fits reused, not double counted.
    output = Path(config["output_dir"])
    summary = pd.read_csv(output / "validation_summary.csv")
    assert list(summary.n_seeds) == [5, 5]
    assert "test" not in " ".join(summary.columns)
    records = [
        json.loads((output / "runs" / f"{run_id}.json").read_text())
        for run_id in result["selected_runs"]
    ]
    for record in records:
        assert record["status"] == "completed"
        assert record["threshold"]["selection_split"] == "validation"
        assert record["test_evaluated"] is False
        assert "test_metrics" not in record
        path = record["artifacts"]["validation_predictions"]["path"]
        with np.load(path) as scores:
            recomputed = binary_metrics(
                scores["label"], scores["score"], threshold=record["threshold"]["value"]
            )
        assert recomputed == record["validation_metrics"]
    modified = copy.deepcopy(records)
    modified.append(records[0])
    with pytest.raises(ValueError, match="unique seeds"):
        aggregate_runs(modified)
    one = aggregate_runs(records[:1])
    assert one.iloc[0].average_precision_std is None


def test_config_cannot_enable_test_evaluation() -> None:
    config = yaml.safe_load(Path("configs/stage2.yaml").read_text())
    config["experiment"]["evaluate_test"] = True
    with pytest.raises(ValueError, match="does not evaluate test"):
        validate_config(config)


def test_invalid_config_writes_failure_record(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/stage2.yaml").read_text())
    config["output_dir"] = str(tmp_path / "results")
    config["experiment"]["threshold_objective"] = "test_f1"
    with pytest.raises(ValueError, match="policy"):
        run_experiment(config)
    logs = list((Path(config["output_dir"]) / "batches").glob("*.json"))
    assert len(logs) == 1
    assert json.loads(logs[0].read_text())["status"] == "failed"

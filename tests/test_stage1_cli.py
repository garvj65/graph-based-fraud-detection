"""CLI smoke and failure logging on explicitly artificial data."""

import json
import subprocess
import sys
from pathlib import Path

import yaml


def _config(raw_dir: Path, output: Path, path: Path) -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["data"]["raw_dir"] = str(raw_dir)
    config["data"]["require_full_dataset"] = False
    config["output_dir"] = str(output)
    path.write_text(yaml.safe_dump(config))


def test_cli_records_config_inputs_and_audit(raw_csvs: Path, tmp_path: Path) -> None:
    output, config_path = tmp_path / "report", tmp_path / "config.yaml"
    _config(raw_csvs, output, config_path)
    run = subprocess.run(
        [sys.executable, "-m", "src.data", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    record = json.loads((output / "latest_run.json").read_text())
    assert record["status"] == "completed"
    assert record["seed"] is None
    assert record["model_metrics"] is None
    assert record["config"]["experiment"]["seeds"] == [0, 1, 2, 3, 4]
    assert len(record["input_files"]) == 3
    assert all(len(entry["sha256"]) == 64 for entry in record["input_files"].values())
    assert record["audit"]["evaluation_edges_touching_train_labeled_nodes"] == 0
    assert record["audit"]["duplicate_directed_edges"] == 0
    assert record["audit"]["self_loops"] == 0
    assert (output / "figures/data_overview.png").stat().st_size > 0


def test_failed_load_leaves_a_failure_log(tmp_path: Path) -> None:
    output, config_path = tmp_path / "report", tmp_path / "config.yaml"
    _config(tmp_path / "missing", output, config_path)
    run = subprocess.run(
        [sys.executable, "-m", "src.data", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    records = list((output / "runs").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "failed"
    assert record["error"]["type"] == "FileNotFoundError"

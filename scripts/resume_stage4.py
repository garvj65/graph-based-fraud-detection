"""Resume an interrupted Stage 4 batch without changing its training protocol.

Completed runs are hash-checked and reused. An unfinished attempt is archived
and restarted with the same seed/config; no partial metric enters selection.
Independent CPU fits may run in separate workers, each with the original
thread limit. Batch metadata is written only by the parent process.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_csv, sha256_file, write_json  # noqa: E402
from src.gnn import prepare_gnn_data  # noqa: E402
from src.gnn_reporting import build_gnn_report  # noqa: E402
from src.gnn_train import fit_gnn_run, validate_gnn_config  # noqa: E402

_RAW = None
_VIEWS: dict[str, Any] = {}


def fit_task(arguments: dict[str, Any]) -> dict[str, Any]:
    """Load/cache immutable inputs per worker; fit one isolated run."""
    global _RAW
    config = arguments["config"]
    if _RAW is None:
        _RAW = load_csv(
            Path(config["data"]["raw_dir"]),
            require_full_dataset=config["data"]["require_full_dataset"],
        )
    group = arguments["group"]
    if group not in _VIEWS:
        _VIEWS[group] = prepare_gnn_data(
            _RAW,
            "all" if group == "all_no_messages" else group,
            config["graph"],
            no_messages=group == "all_no_messages",
        )
    return fit_gnn_run(_VIEWS[group], **arguments)


def resume(batch_path: Path, workers: int) -> dict[str, Any]:
    """Preserve every existing completed fit and reconstruct the declared selection loop."""
    batch = json.loads(batch_path.read_text())
    config, provenance = batch["config"], batch["provenance"]
    validate_gnn_config(config)
    if batch["status"] == "completed":
        return batch
    if config["runtime"]["device"] != "cpu" or workers not in {1, 2}:
        raise ValueError("Resume supports one or two independent CPU workers only.")
    for name, digest in provenance["source_files_sha256"].items():
        assert sha256_file(Path("src") / name) == digest, name
    for name, digest in provenance["input_files_sha256"].items():
        assert sha256_file(Path(config["data"]["raw_dir"]) / name) == digest, name
    for key, digest in provenance["reference_files_sha256"].items():
        assert sha256_file(Path(config["experiment"][key])) == digest, key
    output = Path(config["output_dir"])
    event = {
        "resumed_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "workers": workers,
        "per_worker_threads": config["runtime"]["thread_limit"],
        "settings_and_selection_policy_unchanged": True,
        "interrupted_attempts": [],
    }
    batch.setdefault("resume_events", []).append(event)
    batch["status"] = "running"
    write_json(batch_path, batch)

    def existing_or_arguments(model: str, group: str, index: int, seed: int, phase: str):
        run_id = f"{batch['batch_id']}_{model}_{group}_c{index}_s{seed}"
        path = output / "runs" / f"{run_id}.json"
        if path.exists():
            record = json.loads(path.read_text())
            assert record["config"] == config and record["provenance"] == provenance
            if record["status"] == "completed":
                for artifact in record["artifacts"].values():
                    assert sha256_file(Path(artifact["path"])) == artifact["sha256"]
                return record, None
            if record["status"] != "running":
                raise RuntimeError(f"Review failed run before resuming: {run_id}")
            archive = output / "interrupted_runs"
            archive.mkdir(exist_ok=True)
            attempt = 1
            while (archive / f"{run_id}_attempt_{attempt}.json").exists():
                attempt += 1
            archived_id = f"{run_id}_attempt_{attempt}"
            artifacts = Path(config["artifact_dir"]) / batch["batch_id"] / run_id
            archived_artifacts = artifacts.with_name(archived_id + "_interrupted")
            if artifacts.exists():
                artifacts.rename(archived_artifacts)
            record.update(
                run_id=archived_id,
                resumed_run_id=run_id,
                status="interrupted",
                reason="Active execution session stopped before checkpoint serialization.",
                archived_artifact_directory=str(archived_artifacts),
                excluded_from_selection_and_metrics=True,
            )
            archived_log = archive / f"{archived_id}.json"
            write_json(archived_log, record)
            event["interrupted_attempts"].append(str(archived_log))
            write_json(batch_path, batch)
        return None, {
            "model_name": model,
            "group": group,
            "seed": seed,
            "candidate_index": index,
            "parameters": config["training"]["candidates"][index],
            "phase": phase,
            "config": config,
            "batch_id": batch["batch_id"],
            "provenance": provenance,
        }

    def collect(pool: ProcessPoolExecutor, requests: list[tuple], key: str) -> list[dict]:
        results, pending = {}, {}
        for position, request in enumerate(requests):
            record, arguments = existing_or_arguments(*request)
            if record:
                results[position] = record
            else:
                pending[pool.submit(fit_task, arguments)] = position
        for future in as_completed(pending):
            position = pending[future]
            results[position] = future.result()
            run_id = results[position]["run_id"]
            if run_id not in batch[key]:
                batch[key].append(run_id)
            write_json(batch_path, batch)
        ordered = [results[position] for position in range(len(requests))]
        for record in ordered:
            if record["run_id"] not in batch[key]:
                batch[key].append(record["run_id"])
        write_json(batch_path, batch)
        return ordered

    selected, tuning = [], []
    experiment = config["experiment"]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for model in experiment["models"]:
            for group in experiment["feature_groups"]:
                trials = collect(
                    pool,
                    [
                        (model, group, index, experiment["tuning_seed"], "tuning")
                        for index in range(len(config["training"]["candidates"]))
                    ],
                    "trial_runs",
                )
                tuning.extend(trials)
                winner = max(trials, key=lambda run: run["validation_metrics"]["average_precision"])
                choice = {
                    "candidate_index": winner["candidate_index"],
                    "parameters": winner["parameters"],
                    "selection_seed": experiment["tuning_seed"],
                    "selection_metric": "validation_average_precision",
                }
                key = f"{model}_{group}"
                if key in batch["choices"]:
                    assert batch["choices"][key] == choice
                batch["choices"][key] = choice
                selected.extend(
                    collect(
                        pool,
                        [
                            (
                                model,
                                group,
                                choice["candidate_index"],
                                seed,
                                "tuning" if seed == experiment["tuning_seed"] else "replication",
                            )
                            for seed in experiment["seeds"]
                        ],
                        "selected_runs",
                    )
                )
            if experiment["run_no_message_control"]:
                choice = batch["choices"][f"{model}_all"]
                selected.extend(
                    collect(
                        pool,
                        [
                            (
                                model,
                                "all_no_messages",
                                choice["candidate_index"],
                                seed,
                                "no_message_control_fixed_all_parameters",
                            )
                            for seed in experiment["seeds"]
                        ],
                        "selected_runs",
                    )
                )
    report = build_gnn_report(selected, tuning, batch["choices"], config, output)
    batch.update(status="completed", report=report)
    write_json(batch_path, batch)
    write_json(output / "latest_run.json", batch)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = resume(args.batch, args.workers)
    print(json.dumps({"status": result["status"], "batch_id": result["batch_id"]}))

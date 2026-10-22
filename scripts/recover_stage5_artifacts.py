"""Repair incomplete Stage 5 saves using the original frozen protocol, without tuning."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.atomic_artifacts import write_json, write_npz  # noqa: E402
from src.data import load_csv, sha256_file  # noqa: E402
from src.final_evaluate import evaluate_frozen_entry, make_cache  # noqa: E402
from src.final_reporting import build_final_report  # noqa: E402
from src.inference import predict_frozen, verify_artifact  # noqa: E402


def unchanged_evaluation_logic(before: str, after: str) -> bool:
    """Permit imports and the NPZ writer alias only; require identical remaining AST."""

    class NormalizeWriter(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "write_npz":
                node.func = ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()), attr="savez_compressed", ctx=ast.Load()
                )
            return node

    def normalized(source):
        tree = NormalizeWriter().visit(ast.parse(source))
        tree.body = [
            node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        return ast.dump(tree, include_attributes=False)

    return normalized(before) == normalized(after)


def recover(output: Path) -> dict:
    """Reuse valid saves and reconstruct incomplete ones from unchanged frozen models."""
    plan = json.loads((output / "frozen_protocol.json").read_text())
    config = plan["config"]
    batch_path = output / "batches" / f"{plan['batch_id']}.json"
    batch = json.loads(batch_path.read_text())
    plan_path = Path(batch["protocol"]["path"])
    plan_hash = batch["protocol"]["sha256"]
    assert sha256_file(plan_path) == plan_hash
    evidence = output / "recovery_evidence"
    evidence.mkdir(exist_ok=True)
    write_json(evidence / "failed_batch_before_recovery.json", batch)
    before_sha = plan["critical_source_sha256"]["final_evaluate.py"]
    snapshot = evidence / f"final_evaluate_before_atomic_fix_{before_sha}.txt"
    assert sha256_file(snapshot) == before_sha
    assert unchanged_evaluation_logic(
        snapshot.read_text(), Path("src/final_evaluate.py").read_text()
    )
    amendment = {
        "file": "final_evaluate.py",
        "before_sha256": before_sha,
        "after_sha256": sha256_file(Path("src/final_evaluate.py")),
        "before_source_snapshot": {"path": str(snapshot), "sha256": before_sha},
        "normalized_evaluation_logic_ast_unchanged": True,
        "change_scope": "atomic fsynced JSON/NPZ serialization only; no model or policy change",
    }
    for name, digest in plan["critical_source_sha256"].items():
        if name != "final_evaluate.py":
            assert sha256_file(Path("src") / name) == digest
    data = load_csv(
        Path(config["data"]["raw_dir"]), require_full_dataset=config["data"]["require_full_dataset"]
    )
    cache = make_cache(data, config)
    original_metrics_path = evidence / "original_test_metrics_before_recovery.csv"
    original_metrics = pd.read_csv(original_metrics_path, float_precision="round_trip")
    completed, repaired, reused = [], [], []
    for entry in plan["references"]:
        source = json.loads(verify_artifact(entry["source_record"]).read_text())
        try:
            verify_artifact(entry["validation_predictions"])
            with np.load(entry["validation_predictions"]["path"]) as validation:
                assert len(validation["score"]) > 0
        except (OSError, ValueError, AssertionError, KeyError):
            nodes, scores, _ = predict_frozen(
                cache.observed,
                (25, 34),
                source,
                cache,
                thread_limit=config["runtime"]["thread_limit"],
            )
            path = Path(entry["validation_predictions"]["path"])
            write_npz(
                path,
                tx_id=data.tx_ids[nodes],
                time_step=data.time_step[nodes],
                label=data.y[nodes],
                score=scores,
            )
            # NPZ writing has deterministic array members; saved bytes must match the frozen hash.
            assert sha256_file(path) == entry["validation_predictions"]["sha256"]
        run_id = f"{plan['batch_id']}_{entry['variant']}_s{entry['seed']}"
        run_path = output / "runs" / f"{run_id}.json"
        run = json.loads(run_path.read_text())
        valid = run["status"] == "completed"
        if valid:
            try:
                verify_artifact(run["test_predictions"])
                with np.load(run["test_predictions"]["path"]) as saved:
                    assert len(saved["score"]) > 0
            except (OSError, ValueError, AssertionError, KeyError):
                valid = False
        if valid:
            reused.append(run_id)
        else:
            old = run
            run = evaluate_frozen_entry(
                entry, plan=plan, plan_path=plan_path, plan_hash=plan_hash, data=data, cache=cache
            )
            if "test_predictions" in old:
                assert run["test_predictions"]["sha256"] == old["test_predictions"]["sha256"]
            repaired.append(run_id)
        previous = original_metrics[
            (original_metrics.variant == entry["variant"])
            & (original_metrics.seed == entry["seed"])
        ].iloc[0]
        assert previous.threshold == run["frozen_threshold"]
        for metric in (
            "average_precision",
            "pr_auc_trapezoid",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ):
            assert np.isclose(previous[metric], run["test_metrics"][metric], rtol=0, atol=1e-14)
        completed.append(run)
    report = build_final_report(completed, plan, output)
    audits = {row["time_step"]: row for row in plan["validation_snapshot_audit"]}
    audits.update(cache.audit)
    write_json(output / "snapshot_feature_audit.json", [audits[key] for key in sorted(audits)])
    recovery = {
        "status": "passed",
        "batch_id": plan["batch_id"],
        "original_protocol_sha256": plan_hash,
        "script_sha256": sha256_file(Path(__file__)),
        "atomic_writer_sha256": sha256_file(Path("src/atomic_artifacts.py")),
        "source_amendment": amendment,
        "reused_evaluation_runs": reused,
        "repaired_evaluation_runs": repaired,
        "original_test_metrics": {
            "path": str(original_metrics_path),
            "sha256": sha256_file(original_metrics_path),
        },
        "all_original_test_metrics_reproduced": True,
        "models_refitted": False,
        "thresholds_or_selection_changed": False,
    }
    write_json(output / "artifact_recovery.json", recovery)
    batch.update(
        status="completed",
        test_evaluated=True,
        report=report,
        artifact_recovery={
            "path": str(output / "artifact_recovery.json"),
            "sha256": sha256_file(output / "artifact_recovery.json"),
        },
        reporting_source_sha256=sha256_file(Path("src/final_reporting.py")),
    )
    batch.pop("error", None)
    write_json(batch_path, batch)
    write_json(output / "latest_run.json", batch)
    return recovery


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage5"))
    result = recover(parser.parse_args().output_dir)
    print(
        json.dumps(
            {"status": result["status"], "repaired": len(result["repaired_evaluation_runs"])}
        )
    )

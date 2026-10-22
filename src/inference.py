"""Label-free, fit-free inference for the frozen Elliptic models.

Observed snapshots contain features, IDs, times and topology only. No target
vector is accepted by the feature builders or model scoring interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.data import (
    AGGREGATE_COUNT,
    LOCAL_NUMERIC_COUNT,
    EllipticData,
    GraphView,
    sha256_file,
)
from src.features import (
    NEIGHBOR_NAMES,
    STRUCTURE_NAMES,
    neighbor_score_aggregates,
    snapshot_structure,
)
from src.train import array_digest


@dataclass(frozen=True)
class Observations:
    """Model-visible data; labels are deliberately absent."""

    tx_ids: np.ndarray
    time_step: np.ndarray
    x: np.ndarray
    edge_index: np.ndarray

    @classmethod
    def from_data(cls, data: EllipticData) -> Observations:
        """Drop label access before building any predictors."""
        return cls(data.tx_ids, data.time_step, data.x, data.edge_index)


def observed_view(observed: Observations, steps: tuple[int, int]) -> GraphView:
    """Induce one observed period without exposing or using labels."""
    start, stop = steps
    if not 1 <= start <= stop <= 49:
        raise ValueError("Observed period must be within steps 1–49.")
    mask = (observed.time_step >= start) & (observed.time_step <= stop)
    nodes = np.flatnonzero(mask)
    remap = np.full(len(observed.tx_ids), -1, dtype=np.int64)
    remap[nodes] = np.arange(len(nodes))
    keep = mask[observed.edge_index[0]] & mask[observed.edge_index[1]]
    edges = remap[observed.edge_index[:, keep]]
    if edges.size and np.any(
        observed.time_step[nodes[edges[0]]] != observed.time_step[nodes[edges[1]]]
    ):
        raise ValueError("An edge crosses observed snapshots.")
    return GraphView(nodes, edges)


def raw_inputs(
    observed: Observations,
    nodes: np.ndarray,
    group: str,
    *,
    include_time: bool,
) -> tuple[np.ndarray, list[str]]:
    """Select named released columns; this routine fits no transformations."""
    if group not in {"local", "all"}:
        raise ValueError("Raw predictors require local or all inputs.")
    count = LOCAL_NUMERIC_COUNT + (AGGREGATE_COUNT if group == "all" else 0)
    x = observed.x[nodes, :count]
    names = [f"local_{i:03d}" for i in range(1, LOCAL_NUMERIC_COUNT + 1)]
    if group == "all":
        names += [f"aggregate_{i:03d}" for i in range(1, AGGREGATE_COUNT + 1)]
    if include_time:
        x = np.column_stack([observed.time_step[nodes].astype(np.float32), x])
        names = ["time_step", *names]
    return x.astype(np.float32, copy=False), names


def verify_artifact(artifact: dict[str, str]) -> Path:
    """Require the saved bytes; do not silently refit a missing or changed model."""
    path = Path(artifact["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen artifact: {path}. Reproduce the earlier training stages first; "
            "Stage 5 never refits models or replaces a frozen decision threshold."
        )
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"Frozen artifact hash changed: {path}")
    return path


def load_score_helper(record: dict[str, Any]) -> Any:
    """Load only the Stage 3 helper frozen at step 24; never fit a new helper."""
    if record["fit_cutoff"] != 24 or record["status"] != "completed" or record["cold_start"]:
        raise ValueError("Neighbor scores require the completed, training-only prefix-24 helper.")
    helper = joblib.load(verify_artifact(record["model_artifact"]))
    if int(helper.named_steps["scaler"].n_samples_seen_) != record["fit_labeled_nodes"]:
        raise ValueError("Score-helper scaler disagrees with its frozen training audit.")
    return helper


class SnapshotFeatureCache:
    """Compute structure and frozen-helper scores independently within each snapshot."""

    def __init__(
        self,
        observed: Observations,
        graph_config: dict[str, Any],
        helper: Any,
        *,
        thread_limit: int,
    ) -> None:
        self.observed = observed
        self.graph_config = graph_config
        self.helper = helper
        self.thread_limit = thread_limit
        self.values: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.audit: dict[int, dict[str, Any]] = {}

    def build(self, step: int) -> None:
        """Deterministic current-snapshot transforms, with no labels or future nodes."""
        if step in self.values:
            return
        view = observed_view(self.observed, (step, step))
        structure, graph = snapshot_structure(view, self.graph_config)
        if not len(view.node_indices):
            scores = np.empty(0, dtype=np.float64)
        else:
            local, _ = raw_inputs(self.observed, view.node_indices, "local", include_time=False)
            with threadpool_limits(limits=self.thread_limit):
                scores = self.helper.predict_proba(local)[:, 1].astype(np.float64)
        neighbors = neighbor_score_aggregates(
            graph,
            scores,
            empty_score=self.graph_config["no_neighbor_score"],
        )
        self.values[step] = (view.node_indices, structure, neighbors)
        self.audit[step] = {
            "time_step": step,
            "nodes": len(view.node_indices),
            "raw_directed_edges": view.edge_index.shape[1],
            "structure_sha256": array_digest(structure),
            "neighbor_scores_sha256": array_digest(neighbors),
            "helper_fit_end_step": 24,
            "current_or_future_labels_used": False,
        }

    def aligned(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return requested rows in original transaction order."""
        structure = np.empty((len(nodes), len(STRUCTURE_NAMES)), dtype=np.float32)
        neighbors = np.empty((len(nodes), len(NEIGHBOR_NAMES)), dtype=np.float32)
        times = self.observed.time_step[nodes]
        for step in np.unique(times):
            self.build(int(step))
            source_nodes, values, scores = self.values[int(step)]
            rows = np.flatnonzero(times == step)
            positions = np.searchsorted(source_nodes, nodes[rows])
            if not np.array_equal(source_nodes[positions], nodes[rows]):
                raise ValueError("Snapshot feature rows do not align with transactions.")
            structure[rows], neighbors[rows] = values[positions], scores[positions]
        return structure, neighbors


def model_inputs(
    observed: Observations,
    nodes: np.ndarray,
    record: dict[str, Any],
    cache: SnapshotFeatureCache,
) -> np.ndarray:
    """Reconstruct exactly the recorded input convention, never choosing columns on test."""
    group = record["feature_group"]
    if group in {"all_structure", "all_graph", "structure_only"}:
        structure, neighbors = cache.aligned(nodes)
        names = list(STRUCTURE_NAMES)
        x = structure
        if group == "all_graph":
            x = np.column_stack([x, neighbors])
            names += NEIGHBOR_NAMES
        if group != "structure_only":
            raw, raw_names = raw_inputs(observed, nodes, "all", include_time=True)
            x, names = np.column_stack([raw, x]), [*raw_names, *names]
    else:
        raw_group = "all" if group in {"all_control", "all_no_messages"} else group
        x, names = raw_inputs(observed, nodes, raw_group, include_time=record["include_time"])
    if names != record["feature_names"] or x.shape != (len(nodes), record["n_features"]):
        raise ValueError("Inference columns differ from the frozen training schema.")
    if not np.isfinite(x).all():
        raise ValueError("Inference inputs contain nonfinite values.")
    return x


def predict_frozen(
    observed: Observations,
    steps: tuple[int, int],
    record: dict[str, Any],
    cache: SnapshotFeatureCache,
    *,
    thread_limit: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load verified weights/scalers and score all current nodes without any fit call."""
    if record["status"] != "completed" or record["threshold"]["selection_split"] != "validation":
        raise ValueError("Inference requires a completed model and a validation-frozen threshold.")
    view = observed_view(observed, steps)
    x = model_inputs(observed, view.node_indices, record, cache)
    if record["model"] in {"logreg", "xgboost"}:
        path = verify_artifact(record["artifacts"]["model"])
        if record["model"] == "logreg":
            model = joblib.load(path)
        else:
            model = XGBClassifier(n_jobs=thread_limit)
            model.load_model(path)
        with threadpool_limits(limits=thread_limit):
            scores = model.predict_proba(x)[:, 1].astype(np.float64)
        audit = {"input_sha256": array_digest(x), "graph_nodes": len(x)}
    else:
        import torch

        from src.gnn import SmallGNN, bidirectional_edges

        torch.set_num_threads(thread_limit)
        torch.use_deterministic_algorithms(True)
        checkpoint = torch.load(
            verify_artifact(record["artifacts"]["checkpoint"]),
            map_location="cpu",
            weights_only=True,
        )
        scaler = joblib.load(verify_artifact(record["artifacts"]["scaler"]))
        if (
            array_digest(scaler.mean_) != record["data_audit"]["scaler_mean_sha256"]
            or array_digest(scaler.scale_) != record["data_audit"]["scaler_scale_sha256"]
        ):
            raise ValueError("Frozen GNN scaler differs from its training evidence.")
        values = scaler.transform(x).astype(np.float32)
        edges = bidirectional_edges(view.edge_index)
        if record["feature_group"] == "all_no_messages":
            edges = np.empty((2, 0), dtype=np.int64)
        parameters = record["parameters"]
        model = SmallGNN(
            record["model"],
            record["n_features"],
            parameters["hidden_channels"],
            parameters["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            scores = model(torch.from_numpy(values), torch.from_numpy(edges.copy())).sigmoid()
            scores = scores.numpy().astype(np.float64)
        audit = {
            "input_sha256": array_digest(
                values,
                edges,
                observed.tx_ids[view.node_indices],
                observed.time_step[view.node_indices],
            ),
            "graph_nodes": len(x),
            "message_edges_without_self_loops": edges.shape[1],
            "unknown_nodes_excluded_from_graph": False,
        }
    if scores.shape != (len(view.node_indices),) or not np.isfinite(scores).all():
        raise ValueError("Model produced invalid inference scores.")
    return view.node_indices, scores, audit

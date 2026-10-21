"""Snapshot-local graph features and forward-chained, train-label-only scores.

No held-out label can enter any predictor. Unknown nodes remain in each graph.
PageRank normalization uses the current snapshot only, never the full timeline.
Test-period feature rows remain NaN because Stage 3 does not process them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.data import (
    UNKNOWN,
    EllipticData,
    GraphView,
    make_temporal_masks,
    sha256_file,
    snapshot_graph,
    training_labels,
    write_json,
)
from src.train import array_digest

STRUCTURE_NAMES = [
    "graph_degree",
    "graph_in_degree",
    "graph_out_degree",
    "graph_pagerank",
    "graph_pagerank_times_snapshot_nodes",
    "graph_clustering_undirected",
]
NEIGHBOR_NAMES = ["graph_neighbor_score_mean", "graph_neighbor_score_max"]


@dataclass(frozen=True)
class GraphFeatures:
    """Aligned feature arrays plus audit evidence; unavailable future rows are NaN."""

    structure: NDArray[np.float32]
    neighbor_scores: NDArray[np.float32]
    node_scores: NDArray[np.float64]
    snapshot_audit: list[dict[str, Any]]
    score_audit: list[dict[str, Any]]


def snapshot_structure(
    view: GraphView, graph_config: dict[str, Any]
) -> tuple[NDArray[np.float32], nx.DiGraph]:
    """Compute in/out/total degree, directed PageRank and undirected clustering.

    Edges are unweighted and directed. Duplicate edges and self-loops are
    rejected rather than silently changing their interpretation. Isolates are
    retained. The size-adjusted PageRank is n * PageRank for this snapshot.
    """
    if graph_config["clustering"] != "undirected_projection":
        raise ValueError("Clustering must use the documented undirected projection.")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(view.node_indices)))
    graph.add_edges_from(view.edge_index.T.tolist())
    if graph.number_of_edges() != view.edge_index.shape[1]:
        raise ValueError("Duplicate directed edges are unsupported in graph feature definitions.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Self-loops are unsupported in the raw graph feature definitions.")
    if not len(graph):
        return np.empty((0, len(STRUCTURE_NAMES)), dtype=np.float32), graph
    pagerank = nx.pagerank(
        graph,
        alpha=graph_config["pagerank_alpha"],
        weight=None,
        max_iter=graph_config["pagerank_max_iter"],
        tol=graph_config["pagerank_tolerance"],
    )
    clustering = nx.clustering(graph.to_undirected(), weight=None)
    values = np.asarray(
        [
            [
                graph.degree(node),
                graph.in_degree(node),
                graph.out_degree(node),
                pagerank[node],
                len(graph) * pagerank[node],
                clustering[node],
            ]
            for node in graph
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite graph feature detected.")
    return values, graph


def neighbor_score_aggregates(
    graph: nx.DiGraph, scores: NDArray[np.float64], *, empty_score: float
) -> NDArray[np.float32]:
    """Mean/max scores over unique immediate neighbors, excluding the node itself.

    All neighbors, including unknown-labeled nodes, receive out-of-time model
    scores. This routine receives no labels. Incoming/outgoing reciprocal edges
    count the neighbor once. Isolates receive the explicitly configured sentinel.
    """
    if scores.shape != (len(graph),) or not np.isfinite(scores).all():
        raise ValueError("Each snapshot node must have one finite score.")
    if np.any((scores < 0) | (scores > 1)) or not 0 <= empty_score <= 1:
        raise ValueError("Neighbor scores and the empty-neighbor sentinel must lie in [0, 1].")
    values = np.full((len(graph), len(NEIGHBOR_NAMES)), empty_score, dtype=np.float32)
    for node in graph:
        neighbors = (set(graph.predecessors(node)) | set(graph.successors(node))) - {node}
        if neighbors:
            neighbor_scores = scores[sorted(neighbors)]
            values[node] = [neighbor_scores.mean(), neighbor_scores.max()]
    return values


def forward_chained_scores(
    data: EllipticData,
    visible_train_labels: NDArray[np.int64],
    *,
    max_time: int,
    risk_config: dict[str, Any],
    thread_limit: int,
    artifact_dir: Path | None = None,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Predict each training step from earlier steps; freeze steps 1–24 for validation.

    A deterministic, fixed-C weighted LogReg uses 93 local numeric inputs only.
    Scalers and class weights use known labels at steps < current_step, capped
    at 24. With insufficient class support, use a fixed cold-start score; no
    current/future labels estimate that fallback. No validation tuning occurs.
    """
    if not 1 <= max_time <= 34:
        raise ValueError("Stage 3 score generation may process steps 1–34 only.")
    if visible_train_labels.shape != data.y.shape:
        raise ValueError("Visible label vector must align with all nodes.")
    if np.any(visible_train_labels[data.time_step > 24] != UNKNOWN):
        raise ValueError("Validation/test labels cannot enter neighbor score generation.")
    if not np.isin(visible_train_labels, [-1, 0, 1]).all():
        raise ValueError("Visible labels must be unknown or binary.")
    if risk_config["model"] != "logreg" or risk_config["input_features"] != "local_without_time":
        raise ValueError("The score helper is fixed LogReg on local numeric inputs without time.")
    if risk_config["parameters"].get("class_weight") != "balanced":
        raise ValueError("Score helper must use explicit balanced class weights.")
    fallback = risk_config["cold_start_score"]
    if not 0 <= fallback <= 1:
        raise ValueError("Cold-start score must lie in [0, 1].")
    x, _ = data.features("local", include_time=False)
    scores = np.full(data.num_nodes, np.nan, dtype=np.float64)
    audit: list[dict[str, Any]] = []
    models: dict[int, Pipeline | None] = {}
    fit_metadata: dict[int, dict[str, Any]] = {}
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    for time_step in range(1, max_time + 1):
        current = data.time_step == time_step
        if not current.any():
            continue
        cutoff = min(time_step - 1, 24)
        fit_mask = (data.time_step <= cutoff) & (visible_train_labels != UNKNOWN)
        if cutoff not in models:
            y_fit = visible_train_labels[fit_mask]
            info: dict[str, Any] = {
                "fit_cutoff": cutoff,
                "fit_labeled_nodes": int(fit_mask.sum()),
                "fit_illicit_nodes": int((y_fit == 1).sum()),
                "fit_input_sha256": array_digest(x[fit_mask], y_fit),
                "cold_start": len(np.unique(y_fit)) != 2,
                "config": risk_config,
                "status": "running",
            }
            log_path = artifact_dir / f"prefix_{cutoff:02d}.json" if artifact_dir else None
            if log_path:
                write_json(log_path, info)
            model = None
            try:
                if not info["cold_start"]:
                    model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "classifier",
                                LogisticRegression(
                                    random_state=risk_config["seed"], **risk_config["parameters"]
                                ),
                            ),
                        ]
                    )
                    with (
                        threadpool_limits(limits=thread_limit),
                        warnings.catch_warnings(record=True) as caught,
                    ):
                        warnings.simplefilter("always")
                        model.fit(x[fit_mask], y_fit)
                    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                        raise RuntimeError(
                            "Score helper failed to converge; feature build aborted."
                        )
                    info["warnings"] = [str(item.message) for item in caught]
                    info["scaler_samples_seen"] = int(model.named_steps["scaler"].n_samples_seen_)
                    info["class_weights"] = {
                        str(label): len(y_fit) / (2 * int((y_fit == label).sum()))
                        for label in [0, 1]
                    }
                    if artifact_dir:
                        path = artifact_dir / f"prefix_{cutoff:02d}.joblib"
                        joblib.dump(model, path)
                        info["model_artifact"] = {"path": str(path), "sha256": sha256_file(path)}
                info["status"] = "completed"
                if log_path:
                    write_json(log_path, info)
            except Exception as error:
                info.update(
                    status="failed", error={"type": type(error).__name__, "message": str(error)}
                )
                if log_path:
                    write_json(log_path, info)
                raise
            models[cutoff], fit_metadata[cutoff] = model, info
        model = models[cutoff]
        if model is None:
            scores[current] = fallback
        else:
            with threadpool_limits(limits=thread_limit):
                scores[current] = model.predict_proba(x[current])[:, 1]
        audit.append(
            {
                "time_step": time_step,
                "scored_nodes_including_unknown": int(current.sum()),
                "fit_end_step": cutoff,
                "current_step_labels_used": False,
                "fit": fit_metadata[cutoff],
            }
        )
    return scores, audit


def build_graph_features(
    data: EllipticData,
    config: dict[str, Any],
    *,
    artifact_dir: Path | None = None,
) -> GraphFeatures:
    """Build development-only features and raw train-label coverage diagnostics.

    Label-neighbor coverage is audit-only: it is not supplied to the classifier.
    Using it during training would create a feature availability mismatch.
    """
    max_time = config["graph"]["max_time_step"]
    if isinstance(max_time, bool) or not isinstance(max_time, int) or not 1 <= max_time <= 34:
        raise ValueError("Stage 3 graph statistics may process time steps 1–34 only.")
    if config["graph"]["neighbor_direction"] != "predecessors_union_successors":
        raise ValueError("Neighbor aggregation uses the documented bidirectional union.")
    masks = make_temporal_masks(data)
    visible = training_labels(data, masks)
    scores, score_audit = forward_chained_scores(
        data,
        visible,
        max_time=max_time,
        risk_config=config["risk_model"],
        thread_limit=config["runtime"]["thread_limit"],
        artifact_dir=artifact_dir / "score_models" if artifact_dir else None,
    )
    structure = np.full((data.num_nodes, len(STRUCTURE_NAMES)), np.nan, dtype=np.float32)
    neighbors = np.full((data.num_nodes, len(NEIGHBOR_NAMES)), np.nan, dtype=np.float32)
    audit = []
    for time_step in range(1, max_time + 1):
        view = snapshot_graph(data, time_step)
        if not len(view.node_indices):
            continue
        values, graph = snapshot_structure(view, config["graph"])
        structure[view.node_indices] = values
        neighbors[view.node_indices] = neighbor_score_aggregates(
            graph,
            scores[view.node_indices],
            empty_score=config["graph"]["no_neighbor_score"],
        )
        visible_neighbors = []
        for node in graph:
            adjacent = (set(graph.predecessors(node)) | set(graph.successors(node))) - {node}
            visible_neighbors.append(
                sum(visible[view.node_indices[n]] != UNKNOWN for n in adjacent)
            )
        audit.append(
            {
                "time_step": time_step,
                "nodes_including_unknown": len(graph),
                "directed_edges": graph.number_of_edges(),
                "isolates": nx.number_of_isolates(graph),
                "pagerank_sum": float(values[:, 3].sum(dtype=np.float64)),
                "nodes_with_visible_train_label_neighbor": int(np.count_nonzero(visible_neighbors)),
                "raw_label_neighbor_feature_used": False,
                "graph_scope": "current completed snapshot only",
            }
        )
    return GraphFeatures(structure, neighbors, scores, audit, score_audit)

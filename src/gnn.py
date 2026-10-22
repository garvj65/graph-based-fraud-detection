"""Small PyG models and explicitly isolated temporal graph inputs.

Unknown training-period nodes participate in messages, but never in loss or
scaler fitting. Validation is a separate graph; no test node reaches this API.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from src.data import EllipticData, make_temporal_masks, snapshot_graph, validate_data
from src.features import STRUCTURE_NAMES, snapshot_structure
from src.train import array_digest


@dataclass(frozen=True)
class GraphBatch:
    """One disjoint union of observed snapshots, with separate supervision mask."""

    x: Tensor
    edge_index: Tensor
    y: Tensor
    known: Tensor
    tx_ids: np.ndarray
    times: np.ndarray

    def to(self, device: str) -> GraphBatch:
        """Move tensors without modifying graph membership or evidence arrays."""
        return GraphBatch(
            self.x.to(device),
            self.edge_index.to(device),
            self.y.to(device),
            self.known.to(device),
            self.tx_ids,
            self.times,
        )


@dataclass(frozen=True)
class GNNDevelopmentData:
    """Prepared train/validation tensors and training-only preprocessing evidence."""

    train: GraphBatch
    validation: GraphBatch
    scaler: StandardScaler
    feature_names: list[str]
    audit: dict[str, Any]


def seed_everything(seed: int, *, thread_limit: int, deterministic: bool) -> None:
    """Seed Python, NumPy and Torch; request deterministic kernels explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(thread_limit)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def bidirectional_edges(edges: np.ndarray) -> np.ndarray:
    """Deduplicate both directions; layer-specific root/self treatment is separate."""
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("Edges must have shape [2, E].")
    if edges.size and np.any(edges[0] == edges[1]):
        raise ValueError("Raw self-loops must not silently change GNN semantics.")
    return np.unique(np.concatenate([edges, edges[::-1]], axis=1), axis=1)


def validate_gnn_data(data: GNNDevelopmentData) -> None:
    """Reject future membership, label/mask mismatch, invalid features and edges."""
    for batch, start, end in ((data.train, 1, 24), (data.validation, 25, 34)):
        n = len(batch.tx_ids)
        if batch.x.shape != (n, len(data.feature_names)) or not torch.isfinite(batch.x).all():
            raise ValueError("Finite GNN features must align with IDs and feature names.")
        if batch.times.shape != (n,) or not np.issubdtype(batch.times.dtype, np.integer):
            raise ValueError("Integer timestamps must align with graph nodes.")
        if not n or np.any((batch.times < start) | (batch.times > end)):
            raise ValueError("GNN graph crosses its fixed temporal boundary.")
        if batch.y.shape != (n,) or batch.known.shape != (n,) or batch.known.dtype != torch.bool:
            raise ValueError("Graph labels and boolean supervision mask must align.")
        if not torch.isin(batch.y, torch.tensor([-1, 0, 1], device=batch.y.device)).all():
            raise ValueError("Invalid GNN labels.")
        if not torch.equal(batch.known, batch.y != -1):
            raise ValueError("Unknown labels must be excluded from GNN supervision.")
        if batch.y[batch.known].unique().tolist() != [0, 1]:
            raise ValueError("Each development split requires both known classes.")
        edges = batch.edge_index.cpu().numpy()
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("GNN edges must have shape [2, E].")
        if edges.size and (edges.min() < 0 or edges.max() >= n):
            raise ValueError("Edge references an unavailable node.")
        if edges.size and np.any(batch.times[edges[0]] != batch.times[edges[1]]):
            raise ValueError("A GNN edge crosses snapshots.")
        if len(np.unique(batch.tx_ids)) != n:
            raise ValueError("GNN node IDs must be unique.")
    if np.intersect1d(data.train.tx_ids, data.validation.tx_ids).size:
        raise ValueError("Training and validation graph nodes overlap.")


def prepare_gnn_data(
    data: EllipticData,
    group: str,
    graph_config: dict[str, Any],
    *,
    no_messages: bool = False,
) -> GNNDevelopmentData:
    """Fit scaler on known train rows; build disjoint period graphs including unknowns.

    Structure inputs use the six Stage 3 definitions, computed separately on
    snapshots 1–34. No labels are aggregated. Bidirectional messages deliberately
    discard payment orientation; directed in/out-degree remain structure inputs.
    """
    validate_data(data)
    if graph_config["direction"] != "bidirectional_unique":
        raise ValueError("Only the declared bidirectional message graph is supported.")
    if graph_config["max_time_step"] != 34:
        raise ValueError("Stage 4 cannot prepare test graph statistics.")
    masks = make_temporal_masks(data)
    snapshot_audit = []
    if group == "structure_only":
        x = np.full((data.num_nodes, len(STRUCTURE_NAMES)), np.nan, dtype=np.float32)
        names = list(STRUCTURE_NAMES)
        for step in range(1, 35):
            view = snapshot_graph(data, step)
            values, _ = snapshot_structure(view, graph_config)
            x[view.node_indices] = values
            snapshot_audit.append({"time_step": step, "nodes": len(values)})
        if not np.isnan(x[masks.test_nodes]).all():
            raise ValueError("Test structure was computed during development.")
    elif group in {"local", "all"}:
        x, names = data.features(group, include_time=True)
    else:
        raise ValueError(f"Unsupported GNN feature group: {group}")
    scaler = StandardScaler().fit(x[masks.train_labeled])
    batches, audits = [], {}
    for split in ("train", "validation"):
        mask = masks.node_mask(split)
        nodes = np.flatnonzero(mask)
        remap = np.full(data.num_nodes, -1, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        keep = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        edges = bidirectional_edges(remap[data.edge_index[:, keep]])
        if no_messages:
            edges = np.empty((2, 0), dtype=np.int64)
        values = scaler.transform(x[nodes]).astype(np.float32)
        batch = GraphBatch(
            torch.from_numpy(values),
            torch.from_numpy(edges.copy()),
            torch.from_numpy(data.y[nodes].copy()),
            torch.from_numpy((data.y[nodes] != -1).copy()),
            data.tx_ids[nodes].copy(),
            data.time_step[nodes].copy(),
        )
        batches.append(batch)
        audits[split] = {
            "nodes": len(nodes),
            "unknown_nodes": int((data.y[nodes] == -1).sum()),
            "labeled_nodes": int(batch.known.sum()),
            "raw_directed_edges": int(keep.sum()),
            "message_edges_without_self_loops": edges.shape[1],
            "min_time": int(batch.times.min()),
            "max_time": int(batch.times.max()),
            "input_sha256": array_digest(values, edges, batch.tx_ids, batch.times),
            "supervision_sha256": array_digest(data.y[nodes]),
        }
    result = GNNDevelopmentData(
        *batches,
        scaler,
        names,
        {
            "feature_group": group,
            "no_messages": no_messages,
            "graph_direction": graph_config["direction"],
            "graphs": audits,
            "scaler_fit_scope": "known training rows, steps 1–24 only",
            "scaler_n_samples_seen": int(scaler.n_samples_seen_),
            "scaler_mean_sha256": array_digest(scaler.mean_),
            "scaler_scale_sha256": array_digest(scaler.scale_),
            "structure_snapshots": snapshot_audit,
            "test_features_computed": False,
        },
    )
    validate_gnn_data(result)
    return result


class SmallGNN(nn.Module):
    """Two message-passing layers with ReLU/dropout and one illicit logit per node."""

    def __init__(
        self,
        model: str,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        if model == "gcn":
            # Train and validation have different graphs: never reuse a cached adjacency.
            self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
            self.conv2 = GCNConv(hidden_channels, 1, cached=False)
        elif model == "graphsage":
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean", root_weight=True)
            self.conv2 = SAGEConv(hidden_channels, 1, aggr="mean", root_weight=True)
        else:
            raise ValueError(f"Unsupported GNN architecture: {model}")

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return logits; apply sigmoid only for scoring, never before BCE loss."""
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.conv2(hidden, edge_index).squeeze(-1)


def supervised_loss(logits: Tensor, graph: GraphBatch, *, positive_weight: float) -> Tensor:
    """Exclude unknown targets before evaluating a train-derived weighted BCE."""
    return F.binary_cross_entropy_with_logits(
        logits[graph.known],
        graph.y[graph.known].float(),
        pos_weight=logits.new_tensor(positive_weight),
    )

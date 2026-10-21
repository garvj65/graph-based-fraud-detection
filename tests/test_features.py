"""Test graph semantics and leakage with deliberately small artificial snapshots."""

from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import yaml

from src.data import UNKNOWN, EllipticData, GraphView, make_temporal_masks, training_labels
from src.features import (
    build_graph_features,
    forward_chained_scores,
    neighbor_score_aggregates,
    snapshot_structure,
)


@pytest.fixture
def config() -> dict:
    return yaml.safe_load(Path("configs/stage3.yaml").read_text())


def test_triangle_and_isolate_have_expected_structure(config: dict) -> None:
    view = GraphView(np.arange(4), np.array([[0, 1, 0], [1, 2, 2]]))
    values, graph = snapshot_structure(view, config["graph"])
    np.testing.assert_array_equal(values[:, 0], [2, 2, 2, 0])
    np.testing.assert_array_equal(values[:, 1], [0, 1, 2, 0])
    np.testing.assert_array_equal(values[:, 2], [2, 1, 0, 0])
    np.testing.assert_array_equal(values[:, 5], [1, 1, 1, 0])
    assert values[:, 3].sum() == pytest.approx(1)
    np.testing.assert_allclose(values[:, 4], 4 * values[:, 3])
    assert len(graph) == 4 and graph.out_degree(3) == graph.in_degree(3) == 0


def test_neighbor_scores_exclude_center_and_count_reciprocal_neighbor_once() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (1, 0), (1, 2)])
    score = np.array([0.2, 0.4, 0.8, 0.7])
    values = neighbor_score_aggregates(graph, score, empty_score=0)
    np.testing.assert_allclose(values, [[0.4, 0.4], [0.5, 0.8], [0.4, 0.4], [0, 0]])
    changed = score.copy()
    changed[1] = 0.99
    alternative = neighbor_score_aggregates(graph, changed, empty_score=0)
    np.testing.assert_array_equal(values[1], alternative[1])


@pytest.mark.parametrize("edges", [np.array([[0, 0], [1, 1]]), np.array([[0], [0]])])
def test_duplicate_or_self_edges_rejected(config: dict, edges: np.ndarray) -> None:
    with pytest.raises(ValueError):
        snapshot_structure(GraphView(np.arange(2), edges), config["graph"])


def test_graph_features_never_process_test_and_keep_unknown_nodes(
    data: EllipticData, config: dict
) -> None:
    features = build_graph_features(data, config)
    future = data.time_step >= 35
    assert np.isnan(features.structure[future]).all()
    assert np.isnan(features.neighbor_scores[future]).all()
    assert np.isnan(features.node_scores[future]).all()
    assert np.isfinite(features.structure[(~future) & (data.y == UNKNOWN)]).all()
    assert sum(row["nodes_including_unknown"] for row in features.snapshot_audit) == int(
        (~future).sum()
    )
    for row in features.score_audit:
        assert row["fit_end_step"] < row["time_step"]
        assert row["fit_end_step"] <= 24
        if not row["fit"]["cold_start"]:
            assert row["fit"]["scaler_samples_seen"] == row["fit"]["fit_labeled_nodes"]
    validation = [row for row in features.score_audit if row["time_step"] >= 25]
    assert {row["fit_end_step"] for row in validation} == {24}
    assert len({row["fit"]["fit_input_sha256"] for row in validation}) == 1
    assert all(
        row["nodes_with_visible_train_label_neighbor"] == 0
        for row in features.snapshot_audit
        if row["time_step"] >= 25
    )


def test_heldout_labels_and_future_graph_cannot_change_development_features(
    data: EllipticData,
    config: dict,
) -> None:
    first = build_graph_features(data, config)
    x, y = data.x.copy(), data.y.copy()
    x[data.time_step >= 35] = 1e6
    heldout_known = (data.time_step >= 25) & (y != UNKNOWN)
    y[heldout_known] = 1 - y[heldout_known]
    future_indices = np.flatnonzero(data.time_step == 35)
    edge = np.array([[future_indices[0]], [future_indices[2]]])
    changed = replace(data, x=x, y=y, edge_index=np.column_stack([data.edge_index, edge]))
    second = build_graph_features(changed, config)
    for field in ("structure", "neighbor_scores", "node_scores"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_current_training_labels_cannot_affect_current_step_scores(
    data: EllipticData,
    config: dict,
) -> None:
    visible = training_labels(data, make_temporal_masks(data))
    changed = visible.copy()
    current_known = (data.time_step == 24) & (visible != UNKNOWN)
    changed[current_known] = 1 - changed[current_known]
    args = dict(max_time=24, risk_config=config["risk_model"], thread_limit=1)
    first, _ = forward_chained_scores(data, visible, **args)
    second, _ = forward_chained_scores(data, changed, **args)
    np.testing.assert_array_equal(first, second)
    assert np.all(first[data.time_step == 1] == config["risk_model"]["cold_start_score"])


def test_helper_rejects_validation_labels(data: EllipticData, config: dict) -> None:
    with pytest.raises(ValueError, match="Validation/test labels"):
        forward_chained_scores(
            data, data.y, max_time=34, risk_config=config["risk_model"], thread_limit=1
        )


def test_no_test_graph_feature_request_allowed(data: EllipticData, config: dict) -> None:
    config["graph"]["max_time_step"] = 49
    with pytest.raises(ValueError, match="1–34 only"):
        build_graph_features(data, config)

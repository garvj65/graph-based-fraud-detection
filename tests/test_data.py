"""Guard the actual leakage contracts, not just shapes or implementation details."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    RAW_FILES,
    UNKNOWN,
    EllipticData,
    TemporalSplit,
    assert_no_split_leakage,
    load_csv,
    make_temporal_masks,
    snapshot_graph,
    training_graph,
    training_labels,
    validate_data,
)
from src.eda import per_step_label_summary


def test_labels_join_by_id_despite_reversed_class_csv(data: EllipticData) -> None:
    np.testing.assert_array_equal(data.y[:6], [0, 1, -1, 0, 1, -1])
    np.testing.assert_array_equal(data.edge_index[:, :2], [[0, 1], [1, 2]])


def test_fixed_boundaries_and_chronological_order(data: EllipticData) -> None:
    masks = make_temporal_masks(data)
    assert data.time_step[masks.train_nodes].min() == 1
    assert data.time_step[masks.train_nodes].max() == 24
    assert data.time_step[masks.validation_nodes].min() == 25
    assert data.time_step[masks.validation_nodes].max() == 34
    assert data.time_step[masks.test_nodes].min() == 35
    assert data.time_step[masks.test_nodes].max() == 49
    assert data.time_step[masks.train_nodes].max() < data.time_step[masks.test_nodes].min()
    assert data.time_step[masks.train_nodes].max() < data.time_step[masks.validation_nodes].min()
    assert data.time_step[masks.validation_nodes].max() < data.time_step[masks.test_nodes].min()


def test_every_node_in_exactly_one_split_unknown_included(data: EllipticData) -> None:
    masks = make_temporal_masks(data)
    coverage = sum(masks.node_mask(name).astype(int) for name in ("train", "validation", "test"))
    np.testing.assert_array_equal(coverage, np.ones(data.num_nodes))
    for name in ("train", "validation", "test"):
        assert np.any(masks.node_mask(name) & (data.y == UNKNOWN))
        assert not np.any(masks.labeled_mask(name) & (data.y == UNKNOWN))


def test_masks_do_not_depend_on_row_order(data: EllipticData) -> None:
    # Remove edges here to isolate the row-order/split contract.
    order = np.arange(data.num_nodes)[::-1]
    reversed_data = replace(
        data,
        tx_ids=data.tx_ids[order],
        x=data.x[order],
        y=data.y[order],
        time_step=data.time_step[order],
        edge_index=np.empty((2, 0), dtype=np.int64),
    )
    masks = make_temporal_masks(reversed_data)
    assert reversed_data.time_step[masks.train_nodes].max() == 24


def test_rejects_train_validation_overlap(data: EllipticData) -> None:
    masks = make_temporal_masks(data)
    invalid = replace(masks, train_nodes=masks.train_nodes | masks.validation_nodes)
    with pytest.raises(ValueError, match="boundary"):
        assert_no_split_leakage(data, invalid)


def test_rejects_unknowns_in_supervision(data: EllipticData) -> None:
    masks = make_temporal_masks(data)
    with pytest.raises(ValueError, match="supervision"):
        assert_no_split_leakage(data, replace(masks, train_labeled=masks.train_nodes))


def test_rejects_nonboolean_mask(data: EllipticData) -> None:
    masks = make_temporal_masks(data)
    with pytest.raises(ValueError, match="boolean"):
        assert_no_split_leakage(data, replace(masks, train_nodes=masks.train_nodes.astype(int)))


def test_no_silent_split_change() -> None:
    with pytest.raises(ValueError, match="fixed"):
        TemporalSplit(train=(1, 34))


def test_train_graph_retains_unknown_and_excludes_future(data: EllipticData) -> None:
    masks = make_temporal_masks(data)
    view = training_graph(data, masks)
    assert np.all(data.time_step[view.node_indices] <= 24)
    assert np.any(data.y[view.node_indices] == UNKNOWN)
    global_edges = view.node_indices[view.edge_index]
    assert np.all(data.time_step[global_edges] <= 24)
    assert view.edge_index.shape[1] == 48


def test_test_feature_and_label_changes_cannot_change_training_inputs(data: EllipticData) -> None:
    masks = make_temporal_masks(data)
    changed_x, changed_y = data.x.copy(), data.y.copy()
    changed_x[~masks.train_nodes] = 1e6
    future_known = (~masks.train_nodes) & (data.y != UNKNOWN)
    changed_y[future_known] = 1 - changed_y[future_known]
    changed_edges = data.edge_index[:, masks.train_nodes[data.edge_index[0]]]
    changed = replace(data, x=changed_x, y=changed_y, edge_index=changed_edges)
    validate_data(changed)
    original_view, changed_view = training_graph(data, masks), training_graph(changed, masks)
    np.testing.assert_array_equal(original_view.node_indices, changed_view.node_indices)
    np.testing.assert_array_equal(original_view.edge_index, changed_view.edge_index)
    np.testing.assert_array_equal(
        data.x[original_view.node_indices], changed.x[changed_view.node_indices]
    )
    np.testing.assert_array_equal(training_labels(data, masks), training_labels(changed, masks))


def test_evaluation_labels_never_exposed_to_neighbor_feature(data: EllipticData) -> None:
    masks = make_temporal_masks(data)
    visible = training_labels(data, masks)
    assert np.all(visible[~masks.train_nodes] == UNKNOWN)
    np.testing.assert_array_equal(visible[masks.train_labeled], data.y[masks.train_labeled])


def test_snapshot_contains_only_current_step(data: EllipticData) -> None:
    snapshot = snapshot_graph(data, 35)
    assert len(snapshot.node_indices) == 3
    assert np.all(data.time_step[snapshot.node_indices] == 35)
    assert snapshot.edge_index.min() >= 0
    assert snapshot.edge_index.max() < len(snapshot.node_indices)
    assert not hasattr(snapshot, "y")


@pytest.mark.parametrize("time", [0, 50, 35.5, True])
def test_invalid_snapshot_time_rejected(data: EllipticData, time: int) -> None:
    with pytest.raises(ValueError):
        snapshot_graph(data, time)


@pytest.mark.parametrize("group,count", [("local", 93), ("all", 165)])
def test_feature_schema_time_inclusion_is_explicit(
    data: EllipticData, group: str, count: int
) -> None:
    matrix, names = data.features(group, include_time=False)
    assert matrix.shape == (data.num_nodes, count)
    assert len(names) == count
    assert "time_step" not in names
    with_time, names_with_time = data.features(group, include_time=True)
    assert with_time.shape[1] == count + 1
    assert names_with_time[0] == "time_step"
    np.testing.assert_array_equal(with_time[:, 0], data.time_step)


@pytest.mark.parametrize("problem", ["missing_id", "duplicate_id", "invalid_label"])
def test_rejects_ambiguous_class_csv(raw_csvs: Path, problem: str) -> None:
    path = raw_csvs / RAW_FILES[2]
    frame = pd.read_csv(path, dtype=str)
    if problem == "missing_id":
        frame = frame.iloc[1:]
    elif problem == "duplicate_id":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[0, "class"] = "fraud"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_csv(raw_csvs)


def test_rejects_missing_edge_endpoint(raw_csvs: Path) -> None:
    path = raw_csvs / RAW_FILES[1]
    edges = pd.read_csv(path, dtype=str)
    edges.loc[0, "txId1"] = "not-in-features"
    edges.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_csv(raw_csvs)


@pytest.mark.parametrize("problem", ["future_edge", "time_range", "float_time", "nan_feature"])
def test_rejects_broken_data_contract(data: EllipticData, problem: str) -> None:
    if problem == "future_edge":
        changed = replace(data, edge_index=np.column_stack([data.edge_index, [0, 100]]))
    elif problem == "time_range":
        times = data.time_step.copy()
        times[0] = 50
        changed = replace(data, time_step=times)
    elif problem == "float_time":
        changed = replace(data, time_step=data.time_step.astype(float))
    else:
        features = data.x.copy()
        features[0, 0] = np.nan
        changed = replace(data, x=features)
    with pytest.raises(ValueError):
        validate_data(changed)


def test_unknowns_excluded_from_observed_illicit_rate(data: EllipticData) -> None:
    summary = per_step_label_summary(data)
    assert np.allclose(summary.label_coverage, 2 / 3)
    assert np.allclose(summary.observed_illicit_rate_labeled, 1 / 2)
    assert np.allclose(summary.known_illicit_share_all_nodes, 1 / 3)


def test_no_labels_rate_is_missing_not_zero(data: EllipticData) -> None:
    labels = data.y.copy()
    labels[data.time_step == 49] = UNKNOWN
    summary = per_step_label_summary(replace(data, y=labels))
    last = summary.iloc[-1]
    assert np.isnan(last.observed_illicit_rate_labeled)
    assert last.label_coverage == 0


def test_partial_dataset_cannot_pass_full_integrity_check(data: EllipticData) -> None:
    with pytest.raises(ValueError, match="counts"):
        validate_data(data, require_full_dataset=True)


def test_fractional_raw_timestamp_cannot_round_into_train_split(raw_csvs: Path) -> None:
    path = raw_csvs / RAW_FILES[0]
    frame = pd.read_csv(path, header=None)
    frame[1] = frame[1].astype(float)
    frame.loc[0, 1] = 24.0000001
    frame.to_csv(path, header=False, index=False)
    with pytest.raises(ValueError, match="finite integers"):
        load_csv(raw_csvs)

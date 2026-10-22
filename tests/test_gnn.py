"""Leakage, supervision, graph isolation and checkpoint behavior on artificial data."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.evaluate import binary_metrics  # noqa: E402
from src.gnn import (  # noqa: E402
    SmallGNN,
    bidirectional_edges,
    prepare_gnn_data,
    seed_everything,
    supervised_loss,
    validate_gnn_data,
)
from src.gnn_train import (  # noqa: E402
    EarlyStopping,
    fit_gnn_run,
    run_gnn_experiment,
    validate_gnn_config,
)


@pytest.fixture
def config() -> dict:
    return yaml.safe_load(Path("configs/stage4.yaml").read_text())


@pytest.mark.parametrize("group,n_features", [("local", 94), ("all", 166), ("structure_only", 6)])
def test_gnn_temporal_membership_and_unknown_retention(data, config, group, n_features):
    prepared = prepare_gnn_data(data, group, config["graph"])
    train, val = prepared.train, prepared.validation
    assert train.x.shape == (72, n_features)
    assert val.x.shape == (30, n_features)
    assert train.known.sum() == 48 and val.known.sum() == 20
    assert train.times.max() < val.times.min() < 35
    assert train.edge_index.shape[1] == 96
    assert val.edge_index.shape[1] == 40
    assert prepared.scaler.n_samples_seen_ == 48
    assert prepared.audit["test_features_computed"] is False
    bad = replace(prepared, train=replace(train, times=np.full(72, 35)))
    with pytest.raises(ValueError, match="temporal boundary"):
        validate_gnn_data(bad)


def test_future_changes_cannot_affect_training_or_preprocessing(data, config):
    original = prepare_gnn_data(data, "all", config["graph"])
    x, y = data.x.copy(), data.y.copy()
    x[data.time_step > 24] *= -1000
    future_known = (data.time_step > 24) & (y != -1)
    y[future_known] = 1 - y[future_known]
    changed = prepare_gnn_data(replace(data, x=x, y=y), "all", config["graph"])
    for key in ("x", "y", "known", "edge_index"):
        assert torch.equal(getattr(original.train, key), getattr(changed.train, key))
    np.testing.assert_array_equal(original.scaler.mean_, changed.scaler.mean_)
    # Even unknown training rows cannot influence fitted scaling parameters.
    x[data.y == -1] += 10000
    unknown_changed = prepare_gnn_data(replace(data, x=x), "all", config["graph"])
    np.testing.assert_array_equal(original.scaler.mean_, unknown_changed.scaler.mean_)
    np.testing.assert_array_equal(original.scaler.scale_, unknown_changed.scaler.scale_)


def test_future_edges_cannot_affect_training_or_validation(data, config):
    original = prepare_gnn_data(data, "all", config["graph"])
    future_nodes = np.flatnonzero(data.time_step == 35)
    future_edge = np.array([[future_nodes[0]], [future_nodes[2]]])
    modified = replace(data, edge_index=np.column_stack([data.edge_index, future_edge]))
    changed = prepare_gnn_data(modified, "all", config["graph"])
    for split in ("train", "validation"):
        for key in ("x", "y", "known", "edge_index"):
            assert torch.equal(
                getattr(getattr(original, split), key), getattr(getattr(changed, split), key)
            )


def test_training_cli_dispatches_gcn_with_stage4_defaults(monkeypatch, capsys):
    from src import gnn_train, train

    captured = {}

    def fake_run(config, *, download):
        captured.update(config)
        return {"batch_id": "synthetic_dispatch", "status": "completed", "report": {}}

    monkeypatch.setattr(gnn_train, "run_gnn_experiment", fake_run)
    monkeypatch.setattr("sys.argv", ["train.py", "--model", "gcn", "--seed", "0"])
    train.main()
    assert captured["stage"] == 4
    assert captured["experiment"]["models"] == ["gcn"]
    assert captured["experiment"]["seeds"] == [0]
    assert "synthetic_dispatch" in capsys.readouterr().out


def test_unknown_targets_have_zero_direct_loss_gradient(data, config):
    graph = prepare_gnn_data(data, "local", config["graph"]).train
    logits = torch.zeros(len(graph.y), requires_grad=True)
    loss = supervised_loss(logits, graph, positive_weight=3.0)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~graph.known]) == 0
    assert torch.count_nonzero(logits.grad[graph.known]) == int(graph.known.sum())
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[graph.known],
        graph.y[graph.known].float(),
        pos_weight=torch.tensor(3.0),
    )
    assert loss == expected


@pytest.mark.parametrize("architecture", ["gcn", "graphsage"])
def test_unknown_neighbor_features_participate_in_messages(data, config, architecture):
    graph = prepare_gnn_data(data, "local", config["graph"]).train
    seed_everything(0, thread_limit=1, deterministic=True)
    model = SmallGNN(architecture, graph.x.shape[1], 8, 0.0)
    x = graph.x.clone().requires_grad_()
    logits = model(x, graph.edge_index)
    supervised_loss(logits, graph, positive_weight=1.0).backward()
    assert x.grad[~graph.known].abs().sum() > 0


def test_bidirectional_edges_deduplicate_reciprocal_payments():
    edges = np.array([[0, 1, 1], [1, 0, 2]], dtype=np.int64)
    actual = bidirectional_edges(edges)
    assert set(map(tuple, actual.T)) == {(0, 1), (1, 0), (1, 2), (2, 1)}
    with pytest.raises(ValueError, match="self-loops"):
        bidirectional_edges(np.array([[0], [0]]))


def test_gcn_cannot_reuse_training_adjacency_for_validation(data, config):
    prepared = prepare_gnn_data(data, "all", config["graph"])
    model = SmallGNN("gcn", 166, 8, 0.0).eval()
    fresh = SmallGNN("gcn", 166, 8, 0.0).eval()
    fresh.load_state_dict(model.state_dict())
    with torch.no_grad():
        model(prepared.train.x, prepared.train.edge_index)
        actual = model(prepared.validation.x, prepared.validation.edge_index)
        expected = fresh(prepared.validation.x, prepared.validation.edge_index)
    assert torch.equal(actual, expected)
    assert model.conv1.cached is False and model.conv2.cached is False


def test_no_message_control_has_no_inter_node_edges(data, config):
    standard = prepare_gnn_data(data, "all", config["graph"])
    control = prepare_gnn_data(data, "all", config["graph"], no_messages=True)
    assert control.train.edge_index.shape == (2, 0)
    assert control.validation.edge_index.shape == (2, 0)
    assert torch.equal(control.train.x, standard.train.x)
    assert torch.equal(control.validation.x, standard.validation.x)


def test_early_stopping_uses_ap_and_keeps_earliest_exact_tie():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(1, 0.6)
    assert not stopper.update(2, 0.6)
    assert not stopper.should_stop
    assert not stopper.update(3, 0.5)
    assert stopper.should_stop and stopper.best_epoch == 1
    with pytest.raises(ValueError, match="finite"):
        stopper.update(4, float("nan"))


def test_seed_replay_and_checkpoint_restore_identical_scores(data, config, tmp_path):
    prepared = prepare_gnn_data(data, "all", config["graph"])
    config["output_dir"] = str(tmp_path / "results")
    config["artifact_dir"] = str(tmp_path / "artifacts")
    config["runtime"]["thread_limit"] = 1
    config["training"].update(max_epochs=4, patience=2)
    parameters = {"hidden_channels": 8, "dropout": 0.3, "learning_rate": 0.01}
    records = [
        fit_gnn_run(
            prepared,
            model_name="gcn",
            group="all",
            seed=0,
            candidate_index=0,
            parameters=parameters,
            phase="synthetic_test",
            config=config,
            batch_id=f"synthetic_replay_{i}",
            provenance={"synthetic_fixture": True},
        )
        for i in range(2)
    ]
    with np.load(records[0]["artifacts"]["validation_predictions"]["path"]) as a:
        saved = a["score"].copy()
    with np.load(records[1]["artifacts"]["validation_predictions"]["path"]) as b:
        np.testing.assert_array_equal(saved, b["score"])
    checkpoint = torch.load(records[0]["artifacts"]["checkpoint"]["path"], weights_only=True)
    model = SmallGNN("gcn", 166, 8, 0.3)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        scores = model(prepared.validation.x, prepared.validation.edge_index).sigmoid().numpy()
    np.testing.assert_array_equal(saved, scores)
    history = pd.read_csv(records[0]["artifacts"]["history"]["path"])
    assert records[0]["best_epoch"] == int(history.loc[history.validation_ap.idxmax(), "epoch"])


def test_end_to_end_five_seed_pipeline_and_metrics(raw_csvs, config, tmp_path):
    config["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    config["output_dir"] = str(tmp_path / "results")
    config["artifact_dir"] = str(tmp_path / "artifacts")
    config["experiment"]["feature_groups"] = ["all"]
    config["runtime"]["thread_limit"] = 1
    config["training"].update(
        max_epochs=3,
        patience=2,
        candidates=[{"hidden_channels": 8, "dropout": 0.0, "learning_rate": 0.01}],
    )
    result = run_gnn_experiment(config)
    assert result["status"] == "completed" and result["test_evaluated"] is False
    assert len(result["selected_runs"]) == 20
    assert result["report"]["unique_fit_count"] == 20
    output = Path(config["output_dir"])
    summary = pd.read_csv(output / "validation_summary.csv")
    assert (summary.n_seeds == 5).all()
    assert set(summary.model) == {"gcn", "graphsage"}
    for run_id in result["selected_runs"]:
        record = json.loads((output / "runs" / f"{run_id}.json").read_text())
        assert record["status"] == "completed" and not record["test_evaluated"]
        assert record["threshold"]["selection_split"] == "validation"
        with np.load(record["artifacts"]["validation_predictions"]["path"]) as prediction:
            assert prediction["time_step"].max() == 34
            actual = binary_metrics(
                prediction["label"], prediction["score"], threshold=record["threshold"]["value"]
            )
        assert actual == record["validation_metrics"]
    assert (output / "figures/learning_curves.png").is_file()
    assert len(pd.read_csv(output / "message_passing_lift.csv")) == 10


@pytest.mark.parametrize("mutation", ["test", "metric", "split", "direction"])
def test_config_rejects_protocol_changes(config, mutation):
    config = copy.deepcopy(config)
    if mutation == "test":
        config["experiment"]["evaluate_test"] = True
    elif mutation == "metric":
        config["experiment"]["primary_metric"] = "accuracy"
    elif mutation == "split":
        config["split"]["train"] = [1, 34]
    else:
        config["graph"]["direction"] = "future_global"
    with pytest.raises(ValueError):
        validate_gnn_config(config)


def test_invalid_gnn_config_leaves_failed_batch_log(config, tmp_path):
    config["output_dir"] = str(tmp_path)
    config["experiment"]["evaluate_test"] = True
    with pytest.raises(ValueError, match="test"):
        run_gnn_experiment(config)
    record = json.loads(next((tmp_path / "batches").glob("*.json")).read_text())
    assert record["status"] == "failed"

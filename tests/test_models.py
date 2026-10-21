"""Validate that fitting depends only on known training examples."""

from dataclasses import replace

import numpy as np
import pytest

from src.data import UNKNOWN, EllipticData
from src.models import development_data, fit_tabular, training_class_weights

PARAMETERS = {"class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "C": 1.0}


def test_development_view_contains_no_unknown_or_test_rows(data: EllipticData) -> None:
    development = development_data(data, "all", include_time=False)
    assert len(development.y_train) == 48
    assert len(development.y_validation) == 20
    assert np.all(development.train_times <= 24)
    assert np.all((development.validation_times >= 25) & (development.validation_times <= 34))
    assert np.all(development.y_train != UNKNOWN)
    assert not hasattr(development, "y_test")


def test_scaler_fits_known_training_rows_only(data: EllipticData) -> None:
    changed = data.x.copy()
    changed[(data.y == UNKNOWN) | (data.time_step > 24)] = 1e6
    view = development_data(replace(data, x=changed), "local", include_time=True)
    model, _, _ = fit_tabular("logreg", view, seed=0, parameters=PARAMETERS, thread_limit=1)
    scaler = model.named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(view.y_train)
    np.testing.assert_allclose(scaler.mean_, view.x_train.mean(axis=0, dtype=np.float64))


def test_validation_changes_cannot_change_fitted_model(data: EllipticData) -> None:
    view = development_data(data, "local", include_time=True)
    changed = replace(
        view, x_validation=view.x_validation * 10000, y_validation=1 - view.y_validation
    )
    first, first_weights, _ = fit_tabular(
        "logreg", view, seed=0, parameters=PARAMETERS, thread_limit=1
    )
    second, second_weights, _ = fit_tabular(
        "logreg", changed, seed=0, parameters=PARAMETERS, thread_limit=1
    )
    assert first_weights == second_weights
    np.testing.assert_array_equal(
        first.named_steps["scaler"].mean_, second.named_steps["scaler"].mean_
    )
    np.testing.assert_array_equal(
        first.named_steps["classifier"].coef_, second.named_steps["classifier"].coef_
    )


def test_test_input_changes_cannot_change_development_view(data: EllipticData) -> None:
    first = development_data(data, "all", include_time=False)
    x, y = data.x.copy(), data.y.copy()
    future = data.time_step >= 35
    x[future] = 1e6
    known_future = future & (y != UNKNOWN)
    y[known_future] = 1 - y[known_future]
    second = development_data(replace(data, x=x, y=y), "all", include_time=False)
    for name in ("x_train", "x_validation", "y_train", "y_validation"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_class_weights_from_training_counts() -> None:
    weights = training_class_weights(np.array([0, 0, 0, 1]))
    assert weights == {"licit": 2 / 3, "illicit": 2, "scale_pos_weight": 3}
    with pytest.raises(ValueError, match="known training"):
        training_class_weights(np.array([0, 1, -1]))


def test_xgboost_explicitly_uses_training_imbalance(data: EllipticData) -> None:
    view = development_data(data, "local", include_time=False)
    model, weights, _ = fit_tabular(
        "xgboost",
        view,
        seed=2,
        thread_limit=1,
        parameters={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "n_jobs": 1},
    )
    assert model.get_params()["scale_pos_weight"] == weights["scale_pos_weight"]
    assert model.get_params()["random_state"] == 2

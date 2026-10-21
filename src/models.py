"""Train-only preprocessing and explicitly weighted CPU tabular baselines."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.data import EllipticData, make_temporal_masks


@dataclass(frozen=True)
class DevelopmentData:
    """Only train/validation data can reach the Stage 2 fit/evaluate routine."""

    x_train: NDArray[np.float32]
    y_train: NDArray[np.int64]
    x_validation: NDArray[np.float32]
    y_validation: NDArray[np.int64]
    train_ids: NDArray[np.str_]
    validation_ids: NDArray[np.str_]
    train_times: NDArray[np.int64]
    validation_times: NDArray[np.int64]
    feature_names: list[str]


def development_data(data: EllipticData, group: str, *, include_time: bool) -> DevelopmentData:
    """Slice known training/validation rows; never return test features or labels."""
    if not isinstance(include_time, bool):
        raise ValueError("include_time must be an explicit boolean.")
    masks = make_temporal_masks(data)
    x, names = data.features(group, include_time=include_time)
    train, validation = masks.train_labeled, masks.validation_labeled
    result = DevelopmentData(
        x[train],
        data.y[train],
        x[validation],
        data.y[validation],
        data.tx_ids[train],
        data.tx_ids[validation],
        data.time_step[train],
        data.time_step[validation],
        names,
    )
    validate_development_data(result)
    return result


def validate_development_data(data: DevelopmentData) -> None:
    """Validate the fit boundary even when arrays are supplied outside the loader."""
    for x, y, ids, times, start, stop in (
        (data.x_train, data.y_train, data.train_ids, data.train_times, 1, 24),
        (data.x_validation, data.y_validation, data.validation_ids, data.validation_times, 25, 34),
    ):
        if x.ndim != 2 or x.shape != (len(y), len(data.feature_names)):
            raise ValueError("Feature matrix must align with labeled rows and feature names.")
        if len(y) == 0 or len(ids) != len(y) or times.shape != y.shape:
            raise ValueError("Nonempty IDs, labels and times must align.")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Training and validation each require known labels of both classes.")
        if not np.isfinite(x).all() or not np.issubdtype(times.dtype, np.integer):
            raise ValueError("Features must be finite and timestamps integer.")
        if np.any((times < start) | (times > stop)):
            raise ValueError("Development data crosses its fixed temporal boundary.")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Development IDs must be unique.")
    if set(data.train_ids) & set(data.validation_ids):
        raise ValueError("Train and validation IDs overlap.")


def training_class_weights(labels: NDArray[np.int64]) -> dict[str, float]:
    """Compute imbalance parameters using known training labels only."""
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Class weights require both known training classes.")
    positive, negative = int(labels.sum()), int((labels == 0).sum())
    n = len(labels)
    return {
        "licit": n / (2 * negative),
        "illicit": n / (2 * positive),
        "scale_pos_weight": negative / positive,
    }


def fit_tabular(
    model_name: str,
    data: DevelopmentData,
    *,
    seed: int,
    parameters: dict[str, Any],
    thread_limit: int,
) -> tuple[Pipeline | XGBClassifier, dict[str, float], list[str]]:
    """Fit solely on training arrays; validation is never passed to estimator.fit.

    XGBoost uses fixed configured tree counts and train-only histogram bins.
    Logistic regression standardizes only labeled training rows. Both models
    intentionally use class weighting; output scores are not calibrated.
    """
    validate_development_data(data)
    weights = training_class_weights(data.y_train)
    if model_name == "logreg":
        if parameters.get("class_weight") != "balanced":
            raise ValueError("Logistic regression must explicitly use balanced class weights.")
        model: Pipeline | XGBClassifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=seed, **parameters)),
            ]
        )
    elif model_name == "xgboost":
        if "scale_pos_weight" in parameters or "random_state" in parameters:
            raise ValueError(
                "Weight and seed are derived centrally, not supplied in hyperparameters."
            )
        model = XGBClassifier(
            random_state=seed, scale_pos_weight=weights["scale_pos_weight"], **parameters
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    with threadpool_limits(limits=thread_limit), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data.x_train, data.y_train)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Logistic regression did not converge; run is not reportable.")
    return model, weights, [str(item.message) for item in caught]

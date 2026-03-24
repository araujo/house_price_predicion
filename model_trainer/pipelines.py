"""
Training pipelines: shared preprocessor (Phase 3) + estimators.

``baseline_knn`` mirrors a typical legacy ``create_model`` flow:
``SimpleImputer`` → ``StandardScaler`` on numeric blocks, ``OneHotEncoder`` on
categoricals, then ``KNeighborsRegressor(n_neighbors=5, weights="distance")``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from data_engineer.feature_engineering import FeatureMetadata, build_sklearn_preprocessing_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

EstimatorName = Literal["baseline_knn", "hist_gradient_boosting", "random_forest", "extra_trees"]
TargetMode = Literal["plain", "log1p"]


def build_preprocessor(metadata: FeatureMetadata | None = None):
    """ColumnTransformer from Phase 3 (numeric impute+scale, categorical OHE)."""
    ct = build_sklearn_preprocessing_pipeline(metadata=metadata)
    ct.set_output(transform="pandas")
    return ct


def build_estimator(name: EstimatorName, *, random_state: int = 42):
    """Estimator only (preprocessor is separate in the Pipeline)."""
    if name == "baseline_knn":
        return KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
            algorithm="auto",
            p=2,
            metric="minkowski",
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.08,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown estimator: {name!r}")


def build_supervised_pipeline(
    estimator_name: EstimatorName,
    *,
    metadata: FeatureMetadata | None = None,
    target_mode: TargetMode = "plain",
    random_state: int = 42,
) -> Pipeline | TransformedTargetRegressor:
    """
    Full sklearn object: preprocessor → regressor, optionally with log1p target.

    * ``plain``: target is price in dollars.
    * ``log1p``: ``log1p(price)`` for training; ``predict`` returns dollars via ``expm1``.
    """
    preprocessor = build_preprocessor(metadata=metadata)
    estimator = build_estimator(estimator_name, random_state=random_state)
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ],
    )

    if target_mode == "plain":
        return pipe
    if target_mode == "log1p":
        return TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    raise ValueError(f"Unknown target_mode: {target_mode!r}")

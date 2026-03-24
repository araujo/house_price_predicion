"""
Feature engineering shared by training and inference.

Demographics must already be merged server-side (Phase 2). API clients never
send demographic columns; enrichment stays internal to avoid training-serving skew.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_engineer.constants import (
    DEFAULT_FEATURE_REFERENCE_YEAR,
    ENGINEERED_FEATURE_COLUMNS,
    GEO_CLUSTER_PLACEHOLDER_COLUMN,
    GEO_ZIP_BUCKET_COLUMN,
    HOUSE_CATEGORICAL_COLUMNS,
    HOUSE_NUMERIC_RAW_COLUMNS,
    ZIPCODE_DEMOGRAPHICS_COLUMNS,
)
from data_engineer.preprocessing import build_feature_dataframe

_NUMERIC_EPS = 1e-9

# Preserve sklearn categorical contract when reading model matrices from CSV (pandas infers int).
ZIPCODE_MODEL_CSV_DTYPE: dict[str, type] = {"zipcode": str}


@dataclass(frozen=True)
class FeatureMetadata:
    """
    Contract for column groups and final ordering.

    Use :func:`get_feature_metadata` so training and serving stay aligned.
    """

    house_numeric_raw_columns: tuple[str, ...]
    house_categorical_columns: tuple[str, ...]
    demographic_columns: tuple[str, ...]
    engineered_columns: tuple[str, ...]
    geo_columns: tuple[str, ...]
    final_feature_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary for configs, MLflow tags, or API docs."""
        return {
            "house_numeric_raw_columns": list(self.house_numeric_raw_columns),
            "house_categorical_columns": list(self.house_categorical_columns),
            "demographic_columns": list(self.demographic_columns),
            "engineered_columns": list(self.engineered_columns),
            "geo_columns": list(self.geo_columns),
            "final_feature_columns": list(self.final_feature_columns),
        }

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        """All numeric-like columns (raw house + demographics + engineered + geo)."""
        return (
            *self.house_numeric_raw_columns,
            *self.demographic_columns,
            *self.engineered_columns,
            *self.geo_columns,
        )


def get_feature_metadata() -> FeatureMetadata:
    """Return stable metadata describing feature groups and final column order."""
    final = get_final_feature_column_names()
    geo = (GEO_ZIP_BUCKET_COLUMN, GEO_CLUSTER_PLACEHOLDER_COLUMN)
    return FeatureMetadata(
        house_numeric_raw_columns=HOUSE_NUMERIC_RAW_COLUMNS,
        house_categorical_columns=HOUSE_CATEGORICAL_COLUMNS,
        demographic_columns=ZIPCODE_DEMOGRAPHICS_COLUMNS,
        engineered_columns=ENGINEERED_FEATURE_COLUMNS,
        geo_columns=geo,
        final_feature_columns=final,
    )


def get_final_feature_column_names() -> tuple[str, ...]:
    """
    Deterministic model input column order (stable for MLflow signatures).

    Order: house numeric → house categorical → demographics → engineered → geo.
    """
    return (
        *HOUSE_NUMERIC_RAW_COLUMNS,
        *HOUSE_CATEGORICAL_COLUMNS,
        *ZIPCODE_DEMOGRAPHICS_COLUMNS,
        *ENGINEERED_FEATURE_COLUMNS,
        GEO_ZIP_BUCKET_COLUMN,
        GEO_CLUSTER_PLACEHOLDER_COLUMN,
    )


def _geo_zip_bucket(zipcode: pd.Series) -> pd.Series:
    """Deterministic bucket from normalized zipcode (no trained clustering)."""
    z_str = zipcode.astype(str).str.replace('"', "", regex=False)
    z = pd.to_numeric(z_str, errors="coerce").fillna(0)
    return (z.astype(np.int64) % 1000).astype(np.int32)


def _coerce_floors(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _add_engineered_features(df: pd.DataFrame, *, reference_year: int) -> pd.DataFrame:
    out = df
    yr_built = pd.to_numeric(out["yr_built"], errors="coerce")
    yr_renovated = pd.to_numeric(out["yr_renovated"], errors="coerce").fillna(0)

    house_age = (reference_year - yr_built).clip(lower=0)
    renovated = (yr_renovated > 0) & (yr_renovated > yr_built.fillna(0))
    renovated_flag = renovated.astype(np.int8)
    yr_renovated_np = yr_renovated.to_numpy(dtype=np.float64)
    renovation_age = np.where(
        renovated.to_numpy(),
        np.maximum(np.float64(reference_year) - yr_renovated_np, 0.0),
        0.0,
    ).astype(np.float64)

    sqft_above = pd.to_numeric(out["sqft_above"], errors="coerce").fillna(0)
    sqft_basement = pd.to_numeric(out["sqft_basement"], errors="coerce").fillna(0)
    total_sqft = sqft_above + sqft_basement

    sqft_living = pd.to_numeric(out["sqft_living"], errors="coerce").fillna(0)
    sqft_lot = pd.to_numeric(out["sqft_lot"], errors="coerce").fillna(0)
    living_num = sqft_living.to_numpy(dtype=float)
    living_den = sqft_lot.to_numpy(dtype=float)
    ratio_vals = np.zeros_like(living_num, dtype=np.float64)
    np.divide(
        living_num,
        living_den,
        out=ratio_vals,
        where=living_den > _NUMERIC_EPS,
    )
    living_to_lot_ratio = pd.Series(ratio_vals, index=out.index, dtype=np.float64)

    bathrooms = pd.to_numeric(out["bathrooms"], errors="coerce").fillna(0)
    bedrooms = pd.to_numeric(out["bedrooms"], errors="coerce").fillna(0)
    bed_denom = np.maximum(bedrooms.to_numpy(dtype=float), 1.0)
    bath_bed_ratio = pd.Series(
        bathrooms.to_numpy(dtype=float) / bed_denom,
        index=out.index,
        dtype=np.float64,
    )

    total_np = total_sqft.to_numpy(dtype=float)
    basement_np = sqft_basement.to_numpy(dtype=float)
    basement_vals = np.zeros_like(total_np, dtype=np.float64)
    np.divide(
        basement_np,
        total_np,
        out=basement_vals,
        where=total_np > _NUMERIC_EPS,
    )
    basement_ratio = pd.Series(basement_vals, index=out.index, dtype=np.float64)

    out = out.copy()
    out["house_age"] = house_age
    out["renovated_flag"] = renovated_flag
    out["renovation_age"] = renovation_age
    out["total_sqft"] = total_sqft
    out["living_to_lot_ratio"] = living_to_lot_ratio
    out["bath_bed_ratio"] = bath_bed_ratio
    out["basement_ratio"] = basement_ratio
    return out


def _ensure_zipcode_string(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace('"', "", regex=False).str.strip()


def prepare_model_input_for_prediction(
    X: pd.DataFrame,
    *,
    metadata: FeatureMetadata | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Align categorical dtypes with training before ``Pipeline.predict``.

    CSV round-trips (e.g. batch ``batch_X.csv``) often re-parse ``zipcode`` as int64;
    the fitted ``OneHotEncoder`` expects string categories for ``zipcode``.
    """
    meta = metadata or get_feature_metadata()
    out = X.copy()
    for col in meta.house_categorical_columns:
        if col in out.columns and logger is not None:
            logger.info(
                "categorical feature dtype before coercion: %s=%s",
                col,
                out[col].dtype,
            )
    if "zipcode" in out.columns:
        out["zipcode"] = (
            out["zipcode"].fillna("").astype(str).str.replace('"', "", regex=False).str.strip()
        )
    for col in meta.house_categorical_columns:
        if col in out.columns and logger is not None:
            logger.info(
                "categorical feature dtype after coercion: %s=%s",
                col,
                out[col].dtype,
            )
    return out


def _fill_demographic_nulls(df: pd.DataFrame, demographic_columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in demographic_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def transform_to_model_features(
    df: pd.DataFrame,
    *,
    reference_year: int = DEFAULT_FEATURE_REFERENCE_YEAR,
    strip_metadata: bool = True,
    fill_demographic_na: bool = True,
) -> pd.DataFrame:
    """
    Build the model-ready feature matrix (training or inference).

    Parameters
    ----------
    df:
        Merged house + demographics frame. May include ``id``, ``date``, ``price``.
    reference_year:
        Calendar year used for ``house_age`` / ``renovation_age``. Pass the same
        value in training and serving to avoid skew (e.g. from config or sale date).
    strip_metadata:
        If True, drops ``id`` / ``date`` / ``price`` via :func:`build_feature_dataframe`.
    fill_demographic_na:
        If True, fills missing demographic values with 0 after a left-merge miss.
    """
    working = build_feature_dataframe(df) if strip_metadata else df.copy()
    working = working.copy()

    for col in HOUSE_NUMERIC_RAW_COLUMNS:
        if col == "floors":
            working[col] = _coerce_floors(working[col])
        elif col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    if "zipcode" in working.columns:
        working["zipcode"] = _ensure_zipcode_string(working["zipcode"])

    if fill_demographic_na:
        working = _fill_demographic_nulls(working, ZIPCODE_DEMOGRAPHICS_COLUMNS)

    working = _add_engineered_features(working, reference_year=reference_year)

    working[GEO_ZIP_BUCKET_COLUMN] = _geo_zip_bucket(working["zipcode"])
    working[GEO_CLUSTER_PLACEHOLDER_COLUMN] = np.int8(0)

    final_cols = list(get_final_feature_column_names())
    missing = [c for c in final_cols if c not in working.columns]
    if missing:
        raise ValueError(f"Missing required feature columns after engineering: {missing}")

    out = working[final_cols].copy()
    return out


def build_sklearn_preprocessing_pipeline(
    metadata: FeatureMetadata | None = None,
    *,
    sparse_threshold: float = 0.3,
) -> ColumnTransformer:
    """
    Unfitted sklearn ColumnTransformer (numeric scaling + categorical one-hot).

    Fit on training data only; reuse the fitted transformer at inference.
    """
    meta = metadata or get_feature_metadata()
    numeric = list(meta.numeric_columns)
    categorical = list(meta.house_categorical_columns)

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ],
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
        verbose_feature_names_out=False,
    )

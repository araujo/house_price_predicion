"""Merge demographics, normalize keys, and build feature matrices."""

from pathlib import Path

import pandas as pd

from data_engineer.constants import METADATA_COLUMNS_TO_DROP_FOR_FEATURES
from data_engineer.ingestion import (
    load_future_unseen_examples_dataframe,
    load_kc_house_dataframe,
    load_zipcode_demographics_dataframe,
)
from data_engineer.validation import (
    normalize_zipcode_series,
    run_inference_pipeline_validations,
    run_training_pipeline_validations,
    validate_inference_feature_presence_after_merge,
    validate_merged_training_feature_presence,
)


def merge_demographics_by_zipcode(
    df: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    how: str = "left",
) -> pd.DataFrame:
    """
    Left-join demographics onto `df` using normalized string zipcodes.

    Parameters
    ----------
    df:
        Rows with a ``zipcode`` column (house-level or inference rows).
    demographics:
        Zipcode-level table with ``zipcode`` and demographic columns.
    how:
        Merge mode (default ``left`` keeps all rows from ``df``).
    """
    left = df.copy()
    right = demographics.copy()
    left["zipcode"] = normalize_zipcode_series(left["zipcode"])
    right["zipcode"] = normalize_zipcode_series(right["zipcode"])
    return left.merge(right, on="zipcode", how=how, validate="many_to_one")


def load_training_dataframe(
    raw_dir: Path | str | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load KC sales, merge demographics, ensure string zipcodes.

    Returns a dataframe with ``id``, ``date``, ``price``, house features,
    and demographic columns. Use :func:`split_features_and_target` or
    :func:`build_feature_dataframe` for modeling matrices.
    """
    kc = load_kc_house_dataframe(raw_dir)
    demo = load_zipcode_demographics_dataframe(raw_dir)
    if validate:
        rep = run_training_pipeline_validations(kc, demo)
        if not rep.ok:
            msg = "; ".join(rep.errors)
            raise ValueError(f"training validation failed: {msg}")
    merged = merge_demographics_by_zipcode(kc, demo)
    if validate:
        rep_m = validate_merged_training_feature_presence(merged)
        if not rep_m.ok:
            msg = "; ".join(rep_m.errors)
            raise ValueError(f"merged training validation failed: {msg}")
    return merged


def load_inference_dataframe(
    raw_dir: Path | str | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load future-unseen examples and merge demographics.

    Returned rows match the full feature schema used by the API (house
    features + demographics); there is no ``price`` column.
    """
    inference = load_future_unseen_examples_dataframe(raw_dir)
    demo = load_zipcode_demographics_dataframe(raw_dir)
    if validate:
        rep = run_inference_pipeline_validations(inference)
        if not rep.ok:
            msg = "; ".join(rep.errors)
            raise ValueError(f"inference validation failed: {msg}")
    merged = merge_demographics_by_zipcode(inference, demo)
    if validate:
        rep_m = validate_inference_feature_presence_after_merge(merged)
        if not rep_m.ok:
            msg = "; ".join(rep_m.errors)
            raise ValueError(f"merged inference validation failed: {msg}")
    return merged


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop metadata/target columns and return the feature matrix only.

    Removes ``id``, ``date``, and ``price`` when present so the same
    function works for merged training or inference frames.
    """
    to_drop = [c for c in METADATA_COLUMNS_TO_DROP_FOR_FEATURES if c in df.columns]
    return df.drop(columns=to_drop, errors="ignore")


def split_features_and_target(training_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate ``price`` (target) from merged training data."""
    if "price" not in training_df.columns:
        raise ValueError("training_df must contain a 'price' column")
    y = training_df["price"].copy()
    X = build_feature_dataframe(training_df)
    return X, y


def ensure_zipcode_string(df: pd.DataFrame, column: str = "zipcode") -> pd.DataFrame:
    """Return a copy with ``column`` stored as normalized string dtype."""
    out = df.copy()
    if column in out.columns:
        out[column] = normalize_zipcode_series(out[column])
    return out

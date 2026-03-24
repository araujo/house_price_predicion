"""Unit tests for data_engineer.preprocessing (merge and integration with real paths)."""

from pathlib import Path

import pandas as pd
import pytest
from data_engineer.constants import ZIPCODE_DEMOGRAPHICS_COLUMNS
from data_engineer.preprocessing import (
    build_feature_dataframe,
    load_inference_dataframe,
    load_training_dataframe,
    merge_demographics_by_zipcode,
    split_features_and_target,
)


def test_merge_demographics_by_zipcode_adds_demographic_columns() -> None:
    house = pd.DataFrame(
        {
            "zipcode": [98178, 98042],
            "bedrooms": [3, 4],
            "price": [100_000, 200_000],
        }
    )
    demo_cols = list(ZIPCODE_DEMOGRAPHICS_COLUMNS)
    demo = pd.DataFrame(
        [
            [1.0] * len(demo_cols) + [98178],
            [2.0] * len(demo_cols) + [98042],
        ],
        columns=demo_cols + ["zipcode"],
    )
    merged = merge_demographics_by_zipcode(house, demo)
    assert "ppltn_qty" in merged.columns
    assert merged.shape[0] == 2
    assert str(merged["zipcode"].iloc[0]) == "98178"


def test_merge_demographics_string_zipcode_mismatch_resolved_by_normalization() -> None:
    house = pd.DataFrame({"zipcode": ["98178"], "x": [1]})
    demo = pd.DataFrame({"zipcode": [98178], "ppltn_qty": [100.0]})
    merged = merge_demographics_by_zipcode(house, demo)
    assert merged["ppltn_qty"].iloc[0] == 100.0


def test_split_features_and_target_drops_metadata() -> None:
    df = pd.DataFrame(
        {
            "id": [1],
            "date": ["20140101"],
            "price": [300_000],
            "bedrooms": [2],
            "zipcode": ["98178"],
        }
    )
    X, y = split_features_and_target(df)
    assert "price" not in X.columns
    assert "id" not in X.columns
    assert y.iloc[0] == 300_000


def test_split_features_and_target_raises_without_price() -> None:
    df = pd.DataFrame({"bedrooms": [2]})
    with pytest.raises(ValueError, match="price"):
        split_features_and_target(df)


def test_build_feature_dataframe_inference_like() -> None:
    df = pd.DataFrame({"bedrooms": [2], "zipcode": ["98178"]})
    out = build_feature_dataframe(df)
    assert list(out.columns) == ["bedrooms", "zipcode"]


@pytest.mark.parametrize(
    "raw_dir",
    [None, Path("data/raw")],
)
def test_load_training_and_inference_roundtrip_shapes(raw_dir: Path | None) -> None:
    train = load_training_dataframe(raw_dir)
    inf = load_inference_dataframe(raw_dir)
    X_train, y = split_features_and_target(train)
    X_inf = build_feature_dataframe(inf)
    assert X_train.shape[1] == X_inf.shape[1]
    assert len(y) == train.shape[0]

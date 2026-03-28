"""Unit tests for data_engineer.validation."""

import pandas as pd
from data_engineer.constants import (
    KC_HOUSE_COLUMNS,
    MERGED_TRAINING_FEATURE_COLUMNS,
    ZIPCODE_DEMOGRAPHICS_COLUMNS,
)
from data_engineer.validation import (
    normalize_zipcode_series,
    validate_demographics_schema,
    validate_duplicate_zipcode_rows_demographics,
    validate_kc_house_schema,
    validate_merged_training_feature_presence,
    validate_nulls,
)


def test_normalize_zipcode_series_int_and_float() -> None:
    s = pd.Series([98178, 98042.0, "98028", '"98136"'])
    out = normalize_zipcode_series(s)
    assert list(out) == ["98178", "98042", "98028", "98136"]


def test_normalize_zipcode_series_preserves_na() -> None:
    s = pd.Series([98178, None])
    out = normalize_zipcode_series(s)
    assert pd.isna(out.iloc[1])


def test_validate_kc_house_schema_missing_column() -> None:
    df = pd.DataFrame({c: [0] for c in KC_HOUSE_COLUMNS if c != "price"})
    rep = validate_kc_house_schema(df)
    assert not rep.ok
    assert any("missing columns" in e for e in rep.errors)


def test_validate_demographics_duplicate_zipcode() -> None:
    cols = list(ZIPCODE_DEMOGRAPHICS_COLUMNS)
    df = pd.DataFrame(
        [
            [1.0] * len(cols) + [98178],
            [2.0] * len(cols) + [98178],
        ],
        columns=cols + ["zipcode"],
    )
    rep = validate_duplicate_zipcode_rows_demographics(df)
    assert not rep.ok


def test_validate_nulls_on_price() -> None:
    df = pd.DataFrame({"price": [1.0, None], "zipcode": ["98178", "98178"]})
    rep = validate_nulls(df, ("price",), context="kc")
    assert not rep.ok


def test_validate_merged_training_feature_presence_ok() -> None:
    data = {c: [0.0] for c in MERGED_TRAINING_FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    rep = validate_merged_training_feature_presence(df)
    assert rep.ok


def test_validate_demographics_schema_ok() -> None:
    cols = list(ZIPCODE_DEMOGRAPHICS_COLUMNS)
    df = pd.DataFrame([[1.0] * len(cols) + [98178]], columns=cols + ["zipcode"])
    rep = validate_demographics_schema(df)
    assert rep.ok

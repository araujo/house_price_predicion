"""Tests for feature_engineering — engineered columns and train/infer contract."""

import pandas as pd
import pytest
from data_engineer.constants import ENGINEERED_FEATURE_COLUMNS, ZIPCODE_DEMOGRAPHICS_COLUMNS
from data_engineer.feature_engineering import (
    FeatureMetadata,
    build_sklearn_preprocessing_pipeline,
    get_feature_metadata,
    get_final_feature_column_names,
    transform_to_model_features,
)
from data_engineer.preprocessing import load_inference_dataframe, load_training_dataframe


def _minimal_merged_row() -> pd.DataFrame:
    demo = {c: [1.0] for c in ZIPCODE_DEMOGRAPHICS_COLUMNS}
    row = {
        "bedrooms": [3],
        "bathrooms": [2.0],
        "sqft_living": [1500],
        "sqft_lot": [5000],
        "floors": [1.5],
        "waterfront": [0],
        "view": [0],
        "condition": [3],
        "grade": [7],
        "sqft_above": [1200],
        "sqft_basement": [300],
        "yr_built": [2000],
        "yr_renovated": [2010],
        "zipcode": ["98178"],
        "lat": [47.5],
        "long": [-122.3],
        "sqft_living15": [1400],
        "sqft_lot15": [4800],
    }
    row.update(demo)
    return pd.DataFrame(row)


def test_engineered_columns_present_and_deterministic_order() -> None:
    df = _minimal_merged_row()
    out = transform_to_model_features(df, reference_year=2020)
    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in out.columns
    assert list(out.columns) == list(get_final_feature_column_names())


def test_engineered_values_reasonable() -> None:
    df = _minimal_merged_row()
    out = transform_to_model_features(df, reference_year=2020)
    assert int(out["house_age"].iloc[0]) == 20
    assert int(out["renovated_flag"].iloc[0]) == 1
    assert int(out["renovation_age"].iloc[0]) == 10
    assert float(out["total_sqft"].iloc[0]) == 1500.0
    assert abs(float(out["living_to_lot_ratio"].iloc[0]) - 1500 / 5000) < 1e-9
    assert abs(float(out["bath_bed_ratio"].iloc[0]) - 2 / 3) < 1e-9
    assert abs(float(out["basement_ratio"].iloc[0]) - 300 / 1500) < 1e-9


def test_division_by_zero_and_missing_renovation() -> None:
    demo = {c: [0.0] for c in ZIPCODE_DEMOGRAPHICS_COLUMNS}
    row = {
        "bedrooms": [0],
        "bathrooms": [0.0],
        "sqft_living": [100],
        "sqft_lot": [0],
        "floors": [1.0],
        "waterfront": [0],
        "view": [0],
        "condition": [3],
        "grade": [7],
        "sqft_above": [0],
        "sqft_basement": [0],
        "yr_built": [2000],
        "yr_renovated": [0],
        "zipcode": ["98178"],
        "lat": [47.5],
        "long": [-122.3],
        "sqft_living15": [100],
        "sqft_lot15": [0],
    }
    row.update(demo)
    df = pd.DataFrame(row)
    out = transform_to_model_features(df, reference_year=2020)
    assert float(out["living_to_lot_ratio"].iloc[0]) == 0.0
    assert float(out["bath_bed_ratio"].iloc[0]) == 0.0
    assert float(out["basement_ratio"].iloc[0]) == 0.0
    assert int(out["renovated_flag"].iloc[0]) == 0


def test_train_and_inference_same_columns_and_order() -> None:
    train = load_training_dataframe(None)
    inf = load_inference_dataframe(None)
    ref = 2015
    xt = transform_to_model_features(train, reference_year=ref)
    xi = transform_to_model_features(inf, reference_year=ref)
    assert list(xt.columns) == list(xi.columns)
    assert list(xt.columns) == list(get_final_feature_column_names())


def test_metadata_columns_do_not_skew_features_vs_inference_only() -> None:
    """Same house + demographics: adding id/date/price must match inference transform."""
    base = _minimal_merged_row()
    train_like = base.copy()
    train_like["id"] = [7129300520]
    train_like["date"] = ["20141013T000000"]
    train_like["price"] = [221900]
    ref = 2020
    xt = transform_to_model_features(train_like, reference_year=ref)
    xi = transform_to_model_features(base, reference_year=ref)
    pd.testing.assert_frame_equal(xt.reset_index(drop=True), xi.reset_index(drop=True))


def test_final_column_order_is_stable_tuple() -> None:
    a = get_final_feature_column_names()
    b = get_final_feature_column_names()
    assert a == b
    assert isinstance(a, tuple)
    assert len(a) == len(set(a))


def test_feature_metadata_contract() -> None:
    meta = get_feature_metadata()
    assert isinstance(meta, FeatureMetadata)
    assert meta.final_feature_columns == get_final_feature_column_names()
    assert set(meta.numeric_columns) & set(meta.house_categorical_columns) == set()
    d = meta.to_dict()
    assert "final_feature_columns" in d


def test_demographic_enrichment_columns_present() -> None:
    df = _minimal_merged_row()
    out = transform_to_model_features(df, reference_year=2020)
    for col in ZIPCODE_DEMOGRAPHICS_COLUMNS:
        assert col in out.columns


def test_sklearn_pipeline_builds() -> None:
    ct = build_sklearn_preprocessing_pipeline()
    assert ct.transformers is not None


@pytest.mark.parametrize("reference_year", [2010, 2015, 2024])
def test_geo_zip_bucket_deterministic(reference_year: int) -> None:
    df = _minimal_merged_row()
    out = transform_to_model_features(df, reference_year=reference_year)
    assert int(out["geo_zip_bucket"].iloc[0]) == 178
    assert int(out["geo_cluster_placeholder"].iloc[0]) == 0

"""Data engineering: ingestion, validation, preprocessing."""

from data_engineer.feature_engineering import (
    FeatureMetadata,
    build_sklearn_preprocessing_pipeline,
    get_feature_metadata,
    get_final_feature_column_names,
    transform_to_model_features,
)
from data_engineer.ingestion import (
    load_all_raw,
    load_future_unseen_examples_dataframe,
    load_kc_house_dataframe,
    load_zipcode_demographics_dataframe,
)
from data_engineer.preprocessing import (
    build_feature_dataframe,
    ensure_zipcode_string,
    load_inference_dataframe,
    load_training_dataframe,
    merge_demographics_by_zipcode,
    split_features_and_target,
)
from data_engineer.validation import (
    ValidationReport,
    normalize_zipcode_series,
    run_inference_pipeline_validations,
    run_training_pipeline_validations,
    validate_demographics,
    validate_inference,
    validate_kc_house,
)

__all__ = [
    "FeatureMetadata",
    "ValidationReport",
    "build_sklearn_preprocessing_pipeline",
    "get_feature_metadata",
    "get_final_feature_column_names",
    "transform_to_model_features",
    "build_feature_dataframe",
    "ensure_zipcode_string",
    "load_all_raw",
    "load_future_unseen_examples_dataframe",
    "load_inference_dataframe",
    "load_kc_house_dataframe",
    "load_training_dataframe",
    "load_zipcode_demographics_dataframe",
    "merge_demographics_by_zipcode",
    "normalize_zipcode_series",
    "run_inference_pipeline_validations",
    "run_training_pipeline_validations",
    "split_features_and_target",
    "validate_demographics",
    "validate_inference",
    "validate_kc_house",
]

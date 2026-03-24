"""Schema and quality validation for house price datasets."""

from dataclasses import dataclass, field

import pandas as pd

from data_engineer.constants import (
    INFERENCE_CRITICAL_NON_NULL_COLUMNS,
    INFERENCE_FULL_FEATURE_COLUMNS,
    KC_CRITICAL_NON_NULL_COLUMNS,
    KC_HOUSE_COLUMNS,
    MERGED_TRAINING_FEATURE_COLUMNS,
    ZIPCODE_DEMOGRAPHICS_COLUMNS,
)


@dataclass
class ValidationReport:
    """Aggregated validation outcome."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def merge(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def normalize_zipcode_series(series: pd.Series) -> pd.Series:
    """
    Normalize zipcodes to 5-digit strings.

    Handles int/float/string inputs and quoted strings from CSV.
    """
    out: list[object] = []
    for v in series:
        if pd.isna(v):
            out.append(pd.NA)
            continue
        if isinstance(v, str):
            s = v.strip().strip('"').strip("'")
        else:
            s = str(int(float(v)))
        if s.replace(".", "", 1).isdigit():
            out.append(str(int(float(s))).zfill(5))
        else:
            out.append(s)
    return pd.Series(out, index=series.index, dtype="string")


def _missing_columns(expected: tuple[str, ...], actual: pd.Index) -> list[str]:
    return [c for c in expected if c not in actual]


def _extra_columns(expected: tuple[str, ...], actual: pd.Index) -> list[str]:
    exp = set(expected)
    return [c for c in actual if c not in exp]


def validate_kc_house_schema(df: pd.DataFrame) -> ValidationReport:
    """Check required columns for King County training data."""
    rep = ValidationReport()
    missing = _missing_columns(KC_HOUSE_COLUMNS, df.columns)
    if missing:
        rep.errors.append(f"kc_house: missing columns: {missing}")
    extra = _extra_columns(KC_HOUSE_COLUMNS, df.columns)
    if extra:
        rep.warnings.append(f"kc_house: unexpected extra columns (ignored downstream): {extra}")
    return rep


def validate_demographics_schema(df: pd.DataFrame) -> ValidationReport:
    """Check required columns for zipcode demographics."""
    rep = ValidationReport()
    expected = (*ZIPCODE_DEMOGRAPHICS_COLUMNS, "zipcode")
    missing = _missing_columns(expected, df.columns)
    if missing:
        rep.errors.append(f"demographics: missing columns: {missing}")
    extra = _extra_columns(expected, df.columns)
    if extra:
        rep.warnings.append(f"demographics: unexpected extra columns: {extra}")
    return rep


def validate_inference_schema(df: pd.DataFrame) -> ValidationReport:
    """Check inference rows match the full feature schema (future_unseen_examples)."""
    rep = ValidationReport()
    missing = _missing_columns(INFERENCE_FULL_FEATURE_COLUMNS, df.columns)
    if missing:
        rep.errors.append(f"inference: missing columns: {missing}")
    extra = _extra_columns(INFERENCE_FULL_FEATURE_COLUMNS, df.columns)
    if extra:
        rep.warnings.append(f"inference: unexpected extra columns: {extra}")
    return rep


def validate_nulls(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    context: str = "dataset",
) -> ValidationReport:
    """Report columns that contain null values."""
    rep = ValidationReport()
    for col in columns:
        if col not in df.columns:
            rep.errors.append(f"{context}: null check requested for missing column {col!r}")
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            rep.errors.append(f"{context}: column {col!r} has {int(null_count)} null values")
    return rep


def validate_duplicate_ids(df: pd.DataFrame, id_column: str = "id") -> ValidationReport:
    """Warn or error on duplicate identifiers in training data."""
    rep = ValidationReport()
    if id_column not in df.columns:
        return rep
    dup_mask = df[id_column].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        rep.warnings.append(
            f"kc_house: {n_dup} rows have duplicate {id_column!r} values (non-unique ids)",
        )
    return rep


def validate_duplicate_zipcode_rows_demographics(df: pd.DataFrame) -> ValidationReport:
    """Ensure one row per zipcode in demographics table."""
    rep = ValidationReport()
    if "zipcode" not in df.columns:
        rep.errors.append("demographics: duplicate check requires 'zipcode' column")
        return rep
    dup = df["zipcode"].duplicated(keep=False)
    if dup.any():
        rep.errors.append(
            f"demographics: {int(dup.sum())} rows have duplicate zipcode keys",
        )
    return rep


def validate_duplicate_inference_rows(df: pd.DataFrame) -> ValidationReport:
    """Detect fully duplicated inference rows."""
    rep = ValidationReport()
    dup = df.duplicated()
    if dup.any():
        rep.warnings.append(f"inference: {int(dup.sum())} fully duplicate rows")
    return rep


def validate_merged_training_feature_presence(df: pd.DataFrame) -> ValidationReport:
    """Ensure merged training frame has all expected feature columns (incl. demographics)."""
    rep = ValidationReport()
    missing = _missing_columns(MERGED_TRAINING_FEATURE_COLUMNS, df.columns)
    if missing:
        rep.errors.append(f"merged_training: missing feature columns: {missing}")
    return rep


def validate_inference_feature_presence_after_merge(df: pd.DataFrame) -> ValidationReport:
    """Ensure merged inference frame has full + demographic features."""
    expected = (*INFERENCE_FULL_FEATURE_COLUMNS, *ZIPCODE_DEMOGRAPHICS_COLUMNS)
    rep = ValidationReport()
    missing = _missing_columns(expected, df.columns)
    if missing:
        rep.errors.append(f"merged_inference: missing feature columns: {missing}")
    return rep


def validate_kc_house(df: pd.DataFrame) -> ValidationReport:
    """Run schema, null, duplicate, and zipcode normalization checks for KC training data."""
    rep = ValidationReport()
    rep.merge(validate_kc_house_schema(df))
    rep.merge(validate_nulls(df, KC_CRITICAL_NON_NULL_COLUMNS, context="kc_house"))
    rep.merge(validate_duplicate_ids(df))
    z = normalize_zipcode_series(df["zipcode"]) if "zipcode" in df.columns else None
    if z is not None and z.isna().any():
        rep.errors.append("kc_house: zipcode is null after normalization")
    return rep


def validate_demographics(df: pd.DataFrame) -> ValidationReport:
    """Schema, duplicate zipcode rows, and zipcode null checks."""
    rep = ValidationReport()
    rep.merge(validate_demographics_schema(df))
    rep.merge(validate_duplicate_zipcode_rows_demographics(df))
    rep.merge(validate_nulls(df, ("zipcode",), context="demographics"))
    return rep


def validate_inference(df: pd.DataFrame) -> ValidationReport:
    """Schema, nulls on critical inference columns, duplicate rows."""
    rep = ValidationReport()
    rep.merge(validate_inference_schema(df))
    rep.merge(validate_nulls(df, INFERENCE_CRITICAL_NON_NULL_COLUMNS, context="inference"))
    rep.merge(validate_duplicate_inference_rows(df))
    return rep


def run_training_pipeline_validations(
    kc_df: pd.DataFrame,
    demo_df: pd.DataFrame,
) -> ValidationReport:
    """Validate raw inputs before preprocessing (training path)."""
    rep = ValidationReport()
    rep.merge(validate_kc_house(kc_df))
    rep.merge(validate_demographics(demo_df))
    return rep


def run_inference_pipeline_validations(inference_df: pd.DataFrame) -> ValidationReport:
    """Validate inference examples before preprocessing."""
    return validate_inference(inference_df)

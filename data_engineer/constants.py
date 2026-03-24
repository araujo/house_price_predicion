"""Column names and paths for Seattle house price datasets."""

from pathlib import Path

RAW_DATA_DIR = Path("data/raw")

KC_HOUSE_FILENAME = "kc_house_data.csv"
ZIPCODE_DEMOGRAPHICS_FILENAME = "zipcode_demographics.csv"
FUTURE_UNSEEN_FILENAME = "future_unseen_examples.csv"

# King County sales (training source)
KC_HOUSE_COLUMNS: tuple[str, ...] = (
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
)

# Full feature columns for inference / API "full" endpoint (matches future_unseen_examples.csv)
INFERENCE_FULL_FEATURE_COLUMNS: tuple[str, ...] = (
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
)

# Zipcode-level demographics (excluding merge key)
ZIPCODE_DEMOGRAPHICS_COLUMNS: tuple[str, ...] = (
    "ppltn_qty",
    "urbn_ppltn_qty",
    "sbrbn_ppltn_qty",
    "farm_ppltn_qty",
    "non_farm_qty",
    "medn_hshld_incm_amt",
    "medn_incm_per_prsn_amt",
    "hous_val_amt",
    "edctn_less_than_9_qty",
    "edctn_9_12_qty",
    "edctn_high_schl_qty",
    "edctn_some_clg_qty",
    "edctn_assoc_dgre_qty",
    "edctn_bchlr_dgre_qty",
    "edctn_prfsnl_qty",
    "per_urbn",
    "per_sbrbn",
    "per_farm",
    "per_non_farm",
    "per_less_than_9",
    "per_9_to_12",
    "per_hsd",
    "per_some_clg",
    "per_assoc",
    "per_bchlr",
    "per_prfsnl",
)

# After merge: house features + demographics (single zipcode column)
MERGED_TRAINING_FEATURE_COLUMNS: tuple[str, ...] = (
    *tuple(c for c in KC_HOUSE_COLUMNS if c not in ("id", "date", "price")),
    *ZIPCODE_DEMOGRAPHICS_COLUMNS,
)

# Columns that must not be null for training rows (after merge)
KC_CRITICAL_NON_NULL_COLUMNS: tuple[str, ...] = ("price", "zipcode", "bedrooms", "sqft_living")

# Inference rows must have these non-null before scoring
INFERENCE_CRITICAL_NON_NULL_COLUMNS: tuple[str, ...] = ("zipcode", "bedrooms", "sqft_living")

# Minimal API payload (subset); missing house columns are filled before Phase 3 transform
INFERENCE_MINIMAL_INPUT_COLUMNS: tuple[str, ...] = (
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "zipcode",
    "lat",
    "long",
    "yr_built",
    "grade",
    "condition",
)

METADATA_COLUMNS_TO_DROP_FOR_FEATURES: frozenset[str] = frozenset({"id", "date", "price"})

# --- Feature engineering (Phase 3) — stable names for training/serving & MLflow ---

# Raw house numeric inputs (before engineered totals/ratios)
HOUSE_NUMERIC_RAW_COLUMNS: tuple[str, ...] = (
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
)

HOUSE_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "waterfront",
    "view",
    "condition",
    "grade",
    "zipcode",
)

ENGINEERED_FEATURE_COLUMNS: tuple[str, ...] = (
    "house_age",
    "renovated_flag",
    "renovation_age",
    "total_sqft",
    "living_to_lot_ratio",
    "bath_bed_ratio",
    "basement_ratio",
)

# Deterministic location buckets (no trained clustering — stable for future geo models)
GEO_ZIP_BUCKET_COLUMN: str = "geo_zip_bucket"
GEO_CLUSTER_PLACEHOLDER_COLUMN: str = "geo_cluster_placeholder"

# Default reference year for age features when not supplied (document override in training/serving)
DEFAULT_FEATURE_REFERENCE_YEAR: int = 2015

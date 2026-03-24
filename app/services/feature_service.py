"""Feature preparation: Phase 2 merge + Phase 3 transform (shared with training)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from data_engineer.constants import INFERENCE_FULL_FEATURE_COLUMNS
from data_engineer.feature_engineering import get_feature_metadata, transform_to_model_features
from data_engineer.ingestion import load_zipcode_demographics_dataframe
from data_engineer.preprocessing import merge_demographics_by_zipcode
from data_engineer.validation import normalize_zipcode_series

from app.schemas.prediction import HouseRowFull, HouseRowMinimal

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)


def _coerce_zipcode_series(series: pd.Series) -> pd.Series:
    return normalize_zipcode_series(series)


class FeatureService:
    """Builds the same model matrix as training (merge demographics + Phase 3 features)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def dataframe_from_full_rows(self, rows: list[HouseRowFull]) -> pd.DataFrame:
        """Raw house table (no demographics) from validated full payloads."""
        records = [r.model_dump() for r in rows]
        df = pd.DataFrame(records)
        df["zipcode"] = _coerce_zipcode_series(df["zipcode"])
        self._validate_house_columns(df, mode="full")
        return df

    def dataframe_from_minimal_rows(self, rows: list[HouseRowMinimal]) -> pd.DataFrame:
        """
        Expand minimal payloads to the full house column set using deterministic defaults.

        Defaults are conservative (e.g. no basement, no waterfront) so clients should
        prefer ``/predict/full`` when they have complete listing data.
        """
        out_rows: list[dict] = []
        for r in rows:
            d = r.model_dump()
            z = d["zipcode"]
            d["zipcode"] = str(z).strip().strip('"')
            d["waterfront"] = 0
            d["view"] = 0
            d["sqft_above"] = float(d["sqft_living"])
            d["sqft_basement"] = 0.0
            d["yr_renovated"] = 0
            d["sqft_living15"] = float(d["sqft_living"])
            d["sqft_lot15"] = float(d["sqft_lot"])
            missing = set(INFERENCE_FULL_FEATURE_COLUMNS) - set(d)
            if missing:
                raise ValueError(f"Internal error: missing keys after minimal expansion: {missing}")
            out_rows.append({k: d[k] for k in INFERENCE_FULL_FEATURE_COLUMNS})
        df = pd.DataFrame(out_rows)
        df["zipcode"] = _coerce_zipcode_series(df["zipcode"])
        self._validate_house_columns(df, mode="full")
        return df

    def _validate_house_columns(self, df: pd.DataFrame, *, mode: str) -> None:
        missing = [c for c in INFERENCE_FULL_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required house columns ({mode}): {missing}")

    def enrich_and_transform(self, house_df: pd.DataFrame) -> pd.DataFrame:
        """
        Zipcode-merge demographics (server-side) then Phase 3 ``transform_to_model_features``.

        Same contract as training inference path.
        """
        demo = load_zipcode_demographics_dataframe(self._settings.raw_data_dir)
        merged = merge_demographics_by_zipcode(house_df, demo)
        X = transform_to_model_features(
            merged,
            reference_year=self._settings.feature_reference_year,
            strip_metadata=False,
            fill_demographic_na=True,
        )
        meta = get_feature_metadata()
        cols = list(meta.final_feature_columns)
        if list(X.columns) != cols:
            logger.warning("Feature column order mismatch; reindexing to training contract")
            X = X.reindex(columns=cols)
        return X

    def feature_metadata_dict(self) -> dict:
        return get_feature_metadata().to_dict()

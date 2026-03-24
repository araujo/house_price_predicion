"""Request/response schemas for price prediction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Example row aligned with data/raw/future_unseen_examples.csv (no demographics)
_EXAMPLE_FULL = {
    "bedrooms": 4,
    "bathrooms": 1.0,
    "sqft_living": 1680,
    "sqft_lot": 5043,
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 6,
    "sqft_above": 1680,
    "sqft_basement": 0,
    "yr_built": 1911,
    "yr_renovated": 0,
    "zipcode": "98118",
    "lat": 47.5354,
    "long": -122.273,
    "sqft_living15": 1560,
    "sqft_lot15": 5765,
}


class HouseRowFull(BaseModel):
    """House features only (demographics are merged server-side by zipcode)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE_FULL})

    bedrooms: float = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: float = Field(..., gt=0)
    sqft_lot: float = Field(..., ge=0)
    floors: float
    waterfront: int = Field(..., ge=0)
    view: int = Field(..., ge=0)
    condition: int = Field(..., ge=1, le=5)
    grade: int = Field(..., ge=1, le=13)
    sqft_above: float = Field(..., ge=0)
    sqft_basement: float = Field(..., ge=0)
    yr_built: int = Field(..., ge=1800, le=2100)
    yr_renovated: int = Field(..., ge=0)
    zipcode: str = Field(..., min_length=3)
    lat: float
    long: float
    sqft_living15: float = Field(..., ge=0)
    sqft_lot15: float = Field(..., ge=0)


_EXAMPLE_MINIMAL = {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2220,
    "sqft_lot": 6380,
    "floors": 1.5,
    "zipcode": "98115",
    "lat": 47.6974,
    "long": -122.313,
    "yr_built": 1931,
    "grade": 8,
    "condition": 4,
}


class HouseRowMinimal(BaseModel):
    """Minimum house fields; remaining inputs use documented defaults before enrichment."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE_MINIMAL})

    bedrooms: float = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: float = Field(..., gt=0)
    sqft_lot: float = Field(..., ge=0)
    floors: float
    zipcode: str = Field(..., min_length=3)
    lat: float
    long: float
    yr_built: int = Field(..., ge=1800, le=2100)
    grade: int = Field(..., ge=1, le=13)
    condition: int = Field(..., ge=1, le=5)


class PredictFullRequest(BaseModel):
    """Batch prediction: one or more rows (same schema as future_unseen_examples.csv)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"rows": [_EXAMPLE_FULL, _EXAMPLE_FULL]},
        },
    )

    rows: list[HouseRowFull] = Field(..., min_length=1, max_length=500)


class PredictMinimalRequest(BaseModel):
    """Batch prediction with minimal columns; server fills the rest for the shared pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"rows": [_EXAMPLE_MINIMAL]},
        },
    )

    rows: list[HouseRowMinimal] = Field(..., min_length=1, max_length=500)


class PredictionMeta(BaseModel):
    """Model and feature contract metadata returned with predictions."""

    model_name: str
    model_version: str
    model_source: str = Field(..., description="mlflow_registry | local_artifact")
    feature_reference_year: int
    n_features: int
    feature_columns: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Unified prediction response."""

    predictions: list[float] = Field(..., description="Price estimates in dollars")
    model_name: str
    model_version: str
    model_source: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    meta: PredictionMeta
    feature_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Subset of Phase 3 feature metadata for debugging/clients",
    )

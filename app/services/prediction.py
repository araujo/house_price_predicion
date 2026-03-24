"""Orchestrate feature build + model.predict (inverse transform handled by sklearn pipeline)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from app.schemas.prediction import (
    HouseRowFull,
    HouseRowMinimal,
    PredictionMeta,
    PredictResponse,
)
from app.services.feature_service import FeatureService
from app.services.model_registry import ModelRegistryService

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(
        self,
        settings: Settings,
        features: FeatureService,
        registry: ModelRegistryService,
    ) -> None:
        self._settings = settings
        self._features = features
        self._registry = registry

    def predict_full(self, rows: list[HouseRowFull]) -> PredictResponse:
        house = self._features.dataframe_from_full_rows(rows)
        return self._predict_from_house_df(house)

    def predict_minimal(self, rows: list[HouseRowMinimal]) -> PredictResponse:
        house = self._features.dataframe_from_minimal_rows(rows)
        return self._predict_from_house_df(house)

    def _predict_from_house_df(self, house: pd.DataFrame) -> PredictResponse:
        X = self._features.enrich_and_transform(house)
        lm = self._registry.get()
        logger.info("predict: rows=%s features=%s source=%s", len(X), X.shape[1], lm.source)
        preds = lm.pipeline.predict(X)
        pred_list = [float(x) for x in preds.ravel()]
        meta = self._features.feature_metadata_dict()
        return PredictResponse(
            predictions=pred_list,
            model_name=lm.model_name,
            model_version=lm.model_version,
            model_source=lm.source,
            meta=PredictionMeta(
                model_name=lm.model_name,
                model_version=lm.model_version,
                model_source=lm.source,
                feature_reference_year=self._settings.feature_reference_year,
                n_features=X.shape[1],
                feature_columns=list(meta["final_feature_columns"]),
            ),
            feature_metadata={
                "final_feature_columns": meta["final_feature_columns"],
                "engineered_columns": meta["engineered_columns"],
            },
        )

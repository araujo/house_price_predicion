"""Dependency injection wiring."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.feature_service import FeatureService
from app.services.model_registry import ModelRegistryService
from app.services.prediction import PredictionService


@lru_cache
def _feature_service() -> FeatureService:
    return FeatureService(get_settings())


@lru_cache
def _model_registry_service() -> ModelRegistryService:
    return ModelRegistryService(get_settings())


@lru_cache
def _prediction_service() -> PredictionService:
    return PredictionService(get_settings(), _feature_service(), _model_registry_service())


def get_prediction_service() -> PredictionService:
    return _prediction_service()


def get_feature_service() -> FeatureService:
    return _feature_service()


def get_model_registry_service() -> ModelRegistryService:
    return _model_registry_service()


def get_settings_dep() -> Settings:
    return get_settings()


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
FeatureServiceDep = Annotated[FeatureService, Depends(get_feature_service)]
ModelRegistryServiceDep = Annotated[ModelRegistryService, Depends(get_model_registry_service)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]

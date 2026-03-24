"""Prediction routes (full + minimal)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.dependencies.containers import PredictionServiceDep
from app.schemas.prediction import PredictFullRequest, PredictMinimalRequest, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/full", response_model=PredictResponse)
def predict_full(
    body: PredictFullRequest,
    svc: PredictionServiceDep,
) -> PredictResponse:
    """
    Batch price prediction using the same columns as ``future_unseen_examples.csv``.

    Demographics are **not** accepted from clients; they are merged internally by zipcode.
    """
    try:
        return svc.predict_full(body.rows)
    except RuntimeError as exc:
        logger.exception("predict full failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/minimal", response_model=PredictResponse)
def predict_minimal(
    body: PredictMinimalRequest,
    svc: PredictionServiceDep,
) -> PredictResponse:
    """
    Batch prediction with a reduced payload; missing house fields use documented defaults
    before the shared Phase 3 pipeline runs.
    """
    try:
        return svc.predict_minimal(body.rows)
    except RuntimeError as exc:
        logger.exception("predict minimal failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

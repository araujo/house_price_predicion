"""Aggregate API routers (versioned under ``api_prefix``)."""

from fastapi import APIRouter

from app.api.routes_predict import router as predict_router

api_router = APIRouter()
api_router.include_router(predict_router)

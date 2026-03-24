"""ASGI application entrypoint."""

from fastapi import FastAPI

from app.api.router import api_router
from app.api.routes_health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health_router)
app.include_router(api_router, prefix=settings.api_prefix)

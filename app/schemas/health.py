"""Health check schema."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Standard health payload with service identity."""

    status: str = Field(..., examples=["ok"])
    service: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")

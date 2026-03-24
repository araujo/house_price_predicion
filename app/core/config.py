"""Application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment and optional files."""

    model_config = SettingsConfigDict(
        env_prefix="HPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="house-price-prediction")
    app_version: str = Field(default="0.1.0")
    api_prefix: str = Field(default="/api/v1")
    log_level: str = Field(default="INFO")
    config_dir: str = Field(default="config")

    # Data & features (must match training / Phase 3)
    raw_data_dir: Path = Field(default=Path("data/raw"))
    feature_reference_year: int = Field(default=2015)

    # Model serving: MLflow registry first, then local artifact
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="e.g. file:./mlruns — if unset, local file store default may apply",
    )
    mlflow_registered_model_name: str = Field(default="house_price_regressor")
    local_model_path: Path = Field(default=Path("model/best_model.joblib"))


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

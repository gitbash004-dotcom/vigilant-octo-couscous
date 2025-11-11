from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@db:5432/hr_predictions",
        description="SQLAlchemy database URL",
    )
    model_path: Path = Field(
        default=Path("/app/models/job_change_model.joblib"),
        description="Path to the serialized model artifact",
    )
    data_path: Path = Field(
        default=Path("/app/data/hr_data_sample.csv"),
        description="Fallback dataset used for training if the model artifact is missing",
    )
    create_tables: bool = Field(default=True, description="Create database tables on startup")

    class Config:
        env_file = os.getenv("API_ENV_FILE", "/app/.env")
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()

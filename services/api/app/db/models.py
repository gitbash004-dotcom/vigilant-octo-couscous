from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source: Mapped[str] = mapped_column(String(50), index=True)
    input_features: Mapped[Any] = mapped_column(JSONB)
    prediction: Mapped[float] = mapped_column(Float)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)


class IngestionStatistic(Base):
    __tablename__ = "ingestion_statistics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_name: Mapped[str] = mapped_column(String(255), unique=True)
    ingestion_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    row_count: Mapped[int] = mapped_column(Integer)
    valid_rows: Mapped[int] = mapped_column(Integer)
    invalid_rows: Mapped[int] = mapped_column(Integer)
    criticality: Mapped[str] = mapped_column(String(20))
    summary: Mapped[str] = mapped_column(Text)
    report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    predicted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    good_file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    bad_file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)


class DataQualityIssue(Base):
    __tablename__ = "data_quality_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    statistic_id: Mapped[int] = mapped_column(ForeignKey("ingestion_statistics.id", ondelete="CASCADE"))
    issue_type: Mapped[str] = mapped_column(String(100))
    severity: Mapped[str] = mapped_column(String(20))
    details: Mapped[Any] = mapped_column(JSON)

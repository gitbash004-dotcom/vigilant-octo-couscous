from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, validator

PredictionSource = Literal["webapp", "scheduled", "api"]


class PredictionInput(BaseModel):
    city: str
    city_development_index: float
    gender: str
    relevent_experience: str
    enrolled_university: str
    education_level: str
    major_discipline: str
    experience: float
    company_size: str
    company_type: str
    last_new_job: str
    training_hours: float

    @staticmethod
    def _normalize_numeric(value: Any) -> float:
        if value in (None, ""):
            raise ValueError("Numeric fields cannot be empty")
        if isinstance(value, (int, float)):
            return float(value)
        value = str(value).strip()
        if value.startswith(">"):
            return float(value[1:])
        if value.startswith("<"):
            return float(value[1:])
        return float(value)

    @validator("city_development_index", "experience", "training_hours", pre=True)
    def parse_numeric(cls, value: Any) -> float:
        return cls._normalize_numeric(value)


class PredictRequest(BaseModel):
    source: PredictionSource = Field(default="api")
    records: List[PredictionInput]


class PredictResponseItem(BaseModel):
    prediction: float
    probability: float
    features: PredictionInput
    created_at: datetime


class PredictResponse(BaseModel):
    predictions: List[PredictResponseItem]


class PastPredictionQuery(BaseModel):
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    source: Optional[Literal["webapp", "scheduled", "all", "api"]] = "all"


class PastPredictionItem(BaseModel):
    id: int
    created_at: datetime
    source: str
    prediction: float
    probability: Optional[float]
    features: dict[str, Any]


class PastPredictionResponse(BaseModel):
    items: List[PastPredictionItem]

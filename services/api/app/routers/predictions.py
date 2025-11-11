from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import PredictionRecord
from ..db.session import get_session
from ..model import get_model
from ..schemas.predictions import (
    PastPredictionItem,
    PastPredictionResponse,
    PredictRequest,
    PredictResponse,
    PredictResponseItem,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


def get_db_session() -> Session:
    with get_session() as session:
        yield session


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db_session)) -> PredictResponse:
    model = get_model()
    records = [record.dict() for record in request.records]
    try:
        predictions = model.predict(records)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response_items: List[PredictResponseItem] = []
    for record, (prediction, probability) in zip(request.records, predictions):
        db_record = PredictionRecord(
            source=request.source,
            input_features=record.dict(),
            prediction=float(prediction),
            probability=float(probability),
        )
        db.add(db_record)
        response_items.append(
            PredictResponseItem(
                prediction=float(prediction),
                probability=float(probability),
                features=record,
                created_at=db_record.created_at,
            )
        )
    db.flush()
    return PredictResponse(predictions=response_items)


@router.get("/past", response_model=PastPredictionResponse)
def past_predictions(
    start_date: datetime | None = Query(default=None),
    end_date: datetime | None = Query(default=None),
    source: str | None = Query(default="all"),
    db: Session = Depends(get_db_session),
) -> PastPredictionResponse:
    query = select(PredictionRecord)
    if start_date is not None:
        query = query.where(PredictionRecord.created_at >= start_date)
    if end_date is not None:
        query = query.where(PredictionRecord.created_at <= end_date)
    if source and source != "all":
        query = query.where(PredictionRecord.source == source)

    records = db.execute(query.order_by(PredictionRecord.created_at.desc())).scalars().all()
    items = [
        PastPredictionItem(
            id=record.id,
            created_at=record.created_at,
            source=record.source,
            prediction=record.prediction,
            probability=record.probability,
            features=record.input_features,
        )
        for record in records
    ]
    return PastPredictionResponse(items=items)

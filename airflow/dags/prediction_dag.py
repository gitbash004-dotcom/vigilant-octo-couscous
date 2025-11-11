from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

GOOD_DATA_DIR = os.getenv("GOOD_DATA_DIR", "/opt/airflow/good_data")
API_URL = os.getenv("PREDICTION_API_URL", "http://api:8000/predictions/predict")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@db:5432/hr_predictions",
)


def get_session():
    engine = create_engine(DATABASE_URL)
    return sessionmaker(bind=engine)()


def fetch_pending_statistics(session) -> List[Dict[str, Any]]:
    result = session.execute(
        text(
            """
            SELECT id, file_name, good_file_path
            FROM ingestion_statistics
            WHERE predicted_at IS NULL AND valid_rows > 0
            ORDER BY ingestion_time ASC
            """
        )
    )
    return [dict(row._mapping) for row in result]


def mark_as_predicted(session, statistic_ids: List[int]) -> None:
    if not statistic_ids:
        return
    session.execute(
        text("UPDATE ingestion_statistics SET predicted_at = :ts WHERE id = ANY(:ids)"),
        {"ts": datetime.utcnow(), "ids": statistic_ids},
    )
    session.commit()


with DAG(
    dag_id="predict_hr_data",
    schedule_interval="*/2 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ml-platform"},
    tags=["prediction", "hr"],
) as dag:

    @task()
    def check_for_new_data() -> List[Dict[str, Any]]:
        session = get_session()
        try:
            pending = fetch_pending_statistics(session)
            if not pending:
                raise AirflowSkipException("No new data available")
            return pending
        finally:
            session.close()

    @task()
    def make_predictions(statistics: List[Dict[str, Any]]) -> None:
        session = get_session()
        predicted_ids: List[int] = []
        try:
            for stat in statistics:
                good_path = stat.get("good_file_path") or os.path.join(GOOD_DATA_DIR, stat["file_name"])
                if not os.path.exists(good_path):
                    print(f"Good data file {good_path} not found, skipping")
                    continue
                df = pd.read_csv(good_path)
                if df.empty:
                    continue
                records = df.drop(columns=["target"], errors="ignore").to_dict(orient="records")
                response = requests.post(API_URL, json={"records": records, "source": "scheduled"}, timeout=60)
                response.raise_for_status()
                predicted_ids.append(stat["id"])
            mark_as_predicted(session, predicted_ids)
        finally:
            session.close()

    new_stats = check_for_new_data()
    make_predictions(new_stats)

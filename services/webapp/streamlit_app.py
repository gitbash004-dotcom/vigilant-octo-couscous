from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

import os

API_URL = os.getenv("API_URL") or st.secrets.get("API_URL", "http://api:8000/predictions")

FEATURE_CONFIG = {
    "city": ("text", "city_103"),
    "city_development_index": ("number", 0.92),
    "gender": ("select", ["Male", "Female", "Other"]),
    "relevent_experience": ("select", ["Has relevent experience", "No relevent experience"]),
    "enrolled_university": ("select", ["Full time", "Part time", "no_enrollment"]),
    "education_level": ("select", ["Graduate", "Masters", "Phd", "High School"]),
    "major_discipline": ("select", ["STEM", "Business Degree", "Arts", "Humanities"]),
    "experience": ("number", 5),
    "company_size": ("select", ["<10", "10/49", "50-99", "100-500", "5000-9999", "10000+"]),
    "company_type": ("select", ["Pvt Ltd", "Public Sector", "NGO", "Funded Startup"]),
    "last_new_job": ("select", ["never", ">4", "1", "2", "3"]),
    "training_hours": ("number", 40),
}


def call_api(records: List[Dict[str, Any]], source: str) -> pd.DataFrame:
    response = requests.post(
        f"{API_URL}/predict",
        json={"records": records, "source": source},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()["predictions"]
    df = pd.DataFrame(
        [
            {
                **item["features"],
                "prediction": item["prediction"],
                "probability": item["probability"],
                "created_at": item["created_at"],
            }
            for item in payload
        ]
    )
    return df


def prediction_page() -> None:
    st.header("On-demand predictions")
    st.subheader("Single prediction")
    with st.form("single_prediction"):
        record: Dict[str, Any] = {}
        for feature, (widget, default) in FEATURE_CONFIG.items():
            label = feature.replace("_", " ").title()
            if widget == "text":
                record[feature] = st.text_input(label, value=str(default))
            elif widget == "number":
                record[feature] = st.number_input(label, value=float(default))
            elif widget == "select":
                options = default
                record[feature] = st.selectbox(label, options, index=0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            df = call_api([record], source="webapp")
            st.success("Prediction complete")
            st.dataframe(df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    st.subheader("Batch prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if "target" in batch_df.columns:
                st.warning("The uploaded file contains 'target' column which will be ignored")
                batch_df = batch_df.drop(columns=["target"])
            st.write("Preview", batch_df.head())
            if st.button("Predict uploaded batch"):
                df = call_api(batch_df.to_dict(orient="records"), source="webapp")
                st.dataframe(df)
        except Exception as exc:  # pragma: no cover - streamlit UI feedback
            st.error(f"Failed to read CSV: {exc}")


def past_predictions_page() -> None:
    st.header("Past predictions")
    today = datetime.utcnow()
    start = st.date_input("Start date", value=today.date() - timedelta(days=7))
    end = st.date_input("End date", value=today.date())
    source = st.selectbox("Prediction source", ["all", "webapp", "scheduled", "api"])

    if st.button("Load predictions"):
        params = {
            "start_date": datetime.combine(start, datetime.min.time()).isoformat(),
            "end_date": datetime.combine(end, datetime.max.time()).isoformat(),
            "source": source,
        }
        try:
            response = requests.get(f"{API_URL}/past", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()["items"]
            if not data:
                st.info("No predictions found for the selected filters")
            else:
                df = pd.DataFrame(data)
                st.dataframe(df)
        except Exception as exc:
            st.error(f"Failed to load past predictions: {exc}")


def main() -> None:
    st.set_page_config(page_title="HR Job Change Predictions", layout="wide")
    page = st.sidebar.selectbox("Navigation", ["Predict", "Past predictions"])
    if page == "Predict":
        prediction_page()
    else:
        past_predictions_page()


if __name__ == "__main__":
    main()

"""Train a simple classifier for HR analytics job change prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "hr_data_sample.csv"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "job_change_model.joblib"

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "company_size",
    "company_type",
    "last_new_job",
]
NUMERIC_FEATURES = [
    "city_development_index",
    "experience",
    "training_hours",
]
TARGET = "target"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return preprocess_numeric(df)


def normalize_numeric(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value.startswith(">"):
        value = value[1:]
    if value.startswith("<"):
        value = value[1:]
    return float(value)


def preprocess_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["experience", "training_hours", "city_development_index"]:
        if column in df.columns:
            df[column] = df[column].apply(normalize_numeric)
    return df


def build_pipeline() -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return clf


def train() -> None:
    df = load_dataset(DATA_PATH)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
        },
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()

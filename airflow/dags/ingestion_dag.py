from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import great_expectations as ge
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.utils.trigger_rule import TriggerRule
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "/opt/airflow/raw-data"))
GOOD_DATA_DIR = Path(os.getenv("GOOD_DATA_DIR", "/opt/airflow/good_data"))
BAD_DATA_DIR = Path(os.getenv("BAD_DATA_DIR", "/opt/airflow/bad_data"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/opt/airflow/reports"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@db:5432/hr_predictions",
)

REQUIRED_COLUMNS = [
    "enrollee_id",
    "city",
    "city_development_index",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
    "training_hours",
]

CATEGORICAL_MAP = {
    "gender": {"Male", "Female", "Other"},
    "relevent_experience": {"Has relevent experience", "No relevent experience"},
    "enrolled_university": {"no_enrollment", "Full time", "Part time"},
    "education_level": {"Graduate", "Masters", "Phd", "High School"},
    "major_discipline": {"STEM", "Business Degree", "Arts", "Humanities"},
    "company_size": {"<10", "10/49", "50-99", "100-500", "5000-9999", "10000+", "1000-4999"},
    "company_type": {"Pvt Ltd", "Public Sector", "NGO", "Funded Startup"},
    "last_new_job": {"never", "1", "2", "3", ">4"},
}


class ValidationIssue(Dict[str, Any]):
    pass


def normalize_numeric(value: Any) -> float:
    if value in (None, ""):
        raise ValueError("Numeric value required")
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value.startswith(">"):
        value = value[1:]
    if value.startswith("<"):
        value = value[1:]
    return float(value)


def evaluate_row(row: pd.Series) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for column in REQUIRED_COLUMNS:
        if column not in row or pd.isna(row[column]):
            issues.append({"column": column, "issue": "missing_value", "severity": "high"})

    city_value = row.get("city")
    if isinstance(city_value, str) and not city_value.startswith("city_"):
        issues.append({"column": "city", "issue": "invalid_format", "severity": "low"})

    for column, allowed_values in CATEGORICAL_MAP.items():
        if column in row and row[column] not in allowed_values:
            issues.append({"column": column, "issue": "unexpected_value", "severity": "medium", "allowed": sorted(allowed_values)})

    # Numeric validations
    try:
        cdi = normalize_numeric(row.get("city_development_index"))
        if not 0 <= cdi <= 1:
            issues.append({"column": "city_development_index", "issue": "out_of_range", "severity": "high"})
    except Exception:
        issues.append({"column": "city_development_index", "issue": "invalid_numeric", "severity": "high"})

    try:
        exp = normalize_numeric(row.get("experience"))
        if exp < 0 or exp > 50:
            issues.append({"column": "experience", "issue": "out_of_range", "severity": "medium"})
    except Exception:
        issues.append({"column": "experience", "issue": "invalid_numeric", "severity": "high"})

    try:
        training = normalize_numeric(row.get("training_hours"))
        if training < 0:
            issues.append({"column": "training_hours", "issue": "negative", "severity": "high"})
    except Exception:
        issues.append({"column": "training_hours", "issue": "invalid_numeric", "severity": "high"})

    return issues


def build_html_report(file_name: str, summary: Dict[str, Any], issues: List[ValidationIssue], expectations: List[Dict[str, Any]] | None = None) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    report_path = REPORTS_DIR / f"validation_{Path(file_name).stem}_{timestamp}.html"
    issue_rows = "".join(
        f"<tr><td>{issue['column']}</td><td>{issue['issue']}</td><td>{issue['severity']}</td><td>{json.dumps(issue.get('allowed', ''))}</td></tr>"
        for issue in issues
    )
    expectation_rows = "".join(
        f"<tr><td>{item['name']}</td><td>{'✅' if item['success'] else '❌'}</td><td>{item['details']}</td></tr>"
        for item in (expectations or [])
    )
    html = f"""
    <html>
      <head>
        <title>Data validation report - {file_name}</title>
      </head>
      <body>
        <h1>Validation summary for {file_name}</h1>
        <p><strong>Criticality:</strong> {summary['criticality']}</p>
        <p><strong>Summary:</strong> {summary['summary']}</p>
        <p><strong>Total rows:</strong> {summary['row_count']} - Valid: {summary['valid_rows']} - Invalid: {summary['invalid_rows']}</p>
        <h2>Expectation checks</h2>
        <table border="1" cellspacing="0" cellpadding="5">
          <thead>
            <tr><th>Expectation</th><th>Status</th><th>Details</th></tr>
          </thead>
          <tbody>
            {expectation_rows or '<tr><td colspan="3">No expectations were executed</td></tr>'}
          </tbody>
        </table>
        <h2>Row level issues</h2>
        <table border="1" cellspacing="0" cellpadding="5">
          <thead>
            <tr><th>Column</th><th>Issue</th><th>Severity</th><th>Allowed values / Details</th></tr>
          </thead>
          <tbody>
            {issue_rows or '<tr><td colspan="4">No issues detected</td></tr>'}
          </tbody>
        </table>
      </body>
    </html>
    """
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)


def determine_criticality(invalid_rows: int, row_count: int) -> str:
    if invalid_rows == 0:
        return "low"
    ratio = invalid_rows / max(row_count, 1)
    if ratio > 0.5:
        return "high"
    if ratio > 0.1:
        return "medium"
    return "low"


def aggregate_issues(issues: List[ValidationIssue]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for issue in issues:
        key = issue["issue"]
        summary.setdefault(key, {"count": 0, "severity": issue.get("severity", "medium")})
        summary[key]["count"] += 1
    return summary


def get_db_session():
    engine = create_engine(DATABASE_URL)
    return sessionmaker(bind=engine)()


def register_issues(session, statistic_id: int, issues: List[ValidationIssue], expectations: List[Dict[str, Any]] | None = None) -> None:
    for issue in issues:
        session.execute(
            """
            INSERT INTO data_quality_issues (statistic_id, issue_type, severity, details)
            VALUES (:statistic_id, :issue_type, :severity, :details)
            """,
            {
                "statistic_id": statistic_id,
                "issue_type": issue["issue"],
                "severity": issue.get("severity", "medium"),
                "details": json.dumps(issue),
            },
        )
    if expectations:
        for expectation in expectations:
            if expectation.get("success"):
                continue
            session.execute(
                """
                INSERT INTO data_quality_issues (statistic_id, issue_type, severity, details)
                VALUES (:statistic_id, :issue_type, :severity, :details)
                """,
                {
                    "statistic_id": statistic_id,
                    "issue_type": expectation.get("name", "expectation_failure"),
                    "severity": "medium",
                    "details": json.dumps(expectation),
                },
            )


def save_statistics_record(session, payload: Dict[str, Any]) -> int:
    result = session.execute(
        """
        INSERT INTO ingestion_statistics (file_name, row_count, valid_rows, invalid_rows, criticality, summary, report_path, good_file_path, bad_file_path)
        VALUES (:file_name, :row_count, :valid_rows, :invalid_rows, :criticality, :summary, :report_path, :good_file_path, :bad_file_path)
        RETURNING id
        """,
        payload,
    )
    statistic_id = result.scalar_one()
    session.commit()
    return statistic_id




def _split_dataframe(df: pd.DataFrame, invalid_indices: List[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    invalid_set = set(invalid_indices)
    valid_rows = df[[idx not in invalid_set for idx in range(len(df))]]
    invalid_rows = df[[idx in invalid_set for idx in range(len(df))]]
    return valid_rows, invalid_rows


def _move_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dest)


with DAG(
    dag_id="ingest_hr_data",
    schedule_interval="*/1 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "data-platform"},
    tags=["ingestion", "hr"],
) as dag:

    @task()
    def read_data() -> str:
        files = list(RAW_DATA_DIR.glob("*.csv"))
        if not files:
            raise AirflowSkipException("No raw files to ingest")
        file_path = random.choice(files)
        return str(file_path)

    @task()
    def validate_data(file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        df = pd.read_csv(path)
        ge_df = ge.from_pandas(df)
        expectations: List[Dict[str, Any]] = []

        def add_expectation(expectation_type: str, func, *args, **kwargs) -> None:
            try:
                expectations.append(func(*args, **kwargs))
            except Exception as exc:
                expectations.append(
                    {
                        "expectation_config": {"expectation_type": expectation_type},
                        "success": False,
                        "result": {"exception": str(exc)},
                    }
                )

        expected_columns = sorted(set(REQUIRED_COLUMNS + ["target"]))
        add_expectation("expect_table_columns_to_match_superset", ge_df.expect_table_columns_to_match_superset, expected_columns)
        for column in ["city", "training_hours", "city_development_index"]:
            if column in df.columns:
                if column == "city_development_index":
                    add_expectation("expect_column_values_to_be_between", ge_df.expect_column_values_to_be_between, column, 0, 1)
                else:
                    add_expectation("expect_column_values_to_not_be_null", ge_df.expect_column_values_to_not_be_null, column)
        if "gender" in df.columns:
            add_expectation("expect_column_values_to_be_in_set", ge_df.expect_column_values_to_be_in_set, "gender", list(CATEGORICAL_MAP["gender"]))
        if "experience" in df.columns:
            add_expectation("expect_column_values_to_be_between", ge_df.expect_column_values_to_be_between, "experience", 0, 50, mostly=0.95)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        issues: List[ValidationIssue] = []
        if missing_columns:
            for column in missing_columns:
                issues.append({"column": column, "issue": "missing_column", "severity": "high"})

        invalid_indices: List[int] = []
        if not missing_columns:
            for idx, row in df.iterrows():
                row_issues = evaluate_row(row)
                if row_issues:
                    invalid_indices.append(idx)
                    issues.extend(row_issues)

        row_count = len(df)
        invalid_rows = len(set(invalid_indices))
        valid_rows = row_count - invalid_rows
        criticality = determine_criticality(invalid_rows, row_count)
        summary_text = (
            f"Processed {row_count} rows: {valid_rows} valid, {invalid_rows} invalid."
            if row_count
            else "Empty file"
        )
        expectation_summary = [
            {
                "name": result.get("expectation_config", {}).get("expectation_type", "unknown"),
                "success": result.get("success", False),
                "details": json.dumps(result.get("result", {})) if not result.get("success", False) else "",
            }
            for result in expectations
        ]
        report_path = build_html_report(path.name, {
            "criticality": criticality,
            "summary": summary_text,
            "row_count": row_count,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
        }, issues, expectation_summary)

        return {
            "file_path": str(path),
            "file_name": path.name,
            "row_count": row_count,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "criticality": criticality,
            "summary": summary_text,
            "issues": issues,
            "invalid_indices": invalid_indices,
            "report_path": report_path,
            "expectations": expectation_summary,
        }

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def send_alerts(payload: Dict[str, Any]) -> None:
        webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
        message = {
            "title": f"Data ingestion report for {payload['file_name']}",
            "criticality": payload["criticality"],
            "summary": payload["summary"],
            "report_link": payload["report_path"],
            "issues": aggregate_issues(payload["issues"]),
        }
        if webhook_url:
            try:
                import requests

                requests.post(webhook_url, json={"text": json.dumps(message, indent=2)}, timeout=10)
            except Exception as exc:  # pragma: no cover
                print(f"Failed to send Teams notification: {exc}")
        else:
            print("Notification payload:", json.dumps(message, indent=2))

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def split_and_save_data(payload: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(payload["file_path"])
        df = pd.read_csv(path)
        invalid_indices = payload["invalid_indices"]
        valid_df, invalid_df = _split_dataframe(df, invalid_indices)

        good_path: str | None = None
        bad_path: str | None = None

        if payload["invalid_rows"] == 0:
            dest = GOOD_DATA_DIR / path.name
            _move_file(path, dest)
            good_path = str(dest)
        elif payload["valid_rows"] == 0:
            dest = BAD_DATA_DIR / path.name
            _move_file(path, dest)
            bad_path = str(dest)
        else:
            good_path = str(GOOD_DATA_DIR / path.name)
            bad_path = str(BAD_DATA_DIR / path.name)
            GOOD_DATA_DIR.mkdir(parents=True, exist_ok=True)
            BAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
            valid_df.to_csv(good_path, index=False)
            invalid_df.to_csv(bad_path, index=False)
            path.unlink(missing_ok=True)

        payload.update({"good_file_path": good_path, "bad_file_path": bad_path})
        return payload

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def save_statistics(payload: Dict[str, Any]) -> None:
        session = get_db_session()
        try:
            statistic_id = save_statistics_record(
                session,
                {
                    "file_name": payload["file_name"],
                    "row_count": payload["row_count"],
                    "valid_rows": payload["valid_rows"],
                    "invalid_rows": payload["invalid_rows"],
                    "criticality": payload["criticality"],
                    "summary": payload["summary"],
                    "report_path": payload["report_path"],
                    "good_file_path": payload.get("good_file_path"),
                    "bad_file_path": payload.get("bad_file_path"),
                },
            )
            register_issues(session, statistic_id, payload["issues"], payload.get("expectations"))
        finally:
            session.close()

    file_path = read_data()
    validation = validate_data(file_path)
    enriched_payload = split_and_save_data(validation)
    send_alerts(enriched_payload)
    save_statistics(enriched_payload)

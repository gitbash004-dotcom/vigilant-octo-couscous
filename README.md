# HR Job Change MLOps Platform

This repository contains an end-to-end MLOps reference implementation for the Kaggle **HR Analytics: Job Change of Data Scientists** dataset. The solution is composed of the following containerised services:

| Component | Technology | Description |
|-----------|------------|-------------|
| Web application | Streamlit | Collect on-demand predictions (single or batch) and display past predictions. |
| Model API | FastAPI | Hosts the trained model, records predictions and exposes a past predictions endpoint. |
| Database | PostgreSQL | Stores predictions, ingestion statistics and data quality issues. |
| Ingestion pipeline | Apache Airflow + Great Expectations | Ingests sliced raw data, validates it, alerts on issues and stores clean/bad data splits. |
| Scheduled predictions | Apache Airflow | Checks for newly ingested data and triggers batch predictions via the API. |
| Monitoring | Grafana | Visualises ingestion stats, data quality issues and prediction volumes. |

The high-level data flow matches the specification provided in the assignment brief.

> **Note:** The dataset is not bundled in its entirety. A representative sample (`data/hr_data_sample.csv`) is provided for local testing and the ingestion scripts support the full Kaggle dataset once downloaded.

## Repository structure

```
.
├── airflow/                  # Custom Airflow image, DAGs and dependencies
├── data/                     # Base dataset sample used for development/testing
├── docker-compose.yml        # Orchestrates all services
├── grafana/                  # Grafana provisioning (datasource + dashboard)
├── models/                   # Folder where the trained model artefact is stored
├── notebooks/                # Notebook describing synthetic data issue generation
├── services/
│   ├── api/                  # FastAPI application + training pipeline
│   └── webapp/               # Streamlit UI
├── scripts/                  # Helper utilities (dataset splitter, etc.)
├── raw-data/                 # Source files for the ingestion DAG (mounted in Airflow)
├── good_data/                # Output folder for valid data produced by the ingestion DAG
├── bad_data/                 # Output folder for invalid rows produced by the ingestion DAG
└── great_expectations/       # Minimal GE configuration for validation reports
```

## Getting started

### 1. Prerequisites

* Docker Desktop (with **Docker Compose v2** support) installed and running
* Kaggle dataset downloaded locally (optional but recommended)

### 2. Prepare raw data

Split the Kaggle dataset (or the provided sample) into smaller files to simulate streaming ingestion:

```bash
python scripts/split_dataset.py --dataset data/hr_data_sample.csv --output-dir raw-data --num-files 5 --shuffle
```

The ingestion DAG will pick files randomly from `raw-data/` every minute.

### 3. Launch the stack

Build and start all services with Docker Compose:

```bash
docker compose up --build
```

The first run will download container images, install Python dependencies and apply the Airflow database migrations.

### 4. Access the services

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit web application | http://localhost:8501 | – |
| FastAPI docs | http://localhost:8000/docs | – |
| Airflow UI | http://localhost:8080 | `admin` / `admin` |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| PostgreSQL | localhost:5432 | `postgres` / `postgres` |

## Airflow pipelines

### Ingestion DAG – `ingest_hr_data`

* **Schedule:** every minute (`*/1 * * * *`)
* **Tasks:**
  1. `read_data`: pick a random CSV file from `raw-data/`.
  2. `validate_data`: run rule-based checks (enforced via Great Expectations) to detect seven types of data issues, compute statistics and produce an HTML report.
  3. `split_and_save_data`: move/split files into `good_data/` and `bad_data/` depending on row validity.
  4. `send_alerts`: log (or optionally post to Microsoft Teams via `TEAMS_WEBHOOK_URL`) a summary of the validation results with a link to the HTML report.
  5. `save_statistics`: persist ingestion statistics and detailed issues to PostgreSQL.

The task graph matches the brief (downstream tasks are fanned out from `validate_data`) and the `reports/` directory inside the Airflow container is exposed to Docker so the HTML reports can be inspected locally.

### Prediction DAG – `predict_hr_data`

* **Schedule:** every two minutes (`*/2 * * * *`).
* **Behaviour:**
  1. `check_for_new_data`: queries the ingestion statistics table for ingested files that have not yet been scored. If none are found the **entire DAG run is marked as skipped**, as requested.
  2. `make_predictions`: loads the good data files, calls the FastAPI `/predictions/predict` endpoint once per file and records the outputs in PostgreSQL via the API. Successfully processed statistics are timestamped (`predicted_at`).

## FastAPI model service

* Loads (or trains) a scikit-learn pipeline that predicts whether a candidate is actively looking for a job change.
* `/predictions/predict` accepts a list of feature dictionaries (single and batch prediction use the same endpoint).
* `/predictions/past` exposes historical predictions with optional date/source filters.
* All predictions are stored in PostgreSQL along with the raw features and probability scores.

## Streamlit web application

Two navigation tabs:

1. **Predict** – single prediction form and CSV upload for batch scoring. Results are rendered as a dataframe and echo back the input features.
2. **Past predictions** – date range selector and prediction source filter (`webapp`, `scheduled`, `api`, `all`). Fetches records from the FastAPI service.

Set a custom API endpoint by creating `.streamlit/secrets.toml` or editing the `API_URL` environment variable in `docker-compose.yml`.

## Monitoring with Grafana

Grafana is provisioned with a Postgres datasource and a dashboard (`HR Data Quality & Drift`) that tracks:

* Ingested row counts over time
* Latest data quality issues
* Prediction volume aggregated per hour

The dashboard files are hot-reloaded on startup and can be customised via the Grafana UI (changes will persist in the mounted volume).

## Generating data quality scenarios

Open `notebooks/generate_data_issues.ipynb` to review the seven error types that the ingestion DAG is designed to detect. The notebook explains how to craft faulty raw files (e.g. injecting missing values or invalid categorical levels) before copying them into the `raw-data/` folder.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FERNET_KEY` | autogenerated fallback in `docker-compose.yml` | Shared Airflow Fernet key. |
| `TEAMS_WEBHOOK_URL` | unset | When defined, `send_alerts` posts the validation summary to Microsoft Teams. |
| `API_URL` | `http://api:8000/predictions` | Web application API endpoint. |
| `DATABASE_URL` | `postgresql+psycopg2://postgres:postgres@db:5432/hr_predictions` | Shared database URL for services. |

## Development tips

* Run `python services/api/app/train_model.py` locally (inside a virtualenv with scikit-learn installed) to refresh the model artefact under `models/`.
* Update `scripts/split_dataset.py` to change the ingestion cadence or chunk size.
* Airflow DAGs and Grafana dashboards are mounted as bind volumes so edits are picked up without rebuilding containers.

## Testing the stack

1. Start the services.
2. Open the Streamlit UI to trigger a single prediction and confirm the record appears on the **Past predictions** tab.
3. Drop a few CSV files into `raw-data/` to observe the ingestion DAG move them into `good_data/`/`bad_data/`, log alerts and populate PostgreSQL.
4. Monitor the Grafana dashboard for ingestion and prediction metrics.

## License

This project is provided for educational purposes.

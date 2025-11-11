FROM apache/airflow:2.7.3-python3.10

USER root
RUN mkdir -p /opt/airflow/raw-data /opt/airflow/good_data /opt/airflow/bad_data /opt/airflow/reports
USER airflow

COPY airflow/requirements.txt /requirements.txt
RUN pip install --user --no-cache-dir -r /requirements.txt

COPY airflow/dags /opt/airflow/dags
COPY great_expectations /opt/airflow/great_expectations
COPY data /opt/airflow/data

ENV RAW_DATA_DIR=/opt/airflow/raw-data \
    GOOD_DATA_DIR=/opt/airflow/good_data \
    BAD_DATA_DIR=/opt/airflow/bad_data \
    REPORTS_DIR=/opt/airflow/reports \
    PREDICTION_API_URL=http://api:8000/predictions/predict \
    DATABASE_URL=postgresql+psycopg2://postgres:postgres@db:5432/hr_predictions

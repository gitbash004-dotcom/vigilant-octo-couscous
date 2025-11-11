FROM python:3.10-slim

WORKDIR /app

COPY services/api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY data /app/data
COPY models /app/models
COPY services/api/app /app/app

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

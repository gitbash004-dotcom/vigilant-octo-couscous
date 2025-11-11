FROM python:3.10-slim

WORKDIR /app

COPY services/webapp/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/webapp /app
COPY data /app/data

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "streamlit_app.py"]

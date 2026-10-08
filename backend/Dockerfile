FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts/models && \
    python -m ml.models.train && \
    python -m ml.models.survival && \
    python -m ml.models.train_multi && \
    python -m ml.models.challenger && \
    python -m ml.forecasting.revenue && \
    python -m ml.export_powerbi && \
    python -m ml.observability

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

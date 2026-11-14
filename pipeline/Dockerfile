# ============================================================
# CRIP Streaming Detection Pipeline Dockerfile
# ============================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pipeline/requirements.txt ./pipeline/requirements.txt
RUN pip install --no-cache-dir -r pipeline/requirements.txt

COPY pipeline/ ./pipeline/

RUN useradd -m -u 1000 crip
USER crip

ENV PIP_REDIS_URL=redis://redis:6379 \
    PIP_DB_DSN=postgresql://crip_user:crip_secret@timescaledb:5432/crip \
    PIP_METRICS_PORT=9200

EXPOSE 9200

CMD ["python", "-m", "pipeline.main"]


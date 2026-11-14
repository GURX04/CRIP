# ============================================================
# dashboard/Dockerfile
# Python 3.11-slim image for the CRIP demo dashboard backend.
# ============================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY dashboard/requirements.txt ./dashboard/requirements.txt
RUN pip install --no-cache-dir -r dashboard/requirements.txt

COPY dashboard/ ./dashboard/

RUN useradd -m -u 1000 crip
USER crip

ENV DASH_HOST=0.0.0.0 \
    DASH_PORT=8501 \
    DASH_REDIS_URL=redis://redis:6379 \
    DASH_GENERATOR_URL=http://generator:8000 \
    DASH_LOG_LEVEL=INFO

EXPOSE 8501

CMD ["python", "-m", "dashboard.main"]


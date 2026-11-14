# ============================================================
# CRIP Generator — Docker image
# Python 3.11-slim; runs the tick loop + FastAPI control API
# ============================================================
FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install OS-level dependencies for psycopg2 binary and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY generator/requirements.txt ./generator/requirements.txt
RUN pip install --no-cache-dir -r generator/requirements.txt

# Copy the generator package
COPY generator/ ./generator/

# Non-root user for security
RUN useradd -m -u 1000 crip
USER crip

# Default environment (overridden by docker-compose)
ENV GEN_SINK=redis \
    GEN_REDIS_URL=redis://redis:6379 \
    GEN_API_HOST=0.0.0.0 \
    GEN_API_PORT=8000 \
    GEN_SIM_MINUTES_PER_REAL_SECOND=5 \
    GEN_SEED=42

EXPOSE 8000

CMD ["python", "-m", "generator.main"]


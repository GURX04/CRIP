"""
target_app/main.py
==================
Lightweight FastAPI application — CRIP Phase 1 target, Phase 2 upgraded.

Phase 2 change: /metrics now exposes real Prometheus OpenMetrics text format
via prometheus_client, so Prometheus can scrape it directly.

Endpoints
---------
GET  /health          – liveness/readiness probe
GET  /api/data        – simulates a DynamoDB read (latency + error injection)
POST /api/ingest      – simulates an S3 PUT (write latency)
GET  /metrics         – Prometheus OpenMetrics exposition (text/plain)
"""

import asyncio
import random
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "crip_http_requests_total",
    "Total HTTP requests received by the CRIP target app.",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "crip_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ERROR_COUNT = Counter(
    "crip_http_errors_total",
    "Total HTTP 5xx errors from the CRIP target app.",
    ["endpoint"],
)

ACTIVE_REQUESTS = Gauge(
    "crip_active_requests",
    "Number of requests currently being processed.",
)

INGEST_BYTES = Counter(
    "crip_ingest_bytes_total",
    "Total payload bytes received by the ingest endpoint.",
)

APP_UPTIME = Gauge(
    "crip_app_uptime_seconds",
    "Seconds since the application started.",
)

_start_time: float = time.time()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Target application starting up (Phase 2 — Prometheus enabled)")
    yield
    print("Target application shutting down")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CRIP Target Application",
    description="Phase 1/2 load-test target with Prometheus instrumentation.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: instrument every request with timing + counters
# ---------------------------------------------------------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    endpoint = request.url.path
    method   = request.method

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()

    try:
        response = await call_next(request)
        status   = str(response.status_code)
        elapsed  = time.perf_counter() - start

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)

        return response
    except Exception as exc:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="500").inc()
        raise exc
    finally:
        ACTIVE_REQUESTS.dec()
        APP_UPTIME.set(time.time() - _start_time)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class IngestPayload(BaseModel):
    record_id: str
    payload: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health_check():
    """Liveness and readiness probe used by Docker and Locust."""
    uptime_seconds = round(time.time() - _start_time, 2)
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "uptime_seconds": uptime_seconds,
            "service": "crip-target-app",
            "version": "2.0.0",
        },
    )


@app.get("/api/data", tags=["api"])
async def fetch_data(item_id: str = "default"):
    """
    Simulates a read from a database (DynamoDB).
    Introduces realistic p50/p99 latency variance.
    """
    base_latency = random.uniform(0.005, 0.05)
    spike = random.uniform(0.15, 0.40) if random.random() < 0.05 else 0.0
    await asyncio.sleep(base_latency + spike)

    if random.random() < 0.01:
        ERROR_COUNT.labels(endpoint="/api/data").inc()
        raise HTTPException(status_code=500, detail="Simulated upstream error")

    return {
        "item_id": item_id,
        "value": random.randint(1, 10_000),
        "cached": random.random() > 0.3,
        "timestamp": time.time(),
    }


@app.post("/api/ingest", tags=["api"])
async def ingest_data(payload: IngestPayload):
    """
    Simulates writing a record to object storage (S3).
    Models the higher latency of PUT operations.
    """
    await asyncio.sleep(random.uniform(0.01, 0.08))
    byte_size = len(str(payload.payload))
    INGEST_BYTES.inc(byte_size)

    return {
        "status": "accepted",
        "record_id": payload.record_id,
        "size_bytes": byte_size,
        "timestamp": time.time(),
    }


@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics():
    """
    Prometheus OpenMetrics exposition endpoint (text/plain; version=0.0.4).
    Prometheus scrapes this on the configured interval.
    """
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Entry point (for local runs without Docker)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)

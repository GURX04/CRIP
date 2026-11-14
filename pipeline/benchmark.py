"""
pipeline/benchmark.py
=====================
Microbenchmark for the CRIP detector pipeline.
Feeds synthetic CostEvents directly into the EnsembleDetector and measures
events/sec and per-event latency.

Usage:
    python -m pipeline.benchmark --events 10000 --services 5
"""

from __future__ import annotations

import argparse
import statistics
import time
import uuid
from datetime import datetime, timezone

from pipeline.config import PipelineSettings
from pipeline.ensemble import EnsembleDetector
from pipeline.models import CostEvent

SERVICES = [
    "AmazonApiGateway",
    "EC2 Data Transfer",
    "DynamoDB",
    "CloudWatch Logs",
    "S3 PUT",
]


def make_event(
    service: str, tick: int, run_id: str, cost_mult: float = 1.0
) -> CostEvent:
    sim_h = 10 + (tick // 12) % 14
    sim_m = (tick * 5) % 60
    return CostEvent(
        run_id=run_id,
        event_id=str(uuid.uuid4()),
        tick_index=tick,
        sim_ts=datetime(2026, 11, 12, sim_h, sim_m, tzinfo=timezone.utc),
        wall_ts_ms=int(time.time() * 1000),
        service=service,
        request_count=5000.0 + tick * 0.1,
        usage_amount=5000.0 + tick * 0.1,
        unblended_cost=(5000.0 + tick * 0.1) * 0.0000035 * cost_mult,
        unit_price=0.0000035,
    )


def run_benchmark(n_events: int) -> None:
    settings = PipelineSettings()
    ensemble = EnsembleDetector(settings)
    run_id = str(uuid.uuid4())
    latencies: list[float] = []

    print(
        f"Benchmarking {n_events} events across {len(SERVICES)} services..."
    )
    t0 = time.perf_counter()
    for i in range(n_events):
        svc = SERVICES[i % len(SERVICES)]
        event = make_event(svc, i // len(SERVICES), run_id)
        t_start = time.perf_counter()
        ensemble.process(event)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    total_s = time.perf_counter() - t0
    sorted_lats = sorted(latencies)
    mean_lat = statistics.mean(latencies)
    p50_lat = statistics.median(latencies)
    p95_lat = sorted_lats[int(0.95 * len(latencies))]
    p99_lat = sorted_lats[int(0.99 * len(latencies))]

    print(f"Total events : {n_events}")
    print(f"Total time   : {total_s:.3f}s")
    print(f"Throughput   : {n_events / total_s:.0f} events/sec")
    print(f"Mean latency : {mean_lat:.3f} ms")
    print(f"p50 latency  : {p50_lat:.3f} ms")
    print(f"p95 latency  : {p95_lat:.3f} ms")
    print(f"p99 latency  : {p99_lat:.3f} ms")
    target_ok = mean_lat < 5.0
    print(f"Target (<5ms): {'PASS' if target_ok else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector pipeline benchmark")
    parser.add_argument("--events", type=int, default=10_000)
    args = parser.parse_args()
    run_benchmark(args.events)


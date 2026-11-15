"""
generator/main.py
=================
Entry point for the CRIP continuous event generator.

Starts two concurrent components:
1. Tick loop        — monotonic scheduler; runs in the main thread.
2. FastAPI server   — uvicorn; runs in a background thread.

Graceful shutdown on SIGINT / SIGTERM: tick loop exits, server is stopped.

Usage
-----
    python -m generator.main
    python -m generator.main --sink stdout
    python -m generator.main --sink csv --csv-path reports/gen.csv
    python -m generator.main --seed 99 --speed 10
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import structlog
import uvicorn

from generator.config import GeneratorSettings, get_settings, reset_settings
from generator.clock import SimClock
from generator.traffic_model import TrafficModel
from generator.anomalies import InjectionRegistry
from generator.mitigations import MitigationRegistry
from generator.publisher import Publisher
from generator.api import create_app

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared shutdown event
# ---------------------------------------------------------------------------
_shutdown = threading.Event()


def _signal_handler(signum, frame):
    log.info("shutdown_signal", signal=signum)
    _shutdown.set()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRIP continuous event generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sink",
        choices=["redis", "stdout", "csv"],
        default=None,
        help="Output sink (overrides GEN_SINK env var)",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="CSV output path (overrides GEN_CSV_PATH)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides GEN_SEED)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Sim minutes per real second (overrides GEN_SIM_MINUTES_PER_REAL_SECOND)",
    )
    parser.add_argument(
        "--start-ts",
        default=None,
        help="Sim start timestamp ISO-8601 UTC (overrides GEN_SIM_START_TS)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        default=False,
        help="Disable the FastAPI control server",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """Launch tick loop + API server."""
    args = _parse_args(argv)

    # Apply CLI overrides to environment before settings are loaded
    if args.sink:
        os.environ["GEN_SINK"] = args.sink
    if args.csv_path:
        os.environ["GEN_CSV_PATH"] = args.csv_path
    if args.seed is not None:
        os.environ["GEN_SEED"] = str(args.seed)
    if args.speed is not None:
        os.environ["GEN_SIM_MINUTES_PER_REAL_SECOND"] = str(args.speed)
    if args.start_ts:
        os.environ["GEN_SIM_START_TS"] = args.start_ts

    settings = reset_settings()

    # Register signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Build sim start datetime
    sim_start: Optional[datetime] = None
    if settings.sim_start_ts:
        sim_start = datetime.fromisoformat(settings.sim_start_ts)
        if sim_start.tzinfo is None:
            sim_start = sim_start.replace(tzinfo=timezone.utc)

    # Shared components
    clock = SimClock(
        sim_start_ts=sim_start,
        sim_minutes_per_real_second=settings.sim_minutes_per_real_second,
        seed=settings.seed,
    )
    rng = np.random.default_rng(settings.seed)
    traffic_model = TrafficModel(rng=rng, settings=settings)
    registry = InjectionRegistry()
    mitigation_registry = MitigationRegistry()
    publisher = Publisher(settings=settings, run_id=clock.run_id)

    # Wire ground-truth and mitigation callbacks
    registry.on_start(publisher.write_injection_start)
    registry.on_end(publisher.write_injection_end)
    mitigation_registry.on_start(publisher.write_mitigation_start)
    mitigation_registry.on_end(publisher.write_mitigation_end)

    log.info(
        "generator_starting",
        run_id=clock.run_id,
        seed=settings.seed,
        sink=settings.sink,
        speed=settings.sim_minutes_per_real_second,
    )

    # Start API server in a background thread
    api_thread: Optional[threading.Thread] = None
    uvi_server: Optional[uvicorn.Server] = None

    if not args.no_api:
        app = create_app(
            clock=clock,
            registry=registry,
            publisher=publisher,
            mitigation_registry=mitigation_registry,
        )
        uvi_config = uvicorn.Config(
            app=app,
            host=settings.api_host,
            port=settings.api_port,
            log_level="warning",   # uvicorn noise suppressed; we use structlog
        )
        uvi_server = uvicorn.Server(config=uvi_config)

        def _run_api():
            uvi_server.run()

        api_thread = threading.Thread(target=_run_api, name="api-server", daemon=True)
        api_thread.start()
        log.info("api_started", host=settings.api_host, port=settings.api_port)

    # -----------------------------------------------------------------------
    # Tick loop — monotonic scheduler
    # -----------------------------------------------------------------------
    try:
        _tick_loop(clock, traffic_model, registry, mitigation_registry, publisher, settings)
    finally:
        log.info("generator_shutting_down", run_id=clock.run_id)
        if uvi_server:
            uvi_server.should_exit = True
        publisher.close()
        log.info("generator_stopped")


def _tick_loop(
    clock: SimClock,
    traffic_model: TrafficModel,
    registry: InjectionRegistry,
    mitigation_registry: MitigationRegistry,
    publisher: Publisher,
    settings: GeneratorSettings,
) -> None:
    """
    Main tick loop.  Runs until _shutdown is set.

    Uses ``SimClock.wait_for_next_tick`` which compensates for processing
    drift so the tick rate stays accurate under load.
    """
    while not _shutdown.is_set():
        fired = clock.wait_for_next_tick()
        if not fired:
            # Paused — sleep briefly and re-check
            time.sleep(0.05)
            continue

        tick_state = clock.advance_tick()

        # 1. Baseline traffic
        tick_state = traffic_model.populate(tick_state)

        # 2. Apply anomaly injections
        tick_state = registry.apply_all(tick_state)

        # 3. Apply active mitigations (rate limits / throttles)
        tick_state = mitigation_registry.apply_all(tick_state)

        # 4. Publish events (and labels)
        try:
            publisher.emit_tick(tick_state)
        except Exception as exc:
            log.error(
                "tick_emit_failed",
                tick_index=tick_state.tick_index,
                error=str(exc),
                run_id=tick_state.run_id,
            )

        log.debug(
            "tick_emitted",
            tick_index=tick_state.tick_index,
            sim_ts=tick_state.sim_ts.isoformat(),
            run_id=tick_state.run_id,
            active_injections=registry.active_ids(),
        )


if __name__ == "__main__":
    main()


"""
pipeline/main.py
================
Entry point for the CRIP streaming detector worker.

Usage:
    python -m pipeline.main
    # or via CLI:
    crip detect-live
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from pipeline.config import get_settings
from pipeline.consumer import Consumer
from pipeline.db_writer import DBWriter
from pipeline.ensemble import EnsembleDetector
from pipeline.publisher import Publisher

try:
    import structlog

    logger = structlog.get_logger(__name__)
    _use_structlog = True
except ImportError:
    logger = logging.getLogger(__name__)
    _use_structlog = False


def _log(level: str, msg: str, **kwargs):
    if _use_structlog:
        getattr(logger, level)(msg, **kwargs)
    else:
        extra = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        getattr(logger, level)("%s %s", msg, extra)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CRIP streaming detector")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    settings = get_settings()
    stop_event = threading.Event()

    def _handle_signal(sig, frame):
        _log("info", "shutdown_signal_received", signal=sig)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    db_writer = DBWriter(settings)
    db_writer.start()

    publisher = Publisher(settings)
    ensemble = EnsembleDetector(settings)
    consumer = Consumer(settings, ensemble, publisher, db_writer)

    try:
        consumer.setup_consumer_group()
        _log(
            "info",
            "detector_started",
            consumer=settings.consumer_name,
            group=settings.consumer_group,
            metrics_port=settings.metrics_port,
        )
        consumer.run(stop_event)
    except Exception as exc:
        _log("error", "detector_error", error=str(exc))
        raise
    finally:
        _log("info", "detector_shutting_down")
        consumer.close()
        publisher.close()
        db_writer.stop(timeout=10.0)
        _log("info", "detector_stopped")


def _setup_logging(level: str) -> None:
    if _use_structlog:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.BoundLogger,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
        )


if __name__ == "__main__":
    main()


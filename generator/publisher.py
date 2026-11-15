"""
generator/publisher.py
======================
Event and ground-truth publishing for the CRIP event generator.

Responsibilities
----------------
1. After every tick: emit one JSON event per service to the configured sink
   (Redis XADD, stdout, or CSV).
2. After every tick: emit one label record per service to crip:labels (Redis
   only; labels are never mixed into crip:events so detectors can't cheat).
3. When an injection starts or ends: write ground truth to
   - Redis stream crip:injections
   - TimescaleDB table injections
   - reports/ground_truth.jsonl (append)

Redis reconnection
------------------
Uses exponential back-off (1 → 2 → 4 → 8 → 16 s, cap 30 s) with structured
logging.  The publisher raises after MAX_RECONNECT_ATTEMPTS to let main.py
decide whether to abort or continue with stdout fallback.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import structlog

from generator.clock import TickState
from generator.config import GeneratorSettings

if TYPE_CHECKING:
    import redis as redis_lib

log = structlog.get_logger(__name__)

MAX_RECONNECT_ATTEMPTS = 10
BACKOFF_BASE = 1.0
BACKOFF_MAX = 30.0


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

class Publisher:
    """
    Handles all output for the generator.

    Parameters
    ----------
    settings:
        Generator settings (sink, Redis URL, DB DSN, paths …).
    run_id:
        Current run UUID.  Updated on reset via ``update_run_id``.
    """

    def __init__(self, settings: GeneratorSettings, run_id: str) -> None:
        self._settings = settings
        self._run_id = run_id
        self._redis: Optional["redis_lib.Redis"] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self._db_conn = None
        self._backoff_attempt = 0

        self._setup_sink()
        self._setup_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_run_id(self, run_id: str) -> None:
        """Call after a reset so new events carry the new run_id."""
        self._run_id = run_id

    def emit_tick(self, tick_state: TickState) -> None:
        """
        Emit one event and one label per service for this tick.

        Events go to crip:events (or stdout/csv).
        Labels go to crip:labels (Redis only).
        """
        wall_ts_ms = _now_ms()
        events = []
        labels = []

        for svc_name, sd in tick_state.services.items():
            sd.finalise()
            event = {
                "run_id": self._run_id,
                "event_id": str(uuid.uuid4()),
                "tick_index": tick_state.tick_index,
                "sim_ts": tick_state.sim_ts.isoformat(),
                "wall_ts_ms": wall_ts_ms,
                "service": svc_name,
                "request_count": round(sd.request_count * sd.volume_multiplier, 4),
                "usage_amount": round(sd.usage_amount, 4),
                "unblended_cost": round(sd.unblended_cost, 6),
                "unit_price": round(sd.unit_price * sd.price_multiplier, 8),
            }
            label = {
                "run_id": self._run_id,
                "tick_index": tick_state.tick_index,
                "sim_ts": tick_state.sim_ts.isoformat(),
                "service": svc_name,
                "is_anomalous": sd.is_anomalous,
                "injection_ids": sd.injection_ids,
            }
            events.append(event)
            labels.append(label)

        sink = self._settings.sink
        if sink == "redis":
            self._redis_emit(events, labels)
        elif sink == "stdout":
            for ev in events:
                print(json.dumps(ev))
        elif sink == "csv":
            self._csv_emit(events)

    def write_injection_start(self, inj) -> None:
        """Record injection start to all ground-truth sinks."""
        record = self._injection_record(inj, event="start")
        self._write_ground_truth(record)
        log.info(
            "injection_started",
            injection_id=inj.injection_id,
            type=record["type"],
            service=record["service"],
            run_id=self._run_id,
        )

    def write_injection_end(self, inj) -> None:
        """Record injection end to all ground-truth sinks."""
        record = self._injection_record(inj, event="end")
        self._write_ground_truth(record)
        log.info(
            "injection_ended",
            injection_id=inj.injection_id,
            type=record["type"],
            run_id=self._run_id,
        )

    def write_mitigation_start(self, mitigation) -> None:
        """Record mitigation start to Redis stream."""
        record = mitigation.to_dict()
        record["event"] = "start"
        record["run_id"] = self._run_id
        self._mitigation_redis(record)
        log.info(
            "mitigation_started",
            mitigation_id=mitigation.mitigation_id,
            service=record["service"],
            factor=record["factor"],
            run_id=self._run_id,
        )

    def write_mitigation_end(self, mitigation) -> None:
        """Record mitigation end to Redis stream."""
        record = mitigation.to_dict()
        record["event"] = "end"
        record["run_id"] = self._run_id
        self._mitigation_redis(record)
        log.info(
            "mitigation_ended",
            mitigation_id=mitigation.mitigation_id,
            service=record["service"],
            run_id=self._run_id,
        )

    def _mitigation_redis(self, record: dict) -> None:
        if self._redis is None:
            return
        try:
            stream_name = getattr(self._settings, "redis_stream_mitigations", "crip:mitigations")
            self._redis.xadd(
                stream_name,
                {"payload": json.dumps(record)},
                maxlen=10_000,
                approximate=True,
            )
        except Exception as exc:
            log.error("mitigation_redis_failed", error=str(exc))

    def close(self) -> None:
        """Flush and close all open resources."""
        if self._csv_file:
            self._csv_file.close()
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_sink(self) -> None:
        s = self._settings
        if s.sink == "redis":
            self._connect_redis()
        elif s.sink == "csv":
            self._open_csv()

    def _connect_redis(self) -> None:
        """Attempt Redis connection with exponential back-off."""
        import redis as redis_lib  # noqa: PLC0415

        attempt = 0
        backoff = BACKOFF_BASE
        while attempt < MAX_RECONNECT_ATTEMPTS:
            try:
                r = redis_lib.from_url(
                    self._settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                r.ping()
                self._redis = r
                self._backoff_attempt = 0
                log.info("redis_connected", url=self._settings.redis_url)
                return
            except Exception as exc:
                attempt += 1
                log.warning(
                    "redis_connect_failed",
                    attempt=attempt,
                    backoff_s=backoff,
                    error=str(exc),
                )
                if attempt >= MAX_RECONNECT_ATTEMPTS:
                    raise ConnectionError(
                        f"Could not connect to Redis after {attempt} attempts"
                    ) from exc
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)

    def _open_csv(self) -> None:
        path = Path(self._settings.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "run_id", "event_id", "tick_index", "sim_ts", "wall_ts_ms",
            "service", "request_count", "usage_amount", "unblended_cost", "unit_price",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def _setup_db(self) -> None:
        """Connect to TimescaleDB for ground-truth injection records."""
        try:
            import psycopg2  # noqa: PLC0415
            self._db_conn = psycopg2.connect(self._settings.db_dsn, connect_timeout=1)
            self._db_conn.autocommit = True
            log.info("timescaledb_connected")
        except Exception as exc:
            log.warning("timescaledb_unavailable", error=str(exc))
            self._db_conn = None

    # ------------------------------------------------------------------
    # Emission helpers
    # ------------------------------------------------------------------

    def _redis_emit(self, events: list[dict], labels: list[dict]) -> None:
        """XADD events and labels to Redis streams."""
        if self._redis is None:
            return
        try:
            pipe = self._redis.pipeline(transaction=False)
            maxlen = self._settings.redis_maxlen
            for ev in events:
                pipe.xadd(
                    self._settings.redis_stream_events,
                    {"payload": json.dumps(ev)},
                    maxlen=maxlen,
                    approximate=True,
                )
            for lbl in labels:
                pipe.xadd(
                    self._settings.redis_stream_labels,
                    {"payload": json.dumps(lbl)},
                    maxlen=maxlen,
                    approximate=True,
                )
            pipe.execute()
        except Exception as exc:
            log.error("redis_emit_failed", error=str(exc))
            # Attempt reconnect for the next tick
            try:
                self._connect_redis()
            except Exception:
                pass

    def _csv_emit(self, events: list[dict]) -> None:
        if self._csv_writer:
            for ev in events:
                self._csv_writer.writerow(ev)
            self._csv_file.flush()

    # ------------------------------------------------------------------
    # Ground truth helpers
    # ------------------------------------------------------------------

    def _injection_record(self, inj, event: str) -> dict:
        d = inj.to_dict()
        d["event"] = event          # "start" | "end"
        d["run_id"] = self._run_id
        return d

    def _write_ground_truth(self, record: dict) -> None:
        """Write to Redis stream, TimescaleDB, and JSONL file."""
        self._gt_redis(record)
        self._gt_db(record)
        self._gt_jsonl(record)

    def _gt_redis(self, record: dict) -> None:
        if self._redis is None:
            return
        try:
            self._redis.xadd(
                self._settings.redis_stream_injections,
                {"payload": json.dumps(record)},
                maxlen=10_000,
                approximate=True,
            )
        except Exception as exc:
            log.error("gt_redis_failed", error=str(exc))

    def _gt_db(self, record: dict) -> None:
        if self._db_conn is None:
            return
        try:
            with self._db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO injections (
                        injection_id, run_id, type, service, magnitude,
                        duration_min, sim_start_ts, sim_end_ts,
                        wall_start_ms, wall_end_ms, params
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (injection_id) DO UPDATE SET
                        sim_end_ts   = EXCLUDED.sim_end_ts,
                        wall_end_ms  = EXCLUDED.wall_end_ms,
                        params       = EXCLUDED.params
                    """,
                    (
                        record["injection_id"],
                        record["run_id"],
                        record.get("type", "unknown"),
                        record["service"],
                        record["magnitude"],
                        record["duration_min"],
                        record.get("sim_start_ts"),
                        record.get("sim_end_ts"),
                        record.get("wall_start_ms", 0),
                        record.get("wall_end_ms", 0),
                        json.dumps(record),
                    ),
                )
        except Exception as exc:
            log.error("gt_db_failed", error=str(exc))

    def _gt_jsonl(self, record: dict) -> None:
        path = Path(self._settings.ground_truth_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            log.error("gt_jsonl_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


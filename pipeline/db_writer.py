"""
db_writer.py — Async-safe, non-blocking batched writer to TimescaleDB.

Writes to three tables: cost_events, detections, detector_scores.
The detector loop is never blocked waiting for DB writes: all enqueue_*
calls are non-blocking and drop (with a warning) when the internal queue
is full.

A single daemon thread runs _flush_loop(), which wakes every
db_flush_interval_ms milliseconds OR when the queue reaches
db_flush_batch_size rows, whichever comes first, and executes a
psycopg2 execute_values batch insert.

On psycopg2 errors the rows are put back into the queue and the writer
retries after 2 seconds.  On DB disconnect the writer reconnects with
exponential back-off (1 → 2 → 4 → 8 → 16 → 30 s max).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

try:
    import structlog

    _base_logger = structlog.get_logger(__name__)
    _use_structlog = True
except ImportError:  # pragma: no cover
    _base_logger = logging.getLogger(__name__)
    _use_structlog = False

from pipeline.config import PipelineSettings
from pipeline.models import CostEvent, Detection, ScorePoint

# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_SQL_COST_EVENTS = """
INSERT INTO cost_events (
    sim_ts, wall_ts_ms, run_id, event_id, tick_index, service,
    request_count, usage_amount, unblended_cost, unit_price
)
VALUES %s
ON CONFLICT (sim_ts, event_id) DO NOTHING
"""

_SQL_DETECTIONS = """
INSERT INTO detections (
    detect_ts, detection_id, run_id, service, sim_ts,
    detect_wall_ts_ms, first_event_wall_ts_ms, severity,
    detector_votes, rca, status
)
VALUES %s
ON CONFLICT (detect_ts, detection_id) DO UPDATE SET status = EXCLUDED.status
"""

_SQL_DETECTOR_SCORES = """
INSERT INTO detector_scores (
    sim_ts, run_id, service, cost, expected_cost,
    score_zscore, score_ewma, score_cpr
)
VALUES %s
"""

# Map table name → SQL
_TABLE_SQL: Dict[str, str] = {
    "cost_events": _SQL_COST_EVENTS,
    "detections": _SQL_DETECTIONS,
    "detector_scores": _SQL_DETECTOR_SCORES,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Row = Tuple[Any, ...]
_Item = Tuple[str, _Row]  # (table_name, row_tuple)

_BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 30]  # seconds, capped at 30


def _log(level: str, msg: str, **kw: Any) -> None:
    """Emit a structured or stdlib log line."""
    if _use_structlog:
        getattr(_base_logger, level)(msg, **kw)
    else:
        extra = " ".join(f"{k}={v!r}" for k, v in kw.items())
        getattr(_base_logger, level)("%s %s", msg, extra)


def _event_to_row(event: CostEvent) -> _Row:
    """Convert a CostEvent to its DB row tuple."""
    return (
        event.sim_ts,
        event.wall_ts_ms,
        event.run_id,
        event.event_id,
        event.tick_index,
        event.service,
        event.request_count,
        event.usage_amount,
        event.unblended_cost,
        event.unit_price,
    )


def _detection_to_row(det: Detection) -> _Row:
    """Convert a Detection to its DB row tuple.

    detect_ts maps to det.sim_ts (the simulation timestamp at detection time).
    """
    return (
        det.sim_ts,                                                        # detect_ts
        det.detection_id,
        det.run_id,
        det.service,
        det.sim_ts,                                                        # sim_ts
        det.detect_wall_ts_ms,
        det.first_event_wall_ts_ms,
        det.severity,
        json.dumps([r.__dict__ for r in det.detector_votes]),              # detector_votes (JSONB)
        json.dumps(det.rca.__dict__),                                      # rca (JSONB)
        det.status,
    )


def _scorepoint_to_row(sp: ScorePoint) -> _Row:
    """Convert a ScorePoint to its DB row tuple."""
    return (
        sp.sim_ts,
        sp.run_id,
        sp.service,
        sp.cost,
        sp.expected_cost,
        sp.score_zscore,
        sp.score_ewma,
        sp.score_cpr,
    )


# ---------------------------------------------------------------------------
# DBWriter
# ---------------------------------------------------------------------------


class DBWriter:
    """Non-blocking batched writer to TimescaleDB.

    Usage::

        writer = DBWriter(settings)
        writer.start()
        writer.enqueue_event(cost_event)
        writer.enqueue_detection(detection)
        writer.enqueue_scores(score_point)
        # … at shutdown:
        writer.stop()

    All public enqueue_* methods are non-blocking.  If the internal queue
    is full the row is *dropped* and a warning is logged so that the
    detector loop is never stalled.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """Initialise the writer with pipeline settings.

        Does *not* open a DB connection; that happens inside the
        background thread so the constructor cannot block.

        Args:
            settings: Validated pipeline configuration object.
        """
        self._settings = settings
        self._queue: queue.Queue[_Item] = queue.Queue(
            maxsize=settings.db_queue_maxsize
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._conn: Optional[psycopg2.extensions.connection] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background flush thread.

        Safe to call once.  Subsequent calls are no-ops.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="db-writer-flush",
            daemon=True,
        )
        self._thread.start()
        _log("info", "DBWriter started", dsn=self._settings.db_dsn)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the flush thread to stop, flush remaining rows, then join.

        Args:
            timeout: Seconds to wait for the thread to finish.
        """
        _log("info", "DBWriter stopping", timeout=timeout)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                _log("warning", "DBWriter thread did not finish within timeout",
                     timeout=timeout)
        self._close_conn()
        _log("info", "DBWriter stopped")

    def enqueue_event(self, event: CostEvent) -> None:
        """Enqueue a CostEvent for batch insert into cost_events.

        Non-blocking: if the queue is full the row is dropped with a warning.

        Args:
            event: The cost event to persist.
        """
        self._enqueue("cost_events", _event_to_row(event))

    def enqueue_detection(self, detection: Detection) -> None:
        """Enqueue a Detection for batch insert into detections.

        Non-blocking: if the queue is full the row is dropped with a warning.

        Args:
            detection: The detection result to persist.
        """
        self._enqueue("detections", _detection_to_row(detection))

    def enqueue_scores(self, sp: ScorePoint) -> None:
        """Enqueue a ScorePoint for batch insert into detector_scores.

        Non-blocking: if the queue is full the row is dropped with a warning.

        Args:
            sp: The score point to persist.
        """
        self._enqueue("detector_scores", _scorepoint_to_row(sp))

    def queue_depth(self) -> int:
        """Return the current number of rows waiting to be flushed.

        Intended for use by a Prometheus gauge.

        Returns:
            Approximate queue size (psycopg2 Queue.qsize is advisory).
        """
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, table: str, row: _Row) -> None:
        """Put a (table, row) item on the queue without blocking.

        Drops the item and logs a warning if the queue is at capacity.
        """
        try:
            self._queue.put_nowait((table, row))
        except queue.Full:
            _log(
                "warning",
                "DBWriter queue full — dropping row",
                table=table,
                queue_maxsize=self._settings.db_queue_maxsize,
            )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Open a fresh psycopg2 connection, retrying with exponential back-off.

        Blocks until a connection is established or the stop event is set.
        """
        attempt = 0
        while not self._stop_event.is_set():
            delay = _BACKOFF_SEQUENCE[min(attempt, len(_BACKOFF_SEQUENCE) - 1)]
            try:
                _log(
                    "info",
                    "DBWriter connecting",
                    attempt=attempt + 1,
                    dsn=self._settings.db_dsn,
                )
                conn = psycopg2.connect(self._settings.db_dsn)
                conn.autocommit = False
                self._conn = conn
                self._ensure_schema()
                _log("info", "DBWriter connected", attempt=attempt + 1)
                return
            except psycopg2.OperationalError as exc:
                _log(
                    "warning",
                    "DBWriter connection failed — backing off",
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=str(exc),
                )
                self._stop_event.wait(timeout=float(delay))
                attempt += 1

    def _ensure_schema(self) -> None:
        """Create tables and hypertables if they do not exist yet."""
        if self._conn is None:
            return
        ddl = """
        CREATE TABLE IF NOT EXISTS cost_events (
            sim_ts              TIMESTAMPTZ         NOT NULL,
            wall_ts_ms          BIGINT              NOT NULL,
            run_id              TEXT                NOT NULL,
            event_id            TEXT                NOT NULL,
            tick_index          INT                 NOT NULL,
            service             TEXT                NOT NULL,
            request_count       DOUBLE PRECISION    NOT NULL DEFAULT 0,
            usage_amount        DOUBLE PRECISION    NOT NULL DEFAULT 0,
            unblended_cost      DOUBLE PRECISION    NOT NULL DEFAULT 0,
            unit_price          DOUBLE PRECISION    NOT NULL DEFAULT 0,
            created_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
            CONSTRAINT pk_cost_events PRIMARY KEY (sim_ts, event_id)
        );
        CREATE TABLE IF NOT EXISTS detections (
            detect_ts               TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
            detection_id            TEXT                NOT NULL,
            run_id                  TEXT                NOT NULL,
            service                 TEXT                NOT NULL,
            sim_ts                  TIMESTAMPTZ         NOT NULL,
            detect_wall_ts_ms       BIGINT              NOT NULL,
            first_event_wall_ts_ms  BIGINT              NOT NULL,
            severity                DOUBLE PRECISION    NOT NULL,
            detector_votes          JSONB               NOT NULL DEFAULT '[]'::jsonb,
            rca                     JSONB               NOT NULL DEFAULT '{}'::jsonb,
            status                  TEXT                NOT NULL DEFAULT 'open',
            CONSTRAINT pk_detections PRIMARY KEY (detect_ts, detection_id)
        );
        CREATE TABLE IF NOT EXISTS detector_scores (
            sim_ts          TIMESTAMPTZ         NOT NULL,
            run_id          TEXT                NOT NULL,
            service         TEXT                NOT NULL,
            cost            DOUBLE PRECISION    NOT NULL DEFAULT 0,
            expected_cost   DOUBLE PRECISION    NOT NULL DEFAULT 0,
            score_zscore    DOUBLE PRECISION    NOT NULL DEFAULT 0,
            score_ewma      DOUBLE PRECISION    NOT NULL DEFAULT 0,
            score_cpr       DOUBLE PRECISION    NOT NULL DEFAULT 0
        );
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(ddl)
                try:
                    cur.execute("SELECT create_hypertable('cost_events', 'sim_ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);")
                    cur.execute("SELECT create_hypertable('detections', 'detect_ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);")
                    cur.execute("SELECT create_hypertable('detector_scores', 'sim_ts', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);")
                except Exception:
                    pass
            self._conn.commit()
        except Exception as exc:
            if self._conn:
                self._conn.rollback()
            _log("warning", "DBWriter schema check notice", error=str(exc))

    def _close_conn(self) -> None:
        """Close the psycopg2 connection silently."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    def _ensure_conn(self) -> bool:
        """Return True if the connection is open, reconnect if needed.

        Returns:
            True when a live connection is available, False when the stop
            event fired during reconnection.
        """
        if self._conn is not None:
            try:
                # Lightweight liveness check
                self._conn.cursor().execute("SELECT 1")
                return True
            except psycopg2.Error:
                _log("warning", "DBWriter lost connection — reconnecting")
                self._close_conn()

        self._connect()
        return self._conn is not None

    # ------------------------------------------------------------------
    # Flush loop
    # ------------------------------------------------------------------

    def _drain_queue(self) -> List[_Item]:
        """Drain up to db_flush_batch_size items from the queue.

        Returns:
            A list of (table_name, row_tuple) items.
        """
        batch: List[_Item] = []
        limit = self._settings.db_flush_batch_size
        while len(batch) < limit:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _flush_batch(self, batch: List[_Item]) -> bool:
        """Write a batch of rows to the DB.

        Groups items by table and issues one execute_values call per table.

        Args:
            batch: List of (table_name, row_tuple) items.

        Returns:
            True on success, False on psycopg2 error (caller puts rows back).
        """
        if not batch:
            return True

        # Group by table
        by_table: Dict[str, List[_Row]] = {}
        for table, row in batch:
            by_table.setdefault(table, []).append(row)

        try:
            assert self._conn is not None  # guaranteed by caller
            with self._conn.cursor() as cur:
                for table, rows in by_table.items():
                    sql = _TABLE_SQL[table]
                    psycopg2.extras.execute_values(cur, sql, rows)
            self._conn.commit()
            _log(
                "debug",
                "DBWriter flushed batch",
                total_rows=len(batch),
                tables=list(by_table.keys()),
            )
            return True

        except psycopg2.Error as exc:
            _log(
                "error",
                "DBWriter batch write failed",
                error=str(exc),
                rows_in_batch=len(batch),
            )
            try:
                self._conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            return False

    def _put_back(self, batch: List[_Item]) -> None:
        """Re-queue items that could not be written.

        Items that no longer fit (queue full) are dropped with a warning.
        """
        requeued = 0
        dropped = 0
        for item in batch:
            try:
                self._queue.put_nowait(item)
                requeued += 1
            except queue.Full:
                dropped += 1
        if dropped:
            _log(
                "warning",
                "DBWriter could not re-queue all rows after error",
                requeued=requeued,
                dropped=dropped,
            )

    def _flush_loop(self) -> None:
        """Main loop executed by the background thread.

        Wakes every db_flush_interval_ms ms or when the queue is full to
        the batch size, whichever happens first.  Handles reconnection and
        final flush on shutdown.
        """
        interval_s = self._settings.db_flush_interval_ms / 1000.0
        batch_size = self._settings.db_flush_batch_size

        # Establish initial connection (waits with backoff if DB is down)
        if not self._ensure_conn():
            return  # stop was set during initial connect

        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed = now - last_flush
            queue_len = self._queue.qsize()

            should_flush = (
                elapsed >= interval_s or queue_len >= batch_size
            )

            if not should_flush:
                # Sleep a short slice so we can re-evaluate quickly
                sleep_s = max(0.0, interval_s - elapsed)
                self._stop_event.wait(timeout=min(sleep_s, 0.05))
                continue

            batch = self._drain_queue()
            last_flush = time.monotonic()

            if not batch:
                continue

            if not self._ensure_conn():
                # stop was fired while reconnecting; put rows back for final flush
                self._put_back(batch)
                break

            success = self._flush_batch(batch)
            if not success:
                # Reconnect and retry once after 2 s
                _log("info", "DBWriter retrying failed batch after 2s")
                self._stop_event.wait(timeout=2.0)
                self._close_conn()
                if not self._ensure_conn():
                    self._put_back(batch)
                    break
                retry_ok = self._flush_batch(batch)
                if not retry_ok:
                    self._put_back(batch)
                    _log("error", "DBWriter batch retry also failed — rows re-queued")

        # --- Shutdown flush ---
        _log("info", "DBWriter flushing remaining rows on shutdown")
        remaining = self._drain_queue()
        if remaining and self._conn is not None:
            self._flush_batch(remaining)
        _log("info", "DBWriter flush loop exited")

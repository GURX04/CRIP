"""
consumer.py
-----------
Redis consumer-group loop that reads from `crip:events`, processes each event
through the EnsembleDetector, and publishes results via Publisher and DBWriter.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Optional

import redis
import redis.exceptions

try:
    import structlog

    logger = structlog.get_logger(__name__)
    _use_structlog = True
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)
    _use_structlog = False

from pipeline.config import PipelineSettings
from pipeline.db_writer import DBWriter
from pipeline.ensemble import EnsembleDetector
from pipeline.models import CostEvent, Detection, ScorePoint
from pipeline.publisher import Publisher


# ---------------------------------------------------------------------------
# LRU set — stdlib only
# ---------------------------------------------------------------------------


class _LRUSet:
    """Fixed-capacity LRU set backed by an :class:`~collections.OrderedDict`.

    When the set is at capacity and a new key is inserted the
    least-recently-used key is evicted first.
    """

    def __init__(self, maxsize: int) -> None:
        self._d: OrderedDict[str, None] = OrderedDict()
        self._maxsize = maxsize

    def add(self, key: str) -> None:
        """Insert *key*, refreshing its recency position if already present."""
        if key in self._d:
            self._d.move_to_end(key)
        else:
            self._d[key] = None
            if len(self._d) > self._maxsize:
                self._d.popitem(last=False)

    def __contains__(self, key: str) -> bool:  # type: ignore[override]
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

_BACKOFF_STEPS = (1, 2, 4, 8, 16, 30)
_PENDING_MIN_IDLE_MS = 30_000
_PENDING_COUNT = 100
_PENDING_START = "0-0"


def _bind(event_id: Optional[str] = None, run_id: Optional[str] = None):
    """Return a logger bound with *event_id* / *run_id* when available."""
    if _use_structlog:
        kw = {}
        if run_id is not None:
            kw["run_id"] = run_id
        if event_id is not None:
            kw["event_id"] = event_id
        return logger.bind(**kw)
    return logger


class Consumer:
    """Redis consumer-group worker for the CRIP detection pipeline.

    Reads ``CostEvent`` messages from the ``crip:events`` stream via an
    XREADGROUP consumer group, feeds them through the ensemble detector, and
    fans out results to Redis streams and the database.

    Parameters
    ----------
    settings:
        Runtime configuration (stream names, batch sizes, timeouts, …).
    ensemble:
        Pre-constructed :class:`~pipeline.ensemble.EnsembleDetector` instance.
    publisher:
        Pre-constructed :class:`~pipeline.publisher.Publisher` instance.
    db_writer:
        Pre-constructed :class:`~pipeline.db_writer.DBWriter` instance.
    """

    def __init__(
        self,
        settings: PipelineSettings,
        ensemble: EnsembleDetector,
        publisher: Publisher,
        db_writer: DBWriter,
    ) -> None:
        self._settings = settings
        self._ensemble = ensemble
        self._publisher = publisher
        self._db_writer = db_writer

        self._stream: str = settings.redis_stream_events
        self._group: str = settings.consumer_group
        self._consumer: str = settings.consumer_name
        self._batch_size: int = settings.batch_size
        self._block_ms: int = settings.block_ms

        # Deduplication LRU set
        self._dedup: _LRUSet = _LRUSet(maxsize=settings.dedup_window)

        # Track the active run; reset ensemble on change
        self._current_run_id: Optional[str] = None

        # Redis client — connected with exponential backoff
        self._redis: redis.Redis = None  # type: ignore[assignment]
        self._connect_redis()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_consumer_group(self) -> None:
        """Create the consumer group on the stream, if it does not yet exist.

        Uses ``id="0"`` so that newly-joining consumers re-process any
        messages that were never acknowledged (pending).  ``mkstream=True``
        creates the stream key if it does not yet exist.

        Silently ignores ``BUSYGROUP`` errors (group already exists).
        """
        try:
            self._redis.xgroup_create(
                self._stream,
                self._group,
                id="0",
                mkstream=True,
            )
            _bind().info(
                "consumer_group_created",
                stream=self._stream,
                group=self._group,
            )
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                _bind().debug(
                    "consumer_group_exists",
                    stream=self._stream,
                    group=self._group,
                )
            else:
                raise

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, stop_event: threading.Event) -> None:
        """Block until *stop_event* is set, draining ``crip:events``.

        On each iteration the loop:

        1. Calls :meth:`process_pending` once (at startup) to reclaim stale
           messages from crashed consumers.
        2. Issues ``XREADGROUP`` with ``">"`` (new, undelivered messages).
        3. Calls :meth:`_process_message` for every message in the batch.
        4. ``XACK``-s successfully processed messages.
        5. Updates the publisher queue-depth gauge.

        Parameters
        ----------
        stop_event:
            Caller sets this flag to request a graceful shutdown.  The loop
            finishes the current batch before returning.
        """
        log = _bind()
        log.info("consumer_started", stream=self._stream, group=self._group)

        # Reclaim stale pending messages from previous consumers
        self.process_pending()

        while not stop_event.is_set():
            try:
                raw = self._redis.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams={self._stream: ">"},
                    count=self._batch_size,
                    block=self._block_ms,
                )
            except redis.exceptions.ConnectionError as exc:
                _bind().error("redis_connection_error", error=str(exc))
                self._reconnect()
                continue
            except redis.exceptions.RedisError as exc:
                _bind().error("redis_read_error", error=str(exc))
                time.sleep(1)
                continue

            if not raw:
                # Timed out (block_ms elapsed) — loop to check stop_event
                continue

            # raw shape: [(stream_name, [(msg_id, {field: value}), ...])]
            for _stream_name, messages in raw:
                acked_ids: list[str] = []

                for msg_id, fields in messages:
                    # Decode bytes → str when redis-py returns bytes
                    msg_id_str: str = (
                        msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    )
                    payload_str: str = ""
                    for k, v in fields.items():
                        key = k.decode() if isinstance(k, bytes) else k
                        if key == "payload":
                            payload_str = v.decode() if isinstance(v, bytes) else v
                            break

                    try:
                        self._process_message(msg_id_str, payload_str)
                        acked_ids.append(msg_id_str)
                    except Exception as exc:  # noqa: BLE001
                        _bind().error(
                            "message_processing_failed",
                            msg_id=msg_id_str,
                            error=str(exc),
                            exc_info=True,
                        )
                        # Do NOT ack — leave in PEL for redelivery / XAUTOCLAIM

                if acked_ids:
                    try:
                        self._redis.xack(self._stream, self._group, *acked_ids)
                    except redis.exceptions.RedisError as exc:
                        _bind().error("redis_xack_error", error=str(exc))

                # Update queue-depth gauge
                try:
                    pending_info = self._redis.xpending(self._stream, self._group)
                    depth: int = (
                        pending_info.get("pending", 0)
                        if isinstance(pending_info, dict)
                        else 0
                    )
                    self._publisher.set_queue_depth(depth)
                except Exception:  # noqa: BLE001
                    pass  # Non-critical; gauge update best-effort

        log.info("consumer_stopped", stream=self._stream)

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _process_message(self, msg_id: str, payload_str: str) -> None:
        """Parse, deduplicate and process a single stream message.

        Steps
        -----
        1. Parse ``CostEvent`` from JSON.
        2. Skip duplicate ``event_id`` values (LRU dedup).
        3. Reset ensemble on ``run_id`` change.
        4. Run ensemble detection.
        5. Record end-to-end latency.
        6. Publish score point to Redis stream.
        7. Enqueue event and scores to the DB writer.
        8. If a detection fired, publish it and enqueue it for the DB.
        9. Mark ``event_id`` as seen in the LRU set.

        Parameters
        ----------
        msg_id:
            Redis stream message ID (e.g. ``"1699000000000-0"``).
        payload_str:
            Raw JSON string from the ``payload`` field of the stream entry.
        """
        # 1. Parse
        try:
            event: CostEvent = CostEvent.from_json(payload_str)
        except Exception as exc:  # noqa: BLE001
            _bind().error(
                "event_parse_error",
                msg_id=msg_id,
                error=str(exc),
                payload_preview=payload_str[:200],
            )
            raise  # re-raise so caller can decide not to ack

        log = _bind(event_id=event.event_id, run_id=event.run_id)

        # 2. Dedup
        if event.event_id in self._dedup:
            log.debug("event_duplicate_skipped")
            return

        # 3. Run-id change → reset ensemble
        if event.run_id != self._current_run_id:
            log.info(
                "run_id_changed",
                old_run_id=self._current_run_id,
                new_run_id=event.run_id,
            )
            self._ensemble.reset(event.run_id)
            self._current_run_id = event.run_id

        # 4. Detection
        detection: Optional[Detection]
        score_point: ScorePoint
        detection, score_point = self._ensemble.process(event)

        # 5. Latency
        self._publisher.record_latency(event)

        # 6. Publish scores
        self._publisher.publish_scores(score_point)

        # 7. Enqueue event + scores to DB
        self._db_writer.enqueue_event(event)
        self._db_writer.enqueue_scores(score_point)

        # 8. Detection fanout
        if detection is not None:
            log.warning(
                "detection_fired",
                detection_id=detection.detection_id,
                severity=detection.severity,
                alert_type=detection.rca.alert_type,
                service=detection.service,
                tick_index=detection.tick_index,
            )
            self._publisher.publish_detection(detection)
            self._db_writer.enqueue_detection(detection)

        # 9. Mark seen
        self._dedup.add(event.event_id)

    # ------------------------------------------------------------------
    # Pending / redelivery
    # ------------------------------------------------------------------

    def process_pending(self) -> None:
        """Reclaim and reprocess stale pending messages via ``XAUTOCLAIM``.

        Messages that have been idle for more than 30 seconds (i.e. delivered
        to a consumer that crashed before ``XACK``-ing them) are transferred
        to this consumer and processed normally.
        """
        log = _bind()
        start_id = _PENDING_START

        while True:
            try:
                result = self._redis.xautoclaim(
                    self._stream,
                    self._group,
                    self._consumer,
                    min_idle_time=_PENDING_MIN_IDLE_MS,
                    start_id=start_id,
                    count=_PENDING_COUNT,
                )
            except redis.exceptions.ResponseError as exc:
                # XAUTOCLAIM requires Redis ≥ 6.2; handle gracefully
                log.warning(
                    "xautoclaim_unsupported",
                    error=str(exc),
                )
                return
            except redis.exceptions.RedisError as exc:
                log.error("xautoclaim_error", error=str(exc))
                return

            # result shape (Redis ≥ 7): [next_start_id, [(msg_id, fields), ...], deleted_ids]
            # Redis 6.2 returns a list; handle both
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                next_start_id = result[0]
                claimed_messages = result[1]
            else:
                break

            if isinstance(next_start_id, bytes):
                next_start_id = next_start_id.decode()

            acked_ids: list[str] = []
            for msg_id, fields in (claimed_messages or []):
                msg_id_str: str = (
                    msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                )
                payload_str: str = ""
                for k, v in (fields or {}).items():
                    key = k.decode() if isinstance(k, bytes) else k
                    if key == "payload":
                        payload_str = v.decode() if isinstance(v, bytes) else v
                        break

                log.info("reclaiming_pending_message", msg_id=msg_id_str)
                try:
                    self._process_message(msg_id_str, payload_str)
                    acked_ids.append(msg_id_str)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "pending_message_processing_failed",
                        msg_id=msg_id_str,
                        error=str(exc),
                    )

            if acked_ids:
                try:
                    self._redis.xack(self._stream, self._group, *acked_ids)
                except redis.exceptions.RedisError as exc:
                    log.error("redis_xack_pending_error", error=str(exc))

            # "0-0" means no more pending entries
            if not next_start_id or next_start_id == "0-0":
                break
            start_id = next_start_id

    # ------------------------------------------------------------------
    # Redis connection helpers
    # ------------------------------------------------------------------

    def _connect_redis(self) -> None:
        """Connect to Redis with exponential back-off (1 → 2 → 4 → 8 → 16 → 30 s).

        Blocks until a connection is established.  Each failed attempt is
        logged at ERROR level before sleeping.
        """
        log = _bind()
        for delay in _BACKOFF_STEPS:
            try:
                client = redis.Redis.from_url(
                    self._settings.redis_url,
                    decode_responses=False,  # keep raw bytes; we decode explicitly
                )
                client.ping()
                self._redis = client
                log.info(
                    "redis_connected",
                    url=self._settings.redis_url,
                )
                return
            except redis.exceptions.RedisError as exc:
                log.error(
                    "redis_connect_failed",
                    error=str(exc),
                    retry_in_s=delay,
                )
                time.sleep(delay)

        # Last attempt with no sleep — raise if still failing
        client = redis.Redis.from_url(
            self._settings.redis_url,
            decode_responses=False,
        )
        client.ping()
        self._redis = client
        log.info("redis_connected", url=self._settings.redis_url)

    def _reconnect(self) -> None:
        """Attempt to reconnect after a connection drop."""
        _bind().warning("redis_reconnecting")
        try:
            self._connect_redis()
        except redis.exceptions.RedisError as exc:
            _bind().error("redis_reconnect_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Redis connection cleanly."""
        log = _bind()
        try:
            if self._redis is not None:
                self._redis.close()
                log.info("redis_connection_closed")
        except redis.exceptions.RedisError as exc:
            log.error("redis_close_error", error=str(exc))

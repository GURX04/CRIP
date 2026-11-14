"""publisher.py – Redis stream publisher and Prometheus metrics exporter.

Publishes detection outputs (Detection, ScorePoint) to Redis streams and
exposes Prometheus metrics for the CRIP real-time streaming detection pipeline.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import redis
import redis.exceptions

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server  # type: ignore
    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover
    _HAS_PROMETHEUS = False

    class _NoOpMetric:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set_function(self, *args, **kwargs):
            pass

    def Counter(*args, **kwargs):  # type: ignore
        return _NoOpMetric()

    def Gauge(*args, **kwargs):  # type: ignore
        return _NoOpMetric()

    def Histogram(*args, **kwargs):  # type: ignore
        return _NoOpMetric()

    def start_http_server(*args, **kwargs):  # type: ignore
        pass

from pipeline.config import PipelineSettings
from pipeline.models import CostEvent, Detection, ScorePoint

# ---------------------------------------------------------------------------
# Logger setup – prefer structlog when available, fall back to stdlib logging.
# ---------------------------------------------------------------------------
try:
    import structlog  # type: ignore

    _log = structlog.get_logger(__name__)
    _use_structlog = True
except ImportError:  # pragma: no cover
    _log = logging.getLogger(__name__)  # type: ignore[assignment]
    _use_structlog = False


def _info(msg: str, **kw: object) -> None:
    if _use_structlog:
        _log.info(msg, **kw)
    else:
        _log.info(msg, extra=kw)  # type: ignore[arg-type]


def _warning(msg: str, **kw: object) -> None:
    if _use_structlog:
        _log.warning(msg, **kw)
    else:
        _log.warning(msg, extra=kw)  # type: ignore[arg-type]


def _error(msg: str, **kw: object) -> None:
    if _use_structlog:
        _log.error(msg, **kw)
    else:
        _log.error(msg, extra=kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Prometheus metric definitions (module-level singletons so that re-creating a
# Publisher in tests does not register duplicate collectors).
# ---------------------------------------------------------------------------
_LATENCY_HIST: Optional[Histogram] = None
_EVENTS_CTR: Optional[Counter] = None
_ALERTS_CTR: Optional[Counter] = None
_QUEUE_GAUGE: Optional[Gauge] = None
_DETECTOR_HIST: Optional[Histogram] = None


def _init_metrics() -> tuple[Histogram, Counter, Counter, Gauge, Histogram]:
    """Initialise (or return existing) Prometheus collectors."""
    global _LATENCY_HIST, _EVENTS_CTR, _ALERTS_CTR, _QUEUE_GAUGE, _DETECTOR_HIST

    if _LATENCY_HIST is None:
        _LATENCY_HIST = Histogram(
            "pipeline_process_latency_ms",
            "End-to-end processing latency in milliseconds from event arrival to publish.",
            labelnames=["service"],
            buckets=[0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500],
        )
    if _EVENTS_CTR is None:
        _EVENTS_CTR = Counter(
            "pipeline_events_total",
            "Total number of cost events processed by the pipeline.",
            labelnames=["service"],
        )
    if _ALERTS_CTR is None:
        _ALERTS_CTR = Counter(
            "pipeline_alerts_total",
            "Total number of anomaly detections (alerts) published to Redis.",
            labelnames=["service"],
        )
    if _QUEUE_GAUGE is None:
        _QUEUE_GAUGE = Gauge(
            "pipeline_queue_depth",
            "Current depth of the internal processing queue.",
        )
    if _DETECTOR_HIST is None:
        _DETECTOR_HIST = Histogram(
            "pipeline_detector_runtime_ms",
            "Per-detector update runtime in milliseconds.",
            labelnames=["detector"],
        )

    return _LATENCY_HIST, _EVENTS_CTR, _ALERTS_CTR, _QUEUE_GAUGE, _DETECTOR_HIST


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

_BACKOFF_SCHEDULE = [1, 2, 4, 8, 16, 30]  # seconds


class Publisher:
    """Publishes detection outputs to Redis Streams and exposes Prometheus metrics.

    Parameters
    ----------
    settings:
        Pipeline configuration (redis_url, redis_maxlen, metrics_port, …).
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        self._redis: Optional[redis.Redis] = None  # type: ignore[type-arg]

        # Connect to Redis (blocks until first successful connection).
        self._connect_redis()

        # Prometheus metrics.
        (
            self._latency_hist,
            self._events_ctr,
            self._alerts_ctr,
            self._queue_gauge,
            self._detector_hist,
        ) = _init_metrics()

        # Start the Prometheus HTTP server.
        try:
            start_http_server(settings.metrics_port)
            _info("Prometheus metrics server started", port=settings.metrics_port)
        except OSError as exc:
            _warning(
                "Prometheus HTTP server could not start (port already in use?); "
                "metrics will still be collected but not exported",
                port=settings.metrics_port,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish_detection(self, detection: Detection) -> None:
        """Publish a Detection to the ``crip:detections`` Redis stream.

        The stream entry contains a single ``payload`` field holding the JSON
        string produced by :meth:`Detection.to_redis_dict`.

        Also increments the ``pipeline_alerts_total`` counter for the
        detection's service.

        Parameters
        ----------
        detection:
            Fully assembled detection object ready for publishing.
        """
        payload = detection.to_redis_dict()["payload"]
        self._xadd(
            stream=self._settings.redis_stream_detections,
            fields={"payload": payload},
        )
        self._alerts_ctr.labels(service=detection.service).inc()

    def publish_scores(self, sp: ScorePoint) -> None:
        """Publish a ScorePoint to the ``crip:scores`` Redis stream.

        The stream entry contains a single ``payload`` field with a JSON
        object carrying all score fields plus the current wall-clock timestamp.

        Parameters
        ----------
        sp:
            Score snapshot for a single (run, service, tick).
        """
        payload = json.dumps(
            {
                "run_id": sp.run_id,
                "service": sp.service,
                "sim_ts": sp.sim_ts.isoformat(),
                "cost": sp.cost,
                "expected_cost": sp.expected_cost,
                "score_zscore": sp.score_zscore,
                "score_ewma": sp.score_ewma,
                "score_cpr": sp.score_cpr,
                "wall_ts_ms": int(time.time() * 1000),
            }
        )
        self._xadd(
            stream=self._settings.redis_stream_scores,
            fields={"payload": payload},
        )

    def record_latency(self, event: CostEvent) -> None:
        """Record end-to-end processing latency for *event*.

        Computes ``now - event.wall_ts_ms`` (in milliseconds) and observes it
        in the ``pipeline_process_latency_ms`` histogram labelled by service.
        Also increments ``pipeline_events_total`` for the event's service.

        Parameters
        ----------
        event:
            The cost event whose wall-clock arrival timestamp is used as the
            start of the latency window.
        """
        now_ms = int(time.time() * 1000)
        latency_ms = now_ms - event.wall_ts_ms
        self._latency_hist.labels(service=event.service).observe(latency_ms)
        self._events_ctr.labels(service=event.service).inc()

    def set_queue_depth(self, depth: int) -> None:
        """Update the ``pipeline_queue_depth`` Prometheus gauge.

        Parameters
        ----------
        depth:
            Current number of items waiting in the internal processing queue.
        """
        self._queue_gauge.set(depth)

    def close(self) -> None:
        """Close the Redis connection cleanly."""
        if self._redis is not None:
            try:
                self._redis.close()
                _info("Redis connection closed")
            except Exception as exc:  # noqa: BLE001
                _warning("Error while closing Redis connection", error=str(exc))
            finally:
                self._redis = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect_redis(self) -> None:
        """Attempt to connect to Redis with exponential back-off.

        Retries with delays of 1 → 2 → 4 → 8 → 16 → 30 s (capped), logging
        each attempt.  Returns once a successful PING is received.
        """
        for attempt, delay in enumerate(
            _BACKOFF_SCHEDULE + [_BACKOFF_SCHEDULE[-1]] * 1_000_000, start=1
        ):
            _info(
                "Connecting to Redis",
                url=self._settings.redis_url,
                attempt=attempt,
            )
            try:
                client: redis.Redis = redis.Redis.from_url(  # type: ignore[type-arg]
                    self._settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                client.ping()
                self._redis = client
                _info("Redis connected", url=self._settings.redis_url, attempt=attempt)
                return
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
                _error(
                    "Redis connection failed; retrying",
                    error=str(exc),
                    retry_in_seconds=delay,
                    attempt=attempt,
                )
                time.sleep(delay)

    def _ensure_connected(self) -> bool:
        """Return True if Redis is reachable; attempt a single reconnect if not.

        Returns
        -------
        bool
            ``True`` when a healthy Redis client is available, ``False``
            otherwise (caller should skip the publish operation and retry later).
        """
        if self._redis is None:
            _warning("Redis client is None; attempting reconnect")
            try:
                self._connect_redis()
            except Exception as exc:  # noqa: BLE001
                _error("Reconnect failed", error=str(exc))
                return False
        return self._redis is not None

    def _xadd(self, stream: str, fields: dict[str, str]) -> None:
        """XADD *fields* to *stream* with approximate MAXLEN trimming.

        Silently logs and swallows :class:`redis.exceptions.ConnectionError` so
        that a transient Redis outage does not crash the pipeline; the entry is
        dropped and will be retried on the next call.

        Parameters
        ----------
        stream:
            Redis stream key (e.g. ``crip:detections``).
        fields:
            Mapping of field names to string values for the stream entry.
        """
        if not self._ensure_connected():
            _warning("Skipping XADD – Redis unavailable", stream=stream)
            return
        try:
            self._redis.xadd(  # type: ignore[union-attr]
                stream,
                fields,
                maxlen=self._settings.redis_maxlen,
                approximate=True,
            )
        except redis.exceptions.ConnectionError as exc:
            _error(
                "Redis XADD failed; entry dropped – will retry on next call",
                stream=stream,
                error=str(exc),
            )
            # Mark client as unhealthy so _ensure_connected reconnects next time.
            self._redis = None
        except Exception as exc:  # noqa: BLE001
            _error("Unexpected error during XADD", stream=stream, error=str(exc))

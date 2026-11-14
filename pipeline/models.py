"""
pipeline/models.py
==================
Pure-Python dataclass models shared across every component of the CRIP
real-time streaming detection pipeline.

No third-party dependencies — only the standard library and :mod:`dataclasses`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_to_iso(dt: datetime) -> str:
    """Render a :class:`datetime` as a UTC ISO-8601 string with 'Z' suffix."""
    if dt.tzinfo is None:
        # Treat naïve datetimes as UTC.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _default_json(obj: Any) -> Any:
    """JSON serialiser hook for types not natively handled by :mod:`json`."""
    if isinstance(obj, datetime):
        return _dt_to_iso(obj)
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serialisable")


# ---------------------------------------------------------------------------
# CostEvent
# ---------------------------------------------------------------------------

@dataclass
class CostEvent:
    """A single cost-usage event emitted by the CRIP simulator.

    Attributes
    ----------
    run_id:
        Unique identifier of the simulation run that produced this event.
    event_id:
        Globally unique identifier for this individual event (used for
        deduplication).
    tick_index:
        Zero-based tick counter within the simulation run.
    sim_ts:
        Simulated wall-clock timestamp (always timezone-aware, UTC).
    wall_ts_ms:
        Real wall-clock timestamp in milliseconds since the Unix epoch at the
        moment the event was emitted by the producer.
    service:
        Cloud service name (e.g. ``"EC2"``, ``"S3"``).
    request_count:
        Number of API requests that contributed to this cost entry.
    usage_amount:
        Raw usage quantity (unit depends on the service, e.g. GB-hours).
    unblended_cost:
        Unblended cost in USD for this event.
    unit_price:
        Price per unit of usage in USD.
    """

    run_id: str
    event_id: str
    tick_index: int
    sim_ts: datetime  # UTC
    wall_ts_ms: int
    service: str
    request_count: float
    usage_amount: float
    unblended_cost: float
    unit_price: float

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, payload: str) -> CostEvent:
        """Deserialise a JSON string produced by the CRIP simulator.

        The ``sim_ts`` field is expected to be an ISO-8601 string
        (with or without a timezone designator).  It is always stored
        as a timezone-aware :class:`~datetime.datetime` in UTC.

        Parameters
        ----------
        payload:
            Raw JSON string from the Redis stream entry.

        Returns
        -------
        CostEvent
            A fully-populated :class:`CostEvent` instance.

        Raises
        ------
        json.JSONDecodeError
            If *payload* is not valid JSON.
        KeyError
            If a required field is missing from the parsed object.
        ValueError
            If ``sim_ts`` cannot be parsed as an ISO-8601 datetime string.
        """
        data: dict[str, Any] = json.loads(payload)

        raw_ts: str = data["sim_ts"]
        # datetime.fromisoformat() does not accept the trailing 'Z' in
        # Python < 3.11, so we normalise it first.
        raw_ts = raw_ts.replace("Z", "+00:00")
        sim_ts = datetime.fromisoformat(raw_ts)
        if sim_ts.tzinfo is None:
            sim_ts = sim_ts.replace(tzinfo=timezone.utc)
        else:
            sim_ts = sim_ts.astimezone(timezone.utc)

        return cls(
            run_id=str(data["run_id"]),
            event_id=str(data["event_id"]),
            tick_index=int(data["tick_index"]),
            sim_ts=sim_ts,
            wall_ts_ms=int(data["wall_ts_ms"]),
            service=str(data["service"]),
            request_count=float(data["request_count"]),
            usage_amount=float(data["usage_amount"]),
            unblended_cost=float(data["unblended_cost"]),
            unit_price=float(data["unit_price"]),
        )


# ---------------------------------------------------------------------------
# DetectorResult
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """Output of a single anomaly detector for one tick.

    Attributes
    ----------
    detector_name:
        Human-readable name of the detector (e.g. ``"zscore"``, ``"ewma"``,
        ``"cpr"``).
    score:
        Raw anomaly score emitted by this detector.  The scale is
        detector-specific but higher values always mean *more anomalous*.
    is_anomaly:
        ``True`` when the detector's internal threshold has been exceeded.
    explanation:
        Short, human-readable string describing why the detector fired (or did
        not fire).  Suitable for display in an alert notification.
    """

    detector_name: str
    score: float
    is_anomaly: bool
    explanation: str


# ---------------------------------------------------------------------------
# RCAResult
# ---------------------------------------------------------------------------

@dataclass
class RCAResult:
    """Root-cause analysis result attached to a confirmed :class:`Detection`.

    Attributes
    ----------
    alert_type:
        High-level classification of the anomaly driver:

        * ``"traffic_driven"`` — abnormal request volume, unit price stable.
        * ``"price_driven"`` — unit price spike, volume roughly normal.
        * ``"volume_drop_cost_flat"`` — volume fell but cost did not follow.
        * ``"unknown"`` — could not classify confidently.
    cost_per_1k_now:
        Cost per 1 000 requests in the anomalous window (USD).
    cost_per_1k_baseline:
        Cost per 1 000 requests during the warm-up / baseline window (USD).
    pct_cost_change:
        Percentage change in total cost relative to baseline
        (positive = increase).
    pct_volume_change:
        Percentage change in request volume relative to baseline
        (positive = increase).
    service_contributions:
        Mapping of *service name* → *fraction of total cost deviation*
        (values should sum to ≈ 1.0 across all contributing services).
    """

    alert_type: str  # "traffic_driven" | "price_driven" | "volume_drop_cost_flat" | "unknown"
    cost_per_1k_now: float
    cost_per_1k_baseline: float
    pct_cost_change: float
    pct_volume_change: float
    service_contributions: dict[str, float]  # service → fraction of total cost deviation


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A confirmed anomaly detection emitted by the ensemble detector.

    Attributes
    ----------
    detection_id:
        Globally unique identifier for this detection (UUID recommended).
    run_id:
        Identifier of the simulation run this detection belongs to.
    service:
        Cloud service on which the anomaly was detected.
    sim_ts:
        Simulated timestamp at the tick where the anomaly was confirmed (UTC).
    detect_wall_ts_ms:
        Real wall-clock timestamp (ms since Unix epoch) when the detection was
        emitted by the pipeline.
    first_event_wall_ts_ms:
        Real wall-clock timestamp of the *first* event in the anomalous batch
        (used to measure end-to-end latency).
    severity:
        Normalised severity score in the range ``[0, 100]``.
    detector_votes:
        List of :class:`DetectorResult` objects — one per detector that
        participated in the ensemble vote.
    rca:
        Root-cause analysis result for this detection.
    status:
        Lifecycle status: ``"open"`` (just detected), ``"ongoing"``
        (anomaly persists), or ``"closed"`` (resolved).
    tick_index:
        Zero-based tick index within the simulation run at which the detection
        was emitted.
    """

    detection_id: str
    run_id: str
    service: str
    sim_ts: datetime
    detect_wall_ts_ms: int
    first_event_wall_ts_ms: int
    severity: float
    detector_votes: list[DetectorResult]
    rca: RCAResult
    status: str  # "open" | "ongoing" | "closed"
    tick_index: int

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_redis_dict(self) -> dict[str, str]:
        """Serialise the detection to a flat dict suitable for ``XADD``.

        Redis stream entries must have string keys and string values.
        The entire :class:`Detection` is encoded as JSON and stored under
        the ``"payload"`` key so that consumers can deserialise it in a
        single step.

        Returns
        -------
        dict[str, str]
            Single-entry dict ``{"payload": "<json string>"}`` where the
            JSON object is a full representation of this :class:`Detection`,
            with :class:`~datetime.datetime` fields rendered as UTC
            ISO-8601 strings.
        """
        raw: dict[str, Any] = {
            "detection_id": self.detection_id,
            "run_id": self.run_id,
            "service": self.service,
            "sim_ts": _dt_to_iso(self.sim_ts),
            "detect_wall_ts_ms": self.detect_wall_ts_ms,
            "first_event_wall_ts_ms": self.first_event_wall_ts_ms,
            "severity": self.severity,
            "detector_votes": [
                {
                    "detector_name": v.detector_name,
                    "score": v.score,
                    "is_anomaly": v.is_anomaly,
                    "explanation": v.explanation,
                }
                for v in self.detector_votes
            ],
            "rca": {
                "alert_type": self.rca.alert_type,
                "cost_per_1k_now": self.rca.cost_per_1k_now,
                "cost_per_1k_baseline": self.rca.cost_per_1k_baseline,
                "pct_cost_change": self.rca.pct_cost_change,
                "pct_volume_change": self.rca.pct_volume_change,
                "service_contributions": self.rca.service_contributions,
            },
            "status": self.status,
            "tick_index": self.tick_index,
        }
        return {"payload": json.dumps(raw, default=_default_json)}


# ---------------------------------------------------------------------------
# ScorePoint
# ---------------------------------------------------------------------------

@dataclass
class ScorePoint:
    """Per-tick, per-service score snapshot written to the scores stream.

    Attributes
    ----------
    run_id:
        Identifier of the simulation run.
    service:
        Cloud service this score pertains to.
    sim_ts:
        Simulated timestamp for this tick (UTC).
    cost:
        Observed unblended cost for this tick (USD).
    expected_cost:
        Baseline / predicted cost for this tick as estimated by the detector
        ensemble (USD).
    score_zscore:
        Anomaly score produced by the rolling z-score detector.
    score_ewma:
        Anomaly score produced by the EWMA control-chart detector.
    score_cpr:
        Anomaly score produced by the CPR (change-point regression) detector.
    """

    run_id: str
    service: str
    sim_ts: datetime
    cost: float
    expected_cost: float
    score_zscore: float
    score_ewma: float
    score_cpr: float

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_redis_dict(self) -> dict[str, str]:
        """Serialise the score point to a flat dict suitable for ``XADD``.

        Returns
        -------
        dict[str, str]
            Single-entry dict ``{"payload": "<json string>"}`` where the JSON
            object contains all fields of this :class:`ScorePoint`, with
            :class:`~datetime.datetime` fields as UTC ISO-8601 strings.
        """
        raw: dict[str, Any] = {
            "run_id": self.run_id,
            "service": self.service,
            "sim_ts": _dt_to_iso(self.sim_ts),
            "cost": self.cost,
            "expected_cost": self.expected_cost,
            "score_zscore": self.score_zscore,
            "score_ewma": self.score_ewma,
            "score_cpr": self.score_cpr,
        }
        return {"payload": json.dumps(raw, default=_default_json)}

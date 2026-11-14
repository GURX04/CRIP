"""
pipeline/ensemble.py
====================
Ensemble detector that combines RobustZScoreDetector, EWMADetector, and
CostPerRequestResidualDetector via majority voting to produce Detections and
ScorePoints for the CRIP real-time streaming pipeline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from pipeline.config import PipelineSettings
from pipeline.detectors import (
    CostPerRequestResidualDetector,
    EWMADetector,
    RobustZScoreDetector,
    WarmUpWrapper,
)
from pipeline.models import (
    CostEvent,
    Detection,
    DetectorResult,
    RCAResult,
    ScorePoint,
)

DEFAULT_SERVICES: Tuple[str, ...] = (
    "AmazonApiGateway",
    "EC2 Data Transfer",
    "DynamoDB",
    "CloudWatch Logs",
    "S3 PUT",
)

_BASELINE_ALPHA: float = 0.05  # EW smoothing factor for baseline ewmas


@dataclass
class ServiceState:
    """Per-service mutable state maintained by the ensemble.

    Attributes:
        detectors: Mapping of detector name to its WarmUpWrapper instance.
        cooldown_until_tick: No new alert fires while tick_index <= this value.
        last_event: The most recent CostEvent processed for this service.
        baseline_cost_ewma: Exponentially-weighted mean of unblended_cost
            (alpha=0.05).
        baseline_req_ewma: Exponentially-weighted mean of request_count
            (alpha=0.05).
        last_detection_tick: tick_index at which the last Detection was emitted
            (used by update_detection_status).
        status_emitted_closed: True once "closed" has been returned for the
            most recent detection window (prevents duplicate "closed" returns).
    """

    detectors: Dict[str, WarmUpWrapper]
    cooldown_until_tick: int = 0
    last_event: Optional[CostEvent] = None
    baseline_cost_ewma: float = 0.0
    baseline_req_ewma: float = 0.0
    last_detection_tick: int = -1
    status_emitted_closed: bool = False


def _make_service_state(settings: PipelineSettings) -> ServiceState:
    """Construct a fresh ServiceState with all three detectors wrapped."""
    warmup = settings.warmup_ticks
    detectors: Dict[str, WarmUpWrapper] = {
        "zscore": WarmUpWrapper(
            RobustZScoreDetector(settings),
            warmup,
        ),
        "ewma": WarmUpWrapper(
            EWMADetector(settings),
            warmup,
        ),
        "cpr": WarmUpWrapper(
            CostPerRequestResidualDetector(settings),
            warmup,
        ),
    }
    return ServiceState(detectors=detectors)


class EnsembleDetector:
    """Majority-vote ensemble that emits Detections and ScorePoints.

    One instance is created per pipeline run.  It maintains independent
    per-service state (detectors, baselines, cooldown) and is designed to be
    called from a single consumer thread, though the state dictionary itself is
    protected by an RLock for future multi-threaded consumers.

    Args:
        settings: Fully-configured PipelineSettings instance.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        # Pre-populate one ServiceState per known service
        self._states: Dict[str, ServiceState] = {
            svc: _make_service_state(settings) for svc in DEFAULT_SERVICES
        }
        self._current_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, event: CostEvent) -> Tuple[Optional[Detection], ScorePoint]:
        """Process one CostEvent and return a (Detection | None, ScorePoint).

        Steps:
        1. Retrieve (or lazily create) the ServiceState for event.service.
        2. Run all three wrapped detectors and collect DetectorResults.
        3. Update the per-service baseline EWMAs for cost and request_count.
        4. Build a ScorePoint capturing raw scores and the expected cost.
        5. Count votes (DetectorResult.is_anomaly == True).
        6. If votes >= vote_threshold AND tick_index > cooldown_until_tick,
           compute RCA, build a Detection, arm cooldown, and return it.
        7. Otherwise return (None, ScorePoint).

        Args:
            event: The incoming CostEvent to evaluate.

        Returns:
            A 2-tuple of (Detection or None, ScorePoint).
        """
        with self._lock:
            state = self._get_or_create_state(event.service)

            # --- 2. Run detectors ---
            results: List[DetectorResult] = [
                state.detectors["zscore"].update(event),
                state.detectors["ewma"].update(event),
                state.detectors["cpr"].update(event),
            ]

            # --- 3. Update baseline EWMAs ---
            alpha = _BASELINE_ALPHA
            if state.baseline_cost_ewma == 0.0 and state.baseline_req_ewma == 0.0:
                # Cold-start: seed with first observation
                state.baseline_cost_ewma = event.unblended_cost
                state.baseline_req_ewma = event.request_count
            else:
                state.baseline_cost_ewma = (
                    alpha * event.unblended_cost
                    + (1.0 - alpha) * state.baseline_cost_ewma
                )
                state.baseline_req_ewma = (
                    alpha * event.request_count
                    + (1.0 - alpha) * state.baseline_req_ewma
                )

            # --- 4. Build ScorePoint ---
            score_point = ScorePoint(
                run_id=event.run_id,
                service=event.service,
                sim_ts=event.sim_ts,
                cost=event.unblended_cost,
                expected_cost=state.baseline_cost_ewma,
                score_zscore=results[0].score,
                score_ewma=results[1].score,
                score_cpr=results[2].score,
            )

            # --- 5. Count votes ---
            anomalous = [r for r in results if r.is_anomaly]
            votes = len(anomalous)

            detection: Optional[Detection] = None

            if (
                votes >= self._settings.vote_threshold
                and event.tick_index > state.cooldown_until_tick
            ):
                # --- Compute RCA ---
                rca = self._rca(event, state)

                # severity: mean of anomalous scores * 25, capped at 100
                severity = min(
                    100.0,
                    mean(r.score for r in anomalous) * 25.0,
                )

                # --- Build Detection ---
                detection = Detection(
                    detection_id=str(uuid4()),
                    run_id=event.run_id,
                    service=event.service,
                    sim_ts=event.sim_ts,
                    detect_wall_ts_ms=int(time.time() * 1000),
                    first_event_wall_ts_ms=event.wall_ts_ms,
                    severity=severity,
                    detector_votes=results,
                    rca=rca,
                    status="open",
                    tick_index=event.tick_index,
                )

                # --- Arm cooldown ---
                state.cooldown_until_tick = (
                    event.tick_index + self._settings.cooldown_ticks
                )
                state.last_detection_tick = event.tick_index
                state.status_emitted_closed = False

            # Update last_event after all processing
            state.last_event = event

        return detection, score_point

    def update_detection_status(
        self, service: str, tick_index: int
    ) -> Optional[str]:
        """Return the lifecycle status of the most recent detection for a service.

        Returns:
            * ``"ongoing"``  – a detection was emitted and cooldown is still active.
            * ``"closed"``   – cooldown just expired; returned **once** per detection.
            * ``None``       – no detection has been emitted, or already closed.

        Args:
            service: The service name to query.
            tick_index: Current tick index (used to compare against cooldown).
        """
        with self._lock:
            state = self._states.get(service)
            if state is None or state.last_detection_tick < 0:
                return None

            if tick_index <= state.cooldown_until_tick:
                return "ongoing"

            # Cooldown expired
            if not state.status_emitted_closed:
                state.status_emitted_closed = True
                return "closed"

            return None

    def reset(self, run_id: str) -> None:
        """Clear all ServiceState, resetting the ensemble for a new run.

        Called when a new run_id is detected on the stream so that stale
        per-service state from the previous run does not bleed into the new
        one.

        Args:
            run_id: The new run identifier (stored for reference).
        """
        with self._lock:
            self._current_run_id = run_id
            self._states = {
                svc: _make_service_state(self._settings) for svc in DEFAULT_SERVICES
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_state(self, service: str) -> ServiceState:
        """Return existing ServiceState or create one on first encounter.

        Note: Must be called with self._lock already held.

        Args:
            service: Service name key.

        Returns:
            The ServiceState for that service.
        """
        if service not in self._states:
            self._states[service] = _make_service_state(self._settings)
        return self._states[service]

    def _rca(self, event: CostEvent, state: ServiceState) -> RCAResult:
        """Perform Root-Cause Analysis for an anomalous event.

        Computes cost-per-1k-requests now vs. baseline, percentage changes in
        cost and volume, and classifies the alert type.

        Args:
            event: The anomalous CostEvent.
            state: The current ServiceState (contains baseline EWMAs).

        Returns:
            An RCAResult with classification and contribution map.
        """
        # Cost-per-1k requests
        cost_per_1k_now = (
            event.unblended_cost / max(event.request_count, 1.0) * 1_000.0
        )
        cost_per_1k_baseline = (
            state.baseline_cost_ewma / max(state.baseline_req_ewma, 1.0) * 1_000.0
        )

        # Percentage changes (guard against zero denominators)
        pct_cost_change = (
            (event.unblended_cost - state.baseline_cost_ewma)
            / max(state.baseline_cost_ewma, 1e-9)
            * 100.0
        )
        pct_volume_change = (
            (event.request_count - state.baseline_req_ewma)
            / max(state.baseline_req_ewma, 1e-9)
            * 100.0
        )

        # Classify alert type
        price_ratio_deviation = abs(
            cost_per_1k_now / max(cost_per_1k_baseline, 1e-9) - 1.0
        )

        if pct_volume_change >= 20.0 and price_ratio_deviation < 0.2:
            alert_type = "traffic_driven"
        elif abs(pct_volume_change) < 10.0 and pct_cost_change > 20.0:
            alert_type = "price_driven"
        elif pct_volume_change <= -20.0 and abs(pct_cost_change) < 10.0:
            alert_type = "volume_drop_cost_flat"
        else:
            alert_type = "unknown"

        return RCAResult(
            alert_type=alert_type,
            cost_per_1k_now=cost_per_1k_now,
            cost_per_1k_baseline=cost_per_1k_baseline,
            pct_cost_change=pct_cost_change,
            pct_volume_change=pct_volume_change,
            # Single-service dict; extendable for multi-service alerts
            service_contributions={event.service: 1.0},
        )

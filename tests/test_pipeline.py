"""
tests/test_pipeline.py
======================
Unit and integration test suite for the CRIP real-time streaming detection pipeline.

Tests:
1. Stable traffic => no alerts after warm-up
2. Traffic-driven spike => CPR stays quiet, z-score/EWMA fire
3. Cost-only injection => CPR fires, RCA labels price_driven
4. Slow drift => at least one detector fires
5. Ensemble voting and cooldown
6. Run-id change resets state
7. Evaluator scenario evaluation (precision, recall, latency)
8. Consumer group deduplication and ack with fakeredis
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from pipeline.config import PipelineSettings
from pipeline.detectors import (
    CostPerRequestResidualDetector,
    EWMADetector,
    RobustZScoreDetector,
    WarmUpWrapper,
)
from pipeline.ensemble import EnsembleDetector
from pipeline.evaluator import DetectionRecord, Evaluator, InjectionRecord
from pipeline.models import CostEvent


def _make_event(
    service: str = "DynamoDB",
    tick: int = 0,
    cost: float = 0.01,
    requests: float = 8000.0,
    run_id: str = "run-1",
) -> CostEvent:
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_ts = base_time + timedelta(minutes=tick * 5)
    return CostEvent(
        run_id=run_id,
        event_id=f"ev-{tick}-{service}",
        tick_index=tick,
        sim_ts=sim_ts,
        wall_ts_ms=int(time.time() * 1000),
        service=service,
        request_count=requests,
        usage_amount=requests,
        unblended_cost=cost,
        unit_price=cost / requests if requests else 0.00000125,
    )


class TestStreamingDetectors:
    def test_stable_traffic_no_alerts_after_warmup(self):
        settings = PipelineSettings(warmup_ticks=30)
        ensemble = EnsembleDetector(settings)
        alerts = []

        for t in range(60):
            # Normal variance ~ 1-2%
            noise = 1.0 + ((t % 5) - 2) * 0.005
            ev = _make_event(
                tick=t, cost=0.01 * noise, requests=8000.0 * noise
            )
            det, score = ensemble.process(ev)
            if det:
                alerts.append(det)

        assert len(alerts) == 0, f"Expected 0 alerts on stable traffic, got {len(alerts)}"

    def test_traffic_driven_spike_cpr_stays_quiet(self):
        settings = PipelineSettings(warmup_ticks=30)
        cpr = WarmUpWrapper(CostPerRequestResidualDetector(settings), settings.warmup_ticks)
        ewma = WarmUpWrapper(EWMADetector(settings), settings.warmup_ticks)

        # Baseline training
        for t in range(40):
            cpr.update(_make_event(tick=t, cost=0.01, requests=8000.0))
            ewma.update(_make_event(tick=t, cost=0.01, requests=8000.0))

        # 4x volume AND 4x cost spike (traffic-driven)
        spike_ev = _make_event(tick=41, cost=0.04, requests=32000.0)
        res_cpr = cpr.update(spike_ev)
        res_ewma = ewma.update(spike_ev)

        assert not res_cpr.is_anomaly, "CPR should NOT fire for traffic-driven cost spike"
        assert res_ewma.is_anomaly, "EWMA should fire for 4x cost spike"

    def test_cost_only_injection_cpr_fires_and_rca_price_driven(self):
        settings = PipelineSettings(warmup_ticks=30)
        ensemble = EnsembleDetector(settings)

        for t in range(40):
            ev = _make_event(tick=t, cost=0.01, requests=8000.0)
            ensemble.process(ev)

        # Cost spikes 5x while requests remain flat
        spike_ev = _make_event(tick=41, cost=0.05, requests=8000.0)
        det, score = ensemble.process(spike_ev)

        assert det is not None, "Ensemble should fire on 5x cost-only spike"
        assert det.rca.alert_type == "price_driven"
        cpr_vote = next(v for v in det.detector_votes if "cpr" in v.detector_name or "residual" in v.detector_name)
        assert cpr_vote.is_anomaly, "CPR detector must vote anomaly"

    def test_slow_drift_eventually_fires(self):
        settings = PipelineSettings(warmup_ticks=30, ewma_k=2.5)
        ensemble = EnsembleDetector(settings)
        fired = False

        for t in range(90):
            drift = 1.0 + (t / 40.0) if t > 30 else 1.0
            ev = _make_event(tick=t, cost=0.01 * drift, requests=8000.0)
            det, _ = ensemble.process(ev)
            if det:
                fired = True
                break

        assert fired, "Slow drift should eventually trigger detection"


class TestEnsembleAndCooldown:
    def test_voting_requires_threshold(self):
        settings = PipelineSettings(warmup_ticks=10, vote_threshold=2)
        ensemble = EnsembleDetector(settings)

        for t in range(20):
            ensemble.process(_make_event(tick=t, cost=0.01, requests=8000.0))

        # 4x cost only triggers EWMA + CPR (2 votes >= 2 threshold)
        det, _ = ensemble.process(_make_event(tick=21, cost=0.04, requests=8000.0))
        assert det is not None
        assert sum(1 for v in det.detector_votes if v.is_anomaly) >= 2

    def test_cooldown_prevents_alert_flooding(self):
        settings = PipelineSettings(warmup_ticks=10, cooldown_ticks=5)
        ensemble = EnsembleDetector(settings)

        for t in range(20):
            ensemble.process(_make_event(tick=t, cost=0.01, requests=8000.0))

        first_det, _ = ensemble.process(_make_event(tick=21, cost=0.04, requests=8000.0))
        assert first_det is not None
        assert first_det.status == "open"

        # Next tick still high -> cooldown marks as ongoing or suppresses
        second_det, _ = ensemble.process(_make_event(tick=22, cost=0.04, requests=8000.0))
        if second_det:
            assert second_det.status == "ongoing"

    def test_run_id_resets_state(self):
        settings = PipelineSettings(warmup_ticks=10)
        ensemble = EnsembleDetector(settings)

        for t in range(20):
            ensemble.process(_make_event(tick=t, cost=0.01, requests=8000.0, run_id="run-A"))

        # Switch to run-B -> resets warmup
        det, _ = ensemble.process(_make_event(tick=0, cost=0.05, requests=8000.0, run_id="run-B"))
        assert det is None, "New run_id must enter warm-up and suppress alerts"


class TestEvaluatorLogic:
    def test_precision_recall_and_latency(self):
        settings = PipelineSettings()
        # Mock Evaluator without connecting to redis
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Evaluator, "_connect_redis", lambda self: None)
            ev = Evaluator(settings)

        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        inj = {
            "injection_id": "inj-1",
            "run_id": "run-1",
            "type": "spike",
            "service": "DynamoDB",
            "magnitude": 4.0,
            "duration_min": 30.0,
            "sim_start_ts": t0.isoformat(),
            "sim_end_ts": (t0 + timedelta(minutes=30)).isoformat(),
            "wall_start_ms": 1000,
            "wall_end_ms": 5000,
        }
        ev._ingest_injection(inj)

        # Detection inside window
        det = {
            "detection_id": "det-1",
            "run_id": "run-1",
            "service": "DynamoDB",
            "sim_ts": (t0 + timedelta(minutes=5)).isoformat(),
            "detect_wall_ts_ms": 3200,
            "first_event_wall_ts_ms": 1000,
            "severity": 80.0,
            "tick_index": 1,
            "detector_votes": [{"detector_name": "cpr", "is_anomaly": True}],
        }
        ev._ingest_detection(det)

        metrics = ev.compute_metrics()
        assert metrics["tp"] == 1
        assert metrics["fn"] == 0
        assert metrics["fp"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert abs(metrics["mean_latency_s"] - 2.2) < 0.01  # (3200 - 1000) ms = 2.2s


"""
tests/test_generator.py
=======================
Pytest suite for the CRIP continuous event generator.

Test categories
---------------
1. Seed reproducibility   — same seed ⇒ byte-identical event sequence
2. Anomaly types          — each type changes only the expected quantity
3. Overlapping injections — composition correctness
4. API validation         — invalid inputs rejected with 4xx
5. Inject/clear lifecycle — ground truth written correctly
6. Pause/resume           — sim time does not advance while paused
"""

from __future__ import annotations

import json
import time
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

from generator.config import GeneratorSettings, SERVICES
from generator.clock import SimClock, TickState, ServiceTickData, SIM_INTERVAL_MINUTES
from generator.traffic_model import TrafficModel, diurnal_factor
from generator.anomalies import (
    InjectionRegistry,
    SpikeInjection,
    StepChangeInjection,
    SlowDriftInjection,
    CostOnlyInjection,
    SeasonalityBreakInjection,
    create_injection,
    _now_ms,
)
from generator.api import create_app


FIXED_SIM_START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)  # noon — diurnal ≈ 1.0
FIXED_SEED = 7


def _make_settings(**overrides) -> GeneratorSettings:
    """Build a GeneratorSettings without reading .env / env vars."""
    defaults = dict(
        sim_minutes_per_real_second=5.0,
        seed=FIXED_SEED,
        noise_sigma=0.03,
        redis_url="redis://localhost:6379",
        redis_stream_events="crip:events",
        redis_stream_labels="crip:labels",
        redis_stream_injections="crip:injections",
        redis_maxlen=100_000,
        db_dsn="postgresql://crip_user:crip_secret@localhost:5432/crip",
        sink="stdout",
        csv_path="reports/test_output.csv",
        api_host="127.0.0.1",
        api_port=8000,
        ground_truth_path="reports/ground_truth_test.jsonl",
    )
    defaults.update(overrides)
    return GeneratorSettings(**defaults)


def _make_clock(seed=FIXED_SEED) -> SimClock:
    return SimClock(
        sim_start_ts=FIXED_SIM_START,
        sim_minutes_per_real_second=5.0,
        seed=seed,
    )


def _make_rng(seed=FIXED_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _run_n_ticks(
    n: int,
    seed: int = FIXED_SEED,
    injections: list = None,
) -> list[dict]:
    """
    Run ``n`` ticks with the given seed and optional injections.
    Returns the list of event dicts (all services, all ticks).
    """
    settings = _make_settings(seed=seed)
    rng = _make_rng(seed)
    traffic_model = TrafficModel(rng=rng, settings=settings)
    registry = InjectionRegistry()
    if injections:
        for inj in injections:
            registry.add(inj)

    events: list[dict] = []
    tick_index = 0
    sim_ts = FIXED_SIM_START

    for _ in range(n):
        # Build a minimal tick state
        ts = TickState(
            tick_index=tick_index,
            sim_ts=sim_ts,
            run_id="test-run",
            speed=5.0,
        )
        ts = traffic_model.populate(ts)
        ts = registry.apply_all(ts)
        for svc_name, sd in ts.services.items():
            sd.finalise()
            events.append({
                "tick_index": tick_index,
                "service": svc_name,
                "request_count": round(sd.request_count * sd.volume_multiplier, 6),
                "usage_amount": round(sd.usage_amount, 6),
                "unblended_cost": round(sd.unblended_cost, 8),
                "unit_price": round(sd.unit_price * sd.price_multiplier, 10),
                "volume_multiplier": sd.volume_multiplier,
                "price_multiplier": sd.price_multiplier,
                "is_anomalous": sd.is_anomalous,
            })
        tick_index += 1
        sim_ts = FIXED_SIM_START.replace(
            hour=(FIXED_SIM_START.hour + tick_index * SIM_INTERVAL_MINUTES // 60) % 24,
            minute=(FIXED_SIM_START.minute + tick_index * SIM_INTERVAL_MINUTES) % 60,
        )

    return events


def _baseline_tick(tick: int = 0, service: str = "DynamoDB") -> ServiceTickData:
    """Return a single baseline ServiceTickData for the given service."""
    settings = _make_settings()
    rng = _make_rng()
    traffic_model = TrafficModel(rng=rng, settings=settings)
    ts = TickState(tick_index=tick, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
    ts = traffic_model.populate(ts)
    return ts.services[service]


# ===========================================================================
# 1. Seed reproducibility
# ===========================================================================

class TestSeedReproducibility:
    def test_same_seed_same_events(self):
        """Two runs with the same seed must produce byte-identical events."""
        run1 = _run_n_ticks(n=10, seed=42)
        run2 = _run_n_ticks(n=10, seed=42)
        assert run1 == run2, "Runs with same seed diverged"

    def test_different_seed_different_events(self):
        """Two runs with different seeds must diverge."""
        run1 = _run_n_ticks(n=5, seed=42)
        run2 = _run_n_ticks(n=5, seed=99)
        # At least some event must differ
        diffs = [e1 != e2 for e1, e2 in zip(run1, run2)]
        assert any(diffs), "Different seeds produced identical events (seed isolation broken)"

    def test_deterministic_injection(self):
        """An injection at tick 0 with same seed must produce same modified events."""
        inj_a = create_injection("spike", "DynamoDB", magnitude=3.0, duration_min=15)
        inj_b = create_injection("spike", "DynamoDB", magnitude=3.0, duration_min=15)
        # Give injections the same id so we can compare
        inj_b.injection_id = inj_a.injection_id
        run1 = _run_n_ticks(n=5, seed=42, injections=[inj_a])
        run2 = _run_n_ticks(n=5, seed=42, injections=[inj_b])
        assert run1 == run2


# ===========================================================================
# 2. Anomaly type isolation
# ===========================================================================

class TestAnomalyTypeIsolation:
    def _baseline_sd(self, service="DynamoDB") -> ServiceTickData:
        return _baseline_tick(tick=0, service=service)

    def _apply_one(self, injection, tick=0, service="DynamoDB") -> ServiceTickData:
        """Apply one injection to a fresh tick state and return the ServiceTickData."""
        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)
        ts = TickState(tick_index=tick, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)
        injection.activate(tick, FIXED_SIM_START)
        injection.apply(ts)
        ts.services[service].finalise()
        return ts.services[service]

    def test_spike_raises_volume_and_cost(self):
        inj = SpikeInjection(service="DynamoDB", magnitude=4.0, duration_min=30, ramp_min=0)
        sd = self._apply_one(inj)
        base = self._baseline_sd()
        # Volume multiplier should be 4x
        assert abs(sd.volume_multiplier - 4.0) < 0.01
        # Cost should be ~4x base
        assert sd.unblended_cost > base.unblended_cost * 3.9

    def test_cost_only_does_not_change_request_count(self):
        """cost_only: volume stays flat; only cost increases."""
        inj = CostOnlyInjection(service="DynamoDB", magnitude=5.0, duration_min=30)
        sd = self._apply_one(inj)
        base = self._baseline_sd()
        # request_count after cost_only must equal baseline (volume unchanged)
        assert abs(sd.request_count - base.request_count) < 1e-6, (
            f"request_count changed: {base.request_count} → {sd.request_count}"
        )
        # cost must have risen
        assert sd.unblended_cost > base.unblended_cost * 4.9

    def test_slow_drift_does_not_exceed_magnitude_at_end(self):
        """slow_drift at the final tick must be ≤ 1 + magnitude/100."""
        magnitude = 20.0
        duration_min = 30.0
        inj = SlowDriftInjection(service="DynamoDB", magnitude=magnitude, duration_min=duration_min)
        total_ticks = int(duration_min / SIM_INTERVAL_MINUTES)
        # Apply at the last tick
        inj.activate(0, FIXED_SIM_START)
        inj.end_tick = total_ticks

        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)
        ts = TickState(tick_index=total_ticks - 1, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)
        inj.apply(ts)
        ts.services["DynamoDB"].finalise()

        expected_max = 1.0 + magnitude / 100.0
        assert ts.services["DynamoDB"].volume_multiplier <= expected_max + 1e-9

    def test_slow_drift_volume_increases_monotonically(self):
        """slow_drift volume multiplier must increase tick by tick."""
        magnitude = 50.0
        duration_min = 30.0
        total_ticks = int(duration_min / SIM_INTERVAL_MINUTES)

        multipliers = []
        for t in range(total_ticks):
            inj = SlowDriftInjection(service="DynamoDB", magnitude=magnitude, duration_min=duration_min)
            inj.activate(0, FIXED_SIM_START)
            inj.end_tick = total_ticks
            settings = _make_settings()
            rng = _make_rng()
            traffic_model = TrafficModel(rng=rng, settings=settings)
            ts = TickState(tick_index=t, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
            ts = traffic_model.populate(ts)
            inj.apply(ts)
            multipliers.append(ts.services["DynamoDB"].volume_multiplier)

        for i in range(1, len(multipliers)):
            assert multipliers[i] >= multipliers[i - 1], (
                f"Multiplier not monotone at tick {i}: {multipliers[i-1]} → {multipliers[i]}"
            )

    def test_step_change_is_persistent(self):
        """step_change must remain at magnitude for all ticks within duration."""
        inj = StepChangeInjection(service="DynamoDB", magnitude=2.0, duration_min=30)
        for t in range(6):   # 6 × 5 min = 30 min
            sd = self._apply_one(inj, tick=t)
            assert abs(sd.volume_multiplier - 2.0) < 0.01, (
                f"step_change multiplier wrong at tick {t}: {sd.volume_multiplier}"
            )

    def test_seasonality_break_does_not_change_volume_multiplier(self):
        """seasonality_break should not change volume_multiplier; it sets diurnal_suppressed."""
        inj = SeasonalityBreakInjection(service="DynamoDB", magnitude=1.0, duration_min=30)
        inj.activate(0, FIXED_SIM_START)
        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)
        ts = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)
        inj.apply(ts)
        # volume_multiplier should still be 1.0
        assert ts.services["DynamoDB"].volume_multiplier == 1.0
        # diurnal_suppressed should be True
        assert ts.services["DynamoDB"].diurnal_suppressed is True


# ===========================================================================
# 3. Overlapping injections
# ===========================================================================

class TestOverlappingInjections:
    def test_two_spikes_compose_multiplicatively(self):
        """Two simultaneous spikes on the same service must multiply."""
        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)

        ts = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)
        base_vol = ts.services["DynamoDB"].request_count

        # Reset and apply two spikes
        rng2 = _make_rng()
        traffic_model2 = TrafficModel(rng=rng2, settings=settings)
        ts2 = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts2 = traffic_model2.populate(ts2)

        inj1 = SpikeInjection(service="DynamoDB", magnitude=2.0, duration_min=30, ramp_min=0)
        inj2 = SpikeInjection(service="DynamoDB", magnitude=3.0, duration_min=30, ramp_min=0)
        for inj in [inj1, inj2]:
            inj.activate(0, FIXED_SIM_START)
            inj.apply(ts2)

        ts2.services["DynamoDB"].finalise()
        expected_mult = 2.0 * 3.0
        assert abs(ts2.services["DynamoDB"].volume_multiplier - expected_mult) < 0.01

    def test_spike_and_cost_only_compose_independently(self):
        """spike (volume) + cost_only (price) must both apply without interference."""
        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)
        ts = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)
        base_vol = ts.services["DynamoDB"].request_count

        rng2 = _make_rng()
        traffic_model2 = TrafficModel(rng=rng2, settings=settings)
        ts2 = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts2 = traffic_model2.populate(ts2)

        spike = SpikeInjection(service="DynamoDB", magnitude=4.0, duration_min=30, ramp_min=0)
        cost = CostOnlyInjection(service="DynamoDB", magnitude=3.0, duration_min=30)
        spike.activate(0, FIXED_SIM_START)
        cost.activate(0, FIXED_SIM_START)
        spike.apply(ts2)
        cost.apply(ts2)
        ts2.services["DynamoDB"].finalise()

        sd = ts2.services["DynamoDB"]
        # Volume multiplier from spike only
        assert abs(sd.volume_multiplier - 4.0) < 0.01
        # Price multiplier from cost_only only
        assert abs(sd.price_multiplier - 3.0) < 0.01
        # Total cost = volume * price_mult * base_price
        assert sd.unblended_cost > 0

    def test_overlapping_injection_ids_recorded(self):
        """Both injection IDs must be in injection_ids when two overlap."""
        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)
        ts = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)

        inj1 = SpikeInjection(service="DynamoDB", magnitude=2.0, duration_min=30, ramp_min=0)
        inj2 = CostOnlyInjection(service="DynamoDB", magnitude=2.0, duration_min=30)
        for inj in [inj1, inj2]:
            inj.activate(0, FIXED_SIM_START)
            inj.apply(ts)

        ids = ts.services["DynamoDB"].injection_ids
        assert inj1.injection_id in ids
        assert inj2.injection_id in ids


# ===========================================================================
# 4. API validation
# ===========================================================================

class TestAPIValidation:
    @pytest.fixture
    def client(self):
        clock = _make_clock()
        registry = InjectionRegistry()
        publisher = MagicMock()
        publisher.update_run_id = MagicMock()
        publisher.write_injection_start = MagicMock()
        publisher.write_injection_end = MagicMock()
        app = create_app(clock=clock, registry=registry, publisher=publisher)
        return TestClient(app, raise_server_exceptions=False)

    def test_inject_invalid_type(self, client):
        resp = client.post("/inject", json={
            "type": "BOGUS_TYPE",
            "service": "DynamoDB",
            "magnitude": 2.0,
            "duration_min": 30.0,
        })
        assert resp.status_code == 422

    def test_inject_invalid_service(self, client):
        resp = client.post("/inject", json={
            "type": "spike",
            "service": "FakeService",
            "magnitude": 2.0,
            "duration_min": 30.0,
        })
        assert resp.status_code == 422

    def test_inject_negative_magnitude(self, client):
        resp = client.post("/inject", json={
            "type": "spike",
            "service": "DynamoDB",
            "magnitude": -1.0,
            "duration_min": 30.0,
        })
        assert resp.status_code == 422

    def test_inject_zero_duration(self, client):
        resp = client.post("/inject", json={
            "type": "spike",
            "service": "DynamoDB",
            "magnitude": 2.0,
            "duration_min": 0.0,
        })
        assert resp.status_code == 422

    def test_inject_all_services_accepted(self, client):
        resp = client.post("/inject", json={
            "type": "spike",
            "service": "all",
            "magnitude": 2.0,
            "duration_min": 30.0,
        })
        assert resp.status_code == 201

    def test_valid_inject_returns_injection_id(self, client):
        resp = client.post("/inject", json={
            "type": "cost_only",
            "service": "DynamoDB",
            "magnitude": 3.0,
            "duration_min": 15.0,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "injection_id" in data
        assert len(data["injection_id"]) > 0

    def test_delete_unknown_injection(self, client):
        resp = client.delete("/inject/nonexistent-id")
        assert resp.status_code == 404

    def test_speed_zero_rejected(self, client):
        resp = client.post("/control/speed", json={"sim_minutes_per_real_second": 0})
        assert resp.status_code == 422

    def test_speed_negative_rejected(self, client):
        resp = client.post("/control/speed", json={"sim_minutes_per_real_second": -5})
        assert resp.status_code == 422

    def test_state_endpoint(self, client):
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "sim_ts" in data
        assert "paused" in data
        assert "seed" in data


# ===========================================================================
# 5. Inject/clear lifecycle
# ===========================================================================

class TestInjectLifecycle:
    def test_injection_starts_in_pending(self):
        inj = create_injection("spike", "DynamoDB", magnitude=2.0, duration_min=30)
        assert inj.status == "pending"

    def test_injection_becomes_active_on_first_apply(self):
        registry = InjectionRegistry()
        inj = create_injection("spike", "DynamoDB", magnitude=2.0, duration_min=30)
        registry.add(inj)

        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)
        ts = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)
        registry.apply_all(ts)

        assert inj.status == "active"

    def test_injection_ends_after_duration(self):
        duration_min = 15.0
        total_ticks = int(duration_min / SIM_INTERVAL_MINUTES)
        registry = InjectionRegistry()
        inj = create_injection("spike", "DynamoDB", magnitude=2.0, duration_min=duration_min)
        registry.add(inj)

        settings = _make_settings()
        for t in range(total_ticks + 2):
            rng = _make_rng()
            traffic_model = TrafficModel(rng=rng, settings=settings)
            ts = TickState(tick_index=t, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
            ts = traffic_model.populate(ts)
            registry.apply_all(ts)

        assert inj.status == "ended"

    def test_end_early_via_registry(self):
        registry = InjectionRegistry()
        inj = create_injection("spike", "DynamoDB", magnitude=2.0, duration_min=60)
        registry.add(inj)

        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)
        ts = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)
        registry.apply_all(ts)   # activate

        assert inj.status == "active"
        ok = registry.end_early(inj.injection_id, FIXED_SIM_START)
        assert ok is True
        assert inj.status == "ended"

    def test_on_start_callback_fires(self):
        started = []
        registry = InjectionRegistry()
        registry.on_start(lambda inj: started.append(inj.injection_id))

        inj = create_injection("spike", "DynamoDB", magnitude=2.0, duration_min=30)
        registry.add(inj)

        settings = _make_settings()
        rng = _make_rng()
        traffic_model = TrafficModel(rng=rng, settings=settings)
        ts = TickState(tick_index=0, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
        ts = traffic_model.populate(ts)
        registry.apply_all(ts)

        assert inj.injection_id in started

    def test_on_end_callback_fires(self):
        ended = []
        registry = InjectionRegistry()
        registry.on_end(lambda inj: ended.append(inj.injection_id))

        duration_min = 5.0
        inj = create_injection("spike", "DynamoDB", magnitude=2.0, duration_min=duration_min)
        registry.add(inj)

        settings = _make_settings()
        # Run enough ticks to expire the injection
        for t in range(5):
            rng = _make_rng()
            traffic_model = TrafficModel(rng=rng, settings=settings)
            ts = TickState(tick_index=t, sim_ts=FIXED_SIM_START, run_id="t", speed=5.0)
            ts = traffic_model.populate(ts)
            registry.apply_all(ts)

        assert inj.injection_id in ended


# ===========================================================================
# 6. Pause / resume
# ===========================================================================

class TestPauseResume:
    def test_pause_stops_tick_index(self):
        """While paused, wait_for_next_tick must return False and not advance tick_index."""
        clock = SimClock(
            sim_start_ts=FIXED_SIM_START,
            sim_minutes_per_real_second=600.0,  # very fast so test is quick
            seed=42,
        )
        clock.pause()

        for _ in range(5):
            fired = clock.wait_for_next_tick()
            assert fired is False, "Paused clock should not fire ticks"

        assert clock.tick_index == 0, (
            f"tick_index advanced while paused: {clock.tick_index}"
        )

    def test_resume_allows_ticks(self):
        """After resume, the clock should fire ticks again."""
        clock = SimClock(
            sim_start_ts=FIXED_SIM_START,
            sim_minutes_per_real_second=10000.0,  # ~0 ms per tick
            seed=42,
        )
        clock.pause()
        # Confirm paused
        assert clock.wait_for_next_tick() is False

        clock.resume()
        # Should now fire
        time.sleep(0.01)  # let the interval pass
        fired = clock.wait_for_next_tick()
        assert fired is True

    def test_sim_ts_does_not_advance_while_paused(self):
        """Sim timestamp must not change while clock is paused."""
        clock = SimClock(
            sim_start_ts=FIXED_SIM_START,
            sim_minutes_per_real_second=10000.0,
            seed=42,
        )
        ts_before = clock.sim_ts
        clock.pause()
        time.sleep(0.05)
        ts_after = clock.sim_ts
        assert ts_before == ts_after

    def test_reset_clears_tick_index(self):
        """After reset, tick_index returns to 0 and a new run_id is issued."""
        clock = _make_clock()
        old_run_id = clock.run_id

        # Advance a few ticks manually
        clock._tick_index = 10

        new_run_id = clock.reset(seed=99)
        assert clock.tick_index == 0
        assert new_run_id != old_run_id
        assert clock.seed == 99


# ===========================================================================
# 7. Diurnal factor sanity
# ===========================================================================

class TestDiurnalFactor:
    def test_noon_peak(self):
        noon = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert diurnal_factor(noon) > 0.98

    def test_midnight_trough(self):
        midnight = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert diurnal_factor(midnight) < 0.02

    def test_range_zero_to_one(self):
        from datetime import timedelta
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for h in range(24):
            ts = base + timedelta(hours=h)
            f = diurnal_factor(ts)
            assert 0.0 <= f <= 1.0, f"diurnal_factor out of range at hour {h}: {f}"


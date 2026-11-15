"""
tests/test_generator_mitigate.py
================================
Unit tests for generator mitigation and rate-limiting functionality.
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pytest
from starlette.testclient import TestClient

from generator.clock import SimClock, TickState
from generator.config import GeneratorSettings, SERVICES
from generator.traffic_model import TrafficModel
from generator.anomalies import InjectionRegistry, SpikeInjection
from generator.mitigations import Mitigation, MitigationRegistry
from generator.publisher import Publisher
from generator.api import create_app


def test_mitigation_applies_factor_to_volume_and_cost() -> None:
    """A mitigation with factor 0.6 reduces traffic and cost by exactly 40%."""
    sim_start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    clock = SimClock(sim_start_ts=sim_start, seed=42)
    settings = GeneratorSettings(sink="stdout", noise_sigma=0.0)
    rng = np.random.default_rng(42)
    traffic_model = TrafficModel(rng=rng, settings=settings)

    registry = InjectionRegistry()
    mitigation_registry = MitigationRegistry()

    # Step 1: Unmitigated tick
    tick1 = clock.advance_tick()
    tick1 = traffic_model.populate(tick1)
    tick1 = registry.apply_all(tick1)
    tick1 = mitigation_registry.apply_all(tick1)
    for sd in tick1.services.values():
        sd.finalise()

    base_ddb_volume = tick1.services["DynamoDB"].request_count
    base_ddb_cost = tick1.services["DynamoDB"].unblended_cost

    # Step 2: Add mitigation with factor 0.6
    m = Mitigation(
        mitigation_id="mit-test-1",
        service="DynamoDB",
        factor=0.6,
        duration_min=15.0,
    )
    mitigation_registry.add(m)

    tick2 = clock.advance_tick()
    rng2 = np.random.default_rng(42)
    traffic_model2 = TrafficModel(rng=rng2, settings=settings)
    tick2 = traffic_model2.populate(tick2)
    tick2 = registry.apply_all(tick2)
    tick2 = mitigation_registry.apply_all(tick2)
    for sd in tick2.services.values():
        sd.finalise()

    mit_ddb_volume = tick2.services["DynamoDB"].request_count * tick2.services["DynamoDB"].volume_multiplier
    mit_ddb_cost = tick2.services["DynamoDB"].unblended_cost

    assert np.isclose(mit_ddb_volume, base_ddb_volume * 0.6, rtol=1e-3)
    assert np.isclose(mit_ddb_cost, base_ddb_cost * 0.6, rtol=1e-3)


def test_mitigation_api_endpoints() -> None:
    """Verify POST /mitigate, GET /mitigations, and DELETE /mitigate/{id}."""
    from unittest.mock import MagicMock
    from fastapi import HTTPException
    from generator.api import MitigateRequest

    clock = SimClock()
    registry = InjectionRegistry()
    mitigation_registry = MitigationRegistry()
    publisher = MagicMock()

    app = create_app(
        clock=clock,
        registry=registry,
        publisher=publisher,
        mitigation_registry=mitigation_registry,
    )

    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}

    # 1. POST /mitigate with valid service succeeds
    mitigate_fn = routes["/mitigate"]
    req = MitigateRequest(
        service="AmazonApiGateway",
        factor=0.7,
        duration_min=20,
        mitigation_id="test-mit-123",
    )
    res = mitigate_fn(req)
    assert res.mitigation_id == "test-mit-123"
    assert res.factor == 0.7

    # 2. GET /mitigations
    list_fn = routes["/mitigations"]
    mits = list_fn()
    assert len(mits) == 1
    assert mits[0]["mitigation_id"] == "test-mit-123"

    # 3. DELETE /mitigate/{mitigation_id}
    del_fn = routes["/mitigate/{mitigation_id}"]
    del_res = del_fn("test-mit-123")
    assert del_res["status"] == "ended"

    # 4. Second DELETE fails with 404
    with pytest.raises(HTTPException) as exc_info:
        del_fn("test-mit-123")
    assert exc_info.value.status_code == 404

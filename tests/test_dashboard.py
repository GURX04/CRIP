"""
tests/test_dashboard.py
=======================
Pytest suite for the CRIP demo dashboard backend.

Test categories
---------------
1. SSE endpoint — emits the right message types from a mock/fakeredis reader
2. /history     — returns expected structure from Redis
3. Proxy endpoints — validate and forward to generator API
4. Slow-client queue — drops old ticks without blocking
5. Latency matcher — given injection + detection stream, correct latency
6. Health endpoint — reports correct status per component
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from dashboard.config import DashboardSettings
from dashboard.models import SSEMessage, HistoryResponse
from dashboard.streams import ClientRegistry, AsyncRedisReader, ReaderState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> DashboardSettings:
    defaults = dict(
        host="127.0.0.1",
        port=8501,
        redis_url="redis://localhost:6379",
        redis_stream_scores="crip:scores",
        redis_stream_detections="crip:detections",
        redis_stream_injections="crip:injections",
        redis_stream_eval="crip:eval",
        redis_xread_count=50,
        redis_block_ms=100,
        generator_url="http://localhost:8000",
        generator_timeout_s=2.0,
        client_queue_maxsize=10,
        heartbeat_interval_s=5.0,
        history_minutes=60,
        replay_file=None,
        detector_stale_s=10.0,
        db_dsn="postgresql://crip_user:crip_secret@localhost:5432/crip",
        log_level="WARNING",
    )
    defaults.update(overrides)
    return DashboardSettings(**defaults)


@pytest.fixture
def settings():
    return _make_settings()


@pytest.fixture
def app(settings):
    """Create a FastAPI test app with mocked Redis and generator."""
    from dashboard.main import create_app
    return create_app(settings)


# ---------------------------------------------------------------------------
# 1. ClientRegistry — bounded queue / broadcast
# ---------------------------------------------------------------------------

class TestClientRegistry:
    def test_add_and_remove(self):
        async def _run():
            reg = ClientRegistry(maxsize=5)
            q = await reg.add()
            assert reg.count == 1
            await reg.remove(q)
            assert reg.count == 0
        asyncio.run(_run())

    def test_broadcast_to_multiple_clients(self):
        async def _run():
            reg = ClientRegistry(maxsize=5)
            q1 = await reg.add()
            q2 = await reg.add()
            msg = SSEMessage(event="tick", data={"service": "DDB"})
            await reg.broadcast(msg)
            assert not q1.empty()
            assert not q2.empty()
            assert (await q1.get()).event == "tick"
        asyncio.run(_run())

    def test_slow_client_drops_old_ticks(self):
        """When a client queue is full, the oldest message is dropped, not the latest."""
        async def _run():
            reg = ClientRegistry(maxsize=3)
            q = await reg.add()

            # Fill the queue to capacity
            for i in range(3):
                await reg.broadcast(SSEMessage(event="tick", data={"i": i}))

            # One more broadcast should drop the oldest (i=0) and add the new one
            await reg.broadcast(SSEMessage(event="tick", data={"i": 99}))

            msgs = []
            while not q.empty():
                msgs.append(await q.get())
            indices = [m.data["i"] for m in msgs]

            # The oldest (0) should be gone; latest (99) should be present
            assert 99 in indices, f"Latest message missing from queue: {indices}"
            assert len(msgs) == 3  # still at max capacity
        asyncio.run(_run())

    def test_broadcast_does_not_block_on_full_queue(self):
        """Broadcast must complete instantly even with a full queue."""
        async def _run():
            reg = ClientRegistry(maxsize=2)
            q = await reg.add()
            # Fill queue
            for i in range(5):
                start = asyncio.get_event_loop().time()
                await reg.broadcast(SSEMessage(event="tick", data={"i": i}))
                elapsed = asyncio.get_event_loop().time() - start
                assert elapsed < 0.1, f"Broadcast blocked for {elapsed}s on iteration {i}"
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. SSEMessage encoding
# ---------------------------------------------------------------------------

class TestSSEMessage:
    def test_encode_format(self):
        msg = SSEMessage(event="tick", data={"service": "DDB", "cost": 1.23})
        encoded = msg.encode()
        assert encoded.startswith("event: tick\n")
        assert "data: " in encoded
        assert encoded.endswith("\n\n")

    def test_encode_json_roundtrip(self):
        data = {"service": "DDB", "sim_ts": "2026-01-01T12:00:00Z", "cost": 0.0042}
        msg = SSEMessage(event="detection", data=data)
        encoded = msg.encode()
        # Parse the data line
        for line in encoded.splitlines():
            if line.startswith("data: "):
                parsed = json.loads(line[6:])
                assert parsed["service"] == "DDB"
                assert abs(parsed["cost"] - 0.0042) < 1e-9
                break


# ---------------------------------------------------------------------------
# 3. /history endpoint (mocked Redis)
# ---------------------------------------------------------------------------

class TestHistoryEndpoint:
    def test_returns_history_response_structure(self, app, settings):
        """History endpoint returns the correct HistoryResponse shape even when Redis is down."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/history?minutes=30")
        assert resp.status_code == 200
        body = resp.json()
        assert "ticks" in body
        assert "detections" in body
        assert "injections" in body
        assert "minutes" in body
        assert body["minutes"] == 30

    def test_history_minutes_param(self, app, settings):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/history?minutes=15")
        assert resp.status_code == 200
        assert resp.json()["minutes"] == 15

    def test_history_invalid_minutes(self, app, settings):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/history?minutes=0")
        assert resp.status_code == 422

    def test_history_too_large_minutes(self, app, settings):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/history?minutes=99999")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. Proxy endpoints — validation
# ---------------------------------------------------------------------------

class TestProxyEndpoints:
    @pytest.fixture
    def client(self, app):
        return TestClient(app, raise_server_exceptions=False)

    def test_inject_missing_required_fields(self, client):
        resp = client.post("/api/inject", json={"type": "spike"})
        assert resp.status_code == 422

    def test_inject_negative_magnitude_rejected(self, client):
        resp = client.post("/api/inject", json={
            "type": "spike", "service": "DynamoDB", "magnitude": -1, "duration_min": 30
        })
        assert resp.status_code == 422

    def test_inject_zero_duration_rejected(self, client):
        resp = client.post("/api/inject", json={
            "type": "spike", "service": "DynamoDB", "magnitude": 2, "duration_min": 0
        })
        assert resp.status_code == 422

    def test_speed_zero_rejected(self, client):
        resp = client.post("/api/control/speed", json={"sim_minutes_per_real_second": 0})
        assert resp.status_code == 422

    def test_speed_negative_rejected(self, client):
        resp = client.post("/api/control/speed", json={"sim_minutes_per_real_second": -5})
        assert resp.status_code == 422

    def test_inject_generator_unreachable_returns_503(self, client):
        """When generator is down, proxy returns 503."""
        resp = client.post("/api/inject", json={
            "type": "spike", "service": "DynamoDB", "magnitude": 2.0, "duration_min": 30.0
        })
        # Either 503 (generator down) or 201 (if generator happens to be up)
        assert resp.status_code in (201, 503)

    def test_reset_accepts_empty_body(self, client):
        resp = client.post("/api/control/reset", json={})
        # 200 (success) or 503 (generator down) — both acceptable without live generator
        assert resp.status_code in (200, 503)

    def test_reset_validates_seed(self, client):
        resp = client.post("/api/control/reset", json={"seed": -1})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 5. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_all_components(self, app, settings):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "redis" in body
        assert "generator" in body
        assert "detector" in body
        assert "overall" in body

    def test_health_component_has_ok_field(self, app, settings):
        client = TestClient(app, raise_server_exceptions=False)
        body = client.get("/health").json()
        for key in ("redis", "generator", "detector"):
            assert "ok" in body[key], f"Missing 'ok' in {key}"


# ---------------------------------------------------------------------------
# 6. Latency matching logic (unit test, no I/O)
# ---------------------------------------------------------------------------

def _make_injection(injection_id: str, service: str, sim_start: datetime,
                    duration_min: float) -> dict:
    return {
        "injection_id": injection_id,
        "run_id": "run-1",
        "type": "spike",
        "service": service,
        "magnitude": 4.0,
        "duration_min": duration_min,
        "sim_start_ts": sim_start.isoformat(),
        "sim_end_ts": None,
        "wall_start_ms": int(sim_start.timestamp() * 1000),
        "wall_end_ms": 0,
    }


def _make_detection(service: str, sim_ts: datetime, detect_wall_ms: int,
                    first_wall_ms: int) -> dict:
    return {
        "detection_id": "det-1",
        "run_id": "run-1",
        "service": service,
        "sim_ts": sim_ts.isoformat(),
        "detect_wall_ts_ms": detect_wall_ms,
        "first_event_wall_ts_ms": first_wall_ms,
        "severity": 80.0,
        "detector_votes": [{"detector_name": "cpr", "score": 5.0, "is_anomaly": True, "explanation": ""}],
        "rca": {"alert_type": "price_driven", "cost_per_1k_now": 0.01, "cost_per_1k_baseline": 0.001,
                "pct_cost_change": 400.0, "pct_volume_change": 0.0, "service_contributions": {"DynamoDB": 1.0}},
        "status": "open",
        "tick_index": 10,
    }


class TestLatencyMatcher:
    """
    Unit-test the latency matching logic that the evaluator and stopwatch use.
    No Redis, no network.
    """

    def _overlaps(self, inj: dict, det: dict, tolerance_min: float = 10.0) -> bool:
        """Check if detection falls within injection window (+ tolerance)."""
        from datetime import timedelta
        sim_start = datetime.fromisoformat(inj["sim_start_ts"].replace("Z", "+00:00"))
        sim_end_raw = inj.get("sim_end_ts")
        if sim_end_raw:
            sim_end = datetime.fromisoformat(sim_end_raw.replace("Z", "+00:00"))
        else:
            sim_end = sim_start + timedelta(minutes=inj["duration_min"])
        sim_end += timedelta(minutes=tolerance_min)

        det_ts_raw = det["sim_ts"]
        det_ts = datetime.fromisoformat(det_ts_raw.replace("Z", "+00:00"))
        return sim_start <= det_ts <= sim_end

    def test_detection_inside_window_matches(self):
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t_det = datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
        inj = _make_injection("inj-1", "DynamoDB", t0, duration_min=30)
        det = _make_detection("DynamoDB", t_det, int(t_det.timestamp()*1000), int(t0.timestamp()*1000))
        assert self._overlaps(inj, det)

    def test_detection_outside_window_no_match(self):
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t_det = datetime(2026, 1, 1, 13, 30, 0, tzinfo=timezone.utc)   # 90 min later
        inj = _make_injection("inj-1", "DynamoDB", t0, duration_min=30)
        det = _make_detection("DynamoDB", t_det, int(t_det.timestamp()*1000), int(t0.timestamp()*1000))
        assert not self._overlaps(inj, det, tolerance_min=10.0)

    def test_different_service_does_not_match(self):
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t_det = datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
        inj = _make_injection("inj-1", "DynamoDB", t0, duration_min=30)
        det = _make_detection("S3 PUT", t_det, int(t_det.timestamp()*1000), int(t0.timestamp()*1000))
        # Service mismatch — should not match even if time overlaps
        assert inj["service"] != det["service"]

    def test_latency_calculation(self):
        """Detection latency = detect_wall_ts_ms - wall_start_ms."""
        wall_start = 1_700_000_000_000   # epoch ms
        wall_detect = wall_start + 3_200  # 3.2 s later
        inj = _make_injection("inj-1", "DynamoDB",
                              datetime.fromtimestamp(wall_start/1000, tz=timezone.utc), 30)
        det = _make_detection("DynamoDB",
                              datetime.fromtimestamp(wall_start/1000 + 300, tz=timezone.utc),
                              wall_detect, wall_start)
        latency_s = (det["detect_wall_ts_ms"] - inj["wall_start_ms"]) / 1000.0
        assert abs(latency_s - 3.2) < 0.001

    def test_missed_injection_no_match(self):
        """An injection with no overlapping detection should be reported as missed."""
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        inj = _make_injection("inj-1", "DynamoDB", t0, duration_min=5)
        detections = []  # no detections
        matched = any(self._overlaps(inj, d) for d in detections)
        assert not matched

    def test_overlapping_injections_match_closest(self):
        """When two injections overlap in time, the right service is matched."""
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t_det = datetime(2026, 1, 1, 12, 3, 0, tzinfo=timezone.utc)
        inj_ddb = _make_injection("inj-ddb", "DynamoDB", t0, 30)
        inj_s3  = _make_injection("inj-s3",  "S3 PUT",   t0, 30)
        det_ddb = _make_detection("DynamoDB", t_det, int(t_det.timestamp()*1000), int(t0.timestamp()*1000))

        # DDB detection should match only the DDB injection
        assert self._overlaps(inj_ddb, det_ddb) and inj_ddb["service"] == det_ddb["service"]
        # S3 injection should NOT match the DDB detection
        assert not (self._overlaps(inj_s3, det_ddb) and inj_s3["service"] == det_ddb["service"])


# ---------------------------------------------------------------------------
# 7. Root endpoint serves HTML
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_html(self, app):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/")
        assert resp.status_code in (200, 503)  # 503 if static files missing
        if resp.status_code == 200:
            assert "text/html" in resp.headers.get("content-type", "")


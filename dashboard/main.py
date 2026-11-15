"""
dashboard/main.py
=================
FastAPI application for the CRIP live demo dashboard.

Endpoints
---------
GET  /stream             Server-Sent Events fan-out
GET  /history            Last N simulated minutes of ticks/detections/injections
GET  /health             Redis + generator + detector liveness
POST /api/inject         Proxy to generator POST /inject
DELETE /api/inject/{id}  Proxy to generator DELETE /inject/{id}
POST /api/control/pause  Proxy
POST /api/control/resume Proxy
POST /api/control/speed  Proxy
POST /api/control/reset  Proxy
GET  /api/state          Proxy to generator GET /state
GET  /api/injections     Proxy to generator GET /injections
GET  /                   Serve index.html
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from dashboard.config import DashboardSettings, get_settings
from dashboard.models import (
    HealthStatus, ComponentHealth,
    HistoryResponse, HistoryTick, HistoryDetection, HistoryInjection, HistoryProposal,
    InjectProxyRequest, SpeedProxyRequest, ResetProxyRequest,
    RemediationModeRequest, KillSwitchRequest,
    SSEMessage,
)
from dashboard.streams import AsyncRedisReader, ClientRegistry, ReaderState

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Lifespan — start/stop background reader
# ---------------------------------------------------------------------------

_registry: ClientRegistry = ClientRegistry(200)
_state: ReaderState = ReaderState()
_stop_event: asyncio.Event = asyncio.Event()
_reader_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry, _state, _stop_event, _reader_task
    settings: DashboardSettings = app.state.settings

    _registry = ClientRegistry(maxsize=settings.client_queue_maxsize)
    _state = ReaderState()
    _stop_event = asyncio.Event()

    reader = AsyncRedisReader(settings)
    _reader_task = asyncio.create_task(
        reader.start(_registry, _state, _stop_event)
    )
    log.info("Dashboard stream reader started")
    yield
    _stop_event.set()
    if _reader_task:
        _reader_task.cancel()
        try:
            await _reader_task
        except asyncio.CancelledError:
            pass
    log.info("Dashboard stream reader stopped")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Optional[DashboardSettings] = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="CRIP Live Dashboard",
        description="Real-time FinOps anomaly demo dashboard",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # Serve static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # -----------------------------------------------------------------------
    # SSE stream endpoint
    # -----------------------------------------------------------------------

    @app.get("/stream")
    async def sse_stream(request: Request):
        """Server-Sent Events endpoint. One connection per browser tab."""
        queue = await _registry.add()

        async def event_generator() -> AsyncIterator[str]:
            # Send initial state snapshot
            try:
                snap = await _get_state_snapshot(settings)
                yield SSEMessage(event="state", data=snap).encode()
            except Exception:
                pass

            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg: SSEMessage = await asyncio.wait_for(
                            queue.get(), timeout=settings.heartbeat_interval_s
                        )
                        yield msg.encode()
                    except asyncio.TimeoutError:
                        yield SSEMessage(
                            event="heartbeat", data={"ts_ms": _now_ms()}
                        ).encode()
            finally:
                await _registry.remove(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # -----------------------------------------------------------------------
    # History endpoint
    # -----------------------------------------------------------------------

    @app.get("/history", response_model=HistoryResponse)
    async def get_history(
        minutes: int = Query(default=settings.history_minutes, gt=0, le=1440)
    ):
        """Return the last `minutes` simulated-minutes of data for chart restore."""
        ticks: list[HistoryTick] = []
        detections: list[HistoryDetection] = []
        injections: list[HistoryInjection] = []

        try:
            import redis as redis_lib  # noqa: PLC0415
            r = redis_lib.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.5,
            )
            count = minutes * 12 * 5   # ~12 ticks/sim-min × 5 services

            # Scores stream → ticks
            raw = r.xrevrange(settings.redis_stream_scores, count=count)
            for _id, fields in reversed(raw):
                try:
                    d = json.loads(fields.get("payload", "{}"))
                    ticks.append(HistoryTick(**d))
                except Exception:
                    pass

            # Detections stream
            raw = r.xrevrange(settings.redis_stream_detections, count=500)
            for _id, fields in reversed(raw):
                try:
                    d = json.loads(fields.get("payload", "{}"))
                    detections.append(HistoryDetection(**d))
                except Exception:
                    pass

            # Injections stream
            raw = r.xrevrange(settings.redis_stream_injections, count=200)
            for _id, fields in reversed(raw):
                try:
                    d = json.loads(fields.get("payload", "{}"))
                    injections.append(HistoryInjection(**{
                        k: v for k, v in d.items()
                        if k in HistoryInjection.model_fields
                    }))
                except Exception:
                    pass

            # Proposals stream — keep only the latest snapshot per proposal
            latest_proposals: dict[str, HistoryProposal] = {}
            raw = r.xrevrange(settings.redis_stream_proposals, count=200)

            # XREVRANGE returns newest records first.
            for _id, fields in raw:
                try:
                    d = json.loads(fields.get("payload", "{}"))
                    proposal_id = d.get("proposal_id")

                    if not proposal_id or proposal_id in latest_proposals:
                        continue

                    latest_proposals[proposal_id] = HistoryProposal(**{
                        k: v for k, v in d.items()
                        if k in HistoryProposal.model_fields
                    })
                except Exception:
                    pass

            # Restore chronological order for the dashboard.
            proposals = list(reversed(list(latest_proposals.values())))

        except Exception as exc:
            log.warning("history redis error: %s — returning empty", exc)

        return HistoryResponse(
            ticks=ticks,
            detections=detections,
            injections=injections,
            proposals=proposals,
            minutes=minutes,
        )

    # -----------------------------------------------------------------------
    # Health endpoint
    # -----------------------------------------------------------------------

    @app.get("/health", response_model=HealthStatus)
    async def health():
        redis_ok = False
        redis_detail = ""
        try:
            import redis as redis_lib  # noqa: PLC0415
            r = redis_lib.from_url(settings.redis_url, decode_responses=True,
                                   socket_connect_timeout=0.5)
            r.ping()
            redis_ok = True
        except Exception as exc:
            redis_detail = str(exc)

        gen_ok = False
        gen_detail = ""
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                resp = await client.get(f"{settings.generator_url}/state")
                gen_ok = resp.status_code == 200
        except Exception as exc:
            gen_detail = str(exc)

        age_s = (_now_ms() - _state.last_score_wall_ms) / 1000.0
        det_ok = _state.last_score_wall_ms > 0 and age_s < settings.detector_stale_s
        det_detail = f"last score {age_s:.1f}s ago" if _state.last_score_wall_ms else "no scores yet"

        return HealthStatus(
            redis=ComponentHealth(ok=redis_ok, detail=redis_detail),
            generator=ComponentHealth(ok=gen_ok, detail=gen_detail),
            detector=ComponentHealth(ok=det_ok, detail=det_detail),
            overall=redis_ok and gen_ok,
        )

    # -----------------------------------------------------------------------
    # Generator proxy endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/inject", status_code=201)
    async def proxy_inject(body: InjectProxyRequest):
        return await _proxy_post(settings, "/inject", body.model_dump())

    @app.delete("/api/inject/{injection_id}")
    async def proxy_delete_inject(injection_id: str):
        return await _proxy_delete(settings, f"/inject/{injection_id}")

    @app.post("/api/control/pause")
    async def proxy_pause():
        return await _proxy_post(settings, "/control/pause", {})

    @app.post("/api/control/resume")
    async def proxy_resume():
        return await _proxy_post(settings, "/control/resume", {})

    @app.post("/api/control/speed")
    async def proxy_speed(body: SpeedProxyRequest):
        return await _proxy_post(settings, "/control/speed", body.model_dump())

    @app.post("/api/control/reset")
    async def proxy_reset(body: ResetProxyRequest = ResetProxyRequest()):
        return await _proxy_post(settings, "/control/reset", body.model_dump(exclude_none=True))

    @app.get("/api/state")
    async def proxy_state():
        return await _proxy_get(settings, "/state")

    @app.get("/api/injections")
    async def proxy_injections():
        return await _proxy_get(settings, "/injections")

    # -----------------------------------------------------------------------
    # Remediation control endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/remediation/mode")
    async def set_remediation_mode(body: RemediationModeRequest):
        """Set remediation mode: DRY_RUN | ENFORCE."""
        mode_val = body.mode.upper()
        if mode_val not in ("DRY_RUN", "ENFORCE"):
            raise HTTPException(status_code=400, detail="Invalid mode. Must be DRY_RUN or ENFORCE.")

        try:
            import redis as redis_lib  # noqa: PLC0415
            r = redis_lib.from_url(settings.redis_url, decode_responses=True)
            r.set("crip:remediation:mode", mode_val)
            r.publish("crip:remediation:control", json.dumps({"action": "set_mode", "mode": mode_val}))
        except Exception as exc:
            log.warning("Could not set mode in redis: %s", exc)

        return {"mode": mode_val, "status": "ok"}

    @app.post("/api/remediation/kill-switch")
    async def set_kill_switch(body: KillSwitchRequest):
        """Engage or disengage emergency kill switch."""
        try:
            import redis as redis_lib  # noqa: PLC0415
            r = redis_lib.from_url(settings.redis_url, decode_responses=True)
            r.set("crip:remediation:kill_switch", "1" if body.enabled else "0")
            r.publish("crip:remediation:control", json.dumps({"action": "set_kill_switch", "enabled": body.enabled}))
        except Exception as exc:
            log.warning("Could not set kill switch in redis: %s", exc)

        return {"kill_switch": body.enabled, "status": "ok"}

    @app.get("/api/remediation/state")
    async def get_remediation_state():
        """Get live remediation engine mode and status."""
        mode = "DRY_RUN"
        kill_switch = False
        try:
            import redis as redis_lib  # noqa: PLC0415
            r = redis_lib.from_url(settings.redis_url, decode_responses=True)
            saved_mode = r.get("crip:remediation:mode")
            if saved_mode:
                mode = saved_mode
            saved_ks = r.get("crip:remediation:kill_switch")
            if saved_ks == "1":
                kill_switch = True
        except Exception:
            pass

        return {"mode": mode, "kill_switch": kill_switch}

    
    @app.post("/api/remediation/proposals/{proposal_id}/approve")
    async def approve_remediation_proposal(proposal_id: str):
        """Queue approval for the remediation worker."""
        try:
            import redis as redis_lib  # noqa: PLC0415

            r = redis_lib.from_url(
                settings.redis_url,
                decode_responses=True,
            )

            raw = r.xrevrange(settings.redis_stream_proposals, count=2000)
            target_proposal = None

            for _id, fields in raw:
                try:
                    data = json.loads(fields.get("payload", "{}"))
                    if data.get("proposal_id") == proposal_id:
                        target_proposal = data
                        break
                except Exception:
                    continue

            if not target_proposal:
                raise HTTPException(status_code=404, detail="Proposal not found")

            if target_proposal.get("status") != "pending_approval":
                raise HTTPException(
                    status_code=409,
                    detail="Proposal is not pending approval",
                )

            r.xadd(
                "crip:remediation_approvals",
                {
                    "payload": json.dumps({
                        "proposal_id": proposal_id,
                        "action": "approve",
                    })
                },
                maxlen=10000,
            )

            return {"proposal_id": proposal_id, "status": "approval_queued"}

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/remediation/proposals/{proposal_id}/reject")
    async def reject_remediation_proposal(proposal_id: str):
        """Queue rejection for the remediation worker."""
        try:
            import redis as redis_lib  # noqa: PLC0415

            r = redis_lib.from_url(
                settings.redis_url,
                decode_responses=True,
            )

            raw = r.xrevrange(settings.redis_stream_proposals, count=2000)
            target_proposal = None

            for _id, fields in raw:
                try:
                    data = json.loads(fields.get("payload", "{}"))
                    if data.get("proposal_id") == proposal_id:
                        target_proposal = data
                        break
                except Exception:
                    continue

            if not target_proposal:
                raise HTTPException(status_code=404, detail="Proposal not found")

            if target_proposal.get("status") != "pending_approval":
                raise HTTPException(
                    status_code=409,
                    detail="Proposal is not pending approval",
                )

            r.xadd(
                "crip:remediation_approvals",
                {
                    "payload": json.dumps({
                        "proposal_id": proposal_id,
                        "action": "reject",
                    })
                },
                maxlen=10000,
            )

            return {"proposal_id": proposal_id, "status": "rejection_queued"}

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>CRIP Dashboard — static files not found</h1>", status_code=503)

    return app


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

async def _proxy_post(settings: DashboardSettings, path: str, body: dict):
    try:
        async with httpx.AsyncClient(timeout=settings.generator_timeout_s) as client:
            resp = await client.post(f"{settings.generator_url}{path}", json=body)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Generator unreachable: {exc}")


async def _proxy_delete(settings: DashboardSettings, path: str):
    try:
        async with httpx.AsyncClient(timeout=settings.generator_timeout_s) as client:
            resp = await client.delete(f"{settings.generator_url}{path}")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Generator unreachable: {exc}")


async def _proxy_get(settings: DashboardSettings, path: str):
    try:
        async with httpx.AsyncClient(timeout=settings.generator_timeout_s) as client:
            resp = await client.get(f"{settings.generator_url}{path}")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Generator unreachable: {exc}")


async def _get_state_snapshot(settings: DashboardSettings) -> dict:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.generator_url}/state")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    )
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()


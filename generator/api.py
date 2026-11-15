"""
generator/api.py
================
FastAPI control-plane for the CRIP event generator (port 8000).

Endpoints
---------
POST   /inject                   Schedule a new injection
DELETE /inject/{injection_id}    End an injection early
GET    /injections               List all injections
GET    /state                    Current clock state + active injections
POST   /control/pause
POST   /control/resume
POST   /control/speed            body: {sim_minutes_per_real_second}
POST   /control/reset            body: {seed?, sim_start_ts?}

All request/response bodies use Pydantic models with strict validation.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from generator.config import VALID_SERVICE_NAMES, VALID_INJECTION_TYPES

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class InjectRequest(BaseModel):
    type: str = Field(..., description="Injection type")
    service: str = Field(..., description="Target service name or 'all'")
    magnitude: float = Field(..., gt=0, description="Magnitude multiplier (>0)")
    duration_min: float = Field(..., gt=0, description="Duration in simulated minutes")
    ramp_min: float = Field(default=0.0, ge=0, description="Ramp-in/out in sim minutes")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in VALID_INJECTION_TYPES:
            raise ValueError(
                f"Invalid injection type {v!r}. Valid: {sorted(VALID_INJECTION_TYPES)}"
            )
        return v

    @field_validator("service")
    @classmethod
    def validate_service(cls, v: str) -> str:
        if v != "all" and v not in VALID_SERVICE_NAMES:
            raise ValueError(
                f"Invalid service {v!r}. "
                f"Valid: {sorted(VALID_SERVICE_NAMES)} or 'all'"
            )
        return v


class InjectResponse(BaseModel):
    injection_id: str
    start_sim_ts: Optional[str]  # will be set once injection activates
    wall_ts_ms: int


class SpeedRequest(BaseModel):
    sim_minutes_per_real_second: float = Field(..., gt=0)


class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=None, ge=0)
    sim_start_ts: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC start timestamp for the new run",
    )


class MitigateRequest(BaseModel):
    service: str = Field(..., description="Target service name or 'all'")
    type: str = Field(default="rate_limit", description="Mitigation type")
    factor: float = Field(..., gt=0.0, le=1.0, description="Rate limit multiplier (0 < factor <= 1.0)")
    duration_min: float = Field(..., gt=0.0, description="Duration in simulated minutes")
    mitigation_id: Optional[str] = Field(default=None, description="Optional custom mitigation ID")

    @field_validator("service")
    @classmethod
    def validate_service(cls, v: str) -> str:
        if v != "all" and v not in VALID_SERVICE_NAMES:
            raise ValueError(
                f"Invalid service {v!r}. Valid: {sorted(VALID_SERVICE_NAMES)} or 'all'"
            )
        return v


class MitigateResponse(BaseModel):
    mitigation_id: str
    service: str
    type: str
    factor: float
    duration_min: float
    status: str


# ---------------------------------------------------------------------------
# App factory  (called from main.py so clock/registry refs can be injected)
# ---------------------------------------------------------------------------


def create_app(clock, registry, publisher, mitigation_registry=None) -> FastAPI:
    """
    Build the FastAPI application with shared references to the live
    clock, injection registry, publisher, and mitigation registry.

    Parameters
    ----------
    clock:                generator.clock.SimClock
    registry:             generator.anomalies.InjectionRegistry
    publisher:            generator.publisher.Publisher
    mitigation_registry:  generator.mitigations.MitigationRegistry
    """
    if mitigation_registry is None:
        from generator.mitigations import MitigationRegistry  # noqa: PLC0415
        mitigation_registry = MitigationRegistry()

    app = FastAPI(
        title="CRIP Generator Control API",
        description="Pause/resume/speed/inject/mitigate controls for the CRIP simulation.",
        version="1.0.0",
    )

    # ----- Injection endpoints -----

    @app.post("/inject", response_model=InjectResponse, status_code=status.HTTP_201_CREATED)
    def inject(body: InjectRequest):
        """Schedule a new anomaly injection."""
        from generator.anomalies import create_injection  # noqa: PLC0415

        try:
            inj = create_injection(
                type_=body.type,
                service=body.service,
                magnitude=body.magnitude,
                duration_min=body.duration_min,
                ramp_min=body.ramp_min,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        injection_id = registry.add(inj)
        wall_ts_ms = _now_ms()

        log.info(
            "api_inject",
            injection_id=injection_id,
            type=body.type,
            service=body.service,
            magnitude=body.magnitude,
            duration_min=body.duration_min,
            run_id=clock.run_id,
            tick_index=clock.tick_index,
        )
        return InjectResponse(
            injection_id=injection_id,
            start_sim_ts=clock.sim_ts.isoformat(),
            wall_ts_ms=wall_ts_ms,
        )

    @app.delete("/inject/{injection_id}", status_code=status.HTTP_200_OK)
    def delete_injection(injection_id: str):
        """End an injection early."""
        sim_ts = clock.sim_ts
        ok = registry.end_early(injection_id, sim_ts)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"Injection {injection_id!r} not found or already ended",
            )
        log.info("api_inject_delete", injection_id=injection_id, run_id=clock.run_id)
        return {"injection_id": injection_id, "status": "ended"}

    @app.get("/injections", status_code=status.HTTP_200_OK)
    def list_injections():
        """List all injections (pending, active, ended)."""
        return registry.list_all()

    # ----- Mitigation endpoints -----

    @app.post("/mitigate", response_model=MitigateResponse, status_code=status.HTTP_201_CREATED)
    def mitigate(body: MitigateRequest):
        """Apply a rate-limit / throttle mitigation."""
        from generator.mitigations import Mitigation  # noqa: PLC0415

        mit_id = body.mitigation_id or str(uuid.uuid4())
        m = Mitigation(
            mitigation_id=mit_id,
            service=body.service,
            type=body.type,
            factor=body.factor,
            duration_min=body.duration_min,
        )
        mitigation_registry.add(m)

        log.info(
            "api_mitigate",
            mitigation_id=mit_id,
            service=body.service,
            factor=body.factor,
            duration_min=body.duration_min,
            run_id=clock.run_id,
            tick_index=clock.tick_index,
        )
        return MitigateResponse(
            mitigation_id=mit_id,
            service=body.service,
            type=body.type,
            factor=body.factor,
            duration_min=body.duration_min,
            status=m.status,
        )

    @app.delete("/mitigate/{mitigation_id}", status_code=status.HTTP_200_OK)
    def delete_mitigation(mitigation_id: str):
        """Remove / revert an active mitigation."""
        sim_ts = clock.sim_ts
        ok = mitigation_registry.end_early(mitigation_id, sim_ts)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"Mitigation {mitigation_id!r} not found or already ended",
            )
        log.info("api_mitigate_delete", mitigation_id=mitigation_id, run_id=clock.run_id)
        return {"mitigation_id": mitigation_id, "status": "ended"}

    @app.get("/mitigations", status_code=status.HTTP_200_OK)
    def list_mitigations():
        """List all mitigations."""
        return mitigation_registry.list_all()

    # ----- State endpoint -----

    @app.get("/state", status_code=status.HTTP_200_OK)
    def get_state():
        """Return current simulation state."""
        state = clock.state_dict()
        state["active_injections"] = registry.active_ids()
        state["active_mitigations"] = mitigation_registry.active_ids()
        return state

    # ----- Control endpoints -----

    @app.post("/control/pause", status_code=status.HTTP_200_OK)
    def pause():
        clock.pause()
        log.info("api_pause", run_id=clock.run_id, tick_index=clock.tick_index)
        return {"status": "paused"}

    @app.post("/control/resume", status_code=status.HTTP_200_OK)
    def resume():
        clock.resume()
        log.info("api_resume", run_id=clock.run_id, tick_index=clock.tick_index)
        return {"status": "running"}

    @app.post("/control/speed", status_code=status.HTTP_200_OK)
    def set_speed(body: SpeedRequest):
        try:
            clock.set_speed(body.sim_minutes_per_real_second)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        log.info(
            "api_speed",
            speed=body.sim_minutes_per_real_second,
            run_id=clock.run_id,
        )
        return {"sim_minutes_per_real_second": body.sim_minutes_per_real_second}

    @app.post("/control/reset", status_code=status.HTTP_200_OK)
    def reset(body: ResetRequest = ResetRequest()):
        from generator.anomalies import InjectionRegistry  # noqa: PLC0415

        sim_start: Optional[datetime] = None
        if body.sim_start_ts:
            try:
                sim_start = datetime.fromisoformat(body.sim_start_ts)
                if sim_start.tzinfo is None:
                    sim_start = sim_start.replace(tzinfo=timezone.utc)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Invalid sim_start_ts: {exc}"
                )

        new_run_id = clock.reset(seed=body.seed, sim_start_ts=sim_start)
        registry.clear_all()
        mitigation_registry.clear_all()
        publisher.update_run_id(new_run_id)

        log.info("api_reset", new_run_id=new_run_id, seed=body.seed)
        return {"run_id": new_run_id, "status": "reset"}

    return app


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


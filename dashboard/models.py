"""
dashboard/models.py
===================
Pydantic models for the dashboard backend: SSE messages, history responses,
health status, and proxy request/response shapes.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SSE envelope
# ---------------------------------------------------------------------------

class SSEMessage(BaseModel):
    """An outbound Server-Sent Event."""
    event: str   # tick | detection | injection | eval | state | heartbeat
    data: Dict[str, Any]

    def encode(self) -> str:
        """Render as the SSE wire format: 'event: X\\ndata: {...}\\n\\n'."""
        return f"event: {self.event}\ndata: {json.dumps(self.data)}\n\n"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    ok: bool
    detail: str = ""


class HealthStatus(BaseModel):
    redis: ComponentHealth
    generator: ComponentHealth
    detector: ComponentHealth  # based on last score timestamp age
    overall: bool


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class HistoryTick(BaseModel):
    run_id: str
    service: str
    sim_ts: str
    cost: float
    expected_cost: float
    score_zscore: Optional[float] = None
    score_ewma: Optional[float] = None
    score_cpr: Optional[float] = None


class HistoryDetection(BaseModel):
    detection_id: str
    run_id: str
    service: str
    sim_ts: str
    severity: float
    detector_votes: List[Dict[str, Any]] = Field(default_factory=list)
    rca: Dict[str, Any] = Field(default_factory=dict)
    status: str
    detect_wall_ts_ms: int


class HistoryInjection(BaseModel):
    injection_id: str
    run_id: str
    type: str
    service: str
    magnitude: float
    duration_min: float
    sim_start_ts: Optional[str] = None
    sim_end_ts: Optional[str] = None
    wall_start_ms: int = 0


class HistoryResponse(BaseModel):
    ticks: List[HistoryTick] = Field(default_factory=list)
    detections: List[HistoryDetection] = Field(default_factory=list)
    injections: List[HistoryInjection] = Field(default_factory=list)
    minutes: int


# ---------------------------------------------------------------------------
# Proxy request bodies (forwarded to generator)
# ---------------------------------------------------------------------------

class InjectProxyRequest(BaseModel):
    type: str
    service: str
    magnitude: float = Field(gt=0)
    duration_min: float = Field(gt=0)
    ramp_min: float = Field(default=0.0, ge=0)


class SpeedProxyRequest(BaseModel):
    sim_minutes_per_real_second: float = Field(gt=0)


class ResetProxyRequest(BaseModel):
    seed: Optional[int] = Field(default=None, ge=0)
    sim_start_ts: Optional[str] = None


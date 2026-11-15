"""
generator/mitigations.py
========================
Mitigation management for the CRIP event generator.

Provides:
- Mitigation dataclass representing active or scheduled rate-limits / throttles.
- MitigationRegistry managing active mitigations across ticks.

A mitigation multiplies the request volume (and therefore cost) of a service
by `factor` (e.g. 0.6 = 40% reduction).
Mitigations are logged to the Redis stream `crip:mitigations` upon start and end.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

from generator.clock import TickState, SIM_INTERVAL_MINUTES
from generator.config import VALID_SERVICE_NAMES


@dataclass
class Mitigation:
    """
    A live mitigation applied to one service or 'all'.

    Attributes
    ----------
    mitigation_id:
        Unique identifier (defaults to uuid4 string).
    service:
        Target service name or 'all'.
    type:
        Mitigation type (default 'rate_limit').
    factor:
        Traffic multiplier (> 0.0 and <= 1.0). e.g., 0.6 means 40% reduction.
    duration_min:
        Simulated minutes this mitigation should remain active.
    """
    service: str
    factor: float
    duration_min: float
    type: str = "rate_limit"
    mitigation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    sim_start_ts: Optional[datetime] = None
    sim_end_ts: Optional[datetime] = None
    start_tick: Optional[int] = None
    end_tick: Optional[int] = None
    wall_start_ms: int = 0
    wall_end_ms: int = 0
    _status: str = "pending"

    @property
    def status(self) -> str:
        return self._status

    def activate(self, tick_index: int, sim_ts: datetime) -> None:
        self._status = "active"
        self.start_tick = tick_index
        self.sim_start_ts = sim_ts
        self.wall_start_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        num_ticks = max(1, round(self.duration_min / SIM_INTERVAL_MINUTES))
        self.end_tick = tick_index + num_ticks
        self.sim_end_ts = sim_ts + timedelta(minutes=self.duration_min)

    def apply(self, tick_state: TickState) -> TickState:
        """Multiply volume_multiplier by factor for affected services."""
        if self._status != "active":
            return tick_state

        target_services = (
            list(tick_state.services.keys())
            if self.service == "all"
            else [self.service]
        )

        for svc in target_services:
            if svc in tick_state.services:
                sd = tick_state.services[svc]
                sd.volume_multiplier *= self.factor

        return tick_state

    def try_end(self, tick_index: int, sim_ts: datetime) -> bool:
        if self._status == "active" and self.end_tick is not None and tick_index >= self.end_tick:
            self._status = "ended"
            self.sim_end_ts = sim_ts
            self.wall_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            return True
        return False

    def end_early(self, sim_ts: datetime) -> bool:
        if self._status in ("pending", "active"):
            self._status = "ended"
            self.sim_end_ts = sim_ts
            self.wall_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "mitigation_id": self.mitigation_id,
            "service": self.service,
            "type": self.type,
            "factor": self.factor,
            "duration_min": self.duration_min,
            "status": self._status,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "sim_start_ts": self.sim_start_ts.isoformat() if self.sim_start_ts else None,
            "sim_end_ts": self.sim_end_ts.isoformat() if self.sim_end_ts else None,
            "wall_start_ms": self.wall_start_ms,
            "wall_end_ms": self.wall_end_ms,
        }


class MitigationRegistry:
    """Thread-safe registry managing active mitigations in the generator."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mitigations: Dict[str, Mitigation] = {}
        self._on_start_callbacks: List[Callable[[Mitigation], None]] = []
        self._on_end_callbacks: List[Callable[[Mitigation], None]] = []

    def add(self, mitigation: Mitigation) -> str:
        with self._lock:
            self._mitigations[mitigation.mitigation_id] = mitigation
            return mitigation.mitigation_id

    def get(self, mitigation_id: str) -> Optional[Mitigation]:
        with self._lock:
            return self._mitigations.get(mitigation_id)

    def end_early(self, mitigation_id: str, sim_ts: datetime) -> bool:
        with self._lock:
            m = self._mitigations.get(mitigation_id)
            if not m or m.status == "ended":
                return False
            ended = m.end_early(sim_ts)
        if ended:
            self._fire_end(m)
        return ended

    def apply_all(self, tick_state: TickState) -> TickState:
        with self._lock:
            mits = list(self._mitigations.values())

        newly_activated: List[Mitigation] = []
        newly_ended: List[Mitigation] = []

        for m in mits:
            if m.status == "pending":
                m.activate(tick_state.tick_index, tick_state.sim_ts)
                newly_activated.append(m)

            if m.status == "active":
                tick_state = m.apply(tick_state)
                if m.try_end(tick_state.tick_index + 1, tick_state.sim_ts):
                    newly_ended.append(m)

        for m in newly_activated:
            self._fire_start(m)
        for m in newly_ended:
            self._fire_end(m)

        return tick_state

    def list_all(self) -> List[dict]:
        with self._lock:
            return [m.to_dict() for m in self._mitigations.values()]

    def active_ids(self) -> List[str]:
        with self._lock:
            return [m.mitigation_id for m in self._mitigations.values() if m.status == "active"]

    def clear_all(self) -> None:
        with self._lock:
            self._mitigations.clear()

    def on_start(self, cb: Callable[[Mitigation], None]) -> None:
        self._on_start_callbacks.append(cb)

    def on_end(self, cb: Callable[[Mitigation], None]) -> None:
        self._on_end_callbacks.append(cb)

    def _fire_start(self, m: Mitigation) -> None:
        for cb in self._on_start_callbacks:
            try:
                cb(m)
            except Exception:
                pass

    def _fire_end(self, m: Mitigation) -> None:
        for cb in self._on_end_callbacks:
            try:
                cb(m)
            except Exception:
                pass


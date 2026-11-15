
"""
generator/mitigations.py
========================
Mitigation management for the CRIP event generator.
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
    """A live mitigation applied to one service or 'all'."""

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
    end_reason: Optional[str] = None
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
        if self._status != "active":
            return tick_state

        target_services = (
            list(tick_state.services.keys())
            if self.service == "all"
            else [self.service]
        )

        for svc in target_services:
            if svc in tick_state.services:
                tick_state.services[svc].volume_multiplier *= self.factor

        return tick_state

    def try_end(self, tick_index: int, sim_ts: datetime) -> bool:
        if (
            self._status == "active"
            and self.end_tick is not None
            and tick_index >= self.end_tick
        ):
            self._status = "ended"
            self.sim_end_ts = sim_ts
            self.wall_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            self.end_reason = "expired"
            return True
        return False

    def end_early(self, sim_ts: datetime) -> bool:
        if self._status in ("pending", "active"):
            self._status = "ended"
            self.sim_end_ts = sim_ts
            self.wall_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            self.end_reason = "ended_early"
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
            "end_reason": self.end_reason,
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
            mitigation = self._mitigations.get(mitigation_id)
            if not mitigation or mitigation.status == "ended":
                return False
            ended = mitigation.end_early(sim_ts)

        if ended:
            self._fire_end(mitigation)

        return ended

    def apply_all(self, tick_state: TickState) -> TickState:
        with self._lock:
            mitigations = list(self._mitigations.values())

        newly_activated: List[Mitigation] = []
        newly_ended: List[Mitigation] = []

        for mitigation in mitigations:
            if mitigation.status == "pending":
                mitigation.activate(tick_state.tick_index, tick_state.sim_ts)
                newly_activated.append(mitigation)

            if mitigation.status == "active":
                tick_state = mitigation.apply(tick_state)

                if mitigation.try_end(
                    tick_state.tick_index + 1,
                    tick_state.sim_ts,
                ):
                    newly_ended.append(mitigation)

        for mitigation in newly_activated:
            self._fire_start(mitigation)

        for mitigation in newly_ended:
            self._fire_end(mitigation)

        return tick_state

    def list_all(self) -> List[dict]:
        with self._lock:
            return [m.to_dict() for m in self._mitigations.values()]

    def active_ids(self) -> List[str]:
        with self._lock:
            return [
                m.mitigation_id
                for m in self._mitigations.values()
                if m.status == "active"
            ]

    def clear_all(self) -> None:
        with self._lock:
            self._mitigations.clear()

    def on_start(self, cb: Callable[[Mitigation], None]) -> None:
        self._on_start_callbacks.append(cb)

    def on_end(self, cb: Callable[[Mitigation], None]) -> None:
        self._on_end_callbacks.append(cb)

    def _fire_start(self, mitigation: Mitigation) -> None:
        for callback in self._on_start_callbacks:
            try:
                callback(mitigation)
            except Exception:
                pass

    def _fire_end(self, mitigation: Mitigation) -> None:
        for callback in self._on_end_callbacks:
            try:
                callback(mitigation)
            except Exception:
                pass

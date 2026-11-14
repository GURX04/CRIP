"""
generator/anomalies.py
======================
Composable anomaly injection system for the CRIP event generator.

Each injection type is a dataclass that implements the ``Injection`` protocol:
    apply(tick_state: TickState) -> TickState

The ``InjectionRegistry`` manages all active injections, composing them in
insertion order on every tick, and handles ground-truth persistence.

Injection types
---------------
spike           Multiply request volume by ``magnitude``, trapezoidal shape.
step_change     Persistent multiplicative change in cost from start onward.
slow_drift      Cost increases linearly by ``magnitude`` % over ``duration_min``.
cost_only       Cost-per-request rises by ``magnitude``x; volume stays flat.
seasonality_break  Suppress diurnal factor to 0.5 for the duration.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

from generator.clock import TickState, SIM_INTERVAL_MINUTES
from generator.config import VALID_SERVICE_NAMES


# ---------------------------------------------------------------------------
# Base injection protocol
# ---------------------------------------------------------------------------

class Injection(ABC):
    """Abstract base for all injection types."""

    @abstractmethod
    def apply(self, tick_state: TickState) -> TickState:
        """
        Modify ``tick_state`` in-place for the services this injection targets.

        Must mark each affected ``ServiceTickData`` as anomalous and append
        this injection's ID to ``injection_ids``.
        """

    @property
    @abstractmethod
    def status(self) -> str:
        """'pending' | 'active' | 'ended'"""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialisable representation for API responses and ground truth."""

    def _affected_services(self, tick_state: TickState) -> list[str]:
        """Return the list of service names this injection covers."""
        tgt = getattr(self, "service", "all")
        if tgt == "all":
            return list(tick_state.services.keys())
        if tgt in tick_state.services:
            return [tgt]
        return []


# ---------------------------------------------------------------------------
# Shared mixin for common fields
# ---------------------------------------------------------------------------

@dataclass
class _BaseInjection(Injection, ABC):
    injection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    service: str = "all"       # target service name or "all"
    magnitude: float = 1.0
    duration_min: float = 30.0
    ramp_min: float = 0.0      # ramp-in / ramp-out duration

    # Set at activation time
    start_tick: int = -1
    end_tick: int = -1         # exclusive; -1 = not set yet
    sim_start_ts: Optional[datetime] = None
    sim_end_ts: Optional[datetime] = None
    wall_start_ms: int = 0
    wall_end_ms: int = 0

    _status: str = field(default="pending", init=False, repr=False)

    # ----- lifecycle -----

    def activate(self, tick_index: int, sim_ts: datetime) -> None:
        """Mark injection as active starting at ``tick_index``."""
        self._status = "active"
        self.start_tick = tick_index
        total_ticks = int(self.duration_min / SIM_INTERVAL_MINUTES)
        self.end_tick = tick_index + total_ticks
        self.sim_start_ts = sim_ts
        self.wall_start_ms = _now_ms()

    def try_end(self, tick_index: int, sim_ts: datetime) -> bool:
        """
        Check if this injection should end at ``tick_index``.
        Returns True (and marks ended) if it has expired.
        """
        if self._status != "active":
            return False
        if self.end_tick >= 0 and tick_index >= self.end_tick:
            self._end(sim_ts)
            return True
        return False

    def end_early(self, sim_ts: datetime) -> None:
        """Force-terminate the injection."""
        if self._status == "active":
            self._end(sim_ts)

    def _end(self, sim_ts: datetime) -> None:
        self._status = "ended"
        self.sim_end_ts = sim_ts
        self.wall_end_ms = _now_ms()

    @property
    def status(self) -> str:
        return self._status

    # ----- shared helper: trapezoidal ramp -----

    def _trapezoid_factor(self, tick_index: int) -> float:
        """
        Return a [0..1] scalar for the trapezoidal shape:
        ramps from 0→1 over ramp_min, stays at 1, ramps 1→0 over ramp_min.
        """
        if self.start_tick < 0:
            return 1.0
        ramp_ticks = max(1, int(self.ramp_min / SIM_INTERVAL_MINUTES))
        pos = tick_index - self.start_tick
        total = max(1, self.end_tick - self.start_tick)
        ramp_up = ramp_ticks if self.ramp_min > 0 else 0
        ramp_down = ramp_ticks if self.ramp_min > 0 else 0

        if pos < ramp_up:
            return pos / ramp_up
        elif pos >= total - ramp_down:
            remaining = total - pos
            return max(0.0, remaining / ramp_down) if ramp_down > 0 else 1.0
        return 1.0

    def to_dict(self) -> dict:
        return {
            "injection_id": self.injection_id,
            "type": self.__class__.__name__.lower().replace("injection", ""),
            "service": self.service,
            "magnitude": self.magnitude,
            "duration_min": self.duration_min,
            "ramp_min": self.ramp_min,
            "status": self._status,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "sim_start_ts": self.sim_start_ts.isoformat() if self.sim_start_ts else None,
            "sim_end_ts": self.sim_end_ts.isoformat() if self.sim_end_ts else None,
            "wall_start_ms": self.wall_start_ms,
            "wall_end_ms": self.wall_end_ms,
        }


# ---------------------------------------------------------------------------
# Concrete injection types
# ---------------------------------------------------------------------------

@dataclass
class SpikeInjection(_BaseInjection):
    """
    Multiply request volume (and therefore cost) by ``magnitude``.
    Shape: trapezoidal with optional ``ramp_min`` ramp-in and ramp-out.
    """

    def apply(self, tick_state: TickState) -> TickState:
        factor = self._trapezoid_factor(tick_state.tick_index)
        effective = 1.0 + (self.magnitude - 1.0) * factor
        for svc in self._affected_services(tick_state):
            sd = tick_state.services[svc]
            sd.volume_multiplier *= effective
            sd.is_anomalous = True
            if self.injection_id not in sd.injection_ids:
                sd.injection_ids.append(self.injection_id)
        return tick_state


@dataclass
class StepChangeInjection(_BaseInjection):
    """
    Persistent multiplicative change in cost from activation onward.
    Unlike spike, there is no ramp; it stays at ``magnitude`` forever
    until explicitly ended.
    """

    def apply(self, tick_state: TickState) -> TickState:
        for svc in self._affected_services(tick_state):
            sd = tick_state.services[svc]
            sd.volume_multiplier *= self.magnitude
            sd.is_anomalous = True
            if self.injection_id not in sd.injection_ids:
                sd.injection_ids.append(self.injection_id)
        return tick_state


@dataclass
class SlowDriftInjection(_BaseInjection):
    """
    Cost increases linearly by ``magnitude`` percent over ``duration_min``.
    At tick 0 of injection: multiplier = 1.0.
    At final tick: multiplier ≤ (1 + magnitude/100).
    """

    def apply(self, tick_state: TickState) -> TickState:
        if self.start_tick < 0:
            return tick_state
        total_ticks = max(1, self.end_tick - self.start_tick)
        elapsed = tick_state.tick_index - self.start_tick
        progress = min(1.0, elapsed / total_ticks)
        # magnitude is treated as a percentage: 10 → +10%
        multiplier = 1.0 + (self.magnitude / 100.0) * progress
        for svc in self._affected_services(tick_state):
            sd = tick_state.services[svc]
            sd.volume_multiplier *= multiplier
            sd.is_anomalous = True
            if self.injection_id not in sd.injection_ids:
                sd.injection_ids.append(self.injection_id)
        return tick_state


@dataclass
class CostOnlyInjection(_BaseInjection):
    """
    Cost-per-request rises by ``magnitude``x while request volume stays flat.
    Models a misconfiguration (e.g. wrong tier, expensive scan).
    This is the primary RCA test case.
    """

    def apply(self, tick_state: TickState) -> TickState:
        for svc in self._affected_services(tick_state):
            sd = tick_state.services[svc]
            # Only the price multiplier changes — volume_multiplier stays at 1.0
            sd.price_multiplier *= self.magnitude
            sd.is_anomalous = True
            if self.injection_id not in sd.injection_ids:
                sd.injection_ids.append(self.injection_id)
        return tick_state


@dataclass
class SeasonalityBreakInjection(_BaseInjection):
    """
    Suppress the diurnal factor to a flat 0.5 for the duration.
    Traffic appears constant (no peak/trough pattern).
    """

    def apply(self, tick_state: TickState) -> TickState:
        for svc in self._affected_services(tick_state):
            sd = tick_state.services[svc]
            sd.diurnal_suppressed = True
            sd.is_anomalous = True
            if self.injection_id not in sd.injection_ids:
                sd.injection_ids.append(self.injection_id)
        return tick_state


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_INJECTION_TYPES: dict[str, type[_BaseInjection]] = {
    "spike": SpikeInjection,
    "step_change": StepChangeInjection,
    "slow_drift": SlowDriftInjection,
    "cost_only": CostOnlyInjection,
    "seasonality_break": SeasonalityBreakInjection,
}

VALID_INJECTION_TYPES = set(_INJECTION_TYPES.keys())


def create_injection(
    type_: str,
    service: str,
    magnitude: float,
    duration_min: float,
    ramp_min: float = 0.0,
    injection_id: Optional[str] = None,
) -> _BaseInjection:
    """
    Factory function that validates inputs and returns the correct subclass.

    Raises
    ------
    ValueError
        If ``type_`` or ``service`` is invalid, or ``magnitude`` ≤ 0.
    """
    if type_ not in _INJECTION_TYPES:
        raise ValueError(
            f"Unknown injection type {type_!r}. "
            f"Valid types: {sorted(VALID_INJECTION_TYPES)}"
        )
    if service != "all" and service not in VALID_SERVICE_NAMES:
        raise ValueError(
            f"Unknown service {service!r}. "
            f"Valid services: {sorted(VALID_SERVICE_NAMES)} or 'all'"
        )
    if magnitude <= 0:
        raise ValueError(f"magnitude must be > 0, got {magnitude}")
    if duration_min <= 0:
        raise ValueError(f"duration_min must be > 0, got {duration_min}")

    cls = _INJECTION_TYPES[type_]
    inj = cls(
        service=service,
        magnitude=magnitude,
        duration_min=duration_min,
        ramp_min=max(0.0, ramp_min),
    )
    if injection_id is not None:
        inj.injection_id = injection_id
    return inj


# ---------------------------------------------------------------------------
# Injection registry
# ---------------------------------------------------------------------------

class InjectionRegistry:
    """
    Thread-safe registry of all injections (pending, active, ended).

    The registry is the single source of truth for injection state.
    Publishers and the API both hold a reference to this object.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._injections: Dict[str, _BaseInjection] = {}
        # Callbacks fired when an injection starts or ends (for ground truth)
        self._on_start_callbacks: List[Callable[[_BaseInjection], None]] = []
        self._on_end_callbacks: List[Callable[[_BaseInjection], None]] = []

    # ----- registration / removal -----

    def add(self, injection: _BaseInjection) -> str:
        """Register a new injection.  Returns injection_id."""
        with self._lock:
            self._injections[injection.injection_id] = injection
        return injection.injection_id

    def end_early(self, injection_id: str, sim_ts: datetime) -> bool:
        """
        Force-end an injection.

        Returns True if found and ended, False if not found or already ended.
        """
        with self._lock:
            inj = self._injections.get(injection_id)
            if inj is None or inj.status == "ended":
                return False
            inj.end_early(sim_ts)
        self._fire_end(inj)
        return True

    def clear_all(self) -> None:
        """Remove all injections (called on reset)."""
        with self._lock:
            self._injections.clear()

    # ----- tick processing -----

    def apply_all(self, tick_state: TickState) -> TickState:
        """
        Apply every active injection to ``tick_state`` in insertion order,
        then check for newly activated / expired injections.
        """
        with self._lock:
            injections = list(self._injections.values())

        newly_activated: list[_BaseInjection] = []
        newly_ended: list[_BaseInjection] = []

        for inj in injections:
            if inj.status == "pending":
                inj.activate(tick_state.tick_index, tick_state.sim_ts)
                newly_activated.append(inj)

            if inj.status == "active":
                tick_state = inj.apply(tick_state)
                # Check expiry *after* applying so the final tick is included
                if inj.try_end(tick_state.tick_index + 1, tick_state.sim_ts):
                    newly_ended.append(inj)

        # Fire callbacks outside the lock
        for inj in newly_activated:
            self._fire_start(inj)
        for inj in newly_ended:
            self._fire_end(inj)

        return tick_state

    # ----- queries -----

    def list_all(self) -> List[dict]:
        with self._lock:
            return [inj.to_dict() for inj in self._injections.values()]

    def get(self, injection_id: str) -> Optional[_BaseInjection]:
        with self._lock:
            return self._injections.get(injection_id)

    def active_ids(self) -> List[str]:
        with self._lock:
            return [
                iid for iid, inj in self._injections.items()
                if inj.status == "active"
            ]

    # ----- callback registration -----

    def on_start(self, cb: Callable[[_BaseInjection], None]) -> None:
        self._on_start_callbacks.append(cb)

    def on_end(self, cb: Callable[[_BaseInjection], None]) -> None:
        self._on_end_callbacks.append(cb)

    # ----- private -----

    def _fire_start(self, inj: _BaseInjection) -> None:
        for cb in self._on_start_callbacks:
            try:
                cb(inj)
            except Exception:
                pass  # ground-truth writers must not crash the tick loop

    def _fire_end(self, inj: _BaseInjection) -> None:
        for cb in self._on_end_callbacks:
            try:
                cb(inj)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    """Current epoch time in milliseconds."""
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

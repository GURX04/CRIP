"""
generator/clock.py
==================
Accelerated simulation clock for the CRIP event generator.

The clock advances simulated time in fixed 5-minute increments (one "tick")
while real time passes at a configurable ratio
(SIM_MINUTES_PER_REAL_SECOND → real seconds per tick).

Thread-safety: all mutable state is protected by a reentrant lock so the
FastAPI control plane can read/modify the clock from a different thread.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional


SIM_INTERVAL_MINUTES: int = 5  # fixed simulated interval per tick


class SimClock:
    """
    Maintains the current simulated timestamp and controls playback.

    Parameters
    ----------
    sim_start_ts:
        UTC datetime to start the simulation from.
        Defaults to today 00:00 UTC.
    sim_minutes_per_real_second:
        How many simulated minutes pass per real second.
        Default 5 → one 5-min tick per real second.
    seed:
        Initial seed (stored for introspection / reset).
    """

    def __init__(
        self,
        sim_start_ts: Optional[datetime] = None,
        sim_minutes_per_real_second: float = 5.0,
        seed: int = 42,
    ) -> None:
        self._lock = threading.RLock()

        # Canonical start
        self._origin: datetime = sim_start_ts or self._today_midnight_utc()
        self._seed: int = seed

        # Mutable playback state
        self._speed: float = sim_minutes_per_real_second
        self._tick_index: int = 0
        self._paused: bool = False
        self._run_id: str = str(uuid.uuid4())

        # Monotonic reference: wall-clock time (monotonic) at which the
        # generator loop should fire the *next* tick.
        self._next_tick_mono: float = time.monotonic()

    # ------------------------------------------------------------------
    # Properties (all thread-safe)
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        with self._lock:
            return self._run_id

    @property
    def seed(self) -> int:
        with self._lock:
            return self._seed

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    @property
    def tick_index(self) -> int:
        with self._lock:
            return self._tick_index

    @property
    def sim_ts(self) -> datetime:
        """Current simulated timestamp (UTC, offset-aware)."""
        with self._lock:
            return self._current_sim_ts()

    @property
    def real_seconds_per_tick(self) -> float:
        """Real seconds between ticks at current speed."""
        with self._lock:
            return SIM_INTERVAL_MINUTES / self._speed

    # ------------------------------------------------------------------
    # Control plane
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the simulation clock.  Sim time stops advancing."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """
        Resume from pause.  Re-anchors the monotonic next-tick target to
        *now* so accumulated pause time is not counted as drift.
        """
        with self._lock:
            if self._paused:
                self._paused = False
                # Reset next_tick_mono to now so we don't immediately fire
                # a burst of skipped ticks.
                self._next_tick_mono = time.monotonic()

    def set_speed(self, sim_minutes_per_real_second: float) -> None:
        """
        Change playback speed at runtime.

        The next-tick target is recalculated from *now* using the new speed
        to avoid a sudden burst (if speeding up) or long wait (if slowing).
        """
        if sim_minutes_per_real_second <= 0:
            raise ValueError("Speed must be > 0")
        with self._lock:
            self._speed = sim_minutes_per_real_second
            self._next_tick_mono = time.monotonic()

    def reset(self, seed: Optional[int] = None, sim_start_ts: Optional[datetime] = None) -> str:
        """
        Reset to a fresh run: new run_id, tick_index = 0, optionally new seed/start.

        Returns the new run_id.
        """
        with self._lock:
            if seed is not None:
                self._seed = seed
            self._origin = sim_start_ts or self._today_midnight_utc()
            self._tick_index = 0
            self._paused = False
            self._run_id = str(uuid.uuid4())
            self._next_tick_mono = time.monotonic()
            return self._run_id

    # ------------------------------------------------------------------
    # Tick protocol (called by the main loop)
    # ------------------------------------------------------------------

    def wait_for_next_tick(self) -> bool:
        """
        Block until it is time to fire the next tick.

        Returns True if a tick should be emitted, False if paused (caller
        should call again after a short sleep without advancing state).

        Uses monotonic time to compensate for processing drift: if one tick
        took longer than the interval, the next tick fires immediately (but
        never more than one tick per call — no burst catch-up).
        """
        with self._lock:
            if self._paused:
                return False
            target = self._next_tick_mono
            speed = self._speed

        # Sleep outside the lock so control-plane calls are not blocked.
        now = time.monotonic()
        sleep_s = target - now
        if sleep_s > 0:
            time.sleep(sleep_s)

        with self._lock:
            if self._paused:
                return False
            # Advance next-tick target by exactly one tick interval.
            # If we're late, the next tick will fire immediately (capped at 0).
            interval = SIM_INTERVAL_MINUTES / self._speed
            self._next_tick_mono = target + interval
            self._tick_index += 1
            return True

    def advance_tick(self) -> "TickState":
        """
        Return a snapshot of the current tick *before* advancing.

        Called by the main loop after ``wait_for_next_tick`` returns True.
        """
        with self._lock:
            idx = self._tick_index
            ts = self._current_sim_ts()
            run = self._run_id
            speed = self._speed
        return TickState(
            tick_index=idx,
            sim_ts=ts,
            run_id=run,
            speed=speed,
        )

    def state_dict(self) -> dict:
        """Serialisable snapshot of clock state for GET /state."""
        with self._lock:
            return {
                "run_id": self._run_id,
                "sim_ts": self._current_sim_ts().isoformat(),
                "tick_index": self._tick_index,
                "speed": self._speed,
                "paused": self._paused,
                "seed": self._seed,
                "real_seconds_per_tick": SIM_INTERVAL_MINUTES / self._speed,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_sim_ts(self) -> datetime:
        """Compute current sim timestamp from origin + tick_index."""
        return self._origin + timedelta(
            minutes=SIM_INTERVAL_MINUTES * self._tick_index
        )

    @staticmethod
    def _today_midnight_utc() -> datetime:
        today = datetime.now(tz=timezone.utc).date()
        return datetime(today.year, today.month, today.day, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tick state — immutable value object passed through the pipeline
# ---------------------------------------------------------------------------

class TickState:
    """
    Immutable snapshot passed from the clock through traffic model and
    anomaly injections to the publisher.

    The ``services`` dict is built by traffic_model.py and carried forward.
    Injections operate on copies of the per-service data.
    """

    __slots__ = (
        "tick_index",
        "sim_ts",
        "run_id",
        "speed",
        "services",          # dict[str, ServiceTickData] — filled by traffic_model
        "active_injections", # list[str] injection_ids active this tick
    )

    def __init__(
        self,
        tick_index: int,
        sim_ts: datetime,
        run_id: str,
        speed: float,
    ) -> None:
        self.tick_index: int = tick_index
        self.sim_ts: datetime = sim_ts
        self.run_id: str = run_id
        self.speed: float = speed
        self.services: dict[str, "ServiceTickData"] = {}
        self.active_injections: list[str] = []


class ServiceTickData:
    """
    Mutable per-service data for one tick.

    Fields
    ------
    service:          service name
    request_count:    computed request volume (after noise, before anomaly)
    usage_amount:     same as request_count (aliased; some services use GB etc.)
    unit_price:       cost per request unit (may be mutated by cost_only)
    unblended_cost:   derived as usage_amount * unit_price
    volume_multiplier: anomaly multiplier on volume (spike, step_change, slow_drift)
    price_multiplier:  anomaly multiplier on unit price only (cost_only)
    diurnal_suppressed: flag set by seasonality_break
    is_anomalous:     True if any injection is active for this service
    injection_ids:    list of active injection IDs
    """

    __slots__ = (
        "service",
        "request_count",
        "usage_amount",
        "unit_price",
        "unblended_cost",
        "volume_multiplier",
        "price_multiplier",
        "diurnal_suppressed",
        "is_anomalous",
        "injection_ids",
    )

    def __init__(self, service: str, unit_price: float) -> None:
        self.service = service
        self.request_count: float = 0.0
        self.usage_amount: float = 0.0
        self.unit_price: float = unit_price
        self.unblended_cost: float = 0.0
        self.volume_multiplier: float = 1.0
        self.price_multiplier: float = 1.0
        self.diurnal_suppressed: bool = False
        self.is_anomalous: bool = False
        self.injection_ids: list[str] = []

    def finalise(self) -> None:
        """Apply multipliers and derive cost.  Called after all injections."""
        self.usage_amount = self.request_count * self.volume_multiplier
        self.unblended_cost = self.usage_amount * self.unit_price * self.price_multiplier

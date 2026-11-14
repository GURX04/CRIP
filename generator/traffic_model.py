"""
generator/traffic_model.py
==========================
Baseline traffic model for the CRIP event generator.

Formula per service per tick:
    request_volume(t) = base_volume * diurnal_factor(t) * (1 + noise)

where:
    diurnal_factor(t) = 0.5 * (1 + sin(2π * hour_of_day / 24 - π/2))
                        (peaks at noon UTC, troughs at midnight UTC)
    noise             ~ Normal(0, noise_sigma)

All randomness is drawn from the single seeded numpy Generator that is
passed in, ensuring full reproducibility.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from generator.clock import TickState, ServiceTickData
from generator.config import SERVICES, GeneratorSettings

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Diurnal helper
# ---------------------------------------------------------------------------

def diurnal_factor(sim_ts_utc) -> float:
    """
    Return a value in [0, 1] representing time-of-day traffic intensity.

    Uses a sine wave that:
    - Peaks at 12:00 UTC  → factor ≈ 1.0
    - Troughs at 00:00 UTC → factor ≈ 0.0
    """
    hour_frac = sim_ts_utc.hour + sim_ts_utc.minute / 60.0
    # sin(-π/2) = -1 at midnight, sin(π/2) = 1 at noon
    raw = math.sin(2 * math.pi * hour_frac / 24.0 - math.pi / 2)
    return 0.5 * (1.0 + raw)   # normalise to [0, 1]


# ---------------------------------------------------------------------------
# Traffic model
# ---------------------------------------------------------------------------

class TrafficModel:
    """
    Computes per-service baseline metrics for one tick and populates
    ``tick_state.services``.

    Parameters
    ----------
    rng:
        The seeded numpy random Generator.  Must be the *same* object shared
        across the entire run so the sequence is deterministic.
    settings:
        Generator settings (noise_sigma, service catalogue implicitly via SERVICES).
    """

    def __init__(self, rng: np.random.Generator, settings: GeneratorSettings) -> None:
        self._rng = rng
        self._sigma = settings.noise_sigma

    def populate(self, tick_state: TickState) -> TickState:
        """
        Fill ``tick_state.services`` with one ``ServiceTickData`` per service.

        Noise is drawn in a single batch (one call per tick) for efficiency
        and to maintain a deterministic per-tick RNG advancement even when
        new services are added.
        """
        services = list(SERVICES.values())
        n = len(services)

        # Draw all noise samples in one call → deterministic regardless of
        # how the caller iterates the dict.
        noise_samples: np.ndarray = self._rng.normal(loc=0.0, scale=self._sigma, size=n)

        d_factor = diurnal_factor(tick_state.sim_ts)

        for i, spec in enumerate(services):
            sd = ServiceTickData(service=spec.name, unit_price=spec.unit_price)

            noise = float(noise_samples[i])
            # Clamp so volume is never negative
            noisy_factor = max(0.0, 1.0 + noise)

            if sd.diurnal_suppressed:
                # Anomaly will set this; here it's always False at baseline.
                effective_diurnal = 0.5  # flat mid-point
            else:
                effective_diurnal = d_factor

            sd.request_count = spec.base_volume * effective_diurnal * noisy_factor
            sd.usage_amount = sd.request_count  # injections may change this
            sd.unblended_cost = sd.usage_amount * spec.unit_price

            tick_state.services[spec.name] = sd

        return tick_state

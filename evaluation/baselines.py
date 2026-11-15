"""
evaluation/baselines.py
=======================
Comparison baselines for the CRIP evaluation harness:
1. Rolling refit Facebook Prophet (refit every 6 sim-hours using prior 3 days).
2. Naive static threshold (trailing rolling mean + 3 std).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Naive Static Threshold Baseline
# ---------------------------------------------------------------------------

class NaiveStaticThresholdBaseline:
    """
    Moving window mean + k * std threshold baseline.
    Computes rolling mean and std over trailing `window_ticks` events for each service.
    """

    name: str = "naive_static_threshold"

    def __init__(self, window_ticks: int = 288, k: float = 3.0) -> None:
        self.window_ticks = window_ticks
        self.k = k
        self._buffers: Dict[str, deque] = {}

    def update(self, service: str, sim_ts: pd.Timestamp, cost: float) -> Tuple[bool, float]:
        buf = self._buffers.setdefault(service, deque(maxlen=self.window_ticks))
        if len(buf) < 20:
            buf.append(cost)
            return False, 0.0

        arr = np.array(buf)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        threshold = mu + self.k * max(1e-4, sigma)

        is_anomaly = cost > threshold
        score = max(0.0, (cost - mu) / max(1e-4, sigma))
        buf.append(cost)
        return is_anomaly, score

    def reset(self) -> None:
        self._buffers.clear()


# ---------------------------------------------------------------------------
# Rolling Refit Prophet Baseline
# ---------------------------------------------------------------------------

class RollingProphetBaseline:
    """
    Rolling window Facebook Prophet baseline.
    Refits every `step_ticks` (default 72 ticks = 6h) using trailing `history_ticks`
    (default 864 ticks = 3 days) of data per service.

    Flags when actual cost > yhat_upper.
    Uses only standard daily and weekly seasonality (NO tuned/custom generator seasonality).
    Measures and logs model fit times.
    """

    name: str = "prophet_rolling"

    def __init__(
        self,
        history_ticks: int = 864,    # 3 days at 5-min ticks
        step_ticks: int = 72,        # 6 hours refit cadence
        interval_width: float = 0.95,
    ) -> None:
        self.history_ticks = history_ticks
        self.step_ticks = step_ticks
        self.interval_width = interval_width

        # Per-service tracking
        self._history: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
        self._current_forecasts: Dict[str, pd.DataFrame] = {}
        self._last_fit_tick: Dict[str, int] = {}

        self.fit_times: List[float] = []

    def run_series(
        self,
        events_df: pd.DataFrame,
        service: str,
    ) -> pd.DataFrame:
        """
        Run the rolling Prophet baseline on a full single-service timeseries.
        Returns a DataFrame with columns: ['tick_index', 'service', 'is_anomaly', 'score', 'yhat_upper'].
        """
        svc_df = events_df[events_df["service"] == service].sort_values("tick_index").copy()
        n = len(svc_df)
        flags = np.zeros(n, dtype=bool)
        scores = np.zeros(n, dtype=np.float64)
        uppers = np.zeros(n, dtype=np.float64)

        ticks = svc_df["tick_index"].values
        ts_values = pd.to_datetime(svc_df["sim_ts"]).values
        costs = svc_df["unblended_cost"].values

        # Need at least 24 hours of history to fit first model
        min_fit_ticks = min(288, self.history_ticks // 2)

        next_fit_tick = min_fit_ticks
        forecast_map: Dict[int, float] = {}  # tick -> yhat_upper

        for i in range(n):
            curr_tick = ticks[i]
            cost = costs[i]

            # Trigger rolling refit
            if curr_tick >= next_fit_tick and i >= min_fit_ticks:
                start_idx = max(0, i - self.history_ticks)
                train_df = pd.DataFrame({
                    "ds": pd.to_datetime(ts_values[start_idx:i]).tz_localize(None),
                    "y": costs[start_idx:i],
                })

                t0 = time.perf_counter()
                try:
                    m = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=False,
                        interval_width=self.interval_width,
                    )
                    m.fit(train_df)

                    future_idx = min(n, i + self.step_ticks)
                    future_df = pd.DataFrame({
                        "ds": pd.to_datetime(ts_values[i:future_idx]).tz_localize(None),
                    })
                    fc = m.predict(future_df)
                    for j, r in enumerate(fc.itertuples()):
                        forecast_map[ticks[i + j]] = getattr(r, "yhat_upper")

                except Exception as exc:
                    log.warning("Prophet fit failed at tick %d for %s: %s", curr_tick, service, exc)

                dt = time.perf_counter() - t0
                self.fit_times.append(dt)
                next_fit_tick = curr_tick + self.step_ticks

            # Evaluate against forecast
            if curr_tick in forecast_map:
                yhat_upper = forecast_map[curr_tick]
                uppers[i] = yhat_upper
                if cost > yhat_upper:
                    flags[i] = True
                    scores[i] = max(0.0, (cost - yhat_upper) / max(1e-4, yhat_upper))

        res = svc_df[["sim_ts", "tick_index", "service", "unblended_cost"]].copy()
        res["is_anomaly"] = flags
        res["score"] = scores
        res["yhat_upper"] = uppers
        return res

    @property
    def total_fit_time_s(self) -> float:
        return sum(self.fit_times)

    @property
    def mean_fit_time_s(self) -> float:
        return float(np.mean(self.fit_times)) if self.fit_times else 0.0

    def reset(self) -> None:
        self._history.clear()
        self._current_forecasts.clear()
        self._last_fit_tick.clear()
        self.fit_times.clear()

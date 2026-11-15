"""
evaluation/runner.py
====================
Deterministic offline simulation runner for CRIP anomaly detectors.
Replays generated scenarios through production detectors and baselines using
simulated time only (no Redis, no wall-clock timers, zero label leakage).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation.baselines import NaiveStaticThresholdBaseline, RollingProphetBaseline
from evaluation.detectors_streaming import (
    RiverHSTDetector,
    ServiceDetectorSuite,
    StreamingEnsembleRunner,
    StreamingIsolationForestDetector,
)
from pipeline.config import PipelineSettings
from pipeline.detectors import (
    CostPerRequestResidualDetector,
    EWMADetector,
    RobustZScoreDetector,
    StreamDetector,
)
from pipeline.models import CostEvent

log = logging.getLogger(__name__)


def row_to_cost_event(row: Any) -> CostEvent:
    """Convert a row tuple/dict to a production CostEvent."""
    return CostEvent(
        run_id=getattr(row, "run_id", "offline"),
        event_id=f"off-{getattr(row, 'tick_index', 0)}-{getattr(row, 'service', 'svc')}",
        tick_index=int(getattr(row, "tick_index", 0)),
        sim_ts=pd.to_datetime(getattr(row, "sim_ts")),
        wall_ts_ms=0,
        service=str(getattr(row, "service", "unknown")),
        request_count=float(getattr(row, "request_count", 0.0)),
        usage_amount=float(getattr(row, "usage_amount", 0.0)),
        unblended_cost=float(getattr(row, "unblended_cost", 0.0)),
        unit_price=float(getattr(row, "unit_price", 0.0)),
    )


class OfflineReplayEngine:
    """
    Manages detector execution over an offline scenario DataFrame.
    """

    def __init__(
        self,
        settings: Optional[PipelineSettings] = None,
        warmup_ticks: int = 288,
        cooldown_ticks: int = 6,
    ) -> None:
        self.settings = settings or PipelineSettings(
            warmup_ticks=warmup_ticks,
            cooldown_ticks=cooldown_ticks,
        )
        self.warmup_ticks = warmup_ticks
        self.cooldown_ticks = cooldown_ticks

    def create_single_detector_suite(self, detector_name: str) -> ServiceDetectorSuite:
        """Create a service-partitioned suite for an individual streaming detector."""
        s = self.settings

        if detector_name == "robust_zscore":
            return ServiceDetectorSuite(
                detector_factory=lambda: RobustZScoreDetector(s),
                warmup_ticks=self.warmup_ticks,
            )
        elif detector_name == "ewma":
            return ServiceDetectorSuite(
                detector_factory=lambda: EWMADetector(s),
                warmup_ticks=self.warmup_ticks,
            )
        elif detector_name == "cpr_residual":
            return ServiceDetectorSuite(
                detector_factory=lambda: CostPerRequestResidualDetector(s),
                warmup_ticks=self.warmup_ticks,
            )
        elif detector_name == "river_hst":
            return ServiceDetectorSuite(
                detector_factory=lambda: RiverHSTDetector(),
                warmup_ticks=self.warmup_ticks,
            )
        elif detector_name == "isolation_forest":
            return ServiceDetectorSuite(
                detector_factory=lambda: StreamingIsolationForestDetector(),
                warmup_ticks=self.warmup_ticks,
            )
        else:
            raise ValueError(f"Unknown single detector: {detector_name}")

    def create_ensemble_runner(
        self,
        strategy: str = "hard_vote",
        k: int = 2,
        active_detectors: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        logistic_model: Optional[Any] = None,
    ) -> StreamingEnsembleRunner:
        """Create an ensemble combining multiple streaming detectors."""
        det_names = active_detectors or ["robust_zscore", "ewma", "cpr_residual", "isolation_forest", "river_hst"]
        suites = {name: self.create_single_detector_suite(name) for name in det_names}
        return StreamingEnsembleRunner(
            strategy=strategy,
            suites=suites,
            k=k,
            weights=weights,
            logistic_model=logistic_model,
            cooldown_ticks=self.cooldown_ticks,
        )

    def replay_single_detector(
        self,
        events_df: pd.DataFrame,
        detector_name: str,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Replay scenario through a single streaming detector.
        Returns (flags_df, mean_event_cost_us).
        """
        suite = self.create_single_detector_suite(detector_name)
        n = len(events_df)
        flags = np.zeros(n, dtype=bool)
        scores = np.zeros(n, dtype=np.float64)

        t0 = time.perf_counter()
        for idx, row in enumerate(events_df.itertuples()):
            ev = row_to_cost_event(row)
            det = suite.get(ev.service)
            res = det.update(ev)
            flags[idx] = res.is_anomaly
            scores[idx] = res.score
        total_time_s = time.perf_counter() - t0
        mean_cost_us = (total_time_s / max(1, n)) * 1e6

        out = events_df[["sim_ts", "tick_index", "service", "unblended_cost"]].copy()
        if "is_anomalous" in events_df.columns:
            out["is_anomalous"] = events_df["is_anomalous"]
        out["is_anomaly"] = flags
        out["score"] = scores
        return out, mean_cost_us

    def replay_ensemble(
        self,
        events_df: pd.DataFrame,
        ensemble_runner: StreamingEnsembleRunner,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Replay scenario through a StreamingEnsembleRunner.
        Returns (flags_df, mean_event_cost_us).
        """
        n = len(events_df)
        flags = np.zeros(n, dtype=bool)
        scores = np.zeros(n, dtype=np.float64)

        t0 = time.perf_counter()
        for idx, row in enumerate(events_df.itertuples()):
            ev = row_to_cost_event(row)
            is_anom, score, _ = ensemble_runner.update(ev)
            flags[idx] = is_anom
            scores[idx] = score
        total_time_s = time.perf_counter() - t0
        mean_cost_us = (total_time_s / max(1, n)) * 1e6

        out = events_df[["sim_ts", "tick_index", "service", "unblended_cost"]].copy()
        if "is_anomalous" in events_df.columns:
            out["is_anomalous"] = events_df["is_anomalous"]
        out["is_anomaly"] = flags
        out["score"] = scores
        return out, mean_cost_us

    def replay_naive_baseline(
        self,
        events_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, float]:
        """Replay through NaiveStaticThresholdBaseline."""
        base = NaiveStaticThresholdBaseline(window_ticks=self.warmup_ticks)
        n = len(events_df)
        flags = np.zeros(n, dtype=bool)
        scores = np.zeros(n, dtype=np.float64)

        t0 = time.perf_counter()
        for idx, row in enumerate(events_df.itertuples()):
            is_anom, sc = base.update(row.service, row.sim_ts, float(row.unblended_cost))
            flags[idx] = is_anom
            scores[idx] = sc
        total_time_s = time.perf_counter() - t0
        mean_cost_us = (total_time_s / max(1, n)) * 1e6

        out = events_df[["sim_ts", "tick_index", "service", "unblended_cost"]].copy()
        if "is_anomalous" in events_df.columns:
            out["is_anomalous"] = events_df["is_anomalous"]
        out["is_anomaly"] = flags
        out["score"] = scores
        return out, mean_cost_us

    def replay_prophet_baseline(
        self,
        events_df: pd.DataFrame,
        target_services: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Replay through RollingProphetBaseline.
        Returns (flags_df, mean_event_cost_us, total_fit_time_s).
        """
        base = RollingProphetBaseline(history_ticks=864, step_ticks=72)
        all_services = list(events_df["service"].unique())
        services_to_run = target_services if target_services is not None else all_services

        dfs = []
        t0 = time.perf_counter()
        for svc in services_to_run:
            res_svc = base.run_series(events_df, svc)
            dfs.append(res_svc)

        # For services not targeted, fill with non-anomalous baseline
        for svc in all_services:
            if svc not in services_to_run:
                sub = events_df[events_df["service"] == svc].copy()
                sub["is_anomaly"] = False
                sub["score"] = 0.0
                sub["yhat_upper"] = sub["unblended_cost"] * 1.5
                dfs.append(sub)

        total_time_s = time.perf_counter() - t0

        out = pd.concat(dfs, ignore_index=True).sort_values(["tick_index", "service"]).reset_index(drop=True)
        if "is_anomalous" in events_df.columns:
            lookup = events_df[["sim_ts", "service", "is_anomalous"]].drop_duplicates()
            out = pd.merge(out, lookup, on=["sim_ts", "service"], how="left")

        mean_cost_us = (total_time_s / max(1, len(out))) * 1e6
        return out, mean_cost_us, base.total_fit_time_s

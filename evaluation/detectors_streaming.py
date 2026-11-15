"""
evaluation/detectors_streaming.py
=================================
Streaming anomaly detector wrappers for offline deterministic evaluation.
Reuses the exact production detectors from `pipeline.detectors` and provides
River HST, Streaming IsolationForest, and ensemble voting strategies.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

from pipeline.config import PipelineSettings
from pipeline.detectors import (
    CostPerRequestResidualDetector,
    EWMADetector,
    RobustZScoreDetector,
    StreamDetector,
    WarmUpWrapper,
)
from pipeline.models import CostEvent, DetectorResult

try:
    from river.anomaly import HalfSpaceTrees
    _HAS_RIVER = True
except ImportError:
    _HAS_RIVER = False


# ---------------------------------------------------------------------------
# River HalfSpaceTrees Streaming Detector
# ---------------------------------------------------------------------------

class RiverHSTDetector(StreamDetector):
    """River HalfSpaceTrees online anomaly detector."""

    name: str = "river_hst"

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 15,
        window_size: int = 250,
        threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        if not _HAS_RIVER:
            raise ImportError("river is required for RiverHSTDetector. Install with: pip install river")
        self._n_trees = n_trees
        self._height = height
        self._window_size = window_size
        self._threshold = threshold
        self._seed = seed
        self._model = HalfSpaceTrees(
            n_trees=self._n_trees,
            height=self._height,
            window_size=self._window_size,
            seed=self._seed,
        )

    def update(self, event: CostEvent) -> DetectorResult:
        features = {
            "cost": float(event.unblended_cost),
            "requests": float(event.request_count),
            "usage": float(event.usage_amount),
            "cost_per_req": float(event.unblended_cost / max(1.0, event.request_count)),
        }
        score: float = self._model.score_one(features)
        self._model.learn_one(features)
        is_anomaly = score >= self._threshold
        return DetectorResult(
            detector_name=self.name,
            score=float(score),
            is_anomaly=bool(is_anomaly),
            explanation=f"hst_score={score:.4f}",
        )

    def reset(self) -> None:
        if _HAS_RIVER:
            self._model = HalfSpaceTrees(
                n_trees=self._n_trees,
                height=self._height,
                window_size=self._window_size,
                seed=self._seed,
            )


# ---------------------------------------------------------------------------
# Streaming Isolation Forest (Sliding Window)
# ---------------------------------------------------------------------------

class StreamingIsolationForestDetector(StreamDetector):
    """
    Sliding-window Isolation Forest streaming detector.
    Maintains a rolling window of feature vectors, periodically refitting
    the model while scoring every event online.
    """

    name: str = "isolation_forest"

    def __init__(
        self,
        window_size: int = 288,
        refit_interval: int = 36,
        contamination: float = 0.05,
        score_threshold: float = 0.60,
        seed: int = 42,
    ) -> None:
        self.window_size = window_size
        self.refit_interval = refit_interval
        self.contamination = contamination
        self.score_threshold = score_threshold
        self.seed = seed

        self._window: deque = deque(maxlen=window_size)
        self._model: Optional[IsolationForest] = None
        self._ticks_since_fit: int = 0

    def _extract_features(self, event: CostEvent) -> np.ndarray:
        cpr = event.unblended_cost / max(1.0, event.request_count)
        return np.array([event.unblended_cost, event.request_count, cpr], dtype=np.float64)

    def update(self, event: CostEvent) -> DetectorResult:
        feat = self._extract_features(event)
        self._window.append(feat)
        self._ticks_since_fit += 1

        # Periodic refit
        if self._ticks_since_fit >= self.refit_interval and len(self._window) >= 50:
            X = np.array(self._window)
            self._model = IsolationForest(
                n_estimators=50,
                contamination=self.contamination,
                random_state=self.seed,
                n_jobs=1,
            )
            self._model.fit(X)
            self._ticks_since_fit = 0

        if self._model is not None:
            # -score_samples gives higher value for more anomalous observations
            raw_score = -float(self._model.score_samples(feat.reshape(1, -1))[0])
            is_anomaly = raw_score >= self.score_threshold
            norm_score = float(np.clip((raw_score - 0.4) / 0.4, 0.0, 1.0))
        else:
            norm_score = 0.0
            is_anomaly = False

        return DetectorResult(
            detector_name=self.name,
            score=norm_score,
            is_anomaly=is_anomaly,
            explanation=f"if_score={norm_score:.4f}",
        )

    def reset(self) -> None:
        self._window.clear()
        self._model = None
        self._ticks_since_fit = 0


# ---------------------------------------------------------------------------
# Multi-Service Detector Container
# ---------------------------------------------------------------------------

class ServiceDetectorSuite:
    """
    Manages stateful detector instances partitioned by service name.
    Ensures identical state isolation to production.
    """

    def __init__(
        self,
        detector_factory: Callable[[], StreamDetector],
        warmup_ticks: int = 20,
    ) -> None:
        self.detector_factory = detector_factory
        self.warmup_ticks = warmup_ticks
        self._detectors: Dict[str, WarmUpWrapper] = {}

    def get(self, service: str) -> WarmUpWrapper:
        if service not in self._detectors:
            base = self.detector_factory()
            self._detectors[service] = WarmUpWrapper(base, self.warmup_ticks)
        return self._detectors[service]

    def reset(self) -> None:
        for d in self._detectors.values():
            d.reset()
        self._detectors.clear()


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

class StreamingEnsembleRunner:
    """
    Evaluates streaming ensemble strategies (hard vote, weighted vote, logistic stack)
    over underlying individual detector instances.
    """

    def __init__(
        self,
        strategy: str,
        suites: Dict[str, ServiceDetectorSuite],
        k: int = 2,
        weights: Optional[Dict[str, float]] = None,
        logistic_model: Optional[Any] = None,
        cooldown_ticks: int = 6,
    ) -> None:
        self.strategy = strategy  # "hard_vote" | "weighted_vote" | "logistic_stack"
        self.suites = suites
        self.k = k
        self.weights = weights or {name: 1.0 for name in suites}
        self.logistic_model = logistic_model
        self.cooldown_ticks = cooldown_ticks
        self._cooldown_until: Dict[str, int] = {}

    def update(self, event: CostEvent) -> Tuple[bool, float, Dict[str, DetectorResult]]:
        """
        Ingest event into all detectors for event.service.
        Returns:
            (is_anomaly, ensemble_score, individual_results)
        """
        results: Dict[str, DetectorResult] = {}
        for name, suite in self.suites.items():
            det = suite.get(event.service)
            res = det.update(event)
            results[name] = res

        # Cooldown check
        svc = event.service
        cooldown = self._cooldown_until.get(svc, -1)
        in_cooldown = event.tick_index <= cooldown

        is_anomaly = False
        score = 0.0

        if self.strategy == "hard_vote":
            votes = sum(1 for r in results.values() if r.is_anomaly)
            is_anomaly = votes >= self.k
            score = votes / max(1, len(results))

        elif self.strategy == "weighted_vote":
            total_w = sum(self.weights.get(name, 1.0) for name in results)
            weighted_sum = sum(
                self.weights.get(name, 1.0) for name, r in results.items() if r.is_anomaly
            )
            score = weighted_sum / max(1e-6, total_w)
            is_anomaly = score >= 0.5

        elif self.strategy == "logistic_stack":
            if self.logistic_model is not None:
                # Features: [res.score, float(res.is_anomaly)] for each detector in alphabetical order
                feats = []
                for name in sorted(self.suites.keys()):
                    r = results[name]
                    feats.extend([r.score, 1.0 if r.is_anomaly else 0.0])
                X = np.array(feats).reshape(1, -1)
                prob = float(self.logistic_model.predict_proba(X)[0, 1])
                score = prob
                is_anomaly = prob >= 0.5
            else:
                votes = sum(1 for r in results.values() if r.is_anomaly)
                is_anomaly = votes >= self.k
                score = votes / max(1, len(results))

        if in_cooldown:
            is_anomaly = False

        if is_anomaly:
            self._cooldown_until[svc] = event.tick_index + self.cooldown_ticks

        return is_anomaly, score, results

    def reset(self) -> None:
        for suite in self.suites.values():
            suite.reset()
        self._cooldown_until.clear()

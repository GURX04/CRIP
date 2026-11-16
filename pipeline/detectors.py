"""
pipeline/detectors.py
---------------------
Real-time streaming anomaly detectors for the CRIP cost-monitoring pipeline.

Each detector is stateful, service-scoped, and independent (no global state).
All hyper-parameters are sourced exclusively from ``PipelineSettings``; no
magic numbers appear in the implementation.

Public API
----------
StreamDetector          – Abstract base class every detector must satisfy.
RobustZScoreDetector    – Hour-of-day EWMA + robust z-score with consecutive-
                          tick gating.
EWMADetector            – EWMA control chart (Shewhart-style with EW variance).
CostPerRequestResidualDetector – Online EW linear regression cost ~ f(requests).
WarmUpWrapper           – Suppresses signals until a warm-up period elapses.

Extension point for River HalfSpaceTrees is documented at the bottom of this
file.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List

from pipeline.config import PipelineSettings
from pipeline.models import CostEvent, DetectorResult

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StreamDetector(ABC):
    """Abstract base class for all streaming anomaly detectors.

    Each concrete subclass processes events for **one service** and maintains
    its own internal state.  Instances must never share mutable state.

    Attributes
    ----------
    name : str
        Human-readable identifier used in ``DetectorResult.detector_name``.
    """

    name: str

    @abstractmethod
    def update(self, event: CostEvent) -> DetectorResult:
        """Ingest one event and return an anomaly assessment.

        Parameters
        ----------
        event:
            The incoming cost event to process.

        Returns
        -------
        DetectorResult
            Populated with the detector's name, score, anomaly flag and a
            short human-readable explanation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state to initial conditions.

        Useful for test isolation and pipeline restarts within a run.
        """


# ---------------------------------------------------------------------------
# RobustZScoreDetector
# ---------------------------------------------------------------------------


class RobustZScoreDetector(StreamDetector):
    """Hour-of-day EWMA + robust z-score detector with consecutive-tick gating.

    Strategy
    --------
    - Maintains 24 independent EWMA buckets (one per hour-of-day) so that
      expected cost accounts for intra-day seasonality.
    - Computes a residual against the bucket's current EWMA, then measures
      that residual on a *robust* z-score scale (median / MAD over a rolling
      window) to suppress sensitivity to outliers in the history.
    - Flags an anomaly only after ``zscore_consec`` consecutive ticks exceed
      ``zscore_threshold``, reducing spurious one-off alerts.

    Parameters
    ----------
    settings:
        Pipeline configuration.  Reads ``ewma_alpha``, ``zscore_threshold``,
        ``zscore_consec``.
    residual_window:
        Maximum number of residuals kept in the rolling window for median/MAD
        computation.  Defaults to 20.
    """

    name: str = "robust_zscore"
    _SCORE_CLIP: float = 20.0
    _MAD_SCALE: float = 1.4826  # factor that makes MAD consistent with σ

    def __init__(
        self,
        settings: PipelineSettings,
        residual_window: int = 20,
    ) -> None:
        self._alpha: float = settings.ewma_alpha
        self._threshold: float = settings.zscore_threshold
        self._consec_required: int = settings.zscore_consec
        self._window: int = residual_window
        self._reset_state()

    # ------------------------------------------------------------------
    # StreamDetector interface
    # ------------------------------------------------------------------

    def update(self, event: CostEvent) -> DetectorResult:
        """Process one event and return a robust z-score anomaly result.

        Parameters
        ----------
        event:
            Incoming cost event.

        Returns
        -------
        DetectorResult
            ``is_anomaly`` is ``True`` only when the robust z-score exceeds
            the threshold for at least ``zscore_consec`` consecutive ticks.
        """
        cost: float = event.unblended_cost
        hour: int = event.sim_ts.hour

        # --- seasonal expected value from bucket EWMA ---
        if hour not in self._seeded_hours:
            # Cold bucket: seed with the first observation so the first visit
            # to each hour-of-day does not look like a huge jump from zero.
            self._hourly_ewma[hour] = cost
            self._seeded_hours.add(hour)
        expected: float = self._hourly_ewma[hour]
        residual: float = cost - expected

        # Update the bucket EWMA *after* computing the residual so the
        # expected value is from before this observation.
        self._hourly_ewma[hour] = (
            self._alpha * cost + (1.0 - self._alpha) * expected
        )

        # --- rolling residual history ---
        self._residuals.append(residual)

        # --- robust z-score ---
        sorted_resids: List[float] = sorted(self._residuals)
        median: float = _median_sorted(sorted_resids)
        mad: float = _mad_sorted(sorted_resids, median)
        denom: float = max(self._MAD_SCALE * mad, 1e-9)
        robust_z: float = (residual - median) / denom

        # --- consecutive-tick gate ---
        if abs(robust_z) > self._threshold:
            self._consec_count += 1
        else:
            self._consec_count = 0

        is_anomaly: bool = self._consec_count >= self._consec_required
        score: float = min(abs(robust_z), self._SCORE_CLIP)
        explanation: str = (
            f"robust_z={robust_z:.2f}, consec={self._consec_count}"
        )

        return DetectorResult(
            detector_name=self.name,
            score=score,
            is_anomaly=is_anomaly,
            explanation=explanation,
        )

    def reset(self) -> None:
        """Reset all internal state."""
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        # 24 hour-of-day EWMA buckets, initialised to zero.
        self._hourly_ewma: Dict[int, float] = {h: 0.0 for h in range(24)}
        self._seeded_hours: set = set()
        # Rolling deque of raw residuals for robust statistics.
        self._residuals: Deque[float] = deque(maxlen=self._window if hasattr(self, "_window") else 20)
        # Consecutive ticks above threshold counter.
        self._consec_count: int = 0


# ---------------------------------------------------------------------------
# EWMADetector
# ---------------------------------------------------------------------------


class EWMADetector(StreamDetector):
    """EWMA control chart anomaly detector.

    Maintains an exponentially weighted mean and variance of ``unblended_cost``
    and flags observations that exceed ``ewma_cost + ewma_k * sqrt(ewma_var)``.

    Parameters
    ----------
    settings:
        Pipeline configuration.  Reads ``ewma_alpha``, ``ewma_k``.
    """

    name: str = "ewma"
    _STD_FLOOR_FRAC: float = 0.02

    def __init__(self, settings: PipelineSettings) -> None:
        self._alpha: float = settings.ewma_alpha
        self._k: float = settings.ewma_k
        self._reset_state()

    # ------------------------------------------------------------------
    # StreamDetector interface
    # ------------------------------------------------------------------

    def update(self, event: CostEvent) -> DetectorResult:
        """Process one event and return an EWMA control-chart result.

        Parameters
        ----------
        event:
            Incoming cost event.

        Returns
        -------
        DetectorResult
            ``is_anomaly`` is ``True`` when cost exceeds the upper control
            limit defined by the EWMA mean + k * sqrt(EWMA variance).
        """
        cost: float = event.unblended_cost
        alpha: float = self._alpha

        # --- Cold start: seed the baseline with the first observation ---
        # (starting from 0.0 makes every normal event look anomalous until
        # the EWMA has climbed up to the real cost level)
        if not self._seeded:
            self._ewma_cost = cost
            self._seeded = True

        # --- Check control limit against current baseline ---
        # Floor the std at a small fraction of the level so perfectly flat
        # traffic (var == 0) can still raise an alert on a real jump.
        std: float = max(
            math.sqrt(max(self._ewma_var, 0.0)),
            self._STD_FLOOR_FRAC * abs(self._ewma_cost),
        )
        upper: float = self._ewma_cost + self._k * std
        score: float = max(0.0, (cost - self._ewma_cost) / max(std, 1e-9)) if std > 0 else 0.0
        is_anomaly: bool = cost > upper if std > 0 else False
        explanation: str = (
            f"cost={cost:.4f}, ewma={self._ewma_cost:.4f}, "
            f"upper={upper:.4f}, std={std:.4f}"
        )

        # --- Update EWMA mean & variance (winsorised) ---
        # The innovation is clipped to +/- k*std before it enters the update.
        # A real spike therefore moves the baseline only slowly (it keeps
        # alerting for several ticks), but the baseline and variance never
        # freeze: previously an anomalous tick skipped the variance update and
        # damped the mean, so a normal diurnal ramp latched the detector into
        # "anomaly" for hours.
        delta: float = cost - self._ewma_cost
        if self._ewma_var > 0.0:
            clip: float = self._k * std
            delta_w: float = max(-clip, min(clip, delta))
        else:
            # Variance not bootstrapped yet (e.g. first tick had cost 0):
            # clipping to k*0 would freeze the baseline forever.
            delta_w = delta
        self._ewma_cost += alpha * delta_w
        self._ewma_var = (1.0 - alpha) * (self._ewma_var + alpha * delta_w ** 2)

        return DetectorResult(
            detector_name=self.name,
            score=score,
            is_anomaly=is_anomaly,
            explanation=explanation,
        )

    def reset(self) -> None:
        """Reset all internal state."""
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._ewma_cost: float = 0.0
        self._ewma_var: float = 0.0
        self._seeded: bool = False


# ---------------------------------------------------------------------------
# CostPerRequestResidualDetector
# ---------------------------------------------------------------------------


class CostPerRequestResidualDetector(StreamDetector):
    """Online EW linear-regression detector: cost ~ beta0 + beta1 * requests.

    Fits a simple linear model online using exponentially weighted sufficient
    statistics, so proportionate cost/request increases stay quiet while
    disproportionate cost increases (higher unit price, billing error, etc.)
    trigger an alert.

    Parameters
    ----------
    settings:
        Pipeline configuration.  Reads ``cpr_alpha``, ``cpr_threshold``.

    Notes
    -----
    The EW sufficient statistics follow the recursive update:

        Sxx_{t} = w * Sxx_{t-1} + x_t^2
        Sxy_{t} = w * Sxy_{t-1} + x_t * y_t
        Sx_{t}  = w * Sx_{t-1}  + x_t
        Sy_{t}  = w * Sy_{t-1}  + y_t
        n_eff   = w * n_eff_{t-1} + 1

    where ``w = 1 - alpha`` is the forgetting factor and
    ``x = request_count``, ``y = unblended_cost``.
    """

    name: str = "cost_per_request_residual"

    def __init__(self, settings: PipelineSettings) -> None:
        self._alpha: float = settings.cpr_alpha
        self._threshold: float = settings.cpr_threshold
        self._reset_state()

    # ------------------------------------------------------------------
    # StreamDetector interface
    # ------------------------------------------------------------------

    def update(self, event: CostEvent) -> DetectorResult:
        """Process one event and return a cost-per-request residual result.

        Parameters
        ----------
        event:
            Incoming cost event.

        Returns
        -------
        DetectorResult
            ``is_anomaly`` is ``True`` only when the residual z-score exceeds
            the threshold *and* cost is higher than predicted (not just
            different), so proportionate increases stay quiet.
        """
        x: float = event.request_count
        y: float = event.unblended_cost
        alpha: float = self._alpha
        w: float = 1.0 - alpha  # forgetting / decay factor

        # --- Recursive EW sufficient-statistic update ---
        self._Sxx = w * self._Sxx + x * x
        self._Sxy = w * self._Sxy + x * y
        self._Sx = w * self._Sx + x
        self._Sy = w * self._Sy + y
        self._n_eff = w * self._n_eff + 1.0

        # --- OLS estimates from EW statistics ---
        # Numerator of beta1 uses the centred cross-product identity:
        # Sxy_centred = Sxy - Sx*Sy/n_eff
        n: float = max(self._n_eff, 1.0)
        sxx_centred: float = self._Sxx - self._Sx * self._Sx / n
        sxy_centred: float = self._Sxy - self._Sx * self._Sy / n

        beta1: float = sxy_centred / max(sxx_centred, 1e-9)
        beta0: float = (self._Sy - beta1 * self._Sx) / n

        predicted: float = beta0 + beta1 * x
        residual: float = y - predicted

        # --- EW mean & variance of residuals ---
        delta_r: float = residual - self._ewma_resid
        self._ewma_resid += alpha * delta_r
        self._var_resid = (1.0 - alpha) * (
            self._var_resid + alpha * delta_r ** 2
        )

        std_r: float = math.sqrt(max(self._var_resid, 0.0))
        resid_z: float = residual / max(std_r, 1e-9)

        is_anomaly: bool = (
            abs(resid_z) > self._threshold and residual > 0.0
        )
        score: float = max(0.0, resid_z)
        explanation: str = (
            f"predicted={predicted:.4f}, residual={residual:.4f}, "
            f"resid_z={resid_z:.2f}, beta1={beta1:.6f}"
        )

        return DetectorResult(
            detector_name=self.name,
            score=score,
            is_anomaly=is_anomaly,
            explanation=explanation,
        )

    def reset(self) -> None:
        """Reset all internal state."""
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        # EW sufficient statistics for linear regression
        self._Sxx: float = 0.0
        self._Sxy: float = 0.0
        self._Sx: float = 0.0
        self._Sy: float = 0.0
        self._n_eff: float = 0.0
        # EW mean & variance of residuals
        self._ewma_resid: float = 0.0
        self._var_resid: float = 0.0


# ---------------------------------------------------------------------------
# WarmUpWrapper
# ---------------------------------------------------------------------------


class WarmUpWrapper(StreamDetector):
    """Suppress anomaly signals during a detector's warm-up period.

    Wraps any ``StreamDetector`` and passes all events through, but forces
    ``is_anomaly=False`` and ``score=0.0`` until ``warmup_ticks`` events have
    been seen.  After warm-up the wrapped detector's results are returned
    unmodified.

    Parameters
    ----------
    detector:
        The underlying detector to wrap.
    settings:
        Pipeline configuration.  Reads ``warmup_ticks``.
    """

    def __init__(
        self,
        detector: StreamDetector,
        settings: PipelineSettings | int,
    ) -> None:
        self._detector: StreamDetector = detector
        if isinstance(settings, int):
            self._warmup_ticks: int = settings
        else:
            self._warmup_ticks: int = settings.warmup_ticks
        self._ticks_seen: int = 0
        # Expose the wrapped detector's name so logs are identifiable.
        self.name: str = f"warmup({detector.name})"

    # ------------------------------------------------------------------
    # StreamDetector interface
    # ------------------------------------------------------------------

    def update(self, event: CostEvent) -> DetectorResult:
        """Forward the event to the wrapped detector, suppressing during warm-up.

        Parameters
        ----------
        event:
            Incoming cost event.

        Returns
        -------
        DetectorResult
            During warm-up: ``is_anomaly=False``, ``score=0.0``.
            After warm-up: the wrapped detector's result, unchanged.
        """
        self._ticks_seen += 1
        result: DetectorResult = self._detector.update(event)

        if self._ticks_seen < self._warmup_ticks:
            return DetectorResult(
                detector_name=self.name,
                score=0.0,
                is_anomaly=False,
                explanation=(
                    f"warming up ({self._ticks_seen}/{self._warmup_ticks}): "
                    + result.explanation
                ),
            )

        # Return the real result but rename to reflect the wrapper.
        return DetectorResult(
            detector_name=self.name,
            score=result.score,
            is_anomaly=result.is_anomaly,
            explanation=result.explanation,
        )

    def reset(self) -> None:
        """Reset warm-up counter and delegate to the wrapped detector."""
        self._ticks_seen = 0
        self._detector.reset()


# ---------------------------------------------------------------------------
# Private statistical helpers (module-private, not part of public API)
# ---------------------------------------------------------------------------


def _median_sorted(sorted_values: List[float]) -> float:
    """Return the median of a *pre-sorted* list.

    Parameters
    ----------
    sorted_values:
        A sorted (ascending) list of floats.  Must be non-empty.

    Returns
    -------
    float
        The median value.
    """
    n: int = len(sorted_values)
    if n == 0:
        return 0.0
    mid: int = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _mad_sorted(sorted_values: List[float], median: float) -> float:
    """Return the Median Absolute Deviation for a *pre-sorted* list.

    Parameters
    ----------
    sorted_values:
        A sorted (ascending) list of floats.
    median:
        The pre-computed median of ``sorted_values``.

    Returns
    -------
    float
        The MAD (not scaled); multiply by 1.4826 for consistency with σ.
    """
    if not sorted_values:
        return 0.0
    abs_devs: List[float] = sorted(abs(v - median) for v in sorted_values)
    return _median_sorted(abs_devs)


# ---------------------------------------------------------------------------
# EXTENSION: River HalfSpaceTrees
# ---------------------------------------------------------------------------
# To add River's online anomaly detector as an additional StreamDetector:
#
#   1. Install the dependency:
#          pip install river
#
#   2. Uncomment and adapt the class below.  Add any extra settings fields
#      to PipelineSettings (e.g. hst_n_trees, hst_height, hst_window_size).
#
# from river.anomaly import HalfSpaceTrees
#
# class RiverHSTDetector(StreamDetector):
#     """River HalfSpaceTrees online anomaly detector.
#
#     Uses River's implementation of the streaming Half-Space Trees algorithm
#     (Tan et al., 2011) as an additional ensemble member.
#
#     Parameters
#     ----------
#     settings:
#         Pipeline configuration.  Add ``hst_n_trees``, ``hst_height``,
#         ``hst_window_size`` to PipelineSettings before enabling this class.
#     """
#
#     name: str = "river_hst"
#
#     def __init__(self, settings: PipelineSettings) -> None:
#         self._model = HalfSpaceTrees(
#             n_trees=getattr(settings, "hst_n_trees", 25),
#             height=getattr(settings, "hst_height", 15),
#             window_size=getattr(settings, "hst_window_size", 250),
#         )
#
#     def update(self, event: CostEvent) -> DetectorResult:
#         features = {
#             "cost": event.unblended_cost,
#             "request_count": event.request_count,
#             "usage_amount": event.usage_amount,
#             "unit_price": event.unit_price,
#         }
#         score: float = self._model.score_one(features)
#         self._model.learn_one(features)
#         is_anomaly: bool = score > 0.5  # HST scores are in [0, 1]
#         return DetectorResult(
#             detector_name=self.name,
#             score=score,
#             is_anomaly=is_anomaly,
#             explanation=f"hst_score={score:.4f}",
#         )
#
#     def reset(self) -> None:
#         self._model = HalfSpaceTrees()
# ---------------------------------------------------------------------------
"""
evaluation/metrics.py
=====================
Mathematical and statistical scoring engine for CRIP anomaly detectors.
Provides incident-level matching, latency calculation, false-alarm rates,
event-level metrics, alert fragmentation, bootstrap confidence intervals,
and paired significance testing (Wilcoxon signed-rank test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class IncidentMatchResult:
    injection_id: str
    type: str
    service: str
    magnitude: float
    start_tick: int
    end_tick: int
    detected: bool
    latency_ticks: Optional[int]
    latency_sim_min: Optional[float]
    flag_count: int


@dataclass
class MetricSummary:
    detector: str
    scenario: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    total_injections: int
    false_alarms_per_day: float
    median_latency_min: Optional[float]
    mean_latency_min: Optional[float]
    p90_latency_min: Optional[float]
    missed_fraction: float
    alert_fragmentation: float
    event_precision: float
    event_recall: float
    event_f1: float
    mean_event_cost_us: float = 0.0
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_magnitude: Dict[str, Dict[str, float]] = field(default_factory=dict)


def evaluate_scenario_detections(
    detector_name: str,
    scenario_name: str,
    flags_df: pd.DataFrame,          # columns: ['sim_ts', 'tick_index', 'service', 'is_anomaly', 'score']
    ground_truth: List[Dict[str, Any]],
    duration_days: float,
    tolerance_ticks: int = 2,
    sim_interval_min: int = 5,
    cooldown_ticks: int = 6,
    event_cost_us: float = 0.0,
) -> Tuple[MetricSummary, List[IncidentMatchResult]]:
    """
    Evaluates detector flags against ground truth for a single scenario run.
    """
    # Filter active flags
    active_flags = flags_df[flags_df["is_anomaly"]].copy()

    # Pre-index flags by service and tick
    flags_by_service: Dict[str, List[int]] = {}
    for r in active_flags.itertuples():
        flags_by_service.setdefault(r.service, []).append(r.tick_index)
        flags_by_service.setdefault("all", []).append(r.tick_index)

    incident_results: List[IncidentMatchResult] = []
    tp_count = 0
    fn_count = 0
    matched_flag_indices: set[Tuple[str, int]] = set()
    latencies_min: List[float] = []
    total_inside_flags = 0

    # 1. Match each ground-truth injection
    for inj in ground_truth:
        svc = inj["service"]
        s_tick = inj["start_tick"]
        e_tick = inj["end_tick"]
        w_start = s_tick - tolerance_ticks
        w_end = e_tick + tolerance_ticks

        target_flags: List[int] = []
        if svc == "all":
            target_flags = sorted(list(set(flags_by_service.get("all", []))))
        else:
            target_flags = flags_by_service.get(svc, [])

        inside_flags = [t for t in target_flags if w_start <= t <= w_end]
        detected = len(inside_flags) > 0

        lat_ticks: Optional[int] = None
        lat_min: Optional[float] = None

        if detected:
            tp_count += 1
            first_t = inside_flags[0]
            lat_ticks = max(0, first_t - s_tick)
            lat_min = float(lat_ticks * sim_interval_min)
            latencies_min.append(lat_min)
            total_inside_flags += len(inside_flags)

            for t in inside_flags:
                matched_flag_indices.add((svc, t))
                if svc == "all":
                    for s in flags_by_service.keys():
                        matched_flag_indices.add((s, t))
        else:
            fn_count += 1

        incident_results.append(IncidentMatchResult(
            injection_id=inj["injection_id"],
            type=inj["type"],
            service=svc,
            magnitude=float(inj["magnitude"]),
            start_tick=s_tick,
            end_tick=e_tick,
            detected=detected,
            latency_ticks=lat_ticks,
            latency_sim_min=lat_min,
            flag_count=len(inside_flags),
        ))

    # 2. Count False Positives (clustered by cooldown outside injection windows)
    fp_count = 0
    for svc, tick_list in flags_by_service.items():
        if svc == "all":
            continue
        sorted_ticks = sorted(tick_list)
        last_fp_tick = -9999
        for t in sorted_ticks:
            if (svc, t) not in matched_flag_indices:
                # Outside any valid window. Check cooldown clustering
                if t - last_fp_tick > cooldown_ticks:
                    fp_count += 1
                    last_fp_tick = t

    total_injections = len(ground_truth)
    recall = tp_count / total_injections if total_injections > 0 else (1.0 if fp_count == 0 else 0.0)
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else (1.0 if total_injections == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    fa_per_day = fp_count / max(0.1, duration_days)
    med_lat = float(np.median(latencies_min)) if latencies_min else None
    mean_lat = float(np.mean(latencies_min)) if latencies_min else None
    p90_lat = float(np.percentile(latencies_min, 90)) if latencies_min else None
    missed_frac = fn_count / total_injections if total_injections > 0 else 0.0
    frag = (total_inside_flags / tp_count) if tp_count > 0 else 1.0

    # 3. Event-level metrics
    if "is_anomalous" in flags_df.columns:
        pred_anom = flags_df["is_anomaly"].values
        true_anom = flags_df["is_anomalous"].values
        ev_tp = int(np.sum(pred_anom & true_anom))
        ev_fp = int(np.sum(pred_anom & ~true_anom))
        ev_fn = int(np.sum(~pred_anom & true_anom))
        ev_p = ev_tp / (ev_tp + ev_fp) if (ev_tp + ev_fp) > 0 else 1.0
        ev_r = ev_tp / (ev_tp + ev_fn) if (ev_tp + ev_fn) > 0 else 1.0
        ev_f1 = 2 * ev_p * ev_r / (ev_p + ev_r) if (ev_p + ev_r) > 0 else 0.0
    else:
        ev_p, ev_r, ev_f1 = precision, recall, f1

    # 4. Slices by type and magnitude
    by_type: Dict[str, Dict[str, float]] = {}
    for inj_res in incident_results:
        t = inj_res.type
        st = by_type.setdefault(t, {"total": 0, "detected": 0})
        st["total"] += 1
        if inj_res.detected:
            st["detected"] += 1

    by_type_rec = {
        t: {"recall": d["detected"] / d["total"] if d["total"] > 0 else 0.0, "count": d["total"]}
        for t, d in by_type.items()
    }

    by_mag: Dict[str, Dict[str, float]] = {}
    for inj_res in incident_results:
        m_tier = "low" if inj_res.magnitude <= 1.5 else "medium" if inj_res.magnitude <= 2.5 else "high"
        sm = by_mag.setdefault(m_tier, {"total": 0, "detected": 0})
        sm["total"] += 1
        if inj_res.detected:
            sm["detected"] += 1

    by_mag_rec = {
        m: {"recall": d["detected"] / d["total"] if d["total"] > 0 else 0.0, "count": d["total"]}
        for m, d in by_mag.items()
    }

    summary = MetricSummary(
        detector=detector_name,
        scenario=scenario_name,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=tp_count,
        fp=fp_count,
        fn=fn_count,
        total_injections=total_injections,
        false_alarms_per_day=float(fa_per_day),
        median_latency_min=med_lat,
        mean_latency_min=mean_lat,
        p90_latency_min=p90_lat,
        missed_fraction=float(missed_frac),
        alert_fragmentation=float(frag),
        event_precision=float(ev_p),
        event_recall=float(ev_r),
        event_f1=float(ev_f1),
        mean_event_cost_us=float(event_cost_us),
        by_type=by_type_rec,
        by_magnitude=by_mag_rec,
    )

    return summary, incident_results


# ---------------------------------------------------------------------------
# Statistical Confidence Intervals & Significance
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute mean and non-parametric bootstrap confidence interval [ci_low, ci_high].
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 1 or np.all(arr == arr[0]):
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return float(np.mean(arr)), low, high


def compute_paired_wilcoxon(
    baseline_scores: List[float],
    model_scores: List[float],
) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test between two lists of per-seed F1 scores.
    """
    if len(baseline_scores) != len(model_scores):
        raise ValueError("Score arrays must have equal length for paired test")

    diffs = np.array(model_scores) - np.array(baseline_scores)
    # Check if all zero diffs
    if np.all(diffs == 0.0) or len(diffs) < 3:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "mean_difference": 0.0,
        }

    try:
        stat_res = stats.wilcoxon(diffs, alternative="two-sided")
        pval = float(stat_res.pvalue)
        stat = float(stat_res.statistic)
    except Exception:
        stat, pval = 0.0, 1.0

    return {
        "statistic": stat,
        "p_value": pval,
        "significant": bool(pval < 0.05),
        "mean_difference": float(np.mean(diffs)),
    }

"""
anomaly_detection_v2.py
=======================
CRIP Phase 3 Upgrade — Enhanced Anomaly Detection & Root Cause Analysis

Upgrades over v1
----------------
1. DETECTION — Ensemble of three independent methods:
   a) Rolling Z-score      : fast, explainable, no training required
   b) Isolation Forest     : unsupervised ML, multivariate features
   c) Prophet CI           : seasonality-aware baseline (from v1)
   Anomaly is confirmed when >= VOTE_THRESHOLD methods agree.

2. ROOT CAUSE ANALYSIS — Three layers:
   a) Service Contribution : which AWS service drove the cost excess
   b) Lag-Aware Correlation: which metric best predicts cost at what time offset
      (negative lag = metric LEADS cost = leading indicator of the spike)
   c) Granger Causality    : formal statistical test — does metric X help
      predict future cost Y beyond Y's own history?

3. PERSISTENCE — Anomalies written to detected_anomalies hypertable in
   TimescaleDB for dashboarding and false-positive feedback.

4. SEVERITY SCORING — Composite 0-1 score combining vote count, cost delta
   magnitude, and correlation strength.

Dependencies
------------
    pip install scikit-learn statsmodels scipy

    (pandas, prophet, matplotlib, sqlalchemy, psycopg2-binary from Phase 3 v1)

Usage
-----
    # Standalone (no Docker):
    python anomaly_detection_v2.py --from-csv reports/synthetic_cur.csv

    # Full pipeline (Docker running):
    python anomaly_detection_v2.py

    # Tune ensemble sensitivity:
    python anomaly_detection_v2.py --from-csv reports/synthetic_cur.csv \\
        --zscore-threshold 2.0 --if-contamination 0.05 --vote-threshold 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
import warnings
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

# Suppress noisy library chatter
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("crip.anomaly.v2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOTE_THRESHOLD          = 2     # Methods that must agree to confirm anomaly
ZSCORE_WINDOW           = 12    # Rolling window for Z-score (12 * 5min = 1 hour)
ZSCORE_THRESHOLD        = 2.5   # Standard deviations above mean
IF_CONTAMINATION        = 0.04  # Expected anomaly fraction for Isolation Forest
PROPHET_CI              = 0.95
LAG_RANGE_PERIODS       = 6     # Cross-correlation lags: -6 to +6 intervals (±30 min)
GRANGER_MAXLAG          = 3     # Granger causality max lag (3 * 5min = 15 min)
GRANGER_SIGNIFICANCE    = 0.05  # p-value threshold


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_cost_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load and aggregate CUR CSV into 5-min time buckets."""
    import csv, collections
    log.info("Loading cost series from CSV: %s", csv_path)
    buckets: Dict[str, float] = collections.defaultdict(float)
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ds = row.get("lineItem/UsageStartDate", "")
            cost = row.get("lineItem/UnblendedCost", "0")
            if ds:
                buckets[ds] += float(cost or 0.0)

    df = pd.DataFrame(sorted(buckets.items()), columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"]  = df["y"].astype(float)
    log.info("Loaded %d time buckets from CSV.", len(df))
    return df


def load_cost_from_db(engine, interval: str = "5 minutes") -> pd.DataFrame:
    """Pull aggregated cost series from TimescaleDB using time_bucket."""
    sql = text("""
        SELECT time_bucket(:iv, usage_start) AS ds,
               SUM(unblended_cost)           AS y
        FROM aws_cur_records
        GROUP BY ds ORDER BY ds
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"iv": interval}, parse_dates=["ds"])
    if df.empty:
        log.error("aws_cur_records is empty. Run ingest_to_timescale.py first.")
        sys.exit(1)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"]  = df["y"].astype(float)
    log.info("Loaded %d time buckets from TimescaleDB.", len(df))
    return df


def load_service_breakdown(engine, window_start: pd.Timestamp,
                           window_end: pd.Timestamp) -> pd.DataFrame:
    """Per-service cost for one anomaly window — for contribution analysis."""
    sql = text("""
        SELECT product_code,
               SUM(unblended_cost) AS window_cost
        FROM aws_cur_records
        WHERE usage_start >= :ws AND usage_start < :we
        GROUP BY product_code
        ORDER BY window_cost DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn,
                           params={"ws": window_start.isoformat(),
                                   "we": window_end.isoformat()})


def load_service_breakdown_from_csv(
        csv_path: Path,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Same as above but reads from CSV (offline mode)."""
    import csv, collections
    svc_costs: Dict[str, float] = collections.defaultdict(float)
    ws = window_start.isoformat() + "Z"
    we = window_end.isoformat() + "Z"
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ds = row.get("lineItem/UsageStartDate", "")
            if ws <= ds < we:
                svc = row.get("lineItem/ProductCode", "Unknown")
                svc_costs[svc] += float(row.get("lineItem/UnblendedCost", 0) or 0.0)
    df = pd.DataFrame(
        sorted(svc_costs.items(), key=lambda x: -x[1]),
        columns=["product_code", "window_cost"]
    )
    return df


def load_telemetry_pivot(engine) -> pd.DataFrame:
    """
    Load system_telemetry and pivot into wide format:
    columns = [ts, metric_1, metric_2, ...]
    Useful for multivariate Isolation Forest and lag correlation.
    """
    sql = text("SELECT metric_name, ts, value FROM system_telemetry ORDER BY ts")
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, parse_dates=["ts"])
    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    pivot = df.pivot_table(index="ts", columns="metric_name",
                            values="value", aggfunc="mean")
    pivot.columns.name = None
    pivot = pivot.reset_index()
    return pivot


# ===========================================================================
# FEATURE ENGINEERING
# (Used by Isolation Forest when telemetry is unavailable)
# ===========================================================================

def engineer_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive statistical and cyclical features purely from the cost time series.

    Features
    --------
    y               : raw cost value
    y_rolling_mean  : 1-hour rolling mean
    y_rolling_std   : 1-hour rolling std
    y_zscore        : instantaneous z-score within rolling window
    y_diff          : first difference (rate of change)
    hour_sin/cos    : cyclical encoding of hour-of-day
    cycle_sin/cos   : cyclical encoding of position within 3-hour Locust cycle
    """
    out = df[["ds", "y"]].copy()

    roll = df["y"].rolling(window=ZSCORE_WINDOW, min_periods=3)
    out["y_rolling_mean"] = roll.mean()
    out["y_rolling_std"]  = roll.std().clip(lower=1e-10)
    out["y_zscore"]       = (df["y"] - out["y_rolling_mean"]) / out["y_rolling_std"]
    out["y_diff"]         = df["y"].diff().fillna(0)

    hour_frac = df["ds"].dt.hour + df["ds"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

    # Position within the 3-hour Locust spike cycle (180 minutes)
    minutes_since_midnight = df["ds"].dt.hour * 60 + df["ds"].dt.minute
    cycle_pos = (minutes_since_midnight % 180) / 180.0
    out["cycle_sin"] = np.sin(2 * np.pi * cycle_pos)
    out["cycle_cos"] = np.cos(2 * np.pi * cycle_pos)

    return out.dropna(subset=["y_rolling_mean"])


# ===========================================================================
# DETECTION METHOD 1: Rolling Z-Score
# ===========================================================================

def detect_zscore(
    cost_df: pd.DataFrame,
    window: int = ZSCORE_WINDOW,
    threshold: float = ZSCORE_THRESHOLD,
) -> pd.Series:
    """
    Flag intervals where cost exceeds the rolling mean by `threshold` sigma.

    Returns a boolean Series aligned with cost_df index.
    True = anomaly detected by this method.
    """
    roll  = cost_df["y"].rolling(window=window, min_periods=3, center=False)
    mu    = roll.mean()
    sigma = roll.std().clip(lower=1e-10)
    z     = (cost_df["y"] - mu) / sigma
    flags = z > threshold

    n = flags.sum()
    log.info("[Z-score]        flagged %d / %d  (threshold=%.1f sigma, window=%d)",
             n, len(cost_df), threshold, window)
    return flags.fillna(False)


# ===========================================================================
# DETECTION METHOD 2: Isolation Forest
# ===========================================================================

def detect_isolation_forest(
    cost_df: pd.DataFrame,
    telemetry_pivot: Optional[pd.DataFrame] = None,
    contamination: float = IF_CONTAMINATION,
) -> pd.Series:
    """
    Isolation Forest anomaly detection.

    Features used (in priority order):
      1. Cost-derived statistical features (always available)
      2. Telemetry metrics time-aligned to the cost series (if available)

    Returns a boolean Series aligned with cost_df index.
    """
    feat_df = engineer_cost_features(cost_df)

    # Merge telemetry if available
    if telemetry_pivot is not None and not telemetry_pivot.empty:
        telem = telemetry_pivot.copy()
        # Round telemetry timestamps to 5-min to align with cost buckets
        telem["ds"] = telem["ts"].dt.round("5min")
        telem = telem.drop(columns=["ts"]).groupby("ds").mean().reset_index()
        feat_df = feat_df.merge(telem, on="ds", how="left")
        log.info("[IsolationForest] merged %d telemetry metrics as features.",
                 len(telem.columns) - 1)

    feature_cols = [c for c in feat_df.columns if c not in ("ds", "y")]
    X = feat_df[feature_cols].fillna(feat_df[feature_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    preds = clf.fit_predict(X_scaled)   # -1 = anomaly, +1 = normal
    flags_feat = pd.Series(preds == -1, index=feat_df.index)

    # Re-align to original cost_df index (feat_df may have dropped NaN rows)
    flags = pd.Series(False, index=cost_df.index)
    flags.loc[feat_df.index] = flags_feat

    n = flags.sum()
    log.info("[IsolationForest] flagged %d / %d  (contamination=%.2f, features=%d)",
             n, len(cost_df), contamination, len(feature_cols))
    return flags


# ===========================================================================
# DETECTION METHOD 3: Prophet Confidence Interval (from v1)
# ===========================================================================

def detect_prophet(cost_df: pd.DataFrame, ci: float = PROPHET_CI) -> pd.Series:
    """
    Fit a Prophet model and flag intervals where actual > yhat_upper.
    Returns a boolean Series and the full forecast DataFrame.
    """
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("[Prophet] prophet not installed — skipping this detector.")
        return pd.Series(False, index=cost_df.index), pd.DataFrame()

    model = Prophet(
        interval_width          = ci,
        yearly_seasonality      = False,
        weekly_seasonality      = False,
        daily_seasonality       = True,
        changepoint_prior_scale = 0.15,
        seasonality_mode        = "additive",
    )
    model.add_seasonality(name="three_hourly", period=3/24, fourier_order=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(cost_df[["ds", "y"]])

    future   = model.make_future_dataframe(periods=0, freq="5min",
                                            include_history=True)
    forecast = model.predict(future)
    forecast = forecast.merge(cost_df[["ds", "y"]], on="ds", how="left")

    flags = (forecast["y"] > forecast["yhat_upper"]).fillna(False)
    flags.index = forecast.index   # keep aligned

    # Re-align to cost_df index
    aligned = pd.Series(False, index=cost_df.index)
    aligned.iloc[:len(flags)] = flags.values

    n = aligned.sum()
    log.info("[Prophet]        flagged %d / %d  (CI=%.0f%%)",
             n, len(cost_df), ci * 100)
    return aligned, forecast


# ===========================================================================
# ENSEMBLE: Majority Vote
# ===========================================================================

def run_ensemble(
    cost_df: pd.DataFrame,
    telemetry_pivot: Optional[pd.DataFrame],
    zscore_threshold: float = ZSCORE_THRESHOLD,
    if_contamination: float = IF_CONTAMINATION,
    prophet_ci: float = PROPHET_CI,
    vote_threshold: int = VOTE_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all three detectors and combine via majority vote.

    Returns
    -------
    results_df : cost_df + per-method flags + vote count + is_anomaly
    forecast   : Prophet forecast DataFrame (for plotting)
    """
    log.info("Running ensemble anomaly detection (vote threshold=%d/3)...",
             vote_threshold)

    z_flags  = detect_zscore(cost_df, threshold=zscore_threshold)
    if_flags = detect_isolation_forest(cost_df, telemetry_pivot, if_contamination)
    p_flags, forecast = detect_prophet(cost_df, prophet_ci)

    results = cost_df.copy()
    results["flag_zscore"]    = z_flags.values
    results["flag_iforest"]   = if_flags.values
    results["flag_prophet"]   = p_flags.values
    results["vote_count"]     = (
        results["flag_zscore"].astype(int)
        + results["flag_iforest"].astype(int)
        + results["flag_prophet"].astype(int)
    )
    results["is_anomaly"] = results["vote_count"] >= vote_threshold

    n_confirmed = results["is_anomaly"].sum()
    log.info("Ensemble confirmed %d anomalies (>= %d votes).",
             n_confirmed, vote_threshold)
    return results, forecast


# ===========================================================================
# ROOT CAUSE ANALYSIS 1: Service Contribution
# ===========================================================================

def analyze_service_contribution(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    engine=None,
    csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Break down the anomaly window cost by AWS service.
    Identifies which service contributed the most (top_service).

    Returns a dict with: services, top_service, top_service_pct, total_cost
    """
    if engine is not None:
        breakdown = load_service_breakdown(engine, window_start, window_end)
    elif csv_path is not None:
        breakdown = load_service_breakdown_from_csv(csv_path, window_start, window_end)
    else:
        return {}

    if breakdown.empty:
        return {}

    total = breakdown["window_cost"].sum()
    breakdown["pct"] = (breakdown["window_cost"] / total * 100).round(2)
    top = breakdown.iloc[0]

    return {
        "services":         breakdown.to_dict("records"),
        "top_service":      top["product_code"],
        "top_service_cost": round(float(top["window_cost"]), 8),
        "top_service_pct":  round(float(top["pct"]), 2),
        "total_cost":       round(float(total), 8),
    }


# ===========================================================================
# ROOT CAUSE ANALYSIS 2: Lag-Aware Cross-Correlation
# ===========================================================================

def compute_lag_correlations(
    cost_df: pd.DataFrame,
    telemetry_pivot: pd.DataFrame,
    lag_range: int = LAG_RANGE_PERIODS,
) -> Optional[Dict[str, Any]]:
    """
    For each telemetry metric, compute Pearson correlation against cost at
    lags from -lag_range to +lag_range (in 5-min periods).

    Negative lag: metric LEADS cost (potential leading indicator / root cause).
    Positive lag: metric LAGS cost (potential effect / downstream signal).

    Returns the best (metric, lag, correlation) triple across all metrics.
    """
    if telemetry_pivot is None or telemetry_pivot.empty:
        return None

    # Align telemetry to cost timestamps (round to 5-min)
    telem = telemetry_pivot.copy()
    telem["ds"] = telem["ts"].dt.round("5min") if "ts" in telem.columns else telem["ds"]
    if "ts" in telem.columns:
        telem = telem.drop(columns=["ts"])
    telem = telem.groupby("ds").mean().reset_index()

    aligned = cost_df[["ds", "y"]].merge(telem, on="ds", how="left")
    metric_cols = [c for c in aligned.columns if c not in ("ds", "y")]

    if not metric_cols:
        return None

    best = {"metric": None, "lag": 0, "correlation": 0.0, "all_results": []}
    cost_arr = aligned["y"].values

    for metric in metric_cols:
        m_arr = aligned[metric].fillna(aligned[metric].median()).values
        if np.std(m_arr) < 1e-10:
            continue   # constant series — skip

        for lag in range(-lag_range, lag_range + 1):
            if lag < 0:
                # Metric leads cost: shift metric forward
                c = cost_arr[-lag:]
                m = m_arr[:lag] if lag != 0 else m_arr
            elif lag > 0:
                # Metric lags cost: shift metric backward
                c = cost_arr[:-lag]
                m = m_arr[lag:]
            else:
                c, m = cost_arr, m_arr

            if len(c) < 5:
                continue

            try:
                r, p = scipy_stats.pearsonr(c, m)
                if not np.isfinite(r):
                    continue
                best["all_results"].append({
                    "metric": metric,
                    "lag_periods": lag,
                    "lag_minutes": lag * 5,
                    "correlation": round(float(r), 4),
                    "p_value": round(float(p), 4),
                })
                if abs(r) > abs(best["correlation"]):
                    best["metric"]      = metric
                    best["lag"]         = lag * 5   # convert to minutes
                    best["correlation"] = round(float(r), 4)
            except Exception:
                continue

    if best["metric"] is None:
        return None

    lag_dir = "leads" if best["lag"] < 0 else ("lags" if best["lag"] > 0 else "coincides with")
    log.info(
        "[LagCorrelation] Best: %s %s cost by %d min  (r=%.3f)",
        best["metric"], lag_dir, abs(best["lag"]), best["correlation"]
    )
    return best


# ===========================================================================
# ROOT CAUSE ANALYSIS 3: Granger Causality
# ===========================================================================

def run_granger_causality(
    cost_df: pd.DataFrame,
    telemetry_pivot: pd.DataFrame,
    maxlag: int = GRANGER_MAXLAG,
    significance: float = GRANGER_SIGNIFICANCE,
) -> Optional[Dict[str, Any]]:
    """
    Test whether each telemetry metric Granger-causes cost.

    A metric X Granger-causes cost Y if past values of X contain
    information that helps predict Y, beyond Y's own past values alone.

    Returns the most significant (lowest p-value) Granger-causal metric.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    except ImportError:
        log.warning("[Granger] statsmodels not available — skipping Granger test.")
        return None

    if telemetry_pivot is None or telemetry_pivot.empty:
        return None

    # Align telemetry to cost
    telem = telemetry_pivot.copy()
    if "ts" in telem.columns:
        telem["ds"] = telem["ts"].dt.round("5min")
        telem = telem.drop(columns=["ts"])
    telem = telem.groupby("ds").mean().reset_index()
    aligned = cost_df[["ds", "y"]].merge(telem, on="ds", how="left")
    metric_cols = [c for c in aligned.columns if c not in ("ds", "y")]

    if not metric_cols or len(aligned) < maxlag * 4:
        log.warning("[Granger] Insufficient data for Granger test (%d rows, maxlag=%d).",
                    len(aligned), maxlag)
        return None

    results = []
    for metric in metric_cols:
        series = aligned[["y", metric]].dropna()
        if len(series) < maxlag * 4 or series[metric].std() < 1e-10:
            continue

        # Difference to enforce stationarity (simple approach)
        series_diff = series.diff().dropna()
        if len(series_diff) < maxlag * 3:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_res = grangercausalitytests(
                    series_diff[["y", metric]].values,
                    maxlag=maxlag,
                    verbose=False,
                )
            # Take the minimum p-value across all lags
            min_p = min(
                gc_res[lag][0]["ssr_ftest"][1]
                for lag in range(1, maxlag + 1)
            )
            best_lag = min(
                range(1, maxlag + 1),
                key=lambda lag: gc_res[lag][0]["ssr_ftest"][1],
            )
            results.append({
                "metric":   metric,
                "p_value":  round(float(min_p), 4),
                "best_lag": best_lag,
                "significant": min_p < significance,
            })
        except Exception as exc:
            log.debug("[Granger] Error for metric %s: %s", metric, exc)
            continue

    if not results:
        return None

    results.sort(key=lambda x: x["p_value"])
    best = results[0]

    if best["significant"]:
        log.info(
            "[Granger] %s Granger-causes cost (p=%.4f, lag=%d)",
            best["metric"], best["p_value"], best["best_lag"]
        )
    else:
        log.info("[Granger] No metric significantly Granger-causes cost "
                 "(best p=%.4f for %s).", best["p_value"], best["metric"])

    return {"all": results, "best": best}


# ===========================================================================
# SEVERITY SCORING
# ===========================================================================

def compute_severity(
    vote_count: int,
    cost_delta: float,
    cost_upper_bound: float,
    correlation: Optional[float] = None,
    n_methods: int = 3,
) -> float:
    """
    Composite severity score in [0, 1].

    score = 0.40 * vote_fraction
          + 0.45 * relative_cost_excess (capped at 1)
          + 0.15 * |correlation| (if available)
    """
    vote_component   = vote_count / n_methods
    excess_frac      = min((cost_delta / max(cost_upper_bound, 1e-12)), 1.0)
    excess_component = max(0.0, excess_frac)
    corr_component   = abs(correlation) if correlation is not None else 0.0

    score = 0.40 * vote_component + 0.45 * excess_component + 0.15 * corr_component
    return round(min(max(score, 0.0), 1.0), 4)


# ===========================================================================
# PERSIST ANOMALIES TO TIMESCALEDB
# ===========================================================================

def save_anomalies_to_db(
    engine,
    anomaly_records: List[Dict],
    run_id: str,
) -> int:
    """
    Bulk-insert detected anomalies into detected_anomalies hypertable.
    Returns number of rows inserted.
    """
    if not anomaly_records:
        return 0

    import psycopg2.extras, psycopg2
    rows = []
    for r in anomaly_records:
        gc    = r.get("granger") or {}
        best_gc = gc.get("best") or {}
        rows.append({
            "window_start":           r["window_start"].isoformat(),
            "window_end":             r["window_end"].isoformat(),
            "run_id":                 run_id,
            "actual_cost":            r["actual_cost"],
            "expected_cost":          r.get("expected_cost"),
            "cost_upper_bound":       r.get("cost_upper_bound"),
            "cost_delta":             r.get("cost_delta"),
            "pct_over_bound":         r.get("pct_over_bound"),
            "vote_count":             r["vote_count"],
            "methods_flagged":        r["methods_flagged"],
            "severity_score":         r["severity_score"],
            "top_service":            r.get("top_service"),
            "top_service_cost":       r.get("top_service_cost"),
            "top_service_pct":        r.get("top_service_pct"),
            "top_correlated_metric":  r.get("top_correlated_metric"),
            "top_lag_minutes":        r.get("top_lag_minutes"),
            "top_correlation":        r.get("top_correlation"),
            "granger_metric":         best_gc.get("metric"),
            "granger_p_value":        best_gc.get("p_value"),
            "granger_max_lag":        best_gc.get("best_lag"),
        })

    create_table_sql = """
        CREATE TABLE IF NOT EXISTS detected_anomalies (
            window_start            TIMESTAMPTZ         NOT NULL,
            window_end              TIMESTAMPTZ         NOT NULL,
            detected_at             TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
            run_id                  TEXT,
            actual_cost             DOUBLE PRECISION    NOT NULL,
            expected_cost           DOUBLE PRECISION,
            cost_upper_bound        DOUBLE PRECISION,
            cost_delta              DOUBLE PRECISION,
            pct_over_bound          DOUBLE PRECISION,
            vote_count              SMALLINT            NOT NULL DEFAULT 0,
            methods_flagged         TEXT[]              NOT NULL DEFAULT '{}',
            severity_score          DOUBLE PRECISION,
            top_service             TEXT,
            top_service_cost        DOUBLE PRECISION,
            top_service_pct         DOUBLE PRECISION,
            top_correlated_metric   TEXT,
            top_lag_minutes         INT,
            top_correlation         DOUBLE PRECISION,
            granger_metric          TEXT,
            granger_p_value         DOUBLE PRECISION,
            granger_max_lag         INT,
            is_false_positive       BOOLEAN             DEFAULT FALSE,
            reviewed_by             TEXT,
            notes                   TEXT,
            CONSTRAINT uq_anomaly_window UNIQUE (window_start, run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_anomaly_severity ON detected_anomalies (severity_score DESC, window_start DESC);
        CREATE INDEX IF NOT EXISTS idx_anomaly_service ON detected_anomalies (top_service, window_start DESC);
    """

    insert_sql = """
        INSERT INTO detected_anomalies (
            window_start, window_end, run_id,
            actual_cost, expected_cost, cost_upper_bound,
            cost_delta, pct_over_bound,
            vote_count, methods_flagged, severity_score,
            top_service, top_service_cost, top_service_pct,
            top_correlated_metric, top_lag_minutes, top_correlation,
            granger_metric, granger_p_value, granger_max_lag
        ) VALUES (
            %(window_start)s, %(window_end)s, %(run_id)s,
            %(actual_cost)s, %(expected_cost)s, %(cost_upper_bound)s,
            %(cost_delta)s, %(pct_over_bound)s,
            %(vote_count)s, %(methods_flagged)s, %(severity_score)s,
            %(top_service)s, %(top_service_cost)s, %(top_service_pct)s,
            %(top_correlated_metric)s, %(top_lag_minutes)s, %(top_correlation)s,
            %(granger_metric)s, %(granger_p_value)s, %(granger_max_lag)s
        )
        ON CONFLICT (window_start, run_id) DO NOTHING
    """

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # Auto-create table if not exists
            cur.execute(create_table_sql)
            try:
                cur.execute("SELECT create_hypertable('detected_anomalies', 'window_start', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);")
            except Exception:
                pass
            raw_conn.commit()

            # Bulk insert anomaly records
            psycopg2.extras.execute_batch(cur, insert_sql, rows)
        raw_conn.commit()
        log.info("Persisted %d anomaly records to detected_anomalies.", len(rows))
        return len(rows)
    except Exception as exc:
        raw_conn.rollback()
        log.warning("Could not persist anomalies to DB: %s", exc)
        return 0
    finally:
        raw_conn.close()


# ===========================================================================
# CONSOLE REPORT
# ===========================================================================

_SEP  = "=" * 72
_SSEP = "-" * 72

_SERVICE_LABELS = {
    "AmazonApiGateway": "API Gateway",
    "AmazonEC2":        "EC2 Data Transfer",
    "AmazonDynamoDB":   "DynamoDB",
    "AmazonCloudWatch": "CloudWatch Logs",
    "AmazonS3":         "S3 PUT Requests",
}

_METRIC_LABELS = {
    "cpu_usage_rate_1m":     ("CPU Usage Rate",      lambda v: f"{v*100:.2f}%"),
    "memory_usage_pct":      ("Memory Usage",        lambda v: f"{v*100:.2f}%"),
    "memory_used_bytes":     ("Memory Used",         lambda v: f"{v/1e9:.3f} GB"),
    "http_requests_rate_1m": ("HTTP Request Rate",   lambda v: f"{v:.4f} rps"),
    "http_error_rate_1m":    ("HTTP Error Rate",     lambda v: f"{v:.6f} rps"),
    "active_requests":       ("Active Requests",     lambda v: f"{v:.2f}"),
}


def _fmt_method_flags(flags: List[str]) -> str:
    names = {"zscore": "Z-score", "iforest": "IsolationForest", "prophet": "Prophet"}
    return ", ".join(names.get(f, f) for f in flags)


def print_enhanced_report(anomaly_records: List[Dict], total_cost: float,
                           n_total: int) -> None:
    n = len(anomaly_records)
    anomaly_cost = sum(r["actual_cost"] for r in anomaly_records)

    print()
    print(_SEP)
    print("  CRIP Phase 3 v2 -- Enhanced Cost Anomaly Detection Report")
    print(_SEP)
    print(f"  Total 5-min intervals       : {n_total}")
    print(f"  Confirmed anomalies         : {n}  ({100*n/max(n_total,1):.1f}%)")
    print(f"  Vote threshold              : {VOTE_THRESHOLD}/3 detectors must agree")
    print(f"  Total 24-h cost             : USD {total_cost:.6f}")
    print(f"  Cost in anomaly windows     : USD {anomaly_cost:.6f}"
          f"  ({100*anomaly_cost/max(total_cost,1e-12):.1f}% of total)")
    print(_SSEP)

    if n == 0:
        print("  No anomalies confirmed. All 3 detectors agree cost is within normal range.")
        print(_SEP)
        return

    for i, rec in enumerate(anomaly_records, 1):
        sev_bar = "#" * int(rec["severity_score"] * 20)
        sev_bar_empty = "." * (20 - len(sev_bar))

        print(f"\n  ANOMALY #{i:02d}  |  Severity: [{sev_bar}{sev_bar_empty}] {rec['severity_score']:.3f}")
        print(f"  Window   : {rec['window_start'].strftime('%Y-%m-%d %H:%M')} to "
              f"{rec['window_end'].strftime('%H:%M')} UTC")
        print(f"  Detectors: {_fmt_method_flags(rec['methods_flagged'])}  "
              f"({rec['vote_count']}/3 votes)")
        print()
        print(f"  {'Actual cost':<30} USD {rec['actual_cost']:.8f}")
        if rec.get("expected_cost"):
            print(f"  {'Expected (Prophet yhat)':<30} USD {rec['expected_cost']:.8f}")
            print(f"  {'Upper CI bound':<30} USD {rec['cost_upper_bound']:.8f}")
            print(f"  {'Excess above upper bound':<30} USD {rec['cost_delta']:.8f}"
                  f"  (+{rec['pct_over_bound']:.1f}%)")

        # Service contribution
        if rec.get("top_service"):
            label = _SERVICE_LABELS.get(rec["top_service"], rec["top_service"])
            print()
            print(f"  Root Cause — Service Contribution:")
            print(f"    Top service  : {label}  ({rec['top_service_pct']:.1f}% of window cost)")
            if rec.get("services"):
                for svc in rec["services"][:4]:
                    svc_label = _SERVICE_LABELS.get(svc["product_code"], svc["product_code"])
                    bar = "#" * int(svc["pct"] / 5)
                    print(f"    {svc_label:<25} {svc['pct']:5.1f}%  {bar}")

        # Lag correlation
        if rec.get("top_correlated_metric"):
            lag_min = rec.get("top_lag_minutes", 0)
            lag_dir = ("LEADS cost by" if lag_min < 0
                       else ("LAGS cost by" if lag_min > 0 else "coincides with cost at"))
            m_label = _METRIC_LABELS.get(rec["top_correlated_metric"],
                                          (rec["top_correlated_metric"], None))[0]
            print()
            print(f"  Root Cause -- Lag-Aware Correlation:")
            print(f"    {m_label} {lag_dir} {abs(lag_min)} min"
                  f"  (r={rec['top_correlation']:.3f})")

        # Granger causality
        gc = rec.get("granger") or {}
        best_gc = gc.get("best") or {}
        if best_gc.get("significant"):
            gc_label = _METRIC_LABELS.get(best_gc["metric"],
                                           (best_gc["metric"], None))[0]
            print()
            print(f"  Root Cause -- Granger Causality (p < {GRANGER_SIGNIFICANCE}):")
            print(f"    {gc_label} Granger-causes cost"
                  f"  (p={best_gc['p_value']:.4f}, lag={best_gc['best_lag']} x 5min)")

        print(f"\n  {_SSEP}")

    print()
    print(_SEP)
    print("  End of Report")
    print(_SEP)
    print()


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def plot_enhanced(
    results: pd.DataFrame,
    forecast: pd.DataFrame,
    anomaly_records: List[Dict],
    lag_result: Optional[Dict],
    telemetry_pivot: Optional[pd.DataFrame],
    output_path: Path,
) -> None:
    """
    3-panel diagnostic figure:
    Panel 1 (large) : Forecast ribbon + actuals + colour-coded anomaly markers
    Panel 2 (medium): Per-service cost stacked bars at anomaly timestamps
    Panel 3 (small) : Top lag-correlated metric overlay with phase indicator
    """
    log.info("Generating enhanced plot -> %s ...", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anomaly_df = results[results["is_anomaly"]].copy()
    has_forecast  = not forecast.empty
    has_telemetry = (telemetry_pivot is not None and not telemetry_pivot.empty)
    has_service   = any(r.get("services") for r in anomaly_records)
    n_panels = 1 + int(has_service) + int(has_telemetry and lag_result is not None)

    fig = plt.figure(figsize=(16, 5 * n_panels))
    gs  = gridspec.GridSpec(
        n_panels, 1, figure=fig,
        height_ratios=[3] + [2] * (n_panels - 1),
        hspace=0.35,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]
    ax1  = axes[0]

    # --- Panel 1: Forecast + Actuals + Anomalies ---
    if has_forecast:
        fc = forecast.merge(results[["ds", "vote_count", "is_anomaly"]], on="ds", how="left")
        ax1.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                          alpha=0.2, color="#2196F3", label="95% CI (Prophet)")
        ax1.plot(fc["ds"], fc["yhat"], "--", color="#1565C0",
                  linewidth=1.2, label="Prophet yhat", zorder=2)
        ax1.plot(fc["ds"], fc["yhat_upper"], "-.", color="#F57C00",
                  linewidth=0.8, alpha=0.6, label="CI upper bound", zorder=2)

    ax1.plot(results["ds"], results["y"], color="#37474F",
              linewidth=0.9, alpha=0.85, label="Actual cost", zorder=3)

    # Colour anomalies by vote count
    vote_colors = {1: "#FFA726", 2: "#EF5350", 3: "#B71C1C"}
    for votes, color in vote_colors.items():
        mask = anomaly_df["vote_count"] == votes
        sub  = anomaly_df[mask]
        if not sub.empty:
            ax1.scatter(sub["ds"], sub["y"], color=color, s=70 + votes * 15,
                        zorder=5, edgecolors="black", linewidths=0.6,
                        label=f"Anomaly ({votes}/3 votes)")

    # Vertical guides
    for _, row in anomaly_df.iterrows():
        ax1.axvline(row["ds"], color="#FFCDD2", linewidth=0.4,
                     linestyle=":", alpha=0.7, zorder=0)

    ax1.set_title("CRIP Phase 3 v2 — Ensemble Cost Anomaly Detection\n"
                   f"{results['ds'].min().strftime('%Y-%m-%d %H:%M')} to "
                   f"{results['ds'].max().strftime('%H:%M')} UTC",
                   fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Total 5-min Unblended Cost (USD)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    if not anomaly_df.empty:
        ax1.annotate(f"{len(anomaly_df)} anomalies (>={VOTE_THRESHOLD} votes)",
                      xy=(0.98, 0.96), xycoords="axes fraction",
                      fontsize=8, color="#B71C1C", ha="right", va="top",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                ec="#B71C1C", alpha=0.85))

    panel_idx = 1

    # --- Panel 2: Service Contribution (bar chart at anomaly windows) ---
    if has_service and panel_idx < len(axes):
        ax2 = axes[panel_idx]
        panel_idx += 1

        all_services = list(_SERVICE_LABELS.keys())
        svc_palette  = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#D32F2F"]
        svc_colors   = dict(zip(all_services, svc_palette))

        timestamps = [r["window_start"] for r in anomaly_records if r.get("services")]
        bottoms    = np.zeros(len(timestamps))
        for svc in all_services:
            heights = []
            for rec in anomaly_records:
                if not rec.get("services"):
                    heights.append(0)
                    continue
                row = next((s for s in rec["services"]
                             if s["product_code"] == svc), None)
                heights.append(row["window_cost"] if row else 0)

            if any(h > 0 for h in heights):
                ax2.bar(timestamps, heights, bottom=bottoms,
                         color=svc_colors.get(svc, "grey"),
                         label=_SERVICE_LABELS.get(svc, svc), width=pd.Timedelta("4min"))
                bottoms += np.array(heights)

        ax2.set_title("Service Cost Contribution at Anomaly Windows", fontsize=10, pad=6)
        ax2.set_ylabel("Window Cost (USD)", fontsize=9)
        ax2.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax2.grid(axis="y", alpha=0.25)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    # --- Panel 3: Top Lag-Correlated Metric ---
    if has_telemetry and lag_result is not None and panel_idx < len(axes):
        ax3 = axes[panel_idx]

        best_metric = lag_result.get("metric")
        best_lag    = lag_result.get("lag", 0)
        best_r      = lag_result.get("correlation", 0)

        telem = telemetry_pivot.copy()
        if "ts" in telem.columns:
            telem = telem.rename(columns={"ts": "ds"})
        telem["ds"] = pd.to_datetime(telem["ds"]).dt.round("5min")

        if best_metric and best_metric in telem.columns:
            m_vals = (
                telem.groupby("ds")[best_metric]
                .mean()
                .reindex(results["ds"])
                .interpolate()
                .bfill()
                .ffill()
            )
            norm    = (m_vals - m_vals.min()) / (m_vals.max() - m_vals.min() + 1e-12)
            norm_y  = (results["y"] - results["y"].min()) / (results["y"].max() - results["y"].min())

            ax3.plot(results["ds"], norm_y, color="#37474F",
                      linewidth=0.9, alpha=0.7, label="Cost (normalised)")
            ax3.plot(results["ds"], norm, color="#D32F2F",
                      linewidth=1.1, alpha=0.85,
                      label=f"{_METRIC_LABELS.get(best_metric, (best_metric,None))[0]} (normalised)")

            m_label = _METRIC_LABELS.get(best_metric, (best_metric,))[0]
            lag_dir = "leads" if best_lag < 0 else ("lags" if best_lag > 0 else "aligns with")
            ax3.set_title(
                f"Top Lag-Correlated Metric: {m_label} {lag_dir} cost by "
                f"{abs(best_lag)} min  (r={best_r:.3f})",
                fontsize=10, pad=6,
            )
            ax3.legend(loc="upper left", fontsize=7, framealpha=0.9)
            ax3.grid(alpha=0.2)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    axes[-1].set_xlabel("Time (UTC)", fontsize=9)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved: %s  (%d KB)", output_path,
              output_path.stat().st_size // 1024)


# ===========================================================================
# FULL PIPELINE ORCHESTRATOR
# ===========================================================================

def run_pipeline(args) -> None:
    run_id   = str(uuid.uuid4())[:8]
    csv_path = Path(args.from_csv) if args.from_csv else None
    engine   = None

    # ----- Connect to DB (if not in offline mode) -----
    if not csv_path:
        log.info("Connecting to TimescaleDB: %s", args.db_url)
        engine = create_engine(args.db_url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:
            log.error("Cannot connect to DB: %s", exc)
            log.error("Tip: use --from-csv reports/synthetic_cur.csv for offline mode.")
            sys.exit(1)

    # ----- Load cost series -----
    cost_df = (load_cost_from_csv(csv_path) if csv_path
               else load_cost_from_db(engine, args.interval))

    if len(cost_df) < 20:
        log.error("Too few data points (%d). Need >= 20.", len(cost_df))
        sys.exit(1)

    interval_minutes = int(args.interval.split()[0]) if " " in args.interval else 5

    # ----- Load telemetry pivot (for IF features + RCA) -----
    telemetry_pivot = (load_telemetry_pivot(engine)
                       if engine is not None else None)

    # ----- Run ensemble detection -----
    results, forecast = run_ensemble(
        cost_df, telemetry_pivot,
        zscore_threshold=args.zscore_threshold,
        if_contamination=args.if_contamination,
        prophet_ci=args.ci,
        vote_threshold=args.vote_threshold,
    )

    anomaly_rows = results[results["is_anomaly"]].copy()
    log.info("Building RCA for %d confirmed anomalies...", len(anomaly_rows))

    # ----- Lag correlation (global, across full series) -----
    lag_result = compute_lag_correlations(cost_df, telemetry_pivot)

    # ----- Granger causality (global, across full series) -----
    granger_result = run_granger_causality(cost_df, telemetry_pivot)

    # ----- Build per-anomaly records -----
    anomaly_records = []
    for _, row in anomaly_rows.iterrows():
        ws = row["ds"]
        we = ws + pd.Timedelta(minutes=interval_minutes)

        # Prophet values for this row (if forecast available)
        fc_row = (forecast[forecast["ds"] == ws].iloc[0]
                  if not forecast.empty and len(forecast[forecast["ds"] == ws]) > 0
                  else None)

        cost_delta = float(row["y"] - fc_row["yhat_upper"]) if fc_row is not None else None
        pct_over   = (cost_delta / fc_row["yhat_upper"] * 100
                       if cost_delta and fc_row is not None else None)

        # Service contribution
        svc = analyze_service_contribution(ws, we, engine=engine, csv_path=csv_path)

        methods = []
        if row["flag_zscore"]:  methods.append("zscore")
        if row["flag_iforest"]: methods.append("iforest")
        if row["flag_prophet"]: methods.append("prophet")

        severity = compute_severity(
            vote_count      = int(row["vote_count"]),
            cost_delta      = cost_delta or 0,
            cost_upper_bound= float(fc_row["yhat_upper"]) if fc_row is not None else float(row["y"]),
            correlation     = lag_result.get("correlation") if lag_result else None,
        )

        anomaly_records.append({
            "window_start":          ws,
            "window_end":            we,
            "actual_cost":           float(row["y"]),
            "expected_cost":         float(fc_row["yhat"]) if fc_row is not None else None,
            "cost_upper_bound":      float(fc_row["yhat_upper"]) if fc_row is not None else None,
            "cost_delta":            cost_delta,
            "pct_over_bound":        pct_over,
            "vote_count":            int(row["vote_count"]),
            "methods_flagged":       methods,
            "severity_score":        severity,
            "top_service":           svc.get("top_service"),
            "top_service_cost":      svc.get("top_service_cost"),
            "top_service_pct":       svc.get("top_service_pct"),
            "services":              svc.get("services", []),
            "top_correlated_metric": lag_result.get("metric") if lag_result else None,
            "top_lag_minutes":       lag_result.get("lag") if lag_result else None,
            "top_correlation":       lag_result.get("correlation") if lag_result else None,
            "granger":               granger_result,
        })

    # Sort by severity desc
    anomaly_records.sort(key=lambda x: -x["severity_score"])

    # ----- Print report -----
    print_enhanced_report(
        anomaly_records,
        total_cost=float(results["y"].sum()),
        n_total=len(results),
    )

    # ----- Dispatch Remediation Webhooks -----
    if args.webhook_url and anomaly_records:
        import requests
        log.info("Dispatching remediation webhooks to %s ...", args.webhook_url)
        for i, rec in enumerate(anomaly_records[:args.max_webhooks], 1):
            payload = {
                "timestamp": rec["window_start"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "window_end": rec["window_end"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "actual_cost": round(float(rec["actual_cost"]), 6),
                "expected_cost": round(float(rec.get("expected_cost") or 0.0), 6),
                "cost_delta": round(float(rec.get("cost_delta") or 0.0), 6),
                "pct_over": round(float(rec.get("pct_over_bound") or 0.0), 2),
                "top_service": rec.get("top_service", "AmazonApiGateway"),
                "severity_score": rec.get("severity_score"),
                "methods_flagged": rec.get("methods_flagged"),
            }
            try:
                r = requests.post(args.webhook_url, json=payload, timeout=5.0)
                if r.status_code == 200:
                    log.info("  [Remediation Webhook %02d] OK (200) -> %s", i, r.text[:100])
                else:
                    log.warning("  [Remediation Webhook %02d] HTTP %d: %s", i, r.status_code, r.text[:100])
            except Exception as e:
                log.warning("  [Remediation Webhook %02d] Connection failed: %s", i, e)

    # ----- Persist to DB -----
    if engine is not None and not args.no_persist:
        save_anomalies_to_db(engine, anomaly_records, run_id)

    # ----- Plot -----
    if not args.report_only:
        plot_enhanced(
            results=results,
            forecast=forecast,
            anomaly_records=anomaly_records,
            lag_result=lag_result,
            telemetry_pivot=telemetry_pivot,
            output_path=Path(args.out),
        )

    if engine:
        engine.dispose()
    log.info("Pipeline complete. run_id=%s", run_id)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    default_db = os.getenv(
        "CRIP_DB_URL",
        "postgresql+psycopg2://crip_user:crip_secret@localhost:5432/crip",
    )
    p = argparse.ArgumentParser(
        description="CRIP Phase 3 v2 — Ensemble anomaly detection with RCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db-url",          default=default_db)
    p.add_argument("--from-csv",        default=None,
                    help="Offline mode: path to synthetic_cur.csv.")
    p.add_argument("--interval",        default="5 minutes",
                    choices=["1 minutes","5 minutes","15 minutes","30 minutes"])
    p.add_argument("--ci",              type=float, default=PROPHET_CI)
    p.add_argument("--zscore-threshold",type=float, default=ZSCORE_THRESHOLD)
    p.add_argument("--if-contamination",type=float, default=IF_CONTAMINATION)
    p.add_argument("--vote-threshold",  type=int,   default=VOTE_THRESHOLD,
                    help="Minimum detector votes (1-3) to confirm anomaly.")
    p.add_argument("--out",             default="reports/forecast_plot_v2.png")
    p.add_argument("--webhook-url",      default=os.getenv("REMEDIATION_WEBHOOK_URL"),
                    help="LocalStack Lambda Function URL to send dry-run remediation webhooks.")
    p.add_argument("--max-webhooks",    type=int, default=5,
                    help="Max number of anomalies to dispatch to the webhook.")
    p.add_argument("--report-only",     action="store_true")
    p.add_argument("--no-persist",      action="store_true",
                    help="Skip writing anomalies to detected_anomalies table.")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(_parse_args())

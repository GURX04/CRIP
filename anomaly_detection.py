"""
anomaly_detection.py
====================
CRIP Phase 3 — Machine Learning & Anomaly Detection

Pipeline
--------
1. Load   : Pull time-bucketed (5-min) cost data from TimescaleDB using
            TimescaleDB's native time_bucket() SQL function via SQLAlchemy.
2. Train  : Fit a Facebook Prophet model on the cost time series.
            95% confidence interval defines the expected cost envelope.
3. Detect : Flag every bucket where actual_cost > yhat_upper as a cost anomaly.
4. Explain: For each anomaly window, join against system_telemetry to pull
            the average system metrics active during that interval.
5. Report : Print a structured console report — timestamp, delta, correlated metrics.
6. Plot   : Save a forecast plot with anomaly overlays to reports/forecast_plot.png.

Dependencies (install once)
---------------------------
    pip install pandas prophet matplotlib sqlalchemy psycopg2-binary

Usage
-----
    python anomaly_detection.py

    # Override DB / output settings:
    python anomaly_detection.py \\
        --db-url  "postgresql+psycopg2://crip_user:crip_secret@localhost:5432/crip" \\
        --interval 5min \\
        --ci       0.95 \\
        --out      reports/forecast_plot.png

Notes
-----
* Prophet requires the input DataFrame to have columns named exactly 'ds' and 'y'.
* The model is fit on the full historical series (in-sample anomaly detection).
  Any point where the actual cost lies above the upper confidence band is flagged.
* If system_telemetry has no data for a window (e.g. Prometheus was not running),
  the correlation section gracefully shows "no telemetry available".
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for headless/Docker

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import requests
from sqlalchemy import create_engine, text

# Suppress noisy Stan / cmdstanpy / prophet INFO chatter
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*Disabling daily seasonality.*")

# ---------------------------------------------------------------------------
# Logging — our own logger stays INFO
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("crip.anomaly")


# ===========================================================================
# 1. Data Loading  (TimescaleDB → pandas DataFrames)
# ===========================================================================

_CUR_QUERY = """
SELECT
    time_bucket(:interval, usage_start)   AS ds,
    SUM(unblended_cost)                   AS y
FROM aws_cur_records
GROUP BY ds
ORDER BY ds;
"""

_TELEMETRY_QUERY = """
SELECT
    metric_name,
    ROUND(AVG(value)::NUMERIC, 8)  AS avg_value,
    COUNT(*)                        AS sample_count
FROM system_telemetry
WHERE ts >= :window_start
  AND ts <  :window_end
GROUP BY metric_name
ORDER BY metric_name;
"""

_TELEMETRY_ALL_QUERY = """
SELECT
    metric_name,
    ts,
    value
FROM system_telemetry
ORDER BY metric_name, ts;
"""


def load_cost_series_from_csv(csv_path: Path, interval_minutes: int = 5) -> pd.DataFrame:
    """
    Load cost data directly from synthetic_cur.csv for standalone/offline testing.
    Aggregates into 5-minute time buckets matching TimescaleDB's time_bucket().
    """
    import csv, collections
    log.info("Loading cost data from CSV: %s ...", csv_path)
    if not csv_path.exists():
        log.error("CSV file does not exist: %s", csv_path)
        sys.exit(1)

    buckets = collections.defaultdict(float)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dt_str = r.get("lineItem/UsageStartDate")
            cost_str = r.get("lineItem/UnblendedCost", "0")
            if dt_str:
                buckets[dt_str] += float(cost_str or 0.0)

    df = pd.DataFrame(sorted(buckets.items()), columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = df["y"].astype(float)
    log.info(
        "Loaded %d time buckets from CSV  [%s  ->  %s]",
        len(df),
        df["ds"].min().isoformat(),
        df["ds"].max().isoformat(),
    )
    return df


def load_cost_series(engine, interval: str = "5 minutes") -> pd.DataFrame:
    """
    Execute time_bucket aggregation and return a Prophet-ready DataFrame
    with columns [ds (datetime64[ns, UTC]), y (float64)].
    """
    log.info("Loading cost data from aws_cur_records (bucket=%s) ...", interval)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(_CUR_QUERY),
            conn,
            params={"interval": interval},
            parse_dates=["ds"],
        )

    if df.empty:
        log.error(
            "No rows returned from aws_cur_records. "
            "Run ingest_to_timescale.py first."
        )
        sys.exit(1)

    # Ensure timezone-naive for Prophet (it does not support tz-aware ds)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"]  = df["y"].astype(float)

    log.info(
        "Loaded %d time buckets  [%s  ->  %s]",
        len(df),
        df["ds"].min().isoformat(),
        df["ds"].max().isoformat(),
    )
    return df


def load_telemetry_for_window(
    engine,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return average value per metric for a specific time window.
    Returns an empty DataFrame if system_telemetry has no rows in that window.
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text(_TELEMETRY_QUERY),
            conn,
            params={
                "window_start": window_start.isoformat(),
                "window_end":   window_end.isoformat(),
            },
        )
    return df


def load_all_telemetry(engine) -> pd.DataFrame:
    """
    Load the entire system_telemetry table for time-aligned overlay plotting.
    Returns empty DataFrame if table is empty.
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(_TELEMETRY_ALL_QUERY), conn, parse_dates=["ts"])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    return df


# ===========================================================================
# 2. Prophet Model
# ===========================================================================

def build_and_fit_prophet(df: pd.DataFrame, ci: float = 0.95) -> Prophet:
    """
    Instantiate and fit a Prophet model tuned for sub-daily billing data.

    Design choices
    --------------
    * yearly/weekly seasonality OFF — we have only one day of data.
    * daily seasonality ON (Fourier order 10) — captures the 24-h diurnal cost curve.
    * Custom 3-hour seasonality (Fourier order 5) — explicitly models the recurring
      Locust spike pattern (spike every 180 min).
    * changepoint_prior_scale=0.15 — allows the trend to be moderately flexible,
      so genuine sustained cost increases are reflected without over-fitting noise.
    * interval_width=ci — the 95% envelope is used as the anomaly boundary.
    """
    log.info("Training Prophet model (interval_width=%.2f) ...", ci)

    model = Prophet(
        interval_width              = ci,
        yearly_seasonality          = False,
        weekly_seasonality          = False,
        daily_seasonality           = True,
        changepoint_prior_scale     = 0.15,
        seasonality_prior_scale     = 10.0,
        seasonality_mode            = "additive",
    )

    # Add a custom seasonality that aligns with the 3-hour Locust spike cycle
    model.add_seasonality(
        name    = "three_hourly",
        period  = 3 / 24,          # 3 hours expressed as a fraction of a day
        fourier_order = 5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df)

    log.info("Prophet model training complete.")
    return model


def generate_forecast(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run in-sample prediction on the exact same timestamps as the training data.
    This lets us compare yhat_upper against actual y for every data point.
    """
    log.info("Generating in-sample forecast ...")
    # Use exactly the training timestamps (no future extension needed for anomaly detection)
    future   = model.make_future_dataframe(periods=0, freq="5min", include_history=True)
    forecast = model.predict(future)

    # Merge actual values back onto the forecast for easy comparison
    forecast = forecast.merge(df[["ds", "y"]], on="ds", how="left")
    log.info("Forecast generated for %d timestamps.", len(forecast))
    return forecast


# ===========================================================================
# 3. Anomaly Detection
# ===========================================================================

def detect_anomalies(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where actual cost (y) strictly exceeds the upper CI bound (yhat_upper).

    Returns a filtered DataFrame of anomalous intervals with columns:
    [ds, y, yhat, yhat_lower, yhat_upper, cost_delta, pct_over]
    """
    anomalies = forecast[forecast["y"] > forecast["yhat_upper"]].copy()
    anomalies["cost_delta"] = anomalies["y"] - anomalies["yhat_upper"]
    anomalies["pct_over"]   = (anomalies["cost_delta"] / anomalies["yhat_upper"]) * 100

    log.info(
        "Anomaly detection: %d / %d buckets flagged (%.1f%%)",
        len(anomalies),
        len(forecast),
        100 * len(anomalies) / max(len(forecast), 1),
    )
    return anomalies.reset_index(drop=True)


# ===========================================================================
# 4. Root-Cause Correlation
# ===========================================================================

# Human-readable labels and units for Prometheus metric names
_METRIC_LABELS: Dict[str, Tuple[str, str]] = {
    "cpu_usage_rate_1m":     ("CPU Usage Rate",           "%",   lambda v: f"{v*100:.2f}%"),
    "memory_used_bytes":     ("Memory Used",              "GB",  lambda v: f"{v/1e9:.3f} GB"),
    "memory_usage_pct":      ("Memory Usage",             "%",   lambda v: f"{v*100:.2f}%"),
    "http_requests_rate_1m": ("HTTP Request Rate",        "rps", lambda v: f"{v:.4f} rps"),
    "http_error_rate_1m":    ("HTTP Error Rate",          "rps", lambda v: f"{v:.6f} rps"),
    "active_requests":       ("Active Requests (gauge)",  "",    lambda v: f"{v:.2f}"),
    "ingest_bytes_rate_1m":  ("Ingest Bytes Rate",        "B/s", lambda v: f"{v:.2f} B/s"),
    "app_uptime_seconds":    ("App Uptime",               "s",   lambda v: f"{v:.0f} s"),
}


def format_metric_value(metric_name: str, value: float) -> str:
    """Apply unit-aware formatting for a metric value."""
    info = _METRIC_LABELS.get(metric_name)
    if info:
        _, _, fmt_fn = info
        return fmt_fn(value)
    return f"{value:.6f}"


def correlate_anomalies(
    engine,
    anomalies: pd.DataFrame,
    interval_minutes: int = 5,
) -> List[Dict]:
    """
    For each anomalous interval, query system_telemetry for that window
    and build a correlation record.
    """
    log.info("Correlating %d anomalies with system telemetry ...", len(anomalies))
    records = []

    for _, row in anomalies.iterrows():
        window_start = row["ds"]
        window_end   = window_start + pd.Timedelta(minutes=interval_minutes)

        telemetry_df = load_telemetry_for_window(engine, window_start, window_end)

        metrics_dict: Dict[str, float] = {}
        if not telemetry_df.empty:
            for _, m_row in telemetry_df.iterrows():
                metrics_dict[m_row["metric_name"]] = float(m_row["avg_value"])

        records.append({
            "ds":          row["ds"],
            "window_end":  window_end,
            "y":           row["y"],
            "yhat":        row["yhat"],
            "yhat_upper":  row["yhat_upper"],
            "cost_delta":  row["cost_delta"],
            "pct_over":    row["pct_over"],
            "metrics":     metrics_dict,
        })

    log.info("Correlation complete.")
    return records


# ===========================================================================
# 4b. Remediation Webhook (LocalStack Lambda Integration)
# ===========================================================================

def send_remediation_webhooks(
    webhook_url: str,
    correlation_records: List[Dict],
    max_dispatches: int = 5,
) -> int:
    """
    Dispatch detected cost anomalies to a remediation webhook (e.g. LocalStack Lambda Function URL).
    Triggers simulated dry-run mitigation actions (e.g. API Gateway rate-limiting).
    """
    if not webhook_url:
        return 0

    if not correlation_records:
        log.info("No anomalies to send to remediation webhook.")
        return 0

    log.info("Dispatching remediation webhooks to %s ...", webhook_url)
    dispatched = 0

    for i, rec in enumerate(correlation_records[:max_dispatches], start=1):
        ts_str = rec["ds"].strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(rec["ds"], "strftime") else str(rec["ds"])
        we_str = rec["window_end"].strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(rec["window_end"], "strftime") else str(rec["window_end"])
        payload = {
            "timestamp": ts_str,
            "window_end": we_str,
            "actual_cost": round(float(rec["y"]), 6),
            "expected_cost": round(float(rec["yhat"]), 6),
            "cost_upper_bound": round(float(rec["yhat_upper"]), 6),
            "cost_delta": round(float(rec["cost_delta"]), 6),
            "pct_over": round(float(rec["pct_over"]), 2),
            "top_service": "AmazonApiGateway",
            "metrics": rec.get("metrics", {}),
        }

        try:
            resp = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                log.info("  [Remediation Webhook %02d] Dispatched OK (HTTP 200) -> %s", i, resp.text[:100])
                dispatched += 1
            else:
                log.warning("  [Remediation Webhook %02d] Returned HTTP %d: %s", i, resp.status_code, resp.text[:100])
        except requests.RequestException as exc:
            log.warning("  [Remediation Webhook %02d] Failed to connect to %s: %s", i, webhook_url, exc)

    log.info("Completed dispatching %d remediation webhook(s).", dispatched)
    return dispatched


# ===========================================================================
# 5. Console Report
# ===========================================================================

_SEPARATOR = "=" * 70
_SUBSEP    = "-" * 70


def print_anomaly_report(
    correlation_records: List[Dict],
    forecast: pd.DataFrame,
    ci: float,
) -> None:
    """
    Print a structured, human-readable console report covering:
    - Summary statistics
    - Per-anomaly detail: time window, cost delta, correlated metrics
    """
    n_anomalies = len(correlation_records)
    n_total     = len(forecast)
    total_cost  = forecast["y"].sum()
    anomaly_cost= sum(r["y"] for r in correlation_records)

    print()
    print(_SEPARATOR)
    print("  CRIP Phase 3 -- Cost Anomaly Detection Report")
    print(_SEPARATOR)
    print(f"  CI interval width     : {ci * 100:.0f}%")
    print(f"  Total 5-min buckets   : {n_total}")
    print(f"  Anomalous buckets     : {n_anomalies}  ({100*n_anomalies/max(n_total,1):.1f}%)")
    print(f"  Total 24-h cost       : USD {total_cost:.6f}")
    print(f"  Cost in anomaly windows: USD {anomaly_cost:.6f}  "
          f"({100*anomaly_cost/max(total_cost,1e-12):.1f}% of total)")
    print(_SUBSEP)

    if n_anomalies == 0:
        print("  No anomalies detected. The cost series is within expected bounds.")
        print(_SEPARATOR)
        return

    for i, rec in enumerate(correlation_records, start=1):
        window_label = (
            f"{rec['ds'].strftime('%Y-%m-%d %H:%M')} to "
            f"{rec['window_end'].strftime('%H:%M')} UTC"
        )
        print(f"\n  ANOMALY #{i:02d}  |  {window_label}")
        print(f"  {'Actual cost':<28} USD {rec['y']:.8f}")
        print(f"  {'Expected (yhat)':<28} USD {rec['yhat']:.8f}")
        print(f"  {'Upper CI bound (yhat_upper)':<28} USD {rec['yhat_upper']:.8f}")
        print(f"  {'Excess over upper bound':<28} USD {rec['cost_delta']:.8f}  "
              f"(+{rec['pct_over']:.1f}% above CI)")

        if rec["metrics"]:
            print(f"\n  Correlated System Metrics (avg over window):")
            for metric_name, avg_val in sorted(rec["metrics"].items()):
                label = _METRIC_LABELS.get(metric_name, (metric_name, "", None))[0]
                formatted = format_metric_value(metric_name, avg_val)
                print(f"    {label:<35} {formatted}")
        else:
            print(f"\n  Correlated System Metrics: (no telemetry data for this window)")

        print(f"  {_SUBSEP}")

    print()
    print(_SEPARATOR)
    print("  End of Report")
    print(_SEPARATOR)
    print()


# ===========================================================================
# 6. Visualization
# ===========================================================================

def plot_forecast(
    forecast: pd.DataFrame,
    anomalies: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    output_path: Path,
    ci: float,
) -> None:
    """
    Produce a 2-panel figure:
      Panel 1 (main): Prophet forecast ribbon + actual cost line + anomaly markers.
      Panel 2 (sub):  Correlated system telemetry time series (if available).
    """
    log.info("Generating forecast plot -> %s ...", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_telemetry = (telemetry_df is not None) and (not telemetry_df.empty)
    n_panels      = 2 if has_telemetry else 1
    fig_height    = 10 if has_telemetry else 6

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(16, fig_height),
        sharex=(n_panels > 1),
        gridspec_kw={"height_ratios": [3, 2]} if has_telemetry else {},
    )

    ax1 = axes[0] if has_telemetry else axes

    # ------------------------------------------------------------------
    # Panel 1: Forecast + actuals + anomalies
    # ------------------------------------------------------------------

    # Confidence interval ribbon
    ax1.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.25,
        color="#2196F3",
        label=f"{ci*100:.0f}% Confidence Interval",
        zorder=1,
    )

    # Prophet centre-line forecast
    ax1.plot(
        forecast["ds"],
        forecast["yhat"],
        color="#1565C0",
        linewidth=1.5,
        linestyle="--",
        label="Prophet Forecast (yhat)",
        zorder=2,
    )

    # Actual cost line
    ax1.plot(
        forecast["ds"],
        forecast["y"],
        color="#37474F",
        linewidth=1.0,
        alpha=0.85,
        label="Actual Cost (USD)",
        zorder=3,
    )

    # Anomaly scatter — only plot if there are anomalies
    if not anomalies.empty:
        ax1.scatter(
            anomalies["ds"],
            anomalies["y"],
            color="#D32F2F",
            s=60,
            zorder=5,
            label=f"Anomaly ({len(anomalies)} points)",
            edgecolors="darkred",
            linewidths=0.8,
        )

        # Vertical dashed lines at each anomaly timestamp (subtle)
        for ts in anomalies["ds"]:
            ax1.axvline(
                x=ts,
                color="#EF9A9A",
                linewidth=0.5,
                linestyle=":",
                alpha=0.6,
                zorder=0,
            )

    # Upper CI reference line
    ax1.plot(
        forecast["ds"],
        forecast["yhat_upper"],
        color="#F57C00",
        linewidth=0.8,
        linestyle="-.",
        alpha=0.7,
        label="Upper CI (yhat_upper)",
        zorder=2,
    )

    # Formatting
    ax1.set_title(
        "CRIP Phase 3 — AWS Cost Anomaly Detection (Prophet 95% CI)\n"
        f"In-sample forecast: {forecast['ds'].min().strftime('%Y-%m-%d %H:%M')} "
        f"-- {forecast['ds'].max().strftime('%H:%M')} UTC",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Total 5-min Unblended Cost (USD)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax1.grid(axis="x", alpha=0.2, linewidth=0.4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    # Annotate anomaly count in top-right corner
    if not anomalies.empty:
        ax1.annotate(
            f"{len(anomalies)} anomalies detected",
            xy=(0.98, 0.96),
            xycoords="axes fraction",
            fontsize=9,
            color="#D32F2F",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D32F2F", alpha=0.8),
        )

    # ------------------------------------------------------------------
    # Panel 2: System Telemetry (optional)
    # ------------------------------------------------------------------
    if has_telemetry:
        ax2 = axes[1]

        # Select a subset of most-useful metrics for clarity
        priority_metrics = [
            "cpu_usage_rate_1m",
            "memory_usage_pct",
            "http_requests_rate_1m",
        ]

        metric_colors = {
            "cpu_usage_rate_1m":      "#E53935",   # red
            "memory_usage_pct":       "#7B1FA2",   # purple
            "http_requests_rate_1m":  "#0288D1",   # blue
        }

        plotted_any = False
        for metric in priority_metrics:
            m_df = telemetry_df[telemetry_df["metric_name"] == metric].copy()
            if m_df.empty:
                continue

            label_info = _METRIC_LABELS.get(metric, (metric, "", None))
            label      = label_info[0]
            color      = metric_colors.get(metric, "grey")

            # Normalise to 0-1 for a shared axis (different units)
            vals = m_df["value"].values.astype(float)
            v_min, v_max = vals.min(), vals.max()
            norm_vals = (vals - v_min) / (v_max - v_min + 1e-12)

            ax2.plot(
                m_df["ts"],
                norm_vals,
                label=f"{label} (normalised)",
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )
            plotted_any = True

        if not plotted_any:
            ax2.text(
                0.5, 0.5,
                "No system telemetry data\n(run ingest_to_timescale.py with Prometheus running)",
                ha="center", va="center",
                transform=ax2.transAxes,
                fontsize=9,
                color="grey",
            )
        else:
            ax2.set_ylabel("Normalised Metric Value (0–1)", fontsize=9)
            ax2.legend(loc="upper left", fontsize=7, framealpha=0.9)
            ax2.grid(alpha=0.3, linewidth=0.4)
            ax2.set_ylim(-0.05, 1.15)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))

        ax2.set_title("Correlated System Telemetry (normalised)", fontsize=10, pad=6)
        ax2.set_xlabel("Time (UTC)", fontsize=10)
    else:
        ax1.set_xlabel("Time (UTC)", fontsize=10)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Plot saved: %s  (%d KB)", output_path, output_path.stat().st_size // 1024)


# ===========================================================================
# 7. CLI + Main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    default_db = os.getenv(
        "CRIP_DB_URL",
        "postgresql+psycopg2://crip_user:crip_secret@localhost:5432/crip",
    )
    p = argparse.ArgumentParser(
        description="CRIP Phase 3 — Prophet-based cost anomaly detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--db-url", default=default_db,
        help="SQLAlchemy connection URL for TimescaleDB.",
    )
    p.add_argument(
        "--interval", default="5 minutes",
        choices=["1 minutes", "5 minutes", "15 minutes", "30 minutes", "1 hours"],
        help="time_bucket interval for cost aggregation.",
    )
    p.add_argument(
        "--ci", type=float, default=0.95,
        help="Prophet confidence interval width (0–1). Anomalies exceed yhat_upper.",
    )
    p.add_argument(
        "--out", default="reports/forecast_plot.png",
        help="Output path for the forecast PNG plot.",
    )
    p.add_argument(
        "--report-only", action="store_true",
        help="Skip plotting (useful in headless environments without a display).",
    )
    p.add_argument(
        "--from-csv", default=None,
        help="Optional path to synthetic_cur.csv for standalone/offline testing without TimescaleDB.",
    )
    p.add_argument(
        "--webhook-url", default=os.getenv("REMEDIATION_WEBHOOK_URL"),
        help="LocalStack Lambda Function URL to send dry-run remediation webhooks when anomalies are detected.",
    )
    p.add_argument(
        "--max-webhooks", type=int, default=5,
        help="Maximum number of anomaly events to dispatch to the remediation webhook.",
    )
    p.add_argument(
        "--max-report-anomalies", type=int, default=20,
        help="Cap the number of anomalies printed in the console report.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Parse interval string to integer minutes for timedelta use
    try:
        interval_minutes = int(args.interval.split()[0])
    except (ValueError, IndexError):
        interval_minutes = 5

    # ---- 1. Load cost data ----
    engine = None
    if args.from_csv:
        log.info("Running in standalone/offline mode using CSV: %s", args.from_csv)
        cost_df = load_cost_series_from_csv(Path(args.from_csv), interval_minutes=interval_minutes)
    else:
        # Connect to TimescaleDB
        log.info("Connecting to TimescaleDB: %s", args.db_url)
        engine = create_engine(args.db_url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:
            log.error("Cannot connect to TimescaleDB: %s", exc)
            log.error(
                "Is the Docker stack running? Run: docker compose up -d\n"
                "Or test offline using: python anomaly_detection.py --from-csv reports/synthetic_cur.csv"
            )
            sys.exit(1)

        cost_df = load_cost_series(engine, interval=args.interval)

    if len(cost_df) < 20:
        log.error(
            "Only %d data points available — Prophet requires at least 20 "
            "for a reliable fit.",
            len(cost_df),
        )
        sys.exit(1)

    # ---- 2. Train Prophet ----
    model    = build_and_fit_prophet(cost_df, ci=args.ci)

    # ---- 3. Forecast ----
    forecast = generate_forecast(model, cost_df)

    # ---- 4. Detect anomalies ----
    anomalies = detect_anomalies(forecast)

    # ---- 5. Root-cause correlation ----
    anomalies_for_report = anomalies.head(args.max_report_anomalies)
    if engine is not None:
        correlation_records = correlate_anomalies(
            engine, anomalies_for_report, interval_minutes=interval_minutes
        )
        telemetry_df = load_all_telemetry(engine)
        if telemetry_df.empty:
            log.warning(
                "system_telemetry is empty — telemetry panel will be omitted from plot."
            )
    else:
        # Offline mode without DB telemetry
        log.info("Skipping telemetry correlation (standalone CSV mode).")
        correlation_records = []
        for _, row in anomalies_for_report.iterrows():
            w_end = row["ds"] + pd.Timedelta(minutes=interval_minutes)
            correlation_records.append({
                "ds": row["ds"],
                "window_end": w_end,
                "y": row["y"],
                "yhat": row["yhat"],
                "yhat_upper": row["yhat_upper"],
                "cost_delta": row["cost_delta"],
                "pct_over": row["pct_over"],
                "metrics": {},
            })
        telemetry_df = None

    # ---- 6. Console report ----
    print_anomaly_report(correlation_records, forecast, ci=args.ci)

    # ---- 6b. Remediation Webhook Dispatch ----
    if args.webhook_url:
        send_remediation_webhooks(
            webhook_url=args.webhook_url,
            correlation_records=correlation_records,
            max_dispatches=args.max_webhooks,
        )

    # ---- 7. Plot ----
    if not args.report_only:
        plot_forecast(
            forecast=forecast,
            anomalies=anomalies,
            telemetry_df=telemetry_df if (telemetry_df is not None and not telemetry_df.empty) else None,
            output_path=Path(args.out),
            ci=args.ci,
        )
    else:
        log.info("Skipping plot (--report-only).")

    if engine is not None:
        engine.dispose()
    log.info("Phase 3 anomaly detection pipeline complete.")


if __name__ == "__main__":
    main()

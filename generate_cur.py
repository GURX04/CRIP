"""
generate_cur.py
===============
Synthetic AWS Cost & Usage Report (CUR) Generator — Phase 1 CRIP

Overview
--------
This script generates a 24-hour synthetic AWS CUR CSV file that mathematically
mirrors the traffic pattern produced by the Locust load test defined in
locustfile.py.

Traffic Model
-------------
For each 5-minute billing interval (288 intervals over 24 h) the script
computes a *request volume* using three additive components:

  requests(t) = base_volume
              + sine_wave(t)           ← natural diurnal variance
              + spike(t)               ← Locust-style recurring bursts

  1. base_volume   – constant baseline (e.g. 3 000 req / 5 min)
  2. sine_wave(t)  – A·sin(2π·t/T) models day-night traffic rhythm
                     A = amplitude, T = 24 h period, phase shifts peak to 14:00
  3. spike(t)      – rectangular pulses injected every SPIKE_INTERVAL_MIN,
                     each lasting SPIKE_DURATION_MIN, multiplied by SPIKE_FACTOR

Cost Mapping
------------
Each service's cost is derived from its own pricing formula:

  • API Gateway (REST)   – $3.50 per million requests
  • EC2 Data Transfer    – $0.09 per GB (1 req ≈ 4 KB average payload)
  • DynamoDB             – $1.25 per million read request units
  • CloudWatch Logs      – $0.50 per GB ingested (1 req ≈ 200 B log entry)
  • S3 PUT requests      – $0.005 per 1 000 PUT requests (ingest endpoint)

CUR Header Format
-----------------
Headers are drawn from the official AWS CUR v2 column specification:
  https://docs.aws.amazon.com/cur/latest/userguide/data-dictionary.html

Usage
-----
  python generate_cur.py [--output PATH] [--date YYYY-MM-DD] [--interval-min N]

Examples
  python generate_cur.py
  python generate_cur.py --output reports/my_cur.csv --date 2024-01-15
  python generate_cur.py --interval-min 60     # hourly intervals
"""

import argparse
import csv
import math
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants — tune these to match your Locust test parameters
# ---------------------------------------------------------------------------

# --- Time resolution ---
DEFAULT_INTERVAL_MIN  = 5          # Minutes per billing row (5 min = 288 rows/day)

# --- Traffic model parameters ---
BASE_VOLUME           = 3_000      # Baseline requests per interval
SINE_AMPLITUDE        = 1_200      # ±requests from sine wave (diurnal swing)
SINE_PHASE_HOURS      = 14.0       # Hour of day at which sine peaks (14:00 = 2 PM)

# --- Spike parameters (mirrors SpikyLoadShape in locustfile.py) ---
SPIKE_INTERVAL_MIN    = 180        # One spike cycle every 3 hours
SPIKE_DURATION_MIN    = 30         # Spike lasts 30 minutes
SPIKE_FACTOR          = 4.0        # Peak = base × SPIKE_FACTOR during a spike
SPIKE_RAMP_FRAC       = 0.2        # Fraction of SPIKE_DURATION spent ramping up

# --- Noise ---
NOISE_FRACTION        = 0.03       # ±3% random noise on final volume

# --- AWS pricing (simplified, per-unit costs) ---
PRICE_APIGW_PER_M_REQ = 3.50      # USD per million API Gateway requests
PRICE_EC2_DT_PER_GB   = 0.09      # USD per GB EC2 Data Transfer Out
PRICE_DYNAMO_PER_M_RRU= 1.25      # USD per million DynamoDB read request units
PRICE_CW_LOGS_PER_GB  = 0.50      # USD per GB CloudWatch Logs ingested
PRICE_S3_PUT_PER_K    = 0.005     # USD per 1 000 S3 PUT requests

# --- Payload size assumptions ---
AVG_PAYLOAD_KB        = 4.0        # Average response payload (API Gateway → EC2 DT)
AVG_LOG_BYTES         = 200        # Average CloudWatch log bytes per request
S3_INGEST_FRACTION    = 0.20       # 20% of requests hit the /api/ingest (S3 PUT) endpoint
DYNAMO_READ_FRACTION  = 0.30       # 30% of requests are DynamoDB reads

# --- AWS account / resource identifiers (synthetic) ---
AWS_ACCOUNT_ID        = "123456789012"
AWS_REGION            = "us-east-1"
PAYER_ACCOUNT_ID      = "123456789012"
BILLING_ENTITY        = "AWS"
BILL_TYPE             = "Anniversary"
CURRENCY_CODE         = "USD"

# ---------------------------------------------------------------------------
# CUR column definitions (AWS standard v2)
# ---------------------------------------------------------------------------

CUR_COLUMNS: List[str] = [
    # Identity
    "identity/LineItemId",
    "identity/TimeInterval",
    # Bill
    "bill/InvoiceId",
    "bill/InvoicingEntity",
    "bill/BillingEntity",
    "bill/BillType",
    "bill/PayerAccountId",
    "bill/BillingPeriodStartDate",
    "bill/BillingPeriodEndDate",
    # Line item
    "lineItem/UsageAccountId",
    "lineItem/LineItemType",
    "lineItem/UsageStartDate",
    "lineItem/UsageEndDate",
    "lineItem/ProductCode",
    "lineItem/UsageType",
    "lineItem/Operation",
    "lineItem/AvailabilityZone",
    "lineItem/ResourceId",
    "lineItem/UsageAmount",
    "lineItem/NormalizationFactor",
    "lineItem/NormalizedUsageAmount",
    "lineItem/CurrencyCode",
    "lineItem/UnblendedRate",
    "lineItem/UnblendedCost",
    "lineItem/BlendedRate",
    "lineItem/BlendedCost",
    "lineItem/LineItemDescription",
    # Product
    "product/ProductName",
    "product/servicecode",
    "product/region",
    "product/location",
    "product/usagetype",
    # Pricing
    "pricing/publicOnDemandCost",
    "pricing/publicOnDemandRate",
    "pricing/term",
    "pricing/unit",
    # Reservation (empty for on-demand)
    "reservation/AmortizedUpfrontFeeForBillingPeriod",
    "reservation/AmortizedUpfrontCostForUsage",
    "reservation/RecurringFeeForUsage",
    "reservation/UnusedQuantity",
    "reservation/UnusedRecurringFee",
    # Savings Plans (empty for on-demand)
    "savingsPlan/SavingsPlanARN",
    "savingsPlan/SavingsPlanRate",
    "savingsPlan/UsedCommitment",
    "savingsPlan/SavingsPlanEffectiveCost",
    "savingsPlan/TotalCommitmentToDate",
    "savingsPlan/NetSavingsPlanEffectiveCost",
    # Resource tags
    "resourceTags/user:Project",
    "resourceTags/user:Environment",
    "resourceTags/user:Team",
]


# ---------------------------------------------------------------------------
# Traffic model helpers
# ---------------------------------------------------------------------------

def compute_requests(interval_start: datetime, interval_min: int) -> int:
    """
    Return the modelled request count for a given 5-minute interval.

    Components
    ----------
    1. Baseline constant
    2. Sine wave (diurnal pattern peaking at SINE_PHASE_HOURS)
    3. Spike pulses every SPIKE_INTERVAL_MIN with a trapezoidal envelope
    4. Gaussian noise
    """
    # Minutes elapsed since midnight of the same day
    midnight = interval_start.replace(hour=0, minute=0, second=0, microsecond=0)
    t_min = (interval_start - midnight).total_seconds() / 60.0

    # ---- Component 1: baseline ----
    base = BASE_VOLUME

    # ---- Component 2: sine wave ----
    # Peak at SINE_PHASE_HOURS, 24-hour period
    phase_min = SINE_PHASE_HOURS * 60.0
    sine = SINE_AMPLITUDE * math.sin(
        2 * math.pi * (t_min - phase_min) / (24 * 60)
    )

    # ---- Component 3: spike ----
    spike_position = t_min % SPIKE_INTERVAL_MIN          # 0 … SPIKE_INTERVAL_MIN
    spike_ramp_min = SPIKE_DURATION_MIN * SPIKE_RAMP_FRAC
    spike_drop_start = SPIKE_DURATION_MIN - spike_ramp_min

    if spike_position < spike_ramp_min:
        # Rising edge (linear ramp)
        envelope = spike_position / spike_ramp_min
    elif spike_position < spike_drop_start:
        # Flat-top
        envelope = 1.0
    elif spike_position < SPIKE_DURATION_MIN:
        # Falling edge (linear ramp down)
        envelope = 1.0 - (spike_position - spike_drop_start) / spike_ramp_min
    else:
        envelope = 0.0

    spike_delta = base * (SPIKE_FACTOR - 1.0) * envelope

    # ---- Component 4: noise ----
    noise = (base + sine + spike_delta) * random.uniform(
        -NOISE_FRACTION, NOISE_FRACTION
    )

    total = base + sine + spike_delta + noise
    return max(1, int(round(total)))


# ---------------------------------------------------------------------------
# CUR row builders (one service per function)
# ---------------------------------------------------------------------------

def _fmt_dt(dt: datetime) -> str:
    """ISO-8601 UTC datetime string for CUR timestamps."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _time_interval(start: datetime, end: datetime) -> str:
    return f"{_fmt_dt(start)}/{_fmt_dt(end)}"


def _base_row(
    interval_start: datetime,
    interval_end: datetime,
    billing_start: datetime,
    billing_end: datetime,
    invoice_id: str,
) -> dict:
    """Shared fields common to every CUR line item."""
    return {
        "identity/LineItemId": uuid.uuid4().hex,
        "identity/TimeInterval": _time_interval(interval_start, interval_end),
        "bill/InvoiceId": invoice_id,
        "bill/InvoicingEntity": "Amazon Web Services, Inc.",
        "bill/BillingEntity": BILLING_ENTITY,
        "bill/BillType": BILL_TYPE,
        "bill/PayerAccountId": PAYER_ACCOUNT_ID,
        "bill/BillingPeriodStartDate": _fmt_dt(billing_start),
        "bill/BillingPeriodEndDate": _fmt_dt(billing_end),
        "lineItem/UsageAccountId": AWS_ACCOUNT_ID,
        "lineItem/LineItemType": "Usage",
        "lineItem/UsageStartDate": _fmt_dt(interval_start),
        "lineItem/UsageEndDate": _fmt_dt(interval_end),
        "lineItem/AvailabilityZone": f"{AWS_REGION}a",
        "lineItem/NormalizationFactor": "1",
        "lineItem/CurrencyCode": CURRENCY_CODE,
        "reservation/AmortizedUpfrontFeeForBillingPeriod": "",
        "reservation/AmortizedUpfrontCostForUsage": "",
        "reservation/RecurringFeeForUsage": "",
        "reservation/UnusedQuantity": "",
        "reservation/UnusedRecurringFee": "",
        "savingsPlan/SavingsPlanARN": "",
        "savingsPlan/SavingsPlanRate": "",
        "savingsPlan/UsedCommitment": "",
        "savingsPlan/SavingsPlanEffectiveCost": "",
        "savingsPlan/TotalCommitmentToDate": "",
        "savingsPlan/NetSavingsPlanEffectiveCost": "",
        "resourceTags/user:Project": "CRIP",
        "resourceTags/user:Environment": "load-test",
        "resourceTags/user:Team": "platform-engineering",
        "pricing/term": "OnDemand",
        "product/region": AWS_REGION,
        "product/location": "US East (N. Virginia)",
    }


def row_apigw(requests: int, base: dict) -> dict:
    """API Gateway REST — charged per million requests."""
    usage_amount = requests / 1_000_000
    rate = PRICE_APIGW_PER_M_REQ
    cost = round(usage_amount * rate, 10)

    row = dict(base)
    row.update({
        "lineItem/ProductCode": "AmazonApiGateway",
        "lineItem/UsageType": f"{AWS_REGION}-ApiGatewayRequest",
        "lineItem/Operation": "ApiGatewayHttpRequest",
        "lineItem/ResourceId": f"arn:aws:apigateway:{AWS_REGION}::/restapis/crip1api0",
        "lineItem/UsageAmount": f"{usage_amount:.10f}",
        "lineItem/NormalizedUsageAmount": f"{usage_amount:.10f}",
        "lineItem/UnblendedRate": f"{rate:.10f}",
        "lineItem/UnblendedCost": f"{cost:.10f}",
        "lineItem/BlendedRate": f"{rate:.10f}",
        "lineItem/BlendedCost": f"{cost:.10f}",
        "lineItem/LineItemDescription": (
            f"${rate:.4f} per 1,000,000 API calls received"
        ),
        "product/ProductName": "Amazon API Gateway",
        "product/servicecode": "AmazonApiGateway",
        "product/usagetype": "ApiGatewayRequest",
        "pricing/publicOnDemandCost": f"{cost:.10f}",
        "pricing/publicOnDemandRate": f"{rate:.10f}",
        "pricing/unit": "1M Requests",
    })
    return row


def row_ec2_data_transfer(requests: int, base: dict) -> dict:
    """EC2 Data Transfer Out — derived from request count × avg payload size."""
    gb_transferred = (requests * AVG_PAYLOAD_KB) / (1024 * 1024)   # KB → GB
    rate = PRICE_EC2_DT_PER_GB
    cost = round(gb_transferred * rate, 10)

    row = dict(base)
    row.update({
        "lineItem/ProductCode": "AmazonEC2",
        "lineItem/UsageType": f"{AWS_REGION}-DataTransfer-Out-Bytes",
        "lineItem/Operation": "PublicIP-Out",
        "lineItem/ResourceId": f"i-0crip{random.randint(10000,99999):05d}",
        "lineItem/UsageAmount": f"{gb_transferred:.10f}",
        "lineItem/NormalizedUsageAmount": f"{gb_transferred:.10f}",
        "lineItem/UnblendedRate": f"{rate:.10f}",
        "lineItem/UnblendedCost": f"{cost:.10f}",
        "lineItem/BlendedRate": f"{rate:.10f}",
        "lineItem/BlendedCost": f"{cost:.10f}",
        "lineItem/LineItemDescription": (
            f"${rate:.4f} per GB — data transferred out (Internet)"
        ),
        "product/ProductName": "Amazon Elastic Compute Cloud",
        "product/servicecode": "AmazonEC2",
        "product/usagetype": "DataTransfer-Out-Bytes",
        "pricing/publicOnDemandCost": f"{cost:.10f}",
        "pricing/publicOnDemandRate": f"{rate:.10f}",
        "pricing/unit": "GB",
    })
    return row


def row_dynamodb(requests: int, base: dict) -> dict:
    """DynamoDB — read request units from /api/data endpoint."""
    rru = int(requests * DYNAMO_READ_FRACTION)
    usage_amount = rru / 1_000_000
    rate = PRICE_DYNAMO_PER_M_RRU
    cost = round(usage_amount * rate, 10)

    row = dict(base)
    row.update({
        "lineItem/ProductCode": "AmazonDynamoDB",
        "lineItem/UsageType": f"{AWS_REGION}-ReadRequestUnits",
        "lineItem/Operation": "GetItem",
        "lineItem/ResourceId": (
            f"arn:aws:dynamodb:{AWS_REGION}:{AWS_ACCOUNT_ID}:table/CRIPDataTable"
        ),
        "lineItem/UsageAmount": f"{usage_amount:.10f}",
        "lineItem/NormalizedUsageAmount": f"{usage_amount:.10f}",
        "lineItem/UnblendedRate": f"{rate:.10f}",
        "lineItem/UnblendedCost": f"{cost:.10f}",
        "lineItem/BlendedRate": f"{rate:.10f}",
        "lineItem/BlendedCost": f"{cost:.10f}",
        "lineItem/LineItemDescription": (
            f"${rate:.4f} per 1,000,000 DynamoDB read request units"
        ),
        "product/ProductName": "Amazon DynamoDB",
        "product/servicecode": "AmazonDynamoDB",
        "product/usagetype": "ReadRequestUnits",
        "pricing/publicOnDemandCost": f"{cost:.10f}",
        "pricing/publicOnDemandRate": f"{rate:.10f}",
        "pricing/unit": "1M RRU",
    })
    return row


def row_cloudwatch_logs(requests: int, base: dict) -> dict:
    """CloudWatch Logs — ingested bytes from application log lines."""
    gb_logs = (requests * AVG_LOG_BYTES) / (1024 ** 3)
    rate = PRICE_CW_LOGS_PER_GB
    cost = round(gb_logs * rate, 10)

    row = dict(base)
    row.update({
        "lineItem/ProductCode": "AmazonCloudWatch",
        "lineItem/UsageType": f"{AWS_REGION}-DataProcessing-Bytes",
        "lineItem/Operation": "PutLogEvents",
        "lineItem/ResourceId": (
            f"arn:aws:logs:{AWS_REGION}:{AWS_ACCOUNT_ID}:log-group:/crip/target-app"
        ),
        "lineItem/UsageAmount": f"{gb_logs:.10f}",
        "lineItem/NormalizedUsageAmount": f"{gb_logs:.10f}",
        "lineItem/UnblendedRate": f"{rate:.10f}",
        "lineItem/UnblendedCost": f"{cost:.10f}",
        "lineItem/BlendedRate": f"{rate:.10f}",
        "lineItem/BlendedCost": f"{cost:.10f}",
        "lineItem/LineItemDescription": (
            f"${rate:.4f} per GB — CloudWatch Logs data ingestion"
        ),
        "product/ProductName": "Amazon CloudWatch",
        "product/servicecode": "AmazonCloudWatch",
        "product/usagetype": "DataProcessing-Bytes",
        "pricing/publicOnDemandCost": f"{cost:.10f}",
        "pricing/publicOnDemandRate": f"{rate:.10f}",
        "pricing/unit": "GB",
    })
    return row


def row_s3_puts(requests: int, base: dict) -> dict:
    """S3 PUT requests — from the /api/ingest endpoint."""
    puts = int(requests * S3_INGEST_FRACTION)
    usage_amount = puts / 1_000    # priced per 1 000 PUTs
    rate = PRICE_S3_PUT_PER_K
    cost = round(usage_amount * rate, 10)

    row = dict(base)
    row.update({
        "lineItem/ProductCode": "AmazonS3",
        "lineItem/UsageType": f"{AWS_REGION}-Requests-Tier1",
        "lineItem/Operation": "PutObject",
        "lineItem/ResourceId": (
            f"arn:aws:s3:::crip-ingest-{AWS_ACCOUNT_ID}"
        ),
        "lineItem/UsageAmount": f"{usage_amount:.10f}",
        "lineItem/NormalizedUsageAmount": f"{usage_amount:.10f}",
        "lineItem/UnblendedRate": f"{rate:.10f}",
        "lineItem/UnblendedCost": f"{cost:.10f}",
        "lineItem/BlendedRate": f"{rate:.10f}",
        "lineItem/BlendedCost": f"{cost:.10f}",
        "lineItem/LineItemDescription": (
            f"${rate:.4f} per 1,000 PUT requests — Amazon S3"
        ),
        "product/ProductName": "Amazon Simple Storage Service",
        "product/servicecode": "AmazonS3",
        "product/usagetype": "Requests-Tier1",
        "pricing/publicOnDemandCost": f"{cost:.10f}",
        "pricing/publicOnDemandRate": f"{rate:.10f}",
        "pricing/unit": "Requests",
    })
    return row


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_cur(
    report_date: datetime,
    output_path: Path,
    interval_min: int = DEFAULT_INTERVAL_MIN,
) -> Tuple[int, float]:
    """
    Generate a full 24-hour synthetic CUR CSV.

    Parameters
    ----------
    report_date   : datetime  – The calendar day to generate (UTC midnight start)
    output_path   : Path      – Destination CSV file
    interval_min  : int       – Granularity in minutes (default: 5)

    Returns
    -------
    (row_count, total_cost_usd) : summary statistics
    """
    random.seed(42)   # reproducible output for the same date

    billing_start = report_date.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    billing_end = billing_start + timedelta(days=1)
    invoice_id  = f"INV-CRIP-{billing_start.strftime('%Y%m%d')}"

    intervals: List[datetime] = []
    t = billing_start
    while t < billing_end:
        intervals.append(t)
        t += timedelta(minutes=interval_min)

    rows: List[dict] = []
    total_requests = 0
    total_cost = 0.0

    print(f"  Generating {len(intervals)} intervals x 5 services ...")

    for interval_start in intervals:
        interval_end = interval_start + timedelta(minutes=interval_min)
        req = compute_requests(interval_start, interval_min)
        total_requests += req

        base = _base_row(
            interval_start, interval_end,
            billing_start, billing_end,
            invoice_id,
        )

        service_rows = [
            row_apigw(req, base),
            row_ec2_data_transfer(req, base),
            row_dynamodb(req, base),
            row_cloudwatch_logs(req, base),
            row_s3_puts(req, base),
        ]

        for r in service_rows:
            total_cost += float(r["lineItem/UnblendedCost"])

        rows.extend(service_rows)

    # --- Write CSV ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CUR_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), total_cost


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic AWS CUR CSV for CRIP Phase 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="reports/synthetic_cur.csv",
        help="Output file path (will be created if absent).",
    )
    parser.add_argument(
        "--date",
        default=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        help="Report date in YYYY-MM-DD format (UTC).",
    )
    parser.add_argument(
        "--interval-min",
        type=int,
        default=DEFAULT_INTERVAL_MIN,
        choices=[1, 5, 15, 30, 60],
        help="Billing interval granularity in minutes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    try:
        report_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: Invalid date format '{args.date}'. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    print("=" * 60)
    print("  CRIP Phase 1 -- Synthetic CUR Generator")
    print("=" * 60)
    print(f"  Report date    : {args.date}")
    print(f"  Interval       : {args.interval_min} min")
    print(f"  Output         : {output_path.resolve()}")
    print("-" * 60)

    row_count, total_cost = generate_cur(
        report_date=report_date,
        output_path=output_path,
        interval_min=args.interval_min,
    )

    print("-" * 60)
    print(f"  [OK] Wrote {row_count:,} CUR rows")
    print(f"  [$]  Estimated 24-h cost   : USD {total_cost:,.6f}")
    print(f"  [F]  File                  : {output_path.resolve()}")
    print("=" * 60)

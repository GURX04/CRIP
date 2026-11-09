"""
remediation/remediation_handler.py
==================================
CRIP Automated Cost Anomaly Remediation (Dry-Run Lambda Handler)

Triggered via HTTP POST webhook (Lambda Function URL) from anomaly_detection.py
whenever a cost anomaly is detected.

Actions:
1. Parses incoming anomaly payload (timestamp, actual cost, expected cost, telemetry).
2. Calculates severity and evaluates root-cause signals.
3. Simulates automated remediation actions (e.g., dynamic API Gateway throttling,
   concurrency clamping, circuit breaker tripping).
4. Emits detailed structured audit logs in cloud / LocalStack console.
5. Returns a JSON response summarizing proposed actions.
"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    logger.info("Received anomaly remediation event: %s", json.dumps(event))

    # Parse body if invoked via Lambda Function URL / API Gateway
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if isinstance(body, str):
            try:
                payload = json.loads(body)
            except Exception as e:
                logger.error("Failed to parse JSON body: %s", e)
                payload = {"raw_body": body}
        else:
            payload = body or {}
    else:
        payload = event or {}

    # Extract anomaly parameters
    timestamp = payload.get("timestamp") or payload.get("window_start", datetime.utcnow().isoformat())
    actual_cost = float(payload.get("actual_cost", 0.0))
    expected_cost = float(payload.get("expected_cost", 0.0))
    cost_upper_bound = float(payload.get("cost_upper_bound", 0.0))
    cost_delta = float(payload.get("cost_delta", actual_cost - expected_cost))
    pct_over = float(payload.get("pct_over", payload.get("pct_over_bound", 0.0)))
    metrics = payload.get("metrics") or payload.get("correlated_metrics") or {}

    # Root cause analysis signals
    top_service = payload.get("top_service", "AmazonApiGateway")
    http_rate = metrics.get("http_requests_rate_1m", 0.0)
    cpu_rate = metrics.get("cpu_usage_rate_1m", 0.0)

    # Determine automated dry-run policy
    if http_rate > 50 or actual_cost > 0.05:
        proposed_action = "APIGateway_Throttle_Stage"
        throttle_rate = 50.0  # requests per second
        burst_limit = 100
        remediation_reason = f"High traffic throughput ({http_rate:.2f} rps) driving cost spike"
    else:
        proposed_action = "Lambda_Concurrency_Limit"
        throttle_rate = 10.0
        burst_limit = 20
        remediation_reason = f"Cost exceeded upper bound by USD {cost_delta:.6f} (+{pct_over:.1f}%)"

    # =========================================================================
    # DRY RUN REMEDIATION EXECUTION & LOGGING
    # =========================================================================
    print("=" * 70)
    print(" [DRY RUN REMEDIATION] AUTOMATED COST ANOMALY RESPONSE TRIGGERED")
    print("=" * 70)
    print(f" Timestamp           : {timestamp}")
    print(f" Target Service      : {top_service}")
    print(f" Actual Cost         : USD {actual_cost:.6f}")
    print(f" Expected (Baseline) : USD {expected_cost:.6f}")
    print(f" Excess Cost (Delta) : USD {cost_delta:.6f} (+{pct_over:.1f}%)")
    print("-" * 70)
    print(" Correlated Telemetry:")
    for k, v in metrics.items():
        print(f"   * {k:<25} : {v}")
    print("-" * 70)
    print(f" Proposed Action     : {proposed_action}")
    print(f" Remediation Reason  : {remediation_reason}")
    print(f" Policy Parameters   : RateLimit = {throttle_rate} req/sec, BurstLimit = {burst_limit}")
    print(" Action Status       : DRY-RUN ONLY (No live resources modified)")
    print("=" * 70)

    # Construct response
    response_body = {
        "status": "success",
        "dry_run": True,
        "action_taken": proposed_action,
        "details": {
            "timestamp": timestamp,
            "target_service": top_service,
            "actual_cost": actual_cost,
            "cost_delta": cost_delta,
            "remediation_reason": remediation_reason,
            "simulated_throttle_rate": throttle_rate,
            "simulated_burst_limit": burst_limit,
        },
    }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "X-CRIP-Remediation": "DRY-RUN",
        },
        "body": json.dumps(response_body),
    }


if __name__ == "__main__":
    # Local unit test
    test_event = {
        "timestamp": "2026-10-26T21:15:00Z",
        "actual_cost": 0.070788,
        "expected_cost": 0.053088,
        "cost_upper_bound": 0.061848,
        "cost_delta": 0.008940,
        "pct_over": 14.4,
        "top_service": "AmazonApiGateway",
        "metrics": {
            "http_requests_rate_1m": 68.45,
            "cpu_usage_rate_1m": 0.32,
            "memory_usage_pct": 0.44,
        },
    }
    res = lambda_handler(test_event, None)
    print("Local execution returned:", res["statusCode"])

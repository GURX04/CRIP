"""
tests/test_remediation.py
=========================
Unit and integration tests for the CRIP live remediation engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest

from remediation.config import RemediationSettings
from remediation.executor import RemediationExecutor
from remediation.guardrails import RemediationGuardrails
from remediation.impact import RemediationImpactCalculator
from remediation.models import (
    ActionType,
    ProposalStatus,
    RemediationProposal,
)
from remediation.planner import RemediationPlanner
from remediation.worker import RemediationWorker


# ---------------------------------------------------------------------------
# Test 1: Planner mappings
# ---------------------------------------------------------------------------

def test_planner_traffic_driven_creates_rate_limit() -> None:
    """Traffic-driven anomaly produces RateLimit proposal with estimated savings."""
    settings = RemediationSettings()
    planner = RemediationPlanner(settings)

    detection = {
        "detection_id": "det-1",
        "run_id": "run-1",
        "service": "DynamoDB",
        "severity": 75.0,
        "sim_ts": "2026-01-01T12:00:00+00:00",
        "rca": {
            "alert_type": "traffic_driven",
            "cost_per_1k_now": 0.00125,
            "cost_per_1k_baseline": 0.00125,
            "pct_cost_change": 150.0,
            "pct_volume_change": 150.0,
        },
    }

    proposal = planner.plan_detection(detection)

    assert proposal.action == ActionType.RATE_LIMIT
    assert proposal.service == "DynamoDB"
    assert proposal.classification == "traffic_driven"
    assert 0.49 <= proposal.params["factor"] <= 0.90
    assert proposal.estimated_savings_hourly > 0.0
    assert proposal.estimated_risk > 0.0
    assert "RateLimit" in proposal.reason


def test_planner_price_driven_never_throttles() -> None:
    """Price-driven anomaly produces INVESTIGATE_CONFIG and strictly never a throttle."""
    settings = RemediationSettings()
    planner = RemediationPlanner(settings)

    detection = {
        "detection_id": "det-2",
        "run_id": "run-1",
        "service": "DynamoDB",
        "severity": 85.0,
        "sim_ts": "2026-01-01T12:00:00+00:00",
        "rca": {
            "alert_type": "price_driven",
            "cost_per_1k_now": 0.00375,
            "cost_per_1k_baseline": 0.00125,
            "pct_cost_change": 200.0,
            "pct_volume_change": 0.0,
        },
    }

    proposal = planner.plan_detection(detection)

    assert proposal.action == ActionType.INVESTIGATE_CONFIG
    assert proposal.action != ActionType.RATE_LIMIT
    assert proposal.action != ActionType.CONCURRENCY_CAP
    assert "factor" not in proposal.params
    assert "No throttle recommended" in proposal.reason
    assert "cost_per_1k_delta" in proposal.params


def test_planner_volume_drop_cost_flat() -> None:
    """Volume drop with flat cost produces ALERT_ONLY."""
    settings = RemediationSettings()
    planner = RemediationPlanner(settings)

    detection = {
        "detection_id": "det-3",
        "run_id": "run-1",
        "service": "EC2 Data Transfer",
        "severity": 65.0,
        "rca": {
            "alert_type": "volume_drop_cost_flat",
            "pct_cost_change": 0.0,
            "pct_volume_change": -60.0,
        },
    }

    proposal = planner.plan_detection(detection)
    assert proposal.action == ActionType.ALERT_ONLY
    assert "orphaned provisioned capacity" in proposal.reason


# ---------------------------------------------------------------------------
# Test 2: Guardrails
# ---------------------------------------------------------------------------

def test_guardrail_severity_floor() -> None:
    """Severity below min_severity (60) is blocked."""
    settings = RemediationSettings(min_severity=60.0)
    guardrails = RemediationGuardrails(settings)

    proposal = RemediationProposal(
        detection_id="d-1",
        run_id="r-1",
        service="DynamoDB",
        classification="traffic_driven",
        action=ActionType.RATE_LIMIT,
        params={"factor": 0.6},
        estimated_savings_hourly=10.0,
        estimated_risk=0.05,
        reason="test",
    )

    allowed, logs, status = guardrails.evaluate(proposal, severity=50.0)
    assert not allowed
    assert status == ProposalStatus.BLOCKED_BY_GUARDRAIL
    assert any(g.guardrail == "min_severity" and not g.allowed for g in logs)


def test_guardrail_cooldown() -> None:
    """Subsequent action on the same service within cooldown window is blocked."""
    settings = RemediationSettings(cooldown_sim_min=15.0)
    guardrails = RemediationGuardrails(settings)

    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    proposal = RemediationProposal(
        detection_id="d-1",
        run_id="r-1",
        service="DynamoDB",
        classification="traffic_driven",
        action=ActionType.RATE_LIMIT,
        params={"factor": 0.6},
        estimated_savings_hourly=10.0,
        estimated_risk=0.05,
        reason="test",
    )

    # First evaluation passes
    allowed, _, _ = guardrails.evaluate(proposal, severity=70.0, sim_ts=now)
    assert allowed
    guardrails.record_action_applied(proposal, sim_ts=now)

    # Second evaluation 5 min later is blocked by cooldown
    five_min_later = now + timedelta(minutes=5)
    prop2 = RemediationProposal(
        detection_id="d-2",
        run_id="r-1",
        service="DynamoDB",
        classification="traffic_driven",
        action=ActionType.RATE_LIMIT,
        params={"factor": 0.6},
        estimated_savings_hourly=10.0,
        estimated_risk=0.05,
        reason="test",
    )
    allowed2, logs2, status2 = guardrails.evaluate(prop2, severity=70.0, sim_ts=five_min_later)
    assert not allowed2
    assert status2 == ProposalStatus.BLOCKED_BY_GUARDRAIL
    assert any(g.guardrail == "service_cooldown" and not g.allowed for g in logs2)

    # 16 min later passes
    sixteen_min_later = now + timedelta(minutes=16)
    allowed3, _, _ = guardrails.evaluate(prop2, severity=70.0, sim_ts=sixteen_min_later)
    assert allowed3


def test_guardrail_kill_switch() -> None:
    """Kill switch immediately blocks all actions."""
    settings = RemediationSettings()
    guardrails = RemediationGuardrails(settings)
    guardrails.set_kill_switch(True)

    proposal = RemediationProposal(
        detection_id="d-1",
        run_id="r-1",
        service="DynamoDB",
        classification="traffic_driven",
        action=ActionType.RATE_LIMIT,
        params={"factor": 0.6},
        estimated_savings_hourly=10.0,
        estimated_risk=0.05,
        reason="test",
    )

    allowed, logs, status = guardrails.evaluate(proposal, severity=75.0)
    assert not allowed
    assert status == ProposalStatus.BLOCKED_BY_GUARDRAIL
    assert any(g.guardrail == "kill_switch" and not g.allowed for g in logs)


def test_guardrail_approval_policy() -> None:
    """High severity or high risk routes to PENDING_APPROVAL."""
    settings = RemediationSettings(auto_approve_max_severity=80.0, auto_approve_max_risk=0.15)
    guardrails = RemediationGuardrails(settings)

    # Moderate severity + low risk -> auto-approved
    prop_low = RemediationProposal(
        detection_id="d-low",
        run_id="r-1",
        service="DynamoDB",
        classification="traffic_driven",
        action=ActionType.RATE_LIMIT,
        params={"factor": 0.6},
        estimated_savings_hourly=10.0,
        estimated_risk=0.05,
        reason="test",
    )
    allowed, _, status = guardrails.evaluate(prop_low, severity=70.0)
    assert allowed
    assert status in (ProposalStatus.APPLIED, ProposalStatus.DRY_RUN_RECORDED)

    # High severity (>80) -> PENDING_APPROVAL
    prop_high = RemediationProposal(
        detection_id="d-high",
        run_id="r-1",
        service="DynamoDB",
        classification="traffic_driven",
        action=ActionType.RATE_LIMIT,
        params={"factor": 0.6},
        estimated_savings_hourly=10.0,
        estimated_risk=0.05,
        reason="test",
    )
    allowed2, _, status2 = guardrails.evaluate(prop_high, severity=90.0)
    assert allowed2
    assert status2 == ProposalStatus.PENDING_APPROVAL


# ---------------------------------------------------------------------------
# Test 3: Executor
# ---------------------------------------------------------------------------

def test_executor_dry_run_and_idempotency() -> None:
    """DRY_RUN never calls generator API; execution is idempotent."""
    settings = RemediationSettings(mode="DRY_RUN", localstack_url="")
    executor = RemediationExecutor(settings)

    proposal = RemediationProposal(
        detection_id="d-1",
        run_id="r-1",
        service="DynamoDB",
        classification="traffic_driven",
        action=ActionType.RATE_LIMIT,
        params={"factor": 0.6, "duration_min": 15},
        estimated_savings_hourly=10.0,
        estimated_risk=0.05,
        reason="test",
    )

    with patch("urllib.request.urlopen") as mock_url:
        # First execution in DRY_RUN
        ok = executor.execute(proposal)
        assert ok
        assert proposal.status == ProposalStatus.DRY_RUN_RECORDED
        # Generator was NOT called
        mock_url.assert_not_called()

        # Duplicate execution is an idempotent no-op
        ok2 = executor.execute(proposal)
        assert ok2
        mock_url.assert_not_called()

        # Rollback
        rb_ok = executor.rollback(proposal)
        assert rb_ok
        assert proposal.status == ProposalStatus.ROLLED_BACK


# ---------------------------------------------------------------------------
# Test 4: Impact Calculation
# ---------------------------------------------------------------------------

def test_impact_counterfactual_and_net_benefit() -> None:
    """Validate counterfactual savings, collateral penalty, and net benefit."""
    settings = RemediationSettings(value_per_request=0.0001)
    calc = RemediationImpactCalculator(settings)

    proposal = RemediationProposal(
        detection_id="d-1",
        run_id="r-1",
        service="DynamoDB",
        classification="traffic_driven",
        action=ActionType.RATE_LIMIT,
        params={"factor": 0.5},
        estimated_savings_hourly=20.0,
        estimated_risk=0.05,
        reason="test",
    )

    # Observed actual cost under factor 0.5 was $10.00
    # Counterfactual without mitigation = $10.00 / 0.5 = $20.00
    # Estimated savings = $20.00 - $10.00 = $10.00
    # Baseline requests = 10,000. Risk = 5% -> 500 legitimate requests dropped
    # Collateral penalty = 500 * $0.0001 = $0.05
    # Net benefit = $10.00 - $0.05 = $9.95
    res = calc.calculate_impact(
        proposal=proposal,
        actual_cost_usd=10.0,
        baseline_cost_usd=5.0,
        actual_requests=10_000,
        baseline_requests=10_000,
    )

    assert res.counterfactual_cost_usd == 20.0
    assert res.estimated_savings_usd == 10.0
    assert res.legitimate_requests_dropped == 500.0
    assert res.collateral_penalty_usd == 0.05
    assert res.net_benefit_usd == 9.95
    assert "Counterfactual projection" in res.estimation_method


# ---------------------------------------------------------------------------
# Test 5: End-to-End Worker Process Detection
# ---------------------------------------------------------------------------

def test_worker_process_detection_and_rollback() -> None:
    """Worker consumes detection, creates proposal, executes dry-run, and rolls back on close."""
    settings = RemediationSettings(mode="DRY_RUN")
    mock_redis = MagicMock()
    worker = RemediationWorker(settings=settings, redis_client=mock_redis)

    detection = {
        "detection_id": "det-e2e-1",
        "run_id": "run-e2e",
        "service": "AmazonApiGateway",
        "severity": 72.0,
        "status": "open",
        "sim_ts": "2026-01-01T12:00:00+00:00",
        "rca": {
            "alert_type": "traffic_driven",
            "cost_per_1k_now": 0.0035,
            "cost_per_1k_baseline": 0.0035,
            "pct_cost_change": 120.0,
            "pct_volume_change": 120.0,
        },
    }

    # Step 1: Open detection -> proposal created and recorded
    prop = worker.process_detection(detection)
    assert prop.status == ProposalStatus.DRY_RUN_RECORDED
    assert prop.service == "AmazonApiGateway"
    assert mock_redis.xadd.called

    # Step 2: Closed detection -> triggers auto-rollback
    closing_det = dict(detection)
    closing_det["status"] = "closed"

    rolled_prop = worker.process_detection(closing_det)
    assert rolled_prop.status == ProposalStatus.ROLLED_BACK


# ---------------------------------------------------------------------------
# Test 6: Offline Evaluation Hooks
# ---------------------------------------------------------------------------

def test_offline_remediation_evaluation() -> None:
    """Offline remediation evaluator computes cost avoided and 100% price refusal."""
    from evaluation.remediation_eval import RemediationEvaluator

    evaluator = RemediationEvaluator()
    summary = evaluator.evaluate_scenarios(seeds=[101], scenario_names=["quick_spike", "quick_cost_only"])

    assert summary["price_driven_throttles"] == 0
    assert summary["price_driven_refusal_rate"] == 1.0
    assert summary["total_injections_evaluated"] >= 2
    assert summary["mean_cost_avoided_usd"] >= 0.0

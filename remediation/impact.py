"""
remediation/impact.py
=====================
Counterfactual savings and collateral impact calculation for CRIP mitigations.

Formulas:
1. Counterfactual Cost:
   Expected cost had mitigation NOT been applied:
       counterfactual_cost = actual_cost / factor
2. Estimated Savings:
   Direct dollar cost avoided:
       estimated_savings = max(0.0, counterfactual_cost - actual_cost)
3. Collateral Impact:
   Legitimate requests dropped due to rate-limiting:
       dropped_legitimate_requests = baseline_requests * estimated_risk
       collateral_penalty = dropped_legitimate_requests * value_per_request
4. Net Benefit:
   net_benefit = estimated_savings - collateral_penalty

Outputs are published to Redis stream crip:remediation_results and persisted
into TimescaleDB hypertables.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from remediation.config import RemediationSettings, get_remediation_settings
from remediation.models import (
    RemediationProposal,
    RemediationResult,
)

log = logging.getLogger(__name__)


class RemediationImpactCalculator:
    """
    Computes counterfactual savings, collateral damage, and net financial benefit.
    """

    def __init__(self, settings: Optional[RemediationSettings] = None) -> None:
        self.settings = settings or get_remediation_settings()
        self._db_conn = None
        self._setup_db()

    def _setup_db(self) -> None:
        try:
            import psycopg2  # noqa: PLC0415
            self._db_conn = psycopg2.connect(self.settings.db_dsn, connect_timeout=1)
            self._db_conn.autocommit = True
        except Exception:
            self._db_conn = None

    def calculate_impact(
        self,
        proposal: RemediationProposal,
        actual_cost_usd: float,
        baseline_cost_usd: float,
        actual_requests: float = 10_000.0,
        baseline_requests: float = 8_000.0,
    ) -> RemediationResult:
        """
        Compute honest estimated savings and collateral impact.
        """
        factor = float(proposal.params.get("factor", 1.0))
        if factor <= 0.0 or factor >= 1.0:
            factor = 1.0

        # 1. Counterfactual Cost (what cost would have been without the throttle)
        if factor < 1.0:
            counterfactual_cost = actual_cost_usd / factor
        else:
            counterfactual_cost = actual_cost_usd

        # 2. Estimated Savings
        estimated_savings = max(0.0, counterfactual_cost - actual_cost_usd)

        # 3. Collateral Damage: legitimate traffic impacted
        # Estimated risk gives fraction of legitimate requests that faced rate limiting
        risk_fraction = float(proposal.estimated_risk)
        legitimate_dropped = max(0.0, baseline_requests * risk_fraction)
        collateral_penalty = legitimate_dropped * self.settings.value_per_request

        # 4. Net Benefit
        net_benefit = estimated_savings - collateral_penalty

        method_desc = (
            f"Counterfactual projection (actual / factor {factor:.2f}) "
            f"minus observed cost and {self.settings.value_per_request * 100:.2f}¢/req collateral penalty"
        )

        result = RemediationResult(
            action_id=proposal.remediation_id or proposal.proposal_id,
            proposal_id=proposal.proposal_id,
            service=proposal.service,
            actual_cost_usd=actual_cost_usd,
            counterfactual_cost_usd=counterfactual_cost,
            estimated_savings_usd=estimated_savings,
            legitimate_requests_dropped=legitimate_dropped,
            collateral_penalty_usd=collateral_penalty,
            net_benefit_usd=net_benefit,
            estimation_method=method_desc,
        )

        return result

    def persist_proposal_action(self, proposal: RemediationProposal) -> None:
        """Persist proposal decision to TimescaleDB remediation_actions."""
        if not self._db_conn:
            return
        try:
            with self._db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO remediation_actions (
                        action_id, proposal_id, detection_id, run_id, service,
                        action_type, mode, params, status, reason, guardrail_log,
                        sim_applied_ts, sim_rollback_ts, wall_applied_ms
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        proposal.remediation_id or proposal.proposal_id,
                        proposal.proposal_id,
                        proposal.detection_id,
                        proposal.run_id,
                        proposal.service,
                        proposal.action.value,
                        self.settings.mode,
                        json.dumps(proposal.params),
                        proposal.status.value,
                        proposal.reason,
                        json.dumps([g.__dict__ for g in proposal.guardrail_logs]),
                        proposal.mitigation_applied_sim_ts,
                        proposal.mitigation_rollback_sim_ts,
                        proposal.created_at_wall_ms,
                    ),
                )
        except Exception as exc:
            log.error("db_persist_action_failed: %s", exc)

    def persist_impact_result(self, result: RemediationResult) -> None:
        """Persist impact result to TimescaleDB remediation_results."""
        if not self._db_conn:
            return
        try:
            with self._db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO remediation_results (
                        result_id, action_id, proposal_id, service,
                        actual_cost_usd, counterfactual_cost_usd, estimated_savings_usd,
                        legitimate_requests_dropped, collateral_penalty_usd,
                        net_benefit_usd, estimation_method, calculated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result.result_id,
                        result.action_id,
                        result.proposal_id,
                        result.service,
                        result.actual_cost_usd,
                        result.counterfactual_cost_usd,
                        result.estimated_savings_usd,
                        result.legitimate_requests_dropped,
                        result.collateral_penalty_usd,
                        result.net_benefit_usd,
                        result.estimation_method,
                        result.calculated_at,
                    ),
                )
        except Exception as exc:
            log.error("db_persist_result_failed: %s", exc)


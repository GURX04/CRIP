"""
remediation/planner.py
======================
Translates anomaly detections into structured remediation proposals.

Responsibilities:
1. Parse detection events and evaluate RCA signals (traffic_driven, price_driven, etc.).
2. Map classification and service to appropriate action:
   - traffic_driven -> RateLimit (or ConcurrencyCap) with optimal factor.
   - price_driven   -> Investigate configuration change (NEVER throttles).
   - volume_drop_cost_flat -> Alert only.
3. Compute estimated hourly cost savings and estimated collateral risk.
4. Return a structured RemediationProposal ready for guardrail validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from remediation.config import RemediationSettings, get_remediation_settings
from remediation.models import (
    ActionType,
    ProposalStatus,
    RemediationProposal,
)

log = logging.getLogger(__name__)


class RemediationPlanner:
    """
    Evaluates incoming detections and produces remediation proposals.
    """

    def __init__(self, settings: Optional[RemediationSettings] = None) -> None:
        self.settings = settings or get_remediation_settings()

    def plan_detection(self, detection: Dict[str, Any]) -> RemediationProposal:
        """
        Produce a RemediationProposal from a detection dictionary.

        Parameters
        ----------
        detection:
            Dictionary payload matching pipeline.models.Detection format.
        """
        detection_id = str(detection.get("detection_id", "unknown"))
        run_id = str(detection.get("run_id", "unknown"))
        service = str(detection.get("service", "unknown"))
        severity = float(detection.get("severity", 0.0))
        sim_ts = None
        if detection.get("sim_ts"):
            from datetime import datetime  # noqa: PLC0415
            try:
                sim_ts = datetime.fromisoformat(str(detection["sim_ts"]))
            except Exception:
                pass

        rca = detection.get("rca", {})
        alert_type = str(rca.get("alert_type", "unknown"))
        cost_per_1k_now = float(rca.get("cost_per_1k_now", 0.0))
        cost_per_1k_base = float(rca.get("cost_per_1k_baseline", 0.0))
        pct_cost_change = float(rca.get("pct_cost_change", 0.0))
        pct_volume_change = float(rca.get("pct_volume_change", 0.0))

        # -------------------------------------------------------------
        # 1. Price-driven: NEVER throttle; investigate configuration
        # -------------------------------------------------------------
        if alert_type == "price_driven":
            cost_delta_1k = cost_per_1k_now - cost_per_1k_base
            reason = (
                f"Price-driven anomaly detected on {service}: cost per 1k rose from "
                f"${cost_per_1k_base:.5f} to ${cost_per_1k_now:.5f} (+{pct_cost_change:.1f}%). "
                f"No throttle recommended: throttling cannot fix a price or tier change."
            )
            return RemediationProposal(
                detection_id=detection_id,
                run_id=run_id,
                service=service,
                classification="price_driven",
                action=ActionType.INVESTIGATE_CONFIG,
                params={
                    "service": service,
                    "cost_per_1k_delta": cost_delta_1k,
                    "cost_per_1k_now": cost_per_1k_now,
                    "cost_per_1k_baseline": cost_per_1k_base,
                },
                estimated_savings_hourly=0.0,
                estimated_risk=0.0,
                reason=reason,
                sim_ts=sim_ts,
                status=ProposalStatus.PROPOSED,
            )

        # -------------------------------------------------------------
        # 2. Volume drop with cost flat: Alert only
        # -------------------------------------------------------------
        if alert_type == "volume_drop_cost_flat":
            reason = (
                f"Volume dropped by {abs(pct_volume_change):.1f}% on {service} but cost "
                f"remained flat. Possible orphaned provisioned capacity or fixed charges."
            )
            return RemediationProposal(
                detection_id=detection_id,
                run_id=run_id,
                service=service,
                classification="volume_drop_cost_flat",
                action=ActionType.ALERT_ONLY,
                params={"service": service, "pct_volume_drop": abs(pct_volume_change)},
                estimated_savings_hourly=0.0,
                estimated_risk=0.0,
                reason=reason,
                sim_ts=sim_ts,
                status=ProposalStatus.PROPOSED,
            )

        # -------------------------------------------------------------
        # 3. Traffic-driven anomaly -> RateLimit or ConcurrencyCap
        # -------------------------------------------------------------
        # Compute throttle factor: bring volume back towards normal
        # If volume spiked by 150% (multiplier 2.5), factor = 1.0 / 2.5 = 0.4
        # Clamp to settings.max_throttle_depth (never below 0.5)
        volume_multiplier = max(1.0, 1.0 + (pct_volume_change / 100.0))
        raw_factor = 1.0 / volume_multiplier
        factor = max(self.settings.max_throttle_depth, min(0.9, raw_factor))

        # Estimate savings: hourly excess cost reduction
        # 12 ticks/hr. Approximate current excess rate from pct_cost_change
        # If cost rose by pct_cost_change %, fractional reduction is (1 - factor)
        estimated_excess_hourly = max(
            0.50,
            (pct_cost_change / 100.0) * (cost_per_1k_now * 50.0) * 12.0
        )
        estimated_savings_hourly = estimated_excess_hourly * (1.0 - factor)

        # Estimated risk: legitimate traffic impacted at margin
        # Shallow throttle (<20% reduction) -> risk <= 5%
        # Deeper throttle (>=40% reduction) -> risk <= 12%
        throttle_depth = 1.0 - factor
        estimated_risk = min(0.20, 0.03 + (throttle_depth * 0.15))

        # Service-specific action type
        if service in ("AmazonApiGateway", "EC2 Data Transfer", "S3 PUT", "DynamoDB"):
            action = ActionType.RATE_LIMIT
            action_name = "RateLimit"
        else:
            action = ActionType.CONCURRENCY_CAP
            action_name = "ConcurrencyCap"

        duration_min = min(self.settings.max_mitigation_duration_min, 30.0)
        pct_throttle = (1.0 - factor) * 100.0

        reason = (
            f"High traffic throughput on {service} (+{pct_volume_change:.1f}% volume, "
            f"+{pct_cost_change:.1f}% cost). Proposing {action_name} of {pct_throttle:.0f}% "
            f"(multiplier {factor:.2f}) for {duration_min:.0f} sim-min to avoid "
            f"~USD {estimated_savings_hourly:.2f}/hr excess cost."
        )

        return RemediationProposal(
            detection_id=detection_id,
            run_id=run_id,
            service=service,
            classification="traffic_driven",
            action=action,
            params={
                "service": service,
                "factor": round(factor, 3),
                "duration_min": duration_min,
                "throttle_pct": round(pct_throttle, 1),
            },
            estimated_savings_hourly=estimated_savings_hourly,
            estimated_risk=estimated_risk,
            reason=reason,
            sim_ts=sim_ts,
            status=ProposalStatus.PROPOSED,
        )


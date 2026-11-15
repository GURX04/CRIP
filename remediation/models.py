"""
remediation/models.py
=====================
Data models, enums, and schemas for the CRIP live remediation engine.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    RATE_LIMIT = "rate_limit"
    CONCURRENCY_CAP = "concurrency_cap"
    INVESTIGATE_CONFIG = "investigate_configuration"
    ALERT_ONLY = "alert_only"


class ProposalStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPLIED = "applied"
    BLOCKED_BY_GUARDRAIL = "blocked_by_guardrail"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    DRY_RUN_RECORDED = "dry_run_recorded"


class RemediationMode(str, Enum):
    DRY_RUN = "DRY_RUN"
    ENFORCE = "ENFORCE"


@dataclass
class GuardrailDecision:
    """Audit log for a single guardrail evaluation."""
    guardrail: str
    allowed: bool
    reason: str


@dataclass
class RemediationProposal:
    """
    A proposed remediation action derived from a confirmed Detection.
    """
    detection_id: str
    run_id: str
    service: str
    classification: str  # "traffic_driven" | "price_driven" | "volume_drop_cost_flat" | "unknown"
    action: ActionType
    params: Dict[str, Any]
    estimated_savings_hourly: float
    estimated_risk: float
    reason: str
    status: ProposalStatus = ProposalStatus.PROPOSED
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sim_ts: Optional[datetime] = None
    created_at_wall_ms: int = field(
        default_factory=lambda: int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    )
    guardrail_logs: List[GuardrailDecision] = field(default_factory=list)
    remediation_id: Optional[str] = None  # Generator mitigation_id if applied
    mitigation_applied_sim_ts: Optional[datetime] = None
    mitigation_rollback_sim_ts: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "detection_id": self.detection_id,
            "run_id": self.run_id,
            "service": self.service,
            "classification": self.classification,
            "action": self.action.value,
            "params": self.params,
            "estimated_savings_hourly": round(self.estimated_savings_hourly, 4),
            "estimated_risk": round(self.estimated_risk, 4),
            "reason": self.reason,
            "status": self.status.value,
            "sim_ts": self.sim_ts.isoformat() if self.sim_ts else None,
            "created_at_wall_ms": self.created_at_wall_ms,
            "guardrail_logs": [asdict(g) for g in self.guardrail_logs],
            "remediation_id": self.remediation_id,
            "mitigation_applied_sim_ts": (
                self.mitigation_applied_sim_ts.isoformat()
                if self.mitigation_applied_sim_ts
                else None
            ),
            "mitigation_rollback_sim_ts": (
                self.mitigation_rollback_sim_ts.isoformat()
                if self.mitigation_rollback_sim_ts
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RemediationProposal:
        sim_ts = None
        if data.get("sim_ts"):
            sim_ts = datetime.fromisoformat(data["sim_ts"])
        app_ts = None
        if data.get("mitigation_applied_sim_ts"):
            app_ts = datetime.fromisoformat(data["mitigation_applied_sim_ts"])
        roll_ts = None
        if data.get("mitigation_rollback_sim_ts"):
            roll_ts = datetime.fromisoformat(data["mitigation_rollback_sim_ts"])

        guardrail_logs = [
            GuardrailDecision(**g) for g in data.get("guardrail_logs", [])
        ]

        return cls(
            proposal_id=data["proposal_id"],
            detection_id=data["detection_id"],
            run_id=data["run_id"],
            service=data["service"],
            classification=data["classification"],
            action=ActionType(data["action"]),
            params=data.get("params", {}),
            estimated_savings_hourly=float(data["estimated_savings_hourly"]),
            estimated_risk=float(data["estimated_risk"]),
            reason=data["reason"],
            status=ProposalStatus(data["status"]),
            sim_ts=sim_ts,
            created_at_wall_ms=int(data.get("created_at_wall_ms", 0)),
            guardrail_logs=guardrail_logs,
            remediation_id=data.get("remediation_id"),
            mitigation_applied_sim_ts=app_ts,
            mitigation_rollback_sim_ts=roll_ts,
        )


@dataclass
class RemediationResult:
    """
    Savings & collateral measurement for an applied mitigation.
    """
    action_id: str
    proposal_id: str
    service: str
    actual_cost_usd: float
    counterfactual_cost_usd: float
    estimated_savings_usd: float
    legitimate_requests_dropped: float
    collateral_penalty_usd: float
    net_benefit_usd: float
    estimation_method: str = "Counterfactual detector baseline vs observed cost"
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    calculated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "action_id": self.action_id,
            "proposal_id": self.proposal_id,
            "service": self.service,
            "actual_cost_usd": round(self.actual_cost_usd, 6),
            "counterfactual_cost_usd": round(self.counterfactual_cost_usd, 6),
            "estimated_savings_usd": round(self.estimated_savings_usd, 6),
            "legitimate_requests_dropped": round(self.legitimate_requests_dropped, 2),
            "collateral_penalty_usd": round(self.collateral_penalty_usd, 6),
            "net_benefit_usd": round(self.net_benefit_usd, 6),
            "estimation_method": self.estimation_method,
            "calculated_at": self.calculated_at.isoformat(),
        }


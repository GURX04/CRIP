"""
remediation/guardrails.py
=========================
Independent, configurable safety guardrails protecting the live system.

Guardrails enforced:
1. Kill Switch: Setting or API toggle blocking all actions immediately.
2. Minimum Severity Floor: Anomaly severity must exceed threshold (default 60).
3. Maximum Throttle Depth: Traffic factor cannot drop below floor (default 0.50).
4. Per-Service Cooldown: Minimum simulated interval between actions on the same service (default 15 min).
5. Global Rate Cap: Maximum automated actions permitted per simulated hour (default 6).
6. Human Approval Policy: High severity (>= 80) or high risk (>= 0.15) actions require manual operator approval.
7. Auto-Rollback: Actions revert automatically upon anomaly resolution or timeout.

Every guardrail evaluation produces a structured GuardrailDecision log.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from remediation.config import RemediationSettings, get_remediation_settings
from remediation.models import (
    ActionType,
    GuardrailDecision,
    ProposalStatus,
    RemediationProposal,
)

log = logging.getLogger(__name__)


class RemediationGuardrails:
    """
    Evaluates safety constraints before any remediation action is executed.
    """

    def __init__(self, settings: Optional[RemediationSettings] = None) -> None:
        self.settings = settings or get_remediation_settings()
        self._kill_switch: bool = self.settings.kill_switch

        # Tracking state (isolated by simulation run)
        self._service_last_action_ts: Dict[Tuple[str, str], datetime] = {}
        self._global_action_timestamps: Dict[str, List[datetime]] = {}
        self._active_mitigations: Dict[str, RemediationProposal] = {}
        

    @property
    def kill_switch(self) -> bool:
        return self._kill_switch

    def set_kill_switch(self, enabled: bool) -> None:
        self._kill_switch = enabled
        log.warning("remediation_kill_switch_changed enabled=%s", enabled)

    def evaluate(
        self,
        proposal: RemediationProposal,
        severity: float,
        sim_ts: Optional[datetime] = None,
    ) -> Tuple[bool, List[GuardrailDecision], ProposalStatus]:
        """
        Evaluate all guardrails against a proposal.

        Returns
        -------
        (allowed, decisions, final_status)
        """
        decisions: List[GuardrailDecision] = []
        now = sim_ts or proposal.sim_ts or datetime.now(tz=timezone.utc)
        run_id = proposal.run_id or "default"

        # Non-executable action types (informational only)
        if proposal.action in (ActionType.INVESTIGATE_CONFIG, ActionType.ALERT_ONLY):
            decisions.append(GuardrailDecision(
                guardrail="action_type",
                allowed=True,
                reason=f"Informational action {proposal.action.value} requires no resource modification.",
            ))
            return True, decisions, ProposalStatus.PROPOSED

        # -------------------------------------------------------------------
        # 1. Kill Switch
        # -------------------------------------------------------------------
        if self._kill_switch:
            decisions.append(GuardrailDecision(
                guardrail="kill_switch",
                allowed=False,
                reason="Emergency master kill switch is engaged.",
            ))
            return False, decisions, ProposalStatus.BLOCKED_BY_GUARDRAIL

        decisions.append(GuardrailDecision(
            guardrail="kill_switch",
            allowed=True,
            reason="Master kill switch is disengaged.",
        ))

        # -------------------------------------------------------------------
        # 2. Severity Floor
        # -------------------------------------------------------------------
        if severity < self.settings.min_severity:
            decisions.append(GuardrailDecision(
                guardrail="min_severity",
                allowed=False,
                reason=(
                    f"Anomaly severity {severity:.1f} is below minimum threshold "
                    f"of {self.settings.min_severity:.1f}."
                ),
            ))
            return False, decisions, ProposalStatus.BLOCKED_BY_GUARDRAIL

        decisions.append(GuardrailDecision(
            guardrail="min_severity",
            allowed=True,
            reason=f"Anomaly severity {severity:.1f} exceeds threshold {self.settings.min_severity:.1f}.",
        ))

        # -------------------------------------------------------------------
        # 3. Maximum Throttle Depth
        # -------------------------------------------------------------------
        factor = float(proposal.params.get("factor", 1.0))
        if factor < self.settings.max_throttle_depth:
            decisions.append(GuardrailDecision(
                guardrail="throttle_depth",
                allowed=False,
                reason=(
                    f"Throttle factor {factor:.2f} exceeds maximum allowed depth "
                    f"(minimum floor {self.settings.max_throttle_depth:.2f})."
                ),
            ))
            return False, decisions, ProposalStatus.BLOCKED_BY_GUARDRAIL

        decisions.append(GuardrailDecision(
            guardrail="throttle_depth",
            allowed=True,
            reason=f"Throttle factor {factor:.2f} satisfies maximum depth limit.",
        ))

        # -------------------------------------------------------------------
        # 4. Per-Service Cooldown
        # -------------------------------------------------------------------
        svc = proposal.service
        service_key = (run_id, svc)

        if service_key in self._service_last_action_ts:
            last_ts = self._service_last_action_ts[service_key]

            # Ignore stale timestamps if simulation time moved backwards.
            if now < last_ts:
                self._service_last_action_ts.pop(service_key, None)
            else:
                elapsed_min = (now - last_ts).total_seconds() / 60.0

                if elapsed_min < self.settings.cooldown_sim_min:
                    remaining = self.settings.cooldown_sim_min - elapsed_min
                    decisions.append(GuardrailDecision(
                        guardrail="service_cooldown",
                        allowed=False,
                        reason=(
                            f"Service '{svc}' is in cooldown. Last action was "
                            f"{elapsed_min:.1f} sim-min ago; "
                            f"{remaining:.1f} min remaining."
                        ),
                    ))
                    return False, decisions, ProposalStatus.BLOCKED_BY_GUARDRAIL

        decisions.append(GuardrailDecision(
            guardrail="service_cooldown",
            allowed=True,
            reason=f"Service '{svc}' cooldown satisfied.",
        ))

        # -------------------------------------------------------------------
        # 5. Global Hourly Rate Cap
        # -------------------------------------------------------------------
        one_hour_ago = now - timedelta(hours=1)

        run_actions = self._global_action_timestamps.get(run_id, [])
        recent_actions = [
            t for t in run_actions
            if one_hour_ago <= t <= now
        ]

        self._global_action_timestamps[run_id] = recent_actions

        if len(recent_actions) >= self.settings.max_actions_per_hour:
            decisions.append(GuardrailDecision(
                guardrail="global_rate_cap",
                allowed=False,
                reason=(
                    f"Global limit of {self.settings.max_actions_per_hour} actions/hr "
                    f"reached ({len(recent_actions)} in the last simulated hour)."
                ),
            ))
            return False, decisions, ProposalStatus.BLOCKED_BY_GUARDRAIL

        decisions.append(GuardrailDecision(
            guardrail="global_rate_cap",
            allowed=True,
            reason=(
                f"Global rate cap satisfied "
                f"({len(recent_actions)}/{self.settings.max_actions_per_hour} used)."
            ),
        ))

        # -------------------------------------------------------------------
        # 6. Approval Policy
        # -------------------------------------------------------------------
        is_high_severity = severity > self.settings.auto_approve_max_severity
        is_high_risk = proposal.estimated_risk > self.settings.auto_approve_max_risk
        is_reversible = proposal.action in (ActionType.RATE_LIMIT, ActionType.CONCURRENCY_CAP)

        if is_high_severity or is_high_risk or not is_reversible:
            reasons = []
            if is_high_severity:
                reasons.append(f"severity {severity:.1f} > {self.settings.auto_approve_max_severity:.1f}")
            if is_high_risk:
                reasons.append(f"estimated risk {proposal.estimated_risk:.2f} > {self.settings.auto_approve_max_risk:.2f}")
            if not is_reversible:
                reasons.append("action is irreversible")

            decisions.append(GuardrailDecision(
                guardrail="approval_policy",
                allowed=True,
                reason=f"Requires operator approval because: {', '.join(reasons)}.",
            ))
            return True, decisions, ProposalStatus.PENDING_APPROVAL

        # Auto-approved!
        decisions.append(GuardrailDecision(
            guardrail="approval_policy",
            allowed=True,
            reason="Auto-approval criteria met: low risk, moderate severity, reversible action.",
        ))

        if self.settings.mode.upper() == "ENFORCE":
            final_status = ProposalStatus.APPLIED
        else:
            final_status = ProposalStatus.DRY_RUN_RECORDED

        return True, decisions, final_status

    def record_action_applied(self, proposal: RemediationProposal, sim_ts: Optional[datetime] = None) -> None:
        """Update cooldown and rate limit timestamps after an action is applied."""
        now = sim_ts or proposal.sim_ts or datetime.now(tz=timezone.utc)
        run_id = proposal.run_id or "default"

        self._service_last_action_ts[(run_id, proposal.service)] = now
        self._global_action_timestamps.setdefault(run_id, []).append(now)
        self._active_mitigations[proposal.proposal_id] = proposal

    def record_action_ended(self, proposal_id: str) -> None:
        """Remove from active tracking."""
        self._active_mitigations.pop(proposal_id, None)

    def get_active_proposals(self) -> List[RemediationProposal]:
        return list(self._active_mitigations.values())

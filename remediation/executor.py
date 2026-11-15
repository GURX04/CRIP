"""
remediation/executor.py
=======================
Action executor for CRIP automated remediation.

Guarantees:
- Mode Isolation: In DRY_RUN, zero changes are made to live systems.
- Idempotency: The same proposal_id is never applied twice.
- Reversibility: Every applied action can be rolled back via DELETE /mitigate/{id}.
- Optional AWS LocalStack Integration: Dispatches signed HMAC payload in dry-run.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Set
import urllib.request
import urllib.error

from remediation.config import RemediationSettings, get_remediation_settings
from remediation.models import (
    ActionType,
    ProposalStatus,
    RemediationMode,
    RemediationProposal,
)

log = logging.getLogger(__name__)


class RemediationExecutor:
    """
    Executes and rolls back remediation actions idempotently.
    """

    def __init__(self, settings: Optional[RemediationSettings] = None) -> None:
        self.settings = settings or get_remediation_settings()
        self._applied_proposals: Dict[str, RemediationProposal] = {}
        self._applied_ids: Set[str] = set()

    def execute(
        self,
        proposal: RemediationProposal,
        mode_override: Optional[str] = None,
        sim_ts: Optional[datetime] = None,
    ) -> bool:
        """
        Execute an approved remediation proposal.

        Parameters
        ----------
        proposal:
            The proposal to execute.
        mode_override:
            Optional mode ("DRY_RUN" or "ENFORCE") overriding system settings.
        sim_ts:
            Optional current simulated timestamp.
        """
        mode_str = (mode_override or self.settings.mode).upper()
        mode = RemediationMode.ENFORCE if mode_str == "ENFORCE" else RemediationMode.DRY_RUN

        # Idempotency check: never apply the exact same proposal twice
        if proposal.proposal_id in self._applied_ids:
            log.warning("remediation_already_applied proposal_id=%s", proposal.proposal_id)
            return True

        if proposal.action in (ActionType.INVESTIGATE_CONFIG, ActionType.ALERT_ONLY):
            # Informational proposals require no active system modification
            log.info(
                "remediation_informational_only proposal_id=%s action=%s",
                proposal.proposal_id,
                proposal.action.value,
            )
            return True

        now = sim_ts or proposal.sim_ts or datetime.now(tz=timezone.utc)

        # -------------------------------------------------------------------
        # DRY_RUN Execution
        # -------------------------------------------------------------------
        if mode == RemediationMode.DRY_RUN:
            log.info(
                "remediation_dry_run_recorded proposal_id=%s service=%s action=%s",
                proposal.proposal_id,
                proposal.service,
                proposal.action.value,
            )
            proposal.status = ProposalStatus.DRY_RUN_RECORDED
            proposal.remediation_id = f"dry-run-{proposal.proposal_id[:8]}"
            proposal.mitigation_applied_sim_ts = now
            self._applied_ids.add(proposal.proposal_id)
            self._applied_proposals[proposal.proposal_id] = proposal

            # Optional LocalStack Lambda dry-run notification with HMAC signature
            self._notify_localstack_dry_run(proposal)
            return True

        # -------------------------------------------------------------------
        # ENFORCE Execution -> Generator POST /mitigate
        # -------------------------------------------------------------------
        gen_url = f"{self.settings.generator_url.rstrip('/')}/mitigate"
        mitigation_id = f"mit-{proposal.proposal_id[:8]}"
        payload = {
            "service": proposal.service,
            "type": "rate_limit",
            "factor": float(proposal.params.get("factor", 0.6)),
            "duration_min": float(proposal.params.get("duration_min", 30.0)),
            "mitigation_id": mitigation_id,
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                gen_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status in (200, 201):
                    proposal.status = ProposalStatus.APPLIED
                    proposal.remediation_id = mitigation_id
                    proposal.mitigation_applied_sim_ts = now
                    self._applied_ids.add(proposal.proposal_id)
                    self._applied_proposals[proposal.proposal_id] = proposal

                    log.info(
                        "remediation_enforced_successfully proposal_id=%s mitigation_id=%s",
                        proposal.proposal_id,
                        mitigation_id,
                    )
                    return True
                else:
                    log.error("generator_mitigate_bad_status status=%s", resp.status)
                    return False
        except Exception as exc:
            log.error("generator_mitigate_failed error=%s url=%s", str(exc), gen_url)
            return False

    def rollback(
        self,
        proposal: RemediationProposal,
        sim_ts: Optional[datetime] = None,
    ) -> bool:
        """
        Revert an active remediation action.
        """
        now = sim_ts or datetime.now(tz=timezone.utc)

        if proposal.status == ProposalStatus.DRY_RUN_RECORDED:
            proposal.status = ProposalStatus.ROLLED_BACK
            proposal.mitigation_rollback_sim_ts = now
            self._applied_ids.discard(proposal.proposal_id)
            self._applied_proposals.pop(proposal.proposal_id, None)
            log.info("remediation_dry_run_rolled_back proposal_id=%s", proposal.proposal_id)
            return True

        if proposal.status != ProposalStatus.APPLIED or not proposal.remediation_id:
            log.warning("remediation_rollback_not_applicable status=%s", proposal.status.value)
            return False

        gen_url = f"{self.settings.generator_url.rstrip('/')}/mitigate/{proposal.remediation_id}"

        try:
            req = urllib.request.Request(gen_url, method="DELETE")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status in (200, 204, 404):  # 404 considered already expired
                    proposal.status = ProposalStatus.ROLLED_BACK
                    proposal.mitigation_rollback_sim_ts = now
                    self._applied_ids.discard(proposal.proposal_id)
                    self._applied_proposals.pop(proposal.proposal_id, None)
                    log.info("remediation_rolled_back proposal_id=%s", proposal.proposal_id)
                    return True
                else:
                    log.error("generator_rollback_bad_status status=%s", resp.status)
                    return False
        except Exception as exc:
            log.error("generator_rollback_failed error=%s url=%s", str(exc), gen_url)
            return False

    def _notify_localstack_dry_run(self, proposal: RemediationProposal) -> None:
        """Dispatch optional signed HMAC dry-run audit payload to LocalStack."""
        if not self.settings.localstack_url:
            return

        payload_dict = {
            "proposal_id": proposal.proposal_id,
            "service": proposal.service,
            "action": proposal.action.value,
            "params": proposal.params,
            "dry_run": True,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        raw_body = json.dumps(payload_dict)
        signature = hmac.new(
            self.settings.hmac_secret.encode("utf-8"),
            raw_body.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        target_url = f"{self.settings.localstack_url.rstrip('/')}/remediation"
        try:
            req = urllib.request.Request(
                target_url,
                data=raw_body.encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-CRIP-Signature": signature,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1) as _:
                pass
        except Exception:
            # LocalStack is optional / best-effort in dry-run
            pass

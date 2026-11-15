
"""
remediation/worker.py
=====================
Live streaming remediation worker.

Consumes:
- `crip:detections` via Redis consumer group `crip:remediation:group`.
- `crip:remediation_approvals` for operator decisions.
- `crip:mitigations` for mitigation lifecycle events.

Emits:
- `crip:remediation_proposals`
- `crip:remediation_results`
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import redis

from remediation.config import RemediationSettings, get_remediation_settings
from remediation.executor import RemediationExecutor
from remediation.guardrails import RemediationGuardrails
from remediation.impact import RemediationImpactCalculator
from remediation.models import (
    ProposalStatus,
    RemediationProposal,
)
from remediation.planner import RemediationPlanner

log = logging.getLogger(__name__)


class RemediationWorker:
    """Continuous worker handling detection and remediation events."""

    def __init__(
        self,
        settings: Optional[RemediationSettings] = None,
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        self.settings = settings or get_remediation_settings()
        self.planner = RemediationPlanner(self.settings)
        self.guardrails = RemediationGuardrails(self.settings)
        self.executor = RemediationExecutor(self.settings)
        self.impact = RemediationImpactCalculator(self.settings)

        self.redis = redis_client
        self._running = False
        self._active_proposals: Dict[str, RemediationProposal] = {}
        self._detection_to_proposal: Dict[str, str] = {}

        self._approval_last_id = "0-0"
        self._mitigation_last_id = "0-0"

    def connect_redis(self) -> None:
        if self.redis is None:
            self.redis = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
            )

        try:
            self.redis.xgroup_create(
                self.settings.redis_stream_detections,
                self.settings.redis_group,
                id="$",
                mkstream=True,
            )
            log.info(
                "consumer_group_created group=%s",
                self.settings.redis_group,
            )
        except Exception:
            # Consumer group may already exist.
            pass

    def process_mitigation_event(self, event: dict) -> None:
        """Update proposals when generator mitigations end."""

        if event.get("event") != "end":
            return

        mitigation_id = event.get("mitigation_id")
        end_reason = event.get("end_reason")

        if not mitigation_id or end_reason != "expired":
            return

        for proposal in list(self._active_proposals.values()):
            if proposal.remediation_id != mitigation_id:
                continue

            if proposal.status != ProposalStatus.APPLIED:
                return

            proposal.status = ProposalStatus.EXPIRED

            log.info(
                "remediation_expired proposal_id=%s mitigation_id=%s",
                proposal.proposal_id,
                mitigation_id,
            )

            self.impact.persist_proposal_action(proposal)
            self._publish_proposal(proposal)
            return

    def _process_mitigation_events(self) -> None:
        """Consume generator mitigation lifecycle events."""

        entries = self.redis.xread(
            {
                self.settings.redis_stream_mitigations:
                    self._mitigation_last_id
            },
            count=50,
            block=1,
        )

        for stream_name, messages in entries:
            for msg_id, fields in messages:
                self._mitigation_last_id = msg_id
                raw_payload = fields.get("payload")

                if not raw_payload:
                    continue

                try:
                    event = json.loads(raw_payload)
                    self.process_mitigation_event(event)
                except Exception:
                    log.exception("mitigation_event_processing_failed")

    def process_detection(
        self,
        detection: dict,
    ) -> Optional[RemediationProposal]:
        """Process a detection and create or update remediation."""

        severity = float(detection.get("severity", 0.0))
        sim_ts = None

        if detection.get("sim_ts"):
            try:
                sim_ts = datetime.fromisoformat(
                    str(detection["sim_ts"])
                )
            except Exception:
                pass

        det_status = str(detection.get("status", "open"))
        det_id = str(detection.get("detection_id", ""))

        # 1. Handle closed detections without creating new proposals.
        if det_status == "closed":
            prop_id = self._detection_to_proposal.get(det_id)
            proposal = (
                self._active_proposals.get(prop_id)
                if prop_id
                else None
            )

            if proposal is None:
                log.info(
                    "detection_closed_no_active_proposal detection_id=%s",
                    det_id,
                )
                return None

            if proposal.status == ProposalStatus.APPLIED:
                self._handle_rollback(proposal, sim_ts)

            elif proposal.status == ProposalStatus.PENDING_APPROVAL:
                proposal.status = ProposalStatus.REJECTED
                self._active_proposals.pop(proposal.proposal_id, None)
                self._detection_to_proposal.pop(det_id, None)
                self.impact.persist_proposal_action(proposal)
                self._publish_proposal(proposal)

            else:
                self._active_proposals.pop(proposal.proposal_id, None)
                self._detection_to_proposal.pop(det_id, None)

            return proposal

        # 2. Plan remediation proposal.
        proposal = self.planner.plan_detection(detection)

        # 3. Guardrail evaluation.
        allowed, decisions, initial_status = self.guardrails.evaluate(
            proposal,
            severity,
            sim_ts,
        )

        proposal.guardrail_logs = decisions
        proposal.status = initial_status

        # 4. Track proposals requiring approval or execution.
        if initial_status == ProposalStatus.PENDING_APPROVAL:
            self._active_proposals[proposal.proposal_id] = proposal
            self._detection_to_proposal[
                proposal.detection_id
            ] = proposal.proposal_id

        elif allowed and initial_status in (
            ProposalStatus.APPLIED,
            ProposalStatus.DRY_RUN_RECORDED,
        ):
            executed = self.executor.execute(
                proposal,
                sim_ts=sim_ts,
            )

            if executed:
                self.guardrails.record_action_applied(
                    proposal,
                    sim_ts,
                )
                self._active_proposals[proposal.proposal_id] = proposal
                self._detection_to_proposal[
                    proposal.detection_id
                ] = proposal.proposal_id
            else:
                proposal.status = ProposalStatus.BLOCKED_BY_GUARDRAIL

        # 5. Persist and publish proposal.
        self.impact.persist_proposal_action(proposal)
        self._publish_proposal(proposal)

        return proposal

    def approve_proposal(
        self,
        proposal_id: str,
        sim_ts: Optional[datetime] = None,
    ) -> bool:
        """Operator manually approves a pending proposal."""

        if proposal_id not in self._active_proposals:
            log.warning(
                "proposal_not_found_for_approval proposal_id=%s",
                proposal_id,
            )
            return False

        proposal = self._active_proposals[proposal_id]

        if proposal.status != ProposalStatus.PENDING_APPROVAL:
            return False

        executed = self.executor.execute(
            proposal,
            mode_override="ENFORCE",
            sim_ts=sim_ts,
        )

        if executed:
            self.guardrails.record_action_applied(proposal, sim_ts)
            self.impact.persist_proposal_action(proposal)
            self._publish_proposal(proposal)
            return True

        return False

    def reject_proposal(self, proposal_id: str) -> bool:
        """Operator rejects a pending proposal."""

        if proposal_id not in self._active_proposals:
            return False

        proposal = self._active_proposals[proposal_id]
        proposal.status = ProposalStatus.REJECTED

        self._active_proposals.pop(proposal_id, None)
        self._detection_to_proposal.pop(
            proposal.detection_id,
            None,
        )

        self.impact.persist_proposal_action(proposal)
        self._publish_proposal(proposal)

        return True

    def _handle_rollback(
        self,
        proposal: RemediationProposal,
        sim_ts: Optional[datetime] = None,
    ) -> None:
        """Perform rollback and compute final savings impact."""

        rolled_back = self.executor.rollback(
            proposal,
            sim_ts=sim_ts,
        )

        if not rolled_back:
            log.error(
                "remediation_rollback_failed proposal_id=%s",
                proposal.proposal_id,
            )
            return

        self.guardrails.record_action_ended(proposal.proposal_id)

        self._active_proposals.pop(proposal.proposal_id, None)
        self._detection_to_proposal.pop(
            proposal.detection_id,
            None,
        )

        impact_result = self.impact.calculate_impact(
            proposal=proposal,
            actual_cost_usd=(
                proposal.estimated_savings_hourly / 12.0 * 2.0
            ),
            baseline_cost_usd=(
                proposal.estimated_savings_hourly / 12.0 * 1.5
            ),
        )

        self.impact.persist_impact_result(impact_result)
        self.impact.persist_proposal_action(proposal)

        self._publish_proposal(proposal)
        self._publish_result(impact_result)

    def _publish_proposal(
        self,
        proposal: RemediationProposal,
    ) -> None:
        if self.redis is None:
            return

        try:
            self.redis.xadd(
                self.settings.redis_stream_proposals,
                {"payload": json.dumps(proposal.to_dict())},
                maxlen=10_000,
                approximate=True,
            )
        except Exception as exc:
            log.error("redis_publish_proposal_failed: %s", exc)

    def _publish_result(self, result) -> None:
        if self.redis is None:
            return

        try:
            self.redis.xadd(
                self.settings.redis_stream_results,
                {"payload": json.dumps(result.to_dict())},
                maxlen=10_000,
                approximate=True,
            )
        except Exception as exc:
            log.error("redis_publish_result_failed: %s", exc)

    def _process_approval_commands(self) -> None:
        """Process dashboard approval/rejection commands."""

        entries = self.redis.xread(
            {
                self.settings.redis_stream_approvals:
                    self._approval_last_id
            },
            count=10,
            block=1,
        )

        for stream_name, messages in entries:
            for msg_id, fields in messages:
                self._approval_last_id = msg_id

                try:
                    command = json.loads(
                        fields.get("payload", "{}")
                    )

                    proposal_id = command.get("proposal_id")
                    action = command.get("action")

                    if action == "approve":
                        success = self.approve_proposal(proposal_id)
                    elif action == "reject":
                        success = self.reject_proposal(proposal_id)
                    else:
                        success = False

                    log.info(
                        "approval_command_processed proposal_id=%s "
                        "action=%s success=%s",
                        proposal_id,
                        action,
                        success,
                    )

                except Exception:
                    log.exception("approval_command_failed")

    def run(self) -> None:
        """Main listening loop."""

        self.connect_redis()
        self._running = True

        log.info(
            "remediation_worker_started mode=%s stream=%s group=%s",
            self.settings.mode,
            self.settings.redis_stream_detections,
            self.settings.redis_group,
        )

        while self._running:
            try:
                self._process_approval_commands()
                self._process_mitigation_events()

                entries = self.redis.xreadgroup(
                    groupname=self.settings.redis_group,
                    consumername=self.settings.redis_consumer,
                    streams={
                        self.settings.redis_stream_detections: ">"
                    },
                    count=10,
                    block=1000,
                )

                if not entries:
                    continue

                for stream_name, msg_list in entries:
                    for msg_id, fields in msg_list:
                        raw_payload = fields.get("payload")

                        if raw_payload:
                            try:
                                det_data = json.loads(raw_payload)
                                self.process_detection(det_data)
                            except Exception as ex:
                                log.error(
                                    "detection_processing_failed: %s",
                                    ex,
                                )

                        self.redis.xack(
                            stream_name,
                            self.settings.redis_group,
                            msg_id,
                        )

            except Exception as exc:
                if self._running:
                    log.error("remediation_loop_error: %s", exc)
                    time.sleep(1)

    def stop(self) -> None:
        self._running = False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    worker = RemediationWorker()

    def _sig_handler(sig, frame):
        log.info("remediation_shutdown_signal signal=%s", sig)
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    worker.run()


if __name__ == "__main__":
    main()

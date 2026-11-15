"""
remediation/worker.py
=====================
Live streaming remediation worker.

Consumes:
- `crip:detections` via Redis consumer group `crip:remediation:group`.

Emits:
- `crip:remediation_proposals`
- `crip:remediation_results`
- Interacts with generator /mitigate API when in ENFORCE mode.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
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
    """
    Continuous worker loop handling detection consumption, safety guardrails,
    action execution, and auto-rollback monitoring.
    """

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
        self._active_proposals: Dict[str, RemediationProposal] = {}  # proposal_id -> proposal
        self._detection_to_proposal: Dict[str, str] = {}            # detection_id -> proposal_id

    def connect_redis(self) -> None:
        if self.redis is None:
            self.redis = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
            )

        # Ensure consumer group exists
        try:
            self.redis.xgroup_create(
                self.settings.redis_stream_detections,
                self.settings.redis_group,
                id="$",
                mkstream=True,
            )
            log.info("consumer_group_created group=%s", self.settings.redis_group)
        except Exception:
            # Group already exists
            pass

    def process_detection(self, detection: dict) -> RemediationProposal:
        """
        Process a single detection dictionary and execute if eligible.
        """
        severity = float(detection.get("severity", 0.0))
        sim_ts = None
        if detection.get("sim_ts"):
            try:
                sim_ts = datetime.fromisoformat(str(detection["sim_ts"]))
            except Exception:
                pass

        det_status = str(detection.get("status", "open"))
        det_id = str(detection.get("detection_id", ""))

        # 1. Check for anomaly close -> Auto-Rollback
        if det_status == "closed" and det_id in self._detection_to_proposal:
            prop_id = self._detection_to_proposal[det_id]
            if prop_id in self._active_proposals:
                proposal = self._active_proposals[prop_id]
                self._handle_rollback(proposal, sim_ts)
                return proposal

        # 2. Plan remediation proposal
        proposal = self.planner.plan_detection(detection)

        # 3. Guardrail evaluation
        allowed, decisions, initial_status = self.guardrails.evaluate(proposal, severity, sim_ts)
        proposal.guardrail_logs = decisions
        proposal.status = initial_status

        # 4. Execution if approved
        if allowed and initial_status in (ProposalStatus.APPLIED, ProposalStatus.DRY_RUN_RECORDED):
            executed = self.executor.execute(proposal, sim_ts=sim_ts)
            if executed:
                self.guardrails.record_action_applied(proposal, sim_ts)
                self._active_proposals[proposal.proposal_id] = proposal
                self._detection_to_proposal[proposal.detection_id] = proposal.proposal_id
            else:
                proposal.status = ProposalStatus.BLOCKED_BY_GUARDRAIL

        # 5. Persist to DB and publish proposal to Redis stream
        self.impact.persist_proposal_action(proposal)
        self._publish_proposal(proposal)

        return proposal

    def approve_proposal(self, proposal_id: str, sim_ts: Optional[datetime] = None) -> bool:
        """Operator manually approves a pending_approval proposal."""
        if proposal_id not in self._active_proposals:
            log.warning("proposal_not_found_for_approval proposal_id=%s", proposal_id)
            return False

        proposal = self._active_proposals[proposal_id]
        if proposal.status != ProposalStatus.PENDING_APPROVAL:
            return False

        executed = self.executor.execute(proposal, mode_override="ENFORCE", sim_ts=sim_ts)
        if executed:
            self.guardrails.record_action_applied(proposal, sim_ts)
            self.impact.persist_proposal_action(proposal)
            self._publish_proposal(proposal)
            return True
        return False

    def reject_proposal(self, proposal_id: str) -> bool:
        """Operator rejects a pending_approval proposal."""
        if proposal_id not in self._active_proposals:
            return False

        proposal = self._active_proposals[proposal_id]
        proposal.status = ProposalStatus.REJECTED
        self._active_proposals.pop(proposal_id, None)
        self.impact.persist_proposal_action(proposal)
        self._publish_proposal(proposal)
        return True

    def _handle_rollback(self, proposal: RemediationProposal, sim_ts: Optional[datetime] = None) -> None:
        """Perform rollback and compute final savings impact."""
        self.executor.rollback(proposal, sim_ts=sim_ts)
        self.guardrails.record_action_ended(proposal.proposal_id)
        self._active_proposals.pop(proposal.proposal_id, None)
        self._detection_to_proposal.pop(proposal.detection_id, None)

        # Calculate impact
        impact_result = self.impact.calculate_impact(
            proposal=proposal,
            actual_cost_usd=proposal.estimated_savings_hourly / 12.0 * 2.0,
            baseline_cost_usd=proposal.estimated_savings_hourly / 12.0 * 1.5,
        )

        self.impact.persist_impact_result(impact_result)
        self.impact.persist_proposal_action(proposal)
        self._publish_proposal(proposal)
        self._publish_result(impact_result)

    def _publish_proposal(self, proposal: RemediationProposal) -> None:
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
                entries = self.redis.xreadgroup(
                    groupname=self.settings.redis_group,
                    consumername=self.settings.redis_consumer,
                    streams={self.settings.redis_stream_detections: ">"},
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
                                log.error("detection_processing_failed: %s", ex)
                        self.redis.xack(stream_name, self.settings.redis_group, msg_id)

            except Exception as exc:
                if self._running:
                    log.error("remediation_loop_error: %s", exc)
                    time.sleep(1)

    def stop(self) -> None:
        self._running = False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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


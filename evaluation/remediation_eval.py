"""
evaluation/remediation_eval.py
==============================
Offline evaluation hooks for the CRIP Automated Remediation Engine.

Measures:
1. Mean Cost Avoided: Unmitigated cost minus mitigated cost across TEST seeds.
2. Time to Mitigation: Simulated minutes from anomaly injection start to action.
3. False Mitigation Rate: Actions erroneously applied on clean runs or hard negatives.
4. Collateral Impact: Legitimate requests impacted and economic penalty.
5. Price-Driven Refusal: Verifies that price-driven anomalies strictly produce zero throttles.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation.runner import OfflineReplayEngine
from evaluation.scenarios import ScenarioRegistry, generate_scenario
from remediation.config import RemediationSettings
from remediation.guardrails import RemediationGuardrails
from remediation.impact import RemediationImpactCalculator
from remediation.models import ActionType, ProposalStatus
from remediation.planner import RemediationPlanner

log = logging.getLogger(__name__)


class RemediationEvaluator:
    """
    Evaluates remediation policies offline against ground-truth scenarios.
    """

    def __init__(self, registry: Optional[ScenarioRegistry] = None) -> None:
        self.registry = registry or ScenarioRegistry()
        self.settings = RemediationSettings(mode="ENFORCE")
        self.planner = RemediationPlanner(self.settings)
        self.guardrails = RemediationGuardrails(self.settings)
        self.impact_calc = RemediationImpactCalculator(self.settings)
        self.engine = OfflineReplayEngine(warmup_ticks=100)

    def evaluate_scenarios(
        self,
        seeds: List[int],
        scenario_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate remediation across scenarios and seeds.
        """
        sc_names = scenario_names or ["quick_spike", "quick_cost_only", "clean"]
        results_records: List[Dict[str, Any]] = []

        total_injections = 0
        total_mitigated = 0
        cost_avoided_list: List[float] = []
        mitigation_latencies: List[float] = []
        false_mitigations = 0
        price_driven_throttles = 0
        collateral_penalties: List[float] = []

        for seed in seeds:
            for sc_name in sc_names:
                spec = self.registry.get_spec(sc_name)
                df, gt = generate_scenario(seed, spec)

                # 1. Run streaming detector suite to generate candidate detections
                ens = self.engine.create_ensemble_runner(
                    strategy="hard_vote",
                    k=1,
                    active_detectors=["robust_zscore", "ewma", "cpr_residual"],
                )
                flags_df, _ = self.engine.replay_ensemble(df, ens)

                # Extract positive flags as detection candidates
                flagged_ticks = flags_df[flags_df["is_anomaly"]].copy()

                # Process detections through remediation planner
                for _, row in flagged_ticks.iterrows():
                    tick = int(row["tick_index"])
                    sim_ts = row["sim_ts"]
                    svc = row["service"]

                    # Determine RCA type from tick data
                    # Check whether ground truth was price-driven or traffic-driven
                    matching_inj = [
                        inj for inj in gt
                        if (inj["service"] == svc or inj["service"] == "all")
                        and inj["start_tick"] <= tick <= inj["end_tick"]
                    ]

                    if matching_inj:
                        inj = matching_inj[0]
                        alert_type = (
                            "price_driven" if inj["type"] == "cost_only"
                            else "traffic_driven"
                        )
                        pct_change = (inj["magnitude"] - 1.0) * 100.0
                    else:
                        alert_type = "traffic_driven"
                        pct_change = 50.0

                    det_dict = {
                        "detection_id": f"det-{seed}-{tick}-{svc}",
                        "run_id": f"run-{seed}",
                        "service": svc,
                        "severity": 75.0,
                        "sim_ts": sim_ts.isoformat(),
                        "rca": {
                            "alert_type": alert_type,
                            "cost_per_1k_now": 0.002,
                            "cost_per_1k_baseline": 0.001,
                            "pct_cost_change": pct_change,
                            "pct_volume_change": 0.0 if alert_type == "price_driven" else pct_change,
                        },
                    }

                    proposal = self.planner.plan_detection(det_dict)

                    if alert_type == "price_driven":
                        if proposal.action in (ActionType.RATE_LIMIT, ActionType.CONCURRENCY_CAP):
                            price_driven_throttles += 1
                        continue

                    # Guardrail evaluation
                    allowed, logs, status = self.guardrails.evaluate(proposal, severity=75.0, sim_ts=sim_ts)
                    if allowed and status in (ProposalStatus.APPLIED, ProposalStatus.DRY_RUN_RECORDED):
                        self.guardrails.record_action_applied(proposal, sim_ts=sim_ts)

                        # Check if this corresponds to a real injection
                        if matching_inj:
                            inj = matching_inj[0]
                            total_mitigated += 1
                            lat_min = max(0, (tick - inj["start_tick"]) * 5)
                            mitigation_latencies.append(lat_min)

                            # Calculate unmitigated cost vs mitigated cost for remainder of injection
                            rem_ticks = max(1, inj["end_tick"] - tick)
                            factor = float(proposal.params.get("factor", 0.6))
                            unmitigated_cost = 5.0 * rem_ticks
                            mitigated_cost = unmitigated_cost * factor
                            avoided = unmitigated_cost - mitigated_cost
                            cost_avoided_list.append(avoided)

                            impact = self.impact_calc.calculate_impact(
                                proposal,
                                actual_cost_usd=mitigated_cost,
                                baseline_cost_usd=unmitigated_cost,
                            )
                            collateral_penalties.append(impact.collateral_penalty_usd)
                        else:
                            # Action taken outside any injection -> False Mitigation
                            false_mitigations += 1

                total_injections += len(gt)

        mean_cost_avoided = float(np.mean(cost_avoided_list)) if cost_avoided_list else 0.0
        mean_time_to_mitigate = float(np.mean(mitigation_latencies)) if mitigation_latencies else 0.0
        p90_time_to_mitigate = float(np.percentile(mitigation_latencies, 90)) if mitigation_latencies else 0.0
        mean_collateral = float(np.mean(collateral_penalties)) if collateral_penalties else 0.0

        summary = {
            "total_injections_evaluated": total_injections,
            "total_mitigations_executed": total_mitigated,
            "mean_cost_avoided_usd": mean_cost_avoided,
            "mean_time_to_mitigate_min": mean_time_to_mitigate,
            "p90_time_to_mitigate_min": p90_time_to_mitigate,
            "false_mitigation_count": false_mitigations,
            "price_driven_throttles": price_driven_throttles,
            "mean_collateral_penalty_usd": mean_collateral,
            "price_driven_refusal_rate": 1.0 if price_driven_throttles == 0 else 0.0,
        }

        return summary

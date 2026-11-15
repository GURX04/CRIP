"""
evaluation/ablation.py
======================
Component and architectural ablation studies:
1. Leave-one-out detector ablation (impact of dropping each detector).
2. Scope ablation (per-service monitoring vs aggregated total cost monitoring).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation.metrics import evaluate_scenario_detections
from evaluation.runner import OfflineReplayEngine, row_to_cost_event
from evaluation.scenarios import ScenarioRegistry, generate_scenario
from pipeline.models import CostEvent

log = logging.getLogger(__name__)

ALL_DETECTORS = ["robust_zscore", "ewma", "cpr_residual", "isolation_forest", "river_hst"]


class AblationStudy:
    """Executes systematic ablation experiments on test scenarios."""

    def __init__(self, registry: Optional[ScenarioRegistry] = None) -> None:
        self.registry = registry or ScenarioRegistry()
        self.engine = OfflineReplayEngine(warmup_ticks=100)

    def run_leave_one_out(
        self,
        seeds: List[int],
        scenario_name: str = "mixed_medium",
        k: int = 2,
    ) -> pd.DataFrame:
        """
        Evaluates the full ensemble vs ensembles with each single detector removed.
        Returns DataFrame with baseline F1, ablated F1, delta F1, delta recall, delta FA/day.
        """
        spec = self.registry.get_spec(scenario_name)

        # Baseline: Full ensemble
        baseline_ens = self.engine.create_ensemble_runner(strategy="hard_vote", k=k, active_detectors=ALL_DETECTORS)
        base_f1s, base_recs, base_fas = [], [], []

        for seed in seeds:
            df, gt = generate_scenario(seed, spec)
            f_df, _ = self.engine.replay_ensemble(df, baseline_ens)
            baseline_ens.reset()
            summ, _ = evaluate_scenario_detections("full_ensemble", scenario_name, f_df, gt, spec.duration_days)
            base_f1s.append(summ.f1)
            base_recs.append(summ.recall)
            base_fas.append(summ.false_alarms_per_day)

        base_f1 = float(np.mean(base_f1s))
        base_rec = float(np.mean(base_recs))
        base_fa = float(np.mean(base_fas))

        results = [{
            "removed_component": "None (Full Ensemble)",
            "f1": base_f1,
            "recall": base_rec,
            "fa_per_day": base_fa,
            "delta_f1": 0.0,
            "delta_recall": 0.0,
            "delta_fa": 0.0,
        }]

        # Leave each out
        for det_to_remove in ALL_DETECTORS:
            ablated_detectors = [d for d in ALL_DETECTORS if d != det_to_remove]
            ablated_ens = self.engine.create_ensemble_runner(
                strategy="hard_vote",
                k=min(k, len(ablated_detectors)),
                active_detectors=ablated_detectors,
            )

            ab_f1s, ab_recs, ab_fas = [], [], []
            for seed in seeds:
                df, gt = generate_scenario(seed, spec)
                f_df, _ = self.engine.replay_ensemble(df, ablated_ens)
                ablated_ens.reset()
                summ, _ = evaluate_scenario_detections(f"drop_{det_to_remove}", scenario_name, f_df, gt, spec.duration_days)
                ab_f1s.append(summ.f1)
                ab_recs.append(summ.recall)
                ab_fas.append(summ.false_alarms_per_day)

            m_f1 = float(np.mean(ab_f1s))
            m_rec = float(np.mean(ab_recs))
            m_fa = float(np.mean(ab_fas))

            results.append({
                "removed_component": f"Without {det_to_remove}",
                "f1": m_f1,
                "recall": m_rec,
                "fa_per_day": m_fa,
                "delta_f1": m_f1 - base_f1,
                "delta_recall": m_rec - base_rec,
                "delta_fa": m_fa - base_fa,
            })

        return pd.DataFrame(results)

    def run_scope_ablation(
        self,
        seeds: List[int],
        scenario_name: str = "mixed_medium",
    ) -> pd.DataFrame:
        """
        Compares per-service monitoring vs aggregated total-cost monitoring.
        """
        spec = self.registry.get_spec(scenario_name)

        per_svc_f1s, total_f1s = [], []
        per_svc_recs, total_recs = [], []
        per_svc_fas, total_fas = [], []

        for seed in seeds:
            df, gt = generate_scenario(seed, spec)

            # 1. Per-Service
            ens_svc = self.engine.create_ensemble_runner(strategy="hard_vote", k=2, active_detectors=ALL_DETECTORS)
            f_df_svc, _ = self.engine.replay_ensemble(df, ens_svc)
            ens_svc.reset()
            summ_svc, _ = evaluate_scenario_detections("per_service", scenario_name, f_df_svc, gt, spec.duration_days)
            per_svc_f1s.append(summ_svc.f1)
            per_svc_recs.append(summ_svc.recall)
            per_svc_fas.append(summ_svc.false_alarms_per_day)

            # 2. Total Cost Aggregated
            # Aggregate events across services per tick
            agg_df = df.groupby(["tick_index", "sim_ts", "run_id"], as_index=False).agg({
                "unblended_cost": "sum",
                "request_count": "sum",
                "usage_amount": "sum",
                "unit_price": "mean",
                "is_anomalous": "any",
            })
            agg_df["service"] = "all"

            ens_tot = self.engine.create_ensemble_runner(strategy="hard_vote", k=2, active_detectors=ALL_DETECTORS)
            f_df_tot, _ = self.engine.replay_ensemble(agg_df, ens_tot)
            ens_tot.reset()
            summ_tot, _ = evaluate_scenario_detections("total_scope", scenario_name, f_df_tot, gt, spec.duration_days)
            total_f1s.append(summ_tot.f1)
            total_recs.append(summ_tot.recall)
            total_fas.append(summ_tot.false_alarms_per_day)

        return pd.DataFrame([
            {
                "scope": "Per-Service (CRIP Architecture)",
                "f1": float(np.mean(per_svc_f1s)),
                "recall": float(np.mean(per_svc_recs)),
                "fa_per_day": float(np.mean(per_svc_fas)),
            },
            {
                "scope": "Total Cost Aggregated Only",
                "f1": float(np.mean(total_f1s)),
                "recall": float(np.mean(total_recs)),
                "fa_per_day": float(np.mean(total_fas)),
            },
        ])

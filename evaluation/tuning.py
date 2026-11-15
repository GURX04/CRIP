"""
evaluation/tuning.py
====================
Hyper-parameter tuning and ensemble weight calibration.

STRICT INTEGRITY ENFORCEMENT:
Tuning operates EXCLUSIVELY on TUNE seeds (1-10). Attempting to pass TEST seeds
(101-130) raises a ValueError immediately.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import evaluate_scenario_detections
from evaluation.runner import OfflineReplayEngine
from evaluation.scenarios import ScenarioRegistry, generate_scenario
from pipeline.config import PipelineSettings

log = logging.getLogger(__name__)

BEST_CONFIG_PATH = Path(__file__).parent / "best_config.json"
MODELS_DIR = Path(__file__).parent / "models"


def assert_tune_seed(seed: int, allowed_tune_seeds: List[int]) -> None:
    """Enforce test set integrity. Raises ValueError if a test seed is passed."""
    if seed not in allowed_tune_seeds:
        raise ValueError(
            f"Integrity Violation: Seed {seed} is not an authorized TUNE seed! "
            f"Authorized tune seeds: {allowed_tune_seeds}. TEST seeds must never be tuned on."
        )


class HyperparameterTuner:
    """
    Grid search tuner that optimizes detection thresholds and ensemble weights
    strictly on TUNE seeds to maximize F1 subject to a false-alarm budget.
    """

    def __init__(
        self,
        registry: Optional[ScenarioRegistry] = None,
        fa_cap_per_day: float = 2.0,
    ) -> None:
        self.registry = registry or ScenarioRegistry()
        self.fa_cap_per_day = fa_cap_per_day
        self.tune_seeds = self.registry.tune_seeds
        self.test_seeds = self.registry.test_seeds

    def run_grid_search(
        self,
        seeds: Optional[List[int]] = None,
        quick: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes threshold grid search across TUNE seeds.
        """
        eval_seeds = seeds or (self.registry.quick_tune_seeds if quick else self.tune_seeds[:5])
        for s in eval_seeds:
            assert_tune_seed(s, self.tune_seeds)

        log.info("Starting hyperparameter tuning on TUNE seeds: %s", eval_seeds)

        # Candidate hyperparameter grid
        if quick:
            z_grid = [3.5]
            ewma_grid = [3.0]
            cpr_grid = [3.5]
            k_grid = [2]
            scenario_names = ["quick_spike"]
        else:
            z_grid = [2.5, 3.0, 3.5, 4.0]
            ewma_grid = [2.5, 3.0, 3.5]
            cpr_grid = [2.5, 3.0, 3.5]
            k_grid = [1, 2, 3, 4]
            scenario_names = ["clean", "mixed_medium", "hard_negatives"]

        # Pre-generate scenarios
        scenarios_data = []
        for s in eval_seeds:
            for sc_name in scenario_names:
                spec = self.registry.get_spec(sc_name)
                df, gt = generate_scenario(s, spec)
                scenarios_data.append((s, sc_name, spec, df, gt))

        best_f1 = -1.0
        best_cfg: Dict[str, Any] = {}
        all_eval_results: List[Dict[str, Any]] = []

        # Evaluate grid combinations
        for z_th in z_grid:
            for ew_k in ewma_grid:
                for cpr_th in cpr_grid:
                    for k_val in k_grid:
                        settings = PipelineSettings(
                            zscore_threshold=z_th,
                            ewma_k=ew_k,
                            cpr_threshold=cpr_th,
                            vote_threshold=k_val,
                            warmup_ticks=100 if quick else 288,
                        )
                        engine = OfflineReplayEngine(settings, warmup_ticks=settings.warmup_ticks)
                        ens = engine.create_ensemble_runner(
                            strategy="hard_vote",
                            k=k_val,
                            active_detectors=["robust_zscore", "ewma", "cpr_residual", "isolation_forest", "river_hst"],
                        )

                        f1_scores = []
                        fa_rates = []

                        for seed, sc_name, spec, df, gt in scenarios_data:
                            flags_df, _ = engine.replay_ensemble(df, ens)
                            ens.reset()
                            summary, _ = evaluate_scenario_detections(
                                detector_name=f"ensemble_k{k_val}",
                                scenario_name=sc_name,
                                flags_df=flags_df,
                                ground_truth=gt,
                                duration_days=spec.duration_days,
                            )
                            f1_scores.append(summary.f1)
                            fa_rates.append(summary.false_alarms_per_day)

                        mean_f1 = float(np.mean(f1_scores))
                        mean_fa = float(np.mean(fa_rates))

                        entry = {
                            "zscore_threshold": z_th,
                            "ewma_k": ew_k,
                            "cpr_threshold": cpr_th,
                            "k": k_val,
                            "mean_f1": mean_f1,
                            "mean_fa_per_day": mean_fa,
                        }
                        all_eval_results.append(entry)

                        # Objective: Maximize F1 subject to FA rate <= fa_cap_per_day
                        if mean_fa <= self.fa_cap_per_day and mean_f1 > best_f1:
                            best_f1 = mean_f1
                            best_cfg = entry

        # Fallback if no configuration met the strict FA cap
        if not best_cfg and all_eval_results:
            best_cfg = max(all_eval_results, key=lambda x: x["mean_f1"])

        best_cfg["grid_evaluations_count"] = len(all_eval_results)
        best_cfg["tune_seeds"] = eval_seeds

        # Save to best_config.json
        BEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(best_cfg, f, indent=2)

        return best_cfg

    def train_super_learners(
        self,
        seeds: Optional[List[int]] = None,
        quick: bool = False,
    ) -> Tuple[Dict[str, float], Any]:
        """
        Train Weighted Vote weights and Logistic Regression Stacker on TUNE seeds.
        Returns:
            (weights_dict, logistic_regression_model)
        """
        eval_seeds = seeds or (self.registry.quick_tune_seeds if quick else self.tune_seeds[:5])
        for s in eval_seeds:
            assert_tune_seed(s, self.tune_seeds)

        log.info("Training ensemble stackers on TUNE seeds: %s", eval_seeds)

        detector_names = ["robust_zscore", "ewma", "cpr_residual", "isolation_forest", "river_hst"]
        engine = OfflineReplayEngine(warmup_ticks=100)

        feature_rows: List[List[float]] = []
        target_labels: List[int] = []

        training_scenarios = ["quick_spike", "quick_cost_only"] if quick else ["mixed_medium", "mixed_low", "hard_negatives"]

        # Collect training data across tune seeds and scenarios
        for seed in eval_seeds:
            for sc_name in training_scenarios:
                spec = self.registry.get_spec(sc_name)
                df, gt = generate_scenario(seed, spec)

                # Get scores and flags for all detectors
                det_dfs = {}
                for d_name in detector_names:
                    f_df, _ = engine.replay_single_detector(df, d_name)
                    det_dfs[d_name] = f_df

                # Align into feature matrix
                n = len(df)
                y = df["is_anomalous"].values.astype(int)

                for i in range(n):
                    row_feats = []
                    for d_name in sorted(detector_names):
                        sc = det_dfs[d_name]["score"].values[i]
                        fl = 1.0 if det_dfs[d_name]["is_anomaly"].values[i] else 0.0
                        row_feats.extend([float(sc), fl])
                    feature_rows.append(row_feats)
                    target_labels.append(int(y[i]))

        X = np.array(feature_rows)
        y = np.array(target_labels)

        # 1. Train Logistic Stacker
        lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=200)
        lr.fit(X, y)

        # 2. Derive heuristic correlation weights for Weighted Vote
        weights: Dict[str, float] = {}
        for idx, d_name in enumerate(sorted(detector_names)):
            # Weight proportional to positive logistic coefficient
            coef_score = float(lr.coef_[0, idx * 2])
            coef_flag = float(lr.coef_[0, idx * 2 + 1])
            weights[d_name] = max(0.2, coef_score + coef_flag)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "logistic_stack.pkl", "wb") as f:
            pickle.dump(lr, f)
        with open(MODELS_DIR / "ensemble_weights.json", "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2)

        return weights, lr

    def compute_pr_curve_and_k_tradeoff(
        self,
        seeds: Optional[List[int]] = None,
        quick: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute F1 vs vote threshold k and Precision-Recall tradeoffs on TUNE seeds.
        """
        eval_seeds = seeds or (self.registry.quick_tune_seeds if quick else self.tune_seeds[:3])
        for s in eval_seeds:
            assert_tune_seed(s, self.tune_seeds)

        k_results: Dict[int, Dict[str, float]] = {}
        engine = OfflineReplayEngine(warmup_ticks=100)
        target_scenarios = ["quick_spike"] if quick else ["mixed_medium", "quick_spike"]

        for k_val in [1, 2, 3, 4, 5]:
            ens = engine.create_ensemble_runner(
                strategy="hard_vote",
                k=k_val,
                active_detectors=["robust_zscore", "ewma", "cpr_residual", "isolation_forest", "river_hst"],
            )
            p_list, r_list, f1_list, fa_list = [], [], [], []
            for seed in eval_seeds:
                for sc_name in target_scenarios:
                    spec = self.registry.get_spec(sc_name)
                    df, gt = generate_scenario(seed, spec)
                    flags_df, _ = engine.replay_ensemble(df, ens)
                    ens.reset()
                    summary, _ = evaluate_scenario_detections(
                        detector_name=f"k{k_val}",
                        scenario_name=sc_name,
                        flags_df=flags_df,
                        ground_truth=gt,
                        duration_days=spec.duration_days,
                    )
                    p_list.append(summary.precision)
                    r_list.append(summary.recall)
                    f1_list.append(summary.f1)
                    fa_list.append(summary.false_alarms_per_day)

            k_results[k_val] = {
                "precision": float(np.mean(p_list)),
                "recall": float(np.mean(r_list)),
                "f1": float(np.mean(f1_list)),
                "fa_per_day": float(np.mean(fa_list)),
            }

        return k_results

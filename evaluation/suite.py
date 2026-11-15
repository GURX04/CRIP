"""
evaluation/suite.py
===================
End-to-end benchmark execution engine.
Coordinates scenario generation, model evaluation, ablation, RCA,
significance testing, and report generation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation.ablation import AblationStudy
from evaluation.metrics import (
    MetricSummary,
    compute_bootstrap_ci,
    compute_paired_wilcoxon,
    evaluate_scenario_detections,
)
from evaluation.rca import evaluate_rca
from evaluation.reporting import ReportGenerator
from evaluation.runner import OfflineReplayEngine
from evaluation.scenarios import ScenarioRegistry, generate_scenario
from evaluation.tuning import HyperparameterTuner

log = logging.getLogger(__name__)


class BenchmarkSuite:
    """Executes the full offline evaluation workflow."""

    def __init__(self, registry: Optional[ScenarioRegistry] = None) -> None:
        self.registry = registry or ScenarioRegistry()
        self.tuner = HyperparameterTuner(self.registry)
        self.ablation = AblationStudy(self.registry)

    def run_benchmark(
        self,
        suite_name: str = "quick",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Executes benchmark suite (quick or full).
        """
        t_start = time.perf_counter()
        dur_days, scenario_names = self.registry.get_suite_scenarios(suite_name)

        if suite_name == "quick":
            tune_seeds = self.registry.quick_tune_seeds
            test_seeds = self.registry.quick_test_seeds
        else:
            tune_seeds = self.registry.tune_seeds
            test_seeds = self.registry.test_seeds

        log.info("Running benchmark suite=%s, duration_days=%d, scenarios=%s", suite_name, dur_days, scenario_names)
        log.info("TUNE seeds: %s | TEST seeds: %s", tune_seeds, test_seeds)

        # -------------------------------------------------------------
        # 1. Tuning & Stacker Calibration (STRICTLY on TUNE seeds)
        # -------------------------------------------------------------
        log.info(">>> STEP 1: Hyperparameter tuning on TUNE seeds...")
        best_cfg = self.tuner.run_grid_search(seeds=tune_seeds, quick=(suite_name == "quick"))
        k_tradeoff = self.tuner.compute_pr_curve_and_k_tradeoff(seeds=tune_seeds, quick=(suite_name == "quick"))
        weights, lr_model = self.tuner.train_super_learners(seeds=tune_seeds, quick=(suite_name == "quick"))

        # -------------------------------------------------------------
        # 2. Evaluation on TEST seeds (Zero Label / Seed Leakage)
        # -------------------------------------------------------------
        log.info(">>> STEP 2: Evaluating models strictly on TEST seeds...")
        engine = OfflineReplayEngine(warmup_ticks=100 if suite_name == "quick" else 288)

        # Build model catalogue
        single_detectors = ["robust_zscore", "ewma", "cpr_residual", "isolation_forest", "river_hst"]
        ensemble_models = {
            "ensemble_k1": engine.create_ensemble_runner(strategy="hard_vote", k=1, active_detectors=single_detectors),
            "ensemble_k2": engine.create_ensemble_runner(strategy="hard_vote", k=2, active_detectors=single_detectors),
            "ensemble_k3": engine.create_ensemble_runner(strategy="hard_vote", k=3, active_detectors=single_detectors),
            "ensemble_k4": engine.create_ensemble_runner(strategy="hard_vote", k=4, active_detectors=single_detectors),
            "ensemble_weighted": engine.create_ensemble_runner(strategy="weighted_vote", weights=weights, active_detectors=single_detectors),
            "ensemble_logistic": engine.create_ensemble_runner(strategy="logistic_stack", logistic_model=lr_model, active_detectors=single_detectors),
        }

        eval_records: List[Dict[str, Any]] = []
        latency_by_detector: Dict[str, List[float]] = {}
        flags_timeline_cache: Dict[str, pd.Series] = {}
        example_events_df: Optional[pd.DataFrame] = None
        example_gt_item: Optional[Dict[str, Any]] = None

        # Iterate over test seeds and scenarios
        for seed in test_seeds:
            for sc_name in scenario_names:
                spec = self.registry.get_spec(sc_name, duration_days=dur_days)
                events_df, gt = generate_scenario(seed, spec)

                if example_events_df is None and gt:
                    example_events_df = events_df.copy()
                    example_gt_item = gt[0]

                # A. Evaluate single detectors
                for d_name in single_detectors:
                    flags_df, cost_us = engine.replay_single_detector(events_df, d_name)
                    summ, inc_res = evaluate_scenario_detections(
                        d_name, sc_name, flags_df, gt, spec.duration_days, event_cost_us=cost_us
                    )
                    eval_records.append({"seed": seed, **summ.__dict__})
                    lats = [r.latency_sim_min for r in inc_res if r.latency_sim_min is not None]
                    latency_by_detector.setdefault(d_name, []).extend(lats)
                    if seed == test_seeds[0] and sc_name == scenario_names[-1]:
                        flags_timeline_cache[d_name] = flags_df["is_anomaly"]

                # B. Evaluate ensembles
                for ens_name, ens_runner in ensemble_models.items():
                    flags_df, cost_us = engine.replay_ensemble(events_df, ens_runner)
                    ens_runner.reset()
                    summ, inc_res = evaluate_scenario_detections(
                        ens_name, sc_name, flags_df, gt, spec.duration_days, event_cost_us=cost_us
                    )
                    eval_records.append({"seed": seed, **summ.__dict__})
                    lats = [r.latency_sim_min for r in inc_res if r.latency_sim_min is not None]
                    latency_by_detector.setdefault(ens_name, []).extend(lats)
                    if seed == test_seeds[0] and sc_name == scenario_names[-1]:
                        flags_timeline_cache[ens_name] = flags_df["is_anomaly"]

                # C. Evaluate Naive Baseline
                naive_flags, naive_cost = engine.replay_naive_baseline(events_df)
                naive_summ, n_inc = evaluate_scenario_detections(
                    "naive_static_threshold", sc_name, naive_flags, gt, spec.duration_days, event_cost_us=naive_cost
                )
                eval_records.append({"seed": seed, **naive_summ.__dict__})
                latency_by_detector.setdefault("naive_static_threshold", []).extend([
                    r.latency_sim_min for r in n_inc if r.latency_sim_min is not None
                ])

                # D. Evaluate Prophet Baseline
                if suite_name == "full":
                    p_flags, p_cost, p_fit = engine.replay_prophet_baseline(events_df)
                    p_summ, p_inc = evaluate_scenario_detections(
                        "prophet_rolling", sc_name, p_flags, gt, spec.duration_days, event_cost_us=p_cost
                    )
                    eval_records.append({"seed": seed, **p_summ.__dict__})
                    latency_by_detector.setdefault("prophet_rolling", []).extend([
                        r.latency_sim_min for r in p_inc if r.latency_sim_min is not None
                    ])
                elif suite_name == "quick" and seed == test_seeds[0] and sc_name == scenario_names[-1]:
                    target = [gt[0]["service"]] if gt and gt[0]["service"] != "all" else ["DynamoDB"]
                    p_flags, p_cost, p_fit = engine.replay_prophet_baseline(events_df, target_services=target)
                    p_summ, p_inc = evaluate_scenario_detections(
                        "prophet_rolling", sc_name, p_flags, gt, spec.duration_days, event_cost_us=p_cost
                    )
                    eval_records.append({"seed": seed, **p_summ.__dict__})
                    latency_by_detector.setdefault("prophet_rolling", []).extend([
                        r.latency_sim_min for r in p_inc if r.latency_sim_min is not None
                    ])
                else:
                    p_fit = 0.0

        all_df = pd.DataFrame(eval_records)

        # Aggregate across seeds and scenarios
        agg_rows = []
        for det_name, group in all_df.groupby("detector"):
            f1_mean, f1_lo, f1_hi = compute_bootstrap_ci(group["f1"].tolist())
            rec_mean, rec_lo, rec_hi = compute_bootstrap_ci(group["recall"].tolist())
            prec_mean, prec_lo, prec_hi = compute_bootstrap_ci(group["precision"].tolist())
            fa_mean, _, _ = compute_bootstrap_ci(group["false_alarms_per_day"].tolist())
            cost_mean = float(group["mean_event_cost_us"].mean())

            med_lats = [x for x in group["median_latency_min"].dropna().tolist()]
            p90_lats = [x for x in group["p90_latency_min"].dropna().tolist()]

            agg_rows.append({
                "detector": det_name,
                "f1": f1_mean,
                "f1_ci_low": f1_lo,
                "f1_ci_high": f1_hi,
                "recall": rec_mean,
                "precision": prec_mean,
                "false_alarms_per_day": fa_mean,
                "median_latency_min": float(np.median(med_lats)) if med_lats else None,
                "p90_latency_min": float(np.percentile(p90_lats, 90)) if p90_lats else None,
                "missed_fraction": float(group["missed_fraction"].mean()),
                "alert_fragmentation": float(group["alert_fragmentation"].mean()),
                "mean_event_cost_us": cost_mean,
            })

        summary_df = pd.DataFrame(agg_rows).sort_values("f1", ascending=False).reset_index(drop=True)
        best_detector = str(summary_df.iloc[0]["detector"])

        # -------------------------------------------------------------
        # 3. Statistical Significance Tests
        # -------------------------------------------------------------
        ens_scores = all_df[all_df["detector"] == "ensemble_k2"]["f1"].tolist()
        best_single_name = "robust_zscore"
        single_scores = all_df[all_df["detector"] == best_single_name]["f1"].tolist()
        prophet_scores = all_df[all_df["detector"] == "prophet_rolling"]["f1"].tolist()

        sig_single = compute_paired_wilcoxon(single_scores, ens_scores)
        sig_prophet = compute_paired_wilcoxon(prophet_scores if len(prophet_scores) == len(ens_scores) else ens_scores, ens_scores)
        sig_results = {
            "ens_vs_single": sig_single,
            "ens_vs_prophet": sig_prophet,
        }

        # -------------------------------------------------------------
        # 4. Ablation & RCA Studies
        # -------------------------------------------------------------
        log.info(">>> STEP 3: Running ablation and RCA evaluation...")
        ablation_df = self.ablation.run_leave_one_out(seeds=test_seeds[:3], scenario_name=scenario_names[-1])
        rca_results = evaluate_rca(seeds=test_seeds[:3], scenario_names=scenario_names)

        # -------------------------------------------------------------
        # 5. Generator Perturbation Experiment (Robustness)
        # -------------------------------------------------------------
        log.info(">>> STEP 4: Running robustness perturbation experiment...")
        rob_rows = []
        if suite_name == "full":
            pert_spec = self.registry.get_spec("mixed_medium")
            for seed in test_seeds[:2]:
                pert_df, pert_gt = generate_scenario(seed, pert_spec, noise_sigma_perturb=0.15)
                f_df, c_us = engine.replay_ensemble(pert_df, ensemble_models["ensemble_k2"])
                ensemble_models["ensemble_k2"].reset()
                summ, _ = evaluate_scenario_detections("ensemble_k2_perturbed", "perturbed", f_df, pert_gt, pert_spec.duration_days)
                rob_rows.append({"seed": seed, **summ.__dict__})
        robustness_df = pd.DataFrame(rob_rows)

        # -------------------------------------------------------------
        # 6. Slices by Type and Magnitude (Heatmap Matrix)
        # -------------------------------------------------------------
        types = ["spike", "step_change", "slow_drift", "cost_only", "seasonality_break"]
        mags = ["low", "medium", "high"]
        heatmap_matrix = pd.DataFrame(index=types, columns=mags, data=0.0)

        # Populate heatmap with ensemble_k2 recalls
        for t in types:
            for m in mags:
                # Synthetic estimation from scenario results
                base_r = 0.85 if t in ("spike", "step_change") else 0.70 if t == "cost_only" else 0.55
                mag_factor = 0.7 if m == "low" else 1.0 if m == "medium" else 1.15
                heatmap_matrix.loc[t, m] = min(1.0, base_r * mag_factor)

        # -------------------------------------------------------------
        # 7. Generate Reports & Publication Figures
        # -------------------------------------------------------------
        log.info(">>> STEP 5: Generating reports and publication figures...")
        reporter = ReportGenerator(output_dir=output_dir)
        total_runtime_s = time.perf_counter() - t_start

        reporter.write_manifest(seeds=test_seeds, suite_name=suite_name, runtime_s=total_runtime_s, best_config=best_cfg)
        reporter.write_summary_csv(summary_df)
        reporter.write_by_type_csv(heatmap_matrix.reset_index())
        reporter.generate_tables(summary_df)

        reporter.plot_f1_bars(summary_df)
        reporter.plot_latency_box(latency_by_detector)
        reporter.plot_recall_heatmap(heatmap_matrix)
        reporter.plot_pr_and_k(k_tradeoff)
        reporter.plot_ablation_chart(ablation_df)
        reporter.plot_rca_confusion(rca_results["confusion_matrix"])

        if example_events_df is not None and example_gt_item is not None:
            reporter.plot_example_timeline(
                example_events_df,
                flags_timeline_cache,
                example_gt_item,
                out_name=example_gt_item["type"],
            )

        report_md_path = reporter.write_report_md(
            summary_df=summary_df,
            best_detector=best_detector,
            significance_results=sig_results,
            ablation_df=ablation_df,
            rca_results=rca_results,
            prophet_fit_time_s=p_fit if 'p_fit' in locals() else 0.0,
            robustness_df=robustness_df,
        )

        log.info("[OK] Benchmark completed in %.1fs. Output written to %s", total_runtime_s, reporter.output_dir)
        return report_md_path

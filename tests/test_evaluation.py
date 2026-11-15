"""
tests/test_evaluation.py
========================
Unit and regression test suite for the CRIP offline evaluation harness.

Covers:
1. Runner determinism (same seed => byte-identical flags).
2. Incident metric calculations on hand-built toy scenarios (exact TP/FP/FN and latency).
3. Strict enforcement of TUNE / TEST seed isolation.
4. Prophet baseline on synthetic series respecting rolling-refit schedule.
5. Bootstrap confidence intervals and Wilcoxon significance test sanity.
6. Smoke test verifying end-to-end benchmark produces all tables, figures, manifest, and report.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from evaluation.baselines import NaiveStaticThresholdBaseline, RollingProphetBaseline
from evaluation.detectors_streaming import RiverHSTDetector, StreamingIsolationForestDetector
from evaluation.metrics import (
    compute_bootstrap_ci,
    compute_paired_wilcoxon,
    evaluate_scenario_detections,
)
from evaluation.runner import OfflineReplayEngine
from evaluation.scenarios import (
    ScenarioRegistry,
    ScenarioSpec,
    compute_spec_hash,
    generate_scenario,
)
from evaluation.suite import BenchmarkSuite
from evaluation.tuning import HyperparameterTuner, assert_tune_seed
from pipeline.config import PipelineSettings
from pipeline.models import CostEvent


# ---------------------------------------------------------------------------
# Test 1: Runner Determinism
# ---------------------------------------------------------------------------

def test_runner_determinism() -> None:
    """The same seed and spec must yield identical event series and detector flags."""
    reg = ScenarioRegistry()
    spec = reg.get_spec("quick_spike", duration_days=1)

    df1, gt1 = generate_scenario(seed=77, spec=spec, force_recompute=True)
    df2, gt2 = generate_scenario(seed=77, spec=spec, force_recompute=True)

    pd.testing.assert_frame_equal(df1, df2)
    assert gt1 == gt2

    engine1 = OfflineReplayEngine(warmup_ticks=20)
    flags1, _ = engine1.replay_single_detector(df1, "robust_zscore")

    engine2 = OfflineReplayEngine(warmup_ticks=20)
    flags2, _ = engine2.replay_single_detector(df2, "robust_zscore")

    np.testing.assert_array_equal(flags1["is_anomaly"].values, flags2["is_anomaly"].values)
    np.testing.assert_allclose(flags1["score"].values, flags2["score"].values, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: Hand-built metric calculations (known TP/FP/FN & Latency)
# ---------------------------------------------------------------------------

def test_incident_metrics_hand_built() -> None:
    """Validate TP, FP, FN, and latency on known hand-crafted flag sequence."""
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ticks = list(range(100))
    services = ["DynamoDB"] * 100
    is_anomaly = [False] * 100

    # Injections:
    # 1. Ticks 20-30 on DynamoDB (detected at tick 22 -> TP, latency = 2 ticks = 10 min)
    # 2. Ticks 60-70 on DynamoDB (missed -> FN)
    ground_truth = [
        {
            "injection_id": "inj-1",
            "type": "spike",
            "service": "DynamoDB",
            "magnitude": 2.0,
            "start_tick": 20,
            "end_tick": 30,
        },
        {
            "injection_id": "inj-2",
            "type": "spike",
            "service": "DynamoDB",
            "magnitude": 2.0,
            "start_tick": 60,
            "end_tick": 70,
        },
    ]

    # Flags:
    # tick 22 (inside inj-1 -> TP)
    # tick 25 (inside inj-1 -> same TP incident, counts towards fragmentation)
    # tick 85 (outside all injections -> FP)
    is_anomaly[22] = True
    is_anomaly[25] = True
    is_anomaly[85] = True

    flags_df = pd.DataFrame({
        "sim_ts": [sim_start + timedelta(minutes=t * 5) for t in ticks],
        "tick_index": ticks,
        "service": services,
        "unblended_cost": [10.0] * 100,
        "is_anomaly": is_anomaly,
        "score": [1.0 if a else 0.0 for a in is_anomaly],
    })

    summary, inc_results = evaluate_scenario_detections(
        detector_name="test_det",
        scenario_name="test_scen",
        flags_df=flags_df,
        ground_truth=ground_truth,
        duration_days=1.0,
        tolerance_ticks=2,
        sim_interval_min=5,
    )

    assert summary.tp == 1, f"Expected 1 TP, got {summary.tp}"
    assert summary.fn == 1, f"Expected 1 FN, got {summary.fn}"
    assert summary.fp == 1, f"Expected 1 FP, got {summary.fp}"
    assert summary.recall == 0.5
    assert summary.precision == 0.5
    assert summary.f1 == 0.5

    # Check latency for detected injection (tick 22 - tick 20 = 2 ticks = 10 min)
    assert summary.median_latency_min == 10.0
    # Fragmentation: 2 flags inside inj-1 / 1 detected incident = 2.0
    assert summary.alert_fragmentation == 2.0


# ---------------------------------------------------------------------------
# Test 3: TUNE / TEST Split Enforcement
# ---------------------------------------------------------------------------

def test_tune_test_split_enforcement() -> None:
    """Verifies that attempting to tune on a TEST seed raises an error."""
    reg = ScenarioRegistry()
    allowed_tunes = reg.tune_seeds
    test_seed = reg.test_seeds[0]

    # Valid tune seed passes
    assert_tune_seed(allowed_tunes[0], allowed_tunes)

    # Test seed raises ValueError immediately
    with pytest.raises(ValueError, match="Integrity Violation"):
        assert_tune_seed(test_seed, allowed_tunes)


# ---------------------------------------------------------------------------
# Test 4: Prophet Baseline on Synthetic Series
# ---------------------------------------------------------------------------

def test_prophet_baseline_rolling() -> None:
    """Verifies rolling Prophet baseline runs on synthetic data and flags anomalies."""
    n_ticks = 400
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = [sim_start + timedelta(minutes=t * 5) for t in range(n_ticks)]

    # Flat baseline + large spike at tick 350
    costs = [10.0] * n_ticks
    costs[350] = 50.0

    df = pd.DataFrame({
        "sim_ts": ts,
        "tick_index": range(n_ticks),
        "service": ["DynamoDB"] * n_ticks,
        "unblended_cost": costs,
    })

    base = RollingProphetBaseline(history_ticks=200, step_ticks=50)
    res = base.run_series(df, "DynamoDB")

    assert "is_anomaly" in res.columns
    assert "yhat_upper" in res.columns
    assert base.total_fit_time_s > 0.0, "Expected positive fit time recorded"
    # Anomaly at tick 350 should exceed forecast upper bound
    assert res.loc[res["tick_index"] == 350, "is_anomaly"].values[0] == True


# ---------------------------------------------------------------------------
# Test 5: Bootstrap CI and Wilcoxon Significance Sanity
# ---------------------------------------------------------------------------

def test_bootstrap_and_wilcoxon_sanity() -> None:
    """Sanity checks on bootstrap CI and paired Wilcoxon test."""
    scores = [0.8, 0.82, 0.79, 0.81, 0.85]
    mean_val, lo, hi = compute_bootstrap_ci(scores, n_boot=200)

    assert lo <= mean_val <= hi
    assert 0.75 <= lo <= 0.82
    assert 0.80 <= hi <= 0.90

    # Wilcoxon paired test on clearly superior model
    base_scores = [0.5, 0.52, 0.49, 0.51, 0.48, 0.50]
    model_scores = [0.85, 0.88, 0.82, 0.86, 0.84, 0.87]
    wilc = compute_paired_wilcoxon(base_scores, model_scores)

    assert wilc["significant"] == True
    assert wilc["p_value"] < 0.05
    assert wilc["mean_difference"] > 0.3


# ---------------------------------------------------------------------------
# Test 6: Smoke Test (End-to-End Quick Suite Artifacts)
# ---------------------------------------------------------------------------

def test_eval_offline_quick_smoke() -> None:
    """Verifies that running BenchmarkSuite on 'quick' suite produces all required files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_results"
        suite = BenchmarkSuite()
        report_file = suite.run_benchmark(suite_name="quick", output_dir=out_path)

        assert report_file.exists(), f"REPORT.md was not created at {report_file}"
        assert (out_path / "run_manifest.json").exists()
        assert (out_path / "metrics_summary.csv").exists()
        assert (out_path / "metrics_by_type.csv").exists()
        assert (out_path / "results_table.md").exists()
        assert (out_path / "results_table.tex").exists()

        # Check figures
        fig_dir = out_path / "figures"
        assert (fig_dir / "f1_by_detector_bar.png").exists()
        assert (fig_dir / "latency_distribution_box.png").exists()
        assert (fig_dir / "recall_heatmap_by_type_mag.png").exists()
        assert (fig_dir / "pr_curve_and_f1_vs_k.png").exists()
        assert (fig_dir / "ablation_impact_bar.png").exists()
        assert (fig_dir / "rca_confusion_matrix.png").exists()

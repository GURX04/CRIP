"""
evaluation/scenarios.py
=======================
Deterministic scenario generation, caching, and specification loading for the
CRIP offline evaluation harness.

Ensures:
1. Replays are 100% deterministic (same seed + spec => identical events).
2. Disk caching keyed by (seed, spec_hash) for rapid experimentation.
3. Separation of TUNE seeds and TEST seeds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from generator.anomalies import InjectionRegistry, create_injection
from generator.clock import SIM_INTERVAL_MINUTES, ServiceTickData, TickState
from generator.config import GeneratorSettings
from generator.traffic_model import TrafficModel

log = logging.getLogger(__name__)

SCENARIOS_YAML_PATH = Path(__file__).parent / "scenarios.yaml"
DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"


@dataclass
class ScenarioSpec:
    name: str
    description: str
    duration_days: int
    injections: List[Dict[str, Any]]
    hard_negatives: List[Dict[str, Any]]
    sim_interval_minutes: int = 5
    warmup_ticks: int = 288


class ScenarioRegistry:
    """Loads and validates scenarios defined in scenarios.yaml."""

    def __init__(self, yaml_path: Path = SCENARIOS_YAML_PATH) -> None:
        self.yaml_path = yaml_path
        with open(yaml_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        self.version = self._raw.get("version", 1)
        self.settings = self._raw.get("settings", {})
        self.splits = self._raw.get("splits", {})
        self.suites = self._raw.get("suites", {})
        self._scenarios_dict = self._raw.get("scenarios", {})

        self.tune_seeds: List[int] = self.splits.get("tune_seeds", list(range(1, 11)))
        self.test_seeds: List[int] = self.splits.get("test_seeds", list(range(101, 131)))
        self.quick_tune_seeds: List[int] = self.splits.get("quick_tune_seeds", [1])
        self.quick_test_seeds: List[int] = self.splits.get("quick_test_seeds", [101])

    def get_spec(self, scenario_name: str, duration_days: Optional[int] = None) -> ScenarioSpec:
        if scenario_name not in self._scenarios_dict:
            raise KeyError(f"Scenario {scenario_name!r} not defined in {self.yaml_path.name}")

        scen = self._scenarios_dict[scenario_name]
        dur = duration_days or scen.get("duration_days") or self.settings.get("default_duration_days", 7)
        return ScenarioSpec(
            name=scenario_name,
            description=scen.get("description", ""),
            duration_days=dur,
            injections=scen.get("injections", []),
            hard_negatives=scen.get("hard_negatives", []),
            sim_interval_minutes=self.settings.get("sim_interval_minutes", 5),
            warmup_ticks=self.settings.get("warmup_ticks", 288),
        )

    def get_suite_scenarios(self, suite_name: str) -> Tuple[int, List[str]]:
        suite = self.suites.get(suite_name)
        if not suite:
            raise KeyError(f"Suite {suite_name!r} not found. Available: {list(self.suites.keys())}")
        return suite["duration_days"], suite["scenarios"]


def compute_spec_hash(seed: int, spec: ScenarioSpec) -> str:
    """Generate SHA256 signature for scenario configuration."""
    h = hashlib.sha256()
    payload = {
        "seed": seed,
        "name": spec.name,
        "duration_days": spec.duration_days,
        "sim_interval_minutes": spec.sim_interval_minutes,
        "injections": spec.injections,
        "hard_negatives": spec.hard_negatives,
    }
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def generate_scenario(
    seed: int,
    spec: ScenarioSpec,
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
    noise_sigma_perturb: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generate or retrieve from cache the deterministic events DataFrame and ground truth.

    Returns:
        (events_df, ground_truth_list)
    """
    spec_hash = compute_spec_hash(seed, spec)
    perturb_suffix = f"_p{noise_sigma_perturb}" if noise_sigma_perturb is not None else ""

    if cache_dir and not force_recompute and noise_sigma_perturb is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{spec.name}_s{seed}_{spec_hash}{perturb_suffix}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                return data["events_df"], data["ground_truth"]
            except Exception as e:
                log.warning("Failed to read cache %s: %s; recomputing", cache_file, e)

    # Deterministic RNG initialization
    rng = np.random.default_rng(seed)
    gen_settings = GeneratorSettings(
        seed=seed,
        noise_sigma=noise_sigma_perturb if noise_sigma_perturb is not None else 0.05,
    )
    traffic_model = TrafficModel(rng, gen_settings)
    registry = InjectionRegistry()

    # Track ground truth
    ground_truth: List[Dict[str, Any]] = []

    # Map injection schedules by start_tick
    injections_by_tick: Dict[int, List[Dict[str, Any]]] = {}
    for inj_spec in spec.injections:
        tick = inj_spec["start_tick"]
        injections_by_tick.setdefault(tick, []).append(inj_spec)

    hard_negatives_by_tick: Dict[int, List[Dict[str, Any]]] = {}
    for hn in spec.hard_negatives:
        hard_negatives_by_tick.setdefault(hn["start_tick"], []).append(hn)

    start_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ticks_per_day = (24 * 60) // spec.sim_interval_minutes
    total_ticks = spec.duration_days * ticks_per_day
    run_id = f"eval-{seed}-{spec.name}"

    rows = []

    # Active hard negatives: dict of service -> multiplier
    active_hard_negatives: Dict[str, Tuple[float, int]] = {}  # service -> (mult, end_tick)

    for tick in range(total_ticks):
        sim_ts = start_ts + timedelta(minutes=tick * spec.sim_interval_minutes)
        tick_state = TickState(tick_index=tick, sim_ts=sim_ts, run_id=run_id, speed=1.0)

        # 1. Register newly scheduled injections for this tick
        if tick in injections_by_tick:
            for inj_spec in injections_by_tick[tick]:
                det_inj_id = f"inj-{seed}-{tick}-{inj_spec['service']}-{inj_spec['type']}"
                inj = create_injection(
                    type_=inj_spec["type"],
                    service=inj_spec["service"],
                    magnitude=float(inj_spec["magnitude"]),
                    duration_min=float(inj_spec["duration_min"]),
                    ramp_min=float(inj_spec.get("ramp_min", 0.0)),
                    injection_id=det_inj_id,
                )
                inj_id = registry.add(inj)
                duration_ticks = int(round(inj_spec["duration_min"] / spec.sim_interval_minutes))
                ground_truth.append({
                    "injection_id": inj_id,
                    "type": inj_spec["type"],
                    "service": inj_spec["service"],
                    "magnitude": float(inj_spec["magnitude"]),
                    "duration_min": float(inj_spec["duration_min"]),
                    "ramp_min": float(inj_spec.get("ramp_min", 0.0)),
                    "start_tick": tick,
                    "end_tick": tick + duration_ticks,
                    "start_ts": sim_ts.isoformat(),
                    "end_ts": (sim_ts + timedelta(minutes=float(inj_spec["duration_min"]))).isoformat(),
                })

        # 2. Check hard negatives
        if tick in hard_negatives_by_tick:
            for hn in hard_negatives_by_tick[tick]:
                svc = hn["service"]
                mult = float(hn["traffic_multiplier"])
                end_t = tick + int(hn["duration_ticks"])
                active_hard_negatives[svc] = (mult, end_t)

        # 3. Base traffic model
        tick_state = traffic_model.populate(tick_state)

        # 4. Apply hard negatives (traffic and cost rise proportionally, unit price invariant)
        for svc, (mult, end_t) in list(active_hard_negatives.items()):
            if tick < end_t and svc in tick_state.services:
                sd = tick_state.services[svc]
                sd.request_count *= mult
                sd.usage_amount *= mult
                sd.unblended_cost = sd.usage_amount * sd.unit_price
            elif tick >= end_t:
                active_hard_negatives.pop(svc, None)

        # 5. Apply active injections
        tick_state = registry.apply_all(tick_state)

        # 6. Store row records
        for svc_name, sd in tick_state.services.items():
            rows.append({
                "sim_ts": sim_ts,
                "tick_index": tick,
                "service": svc_name,
                "unblended_cost": float(sd.unblended_cost),
                "request_count": float(sd.request_count),
                "usage_amount": float(sd.usage_amount),
                "unit_price": float(sd.unit_price),
                "is_anomalous": bool(sd.is_anomalous),
                "injection_ids": list(sd.injection_ids),
                "run_id": run_id,
            })

    events_df = pd.DataFrame(rows)

    if cache_dir and noise_sigma_perturb is None:
        try:
            cache_file = cache_dir / f"{spec.name}_s{seed}_{spec_hash}{perturb_suffix}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump({"events_df": events_df, "ground_truth": ground_truth}, f)
        except Exception as e:
            log.warning("Could not write cache file: %s", e)

    return events_df, ground_truth

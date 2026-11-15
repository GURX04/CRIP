"""
generator/config.py
===================
Pydantic-settings configuration for the CRIP event generator.

All values can be overridden via environment variables or a .env file.
The service catalogue (SERVICES) is a pure data dict so traffic_model.py
contains no hard-coded service parameters.
"""

from __future__ import annotations

import os
from typing import Dict, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSpec:
    """Lightweight descriptor for one AWS service in the simulation."""

    __slots__ = ("name", "base_volume", "unit_price", "usage_unit")

    def __init__(
        self,
        name: str,
        base_volume: float,
        unit_price: float,
        usage_unit: str,
    ) -> None:
        self.name = name
        self.base_volume = base_volume   # requests per 5-min interval
        self.unit_price = unit_price     # USD per request / unit
        self.usage_unit = usage_unit     # label only (e.g. "Requests")


# ---------------------------------------------------------------------------
# Canonical service catalogue — single source of truth used by traffic_model
# ---------------------------------------------------------------------------
SERVICES: Dict[str, ServiceSpec] = {
    "AmazonApiGateway": ServiceSpec(
        name="AmazonApiGateway",
        base_volume=5_000,
        unit_price=0.000_003_5,
        usage_unit="Requests",
    ),
    "EC2 Data Transfer": ServiceSpec(
        name="EC2 Data Transfer",
        base_volume=1_000,
        unit_price=0.000_09,
        usage_unit="GB",
    ),
    "DynamoDB": ServiceSpec(
        name="DynamoDB",
        base_volume=8_000,
        unit_price=0.000_001_25,
        usage_unit="ReadCapacityUnit-Hrs",
    ),
    "CloudWatch Logs": ServiceSpec(
        name="CloudWatch Logs",
        base_volume=2_000,
        unit_price=0.000_05,
        usage_unit="GB",
    ),
    "S3 PUT": ServiceSpec(
        name="S3 PUT",
        base_volume=3_000,
        unit_price=0.000_005,
        usage_unit="Requests",
    ),
}

VALID_SERVICE_NAMES = set(SERVICES.keys())
VALID_INJECTION_TYPES = {
    "spike",
    "step_change",
    "slow_drift",
    "cost_only",
    "seasonality_break",
}


class GeneratorSettings(BaseSettings):
    """
    Runtime configuration loaded from environment variables.
    Prefix: GEN_   (e.g. GEN_SEED=42)
    """

    model_config = SettingsConfigDict(
        env_prefix="GEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Simulation clock
    sim_minutes_per_real_second: float = Field(
        default=5.0,
        gt=0,
        description="Simulated minutes that pass per real wall-clock second.",
    )
    sim_start_ts: str = Field(
        default="",
        description="ISO-8601 UTC start timestamp.  Empty = today 00:00 UTC.",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        ge=0,
        description="Master random seed for the numpy Generator.",
    )

    # Noise
    noise_sigma: float = Field(
        default=0.03,
        ge=0,
        description="Gaussian noise std-dev as a fraction of base volume.",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL.",
    )
    redis_stream_events: str = Field(
        default="crip:events",
        description="Redis stream name for cost events.",
    )
    redis_stream_labels: str = Field(
        default="crip:labels",
        description="Redis stream name for anomaly labels (ground-truth side-channel).",
    )
    redis_stream_injections: str = Field(
        default="crip:injections",
        description="Redis stream name for injection lifecycle records.",
    )
    redis_stream_mitigations: str = Field(
        default="crip:mitigations",
        description="Redis stream name for mitigation lifecycle records.",
    )
    redis_maxlen: int = Field(
        default=100_000,
        gt=0,
        description="Approximate MAXLEN for Redis stream trimming.",
    )

    # TimescaleDB
    db_dsn: str = Field(
        default="postgresql://crip_user:crip_secret@localhost:5432/crip",
        description="SQLAlchemy / psycopg2 DSN for TimescaleDB.",
    )

    # Sink
    sink: str = Field(
        default="redis",
        description="Output sink: 'redis' | 'stdout' | 'csv'.",
    )
    csv_path: str = Field(
        default="reports/generator_output.csv",
        description="Output CSV path when sink='csv'.",
    )

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, gt=0, le=65535)

    # Ground truth
    ground_truth_path: str = Field(
        default="reports/ground_truth.jsonl",
        description="Append-mode JSONL file for injection ground truth.",
    )

    @field_validator("sink")
    @classmethod
    def validate_sink(cls, v: str) -> str:
        allowed = {"redis", "stdout", "csv"}
        if v not in allowed:
            raise ValueError(f"sink must be one of {allowed}, got {v!r}")
        return v


# Module-level singleton (lazy); tests may override via env vars before import.
_settings: GeneratorSettings | None = None


def get_settings() -> GeneratorSettings:
    """Return the singleton settings instance, creating it on first call."""
    global _settings
    if _settings is None:
        _settings = GeneratorSettings()
    return _settings


def reset_settings() -> GeneratorSettings:
    """Force re-read of environment variables.  Primarily for tests."""
    global _settings
    _settings = GeneratorSettings()
    return _settings

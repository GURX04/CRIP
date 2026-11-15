"""
remediation/config.py
=====================
Configuration for the CRIP live automated remediation engine.
Loads from environment variables with prefix REM_.
"""

from __future__ import annotations

import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RemediationSettings(BaseSettings):
    """
    Runtime settings for CRIP remediation engine.
    Prefix: REM_ (e.g. REM_MODE=ENFORCE)
    """

    model_config = SettingsConfigDict(
        env_prefix="REM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Operating mode: DRY_RUN | ENFORCE
    mode: str = Field(
        default="DRY_RUN",
        description="Operating mode: DRY_RUN (simulate only) or ENFORCE (call generator).",
    )

    # Master Kill Switch
    kill_switch: bool = Field(
        default=False,
        description="Emergency kill-switch. When True, all actions are blocked immediately.",
    )

    # Redis streams & consumer group
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL.",
    )
    redis_stream_detections: str = Field(
        default="crip:detections",
        description="Stream to read detections from.",
    )
    redis_stream_proposals: str = Field(
        default="crip:remediation_proposals",
        description="Stream to publish remediation proposals to.",
    )
    redis_stream_approvals: str = Field(
        default="crip:remediation_approvals",
        description="Stream for dashboard approval and rejection commands.",
    )
    redis_stream_mitigations: str = Field(
        default="crip:mitigations",
        description="Stream for generator mitigation lifecycle events.",
    )
    redis_stream_results: str = Field(
        default="crip:remediation_results",
        description="Stream to publish remediation results & impact to.",
    )
    redis_stream_scores: str = Field(
        default="crip:scores",
        description="Stream to read detector scores and expected cost baseline from.",
    )
    redis_group: str = Field(
        default="crip:remediation:group",
        description="Redis consumer group name for remediation planner.",
    )
    redis_consumer: str = Field(
        default="remediation_worker_1",
        description="Consumer name within the consumer group.",
    )

    # Generator control plane
    generator_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of CRIP generator API.",
    )

    # Guardrails
    cooldown_sim_min: float = Field(
        default=15.0,
        gt=0,
        description="Per-service cooldown in simulated minutes after applying an action.",
    )
    max_actions_per_hour: int = Field(
        default=6,
        gt=0,
        description="Global rate limit: maximum actions allowed per simulated hour.",
    )
    min_severity: float = Field(
        default=60.0,
        ge=0,
        le=100,
        description="Minimum anomaly severity [0-100] required to consider action.",
    )
    max_throttle_depth: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Maximum throttle depth (factor cannot fall below this, e.g. 0.5 = 50% baseline).",
    )
    auto_approve_max_severity: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Upper severity bound for auto-approval. Higher requires human operator.",
    )
    auto_approve_max_risk: float = Field(
        default=0.15,
        ge=0,
        le=1.0,
        description="Upper risk bound (fraction of legitimate traffic affected) for auto-approval.",
    )
    max_mitigation_duration_min: float = Field(
        default=30.0,
        gt=0,
        description="Maximum duration (simulated minutes) before auto-rollback.",
    )

    # Impact measurement
    value_per_request: float = Field(
        default=0.0001,
        ge=0,
        description="Configurable business value per legitimate request (USD) for collateral impact.",
    )

    # LocalStack Lambda dry-run
    localstack_url: str = Field(
        default="http://localhost:4566",
        description="LocalStack endpoint URL for optional AWS Lambda dry-run trigger.",
    )
    hmac_secret: str = Field(
        default="crip-remediation-hmac-secret-key-321",
        description="Secret key for signing HMAC payloads.",
    )

    # TimescaleDB persistence
    db_dsn: str = Field(
        default="postgresql://crip_user:crip_secret@localhost:5432/crip",
        description="DSN for TimescaleDB audit & results storage.",
    )


_SETTINGS: RemediationSettings | None = None


def get_remediation_settings() -> RemediationSettings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = RemediationSettings()
    return _SETTINGS


def reset_remediation_settings() -> None:
    global _SETTINGS
    _SETTINGS = None


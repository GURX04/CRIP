"""
pipeline/config.py
==================
Centralised configuration for the CRIP real-time streaming detection pipeline.

All settings can be overridden via environment variables (prefix ``PIP_``)
or a ``.env`` file placed next to the process working directory.

Example::

    PIP_REDIS_URL=redis://redis:6379 python -m pipeline.main
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Runtime configuration for every component of the CRIP pipeline.

    Priority (highest → lowest):
    1. Environment variables with the ``PIP_`` prefix.
    2. Values in the ``.env`` file.
    3. Field defaults defined here.
    """

    model_config = SettingsConfigDict(
        env_prefix="PIP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Redis connectivity
    # ------------------------------------------------------------------

    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL used by the ingestion/detection consumer.",
    )

    redis_stream_events: str = Field(
        default="crip:events",
        description="Redis stream key for raw cost events produced by the simulator.",
    )

    redis_stream_detections: str = Field(
        default="crip:detections",
        description="Redis stream key where confirmed anomaly detections are published.",
    )

    redis_stream_scores: str = Field(
        default="crip:scores",
        description="Redis stream key where per-tick per-service score points are published.",
    )

    redis_maxlen: int = Field(
        default=100_000,
        description=(
            "MAXLEN cap (approximate) applied to every XADD on managed streams. "
            "Keeps memory bounded in long-running simulations."
        ),
    )

    # ------------------------------------------------------------------
    # Consumer group / consumer identity
    # ------------------------------------------------------------------

    consumer_group: str = Field(
        default="detectors",
        description="Redis consumer-group name created on the events stream.",
    )

    consumer_name: str = Field(
        default="detector-1",
        description=(
            "Unique name of this consumer within the group. "
            "Override per-replica to avoid duplicate message delivery."
        ),
    )

    # ------------------------------------------------------------------
    # Polling / batching
    # ------------------------------------------------------------------

    batch_size: int = Field(
        default=50,
        description="Maximum number of events fetched per XREADGROUP call (COUNT).",
        ge=1,
    )

    block_ms: int = Field(
        default=500,
        description=(
            "Milliseconds the XREADGROUP call blocks waiting for new messages "
            "before returning an empty result."
        ),
        ge=0,
    )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    dedup_window: int = Field(
        default=10_000,
        description=(
            "Maximum size of the in-memory LRU set used to deduplicate "
            "event_ids seen within a sliding window."
        ),
        ge=1,
    )

    # ------------------------------------------------------------------
    # Database (PostgreSQL)
    # ------------------------------------------------------------------

    db_dsn: str = Field(
        default="postgresql://crip_user:crip_secret@localhost:5432/crip",
        description="SQLAlchemy-compatible DSN for the PostgreSQL persistence store.",
    )

    db_flush_interval_ms: int = Field(
        default=200,
        description=(
            "Maximum time (ms) the async DB writer waits before flushing its "
            "internal buffer even if the buffer has not reached db_flush_batch_size."
        ),
        ge=1,
    )

    db_flush_batch_size: int = Field(
        default=100,
        description="Number of rows that trigger an immediate DB flush.",
        ge=1,
    )

    db_queue_maxsize: int = Field(
        default=5000,
        description=(
            "Capacity of the in-process async queue that buffers detections "
            "before they are written to PostgreSQL."
        ),
        ge=1,
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    metrics_port: int = Field(
        default=9200,
        description="TCP port on which the Prometheus /metrics endpoint is exposed.",
        ge=1024,
        le=65535,
    )

    # ------------------------------------------------------------------
    # Detector warm-up
    # ------------------------------------------------------------------

    warmup_ticks: int = Field(
        default=20,
        description=(
            "Number of ticks consumed before any detector is allowed to emit "
            "an anomaly.  Provides a stable baseline window."
        ),
        ge=0,
    )

    # ------------------------------------------------------------------
    # Z-score detector
    # ------------------------------------------------------------------

    zscore_threshold: float = Field(
        default=3.5,
        description="Number of standard deviations above the rolling mean that triggers a z-score alert.",
        gt=0.0,
    )

    zscore_consec: int = Field(
        default=2,
        description=(
            "Consecutive ticks the z-score must stay above zscore_threshold "
            "before an anomaly is declared (reduces false positives)."
        ),
        ge=1,
    )

    # ------------------------------------------------------------------
    # EWMA control-chart detector
    # ------------------------------------------------------------------

    ewma_alpha: float = Field(
        default=0.1,
        description=(
            "Exponential-smoothing factor for the EWMA detector (0 < α ≤ 1). "
            "Smaller values give a slower-reacting, more stable baseline."
        ),
        gt=0.0,
        le=1.0,
    )

    ewma_k: float = Field(
        default=3.0,
        description=(
            "Control-chart k-sigma multiplier for the EWMA detector. "
            "Alert fires when |EWMA − μ| > k × σ."
        ),
        gt=0.0,
    )

    # ------------------------------------------------------------------
    # CPR (Change-Point Regression) detector
    # ------------------------------------------------------------------

    cpr_alpha: float = Field(
        default=0.05,
        description=(
            "Exponentially-weighted regression forgetting factor (0 < α ≤ 1). "
            "Controls how quickly the model forgets old observations."
        ),
        gt=0.0,
        le=1.0,
    )

    cpr_threshold: float = Field(
        default=3.5,
        description="Residual z-score threshold that declares a CPR anomaly.",
        gt=0.0,
    )

    # ------------------------------------------------------------------
    # Ensemble voting
    # ------------------------------------------------------------------

    vote_threshold: int = Field(
        default=2,
        description=(
            "Minimum number of individual detectors that must independently flag "
            "an anomaly before the ensemble emits a confirmed Detection."
        ),
        ge=1,
    )

    # ------------------------------------------------------------------
    # Cooldown & severity
    # ------------------------------------------------------------------

    cooldown_ticks: int = Field(
        default=12,
        description=(
            "After a Detection is emitted for a service, suppress further "
            "detections on that service for this many ticks (1 tick = 5 "
            "simulated minutes, so 12 = 1 simulated hour). The old default of "
            "1440 ticks was 5 simulated days (~24 min of wall-clock at the "
            "default speed) and silently swallowed every later anomaly."
        ),
        ge=0,
    )

    severity_scale: float = Field(
        default=100.0,
        description=(
            "Multiplicative scale applied when converting a normalised anomaly "
            "score [0, 1] to the human-readable severity value [0, severity_scale]."
        ),
        gt=0.0,
    )

    # ------------------------------------------------------------------
    # Evaluation harness
    # ------------------------------------------------------------------

    eval_redis_url: str = Field(
        default="redis://localhost:6379",
        description=(
            "Redis URL used by the offline evaluation harness. "
            "May point to a separate Redis instance from the live pipeline."
        ),
    )

    eval_fp_tolerance_ticks: int = Field(
        default=2,
        description=(
            "Number of ticks of temporal overlap tolerance used when classifying "
            "a detection as a true positive vs. a false positive during evaluation."
        ),
        ge=0,
    )


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

_settings: Optional[PipelineSettings] = None


def get_settings() -> PipelineSettings:
    """Return the process-wide singleton :class:`PipelineSettings` instance.

    The first call constructs and caches the object; subsequent calls return
    the cached instance without re-reading environment variables or the
    ``.env`` file.  Call :func:`reset_settings` to force a fresh read (useful
    in tests that mutate environment variables between cases).
    """
    global _settings
    if _settings is None:
        _settings = PipelineSettings()
    return _settings


def reset_settings() -> None:
    """Discard the cached settings singleton.

    After calling this, the next :func:`get_settings` call constructs a fresh
    :class:`PipelineSettings` from the current environment.  Intended **only**
    for use in test suites::

        import os
        os.environ["PIP_ZSCORE_THRESHOLD"] = "5.0"
        reset_settings()
        assert get_settings().zscore_threshold == 5.0
    """
    global _settings
    _settings = None
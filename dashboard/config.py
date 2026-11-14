"""
dashboard/config.py
===================
Pydantic-settings configuration for the CRIP demo dashboard backend.

All values can be overridden via environment variables (prefix DASH_) or .env.
"""
from __future__ import annotations
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DASH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8501, gt=0, le=65535)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    redis_stream_events: str = Field(default="crip:events")
    redis_stream_scores: str = Field(default="crip:scores")
    redis_stream_detections: str = Field(default="crip:detections")
    redis_stream_injections: str = Field(default="crip:injections")
    redis_stream_eval: str = Field(default="crip:eval")
    redis_xread_count: int = Field(default=50, gt=0,
        description="Max messages per XREAD call per stream.")
    redis_block_ms: int = Field(default=200, ge=0,
        description="XREAD block timeout ms (0 = non-blocking).")

    # Generator API proxy
    generator_url: str = Field(default="http://localhost:8000",
        description="Base URL of the generator control API.")
    generator_timeout_s: float = Field(default=5.0, gt=0)

    # SSE client queue
    client_queue_maxsize: int = Field(default=200, gt=0,
        description="Per-client bounded SSE queue; old ticks dropped on overflow.")
    heartbeat_interval_s: float = Field(default=5.0, gt=0)

    # History
    history_minutes: int = Field(default=60, gt=0,
        description="Default rolling window for /history endpoint.")

    # Replay
    replay_file: Optional[str] = Field(default=None,
        description="Path to a .jsonl replay file; activates replay mode.")

    # Detector liveness
    detector_stale_s: float = Field(default=10.0, gt=0,
        description="Age in seconds of last score before detector is considered dead.")

    # TimescaleDB (fallback for /history)
    db_dsn: str = Field(
        default="postgresql://crip_user:crip_secret@localhost:5432/crip")

    # Logging
    log_level: str = Field(default="INFO")


_settings: Optional[DashboardSettings] = None


def get_settings() -> DashboardSettings:
    global _settings
    if _settings is None:
        _settings = DashboardSettings()
    return _settings


def reset_settings() -> DashboardSettings:
    global _settings
    _settings = DashboardSettings()
    return _settings


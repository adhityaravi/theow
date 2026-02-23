"""CLI configuration model and profile resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class EngineConfig(BaseModel):
    """Engine-level settings from config file."""

    llm: str | None = None
    llm_secondary: str | None = None
    session_limit: int = 20
    max_tool_calls_per_session: int = 30
    max_tokens_per_session: int = 8192
    archive_llm_attempt: bool = False


class ProfileConfig(BaseModel):
    """Named profile for a recovery scenario."""

    tags: list[str] | None = None
    rules: list[str] | None = None
    collection: str = "default"
    max_retries: int = 3
    max_depth: int = 3
    explore: bool = False
    fallback: bool = True
    plugin: str | None = None
    hint: str | None = None


class TheowConfig(BaseModel):
    """Root config from .theow/config.yaml."""

    engine: EngineConfig = EngineConfig()
    profiles: dict[str, ProfileConfig] = {}


def load_config(theow_dir: Path) -> TheowConfig:
    """Load config from .theow/config.yaml, returning defaults if missing."""
    config_path = theow_dir / "config.yaml"
    if not config_path.exists():
        return TheowConfig()
    data = yaml.safe_load(config_path.read_text()) or {}
    return TheowConfig.model_validate(data)


def resolve_profile(
    config: TheowConfig,
    profile_name: str | None,
    cli_overrides: dict[str, Any],
) -> ProfileConfig:
    """Merge profile defaults with CLI overrides."""
    if profile_name:
        base = config.profiles.get(profile_name)
        if not base:
            raise ValueError(f"Profile not found: {profile_name}")
        profile = base.model_copy()
    else:
        profile = ProfileConfig()

    for key, value in cli_overrides.items():
        if value is not None and hasattr(profile, key):
            setattr(profile, key, value)

    return profile

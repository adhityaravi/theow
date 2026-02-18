"""Tests for CLI config loading and profile resolution."""

import pytest
import yaml

from theow._cli._config import ProfileConfig, TheowConfig, load_config, resolve_profile


def test_load_config_missing_file(tmp_path):
    config = load_config(tmp_path)
    assert config.engine.llm is None
    assert config.profiles == {}


def test_load_config_from_yaml(tmp_path):
    data = {
        "engine": {"llm": "gemini/flash", "session_limit": 10},
        "profiles": {
            "lint": {"tags": ["lint"], "max_retries": 2, "explore": False},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(data))

    config = load_config(tmp_path)
    assert config.engine.llm == "gemini/flash"
    assert config.engine.session_limit == 10
    assert config.profiles["lint"].tags == ["lint"]
    assert config.profiles["lint"].max_retries == 2


def test_resolve_profile_with_overrides():
    config = TheowConfig(profiles={"ci": ProfileConfig(tags=["ci"], max_retries=2, explore=False)})
    profile = resolve_profile(config, "ci", {"explore": True, "plugin": "./fix.py"})

    assert profile.tags == ["ci"]
    assert profile.max_retries == 2
    assert profile.explore is True
    assert profile.plugin == "./fix.py"


def test_resolve_profile_no_profile_uses_defaults():
    config = TheowConfig()
    profile = resolve_profile(config, None, {"explore": True})

    assert profile.explore is True
    assert profile.max_retries == 3
    assert profile.collection == "default"


def test_resolve_profile_not_found():
    config = TheowConfig()
    with pytest.raises(ValueError, match="Profile not found"):
        resolve_profile(config, "nonexistent", {})


def test_resolve_profile_none_overrides_ignored():
    config = TheowConfig(profiles={"ci": ProfileConfig(tags=["ci"], plugin="orig.py")})
    profile = resolve_profile(config, "ci", {"plugin": None})
    assert profile.plugin == "orig.py"

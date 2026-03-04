"""Logfire / OpenTelemetry configuration."""

from __future__ import annotations

from dataclasses import dataclass

import logfire


@dataclass
class LogfireConfig:
    """Configuration for Logfire/OpenTelemetry instrumentation."""

    enabled: bool = False
    send_to_logfire: bool = True  # False = OTel-only via OTEL_* env vars
    service_name: str | None = None  # Defaults to Theow engine name


def configure_logfire(config: LogfireConfig, service_name: str) -> None:
    """Configure Logfire/OTel instrumentation.

    PydanticAI auto-instrumentation is intentionally NOT enabled — Theow's
    @instrumented decorator provides gen_ai.* spans for all gateways
    (PydanticAI, Copilot, etc.) to avoid double-counted token usage.
    """
    if not config.enabled:
        return
    logfire.configure(
        service_name=config.service_name or service_name,
        send_to_logfire=config.send_to_logfire,
        console=False,
    )

"""OpenTelemetry monitoring for the Serenity API."""

from deployment.monitoring.instrument import instrument_app

__all__ = ["instrument_app"]

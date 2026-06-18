"""Application-level OpenTelemetry metrics for NLP, data, and server signals."""

from __future__ import annotations

import time
from typing import Protocol

from opentelemetry import metrics
from opentelemetry.metrics import Observation

_START_TIME = time.time()
_app_metrics: AppMetricsProtocol | None = None


class AppMetricsProtocol(Protocol):
    def record_intent(self, intent: str) -> None: ...

    def record_message_length(self, length: int) -> None: ...

    def record_feedback(self, vote: str) -> None: ...


class _NoOpAppMetrics:
    def record_intent(self, intent: str) -> None:
        return None

    def record_message_length(self, length: int) -> None:
        return None

    def record_feedback(self, vote: str) -> None:
        return None


class AppMetrics:
    """Custom metrics exported alongside auto-instrumented HTTP metrics."""

    def __init__(self, meter: metrics.Meter):
        self._intent_counter = meter.create_counter(
            name="nlp.intent.classified",
            unit="1",
            description="Count of classified user intents",
        )
        self._message_length_histogram = meter.create_histogram(
            name="data.chat.message.length",
            unit="By",
            description="Length of incoming chat messages in characters",
        )
        self._feedback_counter = meter.create_counter(
            name="data.feedback.vote",
            unit="1",
            description="User feedback votes on bot responses",
        )
        meter.create_observable_gauge(
            name="server.process.uptime",
            unit="s",
            description="Seconds since the API process started",
            callbacks=[self._observe_uptime],
        )

    @staticmethod
    def _observe_uptime(options) -> list[Observation]:
        return [Observation(time.time() - _START_TIME)]

    def record_intent(self, intent: str) -> None:
        self._intent_counter.add(1, {"intent": intent})

    def record_message_length(self, length: int) -> None:
        self._message_length_histogram.record(length)

    def record_feedback(self, vote: str) -> None:
        self._feedback_counter.add(1, {"vote": vote})


def init_app_metrics(meter: metrics.Meter) -> AppMetrics:
    global _app_metrics
    _app_metrics = AppMetrics(meter)
    return _app_metrics


def get_app_metrics() -> AppMetricsProtocol:
    return _app_metrics or _NoOpAppMetrics()

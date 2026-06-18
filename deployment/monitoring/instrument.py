from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

from deployment.core.logging import setup_logging
from deployment.monitoring.app_metrics import init_app_metrics
from deployment.monitoring.config import otel_settings
from deployment.monitoring.middleware import OtelMetricsMiddleware
from deployment.monitoring.otel import setup_logs, setup_metrics, setup_traces

logger = setup_logging()
_instrumented = False


def instrument_app(app: FastAPI) -> FastAPI:
    """Attach OpenTelemetry instrumentation to an existing FastAPI app."""
    global _instrumented

    if _instrumented:
        return app

    if not otel_settings.enabled:
        logger.info(
            "OpenTelemetry monitoring disabled (set OTEL_EXPORTER_OTLP_* to enable)"
        )
        return app

    setup_traces(otel_settings)
    meter = setup_metrics(otel_settings)
    init_app_metrics(meter)
    setup_logs(otel_settings)

    LoggingInstrumentor().instrument(set_logging_format=True)
    HTTPXClientInstrumentor().instrument()
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="/health",
    )

    if otel_settings.metrics_endpoint():
        app.add_middleware(OtelMetricsMiddleware, meter=meter)

    _instrumented = True
    logger.info(
        "OpenTelemetry monitoring enabled for service=%s",
        otel_settings.OTEL_SERVICE_NAME,
    )
    return app

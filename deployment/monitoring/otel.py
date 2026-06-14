import logging

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    SERVICE_VERSION,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from deployment.monitoring.config import OtelSettings

logger = logging.getLogger("serenity.monitoring")


def _build_resource(settings: OtelSettings) -> Resource:
    return Resource.create(
        {
            SERVICE_NAME: settings.OTEL_SERVICE_NAME,
            SERVICE_VERSION: settings.OTEL_SERVICE_VERSION,
            DEPLOYMENT_ENVIRONMENT: settings.OTEL_DEPLOYMENT_ENVIRONMENT,
        }
    )


def setup_traces(settings: OtelSettings) -> None:
    endpoint = settings.traces_endpoint()
    if not endpoint:
        logger.warning("OTEL trace export disabled; no OTLP traces endpoint configured")
        return

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers=settings.traces_headers(),
    )
    provider = TracerProvider(resource=_build_resource(settings))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry trace exporter configured: %s", endpoint)


def setup_metrics(settings: OtelSettings) -> metrics.Meter:
    endpoint = settings.metrics_endpoint()
    if not endpoint:
        logger.warning(
            "OTEL metrics export disabled; no OTLP metrics endpoint configured"
        )
        return metrics.get_meter(settings.OTEL_SERVICE_NAME)

    exporter = OTLPMetricExporter(
        endpoint=endpoint,
        headers=settings.metrics_headers(),
    )
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15000)
    provider = MeterProvider(resource=_build_resource(settings), metric_readers=[reader])
    metrics.set_meter_provider(provider)
    logger.info("OpenTelemetry metrics exporter configured: %s", endpoint)
    return metrics.get_meter(settings.OTEL_SERVICE_NAME)


def setup_logs(settings: OtelSettings) -> None:
    endpoint = settings.logs_endpoint()
    if not endpoint:
        logger.warning("OTEL log export disabled; no OTLP logs endpoint configured")
        return

    exporter = OTLPLogExporter(
        endpoint=endpoint,
        headers=settings.logs_headers(),
    )
    provider = LoggerProvider(resource=_build_resource(settings))
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)
    logging.getLogger().addHandler(handler)
    logger.info("OpenTelemetry log exporter configured: %s", endpoint)

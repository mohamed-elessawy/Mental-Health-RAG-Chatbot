from pydantic import ConfigDict
from pydantic_settings import BaseSettings


def parse_otlp_headers(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        key, _, value = pair.partition("=")
        key = key.strip()
        if key:
            headers[key] = value.strip()
    return headers


class OtelSettings(BaseSettings):
    model_config = ConfigDict(extra="ignore")

    OTEL_ENABLED: bool = True
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""
    OTEL_EXPORTER_OTLP_HEADERS: str = ""
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: str = ""
    OTEL_EXPORTER_OTLP_TRACES_HEADERS: str = ""
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: str = ""
    OTEL_EXPORTER_OTLP_METRICS_HEADERS: str = ""
    OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: str = ""
    OTEL_EXPORTER_OTLP_LOGS_HEADERS: str = ""
    OTEL_SERVICE_NAME: str = "serenity-mental-health-api"
    OTEL_SERVICE_VERSION: str = "0.1.0"
    OTEL_DEPLOYMENT_ENVIRONMENT: str = "development"

    @property
    def enabled(self) -> bool:
        if not self.OTEL_ENABLED:
            return False
        return bool(
            self.OTEL_EXPORTER_OTLP_ENDPOINT
            or self.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
            or self.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT
            or self.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT
        )

    def traces_endpoint(self) -> str:
        if self.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:
            return self.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
        if self.OTEL_EXPORTER_OTLP_ENDPOINT:
            return f"{self.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip('/')}/v1/traces"
        return ""

    def metrics_endpoint(self) -> str:
        if self.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT:
            return self.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT
        if self.OTEL_EXPORTER_OTLP_ENDPOINT:
            return f"{self.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip('/')}/v1/metrics"
        return ""

    def logs_endpoint(self) -> str:
        if self.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT:
            return self.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT
        if self.OTEL_EXPORTER_OTLP_ENDPOINT:
            return f"{self.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip('/')}/v1/logs"
        return ""

    def traces_headers(self) -> dict[str, str]:
        if self.OTEL_EXPORTER_OTLP_TRACES_HEADERS:
            return parse_otlp_headers(self.OTEL_EXPORTER_OTLP_TRACES_HEADERS)
        return parse_otlp_headers(self.OTEL_EXPORTER_OTLP_HEADERS)

    def metrics_headers(self) -> dict[str, str]:
        if self.OTEL_EXPORTER_OTLP_METRICS_HEADERS:
            return parse_otlp_headers(self.OTEL_EXPORTER_OTLP_METRICS_HEADERS)
        return parse_otlp_headers(self.OTEL_EXPORTER_OTLP_HEADERS)

    def logs_headers(self) -> dict[str, str]:
        if self.OTEL_EXPORTER_OTLP_LOGS_HEADERS:
            return parse_otlp_headers(self.OTEL_EXPORTER_OTLP_LOGS_HEADERS)
        return parse_otlp_headers(self.OTEL_EXPORTER_OTLP_HEADERS)


otel_settings = OtelSettings()

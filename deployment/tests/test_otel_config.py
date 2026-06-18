from deployment.monitoring.config import OtelSettings


def test_axiom_token_enables_export():
    settings = OtelSettings(
        OTEL_ENABLED=True,
        AXIOM_TOKEN="test-token",
    )
    assert settings.enabled is True


def test_axiom_token_builds_endpoints_and_headers():
    settings = OtelSettings(
        AXIOM_TOKEN="test-token",
        AXIOM_TRACES_DATASET="my-traces",
        AXIOM_METRICS_DATASET="my-metrics",
        AXIOM_LOGS_DATASET="my-logs",
    )
    assert settings.traces_endpoint() == "https://api.axiom.co/v1/traces"
    assert settings.metrics_endpoint() == "https://api.axiom.co/v1/metrics"
    assert settings.logs_endpoint() == "https://api.axiom.co/v1/logs"

    traces_headers = settings.traces_headers()
    assert traces_headers["Authorization"] == "Bearer test-token"
    assert traces_headers["X-Axiom-Dataset"] == "my-traces"

    metrics_headers = settings.metrics_headers()
    assert metrics_headers["X-Axiom-Metrics-Dataset"] == "my-metrics"


def test_disabled_when_no_token_or_endpoints():
    settings = OtelSettings(OTEL_ENABLED=True, AXIOM_TOKEN="")
    assert settings.enabled is False

    settings = OtelSettings(OTEL_ENABLED=False, AXIOM_TOKEN="test-token")
    assert settings.enabled is False

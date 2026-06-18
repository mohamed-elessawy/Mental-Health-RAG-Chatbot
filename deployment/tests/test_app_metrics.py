from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from deployment.monitoring.app_metrics import AppMetrics, init_app_metrics


def test_record_intent_increments_counter():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    app_metrics = init_app_metrics(meter)

    app_metrics.record_intent("greeting")
    app_metrics.record_intent("asking_mental_health_question")

    data = reader.get_metrics_data()
    assert data is not None
    metrics_list = data.resource_metrics[0].scope_metrics[0].metrics
    intent_metric = next(m for m in metrics_list if m.name == "nlp.intent.classified")
    total = sum(point.value for point in intent_metric.data.data_points)
    assert total == 2


def test_record_message_length_records_histogram():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    app_metrics = AppMetrics(meter)

    app_metrics.record_message_length(42)
    app_metrics.record_message_length(100)

    data = reader.get_metrics_data()
    assert data is not None
    metrics_list = data.resource_metrics[0].scope_metrics[0].metrics
    length_metric = next(
        m for m in metrics_list if m.name == "data.chat.message.length"
    )
    assert len(length_metric.data.data_points) >= 1


def test_record_feedback_increments_counter():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    app_metrics = AppMetrics(meter)

    app_metrics.record_feedback("up")
    app_metrics.record_feedback("down")

    data = reader.get_metrics_data()
    assert data is not None
    metrics_list = data.resource_metrics[0].scope_metrics[0].metrics
    feedback_metric = next(m for m in metrics_list if m.name == "data.feedback.vote")
    total = sum(point.value for point in feedback_metric.data.data_points)
    assert total == 2


def test_get_app_metrics_noop_when_not_initialized():
    from deployment.monitoring import app_metrics as app_metrics_module

    original = app_metrics_module._app_metrics
    app_metrics_module._app_metrics = None
    try:
        noop = app_metrics_module.get_app_metrics()
        noop.record_intent("greeting")
        noop.record_message_length(10)
        noop.record_feedback("up")
        noop.record_rag_scores([0.5])
    finally:
        app_metrics_module._app_metrics = original

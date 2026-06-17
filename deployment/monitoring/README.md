# OpenTelemetry Monitoring

OpenTelemetry-based API monitoring for the Serenity Mental Health Chatbot. Exports traces, metrics, and logs via OTLP to any compatible backend (Jaeger, Grafana Cloud, Axiom, Honeycomb, etc.).

**No existing application code is modified.** Instrumentation lives entirely in `deployment/monitoring/` and is activated via a separate entry point.

## What gets monitored

| Signal | What's captured |
|--------|-----------------|
| **Traces** | HTTP requests, FastAPI routes, outbound httpx calls (LLM/Qdrant) |
| **Metrics** | Request count, duration histogram, active requests per route |
| **Logs** | Application logs with trace correlation |

The `/health` endpoint is excluded from trace instrumentation to reduce noise.

## Setup

### 1. Configure OTLP export

Add OpenTelemetry variables to `deployment/.env`. Example with a local [OTel Collector](https://opentelemetry.io/docs/collector/):

```dotenv
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=serenity-mental-health-api
OTEL_SERVICE_VERSION=0.1.0
OTEL_DEPLOYMENT_ENVIRONMENT=development
```

Example with per-signal endpoints (e.g. Axiom, Grafana Cloud):

```dotenv
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://api.axiom.co/v1/traces
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://api.axiom.co/v1/metrics
OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=https://api.axiom.co/v1/logs
OTEL_EXPORTER_OTLP_TRACES_HEADERS=Authorization=Bearer YOUR_TOKEN,X-Axiom-Dataset=serenity-traces
OTEL_EXPORTER_OTLP_METRICS_HEADERS=Authorization=Bearer YOUR_TOKEN,X-Axiom-Metrics-Dataset=serenity-metrics
OTEL_EXPORTER_OTLP_LOGS_HEADERS=Authorization=Bearer YOUR_TOKEN,X-Axiom-Dataset=serenity-logs
OTEL_SERVICE_NAME=serenity-mental-health-api
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the instrumented API

```bash
uv run uvicorn deployment.main_otel:app --reload
```

The original entry point (`deployment.main:app`) is unchanged.

## Docker

```bash
docker build -t mental-health-chatbot .
docker run --env-file deployment/.env -p 8000:8000 \
  mental-health-chatbot \
  uv run uvicorn deployment.main_otel:app --host 0.0.0.0 --port 8000
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OTEL_ENABLED` | No | `true` | Set `false` to disable export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No* | — | Base OTLP/HTTP endpoint (auto-appends `/v1/traces`, etc.) |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | — | Shared headers: `key=value,key2=value2` |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | No* | — | Traces endpoint (overrides base) |
| `OTEL_EXPORTER_OTLP_TRACES_HEADERS` | No | — | Traces-specific headers |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | No* | — | Metrics endpoint (overrides base) |
| `OTEL_EXPORTER_OTLP_METRICS_HEADERS` | No | — | Metrics-specific headers |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | No* | — | Logs endpoint (overrides base) |
| `OTEL_EXPORTER_OTLP_LOGS_HEADERS` | No | — | Logs-specific headers |
| `OTEL_SERVICE_NAME` | No | `serenity-mental-health-api` | Service name in telemetry |
| `OTEL_SERVICE_VERSION` | No | `0.1.0` | Service version |
| `OTEL_DEPLOYMENT_ENVIRONMENT` | No | `development` | Deployment environment label |

\* At least one endpoint (`OTEL_EXPORTER_OTLP_ENDPOINT` or a per-signal endpoint) must be set for export to activate.

## Architecture

```
deployment/main.py          ← unchanged FastAPI app
deployment/main_otel.py     ← imports app + calls instrument_app()
deployment/monitoring/
  ├── config.py             ← OTEL env settings
  ├── otel.py               ← OTLP exporters (traces, metrics, logs)
  ├── middleware.py         ← HTTP request metrics
  └── instrument.py         ← wires everything together
```

## Disabling monitoring

- Set `OTEL_ENABLED=false`, or leave all OTLP endpoints empty
- Run the original entry point: `uvicorn deployment.main:app`

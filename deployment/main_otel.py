"""
Instrumented API entry point.

Imports the existing FastAPI app from deployment.main and attaches OpenTelemetry
instrumentation without modifying any existing application code.

Run with:
    uv run uvicorn deployment.main_otel:app --reload
"""

from deployment.main import app
from deployment.monitoring import instrument_app

app = instrument_app(app)

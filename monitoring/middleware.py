import time

from opentelemetry import metrics
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match


class OtelMetricsMiddleware(BaseHTTPMiddleware):
    """Records HTTP request duration and count via OpenTelemetry metrics."""

    def __init__(self, app, meter: metrics.Meter):
        super().__init__(app)
        self._request_duration = meter.create_histogram(
            name="http.server.request.duration",
            unit="s",
            description="Duration of HTTP requests",
        )
        self._request_count = meter.create_counter(
            name="http.server.request.count",
            unit="1",
            description="Total number of HTTP requests",
        )
        self._active_requests = meter.create_up_down_counter(
            name="http.server.active_requests",
            unit="1",
            description="Number of active HTTP requests",
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        route = self._resolve_route(request)
        attributes = {
            "http.method": request.method,
            "http.route": route,
        }

        self._active_requests.add(1, attributes)
        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            raise
        finally:
            duration = time.perf_counter() - start
            result_attributes = {
                **attributes,
                "http.status_code": status_code,
            }
            self._request_duration.record(duration, result_attributes)
            self._request_count.add(1, result_attributes)
            self._active_requests.add(-1, attributes)

    @staticmethod
    def _resolve_route(request: Request) -> str:
        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                return getattr(route, "path", request.url.path)
        return request.url.path

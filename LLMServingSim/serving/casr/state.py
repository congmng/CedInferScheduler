"""Optional Prometheus-compatible state collection for CASR."""

from __future__ import annotations

import re
from urllib.request import Request, urlopen


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)")


def parse_prometheus_text(payload):
    """Parse numeric Prometheus samples without requiring prometheus_client."""
    metrics = {}
    for line in str(payload).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match:
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                continue
    return metrics


class PrometheusStateCollector:
    """Collect worker and link metrics from configured exporter endpoints.

    Endpoint entries have the form ``{"id": "p0", "url": "http://..."}``.
    The collector is best-effort: a failed endpoint is reported in ``errors``
    while other endpoints remain available to the control tick.
    """

    def __init__(self, config=None):
        config = config or {}
        self.timeout_s = max(0.01, float(config.get("timeout_ms", 100)) / 1000.0)
        self.endpoints = tuple(config.get("endpoints", ()))

    @property
    def enabled(self):
        return bool(self.endpoints)

    def collect(self):
        result = {"enabled": self.enabled, "workers": {}, "links": {}, "errors": []}
        for endpoint in self.endpoints:
            endpoint_id = str(endpoint.get("id", endpoint.get("url", "unknown")))
            url = endpoint.get("url")
            if not url:
                result["errors"].append({"id": endpoint_id, "error": "missing url"})
                continue
            try:
                request = Request(url, headers={"Accept": "text/plain"})
                with urlopen(request, timeout=self.timeout_s) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                metrics = parse_prometheus_text(payload)
                role = str(endpoint.get("role", "worker"))
                target = result["links"] if role == "link" else result["workers"]
                target[endpoint_id] = metrics
            except Exception as exc:
                result["errors"].append({"id": endpoint_id, "error": str(exc)})
        return result

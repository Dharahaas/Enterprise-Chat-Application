"""Prometheus-style metrics collection.

A lightweight in-memory metrics registry that exposes a /metrics endpoint
in Prometheus exposition format. No external dependency — sufficient for
single-instance deployments. For multi-instance, swap with prometheus_client.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import Callable


class Metrics:
    """In-memory metrics registry (counters + histograms)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._started_at = time.time()

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(value)
            # Keep last 1000 samples
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    def counter_value(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def histogram_stats(self, name: str) -> dict[str, float]:
        samples = self._histograms.get(name, [])
        if not samples:
            return {"count": 0, "p50": 0, "p95": 0, "max": 0, "sum": 0}
        sorted_s = sorted(samples)
        n = len(sorted_s)
        return {
            "count": n,
            "p50": sorted_s[n // 2],
            "p95": sorted_s[min(int(n * 0.95), n - 1)],
            "max": sorted_s[-1],
            "sum": sum(sorted_s),
        }

    def expose(self) -> str:
        """Render in Prometheus exposition format."""
        lines = [
            f"# Enterprise Knowledge Assistant metrics",
            f"# Uptime: {int(time.time() - self._started_at)}s",
            "",
        ]
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            for name, samples in sorted(self._histograms.items()):
                stats = self.histogram_stats(name)
                lines.append(f"# TYPE {name} summary")
                lines.append(f'{name}_count {stats["count"]}')
                lines.append(f'{name}_sum {stats["sum"]}')
                lines.append(f'{name}{{"quantile":"0.5"}} {stats["p50"]}')
                lines.append(f'{name}{{"quantile":"0.95"}} {stats["p95"]}')
                lines.append(f'{name}{{"quantile":"1.0"}} {stats["max"]}')
        return "\n".join(lines) + "\n"


# Singleton
metrics = Metrics()

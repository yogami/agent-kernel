"""Telemetry tracking for spans, tokens, cost, latency, and write-gate yield."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TelemetrySpan:
    """Telemetry record for a single execution span or turn."""
    span_id: str
    session_id: str
    span_type: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "SUCCESS"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetryCollector:
    """In-memory and persistent telemetry aggregator for agent operations."""

    def __init__(self) -> None:
        self.spans: list[TelemetrySpan] = []

    def record_span(self, span: TelemetrySpan) -> None:
        """Record an execution span."""
        self.spans.append(span)

    def get_summary_metrics(self) -> dict[str, Any]:
        """Compute aggregate system metrics across recorded spans."""
        if not self.spans:
            return {
                "total_spans": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "success_rate": 1.0,
            }

        total_tokens = sum(s.tokens_in + s.tokens_out for s in self.spans)
        total_cost = sum(s.cost_usd for s in self.spans)
        latencies = sorted([s.latency_ms for s in self.spans])
        avg_lat = sum(latencies) / len(latencies)
        p95_idx = int(len(latencies) * 0.95)
        p95_lat = latencies[min(p95_idx, len(latencies) - 1)]

        success_count = sum(1 for s in self.spans if s.status == "SUCCESS")
        success_rate = success_count / len(self.spans)

        return {
            "total_spans": len(self.spans),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "avg_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "success_rate": success_rate,
        }

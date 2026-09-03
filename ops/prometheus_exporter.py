"""Prometheus Metrics Collector and Exporter for Agent Kernel telemetry."""

from __future__ import annotations

from collections import Counter, defaultdict
import time
from typing import Any


class PrometheusMetricsCollector:
    """Collects request counters, TTFT and latency histograms, and error distributions."""

    def __init__(self) -> None:
        self.latency_buckets = [25.0, 50.0, 100.0, 200.0, 350.0, 500.0, 1000.0, 2500.0, 5000.0]
        self.ttft_buckets = [50.0, 100.0, 200.0, 350.0, 500.0, 1000.0]

        self.requests_total: Counter[str] = Counter()
        self.tokens_in_total: Counter[str] = Counter()
        self.tokens_out_total: Counter[str] = Counter()
        self.cost_usd_total: float = 0.0
        self.errors_total: Counter[str] = Counter()

        # Histogram tracking: (model_id) -> bucket_counts, count, sum
        self.latency_hist: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"counts": [0] * len(self.latency_buckets), "count": 0, "sum": 0.0}
        )
        self.ttft_hist: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"counts": [0] * len(self.ttft_buckets), "count": 0, "sum": 0.0}
        )

    def record_turn(
        self,
        model_id: str,
        latency_ms: float,
        ttft_ms: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        status: str = "success",
        error_class: str | None = None,
    ) -> None:
        """Record telemetry from a single interaction turn."""
        self.requests_total[f"status=\"{status}\",model=\"{model_id}\""] += 1
        self.tokens_in_total[f"model=\"{model_id}\""] += tokens_in
        self.tokens_out_total[f"model=\"{model_id}\""] += tokens_out
        self.cost_usd_total += cost_usd

        if error_class:
            self.errors_total[f"class=\"{error_class}\",model=\"{model_id}\""] += 1

        # Record total latency histogram
        entry = self.latency_hist[model_id]
        entry["count"] += 1
        entry["sum"] += latency_ms
        for i, b in enumerate(self.latency_buckets):
            if latency_ms <= b:
                entry["counts"][i] += 1

        # Record TTFT histogram
        ttft_entry = self.ttft_hist[model_id]
        ttft_entry["count"] += 1
        ttft_entry["sum"] += ttft_ms
        for i, b in enumerate(self.ttft_buckets):
            if ttft_ms <= b:
                ttft_entry["counts"][i] += 1

    def generate_prometheus_format(self) -> str:
        """Generate standard Prometheus exposition text."""
        lines: list[str] = [
            "# HELP agent_requests_total Total count of processed agent requests.",
            "# TYPE agent_requests_total counter",
        ]
        for label, count in self.requests_total.items():
            lines.append(f"agent_requests_total{{{label}}} {count}")

        lines.extend([
            "# HELP agent_tokens_input_total Total input tokens processed.",
            "# TYPE agent_tokens_input_total counter",
        ])
        for label, count in self.tokens_in_total.items():
            lines.append(f"agent_tokens_input_total{{{label}}} {count}")

        lines.extend([
            "# HELP agent_tokens_output_total Total generated output tokens.",
            "# TYPE agent_tokens_output_total counter",
        ])
        for label, count in self.tokens_out_total.items():
            lines.append(f"agent_tokens_output_total{{{label}}} {count}")

        lines.extend([
            "# HELP agent_cost_usd_total Cumulative cost incurred in USD.",
            "# TYPE agent_cost_usd_total counter",
            f"agent_cost_usd_total {self.cost_usd_total:.6f}",
        ])

        if self.errors_total:
            lines.extend([
                "# HELP agent_errors_total Count of execution exceptions by error class.",
                "# TYPE agent_errors_total counter",
            ])
            for label, count in self.errors_total.items():
                lines.append(f"agent_errors_total{{{label}}} {count}")

        # Latency Histograms
        lines.extend([
            "# HELP agent_turn_latency_ms Turn execution latency in milliseconds.",
            "# TYPE agent_turn_latency_ms histogram",
        ])
        for model_id, entry in self.latency_hist.items():
            for i, b in enumerate(self.latency_buckets):
                lines.append(f'agent_turn_latency_ms_bucket{{model="{model_id}",le="{b}"}} {entry["counts"][i]}')
            lines.append(f'agent_turn_latency_ms_bucket{{model="{model_id}",le="+Inf"}} {entry["count"]}')
            lines.append(f'agent_turn_latency_ms_sum{{model="{model_id}"}} {entry["sum"]:.2f}')
            lines.append(f'agent_turn_latency_ms_count{{model="{model_id}"}} {entry["count"]}')

        # TTFT Histograms
        lines.extend([
            "# HELP agent_ttft_ms Time-to-First-Token in milliseconds.",
            "# TYPE agent_ttft_ms histogram",
        ])
        for model_id, entry in self.ttft_hist.items():
            for i, b in enumerate(self.ttft_buckets):
                lines.append(f'agent_ttft_ms_bucket{{model="{model_id}",le="{b}"}} {entry["counts"][i]}')
            lines.append(f'agent_ttft_ms_bucket{{model="{model_id}",le="+Inf"}} {entry["count"]}')
            lines.append(f'agent_ttft_ms_sum{{model="{model_id}"}} {entry["sum"]:.2f}')
            lines.append(f'agent_ttft_ms_count{{model="{model_id}"}} {entry["count"]}')

        return "\n".join(lines) + "\n"


# Global singleton instance
metrics_collector = PrometheusMetricsCollector()

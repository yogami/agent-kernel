"""Prometheus Metrics Collector and Exporter for Agent Kernel telemetry."""

from __future__ import annotations

from collections import Counter, defaultdict
import threading
import time
from typing import Any


class PrometheusMetricsCollector:
    """In-memory Prometheus-compatible metrics aggregator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latency_buckets = [100.0, 300.0, 500.0, 1000.0, 3000.0, 5000.0, 10000.0]
        self.ttft_buckets = [50.0, 100.0, 200.0, 350.0, 500.0, 1000.0]

        self.requests_total: Counter[str] = Counter()
        self.tokens_in_total: Counter[str] = Counter()
        self.tokens_out_total: Counter[str] = Counter()
        self.cost_usd_total: float = 0.0
        self.errors_total: Counter[str] = Counter()

        # Operational counters
        self.tool_executions_total: Counter[str] = Counter()
        self.fsm_transitions_total: Counter[str] = Counter()
        self.memory_admissions_total: Counter[str] = Counter()
        self.spans_total: Counter[str] = Counter()

        # Raw latencies for dynamic percentiles: model_id -> list of latency_ms
        self.raw_latencies: dict[str, list[float]] = defaultdict(list)
        self.max_raw_samples: int = 5000

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
        with self._lock:
            self.requests_total[f'status="{status}",model="{model_id}"'] += 1
            self.tokens_in_total[f'model="{model_id}"'] += tokens_in
            self.tokens_out_total[f'model="{model_id}"'] += tokens_out
            self.cost_usd_total += cost_usd
    
            if error_class:
                self.errors_total[f'class="{error_class}",model="{model_id}"'] += 1
    
            # Track raw latency for P50, P90, P99 calculation
            lat_list = self.raw_latencies[model_id]
            lat_list.append(latency_ms)
            if len(lat_list) > self.max_raw_samples:
                self.raw_latencies[model_id] = lat_list[-self.max_raw_samples :]
    
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

    def record_tool_execution(self, tool_name: str, status: str = "success") -> None:
        """Record invocation outcome of an agent tool."""
        with self._lock:
            self.tool_executions_total[f'tool="{tool_name}",status="{status}"'] += 1

    def record_fsm_transition(self, from_state: str, to_state: str) -> None:
        """Record a state machine transition step."""
        with self._lock:
            self.fsm_transitions_total[f'from="{from_state}",to="{to_state}"'] += 1

    def record_memory_admission(self, decision: str) -> None:
        """Record candidate memory admission decision."""
        with self._lock:
            self.memory_admissions_total[f'decision="{decision}"'] += 1

    def record_span(self, name: str, status: str = "OK") -> None:
        """Record completed tracing span execution."""
        with self._lock:
            self.spans_total[f'name="{name}",status="{status}"'] += 1

    def compute_percentiles(self, model_id: str | None = None) -> dict[str, float]:
        """Compute live P50, P90, and P99 latency percentiles in milliseconds."""
        with self._lock:
            all_lats = []
            if model_id:
                if model_id in self.raw_latencies:
                    all_lats = sorted(self.raw_latencies[model_id])
            else:
                for lats in self.raw_latencies.values():
                    all_lats.extend(lats)
                all_lats.sort()

            if not all_lats:
                return {"p50": 0.0, "p90": 0.0, "p99": 0.0}

            n = len(all_lats)
            def quantile(q: float) -> float:
                idx = int(round(q * (n - 1)))
                return all_lats[max(0, min(idx, n - 1))]
            
            return {
                "p50": quantile(0.50),
                "p90": quantile(0.90),
                "p99": quantile(0.99),
            }

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

        # Live Percentiles
        lines.extend([
            "# HELP agent_turn_latency_p50_ms 50th percentile turn execution latency.",
            "# TYPE agent_turn_latency_p50_ms gauge",
            "# HELP agent_turn_latency_p90_ms 90th percentile turn execution latency.",
            "# TYPE agent_turn_latency_p90_ms gauge",
            "# HELP agent_turn_latency_p99_ms 99th percentile turn execution latency.",
            "# TYPE agent_turn_latency_p99_ms gauge",
        ])
        for model_id in self.raw_latencies:
            p = self.compute_percentiles(model_id)
            lines.append(f'agent_turn_latency_p50_ms{{model="{model_id}"}} {p["p50"]:.2f}')
            lines.append(f'agent_turn_latency_p90_ms{{model="{model_id}"}} {p["p90"]:.2f}')
            lines.append(f'agent_turn_latency_p99_ms{{model="{model_id}"}} {p["p99"]:.2f}')

        # Tool Execution Counters
        if self.tool_executions_total:
            lines.extend([
                "# HELP agent_tool_executions_total Total count of tool invocations.",
                "# TYPE agent_tool_executions_total counter",
            ])
            for label, count in self.tool_executions_total.items():
                lines.append(f"agent_tool_executions_total{{{label}}} {count}")

        # FSM Transition Counters
        if self.fsm_transitions_total:
            lines.extend([
                "# HELP agent_fsm_transitions_total Total state machine transitions.",
                "# TYPE agent_fsm_transitions_total counter",
            ])
            for label, count in self.fsm_transitions_total.items():
                lines.append(f"agent_fsm_transitions_total{{{label}}} {count}")

        # Memory Admissions
        if self.memory_admissions_total:
            lines.extend([
                "# HELP agent_memory_admissions_total Memory admission outcomes.",
                "# TYPE agent_memory_admissions_total counter",
            ])
            for label, count in self.memory_admissions_total.items():
                lines.append(f"agent_memory_admissions_total{{{label}}} {count}")

        # Spans Total
        if self.spans_total:
            lines.extend([
                "# HELP agent_spans_total Completed tracing span records.",
                "# TYPE agent_spans_total counter",
            ])
            for label, count in self.spans_total.items():
                lines.append(f"agent_spans_total{{{label}}} {count}")

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


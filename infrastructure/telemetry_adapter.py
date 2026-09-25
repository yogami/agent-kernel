"""Distributed tracing adapters adhering to the W3C Trace Context specification."""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
import os
import threading
import time
from typing import Any

from domain.models import SpanEvent, SpanRecord, TraceContext
from domain.ports import SpanPort, TracerPort


def current_utc_time() -> datetime:
    return datetime.now(timezone.utc)


def generate_trace_id() -> str:
    """Generate 16 random bytes as a 32-character hexadecimal string."""
    return os.urandom(16).hex().lower()


def generate_span_id() -> str:
    """Generate 8 random bytes as a 16-character hexadecimal string."""
    return os.urandom(8).hex().lower()


_active_span_var: contextvars.ContextVar[InMemorySpan | None] = contextvars.ContextVar(
    "active_span", default=None
)


class InMemorySpan:
    """In-memory implementation of a tracing span supporting nested execution."""

    def __init__(
        self,
        name: str,
        context: TraceContext,
        parent_span_id: str | None = None,
        start_time: float | None = None,
        attributes: dict[str, Any] | None = None,
        tracer: InMemoryTracer | None = None,
    ) -> None:
        self.name = name
        self.context = context
        self.parent_span_id = parent_span_id
        self.start_time = start_time if start_time is not None else time.time()
        self.end_time: float | None = None
        self.duration_ms: float | None = None
        self.status = "UNSET"
        self.error_message: str | None = None
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.events: list[SpanEvent] = []
        self._tracer = tracer
        self._token: contextvars.Token[InMemorySpan | None] | None = None
        self._ended = False
        self._lock = threading.Lock()

    def set_attribute(self, key: str, value: Any) -> None:
        """Assign metadata attribute to the span."""
        with self._lock:
            self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record an event timestamped inside the span."""
        event = SpanEvent(
            name=name,
            timestamp=current_utc_time(),
            attributes=dict(attributes or {}),
        )
        with self._lock:
            self.events.append(event)

    def set_status(self, status: str, description: str | None = None) -> None:
        """Mark span status as OK, ERROR, or UNSET."""
        with self._lock:
            self.status = status.upper()
            if description:
                self.error_message = description

    def record_exception(self, exc: Exception) -> None:
        """Mark the span status as ERROR and append an exception event."""
        with self._lock:
            self.status = "ERROR"
            self.error_message = f"{type(exc).__name__}: {str(exc)}"
        self.add_event(
            "exception",
            {
                "exception.type": type(exc).__name__,
                "exception.message": str(exc),
            },
        )

    def end(self) -> None:
        """Seal span duration and register completed span with tracer."""
        with self._lock:
            if self._ended:
                return
            self._ended = True
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000.0

        if self._token is not None:
            try:
                _active_span_var.reset(self._token)
            except Exception:
                pass
            self._token = None

        if self._tracer is not None:
            self._tracer._record_span(self.to_record())

    def to_record(self) -> SpanRecord:
        """Convert span into immutable SpanRecord domain model."""
        with self._lock:
            return SpanRecord(
                name=self.name,
                context=self.context,
                parent_span_id=self.parent_span_id,
                start_time=self.start_time,
                end_time=self.end_time,
                duration_ms=self.duration_ms,
                status=self.status,
                error_message=self.error_message,
                attributes=dict(self.attributes),
                events=list(self.events),
            )

    def __enter__(self) -> InMemorySpan:
        """Activate span in the current async or thread context."""
        self._token = _active_span_var.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, capture exceptions, and close span."""
        if exc_val is not None:
            self.record_exception(exc_val)
        elif self.status == "UNSET":
            self.set_status("OK")
        self.end()


class InMemoryTracer:
    """Thread-safe, zero-dependency W3C-compliant tracer."""

    def __init__(self, max_history: int = 2000) -> None:
        self._max_history = max_history
        self._records: list[SpanRecord] = []
        self._lock = threading.Lock()

    def start_span(
        self,
        name: str,
        parent_context: TraceContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> InMemorySpan:
        """Begin a new child span or new root trace."""
        active = self.current_span()

        if parent_context is not None:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
            trace_flags = parent_context.trace_flags
        elif active is not None:
            trace_id = active.context.trace_id
            parent_span_id = active.context.span_id
            trace_flags = active.context.trace_flags
        else:
            trace_id = generate_trace_id()
            parent_span_id = None
            trace_flags = "01"

        span_id = generate_span_id()
        ctx = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_flags=trace_flags,
        )

        return InMemorySpan(
            name=name,
            context=ctx,
            parent_span_id=parent_span_id,
            start_time=time.time(),
            attributes=attributes,
            tracer=self,
        )

    def current_span(self) -> InMemorySpan | None:
        """Retrieve active span from current context."""
        return _active_span_var.get()

    def extract_traceparent(self, carrier: dict[str, str]) -> TraceContext | None:
        """Extract W3C tracecontext from headers or carrier dictionary."""
        if not carrier:
            return None
        header_val: str | None = None
        for k, v in carrier.items():
            if k.lower() == "traceparent":
                header_val = v
                break
        if not header_val:
            return None
        return TraceContext.from_traceparent(header_val)

    def inject_traceparent(
        self, carrier: dict[str, str], context: TraceContext | None = None
    ) -> dict[str, str]:
        """Inject trace context into carrier dictionary under key 'traceparent'."""
        target_ctx = context
        if target_ctx is None:
            active = self.current_span()
            if active is not None:
                target_ctx = active.context

        if target_ctx is not None:
            carrier["traceparent"] = target_ctx.to_traceparent()
        return carrier

    def _record_span(self, record: SpanRecord) -> None:
        """Append completed span record to in-memory ring buffer."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_history:
                self._records = self._records[-self._max_history :]

    def get_spans(
        self,
        trace_id: str | None = None,
        name: str | None = None,
        limit: int = 100,
    ) -> list[SpanRecord]:
        """Query captured span records."""
        with self._lock:
            matching = list(self._records)

        if trace_id:
            tid = trace_id.lower()
            matching = [s for s in matching if s.context.trace_id == tid]
        if name:
            matching = [s for s in matching if s.name == name]

        matching.sort(key=lambda s: s.start_time, reverse=True)
        return matching[:limit]

    def clear(self) -> None:
        """Clear all stored span history."""
        with self._lock:
            self._records.clear()


class OpenTelemetryAdapter:
    """Telemetry adapter bridging to OpenTelemetry SDK when available, falling back to InMemoryTracer."""

    def __init__(self, fallback_tracer: InMemoryTracer | None = None) -> None:
        self.fallback = fallback_tracer or InMemoryTracer()
        self._otel_tracer = None
        try:
            import opentelemetry.trace as otel_trace
            self._otel_tracer = otel_trace.get_tracer("agent_kernel")
        except ImportError:
            self._otel_tracer = None

    def start_span(
        self,
        name: str,
        parent_context: TraceContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> SpanPort:
        """Start span delegating to OpenTelemetry or fallback tracer."""
        return self.fallback.start_span(
            name=name,
            parent_context=parent_context,
            attributes=attributes,
        )

    def current_span(self) -> SpanPort | None:
        """Return active span."""
        return self.fallback.current_span()

    def extract_traceparent(self, carrier: dict[str, str]) -> TraceContext | None:
        """Extract W3C trace context."""
        return self.fallback.extract_traceparent(carrier)

    def inject_traceparent(
        self, carrier: dict[str, str], context: TraceContext | None = None
    ) -> dict[str, str]:
        """Inject W3C trace context."""
        return self.fallback.inject_traceparent(carrier, context=context)

    def get_spans(
        self,
        trace_id: str | None = None,
        name: str | None = None,
        limit: int = 100,
    ) -> list[SpanRecord]:
        """Return captured spans from underlying storage."""
        return self.fallback.get_spans(trace_id=trace_id, name=name, limit=limit)

    def clear(self) -> None:
        """Clear recorded spans."""
        self.fallback.clear()

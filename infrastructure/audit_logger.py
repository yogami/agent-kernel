"""Enterprise SIEM Audit Logger with Common Event Format (CEF) support.

Buffers, filters, and formats compliance and security events for streaming
to enterprise SIEM solutions such as Splunk, Datadog, or IBM QRadar.
"""

from __future__ import annotations

import collections
import threading
from typing import Any

from domain.models import AuditEvent, AuditEventSeverity


class CEFAuditLogger:
    """Thread-safe in-memory audit log store with CEF export capabilities."""

    def __init__(self, max_capacity: int = 2000) -> None:
        self._max_capacity = max_capacity
        self._lock = threading.Lock()
        self._events: collections.deque[AuditEvent] = collections.deque(maxlen=max_capacity)

    def log_event(self, event: AuditEvent) -> None:
        """Append an audit event to the log stream."""
        with self._lock:
            self._events.append(event)

    def get_events(
        self,
        tenant_id: str | None = None,
        severity: AuditEventSeverity | str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query collected audit events with optional filtering."""
        with self._lock:
            snapshot = list(self._events)

        if isinstance(severity, str):
            try:
                severity = AuditEventSeverity(severity.upper())
            except ValueError:
                severity = None

        filtered = []
        for ev in reversed(snapshot):
            if tenant_id and ev.tenant_id != tenant_id:
                continue
            if severity and ev.severity != severity:
                continue
            if event_type and ev.event_type != event_type:
                continue
            filtered.append(ev)
            if len(filtered) >= limit:
                break

        return filtered

    def export_cef(
        self,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """Export collected audit events in standard ArcSight Common Event Format (CEF)."""
        events = self.get_events(tenant_id=tenant_id, limit=limit)
        return [ev.to_cef() for ev in events]

    def log_memory_quarantine(
        self,
        tenant_id: str,
        subject: str,
        predicate: str,
        obj: str,
        reason: str,
        trace_id: str | None = None,
    ) -> AuditEvent:
        """Record fact quarantine event caused by contradiction or low confidence."""
        event = AuditEvent(
            tenant_id=tenant_id,
            trace_id=trace_id,
            event_type="memory.contradiction_quarantined",
            severity=AuditEventSeverity.WARNING,
            actor="memory_promoter",
            resource=f"{subject}:{predicate}",
            action="admit_fact",
            outcome="QUARANTINE",
            details={
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "reason": reason,
            },
        )
        self.log_event(event)
        return event

    def log_tool_violation(
        self,
        tenant_id: str,
        tool_name: str,
        violation_type: str,
        detail: str,
        trace_id: str | None = None,
    ) -> AuditEvent:
        """Record tool confinement violation (command injection, path escape, egress)."""
        event = AuditEvent(
            tenant_id=tenant_id,
            trace_id=trace_id,
            event_type="tool.security_violation",
            severity=AuditEventSeverity.CRITICAL,
            actor="tool_sandbox",
            resource=tool_name,
            action="execute_tool",
            outcome="DENY",
            details={
                "tool_name": tool_name,
                "violation_type": violation_type,
                "detail": detail,
            },
        )
        self.log_event(event)
        return event

    def log_causal_denial(
        self,
        tenant_id: str,
        scm_id: str,
        intervention: str,
        violations: list[str],
        trace_id: str | None = None,
    ) -> AuditEvent:
        """Record pre-flight intervention denial caused by falsified SCM dependencies."""
        event = AuditEvent(
            tenant_id=tenant_id,
            trace_id=trace_id,
            event_type="causal.falsification_blocked",
            severity=AuditEventSeverity.CRITICAL,
            actor="causal_pre_flight_gate",
            resource=scm_id,
            action="verify_intervention",
            outcome="DENY",
            details={
                "scm_id": scm_id,
                "intervention": intervention,
                "violation_count": str(len(violations)),
                "violations": "; ".join(violations),
            },
        )
        self.log_event(event)
        return event


# Global singleton logger instance for application
audit_logger = CEFAuditLogger()

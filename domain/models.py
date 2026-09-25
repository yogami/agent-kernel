"""Pure domain entities and value objects for the Agent Kernel."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid

from dataclasses import dataclass, field
Field = field


def generate_uuid() -> str:
    """Generate a string UUID v4."""
    return str(uuid.uuid4())


def current_utc_time() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class AdmissionStatus(str, Enum):
    """Lifecycle status for facts undergoing admission control."""
    PENDING = "PENDING"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class RejectionReason(str, Enum):
    """Reason code when a candidate fact is denied admission."""
    SCHEMA_INVALID = "SCHEMA_INVALID"
    CONTRADICTION_DETECTED = "CONTRADICTION_DETECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    TEMPORALLY_SUPERSEDED = "TEMPORALLY_SUPERSEDED"
    CAUSAL_INCONSISTENCY = "CAUSAL_INCONSISTENCY"
    UNSPECIFIED = "UNSPECIFIED"


class StreamEventType(str, Enum):
    """Event types emitted during streaming agent execution."""
    FSM_STATE = "fsm_state"
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TURN_COMPLETED = "turn_completed"
    ERROR = "error"


class NLILabel(str, Enum):
    """Classification labels for Natural Language Inference pairs."""
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass(kw_only=True)
class NLIResult:
    """Output from an NLI pair classification evaluation."""
    label: NLILabel
    contradiction_score: float = 0.0
    entailment_score: float = 0.0
    neutral_score: float = 0.0
    detail: str = ""


@dataclass(kw_only=True)
class StreamEvent:
    """Standardized event envelope emitted during async streaming execution."""
    event: StreamEventType
    session_id: str
    timestamp: datetime = Field(default_factory=current_utc_time)
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None



@dataclass(kw_only=True)
class ToolCall:
    """Represents an invocation request for a typed tool."""
    tool_id: str = Field(default_factory=generate_uuid)
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


@dataclass(kw_only=True)
class ToolResult:
    """Result emitted after executing a typed tool."""
    tool_id: str
    tool_name: str
    output: Any
    is_error: bool = False
    error_message: str | None = None


@dataclass(kw_only=True)
class Turn:
    """Single interaction turn between user and agent."""
    turn_id: str = Field(default_factory=generate_uuid)
    tenant_id: str = "default_tenant"
    session_id: str
    turn_index: int = 0
    user_input: str
    model_output: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=current_utc_time)


@dataclass(kw_only=True)
class Episode:
    """Immutable collection of interaction turns in a session."""
    episode_id: str = Field(default_factory=generate_uuid)
    tenant_id: str = "default_tenant"
    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=current_utc_time)


@dataclass(kw_only=True)
class CandidateFact:
    """Raw extracted fact held in quarantine pending verification."""
    candidate_id: str = Field(default_factory=generate_uuid)
    tenant_id: str = "default_tenant"
    source_episode_id: str
    session_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    extractor_model: str = "default_extractor"
    status: AdmissionStatus = AdmissionStatus.PENDING
    rejection_reason: RejectionReason | None = None
    rejection_detail: str | None = None
    created_at: datetime = Field(default_factory=current_utc_time)


@dataclass(kw_only=True)
class SemanticFact:
    """Promoted, durable fact with provenance and temporal bounds."""
    fact_id: str = Field(default_factory=generate_uuid)
    tenant_id: str = "default_tenant"
    candidate_id: str
    source_episode_id: str
    session_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    valid_from: datetime = Field(default_factory=current_utc_time)
    valid_until: datetime | None = None
    is_active: bool = True
    retired_at: datetime | None = None
    promoted_at: datetime = Field(default_factory=current_utc_time)
    provenance: dict[str, Any] = Field(default_factory=dict)


@dataclass(kw_only=True)
class ConfigPack:
    """Versioned snapshot of system instructions and tool schemas."""
    version: str
    system_prompt: str
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    sha256_hash: str = ""
    created_at: datetime = Field(default_factory=current_utc_time)


@dataclass(kw_only=True)
class ExecutionContext:
    """Runtime state passed through the Finite State Machine."""
    session_id: str = Field(default_factory=generate_uuid)
    tenant_id: str = "default_tenant"
    step_index: int = 0
    max_steps: int = 5
    cost_usd: float = 0.0
    cost_budget_usd: float = 0.50
    executed_tool_signatures: list[str] = Field(default_factory=list)
    accumulated_tokens: int = 0


@dataclass(kw_only=True)
class QuarantineRetentionPolicy:
    """Retention parameters for purging historical candidate records from quarantine."""
    max_age_days: int = 30
    batch_size: int = 100
    statuses_to_purge: list[AdmissionStatus] = Field(
        default_factory=lambda: [AdmissionStatus.REJECTED]
    )


class ExploitCategory(str, Enum):
    """Adversarial and abnormal execution patterns evaluated in Track T."""
    SOCKET_EGRESS = "socket_egress"
    CREDENTIAL_EXTRACTION = "credential_extraction"
    INFINITE_CYCLE = "infinite_cycle"
    FABRICATED_COMPLETION = "fabricated_completion"
    DEADLINE_OVERRUN = "deadline_overrun"
    PATH_TRAVERSAL = "path_traversal"
    PROCESS_SPAWN_EXPLOSION = "process_spawn_explosion"
    COMMAND_INJECTION = "command_injection"


@dataclass(kw_only=True)
class ToolCapability:
    """Declared capabilities and execution limits for an agent tool."""
    requires_network: bool = False
    requires_filesystem: bool = False
    timeout_seconds: float = 5.0
    max_memory_mb: int = 256
    allowed_paths: list[str] = Field(default_factory=list)
    allow_subprocesses: bool = False
    allow_shell: bool = False


@dataclass(kw_only=True)
class ToolSecurityViolation:
    """Audit record when a tool attempts an unauthorized action or exceeds limits."""
    tool_name: str
    violation_type: str
    detail: str
    occurred_at: datetime = Field(default_factory=current_utc_time)


@dataclass(kw_only=True)
class TraceContext:
    """W3C-compliant trace context specification."""
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    trace_flags: str = "01"

    def to_traceparent(self) -> str:
        """Format as standard W3C traceparent header: 00-{trace_id}-{span_id}-{trace_flags}."""
        return f"00-{self.trace_id.lower()}-{self.span_id.lower()}-{self.trace_flags.lower()}"

    @classmethod
    def from_traceparent(cls, header: str) -> TraceContext | None:
        """Parse standard W3C traceparent header: version-trace_id-parent_id-trace_flags."""
        if not header:
            return None
        parts = header.strip().split("-")
        if len(parts) != 4:
            return None
        version, trace_id, parent_id, trace_flags = parts
        if len(trace_id) != 32 or len(parent_id) != 16 or len(trace_flags) != 2:
            return None
        try:
            int(trace_id, 16)
            int(parent_id, 16)
            int(trace_flags, 16)
        except ValueError:
            return None
        if trace_id == "0" * 32 or parent_id == "0" * 16:
            return None
        return cls(
            trace_id=trace_id.lower(),
            span_id=parent_id.lower(),
            trace_flags=trace_flags.lower(),
        )


@dataclass(kw_only=True)
class SpanEvent:
    """Named event marker within a span duration."""
    name: str
    timestamp: datetime = Field(default_factory=current_utc_time)
    attributes: dict[str, Any] = Field(default_factory=dict)


@dataclass(kw_only=True)
class SpanRecord:
    """Structured representation of a completed or active tracing span."""
    name: str
    context: TraceContext
    parent_span_id: str | None = None
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: str = "UNSET"
    error_message: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)


@dataclass(kw_only=True)
class TenantSecurityPolicy:
    """Security configuration, resource limits, and allowlists for a specific tenant."""
    tenant_id: str
    allowed_egress_domains: list[str] = Field(default_factory=list)
    allowed_tool_capabilities: ToolCapability = Field(default_factory=ToolCapability)
    quarantine_retention_days: int = 30
    command_injection_guard_enabled: bool = True
    path_confinement_enabled: bool = True
    causal_verification_strictness: str = "STRICT"
    rate_limit_rpm: int = 120


class AuditEventSeverity(str, Enum):
    """Severity levels for security and compliance audit events."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(kw_only=True)
class AuditEvent:
    """Compliance audit event record streamable to enterprise SIEM platforms."""
    event_id: str = Field(default_factory=generate_uuid)
    tenant_id: str
    trace_id: str | None = None
    timestamp: datetime = Field(default_factory=current_utc_time)
    event_type: str
    severity: AuditEventSeverity = AuditEventSeverity.INFO
    actor: str = "agent"
    resource: str = ""
    action: str = ""
    outcome: str = "ALLOW"
    details: dict[str, Any] = Field(default_factory=dict)

    def to_cef(self) -> str:
        """Format event as standard Common Event Format (CEF) string."""
        sev_map = {
            AuditEventSeverity.INFO: 3,
            AuditEventSeverity.WARNING: 6,
            AuditEventSeverity.CRITICAL: 9,
        }
        sev_num = sev_map.get(self.severity, 3)

        # Build extension fields
        ext_parts = [
            f"src={self.actor}",
            f"act={self.action}",
            f"res={self.resource}",
            f"outcome={self.outcome}",
            f"cs1={self.tenant_id}",
            "cs1Label=tenant_id",
        ]
        if self.trace_id:
            ext_parts.append(f"cs2={self.trace_id}")
            ext_parts.append("cs2Label=trace_id")

        if self.details:
            detail_msg = "; ".join(f"{k}={v}" for k, v in self.details.items())
            # Escape pipes and equals in msg
            safe_msg = detail_msg.replace("|", "_").replace("=", ":")
            ext_parts.append(f"msg={safe_msg}")

        extension_str = " ".join(ext_parts)
        return (
            f"CEF:0|AgentKernel|SecurityGateway|1.0|"
            f"{self.event_type}|{self.event_type}|{sev_num}|{extension_str}"
        )





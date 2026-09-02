"""Pure domain entities and value objects for the Agent Kernel."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid

from pydantic import BaseModel, Field


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
    UNSPECIFIED = "UNSPECIFIED"


class ToolCall(BaseModel):
    """Represents an invocation request for a typed tool."""
    tool_id: str = Field(default_factory=generate_uuid)
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result emitted after executing a typed tool."""
    tool_id: str
    tool_name: str
    output: Any
    is_error: bool = False
    error_message: str | None = None


class Turn(BaseModel):
    """Single interaction turn between user and agent."""
    turn_id: str = Field(default_factory=generate_uuid)
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


class Episode(BaseModel):
    """Immutable collection of interaction turns in a session."""
    episode_id: str = Field(default_factory=generate_uuid)
    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=current_utc_time)


class CandidateFact(BaseModel):
    """Raw extracted fact held in quarantine pending verification."""
    candidate_id: str = Field(default_factory=generate_uuid)
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


class SemanticFact(BaseModel):
    """Promoted, durable fact with provenance and temporal bounds."""
    fact_id: str = Field(default_factory=generate_uuid)
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


class ConfigPack(BaseModel):
    """Versioned snapshot of system instructions and tool schemas."""
    version: str
    system_prompt: str
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    sha256_hash: str = ""
    created_at: datetime = Field(default_factory=current_utc_time)


class ExecutionContext(BaseModel):
    """Runtime state passed through the Finite State Machine."""
    session_id: str = Field(default_factory=generate_uuid)
    step_index: int = 0
    max_steps: int = 5
    cost_usd: float = 0.0
    cost_budget_usd: float = 0.50
    executed_tool_signatures: list[str] = Field(default_factory=list)
    accumulated_tokens: int = 0

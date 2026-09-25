"""Pure interface protocols (Hexagonal Ports) for the Agent Kernel."""

from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable
from domain.models import (
    AdmissionStatus,
    CandidateFact,
    ConfigPack,
    Episode,
    NLILabel,
    NLIResult,
    RejectionReason,
    SemanticFact,
    SpanRecord,
    ToolCapability,
    ToolResult,
    TraceContext,
    Turn,
)


@runtime_checkable
class EpisodeStorePort(Protocol):
    """Port for append-only storage of episodes and turns."""

    def append_turn(self, turn: Turn) -> None:
        """Persist a single interaction turn immutably."""
        ...

    def get_recent_turns(self, session_id: str, limit: int = 5, tenant_id: str = "default_tenant") -> list[Turn]:
        """Fetch the most recent turns for context assembly."""
        ...

    def get_episode(self, episode_id: str, tenant_id: str = "default_tenant") -> Episode | None:
        """Fetch a full episode by its identifier."""
        ...

    def list_episodes(self, limit: int = 50, tenant_id: str = "default_tenant") -> list[Episode]:
        """List historical episodes."""
        ...


@runtime_checkable
class FactStorePort(Protocol):
    """Port for tri-state memory tables: quarantine and semantic facts."""

    def insert_candidate(self, candidate: CandidateFact) -> None:
        """Write a raw candidate fact strictly into quarantine."""
        ...

    def update_candidate_status(
        self,
        candidate_id: str,
        status: AdmissionStatus,
        reason: RejectionReason | None = None,
        detail: str | None = None,
    ) -> None:
        """Update admission status of a quarantined candidate."""
        ...

    def get_pending_candidates(self, limit: int = 50, tenant_id: str = "default_tenant") -> list[CandidateFact]:
        """Fetch unreviewed candidate facts from quarantine."""
        ...

    def get_quarantine_records(self, limit: int = 100, tenant_id: str = "default_tenant") -> list[CandidateFact]:
        """List all quarantined records with statuses and reasons."""
        ...

    def get_candidate(self, candidate_id: str) -> CandidateFact | None:
        """Fetch a specific quarantined candidate by unique candidate ID."""
        ...

    def insert_semantic_fact(self, fact: SemanticFact) -> None:
        """Commit a verified fact to durable long-term storage."""
        ...

    def get_active_facts_for_subject(self, subject: str, tenant_id: str = "default_tenant") -> list[SemanticFact]:
        """Fetch currently valid semantic facts matching a subject key."""
        ...

    def retire_fact(self, fact_id: str, reason: str = "", tenant_id: str = "default_tenant") -> None:
        """Retire or supersede an existing semantic fact."""
        ...

    def query_all_semantic_facts(self, session_id: str | None = None, tenant_id: str = "default_tenant") -> list[SemanticFact]:
        """Retrieve all active semantic facts, optionally scoped by session."""
        ...

    def purge_quarantine_records(
        self,
        older_than: datetime,
        statuses: list[AdmissionStatus] | None = None,
        tenant_id: str = "default_tenant",
    ) -> int:
        """Purge historical candidate records from quarantine matching retention criteria."""
        ...


@runtime_checkable
class VectorIndexPort(Protocol):
    """Port for semantic neighborhood search with metadata filtering."""

    def upsert(self, item_id: str, vector: list[float], metadata: dict[str, Any], tenant_id: str = "default_tenant") -> None:
        """Index a vector representation alongside metadata."""
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        tenant_id: str = "default_tenant",
    ) -> list[dict[str, Any]]:
        """Find nearest vectors satisfying metadata constraints."""
        ...


@runtime_checkable
class LLMProviderPort(Protocol):
    """Port for language model completion and tool call generation."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Generate a response dictionary containing content and/or tool_calls."""
        ...

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Generate a response dictionary asynchronously."""
        ...

    def stream_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream response chunks asynchronously (deltas for content or tool_calls)."""
        ...

    def get_embedding(self, text: str) -> list[float]:
        """Compute an embedding vector for a given text string."""
        ...


@runtime_checkable
class ToolPort(Protocol):
    """Port contract for executable agent tools."""

    @property
    def name(self) -> str:
        """Unique tool name."""
        ...

    @property
    def description(self) -> str:
        """Human-readable tool description."""
        ...

    @property
    def schema(self) -> dict[str, Any]:
        """JSON Schema / Pydantic schema for tool arguments."""
        ...

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool deterministically and return a typed ToolResult."""
        ...


@runtime_checkable
class EmbeddingProviderPort(Protocol):
    """Port for computing dense vector embeddings."""

    def embed_text(self, text: str) -> list[float]:
        """Compute an embedding vector synchronously."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of strings synchronously."""
        ...

    async def embed_text_async(self, text: str) -> list[float]:
        """Compute an embedding vector asynchronously."""
        ...

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of strings asynchronously."""
        ...


@runtime_checkable
class NLIProviderPort(Protocol):
    """Port for Natural Language Inference evaluation (entailment vs. contradiction)."""

    def classify_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify logical relationship between premise and hypothesis synchronously."""
        ...

    async def classify_pair_async(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify logical relationship between premise and hypothesis asynchronously."""
        ...


@runtime_checkable
class ToolSandboxPort(Protocol):
    """Port for executing tools within isolated security boundaries."""

    def run(
        self,
        tool: ToolPort,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Execute tool within sandbox synchronously."""
        ...

    async def run_async(
        self,
        tool: ToolPort,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Execute tool within sandbox asynchronously."""
        ...


@runtime_checkable
class SpanPort(Protocol):
    """Port representing an active or completed tracing span."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a single metadata attribute on the span."""
        ...

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record a discrete event timestamped inside the span."""
        ...

    def set_status(self, status: str, description: str | None = None) -> None:
        """Set span completion status (OK, ERROR, UNSET)."""
        ...

    def record_exception(self, exc: Exception) -> None:
        """Record an exception and mark span as ERROR."""
        ...

    def end(self) -> None:
        """Complete the span and seal duration."""
        ...

    def __enter__(self) -> SpanPort:
        """Context manager entry."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        ...


@runtime_checkable
class TracerPort(Protocol):
    """Port for distributed tracing and context propagation."""

    def start_span(
        self,
        name: str,
        parent_context: TraceContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> SpanPort:
        """Start a new span, optionally as a child of parent_context or current span."""
        ...

    def current_span(self) -> SpanPort | None:
        """Return the active span in the current execution context."""
        ...

    def extract_traceparent(self, carrier: dict[str, str]) -> TraceContext | None:
        """Extract W3C tracecontext from headers or metadata map."""
        ...

    def inject_traceparent(self, carrier: dict[str, str], context: TraceContext | None = None) -> dict[str, str]:
        """Inject current or given trace context into carrier map as traceparent header."""
        ...


@runtime_checkable
class PromptOptimizerPort(Protocol):
    """Port for automated prompt instruction optimization and few-shot synthesis."""

    def optimize_prompt(
        self,
        base_pack: ConfigPack,
        failure_traces: list[dict[str, Any]],
        eval_callback: Callable[[ConfigPack], float],
    ) -> tuple[ConfigPack, dict[str, Any]]:
        """Synthesize candidate instructions and select the optimal configuration."""
        ...


@runtime_checkable
class MetricsPort(Protocol):
    """Port for operational metrics recording."""

    def record_turn(self, tokens_in: int, tokens_out: int, latency_ms: float, cost_usd: float) -> None:
        """Record usage and latency metrics for a single turn."""
        ...
    
    def record_tool_execution(self, tool_name: str, status: str, latency_ms: float) -> None:
        """Record tool execution outcomes."""
        ...

    def record_fsm_transition(self, from_state: str, to_state: str) -> None:
        """Record FSM transition."""
        ...
        
    def record_span(self, name: str, status: str) -> None:
        """Record span completion."""
        ...


@runtime_checkable
class ToolRegistryPort(Protocol):
    """Port for tool retrieval and registration."""

    def get(self, name: str) -> ToolPort | None:
        """Retrieve a tool by name."""
        ...
    
    def list_tools(self) -> list[ToolPort]:
        """List all registered tools."""
        ...
    
    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        ...

class ContextRAMPort(Protocol):
    def compose_system_prompt(
        self,
        tenant_id: str,
        session_id: str,
        recent_turns: list[dict[str, Any]] | None = None
    ) -> str:
        ...

class OutputGuardrailsPort(Protocol):
    def validate_text_output(self, text: str) -> str:
        ...

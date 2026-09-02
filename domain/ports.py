"""Pure interface protocols (Hexagonal Ports) for the Agent Kernel."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from domain.models import (
    AdmissionStatus,
    CandidateFact,
    Episode,
    RejectionReason,
    SemanticFact,
    ToolResult,
    Turn,
)


@runtime_checkable
class EpisodeStorePort(Protocol):
    """Port for append-only storage of episodes and turns."""

    def append_turn(self, turn: Turn) -> None:
        """Persist a single interaction turn immutably."""
        ...

    def get_recent_turns(self, session_id: str, limit: int = 5) -> list[Turn]:
        """Fetch the most recent turns for context assembly."""
        ...

    def get_episode(self, episode_id: str) -> Episode | None:
        """Fetch a full episode by its identifier."""
        ...

    def list_episodes(self, limit: int = 50) -> list[Episode]:
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

    def get_pending_candidates(self, limit: int = 50) -> list[CandidateFact]:
        """Fetch unreviewed candidate facts from quarantine."""
        ...

    def get_quarantine_records(self, limit: int = 100) -> list[CandidateFact]:
        """List all quarantined records with statuses and reasons."""
        ...

    def insert_semantic_fact(self, fact: SemanticFact) -> None:
        """Commit a verified fact to durable long-term storage."""
        ...

    def get_active_facts_for_subject(self, subject: str) -> list[SemanticFact]:
        """Fetch currently valid semantic facts matching a subject key."""
        ...

    def retire_fact(self, fact_id: str, reason: str = "") -> None:
        """Retire or supersede an existing semantic fact."""
        ...

    def query_all_semantic_facts(self, session_id: str | None = None) -> list[SemanticFact]:
        """Retrieve all active semantic facts, optionally scoped by session."""
        ...


@runtime_checkable
class VectorIndexPort(Protocol):
    """Port for semantic neighborhood search with metadata filtering."""

    def upsert(self, item_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Index a vector representation alongside metadata."""
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
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

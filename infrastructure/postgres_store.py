"""PostgreSQL adapter for multi-tenant episode and fact storage with pgvector."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from domain.serialization import to_dict
from typing import Any

from domain.models import (
    AdmissionStatus,
    CandidateFact,
    Episode,
    RejectionReason,
    SemanticFact,
    ToolCall,
    ToolResult,
    Turn,
)
from domain.ports import EpisodeStorePort, FactStorePort


class PostgresEpisodeStore(EpisodeStorePort):
    """PostgreSQL implementation of EpisodeStorePort with tenant isolation."""

    def __init__(self, connection_url: str | None = None, pool: Any | None = None) -> None:
        self.connection_url = connection_url
        self.pool = pool

    def _get_connection(self) -> Any:
        """Obtain a database connection from pool or driver."""
        if self.pool is not None:
            return self.pool.getconn()
        try:
            import psycopg
            return psycopg.connect(self.connection_url)
        except ImportError:
            raise RuntimeError(
                "PostgreSQL driver 'psycopg' is not installed. Install via 'pip install psycopg[binary]'."
            )

    def append_turn(self, turn: Turn) -> None:
        """Persist a single interaction turn immutably within a tenant scope."""
        conn = self._get_connection()
        tool_calls_json = json.dumps([to_dict(tc) for tc in turn.tool_calls])
        tool_results_json = json.dumps([to_dict(tr) for tr in turn.tool_results])

        query = """
            INSERT INTO turns (
                turn_id, tenant_id, session_id, turn_index, user_input,
                model_output, tool_calls_json, tool_results_json,
                tokens_in, tokens_out, cost_usd, latency_ms, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        params = (
            turn.turn_id,
            turn.tenant_id,
            turn.session_id,
            turn.turn_index,
            turn.user_input,
            turn.model_output,
            tool_calls_json,
            tool_results_json,
            turn.tokens_in,
            turn.tokens_out,
            turn.cost_usd,
            turn.latency_ms,
            turn.created_at.isoformat(),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 5,
        tenant_id: str = "default_tenant",
    ) -> list[Turn]:
        """Fetch recent turns strictly isolated to the specified tenant."""
        conn = self._get_connection()
        query = """
            SELECT
                turn_id, tenant_id, session_id, turn_index, user_input,
                model_output, tool_calls_json, tool_results_json,
                tokens_in, tokens_out, cost_usd, latency_ms, created_at
            FROM turns
            WHERE tenant_id = %s AND session_id = %s
            ORDER BY turn_index DESC
            LIMIT %s;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query, (tenant_id, session_id, limit))
                rows = cur.fetchall()
            turns: list[Turn] = []
            for row in reversed(rows):
                if isinstance(row, dict):
                    r = row
                else:
                    cols = [
                        "turn_id", "tenant_id", "session_id", "turn_index", "user_input",
                        "model_output", "tool_calls_json", "tool_results_json",
                        "tokens_in", "tokens_out", "cost_usd", "latency_ms", "created_at",
                    ]
                    r = dict(zip(cols, row))

                raw_calls = json.loads(r["tool_calls_json"]) if isinstance(r["tool_calls_json"], str) else r["tool_calls_json"]
                raw_results = json.loads(r["tool_results_json"]) if isinstance(r["tool_results_json"], str) else r["tool_results_json"]

                turns.append(
                    Turn(
                        turn_id=str(r["turn_id"]),
                        tenant_id=r["tenant_id"],
                        session_id=r["session_id"],
                        turn_index=r["turn_index"],
                        user_input=r["user_input"],
                        model_output=r["model_output"],
                        tool_calls=[ToolCall(**c) for c in raw_calls],
                        tool_results=[ToolResult(**res) for res in raw_results],
                        tokens_in=r["tokens_in"],
                        tokens_out=r["tokens_out"],
                        cost_usd=float(r["cost_usd"]),
                        latency_ms=float(r["latency_ms"]),
                        created_at=datetime.fromisoformat(str(r["created_at"])),
                    )
                )
            return turns
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def get_episode(self, episode_id: str, tenant_id: str = "default_tenant") -> Episode | None:
        """Fetch an episode entity scoped by tenant."""
        conn = self._get_connection()
        query = """
            SELECT episode_id, tenant_id, session_id, total_cost_usd, total_tokens, created_at
            FROM episodes
            WHERE tenant_id = %s AND episode_id = %s;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query, (tenant_id, episode_id))
                row = cur.fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                r = row
            else:
                cols = ["episode_id", "tenant_id", "session_id", "total_cost_usd", "total_tokens", "created_at"]
                r = dict(zip(cols, row))

            recent_turns = self.get_recent_turns(r["session_id"], limit=100, tenant_id=tenant_id)
            return Episode(
                episode_id=str(r["episode_id"]),
                tenant_id=r["tenant_id"],
                session_id=r["session_id"],
                turns=recent_turns,
                total_cost_usd=float(r["total_cost_usd"]),
                total_tokens=r["total_tokens"],
                created_at=datetime.fromisoformat(str(r["created_at"])),
            )
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def list_episodes(self, limit: int = 50, tenant_id: str = "default_tenant") -> list[Episode]:
        """List episodes belonging strictly to the specified tenant."""
        conn = self._get_connection()
        query = """
            SELECT episode_id, tenant_id, session_id, total_cost_usd, total_tokens, created_at
            FROM episodes
            WHERE tenant_id = %s
            ORDER BY created_at DESC
            LIMIT %s;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query, (tenant_id, limit))
                rows = cur.fetchall()
            episodes: list[Episode] = []
            for row in rows:
                if isinstance(row, dict):
                    r = row
                else:
                    cols = ["episode_id", "tenant_id", "session_id", "total_cost_usd", "total_tokens", "created_at"]
                    r = dict(zip(cols, row))
                episodes.append(
                    Episode(
                        episode_id=str(r["episode_id"]),
                        tenant_id=r["tenant_id"],
                        session_id=r["session_id"],
                        turns=[],
                        total_cost_usd=float(r["total_cost_usd"]),
                        total_tokens=r["total_tokens"],
                        created_at=datetime.fromisoformat(str(r["created_at"])),
                    )
                )
            return episodes
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()


class PostgresFactStore(FactStorePort):
    """PostgreSQL implementation of FactStorePort with pgvector and tenant isolation."""

    def __init__(self, connection_url: str | None = None, pool: Any | None = None) -> None:
        self.connection_url = connection_url
        self.pool = pool

    def _get_connection(self) -> Any:
        if self.pool is not None:
            return self.pool.getconn()
        try:
            import psycopg
            return psycopg.connect(self.connection_url)
        except ImportError:
            raise RuntimeError(
                "PostgreSQL driver 'psycopg' is not installed. Install via 'pip install psycopg[binary]'."
            )

    def insert_candidate(self, candidate: CandidateFact) -> None:
        """Write raw candidate fact to quarantined isolation table."""
        conn = self._get_connection()
        query = """
            INSERT INTO fact_quarantine (
                candidate_id, tenant_id, source_episode_id, session_id,
                subject, predicate, object, confidence,
                extractor_model, status, rejection_reason, rejection_detail, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        params = (
            candidate.candidate_id,
            candidate.tenant_id,
            candidate.source_episode_id,
            candidate.session_id,
            candidate.subject,
            candidate.predicate,
            candidate.object,
            candidate.confidence,
            candidate.extractor_model,
            candidate.status.value,
            candidate.rejection_reason.value if candidate.rejection_reason else None,
            candidate.rejection_detail,
            candidate.created_at.isoformat(),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def update_candidate_status(
        self,
        candidate_id: str,
        status: AdmissionStatus,
        reason: RejectionReason | None = None,
        detail: str | None = None,
    ) -> None:
        """Update admission status of quarantined candidate."""
        conn = self._get_connection()
        query = """
            UPDATE fact_quarantine
            SET status = %s, rejection_reason = %s, rejection_detail = %s
            WHERE candidate_id = %s;
        """
        params = (
            status.value,
            reason.value if reason else None,
            detail,
            candidate_id,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def get_pending_candidates(
        self,
        limit: int = 50,
        tenant_id: str = "default_tenant",
    ) -> list[CandidateFact]:
        """Fetch unreviewed candidates for the tenant."""
        return self._fetch_quarantine_records(
            status_filter=AdmissionStatus.QUARANTINED.value,
            limit=limit,
            tenant_id=tenant_id,
        )

    def get_quarantine_records(
        self,
        limit: int = 100,
        tenant_id: str = "default_tenant",
    ) -> list[CandidateFact]:
        """List quarantine history for the tenant."""
        return self._fetch_quarantine_records(status_filter=None, limit=limit, tenant_id=tenant_id)

    def _fetch_quarantine_records(
        self,
        status_filter: str | None,
        limit: int,
        tenant_id: str,
    ) -> list[CandidateFact]:
        conn = self._get_connection()
        if status_filter:
            query = """
                SELECT candidate_id, tenant_id, source_episode_id, session_id,
                       subject, predicate, object, confidence, extractor_model,
                       status, rejection_reason, rejection_detail, created_at
                FROM fact_quarantine
                WHERE tenant_id = %s AND status = %s
                ORDER BY created_at DESC
                LIMIT %s;
            """
            params: tuple[Any, ...] = (tenant_id, status_filter, limit)
        else:
            query = """
                SELECT candidate_id, tenant_id, source_episode_id, session_id,
                       subject, predicate, object, confidence, extractor_model,
                       status, rejection_reason, rejection_detail, created_at
                FROM fact_quarantine
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
            """
            params = (tenant_id, limit)

        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            records: list[CandidateFact] = []
            for row in rows:
                if isinstance(row, dict):
                    r = row
                else:
                    cols = [
                        "candidate_id", "tenant_id", "source_episode_id", "session_id",
                        "subject", "predicate", "object", "confidence", "extractor_model",
                        "status", "rejection_reason", "rejection_detail", "created_at",
                    ]
                    r = dict(zip(cols, row))
                records.append(
                    CandidateFact(
                        candidate_id=str(r["candidate_id"]),
                        tenant_id=r["tenant_id"],
                        source_episode_id=r["source_episode_id"],
                        session_id=r["session_id"],
                        subject=r["subject"],
                        predicate=r["predicate"],
                        object=r["object"],
                        confidence=float(r["confidence"]),
                        extractor_model=r["extractor_model"],
                        status=AdmissionStatus(r["status"]),
                        rejection_reason=RejectionReason(r["rejection_reason"]) if r["rejection_reason"] else None,
                        rejection_detail=r["rejection_detail"],
                        created_at=datetime.fromisoformat(str(r["created_at"])),
                    )
                )
            return records
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def get_candidate(self, candidate_id: str) -> CandidateFact | None:
        """Fetch a specific quarantined candidate by unique candidate ID."""
        conn = self._get_connection()
        query = """
            SELECT candidate_id, tenant_id, source_episode_id, session_id,
                   subject, predicate, object, confidence, extractor_model,
                   status, rejection_reason, rejection_detail, created_at
            FROM fact_quarantine
            WHERE candidate_id = %s
            LIMIT 1;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query, (candidate_id,))
                row = cur.fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                r = row
            else:
                cols = [
                    "candidate_id", "tenant_id", "source_episode_id", "session_id",
                    "subject", "predicate", "object", "confidence", "extractor_model",
                    "status", "rejection_reason", "rejection_detail", "created_at",
                ]
                r = dict(zip(cols, row))
            return CandidateFact(
                candidate_id=str(r["candidate_id"]),
                tenant_id=r["tenant_id"],
                source_episode_id=r["source_episode_id"],
                session_id=r["session_id"],
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                confidence=float(r["confidence"]),
                extractor_model=r["extractor_model"],
                status=AdmissionStatus(r["status"]),
                rejection_reason=RejectionReason(r["rejection_reason"]) if r["rejection_reason"] else None,
                rejection_detail=r["rejection_detail"],
                created_at=datetime.fromisoformat(str(r["created_at"])),
            )
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def purge_quarantine_records(
        self,
        older_than: datetime,
        statuses: list[AdmissionStatus] | None = None,
        tenant_id: str = "default_tenant",
    ) -> int:
        """Purge historical candidate records from quarantine matching retention criteria."""
        conn = self._get_connection()
        cutoff_iso = older_than.isoformat()
        target_statuses = statuses or [AdmissionStatus.REJECTED]
        placeholders = ",".join("%s" for _ in target_statuses)
        status_values = [s.value for s in target_statuses]

        query = f"""
            DELETE FROM fact_quarantine
            WHERE tenant_id = %s
              AND created_at < %s
              AND status IN ({placeholders});
        """
        params = [tenant_id, cutoff_iso] + status_values
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                deleted_count = cur.rowcount
            conn.commit()
            return max(0, deleted_count)
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def insert_semantic_fact(self, fact: SemanticFact) -> None:
        """Commit an admitted fact to long-term storage."""
        conn = self._get_connection()
        query = """
            INSERT INTO semantic_facts (
                fact_id, tenant_id, candidate_id, source_episode_id, session_id,
                subject, predicate, object, confidence,
                valid_from, valid_until, is_active, retired_at, promoted_at, provenance_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        provenance_json = json.dumps(fact.provenance)
        params = (
            fact.fact_id,
            fact.tenant_id,
            fact.candidate_id,
            fact.source_episode_id,
            fact.session_id,
            fact.subject,
            fact.predicate,
            fact.object,
            fact.confidence,
            fact.valid_from.isoformat(),
            fact.valid_until.isoformat() if fact.valid_until else None,
            fact.is_active,
            fact.retired_at.isoformat() if fact.retired_at else None,
            fact.promoted_at.isoformat(),
            provenance_json,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def get_active_facts_for_subject(
        self,
        subject: str,
        tenant_id: str = "default_tenant",
    ) -> list[SemanticFact]:
        """Fetch active facts matching subject key within tenant scope."""
        conn = self._get_connection()
        query = """
            SELECT fact_id, tenant_id, candidate_id, source_episode_id, session_id,
                   subject, predicate, object, confidence, valid_from, valid_until,
                   is_active, retired_at, promoted_at, provenance_json
            FROM semantic_facts
            WHERE tenant_id = %s AND subject = %s AND is_active = TRUE;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query, (tenant_id, subject))
                rows = cur.fetchall()
            return self._parse_semantic_facts(rows)
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def query_all_semantic_facts(
        self,
        session_id: str | None = None,
        tenant_id: str = "default_tenant",
    ) -> list[SemanticFact]:
        """Query active semantic facts scoped by tenant."""
        conn = self._get_connection()
        if session_id:
            query = """
                SELECT fact_id, tenant_id, candidate_id, source_episode_id, session_id,
                       subject, predicate, object, confidence, valid_from, valid_until,
                       is_active, retired_at, promoted_at, provenance_json
                FROM semantic_facts
                WHERE tenant_id = %s AND session_id = %s AND is_active = TRUE
                ORDER BY promoted_at DESC;
            """
            params: tuple[Any, ...] = (tenant_id, session_id)
        else:
            query = """
                SELECT fact_id, tenant_id, candidate_id, source_episode_id, session_id,
                       subject, predicate, object, confidence, valid_from, valid_until,
                       is_active, retired_at, promoted_at, provenance_json
                FROM semantic_facts
                WHERE tenant_id = %s AND is_active = TRUE
                ORDER BY promoted_at DESC;
            """
            params = (tenant_id,)

        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            return self._parse_semantic_facts(rows)
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def retire_fact(self, fact_id: str, reason: str = "Superseded") -> None:
        """Deactivate an outdated fact."""
        conn = self._get_connection()
        query = """
            UPDATE semantic_facts
            SET is_active = FALSE,
                retired_at = %s,
                valid_until = %s
            WHERE fact_id = %s;
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (now, now, fact_id))
            conn.commit()
        finally:
            if self.pool is not None:
                self.pool.putconn(conn)
            else:
                conn.close()

    def _parse_semantic_facts(self, rows: list[Any]) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        for row in rows:
            if isinstance(row, dict):
                r = row
            else:
                cols = [
                    "fact_id", "tenant_id", "candidate_id", "source_episode_id", "session_id",
                    "subject", "predicate", "object", "confidence", "valid_from", "valid_until",
                    "is_active", "retired_at", "promoted_at", "provenance_json",
                ]
                r = dict(zip(cols, row))
            raw_prov = json.loads(r["provenance_json"]) if isinstance(r["provenance_json"], str) else r["provenance_json"]
            facts.append(
                SemanticFact(
                    fact_id=str(r["fact_id"]),
                    tenant_id=r["tenant_id"],
                    candidate_id=r["candidate_id"],
                    source_episode_id=r["source_episode_id"],
                    session_id=r["session_id"],
                    subject=r["subject"],
                    predicate=r["predicate"],
                    object=r["object"],
                    confidence=float(r["confidence"]),
                    valid_from=datetime.fromisoformat(str(r["valid_from"])),
                    valid_until=datetime.fromisoformat(str(r["valid_until"])) if r["valid_until"] else None,
                    is_active=bool(r["is_active"]),
                    retired_at=datetime.fromisoformat(str(r["retired_at"])) if r["retired_at"] else None,
                    promoted_at=datetime.fromisoformat(str(r["promoted_at"])),
                    provenance=raw_prov,
                )
            )
        return facts

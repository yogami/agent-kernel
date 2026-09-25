"""SQLite implementation of the FactStorePort for tri-state memory management."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from domain.models import (
    AdmissionStatus,
    CandidateFact,
    RejectionReason,
    SemanticFact,
)
from domain.ports import FactStorePort


class SQLiteFactStore(FactStorePort):
    """Relational SQLite store managing candidate quarantine and verified semantic facts."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if self.db_path == ":memory:":
            self._conn: sqlite3.Connection | None = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON;")
        else:
            self._conn = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        with conn:
            # Fact Quarantine Table (writes go here first)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fact_quarantine (
                    candidate_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
                    source_episode_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    extractor_model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rejection_reason TEXT,
                    rejection_detail TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            # Migration check for existing SQLite tables
            cursor = conn.execute("PRAGMA table_info(fact_quarantine);")
            cols = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in cols:
                conn.execute("ALTER TABLE fact_quarantine ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default_tenant';")

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quarantine_tenant_status ON fact_quarantine(tenant_id, status);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quarantine_tenant_session ON fact_quarantine(tenant_id, session_id);
            """)

            # Semantic Facts Table (promoted verified facts only)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    fact_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
                    candidate_id TEXT NOT NULL,
                    source_episode_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    retired_at TEXT,
                    promoted_at TEXT NOT NULL,
                    provenance_json TEXT NOT NULL
                );
            """)
            cursor = conn.execute("PRAGMA table_info(semantic_facts);")
            cols = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in cols:
                conn.execute("ALTER TABLE semantic_facts ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default_tenant';")

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_tenant_subject ON semantic_facts(tenant_id, subject, is_active);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_tenant_session ON semantic_facts(tenant_id, session_id);
            """)
        if self._conn is None:
            conn.close()

    def insert_candidate(self, candidate: CandidateFact) -> None:
        """Write a raw candidate fact strictly into quarantine."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO fact_quarantine (
                    candidate_id, tenant_id, source_episode_id, session_id, subject, predicate,
                    object, confidence, extractor_model, status, rejection_reason,
                    rejection_detail, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    candidate.tenant_id,
                    candidate.source_episode_id,
                    candidate.session_id,
                    candidate.subject.strip().lower(),
                    candidate.predicate.strip().lower(),
                    candidate.object.strip(),
                    candidate.confidence,
                    candidate.extractor_model,
                    candidate.status.value,
                    candidate.rejection_reason.value if candidate.rejection_reason else None,
                    candidate.rejection_detail,
                    candidate.created_at.isoformat(),
                ),
            )
        if self._conn is None:
            conn.close()

    def update_candidate_status(
        self,
        candidate_id: str,
        status: AdmissionStatus,
        reason: RejectionReason | None = None,
        detail: str | None = None,
    ) -> None:
        """Update admission status of a quarantined candidate."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                UPDATE fact_quarantine
                SET status = ?, rejection_reason = ?, rejection_detail = ?
                WHERE candidate_id = ?
                """,
                (
                    status.value,
                    reason.value if reason else None,
                    detail,
                    candidate_id,
                ),
            )
        if self._conn is None:
            conn.close()

    def get_pending_candidates(self, limit: int = 50, tenant_id: str = "default_tenant") -> list[CandidateFact]:
        """Fetch unreviewed candidate facts from quarantine scoped by tenant."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM fact_quarantine
            WHERE tenant_id = ? AND status = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (tenant_id, AdmissionStatus.PENDING.value, limit),
        )
        rows = cursor.fetchall()
        candidates = [self._row_to_candidate(row) for row in rows]
        if self._conn is None:
            conn.close()
        return candidates

    def get_quarantine_records(self, limit: int = 100, tenant_id: str = "default_tenant") -> list[CandidateFact]:
        """List all quarantined records with statuses and reasons scoped by tenant."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM fact_quarantine
            WHERE tenant_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (tenant_id, limit),
        )
        rows = cursor.fetchall()
        candidates = [self._row_to_candidate(row) for row in rows]
        if self._conn is None:
            conn.close()
        return candidates

    def get_candidate(self, candidate_id: str) -> CandidateFact | None:
        """Fetch a specific quarantined candidate by unique candidate ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM fact_quarantine
            WHERE candidate_id = ?
            LIMIT 1
            """,
            (candidate_id,),
        )
        row = cursor.fetchone()
        candidate = self._row_to_candidate(row) if row else None
        if self._conn is None:
            conn.close()
        return candidate

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
        placeholders = ",".join("?" for _ in target_statuses)
        status_values = [s.value for s in target_statuses]

        query = f"""
            DELETE FROM fact_quarantine
            WHERE tenant_id = ?
              AND created_at < ?
              AND status IN ({placeholders});
        """
        params = [tenant_id, cutoff_iso] + status_values
        with conn:
            cursor = conn.execute(query, tuple(params))
            deleted_count = cursor.rowcount

        if self._conn is None:
            conn.close()
        return max(0, deleted_count)

    def insert_semantic_fact(self, fact: SemanticFact) -> None:
        """Commit a verified fact to durable long-term storage."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO semantic_facts (
                    fact_id, tenant_id, candidate_id, source_episode_id, session_id, subject,
                    predicate, object, confidence, valid_from, valid_until,
                    is_active, retired_at, promoted_at, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    fact.tenant_id,
                    fact.candidate_id,
                    fact.source_episode_id,
                    fact.session_id,
                    fact.subject.strip().lower(),
                    fact.predicate.strip().lower(),
                    fact.object.strip(),
                    fact.confidence,
                    fact.valid_from.isoformat(),
                    fact.valid_until.isoformat() if fact.valid_until else None,
                    1 if fact.is_active else 0,
                    fact.retired_at.isoformat() if fact.retired_at else None,
                    fact.promoted_at.isoformat(),
                    json.dumps(fact.provenance),
                ),
            )
        if self._conn is None:
            conn.close()

    def get_active_facts_for_subject(self, subject: str, tenant_id: str = "default_tenant") -> list[SemanticFact]:
        """Fetch currently valid semantic facts matching a subject key within a tenant."""
        clean_subject = subject.strip().lower()
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM semantic_facts
            WHERE tenant_id = ? AND subject = ? AND is_active = 1
            ORDER BY promoted_at DESC
            """,
            (tenant_id, clean_subject),
        )
        rows = cursor.fetchall()
        facts = [self._row_to_semantic_fact(row) for row in rows]
        if self._conn is None:
            conn.close()
        return facts

    def retire_fact(self, fact_id: str, reason: str = "", tenant_id: str = "default_tenant") -> None:
        """Retire or supersede an existing semantic fact."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                UPDATE semantic_facts
                SET is_active = 0, retired_at = ?, valid_until = ?
                WHERE tenant_id = ? AND fact_id = ?
                """,
                (now, now, tenant_id, fact_id),
            )
        if self._conn is None:
            conn.close()

    def query_all_semantic_facts(self, session_id: str | None = None, tenant_id: str = "default_tenant") -> list[SemanticFact]:
        """Retrieve all active semantic facts scoped by tenant, optionally by session."""
        conn = self._get_connection()
        if session_id:
            cursor = conn.execute(
                """
                SELECT * FROM semantic_facts
                WHERE tenant_id = ? AND session_id = ? AND is_active = 1
                ORDER BY promoted_at DESC
                """,
                (tenant_id, session_id),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM semantic_facts
                WHERE tenant_id = ? AND is_active = 1
                ORDER BY promoted_at DESC
                """,
                (tenant_id,),
            )
        rows = cursor.fetchall()
        facts = [self._row_to_semantic_fact(row) for row in rows]
        if self._conn is None:
            conn.close()
        return facts

    def _row_to_candidate(self, row: sqlite3.Row) -> CandidateFact:
        reason_val = row["rejection_reason"]
        return CandidateFact(
            candidate_id=row["candidate_id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else "default_tenant",
            source_episode_id=row["source_episode_id"],
            session_id=row["session_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            extractor_model=row["extractor_model"],
            status=AdmissionStatus(row["status"]),
            rejection_reason=RejectionReason(reason_val) if reason_val else None,
            rejection_detail=row["rejection_detail"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_semantic_fact(self, row: sqlite3.Row) -> SemanticFact:
        return SemanticFact(
            fact_id=row["fact_id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else "default_tenant",
            candidate_id=row["candidate_id"],
            source_episode_id=row["source_episode_id"],
            session_id=row["session_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=(
                datetime.fromisoformat(row["valid_until"])
                if row["valid_until"]
                else None
            ),
            is_active=bool(row["is_active"]),
            retired_at=(
                datetime.fromisoformat(row["retired_at"])
                if row["retired_at"]
                else None
            ),
            promoted_at=datetime.fromisoformat(row["promoted_at"]),
            provenance=json.loads(row["provenance_json"]),
        )

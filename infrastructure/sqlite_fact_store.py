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
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        with conn:
            # Fact Quarantine Table (writes go here first)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fact_quarantine (
                    candidate_id TEXT PRIMARY KEY,
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quarantine_status ON fact_quarantine(status);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quarantine_session ON fact_quarantine(session_id);
            """)

            # Semantic Facts Table (promoted verified facts only)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    fact_id TEXT PRIMARY KEY,
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_facts(subject, is_active);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_session ON semantic_facts(session_id);
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
                    candidate_id, source_episode_id, session_id, subject, predicate,
                    object, confidence, extractor_model, status, rejection_reason,
                    rejection_detail, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
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

    def get_pending_candidates(self, limit: int = 50) -> list[CandidateFact]:
        """Fetch unreviewed candidate facts from quarantine."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM fact_quarantine
            WHERE status = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (AdmissionStatus.PENDING.value, limit),
        )
        rows = cursor.fetchall()
        candidates = [self._row_to_candidate(row) for row in rows]
        if self._conn is None:
            conn.close()
        return candidates

    def get_quarantine_records(self, limit: int = 100) -> list[CandidateFact]:
        """List all quarantined records with statuses and reasons."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM fact_quarantine
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        candidates = [self._row_to_candidate(row) for row in rows]
        if self._conn is None:
            conn.close()
        return candidates

    def insert_semantic_fact(self, fact: SemanticFact) -> None:
        """Commit a verified fact to durable long-term storage."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO semantic_facts (
                    fact_id, candidate_id, source_episode_id, session_id, subject,
                    predicate, object, confidence, valid_from, valid_until,
                    is_active, retired_at, promoted_at, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
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

    def get_active_facts_for_subject(self, subject: str) -> list[SemanticFact]:
        """Fetch currently valid semantic facts matching a subject key."""
        clean_subject = subject.strip().lower()
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM semantic_facts
            WHERE subject = ? AND is_active = 1
            ORDER BY promoted_at DESC
            """,
            (clean_subject,),
        )
        rows = cursor.fetchall()
        facts = [self._row_to_semantic_fact(row) for row in rows]
        if self._conn is None:
            conn.close()
        return facts

    def retire_fact(self, fact_id: str, reason: str = "") -> None:
        """Retire or supersede an existing semantic fact."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                UPDATE semantic_facts
                SET is_active = 0, retired_at = ?, valid_until = ?
                WHERE fact_id = ?
                """,
                (now, now, fact_id),
            )
        if self._conn is None:
            conn.close()

    def query_all_semantic_facts(self, session_id: str | None = None) -> list[SemanticFact]:
        """Retrieve all active semantic facts, optionally scoped by session."""
        conn = self._get_connection()
        if session_id:
            cursor = conn.execute(
                """
                SELECT * FROM semantic_facts
                WHERE session_id = ? AND is_active = 1
                ORDER BY promoted_at DESC
                """,
                (session_id,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM semantic_facts
                WHERE is_active = 1
                ORDER BY promoted_at DESC
                """
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

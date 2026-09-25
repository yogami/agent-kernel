"""SQLite implementation of the append-only EpisodeStorePort."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from domain.serialization import to_dict
import sqlite3
from typing import Any

from domain.models import Episode, ToolCall, ToolResult, Turn
from domain.ports import EpisodeStorePort


class SQLiteEpisodeStore(EpisodeStorePort):
    """Append-only SQLite store for episodes and turns."""

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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    user_input TEXT NOT NULL,
                    model_output TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL,
                    tool_results_json TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            # Migration check for existing SQLite tables
            cursor = conn.execute("PRAGMA table_info(turns);")
            cols = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in cols:
                conn.execute("ALTER TABLE turns ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default_tenant';")

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_tenant_session ON turns(tenant_id, session_id, turn_index);
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
                    session_id TEXT NOT NULL,
                    total_cost_usd REAL NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            cursor = conn.execute("PRAGMA table_info(episodes);")
            cols = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in cols:
                conn.execute("ALTER TABLE episodes ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default_tenant';")

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_tenant_session ON episodes(tenant_id, session_id);
            """)
        if self._conn is None:
            conn.close()

    def append_turn(self, turn: Turn) -> None:
        """Persist a single interaction turn immutably."""
        conn = self._get_connection()
        tool_calls_data = [to_dict(tc) for tc in turn.tool_calls]
        tool_results_data = [to_dict(tr) for tr in turn.tool_results]
        with conn:
            conn.execute(
                """
                INSERT INTO turns (
                    turn_id, tenant_id, session_id, turn_index, user_input, model_output,
                    tool_calls_json, tool_results_json, tokens_in, tokens_out,
                    cost_usd, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id,
                    turn.tenant_id,
                    turn.session_id,
                    turn.turn_index,
                    turn.user_input,
                    turn.model_output,
                    json.dumps(tool_calls_data),
                    json.dumps(tool_results_data),
                    turn.tokens_in,
                    turn.tokens_out,
                    turn.cost_usd,
                    turn.latency_ms,
                    turn.created_at.isoformat(),
                ),
            )
        if self._conn is None:
            conn.close()

    def get_recent_turns(self, session_id: str, limit: int = 5, tenant_id: str = "default_tenant") -> list[Turn]:
        """Fetch the most recent turns for context assembly scoped by tenant."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM turns
            WHERE tenant_id = ? AND session_id = ?
            ORDER BY turn_index DESC
            LIMIT ?
            """,
            (tenant_id, session_id, limit),
        )
        rows = cursor.fetchall()
        turns: list[Turn] = []
        for row in reversed(rows):
            tool_calls_raw = json.loads(row["tool_calls_json"])
            tool_results_raw = json.loads(row["tool_results_json"])
            turn = Turn(
                turn_id=row["turn_id"],
                tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else tenant_id,
                session_id=row["session_id"],
                turn_index=row["turn_index"],
                user_input=row["user_input"],
                model_output=row["model_output"],
                tool_calls=[ToolCall(**tc) for tc in tool_calls_raw],
                tool_results=[ToolResult(**tr) for tr in tool_results_raw],
                tokens_in=row["tokens_in"],
                tokens_out=row["tokens_out"],
                cost_usd=row["cost_usd"],
                latency_ms=row["latency_ms"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            turns.append(turn)
        if self._conn is None:
            conn.close()
        return turns

    def save_episode(self, episode: Episode) -> None:
        """Save high-level episode record and constituent turns within a single transaction."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, tenant_id, session_id, total_cost_usd, total_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.tenant_id,
                    episode.session_id,
                    episode.total_cost_usd,
                    episode.total_tokens,
                    episode.created_at.isoformat(),
                ),
            )
            for turn in episode.turns:
                tool_calls_data = [
                    {"tool_id": tc.tool_id, "tool_name": tc.tool_name, "arguments": tc.arguments}
                    for tc in turn.tool_calls
                ]
                tool_results_data = [
                    {
                        "tool_id": tr.tool_id,
                        "output": tr.output,
                        "error_message": tr.error_message,
                        "is_error": tr.is_error,
                    }
                    for tr in turn.tool_results
                ]
                conn.execute(
                    """
                    INSERT INTO turns (
                        turn_id, tenant_id, episode_id, session_id, turn_index, user_input, model_output,
                        tool_calls_json, tool_results_json, tokens_in, tokens_out,
                        cost_usd, latency_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.turn_id,
                        turn.tenant_id,
                        episode.episode_id,
                        turn.session_id,
                        turn.turn_index,
                        turn.user_input,
                        turn.model_output,
                        json.dumps(tool_calls_data),
                        json.dumps(tool_results_data),
                        turn.tokens_in,
                        turn.tokens_out,
                        turn.cost_usd,
                        turn.latency_ms,
                        turn.created_at.isoformat(),
                    ),
                )
        if self._conn is None:
            conn.close()

    def get_episode(self, episode_id: str, tenant_id: str = "default_tenant") -> Episode | None:
        """Fetch a full episode by its identifier and tenant."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM episodes WHERE tenant_id = ? AND episode_id = ?", (tenant_id, episode_id)
        )
        row = cursor.fetchone()
        if not row:
            if self._conn is None:
                conn.close()
            return None
        session_id = row["session_id"]
        if self._conn is None:
            conn.close()
        turns = self.get_recent_turns(session_id, limit=1000, tenant_id=tenant_id)
        return Episode(
            episode_id=row["episode_id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else tenant_id,
            session_id=session_id,
            turns=turns,
            total_cost_usd=row["total_cost_usd"],
            total_tokens=row["total_tokens"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_episodes(self, limit: int = 50, tenant_id: str = "default_tenant") -> list[Episode]:
        """List historical episodes filtered by tenant."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM episodes WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?", (tenant_id, limit)
        )
        rows = cursor.fetchall()
        episodes: list[Episode] = []
        for row in rows:
            episodes.append(
                Episode(
                    episode_id=row["episode_id"],
                    tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else tenant_id,
                    session_id=row["session_id"],
                    turns=[],
                    total_cost_usd=row["total_cost_usd"],
                    total_tokens=row["total_tokens"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        if self._conn is None:
            conn.close()
        return episodes

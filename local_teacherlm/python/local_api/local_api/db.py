from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from local_api.config import Settings, get_settings


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class SQLiteStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self._ensure_dirs()
        with self._locked_conn() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate(conn)
            conn.commit()
        self._seed_default_provider()
        self._seed_generator_registry()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        uploaded_file_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(uploaded_files)").fetchall()
        }
        if "chunk_count" not in uploaded_file_columns:
            conn.execute("ALTER TABLE uploaded_files ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 0")
        node_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(knowledge_graph_nodes)").fetchall()
        }
        for column, ddl in {
            "node_key": "ALTER TABLE knowledge_graph_nodes ADD COLUMN node_key TEXT NOT NULL DEFAULT ''",
            "description": "ALTER TABLE knowledge_graph_nodes ADD COLUMN description TEXT NOT NULL DEFAULT ''",
            "ref_id": "ALTER TABLE knowledge_graph_nodes ADD COLUMN ref_id TEXT",
            "source_chunk_ids_json": "ALTER TABLE knowledge_graph_nodes ADD COLUMN source_chunk_ids_json TEXT NOT NULL DEFAULT '[]'",
            "active": "ALTER TABLE knowledge_graph_nodes ADD COLUMN active INTEGER NOT NULL DEFAULT 1",
            "created_at": "ALTER TABLE knowledge_graph_nodes ADD COLUMN created_at TEXT",
            "updated_at": "ALTER TABLE knowledge_graph_nodes ADD COLUMN updated_at TEXT",
        }.items():
            if column not in node_columns:
                conn.execute(ddl)
        edge_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(knowledge_graph_edges)").fetchall()
        }
        for column, ddl in {
            "relation_type": "ALTER TABLE knowledge_graph_edges ADD COLUMN relation_type TEXT NOT NULL DEFAULT ''",
            "confidence": "ALTER TABLE knowledge_graph_edges ADD COLUMN confidence REAL NOT NULL DEFAULT 0.6",
            "source_chunk_ids_json": "ALTER TABLE knowledge_graph_edges ADD COLUMN source_chunk_ids_json TEXT NOT NULL DEFAULT '[]'",
            "active": "ALTER TABLE knowledge_graph_edges ADD COLUMN active INTEGER NOT NULL DEFAULT 1",
            "created_at": "ALTER TABLE knowledge_graph_edges ADD COLUMN created_at TEXT",
            "updated_at": "ALTER TABLE knowledge_graph_edges ADD COLUMN updated_at TEXT",
        }.items():
            if column not in edge_columns:
                conn.execute(ddl)
        if "relation_type" not in edge_columns and "edge_type" in edge_columns:
            conn.execute("UPDATE knowledge_graph_edges SET relation_type = edge_type WHERE relation_type = ''")
        course_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(coursebuilder_courses)").fetchall()
        }
        for column, ddl in {
            "status": "ALTER TABLE coursebuilder_courses ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'",
            "build_id": "ALTER TABLE coursebuilder_courses ADD COLUMN build_id TEXT",
            "source_fingerprint": "ALTER TABLE coursebuilder_courses ADD COLUMN source_fingerprint TEXT NOT NULL DEFAULT ''",
            "quality_mode": "ALTER TABLE coursebuilder_courses ADD COLUMN quality_mode TEXT NOT NULL DEFAULT 'fallback'",
            "error": "ALTER TABLE coursebuilder_courses ADD COLUMN error TEXT",
            "created_at": "ALTER TABLE coursebuilder_courses ADD COLUMN created_at TEXT",
        }.items():
            if column not in course_columns:
                conn.execute(ddl)

    def _ensure_dirs(self) -> None:
        data_dir = self.settings.data_dir
        paths = [
            data_dir,
            data_dir / "objects" / "uploads",
            data_dir / "objects" / "parsed",
            data_dir / "objects" / "cleaned",
            data_dir / "artifacts" / "quizzes",
            data_dir / "artifacts" / "mindmaps",
            data_dir / "artifacts" / "podcasts",
            data_dir / "artifacts" / "presentations",
            data_dir / "artifacts" / "reports",
            data_dir / "artifacts" / "charts",
            data_dir / "indexes" / "vector",
            data_dir / "indexes" / "bm25",
            data_dir / "indexes" / "graph",
            data_dir / "models" / "embeddings",
            data_dir / "models" / "rerankers",
            data_dir / "models" / "tts",
            data_dir / "logs",
            data_dir / "traces",
            data_dir / "plugins" / "generators",
            data_dir / "plugins" / "mcp",
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.settings.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    def _locked_conn(self) -> sqlite3.Connection:
        self._lock.acquire()
        return _LockedConnection(self)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._locked_conn() as conn:
            conn.execute(sql, tuple(params))
            conn.commit()

    def executemany(self, sql: str, params: Iterable[Iterable[Any]]) -> None:
        with self._locked_conn() as conn:
            conn.executemany(sql, params)
            conn.commit()

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._locked_conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def create_conversation(self, title: str = "New course") -> dict[str, Any]:
        conversation_id = new_id("conv")
        now = utc_now()
        self.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title, now, now),
        )
        self.save_learner_state(conversation_id, {"conversation_id": conversation_id})
        return self.get_conversation(conversation_id) or {"id": conversation_id, "title": title}

    def list_conversations(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM conversations ORDER BY updated_at DESC")

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM conversations WHERE id = ?", (conversation_id,))

    def update_conversation(self, conversation_id: str, *, title: str | None = None) -> dict[str, Any] | None:
        values: dict[str, Any] = {}
        if title is not None:
            values["title"] = title
        if not values:
            return self.get_conversation(conversation_id)
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        self.execute(
            f"UPDATE conversations SET {assignments} WHERE id = ?",
            (*values.values(), conversation_id),
        )
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return False

        data_dir = self.settings.data_dir.resolve()
        upload_rows = self.query(
            """
            SELECT stored_path
            FROM uploaded_files
            WHERE conversation_id = ?
              AND stored_path NOT IN (
                SELECT stored_path
                FROM uploaded_files
                WHERE conversation_id != ?
              )
            """,
            (conversation_id, conversation_id),
        )
        file_ids = [row["id"] for row in self.query("SELECT id FROM uploaded_files WHERE conversation_id = ?", (conversation_id,))]
        artifact_rows = self.query("SELECT local_key FROM artifacts WHERE conversation_id = ?", (conversation_id,))

        with self._locked_conn() as conn:
            conn.execute(
                """
                DELETE FROM generated_chunk_questions
                WHERE chunk_id IN (SELECT id FROM search_chunks WHERE conversation_id = ?)
                """,
                (conversation_id,),
            )
            conn.execute("DELETE FROM search_chunks_fts WHERE conversation_id = ?", (conversation_id,))
            for table in (
                "course_documents",
                "course_sections",
                "search_chunks",
                "formulas",
                "tables",
                "timeline_events",
                "concept_inventory",
                "learning_phases",
                "learning_objectives",
                "knowledge_graph_nodes",
                "knowledge_graph_edges",
                "learner_state",
                "review_windows",
                "knowledge_checks",
                "coursebuilder_plans",
                "coursebuilder_quiz_attempts",
                "coursebuilder_progress",
                "coursebuilder_courses",
                "artifacts",
                "generator_run_traces",
                "hyde_traces",
                "messages",
                "uploaded_files",
            ):
                conn.execute(f"DELETE FROM {table} WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()

        cleanup_paths: list[Path] = []
        cleanup_paths.extend(Path(row["stored_path"]) for row in upload_rows)
        for file_id in file_ids:
            cleanup_paths.append(data_dir / "objects" / "parsed" / f"{file_id}.md")
            cleanup_paths.append(data_dir / "objects" / "cleaned" / f"{file_id}.txt")
        cleanup_paths.extend(data_dir / row["local_key"] for row in artifact_rows)
        for path in cleanup_paths:
            try:
                resolved = path.resolve()
                if resolved.is_file() and resolved.is_relative_to(data_dir):
                    resolved.unlink()
            except OSError:
                continue
        return True

    def touch_conversation(self, conversation_id: str) -> None:
        self.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (utc_now(), conversation_id))

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        output_type: str = "text",
        artifacts: list[dict[str, Any]] | None = None,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = new_id("msg")
        now = utc_now()
        self.execute(
            """
            INSERT INTO messages
              (id, conversation_id, role, content, output_type, artifacts_json, sources_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                output_type,
                json.dumps(artifacts or []),
                json.dumps(sources or []),
                json.dumps(metadata or {}),
                now,
            ),
        )
        self.touch_conversation(conversation_id)
        return self.one("SELECT * FROM messages WHERE id = ?", (message_id,)) or {}

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        for row in rows:
            row["artifacts"] = _json(row.pop("artifacts_json"), [])
            row["sources"] = _json(row.pop("sources_json"), [])
            row["metadata"] = _json(row.pop("metadata_json"), {})
        return rows

    def load_learner_state(self, conversation_id: str) -> dict[str, Any]:
        row = self.one("SELECT state_json FROM learner_state WHERE conversation_id = ?", (conversation_id,))
        if row is None:
            return {"conversation_id": conversation_id}
        state = _json(row["state_json"], {})
        state.setdefault("conversation_id", conversation_id)
        return state

    def save_learner_state(self, conversation_id: str, state: dict[str, Any]) -> None:
        payload = json.dumps({**state, "conversation_id": conversation_id})
        now = utc_now()
        self.execute(
            """
            INSERT INTO learner_state (conversation_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
              state_json = excluded.state_json,
              updated_at = excluded.updated_at
            """,
            (conversation_id, payload, now),
        )

    def create_uploaded_file(
        self,
        conversation_id: str,
        filename: str,
        stored_path: Path,
        *,
        mime_type: str = "",
        size_bytes: int = 0,
    ) -> dict[str, Any]:
        file_id = new_id("file")
        now = utc_now()
        self.execute(
            """
            INSERT INTO uploaded_files
              (id, conversation_id, filename, stored_path, mime_type, size_bytes, status, chunk_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'uploaded', 0, ?, ?)
            """,
            (file_id, conversation_id, filename, str(stored_path), mime_type, size_bytes, now, now),
        )
        return self.get_file(file_id) or {}

    def update_file(self, file_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        self.execute(
            f"UPDATE uploaded_files SET {assignments} WHERE id = ?",
            (*values.values(), file_id),
        )

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM uploaded_files WHERE id = ?", (file_id,))

    def get_file_for_conversation(self, conversation_id: str, file_id: str) -> dict[str, Any] | None:
        return self.one(
            "SELECT * FROM uploaded_files WHERE id = ? AND conversation_id = ?",
            (file_id, conversation_id),
        )

    def list_files(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM uploaded_files WHERE conversation_id = ? ORDER BY created_at DESC",
            (conversation_id,),
        )

    def clear_file_content(self, conversation_id: str, file_id: str) -> None:
        with self._locked_conn() as conn:
            conn.execute(
                "DELETE FROM generated_chunk_questions WHERE chunk_id IN (SELECT id FROM search_chunks WHERE source_file_id = ?)",
                (file_id,),
            )
            conn.execute("DELETE FROM search_chunks_fts WHERE source_file_id = ?", (file_id,))
            for table in (
                "search_chunks",
                "course_sections",
                "course_documents",
                "formulas",
                "tables",
                "timeline_events",
            ):
                conn.execute(f"DELETE FROM {table} WHERE source_file_id = ?", (file_id,))
            conn.execute("DELETE FROM concept_inventory WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM knowledge_graph_edges WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM knowledge_graph_nodes WHERE conversation_id = ?", (conversation_id,))
            conn.commit()

    def delete_uploaded_file(self, conversation_id: str, file_id: str) -> dict[str, Any] | None:
        record = self.get_file_for_conversation(conversation_id, file_id)
        if record is None:
            return None

        data_dir = self.settings.data_dir.resolve()
        cleanup_paths = [
            Path(record["stored_path"]),
            data_dir / "objects" / "parsed" / f"{file_id}.md",
            data_dir / "objects" / "cleaned" / f"{file_id}.txt",
        ]

        self.clear_file_content(conversation_id, file_id)
        self.execute("DELETE FROM uploaded_files WHERE id = ? AND conversation_id = ?", (file_id, conversation_id))

        for path in cleanup_paths:
            try:
                resolved = path.resolve()
                if resolved.is_file() and resolved.is_relative_to(data_dir):
                    resolved.unlink()
            except OSError:
                continue
        return record

    def replace_chunks_for_file(self, file_id: str, chunks: list[dict[str, Any]]) -> None:
        with self._locked_conn() as conn:
            conn.execute("DELETE FROM generated_chunk_questions WHERE chunk_id IN (SELECT id FROM search_chunks WHERE source_file_id = ?)", (file_id,))
            conn.execute("DELETE FROM search_chunks_fts WHERE source_file_id = ?", (file_id,))
            conn.execute("DELETE FROM search_chunks WHERE source_file_id = ?", (file_id,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO search_chunks
                      (id, conversation_id, source_file_id, document_id, section_id, chunk_index, text,
                       source_filename, metadata_json, embedding_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["conversation_id"],
                        chunk["source_file_id"],
                        chunk.get("document_id"),
                        chunk.get("section_id"),
                        chunk.get("chunk_index", 0),
                        chunk["text"],
                        chunk["source_filename"],
                        json.dumps(chunk.get("metadata", {})),
                        json.dumps(chunk.get("embedding")) if chunk.get("embedding") else None,
                        utc_now(),
                    ),
                )
                metadata = chunk.get("metadata", {})
                conn.execute(
                    """
                    INSERT INTO search_chunks_fts (id, conversation_id, source_file_id, source_filename, text, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["conversation_id"],
                        chunk["source_file_id"],
                        chunk["source_filename"],
                        chunk["text"],
                        json.dumps(metadata),
                    ),
                )
                for question in metadata.get("generated_questions", []):
                    conn.execute(
                        "INSERT INTO generated_chunk_questions (id, chunk_id, question, created_at) VALUES (?, ?, ?, ?)",
                        (new_id("q"), chunk["id"], question, utc_now()),
                    )
            conn.commit()

    def update_chunk_embeddings(self, embeddings: dict[str, list[float]], *, model: str, dim: int) -> None:
        if not embeddings:
            return
        with self._locked_conn() as conn:
            for chunk_id, vector in embeddings.items():
                row = conn.execute(
                    "SELECT metadata_json FROM search_chunks WHERE id = ?",
                    (chunk_id,),
                ).fetchone()
                metadata = _json(row["metadata_json"], {}) if row else {}
                metadata["embedding_model"] = model
                metadata["embedding_dim"] = dim
                conn.execute(
                    "UPDATE search_chunks SET embedding_json = ?, metadata_json = ? WHERE id = ?",
                    (json.dumps(vector), json.dumps(metadata), chunk_id),
                )
            conn.commit()

    def list_chunks(
        self,
        conversation_id: str,
        *,
        source_file_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM search_chunks WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if source_file_ids:
            placeholders = ",".join("?" for _ in source_file_ids)
            sql += f" AND source_file_id IN ({placeholders})"
            params.extend(source_file_ids)
        sql += " ORDER BY source_filename ASC, chunk_index ASC"
        rows = self.query(sql, params)
        for row in rows:
            row["metadata"] = _json(row.pop("metadata_json"), {})
            row["embedding"] = _json(row.pop("embedding_json"), None)
        return rows

    def insert_artifact(
        self,
        *,
        artifact_id: str,
        conversation_id: str,
        artifact_type: str,
        filename: str,
        local_key: str,
        mime_type: str,
        source_message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        self.execute(
            """
            INSERT INTO artifacts
              (id, conversation_id, source_message_id, type, filename, local_key, mime_type, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                conversation_id,
                source_message_id,
                artifact_type,
                filename,
                local_key,
                mime_type,
                json.dumps(metadata or {}),
                now,
            ),
        )
        return self.get_artifact(artifact_id) or {}

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self.one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        if row:
            row["metadata"] = _json(row.pop("metadata_json"), {})
        return row

    def list_artifacts(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM artifacts WHERE conversation_id = ? ORDER BY created_at DESC",
            (conversation_id,),
        )
        for row in rows:
            row["metadata"] = _json(row.pop("metadata_json"), {})
        return rows

    def log_hyde_trace(self, trace: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO hyde_traces
              (id, conversation_id, query, provider_id, hyde_preview, hyde_hash, status, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("hyde"),
                trace.get("conversation_id"),
                trace.get("query"),
                trace.get("provider_id"),
                trace.get("hyde_preview"),
                trace.get("hyde_hash"),
                trace.get("status", "ok"),
                json.dumps(trace.get("metadata", {})),
                utc_now(),
            ),
        )

    def _seed_default_provider(self) -> None:
        existing = self.one("SELECT id FROM llm_providers WHERE provider_type = 'ollama' LIMIT 1")
        if existing:
            return
        now = utc_now()
        self.execute(
            """
            INSERT INTO llm_providers
              (id, display_name, provider_type, base_url, model_name, api_key_ciphertext,
               is_default_chat, is_default_embedding, status, metadata_json, created_at, updated_at)
            VALUES (?, ?, 'ollama', ?, ?, NULL, 1, 0, 'unknown', '{}', ?, ?)
            """,
            (
                "provider_ollama_default",
                "Ollama",
                self.settings.default_ollama_base_url,
                self.settings.default_ollama_model,
                now,
                now,
            ),
        )

    def _seed_generator_registry(self) -> None:
        path = self.settings.generators_registry_path
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        generators = data.get("generators", [])
        with self._locked_conn() as conn:
            for manifest in generators:
                generator_id = manifest["generator_id"]
                conn.execute(
                    """
                    INSERT INTO generator_registry (id, manifest_json, enabled, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      manifest_json = excluded.manifest_json,
                      enabled = excluded.enabled,
                      updated_at = excluded.updated_at
                    """,
                    (
                        generator_id,
                        json.dumps(manifest),
                        1 if manifest.get("enabled", True) else 0,
                        utc_now(),
                    ),
                )
            conn.commit()


class _LockedConnection:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def __enter__(self) -> sqlite3.Connection:
        return self.store._connect()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.store._lock.release()


def _json(raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


_store: SQLiteStore | None = None


def get_store() -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore()
    return _store


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  output_type TEXT NOT NULL DEFAULT 'text',
  artifacts_json TEXT NOT NULL DEFAULT '[]',
  sources_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uploaded_files (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  original_path TEXT,
  stored_path TEXT NOT NULL,
  mime_type TEXT NOT NULL DEFAULT '',
  size_bytes INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  parser_used TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS course_documents (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  source_file_id TEXT NOT NULL,
  title TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS course_sections (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  source_file_id TEXT NOT NULL,
  heading_path_json TEXT NOT NULL DEFAULT '[]',
  summary TEXT NOT NULL DEFAULT '',
  order_index INTEGER NOT NULL DEFAULT 0,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_chunks (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  source_file_id TEXT NOT NULL,
  document_id TEXT,
  section_id TEXT,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  source_filename TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  embedding_json TEXT,
  created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS search_chunks_fts USING fts5(
  id UNINDEXED,
  conversation_id UNINDEXED,
  source_file_id UNINDEXED,
  source_filename,
  text,
  metadata
);

CREATE TABLE IF NOT EXISTS generated_chunk_questions (
  id TEXT PRIMARY KEY,
  chunk_id TEXT NOT NULL,
  question TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS formulas (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  source_file_id TEXT,
  chunk_id TEXT,
  label TEXT,
  expression TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tables (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  source_file_id TEXT,
  chunk_id TEXT,
  caption TEXT,
  content_json TEXT NOT NULL DEFAULT '{}',
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS timeline_events (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  source_file_id TEXT,
  chunk_id TEXT,
  label TEXT NOT NULL,
  event_date TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS concept_inventory (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  name TEXT NOT NULL,
  aliases_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS learning_phases (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  title TEXT NOT NULL,
  order_index INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS learning_objectives (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  phase_id TEXT,
  objective_text TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  label TEXT NOT NULL,
  node_type TEXT NOT NULL DEFAULT 'concept',
  node_key TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  ref_id TEXT,
  source_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT,
  updated_at TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  source_node_id TEXT NOT NULL,
  target_node_id TEXT NOT NULL,
  edge_type TEXT NOT NULL,
  relation_type TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT 0.6,
  source_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT,
  updated_at TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS learner_state (
  conversation_id TEXT PRIMARY KEY,
  state_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_windows (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_checks (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coursebuilder_courses (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'ready',
  build_id TEXT,
  source_fingerprint TEXT NOT NULL DEFAULT '',
  quality_mode TEXT NOT NULL DEFAULT 'fallback',
  error TEXT,
  created_at TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coursebuilder_plans (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL UNIQUE,
  plan_id TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'draft',
  source_fingerprint TEXT NOT NULL DEFAULT '',
  quality_mode TEXT NOT NULL DEFAULT 'fallback',
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_coursebuilder_plans_conversation
  ON coursebuilder_plans(conversation_id, updated_at);

CREATE TABLE IF NOT EXISTS coursebuilder_progress (
  conversation_id TEXT PRIMARY KEY,
  course_id TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL DEFAULT '',
  progress_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coursebuilder_quiz_attempts (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  course_id TEXT NOT NULL,
  quiz_id TEXT NOT NULL,
  answers_json TEXT NOT NULL DEFAULT '[]',
  score REAL NOT NULL DEFAULT 0,
  passed INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_coursebuilder_attempts_conversation
  ON coursebuilder_quiz_attempts(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  source_message_id TEXT,
  type TEXT NOT NULL,
  filename TEXT NOT NULL,
  local_key TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generator_registry (
  id TEXT PRIMARY KEY,
  manifest_json TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS connected_external_agents (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  manifest_json TEXT NOT NULL DEFAULT '{}',
  permissions_json TEXT NOT NULL DEFAULT '{}',
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generator_run_traces (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  generator_id TEXT NOT NULL,
  trace_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS background_jobs (
  id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_providers (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  provider_type TEXT NOT NULL,
  base_url TEXT NOT NULL,
  model_name TEXT NOT NULL,
  api_key_ciphertext TEXT,
  is_default_chat INTEGER NOT NULL DEFAULT 0,
  is_default_embedding INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'unknown',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parser_settings (
  id TEXT PRIMARY KEY CHECK (id = 'default'),
  llama_cloud_api_key_ciphertext TEXT,
  use_local_parsers_only INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'local',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coursebuilder_settings (
  id TEXT PRIMARY KEY CHECK (id = 'default'),
  sequential_unlocking_enabled INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generator_settings (
  id TEXT PRIMARY KEY CHECK (id = 'default'),
  podcast_audio_enabled INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_settings (
  id TEXT PRIMARY KEY CHECK (id = 'default'),
  embedding_model TEXT NOT NULL,
  embedding_dim INTEGER NOT NULL,
  embedding_batch_size INTEGER NOT NULL,
  retrieval_top_k INTEGER NOT NULL,
  retrieval_dense_candidate_k INTEGER NOT NULL,
  retrieval_sparse_candidate_k INTEGER NOT NULL,
  retrieval_hyde_enabled INTEGER NOT NULL,
  retrieval_rerank_enabled INTEGER NOT NULL,
  retrieval_reranker_model TEXT NOT NULL,
  retrieval_graph_enabled INTEGER NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hyde_traces (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  query TEXT NOT NULL,
  provider_id TEXT,
  hyde_preview TEXT,
  hyde_hash TEXT,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
  id TEXT PRIMARY KEY,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite storage layer for metadata and experiment logs.

This module keeps structured data in SQLite while vectors remain in FAISS.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "storage" / "rag_system.db"


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _to_json(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


class RAGStorageDB:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    storage_path TEXT,
                    index_file TEXT,
                    doc_file TEXT,
                    embedding_model TEXT,
                    vector_dimension INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    source TEXT DEFAULT 'upload',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(collection_id, file_name),
                    FOREIGN KEY(collection_id) REFERENCES knowledge_collections(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER NOT NULL,
                    document_id INTEGER,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT,
                    char_length INTEGER DEFAULT 0,
                    faiss_row_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(collection_id) REFERENCES knowledge_collections(id) ON DELETE CASCADE,
                    FOREIGN KEY(document_id) REFERENCES source_documents(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_collection
                ON document_chunks(collection_id, chunk_index);

                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    collection_name TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    final_score REAL DEFAULT 0,
                    final_step TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS query_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER NOT NULL,
                    intent_json TEXT,
                    planner_json TEXT,
                    reasoning_json TEXT,
                    evaluation_json TEXT,
                    optimization_suggestions_json TEXT,
                    optimization_report TEXT,
                    raw_answer TEXT,
                    optimized_answer TEXT,
                    retrieve_rounds INTEGER DEFAULT 0,
                    called_tools_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(query_id) REFERENCES query_logs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_run_id INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_payload_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(query_run_id) REFERENCES query_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS evaluation_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_run_id INTEGER NOT NULL UNIQUE,
                    retrieval_relevance INTEGER DEFAULT 0,
                    answer_accuracy INTEGER DEFAULT 0,
                    answer_completeness INTEGER DEFAULT 0,
                    reasoning_effectiveness INTEGER DEFAULT 0,
                    tool_call_appropriateness INTEGER DEFAULT 0,
                    result_fusion_quality INTEGER DEFAULT 0,
                    answer_optimization_effect INTEGER DEFAULT 0,
                    suggestion TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(query_run_id) REFERENCES query_runs(id) ON DELETE CASCADE
                );
                """
            )
            self._migrate_legacy_schema(conn)

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> List[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return [str(row["name"]) for row in rows]

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _migrate_legacy_schema(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "tool_calls"):
            return

        tool_call_columns = self._table_columns(conn, "tool_calls")
        if "query_run_id" in tool_call_columns and "tool_payload_json" in tool_call_columns:
            return

        logger.warning("Detected legacy tool_calls schema, migrating to the current layout.")

        backup_name = "tool_calls_legacy_backup"
        if self._table_exists(conn, backup_name):
            conn.execute(f"DROP TABLE {backup_name}")

        conn.execute("ALTER TABLE tool_calls RENAME TO tool_calls_legacy_backup")
        conn.execute(
            """
            CREATE TABLE tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_run_id INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                tool_payload_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(query_run_id) REFERENCES query_runs(id) ON DELETE CASCADE
            )
            """
        )

    def upsert_collection(
        self,
        name: str,
        storage_path: Optional[str] = None,
        index_file: Optional[str] = None,
        doc_file: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_dimension: int = 0,
        chunk_count: int = 0,
    ) -> int:
        now = _utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO knowledge_collections
                    (name, storage_path, index_file, doc_file, embedding_model, vector_dimension, chunk_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    storage_path = excluded.storage_path,
                    index_file = excluded.index_file,
                    doc_file = excluded.doc_file,
                    embedding_model = excluded.embedding_model,
                    vector_dimension = excluded.vector_dimension,
                    chunk_count = excluded.chunk_count,
                    updated_at = excluded.updated_at
                """,
                (
                    name,
                    storage_path,
                    index_file,
                    doc_file,
                    embedding_model,
                    vector_dimension,
                    chunk_count,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM knowledge_collections WHERE name = ?",
                (name,),
            ).fetchone()
            return int(row["id"])

    def replace_collection_documents(
        self,
        collection_name: str,
        documents: Iterable[Dict[str, Any]],
        chunks: Iterable[str],
        storage_path: Optional[str] = None,
        index_file: Optional[str] = None,
        doc_file: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_dimension: int = 0,
    ) -> int:
        chunks = list(chunks)
        collection_id = self.upsert_collection(
            name=collection_name,
            storage_path=storage_path,
            index_file=index_file,
            doc_file=doc_file,
            embedding_model=embedding_model,
            vector_dimension=vector_dimension,
            chunk_count=len(chunks),
        )
        now = _utc_now()

        with self.connect() as conn:
            conn.execute("DELETE FROM source_documents WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM document_chunks WHERE collection_id = ?", (collection_id,))

            document_ids: List[int] = []
            for doc in documents:
                cursor = conn.execute(
                    """
                    INSERT INTO source_documents
                        (collection_id, file_name, file_path, file_type, file_size, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        collection_id,
                        doc.get("file_name", "unknown"),
                        doc.get("file_path"),
                        doc.get("file_type"),
                        doc.get("file_size"),
                        doc.get("source", "upload"),
                        now,
                        now,
                    ),
                )
                document_ids.append(int(cursor.lastrowid))

            fallback_document_id = document_ids[0] if document_ids else None
            for idx, content in enumerate(chunks):
                conn.execute(
                    """
                    INSERT INTO document_chunks
                        (collection_id, document_id, chunk_index, content, char_length, faiss_row_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        collection_id,
                        fallback_document_id,
                        idx,
                        content,
                        len(content),
                        idx,
                        now,
                    ),
                )

            conn.execute(
                """
                UPDATE knowledge_collections
                SET chunk_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (len(chunks), now, collection_id),
            )

        return collection_id

    def log_query_result(self, result: Dict[str, Any], collection_name: Optional[str] = None) -> int:
        now = _utc_now()
        reasoning = result.get("reasoning", {}) or {}
        evaluation = result.get("evaluation", {}) or {}
        with self.connect() as conn:
            query_cursor = conn.execute(
                """
                INSERT INTO query_logs
                    (query_text, collection_name, success, retry_count, final_score, final_step, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.get("query", ""),
                    collection_name,
                    1 if result.get("success") else 0,
                    int(result.get("retry_count", 0) or 0),
                    float(result.get("final_score", 0) or 0),
                    result.get("step"),
                    result.get("error"),
                    now,
                ),
            )
            query_id = int(query_cursor.lastrowid)

            run_cursor = conn.execute(
                """
                INSERT INTO query_runs
                    (query_id, intent_json, planner_json, reasoning_json, evaluation_json, optimization_suggestions_json,
                     optimization_report, raw_answer, optimized_answer, retrieve_rounds, called_tools_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    _to_json(result.get("intent")),
                    _to_json(result.get("planner")),
                    _to_json(reasoning),
                    _to_json(evaluation),
                    _to_json(result.get("optimization_suggestions", [])),
                    result.get("optimization_report"),
                    reasoning.get("raw_answer"),
                    reasoning.get("optimized_answer"),
                    int(reasoning.get("retrieve_rounds", 0) or 0),
                    _to_json(reasoning.get("called_tools", [])),
                    now,
                ),
            )
            query_run_id = int(run_cursor.lastrowid)

            for tool_name in reasoning.get("called_tools", []) or []:
                conn.execute(
                    """
                    INSERT INTO tool_calls (query_run_id, tool_name, tool_payload_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        query_run_id,
                        tool_name,
                        _to_json(reasoning),
                        now,
                    ),
                )

            if evaluation:
                conn.execute(
                    """
                    INSERT INTO evaluation_scores
                        (query_run_id, retrieval_relevance, answer_accuracy, answer_completeness,
                         reasoning_effectiveness, tool_call_appropriateness, result_fusion_quality,
                         answer_optimization_effect, suggestion, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_run_id,
                        int(evaluation.get("retrieval_relevance", 0) or 0),
                        int(evaluation.get("answer_accuracy", 0) or 0),
                        int(evaluation.get("answer_completeness", 0) or 0),
                        int(evaluation.get("reasoning_effectiveness", 0) or 0),
                        int(evaluation.get("tool_call_appropriateness", 0) or 0),
                        int(evaluation.get("result_fusion_quality", 0) or 0),
                        int(evaluation.get("answer_optimization_effect", 0) or 0),
                        evaluation.get("suggestion"),
                        now,
                    ),
                )

        return query_id

    def get_summary(self) -> Dict[str, int]:
        with self.connect() as conn:
            collections = conn.execute("SELECT COUNT(*) AS cnt FROM knowledge_collections").fetchone()["cnt"]
            documents = conn.execute("SELECT COUNT(*) AS cnt FROM source_documents").fetchone()["cnt"]
            chunks = conn.execute("SELECT COUNT(*) AS cnt FROM document_chunks").fetchone()["cnt"]
            queries = conn.execute("SELECT COUNT(*) AS cnt FROM query_logs").fetchone()["cnt"]
            return {
                "collections": int(collections),
                "documents": int(documents),
                "chunks": int(chunks),
                "queries": int(queries),
            }


_db_instance: Optional[RAGStorageDB] = None


def get_storage_db() -> RAGStorageDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = RAGStorageDB()
    return _db_instance


if __name__ == "__main__":
    db = get_storage_db()
    print(json.dumps(db.get_summary(), ensure_ascii=False, indent=2))

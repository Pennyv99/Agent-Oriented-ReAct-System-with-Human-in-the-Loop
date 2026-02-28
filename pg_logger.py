# pg_logger.py
import os
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from typing import Any, Dict, Optional


class PGLogger:
    def __init__(self) -> None:
        self.conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            dbname=os.getenv("PG_DB", "agent"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", "password"),
        )
        self.conn.autocommit = True
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    meta JSONB,
                    created_at TIMESTAMP NOT NULL
                );
                """
            )

    def log_chat(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_logs(session_id, user_message, agent_response, meta, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, user_message, agent_response, Json(meta or {}), datetime.utcnow()),
            )
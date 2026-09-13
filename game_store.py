"""SQLite-backed persistence for game state.

Writes whole-snapshot or per-game JSON blobs keyed by game_id. The
in-memory snapshot is authoritative: save_all() does a full-table replace
so in-memory deletions propagate. All IO happens on the same event-loop
thread in practice, but a lock + check_same_thread=False keep it safe if a
call ever lands on another thread.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id    TEXT PRIMARY KEY,
    blob       TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class GameStore:
    def __init__(self, db_path, timeout=5.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), timeout=timeout, check_same_thread=False
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def save_all(self, games):
        """Replace the whole table with {game_id: to_persistable dict}.

        Deletions propagate because rows for games no longer in memory are
        dropped. No-op when there is nothing to write (e.g. the in-memory
        manager is empty)."""
        if not games:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                self._conn.execute("DELETE FROM games")
                self._conn.executemany(
                    "INSERT INTO games (game_id, blob, updated_at) VALUES (?, ?, ?)",
                    [(gid, json.dumps(blob), now) for gid, blob in games.items()],
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def upsert(self, game_id, blob):
        with self._lock:
            self._conn.execute(
                "INSERT INTO games (game_id, blob, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(game_id) DO UPDATE SET "
                "blob = excluded.blob, updated_at = excluded.updated_at",
                (
                    game_id,
                    json.dumps(blob),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            self._conn.commit()

    def load_all(self):
        """-> {game_id: persisted dict}. Corrupt rows are logged and skipped."""
        out = {}
        with self._lock:
            rows = self._conn.execute("SELECT game_id, blob FROM games").fetchall()
        for gid, blob in rows:
            try:
                out[gid] = json.loads(blob)
            except json.JSONDecodeError:
                logger.exception("ignoring corrupt persisted state for %s", gid)
        return out

    def close(self):
        with self._lock:
            self._conn.close()
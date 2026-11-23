"""SQLite persistence for compressed furnace archives.

The store is safe to use from several threads *and* from several OS processes
that share one database file (for example two service replicas mounting the
same volume).

Every write runs inside ``BEGIN IMMEDIATE`` so that SQLite's own write lock --
not a process-local mutex -- decides who writes first.  The "was there already
an archive?" question is answered by the very transaction that writes, which
makes ``replaced`` exact even under contention: exactly one of several racing
submissions of a brand-new run sees ``replaced=False``.

The write transaction is deliberately tiny.  Compression, JSON parsing and
response building all happen outside it, so a long curve being compressed never
blocks reads or short archives for other runs.

Failed or rejected writes never reach :meth:`put`; if the database itself fails
mid-write the transaction is rolled back and the previous archive survives.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

DEFAULT_DB_PATH = os.environ.get("FURNACE_DB_PATH", os.path.join("data", "archive.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS archives (
    run_id       TEXT PRIMARY KEY,
    segments     TEXT NOT NULL,
    limit_value  TEXT NOT NULL,
    tolerance    TEXT NOT NULL,
    max_points   INTEGER NOT NULL,
    point_count  INTEGER NOT NULL,
    source_point_count INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""

_UPSERT = """
INSERT INTO archives
    (run_id, segments, limit_value, tolerance, max_points,
     point_count, source_point_count, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(run_id) DO UPDATE SET
    segments = excluded.segments,
    limit_value = excluded.limit_value,
    tolerance = excluded.tolerance,
    max_points = excluded.max_points,
    point_count = excluded.point_count,
    source_point_count = excluded.source_point_count,
    updated_at = excluded.updated_at
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ArchiveStore:
    """A tiny SQLite-backed archive keyed by furnace run id."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        # Serialises schema creation and writes between threads of this
        # process.  Cross-process serialisation is done by SQLite itself.
        self._lock = threading.Lock()
        self._initialised = False

    # -- connection handling ------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # Autocommit mode: transactions are opened explicitly below, so a
        # SELECT never leaves an implicit read transaction open on the
        # connection and locks are held for exactly as long as intended.
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            parent = os.path.dirname(os.path.abspath(self.db_path))
            os.makedirs(parent, exist_ok=True)
            with self._connection() as conn:
                # WAL lets readers (GET) proceed while another writer commits.
                # It is persistent in the database file; if it cannot be
                # enabled (odd filesystem) the rollback journal still works.
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.OperationalError:
                    pass
                # ``IMMEDIATE`` takes the write lock before we look at the
                # schema, so two processes starting at once cannot both run
                # the same migration.
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute(_SCHEMA)
                    # Lightweight in-place migration for archives created by an
                    # older build that did not track the source sample count.
                    columns = {
                        row["name"]
                        for row in conn.execute(
                            "PRAGMA table_info(archives)"
                        ).fetchall()
                    }
                    if "source_point_count" not in columns:
                        conn.execute(
                            "ALTER TABLE archives ADD COLUMN source_point_count "
                            "INTEGER NOT NULL DEFAULT 0"
                        )
                    conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
            self._initialised = True

    # -- reads ---------------------------------------------------------

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return the last successfully stored archive for ``run_id``."""
        self._ensure_schema()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM archives WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list_runs(self) -> list[dict[str, Any]]:
        self._ensure_schema()
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM archives ORDER BY updated_at DESC, run_id ASC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # -- writes --------------------------------------------------------

    def put(
        self,
        run_id: str,
        segments: list[list[dict[str, int]]],
        limit_value: str,
        tolerance: str,
        max_points: int,
        point_count: int,
        source_point_count: int,
    ) -> dict[str, Any]:
        """Store ``run_id`` and return a snapshot of exactly this write.

        ``replaced`` is decided inside the write transaction: ``False`` only
        when this transaction inserted the row, ``True`` when it updated an
        archive that was already committed.  The returned record is built from
        the values written here, so a concurrent replacement cannot leak into
        the caller's copy of the archive.
        """
        self._ensure_schema()
        now = _utcnow()
        segments_json = json.dumps(segments, separators=(",", ":"), ensure_ascii=False)
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT created_at FROM archives WHERE run_id = ?", (run_id,)
                ).fetchone()
                replaced = existing is not None
                created = existing["created_at"] if replaced else now
                conn.execute(
                    _UPSERT,
                    (
                        run_id,
                        segments_json,
                        limit_value,
                        tolerance,
                        max_points,
                        point_count,
                        source_point_count,
                        created,
                        now,
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return {
            "run_id": run_id,
            "segments": segments,
            "limit": limit_value,
            "tolerance": tolerance,
            "max_points": max_points,
            "point_count": point_count,
            "source_point_count": source_point_count,
            "created_at": created,
            "updated_at": now,
            "replaced": replaced,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        # ``limit`` and ``tolerance`` are stored as exact rational literals so
        # no precision is lost on the round trip; callers convert them.
        return {
            "run_id": row["run_id"],
            "segments": json.loads(row["segments"]),
            "limit": row["limit_value"],
            "tolerance": row["tolerance"],
            "max_points": row["max_points"],
            "point_count": row["point_count"],
            "source_point_count": row["source_point_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

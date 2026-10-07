"""SQLite-backed persistence for device slot manifests and image bytes.

Everything the recovery adjudicator needs is durable *before* a simulated
power cut is reported:

* the full slot roster (A/B) with stage, version, claimed/measured digest,
* the candidate staging progress (``written`` bytes),
* the confirmation generation and the qualification token,
* append-only diagnostic evidence,
* **each slot's own image bytes in a separate ``(device_id, slot) row** --
  staging a candidate into the inactive slot can never overwrite the active
  slot's persisted image.

Mutations run under ``BEGIN IMMEDIATE`` so concurrent candidate submissions are
serialised by SQLite itself; the service layer then applies the generation
qualification rule on the freshly read row.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .models import Device

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id   TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    powered_on  INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS blobs (
    device_id TEXT NOT NULL,
    slot      TEXT NOT NULL,
    content   BLOB NOT NULL,
    PRIMARY KEY (device_id, slot)
);
"""

# Matches a composite primary key on (device_id, slot), tolerating whitespace.
_COMPOSITE_PK = re.compile(
    r"primary\s+key\s*\(\s*device_id\s*,\s*slot\s*\)", re.IGNORECASE
)


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._migrate_blob_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    def _migrate_blob_schema(self) -> None:
        """Upgrade databases created by the early single-blob-per-device schema.

        The original ``blobs`` table keyed rows by ``device_id`` alone, so
        writing a candidate into the inactive slot silently replaced the
        active slot's image while its manifest kept claiming the old digest.
        Such a database is copied row by row into the composite-key layout;
        recovery then re-measures every slot and refuses any slot whose
        persisted content cannot be verified (no guesswork, no rollback).
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='blobs'"
        ).fetchone()
        if row is not None and _COMPOSITE_PK.search(row["sql"] or ""):
            return
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                CREATE TABLE blobs_v2 (
                    device_id TEXT NOT NULL,
                    slot      TEXT NOT NULL,
                    content   BLOB NOT NULL,
                    PRIMARY KEY (device_id, slot)
                )
                """
            )
            # One row per device under the old key; keep it tagged with the
            # slot it was last written for and let recovery adjudicate it.
            self._conn.execute(
                """
                INSERT OR IGNORE INTO blobs_v2 (device_id, slot, content)
                SELECT device_id, slot, content FROM blobs
                """
            )
            self._conn.execute("DROP TABLE blobs")
            self._conn.execute("ALTER TABLE blobs_v2 RENAME TO blobs")
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ #
    def list_device_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT device_id FROM devices ORDER BY device_id"
            ).fetchall()
            return [r["device_id"] for r in rows]

    def is_powered_on(self, device_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT powered_on FROM devices WHERE device_id=?", (device_id,)
            ).fetchone()
            if row is None:
                raise KeyError(device_id)
            return bool(row["powered_on"])

    def load_raw(self, device_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data, powered_on FROM devices WHERE device_id=?",
                (device_id,),
            ).fetchone()
            if row is None:
                return None
            data = json.loads(row["data"])
            data["_powered_on"] = bool(row["powered_on"])
            return data

    def load(self, device_id: str) -> Device:
        raw = self.load_raw(device_id)
        if raw is None:
            raise KeyError(device_id)
        raw.pop("_powered_on", None)
        return Device.from_dict(raw)

    def load_all(self) -> list[Device]:
        return [self.load(did) for did in self.list_device_ids()]

    def get_blob(self, device_id: str, slot: str) -> Optional[bytes]:
        """Read one slot's persisted image outside a transaction (tests/diag)."""
        with self._lock:
            return self.read_blob(self._conn, device_id, slot)

    # ------------------------------------------------------------------ #
    def transaction(self):
        """Context manager yielding a fresh in-transaction connection.

        Callers must re-read the device inside the transaction and persist it
        through :meth:`save` before exiting -- ``COMMIT`` is the single atomic
        switch point, exactly like the bootloader's manifest commit.
        """
        return _Tx(self)

    def save(self, conn: sqlite3.Connection, device: Device) -> None:
        payload = json.dumps(device.to_dict(), separators=(",", ":"))
        conn.execute(
            """
            INSERT INTO devices (device_id, data, powered_on)
            VALUES (?, ?, 1)
            ON CONFLICT(device_id) DO UPDATE SET data=excluded.data
            """,
            (device.device_id, payload),
        )

    def insert_device(
        self, conn: sqlite3.Connection, device: Device, powered_on: bool = True
    ) -> None:
        payload = json.dumps(device.to_dict(), separators=(",", ":"))
        conn.execute(
            "INSERT INTO devices (device_id, data, powered_on) VALUES (?, ?, ?)",
            (device.device_id, payload, 1 if powered_on else 0),
        )

    def set_powered(
        self, conn: sqlite3.Connection, device_id: str, powered_on: bool
    ) -> None:
        conn.execute(
            "UPDATE devices SET powered_on=? WHERE device_id=?",
            (1 if powered_on else 0, device_id),
        )

    @staticmethod
    def read_blob(
        conn: sqlite3.Connection, device_id: str, slot: str
    ) -> Optional[bytes]:
        row = conn.execute(
            "SELECT content FROM blobs WHERE device_id=? AND slot=?",
            (device_id, slot),
        ).fetchone()
        return row["content"] if row is not None else None

    def write_blob(
        self, conn: sqlite3.Connection, device_id: str, slot: str, content: bytes
    ) -> None:
        conn.execute(
            """
            INSERT INTO blobs (device_id, slot, content) VALUES (?, ?, ?)
            ON CONFLICT(device_id, slot) DO UPDATE SET
                content=excluded.content
            """,
            (device_id, slot, content),
        )

    def append_blob(
        self, conn: sqlite3.Connection, device_id: str, slot: str, chunk: bytes
    ) -> int:
        """Persist a streaming write to one slot; returns the new byte count."""
        previous = self.read_blob(conn, device_id, slot) or b""
        content = previous + chunk
        conn.execute(
            """
            INSERT INTO blobs (device_id, slot, content) VALUES (?, ?, ?)
            ON CONFLICT(device_id, slot) DO UPDATE SET
                content=excluded.content
            """,
            (device_id, slot, content),
        )
        return len(content)


class _Tx:
    def __init__(self, store: Store):
        self._store = store
        self._lock = store._lock
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        conn = self._store._conn
        conn.execute("BEGIN IMMEDIATE")
        self.conn = conn
        return conn

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.conn.execute("COMMIT")
            else:
                # Any error mid-stage rolls back exactly the un-committed
                # switch -- a real power cut would discard the same page cache.
                self.conn.execute("ROLLBACK")
        finally:
            self.conn = None
            self._lock.release()

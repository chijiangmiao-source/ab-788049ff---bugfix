"""Migration from the early single-blob-row-per-device layout."""
from __future__ import annotations

import json
import sqlite3


def _old_payload(image_a: bytes, digest: str) -> str:
    slot_a = {
        "name": "A", "status": "CONFIRMED", "version": "1.0.0",
        "digest": digest, "actual_digest": digest,
        "size": len(image_a), "written": len(image_a), "confirmed_generation": 1,
    }
    slot_b = {"name": "B", "status": "EMPTY", "version": None, "digest": None,
              "actual_digest": None, "size": None, "written": 0,
              "confirmed_generation": None}
    return json.dumps({
        "device_id": "d", "active_slot": "A", "generation": 1,
        "qualified_generation": None, "qualified_request": None,
        "qualified_slot": None, "slots": {"A": slot_a, "B": slot_b},
        "last_recovery": None, "recovery_history": [], "evidence": [], "evidence_seq": 0,
    }, separators=(",", ":"))


def test_legacy_single_blob_row_migrates_and_keeps_active_image(tmp_path):
    import hashlib
    import os

    db = tmp_path / "legacy.db"
    image_a = b"legacy-active-image"
    digest = hashlib.sha256(image_a).hexdigest()

    raw = sqlite3.connect(db)
    raw.executescript(
        """
        CREATE TABLE devices (device_id TEXT PRIMARY KEY, data TEXT NOT NULL,
                              powered_on INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE blobs (device_id TEXT PRIMARY KEY, slot TEXT NOT NULL,
                            content BLOB NOT NULL);
        """
    )
    raw.execute("INSERT INTO devices (device_id, data, powered_on) VALUES (?,?,0)",
                ("d", _old_payload(image_a, digest)))
    raw.execute("INSERT INTO blobs VALUES (?,?,?)", ("d", "A", image_a))
    raw.commit()
    raw.close()

    os.environ["DATA_PATH"] = str(db)
    from backend.store import Store

    store = Store(db)  # triggers migration
    try:
        with store.transaction() as conn:
            assert store.read_blob(conn, "d", "A") == image_a
            assert store.read_blob(conn, "d", "B") is None

        from backend.service import UpgradeService

        svc = UpgradeService(store)
        out = svc.power_on("d")
        assert out["recovery"]["active_slot"] == "A"
    finally:
        store.close()

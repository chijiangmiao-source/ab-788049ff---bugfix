"""Slot image isolation: unconfirmed candidates must never alter the active slot.

Every check runs against an isolated database (the ``client`` fixture uses a
per-test ``tmp_path``). Two independent guarantees are audited:

1. During all three interrupted candidate phases (write / digest check /
   verified-but-unconfirmed) the *old active slot's persisted bytes* stay
   byte-identical to its manifest, not just its metadata.
2. After a committed switch the new slot is the unique active slot with
   matching content, manifest digest and confirmation generation, and the
   next upgrade attempt keeps preserving the current active slot until that
   switch commits.

A separate test fabricates the legacy single-blob-per-device database that
was previously affected: such data must be identified on reopen and refused,
never booted and never rolled back through to a superseded version.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

from tests.conftest import make_device


def _payload_digest(version: str) -> str:
    tag = f"payload-image::{version}::".encode()
    data = (tag + bytes((i * 7 + len(version)) & 0xFF for i in range(256 - len(tag))))[:256]
    return hashlib.sha256(data).hexdigest()


def _slot(view, name):
    return view["slots"][name]


def test_old_slot_bytes_unchanged_after_interrupted_candidate_write(client):
    dev = make_device(client)
    a0 = _slot(dev, "A")
    assert a0["stored_digest"] == a0["digest"]
    a_digest = a0["digest"]

    r = client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1", "fault_point": "candidate_write",
    })
    assert r.status_code == 200
    # Even while powered off, the persisted bytes of A are still its image.
    assert _slot(r.json()["device"], "A")["stored_digest"] == a_digest

    r = client.post("/api/devices/dev-1/power-on")
    rec, reopened = r.json()["recovery"], r.json()["device"]
    assert rec["active_slot"] == "A"
    a = _slot(reopened, "A")
    assert a["version"] == "1.0.0"
    assert a["status"] == "CONFIRMED"
    assert a["stored_digest"] == a_digest              # actual bytes == manifest
    assert a["stored_matches_manifest"] is True
    b = _slot(reopened, "B")
    assert b["stored_size"] == b["written"] < b["size"]  # only the candidate slot is torn


def test_old_slot_bytes_unchanged_after_interrupted_digest_check(client):
    dev = make_device(client)
    a_digest = _slot(dev, "A")["digest"]

    client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1", "fault_point": "digest_check",
    })
    r = client.post("/api/devices/dev-1/power-on")
    rec, reopened = r.json()["recovery"], r.json()["device"]
    assert rec["active_slot"] == "A"
    a = _slot(reopened, "A")
    assert a["stored_digest"] == a_digest
    assert a["stored_matches_manifest"] is True
    b = _slot(reopened, "B")
    # Candidate bytes are fully present in B, but the verdict was never
    # committed; A's image is what actually boots.
    assert b["stored_size"] == b["size"]
    assert b["status"] == "CANDIDATE"
    assert b["actual_digest"] is None


def test_old_slot_bytes_unchanged_while_candidate_verified_but_unconfirmed(client):
    dev = make_device(client)
    a_digest = _slot(dev, "A")["digest"]

    staged = client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1",
    })
    assert staged.json()["outcome"] == "staged"
    # Normal verification succeeded; power off before any confirmation.
    client.post("/api/devices/dev-1/power-off")
    r = client.post("/api/devices/dev-1/power-on")
    rec, reopened = r.json()["recovery"], r.json()["device"]
    assert rec["active_slot"] == "A"
    assert rec["generation"] == 1
    diag_b = next(d for d in rec["diagnoses"] if d["slot"] == "B")
    assert diag_b["reason"] == "unconfirmed_candidate"
    a = _slot(reopened, "A")
    assert a["stored_digest"] == a_digest
    assert a["stored_matches_manifest"] is True
    b = _slot(reopened, "B")
    assert b["status"] == "VERIFIED"
    assert b["stored_digest"] == _payload_digest("2.0.0")

    # The pending candidate can still be confirmed after the recovery.
    sw = client.post("/api/devices/dev-1/confirm", json={}).json()
    assert sw["outcome"] == "switched" and sw["generation"] == 2


def test_switch_commits_content_digest_and_generation_then_survives_reopen(client):
    make_device(client)
    client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1",
    })
    sw = client.post("/api/devices/dev-1/confirm", json={}).json()
    assert sw["active_slot"] == "B" and sw["generation"] == 2
    b = _slot(sw["device"], "B")
    expected = _payload_digest("2.0.0")
    assert b["status"] == "CONFIRMED"
    assert b["confirmed_generation"] == 2
    assert b["digest"] == expected
    assert b["stored_digest"] == expected
    assert b["stored_matches_manifest"] is True
    assert _slot(sw["device"], "A")["status"] == "SUPERSEDED"

    client.post("/api/devices/dev-1/power-off")
    reopened = client.post("/api/devices/dev-1/power-on").json()["device"]
    b2 = _slot(reopened, "B")
    assert reopened["active_slot"] == "B"
    assert reopened["generation"] == 2
    for field, want in (
        ("status", "CONFIRMED"),
        ("version", "2.0.0"),
        ("digest", expected),
        ("stored_digest", expected),
        ("confirmed_generation", 2),
    ):
        assert b2[field] == want, field


def test_next_upgrade_keeps_current_active_slot_until_new_switch_commits(client):
    make_device(client)
    client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1",
    })
    client.post("/api/devices/dev-1/confirm", json={})
    healthy = client.get("/api/devices/dev-1").json()["device"]
    b_digest = _slot(healthy, "B")["stored_digest"]

    # Stage 3.0.0 into A and cut power during its write; B must remain intact.
    client.post("/api/devices/dev-1/candidate", json={
        "version": "3.0.0", "request_id": "r2", "fault_point": "candidate_write",
    })
    r = client.post("/api/devices/dev-1/power-on")
    rec, reopened = r.json()["recovery"], r.json()["device"]
    assert rec["active_slot"] == "B" and rec["generation"] == 2
    assert _slot(reopened, "B")["stored_digest"] == b_digest
    assert _slot(reopened, "B")["stored_matches_manifest"] is True

    # Complete the upgrade: A becomes the unique active slot at generation 3.
    client.post("/api/devices/dev-1/candidate", json={
        "version": "3.0.0", "request_id": "r3",
    })
    sw = client.post("/api/devices/dev-1/confirm", json={}).json()
    assert sw["active_slot"] == "A" and sw["generation"] == 3
    a = _slot(sw["device"], "A")
    assert a["stored_digest"] == _payload_digest("3.0.0")
    assert a["confirmed_generation"] == 3
    assert _slot(sw["device"], "B")["status"] == "SUPERSEDED"


def test_tampered_active_image_is_refused_without_rollback(client):
    make_device(client)
    client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1",
    })
    client.post("/api/devices/dev-1/confirm", json={})

    # Simulate silent flash corruption of the active slot's own bytes.
    import backend.api as api
    db = api.store.path
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA busy_timeout=5000")
    row = conn.execute(
        "SELECT content FROM blobs WHERE device_id=? AND slot=?", ("dev-1", "B")
    ).fetchone()
    bad = bytes([row[0][0] ^ 0x01]) + row[0][1:]
    conn.execute(
        "UPDATE blobs SET content=? WHERE device_id=? AND slot=?",
        (bad, "dev-1", "B"),
    )
    conn.commit()
    conn.close()

    client.post("/api/devices/dev-1/power-off")
    r = client.post("/api/devices/dev-1/power-on")
    body = r.json()
    assert body["outcome"] == "unbootable"
    assert body["recovery"]["active_slot"] is None
    diag = {d["slot"]: d["reason"] for d in body["recovery"]["diagnoses"]}
    assert diag["B"] == "image_unverifiable"
    assert diag["A"] == "superseded_no_rollback"  # never fall back to v1
    dev = body["device"]
    assert dev["slots"]["B"]["status"] == "REJECTED"
    assert dev["slots"]["A"]["status"] == "SUPERSEDED"
    ev = client.get("/api/devices/dev-1/evidence").json()["evidence"]
    assert any(e["reason"] == "image_unverifiable" for e in ev)

    # A later reopen keeps refusing it: no resurrection, no rollback.
    r2 = client.post("/api/devices/dev-1/power-on")
    assert r2.json()["recovery"]["active_slot"] is None


def _build_legacy_db(path: str, *, current_corrupt: bool) -> str:
    """Fabricate a database written by the old single-blob-per-device schema."""
    from backend.models import Device, Slot, SlotStatus

    def image(v: str) -> bytes:
        tag = f"payload-image::{v}::".encode()
        return (tag + bytes((i * 7 + len(v)) & 0xFF for i in range(256 - len(tag))))[:256]

    d1 = hashlib.sha256(image("1.0.0")).hexdigest()
    d2 = hashlib.sha256(image("2.0.0")).hexdigest()
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE devices (device_id TEXT PRIMARY KEY, data TEXT NOT NULL, "
        "powered_on INTEGER NOT NULL DEFAULT 1)"
    )
    conn.execute(
        "CREATE TABLE blobs (device_id TEXT PRIMARY KEY, slot TEXT NOT NULL, "
        "content BLOB NOT NULL)"
    )
    if current_corrupt:
        # v2 confirmed, v1 superseded; the lone old-schema blob row was last
        # overwritten by a torn 3.0.0 candidate write aimed at slot A.
        slot_a = Slot("A", SlotStatus.SUPERSEDED, "1.0.0", d1, d1, 256, 256, 1)
        slot_b = Slot("B", SlotStatus.CONFIRMED, "2.0.0", d2, d2, 256, 256, 2)
        dev = Device("dev", {"A": slot_a, "B": slot_b}, "B", generation=2)
        payload = json.dumps(dev.to_dict(), separators=(",", ":"))
        conn.execute("INSERT INTO devices VALUES (?,?,0)", ("dev", payload))
        conn.execute("INSERT INTO blobs VALUES (?,?,?)", ("dev", "A", image("3.0.0")[:96]))
    else:
        # v1 still active; the lone row was overwritten by a partial v2 write.
        slot_a = Slot("A", SlotStatus.CONFIRMED, "1.0.0", d1, d1, 256, 256, 1)
        slot_b = Slot("B", SlotStatus.CANDIDATE, "2.0.0", d2, None, 256, 128)
        dev = Device("dev", {"A": slot_a, "B": slot_b}, "A", generation=1)
        payload = json.dumps(dev.to_dict(), separators=(",", ":"))
        conn.execute("INSERT INTO devices VALUES (?,?,0)", ("dev", payload))
        conn.execute("INSERT INTO blobs VALUES (?,?,?)", ("dev", "B", image("2.0.0")[:128]))
    conn.commit()
    conn.close()
    return path


def test_legacy_affected_data_is_refused_not_booted(tmp_path):
    from backend.service import UpgradeService
    from backend.store import Store

    path = _build_legacy_db(str(tmp_path / "legacy.db"), current_corrupt=False)
    store = Store(path)
    try:
        res = UpgradeService(store).power_on("dev")
        assert res["outcome"] == "unbootable"
        rec = res["recovery"]
        assert rec["active_slot"] is None
        diag = {d["slot"]: d["reason"] for d in rec["diagnoses"]}
        assert diag["A"] == "image_unverifiable"  # manifest claimed v1, bytes gone
        assert res["device"]["slots"]["A"]["status"] == "REJECTED"
        # Stays refused on every subsequent open.
        assert UpgradeService(store).power_on("dev")["recovery"]["active_slot"] is None
    finally:
        store.close()


def test_legacy_affected_current_slot_never_rolls_back(tmp_path):
    from backend.service import UpgradeService
    from backend.store import Store

    path = _build_legacy_db(str(tmp_path / "legacy2.db"), current_corrupt=True)
    store = Store(path)
    try:
        res = UpgradeService(store).power_on("dev")
        assert res["outcome"] == "unbootable"
        assert res["recovery"]["active_slot"] is None
        dev = res["device"]
        assert dev["slots"]["B"]["status"] == "REJECTED"
        assert dev["slots"]["A"]["status"] == "SUPERSEDED"
        assert dev["active_slot"] == "B"  # pointer unchanged, but B is not booted
        diag = {d["slot"]: d["reason"] for d in res["recovery"]["diagnoses"]}
        assert diag["B"] == "image_unverifiable"
        assert diag["A"] == "superseded_no_rollback"
    finally:
        store.close()

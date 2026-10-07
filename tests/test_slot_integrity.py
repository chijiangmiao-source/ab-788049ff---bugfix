"""Slot image integrity across candidate staging, interruption and switch.

These tests judge each slot by its *own persisted image bytes* (re-measured via
the diagnostic ``/slot-images`` report), not only by manifest fields:

* while an unconfirmed candidate is staged into the inactive slot -- interrupted
  at any of the three fault points, or merely verified -- the currently active
  slot's stored content must stay byte-identical to its manifest image;
* after a confirmed switch the new slot is the unique CONFIRMED slot whose
  measured digest equals its manifest digest at the new generation;
* a previously confirmed slot whose durable image was later replaced/corrupted
  is safely identified and refused on every reopen, without ever falling back
  to a SUPERSEDED version.
"""
from __future__ import annotations

import os

from tests.conftest import make_device


def _images(client, dev="dev-1"):
    return client.get(f"/api/devices/{dev}/slot-images").json()


def _baseline_active_digest(client):
    rep = _images(client)
    a = rep["slots"]["A"]
    assert a["status"] == "CONFIRMED"
    assert a["digest_matches"] is True
    assert a["content_complete"] is True
    return a["manifest_digest"], a["measured_digest"]


def test_active_slot_bytes_survive_candidate_write_interruption(client):
    make_device(client)
    manifest_d, measured_d = _baseline_active_digest(client)

    client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1", "fault_point": "candidate_write",
    })
    rec = client.post("/api/devices/dev-1/power-on").json()["recovery"]
    assert rec["active_slot"] == "A"

    img = _images(client)
    a = img["slots"]["A"]
    # The inactive-slot staging must never have touched the active image.
    assert a["status"] == "CONFIRMED"
    assert a["measured_digest"] == measured_d == manifest_d
    assert a["digest_matches"] is True
    assert a["content_complete"] is True
    b = img["slots"]["B"]
    assert b["stored_size"] < b["manifest_size"]
    assert b["digest_matches"] is False


def test_active_slot_bytes_survive_digest_check_interruption(client):
    make_device(client)
    manifest_d, measured_d = _baseline_active_digest(client)

    client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1", "fault_point": "digest_check",
    })
    rec = client.post("/api/devices/dev-1/power-on").json()["recovery"]
    assert rec["active_slot"] == "A"

    img = _images(client)
    a = img["slots"]["A"]
    assert a["measured_digest"] == measured_d == manifest_d
    assert a["digest_matches"] is True and a["content_complete"] is True
    b = img["slots"]["B"]
    # Candidate bytes are fully present in B but unproven; A is untouched.
    assert b["stored_size"] == b["manifest_size"]
    assert b["status"] == "CANDIDATE"


def test_active_slot_bytes_survive_verified_but_unconfirmed_candidate(client):
    make_device(client)
    manifest_d, measured_d = _baseline_active_digest(client)

    staged = client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1",
    })
    assert staged.json()["outcome"] == "staged"
    # Verified candidate, still unconfirmed -> explicit power cycle.
    client.post("/api/devices/dev-1/power-off")
    rec = client.post("/api/devices/dev-1/power-on").json()["recovery"]
    assert rec["active_slot"] == "A"
    assert next(d for d in rec["diagnoses"] if d["slot"] == "B")["reason"] \
        == "unconfirmed_candidate"

    img = _images(client)
    a = img["slots"]["A"]
    assert a["status"] == "CONFIRMED"
    assert a["measured_digest"] == measured_d == manifest_d
    assert a["digest_matches"] is True
    # And a power cut right at confirm keeps the active image intact too.
    client.post("/api/devices/dev-1/confirm", json={"fault_point": "confirm_switch"})
    rec = client.post("/api/devices/dev-1/power-on").json()["recovery"]
    assert rec["active_slot"] == "A"
    img = _images(client)
    assert img["slots"]["A"]["measured_digest"] == manifest_d
    assert img["slots"]["A"]["digest_matches"] is True


def test_confirmed_switch_content_digest_and_generation_consistent(client):
    make_device(client)
    client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1",
    })
    sw = client.post("/api/devices/dev-1/confirm", json={}).json()
    assert sw["outcome"] == "switched" and sw["generation"] == 2

    img = _images(client)
    assert img["active_slot"] == "B" and img["generation"] == 2
    b = img["slots"]["B"]
    assert b["status"] == "CONFIRMED"
    assert b["digest_matches"] is True and b["content_complete"] is True
    assert b["confirmed_generation"] == 2
    assert img["slots"]["A"]["status"] == "SUPERSEDED"

    # Survives reopen with identical content digest and confirmation generation.
    client.post("/api/devices/dev-1/power-off")
    client.post("/api/devices/dev-1/power-on")
    img2 = _images(client)
    b2 = img2["slots"]["B"]
    assert b2["measured_digest"] == b["measured_digest"] == b["manifest_digest"]
    assert b2["confirmed_generation"] == 2 == img2["generation"]
    assert img2["slots"]["A"]["status"] == "SUPERSEDED"


def test_next_upgrade_keeps_current_active_slot_until_switch(client):
    make_device(client)
    client.post("/api/devices/dev-1/candidate", json={
        "version": "2.0.0", "request_id": "r1",
    })
    client.post("/api/devices/dev-1/confirm", json={})
    active_before = _images(client)["slots"]["B"]["manifest_digest"]

    # Second upgrade targets slot A while B is the healthy active slot. A cut
    # mid staging must leave B's bytes untouched and still boot B.
    client.post("/api/devices/dev-1/candidate", json={
        "version": "3.0.0", "request_id": "r2", "fault_point": "candidate_write",
    })
    rec = client.post("/api/devices/dev-1/power-on").json()["recovery"]
    assert rec["active_slot"] == "B" and rec["generation"] == 2
    img = _images(client)
    assert img["slots"]["B"]["status"] == "CONFIRMED"
    assert img["slots"]["B"]["measured_digest"] == active_before
    assert img["slots"]["B"]["digest_matches"] is True
    assert img["slots"]["A"]["stored_size"] < img["slots"]["A"]["manifest_size"]


def test_previously_affected_active_image_is_refused_without_rollback(client):
    make_device(client)
    # Take the durable store through the same DB file and replace the active
    # slot's bytes -- emulating data already affected by the old overwrite bug.
    from backend.store import Store

    store = Store(os.environ["DATA_PATH"])
    try:
        with store.transaction() as conn:
            store.write_blob(conn, "dev-1", "A", b"partial-or-foreign-write")
    finally:
        store.close()

    out = client.post("/api/devices/dev-1/power-on").json()
    rec = out["recovery"]
    assert rec["active_slot"] is None  # refuses to boot the mismatched image
    diag_a = next(d for d in rec["diagnoses"] if d["slot"] == "A")
    assert diag_a["reason"] == "active_image_corrupt"
    assert out["device"]["slots"]["A"]["status"] == "REJECTED"

    # Persisted refusal: still rejected on the next open.
    client.post("/api/devices/dev-1/power-off")
    again = client.post("/api/devices/dev-1/power-on").json()["recovery"]
    assert again["active_slot"] is None
    assert next(d for d in again["diagnoses"] if d["slot"] == "A")["reason"] \
        == "digest_mismatch"

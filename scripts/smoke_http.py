#!/usr/bin/env python3
"""HTTP smoke test for power-loss recovery and concurrent adjudication.

Talks to a *live* uvicorn server (plain stdlib only) and exits non-zero on the
first failed expectation.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
checks = 0


def call(method: str, path: str, body=None, expect=None):
    global checks
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status, payload = resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        status, payload = exc.code, json.loads(exc.read().decode())
    if expect is not None:
        checks += 1
        assert status == expect, f"{method} {path}: expected {expect}, got {status} {payload}"
    return status, payload


def assert_that(cond, text):
    global checks
    checks += 1
    assert cond, text


def power_cycle(dev, expected_slot, expected_gen):
    call("POST", f"/api/devices/{dev}/power-off", {}, 200)
    _, body = call("POST", f"/api/devices/{dev}/power-on", {}, 200)
    rec, d = body["recovery"], body["device"]
    assert_that(rec["active_slot"] == expected_slot,
                f"{dev}: recovery selected {rec['active_slot']}, want {expected_slot}")
    assert_that(rec["generation"] == expected_gen,
                f"{dev}: generation {rec['generation']}, want {expected_gen}")
    return rec, d


def images(dev):
    _, body = call("GET", f"/api/devices/{dev}/slot-images", expect=200)
    return body


def assert_active_untouched(img, slot, expected_digest):
    s = img["slots"][slot]
    assert_that(s["status"] == "CONFIRMED", f"{slot} must stay CONFIRMED")
    assert_that(s["measured_digest"] == expected_digest,
                f"{slot} actual image digest changed to {s['measured_digest']}")
    assert_that(s["digest_matches"] is True, f"{slot} digest no longer matches manifest")
    assert_that(s["content_complete"] is True, f"{slot} image no longer complete")


def clobber_blob(db_path, dev, slot, content):
    """Overwrite one slot's durable bytes out-of-band.

    Emulates data that was *already affected* before this open: the manifest
    still claims the confirmed digest while the stored image is foreign.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "UPDATE blobs SET content=? WHERE device_id=? AND slot=?",
            (content, dev, slot),
        )
        assert conn.total_changes >= 1, "blob row to clobber not found"
        conn.commit()
    finally:
        conn.close()


def check_previously_affected_data():
    db_path = os.environ.get("DATA_PATH")
    call("POST", "/api/devices", {"device_id": "s3", "version": "1.0.0"}, 201)
    assert db_path, "DATA_PATH must point at the server's SQLite file"
    call("POST", "/api/devices/s3/power-off", {}, 200)
    # While off, damage the active slot's actual image after it was confirmed.
    clobber_blob(db_path, "s3", "A", b"foreign-or-partial-candidate-bytes")

    _, body = call("POST", "/api/devices/s3/power-on", {}, 200)
    rec, d = body["recovery"], body["device"]
    assert_that(body["outcome"] == "unbootable", "mismatched active image must refuse boot")
    assert_that(rec["active_slot"] is None, "no slot may be selected from a corrupt active image")
    diag_a = next(x for x in rec["diagnoses"] if x["slot"] == "A")
    assert_that(diag_a["reason"] == "active_image_corrupt",
                f"want active_image_corrupt, got {diag_a['reason']}")
    assert_that(d["slots"]["A"]["status"] == "REJECTED", "damaged confirmed slot must be REJECTED")
    img = images("s3")["slots"]["A"]
    assert_that(img["digest_matches"] is False, "measured digest must not match manifest")

    # Persisted refusal: a later open still rejects it (never silently boots).
    call("POST", "/api/devices/s3/power-off", {}, 200)
    _, body = call("POST", "/api/devices/s3/power-on", {}, 200)
    assert_that(body["recovery"]["active_slot"] is None, "reopen must still refuse boot")
    again = next(x for x in body["recovery"]["diagnoses"] if x["slot"] == "A")
    assert_that(again["reason"] == "digest_mismatch", "reopen must keep digest_mismatch refusal")


def main() -> int:
    # --- health + built page ------------------------------------------------
    _, health = call("GET", "/api/health", expect=200)
    assert_that(health["status"] == "ok", "health status not ok")
    with urllib.request.urlopen(BASE + "/", timeout=10) as resp:
        html = resp.read().decode()
    assert_that("轨道载荷" in html and "/src/main.js" not in html,
                "served page is not the built bundle")

    # --- power-loss scenarios on device s1 ----------------------------------
    call("POST", "/api/devices", {"device_id": "s1", "version": "1.0.0"}, 201)
    # Record the active slot's actual persisted image digest before staging.
    active_digest = images("s1")["slots"]["A"]["measured_digest"]

    # fault 1: cut during candidate write -> incomplete, old slot boots
    _, b = call("POST", "/api/devices/s1/candidate",
                {"version": "2.0.0", "request_id": "w1", "fault_point": "candidate_write"}, 200)
    assert_that(b["outcome"] == "power_cut" and b["fault_point"] == "candidate_write",
                "candidate_write fault not injected")
    rec, d = power_cycle("s1", "A", 1)
    diag = {x["slot"]: x for x in rec["diagnoses"]}
    assert_that(diag["B"]["reason"] == "incomplete_write", "want incomplete_write diagnosis")
    assert_that(d["slots"]["B"]["written"] < d["slots"]["B"]["size"], "partial write expected")
    assert_active_untouched(images("s1"), "A", active_digest)

    # fault 2: cut during digest check -> fully written but unproven
    _, b = call("POST", "/api/devices/s1/candidate",
                {"version": "2.0.0", "request_id": "w2", "fault_point": "digest_check"}, 200)
    assert_that(b["outcome"] == "power_cut", "digest_check fault not injected")
    rec, d = power_cycle("s1", "A", 1)
    diag = {x["slot"]: x for x in rec["diagnoses"]}
    assert_that(diag["B"]["reason"] == "unverified_candidate", "want unverified_candidate")
    assert_that(d["slots"]["B"]["actual_digest"] is None, "no verdict may be committed")
    assert_active_untouched(images("s1"), "A", active_digest)

    # corrupt candidate -> REJECTED, evidence kept, never boots
    _, b = call("POST", "/api/devices/s1/candidate",
                {"version": "2.0.0", "request_id": "w3", "corrupt": True}, 200)
    assert_that(b["outcome"] == "verification_failed", "corrupt candidate must fail verification")
    assert_that(b["claimed_digest"] != b["actual_digest"], "digests must differ")
    rec, d = power_cycle("s1", "A", 1)
    assert_that(d["slots"]["B"]["status"] == "REJECTED", "corrupt slot must stay REJECTED")
    assert_active_untouched(images("s1"), "A", active_digest)

    # normal stage, fault 3: cut at confirm -> old version, candidate unconfirmed
    call("POST", "/api/devices/s1/candidate",
         {"version": "2.0.0", "request_id": "w4"}, 200)
    _, b = call("POST", "/api/devices/s1/confirm", {"fault_point": "confirm_switch"}, 200)
    assert_that(b["outcome"] == "power_cut", "confirm_switch fault not injected")
    rec, d = power_cycle("s1", "A", 1)
    assert_that(d["slots"]["B"]["status"] == "VERIFIED", "candidate remains VERIFIED")
    assert_active_untouched(images("s1"), "A", active_digest)

    # commit for real -> B active, generation 2, no rollback after reopen
    _, b = call("POST", "/api/devices/s1/confirm", {}, 200)
    assert_that(b["outcome"] == "switched" and b["generation"] == 2, "switch not committed")
    digest_b = b["device"]["slots"]["B"]["digest"]
    rec, d = power_cycle("s1", "B", 2)
    assert_that(d["slots"]["B"]["version"] == "2.0.0", "reopen version mismatch")
    assert_that(d["slots"]["B"]["digest"] == digest_b, "reopen digest mismatch")
    assert_that(d["slots"]["A"]["status"] == "SUPERSEDED", "old slot must be SUPERSEDED")
    assert_that(any("防回退" in x for x in rec["rationale"]), "no-rollback rationale missing")
    new_img = images("s1")["slots"]["B"]
    assert_that(new_img["status"] == "CONFIRMED" and new_img["digest_matches"] is True,
                "new active slot must be CONFIRMED with measured digest matching manifest")
    assert_that(new_img["measured_digest"] == digest_b
                and new_img["confirmed_generation"] == 2,
                "new active content/digest/generation inconsistent after reopen")

    # --- previously affected durable data is safely refused (no rollback) ----
    check_previously_affected_data()
    # --- concurrent different candidates on device s2 -----------------------
    call("POST", "/api/devices", {"device_id": "s2", "version": "1.0.0"}, 201)
    barrier = threading.Barrier(2)
    outcomes = []

    def concurrent_submit(version, request_id):
        barrier.wait()
        outcomes.append(call("POST", "/api/devices/s2/candidate",
                             {"version": version, "request_id": request_id}))

    t1 = threading.Thread(target=concurrent_submit, args=("2.0.0", "page-one"))
    t2 = threading.Thread(target=concurrent_submit, args=("2.1.0", "page-two"))
    t1.start(); t2.start(); t1.join(); t2.join()
    statuses = sorted(code for code, _ in outcomes)
    assert_that(statuses == [200, 409], f"concurrent statuses {statuses}")
    winner = next((p for code, p in outcomes if code == 200), None)
    loser = next((p for code, p in outcomes if code == 409), None)
    assert_that(loser["error"]["code"] == "upgrade_conflict", "loser must be upgrade_conflict")
    assert_that(loser["error"]["holder_request"] == winner["request_id"], "holder mismatch")
    _, d = call("GET", "/api/devices/s2", expect=200)
    d = d["device"]
    assert_that(d["active_slot"] == "A" and d["slots"]["A"]["version"] == "1.0.0",
                "loser must not rewrite active version")

    # winner confirms; reopen shows identical slot/version/digest/generation
    _, b = call("POST", "/api/devices/s2/confirm", {}, 200)
    assert_that(b["generation"] == 2 and b["active_slot"] == "B", "winner switch failed")
    before = b["device"]
    rec, after = power_cycle("s2", "B", 2)
    for field in ("status", "version", "digest", "confirmed_generation"):
        assert_that(after["slots"]["B"][field] == before["slots"]["B"][field],
                    f"field {field} changed across reopen")
    assert_that(after["generation"] == before["generation"], "generation changed across reopen")

    print(f"SMOKE OK: {checks} HTTP assertions passed "
          f"(power-loss x3, corrupt candidate, concurrent 409, reopen consistency)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

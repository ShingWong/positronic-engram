# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================

"""Schema-driven ingestion: every field reaches the engine, both axes land.

The defect this exists for: the old harness flattened each record to one
string, so identity fields never reached the engine and every object-layer
claim was void. These tests assert on absolute structure -- named objects,
named relations, real fields in features -- because "it ingested N events"
would pass while the defect was still present.
"""

from __future__ import annotations

from memeng.engine import MemoryEngine
from memeng.ingest import CARRIED, SESSION_KIND, ingest_records
from memeng.store import SQLiteStore

SCHEMA = {
    "software": "x",
    "identity_key": ["ident", "pid"],
    "time_field": "stamp",
    "record_layers": {},
    "classes": [],
    "message_key": {
        "name": "queue_id",
        "field": "message",
        "pattern": r"\b([0-9A-F]{10,13})\b",
    },
}

RECS = [
    {"ident": "postfix/smtpd", "pid": "11", "stamp": "2026-01-01T00:00:00",
     "message": "connect from x"},
    {"ident": "postfix/smtpd", "pid": "11", "stamp": "2026-01-01T00:00:01",
     "message": "2C79E142DE806: client=y"},
    {"ident": "postfix/qmgr", "pid": "22", "stamp": "2026-01-01T00:00:02",
     "message": "2C79E142DE806: from=<a@b.c>"},
    {"ident": "postfix/qmgr", "pid": "22", "stamp": "not a timestamp",
     "message": "2C79E142DE806: removed"},
]


def _engine():
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s, {"entity_extraction": False})
    e.init_database()
    e.register_domain("d", retention_profile="balanced")
    e.attach_stream("d:s", "d")
    return e


def test_identity_objects_converge_across_records():
    e = _engine()
    rep = ingest_records(e, SCHEMA, RECS, stream="d:s")
    assert rep.ingested == 3
    assert rep.skipped_no_clock == 1
    assert rep.sessions == 2          # two (ident, pid) pairs, not 3 lines
    assert rep.messages == 1          # one queue id across two records
    rows = e.store.conn.execute(
        "SELECT canonical_name FROM object WHERE kind=?",
        (SESSION_KIND,)).fetchall()
    assert sorted(r[0] for r in rows) == [
        "postfix/qmgr:22", "postfix/smtpd:11"]


def test_session_carried_message_relation_is_written():
    e = _engine()
    rep = ingest_records(e, SCHEMA, RECS, stream="d:s")
    assert rep.relations == 2
    edges = e.store.conn.execute(
        "SELECT predicate, provenance FROM object_relation").fetchall()
    assert all(r[0] == CARRIED and r[1] == "schema" for r in edges)
    # The message is reachable from either session that handled it.
    sess = e.store.conn.execute(
        "SELECT id FROM object WHERE canonical_name=?",
        ("postfix/smtpd:11",)).fetchone()[0]
    g = e.store.object_graph(sess, depth=2)
    assert any(n.get("canonical_name") == "queue:2C79E142DE806"
               for n in g["nodes"])


def test_every_field_reaches_features_not_one_string():
    e = _engine()
    ingest_records(e, SCHEMA, RECS[:1], stream="d:s")
    feats = e.store.conn.execute("SELECT features_json FROM episode").fetchone()[0]
    import json
    f = json.loads(feats)
    assert f["ident"] == "postfix/smtpd"
    assert f["pid"] == "11"
    assert f["message"] == "connect from x"
    assert f["identity_canonical"] == "postfix/smtpd:11"
    assert f["message_ids"] == []


def test_arousal_is_inverse_frequency_not_zero():
    e = _engine()
    rep = ingest_records(
        e, SCHEMA, RECS[:3], stream="d:s",
        classify=lambda r: "connect" if r["message"] == "connect from x"
        else "queue")
    assert rep.arousal_source.startswith("inverse")
    import json
    rows = e.store.conn.execute(
        "SELECT features_json FROM episode ORDER BY wall").fetchall()
    a = [json.loads(r[0])["arousal"] for r in rows]
    # queue class has 2 members, connect has 1: the common class scores lower.
    assert a[0] == 1.0 - 1 / 2 and a[1] == 0.0 and a[2] == 0.0


def test_without_classifier_the_report_says_so():
    e = _engine()
    rep = ingest_records(e, SCHEMA, RECS[:1], stream="d:s")
    assert rep.arousal_source == "unset"
    assert any("no classifier" in n for n in rep.notes)


def test_unattached_stream_is_an_error_not_silent_default():
    e = _engine()
    try:
        ingest_records(e, SCHEMA, RECS[:1], stream="nope:stream")
    except ValueError as ex:
        assert "not attached" in str(ex)
    else:
        raise AssertionError("ingested into a stream that does not exist")


def test_bad_pattern_is_an_error_not_zero_ids():
    e = _engine()
    bad = dict(SCHEMA, message_key={"name": "q", "field": "message",
                                    "pattern": r"(a)(b)"})
    try:
        ingest_records(e, bad, RECS[:1], stream="d:s")
    except ValueError as ex:
        assert "exactly one" in str(ex)
    else:
        raise AssertionError("two-group pattern accepted silently")

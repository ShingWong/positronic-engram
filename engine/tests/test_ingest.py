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


def _did(e):
    return int(e.store.get_stream("d:s")["domain_id"])


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


# ------------------------------------------------- participants

def _ev(stream="d:s", persons=None, objects=None, text="hello world"):
    from datetime import datetime, timezone

    from memeng.models import Event
    return Event(stream=stream, kind="message",
                 wall=datetime(2026, 1, 1, tzinfo=timezone.utc),
                 persons=persons or [], objects=objects or [],
                 features={"subject_norm": text, "body_text": text,
                           "arousal": 0.0})


def test_object_salience_feeds_gate_gain():
    e = _engine()
    # A salient session pulls a mediocre episode over the line; without the
    # object the same text and no people scores gain 0.5.
    oid = e.store.get_or_create_object(
        domain_id=_did(e), kind="session", canonical_name="s:1",
        wall=_ev().wall, tau=0.0)
    e.store.conn.execute("UPDATE object SET salience=1.0 WHERE id=?", (oid,))
    e.store._commit()
    v_obj = e.new_event(_ev(objects=[("session", "s:1")]))
    v_plain = e.new_event(_ev())
    assert v_obj.verdict.score >= v_plain.verdict.score


def test_encode_links_sightings_and_credits_salience():
    e = _engine()
    e.new_event(_ev(objects=[("session", "s:9")]))
    o = e.store.find_object(domain_id=_did(e), kind="session", canonical_name="s:9")
    assert o is not None and float(o["salience"]) > 0.5
    links = e.store.conn.execute(
        "SELECT * FROM object_sighting WHERE object_id=?",
        (o["id"],)).fetchall()
    assert len(links) == 1


def test_dedup_separates_same_text_on_different_sessions():
    # Stored directly to isolate prune from the gate: the gate would reject
    # the second identical text on novelty before prune ever saw it, which
    # is correct gate behavior and a different test.
    import uuid

    from memeng.models import Provenance
    from memeng.store import EpisodeRecord, Tier
    e = _engine()
    for sess in ("s:A", "s:B"):
        e.store.insert_episode(EpisodeRecord(
            id=uuid.uuid4(), stream="d:s", kind="message",
            wall=_ev().wall, mono=1, tau=1.0, persons=[],
            subject_norm="hello world", salience=0.5, tier=Tier.NORMAL,
            strength=1.0, provenance=Provenance.WITNESSED,
            fuzz_lo=None, fuzz_hi=None, precision_src="exact",
            is_anchor=False,
            features={"subject_norm": "hello world",
                      "body_text": "hello world",
                      "object_refs": [f"session:{sess}"]}), _did(e))
    # prune takes tau_now positionally; the domain goes by keyword. Passing
    # "d" positionally once dated the whole sweep to the string "d".
    rep = e.prune(domain="d")
    assert rep.deduped == 0
    n = e.store.conn.execute(
        "SELECT COUNT(*) FROM episode WHERE level='event'").fetchone()[0]
    assert n == 2


def test_fusion_boosts_shared_message():
    e = _engine()
    # Different wording per record: identical text would be gate-rejected on
    # novelty before fusion ever saw it, which is correct gate behavior and a
    # different test. Same message object, different words.
    r1 = e.new_event(_ev(objects=[("session", "s:1"), ("message", "queue:Q1")],
                      text="delivery completed successfully"))
    r2 = e.new_event(_ev(objects=[("session", "s:2"), ("message", "queue:Q1")],
                      text="bounce notification for the morning batch"))
    e.new_event(_ev(text="unrelated weather discussion today"))
    out = e.activate({"text": "delivery completed",
                      "objects": ["message:queue:Q1"]})
    by_id = {str(x["episode_id"]): x for x in out}
    q1 = [str(r1.episode_id), str(r2.episode_id)]
    assert all(q in by_id for q in q1)
    assert all(by_id[q]["participant_boost"] == 1.3 for q in q1)
    # Without the identity cue, r2 shares no text with the query and must
    # NOT surface: this is what proves the object channel did the work,
    # rather than the test passing on lexical overlap alone.
    text_only = {str(x["episode_id"]) for x in
                 e.activate({"text": "delivery completed"})}
    assert str(r2.episode_id) not in text_only


def test_prune_decays_object_salience():
    e = _engine()
    e.new_event(_ev(objects=[("session", "s:7")]))
    o = e.store.find_object(domain_id=_did(e), kind="session", canonical_name="s:7")
    before = float(o["salience"])
    assert before > 0.5
    e.store.decay_object_salience(rate=0.5)
    after = float(e.store.find_object(
        domain_id=_did(e), kind="session",
        canonical_name="s:7")["salience"])
    assert after == 0.5 + (before - 0.5) * 0.5


# ------------------------------------------------- range-and-key recall

def test_recall_object_returns_a_message_life_in_order():
    # Maximally distinct wording per record: near-identical texts would be
    # gate-rejected on novelty before recall ever saw them, which is correct
    # gate behavior and not what this test is about.
    e = _engine()
    for objs, txt in (([("session", "s:1"), ("message", "queue:Q9")],
                       "queue manager accepted the parcel"),
                      ([("session", "s:2"), ("message", "queue:Q9")],
                       "antivirus verdict arrived clean"),
                      ([("session", "s:2"), ("message", "queue:Q9")],
                       "mailbox write finished at dawn")):
        r = e.new_event(_ev(objects=objs, text=txt))
        assert r.verdict.encoded, txt
    e.new_event(_ev(text="unrelated weather discussion today"))
    # The ingest path writes carried edges; here the event path does not, so
    # write the two the scenario needs the way ingest would.
    did = _did(e)
    s1 = e.store.find_object(domain_id=did, kind="session",
                             canonical_name="s:1")["id"]
    s2 = e.store.find_object(domain_id=did, kind="session",
                             canonical_name="s:2")["id"]
    m = e.store.find_object(domain_id=did, kind="message",
                            canonical_name="queue:Q9")["id"]
    e.store.add_object_relation(subject_id=s1, predicate="carried",
                                object_id=m, provenance="test")
    e.store.add_object_relation(subject_id=s2, predicate="carried",
                                object_id=m, provenance="test")
    out = e.recall_object("message", "queue:Q9")
    assert out["found"] is True
    assert len(out["episodes"]) == 3
    walls = [x["wall"] for x in out["episodes"]]
    assert walls == sorted(walls)
    assert sorted(out["carried_by"]) == ["s:1", "s:2"]
    assert all("message:queue:Q9" in x["object_refs"] for x in out["episodes"])


def test_recall_object_window_bounds_are_honored():
    e = _engine()
    for day, txt in ((1, "zebra alpha embarkation"), (2, "quarry beta manifest"),
                      (3, "fjord gamma departure")):
        from datetime import datetime, timezone

        from memeng.models import Event
        r = e.new_event(Event(
            stream="d:s", kind="message",
            wall=datetime(2026, 1, day, tzinfo=timezone.utc),
            objects=[("session", "s:W")],
            features={"subject_norm": txt, "body_text": txt, "arousal": 0.0}))
        assert r.verdict.encoded, txt
    out = e.recall_object("session", "s:W", since="2026-01-02T00:00:00+00:00",
                          until="2026-01-02T23:59:59+00:00")
    assert [x["subject"] for x in out["episodes"]] == ["quarry beta manifest"]


def test_recall_object_unknown_identity_is_empty_not_error():
    e = _engine()
    out = e.recall_object("message", "queue:DOESNOTEXIST")
    assert out == {"kind": "message", "canonical": "queue:DOESNOTEXIST",
                   "found": False, "episodes": [], "carried_by": [],
                   "carried": []}


def test_episodes_in_range_reads_a_window_as_narrative():
    e = _engine()
    for day, txt in ((1, "harbor opens at sunrise"), (2, "cargo sways midday"),
                      (3, "gates close after dusk")):
        from datetime import datetime, timezone

        from memeng.models import Event
        r = e.new_event(Event(
            stream="d:s", kind="message",
            wall=datetime(2026, 1, day, tzinfo=timezone.utc),
            features={"subject_norm": txt, "body_text": txt, "arousal": 0.0}))
        assert r.verdict.encoded, txt
    ids = e.store.episodes_in_range(stream="d:s", since="2026-01-01T00:00:00+00:00",
                                    until="2026-01-02T12:00:00+00:00")
    assert len(ids) == 2
    walls = [e.store.get_episode(i).wall.isoformat() for i in ids]
    assert walls == sorted(walls)

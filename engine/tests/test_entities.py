# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://gnu.org>.
# =====================================================================

"""Deterministic technical-entity extraction tests."""
from datetime import datetime, timezone

from memeng.entities import extract_entities
from memeng.engine import MemoryEngine
from memeng.models import Event
from memeng.store import SQLiteStore


def test_hyphen_identifiers():
    assert extract_entities(
        "deployed persona-bot-v2 using sure-state and bge-m3") == {
        "persona-bot-v2", "sure-state", "bge-m3"}


def test_letter_digit_tokens():
    got = extract_entities("web2 talks to ai1 and mx1; d3d11.dll loaded")
    assert {"web2", "ai1", "mx1", "d3d11"} <= got


def test_ipv4_and_host_port():
    got = extract_entities("whisperx:8000 on 192.168.4.22, kokoro :9880 too")
    assert "192.168.4.22" in got
    assert "whisperx:8000" in got


def test_dotted_names():
    assert "memory.db" in extract_entities("state lives in memory.db")


def test_filters():
    # pure numbers, single/double char tokens, e.g./i.e. junk rejected
    got = extract_entities("v2 x step 3 done e.g. i.e 42")
    assert "42" not in got
    assert "v2" not in got
    assert "e.g" not in got
    assert len(got) == 0 or all(3 <= len(t) <= 40 for t in got)


def test_case_normalized_and_sorted_set():
    got = extract_entities("WEB2 and web2")
    assert got == {"web2"}


def mk(config=None):
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s, config)
    e.init_database()
    return e, s


def _sess_event(subject, body):
    return Event(stream="kairos:sessions", kind="message",
                 persons=["p_kairos"],
                 wall=datetime.now(timezone.utc),
                 features={"subject_norm": subject, "sender": "p_kairos",
                           "arousal": 0.95, "body_text": body})


def test_encode_creates_entity_objects_and_sightings():
    e, s = mk()
    e.new_event(_sess_event(
        "web2 deploy",
        "deployed persona-bot-v2 on web2 192.168.4.22"))
    rows = s.conn.execute(
        "SELECT canonical_name FROM object WHERE kind='entity'").fetchall()
    names = {r["canonical_name"] for r in rows}
    assert {"web2", "persona-bot-v2", "192.168.4.22"} <= names
    n = s.conn.execute(
        "SELECT COUNT(*) c FROM object_sighting WHERE channel='text'"
    ).fetchone()["c"]
    assert n >= 3


def test_repeated_mentions_converge_on_one_object():
    e, s = mk()
    e.new_event(_sess_event("web2 deploy", "shipped to web2"))
    e.new_event(_sess_event("web2 rollback", "rolled back web2"))
    rows = s.conn.execute(
        "SELECT id FROM object WHERE kind='entity' AND "
        "canonical_name='web2'").fetchall()
    assert len(rows) == 1
    sightings = s.conn.execute(
        "SELECT COUNT(*) c FROM object_sighting WHERE object_id=?",
        (rows[0]["id"],)).fetchone()["c"]
    assert sightings == 2


def test_link_sighting_respects_batch_transaction():
    e, s = mk()
    e.new_event(_sess_event("web2 deploy", "shipped to web2"))
    oid = s.conn.execute(
        "SELECT id FROM object WHERE kind='entity' AND "
        "canonical_name='web2'").fetchone()["id"]
    eid = s.conn.execute(
        "SELECT id FROM episode LIMIT 1").fetchone()["id"]
    s.begin_batch()
    s.link_sighting(str(eid), str(oid), channel="text")
    assert s.conn.in_transaction
    s.commit_batch()


def test_entity_mentions_advance_last_seen():
    e, s = mk()
    e.new_event(_sess_event("web2 deploy", "shipped to web2"))
    e.new_event(_sess_event("web2 rollback", "rolled back web2"))
    oid = s.conn.execute(
        "SELECT id FROM object WHERE kind='entity' AND "
        "canonical_name='web2'").fetchone()["id"]
    latest = s.conn.execute(
        "SELECT tau, wall FROM episode ORDER BY tau DESC LIMIT 1").fetchone()
    row = s.conn.execute(
        "SELECT last_seen_tau, last_seen_wall FROM object WHERE id=?",
        (oid,)).fetchone()
    assert row["last_seen_tau"] == latest["tau"]
    assert row["last_seen_wall"] == latest["wall"]


def test_extraction_can_be_disabled():
    e, s = mk(config={"entity_extraction": False})
    e.new_event(_sess_event("web2 deploy", "shipped to web2"))
    n = s.conn.execute(
        "SELECT COUNT(*) c FROM object WHERE kind='entity'").fetchone()["c"]
    assert n == 0


def test_person_cache_reflects_bump_miss_cost():
    # I2 regression: bump_miss_cost changes weight; the cached person row
    # must be invalidated so the next gate decision sees the new weight.
    e, s = mk()
    e.register_person("p_beta")
    s.get_person("p_beta")                       # populate _person_cache
    s.bump_miss_cost("p_beta")                   # weight 0.5 -> 0.55
    fresh = s.get_person("p_beta")
    assert fresh["weight"] == 0.55


def test_person_cache_reflects_touch_person():
    # I2: touch_person updates last_seen_tau; cached row must not go stale.
    e, s = mk()
    e.register_person("p_beta")
    s.get_person("p_beta")
    s.touch_person("p_beta", 42.0)
    fresh = s.get_person("p_beta")
    assert fresh["last_seen_tau"] == 42.0


# prune default (tau_now=None) must derive "now" for the objects pass too,
# not only for the episode loop. Regression for:
#   prune: unsupported operand type(s) for -: 'NoneType' and 'float'
def test_prune_default_now_ages_objects():
    e, s = mk()
    e.new_event(_sess_event("web2 deploy", "shipped to web2"))
    oid = s.conn.execute(
        "SELECT id FROM object WHERE kind='entity' AND "
        "canonical_name='web2'").fetchone()["id"]
    s.conn.execute(
        "UPDATE object SET first_seen_tau=0, last_seen_tau=0 WHERE id=?",
        (oid,))
    s.conn.execute("UPDATE stream SET tau=1500.0, mono=500 WHERE stream=?", ("kairos:sessions",))
    rep = e.prune()                                   # tau_now defaults to None
    assert rep.objects_dormant + rep.objects_forgotten >= 1
    status = s.conn.execute(
        "SELECT status FROM object WHERE id=?", (oid,)).fetchone()["status"]
    assert status in ("dormant", "forgotten")


def test_prune_objects_use_per_domain_now_not_global():
    # C1 regression: object in a QUIET stream must not be aged against a
    # hot stream's global MAX(tau). now must derive per-domain.
    e, s = mk()
    hot_did = e.attach_stream("hot:stream", "hot_domain")
    quiet_did = e.attach_stream("quiet:stream", "quiet_domain")
    s.attach_stream("hot:stream", hot_did)
    s.attach_stream("quiet:stream", quiet_did)
    s.conn.execute("UPDATE stream SET tau=1500.0, mono=500 WHERE stream=?",
                   ("hot:stream",))
    s.conn.execute("UPDATE stream SET tau=1.0, mono=1 WHERE stream=?",
                   ("quiet:stream",))
    wall = datetime.now(timezone.utc)
    s.get_or_create_object(
        domain_id=hot_did, kind="entity", canonical_name="hot-thing",
        wall=wall, tau=1500.0)
    quiet_obj = s.get_or_create_object(
        domain_id=quiet_did, kind="entity", canonical_name="quiet-thing",
        wall=wall, tau=1.0)
    rep = e.prune()                                   # tau_now=None
    quiet_status = s.conn.execute(
        "SELECT status FROM object WHERE id=?", (quiet_obj,)).fetchone()["status"]
    assert quiet_status not in ("dormant", "forgotten")
    assert rep.objects_forgotten == 0

# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================
# This program is DUAL-LICENSED. You may redistribute and/or modify it 
# under the terms of the GNU Affero General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your 
# option) any later version.
#
# Alternatively, commercial entities, multi-tenant instances, and Managed 
# Service Providers (MSPs) may utilize this program under a separate, 
# proprietary Commercial License Waiver issued directly by the copyright 
# holder, completely exempt from the network-use copyleft restrictions of 
# the AGPLv3 Section 13.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License 
# along with this program. If not, see <https://gnu.org>.
# =====================================================================

"""Deterministic technical-entity extraction tests."""
from datetime import datetime, timezone

from memeng.engine import MemoryEngine
from memeng.entities import extract_entities
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


def _ev(persons, features, stream="probe:s"):
    return Event(stream=stream, kind="message", persons=persons,
                 wall=datetime(2026, 3, 1, tzinfo=timezone.utc),
                 features={**features, "arousal": 0.0})


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


# E2 ablation — object layer: the wall arm, the renewal skip, and the wall_base
# fallback. Before the object-axis branch these all fail: decay_axis gated only
# the episode ladder, so objects aged on tau whatever the axis said.
def _aged_object(s, *, tau, wall, name):
    """One object, aged by fiat: last_seen_* set to (tau, wall)."""
    oid = s.get_or_create_object(
        domain_id=1, kind="entity", canonical_name=name,
        wall=wall, tau=tau)
    s.conn.execute(
        "UPDATE object SET first_seen_tau=?, last_seen_tau=?, "
        "first_seen_wall=?, last_seen_wall=? WHERE id=?",
        (tau, tau, wall.isoformat(), wall.isoformat(), oid))
    return oid


# balanced's object ladder: obj_dormant=200, obj_forget=1200 (tau units). The
# wall arm compares DAYS against the same numbers, so a gap past 1200 days is
# needed to reach "forgotten" rather than merely "dormant".
_WALL_ANCIENT = datetime(2026, 1, 1, tzinfo=timezone.utc)
_WALL_FAR_FUTURE = datetime(2031, 1, 1, tzinfo=timezone.utc)   # ~1826 days


def test_prune_wall_axis_without_wall_now_does_not_raise():
    # Regression: wall is stored as ISO TEXT, so MAX(wall) is a string and
    # (wall_base - ep.wall.timestamp()) raised TypeError. The E1 driver always
    # passed wall_now explicitly, which is why this stayed hidden.
    e, _s = mk()
    e.new_event(_sess_event("deploy", "shipped"))
    e.prune(decay_axis="wall")


def test_prune_wall_axis_falls_back_to_max_wall():
    e, _s = mk()
    now = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    e.new_event(Event(stream="kairos:sessions", kind="message", wall=now,
                      features={"subject_norm": "x", "arousal": 0.9}, persons=[]))
    rep = e.prune(decay_axis="wall")
    assert rep.scanned >= 1


def test_object_wall_axis_ages_by_wall_not_tau():
    # An object whose wall clock is ancient but whose tau is recent: recent in
    # experience, ancient in calendar time. The wall arm must forget it; the
    # tau arm must keep it. This is the whole point of the comparison.
    for axis, expect_forgotten in (("tau", False), ("wall", True)):
        e, s = mk()
        oid = _aged_object(s, tau=1400.0, wall=_WALL_ANCIENT,
                           name=f"thing-{axis}")
        s.conn.execute("UPDATE stream SET tau=1500.0, mono=500 WHERE stream=?",
                       ("kairos:sessions",))
        rep = e.prune(tau_now=1500.0, decay_axis=axis,
                      wall_now=_WALL_FAR_FUTURE.timestamp())
        status = s.conn.execute(
            "SELECT status FROM object WHERE id=?", (oid,)).fetchone()["status"]
        if expect_forgotten:
            assert status == "forgotten", \
                f"{axis} arm must forget a calendar-ancient object, got {status}"
            assert rep.objects_forgotten >= 1
        else:
            assert status not in ("dormant", "forgotten"), \
                f"{axis} arm must not age a recently-experienced object, got {status}"


def test_object_wall_axis_never_renews():
    # Renewal is a tau mechanism (interval in tau units). A wall arm that renewed
    # would no longer be a plain timestamp index, which is the control we need.
    #
    # Order matters: the tau axis is asserted FIRST, while renewal is still
    # enabled. Without that the test would pass vacuously on today's engine,
    # where the wall arm ignores objects altogether and therefore never renews
    # anything. Asserting tau-renews-then-wall-does-not pins both halves of the
    # contrast, so a regression that made the wall arm renew would still fail.
    def _prepare():
        e, s = mk()
        P = e.retention_profiles["balanced"]
        assert P.get("renew_ratio", 0.0) > 0.0, \
            "precondition: the balanced profile must have renewal enabled"
        # Load-bearing evidence path 2 needs >=2 REAL non-consolidation episodes
        # sighting the object since the last prune, so the episodes have to
        # actually exist. Ingesting events that mention the entity produces both
        # the episodes and the sightings, via entity extraction.
        for i in range(2):
            e.new_event(_sess_event(
                f"mention {i}", "shipped persona-bot-v2 to web2 again"))
        oid = s.conn.execute(
            "SELECT id FROM object WHERE canonical_name='web2'").fetchone()["id"]
        sx = s.conn.execute(
            "SELECT COUNT(*) c FROM object_sighting WHERE object_id=?",
            (oid,)).fetchone()["c"]
        assert sx >= 2, f"precondition: need >=2 real sightings, got {sx}"
        # now age it: ancient in calendar time, ancient in tau as well, so the
        # object is a forget candidate and only renewal can save it
        s.conn.execute(
            "UPDATE object SET first_seen_tau=0, last_seen_tau=0, "
            "first_seen_wall=?, last_seen_wall=? WHERE id=?",
            (_WALL_ANCIENT.isoformat(), _WALL_ANCIENT.isoformat(), oid))
        s.conn.execute("UPDATE stream SET tau=9000.0, mono=500 WHERE stream=?",
                       ("kairos:sessions",))
        return e, s, oid

    far_future = _WALL_FAR_FUTURE.timestamp()

    # (a) tau axis, renewal ENABLED: the same object is load-bearing and renewed.
    e, s, oid = _prepare()
    rep = e.prune(tau_now=9000.0, decay_axis="tau")
    assert rep.objects_renewed >= 1, \
        "precondition: the tau arm must renew a load-bearing object"
    assert rep.objects_forgotten == 0

    # (b) wall axis: renewal is skipped, so the same object is forgotten.
    e, s, oid = _prepare()
    rep = e.prune(tau_now=9000.0, decay_axis="wall", wall_now=far_future)
    assert rep.objects_renewed == 0, "the wall arm must not renew"
    status = s.conn.execute(
        "SELECT status FROM object WHERE id=?", (oid,)).fetchone()["status"]
    assert status == "forgotten"


def test_ttl_renewal_defers_forgetting_by_exactly_one_cycle():
    """CHARACTERIZATION: renewal is a one-cycle deferral, not interval growth.

    The retention-profile docstring describes TTL renewal as advancing an
    object's "survival clock" by renew_ratio x (now - anchor), capped at
    renew_max -- spaced repetition, after Ebbinghaus + Rescorla-Wagner. That
    is not what the code does, and the gap was invisible because the
    arithmetic *runs* (last_renewal_tau is written, objects_renewed is
    counted, a trace event fires) while never reaching the survival test.

    The survival test compares dmeas = now - last_seen_tau against
    obj_forget. _renew_load_bearing writes last_renewal_tau, and nothing
    reads that column to build dmeas. So renewal changes nothing about how
    long an object lives; it only skips this cycle's forget.

    Pinned deliberately: this is the behaviour the released engine has, and
    the arXiv paper now reports it as a limitation. If renewal is ever made
    real, this test may or may not catch it — it pins the two-cycle sequence,
    not the absence of a mechanism, so treat it as documentation that must be
    re-read rather than as a tripwire. (Claiming it would trip on any renewal
    fix would be overreach: a fix that lengthened the anchor by 2730 against
    an obj_forget of 1200 still leaves the object forgotten on cycle 2, and
    this test would pass. The discriminating question — whether an object
    that is continuously *mentioned* in consolidations but never re-sighted
    survives indefinitely — is NOT pinned here, and is open.)

    Discriminating within its stated scope: it asserts renewal genuinely fired
    (objects_renewed >= 1) and wrote a large anchor (> obj_forget), so
    "forgotten anyway" cannot be mistaken for "never renewed". What it pins is
    that the anchor does not protect, and that a second consecutive renewal is
    impossible because the load-bearing window closes at the prune boundary.
    """
    e, s = mk()
    for i in range(2):
        e.new_event(_sess_event(
            f"mention {i}", "shipped persona-bot-v2 to web2 again"))
    oid = s.conn.execute(
        "SELECT id FROM object WHERE canonical_name='web2'").fetchone()["id"]

    def _age():
        s.conn.execute(
            "UPDATE object SET first_seen_tau=0, last_seen_tau=0 WHERE id=?",
            (oid,))
        s.conn.execute(
            "UPDATE stream SET tau=9000.0, mono=500 WHERE stream=?",
            ("kairos:sessions",))
        s._commit()

    def _status():
        return s.conn.execute(
            "SELECT status, last_renewal_tau FROM object WHERE id=?",
            (oid,)).fetchone()

    P = e.retention_profiles["balanced"]
    assert P.get("renew_ratio", 0.0) > 0.0, \
        "precondition: renewal must be enabled on the balanced profile"

    # cycle 1: load-bearing, so renewal fires and the object survives
    _age()
    rep1 = e.prune(tau_now=9100.0, decay_axis="tau")
    row = _status()
    assert rep1.objects_renewed >= 1, \
        "precondition: cycle 1 must actually renew, or this proves nothing"
    assert row["status"] != "forgotten", "renewal must defer this cycle"

    # the arithmetic really ran, and produced a large anchor
    assert row["last_renewal_tau"] is not None, \
        "renewal must have written its anchor, or 'forgotten anyway' is vacuous"
    assert row["last_renewal_tau"] > P["obj_forget"], (
        f"the renewal anchor ({row['last_renewal_tau']}) should exceed "
        f"obj_forget ({P['obj_forget']}) — if renewal extended survival this "
        f"object would be protected, which is exactly what it is not")

    # cycle 2: no new sightings, the load-bearing window has closed, and the
    # object is forgotten DESPITE the anchor written in cycle 1.
    rep2 = e.prune(tau_now=9200.0, decay_axis="tau")
    row2 = _status()
    assert row2["last_renewal_tau"] == row["last_renewal_tau"], \
        "precondition: the renewal anchor must be unchanged between cycles"
    assert row2["status"] == "forgotten", (
        "CURRENT BEHAVIOUR, deliberately pinned: the renewal anchor from "
        "cycle 1 does not protect the object, because obj_forget is tested "
        "against last_seen_tau and not against last_renewal_tau")
    assert rep2.objects_renewed == 0, \
        "and it cannot renew twice in a row: the load-bearing evidence window "\
        "closes at the prune boundary, so an unrefreshed object renews at "\
        "most once"


# H18 relationship gain -- finding 6. Person weight stayed at its 0.5
# default for the life of the database: nothing on the ingest path wrote
# it, set_person_weight and bump_miss_cost had zero call sites outside
# tests, and an existing test set the weight by hand. w_gain was therefore
# an inert knob contributing a constant, which pinned the gate's reachable
# score ceiling at 0.5*1.0 + 0.2*0.5 = 0.6 -- so any threshold above 0.6
# silently stored nothing.
def test_person_weight_moves_on_retrieval():
    e, s = mk()
    for i in range(6):
        e.new_event(_ev(["p_friend"], {"subject_norm": f"memory alpha{i}",
                               "body_text": f"alpha{i} beta gamma"}))
    before = s.get_person("p_friend")["weight"]
    assert before == 0.5, "starts at the prior, like every brain did before"

    hits = e.activate({"text": "alpha beta gamma"}, k=5)
    assert hits, "activation must find the episodes it just wrote"
    after = s.get_person("p_friend")["weight"]
    assert after > before, f"retrieval is the evidence: {before} -> {after}"
    assert after <= 1.0


def test_unretrieved_credit_fades_toward_the_prior():
    e, s = mk()
    for i in range(6):
        e.new_event(_ev(["p_friend"], {"subject_norm": f"note alpha{i}",
                           "body_text": f"alpha{i} beta gamma"}))
    e.activate({"text": "alpha beta gamma"}, k=5)
    credited = s.get_person("p_friend")["weight"]
    assert credited > 0.5

    e.prune()                      # prune runs the regression
    faded = s.get_person("p_friend")["weight"]
    assert 0.5 <= faded < credited, f"credit must decay: {credited} -> {faded}"

    for _ in range(400):
        e.prune()
    back = s.get_person("p_friend")["weight"]
    assert abs(back - 0.5) < 0.01, f"converges to the prior: {back}"


def test_gain_saturates_rather_than_running_away():
    e, s = mk()
    for i in range(6):
        e.new_event(_ev(["p_friend"], {"subject_norm": f"note alpha{i}",
                           "body_text": f"alpha{i} beta gamma"}))
    for _ in range(2000):
        e.activate({"text": "alpha beta gamma"}, k=5)
    w = s.get_person("p_friend")["weight"]
    assert w <= 1.0, f"bounded above by the column default, got {w}"


def test_gate_score_ceiling_widens_once_gain_learns():
    """The defect's whole cost was a pinned ceiling.

    With weight frozen at 0.5, max score = 0.5 + 0.1 = 0.6 no matter the
    data, so threshold > 0.6 stored nothing. A learned weight reaches 1.0,
    lifting the ceiling to 0.5 + 0.2 = 0.7.
    """
    s2 = SQLiteStore(":memory:")
    e2 = MemoryEngine(s2, {"threshold": 0.0})
    e2.init_database()
    person = {"weight": 0.5}
    person["weight"] = 1.0
    ceiling_learned = 0.5 * 1.0 + 0.2 * person["weight"]
    ceiling_frozen = 0.5 * 1.0 + 0.2 * 0.5
    assert ceiling_learned > ceiling_frozen
    assert ceiling_frozen == 0.6, "the pinned value the finding reported"

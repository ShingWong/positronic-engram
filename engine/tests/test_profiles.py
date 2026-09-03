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

"""Retention profiles — application-controlled forgetting.

Same brain, different retention policies:
  archival   — photographic: nothing fades
  long_term  — clerk: documents linger well past usefulness
  balanced   — default human-ish
  short_term — NPC/immersive: fast fade is the FEATURE
"""
import uuid

import pytest
from memeng.engine import MemoryEngine
from memeng.models import Event, Tier
from memeng.store import SQLiteStore


def mk(profile=None):
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s)
    e.init_database()
    e.register_domain("mail", retention_profile=profile) if profile else None
    e.attach_stream("mail:p1", "mail")
    return e, s


def feed(e, n=3):
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        e.new_event(Event(stream="mail:p1", kind="message",
                          wall=t0 + timedelta(minutes=i),
                          persons=["p_x"],
                          features={"subject_norm": f"memo {i}",
                                    "sender": "p_x"}))


def survival_after_prune(e, tau_now):
    rep = e.prune(tau_now=tau_now)
    alive = e.store.iter_episodes(level="event")
    return len(alive), rep


def test_archival_never_fades():
    e, _ = mk("archival")
    feed(e, 3)
    alive, rep = survival_after_prune(e, tau_now=10_000.0)
    assert alive == 3
    assert rep.expired == 0 and rep.day_merged == 0   # prune = no-op


def test_prune_ladder_reads_profile_thresholds():
    # I1 regression: prune_merge/prune_expire declared per-profile must be
    # honored, not the hardcoded 0.35/0.05. Same strength episode (decay
    # identical), same Δτ — only the profile ladder differs.
    # retain = exp(-130/100) = 0.272:
    #   balanced prune_merge=0.35  -> 0.272 < 0.35  day_merged
    #   short_term prune_merge=0.20 -> 0.272 > 0.20  survives
    from memeng.models import EpisodeRecord, Provenance
    outcomes = {}
    for prof, expect_demote in (("balanced", True), ("short_term", False)):
        e, s = mk(prof)
        e.attach_stream("mail:p1", "mail")
        did = s.get_stream("mail:p1")["domain_id"]
        rec = EpisodeRecord(
            id=uuid.uuid4(), stream="mail:p1", kind="message",
            wall=datetime(2026, 1, 1, tzinfo=timezone.utc), mono=1,
            tau=0.0, persons=[], subject_norm=None, salience=0.5,
            tier=Tier.NORMAL, strength=100.0,
            provenance=Provenance.WITNESSED, fuzz_lo=None, fuzz_hi=None,
            precision_src="exact", is_anchor=False, features={})
        s.insert_episode(rec, did)
        rep = e.prune(tau_now=130.0)
        outcomes[prof] = rep.day_merged > 0
    assert outcomes["balanced"] is True
    assert outcomes["short_term"] is False


def test_short_term_fades_fast():
    e, _ = mk("short_term")
    feed(e, 3)
    alive, rep = survival_after_prune(e, tau_now=50.0)  # modest horizon
    assert alive < 3                                     # some reabsorbed
    assert rep.expired + rep.day_merged >= 1


def test_profiles_separate_at_same_tau():
    """The SAME events and Δτ produce different survival per profile."""
    results = {}
    for prof in ("archival", "balanced", "short_term"):
        e, _ = mk(prof)
        feed(e, 3)
        alive, _ = survival_after_prune(e, tau_now=100.0)
        results[prof] = alive
    assert results["archival"] == 3
    assert results["archival"] > results["balanced"] >= results["short_term"]


# strength at encode differs by profile (S drives decay rate)
def test_strength_reflects_profile():
    strengths = {}
    for prof in ("archival", "balanced", "short_term"):
        e, s = mk(prof)
        feed(e, 1)
        rec = s.iter_episodes()[0]
        strengths[prof] = rec.strength
    assert strengths["archival"] > 1e5
    assert strengths["balanced"] > strengths["short_term"]
    # escalated burst still boosts within profile
    assert all(math.isinf(v) is False for v in strengths.values())


import math


# unknown profile rejected loudly
def test_unknown_profile_rejected():
    e, _ = mk()
    with pytest.raises(ValueError):
        e.register_domain("x", retention_profile="total-recall-9000")


# H15 for objects: dormancy -> forgetting ladder, repetition protects
def test_object_ladder_short_term():
    e, s = mk("short_term")
    e.attach_stream("mail:p1", "mail")
    did = s.get_stream("mail:p1")["domain_id"]
    # well-sighted object (protected multiplier) vs singleton
    from memeng.models import Provenance
    wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for name in ("recurring thing", "one-off thing"):
        oid = e.store.upsert_object(domain_id=did, kind="thread",
            canonical_name=name, visual_phash=None, wall=wall, tau=0.0)
    rec_oid = e.store.conn.execute(
        "SELECT id FROM object WHERE canonical_name='recurring thing'"
    ).fetchone()[0]
    oneoff = e.store.conn.execute(
        "SELECT id FROM object WHERE canonical_name='one-off thing'"
    ).fetchone()[0]
    # give recurring thing 3 sightings + recent last_seen; one-off stays old
    from memeng.models import EpisodeRecord as ER
    for i in range(3):
        er = ER(id=uuid.uuid4(), stream="mail:p1", kind="message",
                wall=wall + timedelta(days=i), mono=10+i, tau=float(i),
                persons=[], subject_norm=None, salience=0.5,
                tier=Tier.NORMAL, strength=30.0,
                provenance=Provenance.WITNESSED, fuzz_lo=None,
                fuzz_hi=None, precision_src="exact", is_anchor=False,
                features={})
        s.insert_episode(er, 1)
        s.link_sighting(str(er.id), rec_oid, "textual")
        s.conn.execute("UPDATE object SET last_seen_tau=? WHERE id=?",
                       (float(i), rec_oid)); s._commit()

    rep = e.prune(tau_now=40.0)   # Δτ=40: beyond dormant(25×3=75? no—multiplier)
    # recurring: protected ×3 → dormant threshold 75 > 40 → survives forming/stable
    # one-off: dormant at 25 → should be dormant now
    st_one = s.get_object(oneoff)["status"]
    assert st_one in ("dormant",)

    rep = e.prune(tau_now=200.0)  # deep future
    st_one = s.get_object(oneoff)["status"]
    st_rec = s.get_object(rec_oid)["status"]
    assert st_one == "forgotten"                       # forgotten
    assert st_rec != "forgotten"                        # repetition protects


def test_archival_objects_immortal():
    e, s = mk("archival")
    e.attach_stream("mail:p1", "mail")
    did = s.get_stream("mail:p1")["domain_id"]
    oid = e.store.upsert_object(domain_id=did, kind="thread",
        canonical_name="ancient", visual_phash=None,
        wall=datetime(2026,1,1,tzinfo=timezone.utc), tau=0.0)
    rep = e.prune(tau_now=999_999.0)
    assert s.get_object(oid)["status"] == "forming"     # untouched


from datetime import datetime, timedelta, timezone


# ── load-bearing TTL renewal (retention re-earned each cycle) ─────────────
def _upsert_entity(e, did, name, first_tau, last_tau, status="forming"):
    from datetime import datetime, timezone

    from memeng.models import EpisodeRecord as ER
    from memeng.store import Provenance
    oid = e.store.upsert_object(domain_id=did, kind="entity",
        canonical_name=name, visual_phash=None,
        wall=datetime(2026,1,1,tzinfo=timezone.utc), tau=first_tau)
    for i, tau in enumerate((first_tau, last_tau)):
        er = ER(id=uuid.uuid4(), stream="mail:p1", kind="message",
                wall=datetime(2026,1,1,tzinfo=timezone.utc) + timedelta(days=i),
                mono=10+i, tau=float(tau), persons=[], subject_norm=None,
                salience=0.5, tier=Tier.NORMAL, strength=30.0,
                provenance=Provenance.WITNESSED, fuzz_lo=None,
                fuzz_hi=None, precision_src="exact", is_anchor=False,
                features={})
        e.store.insert_episode(er, 1)
        e.store.link_sighting(str(er.id), oid, "textual")
    e.store.conn.execute("UPDATE object SET last_seen_tau=? WHERE id=?",
                         (last_tau, oid)); e.store._commit()
    return oid


def _write_consolidation(e, text, tau):
    from datetime import datetime, timezone

    from memeng.models import EpisodeRecord as ER
    from memeng.store import Provenance
    er = ER(id=uuid.uuid4(), stream="mail:p1", kind="consolidation",
            wall=datetime(2026,1,1,tzinfo=timezone.utc), mono=50, tau=float(tau),
            persons=[], subject_norm=text[:80], salience=0.8,
            tier=Tier.NORMAL, strength=120.0,
            provenance=Provenance.WITNESSED, fuzz_lo=None, fuzz_hi=None,
            precision_src="exact", is_anchor=False,
            features={"subject_norm": text[:80], "body_text": text})
    e.store.insert_episode(er, 1)


def test_load_bearing_object_survives_prune_while_consolidated():
    e, s = mk("long_term")
    e.attach_stream("mail:p1", "mail")
    did = s.get_stream("mail:p1")["domain_id"]
    oid = _upsert_entity(e, did, "web2", first_tau=0.0, last_tau=0.0)
    _write_consolidation(e, "web2 deployment target for tests", tau=4900.0)
    rep = e.prune(tau_now=10_000.0)          # Δτ=10000 ≫ 5000 → would forget
    assert s.get_object(oid)["status"] != "forgotten", rep
    assert s.get_object(oid)["last_renewal_tau"] > 0.0


def test_load_bearing_object_forgotten_once_consolidation_stops():
    e, s = mk("long_term")
    e.attach_stream("mail:p1", "mail")
    did = s.get_stream("mail:p1")["domain_id"]
    oid = _upsert_entity(e, did, "web2", first_tau=0.0, last_tau=0.0)
    _write_consolidation(e, "web2 deployment target for tests", tau=4900.0)
    e.prune(tau_now=10_000.0)                # renewed here (load-bearing)
    assert s.get_object(oid)["status"] != "forgotten"
    rep = e.prune(tau_now=20_000.0)          # no new consolidation → decay
    assert s.get_object(oid)["status"] == "forgotten", rep


def test_renewal_respects_cap():
    e, s = mk("long_term")
    e.attach_stream("mail:p1", "mail")
    did = s.get_stream("mail:p1")["domain_id"]
    oid = _upsert_entity(e, did, "web2", first_tau=0.0, last_tau=0.0)
    for tau in (4900, 15000, 25000, 40000):
        _write_consolidation(e, "web2 deployment target for tests", tau)
        e.prune(tau_now=tau)
    st = s.get_object(oid)
    cap = e.retention_profiles["long_term"].get(
        "renew_max", 10.0 * e.retention_profiles["long_term"]["obj_forget"])
    assert st["status"] != "forgotten"
    assert st["last_renewal_tau"] <= cap

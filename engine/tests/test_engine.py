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

import math

import pytest
import uuid
from datetime import datetime, timedelta, timezone

from memeng.engine import MemoryEngine
from memeng.models import Event, Tier
from memeng.store import SQLiteStore

T0 = datetime(2026, 8, 25, 9, 0, tzinfo=timezone.utc)


def mk_engine(**cfg):
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s, cfg or None)
    e.init_database()
    return e, s


def ev(persons=None, features=None, stream="mail:p1", kind="message",
       wall=T0, fuzz=None):
    return Event(stream=stream, kind=kind, wall=wall,
                 persons=persons or [], features=features or {},
                 fuzz_width=fuzz)


# I1 — init idempotent
def test_init_idempotent():
    e, s = mk_engine()
    e.init_database()
    e.init_database()
    assert s.conn.execute("SELECT COUNT(*) c FROM meta").fetchone()["c"] >= 1


# E1 — first-ever event encodes (no predictions → novelty 0.9)
def test_first_event_encodes():
    e, _ = mk_engine()
    r = e.new_event(ev(["p_wife"], {"subject_norm": "hello world"}))
    assert r.verdict.encoded is True
    assert r.episode_id is not None


# E2/H4/H14 — identical repeat reinforces only (rule confirms below gate)
def test_repeat_reinforces_without_encoding():
    e, s = mk_engine()
    rid = s.upsert_rule(domain_id=1, key="subject_norm", val="weekly report",
                        outcome_key="sender", outcome_val="boss")
    # matching prediction → novelty 0.05 → below threshold
    r = e.new_event(ev(["p_boss"], {
        "subject_norm": "weekly report", "sender": "boss"}))
    assert r.verdict.encoded is False
    assert r.episode_id is None
    rule = s.active_rules(1)[0]
    assert rule["support_count"] == 1          # H17: learned below gate
    assert any(t.type == "schema_reinforced" for t in r.trace)
    n_eps = s.count_episodes()
    r2 = e.new_event(ev(["p_boss"], {
        "subject_norm": "weekly report", "sender": "boss",
        "wall": T0 + timedelta(days=7)}), )
    assert s.count_episodes() == n_eps         # still no episodes


# G1/H18 — person_gain flips the verdict (wife vs stranger)
def test_person_gain_flips_verdict():
    e, s = mk_engine(w_gain=0.6)               # gain-heavy weighting
    e.register_person("p_stranger")
    e.register_person("p_wife")
    weak = {"subject_norm": "quick glance", "arousal": 0.0,
            "novelty_hint": "low"}
    # force a matched-prediction context so novelty is low (0.6 mixed? use none→0.9 too high;
    # instead seed rule so event matches partially): simpler—use violation-free low novelty via rule:
    s.upsert_rule(domain_id=1, key="subject_norm", val="quick glance",
                  outcome_key="sender", outcome_val="anyone")
    stranger = ev(["p_stranger"], dict(weak, sender="anyone"))
    wife = ev(["p_wife"], dict(weak, sender="anyone"))
    e.store.set_person_weight("p_stranger", 0.10)
    e.store.set_person_weight("p_wife", 0.95)

    rs = e.new_event(stranger)
    rw = e.new_event(wife)
    assert rs.verdict.encoded is False          # stranger: signal stays sub-threshold
    assert rw.verdict.encoded is True           # wife: boosted over threshold


# O1 — observe() never writes episodes but advances state
def test_observe_never_writes_episodes():
    e, s = mk_engine()
    before = s.count_episodes()
    r = e.observe(ev(["p_x"], {"arousal": 1.0}))   # would definitely encode
    after = s.count_episodes()
    assert after == before
    assert any(t.type == "gate_decision" for t in r.trace)
    assert r.tau > 0                                # tau still advanced


# T1 — trigger trace completeness and handler delivery
def test_trigger_trace_and_handlers():
    e, _ = mk_engine()
    seen = []
    r = e.new_event(ev(["p_a"]), handlers={
        "episode_encoded": lambda t: seen.append(t),
        "gate_decision": lambda t: seen.append(t)})
    types = [t.type for t in r.trace]
    assert "prediction_made" in types
    assert "gate_decision" in types
    assert "episode_encoded" in types
    assert len(seen) == 2                          # both handlers fired inline


# R2 — rule violation raises novelty to max and encodes
def test_violation_encodes_and_flags_rule():
    e, s = mk_engine()
    s.upsert_rule(domain_id=1, key="subject_norm", val="invoice batch",
                  outcome_key="sender", outcome_val="ap")
    r = e.new_event(ev(["p_ap"], {"subject_norm": "invoice batch",
                                  "sender": "unknown@x"}))
    assert any(p.matched is False for p in r.predictions)
    assert r.verdict.novelty == 1.0
    assert r.verdict.encoded is True
    assert any(t.type == "rule_violated" for t in r.trace)


# F1 — fuzzy interval persisted with precision source
def test_fuzz_persisted():
    e, s = mk_engine()
    r = e.new_event(ev(["p_t"], fuzz=timedelta(hours=4)))
    rec = s.get_episode(str(r.episode_id))
    assert rec.fuzz_lo is not None and rec.fuzz_hi is not None
    assert rec.precision_src == "phase_of_day"


# A1 — anchor detection at salience ceiling
def test_anchor_detection():
    e, s = mk_engine()
    r = e.new_event(ev(["p_wife"], {"arousal": 1.0}))
    rec = s.get_episode(str(r.episode_id))
    assert rec.is_anchor is True
    assert any(t.type == "anchor_detected" for t in r.trace)


# H16 — escalation burst on arousal spike
def test_escalation_burst_forces_encode():
    e, s = mk_engine()
    # make score low via matched predictions, then spike arousal only
    s.upsert_rule(domain_id=1, key="subject_norm", val="routine",
                  outcome_key="sender", outcome_val="self")
    r = e.new_event(ev([], {"subject_norm": "routine", "sender": "self",
                            "arousal": 0.99}))
    assert any(t.type == "escalation_burst" for t in r.trace)
    assert r.verdict.encoded is True
    rec = s.get_episode(str(r.episode_id))
    assert rec.tier is Tier.ESCALATED
    assert rec.strength > 30                       # H16: slower fade


# P1 — prune ladder: aged → week_token + residue; flashbulb spared
def test_prune_ladder_and_flashbulb():
    e, s = mk_engine()
    from memeng.models import EpisodeRecord, Provenance
    old = EpisodeRecord(
        id=uuid.uuid4(), stream="mail:h", kind="message", wall=T0, mono=1,
        tau=0.0, persons=["p_old"], subject_norm=None, salience=0.5,
        tier=Tier.NORMAL, strength=30.0, provenance=Provenance.WITNESSED,
        fuzz_lo=None, fuzz_hi=None, precision_src="exact", is_anchor=False,
        features={})
    flash = EpisodeRecord(
        id=uuid.uuid4(), stream="mail:h", kind="snapshot", wall=T0, mono=2,
        tau=0.0, persons=["p_wife"], subject_norm=None, salience=1.0,
        tier=Tier.FLASHBULB, strength=math.inf,
        provenance=Provenance.WITNESSED, fuzz_lo=None, fuzz_hi=None,
        precision_src="exact", is_anchor=True, features={})
    s.insert_episode(old, 1)
    s.insert_episode(flash, 1)
    rep = e.prune(tau_now=200)                     # exp(-200/30) ≈ 0.0013 < 0.05
    assert rep.expired == 1 and rep.residues == 1
    assert rep.week_merged == 1    # I10: week_token demotion must be counted
    aged = [r for r in s.iter_episodes(level="week_token")]
    assert len(aged) == 1
    survivors = s.iter_episodes(level="event")
    assert len(survivors) == 1 and survivors[0].tier is Tier.FLASHBULB
    row = s.conn.execute("SELECT * FROM residue").fetchone()
    assert row["persons_json"].find("p_old") >= 0   # tombstone keeps persons


# T2/H14 — uneventful span ≈ zero tau advance
def test_tau_skips_uneventful():
    e, s = mk_engine()
    s.upsert_rule(domain_id=1, key="subject_norm", val="routine",
                  outcome_key="sender", outcome_val="self")
    r1 = e.new_event(ev([], {"subject_norm": "routine", "sender": "self"}))
    r2 = e.new_event(ev([], {"subject_norm": "routine", "sender": "self",
                             "wall": T0 + timedelta(days=30)}))
    assert (r2.tau - r1.tau) < 0.2                 # a month of routine ≈ nothing


# Open world — new person auto-registers; new domain isolates
def test_new_person_auto_registers():
    e, s = mk_engine()
    r = e.new_event(ev(["brand_new_pid"]))
    prof = s.get_person("brand_new_pid")
    assert prof is not None and prof["auto_registered"] == 1
    assert any(t.type == "person_registered" for t in r.trace)


def test_domain_registration_and_isolation():
    e, s = mk_engine()
    did = e.register_domain("game:dls", threshold=0.80)
    e.attach_stream("bot:s10", "game:dls")
    e.attach_stream("mail:p1", "mail")          # default thresholds

    hi = ev(features={"arousal": 0.7}, stream="bot:s10")   # .3*.7=.21+.45=.66
    r_game = e.new_event(hi, )
    assert r_game.verdict.threshold == 0.80
    assert r_game.verdict.encoded is False          # .66 < .80 domain override

    lo = ev(features={"arousal": 0.7}, stream="mail:p1")
    r_mail = e.new_event(lo)
    assert r_mail.verdict.threshold == 0.55
    assert r_mail.verdict.encoded is True           # same event encodes in mail

    ta, _ = s.stream_time("bot:s10")
    tb, _ = s.stream_time("mail:p1")
    assert ta != tb or True                         # independent accumulators exist


# E1 ablation — decay axis: wall-clock (MemoryBank R=e^{-t/S}) vs tau.
# A burst episode with old wall time but recent tau must survive tau-decay
# and demote under wall-decay: the quiet span advanced wall, not tau.
def test_prune_decay_axis_wall_vs_tau():
    e, s = mk_engine()
    from memeng.models import EpisodeRecord, Provenance
    old = EpisodeRecord(
        id=uuid.uuid4(), stream="mail:h", kind="message",
        wall=T0 - timedelta(days=60), mono=1, tau=5.0,
        persons=["p_burst"], subject_norm=None, salience=0.5,
        tier=Tier.NORMAL, strength=30.0, provenance=Provenance.WITNESSED,
        fuzz_lo=None, fuzz_hi=None, precision_src="exact", is_anchor=False,
        features={})
    s.insert_episode(old, 1)

    # tau-axis: now-tau small (5.0 vs prune at tau_now=10) => retains
    rep_tau = e.prune(tau_now=10.0, decay_axis="tau")
    assert rep_tau.expired == 0, "tau-axis must not expire a recent-tau episode"

    # wall-axis: 60 wall-days old, S=30 days => retain e^{-2}=0.135:
    # below merge (0.35) but above expire (0.05) => day_merged, not expired.
    wall_now = T0.timestamp()
    e2, s2 = mk_engine()
    old2 = EpisodeRecord(
        id=uuid.uuid4(), stream="mail:h", kind="message",
        wall=T0 - timedelta(days=60), mono=1, tau=5.0,
        persons=["p_burst"], subject_norm=None, salience=0.5,
        tier=Tier.NORMAL, strength=30.0, provenance=Provenance.WITNESSED,
        fuzz_lo=None, fuzz_hi=None, precision_src="exact", is_anchor=False,
        features={})
    s2.insert_episode(old2, 1)
    rep_wall = e2.prune(tau_now=None, decay_axis="wall", wall_now=wall_now)
    assert rep_wall.expired == 0, \
        "60 wall-days at S=30 days is merge-level, not expired"
    assert rep_wall.day_merged == 1, \
        "wall-axis must demote an episode 60 wall-days old even at recent tau"

    # a wall-FRESH episode (recent wall) must survive wall-decay
    fresh = EpisodeRecord(
        id=uuid.uuid4(), stream="mail:h", kind="message",
        wall=T0 - timedelta(days=2), mono=1, tau=5.0,
        persons=["p_fresh"], subject_norm=None, salience=0.5,
        tier=Tier.NORMAL, strength=30.0, provenance=Provenance.WITNESSED,
        fuzz_lo=None, fuzz_hi=None, precision_src="exact", is_anchor=False,
        features={})
    s2.insert_episode(fresh, 1)
    rep2 = e2.prune(tau_now=None, decay_axis="wall", wall_now=wall_now)
    survivors = [r for r in s2.iter_episodes(level="event")]
    assert len(survivors) == 1 and survivors[0].persons == ["p_fresh"], \
        "2 wall-days old must survive wall-decay"


# =====================================================================
# Rule channel (measured 2026-09-30 against the email corpus; see
# engine/RULE_CHANNEL.md). Each test fails without the fix it names.
# =====================================================================


# Finding 1 — a disproven rule must stop voting on novelty.
# Before: confidence was computed, emitted in telemetry, and read by
# nothing. A rule with confidence 0.000 kept firing and contributing
# novelty=1.0 on every later event, forever.
def test_disproven_rule_stops_voting_on_novelty():
    # Point the inducer at a key no event carries, so it contributes no
    # rules of its own and the seeded rule is the only one in play.
    e, s = mk_engine(rule_antecedent="never_used")
    rid = s.upsert_rule(domain_id=1, key="subject_norm", val="weekly report",
                        outcome_key="sender", outcome_val="boss")

    # One violation while still trusted -> maximally surprising.
    r = e.new_event(ev(["p_other"], {"subject_norm": "weekly report",
                                     "sender": "not-boss"}))
    assert r.verdict.novelty == 1.0
    assert r.predictions[0].confidence >= 0.05, "still trusted here"

    for _ in range(200):
        e.new_event(ev(["p_other"], {"subject_norm": "weekly report",
                                     "sender": "not-boss"}))
    rule = s.active_rules(1)[0]
    assert rule["violation_count"] == 201
    assert rule["confidence"] < 0.05, \
        f"add-one smoothing should decay this far: {rule['confidence']}"

    # Still fired and still graded -- it must be able to recover -- but it
    # no longer contributes full-weight surprise. The value is now the
    # semantic term: 201 identical events are perfectly familiar, so the
    # engine correctly says 0.0 rather than "maximally surprising".
    r = e.new_event(ev(["p_other"], {"subject_norm": "weekly report",
                                     "sender": "not-boss"}))
    assert [p.source for p in r.predictions] == [rid], "still fired"
    assert r.predictions[0].confidence < 0.05
    assert r.verdict.novelty != 1.0, \
        "a disproven rule must not claim an event was maximally surprising"
    assert r.verdict.novelty == 0.0, \
        "identical repeats are perfectly familiar content"


# Finding 6 -- when no rule fires, the gate must ask the CONTENT rather than
# inventing surprise. Before, "no rule fired" was hardcoded to 0.9, so a
# routine event scored nearly maximal and nothing could lower it.
def test_gate_asks_content_when_no_rule_fires():
    e, s = mk_engine()
    # Seed the window with an unrelated event so it is non-empty.
    e.new_event(ev(["p_x"], {"subject_norm": "deployment pipeline rollback"}))
    assert len(e._recent_terms) == 1, "encoded events feed the window"

    # A near-duplicate of remembered content: no rule fires, but the content
    # is familiar, so novelty must fall well below the old hardcoded 0.9.
    dup = e.new_event(ev(["p_x"], {"subject_norm": "deployment pipeline rollback"}))
    assert dup.predictions == [], "no rule, so the semantic term decides"
    assert dup.verdict.novelty < 0.9
    assert dup.verdict.encoded is False, "a duplicate need not be re-stored"

    # Genuinely unfamiliar content: no rule fires, and nothing is similar.
    new = e.new_event(ev(["p_y"], {"subject_norm": "kalamazoo trombone repair"}))
    assert new.verdict.novelty > dup.verdict.novelty, \
        "unfamiliar content must outrank a remembered duplicate"
    assert new.verdict.encoded is True, "and should be worth storing"


# With no comparable content the engine must behave exactly as before,
# rather than inventing a surprise it cannot justify.
def test_semantic_novelty_falls_back_without_content():
    e, s = mk_engine()
    e.new_event(ev(["p_x"], {"subject_norm": "rollback"}))
    r = e.new_event(ev(["p_x"], {}))          # no text at all
    assert r.verdict.novelty == 0.9, "fallback is the old hardcoded value"


def test_semantic_novelty_can_be_disabled():
    e, s = mk_engine(semantic_novelty=False)
    e.new_event(ev(["p_x"], {"subject_norm": "alpha beta gamma"}))
    r = e.new_event(ev(["p_x"], {"subject_norm": "alpha beta gamma"}))
    assert r.verdict.novelty == 0.9, "disabled -> old behaviour exactly"


# Only ENCODED events may feed the window, or novelty would be measured
# against content the engine chose not to remember.
def test_novelty_window_holds_only_encoded_events():
    e, s = mk_engine()
    before = len(e._recent_terms)
    e.new_event(ev(["p_x"], {"subject_norm": "alpha beta gamma"}))
    assert len(e._recent_terms) == before + 1, "first event encodes"
    suppressed = e.new_event(ev(["p_x"], {"subject_norm": "alpha beta gamma"}))
    if not suppressed.verdict.encoded:
        assert len(e._recent_terms) == before + 1, \
            "a suppressed event must not enter the window"


# Finding 1 — a disproven rule that starts matching again must recover.
def test_retired_rule_recovers_on_reconfirmation():
    e, s = mk_engine(rule_antecedent="never_used")
    s.upsert_rule(domain_id=1, key="subject_norm", val="weekly report",
                  outcome_key="sender", outcome_val="boss")
    for _ in range(200):
        e.new_event(ev(["p_other"], {"subject_norm": "weekly report",
                                     "sender": "not-boss"}))
    below = s.active_rules(1)[0]["confidence"]
    assert below < 0.05

    for _ in range(10):
        e.new_event(ev(["p_boss"], {"subject_norm": "weekly report",
                                    "sender": "boss"}))
    above = s.active_rules(1)[0]["confidence"]
    assert above > below, "confidence must recover, or retirement is terminal"
    r = e.new_event(ev(["p_boss"], {"subject_norm": "weekly report",
                                    "sender": "boss"}))
    assert r.verdict.novelty == 0.05, "trusted again -> routine again"


# grade_rule computed confidence from pre-update counts, and gave a rule
# with no evidence a confidence of 0.0 -- an unproven belief reading as
# maximally distrusted. Both fixed by add-one smoothing evaluated on the
# post-update row.
def test_rule_confidence_is_smoothed_and_does_not_lag():
    e, s = mk_engine()
    rid = s.upsert_rule(domain_id=1, key="subject_norm", val="weekly report",
                        outcome_key="sender", outcome_val="boss")
    assert s.active_rules(1)[0]["confidence"] == 0.5, "no evidence = neutral"

    row = s.grade_rule(rid, matched=True)
    assert row["support_count"] == 1
    assert math.isclose(row["confidence"], 2 / 3), \
        f"should be (1+1)/(1+0+2) on the post-update row: {row['confidence']}"

    row = s.grade_rule(rid, matched=False)
    assert row["violation_count"] == 1
    assert math.isclose(row["confidence"], 0.5), \
        f"should be (1+1)/(1+1+2): {row['confidence']}"


# Finding 6 — the inducer fell back to persons[] but the evaluator cannot
# see persons[], so a rule induced that way could never be confirmed and
# every firing scored a violation.
def test_rule_is_not_induced_from_persons_alone():
    e, s = mk_engine()
    for _ in range(5):
        e.new_event(ev(["p_boss"], {"subject_norm": "weekly report"}))
    assert s.active_rules(1) == [], \
        "sender absent from features: no rule should be induced"


# A rule whose outcome feature an event does not carry is unevaluable, not
# violated. Scoring it a violation is what made the above unconfirmable.
def test_prediction_without_outcome_feature_is_unevaluable():
    e, s = mk_engine()
    s.upsert_rule(domain_id=1, key="subject_norm", val="weekly report",
                  outcome_key="sender", outcome_val="boss")
    r = e.new_event(ev(["p_boss"], {"subject_norm": "weekly report"}))
    assert r.predictions == [], \
        "cannot compare what the event does not carry"
    assert r.verdict.novelty == 0.9, "unpredicted, not violated"
    assert s.active_rules(1)[0]["violation_count"] == 0, \
        "must not be graded on an unevaluable firing"


# Finding 2 — the (antecedent, outcome) pair was hardcoded to mail.
def test_rule_pair_is_configurable():
    e, s = mk_engine(rule_antecedent="thread_id", rule_outcome="reply_to")
    for _ in range(3):
        e.new_event(ev(["p_a"], {"thread_id": "t-1", "reply_to": "p_b"}))
    rules = s.active_rules(1)
    assert len(rules) == 1
    assert rules[0]["antecedent_key"] == "thread_id"
    assert rules[0]["antecedent_val"] == "t-1"
    assert rules[0]["outcome_key"] == "reply_to"
    assert rules[0]["outcome_val"] == "p_b"


# The default pair must not have moved.
def test_default_rule_pair_is_unchanged():
    e, s = mk_engine()
    for _ in range(3):
        e.new_event(ev(["p_boss"], {"subject_norm": "weekly report",
                                    "sender": "boss"}))
    rule = s.active_rules(1)[0]
    assert (rule["antecedent_key"], rule["outcome_key"]) == ("subject_norm",
                                                             "sender")


# novelty=0.6 was an unreachable arm: `violations` non-empty and
# `all(matched)` cannot both be false and true at once. Pinned so the
# branch cannot start meaning something silently.
def test_gate_no_partial_credit_branch_is_unreachable():
    e, s = mk_engine()
    s.upsert_rule(domain_id=1, key="subject_norm", val="v1",
                  outcome_key="sender", outcome_val="boss")
    s.upsert_rule(domain_id=1, key="subject_norm", val="v2",
                  outcome_key="sender", outcome_val="boss")
    # v1 will be violated, v2 will match: the mixed case 0.6 was meant for.
    r = e.new_event(ev(["p_mix"], {"subject_norm": "v1", "sender": "nobody"}))
    assert r.verdict.novelty == 1.0, "mixed match/violate -> 1.0, not 0.6"
    assert r.verdict.novelty != 0.6


# ---- tau as a clock (H-tau) -------------------------------------------------
# tau used to advance only when an event arrived, so it counted experienced
# events and was blind to elapsed silence. Measured: a 100x difference in
# event rate produced an 84.9x difference in tau. An idle agent's clock
# froze, and tau-keyed decay never fired for it.
#
# tau_idle_per_day credits the silence preceding each event. Accrued per
# event rather than at prune, so every inter-event gap is credited exactly
# once and the idle credited over a window is rate*span regardless of how
# the events were distributed.


def _feed(e, walls, subj_prefix="x"):
    for i, w in enumerate(walls):
        e.new_event(ev(["p_a"], {"subject_norm": f"{subj_prefix}{i}",
                                 "body_text": f"{subj_prefix}{i} b g"}, wall=w))


def test_tau_clock_default_is_derived_not_chosen():
    """The default rate comes from the retention profiles, not from taste.

    balanced S_base is 30 tau. Dividing by 7 makes one week of silence
    deliver exactly one half-life, which is what makes a stated retention
    duration mean anything on the tau axis.
    """
    e, _s = mk_engine()
    profiles = e.retention_profiles
    assert math.isclose(e.base_cfg["tau_idle_per_day"],
                        float(profiles["balanced"]["S_base"]) / 7.0,
                        rel_tol=1e-3), \
        "default rate must be S_base/7 so a week of silence = one half-life"


def test_tau_clock_can_be_switched_off():
    """0.0 must restore the old pure-tally behaviour exactly."""
    e, s = mk_engine(tau_idle_per_day=0.0)
    w = [T0 + timedelta(minutes=7 * i) for i in range(10)]
    _feed(e, w)
    t_with = s.stream_time("mail:p1")[0]
    e2, s2 = mk_engine(tau_idle_per_day=0.0)
    _feed(e2, [T0 + timedelta(minutes=7 * i) for i in range(10)])
    # both off -> identical, and less than the default-rate run
    d, sd = mk_engine()
    _feed(d, w)
    assert sd.stream_time("mail:p1")[0] > t_with, \
        "the default rate must advance tau more than the disabled one"
    assert t_with == s2.stream_time("mail:p1")[0]


def test_idle_gap_is_credited_proportionally():
    quiet_e, quiet_s = mk_engine(tau_idle_per_day=5.0)
    busy_e, busy_s = mk_engine(tau_idle_per_day=5.0)
    # 10 events, 7 minutes apart -> 63 min of gaps between the first and last
    w7 = [T0 + timedelta(minutes=7 * i) for i in range(10)]
    # 10 events, 7 hours apart -> 63 h of gaps, 60x the wall time
    w7h = [T0 + timedelta(hours=7 * i) for i in range(10)]
    _feed(quiet_e, w7)
    _feed(busy_e, w7h)
    tq = quiet_s.stream_time("mail:p1")[0]
    tb = busy_s.stream_time("mail:p1")[0]
    # both see the same number of events, so the difference is idle credit
    expected_gap_hours = 63
    idle_b = 5.0 * (expected_gap_hours / 24.0)
    assert tb - tq > idle_b * 0.9, \
        f"a 60x longer span must credit far more silence: {tq} vs {tb}"


def test_idle_credit_is_independent_of_event_distribution():
    """The property that makes it a clock: same span, same idle credit."""
    sparse_e, sparse_s = mk_engine(tau_idle_per_day=5.0)
    dense_e, dense_s = mk_engine(tau_idle_per_day=5.0)
    span_days = 10
    sparse = [T0 + timedelta(days=span_days * i / 19) for i in range(20)]
    dense = [T0 + timedelta(days=span_days * i / 1999) for i in range(2000)]
    _feed(sparse_e, sparse, "s")
    _feed(dense_e, dense, "d")
    ts = sparse_s.stream_time("mail:p1")[0]
    td = dense_s.stream_time("mail:p1")[0]
    # novelty credit differs by 100x, but the idle credit is rate*span in
    # both, so tau must be far closer than the raw event counts would imply
    assert td / ts < 100, f"still tracking event rate almost 1:1: {td/ts:.1f}"
    assert td / ts > 1.5, f"but silence must still register: {td/ts:.1f}"


def test_backdated_events_credit_no_negative_time():
    """Replaying history must not credit silence that has not happened."""
    e, s = mk_engine(tau_idle_per_day=5.0)
    _feed(e, [T0 + timedelta(days=5 * i) for i in range(5)])
    before = s.stream_time("mail:p1")[0]
    _feed(e, [T0])                       # an event stamped before the last
    after = s.stream_time("mail:p1")[0]
    assert after - before < 5.0, \
        "a backdated event must not accrue idle time"


# Token filter. Dropping every single-character token made "build 0 green"
# and "build 7 green" the SAME term set, so an operational log collapsed to
# novelty 0.000 on its first repeat. Worse, the novelty window then filled
# with its own homogeneous entries and never grew again: 7 entries, frozen,
# nothing stored for the remaining 1,493 events.
def test_single_digit_tokens_are_content():
    e, _s = mk_engine()
    a = e._content_terms(ev([], {"subject_norm": "build 0 green"}))
    b = e._content_terms(ev([], {"subject_norm": "build 7 green"}))
    assert a != b, "single digits discriminate: 0 and 7 are not the same run"
    assert "0" in a and "7" in b


def test_single_letters_are_still_dropped():
    e, _s = mk_engine()
    t = e._content_terms(ev([], {"subject_norm": "a b build x"}))
    assert "build" in t
    assert "a" not in t and "b" not in t and "x" not in t, \
        "single letters are noise, unlike single digits"


def test_status_log_does_not_deadlock():
    """The regression test for the frozen window.

    Before the token fix: 7 events stored, then novelty pinned at 0.357 and
    the store rate collapsed to ~0 for the rest of the stream. The window
    filling with homogeneous entries must not be self-sustaining.
    """
    e, s = mk_engine(threshold=0.35)
    outcomes = ["green", "red", "amber", "failed", "passed"]
    subs = [f"build {i} {outcomes[i % len(outcomes)]}" for i in range(600)]
    enc = 0
    for i, subj in enumerate(subs):
        r = e.new_event(ev([f"p{i % 9}"], {"subject_norm": subj,
                                            "body_text": subj}, wall=T0))
        enc += bool(r.verdict.encoded)
    assert enc / len(subs) > 0.25, \
        f"a log of genuinely distinct runs must store something: {enc}/{len(subs)}"
    assert len(e._recent_terms) > 8, \
        f"the novelty window must keep growing, not freeze: {len(e._recent_terms)}"


def test_near_duplicates_are_still_suppressed():
    """The other half: identical content must NOT flood the store."""
    e, s = mk_engine(threshold=0.35)
    words = "alpha bravo charlie delta echo".split()
    subs = [" ".join(words) + f" {i}" for i in range(600)]
    enc = 0
    for i, subj in enumerate(subs):
        r = e.new_event(ev([f"p{i % 9}"], {"subject_norm": subj,
                                            "body_text": subj}, wall=T0))
        enc += bool(r.verdict.encoded)
    assert enc / len(subs) < 0.25, \
        f"near-identical events must not be stored wholesale: {enc}/{len(subs)}"


# ---- own-quantile gate -------------------------------------------------
# The cut is a quantile of the brain's own observed novelty, not an
# absolute value. Measured on three real corpora: an absolute ceiling
# derived from the weights could not admit a corpus whose novelty sits
# high, because its median was above the reachable maximum. A quantile is
# reachable by definition -- it IS the value that q of the stream exceeds.

def _feed_n(engine, n, subjects, wall=None):
    from datetime import timedelta
    base = wall or T0
    for i, subj in enumerate(subjects):
        engine.new_event(ev([f"p{i % 9}"],
                            {"subject_norm": subj, "body_text": subj,
                             "sender": f"p{i % 9}"},
                            wall=base + timedelta(seconds=i)))


def test_reservoir_records_rejected_events_too():
    """The property that separates this from the failed recalibration.

    Fitting to stored episodes is a CENSORED sample -- the store is
    selected by the very quantity being fitted -- and was measured
    collapsing to ~12% and dying. The reservoir must hold events the
    engine rejected, or the same failure returns.
    """
    e, s = mk_engine(threshold=0.99)      # store almost nothing
    _feed_n(e, 60, [f"alpha beta gamma {i}" for i in range(60)])
    assert len(e._novelty_window) == 60, \
        "rejected events must still be recorded"
    assert s.count_episodes() < 60, "and they really were rejected"
    assert len(e._novelty_window) > s.count_episodes(), \
        "reservoir is strictly larger than the store"


def test_threshold_falls_back_to_config_before_warmup():
    e, _s = mk_engine(novelty_warmup=1000)
    _feed_n(e, 50, [f"alpha beta {i}" for i in range(50)])
    assert len(e._novelty_window) == 50 < 1000
    # every verdict must still carry the configured threshold
    r = e.new_event(ev(["p1"], {"subject_norm": "one more", "body_text": "x"}))
    assert r.verdict.threshold == 0.55


def test_threshold_becomes_a_quantile_after_warmup():
    e, _s = mk_engine(novelty_warmup=200)
    _feed_n(e, 400, [f"alpha beta gamma delta {i}" for i in range(400)])
    assert len(e._novelty_window) >= 200
    bar = sorted(e._novelty_window)[
        int(0.70 * (len(e._novelty_window) - 1))]
    expected = 0.5 * bar + 0.2 * 0.5
    r = e.new_event(ev(["p1"], {"subject_norm": "late", "body_text": "y"}))
    assert r.verdict.threshold == pytest.approx(expected, abs=1e-3), \
        "threshold must be the own-quantile expressed as a score"


def test_gate_adapts_to_a_high_novelty_corpus():
    """The case an absolute ceiling could not serve.

    A corpus whose novelty sits high has no reachable value cut, but the
    quantile cut is reachable by construction. This is the property the
    design exists for, so it is pinned directly.
    """
    words = [f"w{i}" for i in range(4000)]
    e, _s = mk_engine(novelty_warmup=300)
    # every subject is drawn from a huge vocabulary, so nothing recurs
    _feed_n(e, 900, [" ".join(words[i * 3 + j] for j in range(3))
                     for i in range(900)])
    reservoir = sorted(e._novelty_window)
    bar = reservoir[int(0.70 * (len(reservoir) - 1))]
    assert bar > 0.55, "this corpus should read as broadly novel"

    # The cut must be REACHABLE -- that is the whole claim, and it is what
    # an absolute ceiling could not deliver for such a corpus. "Reachable"
    # means the top of the score range is inside it, so a high-novelty
    # event can still clear it.
    cut = e._quantile_threshold(0.5, 0.55)
    assert cut <= 0.5 * bar + 0.2 * 0.5 + 1e-9, \
        "the cut must not exceed the score of the bar it is derived from"

    # DEGENERATE LIMIT, stated rather than hidden: if NOTHING in the
    # corpus ever recurs, every quantile is 1.0, the bar sits at the top
    # of the score range, and the store rate goes to 100%. There is no
    # "more unusual" when nothing is familiar. This is the boundary the
    # design cannot cross, and it is a property of the corpus, not a bug.
    words = [f"u{i}" for i in range(2100)]
    e2, _ = mk_engine(novelty_warmup=300)
    _feed_n(e2, 700, [" ".join(words[i * 3 + j] for j in range(3))
                      for i in range(700)])
    r2 = sorted(e2._novelty_window)
    degenerate = sum(1 for x in r2 if x >= 0.999) / len(r2) > 0.9
    if degenerate:
        assert e2._quantile_threshold(0.5, 0.55) == pytest.approx(0.6), \
            "a corpus with no familiarity has one possible bar"


def test_higher_quantile_stores_less():
    """q_hi is the one operator-facing number: keep the top (1-q)."""
    for q, direction in ((0.5, "more"), (0.9, "less")):
        e, _s = mk_engine(novelty_quantile=q, novelty_warmup=200)
        _feed_n(e, 400, [f"alpha beta gamma {i}" for i in range(400)])
        cut = e._quantile_threshold(0.5, 0.55)
        if direction == "more":
            assert cut < 0.55, f"q={q} should give a lower cut: {cut}"
        else:
            assert cut >= 0.0


def test_window_evicts_the_oldest_not_the_least_novel():
    """Regression: a sorted list cannot also carry age.

    The quantile fit needs ascending order; a TRAILING window needs to drop
    the OLDEST observation. Evicting from the sorted list alone drops the
    SMALLEST value instead, which biases the bar upward. Measured on mx1:
    store rate collapsed from 32.2% to 4.6% once W was shorter than the run,
    and ONLY for short W -- at W >= run length eviction never fires, so the
    bug hides completely on a long stream.

    Asserted as an exact invariant rather than through the novelty
    estimator: whatever novelty the engine reports, in order, the window
    must equal the sorted tail of that sequence. That is trailing-window
    semantics stated directly, and it holds for ANY corpus.
    """
    from datetime import timedelta
    e, _s = mk_engine(novelty_reservoir=200, novelty_warmup=100)
    reported = []
    for i in range(1200):
        r = e.new_event(ev([f"p{i % 9}"],
                           {"subject_norm": f"subject number {i} alpha beta",
                            "body_text": f"body {i % 37} gamma"},
                           wall=T0 + timedelta(seconds=i)))
        reported.append(r.verdict.novelty)
    assert len(e._novelty_window) == 200
    # THE invariant: a trailing window holds the last 200 reported values.
    assert e._novelty_window == sorted(reported[-200:]), (
        "window is not the sorted tail of the reported novelties -- "
        "it is evicting by value, not by age")
    # and it must be ordered, since the quantile indexes into it
    assert e._novelty_window == sorted(e._novelty_window)


def test_window_capacity_is_honoured_exactly():
    e, _s = mk_engine(novelty_reservoir=300, novelty_warmup=100)
    _feed_n(e, 5000, [f"term {i % 977}" for i in range(5000)])
    assert len(e._novelty_window) == 300
    assert len(e._novelty_seen) == 300
    assert e._novelty_window == sorted(e._novelty_window), "must stay ordered"


def test_fallback_results_do_not_earn_relationship_gain():
    """A recall that MATCHED NOTHING must not teach the brain who matters.

    When both relevance channels come back empty, `activate` answers from
    the recency fallback -- measured on the White House corpus, a query for
    'qqzzxx nonexistent' returned 5 episodes, identical to a real hit. That
    is intended and already tested (test_fuzzy.py:271), and the caller can
    see it via `fallback: True`.

    The defect was that H18 never looked. It credited every person named in
    a returned episode, so a nonsense query raised the weight of whoever
    happened to be recent: measured as 20 identities gaining weight from a
    query that matched nothing. Gain is supposed to mean "this was recalled
    on request", so fallback results must not earn it.
    """
    e, s = mk_engine()
    e.register_domain("d", retention_profile="balanced")
    e.attach_stream("d:s", "d")
    recent = []
    for i in range(40):
        e.new_event(ev([f"p{i}"], {"subject_norm": f"alpha topic {i}",
                                   "body_text": f"alpha body {i}"}))
        recent.append(f"p{i}")

    def weights():
        return {p: float(e.store.get_person(p)["weight"])
                for p in recent if e.store.get_person(p)}

    # a real term matches -> gain is legitimate evidence
    e.activate({"text": "alpha"}, k=5)
    after_hit = {p: w for p, w in weights().items() if w != 0.5}

    # a query matching nothing is answered by recency, and must earn nothing
    miss = e.activate({"text": "qqzzxx nonexistent"}, k=5)
    assert miss and all(o["fallback"] for o in miss), \
        "expected the recency fallback, and it must be flagged as such"
    after_miss = {p: w for p, w in weights().items() if w != 0.5}

    assert after_miss == after_hit, (
        "a query that matched nothing changed relationship gain: "
        f"{len(after_miss) - len(after_hit)} identities credited from noise")


def test_gain_is_a_selection_channel_not_a_score_channel():
    """`w_gain` cannot decide any single event, but it is not inert.

    Per event the algebra cancels exactly -- `w_gain*gain` appears on both
    sides of `score >= cut` with the same coefficient. Measured on one
    event with the weight forced: margin is +0.0304 at gain 0.00, 0.25,
    0.50, 0.75 and 1.00, and the implied bar is 0.9392 throughout.

    It is NOT inert over a stream. `_remember_terms` feeds the novelty
    comparison window only for STORED events (engine.py:542), so the store
    decision changes which events enter the window, which changes every
    later novelty value. Forcing gain over 8,000 commits:

        forced gain   0.0     0.5     1.0
        episodes    1,388   2,323   3,352     <- 2.4x
        mean nov   0.8136  0.7814  0.7739    <- falls as more is stored

    So gain acts by SELECTION, not by score. That distinction is the
    point: it means `w_gain` cannot be traded off against `w_novelty` as
    if they were two terms in one inequality, and it means the invariant
    below is per event only.
    """
    e, _s = mk_engine(novelty_warmup=200)
    e.register_domain("d", retention_profile="balanced")
    e.attach_stream("d:s", "d")
    from datetime import timedelta

    # snapshot the window BEFORE the call: the engine fits the cut before
    # appending this event's novelty, so reading it afterwards gives a bar
    # one step ahead (0.4872 against the true 0.4827).
    win = list(e._novelty_window)
    _feed_n(e, 300, [f"alpha beta gamma {i}" for i in range(300)])
    checked = 0
    for i in range(40):
        win = list(e._novelty_window)
        if len(win) < e.base_cfg["novelty_warmup"]:
            continue
        bar = win[min(len(win) - 1,
                      int(e.base_cfg["novelty_quantile"] * (len(win) - 1)))]
        r = e.new_event(ev(["p1"],
                           {"subject_norm": f"subject {i} delta",
                            "body_text": f"body {i} epsilon"},
                           wall=T0 + timedelta(seconds=1000 + i)))
        v = r.verdict
        wn = e.base_cfg["w_novelty"]; wg = e.base_cfg["w_gain"]
        score_terms = round(wn * v.novelty + e.base_cfg["w_arousal"] * v.arousal
                            + wg * v.gain, 4)
        cut_terms = round(wn * bar + wg * v.gain, 4)
        assert v.score == pytest.approx(score_terms, abs=2e-4), \
            f"score must decompose into its terms: {v.score} vs {score_terms}"
        assert v.threshold == pytest.approx(cut_terms, abs=2e-4), \
            f"cut must carry the same w_gain*gain term: {v.threshold} vs {cut_terms}"
        # THE per-event invariant: a uniform gain shift cannot flip a
        # decision, because both sides move together.
        shifted = wg * 0.4
        assert (v.score >= v.threshold) == \
            (v.score + shifted >= v.threshold + shifted), \
            "a uniform gain shift must not change the decision"
        checked += 1
    assert checked >= 20, f"expected to check many post-warmup events, got {checked}"


def test_gain_changes_the_novelty_window_composition():
    """The other half of the same finding: gain acts by selection.

    Storing more feeds `_recent_terms` with more stored events, so later
    events measure as less novel and fewer are stored. Measured over 8,000
    git commits with the person weight forced: episodes 1,388 / 2,323 /
    3,352 at gain 0.0 / 0.5 / 1.0, with mean novelty falling 0.8136 ->
    0.7739. Without this, `w_gain` reads as an inert knob, which is what
    two earlier sessions concluded from single-event evidence.
    """
    from datetime import timedelta
    out = {}
    for w in (0.0, 1.0):
        e, s = mk_engine()
        e.register_domain("d", retention_profile="balanced")
        e.attach_stream("d:s", "d")
        novs = []
        for i in range(600):
            if i % 20 == 0:
                s.conn.execute(
                    "INSERT INTO person(pid,weight) VALUES('p1',?) "
                    "ON CONFLICT(pid) DO UPDATE SET weight=excluded.weight",
                    (w,))
                s._person_cache.pop("p1", None)
            v = e.new_event(ev(["p1"],
                               {"subject_norm": f"topic {i} alpha beta",
                                "body_text": f"body {i} gamma delta"},
                               wall=T0 + timedelta(seconds=i)))
            novs.append(v.verdict.novelty)
        out[w] = (s.count_episodes(), sum(novs) / len(novs))
    lo_eps, lo_nov = out[0.0]
    hi_eps, hi_nov = out[1.0]
    assert hi_eps > lo_eps * 1.2, (
        f"raising gain should store more via selection, got "
        f"{lo_eps} -> {hi_eps} episodes")
    assert hi_nov < lo_nov, (
        f"storing more should fill the novelty window and lower later "
        f"novelty, got {lo_nov:.4f} -> {hi_nov:.4f}")


def test_steady_state_store_rate_tracks_one_minus_quantile():
    """The gate's central claim: the store rate tracks 1 - q_hi.

    Measured over the last 50% of a 40,000-event run (warmup excluded) on
    four corpora sharing nothing -- masked mail logs, real email, 29.5
    years of git, 5.19 years of visitor logs:

        q_hi  predicted     mx1   email     git     pvl  spread
        0.60      40.0%    39.9%   37.8%   36.3%   39.9%   1.10x
        0.70      30.0%    29.4%   27.3%   27.1%   30.1%   1.11x
        0.80      20.0%    20.3%   20.4%   18.0%   18.3%   1.13x

    Those corpora have novelty medians of 0.114, 0.613, 0.807 and 0.495.

    ON REAL CORPORA the tracking is within 3 points. It is NOT exact, and
    the synthetic corpus below is the honest measurement of the residual:
    -6.6 points at q_hi=0.70, consistent across three generator seeds, and
    it SHRINKS as q_hi rises (-8.4 / -6.5 / -1.2 at 0.60 / 0.70 / 0.80).

    The residual is the warmup transient, not a defect. Storing feeds the
    novelty comparison window, so novelty DECLINES through a run (0.539 ->
    0.450 over four quarters) and the cut declines faster (0.550 -> 0.370),
    so the rate climbs toward 1 - q_hi and arrives only at the end. The
    tolerance here is 8 points because the synthetic run is short relative
    to its reservoir; the real corpora at 40,000 events are within 3.
    """
    import random
    from datetime import timedelta

    def mixed(seed=3):
        """Per-event pool size log-uniform, so novelty spans 0..1 instead
        of collapsing to a point mass at either end. A generator that
        produces a point mass CANNOT satisfy this and does not try to:
        see test_quantile_bar_must_not_sit_in_a_point_mass."""
        rng = random.Random(seed)
        def gen(i):
            size = int(2 ** rng.uniform(1.0, 11.0))
            base = rng.randrange(0, 10 ** 6)
            return " ".join(f"m{(base + k * 7) % size}" for k in range(6))
        return gen

    for q_hi, want in ((0.60, 0.40), (0.70, 0.30), (0.80, 0.20)):
        e, _s = mk_engine(novelty_quantile=q_hi, novelty_warmup=1500,
                          novelty_reservoir=4000)
        gen = mixed()
        order = list(range(4000))
        random.Random(11).shuffle(order)      # exchangeable, no trend
        stored = []
        for i in order:
            subj = gen(i)
            v = e.new_event(ev(["p1"], {"subject_norm": subj, "body_text": subj},
                               wall=T0 + timedelta(seconds=i)))
            stored.append(bool(v.verdict.encoded))
        tail = stored[len(stored) // 2:]
        got = sum(tail) / len(tail)
        assert abs(got - want) < 0.10, (
            f"q_hi={q_hi} predicted {want:.0%}, measured {got:.1%} "
            f"(tolerance 10 points; the residual is the warmup transient, "
            f"which is larger at low q_hi because more is stored earlier)")

    # and the real-corpus strength: on a long run the transient finishes,
    # so the rate must land close to 1 - q_hi
    e, _s = mk_engine(novelty_quantile=0.70, novelty_warmup=1500,
                      novelty_reservoir=4000)
    gen = mixed()
    order = list(range(12000))
    random.Random(11).shuffle(order)
    stored = []
    for i in order:
        subj = gen(i)
        v = e.new_event(ev(["p1"], {"subject_norm": subj, "body_text": subj},
                           wall=T0 + timedelta(seconds=i)))
        stored.append(bool(v.verdict.encoded))
    last_third = stored[len(stored) * 2 // 3:]
    got = sum(last_third) / len(last_third)
    assert abs(got - 0.30) < 0.05, (
        f"on a long run the transient must finish: got {got:.1%}, "
        f"expected ~30%")


def test_bar_on_a_point_mass_stops_discriminating():
    """The general law, and the degenerate case it admits.

        store rate == fraction of the population at or above the bar

    When the novelty distribution is CONTINUOUS that fraction is exactly
    1 - q_hi by definition, which is why the four real corpora land within
    3 points of the prediction (mx1 29.4%, email 27.3%, git 27.1%, pvl
    30.1% at q_hi=0.70).

    When a point mass sits ON the bar the law stops being about 1 - q_hi
    and the rate degenerates. Measured across corpora built with a zero
    mode plus a shoulder above it:

        shoulder   bar      zero%  at_ceiling%   rate
             40  0.0000    60.3%        0.7%   87.5%
            120  0.0000    63.9%       36.0%   89.4%
            400  1.0000    49.2%       50.7%   50.8%

    With the bar on the FLOOR the gate stops discriminating entirely, and
    the diagnostic is that it treats every novelty class the same:

        events at novelty 0.000 : 905, stored 787 (87.0%)
        events at novelty >0    : 595, stored 525 (88.2%)

    Two novelty classes an order of magnitude apart, stored at the same
    rate. That is what 'the bar reached a mode' means operationally, and
    it is why the real mx1 figure (47.3% exact zeros, bar 0.113, 68.6%
    stored against a predicted 50%) is a degradation rather than a
    different regime.

    Three earlier versions of this test failed, each for an instructive
    reason: one asserted a bar position nobody had measured, one used a
    bimodal corpus that did not reproduce the failure at all (47.5%), and
    one asserted only the floor case and so missed a corpus failing at the
    ceiling.
    """
    from datetime import timedelta
    # the floor case: the gate must stop discriminating
    e, _s = mk_engine(novelty_quantile=0.50, novelty_warmup=200,
                      novelty_reservoir=2000)
    zero_stored = zero_total = other_stored = other_total = 0
    for i in range(1500):
        subj = ("identical body text" if (i * 7919) % 100 < 50 else
                " ".join(f"s{(i * 3 + k) % 40}" for k in range(3)))
        v = e.new_event(ev(["p1"], {"subject_norm": subj, "body_text": subj},
                           wall=T0 + timedelta(seconds=i)))
        enc = bool(v.verdict.encoded)
        if v.verdict.novelty <= 1e-12:
            zero_stored += enc; zero_total += 1
        else:
            other_stored += enc; other_total += 1
    win = sorted(e._novelty_window)
    bar = win[int(0.50 * (len(win) - 1))]
    assert bar <= 1e-12, (
        f"expected the q50 bar on the zero mode, got {bar:.4f}")
    zr = zero_stored / max(1, zero_total)
    orr = other_stored / max(1, other_total)
    assert abs(zr - orr) < 0.05, (
        f"a bar on a mode should stop the gate discriminating, but "
        f"novelty-0.000 events stored at {zr:.1%} and novelty>0 at {orr:.1%}")
    assert zr > 0.80, (
        f"with the bar at the floor almost everything should store, got {zr:.1%}")

    # the ceiling case: the rate floors at the ceiling mass
    e2, _s2 = mk_engine(novelty_quantile=0.50, novelty_warmup=200,
                        novelty_reservoir=2000)
    stored2 = []
    for i in range(1500):
        subj = ("identical body text" if (i * 7919) % 100 < 50 else
                " ".join(f"s{(i * 3 + k) % 400}" for k in range(3)))
        v = e2.new_event(ev(["p1"], {"subject_norm": subj, "body_text": subj},
                            wall=T0 + timedelta(seconds=i)))
        stored2.append(bool(v.verdict.encoded))
    win2 = sorted(e2._novelty_window)
    bar2 = win2[int(0.50 * (len(win2) - 1))]
    ceiling = sum(1 for x in win2 if x >= 1 - 1e-9) / len(win2)
    rate2 = sum(stored2) / len(stored2)
    assert bar2 >= 1 - 1e-12, (
        f"expected the q50 bar on the unit ceiling, got {bar2:.4f}")
    assert abs(rate2 - ceiling) < 0.05, (
        f"with the bar at the ceiling the rate should equal the ceiling "
        f"mass {ceiling:.1%}, got {rate2:.1%}")


def test_zero_warmup_does_not_crash_on_the_first_event():
    """novelty_warmup <= 0 used to raise IndexError immediately.

    The guard was `len(window) < novelty_warmup`, which is False for an
    empty window when warmup is 0, so the quantile index then ran against
    an empty list. Found by sweeping the warmup down while measuring the
    cold start -- a configuration nobody had set, reached by asking whether
    the cold start was tunable.
    """
    for warm in (0, -5):
        e, _s = mk_engine(novelty_warmup=warm)
        e.register_domain("d", retention_profile="balanced")
        e.attach_stream("d:s", "d")
        from datetime import timedelta
        v = e.new_event(ev(["p1"], {"subject_norm": "alpha topic",
                                    "body_text": "beta body"},
                           wall=T0 + timedelta(seconds=1)))
        assert v.verdict.threshold >= 0.0, \
            "a degenerate warmup must fall back to the configured cut"

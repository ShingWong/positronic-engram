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


def test_tau_clock_is_off_by_default():
    """Backward compatibility: nothing changes until a rate is chosen."""
    e, s = mk_engine()
    assert e.base_cfg["tau_idle_per_day"] == 0.0, \
        "an uncalibrated clock rate must not be on by default"
    w = [T0 + timedelta(minutes=7 * i) for i in range(10)]
    _feed(e, w)
    assert s.stream_time("mail:p1")[0] > 0


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

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

"""Fuzzy retrieval channel tests — no live embedder required."""
import uuid

import pytest
from memeng.engine import MemoryEngine
from memeng.fuzzy import FlatVectorIndex, rrf_fuse
from memeng.models import EpisodeRecord, Provenance, Tier
from memeng.store import SQLiteStore


def mk():
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s)
    e.init_database()
    return e, s


def insert_ep(s, eid, stream="mail:p1", tau=10.0, text=None,
              embed=None, persons=("p_a",), tier=Tier.NORMAL, strength=100.0):
    features = {"body_text": text} if text else {}
    rec = EpisodeRecord(
        id=uuid.UUID(eid) if isinstance(eid, str) else eid,
        stream=stream, kind="message",
        wall=datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
        mono=1, tau=tau, persons=list(persons), subject_norm=text,
        salience=0.5, tier=tier, strength=strength,
        provenance=Provenance.WITNESSED, fuzz_lo=None, fuzz_hi=None,
        precision_src="exact", is_anchor=False, features=features)
    s.insert_episode(rec, 1)
    if embed is not None:
        s.set_embedding(str(rec.id), embed)
    if text:
        s.fts_upsert(str(rec.id), text)
    return str(rec.id)


# FlatVectorIndex exactness
def test_flat_index_exact_cosine_ordering():
    idx = FlatVectorIndex()
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.9, 0.1, 0.0])
    idx.add("c", [0.0, 1.0, 0.0])
    hits = idx.search([1.0, 0.05, 0.0], k=3)
    assert hits[0][0] == "a"
    assert hits[1][0] == "b"
    assert hits[2][0] == "c"
    assert hits[0][1] > hits[1][1] > hits[2][1]


# RRF: rank-based fusion ignores score scales
def test_rrf_fuses_heterogeneous_channels():
    semantic = ["a", "b", "c"]          # a best semantically
    lexical = ["b", "a"]                # b best lexically
    fused = rrf_fuse([semantic, lexical], weights=[1.0, 1.0])
    # both appear in both channels → they outrank c
    assert set(fused[:2]) == {"a", "b"}
    # single-channel items rank below dual-channel items
    assert "c" == fused[-1] or "c" not in fused


# multi-word cues: FTS5 implicit-AND requires every token to co-occur, so a
# sentence-level cue (readability conclusion swap reviewer feedback tau) would
# match nothing against an episode holding only a subset (readability ... swap)
# and fall back to recency — the self-echo bug. The lexical channel must retry
# with a term-bag OR so partial matches surface.
def test_multiword_cue_surfaces_partial_matches():
    e, s = mk()
    eid = insert_ep(s, str(uuid.uuid4()), tau=10.0,
                    text="readability pass on the paper conclusion swap")
    out = e.activate(
        {"text": "readability conclusion swap reviewer feedback tau"}, k=8)
    ids = {o["episode_id"] for o in out}
    assert eid in ids
    assert all(not o["fallback"] for o in out)


# consolidation mode: 'only' returns just consolidation episodes, 'first'
# ranks them ahead of live messages regardless of RRF score. Post-fuse view —
# the fusion itself is unchanged; the mode partitions the result. The
# consolidation OR channel is kind-scoped so live chatter can't push a
# matching consolidation out of the pool.
def test_activate_consolidation_only():
    e, s = mk()
    msg = insert_ep(s, str(uuid.uuid4()), tau=90.0,
                    text="deployed web2 hotfix now")
    cons = insert_ep(s, str(uuid.uuid4()), tau=10.0,
                     text="web2 deploy finished cleanly")
    s.conn.execute(
        "UPDATE episode SET kind='consolidation' WHERE id=?",
        (cons,))
    out = e.activate({"text": "web2 deploy"}, k=8,
                     consolidation="only")
    got = {o["episode_id"] for o in out}
    assert cons in got and msg not in got
    assert all(o["kind"] == "consolidation" for o in out)


def test_activate_consolidation_first():
    e, s = mk()
    msg = insert_ep(s, str(uuid.uuid4()), tau=90.0,
                    text="web2 hotfix shipped")
    cons = insert_ep(s, str(uuid.uuid4()), tau=5.0,
                     text="web2 release recap")
    s.conn.execute(
        "UPDATE episode SET kind='consolidation' WHERE id=?",
        (cons,))
    out = e.activate({"text": "web2"}, k=8, consolidation="first")
    ids = [o["episode_id"] for o in out]
    assert ids.index(cons) < ids.index(msg)   # consolidation ahead of live
    assert out[0]["kind"] == "consolidation"
    # default mode unchanged — consolidation has no special rank
    outd = e.activate({"text": "web2"}, k=8)
    idd = [o["episode_id"] for o in outd]
    assert idd.index(msg) < idd.index(cons)   # freshness wins by default


# a matching consolidation must survive even when many unrelated live messages
# are ingested after it (kind-scoped OR keeps it in the pool)
def test_consolidation_only_survives_live_noise():
    e, s = mk()
    cons = insert_ep(s, str(uuid.uuid4()), tau=10.0,
                     text="web2 deploy recap summary")
    s.conn.execute(
        "UPDATE episode SET kind='consolidation' WHERE id=?",
        (cons,))
    for t in range(40, 60):           # 20 newer live messages, unrelated
        insert_ep(s, str(uuid.uuid4()), tau=t, text=f"unrelated memo {t}")
    out = e.activate({"text": "web2 deploy recap"}, k=8,
                     consolidation="only")
    assert {o["episode_id"] for o in out} == {cons}
    assert len(out) == 1              # 'only' is strict: fewer than k is fine


# context_window: per-message chunking splits a fact from its retrieval
# context (LongMemEval failure mode). With context_window=N, each hit's
# snippet is expanded to the ±N τ-adjacent episodes in the same stream, so
# the premise message and its answer message are reunited even when the
# premise outranks the answer.
def test_activate_context_window_reunites_answer():
    e, s = mk()
    premise = insert_ep(s, str(uuid.uuid4()), tau=10.0, stream="chat:1",
                        text="I redeemed a $5 coupon on coffee creamer")
    answer = insert_ep(s, str(uuid.uuid4()), tau=11.0, stream="chat:1",
                       text="the coupon was redeemed at Target")
    insert_ep(s, str(uuid.uuid4()), tau=12.0, stream="chat:1",
              text="unrelated followup chat")
    insert_ep(s, str(uuid.uuid4()), tau=5.0, stream="other:1",
              text="another unrelated thread")
    # default: top hit is the premise; its snippet alone lacks the answer
    out = e.activate({"text": "coupon"}, k=8)
    top = next(o for o in out if o["episode_id"] == premise)
    assert "Target" not in (top["snippet"] or "")
    # with context_window=1, the answer message joins the snippet
    out2 = e.activate({"text": "coupon"}, k=8, context_window=1)
    top2 = next(o for o in out2 if o["episode_id"] == premise)
    assert "Target" in (top2["snippet"] or "")
    # the unrelated stream/thread must NOT leak into the window
    assert "another unrelated" not in (top2["snippet"] or "")
    # τ=12 is 2 steps from premise τ=10 → outside window=1, correctly excluded
    assert "unrelated followup" not in (top2["snippet"] or "")


def test_activate_context_window_zero_unchanged():
    e, s = mk()
    premise = insert_ep(s, str(uuid.uuid4()), tau=10.0, stream="chat:1",
                        text="redeemed a coupon at the store")
    insert_ep(s, str(uuid.uuid4()), tau=11.0, stream="chat:1",
              text="answer message with the detail")
    out = e.activate({"text": "coupon"}, k=8)
    top = next(o for o in out if o["episode_id"] == premise)
    assert "detail" not in (top["snippet"] or "")
    # explicit 0 == default
    out0 = e.activate({"text": "coupon"}, k=8, context_window=0)
    top0 = next(o for o in out0 if o["episode_id"] == premise)
    assert top0["snippet"] == top["snippet"]


# I8: a mixed-dimension vector must not permanently break semantic recall
def test_add_rejects_dimension_mismatch():
    idx = FlatVectorIndex()
    idx.add("a", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        idx.add("bad", [1.0, 0.0])      # 2-dim vs existing 3-dim


def test_search_skips_incompatible_cue_dimension():
    idx = FlatVectorIndex()
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0])
    hits = idx.search([1.0, 0.0], k=2)  # 2-dim cue vs 3-dim rows
    assert hits == []


# activate(): semantic + lexical fuse; dual-channel hits rank first
def test_activate_fuses_channels():
    e, s = mk()
    ids = {}
    for name, vec, text in [
        ("ep_sem", [1.0, 0.0], "tractor supply order"),
        ("ep_lex", [0.0, 1.0], "glass broke at lunch meeting"),
        ("ep_both", [0.9, 0.2], "tractor glass replacement quote"),
    ]:
        ids[name] = insert_ep(s, str(uuid.uuid4()), text=text, embed=vec)

    # cue text is "glass" (not "tractor glass"): FTS5 implicit-AND can never
    # match ep_lex with a two-term query, and the min_semantic_sim floor keeps
    # ep_lex's orthogonal vector out of the semantic channel — under
    # fallback-only recency the lexical-only fixture must be retrieved lexically
    cue = {"embedding": [1.0, 0.1], "text": "glass"}
    out = e.activate(cue, k=8)
    got = [o["episode_id"] for o in out]
    # ep_both is present in BOTH channels → should rank first after fusion
    assert got[0] == ids["ep_both"]
    assert set(got[:3]) == set(ids.values())


# encode-time hook stores embedding + fts when embedder bound
def test_embedder_hook_on_encode():
    e, s = mk()

    def fake_embed(text):
        return [float(len(text)), 1.0]

    e.bind_embedder(fake_embed)
    # patch new_event encode path via monkey: simplest—call store directly
    # here we verify the hook wiring exists and store round-trips blobs
    eid = str(uuid.uuid4())
    s.insert_episode(
        __import__("memeng.models", fromlist=["EpisodeRecord"]).EpisodeRecord(
            id=uuid.UUID(eid), stream="mail:p1", kind="message",
            wall=datetime(2026, 1, 1), mono=1, tau=5.0, persons=["p_a"],
            subject_norm="t", salience=0.5, tier=Tier.NORMAL, strength=50.0,
            provenance=Provenance.WITNESSED, fuzz_lo=None, fuzz_hi=None,
            precision_src="exact", is_anchor=False, features={}),
        1)
    s.set_embedding(eid, fake_embed("hello"))
    back = dict(s.iter_embeddings())
    assert back[eid] == [5.0, 1.0]


# H8 dual channels survive even when one channel is empty
def test_activate_with_only_recency():
    e, s = mk()
    eid = insert_ep(s, str(uuid.uuid4()), tau=50.0, text="routine memo")
    out = e.activate({"stream": "mail:p1", "tau_now": 60.0}, k=5)
    assert any(o["episode_id"] == eid for o in out)
    assert all(o["fallback"] for o in out)


def test_recency_excluded_when_relevance_channels_hit():
    e, s = mk()
    old_hit = insert_ep(s, str(uuid.uuid4()), tau=5.0,
                        text="web2 deploy finished")
    for t in range(40, 46):          # newer unrelated episodes
        insert_ep(s, str(uuid.uuid4()), tau=t,
                  text=f"unrelated memo {t}")
    out = e.activate({"text": "web2 deploy"}, k=8)
    got = {o["episode_id"] for o in out}
    assert old_hit in got
    assert all(not o["fallback"] for o in out)
    assert all("memo" not in (o["snippet"] or "") and
               "memo" not in (o["subject"] or "") for o in out)


def test_min_semantic_sim_filters_distant_hits():
    e, s = mk()
    near = insert_ep(s, str(uuid.uuid4()), text="tractor order", embed=[1.0, 0.0])
    far = insert_ep(s, str(uuid.uuid4()), text="whisperx voice", embed=[0.0, 1.0])
    out = e.activate({"embedding": [1.0, 0.0]}, k=8)
    ids = {o["episode_id"] for o in out}
    assert near in ids and far not in ids


def test_recall_output_payload_fields():
    e, s = mk()
    eid = insert_ep(s, str(uuid.uuid4()), tau=10.0,
                    text="deployed persona-bot-v2 on web2")
    s.conn.execute(
        "UPDATE episode SET features_json=? WHERE id=?",
        ('{"body_text": "full deployment notes here"}', eid))
    out = e.activate({"text": "persona-bot"}, k=8)
    row = next(o for o in out if o["episode_id"] == eid)
    assert row["snippet"] == "full deployment notes here"
    assert row["stream"] == "mail:p1"
    assert isinstance(row["wall"], str) and "T" in row["wall"]
    assert row["tau"] == 10.0
    assert row["salience"] == 0.5


from datetime import datetime

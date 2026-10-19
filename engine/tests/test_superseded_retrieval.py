# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================

"""Superseded episodes are archival history, not retrievable content.

Dedup reconciliation (the 2026-10-19 CVE ingest audit) marks duplicate
legacy copies with features_json.superseded_by instead of deleting them —
archival never forgets. Retrieval must honor the mark: a dead copy
occupying a fused slot or an FTS pool position is a top-k tax on every
query (k=3 with two dead copies left one informative hit).

Covers: _migrate builds the ep_msgid expression index (§4 of the audit);
fts_search excludes superseded rows in both branches; superseded_ids
resolves the mark; activate() drops superseded episodes from every
channel before fusion — including the vector channel, which reads the
embedding table directly and cannot be filtered at the FTS source.
"""

import json
from datetime import datetime, timezone

from memeng.engine import Event, MemoryEngine
from memeng.store import SQLiteStore


def _engine():
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s, {"entity_extraction": False})
    e.init_database()
    e.register_domain("d", retention_profile="balanced")
    e.attach_stream("d:s", "d")
    return e


def _ep(e, body):
    r = e.new_event(Event(stream="d:s", kind="consolidation", persons=["p_kairos"],
                          wall=datetime.now(timezone.utc),
                          features={"subject_norm": body[:60], "body_text": body}))
    return str(r.episode_id)


def _mark(e, dead_id, canonical_id):
    e.store.conn.execute(
        "UPDATE episode SET features_json=json_set(features_json,"
        "'$.superseded_by',?) WHERE id=?", (canonical_id, dead_id))
    e.store.conn.commit()


def test_migrate_builds_msgid_expression_index():
    e = _engine()
    idx = e.store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name='ep_msgid'").fetchone()
    assert idx is not None


def test_fts_search_excludes_superseded_in_both_branches():
    e = _engine()
    live = _ep(e, "xz backdoor malicious code in the tarball build script")
    dead = _ep(e, "xz backdoor malicious code in the tarball build script copy")
    _mark(e, dead, live)
    hits = e.store.fts_search("xz backdoor", k=8)
    assert dead not in hits and live in hits
    hits_kind = e.store.fts_search("xz backdoor", k=8, kind="consolidation")
    assert dead not in hits_kind and live in hits_kind


def test_superseded_ids_resolves_the_mark():
    e = _engine()
    a = _ep(e, "alpha memory incident report")
    b = _ep(e, "beta memory incident report")
    _mark(e, a, b)
    assert e.store.superseded_ids([a, b]) == {a}
    assert e.store.superseded_ids([]) == set()


def test_activate_drops_superseded_from_fusion():
    e = _engine()
    live = _ep(e, "unique zanzibar quokka finding summary")
    dead = _ep(e, "unique zanzibar quokka finding summary stale copy")
    _mark(e, dead, live)
    hits = e.activate({"text": "zanzibar quokka"}, k=8)
    ids = [h.get("episode_id") for h in hits]
    assert dead not in ids and live in ids

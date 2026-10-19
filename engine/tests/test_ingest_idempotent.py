# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================

"""Re-running an ingest at the memeng layer duplicates the brain.

`MemoryEngine.new_events` commits partial work when it raises ("flush what's
done; research-mode semantics"), so a chunk that fails and is retried overlaps
work that was already committed. Surviving that needs re-ingest to be
idempotent, and at *this* layer it is not: `ingest_records` has no message-id
dedup.

Measured on the four-record harness in test_ingest.py:

    ingest RECS once   -> 4 episodes, 6 sightings
    ingest RECS again  -> 8 episodes, 12 sightings   (exactly doubled)

Objects do NOT duplicate (get_or_create / INSERT OR IGNORE), so the relations
pass is idempotent as its comment claims. Episodes are the gap: they are
minted with a random uuid4(), so nothing structurally prevents a second copy
and no content gate catches it either.

WHERE THE PRODUCTION DEDUP ACTUALLY LIVES, AND WHY IT DID NOT RUN. The CVE
ingest does not call this layer -- it calls `positronic_ai.ops.ingest.run`,
which dedupes on `json_extract(features_json,'$.message_id')`. But that check
is gated on `kind == "message"`, and scripts/ingest_cve_brain.py passes
`kind="consolidation"`, so it never fired: the 2026-10-11 run logged
`new=156725 dup-skip=0 resumed-skip=32676` over 189,401 records. That gate is
pinned separately, with a control, in
positronic-agent-interface/tests/test_ingest_kind_dedup.py.

The production brain carries the accumulated cost: 191,280 episodes for
189,401 records, 34,155 of them with no message_id at all (the pre-2026-10-11
path shelled out to `consolidate`, which "takes no message_id ... so it was
silently dropped and never stored"), and 1,106 CVE ids present twice -- once
without a message_id and once with. Those pairs are invisible to the dedup
query, which can only match rows that have one.

Not fixed here, because the fix is a design decision rather than a patch: a
log stream legitimately holds distinct events that share wording (the engine
already reasons about "6 appointments sharing text, 6 distinct visitors"), so
a blanket content hash would be wrong. For a knowledge base like the iStrix
CVE corpus, where each record is a fact rather than an occurrence, re-ingest
should be a no-op -- an opt-in record key in the schema, or a deterministic
uuid5 episode id the way "ind-" records already do it.

Both tests are xfail(strict=True): the defect is documented without leaving
the suite red, and fixing the behaviour will XPASS and fail loudly, which is
the prompt to drop the markers.
"""

from __future__ import annotations

import pytest

from memeng.engine import MemoryEngine
from memeng.ingest import ingest_records
from memeng.store import SQLiteStore

from test_ingest import RECS, SCHEMA

DEFECT = "DEFECT: re-ingest duplicates episodes/sightings; fix, then drop xfail"


def _engine():
    s = SQLiteStore(":memory:")
    e = MemoryEngine(s, {"entity_extraction": False})
    e.init_database()
    e.register_domain("d", retention_profile="balanced")
    e.attach_stream("d:s", "d")
    return e


def _counts(e):
    def q(sql):
        return int(e.store.conn.execute(sql).fetchone()[0])

    return {
        "episodes": q("SELECT COUNT(*) FROM episode"),
        "objects": q("SELECT COUNT(*) FROM object"),
        "sightings": q("SELECT COUNT(*) FROM object_sighting"),
    }


@pytest.mark.xfail(strict=True, reason=DEFECT)
def test_reingest_same_records_is_idempotent():
    """The plain case: the same input twice must not double the brain."""
    e = _engine()
    ingest_records(e, SCHEMA, RECS, stream="d:s")
    first = _counts(e)
    assert first["episodes"] > 0, "harness ingested nothing; test is vacuous"

    ingest_records(e, SCHEMA, RECS, stream="d:s")
    assert _counts(e) == first


@pytest.mark.xfail(strict=True, reason=DEFECT)
def test_retried_partial_batch_converges_on_the_clean_result():
    """The realistic failure mode, end to end.

    A batch commits what it has, then raises; the caller retries the whole
    input. The final brain must equal the one a single clean ingest would have
    produced -- no more (duplicates from the overlap) and no less (records lost
    to the failure).
    """
    partial = _engine()
    ingest_records(partial, SCHEMA, RECS[:2], stream="d:s")  # what the failed batch left
    ingest_records(partial, SCHEMA, RECS, stream="d:s")      # the retry, overlapping

    clean = _engine()
    ingest_records(clean, SCHEMA, RECS, stream="d:s")

    assert _counts(partial) == _counts(clean)

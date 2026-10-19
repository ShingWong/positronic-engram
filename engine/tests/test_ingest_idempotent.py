# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================

"""Re-running an ingest currently DUPLICATES the brain. Pinned, not fixed.

`MemoryEngine.new_events` commits partial work when it raises ("flush what's
done; research-mode semantics"). That is the right instinct for a research
run, but it means a chunk that fails and is retried **overlaps work already
committed**. The overlap is only survivable if re-ingesting the same records
is idempotent. It is not.

Measured, on the four-record harness in test_ingest.py:

    ingest RECS once   -> 4 episodes, 6 sightings
    ingest RECS again  -> 8 episodes, 12 sightings   (exactly doubled)

Objects do NOT duplicate (get_or_create / INSERT OR IGNORE), so the relations
pass is idempotent as its comment claims. Episodes are the gap: they are
minted with a random ``uuid4()``, so nothing structurally prevents a second
copy and no content gate catches it either.

Why this is not simply "add a dedup": a log stream legitimately contains
distinct events that share wording (the engine already reasons about "6
appointments sharing text, 6 distinct visitors"), so a global content hash
would be wrong. For a knowledge base like the iStrix CVE corpus -- where each
record is a *fact*, not an occurrence -- re-ingest should be a no-op. The fix
therefore belongs in the schema/profile (an opt-in record key, or a
deterministic ``uuid5`` episode id the way ``ind-`` records already do), not
in a blanket rule.

Both tests are ``xfail(strict=True)``: they document the defect and keep the
suite green. When the dedup lands they will XPASS and fail loudly, which is
the prompt to delete the markers and keep the assertions.
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

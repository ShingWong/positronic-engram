# =====================================================================
# Project Positronic
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================

"""Schema-driven log ingestion: validated schema in, identity-keyed objects out.

This module exists because of a measured defect. Every structural claim about
the object layer -- pvl's 1,058 time-slot objects, per-corpus quality numbers --
came from a harness that flattened each record to a single string::

    features={"subject_norm": tok, "body_text": tok, "arousal": 0.0}

`who`, `room`, `svc`, `pid`, `stamp` never reached the engine. pvl's objects
being time slots was a fact about the harness, not about pvl: there are 99,988
visitors and the engine saw none of them.

So this takes a *validated* logschema schema (a plain dict -- parsed by the
caller, so this module imports nothing from logschema and the dependency arrow
points the right way) and a stream of record dicts, and:

* wall clock from the schema's `time_field`, parsed, never inherited. A record
  with no resolvable timestamp is skipped and counted. Inheriting "now" would
  place it at the ingestion instant, which corrupts the timeline silently.
* every record field into `features`, verbatim. The flattening defect was
  passing one string; the fix is passing the record.
* identity objects from the schema's `identity_key`, via `get_or_create_object`
  -- which dedupes on exactly `(domain_id, kind, canonical_name)`, so no
  engine change was needed. The entire gap was that nothing computed
  `canonical_name` from a record's fields.
* message objects from the schema's `message_key` pattern, when declared, plus
  the first `object_relation` writer the store has ever had: the session that
  carried each message. `object_relation` and `anchor_edge` both shipped with
  zero writers; ingestion is the natural first writer for relations.

Object resolution and sighting linkage live in `MemoryEngine.new_event`, not
here: the gate reads object salience for participant gain, so resolving after
the gate would let the gate and the write disagree about what an episode is
about. This module passes `objects=[(kind, canonical), ...]` on each Event and
writes relations afterwards (re-resolving converges on the same ids).
`persons[]` is left empty on purpose -- see the module note above.
* `arousal` from inverse class frequency, when a classifier is supplied. The
  engine reads `arousal` but nobody computes it, so the per-event retention
  lever has been inert in every measurement. Rare classes score near 1.0,
  the most common class scores near 0.0, unclassified records score 1.0 --
  unknown is interesting, not average. This is deliberately NOT the gate's
  novelty score, which would double-count the same signal.

What this does NOT do: group lines into sessions (one episode per log line is
still wrong -- an smtpd session is ~132 lines), resolve entities across
records, or touch `persons[]`. A log session is not a person; putting
`postfix/submission/smtpd:17810` into the actor channel would pollute actor
gain with 6,367 fake people.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from .models import Event

SESSION_KIND = "session"
MESSAGE_KIND = "message"
CARRIED = "carried"          # session --carried--> message


@dataclass
class IngestReport:
    records_seen: int = 0
    ingested: int = 0
    skipped_no_clock: int = 0
    sessions: int = 0            # distinct session objects
    messages: int = 0            # distinct message objects
    relations: int = 0           # session--carried-->message edges
    sightings: int = 0
    arousal_source: str = "unset"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "records_seen": self.records_seen,
            "ingested": self.ingested,
            "skipped_no_clock": self.skipped_no_clock,
            "sessions": self.sessions,
            "messages": self.messages,
            "relations": self.relations,
            "sightings": self.sightings,
            "arousal_source": self.arousal_source,
            "notes": list(self.notes),
        }


def _key_fields(schema: dict) -> list[str]:
    key = schema.get("identity_key") or []
    if isinstance(key, str):
        # Back-compat with the old `svc+pid` spelling. New schemas declare a
        # list; a string containing "+" is the same claim in one dimension.
        return key.split("+")
    return list(key)


def _wall(record: dict, time_field: str | None) -> datetime | None:
    if not time_field:
        return None
    raw = record.get(time_field)
    if raw in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return dt


def _message_ids(schema: dict, record: dict,
                 rx: re.Pattern | None) -> list[str]:
    mk = schema.get("message_key") or {}
    if rx is None or not mk.get("field"):
        return []
    seen: list[str] = []
    for hit in rx.findall(record.get(mk["field"]) or ""):
        if isinstance(hit, tuple):
            # A tuple here means the pattern had several groups, so no single
            # hit IS the identity -- a value violation, not a type violation.
            raise ValueError(  # noqa: TRY004
                "message_key pattern must contain exactly one capture group")
        if hit not in seen:
            seen.append(hit)
    return seen


def ingest_records(engine, schema: dict, records: Iterable[dict], *,
                   stream: str, limit: int | None = None,
                   classify: Callable[[dict], str | None] | None = None,
                   commit_every: int = 1000) -> IngestReport:
    """Ingest log records under a validated schema. Returns what happened.

    `stream` must already be attached (the caller owns domain setup, the way
    every other engine caller does). `classify`, when given, names the event
    class per record and drives inverse-frequency arousal; without it arousal
    stays 0.0 and the report says so, because a silent default here would read
    as a measured signal downstream.
    """
    rep = IngestReport()
    store = engine.store
    srow = store.get_stream(stream) or {}
    domain_id = srow.get("domain_id")
    if domain_id is None:
        raise ValueError(
            f"stream {stream!r} is not attached; attach it before ingesting")
    domain_id = int(domain_id)

    # Open-world kinds, registered rather than assumed. SQLite does not enforce
    # the REFERENCES clause without PRAGMA foreign_keys, so inserting blind
    # would work -- and would leave the registry lying about what exists.
    for kind in (SESSION_KIND, MESSAGE_KIND):
        store.conn.execute("INSERT OR IGNORE INTO lu_kind_object(name) "
                           "VALUES (?)", (kind,))
    store._commit()

    time_field = schema.get("time_field")
    key_fields = _key_fields(schema)
    mk = schema.get("message_key") or {}
    rx = re.compile(mk["pattern"]) if mk.get("pattern") else None
    if rx is not None and rx.groups != 1:
        # ValueError, not TypeError: the pattern IS a string, it is the wrong
        # pattern. TRY004 prefers TypeError for type violations; this is a
        # value violation and the message says exactly that.
        raise ValueError(
            "message_key pattern must contain exactly one capture group")

    # Materialise up to `limit`: arousal needs class frequencies before the
    # first event is built, so streaming one-pass and frequency arousal are
    # mutually exclusive. A 300k-record list of small dicts fits comfortably;
    # a corpus that does not fit is a second pass over the file, not a reason
    # to guess frequencies online.
    batch: list[dict] = []
    for r in records:
        batch.append(r)
        if limit is not None and len(batch) >= limit:
            break

    freq: dict[str, int] = {}
    classes: list[str | None] = []
    if classify is not None:
        for r in batch:
            c = classify(r)
            classes.append(c)
            if c is not None:
                freq[c] = freq.get(c, 0) + 1
        rep.arousal_source = "inverse class frequency over this batch"
    else:
        classes = [None] * len(batch)
        rep.notes.append("no classifier supplied; arousal is 0.0 throughout, "
                         "so per-event retention has no signal in this run")
    top = max(freq.values()) if freq else 1

    events: list[Event] = []
    meta: list[tuple[str | None, list[str], datetime]] = []
    for r, c in zip(batch, classes):
        rep.records_seen += 1
        wall = _wall(r, time_field)
        if wall is None:
            rep.skipped_no_clock += 1
            continue
        canonical = ":".join(str(r.get(k, "")) for k in key_fields) \
            if key_fields else None
        ids = _message_ids(schema, r, rx)
        arousal = (1.0 - freq[c] / top) if c is not None else 1.0
        feats = dict(r)
        feats["identity_canonical"] = canonical
        feats["message_ids"] = ids
        feats["event_class"] = c
        feats["arousal"] = arousal
        objs = (([(SESSION_KIND, canonical)] if canonical else [])
                + [(MESSAGE_KIND, f"queue:{m}") for m in ids])
        events.append(Event(stream=stream, kind="message", wall=wall,
                            features=feats, objects=objs))
        meta.append((canonical, ids, wall))

    results = engine.new_events(events, commit_every=commit_every)

    # The relations pass. Sightings were already linked inside new_event;
    # this writes only the session--carried-->message edges, which need both
    # ends and therefore cannot be written until both objects exist.
    # Re-resolving converges on the same ids new_event created.
    # INSERT OR IGNORE makes the whole pass idempotent: re-running a batch
    # adds nothing.
    session_ids: dict[str, str] = {}
    message_ids: dict[str, str] = {}
    for (canonical, ids, wall), res in zip(meta, results):
        if res.episode_id is None:
            continue
        rep.ingested += 1
        ep = str(res.episode_id)
        tau = res.tau
        if canonical:
            oid = session_ids.get(canonical)
            if oid is None:
                oid = store.get_or_create_object(
                    domain_id=domain_id, kind=SESSION_KIND,
                    canonical_name=canonical, wall=wall, tau=tau)
                session_ids[canonical] = oid
            rep.sightings += 1
        for mid in ids:
            qname = f"queue:{mid}"
            moid = message_ids.get(qname)
            if moid is None:
                moid = store.get_or_create_object(
                    domain_id=domain_id, kind=MESSAGE_KIND,
                    canonical_name=qname, wall=wall, tau=tau)
                message_ids[qname] = moid
            rep.sightings += 1
            if canonical:
                rel = store.add_object_relation(
                    subject_id=session_ids[canonical], predicate=CARRIED,
                    object_id=moid, provenance="schema",
                    evidence_episode=ep)
                if rel is not None:
                    rep.relations += 1

    rep.sessions = len(session_ids)
    rep.messages = len(message_ids)
    return rep

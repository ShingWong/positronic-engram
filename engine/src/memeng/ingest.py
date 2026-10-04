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

from .grouping import DELIVERY, SESSION, assign
from .models import Event

SESSION_KIND = "session"
MESSAGE_KIND = "message"
CARRIED = "carried"          # session --carried--> message
GROUP_TEXT_LINES = 100       # member lines quoted in a group episode
GROUP_TEXT_BYTES = 20_000


@dataclass
class IngestReport:
    records_seen: int = 0
    ingested: int = 0
    skipped_no_clock: int = 0
    sessions: int = 0            # distinct session objects
    messages: int = 0            # distinct message objects
    relations: int = 0           # session--carried-->message edges
    sightings: int = 0
    groups: int = 0              # group episodes (sessions + deliveries)
    grouped_members: int = 0     # member episodes carrying member_of
    singletons: int = 0          # groups of one: honest, not failures
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
            "groups": self.groups,
            "grouped_members": self.grouped_members,
            "singletons": self.singletons,
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


def _group_text(texts: list[str]) -> tuple[str, bool]:
    """Member lines joined for the group episode, capped.

    Returns (text, truncated). The cap bounds group-episode size: a 7,260-line
    daemon group must not become a 700KB episode. Retrieval reads the members
    for detail; the group carries the shape of the session, not its bytes.
    """
    kept: list[str] = []
    size = 0
    for t in texts:
        if len(kept) >= GROUP_TEXT_LINES or size >= GROUP_TEXT_BYTES:
            break
        kept.append(t)
        size += len(t) + 1
    truncated = len(kept) < len(texts)
    if truncated:
        kept.append(f"... +{len(texts) - len(kept)} more lines; "
                    f"see member episodes")
    return "\n".join(kept), truncated


def _content_field(schema: dict) -> str | None:
    for layer in ("envelope", "payload"):
        for e in (schema.get("record_layers") or {}).get(layer, []) or []:
            if isinstance(e, dict) and e.get("role") == "content":
                return e.get("path")
    return None


def _group_event(stream: str, g, *, recs: list[dict],
                 classes: list, arousals: list[float],
                 content_field: str | None) -> Event:
    """One episode for the whole group: session or delivery.

    The wall is the first member's wall, so the group leads its members in
    tau order. The arousal is the members' mean -- a session containing one
    rare line scores high, which is how the 49 bounces in 1.5M lines stay
    visible at session granularity instead of dissolving into routine means.
    """
    texts = [str(r.get(content_field) or "") for r in recs] \
        if content_field else [str(r) for r in recs]
    body, truncated = _group_text([t for t in texts if t])
    hist: dict[str, int] = {}
    for c in classes:
        if c is not None:
            hist[c] = hist.get(c, 0) + 1
    kind = SESSION if g.kind == SESSION else DELIVERY
    canon = g.canonical
    objs = [(SESSION_KIND if g.kind == SESSION else MESSAGE_KIND, canon)]
    feats = {
        "episode_role": "group",
        "group_canonical": canon,
        "group_kind": g.kind,
        "member_count": len(g.members),
        "span_seconds": round(g.span_seconds, 1),
        "class_histogram": hist,
        "truncated": truncated,
        "subject_norm": f"{g.kind} {canon}: {len(g.members)} records",
        "body_text": body,
        "arousal": (sum(arousals) / len(arousals)) if arousals else 0.0,
    }
    return Event(stream=stream, kind=kind, wall=g.first_wall,
                 features=feats, objects=objs)


def ingest_records(engine, schema: dict, records: Iterable[dict], *,
                   stream: str, limit: int | None = None,
                   classify: Callable[[dict], str | None] | None = None,
                   commit_every: int = 1000,
                   group: bool = True,
                   gap_seconds: int = 3600) -> IngestReport:
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

    # Per-record pass: wall, identity, message ids, arousal, class.
    walls: list = []
    canonicals: list = []
    id_lists: list = []
    arousals: list = []
    kept: list[int] = []          # batch indices that survived the clock
    for bi, (r, c) in enumerate(zip(batch, classes)):
        rep.records_seen += 1
        wall = _wall(r, time_field)
        if wall is None:
            rep.skipped_no_clock += 1
            continue
        canonical = ":".join(str(r.get(k, "")) for k in key_fields) \
            if key_fields else None
        ids = _message_ids(schema, r, rx)
        arousal = (1.0 - freq[c] / top) if c is not None else 1.0
        walls.append(wall)
        canonicals.append(canonical)
        id_lists.append(ids)
        arousals.append(arousal)
        kept.append(bi)

    content_field = _content_field(schema)

    # Grouping pass: sessions and deliveries over the surviving records.
    groups = []
    group_of: dict[int, object] = {}
    if group:
        recs = [batch[bi] for bi in kept]
        groups = []
        idx_of = {id(r): n for n, r in enumerate(recs)}
        groups = assign(
            recs,
            session_of=lambda r: canonicals[idx_of[id(r)]],
            messages_of=lambda r: id_lists[idx_of[id(r)]],
            wall_of=lambda r: walls[idx_of[id(r)]],
            gap_seconds=gap_seconds)
        for g in groups:
            for pos in g.members:
                group_of[kept[pos]] = g
        rep.groups = sum(1 for g in groups if len(g.members) > 1)
        rep.singletons = sum(1 for g in groups if len(g.members) == 1)

    events: list[Event] = []
    # meta aligns with events: (canonical, ids, wall, kind, group, is_group)
    meta: list = []
    emitted_group: set[int] = set()
    for n, bi in enumerate(kept):
        r = batch[bi]
        g = group_of.get(bi)
        feats = dict(r)
        feats["identity_canonical"] = canonicals[n]
        feats["message_ids"] = id_lists[n]
        feats["event_class"] = classes[bi]
        feats["arousal"] = arousals[n]
        objs = (([(SESSION_KIND, canonicals[n])] if canonicals[n] else [])
                + [(MESSAGE_KIND, f"queue:{m}") for m in id_lists[n]])
        if g is not None:
            feats["member_of"] = g.canonical
            feats["group_rank"] = g.members.index(n)
            rep.grouped_members += 1
            if id(g) not in emitted_group:
                # The group episode leads its members: same file position as
                # the first member, so tau order tells the story in order.
                # Singletons emit nothing: a group of one duplicates its
                # member exactly, and the gate rejects the copy on novelty --
                # correctly, which is why this is a rule rather than a hope.
                # The member still carries member_of, naming the group that
                # logically contains it. Note the member event below is
                # appended unconditionally: skipping the GROUP is not
                # skipping the MEMBER.
                emitted_group.add(id(g))
                if len(g.members) > 1:
                    events.append(_group_event(
                        stream, g,
                        recs=[batch[kept[p]] for p in g.members],
                        classes=[classes[kept[p]] for p in g.members],
                        arousals=[arousals[p] for p in g.members],
                        content_field=content_field))
                    meta.append((None, [], walls[n], "group", g, True))
        events.append(Event(stream=stream, kind="message", wall=walls[n],
                            features=feats, objects=objs))
        meta.append((canonicals[n], id_lists[n], walls[n], "message",
                     g, False))

    results = engine.new_events(events, commit_every=commit_every)

    # The relations pass. Sightings were already linked inside new_event;
    # this writes only the session--carried-->message edges, which need both
    # ends and therefore cannot be written until both objects exist.
    # Re-resolving converges on the same ids new_event created.
    # INSERT OR IGNORE makes the whole pass idempotent: re-running a batch
    # adds nothing.
    session_ids: dict[str, str] = {}
    message_ids: dict[str, str] = {}
    # Groups need their kinds registered the same way objects did: the
    # registry should name what exists rather than discover it later.
    for kind in (SESSION, DELIVERY):
        store.conn.execute("INSERT OR IGNORE INTO lu_kind(name) VALUES (?)",
                           (kind,))
    store._commit()
    for (canonical, ids, wall, _kind, _g, _is_group), res in zip(meta, results):
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

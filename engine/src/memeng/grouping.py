# =====================================================================
# Project Positronic
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================

"""Session grouping: lines in, sessions and deliveries out.

One episode per log line is the wrong unit. An smtpd session is ~132 lines; a
pvl visit is a (who, room, time-window) span; a qmgr daemon writes 7,260 lines
in one process lifetime and means none of it as a session. The gate scoring
lines, novelty measured per line, and 21,219 carried edges pointing at
fragments are all the same defect: the unit of ingestion is not the unit of
the phenomenon.

The rule, measured on 300,000 real lines rather than asserted:

* a record carrying message ids belongs to its FIRST message id's group --
  a delivery. The re-injection line naming two ids lands in the old id's
  group while announcing the new one; the carried relation already captures
  the handoff, so the group does not have to be in two places at once.
* a record with no message id belongs to its session key's group -- a
  session -- split whenever the gap since the previous line exceeds
  `gap_seconds`. Measured: p90 intra-group gap 300s, p99 11h (pid reuse
  across the day boundary). The default 3600s splits reuse without
  shredding real sessions.
* session groups are capped at `max_span_seconds` / `max_lines`, splitting
  with a sequence suffix. Without the cap, one qmgr daemon lifetime is a
  single 7,260-line group -- a true fact (one process wrote them) and useless
  for retrieval. The cap bounds group-episode text; the message groups carry
  the per-delivery meaning for daemon output anyway.

What grouping does NOT do: it never merges across keys (a false merge is not
recoverable), it never reorders (members keep file order), and singletons stay
singletons -- 1,487 of 6,368 groups on the measured corpus. A group of one is
not a failure; forcing singletons together would be the failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

SESSION = "session"
DELIVERY = "delivery"

DEFAULT_GAP_SECONDS = 3600
DEFAULT_MAX_SPAN_SECONDS = 86400
DEFAULT_MAX_LINES = 2000


@dataclass
class Group:
    kind: str                      # session | delivery
    key: str                       # session canonical or queue:<id>
    members: list[int] = field(default_factory=list)   # record indices
    first_wall: datetime | None = None
    last_wall: datetime | None = None
    seq: int = 1                   # split sequence within one key

    @property
    def canonical(self) -> str:
        return self.key if self.seq == 1 else f"{self.key}#{self.seq}"

    @property
    def span_seconds(self) -> float:
        if self.first_wall is None or self.last_wall is None:
            return 0.0
        return (self.last_wall - self.first_wall).total_seconds()


def assign(records: list[dict], *, session_of, messages_of, wall_of,
           gap_seconds: int = DEFAULT_GAP_SECONDS,
           max_span_seconds: int = DEFAULT_MAX_SPAN_SECONDS,
           max_lines: int = DEFAULT_MAX_LINES) -> list[Group]:
    """Group record indices. Pure function: no I/O, no engine, no schema.

    `session_of(r)` -> canonical session key or None; `messages_of(r)` ->
    list of message ids (may be empty); `wall_of(r)` -> datetime. Records are
    visited in file order and members keep file order, so replaying a file
    twice assigns identically.
    """
    open_groups: dict[str, Group] = {}
    done: list[Group] = []
    seq_of: dict[str, int] = {}
    # Closed group identities. The flush below must not re-add a group that
    # split or rotated out of `open_groups`; testing membership by scanning
    # `done` is O(open x done) -- measured as a hang (zero progress, flat
    # RSS) on 100k single-visit groups, roughly 30 billion comparisons.
    closed_ids: set[int] = set()

    def current(key: str, kind: str, wall: datetime) -> Group:
        g = open_groups.get(key)
        if g is not None and g.kind == kind:
            gap = (wall - g.last_wall).total_seconds() \
                if g.last_wall is not None else 0.0
            if gap <= gap_seconds and \
                    g.span_seconds <= max_span_seconds and \
                    len(g.members) < max_lines:
                return g
            closed_ids.add(id(g))
            # Split: same key, new group. The old one is done. first_wall is
            # left unset so the current record sets it -- carrying the old
            # group's start would inflate the new span and re-split at once.
            done.append(g)
            seq_of[key] = g.seq + 1
            nxt = Group(kind=kind, key=key, seq=seq_of[key])
            open_groups[key] = nxt
            return nxt
        if g is not None:
            done.append(g)
            closed_ids.add(id(g))
        # A split key keeps counting across splits AND reopenings: the third
        # visit to a key is #3 whether the second group closed by gap or by
        # cap. Tracked in a dict, not scanned, so 10,000 groups do not cost
        # 100M comparisons.
        seq_of[key] = seq_of.get(key, 0) + 1
        nxt = Group(kind=kind, key=key, seq=seq_of[key])
        open_groups[key] = nxt
        return nxt

    for i, r in enumerate(records):
        wall = wall_of(r)
        ids = messages_of(r) or []
        if ids:
            key = f"queue:{ids[0]}"
            g = current(key, DELIVERY, wall)
        else:
            sess = session_of(r)
            if sess is None:
                # No identity on either axis. A group of one, honestly: the
                # alternative is merging the unidentifiable together, which
                # manufactures sessions out of nothing.
                done.append(Group(kind=SESSION, key=f"ungrouped:{i}",
                                  members=[i], first_wall=wall,
                                  last_wall=wall))
                continue
            g = current(sess, SESSION, wall)
        g.members.append(i)
        if g.first_wall is None:
            g.first_wall = wall
        g.last_wall = wall

    done.extend(g for g in open_groups.values() if id(g) not in closed_ids)
    # Deterministic order: first appearance in the file.
    done.sort(key=lambda g: min(g.members) if g.members else 0)
    return done

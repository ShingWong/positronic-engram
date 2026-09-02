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

"""MemoryEngine v0 — data models."""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Kind(str, Enum):
    MESSAGE = "message"
    SNAPSHOT = "snapshot"
    DAY_TOKEN = "day_token"
    WEEK_TOKEN = "week_token"
    PERIOD_REF = "period_ref"


class Tier(str, Enum):
    NORMAL = "normal"
    ESCALATED = "escalated"
    FLASHBULB = "flashbulb"


class Provenance(str, Enum):
    WITNESSED = "witnessed"
    BOUNDARY = "boundary"
    RECONSTRUCTED = "reconstructed"


@dataclass
class Event:
    stream: str
    kind: str                      # lookup key; validated against lu_kind
    wall: datetime | None = None   # None => store clock
    persons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    fuzz_width: timedelta | None = None

    def __post_init__(self) -> None:
        self.wall = self.wall or utcnow()


@dataclass
class Prediction:
    source: str                    # rule id or cadence stat name
    feature: str                   # expected feature key
    expected: Any
    actual: Any | None = None
    matched: bool | None = None    # filled at gate time


@dataclass
class GateVerdict:
    score: float
    novelty: float
    arousal: float
    gain: float
    threshold: float
    encoded: bool


@dataclass(frozen=True)
class Trigger:
    type: str
    tau: float
    payload: dict[str, Any]


Handler = Callable[[Trigger], None]
Handlers = dict[str, Handler]


@dataclass
class EpisodeRecord:
    id: uuid.UUID
    stream: str
    kind: str
    wall: datetime
    mono: int
    tau: float
    persons: list[str]
    subject_norm: str | None
    salience: float
    tier: Tier
    strength: float                # decay constant S (None/inf => flashbulb)
    provenance: Provenance
    fuzz_lo: datetime | None
    fuzz_hi: datetime | None
    precision_src: str             # exact | phase_of_day | relative_anchor | inferred
    is_anchor: bool
    features: Any = None
    domain_id: int | None = None


@dataclass
class EventResult:
    verdict: GateVerdict
    tau: float
    episode_id: uuid.UUID | None
    predictions: list[Prediction]
    trace: list[Trigger]


@dataclass
class PruneReport:
    scanned: int
    day_merged: int
    week_merged: int
    expired: int
    residues: int = 0
    objects_dormant: int = 0
    objects_forgotten: int = 0

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

"""Deterministic technical-entity extraction from free text.

Pattern families: hyphen/underscore identifiers (persona-bot-v2),
letter+digit tokens (web2, d3d11), IPv4, host:port pairs (whisperx:8000),
dotted names (memory.db). Lowercase-only; junk filtered by length bounds
and a small stoplist. Pure stdlib — safe to run inside the encode path.
"""
from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+)+\b"),
    re.compile(r"\b(?![0-9])[a-z][a-z0-9]*\d[a-z0-9]*\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?:\d{2,5}\b"),
    re.compile(r"\b[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+\b"),
]
_STOPWORDS = frozenset({"e.g", "i.e"})
_MIN_LEN, _MAX_LEN = 3, 40


def extract_entities(text: str) -> set[str]:
    """Lowercase technical identifiers mentioned in `text`."""
    if not text:
        return set()
    low = text.lower()
    out: set[str] = set()
    for pat in _PATTERNS:
        for m in pat.finditer(low):
            name = m.group(0)
            if (_MIN_LEN <= len(name) <= _MAX_LEN
                    and name not in _STOPWORDS):
                out.add(name)
    return out

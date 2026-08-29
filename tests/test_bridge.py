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

"""Bridge-level tests: kairos_brain.ask() against a temp store."""
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
for p in (HERE, Path("/usr/local/devel/positronic/positronic-private")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(HERE / "engine" / "src"))

import kairos_brain as kb  # noqa: E402
from memeng.models import Event  # noqa: E402


def _fresh(tmp_path, monkeypatch):
    kb._engine = None                      # reset singleton
    monkeypatch.setattr(kb, "DB", tmp_path / "mem.db")
    return kb.brain()


def _sess_event(subject, body):
    return Event(stream="kairos:sessions", kind="message",
                 persons=["p_kairos"],
                 wall=datetime.now(timezone.utc),
                 features={"subject_norm": subject, "sender": "p_kairos",
                           "arousal": 0.95, "body_text": body})


def test_ask_returns_dossier_with_sightings(tmp_path, monkeypatch):
    e = _fresh(tmp_path, monkeypatch)
    e.new_event(_sess_event("web2 deploy",
                            "deployed persona-bot-v2 on web2 192.168.4.22"))
    d = kb.ask("WEB2")                     # case-insensitive
    assert d["found"] is True
    assert d["object"]["canonical_name"] == "web2"
    assert d["sightings"][0]["subject_norm"] == "web2 deploy"


def test_ask_unknown_falls_back_to_recall(tmp_path, monkeypatch):
    e = _fresh(tmp_path, monkeypatch)
    e.new_event(_sess_event("liqui-fire rx quote",
                            "quoted liqui-fire rx pump seal kit"))
    d = kb.ask("nonexistent-hostname-xyz")
    assert d["found"] is False
    assert isinstance(d["episodes"], list)

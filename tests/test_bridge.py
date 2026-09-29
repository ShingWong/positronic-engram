# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================
# This program is DUAL-LICENSED. You may redistribute and/or modify it 
# under the terms of the GNU Affero General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your 
# option) any later version.
#
# Alternatively, commercial entities, multi-tenant instances, and Managed 
# Service Providers (MSPs) may utilize this program under a separate, 
# proprietary Commercial License Waiver issued directly by the copyright 
# holder, completely exempt from the network-use copyleft restrictions of 
# the AGPLv3 Section 13.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License 
# along with this program. If not, see <https://gnu.org>.
# =====================================================================

"""Bridge-level tests: kairos_brain.ask() against a temp store."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "engine" / "src"))

# kairos_brain lives in positronic-private, which is deliberately not a
# dependency of this public repo: it holds PII and is behind a gitignore
# firewall. So there is no default location to fall back on -- a path baked in
# here would only ever work on the machine that wrote it, and would name that
# machine's layout. Point POSITRONIC_PRIVATE_DIR at the checkout, the same
# variable name consumers/benchmarks/harness/paths.py already uses.
_PRIVATE = os.environ.get("POSITRONIC_PRIVATE_DIR", "").strip()
if _PRIVATE and Path(_PRIVATE).is_dir() and _PRIVATE not in sys.path:
    sys.path.insert(0, _PRIVATE)

import pytest

kb = pytest.importorskip(
    "kairos_brain",
    reason="needs positronic-private; set POSITRONIC_PRIVATE_DIR to its checkout",
)
from memeng.models import Event


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
